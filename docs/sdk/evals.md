# Evalsets — Eval-Driven Development

> Status: local loop. Xybrid's eval harness turns a bad result into a test case,
> a leaderboard, a CI wall, and a guardrail — without an ML team. This document
> specifies the **on-disk evalset format**, the **task-implied graders**, the
> **run/score record**, and the **`xybrid eval`** CLI. The file is the source of
> truth; a remote backend holds a synced copy.

## Directory layout

An evalset lives in `evals/<name>/`:

```
evals/intent-classifier/
├── evalset.yaml        # manifest
├── cases.jsonl         # one JSON case per line
└── clips/              # optional binary payload sidecars (audio/image)
    └── x.wav
```

Binary payloads are sibling files referenced by **relative path** (`file:clips/x.wav`)
so an evalset is a normal git directory — versionable, diffable, reviewable in PRs.

> **Security:** payload references are validated on load. A reference that
> escapes the evalset directory (absolute path, `..`, or an escaping symlink) is
> rejected — an imported or flagged evalset cannot read or exfiltrate host files.

## Manifest — `evalset.yaml`

```yaml
name: intent-classifier
task: classify            # classify | chat | summarize | extract | asr | tts | embedding | vlm
version: 3                # bumped when cases change; every run records it
kind: quality             # quality | safety | performance  (default: quality)
labels: [refund, cancel, question, other]   # classify: allowed labels + alias source
gate:                     # optional; consumed by `eval gate` and over-the-air promotion
  min_quality: 0.92
  max_p95_latency_ms: 800
  min_cases: 30           # below this → inconclusive
  non_inferiority_margin: 0.02
# Tier 3 only:
# grader: { judge_model: …, rubric: …, custom: wasm:./graders/my_metric.wasm }
```

Tier 1 never writes this file — `xybrid eval init` and `xybrid eval pull`
generate and maintain it.

## Case — `cases.jsonl`

One JSON object per line:

```json
{ "id": "c_01HF8Q", "input": { "text": "I want my money back" },
  "expected": { "label": "refund" },
  "source": "flagged", "trace_id": "tr_9a31", "added": "2026-06-10" }
```

| Field | Meaning |
|---|---|
| `id` | Stable case id. |
| `input` | `{"text": …}`, `{"audio": "file:…"}`, or `{"image": "file:…"}` (mirrors `Envelope`). |
| `expected` | `{"label": …}`, `{"text": …}`, or `{"json": …}`. **Optional** — absent ⇒ golden mode. |
| `source` | `flagged` \| `authored` \| `imported` \| `golden`. |
| `trace_id` | Originating inference trace (when `source = flagged`). |
| `added` | ISO date. |

### Case governance (data trust)

A production-fed evalset is only trustworthy if it is curated. Every case can
carry lifecycle state (all optional, with safe defaults):

`review_status` (`unreviewed`\|`reviewed`\|`golden`) · `severity` · `weight`
(default 1.0) · `cluster_id` · `dedupe_hash` · `privacy_class`
(`captured`\|`redacted`\|`metadata-only`) · `source_confidence` · `split`
(`dev`\|`regression`) · `owner` · `expires_at` · `quarantine_reason`.

Quarantined or expired cases are **excluded from gate runs but retained for
audit**. Gate runs use the `regression` split. Splits are auto-assigned, never
asked of a Tier 1 developer.

## Graders (the task implies the metric)

Quality is **always** normalized to a `0..=1` score plus a per-case verdict
(`pass` \| `fail` \| `unblessed`), so the CLI, the console, and the gate never
depend on which grader produced the number.

| `task` | Default grader | Needs `expected`? |
|---|---|---|
| `classify` | Normalized label match (case/whitespace-insensitive; aliases from `labels`). Unmatched → fail. | Yes |
| `asr` | Word Error Rate; `quality = clamp(1 - WER, 0, 1)`. | Yes |
| `extract` | Per-field match over a reference JSON object. | Yes |
| `chat` / `summarize` / `vlm` | LLM-as-judge (currently an offline deterministic stand-in). | Optional |
| `tts` | Golden-output + duration/RTF sanity. *(deferred)* | No |
| `embedding` | recall@k over labeled pairs. *(deferred)* | Yes |
| any, no `expected` | **Golden mode** (see below). | No |

### Golden mode

A case with no `expected` cannot be scored until a good output is **blessed** as
its reference. Until then it grades as `unblessed` (neither pass nor fail).
Blessing writes the chosen output into `expected` and sets
`review_status: golden`; later runs diff against it (currently an exact-diff for
deterministic tasks; judge-equivalence is deferred to the calibrated judge).

## Run & scores

A run is `evalset × candidate → verdicts`, stored under
`~/.xybrid/eval-runs/<run_id>/run.json`:

```json
{
  "run_id": "run_01HF", "evalset": "intent-classifier", "evalset_version": 3,
  "candidate": { "model_id": "qwen3.5-0.8b", "config": { "temperature": 0.0, "seed": 42 } },
  "environment": { "host": "macos-arm64", "backend": "llamacpp", "execution_provider": "metal", "sdk_version": "0.1.2" },
  "scores": {
    "quality": 0.84, "pass": 42, "fail": 8, "verdict": "fail",
    "ci": { "low": 0.78, "high": 0.90, "n": 50, "repeats": 1 },
    "latency_p50_ms": 210, "latency_p95_ms": 640,
    "ttft_p95_ms": null, "cold_start_ms": null, "peak_memory_mb": null,
    "crash_or_timeout": 0, "scorer_version": "eval-scorer-v0"
  },
  "cases": [ { "id": "c1", "verdict": "fail", "score": 0.0, "latency_ms": 188 } ]
}
```

The `scores` block reserves the full on-device SLO field set (TTFT/ITL, cold
start, energy/thermal, memory, bundle, offline) and judge-identity fields
(`grader_id`, `rubric_version`, `judge_model`, `judge_prompt_hash`, seed,
temperature) — populated incrementally, present in the schema from day one.

## The gate is statistical

`eval gate` returns **pass / fail / inconclusive** — never a false green on
noise. Evaluation order is fixed: **minimum case count → latency SLO → quality
vs. confidence interval → non-inferiority vs. baseline**. A delta within the
non-inferiority margin, a sample below `min_cases`, or a confidence interval that
straddles the threshold all resolve to `inconclusive` (CI-neutral). Confidence
intervals come from a deterministic bootstrap (fixed seed, reproducible across
machines). Candidates that score erratically across repeats are flagged flaky and
excluded from ranking.

Exit codes: `pass` → 0, `inconclusive` → 0 (or 2 with `--strict`), `fail` → 2.

## CLI

| Command | Tier | Behavior |
|---|---|---|
| `xybrid eval` | 1 | Discover `evals/` and list evalsets. |
| `xybrid eval init <task> [--name N]` | 1 | Scaffold `evals/<name>/`. |
| `xybrid eval inspect <path>` | 1 | Validate + summarize an evalset. |
| `xybrid eval run <evalset> --model <id> [--limit N] [--no-capture]` | 1–3 | Score a candidate; persist the run. |
| `xybrid eval compare <evalset> --model <id>… [--auto]` | 1 | Leaderboard + recommended winner (hard-constraint filter → quality → tie-breakers). |
| `xybrid eval gate <evalset> [--run <id> \| --model <id>] [--strict]` | 2 | Pass/fail/inconclusive with a CI-aware exit code (CI primitive). |
| `xybrid eval show <run_id>` / `diff <a> <b>` | 2–3 | Re-print / side-by-side delta. |

`run` / `compare` / `gate --model` execute the candidate through the same path as
`xybrid run`, so they require a platform preset (`--features platform-*`) built
in. The offline commands need no backend.

## Determinism

Eval runs pin seed (default 42) and use greedy decoding by default, and record
model id, environment (host/backend/EP), and SDK version — so a number can always
be reproduced and a regression always attributed.

## See also

- Telemetry / feedback event: [`telemetry.md`](telemetry.md)
