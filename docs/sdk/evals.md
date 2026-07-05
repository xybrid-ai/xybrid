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
> rejected, and the target must exist as a regular file — an imported or flagged
> evalset cannot read or exfiltrate host files. `evalset.yaml` and `cases.jsonl`
> must also be regular files before they are read.

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
  repeats: 3              # rerun the full evalset; flaky repeats cannot pass
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

All cases still execute and are recorded in a run. Only cases with
`split: regression` that are not quarantined or expired are marked
`counts_for_gate: true` and feed gate scores, latency SLOs, ship, and promote.
Splits are auto-assigned, never asked of a Tier 1 developer.

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
    "flaky": false, "repeat_qualities": [0.84, 0.83, 0.86],
    "latency_p50_ms": 210, "latency_p95_ms": 640,
    "ttft_p95_ms": null, "cold_start_ms": null, "peak_memory_mb": null,
    "crash_or_timeout": 0, "scorer_version": "eval-scorer-v0"
  },
  "cases": [ { "id": "c1", "verdict": "fail", "score": 0.0, "latency_ms": 188, "counts_for_gate": true } ]
}
```

Per-case outputs are not captured by default. Pass `--capture` to `run`,
`compare`, `gate --model`, or `ship --model` when a local run record needs the
raw model output for debugging.

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
machines). When `gate.repeats > 1`, the full evalset is executed that many
times; the stored per-case rows come from the first repeat, latency pools across
repeats, and per-repeat mean qualities are stored in `repeat_qualities`.
Candidates whose repeat-quality sample standard deviation exceeds the flaky
threshold are marked `flaky` and resolve no better than `inconclusive`.

Exit codes: `pass` → 0, `inconclusive` → 0 (or 2 with `--strict`), `fail` → 2.

## CLI

| Command | Tier | Behavior |
|---|---|---|
| `xybrid eval` | 1 | Discover `evals/` and list evalsets. |
| `xybrid eval init <task> [--name N]` | 1 | Scaffold `evals/<name>/`. |
| `xybrid eval inspect <path>` | 1 | Validate + summarize an evalset. |
| `xybrid eval pull <evalset> [--accept-all] [--dry-run]` | 1 | Drain platform-curated or local flagged cases into the evalset via a review queue. |
| `xybrid eval inbox [--period 7d] [--model <id>] [--source report\|signal] [--rating up\|down]` | 1 | View the platform failure inbox (flagged results + monitor auto-flags) in the terminal — the read side of collect. Needs `XYBRID_API_KEY`. |
| `xybrid eval run <evalset> --model <id> [--limit N] [--capture]` | 1–3 | Score a candidate; persist the run. |
| `xybrid eval compare <evalset> --model <id>… [--auto] [--capture]` | 1 | Leaderboard + recommended winner (hard-constraint filter → quality → tie-breakers). |
| `xybrid eval gate <evalset> [--run <id> \| --model <id>] [--strict] [--capture]` | 2 | Pass/fail/inconclusive with a CI-aware exit code (CI primitive). |
| `xybrid eval ship <evalset> [--run <id> \| --model <id>] [--capture]` | 2 | Record a promotion only when the current gate passes. |
| `xybrid eval show <run_id>` / `diff <a> <b>` | 2–3 | Re-print / side-by-side delta. |

`run` / `compare` / `gate --model` / `ship --model` execute the candidate
through the same path as `xybrid run`, so they require a platform preset
(`--features platform-*`) built in. The offline commands need no backend.
`--no-capture` is accepted as a hidden deprecated alias for older scripts, but
capture is already off unless `--capture` is set.

### Pulling curated cases

`xybrid eval pull <evalset>` reads the evalset manifest name, then uses
`XYBRID_API_KEY` (or global `--api-key`) plus the platform URL to fetch pending
cases from `/v1/evals/cases?evalset=<name>&status=pending`. The same review
queue is used for remote and local cases:

- accept appends the case to `cases.jsonl` first, then reconciles the remote
  case to accepted;
- skip leaves the remote case pending for a later pull;
- discard keeps it out of `cases.jsonl` and reconciles the remote case to
  discarded.

Accepted remote cases are written as production regressions: `source: flagged`,
`review_status: reviewed`, `split: regression`, with trace metadata preserved
when the platform case carries it. The local `cases.jsonl` is the source of
truth because gates run against it: accepted cases are appended and the evalset
version is bumped before any remote status PATCH is attempted. If the local
write fails, remote status is not patched. `--accept-all` accepts every new
pending case without prompting. `--dry-run` shows what would be appended but
does not write files or patch remote state.

Remote status reconciliation is best-effort after the local write. A pending
remote case is not appended again when its id or dedupe key is already present
in `cases.jsonl`, but `pull` still retries the remote acceptance PATCH so a
prior local-ahead crash converges on the next run. HTTP 409 means the case was
already resolved on the platform; `pull` logs that case and continues. Any other
PATCH failure makes the command fail after it has tried the remaining cases,
with local cases still saved; re-running `xybrid eval pull` retries the
incomplete remote reconciliation. HTTP 501 means the platform backend does not
support eval telemetry.

If remote listing fails for any reason, including offline transport, auth
rejection, or a platform backend without eval telemetry support, `pull` warns
with the specific reason and falls back to the local inbox file
`~/.xybrid/inbox/<evalset>.jsonl`; passing `--inbox <path>` forces that local
mode.

### Feedback capture

`InferenceResult::report(Feedback::...)` emits metadata only by default. Calling
`Feedback::capture()` opts a single report into payload capture. Captured
feedback may include the text input, text output, expected correction, and note;
each field is truncated before telemetry publication.

Only plain text inference inputs are captured. Audio, image, embedding, and
multi-part inputs remain metadata-only unless the caller also supplies a text
correction or note.

The platform can promote only flagged events that include a captured text
`input` string. `xybrid eval pull` skips SDK-flagged platform cases without that
input because there is no runnable eval case to write. Signal-sourced or
otherwise no-input cases are not drainable by the CLI; they are left pending for
platform-side triage rather than patched to discarded.

## Determinism

Eval runs pin seed (default 42) and use greedy decoding by default, and record
model id, environment (host/backend/EP), and SDK version — so a number can always
be reproduced and a regression always attributed.

## See also

- Telemetry / feedback event: [`telemetry.md`](telemetry.md)
