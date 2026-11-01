# HF reference for LFM3 d1 decisions (LiquidAI's own code)

The independent side of the decision gates: LiquidAI's own d1 code and weights,
`LiquidAI/d1-2B` at `10e7671d44e4b37700020b1eed3f594c5f2af0b4` (artifact
`d1-2b-hf`), run on decision cases one question per sequence
(`set_size = 1`), the way llama.cpp's server reads every question. It records,
per case, the prompt the reference renders, its token ids, the label groups,
the temperature, the answer slot's raw logits and vocabulary log-probabilities,
and the answers. It is G1's reference (contract 5). Where the reference and
llama.cpp disagree, [`DIFFERENCES.md`](DIFFERENCES.md) lists every known
difference with a case.

Nothing here is a hand port: the runner imports the snapshot's own `prompt.py`,
`api.py`, `attention.py`, `runner.py`, `calibration.py` and model code, calls
its public `model.system_one(state, {id: question}, set_size=1)`, and reads
what that call evaluated through hooks.

## Run it from a clean checkout

Needs `curl`, `jq`, [`uv`](https://docs.astral.sh/uv/) and about 4 GB of disk.
The runner peaks at about 6.7 GB resident: 3.8 GB of that is the memory-mapped
checkpoint, plus at most about 2.3 GB of its own memory between calls and the
transient float32 copies during a forward. Linux x86_64 is the reference
platform (the lock pins CPU-only torch for it).

```bash
tools/conformance/decision_hf/download.sh   # the pinned snapshot, size and sha256 checked (3.9 GB)
tools/conformance/decision_hf/run.sh --check-f32-parity integration-tests/fixtures/decision/hf/cases
# any case files in the decision case schema, e.g. the upstream oracle's:
tools/conformance/decision_hf/run.sh --out /tmp/hf-obs integration-tests/fixtures/decision/cases
```

`download.sh` fetches every file `pins.json` lists from its immutable URL
(`resolve/<revision>`) into `$XYBRID_DECISION_HF_CACHE/snapshots/<revision>/d1-2B/`,
keeping only files whose size and sha256 match. The default cache is `.cache/`
in this folder: git ignores it, and `clean.sh` (run by the root `clean.sh`)
removes it with the module cache, the cross-checks' GGUFs and llama.cpp build,
and the run logs. A cache moved with `XYBRID_DECISION_HF_CACHE` is yours to
clean. `run.sh` builds the pinned environment (Python 3.12 with
`requirements.txt`, through uv; on Linux it points uv at the PyTorch CPU index),
runs offline, and hands its arguments to `run_reference.py`:

| Option | Default | Meaning |
|---|---|---|
| `CASE...` | | case files, or directories of them |
| `--out DIR` | `integration-tests/fixtures/decision/hf/observations` (`…/packed` with `--set-size default`) | one `<id>.json` observation per case |
| `--run-log FILE` | none | what the observations were made from, recorded when the run starts (pins, verified snapshot files, lock, recipe sources, request digests), the digest of every observation written, the parity coverage, environment, hardware, timings and memory |
| `--threads N` | 1 | torch threads; the committed observations use 1 (see Precision) |
| `--precision` | `f32` | `f32`: float32 forward over the bf16 weights. `f32-load`: the model card's `dtype=torch.float32` (about 9 GB). `bf16`: the bfloat16 forward |
| `--set-size` | `1` | `default`: the checkpoint's own set size (16), every question of a request packed after one state |
| `--check-f32-parity` | off | replay every module call of every forward on its float32 copy; any byte that differs, any floating input that is not float32, or a hooked module never verified fails the run (exit 1, nothing written) |

The runner reads only `id` and `request_text` from a case, so it takes the
upstream oracle's cases and this folder's alike. It writes its observations
only once every case has run and the parity proof (when asked for) holds: any
broken invariant exits 1 with nothing written. A run of the 46 committed cases
takes about 7 minutes on 4 vCPUs at one thread: most questions take 5 to 9 s,
and the 3,605-token case takes 86 s. The parity proof roughly doubles that.

`synthetic_median.py` (the score-median cases) needs no weights:
`download.sh --no-weights` fetches the 33 other pinned files (about 18 MB) and
`run.sh synthetic_median.py` verifies exactly those.

## What an observation holds

One file per case, `<id>.json`:

```jsonc
{
  "schema_version": 1, "case": "<id>", "source": "hf-reference",
  "artifact": "d1-2b-hf", "settings": "hf-cpu-f32-t1",
  "request_text_sha256": "…",
  "hf":     { "parse": …, "prepare": …, "evaluate": …, "finalize": … },
  "detail": { "parse": …, "prepare": […], "evaluate": […] }
}
```

`hf` has the stages and the shapes of the case schema's `upstream` block
(`integration-tests/fixtures/decision/schema.json`, `$defs/upstream`), so the
two references read alike:

- `parse.questions`: id, type and option keys in model order (yes/no: `true`
  then `false`, the reference's own order).
- `prepare.passes[i]`: the exact prompt, its token ids, `output_index` (the
  answer slot, the last token) and `select` (the label tokens of every group).
- `prepare.answers[j]`: groups (`key`, `code`, `range` of `select`) and the
  temperature. `temperature` is the repr of the float64 the reference reads
  from `calibration.json`; `temperature_f32` is the f32 nearest to it, which is
  not what the reference computes with (`detail.prepare[j].temperature_f64` is).
- `evaluate.raw_scores[i]`: **raw logits**, the LM head's output at the slot
  for the `select` tokens, before any normalization. `scores[i]`: each group's
  maximum. This is upstream's quantity.
- `finalize`: the answers `system_one` returns, and the usage.

`detail` holds what only the reference has. `evaluate[i].raw_logprobs` and
`logprob_scores` are the **vocabulary log-probabilities** the reference
actually reads (`runner.py` `_slot_logits`: logits minus their logsumexp, in
float32) and their group maxima. `logsumexp` is the normalizer;
`distribution` is the option distribution `prompt.readouts` computes from them
(group maxima over the temperature, softmaxed in float64). `parse` notes what
Python's json module did (repeated keys, `NaN`, integers past int64, lone
surrogates) and which fields were ignored.

The reference has no HTTP layer, so a refusal is
`{"refusal": {"question"?, "exception", "message"}}`: the Python exception
and the reference's own message. Values use the case schema's encoding: f32 as
bit patterns (`"0x3fab2fdc"`), f64 as numbers with 17 significant digits,
non-finite values as strings. A request with typed questions (reads,
estimates, boxes, `multiple` choices), which have no option groups, leaves
`hf` empty and records the reference's answers in `detail.typed`. With
`--set-size default`, `detail.packed` holds the packed sequence and each
question's slot values, and `hf.finalize` the answers.

## The reference, exactly

- **Code.** transformers 5.19.0, the version the checkpoint records, ships no
  LFM3 classes, so the snapshot's vendored copies run. The runner refuses a
  transformers that brings its own, and checks every snapshot module the
  process imported against `pins.json`. transformers loads remote code twice
  (config, then model), in two hash-named copies; each copy is checked.
- **Call.** `system_one(state, {id: question}, set_size=1)` for each question:
  one sequence, a batch of one, no padding. The checkpoint's `set` attention
  then reads the whole lone prompt both ways on full-attention layers, while
  the sliding-window (256) and short-convolution layers stay causal: the
  reading `attention.py` documents for a lone question. Every question is
  rendered and labeled before any forward pass, as the reference does, so a
  request that fails preparation evaluates nothing.
- **Stages.** parse: `json.loads` (the reference takes Python objects), then
  the reference's own `check_question`, question by question. prepare:
  `_prefix`, `_suffixes`, `_sequences`, `readout_ids`, `aliases` and
  `Calibration.temperature`; the runner checks that the evaluated sequence is
  the rendered text, tokenized. evaluate and finalize: the hooks on the
  backbone input, the LM head and `_readouts`.
- **Temperature.** `calibration.json` `default` (1.337398453966105, f64), for
  every type. The GGUF stores the nearest f32, and llama.cpp reads it back
  through a 6-decimal string, so its T is 3.0e-7 lower (`DIFFERENCES.md` 1.3).

## Precision

The checkpoint stores bfloat16 weights (`config.json` `dtype`), so a float32
forward computes with their exact float32 widenings. The model card's CPU
recipe loads the whole model in float32. That needs about 9 GB, and this
sandbox kills a process near its 8 GiB limit whatever the swap: the load was
killed at 7.8 GB resident. `--precision f32` computes exactly the same
numbers within the limit. A pre-hook on each decoder layer, the embeddings,
the final norm and the LM head swaps in float32 widenings of that module's own
parameters for the duration of its forward, and a forward hook swaps the
bfloat16 data back. The hooked modules cover every text-model parameter, and
the media encoders are guarded so that they never run.

That this is a float32 load's forward is checked, not assumed, and the check
fails closed. `--check-f32-parity` records each hooked module's inputs and
output during real forwards, then replays every call on a float32 deep copy of
the module, the parameters a float32 load holds. The run stops (exit 1, nothing
written) if:

- a floating input is not float32, or an input or output holds a value the
  checks cannot read;
- the replay's output differs from the recorded one in structure (types,
  lengths, keys), dtype, shape or a single byte;
- checking a call fails at all (the replay, an input check, a byte view, a
  comparison or its message): that proves nothing, so it is a failure too;
- the run verified no call, or left a hooked module unverified.

The first failure stands: every hook is guarded and records it, so neither a
later call nor the coverage count passes after it, even if the reference
caught the exception on its way out. Only the reference's own code refuses a
request, and only before its model runs (its question checks, prompt building
and tokenizer). Whatever is raised once a forward has begun, in this runner's
hooks, or by this machine's limits (memory, recursion) ends the run instead of
becoming a recorded refusal.

The modules chain, so a passing proof makes the whole forward a float32
load's. The committed observation and packed runs ran under it; their
coverage is in `hf/provenance.json` under `verification.float32_parity`, and
`provenance.py` refuses an observation run without a passing proof.
`run.sh tests` holds the gate's regression tests (a changed output, one
flipped bit, an extra output, a bfloat16 input, an empty or partial proof,
failures inside the checks, and a failed run writing nothing). Its production
path tests run `main` and the real hooks around a tiny model with the same 27
hooked modules: two cases, the first verifying every module and the second
failing inside the comparison (or any hook, or the model once it ran), write
nothing and exit 1, while refusals before the model runs are still recorded.
They use torch, not the weights.

Determinism:

- **Repeat runs.** Bit-identical at one thread.
- **Thread count.** Four threads are not bit-identical to one (MKL's
  reduction order): logits move by up to 5.2e-6 and probabilities by up to
  7.3e-7, with no answer changes. The settings id pins one thread, and this is
  the reference's own floor.
- **bf16 forward.** It moves logits by up to 0.11 and probabilities by up to
  0.010 on the pilot cases (below), with no answer changes.

Settings ids name these runs (the coordinator's `cpu-t4-fa-off` and
`cpu-t4-fa-on` name llama.cpp contexts). An id stands for every value a run
log records under `numerics`, and `provenance.py` holds each run to all of
them, not to the id alone (`SETTINGS` and `VERIFICATION_SETTINGS` there):

| id | Meaning |
|---|---|
| `hf-cpu-f32-t1` | CPU, 1 thread, float32 forward as above, `set_size = 1`, batch 1, no padding, deterministic algorithms, eval and inference mode |
| `hf-cpu-f32-t1-packed` | the same at the checkpoint's own set size (16): a request's questions packed after one state |
| `hf-cpu-f32-t4`, `hf-cpu-bf16-t1` | verification runs only: 4 threads, or the bfloat16 forward |

## Regenerate the committed fixtures

Only when the pins, the environment, a recipe source or a case changes:
`provenance.py --check` names what changed. Building `llama-server` for the
cross-checks takes about 15 minutes; everything else about an hour, most of it
the gated observation run and the bf16 comparison.

```bash
H=tools/conformance/decision_hf FIX=integration-tests/fixtures/decision/hf RUNS=$H/.cache/runs
python3 -B $H/make_cases.py                                                   # cases/
$H/run.sh --check-f32-parity --run-log $RUNS/observations.run.json $FIX/cases  # observations/
$H/run.sh --check-f32-parity --set-size default --run-log $RUNS/packed.run.json \
    $FIX/cases/hf-diff-packing.json $FIX/cases/hf-card-three.json             # packed/
$H/run.sh synthetic_median.py --run-log $RUNS/synthetic.run.json              # synthetic/median.json
# verification runs, under $RUNS, then compared with the committed observations:
$H/run.sh --out $RUNS/repeat --run-log $RUNS/repeat.run.json <a few cases>
$H/run.sh --threads 4 --out $RUNS/threads --run-log $RUNS/threads.run.json <a few cases>
$H/run.sh --precision bf16 --out $RUNS/bf16 --run-log $RUNS/bf16.run.json <the pilot cases>
for v in repeat threads bf16; do
    python3 -B $H/compare_observations.py $FIX/observations $RUNS/$v --json $RUNS/$v.compare.json
done
python3 -B $H/compare_observations.py $FIX/observations $FIX/packed --json $RUNS/packed.compare.json
# cross-checks (crosscheck/<artifact>.<settings>.json): llama-server from vendor/llama-cpp
cmake -S vendor/llama-cpp -B $H/.cache/llama-build -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build $H/.cache/llama-build -j 3 --target llama-server
$H/.cache/llama-build/bin/llama-server -m $H/.cache/gguf/d1-2B-F16.gguf \
    -t 4 -c 4096 -b 4096 -ub 4096 -fa off -np 1 --port 8089 &
python3 -B $H/crosscheck_server.py --url http://127.0.0.1:8089 --hf $FIX/observations \
    --temperature 1.3373980522155762 --artifact d1-2b-f16 --settings cpu-t4-fa-off \
    --label '<how the server was built and started>' --out $FIX/crosscheck/d1-2b-f16.cpu-t4-fa-off.json $FIX/cases
# stop it; likewise d1-2b-bf16 (cpu-t4-fa-off on every case, cpu-t4-fa-on) and d1-2b-q4km on the pilot cases
python3 -B $H/provenance.py --update --runs $RUNS --llama-build $H/.cache/llama-build
python3 -B $H/provenance.py --check
$H/run.sh tests
```

The GGUFs are the coordinator's pinned artifacts
(`integration-tests/fixtures/decision/artifacts.json`), downloaded from their
`url` into `$H/.cache/gguf/` and checked against their `sha256`. `lock.sh`
relocks `requirements.in`; a relock changes the lock's digest, so
`provenance.py` refuses the old observations until they are regenerated.

## Provenance

Every committed file records, or is bound to, what made it:

- **Run logs.** Each producer records its inputs when it starts. A
  `run_reference.py` run log has the pins, the snapshot files it verified, the
  lock, its recipe (`run_reference.py`, `run.sh`), the digest of each request
  it read, the digest of each observation it wrote, its complete numerics (the
  settings id, precision, threads, set size, deterministic algorithms, mode and
  the parity gate) and the package versions, CPU capability and hardware it
  ran on. `synthetic_median.py --run-log` records the same inputs and versions
  over its weight-free subset.
- **Outputs.** `crosscheck_server.py` and `compare_observations.py --json` write
  their own sha256 and the digests of the observations they compared into
  their output.
- **`provenance.py --update`** carries those identities into `provenance.json`
  and never substitutes what is on disk for what a run recorded. It refuses
  (exit 1, nothing written) if:
  - any recorded pin, snapshot file, lock or recipe source differs from the
    current one;
  - a run's numerics are not its role's complete settings: a log labelled
    `hf-cpu-f32-t1` that ran in bfloat16, at 4 threads, packed or without the
    gate is refused, as is an observation or packed run without the gate;
  - a run imported other package versions than the lock pins, or another
    Python than `run.sh`'s, or ran with other versions, CPU capability or
    hardware than the observation run;
  - a committed observation is not the bytes its run wrote, or came from
    another request than its case's;
  - an observation or packed run has no passing parity proof;
  - a comparison or cross-check was computed from other observations or by
    another version of its script;
  - `make_cases.py --check` fails: the cases must be byte for byte what it
    generates, valid against `../schema.json`, with `artifact` and `settings`
    ids from `../artifacts.json`.
- **`provenance.py --check`** needs no run logs. It recomputes every file
  digest and the pins, lock and recipe digests, including `make_cases.py` and
  `provenance.py` itself (recorded at `--update`, as the writer). It also
  re-binds observations to their cases' requests and settings and cross-checks
  to observations, checks the recorded parity proofs, holds the published
  settings to `SETTINGS`, every published run's numerics to its role's and the
  published versions to the lock, and reruns `make_cases.py --check`. A change
  to any recipe source, such as the float32 hooks, fails it until the fixtures
  are regenerated.
- **Tests.** `tests/test_provenance.py` holds these refusals (standard library:
  `python3 -B -m unittest discover -s tools/conformance/decision_hf/tests -p test_provenance.py`).

## G1 on this hardware

G1 is feasible here. Two conclusions shaped the plan: the first is now contract
C10, the second is the method P11 applies.

- **The reference runs.** The `set_size = 1` float32 forward fits the 8 GiB
  machine and is bit-reproducible at one thread. Without the parity proof (the
  run first committed, at `aea334e`), a question takes a median of 6.1 s (5.2
  to 9.4 s for 29 to 234 prompt tokens), 28.5 s at 1,064 tokens and 86 s at
  3,605, near the 4,096-token budget of `cpu-t4-fa-off`. The proof the
  committed runs carry triples that: a median of 18.6 s a case, and 224 s for
  the 3,605-token case, at a peak of 7.1 GiB resident. Four threads are about
  1.9× faster but not bit-identical to one, so fixtures keep one thread. A full
  float32 load does not fit; the parity proof stands in for it.
- **The pilot cases agree.** The cross-checks were first summarized on 24
  pilot cases (26 questions): `hf-*` without `hf-diff-*`, a selection by name
  that is historical, not the G1 allowlist (P11's, contract C16; below). On
  them the stock `llama-server` (v0.8.0, `cpu-t4-fa-off`) and the reference
  tokenize every prompt to the same length and agree on every answer,
  including the 3,605-token case. The largest gap (difference of max-pooled
  logit differences) is 0.0033 for `d1-2b-f16` and 0.027 for `d1-2b-bf16`. The
  largest probability difference is 0.00049 and 0.0042. Flash attention on
  leaves BF16 in the same range (0.031); no F16 run with flash attention on
  was made here (C16 holds `cpu-t4-fa-on` to its cap as a hypothesis under
  test, not a measured bound). `d1-2b-q4km` drifts by up to 0.87 in
  gap and 0.097 in probability, still with no answer change: G3's report, not
  G1's. With C16's two `hf-diff-*` cases added (26 cases, 32 questions),
  `d1-2b-f16` still tokenizes alike and agrees on every answer; its largest
  gap becomes 0.0036 (`hf-diff-packing`) and its largest probability
  difference stays 0.00049 (BF16: 0.027 and 0.0042). On this CPU the server
  takes about 2 s a short question but 2 to 3 minutes for the 3,605-token case
  on BF16 and F16 (124 to 173 s across two runs; 83 to 109 s on Q4_K_M). That
  matters for the measured budgets of P2 and G3.
- **Prompt differences stand out, against F16.** The cases whose prompts
  differ, where both sides answer, show gaps from 0.053 (single against double
  quotes in object instructions) to 7.9, and different token counts wherever
  the prompt length changes. 0.053 is about 15 times F16's noise (0.0036) but
  only twice BF16's.
- **`d1-2b-f16` is the G1 artifact (contract C10).** ggml rounds a matmul's
  activations to the weight type: bf16 (8 significant bits) for the BF16 GGUF,
  f16 (11) for the F16 GGUF. F16 therefore tracks the float32 reference about
  8× more tightly. C16 (quoted below) fixes the cap at C = 0.020 logits for
  `cpu-t4-fa-off` and `cpu-t4-fa-on` alike, with a bound B_s per setting from
  its headroom rule. On this folder's gap metric, which is not C16's AGE,
  every prompt difference measured here (0.053 and up) lies above C, and the
  BF16 GGUF's noise (0.027) does too. Every case here names `d1-2b-f16` as its
  `artifact`, and the BF16 and Q4_K_M cross-checks are reports.
- **For P11: the comparison method.**
  - What G1 accepts is contract C16, verbatim (framework-v2.md):

    > C16. **G1 acceptance contract (after Codex rb2, `review-p11-plan.md`; P11 owns the implementation).** G1's numeric domain is P11's **explicit semantic allowlist** with verified exclusion kinds, replacing C10's blanket `hf-diff-` prefix exclusion (`hf-diff-packing` from `observations/` with `set_size = 1`, and `hf-diff-duplicate-keys` as reference-to-reference evidence, are included; xybrid still refuses duplicate keys at parse). Metric: anchored gap error (AGE) over all offered groups, gated `AGE ≤ B_s` per setting on its own; near-tie certificates are necessary conditions only. Cap C = 0.020 logits for both `cpu-t4-fa-off` and `cpu-t4-fa-on` as a predeclared quality requirement (a hypothesis under test, not a measured FA-on bound); clipped headroom rule `M_s ≤ C/2`, `B_s = min(C, max(0.005, ⌈3·M_s⌉₀.₀₀₁))`, B/M ≥ 2× recorded. A failing FA-on result is a blocking failure for the product default (no omission, no silent switch to FA-off, no borrowed Q4 allowance); a changed target is a new experiment. Held-out: recipe + literal seed committed before representative observations, bytes generated after the freeze, structural strata (36-option fallback, 63/64/65-token FA tiling boundary, near-default-budget length); a targeted regression suite, not a statistical certification. G0 has no automatic threshold exemptions.

  - P11's allowlist is 26 cases, 32 question observations and 26 distinct
    token sequences: the 24 cases not named `hf-diff-`, plus
    `hf-diff-packing` (5 questions, read from `observations/`; `packed/`
    holds the difference) and `hf-diff-duplicate-keys` (both references keep
    the last value). Its 20 exclusions each have a verified kind:
    `prompt_differs` ×9, `upstream_refuses` ×7, `hf_refuses` ×3 and
    `hf_no_groups` ×1 (`hf-diff-choice-multiple`). This folder's
    observations and F16 cross-check agree with every one. The names `hf-*`
    and `hf-diff-*` say why a case was written, not whether it is in the
    allowlist; the 24-case pilot subset in `hf/crosscheck/`,
    `provenance.json` and `DIFFERENCES.md` section 6 is historical, a summary
    rather than the allowlist.
  - Compare logits or logit gaps, not log-probabilities or probabilities at
    the publisher temperature. The temperatures differ by 3.0e-7 relative,
    enough to flip a median at a boundary (`hf/synthetic/median.json`), and
    the reference reads log-probabilities where upstream reads logits. This
    folder's "gap" (the difference of max-pooled logit differences) is not
    C16's AGE: its numbers are evidence, not a B_s.
  - Report discrete winner and median disagreements separately from the
    numeric bounds.
  - Include the threshold-sensitive cases (`hf-choice-ambiguous`,
    `hf-score-borderline`, `hf-choice-27`); held-out cases follow C16's
    freeze (recipe and seed committed before the observations).

The measurements are the cross-check files under `hf/crosscheck/`, summarized
in `DIFFERENCES.md` section 6. They are a black-box check through the stock
server. G1 proper compares the upstream oracle's `raw_scores` with
`hf.evaluate.raw_scores`, case by case, under the same settings.

## Files

| File | Role |
|---|---|
| `pins.json` | The snapshot: repo, revision, every file's size and sha256 |
| `download.sh` | Fetches and checks the pinned snapshot |
| `requirements.in`, `requirements.txt`, `lock.sh` | The environment: direct pins, the full lock (Linux x86_64, CPU torch), how to relock it |
| `run.sh` | Runs `run_reference.py`, `synthetic_median.py` or the tests in that environment |
| `run_reference.py` | The runner |
| `make_cases.py` | Writes `hf/cases/` (stdlib) |
| `synthetic_median.py` | The score-median cases of `DIFFERENCES.md` 1.3 and 1.5 |
| `compare_observations.py` | Compares two observation sets (stdlib) |
| `crosscheck_server.py` | Sends cases to a running `llama-server` and compares (stdlib) |
| `provenance.py` | Writes and checks `hf/provenance.json` (stdlib) |
| `tests/` | The parity gate's tests (`test_parity.py`: torch, no weights) and the provenance refusals (`test_provenance.py`: stdlib); `run.sh tests` runs both |
| `clean.sh`, `.gitignore` | The folder's outputs, `.cache/` (and any bytecode), ignored by git and removed by `clean.sh` |
| `DIFFERENCES.md` | The reference-difference matrix |

Everything this folder produces goes to `.cache/` here (snapshot, module
cache, GGUFs, llama.cpp build, run logs) or to
`integration-tests/fixtures/decision/hf/`. `run.sh` runs Python with `-B`, so
no bytecode lands next to the scripts. `clean.sh --apply` (or the root
`clean.sh`) removes `.cache/`.
