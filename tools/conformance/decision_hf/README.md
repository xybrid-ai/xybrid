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
tools/conformance/decision_hf/run.sh integration-tests/fixtures/decision/hf/cases
# any case files in the decision case schema, e.g. the upstream oracle's:
tools/conformance/decision_hf/run.sh --out /tmp/hf-obs integration-tests/fixtures/decision/cases
```

`download.sh` fetches every file `pins.json` lists from its immutable URL
(`resolve/<revision>`) into `$XYBRID_DECISION_HF_CACHE/snapshots/<revision>/d1-2B/`
(default `target/decision-hf`), keeping only files whose size and sha256 match.
`run.sh` builds the pinned environment (Python 3.12 with `requirements.txt`,
through uv; on Linux it points uv at the PyTorch CPU index), runs offline, and
hands its arguments to `run_reference.py`:

| Option | Default | Meaning |
|---|---|---|
| `CASE...` | | case files, or directories of them |
| `--out DIR` | `integration-tests/fixtures/decision/hf/observations` (`…/packed` with `--set-size default`) | one `<id>.json` observation per case |
| `--run-log FILE` | none | environment, hardware, per-case seconds and tokens, model load time, peak memory |
| `--threads N` | 1 | torch threads; the committed observations use 1 (see Precision) |
| `--precision` | `f32` | `f32`: float32 forward over the bf16 weights. `f32-load`: the model card's `dtype=torch.float32` (about 9 GB). `bf16`: the bfloat16 forward |
| `--set-size` | `1` | `default`: the checkpoint's own set size (16), every question of a request packed after one state |
| `--check-f32-parity` | off | replay every module of every forward as its float32 copy; report bit-identity |

It reads only `id` and `request_text` from a case, so it takes the upstream
oracle's cases and this folder's alike. A run of the 46 committed cases takes
about 7 minutes on 4 vCPUs at one thread: most questions take 5 to 9 s, and
the 3,605-token case takes 86 s.

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

That this is a float32 load's forward is checked, not assumed.
`--check-f32-parity` records each hooked module's inputs and output during
real forwards. It then replays the call on a float32 deep copy of the module,
the parameters a float32 load holds, and requires a bit-identical output and
float32 inputs. On six cases (162 module calls) all 162 were bit-identical,
with no non-float32 input. The modules chain, so the whole forward is a
float32 load's. The numbers are in `hf/provenance.json` under `verification`.

Determinism:

- **Repeat runs.** Bit-identical at one thread.
- **Thread count.** Four threads are not bit-identical to one (MKL's
  reduction order): logits move by up to 5.2e-6 and probabilities by up to
  7.3e-7, with no answer changes. The settings id pins one thread, and this is
  the reference's own floor.
- **bf16 forward.** It moves logits by up to 0.11 and probabilities by up to
  0.010 on the shared-domain cases, with no answer changes.

Settings ids name these runs (the coordinator's `cpu-t4-fa-off` and
`cpu-t4-fa-on` name llama.cpp contexts):

| id | Meaning |
|---|---|
| `hf-cpu-f32-t1` | CPU, 1 thread, float32 forward as above, `set_size = 1`, batch 1, no padding, deterministic algorithms, eval and inference mode |
| `hf-cpu-f32-t1-packed` | the same at the checkpoint's own set size (16): a request's questions packed after one state |

## Regenerate the committed fixtures

Only when the pins, the environment, the runner or a case changes. Building
`llama-server` (for the cross-checks) takes about 15 minutes; everything else
about 40.

```bash
H=tools/conformance/decision_hf FIX=integration-tests/fixtures/decision/hf RUNS=target/decision-hf/runs
python3 $H/make_cases.py                                         # cases/ (stdlib)
$H/run.sh --run-log $RUNS/observations.run.json $FIX/cases       # observations/
$H/run.sh --set-size default --run-log $RUNS/packed.run.json \
    $FIX/cases/hf-diff-packing.json $FIX/cases/hf-card-three.json  # packed/
$H/run.sh synthetic_median.py                                    # synthetic/median.json
# verification runs, written under $RUNS, compared with compare_observations.py --json:
#   --check-f32-parity (parity.run.json, repeat.compare.json), --threads 4 (threads.compare.json),
#   --precision bf16 (bf16.compare.json), and observations/ vs packed/ (packed.compare.json)
cmake -S vendor/llama-cpp -B target/decision-hf/llama-build -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build target/decision-hf/llama-build -j 3 --target llama-server
# for each GGUF and settings (crosscheck/<artifact>.<settings>.json):
target/decision-hf/llama-build/bin/llama-server -m <gguf> -t 4 -c 4096 -b 4096 -ub 4096 -fa off -np 1 --port 8089 &
python3 $H/crosscheck_server.py --url http://127.0.0.1:8089 --hf $FIX/observations \
    --temperature 1.3373980522155762 --label '<what ran>' --out $FIX/crosscheck/<artifact>.<settings>.json <cases>
python3 $H/provenance.py --update --runs $RUNS                   # provenance.json
python3 $H/provenance.py --check
```

The GGUFs are the coordinator's pinned artifacts
(`integration-tests/fixtures/decision/artifacts.json`), downloaded from their
`url` and checked against their `sha256`. `lock.sh` relocks
`requirements.in`; relocking changes `provenance.json`'s environment digest,
so regenerate after it.

## G1 on this hardware

G1 is feasible here, with two adjustments to the plan.

- **The reference runs.** The `set_size = 1` float32 forward fits the 8 GiB
  machine and is bit-reproducible at one thread. A question takes a median of
  6.1 s (5.2 to 9.4 s for 29 to 234 prompt tokens), 28.5 s at 1,064 tokens and
  86 s at 3,605, near the 4,096-token budget of `cpu-t4-fa-off`. Four threads
  are about 1.9× faster but not bit-identical to one, so fixtures keep one
  thread. A full float32 load does not fit; the parity check stands in for it.
- **The shared domain agrees.** On the 24 shared-domain cases (26 questions),
  the stock `llama-server` (v0.8.0, `cpu-t4-fa-off`) and the reference
  tokenize every prompt to the same length and agree on every answer,
  including the 3,605-token case. The largest gap (difference of max-pooled
  logit differences) is 0.0033 for `d1-2b-f16` and 0.027 for `d1-2b-bf16`. The
  largest probability difference is 0.00049 and 0.0042. Flash attention on
  leaves BF16 in the same range (0.031). `d1-2b-q4km` drifts by up to 0.87 in
  gap and 0.097 in probability, still with no answer change: G3's report, not
  G1's. On this CPU the server takes about 2 s a short question but 124 to
  159 s for the 3,605-token case on BF16 and F16 (83 s on Q4_K_M). That
  matters for the measured budgets of P2 and G3.
- **Prompt differences stand out, against F16.** The difference cases that
  both sides answer show gaps from 0.055 (single against double quotes in
  object instructions) to 7.9, and different token counts wherever the prompt
  length changes. 0.055 is 17 times F16's noise but only twice BF16's.
- **Adjustment 1: use `d1-2b-f16` as the G1 artifact on CPU.** ggml rounds a
  matmul's activations to the weight type: bf16 (8 significant bits) for the
  BF16 GGUF, f16 (11) for the F16 GGUF. F16 therefore tracks the float32
  reference about 8× more tightly, and a cap of a few times its noise (around
  0.01 to 0.02 in gap) still catches every prompt difference measured here.
  BF16 bounds would have to sit near the smallest prompt-induced gap.
- **Adjustment 2: compare logit gaps or logits, not log-probabilities or
  probabilities at the publisher temperature.** The temperatures differ by
  3.0e-7 relative, which is enough to flip a median at a boundary
  (`hf/synthetic/median.json`), and the reference reads log-probabilities
  where upstream reads logits. Leave the `hf-diff-*` cases out of the numeric
  comparison, since their prompts differ by design. Report winner and median
  disagreements separately. Include the threshold-sensitive cases
  (`hf-choice-ambiguous`, `hf-score-borderline`, `hf-choice-27`) and cases the
  bounds were not fitted on.

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
| `run.sh` | Runs `run_reference.py` (or `synthetic_median.py`) in that environment |
| `run_reference.py` | The runner |
| `make_cases.py` | Writes `hf/cases/` (stdlib) |
| `synthetic_median.py` | The score-median cases of `DIFFERENCES.md` 1.3 and 1.5 |
| `compare_observations.py` | Compares two observation sets (stdlib) |
| `crosscheck_server.py` | Sends cases to a running `llama-server` and compares (stdlib) |
| `provenance.py` | Writes and checks `hf/provenance.json` (stdlib) |
| `DIFFERENCES.md` | The reference-difference matrix |

Every output goes to `target/decision-hf/` (snapshot, module cache, runs,
llama.cpp build) or to `integration-tests/fixtures/decision/hf/`, and `run.sh`
writes no bytecode here. The root `clean.sh` owns `target/`, so this folder
needs no `clean.sh` of its own.
