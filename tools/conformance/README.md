# Choice-scoring conformance tooling

Reproduces every input of the choice-scoring conformance suite: the
CUA-S1-FORMS ONNX export, the Qwen3.5-0.8B GGUF conversions, the reference
goldens and the fake ONNX scorers. The committed results live in
[`integration-tests/fixtures/choice/`](../../integration-tests/fixtures/choice),
pinned by `provenance.json` there. Three TTS tools that reuse the same pinned
environment live here too: two that add a duration output to a real model, and
one that writes the committed fake TTS fixtures. See
[TTS duration output](#tts-duration-output).

The scripts need PyTorch, ONNX and Transformers, so they live outside
`tools/scripts`, whose stdlib-only unittest run must never import them. Each
runs in a pinned [`uv`](https://docs.astral.sh/uv/) environment; nothing is
installed into the system Python.

## Prepare the models

```bash
tools/conformance/prepare.sh cua-s1-forms qwen3.5-0.8b-f16-cf qwen3.5-0.8b-q4km-cf
python3 tools/scripts/check_choice_provenance.py --staged cua-s1-forms qwen3.5-0.8b-f16-cf qwen3.5-0.8b-q4km-cf
```

`prepare.sh` is the one entry point for local runs and CI
(`.github/workflows/test-choice-conformance.yml`), and
`integration-tests/download.sh <id>` delegates to it. For each model's
`derived` entry in `integration-tests/fixtures/models/models.json`, it keeps a
staged artifact whose sha256 matches. Otherwise it downloads every input from
its immutable URL (`resolve/<commit>`) into the cache, checks its sha256,
rebuilds the artifact with the pinned tool and environment, and stages it into
`integration-tests/fixtures/models/<id>/` only if its sha256 matches the pin.

It needs `curl`, `jq` and `uv`; the GGUFs also need `cmake` and
`vendor/llama-cpp` at the pinned commit (`git submodule update --init vendor/llama-cpp`).
Inputs, the llama-quantize build and logs go to `$XYBRID_CONFORMANCE_CACHE`
(default `target/choice-conformance`). `--inputs` stages the inputs even when
nothing needs rebuilding, for regenerating goldens.

| Model id | Artifact | Rebuilt by | From |
|---|---|---|---|
| `cua-s1-forms` | `cua-s1-forms.onnx` | `export_forms_onnx.py` | `cua-ai/cua-s1-forms` safetensors pair + `cua_s1` from `trycua/cua` |
| `qwen3.5-0.8b-f16-cf` | `Qwen3.5-0.8B-F16.gguf` | `convert_gguf.sh f16` | `Qwen/Qwen3.5-0.8B` snapshot |
| `qwen3.5-0.8b-q4km-cf` | `Qwen3.5-0.8B-Q4_K_M.gguf` | `convert_gguf.sh q4_k_m` | the F16 GGUF |

`cua-s1-forms` commits its `model_metadata.json` (a `ChoiceScorer`), so the
SDK loads it from `integration-tests/fixtures/models/cua-s1-forms` once the
export is staged; the GGUF directories hold only their artifacts. Gate a test
on the artifact file itself (`testing::model_fixtures::staged_artifact`):
`integration-tests`' `model_available` looks only for the metadata file, so it
reports FORMS present before the export is staged and the GGUFs missing after.

## Why regenerated, not downloaded

Publishing the derived ONNX and GGUF files under an xybrid-ai Hugging Face
repository needs explicit authorization, so they are regenerated from pinned
public inputs, which their licences permit: CUA-S1-FORMS and `trycua/cua` are
MIT, Qwen3.5-0.8B is Apache-2.0 and llama.cpp is MIT (`provenance.json` pins
each licence file). Full weights are never committed. If publication is
authorized later, the `models.json` entries become ordinary `url` entries for
the published `resolve/<commit>` files, with the same sha256.

CI can rebuild instead of download because regeneration is reproducible to the
byte:

- The ONNX export strips exporter metadata (source locations, producer). The
  pinned environment gives the same bytes on macOS arm64 (torch 2.14.0) and
  Linux x86_64 (torch 2.14.0+cpu).
- `convert_hf_to_gguf.py` derives `general.basename`/`finetune`/`size_label`
  from the snapshot's directory name, so `convert_gguf.sh` converts from a view
  named after the upstream repository (`Qwen3.5-0.8B`).
- `llama-quantize` is built CPU-only, without `-march=native` and with
  `-ffp-contract=off` to avoid compiler-dependent fused multiply-adds.
  `provenance.json` records the platforms verified for the current
  conversion.
- The Qwen F16 is reproducible per CPU architecture, not across them: the
  converter folds `ssm_a = -exp(A_log)` with torch's vectorized `exp`, whose
  last bit differs between arm64 and x86_64. The pinned digests come from
  the CI platform (macOS arm64); an x86_64 rebuild differs only in those
  16-element `ssm_a` tensors.

The llama.cpp v0.6.0 migration keeps the Qwen text-trunk fixture by passing
`--no-mtp`, which excludes the optional prediction head the new converter
otherwise exports. The F16 digest changes with new
`qwen35.attention.recurrent_layers` and `tokenizer.ggml.add_eos_token` metadata,
and the Q4_K_M fixture is regenerated from that F16. Reference goldens,
templates and scoring tolerances are preserved.

The v0.7.0 migration adds `tokenizer.ggml.normalizer.nfc = true`, matching the
pinned tokenizer's NFC normalizer. Rebuilt on macOS arm64 with AppleClang 15.0,
both GGUFs gain 32 bytes of metadata while all 320 tensors remain byte-identical
to v0.6.0. Their artifact digests and the Q4_K_M fixture's F16 input digest are
updated together; templates, reference goldens and scoring bounds are preserved.

The v0.8.0 migration adds `tokenizer.ggml.clean_spaces = false`, matching
`clean_up_tokenization_spaces` in the pinned tokenizer configuration. Both
GGUFs gain 64 bytes of metadata. Their artifact digests and the Q4_K_M
fixture's F16 input digest are updated together from macOS arm64 rebuilds,
with the embedded chat templates verified against the committed template.
Reference goldens and scoring bounds are preserved.

## Environments

Each environment is Python 3.12 plus a full lock. `requirements-<env>.in` holds
the direct pins; `lock.sh` resolves it into `requirements-<env>.txt`, which
pins every transitive package too and is what the scripts install.
`provenance.json` pins each lock's sha256, so relock only deliberately, then
regenerate what that environment builds. The locks resolve for macOS arm64;
Linux x86_64 with the CPU-only torch index resolves the same versions, with
torch as the `+cpu` build the `==` pin accepts.

| Environment | Used by |
|---|---|
| `forms` | `export_forms_onnx.py`, `gen_forms_goldens.py`, `gen_fake_scorers.py`, `expose_tts_duration.py`, `verify_tts_duration.py`, `gen_fake_tts.py` |
| `labels` | `gen_label_goldens.py` |
| `convert` | `convert_gguf.sh` (compatible pins for the Qwen fixture converter) |

```bash
uv run --no-project --python 3.12 --with-requirements tools/conformance/requirements-forms.txt python <script>
```

On Linux, add `--index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match`
(or set `UV_INDEX` / `UV_INDEX_STRATEGY`, as CI does) to get CPU-only torch.

## Regenerate the committed fixtures

Only when an input or a case deliberately changes. Stage the inputs first:

```bash
tools/conformance/prepare.sh --inputs cua-s1-forms qwen3.5-0.8b-f16-cf
IN=target/choice-conformance/inputs
forms() { uv run --no-project --python 3.12 --with-requirements tools/conformance/requirements-forms.txt python "$@"; }
labels() { uv run --no-project --python 3.12 --with-requirements tools/conformance/requirements-labels.txt python "$@"; }

forms tools/conformance/gen_fake_scorers.py
forms tools/conformance/gen_forms_goldens.py --cua-src "$IN/cua-s1-forms" \
    --checkpoint "$IN/cua-s1-forms/cua-s1-forms.safetensors"
labels tools/conformance/gen_label_goldens.py --snapshot "$IN/qwen3.5-0.8b-f16-cf"
python3 tools/scripts/check_choice_provenance.py --update
```

Both golden generators refuse to run unless every input matches its pin in
`provenance.json`, and they record the verified list in the golden.
`gen_label_goldens.py` also refuses unless the text model loads with no
missing, unexpected or mismatched weights. `--update` rewrites only the fixture
digest table, and only if everything else then checks out; review the diff
before committing. A new artifact digest goes into both `models.json` and
`provenance.json` by hand, and `--check` fails until they agree.

## What the fixtures hold

| Path | Contents |
|---|---|
| `provenance.json` | Pinned sources (full commits, per-file sha256, licences), environments, artifact digests and recipes, gates, and the digest of every other file here |
| `specs/cua-s1-forms.json` | The standalone `OnnxByteOptionScorer` spec for the export: tensor names, byte fields (offset 1, 224/96 bytes, `Truncate`), `max_choices` 64, fixed choices `check`, `click`, `skip` |
| `cases/forms.json` | FORMS requests, including UTF-8 characters split at bytes 224 and 96, the 64-choice maximum, fixed choices only, an exact tie and a near tie |
| `cases/labels.json` | Label-token requests and their prompt templates, including 26 labels, a system message, non-ASCII text and a near tie |
| `goldens/forms.json` | G0 encoded byte ids and the PyTorch reference logits, scores, best and margin |
| `goldens/labels.json` | G0 rendered prompt, token ids and label ids, and the Transformers reference label logits, logsumexp, scores and `label_mass` |
| `templates/qwen3.5.jinja` | The pinned Qwen3.5-0.8B chat template, byte-identical to the one both GGUFs embed |
| `fake/*.onnx` | Five-input fake scorers: dynamic, static with a NaN-masked tail, an N-1 output, a float64 output, a byte lookup that fails inside ONNX Runtime on any byte id of 64 or more, and an extra output whose length is the first context byte id, which ONNX Runtime warns about |

When `enable_thinking` is omitted, the Qwen3.5-0.8B template ends a generation
prompt with `<think>\n\n</think>\n\n`, unlike the Qwen3.5-4B template, which
ends it with `<think>\n`. The goldens record the 0.8B bytes.

## Export gates

`export_forms_onnx.py` refuses to write the ONNX unless, on the committed cases
plus 200 seeded random ones, ONNX Runtime 1.23.2 (the version `ort-sys` ships)
matches PyTorch on the same fixed-length tensors, and the fixed 224/96-byte
padding matches the upstream collator's batch-longest padding. Each check
bounds candidate scores at 1e-5 and logits at 5e-5.

The logit bound sits just above the measured float32 floor: logits reach |37|,
and PyTorch's fused and unfused attention paths alone disagree by up to 1.3e-5
on these cases, so a 1e-5 logit bound fails on rounding. Measured maxima over
213 cases, also recorded under the artifact's `export_checks` in
`provenance.json`:

| Check | macOS arm64 | Linux x86_64 |
|---|---|---|
| ONNX Runtime vs PyTorch, logits / scores | 2.3e-5 / 1.3e-6 | 1.6e-5 / 2.2e-6 |
| Padding (fixed vs collator), logits / scores | 1.9e-6 / 5e-11 | 1.1e-5 / 1.2e-6 |

Padding is not exempt: on x86_64 it alone exceeds 1e-5 on logits.

## Runtime gates

Each track in `provenance.json`'s `gates` (`forms`, `labels`) has its own
`status`. `provisional` bounds are the plan's starting values; `frozen` bounds
were measured through xybrid's own runtime path, and loosening one needs a
written justification. The runtime test of a frozen track fails if its status
reverts to `provisional`.

`forms` is frozen. Against the PyTorch reference, xybrid's ONNX path reaches
|Δlogit| 1.9e-5 and |Δscore| 4.4e-7 on macOS arm64, and 1.2e-5 and 2.0e-6 on
Linux x86_64, the same at 1, 4 and 8 threads and bit-identical across repeats.
G1 allows 8e-5 and 1e-5: the larger of the export bound and four times the
worst measurement. `labels` stays provisional until the label scorer's runtime
tests freeze it.

One input is outside what these gates prove: an empty context. With every
context position masked, the model averages over all padding positions, and its
position embeddings make that average depend on the padded length (224 in the
export, 1 in the reference collator), so the logits differ by about 1.5. xybrid
refuses an empty context instead.

## Same-artifact harness

`llama_harness/` is a `llama.h`-only C++ program that scores a prompt the way
`xybrid-llama`'s `LlamaContext::score_final_position` does: the same context
settings, a cleared memory, the prompt decoded in chunks of
`llama_n_batch(ctx)` with output on the last token only, and the log-sum-exp
of the final row summed in `f64` in index order. Linked against the same
llama.cpp libraries as the Rust test, it must print the same final row bit for
bit, so a mismatch points at the Rust readout rather than at the model or
llama.cpp.

```bash
tools/conformance/prepare.sh qwen3.5-0.8b-f16-cf
XYBRID_LLAMA_HARNESS="$(tools/conformance/llama_harness/build.sh)" \
  cargo test -p xybrid-llama --features bindings --test choice_logits
```

`build.sh [FEATURES]` builds that test, without running it, with
`xybrid-llama`'s features `FEATURES` (`bindings` by default), then links the
harness against exactly the archives and system libraries cargo reports for
llama.cpp: a downloaded or staged natives slice, or a source build. Pass the
features you test with, such as `build.sh bindings,vision`. It needs `python3`
and writes `target/llama-harness/llama_harness`.

## TTS duration output

The SDK reports per-phoneme timings for an ONNX TTS model that declares a
per-token `duration` output (see
[`TTSAudioEncode`](../../docs/sdk/MODEL_METADATA.md#ttsaudioencode)). Stock
Kokoro-82M computes those durations but keeps them inside its graph.
`expose_tts_duration.py` adds the output and `verify_tts_duration.py` proves
the audio did not change. They work on paths you pass them; nothing they write
is a committed fixture. The third tool, `gen_fake_tts.py`, writes the committed
fixtures the SDK tests load; see [Fake TTS fixtures](#fake-tts-fixtures).

```bash
SRC=path/to/kokoro-82m          # stock model directory: model.onnx, tokens.txt, voices.bin, ...
DST=path/to/kokoro-82m-timed
forms() { uv run --no-project --python 3.12 --with-requirements tools/conformance/requirements-forms.txt python "$@"; }

forms tools/conformance/expose_tts_duration.py "$SRC/model.onnx" "$DST/model.onnx"
forms tools/conformance/verify_tts_duration.py "$SRC/model.onnx" "$DST/model.onnx" "$SRC/tokens.txt"
```

`expose_tts_duration.py` finds the duration tensor by structure, not by name:
the single INT64 `Cast` whose output feeds a `Gather` that feeds the `CumSum`
building the alignment (`/encoder/Cast_output_0` in the Kokoro-82M v1.0
export). It refuses a model that does not have exactly one output, that
already uses the name `duration`, or where that search finds anything other
than one tensor. It appends an `Identity` output named `duration` (INT64,
`[1, sequence_length]`), and after saving it checks that the outputs are the
original one followed by `duration` and that the initializers hash the same as
before. It
also records `xybrid.duration_*` model properties for reference; the SDK does
not read them and finds the output by its name.

`verify_tts_duration.py` runs both models in ONNX Runtime 1.23.2 (the version
xybrid ships) on three Misaki phoneme strings at speeds 0.8, 1.0 and 1.25,
with one thread and with the default thread count, and a fixed seeded style
vector. It builds token ids as the `Phonemize` step does: pad 0, one id per
phoneme character from `tokens.txt`, pad 0. It fails unless `audio` is
byte-identical between the two models and `duration` has shape
`[1, len(tokens)]` with every value at least 1. It feeds inputs by Kokoro's
names (`tokens`, `style`, `speed`).

It also prints the waveform length over `600 × sum(duration)`;
`--frame-samples` changes the 600 (on `expose_tts_duration.py` the same flag
only sets the recorded property). For Kokoro-82M that ratio is exactly 1 at
every speed: one frame is 600 samples, 25 ms at 24 kHz. The SDK does not rely
on it; it scales durations to the waveform it gets.

To use the result, copy the rest of `$SRC` into `$DST` and give its
`model_metadata.json` a new `model_id`, as
[`MODEL_METADATA.md`](../../docs/sdk/MODEL_METADATA.md#ttsaudioencode)
describes.

### Fake TTS fixtures

`gen_fake_tts.py` writes the two tiny models in
[`integration-tests/fixtures/tts/fake/`](../../integration-tests/fixtures/tts/fake)
that `integration-tests/tests/tts_phoneme_timings.rs` runs through the SDK:
`with_duration/` declares a `duration` output before `audio`, and
`without_duration/` is the same graph with `audio` only, so both must return
the same audio. Each directory also holds the `tokens.txt`,
`misaki/us_gold.json` and `model_metadata.json` the test needs to run offline.
Regenerate them only when the fakes deliberately change, in the `forms`
environment:

```bash
forms() { uv run --no-project --python 3.12 --with-requirements tools/conformance/requirements-forms.txt python "$@"; }

forms tools/conformance/gen_fake_tts.py        # --out DIR writes somewhere else
git status --short integration-tests/fixtures/tts/fake
```

The models are built with the ONNX helper API, so their bytes depend on the
pinned `onnx` (1.23.0); another `onnx` version can write different bytes. In
the pinned environment, regenerating reproduces the committed files byte for
byte (checked on macOS arm64). Before writing, the script runs both models in
the pinned ONNX Runtime and checks the durations, the waveform and that the
two models' audio is byte-identical, then prints each file's sha256. Unlike the
choice fakes, these files are not pinned by a `provenance.json`, so review the
diff before committing, and keep the constants `tts_phoneme_timings.rs` copies
from the script (`FRAME_SAMPLES`, the base frame counts) in step with it.
