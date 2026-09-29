# Choice-scoring conformance tooling

Reproduces every input of the choice-scoring conformance suite: the
CUA-S1-FORMS ONNX export, the Qwen3.5-0.8B GGUF conversions, the reference
goldens and the fake ONNX scorers. The committed results live in
[`integration-tests/fixtures/choice/`](../../integration-tests/fixtures/choice),
pinned by `provenance.json` there.

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

The staged directories hold only the artifacts, with no `model_metadata.json`
until the scorer template exists. Gate a test on the artifact file itself:
`integration-tests`' `model_available` requires the metadata file and would
report these models missing.

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
  `-ffp-contract=off`. Without that flag GCC and clang fuse multiply-adds
  differently and produce different Q4_K_M bytes; with it, Apple clang 21
  (macOS arm64) and GCC 13.3 (Linux arm64 and x86_64) agree, at any thread
  count.

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
| `forms` | `export_forms_onnx.py`, `gen_forms_goldens.py`, `gen_fake_scorers.py` |
| `labels` | `gen_label_goldens.py` |
| `convert` | `convert_gguf.sh` (llama.cpp's own converter pins) |

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
| `fake/*.onnx` | Five-input fake scorers: dynamic, static with a NaN-masked tail, an N-1 output, and a float64 output |

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

Padding is not exempt: on x86_64 it alone exceeds 1e-5 on logits. The G1/G2
bounds in `provenance.json`'s `gates` are the plan's starting values
(`"status": "provisional"`); the runtime conformance tests measure and freeze
them.
