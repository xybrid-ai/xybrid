# MLX Backend

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning on Apple Silicon. Xybrid can run SafeTensors LLM and embedding models on it, next to the llama.cpp backend that runs GGUF.

The MLX backend is a **preview**:

- It runs on **Apple Silicon macOS** only. iOS builds compile the non-linking tier and cannot run MLX (see [iOS](#ios)).
- It is an **opt-in source build** (`llm-mlx-runtime`). No released SDK, CLI or binding artifact contains it yet.
- It does less than llama.cpp. Read [What MLX does not do](#what-mlx-does-not-do) before choosing it for a model.

## What it runs

| Architecture (`execution_template.architecture`) | Models | Evidence |
|---|---|---|
| `qwen3` | Qwen 3 | CI runs it on synthetic weights and on real ones (Qwen3 0.6B, MLX 4-bit). Full-precision Qwen3 0.6B passes the same tests locally. |
| `gemma4` | Gemma 4 E2B | CI runs it on synthetic weights. Gemma 4 E2B (MLX 4-bit) passes the real-weight tests locally. E4B has the same architecture and has not been run. |
| `lfm2`, `lfm`, `lfm3` | LFM2, LFM2.5 | CI runs it on synthetic weights. LFM2 350M (BF16) and LFM2.5 1.2B (MLX 4-bit) pass the real-weight tests locally. No LFM3 SafeTensors bundle has been run. |
| `bert`, `nomic_bert` | BERT, nomic-embed-text v1.5 (embeddings) | CI runs it on synthetic weights. nomic-embed-text v1.5 passes the real-weight tests locally. |

Weights are SafeTensors: one `model.safetensors`, or a Hugging Face `model.safetensors.index.json` with its `model-*.safetensors` shards. LLM bundles can be full precision (F32, F16, BF16) or MLX affine-quantized (the `mlx-community/*-4bit` layout). Embedding bundles must be full precision.

A bundle whose architecture or quantization MLX cannot load fails at load with an error that says so; nothing falls back to another backend.

## Getting an MLX build

MLX is behind two Cargo features:

| Feature | What it compiles | Where |
|---|---|---|
| `llm-mlx` | Everything except the calls into MLX: backend selection, registry formats, config and weight validation, chat templates, sampling. Running a model returns an error that names `llm-mlx-runtime`. | Every target. Included in `platform-macos` and `platform-ios`. |
| `llm-mlx-runtime` | The MLX runtime itself. Links `mlx.xcframework`. | `aarch64-apple-darwin` only; a compile error anywhere else. Never part of a preset. |

```sh
# 1. Materialize the MLX xcframework (gitignored) under vendor/mlx-apple/.
./tools/scripts/build-local-mlx-xcframework.sh    # from the pinned sources; needs Xcode and CMake

# 2. Build with the runtime.
cargo build -p xybrid-cli --features platform-macos,llm-mlx-runtime
```

The pins are in [`vendor/mlx-apple/UPSTREAM_VERSIONS.txt`](../../vendor/mlx-apple/UPSTREAM_VERSIONS.txt). No prebuilt xcframework is published yet (`release=unpublished`), so `fetch-mlx-xcframework.sh` has nothing to download and the source build is the only path; see [`vendor/mlx-apple/README.md`](../../vendor/mlx-apple/README.md). A build outside this repository sets `MLX_XCFRAMEWORK_PATH` to an `mlx.xcframework`.

### Shipping a binary that links MLX

MLX loads its Metal shader library, `mlx.metallib`, when it first uses the GPU. It looks next to the binary image it is linked into (the executable, or the dynamic library that embeds it): `mlx.metallib` in that directory, then `Resources/mlx.metallib`.

- `cargo run` and `cargo test` from this repository need nothing: the first MLX call links the vendored library into `Resources/` beside the built binary.
- A binary you distribute must ship `mlx.metallib` (from `mlx.xcframework/macos-arm64/Resources/`) at one of those two places, or the user sets `XYBRID_MLX_METALLIB` to a copy of it.

Without one, the first MLX call fails with an error that names the directory it searched.

## How a model reaches MLX

### A local bundle

A directory or bundle whose `model_metadata.json` has a `SafeTensors` execution template and one of the architectures above runs on MLX. `backend: mlx` in the metadata pins it; so does leaving `backend` out or setting `auto`. An embedding bundle without `backend: mlx` must also declare an embedding task, so other SafeTensors BERT models are not taken over.

### A registry model

By default a registry model loads the registry's default variant, which is GGUF for LLMs. That is true on every build that cannot run MLX, and such a build makes no extra registry request.

On a build that can run MLX (Apple Silicon, `llm-mlx-runtime`, Metal available), an LLM or embedding model for which the registry lists a `safetensors` variant loads that variant on MLX instead. The registry lists no SafeTensors variants today, so this changes nothing yet. When it does, a SafeTensors variant is a promise that MLX can load it: there is no fallback to GGUF if it cannot.

A SafeTensors variant is cached separately from the default variant, so the two never shadow each other.

### Choosing the backend yourself

`backend` takes `auto` (the default), `mlx`, `llamacpp` or `mistral`. An explicit backend is a hard requirement: if this build or host cannot provide it, or it cannot run the model's format, the load fails with an error that names the target and the backends that are available. For a registry model, `mlx` asks the registry for the `safetensors` variant and `llamacpp` for the `gguf` one.

Pipeline YAML, per stage:

```yaml
name: llm-chat
stages:
  - model: qwen3.5-0.8b
    backend: llamacpp
```

CLI (`run`, `repl` and `fetch` take `--backend`; on `run --config` it applies to every local stage):

```sh
xybrid run --directory ./qwen3-mlx --backend mlx --input-text "Hello"
xybrid run --model qwen3.5-0.8b --backend llamacpp --input-text "Hello"
```

Rust:

```rust
use xybrid_sdk::{BackendChoice, ModelLoader};

let model = ModelLoader::from_directory("./qwen3-mlx")?
    .with_backend(BackendChoice::Mlx)
    .load()?;
```

Flutter:

```dart
final loader = XybridModelLoader.fromDirectory('/path/to/qwen3-mlx');
final model = await loader.load(backend: XybridBackend.mlx);
```

The Swift, Kotlin, Python, Unity and React Native SDKs have no backend override yet.

Some pairs can never work and are rejected up front: `llamacpp` or `mistral` with a SafeTensors bundle or a registry embedding model, `mlx` with a GGUF model, and anything but `llamacpp` with a decision model.

## What MLX does not do

A request for something on this list fails with an error; it is never answered without the feature.

- **Tool calling.** A request with `tools` is refused. Run it on llama.cpp (`backend: llamacpp`).
- **Grammars and JSON-schema output.** A request with `grammar` is refused, for the same reason.
- **Images.** MLX runs text models only.
- **Decision models.** They run on the llama.cpp decision runtime.

Other differences from llama.cpp:

- **One MLX operation at a time.** MLX does not tolerate two threads evaluating at once, so loads, generations and embedding runs are serialized across the whole process. Two MLX models never run in parallel; the second waits. A streaming generation holds its turn until it ends, so a token callback must not block on another thread's MLX call (an embedding of the text so far, say): that call is waiting for the stream to finish. Do such work on the callback's own thread, or after the stream.
- **Unloading frees memory when MLX is idle.** A dropped or unloaded model's memory goes back to the system at once, or, if another MLX model is mid-run, when that run finishes.
- **No prompt reuse across turns.** Every call evaluates its whole prompt again. A long conversation costs more per turn than on llama.cpp, which keeps the shared prefix.
- **Gemma reasoning.** A reasoning model's thinking is split off the answer and returned as `reasoning_content`, as on llama.cpp. Gemma 4 always answers directly, on both backends.

`seed` in the generation config is honored by MLX and makes sampled output reproducible. Without a seed, sampled output differs on every call, as on llama.cpp. Greedy decoding (temperature 0, the default) is deterministic either way.

## Testing

CI runs the MLX runtime on Apple Silicon in two lanes, both only when something that reaches the runtime changed. The non-linking tier is checked on Linux for every code change.

- `check (llm-mlx-runtime, macos)` in `ci.yml` uses synthetic, randomly initialized bundles written by the tests themselves, through the adapters, the executor, the SDK and the CLI.
- `Test MLX (integration, real model)` in `test-mlx.yml` downloads one small real bundle, the `qwen3-0.6b-4bit-mlx` fixture (335 MB, pinned by revision and checksum), and runs the chat, streaming, cancellation, two-models-at-once and memory-release tests on it, then the bundle through the CLI. `tools/scripts/run-mlx-real-tests.sh` runs each test in its own process and then all of them in one, and prints the crash report's backtrace if a process dies.

The other real-weight tests are for bundles too large to download in CI. They run only when an environment variable points at a bundle staged on disk, and skip otherwise:

| Fixture (`integration-tests/fixtures/models/models.json`) | Variable | Files |
|---|---|---|
| `qwen3-4b-mlx` (or any Qwen 3 bundle) | `XYBRID_MLX_QWEN_4B_DIR` / `XYBRID_MLX_QWEN_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, weights |
| an MLX 4-bit Qwen 3 bundle | `XYBRID_MLX_QWEN_4BIT_DIR` | same |
| `gemma4-2b` | `XYBRID_MLX_GEMMA_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, weights |
| `lfm2-350m-bf16` | `XYBRID_MLX_LFM_DIR` | `config.json`, `tokenizer.json`, weights |
| `lfm2.5-1.2b-instruct-mlx` | `XYBRID_MLX_LFM25_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, weights |
| `nomic-embed-text-v1.5` | `XYBRID_MLX_NOMIC_DIR` | `config.json`, `tokenizer.json`, weights |

```sh
XYBRID_MLX_QWEN_DIR=/path/to/Qwen3-0.6B \
  cargo test -p xybrid-core --no-default-features --features llm-mlx-runtime,llm-llamacpp \
  --test mlx_llm_chat
```

`./integration-tests/download.sh qwen3-0.6b-4bit-mlx` downloads the CI fixture; point `XYBRID_MLX_QWEN_DIR` and `XYBRID_MLX_QWEN_4BIT_DIR` at `integration-tests/fixtures/models/qwen3-0.6b-4bit-mlx`. For a staged fixture, `download.sh <fixture-id>` prints where it comes from and does not download it.

`tools/scripts/capture-golden-tokens.sh --self` records the tokens a staged bundle decodes to at the current commit; `--test mlx_golden_parity` then fails if a later change alters them. Capture before changing decode code.

## Troubleshooting

### `MLX backend requested but not available on this platform`

`backend: mlx` is a hard requirement and this build or host cannot meet it. The error lists the backends that are available. The usual causes:

- the build has `llm-mlx` but not `llm-mlx-runtime` (every `platform-macos` build without the extra feature);
- the host is not Apple Silicon macOS;
- Metal is unreachable, which happens in some virtual machines.

Use `backend: auto`, or build with the runtime.

### `no MLX Metal library found`

MLX could not find `mlx.metallib`. See [Shipping a binary that links MLX](#shipping-a-binary-that-links-mlx).

### `mlx.xcframework not found` at build time

`llm-mlx-runtime` needs the xcframework under `vendor/mlx-apple/`, or `MLX_XCFRAMEWORK_PATH`. Run `./tools/scripts/build-local-mlx-xcframework.sh`.

### `the MLX backend does not support tool calling` / `grammar-constrained generation`

See [What MLX does not do](#what-mlx-does-not-do). Load the model with `backend: llamacpp`.

### `MLX sharded SafeTensors bundles require model.safetensors.index.json`

The directory has `model-*.safetensors` shards without the index that maps tensors to them. Download the index with the shards.

### `unsupported quantization`

An embedding bundle is quantized, or an LLM bundle uses a quantization layout other than MLX affine. Use a full-precision bundle, or the model's GGUF variant on llama.cpp.

## iOS

Upstream MLX builds without its Metal backend for iOS, so there is nothing to run a model with. `platform-ios` carries `llm-mlx` so that metadata, selection and registry code compile the same on both Apple platforms; `llm-mlx-runtime` is a compile error for iOS. Use GGUF models on llama.cpp there.

## Further reading

- [`vendor/mlx-apple/README.md`](../../vendor/mlx-apple/README.md): where the xcframework comes from and how to refresh it.
- [`FEATURE_MATRIX.md`](../FEATURE_MATRIX.md): every feature and platform preset.
- [`crates/xybrid-mlx/`](../../crates/xybrid-mlx/): the safe Rust wrapper over `mlx-c`.
- [`crates/xybrid-core/src/runtime_adapter/mlx/`](../../crates/xybrid-core/src/runtime_adapter/mlx/): the LLM and embedding adapters.
- Upstream: [ml-explore/mlx](https://github.com/ml-explore/mlx), [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c).
