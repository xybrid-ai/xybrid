# MLX Backend

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning on Apple Silicon. Xybrid can run SafeTensors LLM and embedding models on it, next to the llama.cpp backend that runs GGUF.

The MLX backend is a **preview**:

- It runs on **Apple Silicon macOS** only. iOS cannot run MLX (see [iOS](#ios)).
- It is an **opt-in source build** (`llm-mlx-runtime`). No released SDK, CLI or binding artifact contains it yet, and no platform preset enables it.
- It runs **only when you name it**. A registry model loads on MLX when the stage, the loader or the command line says `mlx`; nothing selects it for you. See [How a model reaches MLX](#how-a-model-reaches-mlx).
- It does less than llama.cpp. Read [What MLX does not do](#what-mlx-does-not-do) before choosing it for a model.

## What it runs

| Architecture (`execution_template.architecture`) | Models | Evidence |
|---|---|---|
| `qwen3` | Qwen 3 | CI runs it on synthetic weights and on real ones (Qwen3 0.6B, MLX 4-bit). Full-precision Qwen3 0.6B passes the same tests locally. |
| `gemma4` | Gemma 4 E2B | CI runs it on synthetic weights. Gemma 4 E2B (MLX 4-bit) passes the real-weight tests locally, tool calling included. E4B has the same architecture and has not been run. |
| `lfm2`, `lfm` | LFM2, LFM2.5 | CI runs it on synthetic weights. LFM2 350M (BF16) and LFM2.5 1.2B (MLX 4-bit) pass the real-weight tests locally, tool calling included for LFM2.5. |
| `bert`, `nomic_bert` | BERT, nomic-embed-text v1.5 (embeddings) | CI runs it on synthetic weights. nomic-embed-text v1.5 passes the real-weight tests locally. |

Weights are SafeTensors: one `model.safetensors`, or a Hugging Face `model.safetensors.index.json` with its `model-*.safetensors` shards. LLM bundles can be full precision (F32, F16, BF16) or MLX affine-quantized (the `mlx-community/*-4bit` layout). Embedding bundles must be full precision.

A bundle whose architecture or quantization MLX cannot load fails at load with an error that says so; nothing falls back to another backend.

**LFM3 is not supported.** It is a different network from LFM2 (sliding-window attention layers, sandwich norms), and an `lfm3` bundle is refused with an error that points at llama.cpp, which runs LFM3 from GGUF.

## Getting an MLX build

MLX is behind two Cargo features:

| Feature | What it compiles | Where |
|---|---|---|
| `llm-mlx` | Everything except the calls into MLX: config and weight validation, chat templates, sampling. Running a model returns an error that names `llm-mlx-runtime`. It exists so the adapter can be compiled and tested without Apple Silicon; a build has no other use for it. | Every target. Not part of any preset. |
| `llm-mlx-runtime` | The MLX runtime itself. Links `mlx.xcframework`. Brings `llm-mlx` with it. | `aarch64-apple-darwin` only; a compile error anywhere else. Not part of any preset. |

```sh
# 1. Materialize the MLX xcframework (gitignored) under vendor/mlx-apple/.
./tools/scripts/build-local-mlx-xcframework.sh    # from the pinned sources; needs Xcode and CMake

# 2. Build with the runtime.
cargo build -p xybrid-cli --features platform-macos,llm-mlx-runtime
```

The pins are in [`vendor/mlx-apple/UPSTREAM_VERSIONS.txt`](../../vendor/mlx-apple/UPSTREAM_VERSIONS.txt): MLX v0.32.4 and mlx-c v0.8.0. No prebuilt xcframework is published yet (`release=unpublished`), so `fetch-mlx-xcframework.sh` has nothing to download and the source build is the only path; see [`vendor/mlx-apple/README.md`](../../vendor/mlx-apple/README.md). A build outside this repository sets `MLX_XCFRAMEWORK_PATH` to an `mlx.xcframework`.

### Shipping a binary that links MLX

MLX loads its Metal shader library, `mlx.metallib`, when it first uses the GPU. It looks next to the binary image it is linked into (the executable, or the dynamic library that embeds it): `mlx.metallib` in that directory, then `Resources/mlx.metallib`.

- `cargo run` and `cargo test` from this repository need nothing: the first MLX call links the vendored library into `Resources/` beside the built binary.
- A binary you distribute must ship `mlx.metallib` (from `mlx.xcframework/macos-arm64/Resources/`) at one of those two places, or the user sets `XYBRID_MLX_METALLIB` to a copy of it.

Without one, the first MLX call fails with an error that names the directory it searched.

## How a model reaches MLX

### A local bundle

A directory or bundle whose `model_metadata.json` has a `SafeTensors` execution template and one of the architectures above runs on MLX. `backend: mlx` in the metadata pins it; so does leaving `backend` out or setting `auto`. An embedding bundle without `backend: mlx` must also declare an embedding task, so other SafeTensors BERT models are not taken over.

### A registry model

A registry model loads the registry's default variant, which is GGUF for LLMs, unless you name a backend. That holds on every build, one that can run MLX included: such a build resolves the same artifact as one that cannot, and makes no extra registry request. MLX refuses grammars and images, and a request that works on llama.cpp must not start failing because the build gained a runtime.

Naming `mlx` (below) asks the registry for the model's `safetensors` variant. The registry lists no SafeTensors variants today, so for now MLX runs local bundles only. A SafeTensors variant is cached separately from the default variant, and a copy left by a `backend: mlx` load is not picked up by a later load that names no backend.

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

The Flutter, Swift, Kotlin, Python, Unity and React Native SDKs have no backend override: none of their published native libraries contains the MLX runtime, so the option could only fail there. They get it together with the runtime.

Some pairs can never work and are rejected up front: `llamacpp` or `mistral` with a SafeTensors bundle or a registry embedding model, `mlx` with a GGUF model, and anything but `llamacpp` with a decision model.

## Tool calling

Tool calling works as it does on llama.cpp, through the same code: the request's `tools` are rendered into the prompt by the bundle's own chat template, calls are parsed from the model's output, a streamed turn holds the call back and delivers it parsed on the terminal token, and a tool result goes back with `Envelope::tool_results`. See the [tool calling guide](../content/docs/guides/tool-calling.mdx).

| Family | Call notation | Evidence |
|---|---|---|
| LFM2.5 | `<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>` | LFM2.5 1.2B Instruct (MLX 4-bit): calls the tool and answers from its result, locally. The same call llama.cpp gets from the Q4_K_M GGUF. |
| Gemma 4 | `<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>` | Gemma 4 E2B (MLX 4-bit): calls the tool and answers from its result, locally. |
| Qwen 3 | | Not run with tools. |

CI covers the plumbing on synthetic weights; the two real-weight cycles above run only with the bundles staged (see [Testing](#testing)).

Two things to know:

- **The bundle's template has to list tools.** A bundle whose chat template never mentions `tools` (`mlx-community/LFM2-350M-4bit` is one) is refused for a tool-bearing request, with an error that says so, instead of answering as if no tools had been offered.
- **Small 4-bit models decide differently.** Asked "What is the weather in Paris right now?", LFM2.5 1.2B answered in prose from the MLX 4-bit weights and called the tool from the Q4_K_M GGUF; asked "What's the weather in Paris?", both called it. The two files are different quantizations of the model. If a model is on the edge of calling a tool, say in the system prompt when it should.

## What MLX does not do

A request for something on this list fails with an error; it is never answered without the feature.

- **Grammars and JSON-schema output.** A request with `grammar` is refused. Run it on llama.cpp (`backend: llamacpp`).
- **Images.** MLX runs text models only.
- **Decision models.** They run on the llama.cpp decision runtime.

Other differences from llama.cpp:

- **One MLX operation at a time.** MLX does not tolerate two threads evaluating at once, so loads, generations and embedding runs are serialized across the whole process. Two MLX models never run in parallel; the second waits. A streaming generation holds its turn until it ends, so a token callback must not block on another thread's MLX call (an embedding of the text so far, say): that call is waiting for the stream to finish. Do such work on the callback's own thread, or after the stream.
- **Unloading frees memory when MLX is idle.** A dropped or unloaded model's memory goes back to the system at once, or, if another MLX model is mid-run, when that run finishes.
- **A GPU failure is an error, not a crash.** If Metal rejects the work (the system is out of GPU memory, most often), the load or run that was waiting for it fails with `[METAL] Command buffer execution failed` and Metal's reason. The model stays loaded and the next run starts clean.
- **No prompt reuse across turns.** Every call evaluates its whole prompt again. A long conversation costs more per turn than on llama.cpp, which keeps the shared prefix.
- **Gemma reasoning.** A reasoning model's thinking is split off the answer and returned as `reasoning_content`, as on llama.cpp. Gemma 4 always answers directly, on both backends.

`seed` in the generation config is honored by MLX and makes sampled output reproducible. Without a seed, sampled output differs on every call, as on llama.cpp. Greedy decoding (temperature 0, the default) is deterministic either way.

## Performance

The reason to choose MLX is decode speed on small models; the reasons not to are time to first token and, for Gemma 4, memory. Measured on an Apple M4 Max with each backend's usual 4-bit artifact of the same model (MLX 4-bit against a Q4_K_M GGUF, which are different quantizations), 128 tokens, greedy:

| Model | MLX decode | llama.cpp decode | Time to first token, MLX / llama.cpp | Peak memory, MLX / llama.cpp |
|---|---|---|---|---|
| Qwen 3 0.6B | 620 to 655 tok/s | 160 to 430 tok/s | 31 ms / 3 to 5.5 ms | 0.8 GB / 1.1 GB |
| LFM2.5 1.2B | 500 to 510 tok/s | 255 to 490 tok/s | 51 ms / 47 to 53 ms | 1.5 GB / 1.3 GB |
| Gemma 4 E2B | 104 to 109 tok/s | 54 to 73 tok/s | 135 ms / 13 to 25 ms | 7.0 GB / 3.6 GB |

Read it with these in mind:

- **The ranges are over repeated passes (four, three for Gemma), and the machine was not idle.** MLX's rate did not move between passes; llama.cpp's did, by up to 2.7x, with other GPU work running. Against llama.cpp's best pass MLX is about 1.5x faster on Qwen 3 0.6B and level on LFM2.5 1.2B. For Gemma 4 no ratio is claimed: llama.cpp's warm-up rounds, which the benchmark discards, reached MLX's rate.
- **llama.cpp answers a repeated prompt sooner.** It keeps the prompt it evaluated last time; MLX evaluates the whole prompt again on every call. In a long conversation that gap grows with the history.
- **At full precision the engines are level, or MLX is behind.** On identical BF16 weights Qwen 3 4B measured 1.03x llama.cpp. Gemma 4 and LFM2.5 at BF16 measured far below llama.cpp in May 2026 and have not been re-measured.

Every pass, the method and how to reproduce it are in [`docs/benchmarks/llm-backends-baseline.md`](../benchmarks/llm-backends-baseline.md).

## Testing

The adapter is tested on three levels.

- **Every code change, on Linux:** `check (llm-mlx non-linking, ubuntu-latest)` in `ci.yml` compiles and lints the `llm-mlx` tier and runs its whole core, SDK and CLI test suites: everything but the calls into MLX. A change to shared code that breaks the adapter fails here.
- **A change to the MLX backend, on Apple Silicon:** `check (llm-mlx-runtime, macos)` in `ci.yml` links MLX and runs the adapters, the executor, the SDK and the CLI on synthetic, randomly initialized bundles the tests write themselves. It runs when the `mlx_runtime` lane in `tools/scripts/ci_gate.py` matches, and never for a pull request that does not touch the backend.
- **Every night, on master:** `test-mlx.yml` runs those steps again, plus `Test MLX (integration, real model)`, which downloads two small real bundles (`qwen3-0.6b-4bit-mlx` and `lfm2.5-1.2b-instruct-4bit-mlx`, pinned by revision and checksum) and runs the chat, streaming, cancellation, two-models-at-once, memory-release and tool-calling tests on them, then the bundle through the CLI. That is where the change to shared code that compiles on every tier and breaks MLX at run time shows up.

The MLX-linked steps are one script, `tools/scripts/check-mlx-runtime.sh`, so the pull-request job and the nightly one cannot drift; run it locally to reproduce either. `tools/scripts/run-mlx-real-tests.sh` runs each real-weight test in its own process and then all of them in one, and prints the crash report's backtrace if a process dies. A hosted runner's GPU is virtual and rejects a command buffer now and then; both scripts retry that one failure, by MLX's message, and print Metal's reason each time.

The other real-weight tests are for bundles too large to download in CI. They run only when an environment variable points at a bundle staged on disk, and skip otherwise:

| Fixture (`integration-tests/fixtures/models/models.json`) | Variable | Files |
|---|---|---|
| `qwen3-4b-mlx` (or any Qwen 3 bundle) | `XYBRID_MLX_QWEN_4B_DIR` / `XYBRID_MLX_QWEN_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, weights |
| an MLX 4-bit Qwen 3 bundle | `XYBRID_MLX_QWEN_4BIT_DIR` | same |
| `gemma4-2b` | `XYBRID_MLX_GEMMA_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, weights |
| `gemma-4-e2b-it-4bit-mlx` (3.55 GB, downloadable) | `XYBRID_MLX_GEMMA_4BIT_DIR` | same; the tool-calling tests use it |
| `lfm2-350m-bf16` | `XYBRID_MLX_LFM_DIR` | `config.json`, `tokenizer.json`, weights |
| `lfm2.5-1.2b-instruct-mlx` | `XYBRID_MLX_LFM25_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, weights |
| `lfm2.5-1.2b-instruct-4bit-mlx` (659 MB, downloadable) | `XYBRID_MLX_LFM25_4BIT_DIR` | same; the tool-calling tests use it |
| `nomic-embed-text-v1.5` | `XYBRID_MLX_NOMIC_DIR` | `config.json`, `tokenizer.json`, weights |

```sh
XYBRID_MLX_QWEN_DIR=/path/to/Qwen3-0.6B \
  cargo test -p xybrid-core --no-default-features --features llm-mlx-runtime,llm-llamacpp \
  --test mlx_llm_chat
```

`./integration-tests/download.sh qwen3-0.6b-4bit-mlx` downloads the CI fixture; point `XYBRID_MLX_QWEN_DIR` and `XYBRID_MLX_QWEN_4BIT_DIR` at `integration-tests/fixtures/models/qwen3-0.6b-4bit-mlx`. For a staged fixture, `download.sh <fixture-id>` prints where it comes from and does not download it.

`tools/scripts/capture-golden-tokens.sh --self` records the tokens a staged bundle decodes to at the current commit; `--test mlx_golden_parity` then fails if a later change alters them. Capture before changing decode code.

`tools/scripts/capture-upstream-goldens.py` records what Python mlx-lm decodes for the same prompts, and the ignored `golden_upstream_audit` test compares the two. Streams that part at a near-tie are expected: the Qwen and LFM adapters compute in the checkpoint's own dtype, as mlx-lm does by default, and in bfloat16 each stack's logits are off by a few rounding steps in its own way. The reference is therefore computed in float32, to measure how close the two candidates really were. A difference at the first token, or where that margin is wide, is a numerics bug. Run it when the MLX pin moves. On the current pin, Qwen3 0.6B matches mlx-lm token for token up to a near-tie on both canonical prompts: margins of 0.045 and 0.006 in log-probability for the MLX 4-bit bundle, 0.043 and 0.005 for the BF16 one.

## Troubleshooting

### `MLX backend requested but not available on this platform`

`backend: mlx` is a hard requirement and this build or host cannot meet it. The error lists the backends that are available. The usual causes:

- the build has no `llm-mlx-runtime` (every released artifact, and every `platform-macos` build without the extra feature);
- the host is not Apple Silicon macOS;
- Metal is unreachable, which happens in some virtual machines.

Use `backend: auto`, or build with the runtime.

### `no MLX Metal library found`

MLX could not find `mlx.metallib`. See [Shipping a binary that links MLX](#shipping-a-binary-that-links-mlx).

### `mlx.xcframework not found` at build time

`llm-mlx-runtime` needs the xcframework under `vendor/mlx-apple/`, or `MLX_XCFRAMEWORK_PATH`. Run `./tools/scripts/build-local-mlx-xcframework.sh`.

### `the MLX backend does not support grammar-constrained generation`

See [What MLX does not do](#what-mlx-does-not-do). Load the model with `backend: llamacpp`.

### `this bundle's chat template does not render tool definitions`

The request offers tools and the bundle's chat template has no place for them, so the model could not know they exist. Use a bundle whose template lists tools (the model's own `chat_template.jinja` usually does; some community conversions replace it with a bare one), or drop `tools` from the request.

### `unsupported MLX model architecture`

MLX runs Qwen 3, Gemma 4 and LFM2 / LFM2.5. Anything else, LFM3 included, runs from GGUF on llama.cpp.

### `MLX sharded SafeTensors bundles require model.safetensors.index.json`

The directory has `model-*.safetensors` shards without the index that maps tensors to them. Download the index with the shards.

### `[METAL] Command buffer execution failed`

Metal could not run the work MLX gave it; the rest of the message is Metal's reason. The process and the loaded model survive the error.

- `Insufficient Memory`: the model and its working memory do not fit in what the GPU can use right now. Close other GPU-heavy applications, use a smaller or more heavily quantized bundle, or run the model's GGUF variant on llama.cpp.
- `Caused GPU Timeout Error`: the GPU did not finish the work in time. Virtual GPUs do this under load (the hosted macOS CI runners hit it now and then), and a real one can when it is heavily contended. Run the request again.

### `unsupported quantization`

An embedding bundle is quantized, or an LLM bundle uses a quantization layout other than MLX affine. Use a full-precision bundle, or the model's GGUF variant on llama.cpp.

## iOS

Upstream MLX builds without its Metal backend for iOS, so there is nothing to run a model with. `llm-mlx-runtime` is a compile error for iOS, as for every target but Apple Silicon macOS (`tools/scripts/test-mlx-runtime-gate.sh` keeps it one), and `platform-ios` carries nothing of MLX. Use GGUF models on llama.cpp there.

## Further reading

- [`vendor/mlx-apple/README.md`](../../vendor/mlx-apple/README.md): where the xcframework comes from and how to refresh it.
- [`FEATURE_MATRIX.md`](../FEATURE_MATRIX.md): every feature and platform preset.
- [`crates/xybrid-mlx/`](../../crates/xybrid-mlx/): the safe Rust wrapper over `mlx-c`.
- [`crates/xybrid-core/src/runtime_adapter/mlx/`](../../crates/xybrid-core/src/runtime_adapter/mlx/): the LLM and embedding adapters.
- Upstream: [ml-explore/mlx](https://github.com/ml-explore/mlx), [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c).
