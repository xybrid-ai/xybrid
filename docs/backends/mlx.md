# MLX Backend

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning on Apple Silicon. Xybrid uses MLX as a third LLM and embedding backend on Apple Silicon Macs, complementing the always-available llama.cpp path. iOS builds carry the non-linking `llm-mlx` tier today; the real `llm-mlx-runtime` path is not treated as iOS-ready until upstream MLX produces a Metal-enabled iOS xcframework slice.

This document covers when MLX runs, how to opt in or out per pipeline, and how to troubleshoot common issues.

## When MLX is used

MLX is selected automatically for local SafeTensors LLM bundles when the metadata uses `backend: auto` (or omits `backend`) and the `execution_template.architecture` is one of the MLX text-generation families (`qwen3`, `gemma4`, `lfm2`, `lfm`, `lfm3`). Local embedding metadata with an embedding task and a supported BERT/nomic-BERT architecture follows the same automatic MLX path. Explicit `backend: mlx` still hard-requires MLX.

The lower-level runtime backend selector (`xybrid_core::runtime_adapter::selector`) also drives automatic SDK and CLI registry loading for backend-selectable LLM and embedding models. That selector can choose MLX when **all** of the following are true:

1. The build includes the `llm-mlx-runtime` feature.
2. The host is Apple Silicon (`aarch64-apple-darwin`; iOS is still staged behind a Metal-enabled upstream slice).
3. The Metal runtime probe succeeds at process startup (a reachable Metal device and a functional `mlx.xcframework`).
4. The registry model detail exposes a `format: safetensors` variant for the requested LLM or embedding model.

If any of these are false, LLM registry loaders can fall through to `llm-llamacpp` and request a `format: gguf` variant when the registry advertises one; embedding registry loaders keep the registry default, typically ONNX, instead of requesting GGUF because the local llama.cpp adapter does not currently produce embedding envelopes. The current `TemplateExecutor` path wires local SafeTensors MLX bundle routing. Explicit SDK/CLI backend overrides for registry LLM models still ask the registry for the matching artifact format (`mlx` -> `safetensors`, `llamacpp` -> `gguf`) before loading, and unavailable explicit backends fail with the selector's platform-specific error. Known registry embedding models reject explicit non-MLX local backends until a non-MLX embedding runtime exists. Outside Apple Silicon macOS, including current iOS builds, the non-linking `llm-mlx` tier can compile for metadata and selector checks, but `llm-mlx-runtime` is a compile-time error because the real runtime is validated only for `aarch64-apple-darwin`.

### Model coverage

MLX only accelerates models with a registered MLX variant. The current implementation supports:

| Architecture | Status | Notes |
|--------------|--------|-------|
| Qwen 3 | 🟢 Runtime | SafeTensors load, persistent runtime weight residency, per-layer K/V cache append/read for incremental decode, chat formatting, tokenization, sampling, stop handling, GPU-default dispatch with CPU fallback, telemetry, and streaming callback path are wired. |
| Gemma 4 (2B / 4B) | 🟡 Partial runtime | SafeTensors load, persistent runtime weight residency, per-layer K/V cache append/read for incremental decode, Gemma RMSNorm `(1 + weight)`, tanh-GeLU FFN, mixed local/global attention with per-layer head dimensions, tokenization, sampling, stop handling, telemetry, and streaming callback path are wired. Synthetic generation coverage is in-tree; nested `text_config`, `language_model.`-prefixed tensor layouts, `chat_template.jinja` fallback, macro-capable chat-template rendering, and indexed SafeTensors shards are supported. The public full Gemma BF16 bundle passes the env-gated real generation smoke locally when `XYBRID_MLX_GEMMA_DIR` points at the staged indexed SafeTensors shard bundle. Gemma 4 also has exact BF16 MLX/GGUF benchmark rows recorded as informational because current MLX decode throughput is below the parity floor. CI checks the staged fixture manifest and compiles the runtime harness; the real bundle smoke remains env-gated because the payload is too large for the repo. |
| LFM 2 / 2.5 / 3.5 | 🟡 Partial runtime | SafeTensors load, persistent runtime weight residency, attention K/V append/read, short-conv recurrent state, hybrid short-conv/full-attention blocks, SwiGLU FFN, tokenization, sampling, stop handling, telemetry, and streaming callback path are wired. Synthetic generation coverage is in-tree, including the `lfm3` dispatcher shape. Public LFM2 350M and LFM2.5 1.2B Instruct BF16 bundles pass env-gated real generation smokes locally when staged; the LFM2.5 bundle also has exact BF16 MLX/GGUF benchmark rows recorded as informational because current MLX decode throughput is below the parity floor. No public LFM3.5 text-generation SafeTensors fixture was found during the current staging pass, so LFM3 remains synthetic-only evidence. CI checks the staged fixture manifest and compiles the runtime harness; real LFM bundle smoke remains env-gated because the payloads are too large for the repo. |
| BERT / nomic-bert (embeddings) | 🟡 Partial runtime | Canonical BERT and the current `nomic-ai/nomic-embed-text-v1.5` SafeTensors layouts now materialize resident weights and execute encoder-block runtime paths, including both single-file and HuggingFace indexed shard manifests. `TemplateExecutor` routes embedding metadata through the cached `MlxEmbeddingStrategy`; runtime CI runs external-fixture-free synthetic BERT smokes through the core TemplateExecutor integration harness, the SDK `ModelLoader::from_directory(...).load().run(...)` path, and the CLI local-directory path for both supported SafeTensors layouts, so the checked-in tests prove real MLX encoder execution even when no staged Nomic bundle is present. Real Nomic fixture assertions pass locally when `XYBRID_MLX_NOMIC_DIR` points at the staged SafeTensors bundle. CI checks the staged fixture manifest and compiles the runtime harness plus `mlx_embedding` benchmark; measured embedding rows are still env-gated because the payload is too large for the repo. |

LLM models without a SafeTensors MLX variant continue to load through llama.cpp or the registry default; embedding models without a SafeTensors MLX variant keep the registry default. There is no change in behaviour for non-MLX variants.

## Runtime selection

Local `model_metadata.json` routing is transparent once the bundle declares a supported MLX SafeTensors architecture. Existing `TemplateExecutor` / pipeline calls route `backend: mlx` and auto Qwen/Gemma/LFM SafeTensors bundles through `MlxLlmAdapter`; embedding metadata with `backend: mlx`, or `backend: auto` plus an embedding task and a supported BERT/nomic-BERT architecture, routes through `MlxEmbeddingAdapter`. Registry-driven `Xybrid.model(id).load()`, SDK pipeline loading, `xybrid run`, and `xybrid fetch` now use the selector for automatic LLM and embedding registry formats; on Apple Silicon with `llm-mlx-runtime` and a working Metal probe they request `format=safetensors`, LLM fallback requests `format=gguf` when available, and embedding fallback keeps the registry default. `Xybrid.model(id).with_backend(BackendChoice::Mlx).load()` and explicit CLI/pipeline `backend: mlx` overrides hard-require the MLX backend and request `safetensors` before loading. Explicit `llamacpp` / `mistral` overrides for known registry embedding tasks fail before artifact resolution because those adapters do not currently return embedding envelopes.

When the Apple runtime tier is enabled, Qwen, Gemma, and LFM weights are materialized into adapter state during model load and reused across generation calls. Each runtime resets resident generation state per call, writes prompt state during prefill, then forwards only the newest token at the absolute decode position. Qwen and Gemma keep per-layer K/V tensors. LFM keeps K/V tensors for attention layers and the previous `conv_L_cache - 1` `B * x` activations for short-conv layers. Embedding weights are held by the executor's cached `MlxEmbeddingStrategy`, so repeated local embedding calls reuse the loaded BERT/nomic-BERT adapter when the model path and embedding config match. The generation loop explicitly requests MLX's default GPU stream for prefill and decode. If that stream cannot be initialized, it logs `mlx.generate.gpu_stream_unavailable_falling_back_to_cpu` and retries on MLX's default CPU stream instead of aborting generation.

### Real bundle fixture staging

The in-tree MLX LLM runtime tests always run synthetic Qwen/Gemma/LFM bundles, and the MLX embedding integration harness always runs synthetic canonical-BERT TemplateExecutor smokes for both single-file and indexed-shard SafeTensors layouts. The SDK runtime tests build both synthetic canonical-BERT layouts and run them through `ModelLoader::from_directory(...).load().run(...)`; the CLI runtime tests run both layouts through `xybrid run`'s local-directory handler and verify the saved embedding output. Real SafeTensors fixtures are manual/staged today; registry publication is separate from runtime validation and is not required for source-build CI or local Apple Silicon checks. The fixture manifest records the expected env var and files for each staged model, and `integration-tests/download.sh --list` / `--check` surfaces that state, including missing shards referenced by `model.safetensors.index.json`. Runtime loading accepts either a single `model.safetensors` file or a HuggingFace `model.safetensors.index.json` manifest with the referenced `model-*.safetensors` shard files present in the same directory. Registry passthrough fetches use the same contract: `xybrid fetch`, SDK registry loads, and pipeline fetches download the resolved file, every file listed in inline `model_metadata.json`, and every shard referenced by a downloaded SafeTensors index, while rejecting absolute or parent-directory shard paths. The MLX-vs-llama.cpp LLM benchmark and MLX embedding benchmark both honor these staged env vars, so large public fixtures can stay outside the repo checkout.

| Model fixture | Runtime test env var | Required files |
|---------------|----------------------|----------------|
| `qwen3-4b-mlx` | `XYBRID_MLX_QWEN_4B_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors.index.json`, referenced `model-*.safetensors` shards |
| `gemma4-2b` | `XYBRID_MLX_GEMMA_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `model.safetensors.index.json`, referenced `model-*.safetensors` shards |
| `lfm2-350m-bf16` | `XYBRID_MLX_LFM_DIR` | `config.json`, `tokenizer.json`, `model.safetensors` |
| `lfm2.5-1.2b-instruct-mlx` | `XYBRID_MLX_LFM25_DIR` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `model.safetensors.index.json`, referenced `model.safetensors` |
| `nomic-embed-text-v1.5` | `XYBRID_MLX_NOMIC_DIR` | `config.json`, `tokenizer.json`, `model.safetensors` or `model.safetensors.index.json` plus referenced shard files |

Use `./integration-tests/download.sh <fixture-id>` to print the exact local staging hint for a fixture. The script still returns non-zero for staged entries because it does not download them, and `--all` intentionally skips them.

### Forcing a backend per pipeline

To override the automatic selection, set `backend` on a model stage in your pipeline YAML:

```yaml
name: llm-chat
stages:
  - model: qwen3-4b
    backend: mlx          # "auto" | "mlx" | "llamacpp" | "mistral"
```

- `auto` (default) — local SafeTensors LLM or embedding metadata routes to MLX when the architecture is supported; registry LLM and embedding loads use the selector to request SafeTensors on Apple Silicon runtime builds, while only LLM fallbacks request GGUF variants.
- `mlx` — hard-require MLX. The pipeline fails to load on non-runtime targets, including current iOS non-linking builds, or when the Metal probe fails, with a pointed `SelectorError::ExplicitBackendUnavailable` error that names the target.
- `llamacpp` — pin LLM registry loads to llama.cpp even on Apple Silicon (useful for parity testing or when a specific GGUF quantization is preferred). Known registry embedding models reject this override until a llama.cpp embedding path exists.

For one-off CLI runs, use `--backend` with registry, bundle, directory,
HuggingFace, GGUF file, or pipeline inputs:

```bash
xybrid run --directory ./qwen-mlx --backend mlx --input-text "Hello"
xybrid run --config llm.yaml --backend llamacpp --input-text "Hello"
```

To pre-download the same backend-specific registry artifact, pass the backend
to `fetch` as well. Pipeline fetches also honor per-stage `backend:` fields,
with a CLI `--backend` override applying to all fetched device stages:

```bash
xybrid fetch --model qwen3-4b --backend mlx
xybrid fetch llm.yaml
```

The same knob is exposed as a per-call override in the Rust SDK:

```rust
use xybrid_sdk::{Xybrid, BackendChoice};

let model = Xybrid::model("qwen3-4b")
    .with_backend(BackendChoice::Mlx)
    .load()?;
```

For registry LLM models this override requests a format-specific registry
variant before loading (`mlx` -> `safetensors`, `llamacpp` -> `gguf`) and then
pins the in-memory metadata backend. For known registry embedding tasks, MLX is
the only explicit local backend accepted today. If the registry has no
compatible SafeTensors variant, the MLX load fails instead of silently falling
back to GGUF.

For local directories and bundles, explicit backend overrides are also checked
against the artifact format before execution. `--backend mlx` is rejected for
local GGUF metadata, and `--backend llamacpp` / `--backend mistral` are rejected
for local SafeTensors metadata. Use `auto` when you want the metadata and runtime
selector to choose the compatible local path.

## Vendor xcframework

MLX native code lives in `vendor/mlx-apple/mlx.xcframework/`, materialized either by downloading a pinned artifact or by source-building the macOS runtime slice from the pinned SHAs in [`vendor/mlx-apple/UPSTREAM_VERSIONS.txt`](../../vendor/mlx-apple/UPSTREAM_VERSIONS.txt). The full packaging workflow [`build-mlx-xcframework.yml`](../../.github/workflows/build-mlx-xcframework.yml) compiles `ml-explore/mlx` + `ml-explore/mlx-c` into a fat xcframework with `iphoneos-arm64`, `iphonesimulator-arm64`, and `macos-arm64` slices.

The macOS slice is the runtime-critical slice and must include `Resources/mlx.metallib`. Current pinned upstream MLX forces `MLX_BUILD_METAL=OFF` for `CMAKE_SYSTEM_NAME=iOS`, so iOS slices are packaged for link-layout compatibility only and are expected to be CPU/no-Metal until that upstream gate changes. Do not describe iOS MLX runtime readiness from the presence of an iOS slice alone.

For plain CLI and test binaries, `xybrid-mlx` resolves the vendored macOS `mlx.metallib` at build time and creates a `Resources/mlx.metallib` link next to the running binary before the first MLX FFI call. That keeps runtime execution independent of the disposable upstream source-build directory that produced `libmlx-combined.a`.

Downstream consumers should not need CMake, Metal tooling, or a macOS host once a downloadable xcframework artifact pin is available. The download path is `tools/scripts/fetch-mlx-xcframework.sh`, which pulls the matching `mlx-v<version>.xcframework.zip` from the GitHub Release for the pinned commit and verifies its SHA256 against `UPSTREAM_VERSIONS.txt`. The script is idempotent: a matching `.installed-sha256` marker short-circuits the download.

The runtime path is not blocked on a downloadable xcframework artifact. When `release=unpublished` / `sha256=unpublished`, CI builds the macOS arm64 runtime slice directly from the pinned `mlx=` and `mlx-c=` SHAs with `tools/scripts/build-local-mlx-xcframework.sh`, installs it into `vendor/mlx-apple/mlx.xcframework/`, and records `.installed-source-pin`. This source-build fallback is also the local Apple Silicon validation path while no download pin is available.

```sh
# Apple Silicon runtime validation path from source pins
./tools/scripts/build-local-mlx-xcframework.sh

# Download-based setup once release= and sha256= point at an artifact
./tools/scripts/fetch-mlx-xcframework.sh
```

The installed xcframework lives at `vendor/mlx-apple/mlx.xcframework/` and is gitignored. See [`vendor/mlx-apple/README.md`](../../vendor/mlx-apple/README.md) for refresh instructions and the upstream license attribution.

## Feature flags

See [`FEATURE_MATRIX.md`](../FEATURE_MATRIX.md) for the full feature reference. MLX adds two flags:

| Flag | Purpose | Default |
|------|---------|---------|
| `llm-mlx` | Non-linking MLX SafeTensors tier for config parsing, tokenizer/chat-template handling, weight validation, registry selection, and local adapter routing. It compiles on every target; forward-pass calls return a pointed runtime-gate error unless `llm-mlx-runtime` is also enabled on Apple Silicon macOS. | Included in `platform-macos` and `platform-ios` |
| `llm-mlx-runtime` | Real MLX LLM and embedding forward pass via `xybrid-mlx` on Apple Silicon macOS. Requires `mlx.xcframework` on the link path. | Explicit opt-in |

The two-tier split exists so that Linux / Windows / Android / iOS CI jobs can still build with `llm-mlx` in the feature set (for cross-platform config parsing, local routing checks, and registry round-trips) without linking against Metal. `llm-mlx-runtime` is the gate that pulls in `xybrid-mlx/bindings` and hard-requires the macOS arm64 xcframework slice.

On macOS, `platform-macos` alone is not an MLX runtime build. Use
`--features platform-macos,llm-mlx-runtime` when you need actual MLX
SafeTensors execution through the SDK or CLI. From a local clone, populate
`vendor/mlx-apple/mlx.xcframework/` with `./tools/scripts/fetch-mlx-xcframework.sh`
when a download pin is available, or with
`./tools/scripts/build-local-mlx-xcframework.sh` from pinned source SHAs. The
runtime path and CI checks do not require a downloadable xcframework artifact
while the pin remains unpublished. From an external build environment, set
`MLX_XCFRAMEWORK_PATH` to the prebuilt `mlx.xcframework`.

### Disabling MLX in a build

```bash
# Explicitly disable both flags
cargo build --no-default-features --features "ort-download,llm-llamacpp"
```

Or, from a platform preset, use `platform-desktop` which intentionally omits MLX:

```bash
cargo build --features platform-desktop
```

## Troubleshooting

### `mlx_runtime_available` returns `false`

The Metal probe failed at startup. Common causes:

- The build doesn't include `llm-mlx-runtime` — confirm with `cargo build -v 2>&1 | grep llm-mlx`.
- The host is not Apple Silicon macOS — the shipping MLX runtime path requires `aarch64-apple-darwin`. Intel Macs (`x86_64-apple-darwin`) and current iOS non-linking builds fall through to llama.cpp; enabling `llm-mlx-runtime` for iOS is a compile-time error.
- The `mlx.xcframework` is missing or corrupt — rerun `./tools/scripts/fetch-mlx-xcframework.sh` when a download pin is available, or `./tools/scripts/build-local-mlx-xcframework.sh` for the local source-build path. If the SHA256 check fails, delete `vendor/mlx-apple/mlx.xcframework/` and retry the fetch path.
- The Metal device is unreachable — happens in some VM setups and certain CI runners. Check `ioreg -l | grep -i metal` returns a device.

When the probe fails, automatic selection records the fallback decision and proceeds with llama.cpp; there is no crash.

### Metal shader compile cache

MLX caches compiled Metal kernels in `~/Library/Caches/com.apple.metal/` on macOS. A stale cache can cause first-run latency of 2–5 seconds as shaders recompile. To clear it:

```sh
# macOS
rm -rf ~/Library/Caches/com.apple.metal/
```

This is rarely needed in practice — the cache survives app updates and only invalidates on macOS upgrades.

### iOS runtime status

The current xcframework workflow can produce `iphoneos-arm64` and `iphonesimulator-arm64` static-library slices, but upstream MLX disables the Metal backend for `CMAKE_SYSTEM_NAME=iOS`. In local validation that produced `backend/metal/no_metal.cpp`, no `.metallib`, and availability warnings for Accelerate BLAS/LAPACK APIs below iOS 16.4.

Until a Metal-enabled iOS slice is available and verified on device, use llama.cpp GGUF variants for iOS LLM execution. Keep `llm-mlx` in the iOS preset for metadata, registry, and selector parity only; do not enable `llm-mlx-runtime` for iOS release builds.

Once upstream iOS Metal support lands, the iOS gate should be revalidated with a device run, a simulator link check, and memory-pressure tests for Qwen 0.8B before the docs call it ready.

### `ExplicitBackendUnavailable` on a non-Apple host

Pipeline YAML with `backend: mlx` on a Linux / Windows / Android target will fail to load with:

```text
MLX backend requested but not available on this platform (target: linux-x86_64)
```

This is intentional — `backend: mlx` is a hard constraint. Change the YAML to `backend: auto` (the default) or remove the field entirely to fall through to llama.cpp.

### Sharded SafeTensors bundles

MLX loading accepts both single-file `model.safetensors` bundles and HuggingFace indexed shard layouts:

```text
model.safetensors.index.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
```

The index is required because it maps tensor names to shard files. A directory that contains shard files without `model.safetensors.index.json` is rejected before runtime initialisation with:

```text
MLX sharded SafeTensors bundles require `model.safetensors.index.json`: found orphan shard `<path>` in `<dir>`.
```

### Quantized SafeTensors bundles

MLX variants currently require dequantized SafeTensors weights. Bundles with `config.json` quantization metadata, including common 4-bit MLX-LM bundles, are rejected during header validation before Metal initialisation with:

```text
unsupported MLX quantization for `<model_type>`: <bits>-bit/group=<group_size>. <reason>. Register a GGUF fallback variant or republish this MLX variant as dequantized SafeTensors.
```

Use a GGUF llama.cpp variant for quantized distribution until `mlx_fast_quantized_matmul` is wired through `xybrid-mlx` and the architecture forward passes.

## Further reading

- [`vendor/mlx-apple/README.md`](../../vendor/mlx-apple/README.md) — xcframework sourcing and refresh.
- [`FEATURE_MATRIX.md`](../FEATURE_MATRIX.md) — full feature reference and platform presets.
- [`crates/xybrid-mlx/`](../../crates/xybrid-mlx/) — safe Rust wrapper around `mlx-c`.
- [`crates/xybrid-core/src/runtime_adapter/mlx/`](../../crates/xybrid-core/src/runtime_adapter/mlx/) — `MlxLlmAdapter`, `MlxEmbeddingAdapter`, tokenizer, chat template, sampler, generate loop.
- MLX upstream: [ml-explore/mlx](https://github.com/ml-explore/mlx), [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c).
