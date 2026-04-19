# MLX Backend

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning on Apple Silicon. Xybrid uses MLX as a third LLM and embedding backend on macOS and iOS, complementing the always-available llama.cpp path.

This document covers when MLX runs, how to opt in or out per pipeline, and how to troubleshoot common issues.

## When MLX is used

MLX is selected automatically by the runtime backend selector (`xybrid_core::runtime_adapter::selector`) when **all** of the following are true:

1. The build includes the `llm-mlx-runtime` feature (the Apple-platform presets `platform-macos` and `platform-ios` both do).
2. The host is Apple Silicon (`aarch64-apple-darwin` or `aarch64-apple-ios`).
3. The Metal runtime probe succeeds at process startup (a reachable Metal device and a functional `mlx.xcframework`).
4. The resolved model bundle declares an `mlx` variant in `registry.json` **and** the variant is present in the cache.

If any of these are false, the selector falls through to `llm-llamacpp` when a GGUF variant is available. On non-Apple platforms MLX is compiled out entirely — `llm-mlx` is a compile-time error on Android, Linux, and Windows.

### Model coverage

MLX only accelerates models with a registered MLX variant. As of the current release, MLX supports:

| Architecture | Status | Notes |
|--------------|--------|-------|
| Qwen 3 / Qwen 3.5 | ✅ Shipping | End-to-end generate / streaming / chat |
| Gemma 4 (2B / 4B) | 🟡 Skeleton | Config + weight enumeration; forward pass returns `NotImplemented` pending US-014 follow-up |
| LFM 3.5 | 🟡 Skeleton | Hybrid conv+attention skeleton; forward pass pending |
| BERT / nomic-bert (embeddings) | ✅ Shipping | `MlxEmbeddingAdapter` for retrieval / similarity |

Models without an `mlx` variant continue to load through llama.cpp — no change in behaviour.

## Runtime selection

The selector runs at model load time and is transparent to application code. You do not need to change your Flutter, Kotlin, Swift, Unity, or Rust code to opt in — existing `Xybrid.model(id).load()` / pipeline calls pick up MLX automatically on Apple Silicon.

### Forcing a backend per pipeline

To override the automatic selection, set `backend` on a model stage in your pipeline YAML:

```yaml
name: llm-chat
stages:
  - model: qwen3.5-0.8b
    backend: mlx          # "auto" | "mlx" | "llamacpp"
```

- `auto` (default) — runtime selector chooses based on availability and variant.
- `mlx` — hard-require MLX. The pipeline fails to load on non-Apple targets or when the Metal probe fails, with a pointed `SelectorError::ExplicitBackendUnavailable` error that names the target.
- `llamacpp` — pin to llama.cpp even on Apple Silicon (useful for parity testing or when a specific GGUF quantization is preferred).

The same knob is exposed as a per-call override in the Rust SDK:

```rust
use xybrid_sdk::{Xybrid, BackendChoice};

let model = Xybrid::model("qwen3.5-0.8b")
    .with_backend(BackendChoice::Mlx)
    .load()?;
```

## Vendor xcframework

MLX native code lives in a prebuilt `mlx.xcframework` produced by the [`build-mlx-xcframework.yml`](../../.github/workflows/build-mlx-xcframework.yml) workflow. The workflow compiles `ml-explore/mlx` + `ml-explore/mlx-c` from pinned SHAs stored in [`vendor/mlx-apple/UPSTREAM_VERSIONS.txt`](../../vendor/mlx-apple/UPSTREAM_VERSIONS.txt) into a fat xcframework with `iphoneos-arm64`, `iphonesimulator-arm64`, and `macos-arm64` slices.

Downstream consumers do **not** need CMake, Metal tooling, or a macOS host — the xcframework is fetched by `tools/scripts/fetch-mlx-xcframework.sh`, which pulls the matching `mlx-v<version>.xcframework.zip` from the GitHub Release for the pinned commit and verifies its SHA256 against `UPSTREAM_VERSIONS.txt`. The script is idempotent: a matching `.installed-sha256` marker short-circuits the download.

```sh
# One-time setup on a fresh clone (only needed on macOS / CI)
./tools/scripts/fetch-mlx-xcframework.sh
```

The fetched xcframework lives at `vendor/mlx-apple/mlx.xcframework/` and is gitignored. See [`vendor/mlx-apple/README.md`](../../vendor/mlx-apple/README.md) for refresh instructions and the upstream license attribution.

## Feature flags

See [`FEATURE_MATRIX.md`](../FEATURE_MATRIX.md) for the full feature reference. MLX adds two flags:

| Flag | Purpose | Default |
|------|---------|---------|
| `llm-mlx` | Skeleton (config parsing, weight enumeration, registry wiring) — compiles on every target but refuses to run a forward pass. Safe default. | Included in `platform-macos` and `platform-ios` |
| `llm-mlx-runtime` | Real MLX forward pass via `xybrid-mlx`. Requires `mlx.xcframework` on the link path. | Included in `platform-macos` and `platform-ios` |

The two-tier split exists so that Linux / Windows / Android CI jobs can still build with `llm-mlx` in the feature set (for cross-platform config parsing and registry round-trips) without linking against Metal. `llm-mlx-runtime` is the gate that pulls in `xybrid-mlx/bindings` and hard-requires the xcframework.

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
- The host is not Apple Silicon — MLX requires `aarch64-apple-darwin` or `aarch64-apple-ios`. Intel Macs (`x86_64-apple-darwin`) fall through to llama.cpp.
- The `mlx.xcframework` is missing or corrupt — rerun `./tools/scripts/fetch-mlx-xcframework.sh`. If the SHA256 check fails, delete `vendor/mlx-apple/mlx.xcframework/` and retry.
- The Metal device is unreachable — happens in some VM setups and certain CI runners. Check `ioreg -l | grep -i metal` returns a device.

When the probe fails, the selector logs a `tracing::warn!` and proceeds with llama.cpp; there is no crash.

### Metal shader compile cache

MLX caches compiled Metal kernels in `~/Library/Caches/com.apple.metal/` (macOS) or the app's sandbox cache on iOS. A stale cache can cause first-run latency of 2–5 seconds as shaders recompile. To clear it:

```sh
# macOS
rm -rf ~/Library/Caches/com.apple.metal/

# iOS (via Xcode device console)
# Delete the app and reinstall — the sandbox cache is reset on reinstall.
```

This is rarely needed in practice — the cache survives app updates and only invalidates on macOS / iOS upgrades.

### Out-of-memory on iOS

MLX arrays are backed by Metal buffers, which share the app's memory budget. iOS aggressively terminates apps that cross the system memory limit (typically 1–3 GB on older devices). Symptoms:

- `MlxError::OutOfMemory` from `MlxArray::from_slice_f32`.
- App termination with Xcode console reporting "Jetsam" or "EXC_RESOURCE RESOURCE_TYPE_MEMORY".

Mitigations:

- Use smaller models (Qwen 3.5 0.8B instead of 2B) on iOS.
- Reduce `max_seq_len` — the KV cache grows linearly with sequence length × layers. The default 4096 tokens is appropriate for desktop; drop to 1024 or 2048 on iOS.
- Reduce concurrent pipelines. The runtime selector pools one MLX context per loaded model; unloading models (`Xybrid.unload(id)`) releases Metal buffers.

MLX exposes no direct memory-cap knob — budget is enforced at the Metal driver level. For hard-limiting scenarios, prefer llama.cpp with a quantized GGUF (Q4_K_M), which has more predictable memory usage.

### `ExplicitBackendUnavailable` on a non-Apple host

Pipeline YAML with `backend: mlx` on a Linux / Windows / Android target will fail to load with:

```text
MLX backend requested but not available on this platform (target: linux-x86_64)
```

This is intentional — `backend: mlx` is a hard constraint. Change the YAML to `backend: auto` (the default) or remove the field entirely to fall through to llama.cpp.

## Further reading

- [`vendor/mlx-apple/README.md`](../../vendor/mlx-apple/README.md) — xcframework sourcing and refresh.
- [`FEATURE_MATRIX.md`](../FEATURE_MATRIX.md) — full feature reference and platform presets.
- [`crates/xybrid-mlx/`](../../crates/xybrid-mlx/) — safe Rust wrapper around `mlx-c`.
- [`crates/xybrid-core/src/runtime_adapter/mlx/`](../../crates/xybrid-core/src/runtime_adapter/mlx/) — `MlxLlmAdapter`, `MlxEmbeddingAdapter`, tokenizer, chat template, sampler, generate loop.
- MLX upstream: [ml-explore/mlx](https://github.com/ml-explore/mlx), [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c).
