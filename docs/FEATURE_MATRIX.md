# Feature Matrix

This document provides a comprehensive reference for all feature flags, platform presets, and valid combinations across the xybrid crate hierarchy.

## Table of Contents

1. [xybrid-core Feature Flags](#xybrid-core-feature-flags)
2. [xybrid-sdk Feature Flags](#xybrid-sdk-feature-flags)
3. [xybrid-cli Feature Flags](#xybrid-cli-feature-flags)
4. [Platform Presets](#platform-presets)
5. [Feature-Gated Types and Modules](#feature-gated-types-and-modules)
6. [Invalid Feature Combinations](#invalid-feature-combinations)
7. [Release Gates](#release-gates)
8. [ORT Loading Strategy](#ort-loading-strategy)
9. [xtask Commands](#xtask-commands)
10. [Build Architecture](#build-architecture)

---

## xybrid-core Feature Flags

| Feature | Description | Enables |
|---------|-------------|---------|
| **default** | Default features | `ort-download` (llama.cpp opted into via `llm-llamacpp` or platform preset) |
| **ort-download** | Download prebuilt ONNX Runtime binaries | `ort/download-binaries`, `ort/tls-native` |
| **ort-dynamic** | Load ONNX Runtime .so at runtime | `ort/load-dynamic` |
| **ort-coreml** | Apple Neural Engine acceleration | `ort/coreml` |
| **candle** | Pure Rust ML framework — the SafeTensors Whisper path. **Opt-in only**: no `platform-*` preset enables it (`asr-whispercpp` / whisper.cpp is the shipped ASR backend). Android-compatible | `candle-core`, `candle-nn`, `candle-transformers`, `safetensors`, `byteorder`, `num-traits`, `rayon` |
| **candle-hub** | Candle + HuggingFace Hub download support | `candle`, `hf-hub` (requires OpenSSL — **not for Android**) |
| **candle-metal** | Candle with Metal GPU acceleration | `candle`, `candle-core/metal`, `candle-nn/metal` |
| **candle-cuda** | Candle with CUDA GPU acceleration | `candle`, `candle-core/cuda` |
| **llm-mistral** | mistral.rs LLM backend (CPU) | `mistralrs` |
| **llm-mistral-metal** | mistral.rs with Metal acceleration | `llm-mistral`, `mistralrs/metal` |
| **llm-mistral-cuda** | mistral.rs with CUDA acceleration | `llm-mistral`, `mistralrs/cuda` |
| **vision** | Image envelope primitives and image preprocessing | *(no additional dependencies; uses the always-present `image` crate)* |
| **llm-llamacpp** | llama.cpp backend (cmake build + link) | `llama-cpp-sys/bindings`, `xybrid-llama/bindings` |
| **llm-llamacpp-vision** | Native llama multimodal (`mmproj` / `mtmd`) backend on top of `llm-llamacpp` | `llm-llamacpp`, `xybrid-llama-sys/vision`, `xybrid-llama/vision` |
| **asr-whispercpp** | whisper.cpp speech recognition. Requires `llm-llamacpp` — whisper.cpp links the ggml that llama.cpp builds rather than a second copy | `llm-llamacpp`, `xybrid-whisper/bindings` |

### Notes

- Enabling **`llm-llamacpp`** activates `llama-cpp-sys/bindings` (the cmake
  build of llama.cpp + the `wrapper.cpp` shim) and `xybrid-llama/bindings`
  (safe RAII wrappers). It is **not** enabled by default — it requires cmake,
  a C++ toolchain, and a llama.cpp source clone. All four `platform-*` presets
  on `xybrid-sdk` depend on it. Builds without the feature simply don't expose
  the llama.cpp backend types.
- The 3-layer crate shape:
  `llama-cpp-sys` (raw FFI + cmake build) → `xybrid-llama` (safe wrappers,
  typed errors) → `xybrid-core::runtime_adapter::llama_cpp` (thin adapter).
- `vision` alone enables image envelopes and image preprocessing. Local llama.cpp
  VLM generation requires `llm-llamacpp-vision`, which composes `vision` with
  the llama.cpp backend and links the vendored `mtmd` helpers.

---

## xybrid-sdk Feature Flags

| Feature | Description | Forwards to xybrid-core |
|---------|-------------|-------------------------|
| **default** | No default features | *(none)* |
| **platform-android** | Android preset | `ort-dynamic`, `llm-llamacpp-vision`, `asr-whispercpp` |
| **platform-ios** | iOS preset | `ort-download`, `ort-coreml`, `llm-llamacpp-vision`, `asr-whispercpp` |
| **platform-macos** | macOS preset | `ort-download`, `ort-coreml`, `llm-llamacpp-vision`, `asr-whispercpp` |
| **platform-desktop** | Desktop (Linux/Windows) preset | `ort-download`, `llm-llamacpp-vision`, `asr-whispercpp` |
| **ort-download** | Forward to core | `xybrid-core/ort-download` |
| **ort-dynamic** | Forward to core | `xybrid-core/ort-dynamic` |
| **ort-coreml** | Forward to core | `xybrid-core/ort-coreml` |
| **candle** | Forward to core | `xybrid-core/candle` |
| **candle-hub** | Forward to core | `xybrid-core/candle-hub` |
| **candle-metal** | Forward to core | `xybrid-core/candle-metal` |
| **candle-cuda** | Forward to core | `xybrid-core/candle-cuda` |
| **llm-mistral** | Forward to core | `xybrid-core/llm-mistral` |
| **llm-mistral-metal** | Forward to core | `xybrid-core/llm-mistral-metal` |
| **llm-mistral-cuda** | Forward to core | `xybrid-core/llm-mistral-cuda` |
| **llm-llamacpp** | Forward to core | `xybrid-core/llm-llamacpp` |
| **asr-whispercpp** | Forward to core (pulls `llm-llamacpp` transitively) | `xybrid-core/asr-whispercpp` |
| **vision** | Forward to core | `xybrid-core/vision` |
| **llm-llamacpp-vision** | Forward to core VLM path | `xybrid-core/llm-llamacpp-vision`, `llm-llamacpp`, `vision` |

---

## xybrid-cli Feature Flags

| Feature | Description | Enables |
|---------|-------------|---------|
| **default** | CLI defaults to image-bearing input support so `xybrid run --input-image` works in a `cargo install xybrid-cli` build with no extra flags | `vision` |
| **huggingface** | Direct HuggingFace loading for `xybrid run --huggingface` | `xybrid-sdk/huggingface` |
| **onnx-inspect** | ONNX metadata inspection for `xybrid init` | `xybrid-sdk/onnx-inspect` |
| **vision** | `xybrid run --input-image` and REPL `/image` envelope construction for VLM turns | `xybrid-core/vision`, `xybrid-sdk/vision` |
| **llm-llamacpp-vision** | Native llama.cpp VLM runtime for image turns | `llm-llamacpp`, `xybrid-sdk/llm-llamacpp-vision` |
| **asr-whispercpp** | whisper.cpp speech recognition for `xybrid run --input-audio` | `xybrid-sdk/asr-whispercpp` |
| **platform-android** | Android release preset — forwards to `xybrid-sdk/platform-android` | `ort-dynamic`, `llm-llamacpp-vision`, `asr-whispercpp`, `llm-llamacpp`, `huggingface` |
| **platform-ios** | iOS release preset — forwards to `xybrid-sdk/platform-ios` | `ort-download`, `ort-coreml`, `llm-llamacpp-vision`, `asr-whispercpp`, `llm-llamacpp`, `huggingface` |
| **platform-macos** | macOS release preset — forwards to `xybrid-sdk/platform-macos` | `ort-download`, `ort-coreml`, `llm-llamacpp-vision`, `asr-whispercpp`, `llm-llamacpp`, `huggingface` |
| **platform-desktop** | Linux/Windows release preset — forwards to `xybrid-sdk/platform-desktop` | `ort-download`, `llm-llamacpp-vision`, `asr-whispercpp`, `llm-llamacpp`, `huggingface` |

---

## Platform Presets

Platform presets are the **single source of truth** for platform-specific feature combinations. They are defined in `xybrid-sdk/Cargo.toml` and forwarded through the crate hierarchy.

All four platform presets ship the vision-language llama.cpp path (`llm-llamacpp-vision`, ~0.7 MiB stripped on the Android `.so` proxy / ~1.5 MiB on the iOS static-lib proxy) **and** whisper.cpp speech recognition (`asr-whispercpp`, ~0.2 MiB stripped). VLM and ASR work out of the box — there is nothing extra to compose.

Candle is **not** in any preset. It cost ~1.3 MiB stripped on the same Android proxy and ~1.9 MiB on the Apple shape (which also links `candle-metal`), i.e. 6.5-9.5x its replacement, and on a Pixel 8 reached a first partial in 9871 ms against whisper.cpp's 2724 ms. The `candle*` features stay declared and buildable, so anyone who wants the SafeTensors path can opt in explicitly — they are simply not on by default any more.

| Preset | Target Platform | Core Features Enabled | VLM Default | ASR Default | Rationale |
|--------|-----------------|----------------------|-------------|-------------|-----------|
| **platform-android** | Android (all ABIs) | `ort-dynamic`, `llm-llamacpp-vision`, `asr-whispercpp` | On | whisper.cpp | Dynamic ORT loading for AAR distribution; whisper.cpp for Whisper ASR on the ggml llama.cpp already links (+0.2 MiB stripped); llama.cpp has runtime SIMD detection; mistral.rs causes SIGILL on devices without ARMv8.2-A FP16 |
| **platform-ios** | iOS (arm64, simulator) | `ort-download`, `ort-coreml`, `llm-llamacpp-vision`, `asr-whispercpp` | On | whisper.cpp | Static ORT linking; CoreML for ANE acceleration; Metal for GPU via ggml |
| **platform-macos** | macOS (arm64, x86_64) | `ort-download`, `ort-coreml`, `llm-llamacpp-vision`, `asr-whispercpp` | On | whisper.cpp | Same as iOS - unified Apple platform features |
| **platform-desktop** | Linux, Windows | `ort-download`, `llm-llamacpp-vision`, `asr-whispercpp` | On | whisper.cpp | Static ORT linking; llama.cpp for LLM inference and whisper.cpp for ASR (unified across all platforms) |

> **Note**: The CLI (`xybrid-cli`) adds `huggingface` to all its platform presets so `xybrid run --huggingface` works in release builds. SDK/FFI presets do not include `huggingface` by default — add it individually if needed.

The presets already carry `llm-llamacpp-vision` and `asr-whispercpp`, so a plain preset build is a VLM + ASR build:

```bash
cargo build -p xybrid-cli --features platform-macos
cargo check -p xybrid-sdk --features platform-desktop
```

To opt back into the Candle SafeTensors path on top of a preset:

```bash
cargo check -p xybrid-sdk --features platform-macos,candle-metal
```

### Why llm-mistral is NOT on Android

mistral.rs compiles with `+fp16` target feature on ARM, which requires ARMv8.2-A FP16 extensions. Many Android devices (including popular Samsung and Pixel devices) do not have these extensions, causing **SIGILL** (illegal instruction) crashes at runtime.

llama.cpp uses **runtime SIMD detection** via ggml, making it safe for all Android devices.

---

## Feature-Gated Types and Modules

The following types and modules are conditionally compiled based on feature flags:

### runtime_adapter/mod.rs

| Module | Condition | Description |
|--------|-----------|-------------|
| `coreml` | `target_os = "macos" OR target_os = "ios" OR test` | CoreML runtime adapter |
| `candle` | `feature = "candle"` | Candle (pure Rust) runtime adapter — opt-in, not in any preset |
| `whisper_cpp` | `feature = "asr-whispercpp"` | whisper.cpp ASR runtime adapter (shares llama.cpp's ggml) |
| `llm` | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | Shared LLM types and adapter |
| `mistral` | `feature = "llm-mistral"` | MistralBackend implementation |
| `llama_cpp` | `feature = "llm-llamacpp"` | LlamaCppBackend implementation |

### execution/executor.rs

| Item | Condition | Description |
|------|-----------|-------------|
| `LlmRuntimeAdapter` import | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | LLM adapter import |
| `llm_adapter_cache` field | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | Cached LLM adapter in TemplateExecutor |
| `ExecutionTemplate::Gguf` handling | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | GGUF model execution path |
| `ExecutionTemplate::GgmlWhisper` handling | `feature = "asr-whispercpp"` | GGML Whisper (whisper.cpp) execution path. Fields: `model_file`, `language` (`null` = auto-detect), `audio_ctx` (`0` = no truncation), `translate`. Without the feature the arm errors `GGML Whisper execution requires the 'asr-whispercpp' feature` |
| `ExecutionTemplate::SafeTensors` handling | `feature = "candle"` | Candle SafeTensors path. Without the feature the arm errors `SafeTensors execution requires the 'candle' feature…` and points at a GGML bundle instead |
| `execute_streaming()` full impl | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | Streaming with callback |
| `execute_streaming()` stub | `NOT (llm-mistral OR llm-llamacpp)` | Falls back to regular execution |
| `execute_streaming_with_context()` | Same as above | Streaming with conversation context |
| `execute_llm()` | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | Internal LLM execution |
| `execute_llm_streaming()` | Same as above | Internal streaming execution |

### Re-exports in runtime_adapter/mod.rs

| Export | Condition |
|--------|-----------|
| `ONNXMobileRuntimeAdapter` | `target_os = "android" OR test` |
| `CoreMLRuntimeAdapter` | `target_os = "macos" OR target_os = "ios" OR test` |
| `CandleBackend`, `CandleRuntimeAdapter` | `feature = "candle"` |
| `WhisperCppRuntime` | `feature = "asr-whispercpp"` |
| `ChatMessage`, `GenerationConfig`, `GenerationOutput`, `LlmBackend`, `LlmConfig`, `LlmResult`, `LlmRuntimeAdapter` | `feature = "llm-mistral" OR feature = "llm-llamacpp"` |
| `MistralBackend` | `feature = "llm-mistral"` |
| `LlamaCppBackend` | `feature = "llm-llamacpp"` |
| `llama_log_get_verbosity`, `llama_log_set_verbosity` | `feature = "llm-llamacpp"` |

---

## Invalid Feature Combinations

The following feature combinations are invalid and should produce compile-time errors:

| Combination | Reason | Recommended Alternative |
|-------------|--------|------------------------|
| `llm-mistral` on `target_os = "android"` | SIGILL crash on devices without ARMv8.2-A FP16 | Use `llm-llamacpp` or a platform preset instead |
| `ort-download` AND `ort-dynamic` | Mutually exclusive ORT loading strategies | Choose one based on platform |
| `candle-metal` on non-Apple targets | Metal is Apple-only | Use `candle` (CPU) or `candle-cuda` |
| `candle-cuda` on Apple targets | CUDA not available on Apple | Use `candle-metal` |
| `ort-coreml` on non-Apple targets | CoreML is Apple-only | Use `ort-download` |
| `cargo … --all-features` | Target-dependent: on every supported triple `--all-features` triggers at least one row above (ORT load-mode conflict is universal; the Candle Metal/CUDA + ORT CoreML rows fire on the opposite of their supported target). It also enables the marker-only `llm-mistral*` features whose backing crate is currently commented out of the workspace, so the build fails on the missing `mistralrs` import regardless of target. | Use a [release gate](#release-gates) below; never `--all-features` as a CI gate. |

**Note**: The per-row `compile_error!` guards listed in the table above are **implemented** in [`crates/xybrid-core/src/lib.rs`](../crates/xybrid-core/src/lib.rs). Each conflict fires a typed compile error with a remediation message — see `compile_error!` blocks for `llm-mistral` on Android, `ort-download` vs `ort-dynamic`, `candle-metal` off Apple, `candle-cuda` on Apple, and `ort-coreml` off Apple. The `--all-features` row is enforced through these per-row guards plus the marker-only `llm-mistral*` build break.

---

## Release Gates

These are the canonical feature combinations CI must run to gate a release. Any acceptance criterion that asks for `cargo … --all-features -- -D warnings` is wrong (see [Invalid Feature Combinations](#invalid-feature-combinations) above) — point reviewers here instead.

### Workspace-wide clippy

| Gate | Command | Covers |
|------|---------|--------|
| Default-features workspace clippy | `cargo clippy --workspace -- -D warnings` | Default `ort-download` shape; vendored crates compile cleanly with nothing else enabled. |
| Vision umbrella workspace clippy | `cargo clippy --workspace --features llm-llamacpp-vision --tests --examples -- -D warnings` | The full VLM path through llama.cpp `mtmd`, including vision tests/examples that gate on `llm-llamacpp-vision`. |
| **`--all-features` is forbidden.** | — | See conflict table above. |

### Platform preset matrix

Run on each target host (or in CI matrix jobs). Each row matches what the release workflow actually builds — i.e. the artifact that ships, built the way CI builds it. Mismatching this on a local box (e.g. clippy-ing the host triple instead of cross-compiling) misses real platform-gated bugs.

| Platform | Build host | Canonical gate |
|---------|-----------|---------|
| macOS arm64 / x86_64 | macOS | `cargo clippy --workspace --features platform-macos -- -D warnings` + `cargo test --workspace --features platform-macos` |
| iOS arm64 + simulator | macOS | `bazel build --config=ios //bindings/apple:XybridFFI` (rules_apple xcframework, device + simulator slices). See [`.github/workflows/build-apple.yml`](../.github/workflows/build-apple.yml) for the CI variant. |
| Android arm64-v8a / armeabi-v7a / x86_64 | Any (Bazel downloads its own NDK) | `bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar` (feature-complete 3-ABI AAR). See [`.github/workflows/build-android.yml`](../.github/workflows/build-android.yml) for the CI variant. |
| Desktop Linux x86_64 | Linux | `cargo clippy --workspace --features platform-desktop -- -D warnings` + `cargo test --workspace --features platform-desktop` |
| Desktop Windows x86_64 | Windows | same as Linux desktop |

For a vision-language CI gate on iOS or Android, the canonical xtask commands above must compose with the `llm-llamacpp-vision` feature on `xybrid-uniffi`. The build-apple/build-android workflows already accept this composition — do not invent a new local clippy invocation; use what CI uses.

### Format and diff gates

These run on every host and produce no platform-specific artifacts:

```bash
cargo fmt --all --check
git diff --check          # no whitespace errors
```

### Quick verification on an Apple Silicon dev box

The first three are the canonical local sweep before opening a PR:

```bash
cargo fmt --all --check
cargo clippy --workspace --features llm-llamacpp-vision --tests --examples -- -D warnings
cargo test --workspace --features llm-llamacpp-vision
```

This sweep was green on the `codex/vision-models-support` branch in 2026-05-23 (2m 20s for clippy, well under the timing budget for a pre-PR sanity check). Reproducing this set is the minimum bar before pushing.

---

## ORT Loading Strategy

ONNX Runtime loading varies by platform:

| Platform | Strategy | Feature | Environment Variable | Notes |
|----------|----------|---------|---------------------|-------|
| Desktop (Linux/Windows) | Download prebuilt | `ort-download` | - | Downloads ORT binaries at build time |
| macOS | Download prebuilt | `ort-download` | - | Downloads ORT binaries at build time |
| iOS | XCFramework | `ort-download` | `ORT_IOS_XCFWK_LOCATION` | Must point to `onnxruntime.xcframework` |
| Android | Dynamic loading | `ort-dynamic` | - | Loads `libonnxruntime.so` from AAR at runtime |

### iOS XCFramework Setup

For iOS builds, you must set `ORT_IOS_XCFWK_LOCATION` to point to a prebuilt ONNX Runtime iOS XCFramework:

```bash
# Option 1: Download from VOICEVOX
# https://github.com/VOICEVOX/onnxruntime-builder/releases

# Option 2: Download from HuggingFace
# https://huggingface.co/csukuangfj/ios-onnxruntime

# Option 3: Build from source
# https://onnxruntime.ai/docs/build/ios.html

export ORT_IOS_XCFWK_LOCATION=/path/to/onnxruntime.xcframework
```

---

## xtask Commands

The `xtask` crate provides build automation commands. Run `cargo xtask --help` for full documentation.

| Command | Purpose | Platform | Example |
|---------|---------|----------|---------|
| `setup-test-env` | Download models for integration tests | Any | `cargo xtask setup-test-env` |
| `build-flutter` | Build Flutter native libraries | Varies | `cargo xtask build-flutter --platform macos` |

### xtask to Feature Preset Mapping

| xtask Command | Platform Preset Used | Targets Built |
|---------------|---------------------|---------------|
| `build-flutter --platform ios` | `platform-ios` | aarch64-apple-ios, aarch64-apple-ios-sim |
| `build-flutter --platform android` | `platform-android` | aarch64-linux-android, armv7-linux-androideabi, x86_64-linux-android |
| `build-flutter --platform macos` | `platform-macos` | aarch64-apple-darwin, x86_64-apple-darwin |
| `build-flutter --platform linux` | `platform-desktop` | x86_64-unknown-linux-gnu |
| `build-flutter --platform windows` | `platform-desktop` | x86_64-pc-windows-msvc |

These automatic xtask mappings use the text-only platform presets above. A VLM
build must add `llm-llamacpp-vision` explicitly in the Cargo feature set used
for that build path.

---

## Build Architecture

Xybrid uses a **two-layer build architecture**:

### Layer 1: xtask (Orchestration)

**Location**: `xtask/src/main.rs`

**Responsibilities**:
- Cross-compilation target selection
- Multi-target builds (e.g., all Android ABIs)
- Platform-specific tooling (lipo, xcodebuild, cargo-ndk)
- Packaging and distribution (zip, tar.gz)
- CI/CD integration

**Does NOT handle**:
- Native dependency compilation
- Linker configuration
- CMake invocation

### Layer 2: llama-cpp-sys build.rs (Compilation)

**Location**: `crates/llama-cpp-sys/build.rs`

**Responsibilities**:
- Compiling vendored llama.cpp via CMake
- Detecting Android NDK for CMake toolchain
- Platform-specific linking (Metal, Accelerate, etc.)
- Setting `cargo:rustc-link-lib` and `cargo:rustc-link-search`

**Triggered by**:
- The `llama-cpp-sys/bindings` feature, reached through `xybrid-core/llm-llamacpp`
- Cargo's build process when llm-llamacpp is compiled

## Quick Reference

### Minimal Build (No LLM)

```bash
cargo check -p xybrid-core --no-default-features --features ort-download
```

### macOS Development

```bash
cargo build -p xybrid-core --features "ort-download,ort-coreml,llm-llamacpp"
```

### macOS Vision-Language Development

```bash
cargo build -p xybrid-core --features "ort-download,ort-coreml,llm-llamacpp-vision"
```

### Android Build

```bash
# Bazel downloads its own pinned NDK — no machine setup
bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar
```

### Full Feature Check

```bash
# macOS only (includes Metal features)
cargo check -p xybrid-core --features "ort-download,ort-coreml,candle-metal,llm-llamacpp"
```
