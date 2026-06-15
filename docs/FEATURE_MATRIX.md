# Feature Matrix

This document provides a comprehensive reference for all feature flags, platform presets, and valid combinations across the xybrid crate hierarchy.

## Table of Contents

1. [xybrid-core Feature Flags](#xybrid-core-feature-flags)
2. [xybrid-sdk Feature Flags](#xybrid-sdk-feature-flags)
3. [xybrid-ffi Feature Flags](#xybrid-ffi-feature-flags)
4. [xybrid-cli Feature Flags](#xybrid-cli-feature-flags)
5. [Platform Presets](#platform-presets)
6. [Feature-Gated Types and Modules](#feature-gated-types-and-modules)
7. [Invalid Feature Combinations](#invalid-feature-combinations)
8. [Release Gates](#release-gates)
9. [ORT Loading Strategy](#ort-loading-strategy)
10. [xtask Commands](#xtask-commands)
11. [Build Architecture](#build-architecture)

---

## xybrid-core Feature Flags

| Feature | Description | Enables |
|---------|-------------|---------|
| **default** | Default features | `ort-download` (llama.cpp opted into via `llm-llamacpp` or platform preset) |
| **ort-download** | Download prebuilt ONNX Runtime binaries | `ort/download-binaries`, `ort/tls-native` |
| **ort-dynamic** | Load ONNX Runtime .so at runtime | `ort/load-dynamic` |
| **ort-coreml** | Apple Neural Engine acceleration | `ort/coreml` |
| **candle** | Pure Rust ML framework (Whisper) — Android-compatible | `candle-core`, `candle-nn`, `candle-transformers`, `safetensors`, `byteorder`, `num-traits` |
| **candle-hub** | Candle + HuggingFace Hub download support | `candle`, `hf-hub` (requires OpenSSL — **not for Android**) |
| **candle-metal** | Candle with Metal GPU acceleration | `candle`, `candle-core/metal`, `candle-nn/metal` |
| **candle-cuda** | Candle with CUDA GPU acceleration | `candle`, `candle-core/cuda` |
| **llm-mistral** | mistral.rs LLM backend (CPU) | `mistralrs` |
| **llm-mistral-metal** | mistral.rs with Metal acceleration | `llm-mistral`, `mistralrs/metal` |
| **llm-mistral-cuda** | mistral.rs with CUDA acceleration | `llm-mistral`, `mistralrs/cuda` |
| **vision** | Image envelope primitives and image preprocessing | *(no additional dependencies; uses the always-present `image` crate)* |
| **llm-llamacpp** | llama.cpp backend (cmake build + link) | `llama-cpp-sys/bindings`, `xybrid-llama/bindings` |
| **llm-llamacpp-vision** | llama.cpp VLM path with `mmproj` / `mtmd` support | `llm-llamacpp`, `vision`, `llama-cpp-sys/vision`, `xybrid-llama/vision` |

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
| **platform-android** | Android preset | `ort-dynamic`, `candle`, `llm-llamacpp` |
| **platform-ios** | iOS preset | `ort-download`, `ort-coreml`, `candle-metal`, `candle-hub`, `llm-llamacpp` |
| **platform-macos** | macOS preset | `ort-download`, `ort-coreml`, `candle-metal`, `candle-hub`, `llm-llamacpp` |
| **platform-desktop** | Desktop (Linux/Windows) preset | `ort-download`, `llm-llamacpp` |
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
| **vision** | Forward to core | `xybrid-core/vision` |
| **llm-llamacpp-vision** | Forward to core VLM path | `xybrid-core/llm-llamacpp-vision`, `llm-llamacpp`, `vision` |

---

## xybrid-ffi Feature Flags

| Feature | Description | Forwards to xybrid-sdk |
|---------|-------------|------------------------|
| **default** | No default features | *(none)* |
| **csharp** | Generate C# bindings for Unity | *(build-time only)* |
| **platform-android** | Android preset | `xybrid-sdk/platform-android` |
| **platform-ios** | iOS preset | `xybrid-sdk/platform-ios` |
| **platform-macos** | macOS preset | `xybrid-sdk/platform-macos` |
| **platform-desktop** | Desktop preset | `xybrid-sdk/platform-desktop` |
| **ort-download** | Forward to SDK | `xybrid-sdk/ort-download` |
| **ort-dynamic** | Forward to SDK | `xybrid-sdk/ort-dynamic` |
| **ort-coreml** | Forward to SDK | `xybrid-sdk/ort-coreml` |
| **candle** | Forward to SDK | `xybrid-sdk/candle` |
| **candle-metal** | Forward to SDK | `xybrid-sdk/candle-metal` |
| **candle-cuda** | Forward to SDK | `xybrid-sdk/candle-cuda` |
| **llm-mistral** | Forward to SDK | `xybrid-sdk/llm-mistral` |
| **llm-mistral-metal** | Forward to SDK | `xybrid-sdk/llm-mistral-metal` |
| **llm-mistral-cuda** | Forward to SDK | `xybrid-sdk/llm-mistral-cuda` |
| **llm-llamacpp** | Forward to SDK | `xybrid-sdk/llm-llamacpp` |
| **vision** | Forward to SDK image envelope primitives | `xybrid-sdk/vision` |
| **llm-llamacpp-vision** | Forward to SDK llama.cpp VLM path | `xybrid-sdk/llm-llamacpp-vision` |
| **huggingface** | Forward to SDK registry/HuggingFace loading | `xybrid-sdk/huggingface` |

---

## xybrid-cli Feature Flags

| Feature | Description | Enables |
|---------|-------------|---------|
| **default** | CLI defaults to image-bearing input support so `xybrid run --input-image` works in a `cargo install xybrid-cli` build with no extra flags | `vision` |
| **huggingface** | Direct HuggingFace loading for `xybrid run --huggingface` | `xybrid-sdk/huggingface` |
| **onnx-inspect** | ONNX metadata inspection for `xybrid init` | `xybrid-sdk/onnx-inspect` |
| **vision** | `xybrid run --input-image` and REPL `/image` envelope construction for VLM turns | `xybrid-core/vision`, `xybrid-sdk/vision` |
| **llm-llamacpp-vision** | llama.cpp VLM runtime plus CLI image input support | `llm-llamacpp`, `vision`, `xybrid-sdk/llm-llamacpp-vision` |
| **platform-android** | Android release preset | `ort-dynamic`, `llm-llamacpp`, `candle`, `huggingface` |
| **platform-ios** | iOS release preset | `ort-download`, `ort-coreml`, `candle-metal`, `candle-hub`, `llm-llamacpp`, `huggingface` |
| **platform-macos** | macOS release preset | `ort-download`, `ort-coreml`, `candle-metal`, `candle-hub`, `llm-llamacpp`, `huggingface` |
| **platform-desktop** | Linux/Windows release preset | `ort-download`, `llm-llamacpp`, `huggingface` |

---

## Platform Presets

Platform presets are the **single source of truth** for platform-specific feature combinations. They are defined in `xybrid-sdk/Cargo.toml` and forwarded through the crate hierarchy.

All current platform presets default to **text-only** llama.cpp support. Vision-language builds must compose the platform preset with `llm-llamacpp-vision`; use `vision` alone only when a crate needs image envelope/preprocessing types without the llama.cpp VLM runtime.

| Preset | Target Platform | Core Features Enabled | VLM Default | Rationale |
|--------|-----------------|----------------------|-------------|-----------|
| **platform-android** | Android (all ABIs) | `ort-dynamic`, `candle`, `llm-llamacpp` | Off; add `llm-llamacpp-vision` | Dynamic ORT loading for AAR distribution; Candle (CPU) for Whisper ASR; llama.cpp has runtime SIMD detection; mistral.rs causes SIGILL on devices without ARMv8.2-A FP16 |
| **platform-ios** | iOS (arm64, simulator) | `ort-download`, `ort-coreml`, `candle-metal`, `candle-hub`, `llm-llamacpp` | Off; add `llm-llamacpp-vision` | Static ORT linking; CoreML for ANE acceleration; Metal for GPU |
| **platform-macos** | macOS (arm64, x86_64) | `ort-download`, `ort-coreml`, `candle-metal`, `candle-hub`, `llm-llamacpp` | Off; add `llm-llamacpp-vision` | Same as iOS - unified Apple platform features |
| **platform-desktop** | Linux, Windows | `ort-download`, `llm-llamacpp` | Off; add `llm-llamacpp-vision` | Static ORT linking; llama.cpp for LLM inference (unified across all platforms) |

> **Note**: The CLI (`xybrid-cli`) adds `huggingface` to all its platform presets so `xybrid run --huggingface` works in release builds. SDK/FFI presets do not include `huggingface` by default — add it individually if needed.

Example VLM builds:

```bash
cargo build -p xybrid-cli --features platform-macos,llm-llamacpp-vision
cargo check -p xybrid-sdk --features platform-desktop,llm-llamacpp-vision
cargo check -p xybrid-ffi --features platform-ios,llm-llamacpp-vision
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
| `candle` | `feature = "candle"` | Candle (pure Rust) runtime adapter |
| `llm` | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | Shared LLM types and adapter |
| `mistral` | `feature = "llm-mistral"` | MistralBackend implementation |
| `llama_cpp` | `feature = "llm-llamacpp"` | LlamaCppBackend implementation |

### execution/executor.rs

| Item | Condition | Description |
|------|-----------|-------------|
| `LlmRuntimeAdapter` import | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | LLM adapter import |
| `llm_adapter_cache` field | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | Cached LLM adapter in TemplateExecutor |
| `ExecutionTemplate::Gguf` handling | `feature = "llm-mistral" OR feature = "llm-llamacpp"` | GGUF model execution path |
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
| iOS arm64 + simulator | macOS | `cargo xtask build-xcframework --release` (cross-compiles `xybrid-uniffi` for `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios`). See [`.github/workflows/build-apple.yml`](../.github/workflows/build-apple.yml) for the CI variant including the vision matrix job. |
| Android arm64-v8a / armeabi-v7a / x86_64 | Linux or macOS with NDK | `cargo xtask build-android --release` (drives `cargo ndk` against `xybrid-uniffi` for all three ABIs). See [`.github/workflows/build-android.yml`](../.github/workflows/build-android.yml) for the matrix-parallelised CI variant. |
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
| `build-ffi` | Build xybrid-ffi library (C ABI) | Any | `cargo xtask build-ffi --release` |
| `build-xcframework` | Build Apple XCFramework via boltffi (Swift bindings + xcframework) | macOS only | `cargo xtask build-xcframework --release` |
| `build-android` | Build Android .so files | Any | `cargo xtask build-android --release` |
| `build-flutter` | Build Flutter native libraries | Varies | `cargo xtask build-flutter --platform macos` |
| `setup-targets` | Install Rust cross-compilation targets | Any | `cargo xtask setup-targets` |
| `build-all` | Build all platforms | Varies | `cargo xtask build-all --release` |
| `package` | Package artifacts for distribution | Any | `cargo xtask package --version 0.1.2` |

### xtask to Feature Preset Mapping

| xtask Command | Platform Preset Used | Targets Built |
|---------------|---------------------|---------------|
| `build-xcframework` | `platform-macos` / `platform-ios` | iOS arm64, iOS Simulator (arm64, x86_64), macOS (arm64, x86_64) |
| `build-android` | `platform-android` | arm64-v8a, armeabi-v7a, x86_64 |
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

### NDK Detection Duplication

Both xtask and build.rs need to detect the Android NDK:

| Component | Purpose | Environment Variables Checked |
|-----------|---------|------------------------------|
| **xtask** | Locate NDK for `cargo-ndk` invocation | `ANDROID_NDK_HOME`, checks for `cargo ndk --version` |
| **llama-cpp-sys build.rs** | Locate NDK for CMake toolchain file | `ANDROID_NDK_HOME`, `NDK_HOME`, `CC_*`, `ANDROID_HOME`, `ANDROID_SDK_ROOT`, common paths |

This duplication exists because:
1. xtask runs **before** cargo builds the crate
2. build.rs runs **during** the cargo build
3. cargo-ndk sets up the Rust cross-compiler but doesn't pass NDK location to CMake

### Build Flow Diagram

```
User runs: cargo xtask build-android --release

┌─────────────────────────────────────────────────────────────┐
│ xtask (Orchestration)                                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Parse command-line arguments                             │
│ 2. Detect NDK (for cargo-ndk)                               │
│ 3. For each ABI (arm64-v8a, armeabi-v7a, x86_64):           │
│    └─ Run: cargo ndk --target <rust-target> build           │
│       ├─ cargo-ndk sets CC/CXX environment variables        │
│       └─ cargo-ndk invokes cargo build                      │
│ 4. Copy .so files to bindings/kotlin/libs/                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ llama-cpp-sys build.rs (Compilation) - runs for each target │
├─────────────────────────────────────────────────────────────┤
│ 1. Runs when llama-cpp-sys/bindings is enabled              │
│ 2. If enabled:                                              │
│    a. Find Android NDK (from CC env var or ANDROID_NDK_HOME)│
│    b. Configure CMake with NDK toolchain file               │
│    c. Build llama.cpp static libraries                      │
│    d. Build wrapper.cpp                                     │
│    e. Output cargo:rustc-link-lib directives                │
│ 3. Cargo links everything together                          │
└─────────────────────────────────────────────────────────────┘
```

---

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
# Requires: Android NDK, cargo-ndk
cargo xtask build-android --release
```

### Full Feature Check

```bash
# macOS only (includes Metal features)
cargo check -p xybrid-core --features "ort-download,ort-coreml,candle-metal,llm-llamacpp"
```
