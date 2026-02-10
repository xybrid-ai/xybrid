# Feature Matrix

This document provides a comprehensive reference for all feature flags, platform presets, and valid combinations across the xybrid crate hierarchy.

## Table of Contents

1. [xybrid-core Feature Flags](#xybrid-core-feature-flags)
2. [xybrid-sdk Feature Flags](#xybrid-sdk-feature-flags)
3. [xybrid-ffi Feature Flags](#xybrid-ffi-feature-flags)
4. [Platform Presets](#platform-presets)
5. [Feature-Gated Types and Modules](#feature-gated-types-and-modules)
6. [Invalid Feature Combinations](#invalid-feature-combinations)
7. [ORT Loading Strategy](#ort-loading-strategy)
8. [xtask Commands](#xtask-commands)
9. [Build Architecture](#build-architecture)

---

## xybrid-core Feature Flags

| Feature | Description | Enables |
|---------|-------------|---------|
| **default** | Default features | `ort-download`, `llm-llamacpp` |
| **ort-download** | Download prebuilt ONNX Runtime binaries | `ort/download-binaries`, `ort/tls-native` |
| **ort-dynamic** | Load ONNX Runtime .so at runtime | `ort/load-dynamic` |
| **ort-coreml** | Apple Neural Engine acceleration | `ort/coreml` |
| **candle** | Pure Rust ML framework (Whisper) | `candle-core`, `candle-nn`, `candle-transformers`, `safetensors`, `hf-hub`, `byteorder`, `num-traits` |
| **candle-metal** | Candle with Metal GPU acceleration | `candle`, `candle-core/metal`, `candle-nn/metal` |
| **candle-cuda** | Candle with CUDA GPU acceleration | `candle`, `candle-core/cuda` |
| **llm-mistral** | mistral.rs LLM backend (CPU) | `mistralrs` |
| **llm-mistral-metal** | mistral.rs with Metal acceleration | `llm-mistral`, `mistralrs/metal` |
| **llm-mistral-cuda** | mistral.rs with CUDA acceleration | `llm-mistral`, `mistralrs/cuda` |
| **llm-llamacpp** | llama.cpp backend (Android-compatible) | *(marker feature - triggers build.rs)* |

### Notes

- `llm-llamacpp` is a **marker feature** - it doesn't enable external crate dependencies but instead:
  1. Triggers `build.rs` to compile vendored llama.cpp via CMake
  2. Gates source code with `#[cfg(feature = "llm-llamacpp")]` blocks
  3. Requires `vendor/llama.cpp` directory with cloned llama.cpp source

---

## xybrid-sdk Feature Flags

| Feature | Description | Forwards to xybrid-core |
|---------|-------------|-------------------------|
| **default** | No default features | *(none)* |
| **platform-android** | Android preset | `ort-dynamic`, `llm-llamacpp` |
| **platform-ios** | iOS preset | `ort-download`, `ort-coreml`, `candle-metal`, `llm-llamacpp` |
| **platform-macos** | macOS preset | `ort-download`, `ort-coreml`, `candle-metal`, `llm-llamacpp` |
| **platform-desktop** | Desktop (Linux/Windows) preset | `ort-download`, `llm-llamacpp` |
| **ort-download** | Forward to core | `xybrid-core/ort-download` |
| **ort-dynamic** | Forward to core | `xybrid-core/ort-dynamic` |
| **ort-coreml** | Forward to core | `xybrid-core/ort-coreml` |
| **candle** | Forward to core | `xybrid-core/candle` |
| **candle-metal** | Forward to core | `xybrid-core/candle-metal` |
| **candle-cuda** | Forward to core | `xybrid-core/candle-cuda` |
| **llm-mistral** | Forward to core | `xybrid-core/llm-mistral` |
| **llm-mistral-metal** | Forward to core | `xybrid-core/llm-mistral-metal` |
| **llm-mistral-cuda** | Forward to core | `xybrid-core/llm-mistral-cuda` |
| **llm-llamacpp** | Forward to core | `xybrid-core/llm-llamacpp` |

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

---

## Platform Presets

Platform presets are the **single source of truth** for platform-specific feature combinations. They are defined in `xybrid-sdk/Cargo.toml` and forwarded through the crate hierarchy.

| Preset | Target Platform | Core Features Enabled | Rationale |
|--------|-----------------|----------------------|-----------|
| **platform-android** | Android (all ABIs) | `ort-dynamic`, `llm-llamacpp` | Dynamic ORT loading for AAR distribution; llama.cpp has runtime SIMD detection; mistral.rs causes SIGILL on devices without ARMv8.2-A FP16 |
| **platform-ios** | iOS (arm64, simulator) | `ort-download`, `ort-coreml`, `candle-metal`, `llm-llamacpp` | Static ORT linking; CoreML for ANE acceleration; Metal for GPU |
| **platform-macos** | macOS (arm64, x86_64) | `ort-download`, `ort-coreml`, `candle-metal`, `llm-llamacpp` | Same as iOS - unified Apple platform features |
| **platform-desktop** | Linux, Windows | `ort-download`, `llm-llamacpp` | Static ORT linking; llama.cpp for LLM inference (unified across all platforms) |

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
| `llm-mistral` on `target_os = "android"` | SIGILL crash on devices without ARMv8.2-A FP16 | Use `llm-llamacpp` instead |
| `ort-download` AND `ort-dynamic` | Mutually exclusive ORT loading strategies | Choose one based on platform |
| `candle-metal` on non-Apple targets | Metal is Apple-only | Use `candle` (CPU) or `candle-cuda` |
| `candle-cuda` on Apple targets | CUDA not available on Apple | Use `candle-metal` |
| `ort-coreml` on non-Apple targets | CoreML is Apple-only | Use `ort-download` |

**Note**: As of this writing, these compile_error! guards are planned but not yet implemented. See US-006 in the feature cascade fix PRD.

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
| `build-uniffi` | Build xybrid-uniffi library | Any | `cargo xtask build-uniffi --release` |
| `build-ffi` | Build xybrid-ffi library (C ABI) | Any | `cargo xtask build-ffi --release` |
| `generate-bindings` | Generate Swift/Kotlin bindings | Any | `cargo xtask generate-bindings --language swift` |
| `build-xcframework` | Build Apple XCFramework | macOS only | `cargo xtask build-xcframework --release` |
| `build-android` | Build Android .so files | Any | `cargo xtask build-android --release` |
| `build-flutter` | Build Flutter native libraries | Varies | `cargo xtask build-flutter --platform macos` |
| `setup-targets` | Install Rust cross-compilation targets | Any | `cargo xtask setup-targets` |
| `build-all` | Build all platforms | Varies | `cargo xtask build-all --release` |
| `package` | Package artifacts for distribution | Any | `cargo xtask package --version 0.1.0` |

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

### Layer 2: build.rs (Compilation)

**Location**: `crates/xybrid-core/build.rs`

**Responsibilities**:
- Compiling vendored llama.cpp via CMake
- Detecting Android NDK for CMake toolchain
- Platform-specific linking (Metal, Accelerate, etc.)
- Setting `cargo:rustc-link-lib` and `cargo:rustc-link-search`

**Triggered by**:
- `#[cfg(feature = "llm-llamacpp")]` in the build script
- Cargo's build process when xybrid-core is compiled

### NDK Detection Duplication

Both xtask and build.rs need to detect the Android NDK:

| Component | Purpose | Environment Variables Checked |
|-----------|---------|------------------------------|
| **xtask** | Locate NDK for `cargo-ndk` invocation | `ANDROID_NDK_HOME`, checks for `cargo ndk --version` |
| **build.rs** | Locate NDK for CMake toolchain file | `ANDROID_NDK_HOME`, `NDK_HOME`, `CC_*`, `ANDROID_HOME`, `ANDROID_SDK_ROOT`, common paths |

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
│ build.rs (Compilation) - runs for each target               │
├─────────────────────────────────────────────────────────────┤
│ 1. Check #[cfg(feature = "llm-llamacpp")]                   │
│ 2. If enabled:                                              │
│    a. Find Android NDK (from CC env var or ANDROID_NDK_HOME)│
│    b. Configure CMake with NDK toolchain file               │
│    c. Build llama.cpp static libraries                      │
│    d. Build llama_wrapper.cpp                               │
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
