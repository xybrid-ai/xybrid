# Changelog

## 0.1.0-beta11

* Added LLM streaming telemetry: TTFT, decode/prefill TPS, and ITL now exposed via the SDK for both `llama_cpp` and `mistral` backends
* Added `Device` struct with a stable cross-platform device identifier
* Added NeuTTS codec TTS support
* Improved offline behavior: actionable errors and cached-models fallback when the registry is unreachable

## 0.1.0-beta10

* Version bump to track core release. No Flutter API changes.

## 0.1.0-beta9

* Added `fromDirectory()` for loading custom local models
* Added `fromHuggingFace()` for loading models directly from HuggingFace Hub
* Fixed cargokit version hash not triggering rebuilds across releases

## 0.1.0-beta8

* Fixed LLM model loading failing with "Unknown frame descriptor" on all platforms — passthrough GGUF models now load correctly (#16)

## 0.1.0-beta7

* Fixed `libc++_shared.so` missing from Android APK — replaced symlinks with NDK copy task
* Fixed Android 16KB page alignment for newer devices

## 0.1.0-beta6

* Version bump to track core release. No Flutter API changes.

## 0.1.0-beta5

* Qwen 3.5 model support via updated llama.cpp backend
* Automatic `<think>` tag stripping for reasoning models

## 0.1.0-beta4

* Version bump to track core release. No Flutter API changes.

## 0.1.0-beta3

* Version bump to track core release. No Flutter API changes.

## 0.1.0-beta2

* Version bump — core runtime fix (reverted ORT to `2.0.0-rc.11`). No Flutter API changes.

## 0.1.0-beta1

* Version bump to track core release. No Flutter API changes.

## 0.1.0-alpha8

* Version bump to track core release. No Flutter API changes.

## 0.1.0-alpha7

### Features

* **GenerationConfig**: Control LLM generation parameters (temperature, top_p, max_tokens, etc.) via optional `config` parameter on all `XybridModel` run and streaming methods
* **GenerationConfig presets**: `GenerationConfig.greedy()` and `GenerationConfig.creative()` named constructors for common configurations

## 0.1.0-alpha6

### Features

* Xybrid Studio video polish and UI improvements

## 0.1.0-alpha5

### Features

* **Registry model loading**: Load models directly from the xybrid registry with `Xybrid.model(modelId: '...')`
* **LLM chat streaming**: Real-time token-by-token streaming for LLM inference
* **Conversation context**: Multi-turn conversation memory with `ConversationContext`
* **Pipeline execution**: Run multi-stage ML pipelines from YAML definitions
* **5-platform support**: macOS, iOS, Android, Linux, Windows

### Improvements

* Remote model usage example added to Flutter example app
* Updated LLM demo screen in Flutter example app
* Kotlin SDK published to Maven Central (`ai.xybrid:xybrid-kotlin:0.1.0-alpha3`)

## 0.1.0-alpha4

### Features

* **TTS quality improvements**: Silence token handling, center-break chunking, voice mixing, CJK punctuation, inter-chunk crossfading, configurable speed
* **Composable model system**: Metadata-driven TTS input mapping, voice selection strategy
* **KittenTTS phonemizer fix**: Switched from CmuDict to MisakiDictionary for correct phoneme output

### Improvements

* Model naming convention standardized (e.g., `kitten-tts-nano-0.2`)
* TTS registry cleaned up with proper model versioning

## 0.1.0-alpha3

### Features

* **LLM hardening**: Thread-safe llama.cpp wrapper, multi-token EOG, min_p sampling
* **Windows support**: MSVC CRT mismatch resolved, Git Bash CFLAGS fix
* **Unity iOS build**: C FFI library building for iOS targets

### Improvements

* Release CI fixes across all platforms
* Test CI and release workflow updates
* Metadata generation tooling for automated model config

## 0.1.0-alpha2

### Features

* **Conversation memory**: `ConversationContext` with configurable FIFO pruning, `ChatTemplateFormatter` (ChatML, Llama 2)
* **Unified ORT iOS**: Shared `vendor/ort-ios/` xcframework across all build paths
* **xtask auto-detection**: Build commands automatically select platform features based on target triple

### Breaking Changes

* Feature flag cascade fix: `ort-download` + `ort-dynamic` now caught at compile time
* Platform presets renamed for clarity

## 0.1.0-alpha1

### Features

* **Platform SDK restructure**: UniFFI bindings (Swift/Kotlin), xybrid-ffi (C API)
* **Thin Flutter FFI**: ~150 LOC Dart bridge via flutter_rust_bridge
* **xtask build commands**: `cargo xtask build-ffi`, `build-uniffi`, `build-xcframework`, `build-android`, `build-flutter`
* **GitHub Actions CI**: Automated builds for all platforms

### Breaking Changes

* `xybrid_core::llm` module renamed to `xybrid_core::cloud`
* `PipelineLoader` renamed to `PipelineRef`
* `XybridPipeline` renamed to `Pipeline`
* Direct TTS API removed (use pipeline execution instead)

### Platform Support

| Platform | ONNX Runtime | Candle | LLM |
|----------|-------------|--------|-----|
| macOS | download-binaries | Metal | llama.cpp |
| iOS | vendor/ort-ios/ | Metal | llama.cpp |
| Android | load-dynamic | - | llama.cpp |
| Linux | download-binaries | CPU | llama.cpp |
| Windows | download-binaries | CPU | llama.cpp |
