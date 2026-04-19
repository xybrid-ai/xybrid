# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned (0.1.x)

- **OpenUPM registry**: Publish Unity SDK to [openupm.com](https://openupm.com) for scoped registry install

---

## [0.1.0-beta11] - 2026-04-19

### Added

- **LLM streaming telemetry** (#40): TTFT, decode/prefill TPS, and ITL metrics emitted by the SDK for both `llama_cpp` and `mistral` backends. Streaming paths in both backends hardened with regression coverage for `<think>...</think>` tag filtering.
- **Device struct** (#42): SDK exposes a `Device` struct with a stable cross-platform device identifier for telemetry and routing.
- **NeuTTS codec TTS** (#43): Codec-based TTS integration via the llama.cpp runtime adapter.
- **Actionable offline errors and cached-models fallback**: Offline load no longer trips the circuit breaker, `SdkError::Offline` now propagates through `xybrid-uniffi`, and the SDK falls back to the local cache with a clear error when the registry is unreachable.

### Fixed

- **Rust 1.95 clippy clean** (#41): Satisfied `collapsible_match` and `unnecessary_sort_by` lints introduced in Rust 1.95.

---

## [0.1.0-beta10] - 2026-04-07

### Added

- **CLI install scripts** (#19): Shell and PowerShell install scripts with installation guide
- **Chinese documentation**: Full Chinese localization for all documentation pages
- **CI hardening**: Dependabot grouping, concurrency groups, OpenSSF Scorecard audit

### Fixed

- **Passthrough model download**: Fix bare model download unpack for passthrough GGUF models
- **HuggingFace CLI feature**: Enable `huggingface` feature in CLI for all platforms
- **Install script release selection**: Fix release binary selection in install scripts

---

## [0.1.0-beta9] - 2026-04-02

### Added

- **Custom model loading** (#15): `fromDirectory()` exposed in all SDK bindings (Flutter, Kotlin, Swift, Unity), `fromHuggingFace()` in Rust SDK with auto-generated `model_metadata.json`, JSON Schema published
- **`xybrid init` command** (#18): Auto-generate `model_metadata.json` by inspecting ONNX/GGUF/SafeTensors model files
- **HuggingFace models in REPL mode** (#20): Use HuggingFace models directly in interactive CLI sessions
- **`xybrid run <file>` command**: Run inference directly on a model file
- **LFM2.5-350M model spec**: Liquid AI passthrough model added to registry
- **CLI UI refresh**: Updated welcome screen, improved token/latency display, general UI cleanup

### Fixed

- **Android arm64 performance regression**: Enable dotprod GEMM kernels (`GGML_CPU_ARM_ARCH=8.2`) — fixes 3-5x throughput drop on Cortex-A76+ devices (Snapdragon 855+, Tensor G1+)
- **Smart GGUF variant selection**: Better variant matching when loading from HuggingFace
- **Flutter binding version hash**: Fixed cargokit static hash not updating across releases
- **CI test gating**: Fixture validation and init tests now skip gracefully when model files are unavailable

---

## [0.1.0-beta8] - 2026-03-24

### Fixed

- **LLM model loading on Android**: Fixed "Unknown frame descriptor" error when loading passthrough GGUF models (gemma-3-1b, smollm2-360m, qwen-3.5-0.8b) — `load_from_registry_api` now uses `fetch_extracted()` to correctly handle both `.xyb` bundles and passthrough variants (#16)

---

## [0.1.0-beta7] - 2026-03-21

### Added

- **Android sample app overhaul**: Audio playback via `PcmPlayer`, improved `InferenceCard` and `ModelLoadingCard` UI components, updated model catalog

### Fixed

- **Android 16KB page alignment**: Added `-Wl,-z,max-page-size=16384` linker flag for Android targets to support 16KB page size devices
- **Android `libc++_shared.so` missing from APK**: Replaced broken symlinks with a `copyNdkLibs` Gradle task that copies from the NDK at build time (fixes pub.dev packaging)

---

## [0.1.0-beta6] - 2026-03-18

### Fixed

- **Android FP16 assembler**: Fixed FP16 assembler issue on Android builds
- **Android OpenSSL**: Fixed whisper OpenSSL pull on Android
- **Unity macOS and Linux builds**: Fixed build failures for Unity on macOS and Linux
- **Unity missing meta files**: Added missing Unity `.meta` files
- **Candle device module**: Added missing `Debug` import in candle device module
- **Android bindings cleanup**: Removed committed `.gradle/` files from Android bindings

### Changed

- **CI: Strip Android and Linux builds for Unity**: Reduced binary size by stripping symbols
- **CI: Removed iOS from UPM build** temporarily
- **Documentation updates**: Cleaned up READMEs, added X social link

---

## [0.1.0-beta5] - 2026-03-11

### Added

- **Qwen 3.5 support**: Updated vendored llama.cpp to support `qwen35` architecture (0.8B and 2B models)
- **Think tag stripping**: Automatically strips `<think>...</think>` reasoning blocks from Qwen 3.5 and similar models in both batch and streaming generation
- **`XYBRID_LLAMACPP_VERBOSITY` env var**: Surface llama.cpp C++ logs for debugging model load failures (set to 4 for full debug output)

### Changed

- **Vendored llama.cpp**: Updated from Jan 30 to Mar 11 2026 (adds qwen35, qwen3next architectures)
- **Improved error messages**: Model load failures now include file path and hint about unsupported architectures

---

## [0.1.0-beta4] - 2026-03-10

### Added

- **Swift SDK (UniFFI)**: Regenerated UniFFI bindings with full API surface — `XybridVoiceInfo`, `XybridGenerationConfig`, expanded error enum
- **Swift Package.swift binaryTarget**: Replaced source-based FFI target with `.binaryTarget` pointing to local XCFramework
- **XCFramework headers**: Added `-headers` flag to `xcodebuild -create-xcframework` in xtask for SPM module resolution
- **Module map**: Created `module.modulemap` for `xybrid_uniffiFFI` clang module
- **System framework linking**: Metal, MetalPerformanceShaders, MetalPerformanceShadersGraph, CoreML, Accelerate, Security, libc++
- **iOS example app rewrite**: Real SDK integration — removed all mock/simulated code, uses real `import Xybrid` with model loading, voice picker, and audio playback
- **Swift release workflow**: `publish-swift` job builds XCFramework, computes checksum, publishes URL-based Package.swift to `swift` orphan branch
- **Unity platform .meta files**: Import settings for all target platforms (Windows, Linux, Android ABIs, iOS static lib)
- **UPM branch CI**: Publishes `upm` branch with pre-built native libraries for Unity Package Manager

### Changed

- **Unity CI**: Updated `build-unity.yml` to wait for release to exist before publishing
- **xtask**: Updated library name detection to use target triple instead of host OS for cross-compilation

---

## [0.1.0-beta3] - 2026-03-07

### Added

- **Unity CI pipeline** (`build-unity.yml`): Automated native library builds for all Unity platforms (macOS arm64, Windows x86_64, Linux x86_64, iOS arm64, Android arm64/armv7/x86_64)
- **UPM branch distribution**: CI publishes `upm` branch with pre-built native libraries for Unity Package Manager Git URL install (`https://github.com/xybrid-ai/xybrid.git#upm`)
- **Unity platform .meta files**: Import settings for all target platforms (Windows, Linux, Android ABIs, iOS static lib)
- **C# bindings sync check**: CI validates `NativeMethods.g.cs` stays in sync with `xybrid-ffi`
- **`cargo xtask build-unity`**: New subcommand for building Unity native libraries locally
- **Unity SDK packaging**: `xybrid-unity-sdk-<version>.tar.gz` attached to GitHub Releases
- **Android cargo-ndk support**: `build-ffi` uses cargo-ndk for Android cross-compilation (matches Kotlin/Flutter CI)

### Fixed

- Fixed library name detection in xtask to use target triple instead of host OS for cross-compilation
- Removed x86_64-apple-darwin from Unity targets (ORT has no prebuilt binaries; arm64 via Rosetta 2)

---

## [0.1.0-beta2] - 2026-03-06

### Fixed

- Reverted to ort `2.0.0-rc.11` to resolve compatibility regressions from rc.12

---

## [0.1.0-beta1] - 2026-03-04

### Added

- **CLI modular refactor**: Split monolithic `main.rs` into modular command files
- **Bundle download**: `xybrid fetch` now supports direct `.xyb` bundle downloads
- **Missing model warnings**: CLI warns when referenced models are not cached
- **Pass-through model resolution**: Models resolve through registry transparently
- **ORT upgrade**: Upgraded to ort 2.0.0-rc.12 (reverted in beta2)
- Chinese README translation (`README.zh-CN.md`)

### Changed

- Updated version-sync.sh tooling
- Updated API reference documentation
- Unity build artifacts updated

---

## [0.1.0-alpha8] - 2026-03-01

### Added

- **OpenPhonemizer support**: New phonemizer backend option (#10)
- **Per-model chunk sizing**: Model metadata can now specify chunk size for execution
- **Unified API contract**: Added `api-surface.yaml` and `api-contract-check.sh` for SDK contract validation
- **Telemetry integration tests**: OpenTelemetry span exporter (`UreqSpanExporter`) in xybrid-sdk
- **OpenTelemetry API**: Added tracing API to xybrid-core
- **KittenTTS Micro 0.8**: New model fixture
- Chinese README (`README.zh-CN.md`)

### Changed

- Improved G2P / dictionary quality
- Adaptive LLM defaults for Android (performance tuning)
- Telemetry cleanup and integration test improvements

### Fixed

- Removed `opt_level()` and environment variable usage in tests
- Fixed integration test model fixtures

---

## [0.1.0-alpha7] - 2026-02-18

### Added

- **GenerationConfig SDK propagation**: Surfaced `GenerationConfig` (temperature, top_p, max_tokens, min_p, top_k, repetition_penalty, stop_sequences) through all three SDK bindings
  - Flutter/Dart: `GenerationConfig` class with `greedy()` / `creative()` presets, optional `config` parameter on all run/streaming methods
  - Kotlin/Android: `XybridGenerationConfig` UniFFI Record with `GenerationConfigs.greedy()` / `creative()` presets
  - Unity/C#: `GenerationConfig : IDisposable` with opaque handle pattern, setter methods, `Greedy()` / `Creative()` factories

### Fixed

- Rust SDK `run_async()` now accepts `Option<&GenerationConfig>` (was hard-coded to `None`, blocking Kotlin config passthrough)
- Fixed LLM generation max tokens on macOS

---

## [0.1.0-alpha6] - 2026-02-20

### Added

- **Flutter pub.dev preparation**: Prepared `xybrid_flutter` for pub.dev publication
- **Flutter model status APIs**: Exposed model status query APIs in Flutter SDK
- **ORT binary externalization**: Externalized ORT binaries from Flutter package (36MB → 137KB)

### Fixed

- Flutter publish configuration fixes
- Model offloading memory issue resolved

---

## [0.1.0-alpha5] - 2026-02-16

### Added

- **TTS quality improvements** (#9): Silence tokens, center-break chunking, voice mixing, CJK punctuation, inter-chunk crossfading, configurable speed
- **KittenTTS Integration V1.0 Prep**: Fixed phonemizer mismatch (CmuDict → Misaki), Python parity validation
- **Composable model system**: Pluggable phonemizer backends for TTS

### Fixed

- Phonemization token mapping fixes
- Backend phonemization boundary fixes
- Regenerated UniFFI Kotlin bindings

---

## [0.1.0-alpha4] - 2026-02-14

### Added

- **Kokoro TTS quality parity**: Closed quality gap with official Python pipeline (#8)
- **Swift/Kotlin voice selection**: Voice selection support in Apple and Kotlin SDKs
- **Unity TTS and voice support**: Full TTS pipeline with voice selection in Unity SDK

### Fixed

- Resolved chat template token leaks in LLM output
- Converted broken doctests from `no_run` to `ignore` across all crates
- Resolved all CI clippy failures

### Changed

- Documentation cleanup across README, Kotlin docs
- CI workflow updates

---

## [0.1.0-alpha3] - 2026-02-12

### Added

- **Kotlin Android SDK**: Real inference via UniFFI + TemplateExecutor with ORT bundling
- **Metadata generation tooling**: Automated model metadata generation
- **Flutter remote usage example**: Example demonstrating remote model loading
- **Unity iOS build support**: C FFI library building for iOS targets
- **min_p sampling**: Added to llama.cpp sampler chain (default 0.05)

### Fixed

- **Thread safety**: Removed unsafe `impl Sync for LlamaContext`, added Mutex wrapping
- **Multi-token EOG**: `llama_vocab_is_eog()` for Llama 3, Gemma, Qwen end-of-generation detection
- **llama.cpp audit fixes #4–#13**: Comprehensive wrapper audit
- **Hot loop optimization**: Hoisted `candidates_data` allocation out of generation hot loop
- **Callback ordering**: Check end-of-generation BEFORE emitting to callback
- **flash_attn_type**: Use enum for context params instead of raw values
- **Windows CRT mismatch**: Static CRT (/MT) for llama_wrapper to match esaxx-rs
- **Windows MSVC CRT**: Resolved CRT mismatch for CLI builds
- **Git Bash CFLAGS**: Use `-MD` not `/MD` to prevent path mangling
- Unity build folder output directories corrected
- llama.cpp pub cache failure resolved
- Release build failures across all platforms (#6)

### Changed

- Updated Kotlin bindings publish configuration
- Updated `libxybrid_ffi.dylib` for latest SDK
- Updated LLM demo screen in Flutter example app
- CI workflow updates (test-ci.yml, release.yml)

---

## [0.1.0-alpha2] - 2026-02-10

### Added

- Unity macOS build artifacts
- Sample and integration test cleanup

### Fixed

- Prevented heap corruption in llama.cpp when prompt exceeds 512 tokens

---

## [0.1.0-alpha1] - 2026-02-09

### Added

- **Version bump tooling**: `version-sync.sh`, `just version`, `just bump-version`
- **Unity C# SDK**: Exposed xyb bundler to C# library, updated to latest APIs
- **Open source community files**: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, GitHub templates
- **README overhaul**: SDK hierarchy, Quick Start, Models by task, Features matrix
- **Documentation lean-down**: Restructured internals to concepts, cleaned up docs
- **CI infrastructure**: sccache, FRB binary caching, FRB staleness check, workflow_dispatch
- **Flutter precompile configuration**

### Fixed

- Force `/MD` (dynamic CRT) on Windows builds to fix esaxx-rs
- Added missing `-std=c++17` in cc-rs build
- FRB install fixes
- Removed `NativeMethods.Bundle.cs`

### Changed

- Converted llama.cpp to submodule
- Replaced cloning with submodule in builds

---

## [0.1.0] - 2026-01-27

First production release of xybrid - a hybrid cloud-edge ML inference orchestrator.

### Added

#### CLI

- `xybrid models list` - List models from registry
- `xybrid models search <query>` - Search models
- `xybrid models info <id>` - Show model details
- `xybrid plan <pipeline.yaml>` - Show execution plan
- `xybrid fetch --model <id>` - Download model with progress
- `xybrid fetch <pipeline.yaml>` - Pre-download pipeline models
- `xybrid cache list` - Show cached models
- `xybrid cache status` - Cache statistics
- `xybrid cache clear` - Clear cache
- `xybrid run <pipeline.yaml>` - Execute pipeline
- `xybrid run --model <id>` - Direct model execution from registry
- `xybrid run --voice <index>` - TTS voice selection
- `xybrid run --output <file>` - Save output (WAV/text/JSON)
- `xybrid run --trace` - Execute with tracing

#### Core Runtime

- ONNX Runtime execution with preprocessing/postprocessing
- Whisper ASR with Metal acceleration (macOS/iOS)
- Metadata-driven model execution
- Policy-based orchestration with offline-first routing
- CoreML/ANE acceleration for Apple devices

#### LLM Inference

- Local LLM execution for GGUF models
- Desktop: CPU, Metal (macOS), CUDA (Linux/Windows)
- Android: Optimized for ARM devices
- Runtime backend selection via model metadata

#### SDK

- `PipelineRef::from_yaml()` - Instant YAML parsing
- `Pipeline::load_models()` - Model preloading with progress
- `Pipeline::run()` - Execute inference
- `RegistryClient` - Model discovery, resolution, and caching
- Telemetry with batching

#### Preprocessing

- `AudioDecode` - WAV bytes to float samples
- `Phonemize` - Text to phoneme tokens
- `Tokenize` - Text tokenization

#### Postprocessing

- `CTCDecode` - Logits to text transcription
- `TTSAudioEncode` - Waveform to PCM audio bytes
- `ArgMax` - Classification output

### Models Supported

- **Kokoro-82M** (TTS) - 24 voices
- **KittenTTS-nano** (TTS) - Lightweight
- **Whisper-tiny** (ASR) - Real-time capable
- **Wav2Vec2-base-960h** (ASR) - English
- **all-MiniLM-L6-v2** (Embeddings) - 384-dim vectors
- **MobileNetV2** (Vision) - 6.8x ANE speedup
- **Qwen 2.5 0.5B** (LLM) - On-device chat

### Platform Support

| Platform | ASR/TTS/Vision | LLM | Hardware Acceleration |
|----------|----------------|-----|----------------------|
| macOS arm64 | ✅ | ✅ | CoreML ANE, Metal GPU |
| macOS x86_64 | ✅ | ✅ | CoreML GPU |
| Linux x86_64 | ✅ | ✅ | CUDA |
| Windows x86_64 | ✅ | ✅ | CUDA |
| Android arm64 | ✅ | ✅ | CPU (NNAPI planned) |
| iOS arm64 | ✅ | Planned | CoreML ANE, Metal GPU |

## [Unreleased]

### Planned

- Android NNAPI execution provider
- MLX runtime for Apple Silicon
- Voice cloning support
- Streaming TTS
