# Changelog

## 0.2.0-rc1

Release candidate for `0.2.0`, published so consumers can validate the vision
binding against real integrations ahead of the stable tag. No functional
changes from `0.2.0` — see the `0.2.0` entry below for the full change set.

## 0.2.0-alpha

Prerelease of `0.2.0`, published to validate the release pipeline ahead of the
stable tag. No functional changes from `0.2.0` — see the `0.2.0` entry below
for the full change set.

## 0.2.0

The vision release. The binding gains on-device multimodal input and the
real-time camera vision primitives behind Studio's live loop.

* On-device vision (VLM): new `XybridEnvelope.image` (encoded PNG/JPEG/WebP), `XybridEnvelope.imageRaw` (raw camera/canvas pixel frames), and `XybridEnvelope.multiPart` (user-role message with image attachments) for running vision-language models from Dart (xybrid-ai/xybrid#245, #265)
* Reachable streaming cancellation: new `CancellationToken` whose `cancel()` drives a real runtime abort end-to-end — the generation halts at the next token and releases the model lock, instead of the old behavior where "stop" only unsubscribed while the runtime kept generating (xybrid-ai/xybrid#245)
* Live-loop run options on the model handle: `preempt` (latest-frame-wins — a new run preempts the in-flight one so a live loop no longer head-of-line-blocks behind a stale frame) and `frameSessionId` for tagging live inferences (xybrid-ai/xybrid#245)
* Raw-frame path avoids per-frame JPEG re-encoding: `imageRaw` packs RGB pixel buffers straight through to the multimodal runtime; the encoded `image` path remains the fallback (xybrid-ai/xybrid#245)
* Streaming TTS support on top of the new audio generation path (xybrid-ai/xybrid#245)
* Live-mode telemetry is rate-limited by a per-session sampler (≈1 row/sec/session), so live camera sessions no longer emit a telemetry row per frame (xybrid-ai/xybrid#245)
* `XybridModel.warmup` / `unload` are now exposed on the Flutter binding, completing the sync/async method symmetry (xybrid-ai/xybrid#293)
* Fixed: TTS text chunking is now UTF-8-safe — multi-byte codepoints are no longer split mid-character (xybrid-ai/xybrid#249)
* Fixed: `.npz` voice files are detected by magic header rather than file extension (xybrid-ai/xybrid#252)
* Fixed: `tokens_out` is now emitted on local LLM telemetry paths (xybrid-ai/xybrid#253)

## 0.1.2

* Audio inputs now detect MP3, OGG, and FLAC in addition to WAV, and mono audio is upmixed to stereo when a model expects two channels (xybrid-ai/xybrid#132, #141)
* Robustness: the underlying SDK/core no longer panics on poisoned locks, unchecked length headers, or non-contiguous ONNX output tensors — these are recovered or handled gracefully (xybrid-ai/xybrid#233, #234, #235, #231, #232, #237)
* The Xybrid API key is no longer placed in the process environment (xybrid-ai/xybrid#214)
* Registry requests now honor `Retry-After` on `429` responses (xybrid-ai/xybrid#134)

## 0.1.1

* New bundled `init()` entry point starts anonymous-by-default telemetry from an API key; the standalone `initTelemetry` is now legacy (xybrid-ai/xybrid#188, #195)
* `PlatformEvent` payloads now carry `sdk_version` and `binding`, so telemetry is attributable to the SDK build and the Flutter binding that emitted it (xybrid-ai/xybrid#183)
* Fixed: the SDK no longer leaks the leading bytes of its own API key into emitted telemetry (xybrid-ai/xybrid#209)
* Fixed: cache TTL handling is panic-safe — a backwards system clock no longer panics the cache layer (xybrid-ai/xybrid#203)
* Example app now reads `XYBRID_API_KEY` from the environment at init (xybrid-ai/xybrid#207)

## 0.1.0

Production release of the 0.1.0 line. No Flutter-binding code changes since rc4 — closes the rc series.

Cumulative since the last published-to-pub.dev release (rc3):

* `XybridResult` now exposes typed `InferenceMetrics` (CPU / memory / GPU / wall-clock per inference); the underlying telemetry is also surfaced in the bundled Flutter demos
* Streaming-LLM cloud fallback now routes off live device pressure signals (CPU / memory / thermal) instead of static thresholds
* `ModelWarmup` events emit from `XybridModel.warmup` and arrive in the binding's telemetry stream, so first-token latency is attributable to warmup vs. inference
* `streaming` is now a top-level field on `PlatformEvent` payloads instead of nested under metadata
* GGUF bundles without an explicit backend annotation now report `llamacpp` in telemetry instead of `unknown`
* New `Denormalize` postprocessing step in the SDK core (mirror of `Normalize`), useful for round-tripping model output back into input-space coordinates
* Fixed: `ModelComplete` events were dropped on streaming fast-path inference; now emitted on every code path
* Fixed: internal orchestrator pipeline-frame events no longer leak to the binding as opaque payloads

## 0.1.0-rc4

* `XybridResult` now exposes typed `InferenceMetrics` (CPU / memory / GPU / wall-clock per inference); the underlying telemetry is also surfaced in the bundled Flutter demos
* Streaming-LLM cloud fallback now routes off live device pressure signals (CPU / memory / thermal) instead of static thresholds
* `ModelWarmup` events emit from `XybridModel.warmup` and arrive in the binding's telemetry stream, so first-token latency is attributable to warmup vs. inference
* `streaming` is now a top-level field on `PlatformEvent` payloads instead of nested under metadata
* GGUF bundles without an explicit backend annotation now report `llamacpp` in telemetry instead of `unknown`
* New `Denormalize` postprocessing step in the SDK core (mirror of `Normalize`), useful for round-tripping model output back into input-space coordinates
* Fixed: `ModelComplete` events were dropped on streaming fast-path inference; now emitted on every code path
* Fixed: internal orchestrator pipeline-frame events no longer leak to the binding as opaque payloads

## 0.1.0-rc3

* Adaptive cloud fallback for streaming LLM: pipelines can now transparently fall back to a cloud runtime when on-device streaming generation stalls or errors mid-stream; configurable via new run options on the underlying SDK
* Streaming and chat-context LLM telemetry spans now include backend and quantization tags (previously dropped on these code paths)
* Hybrid LLM architectures (Mamba / SSM-style) now load and run cleanly through the bundled llama.cpp runtime

## 0.1.0-rc2

* Republishes 0.1.0-rc1 — the rc1 pub.dev publish was skipped due to an upstream compile failure in `xybrid-core` on `aarch64-linux-android` (fixed in xybrid-ai/xybrid#112). No API or behavior changes in the Flutter binding itself.

## 0.1.0-rc1

* Registry calls now send the `X-Xybrid-Client` telemetry header identifying the Flutter binding, SDK / core versions, platform, and enabled backends; respects the `XYBRID_TELEMETRY_OPTOUT` env var
* Per-inference resource telemetry: CPU / memory / GPU pressure metrics now flow into telemetry events from the underlying SDK
* Cloud LLM telemetry exposes provider-agnostic prompt-cache token counts (`cache_creation` / `cache_read`)

## 0.1.0-beta12

* LLM telemetry expansion: swim-lane spans, device profile metadata, and Pipeline::run hardening on top of beta11's streaming telemetry
* Fixed Windows precompile path mangling that was blocking native binaries from publishing to pub.dev

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
