# Changelog

## Unreleased

* Added: complete model-cache management through `Xybrid`, including aggregate
  status, physical entries, preferred paths, ready-model IDs,
  per-model deletion, and full clearing. These new operations return Futures
  and run off the UI isolate. Expiry cleanup reports an error until persistent
  retention is supported. The existing `isModelCached` retains
  its ready-to-load meaning; `hasCachedModelData` also counts archives and
  shared downloads that still need extraction (xybrid-ai/xybrid#505)

## 0.7.0

Streaming tool loops can now keep both live output and conversation history,
and apps can release idle model memory without throwing away model handles.

* Added: a terminal streaming token carries typed `toolCalls`,
  `finishReason: "tool_calls"`, and `rawText`, so callers dispatch a tool without
  parsing protocol text and retain the exact assistant turn needed for the
  continuation (xybrid-ai/xybrid#542)
* Added: `tool_results` continuations work through
  `runStreamingWithContext`, preserving history and token-by-token output on the
  second turn (xybrid-ai/xybrid#542)
* Added: `Xybrid.releaseMemory`, `Xybrid.setAutoRelease`, and
  `Xybrid.isAutoReleaseEnabled`. Explicit release skips busy models and an
  evicted model reloads itself on its next use; automatic release is off by
  default (xybrid-ai/xybrid#539)
* Fixed: a terminal tool call is delivered exactly once. The native terminal
  token and the completion event previously exposed the same call, which could
  make a straightforward `hasToolCalls` loop dispatch every tool twice
  (xybrid-ai/xybrid#542)
* Fixed: cloud fallback preserves `maxTokens`, `temperature`, `topP`, and exact
  `stopSequences` values instead of silently using gateway defaults or
  corrupting whitespace- and comma-bearing stops (xybrid-ai/xybrid#545)
* Changed: image-bearing tool continuations still fail closed, with an error
  that explains image embeddings cannot be reconstructed from replayed text
  (xybrid-ai/xybrid#542)

## 0.6.0

Structured output, reasoning text, and tool-capability reporting reach the Dart
surface, and the package finally ships a runnable example app.

* Added: `GenerationConfig.grammar` and `jsonSchemaToGbnf` — constrain
  generation to a JSON Schema, with the greedy and creative presets taking an
  optional grammar so the usual extraction shape is a single call. Both had
  reached the generated layer and stopped at the hand-written wrapper
  (xybrid-ai/xybrid#511)
* Added: `XybridResult.reasoningContent`, what a thinking model emits separately
  from its answer (xybrid-ai/xybrid#511)
* Added: `XybridModel.supportsToolCalling`, the bundle's tool-calling metadata
  flag as an advisory tri-state — `null` means the bundle says nothing
  (xybrid-ai/xybrid#515)
* Added: a runnable single-screen example app in `example/`, replacing the
  `example.md` snippet file. The snippets themselves remain in the package
  README (xybrid-ai/xybrid#525, xybrid-ai/xybrid#152)
* Fixed: `flutter build macos` failed with 1287 duplicate `ggml_*` symbols — one
  Rust staticlib carried two copies of ggml (xybrid-ai/xybrid#528)
* Changed: the Dart unit tests under `test/` run in CI for the first time, so
  wrapper-level regressions are caught before release (xybrid-ai/xybrid#511)

## 0.5.0

Live streaming speech recognition, now on the whisper.cpp backend, plus the
speculative-cloud surface.

* Added: live streaming ASR — `XybridStreamSession` with a rolling window over
  fed audio, partial and final transcript events, and a demo screen in the
  example app (xybrid-ai/xybrid#453)
* Added: `XybridStreamSession.fromModel(..., audioCtx:)`, an optional Whisper
  encoder-context override in mel frames. `null` keeps the model bundle's
  default (xybrid-ai/xybrid#481)
* Added: speculative cloud fallback and download visibility —
  `fromRegistrySpeculative`, `willSpeculate`, `isCloudServing`,
  `downloadStatus`, `downloadProgress` (a push `StreamSink` of load events),
  `setPlatformUrl`, and `setSpeculativeCloud`. Results carry
  `executionTarget`, so a device answer is distinguishable from a gateway one;
  a pipeline reports its final stage's target (xybrid-ai/xybrid#459)
* Changed: ASR runs on whisper.cpp rather than Candle — 3.6x faster to a first
  partial on a Pixel 8, and the first partial itself arrives ~3.5 s sooner
  thanks to warm-up windowing. **The registry id `whisper-tiny` no longer
  loads**; use `whisper-tiny-ggml` (xybrid-ai/xybrid#462, xybrid-ai/xybrid#465,
  xybrid-ai/xybrid#476, xybrid-ai/xybrid#458)
* Changed: transcripts no longer contain bracketed non-speech annotations such
  as `[BLANK_AUDIO]`, audio longer than 30 seconds no longer repeats text past
  its end, and unsupported `prompt` metadata now returns an input error instead
  of being ignored (xybrid-ai/xybrid#484)
* Fixed: building the example for Android from source failed in
  `xybrid-whisper-sys` with `'stdio.h' file not found`; bindgen now gets the
  NDK sysroot (xybrid-ai/xybrid#468)

## 0.4.1

No Dart API changes. Fixes to the iOS build and to diagnostics on both mobile
platforms.

* Fixed: iOS builds linked the device ONNX Runtime slice when targeting the simulator and failed with `building for 'iOS-simulator', but linking in object file built for 'iOS'`. The simulator slice is now preferred, falling back to a checksum-pinned fetch of the same artifact (xybrid-ai/xybrid#450)
* Fixed: telemetry export failures were discarded silently and the binding installed no native log sink, so nothing reached logcat or the unified log. Export errors now log with attempt count and cause, and `Xybrid.init` installs `android_logger` on Android and `oslog` on iOS (xybrid-ai/xybrid#448)
* Fixed: registry URL failover logged at `debug`, below the default mobile log level, so a failing registry looked like a silent hang. It now logs at `warn` (xybrid-ai/xybrid#449)

## 0.4.0

Stable release of the 0.4.0 line. No Flutter API changes since `0.4.0-rc1`.

* Changed: the precompiled Windows native is now an MSVC-ABI DLL plus import library produced by the Bazel release graph, includes the llama.cpp vision path, and is load-tested on Windows before release (xybrid-ai/xybrid#416, xybrid-ai/xybrid#418, xybrid-ai/xybrid#419, xybrid-ai/xybrid#420)

## 0.4.0-rc1

Release candidate for 0.4.0.

* Fixed: the package would not build for anyone with a Rust toolchain installed — it failed with `error inheriting 'edition' from workspace root manifest`. cargokit disables precompiled binaries whenever `rustup` is on `PATH`, and this package cannot be built from source, so it fell through to a build that could never succeed. The published package now always uses its precompiled binaries, and a source build that cannot succeed fails with an explanation instead of a cargo error. `use_precompiled_binaries` still overrides (xybrid-ai/xybrid#338)

## 0.4.0-alpha

Prerelease exercising the new release pipeline.

* Fixed: `libxybrid_flutter.so` not found when compiling from source on Linux (xybrid-ai/xybrid#340)
* Changed: the precompiled binaries (desktop + mobile) are now built by Bazel with hermetic toolchains — same download, naming, and signature; the Linux `.so` now loads on older-glibc distros than before (xybrid-ai/xybrid#369, xybrid-ai/xybrid#371)

## 0.3.0

* Fixed: model cache clearing now reports the number of cache roots actually removed (previously counted scanned `.xyb` entries, ~0 for the nested registry-bundle layout), so "clear cache" no longer reports success when nothing was cached; `extracted/`, `hf/`, and `hf-hub/` stay co-located under a relative cache root (xybrid-ai/xybrid#309)

## 0.2.2

Structured output on Flutter. Local llama generation can now be constrained to a
JSON Schema so small models emit guaranteed-valid JSON for on-device data
extraction: `FfiGenerationConfig` gains a `grammar` field, and a new
`jsonSchemaToGbnf` helper converts a JSON Schema to the GBNF grammar the backend
enforces (xybrid-ai/xybrid#310, xybrid-ai/xybrid#311).

## 0.2.1

Vision (VLM) now runs on every Flutter target. `0.2.0` shipped on-device
vision on Android and iOS; `0.2.1` brings the native VLM backend
(`llm-llamacpp-vision`, llama.cpp's mtmd) to the **desktop** targets too —
macOS, Linux, and Windows — so a Flutter desktop app can run a vision-language
model out of the box, matching mobile (xybrid-ai/xybrid#296).

* Fixed: GGUF models with custom or non-standard chat templates now load and run — when llama.cpp's built-in template matcher rejects the embedded template, it is rendered via a real Jinja engine (minijinja) instead of failing (xybrid-ai/xybrid#304)

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
