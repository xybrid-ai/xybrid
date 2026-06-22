# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- **OpenUPM registry**: Publish Unity SDK to [openupm.com](https://openupm.com) for scoped registry install
- **Multimodal KV-prefix reuse**: the per-frame prefill cost lever for live vision — **deferred** from 0.2.0, not yet implemented.

---

## [0.2.0-rc1] - 2026-06-21

Release candidate for `0.2.0`. This is the stable `0.2.0` tree under a
prerelease tag — published across every distribution channel (crates.io,
pub.dev, Maven Central, SPM) so consumers can validate the vision/BoltFFI
release against real integrations before the final tag. No functional changes
from the `0.2.0` candidate — see the [0.2.0] entry below for the full change
set.

---

## [0.2.0-alpha] - 2026-06-19

Prerelease of `0.2.0` cut to validate the release pipeline and exercise the
new BoltFFI binding surface across every distribution channel (crates.io,
pub.dev, Maven Central, SPM) ahead of the stable tag. No functional changes
from the `0.2.0` candidate — see the [0.2.0] entry below for the full change
set.

---

## [0.2.0] - 2026-06-17

The vision release. xybrid gains an on-device multimodal stack — VLM inference,
real-time camera vision primitives, and streaming TTS — and the FFI surface is
re-platformed from UniFFI onto BoltFFI through a single shared facade. This is a
**breaking release** for binding consumers: the Swift / Kotlin / Java / C# / RN
bindings are now generated through `xybrid-bolt` + `xybrid-ffi-facade` rather
than UniFFI, and the run/envelope call shapes changed accordingly.

### Added

- **On-device vision foundation** (#245): VLM inference, real-time camera vision
  primitives, and streaming TTS land in the runtime. The vision pipeline is now
  unconditional rather than feature-gated (#263).
- **Vision envelopes through bolt** (#265): `Image` / `MultiPart` envelopes and
  typed capability errors are threaded through the BoltFFI bindings; generation
  config is now plumbed through `XybridModel.run` (#262).
- **Reachable streaming cancellation**: cancelling a streaming generation drives a
  real runtime abort end-to-end (`FfiCancellationToken` + options-aware streaming
  routing + sink-closed-as-cancel), so generation halts at the next token and
  releases the model lock. `UserCancelled` is the default abort outcome.
- **Preemptive cancel-and-replace slot** on the model handle: a new run can preempt
  the in-flight run (latest-frame-wins), so a live loop no longer head-of-line-blocks
  behind a stale frame.
- **Raw-frame `mtmd` path + `imageRaw` binding**: a packed-RGB `mtmd_bitmap_init`
  shim routes `ImageSource::Raw` through `mtmd` without per-frame JPEG re-encoding;
  the `imageRaw` envelope binding is exposed to Dart/FRB. The encoded `image` path
  is unchanged and remains the fallback.
- **Live-mode telemetry tagging + per-session sampler**: live inferences are tagged
  (`live_mode` + `frame_session_id`) and rate-limited by a per-session sampler
  (≈1 row/sec/session, TTL-bounded), so live sessions don't emit a telemetry row
  per frame.
- **Speculative cloud loader decision layer** (#250): `set_speculative_cloud` +
  `ModelLoader::with_speculative_cloud` / `will_speculate` let the loader begin a
  cloud execution while the local model is still downloading.
- **React Native binding** (#93, #260): a React Native binding, now ported onto
  BoltFFI alongside the other foreign-language bindings.
- **Async/suspend conveniences restored** (#269) for Swift and Kotlin load + run.

### Changed

- **FFI bindings migrated from UniFFI to BoltFFI** (#205) via a shared
  `xybrid-ffi-facade` — one canonical SDK→foreign-language translation feeding the
  Swift / Kotlin / Java / C# / WASM bindings. **Breaking** for binding consumers.
- **Executor decomposition**: LLM envelope and gen-config helpers deduped (#261),
  LLM telemetry extracted into `execution::llm_telemetry` (#251), and TTS chunking
  + audio crossfade extracted from the executor (#239).
- **iOS LiveVision example** migrated to the bolt `run()` shape (#267).
- **Docs**: docs site refreshed — restored deploys, surfaced hidden nav, added
  missing pages (#254); local-first foundation vs additive platform layer
  clarified (#248).
- **Release/CI**: `llama-cpp-sys` renamed to `xybrid-llama-sys` (#247) and both
  `xybrid-llama-sys` + `xybrid-llama` now publish to crates.io (#246); native
  build cache is warmed on master pushes (#268).

### Fixed

- **Kotlin image format validation** restored and `EnvelopeTest` fixed for the bolt
  envelope shape (#266).
- **`tokens_out` emitted** on local LLM telemetry paths (#253).
- **`.npz` voice files detected** by magic header rather than extension (#252).
- **TTS text chunking is UTF-8-safe** (#249) — no longer splits multi-byte
  codepoints mid-character.

---

## [0.1.2] - 2026-06-06

A robustness and supply-chain hardening release. The headline is a sweeping
panic-safety pass across `xybrid-core` and `xybrid-sdk` — poisoned locks,
unchecked arithmetic, and non-contiguous tensors no longer abort the process —
plus a wider set of audio input formats and a leaner, restructured native build.
No public API changes.

### Added

- **Audio format detection for MP3, OGG, and FLAC** (#132): `AudioFormat::detect_format`
  now recognizes these container formats in addition to WAV.
- **Mono → stereo upmixing** in `prepare_audio_samples` (#141): mono inputs are
  upmixed to stereo when a model expects two channels.

### Changed

- **`llama.cpp` integration split** into `llama-cpp-sys` + `xybrid-llama` crates (#166),
  separating the `-sys` build from the higher-level backend.
- **`resolve_file_path` consolidated** into `execution::path` (#238); the SDK chains
  error causes via `#[source]` instead of stringizing them (#220).
- **Generated native libraries (~125MB) are no longer committed** (#226): they are
  built/downloaded rather than vendored into git.
- **CI**: each release now ships a CycloneDX SBOM (#230); the release flow unblocks
  pub.dev publishing, the merge gate, and draft re-creation (#218).
- **Deps**: `console` 0.15 → 0.16 (#29); `base64` 0.21 → 0.22 (#34).

### Fixed

- **Panic-safety hardening across core and SDK**: poisoned-lock recovery instead of
  panicking in the llama.cpp `is_loaded` context lock (#237), the telemetry-session
  lock (#236), the SDK telemetry locks (#234), the event-bus locks (#233), and the
  routing-engine lock (#228); `with_retry` no longer panics when the circuit is open
  for every attempt (#227).
- **Checked arithmetic** in the WAV chunk parser (#232) and the voice-codes length
  header (#231), and **non-contiguous ONNX output tensors** are now handled without
  panicking (#235).
- **Keep the Xybrid API key out of the process environment** (#214).
- **Keep the test-fixtures fallback out of release builds** (#225).
- **Honor `Retry-After` on registry `429` responses** (#134).

### Docs

- Added governance, maintainers, dependency, and release-verification docs (#224),
  plus an OpenSSF Best Practices badge (#221).
- Documented the two `candle` unsafe blocks with SAFETY comments (#229).
- Examples inject `apiKey` + `ingestUrl` via platform-native env vars (#219); install
  versions synced and the stale Pipelines concept page removed (#222).

### Known issues

- **iOS Simulator slice still missing from the published xcframework**
  ([#179](https://github.com/xybrid-ai/xybrid/issues/179)): unchanged from 0.1.0.
  Swift consumers building against the iOS Simulator on Apple Silicon still need the
  `useLocalNatives = true` workaround after vendoring the ORT iOS simulator slice.

### Consumer install lines

```swift
// Swift Package Manager
.package(url: "https://github.com/xybrid-ai/xybrid", from: "0.1.2")
```

```yaml
# Flutter / pub.dev
xybrid_flutter: ^0.1.2
```

---

## [0.1.1] - 2026-05-30

First patch on the 0.1.0 line. Headline is the new `Xybrid.init()` entry point —
anonymous-by-default telemetry wired up uniformly across every binding — plus a
round of FFI soundness/safety hardening across the C ABI.

### Added

- **`Xybrid.init()` builder with anonymous-by-default telemetry** (#188): a single
  SDK entry point that starts telemetry from an API key, anonymous unless configured
  otherwise. Brought to every binding in lockstep: Swift `Xybrid.initialize()` (#196),
  Kotlin `Xybrid.init()` (#201), Unity `XybridClient.Initialize()` (#202), and the
  Flutter bundled `init()` (#195, which also marks the old `initTelemetry` legacy).
- **Error retryability across bindings**: inherent `SdkError::is_retryable` /
  `retry_after` (#198), surfaced to Swift and Kotlin through UniFFI (#200).
- **Typed `XybridOutputType` enum** for the result output kind in the C FFI (#194).
- **Telemetry stamps `sdk_version` and `binding`** on every `PlatformEvent` (#183),
  so events are attributable to the SDK build and language binding that emitted them.

### Changed

- **SDK**: one shared blocking body backs pipeline `run` / `run_async` (#210);
  platform detection deduplicated to a single `cfg` ladder (#206).
- **FFI**: handle-lifecycle helpers consolidated behind a macro (#192).
- **Docs**: READMEs and reference docs aligned with the bundled `init()` telemetry
  (#204); the Flutter example reads `XYBRID_API_KEY` at init (#207); SAFETY comments
  added to every `llama_cpp` unsafe block and impl (#191).
- **CI**: workflow token permissions scoped to least privilege (#211); native build
  workflows skipped on markdown-only changes (#208); docs deploy only when `docs/`
  changes (#186); apple release-prep jobs parallelized, NDK cached (#184); verify-release
  SPM + Flutter version parsing tightened (#182).

### Fixed

- **Redact Xybrid's own api-key prefix in telemetry** (#209): the SDK no longer leaks
  the leading bytes of its own key into emitted events.
- **Cache TTL clock handling is now panic-safe** (#203): a backwards clock no longer
  panics the cache layer.
- **FFI soundness and panic-safety**:
  - removed the unsound `unsafe impl Sync` from `StreamCallbackCtx` (#187);
  - every `extern "C"` body now guards against panics unwinding across the C ABI (#185);
  - accessor strings are cached in handle state to fix a use-after-free contract (#189).

### Known issues

- **iOS Simulator slice still missing from the published xcframework**
  ([#179](https://github.com/xybrid-ai/xybrid/issues/179)): unchanged from 0.1.0.
  Swift consumers building against the iOS Simulator on Apple Silicon still need the
  `useLocalNatives = true` workaround after vendoring the ORT iOS simulator slice.

### Consumer install lines

```swift
// Swift Package Manager
.package(url: "https://github.com/xybrid-ai/xybrid", from: "0.1.1")
```

```yaml
# Flutter / pub.dev
xybrid_flutter: ^0.1.1
```

---

## [0.1.0] - 2026-05-27

Production release of the 0.1.0 line. No code changes since rc4 — this release closes the rc series and finalizes the release toolchain that was iterated through rc1 → rc4.

### Release infrastructure (since rc4)

- **SLSA build provenance attestations** (#178): Every release asset (XCFramework zip, Android `.so` zip, all CLI binaries) is now signed and recorded in GitHub's transparency log via Sigstore. Consumers verify with `gh attestation verify <file> --repo xybrid-ai/xybrid`.
- **Consumer-side resolution verification** (#177): `just verify-release <version>` spins up minimal consumer projects in a tmp dir for each registry (SPM / Cargo / Flutter pub.dev / Maven Central) and runs end-to-end resolution against the published artifacts. Also exercises an iOS Simulator xcodebuild against `examples/ios/XybridExample`.
- **pub.dev OIDC binding moved to GitHub Actions environment** (#176): The trusted-publisher binding now gates on a `pub-dev-publish` environment claim rather than a tag-pattern claim, decoupling pub.dev publishes from the workflow trigger type. (See [#179](https://github.com/xybrid-ai/xybrid/issues/179) follow-up — full automation of pub.dev publishes pending.)
- **`workflow_dispatch` recovery path on `release-publish.yml`** (#175): If the `pull_request: closed` event doesn't reach Actions (race condition, deleted PR, etc.) the publish flow can be re-run manually with `gh workflow run release-publish.yml --field tag=v<version>`. The publish-release step is gated on `isDraft=true` so it's a no-op when the release is already live.

### Cumulative highlights — what 0.1.0 ships (vs. 0.1.0-rc3)

Everything that landed in rc4 is in 0.1.0:

- **`InferenceMetrics` across every binding** (INF-15 series, #120, #131, #135, #138, #139, #142): typed per-inference CPU / memory / GPU / wall-clock metrics now visible from Rust SDK, Kotlin + Swift (UniFFI), Dart (`XybridResult`), and Unity (C FFI accessors). Surfaced in the bundled Flutter demos and Unity docs.
- **Streaming-LLM cloud fallback uses live device signals** (#121): real CPU / memory / thermal pressure feeds the routing decision instead of static thresholds.
- **`ModelWarmup` telemetry events** (#158 + #164): `XybridModel::warmup` emits dedicated `ModelWarmup` spans; warmup events drain on event boundaries so they don't bleed into subsequent inferences.
- **`streaming` field hoisted to top-level `PlatformEvent`** (#162): downstream consumers no longer descend into metadata to filter streaming events.
- **GGUF backend label defaults to `llamacpp`** (#119): unannotated GGUF bundles attribute correctly in telemetry instead of showing `unknown`.
- **`Denormalize` postprocessing step** (#133): inverse of `Normalize`, useful for round-tripping model output back into input-space coordinates.
- **Release-branch flow** (#169, #171, #173): replaces the tag-driven release. `release-prep.yml` + `release-publish.yml` keep master's SPM checksum in sync, eliminate force-moved tags, and stage every release through a reviewable PR + draft release.

### Fixed

- **SPM `branch: "master"` consumers** unblocked (#167, #169): the new release-branch flow keeps master's `Package.swift` `xybridFFIChecksum` in sync with the released xcframework. The recommended consumer line is now `from: "0.1.0"`, but `branch: "master"` works too.
- Streaming fast-path `ModelComplete` events restored (#137), orchestrator pipeline-frame events filtered at SDK bridge (#146), CLI REPL routes cached models locally (#165), warmup span collector drains on event boundary (#164) — all from rc4.

### Known issues — deferred to v0.1.1

- **iOS Simulator slice missing from the published xcframework** ([#179](https://github.com/xybrid-ai/xybrid/issues/179)): Swift consumers cannot build against the iOS Simulator on Apple Silicon without a workaround. Pre-existed in rc1 through rc4. Workaround: build locally with `useLocalNatives = true` after vendoring the ORT iOS simulator slice.
- **pub.dev publish requires one manual step**: `flutter pub publish -f` from a maintainer's machine after merging the release PR. Refactor tracked separately.

### Consumer install lines

```swift
// Swift Package Manager
.package(url: "https://github.com/xybrid-ai/xybrid", from: "0.1.0")
```

```yaml
# Flutter / pub.dev
xybrid_flutter: ^0.1.0
```

```toml
# Rust / crates.io
xybrid = "0.1.0"
```

```kotlin
// Kotlin / Maven Central
implementation("ai.xybrid:xybrid-kotlin:0.1.0")
```

```sh
# Unity / UPM
https://github.com/xybrid-ai/xybrid.git#upm
```

---

## [0.1.0-rc4] - 2026-05-26

### Added

- **`InferenceMetrics` on result types across every binding** (INF-15 — #120, #131, #135, #138): Typed per-inference metrics (CPU / memory / GPU / wall-clock) are now exposed on the SDK result type and threaded through to Kotlin + Swift (UniFFI), Dart (`XybridResult`), and Unity (C FFI accessors). Flutter demos and Unity docs now surface them end-to-end (#139, #142).
- **Live-signal routing for streaming cloud fallback** (#121): The streaming-LLM fallback policy now consumes real-time device pressure signals (CPU / memory / thermal) instead of static thresholds when deciding whether to spill to cloud.
- **`ModelWarmup` telemetry event** (#158): `XybridModel::warmup` now emits a dedicated `ModelWarmup` span; the CLI REPL routes its warmup through this event so first-token latency is attributable to warmup vs. inference.
- **`streaming` field hoisted to `PlatformEvent` top-level payload** (#162): Previously nested under metadata, now a top-level field so downstream consumers don't have to descend into the payload to filter streaming events.
- **GGUF backend label defaults to `llamacpp` on unannotated bundles** (#119): Telemetry events from bundles that don't carry an explicit backend tag now default to `llamacpp` rather than `unknown`, so dashboards correctly attribute GGUF traffic.
- **`Denormalize` postprocessing step in core** (#133): New core postprocessing primitive that inverts a `Normalize` step, useful for round-tripping model output back into input-space coordinates.

### Fixed

- **`ModelComplete` events on streaming fast-path inference** (#137): The streaming fast-path was skipping the `ModelComplete` emission, leaving downstream consumers waiting on a terminal event that never arrived. Now emitted on every path.
- **Orchestrator pipeline-frame events filtered at SDK bridge** (#146): Internal `PipelineFrame` events from the orchestrator no longer leak to binding consumers as opaque payloads.
- **REPL routes cached models locally** (#165): The CLI REPL was occasionally re-resolving cached models through the cloud router; it now short-circuits to the local cache when the model is present on disk.
- **`ModelWarmup` span collector drained on event boundary** (#164): Warmup spans were leaking into the subsequent event's batch; the span collector is now drained when `ModelWarmup` is published.
- **SPM consumers on `branch: "master"` no longer hit checksum mismatch** (#167, #169): The new release-branch flow keeps master's `Package.swift` `xybridFFIChecksum` in sync with the released xcframework asset. Tag-pinned (`exact:` / `from:`) and `branch: "swift"` consumers were unaffected; this fixes the `branch: "master"` case that had been silently broken since rc1.

### Build / CI

- **Release-branch flow** (#169, #171): New `release-prep.yml` + `release-publish.yml` workflows. A maintainer cuts `release/v<version>`, runs `just bump-version`, and pushes — CI builds every artifact, patches the SPM checksum back to the branch, creates a draft GitHub Release with all assets, and opens a PR to master. Merging the PR publishes the draft (tag created at merge commit) and publishes to crates.io / pub.dev / Maven Central. The legacy `release.yml` is kept as a `workflow_dispatch`-only break-glass.
- **`version-sync.sh` now bumps `bindings/flutter/rust/Cargo.toml`** (#173): `just bump-version` was silently leaving the Flutter rust crate behind because the crate hardcodes its version (cargokit hashes the file). The bump script now keeps it in sync; master's previously-stale rc1 version is brought up too.
- **`publish-crates` job pushes the four crates to crates.io** (#143, #145): `xybrid-macros`, `xybrid-core`, `xybrid-sdk`, and the `xybrid` umbrella now publish from the release workflow.
- **Discord notifications + contributor welcome workflow** (#147, #148): Release publish notifies the project Discord; new contributors get a welcome message on their first PR.

### Docs

- **Vision envelopes + multi-part user messages** (#123): SDK docs now cover the input shape for vision payloads and the multi-part message format.
- **`XYBRID_LLAMACPP_VERBOSITY` env var documented** (#156).
- **Doctest examples compile under `no_run`** (#168): All public-API doctests now compile cleanly even without runtime dependencies present, so `cargo test --doc` runs green in CI.
- **README install snippets bumped to 0.1.0-rc4** (this release, see also #157 for the rc3 equivalent).
- **New-contributor pointers** (#130): READMEs now point first-time contributors at the `good-first-issue` and area labels.

---

## [0.1.0-rc3] - 2026-05-16

### Added

- **Adaptive cloud fallback for streaming LLM** (#114): Streaming LLM pipelines can now transparently fall back to a cloud runtime when on-device generation stalls or errors mid-stream. New `RunOptions` controls expose the fallback policy on the SDK; the cloud runtime adapter, llama.cpp adapter, mistral adapter, and orchestrator authority layer all participate in the new flow.

### Fixed

- **Backend and quantization tags on streaming LLM spans** (#118): Telemetry spans emitted from streaming and chat-context LLM execution now carry backend and quantization labels (previously dropped on these code paths), so dashboards correctly attribute traffic to the runtime that actually served the request.
- **Hybrid LLM architecture support in llama.cpp adapter** (#109, #117): Skip KV prefix-reuse and broaden the recurrent-state gate so hybrid (Mamba / SSM-style) architectures load and run cleanly through the llama.cpp runtime adapter. Adds an `llm_context_integration` test to lock in the behavior.

### Build / CI

- **Inline Flutter publish in release workflow** (#116): The release workflow no longer composes `publish-flutter` as a reusable job — inlining it avoids the `actions/checkout@v6` ref-moved guard that was triggered by the rc2 manifest-checksum self-patch (which force-moves the tag).

---

## [0.1.0-rc2] - 2026-05-14

### Fixed

- **Apple device detection compile on non-Apple aarch64 targets** (#112): `detect_apple_device` in `xybrid-core` now compiles on `aarch64-linux-android` and `aarch64-unknown-linux-gnu`. The missing tail expression triggered E0317 on those targets, which silently blocked `Build Unity → Build Android Libraries` and the Release workflow's `Precompile Flutter (linux)` job in 0.1.0-rc1.

### Release

- **Flutter (pub.dev) and Unity (UPM)** ship for 0.1.0-rc2. Both were skipped in 0.1.0-rc1 because the aarch64 compile failure above blocked their upstream build jobs; no code/API changes in the Flutter or Unity bindings themselves.

### Build / CI

- **Release workflow self-patches the SPM manifest checksum** (#111): the XCFramework SHA computed by CI is now written into `xybridFFIChecksum` in `Package.swift`, committed, and the tag is force-moved to the patched commit. Removes the chicken-and-egg between the tag-time manifest and the CI-rebuilt zip bytes.

---

## [0.1.0-rc1] - 2026-04-30

### Added

- **Per-inference resource telemetry** (#53): `xybrid-core` exposes CPU / memory / GPU pressure metrics per inference and `xybrid-sdk` folds them into telemetry events; new `device::resource` module with pressure sampling and a Criterion bench.
- **Provider-agnostic prompt-cache token counts** (#52): Cloud LLM responses report `cache_creation` / `cache_read` token counts uniformly across providers; legacy field names continue to deserialize.
- **Registry telemetry header** (#60): `X-Xybrid-Client` header on registry calls advertises binding, SDK version, core version, platform, and enabled backends; honors `XYBRID_TELEMETRY_OPTOUT`. Binding identifier is wired through Flutter, Kotlin, Swift, Unity, and Rust; CLI gains `xybrid telemetry status`.
- **Unity / C# telemetry surface** (#56): `TelemetryConfig` API with Editor domain-reload guard, configuration sample scene, and Editor tests; C FFI exposes telemetry init / config / event hooks.
- **Swift Package Manager root manifest** (#62): Top-level `Package.swift` with a `useLocalNatives` toggle for switching between published binaries and local-built XCFrameworks; `set-natives-mode.sh` and `sync-spm-checksum.sh` helpers.
- **Localized docs**: Japanese README (#58) and Chinese localization for newly added documentation pages (#55).

### Fixed

- **Apple SPM MPS link** (#62): Corrected Metal Performance Shaders linker flags so the Apple SPM target builds cleanly against the unified XCFramework.

---

## [0.1.0-beta12] - 2026-04-20

### Added

- **LLM telemetry expansion** (#45): Swim-lane spans, device profile metadata, and Pipeline::run hardening build on the streaming telemetry landed in beta11.

### Fixed

- **Flutter Windows publish** (#48): Fixed backslash mangling in the Windows precompile CI job that was stripping `\a`, `\x` from `${{ github.workspace }}` when bash parsed the path, preventing Flutter native binaries from being published to pub.dev.
- **Example crate name collision** (#49): Renamed `voice_assistant_demo` in `xybrid-core` examples to avoid a conflict with the example of the same name in `xybrid-sdk`.

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
