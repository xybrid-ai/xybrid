# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- **Multimodal KV-prefix reuse**: the per-frame prefill cost lever for live vision — **deferred** from 0.2.0, not yet implemented.

---

## [0.6.0] - 2026-08-25

Tool calling reaches every binding, and an external `cargo build --features
llm-llamacpp` stops compiling llama.cpp. FunctionGemma joins the local
tool-calling backends, tool calls cross the FFI boundary into Swift, Kotlin,
Python, Unity C# and Dart, and `supportsToolCalling` lets an app gate its tool
UI on what a bundle actually declares. Separately, `xybrid-llama-sys` resolves a
prebuilt, SHA-256-verified llama.cpp slice over plain HTTPS — no oras, no
environment variable, no CMake — turning the dominant cold-build cost from
roughly twenty minutes into a download.

**Upgrade notes.** The HuggingFace cache moves to a repository-hash layout, so
caches written by 0.5.0 and earlier are not reused: the next
`from_huggingface(...)` load re-downloads the model, and the old copy stays on
disk until it is evicted. It remains listed under its raw on-disk label
(`owner--repo` under `hf/`, `models--owner--repo` under `hf-hub/`) and is
removed by a targeted eviction on that label. Kotlin callers should drop
`import ai.xybrid.reasoningContent` — `reasoningContent` is now a member of
`XybridResult` rather than an extension property, which is source-compatible at
the call site but needs a recompile against the new AAR.

### Added

- **Prebuilt llama.cpp for a plain `cargo build`.** `xybrid-llama-sys` resolves
  its native archives in three steps — an explicitly staged
  `XYBRID_NATIVES_PREBUILT_DIR/<target>`, then a slice named in the generated
  `natives-manifest.txt`, then the CMake source build. The middle step is new:
  it fetches the layer over plain HTTPS from `ghcr.io/xybrid-ai/llama-natives`
  and verifies its SHA-256, needing no oras, no environment variable and no
  CMake, which takes the dominant cold-build cost for an external
  `--features llm-llamacpp` consumer from roughly twenty minutes to a download.
  Slices are cached by digest under `$CARGO_HOME/xybrid-natives/` and shared
  across projects. Every miss falls through to the source build, so the fast
  path can only fail to accelerate a build, never break one; the manifest pins
  the sha256 of `build.rs`, `wrapper.cpp`, `wrapper.h` and the llama.cpp commit,
  so a local edit to any of them disarms every row until CI republishes. Set
  `XYBRID_NATIVES_FORCE_SOURCE=1` to opt out (#526).
- **FunctionGemma tool calling.** The local llama.cpp backend recognizes
  FunctionGemma's call protocol, including its space-separated form, alongside
  the existing tool-calling models (#512).
- **Tool calls cross the FFI boundary.** Tool definitions travel out on
  `RunOptions` and parsed calls come back on the result, so Swift, Kotlin,
  Python and Unity C# callers drive a tool loop turn by turn — the caller
  executes the tool and issues another `run`, with no cross-boundary callback
  (#513).
- **`supportsToolCalling` on every binding.** `XybridModel` surfaces the
  bundle's `tool_calling` metadata flag as an advisory tri-state — `null` means
  the bundle says nothing — so an app can gate its tool UI on model capability.
  Available on Swift, Kotlin, Python, Unity C# and Flutter. Enforcement stays at
  run time: a tools-bearing request against a model whose chat template has no
  tool support fails either way (#515).
- **Grammar, `jsonSchemaToGbnf` and `reasoningContent` in Dart.** All three had
  reached the generated layer and stopped at the hand-written Flutter wrapper.
  `GenerationConfig.grammar` now passes through, the greedy and creative presets
  take an optional grammar so the usual extraction shape is one call,
  `jsonSchemaToGbnf` is re-exported, and `XybridResult.reasoningContent` is
  readable. The Dart unit tests under `bindings/flutter/test` also run in CI for
  the first time (#511).
- **`TemperatureSample` postprocessing step.** Pipeline templates can sample a
  token from logits with temperature, optional top-k and optional top-p, rather
  than taking the argmax — sampling from the final sequence position, with
  temperature zero preserved as exact argmax (#521).
- **A runnable Flutter example app.** `xybrid_flutter` ships a real
  single-screen app in `example/` instead of a snippet file, so a consumer of
  the published package has something to run (#525, #152).
- **Reasoning text as a typed field on the Bolt bindings.** `XybridResult`
  gains `reasoningContent`, carrying what a thinking model emits separately
  from its answer, on Swift, Kotlin, Python, and Unity C#. It is appended last
  on the wire — `#[data]` PODs serialize in declaration order — and the
  envelope keeps its `reasoning_content` metadata, so a consumer reading that
  metadata is unaffected. Each generator's decoder probes for the tail before
  reading it and falls back to the metadata when it is absent (#508).
- **Conversation history readback.** `ConversationContext.history()` returns
  the turns a context holds, excluding the persistent system envelope, on
  Swift, Kotlin, Python, and Unity C# (#508).
- **Revision-pinned HuggingFace loading.** `from_huggingface_with_revision`
  resolves a branch, tag, or commit SHA to an immutable commit and pins the
  load to it. Pinned and mutable refs occupy separate cache namespaces, so
  `main` and a commit of `main` no longer alias, and a resolved revision that
  is already materialized still loads when the Hub is unreachable (#508).
- **The resolved default generation config.** `defaultGenerationConfig` returns
  what a `run*` call uses when given no explicit config — template
  `generation_params` layered over global defaults, including the
  reasoning-budget floor — so a caller building an explicit config can start
  from the model's own defaults rather than `GenerationConfig::default()`. It
  no longer requires an LLM backend feature to be compiled in, and reads
  without waiting on an in-flight run (#508).

### Changed

- **The HuggingFace cache is keyed by repository hash.** Both `hf/` and
  `hf-hub/` directories move from a slash-to-`--` encoding (`owner--repo`,
  `models--owner--repo`) to `repo--<sha256(repo)>`, with the repository id
  recorded in a `.repo-id` marker inside each directory. The old encoding was
  not injective — `a/b--c` and `a--b/c` both wrote to `a--b--c` — so two
  unrelated repositories could share one cache. Directories under the old
  layout are not adopted, because their label cannot be decoded back to a
  repository id; they stay visible to cache listing and evictable under that
  raw label. See the upgrade note above (#508).
- **Envelope metadata crosses the FFI boundary in a deterministic order.**
  `XybridEnvelope.metadata` is sorted by key rather than emitted in `HashMap`
  iteration order, so two conversions of the same envelope produce identical
  wire bytes (#508).

### Fixed

- **Streaming stop-boundary panics.** Three fixes in `StreamingTextFilter`'s
  `last_emitted_len` invariant: a stop sequence completing across chunks could
  land the emission boundary below already-emitted text, after which the
  cumulative slice went out of bounds and panicked mid-stream. The boundary is
  now clamped after a complete-stop truncation, and the potential-stop scan
  holds the longest matching tail prefix per pattern and the earliest hold
  across patterns, rather than the shortest prefix of the first pattern that
  matched — the under-hold that let a later chunk complete a stop behind the
  boundary (#518).
- **A second ggml bundled into staticlib targets.** `flutter build macos` failed
  on the example app with 1287 duplicate `ggml_*` symbols: one Rust staticlib
  carried two copies, from `whisper-cpp-sys`'s deliberate ggml re-emission on
  top of the one llama.cpp already links (#528).
- **Published native slices are portable.** llama.cpp's `GGML_NATIVE` defaults
  on for non-cross builds, so each published slice inherited the CPU of the
  runner that built it — the x86_64 Linux slice carried AVX-512 and AMX, and the
  aarch64 Linux slice carried SVE, either of which is an illegal-instruction
  trap on a consumer without them, while the cross-built darwin x86_64 slices
  had the inverse problem and shipped scalar ggml. Slices now pin an explicit
  x86-64-v3 baseline (SSE4.2/AVX/AVX2/BMI2/FMA/F16C) on non-Android x86_64 and
  plain armv8-a on aarch64, so no consumer traps and the cross-built slices
  regain SIMD (#534).
- **Windows slice publishing and darwin cross-arch linking.** GNU tar on the
  Windows runner read the `D:` drive-letter prefix as a remote host, so neither
  `x86_64-pc-windows-msvc` slice ever published; and llama.cpp's vendored
  cpp-httplib defaults `LLAMA_OPENSSL` on, so the arm64 macOS runner
  cross-building x86_64 picked up an arm64 libcrypto and failed to link. The
  slices ship only static archives, so tool-binary TLS is now off (#529).
- **`clippy::chunks_exact_to_as_chunks` under Rust 1.98.** Constant-size slice
  chunking moves to `as_chunks`, restoring a green `-D warnings` build on the
  stable toolchain (#522).
- **crates.io publish ordering.** The whisper crates publish before
  `xybrid-core`, which is what broke v0.5.0's publish run, and the generated
  Python binding's `PACKAGE_VERSION` is synced by the version bump rather than
  being left behind for `gen_python_bolt.py --check` to reject (#488, #490).

### Performance

- **Metal for whisper.cpp on Apple silicon.** Measured inference latency drops
  2.7-7.8x on an M1 Pro with no word-level regressions. Other targets stay on
  CPU until measured independently (#491).
- **Cached stream language detection.** A streaming session detects its language
  once and reuses the result across windows instead of re-detecting per chunk
  (#493).

---

## [0.5.0] - 2026-08-11

Speech recognition moves off Candle and onto whisper.cpp, running on the same
ggml that llama.cpp already links — 3.6x faster to a first partial on a Pixel 8
for 0.2 MiB of binary, and the first time Linux and Windows have local Whisper
at all. Candle is retired from all four platform presets as a result; the
features remain opt-in, so this removes a default, not a capability. Live
streaming ASR lands on Flutter with warm-up windowing and a span-reconciled
transcript, every binding gains the speculative-cloud surface that 0.4.1 shipped
Rust-only, all bindings move to BoltFFI 0.29.3 (the Python SDK is now generated
rather than hand-ported), and desktop Linux gets an opt-in Vulkan build of
llama.cpp.

**Upgrade notes.** Registry model id `whisper-tiny` resolves to a
SafeTensors/Candle bundle and no longer runs on a default build — use
`whisper-tiny-ggml`, a multilingual Q5_1 GGML bundle served as its own id, or
build with `--features candle`. The Python SDK now requires **Python >= 3.10**
and ships one wheel per interpreter ABI.

### Added

- **whisper.cpp speech recognition on the shared ggml.** A three-layer stack
  mirroring the llama.cpp one — `crates/whisper-cpp-sys` (raw FFI, compiled
  with `cc` against llama's ggml headers, never whisper's bundled copy),
  `crates/xybrid-whisper` (safe RAII handle, no `unsafe` on the public surface),
  and a `WhisperCppRuntime` in core behind `ExecutionTemplate::GgmlWhisper`.
  Measured on a Pixel 8 (5 s window, tiny.en Q5_1, 4 threads): 744 ms against
  ~3.5 s for the Candle path. Note the container: whisper.cpp verifies
  `GGML_FILE_MAGIC` and never moved to GGUF, so these are `ggml-*.bin` files
  (#462).
- **whisper.cpp is built by Bazel and enabled in every platform preset.**
  `//:whisper` is a plain `cc_library` on `//:llama` rather than a second
  `cmake()` target, so "exactly one ggml" is a property of the build graph and
  CI fails if that stops being true. Cost on the stripped cdylib: +0.2 MiB
  (#465).
- **Live streaming ASR on Flutter**, with the rolling-window streaming API
  bound through `flutter_rust_bridge` and a demo screen in the example app
  (#453).
- **Warm-up windows and a span-reconciled transcript for live streaming.** The
  first chunks use growing windows (1.5 s, 3 s) instead of waiting for a full
  5 s one, landing the first partial ~3.5 s sooner, and the accumulator tracks
  the audio span each segment covers so overlapping windows replace rather than
  repeat earlier text (#458).
- **An `audio_ctx` streaming override**, threaded through core `StreamConfig`,
  the SDK builder, and the Flutter binding, so a caller can trim the Whisper
  encoder window without repackaging the model bundle. `None` keeps the bundle
  default (#481).
- **Speculative cloud, download progress, and result provenance on every
  binding.** `ExecutionProvenance` on `InferenceResult` tells a device answer
  from a gateway answer, `DownloadState`/`DownloadStatus` plus
  `download_status`/`await_download`/`is_cloud_serving` expose the background
  download, and `set_speculative_cloud` / `from_registry_speculative` /
  `set_platform_url` reach Swift, Kotlin, Unity C#, Python, React Native, and
  Dart. Progress is polled everywhere except Flutter, which gets a push
  `StreamSink`: FRB sinks are safe, while boltffi's inline-closure return ABI
  is not, so no closure crosses that boundary (#459).
- **llama.cpp Vulkan builds for desktop Linux**: consumers can now set
  `XYBRID_LLAMA_CPP_VULKAN=1` when building `platform-desktop` to compile the
  bundled backend with `GGML_VULKAN=ON`; local LLM telemetry reports `vulkan`
  for those builds. Windows is not supported yet — ggml builds its GLSL
  compiler as a nested CMake project whose paths exceed Windows' 260-character
  limit under cargo's `OUT_DIR`, so the build fails wherever the repo is
  checked out. Every other target rejects the variable with a clear error.
- **A Vulkan lane on the Bazel graph** (`--config=linux-vulkan`). The Linux CLI
  builds from Bazel, which writes llama.cpp's cmake defines itself and never
  reads `XYBRID_LLAMA_CPP_VULKAN` — so the environment variable above reaches
  cargo builds only. The new `--//:vulkan` flag selects `//:llama_vulkan`,
  mirroring how `--//:metal` selects `//:llama_metal`, and a CI job builds and
  smokes the resulting CLI (Vulkan backend symbols present, `libvulkan.so.1` a
  real link dependency). That target runs on the local machine rather than a
  remote worker, because ggml's Vulkan build needs `glslc`, the Vulkan headers
  and a host compiler where cmake runs, and the remote image has none of them.
  Published Linux binaries are unchanged and stay CPU-only: a Vulkan build
  requires a Vulkan loader on the machine that runs it, so it belongs in a
  separate artifact rather than in place of the default one.
- **A warning when a GPU offload request cannot be honored.** `gpu_layers`
  defaults to 99 and llama.cpp keeps every layer on the CPU when no GPU backend
  was compiled in, so until now the only symptom of a CPU-only build on a
  machine with an idle GPU was that inference ran slowly. Model load now logs
  this once, pointing at the platform's GPU build options (#485).
- **A Device Logs guide** — where SDK logs land per platform (logcat tag
  `xybrid`, unified-log subsystem `dev.xybrid.sdk`, host-registered logger on
  desktop), the commands to read them, and what the telemetry-export and
  registry-failover warnings look like (#457).

### Changed

- **Candle is retired from all four platform presets.** whisper.cpp supersedes
  it for ASR at a fraction of the cost — measured on the stripped cdylib,
  Candle is 1.3–1.9 MiB against whisper.cpp's 0.2 MiB, 6.5–9.5x its
  replacement — and Whisper was its only live model here. The `candle*`
  features stay declared and buildable as an opt-in. Consequence for callers:
  the registry id `whisper-tiny` (a SafeTensors bundle) no longer loads on a
  default build, and fails with a message naming both the feature and the GGML
  alternative rather than a bare `Runtime 'candle' not configured`. The demos
  and examples point at `whisper-tiny-ggml` (#476).
- **whisper.cpp transcription is hardened**, with three user-visible changes:
  bracketed non-speech annotations such as `[BLANK_AUDIO]` are suppressed
  rather than emitted as transcript text (a caller-facing opt-out is tracked in
  #483); timestamp tokens are decoded for audio longer than 30 seconds, so a
  padded final window no longer repeats text past the end of the real audio;
  and unsupported `prompt` metadata returns `InvalidInput` instead of being
  silently ignored, without echoing the prompt back. Twelve real-model
  regression cases move off Candle onto a checksum-pinned multilingual GGML
  model, covering WER, window boundaries, 66-second input, per-request language
  and translation, and the two behaviours above (#484).
- **Streaming sessions share the model's loaded executor.** Each session used
  to build its own, reloading whisper weights from disk on every open (~2.5 s),
  and then ran a full silent warm-up inference even when the weights were
  already resident. `ModelHandle.executor` is now shared, `ModelRuntime` gained
  `is_loaded`, and the warm-up returns early when the executor is already warm,
  so second and later sessions open with no hidden compute. Unloading mid-stream
  swaps in a fresh executor rather than pulling weights out from under a running
  transcription (#461, #464).
- **BoltFFI 0.25.3 → 0.29.3.** Every exported C symbol is renamed
  (`boltffi_set_api_key` → `boltffi_function_xybrid_bolt_set_api_key`), so all
  generated bindings — Swift, Kotlin, Unity C#, Python — were regenerated
  together; the wire format itself is unchanged. The Cargo package is renamed
  `xybrid-bolt` → `xybrid_bolt` because 0.29's C# generator rejects the
  hyphenated name; the cdylib and Bazel `crate_name` were already
  `xybrid_bolt`, so no artifact moved. Unity's 576-line hand-ported inference
  supplement collapsed to ~85 lines, since 0.29 emits the inference path 0.25.3
  dropped.
- **Python SDK on generated bindings.** The 1,900-line hand-ported ctypes wire
  layer is gone; `xybrid/_bolt/` is boltffi output, regenerated by
  `tools/scripts/gen_python_bolt.py` and byte-compared in CI. The SDK's
  Pythonic surface (envelope factories, `result.text`, model properties, typed
  `xybrid.ModelNotFound`-style exceptions) is unchanged for callers but now
  lives in `xybrid/_sugar.py` and `xybrid/_errors.py`. **Breaking for the
  distribution**: 0.29's Python target compiles a CPython extension rather than
  emitting ctypes, so the SDK requires **Python >= 3.10** and ships one wheel
  per interpreter ABI instead of a single `py3-none` wheel.
- **Docs corrected for the Candle retirement.** Preset tables, platform
  rationales, and every runnable example that named the now-unloadable
  `whisper-tiny`; the `candle` feature-reference rows stay and now say
  explicitly that they are opt-in. Also fills gaps that predate the change:
  `asr-whispercpp` and `ExecutionTemplate::GgmlWhisper` were absent from these
  docs despite shipping in every preset, telemetry's `backend` enum lacked
  `whispercpp`, and the feature matrix claimed every preset is text-only when
  they have carried the llama.cpp vision path for far longer (#479).

### Fixed

- **The READMEs claimed CUDA acceleration on Linux and Windows, which no build
  provides.** `GGML_CUDA` is `OFF` on every target in both the cargo and Bazel
  paths, `llm-mistral-cuda` is a marker feature whose backing crate is commented
  out of the workspace, and Candle — the one component with a real CUDA path —
  was retired from the platform presets. The hardware-acceleration table now
  reads CPU for Linux (with Vulkan as a build-time opt-in) and CPU for Windows,
  in all three translations (#485).
- **A gateway request with no model no longer silently runs OpenAI.** Both
  request builders fell back to `gpt-4o-mini`, so a caller who forgot to set a
  model paid for a third-party provider and saw the resulting failure as an
  opaque gateway 502 rather than a client bug. It is now an `InvalidInput`
  error. The per-provider defaults stay, since there the caller has already
  chosen the provider (#463).
- **Swift and Kotlin SDKs register a native log sink.** `xybrid-bolt` never
  registered a log backend, so every `log::warn!` in the SDK — telemetry send
  failures, registry failovers — was discarded on those two SDKs, the same hole
  #448 closed for Flutter (#452).
- **The cargo-built CLI had no whisper.cpp on any platform.** Its manifest
  re-listed its own backend sets instead of forwarding to the SDK presets, so
  `asr-whispercpp` never reached it and only the Bazel-built CLI had ASR. It
  now forwards, removing the drift class rather than this instance of it
  (#476).
- **`asr-whispercpp` is advertised in the registry client's backend list.** It
  was missing from `ALL_FEATURES`, so the `backends=` header under-reported the
  backend set for as long as the backend had existed (#476).
- **Cross-compiling whisper.cpp gave bindgen the wrong sysroot.** A cargo
  Android build — which is what cargokit does for Flutter — died on
  `ggml.h:214:10: fatal error: 'stdio.h' file not found`, because libclang
  resolves headers against the host sysroot. `xybrid-llama-sys` now emits
  `cargo:ndk=<path>` and the whisper crate consumes it, keeping one
  NDK-resolution implementation in the workspace; the Apple `-isysroot` and
  simulator-triple handling come along in the same helper. CI missed it because
  every Android lane goes through Bazel, which compiles committed bindings and
  never runs bindgen (#468).
- **`gen_python_bolt.py` no longer deletes the staged native artifacts.** It
  pruned every file it had not generated, so regenerating removed the compiled
  bridge and the cdylib beside it and broke `import xybrid`.
- **`XYBRID_FEATURES` reaches the Python native build again.** The staged
  cdylib comes from the wheel `boltffi pack python` builds, so features are now
  forwarded to that build (`--cargo-arg`) instead of only to a separate
  `cargo build` whose output was discarded.
- **Three workflows no longer fail in every fork.** Unity Editor CI, the
  Bazel Windows Flutter DLL smoke, and the weekly Build Natives cron each
  assumed a non-`pull_request` event implies secrets are present, which is
  false for a contributor syncing their fork's master and for inherited crons.
  None of them gated anything here, but they buried real failures in forks
  under noise a contributor cannot fix (#480).

---

## [0.4.1] - 2026-08-05

Patch release on the 0.4.0 line. The headline is the explicit model-loading
lifecycle restored on Swift and Kotlin: the documented
`Xybrid.model(...).load()` flow existed on Flutter, Rust, and Unity but not on
those two SDKs, so their published quickstarts described an API 0.4.0 did not
ship. Token streaming also reaches Apple, Kotlin, and React Native, speculative
cloud fallback lands, and the mobile telemetry pipeline is now device-verified
on both platforms.

### Added

- **Token streaming on Apple, Kotlin, and React Native.** The pull-based
  streaming sessions that 0.4.0 introduced on the bolt surface (`run_stream` →
  `stream_next` / `stream_result` / `stream_close`, shipped for Unity) now
  reach the remaining bindings: Apple `model.streamTokens()`
  (`AsyncThrowingStream`), Kotlin `model.streamTokens()` (`Flow`), and React
  Native `model.runStreaming()` (async generator). The regenerated Swift and
  Kotlin bolt bindings surface the streaming methods. Stopping consumption —
  breaking out of the loop, cancelling the task/coroutine, or releasing the
  model — closes the session and aborts generation at the next token boundary
  instead of running to `max_tokens`; mid-stream failures carry the same typed
  errors as `run` (#438).
- **Speculative cloud fallback.** When a registry model has not been downloaded
  yet and an API key is present, the SDK serves inference from the platform
  gateway while the weights download in the background, then switches to local
  transparently — exposed as `run`/`repl --speculative-cloud`. Both this path
  and the existing reactive cloud fallback now send the registry model id to
  the gateway so it routes to the same model running on the CPU cluster,
  instead of defaulting to a hosted third-party model (#264).
- **Grammar-constrained output on React Native.** `GenerationConfig.grammar`
  now crosses the TurboModule boundary instead of being silently dropped, and
  `jsonSchemaToGbnf()` is exposed in JavaScript, closing the structured-output
  gap that Swift, Kotlin, C, and Dart already had (#442).

### Fixed

- **Swift and Kotlin regain an explicit model-loading step.** `Xybrid.model(id)`
  and `Xybrid.model(source)` return a cheap, unloaded `ModelLoader`; all
  network, disk, and runtime initialization happens at a named `load()`
  boundary, with `loadSync()` / `loadBlocking()` for worker threads. Registry,
  bundle, directory, and Hugging Face sources are strongly typed through
  `ModelSource`. The direct constructors and factories still work, so existing
  code compiles unchanged; the hidden-loading async factories are deprecated in
  favour of the loader flow (#451).
- **Flutter's iOS build linked the wrong ONNX Runtime slice.** Simulator builds
  resolved to the device `ios-arm64` slice and failed to link. The build now
  prefers the simulator slice, falling back to a checksum-pinned fetch of the
  same artifact the Bazel graph uses (#450).
- **Telemetry export failures were silently discarded.** Exporter transport
  errors are now logged with attempt count and cause, and the Flutter binding
  installs a native log sink on both platforms (`android_logger` on Android,
  `oslog` on iOS) so those lines reach logcat and the unified log (#448).
- **Registry URL failover was invisible on mobile.** It now logs at `warn`
  rather than `debug`, which the default mobile log level hides. Both mobile
  platforms are now device-verified end to end for the metrics pipeline (#449).
- **Unity release packaging failed on staged Bazel outputs.** Bazel's outputs
  are read-only; they are now made owner-writable before post-processing, which
  unblocks the Linux, macOS, and iOS strip steps (#432).

### Changed

- **`xybrid-llama-sys` declares `links = "llama"`.** A dependency graph pulling
  in both this crate and crates.io's `llama-cpp-sys-2` now fails at resolution
  time, naming both packages, instead of static-linking two copies of the same
  ggml/llama archives into one binary. No effect on compiled output (#441).
- Dependency bumps: `schemars` 0.8.22 → 1.2.2, `tokenizers` 0.19.1 → 0.22.2,
  `dialoguer` 0.11.0 → 0.12.0.

---

## [0.4.0] - 2026-07-31

Stable release of the 0.4.0 line. Since `0.4.0-rc1`, local Whisper gains
correct long-audio and per-request language/task handling, the Windows Flutter
and Unity natives move onto the tested MSVC Bazel path, and Kotlin callers get
a compatibility shim for the loader API removed during the BoltFFI migration.
The `0.4.0-rc1` and `0.4.0-alpha` entries below cover the rest of the changes
since 0.3.0.

### Fixed

- **Candle Whisper now transcribes audio of any length** instead of failing at
  roughly 15 seconds or truncating past 30 seconds. Audio is decoded in
  correctly trimmed windows, padding-only output is discarded, Whisper's
  suppress-token and no-speech rules are applied, and model-backed regression
  tests cover clips through 66 seconds (#426).
- **Whisper honors each request's language and task.** Transcription and
  translation select the correct decode prefix without leaking state between
  calls; unsupported languages, prompts, non-zero temperatures, and timestamp
  granularities now return an explicit input error instead of being silently
  ignored (#426).
- **Kotlin's migration docs referenced a removed `XybridModelLoader`.** The
  examples now use `XybridModel` directly, and a deprecated compatibility shim
  keeps applications written against the old loader shape compiling through
  the 0.4.x line (#329, #412).

### Changed

- **Flutter's Windows native is now MSVC ABI from end to end.** The shipped DLL
  and import library are cross-compiled by the Bazel release graph, include the
  llama.cpp vision path, and are load-tested on Windows before release
  (#416–#420).
- **Unity's Windows native now comes from the same Bazel graph** and is checked
  through real IL2CPP runtime smokes on Windows and Linux (#414, #421).
- **Release builds resolve exact Bazel outputs through one checked helper** and
  share one remote-execution configuration, removing ambiguous `bazel-bin`
  lookups and drift between shipping jobs (#424, #427).

---

## [0.4.0-rc1] - 2026-07-28

Release candidate for 0.4.0. Two consumer-visible fixes on top of `0.4.0-alpha`
— the published Flutter package could not be built by anyone with a Rust
toolchain installed, and Linux `.so`s linked the C++ runtime dynamically — plus
the Unity SDK moving fully onto BoltFFI and the Bazel graph growing from
"builds the artifacts" to "builds and tests them".

### Fixed

- **Flutter: the published pub.dev package would not build** for any consumer
  with a Rust toolchain installed. cargokit disables precompiled binaries
  whenever `rustup` is on `PATH`, and the published package cannot be built
  from source — its Rust crate inherits `edition` from a workspace root that is
  not published and depends on sibling crates by path — so the build died on an
  unrelated cargo manifest error. The choice is now made from the crate's
  location: published package always precompiled, monorepo checkout still
  source-built so edits to `xybrid-core`/`xybrid-sdk`/`xybrid-ffi-facade` are
  picked up (#338, #408, #409).
- **Linux `.so`s linked `libstdc++` dynamically** while binaries got it
  statically, so the shipped cdylibs required a C++ runtime on the host
  (#407).
- **Unity shipped debug natives** in the release bundles (#391), and desktop
  ONNX Runtime was missing from them (#379).

### Changed

- **Unity is fully on BoltFFI.** The managed layer, run/stream/context,
  telemetry, bundle and model-file paths all go through `xybrid_bolt`, and its
  natives for macOS, Android, Linux and iOS are built by Bazel
  (#380–#390, #392–#394).
- **Bazel now runs the test suite**, not just the builds — 38 test targets
  covering the `tests/` binaries across `xybrid-core`, `xybrid-sdk`,
  `xybrid-llama`, `xtask`, the CLI and integration-tests (#397–#403).
- **Unity CI runs EditMode tests in a real Unity Editor** on Linux, plus a
  Windows IL2CPP gate for the Bazel-built DLL (#396, #406).

### Removed

- **`xybrid-ffi` crate (pre-bolt C ABI)**: with Unity migrated onto BoltFFI,
  the C-ABI crate, its cbindgen header, and the csbindgen C# generation
  (`Runtime/Native/NativeMethods.g.cs`) are gone. The Unity SDK now loads
  `xybrid_bolt` on every platform (`cargo xtask build-ffi` builds bolt).

---

## [0.4.0-alpha] - 2026-07-18

Prerelease. The headline is invisible on purpose: nearly every shipped artifact
is now produced by one Bazel build graph on remote execution instead of
per-platform cargo builds — same names, same delivery, same signatures. This
alpha exists to exercise that new release pipeline end-to-end before a stable
cut. Alongside it: a browser SDK preview, a Python SDK, and reasoning-model
fixes.

### Added

- **Browser SDK preview (`@xybrid/web`)** backed by LiteRT.js and LiteRT-LM (#346).
- **Python SDK** — BoltFFI-based, ctypes over the `xybrid-bolt` cdylib (#327).
- **Reasoning capture**: `<thinking>` and gemma-4 reasoning formats recognized
  via a marker table and surfaced separately from the answer (#336, docs #331).
- **Bonsai-27B 1-bit runtime**: Qwen3VL companion artifacts and text-only VLM
  routing (#356).

### Fixed

- **Reasoning models silently produced empty answers** when the entire output
  was a thinking block (#355, #358).
- **SwiftPM manifest honesty**: the package no longer advertises a macOS slice
  that never shipped, and the iOS floor is `.v16` to match the binary — both
  previously failed at link time instead of resolve time (#357).
- **Flutter on Linux**: `libxybrid_flutter.so` not found when compiling from
  source (#340).

### Changed

- **The build factory: cargo → Bazel + remote execution** for every release
  artifact — the CLI on Linux (#347), macOS (#348, #350), and Windows (#352,
  #354), the Android AAR (#341), the iOS XCFramework with device + simulator
  slices (#362–#367), and the Flutter precompiled binaries (#369, #371).
  Consumer-visible effects: the Windows CLI switches toolchain flavor
  MSVC → MinGW (behaviorally identical for a self-contained CLI), and Linux
  artifacts now carry a hermetic glibc ≤ 2.31 floor, so they load on older
  distros than the previous runner-glibc builds. Bazel is also the required
  CI gate; the duplicate cargo CLI jobs are retired (#360, #361).
- **Dead execution strategies removed** from the core resolver
  (Standard/Tts/Llm) (#353).

---

## [0.3.0] - 2026-07-06

Local tool calling, Unity on OpenUPM, and honest cache clearing. The local
llama.cpp backend gains function/tool calling; the Unity SDK is re-platformed
onto a managed-only OpenUPM package that fetches its natives at import; and the
model-cache clear/discovery paths are corrected to report what they actually
remove.

### Changed

- **Unity SDK distribution moved to OpenUPM** (#321, #324). The Unity package now
  ships managed-only via the OpenUPM scoped registry (`ai.xybrid`); per-platform
  native libraries are downloaded from the GitHub Release at import by an editor
  resolver (SHA-256 verified) into `Assets/Xybrid/Plugins/`, **including the
  ~326 MB iOS slice** that previously required manual setup. **Breaking for Unity
  consumers:** the `#upm` git-branch install is replaced (install via OpenUPM or
  the `?path=/bindings/unity` git URL), and the `publish-upm` CI job is retired.
- **Model cache clearing reports what it removed** (#309). **Breaking:** `clear()`
  / `clear_model_roots()` now return the number of cache *roots* removed (was the
  scanned `.xyb` entry count, ~0 for the nested registry-bundle layout), and the
  CLI now warns when nothing was cached instead of always reporting success.
  `cache_root()` keeps `extracted/`, `hf/`, and `hf-hub/` co-located under the
  cache root for a bare relative `models` root instead of resolving them
  CWD-relative. `clear*` operations are documented as unsafe against concurrent
  loads.

### Added

- **Local tool calling for the llama.cpp backend** (#323): function/tool calls
  are parsed from local LLM output for LFM2 and Gemma-family models, with
  streaming tool-call continuation, an example, and a CLI REPL. See the
  tool-calling guide.
- **Unity native-library resolver + release bundles** (#321). An editor resolver
  downloads/verifies the per-platform natives on import and before player builds;
  CI publishes `xybrid-unity-native-<platform>-v<version>.zip` bundles + a
  SHA-256 manifest as release assets (managed by `cargo xtask package-unity-natives`).

### Fixed

- **Internal path-dep pins** realigned to the workspace version (#318).

---

## [0.2.2] - 2026-07-04

Structured output on-device. The local llama.cpp backend can now be constrained
to a grammar so small models (e.g. LFM2.5-230M) emit guaranteed-valid JSON for
data-extraction workloads, and that capability is exposed across every binding.

### Added

- **JSON-Schema / GBNF constrained decoding for the local llama backend**
  (#310): `GenerationConfig` gains a `grammar` field with chainable
  `with_grammar` / `with_json_schema` builders, backed by a new
  JSON-Schema→GBNF converter (`runtime_adapter::grammar`) covering the
  object / array / scalar / enum subset, including nullable (`["string","null"]`)
  fields and `\uXXXX` escapes. The grammar is prepended to the llama.cpp sampler
  chain at the single shared chokepoint, so all generate paths are constrained
  with no new type crossing the ABI. Ships with an end-to-end
  `lfm2_230m_grammar` example proving schema-valid receipt→JSON extraction on
  LFM2.5-230M where the unconstrained baseline fails. New
  `XybridError::Grammar` variant.
- **Grammar constraint exposed across all FFI surfaces** (#311): structured
  output now works from Swift, Kotlin, C, and Dart. The SDK re-exports
  `json_schema_to_gbnf` / `json_schema_str_to_gbnf` / `GrammarError`; the schema
  crosses the FFI boundary as text and is converted natively. Bolt (Swift /
  Kotlin), the C ABI (`xybrid_generation_config_set_grammar`,
  `xybrid_json_schema_to_gbnf`), and Flutter (`FfiGenerationConfig.grammar`,
  `jsonSchemaToGbnf`) all gain the `grammar` field and converter; committed
  Swift/Kotlin wrappers, the C header, and the FRB bindings are regenerated.

### Fixed

- **Compact JSON from schema→GBNF** (#310): the converter's whitespace rule
  allowed unbounded inter-token whitespace, letting a greedy model emit newlines
  until `max_tokens` (truncated output, `finish_reason=length`). The converter
  now emits compact (minified) JSON to remove the trap; output stays valid JSON.
- **Grammar converter robustness** (#310): NULL-check the llama sampler chain
  before use; error on non-object `properties` instead of silently matching
  `{}`; JSON-escape object keys before GBNF-escaping so control characters match
  their JSON-escaped form.

---

## [0.2.1] - 2026-06-25

The native VLM ships enabled. `0.2.0` landed the vision *foundation* — image
envelopes, preprocessing, and the mtmd backend in the codebase — but kept the
native VLM backend opt-in, so the default mobile/desktop binaries could not
actually run a vision-language model. `0.2.1` turns it on in every platform
preset: vision-language inference works out of the box, at a measured
~0.7–1.5 MiB stripped size cost.

### Added

- **Native VLM backend shipped enabled** (#296): `llm-llamacpp-vision`
  (llama.cpp's mtmd/clip) is now part of every platform preset
  (`platform-android` / `platform-ios` / `platform-macos` / `platform-desktop`),
  so the default XCFramework, Android AAR, Flutter/React Native natives, and CLI
  run vision-language models with no build-from-source step. The prebuilt-natives
  CI now publishes `vision` slices alongside `base`, so vision builds stay on the
  fast cached path instead of recompiling llama.cpp.

### Fixed

- **Unrecognized GGUF chat templates now render** (#304): when llama.cpp's
  hardcoded template matcher rejects a model's embedded chat template, xybrid
  falls back to a real Jinja engine (minijinja) to render it instead of failing
  — so GGUF models with custom or non-standard chat templates load and run
  correctly. Gated into `llm-llamacpp`, so non-llama builds pay zero cost.
- **Readable Apple FFI errors** (#296): `FfiError` now conforms to
  `LocalizedError`, so model-load and other low-level FFI failures surface their
  real message (e.g. a registry `ModelNotFound`) instead of the opaque
  "The operation couldn't be completed. (Xybrid.FfiError error 1.)".
- **Release tooling**: `version-sync` is now React-Native-aware (#298), and a
  spurious `bindings/flutter/rust/Cargo.lock` that broke dependency resolution
  was removed (#300).

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
  BoltFFI alongside the other foreign-language bindings, with a runnable Expo
  example and an Android build-from-source CI gate (#294).
- **Async/suspend conveniences restored** (#269) for Swift and Kotlin load + run.
- **Model `warmup` / `unload` exposed on Flutter** (#293), filling the sync/async
  symmetry across the binding surface.

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
  build cache is warmed on master pushes (#268); Swift + Kotlin wrapper compiles
  are gated in CI (#275). A prebuilt-llama.cpp-slices pipeline on ghcr
  (compile-once/link-many) now covers Android (3 ABIs), iOS device + simulator,
  and Linux x86_64, cutting native build time from ~25 min to seconds
  (#281, #284–#286, #288, #289, #291).

### Fixed

- **BoltFFI CLI/runtime aligned to 0.25.3** (#276): a CLI/runtime skew mis-generated
  unit-ok `Result<(), E>` exports (model warmup/unload) as a `void` foreign function
  that dropped the error and leaked the result buffer; pinned in lockstep.
- **Android `.so` is strip-safe and 16 KB-page aligned** (#287): the bolt `.so` now
  links `c++_shared` + 16 KB alignment via a clang shim instead of a post-link
  patchelf step (which appended a LOAD segment AGP strip corrupted, crashing
  `dlopen` on 16 KB-page devices); guarded by a dlopen CI gate.
- **Kotlin image format validation** restored and `EnvelopeTest` fixed for the bolt
  envelope shape (#266); `displayMessage` `when()` made exhaustive over the new
  error variants (#273).
- **iOS-simulator bindgen** now passes clang the canonical simulator triple (#274),
  unblocking the cross-compile vision compile-check.
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
