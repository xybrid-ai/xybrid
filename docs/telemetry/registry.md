# Registry Call Telemetry

The Xybrid SDK attaches a single header — `X-Xybrid-Client` — to every metadata request it makes against the model registry. The header reports the binding that originated the call, the SDK and core versions, the build platform, and the active backend feature set. It is the only telemetry surface on registry traffic; outcome reporting (`/v1/anon` and friends) is intentionally not part of this path.

This page documents exactly what is sent, why it exists, and how to opt out.
Resolve requests also carry functional compatibility parameters, including
an optional runtime OS token; these are separate from the telemetry header.

## Opting out

Set the environment variable before any registry call is made:

```sh
export XYBRID_TELEMETRY_OPTOUT=1
```

When the variable is set to any of `1`, `true`, or `yes` (case-insensitive, surrounding whitespace ignored), the SDK omits `X-Xybrid-Client` from every registry request. The opt-out also disables the platform telemetry exporter described in [telemetry.md](../sdk/telemetry.md). It does not remove the `backends` or optional `os` query parameters from resolve requests, because they decide which file the registry serves; see [Resolve compatibility](#resolve-compatibility).

The value is read once per process and cached, so changing it after the first registry call has no effect for that run.

### Verifying opt-out

The CLI reports the resolved opt-out state:

```sh
$ xybrid telemetry status
registry telemetry: enabled
```

```sh
$ XYBRID_TELEMETRY_OPTOUT=1 xybrid telemetry status
registry telemetry: disabled (XYBRID_TELEMETRY_OPTOUT=1)
```

The command exits 0 in both cases — it is a status report, not an error.

## What is sent

The header is a single line, semicolon-separated:

```
X-Xybrid-Client: binding=flutter; sdk_version=0.11.0; core_version=0.11.0; platform=ios-arm64; backends=asr-whispercpp,asr-whispercpp-coreml,llm-llamacpp,llm-llamacpp-vision,ort-coreml,ort-download,vision
```

| Field | Type | Description |
|-------|------|-------------|
| `binding` | enum string | The platform binding that made the call. One of `rust`, `cli`, `flutter`, `kotlin`, `python`, `react-native`, `swift`, `unity`, `web`. Defaults to `rust` when no binding is registered. |
| `sdk_version` | semver-ish string | `xybrid-sdk` package version, baked in at compile time via `CARGO_PKG_VERSION`. |
| `core_version` | semver-ish string | `xybrid-core` package version, baked in via the same mechanism. Usually equal to `sdk_version` but reported independently so version skews surface. |
| `platform` | enum string | Compile-time target triple summary. See the table below. |
| `backends` | comma-separated list | Alphabetical list of enabled runtime feature flags from `xybrid_core::features::enabled()`. Never empty: `vision` is reported unconditionally because the vision pipeline is always compiled in. |

### `binding` values

| Value | Source | Set by |
|-------|--------|--------|
| `rust` | Default — used when nothing registers a binding, e.g. a Rust app built on `xybrid-sdk`. | `xybrid_sdk::DEFAULT_BINDING` |
| `cli` | The `xybrid` command-line tool. | `main()` (`crates/xybrid-cli/src/main.rs`) |
| `flutter` | Flutter plugin via `flutter_rust_bridge`. | `XybridRustLib.init()`, through the `#[frb(init)]` hook `init_app` (`bindings/flutter/rust/src/api/sdk_client.rs`) |
| `kotlin` | Android library via BoltFFI. | `Xybrid.init(context)` (`bindings/kotlin/src/main/kotlin/ai/xybrid/Xybrid.kt`); until then, the native library's Android fallback (`crates/xybrid-bolt/src/lib.rs`) |
| `python` | Python package via BoltFFI. | `import xybrid` (`bindings/python/xybrid/__init__.py`) |
| `react-native` | React Native package, wrapping the Swift and Kotlin SDKs. | Every bridged call, before the SDK it wraps (`bindings/react-native/ios/XybridModuleImpl.swift`, `bindings/react-native/android/src/main/java/ai/xybrid/reactnative/XybridModule.kt`) |
| `swift` | iOS / macOS Swift package via BoltFFI. | `Xybrid.initialize()` (`bindings/apple/Sources/Xybrid/Xybrid.swift`); until then, the native library's Apple-platform fallback, set by any model, download or pipeline constructor (`crates/xybrid-bolt/src/lib.rs`) |
| `unity` | Unity / C# binding via BoltFFI. | Static constructors on the generated types that reach native code (`bindings/unity/Runtime/BoltSupplement/BindingRegistration.cs`) |
| `web` | Browser SDK (`@xybrid/web`). It sends only `binding`, `sdk_version` and `platform`, and has no opt-out: browsers have no environment variables. | Every registry resolve request (`bindings/web/src/internal/registry.ts`) |

The first registration in a process wins. The native library's `swift` / `kotlin` fallback on Apple platforms and Android is not a registration: it applies only until an SDK registers, so React Native, Unity and Python report their own names even if a native call ran first.

Any value containing characters outside `[a-z0-9_-]` is replaced with `rust` before being placed in the header. This is a defensive sanitization step — it prevents user-controlled strings from injecting additional fields or terminators into the header.

### `platform` values

| Value | Target |
|-------|--------|
| `macos-arm64` | macOS on Apple Silicon |
| `macos-x86_64` | macOS on Intel |
| `ios-arm64` | iOS / iPadOS device builds |
| `android-arm64` | Android arm64-v8a |
| `android-arm` | Android armeabi-v7a |
| `linux-x86_64` | Linux on x86_64 |
| `linux-arm64` | Linux on aarch64 |
| `windows-x86_64` | Windows on x86_64 |
| `web` | Browser (`@xybrid/web`) |
| `unknown` | Any target outside the table above |

Source: `xybrid_sdk::current_platform()` (`crates/xybrid-sdk/src/platform.rs`). The string is decided at compile time; switching architectures requires rebuilding.

### `backends` values

| Value | Cargo feature | Notes |
|-------|---------------|-------|
| `asr-whispercpp` | `asr-whispercpp` | whisper.cpp speech recognition; in every `platform-*` preset. Pulls `llm-llamacpp` transitively — whisper.cpp links the ggml that llama.cpp builds |
| `asr-whispercpp-coreml` | `asr-whispercpp` + `coreml`, iOS only | The build can run whisper.cpp's encoder through Core ML (iOS 16 and later). The registry serves the `ios-arm64` Whisper bundles, which carry the encoder, only to clients that list it, so `resolve` also sends it in its query (see [When the header is sent](#when-the-header-is-sent)) |
| `candle-cuda` | `candle-cuda` | Candle backend with CUDA acceleration. Opt-in only — no `platform-*` preset enables it |
| `candle-metal` | `candle-metal` | Candle backend with Metal acceleration. Opt-in only — no `platform-*` preset enables it |
| `espeak` | `espeak` | espeak-ng phonemizer (multi-language TTS) |
| `llm-llamacpp` | `llm-llamacpp` | llama.cpp backend (universal LLM runtime) |
| `llm-llamacpp-vision` | `llm-llamacpp-vision` | Native llama multimodal (mtmd) VLM backend layered on `llm-llamacpp`; in every `platform-*` preset |
| `llm-mistral` | `llm-mistral` | mistral.rs backend |
| `ort-coreml` | `ort-coreml` | ONNX Runtime with CoreML execution provider |
| `ort-cuda` | `ort-cuda` | ONNX Runtime with CUDA execution provider |
| `ort-download` | `ort-download` | ONNX Runtime resolved via prebuilt downloads |
| `ort-dynamic` | `ort-dynamic` | ONNX Runtime loaded dynamically at runtime (Android) |
| `tts-zzz` | `tts-zzz` | Kitten TTS 2 and Phonon through the pinned zzz CPU engine (macOS ARM64, Linux x86-64-v3, Android ARM64). Opt-in only — no `platform-*` preset enables it |
| `vision` | _(none — always on)_ | Vision envelopes + preprocessing pipeline; compiled in unconditionally, so this value appears on every build |
| `zzz-phonon-31m` | `tts-zzz` | The build's pinned zzz engine loads Gradium's `phonon-31m-en` ([Phonon speech](../sdk/PHONON.md)). 0.12.0 builds report `tts-zzz` too, but their engine rejects that model, so the registry serves `phonon-31m-en` only to clients that list `zzz-phonon-31m`, and `resolve` also sends it in its query (see [When the header is sent](#when-the-header-is-sent)) |

Source: `xybrid_core::features::enabled()` (`crates/xybrid-core/src/features.rs`). The list is computed once per process from `cfg!(feature = "...")` checks and cached.

The full set of mutually compatible combinations is documented in [`docs/FEATURE_MATRIX.md`](../FEATURE_MATRIX.md).

## What is NOT collected

The header carries only the fields above. It does **not** carry:

- No personal identifiers (no username, no hostname, no email)
- No network identifiers (no IP address, no MAC address)
- No model identifiers (the model ID requested on `/v1/models/{mask}` is part of the URL path, not the header — but the header itself adds no further model context)
- No device fingerprints (no chip family, no RAM size, no OS version, no kernel version)
- No prompts, no inputs, no outputs, no error messages
- No `device_id` or any session/trace identifier

The optional `os` query parameter does report a runtime OS or GNU libc version
for resolve compatibility. It is separate from the header, is excluded from
registry analytics, and remains visible in Worker request logs. Opting out of
telemetry does not suppress it; see [Resolve compatibility](#resolve-compatibility).

The platform-ingest exporter described in [telemetry.md](../sdk/telemetry.md) is a separate, opt-in surface that does collect device context for inference performance analysis. Registry-call telemetry — the subject of this document — is the smaller surface used for fleet attribution and is enabled by default unless opted out.

## Why this exists

We use the aggregate distribution of `binding`, `platform`, and `backends` to prioritize roadmap work:

- Which bindings actually see usage (and which can be deprioritized)
- Which platforms are deployed in the wild (so we know which CI matrices matter)
- Which backend combinations users assemble (so we know which feature flag combinations to keep working)

The data is summarized at the fleet level. The registry server stores parsed values bounded to a small allowlist — anything outside the allowlist is recorded as the literal string `unknown` so cardinality stays bounded and a single noisy app cannot pollute the dataset.

## When the header is sent

The SDK adds `X-Xybrid-Client` to the three metadata calls:

| Method | Path | Purpose |
|--------|------|---------|
| `RegistryClient::list_models` | `GET /v1/models` | List published models |
| `RegistryClient::get_model` | `GET /v1/models/{mask}` | Fetch metadata for one model |
| `RegistryClient::resolve` | `GET /v1/models/{mask}/resolve?platform=…` | Resolve a model+platform to a bundle URL |

`resolve` also lists, as `&backends=` in its query, the backends that decide which file the registry serves; today those are `asr-whispercpp-coreml` and `zzz-phonon-31m`. Unlike the header, it is sent when telemetry is opted out too: without it, an opted-out iOS client would get the plain `.bin` instead of the bundle with the Core ML encoder, and an opted-out `tts-zzz` client could not resolve `phonon-31m-en`. An optional `&os=` supplies the runtime version for OS compatibility. The registry does not record either query parameter's value in analytics.

Bundle download requests are intentionally **not** instrumented in this version. They are downloads, not metadata calls, and a per-byte header is not interesting in aggregate.

## Resolve compatibility

The Rust SDK sends `os` when it can detect the runtime version and the
requested platform matches the current platform. It omits the parameter
when the version is unknown or the caller explicitly resolves a different
platform. These tokens describe the running environment, not a build target's
deployment minimum:

| Runtime | Token | Example |
|---|---|---|
| Android | `android-api-N` | `android-api-35` |
| iOS / iPadOS | `ios-X.Y` | `ios-18.5` |
| macOS | `macos-X.Y` | `macos-15.4` |
| Linux with GNU libc | `glibc-X.Y` | `glibc-2.34` |

On Linux, `X.Y` is the GNU libc runtime version, never the kernel release.
For example, a native Android request can be:

```text
GET /v1/models/phonon-31m-en/resolve?platform=android-arm64&backends=zzz-phonon-31m&os=android-api-35
```

The `os` and `backends` parameters are functional compatibility inputs,
independent of `XYBRID_TELEMETRY_OPTOUT`. The `X-Xybrid-Client` header has no
OS field. Browser registry requests follow the separate web client described
in the binding table above.

Model-detail variants and the selected `resolved` record may include
`min_os`, an array of OS floor tokens. Rust's `VariantInfo` and
`ResolvedVariant` preserve it as `Vec<String>`, defaulting an omitted array to
empty. Missing floors mean no OS minimum was declared, not that native execution was
qualified. Floors are per variant and are not inherited from another entry.
The registry compares numeric versions only for matching OS names; absent,
malformed or unrelated OS tokens preserve legacy selection without enforcing
that floor.

Resolution tries exact platform, platform family, then `universal`, skipping
incompatible candidates. If no candidate serves the client, a format-matching
candidate's compatibility refusal keeps HTTP 404 and adds structured fields
inside `error`:

| `error.code` | Diagnostic fields alongside `message` and `status` |
|---|---|
| `BACKEND_REQUIRED` | `platform`, `requires_backends` (required backends the client did not advertise) |
| `OS_VERSION_UNSUPPORTED` | `platform`, `os` (the client's token), `min_os` (unmet floors) |

The first refusing candidate in selection order supplies the diagnostic; an
unmet OS floor takes precedence over missing backends on that candidate.
Ordinary missing models or variants still return `NOT_FOUND`.
The registry gives the client header's platform precedence over the legacy
`platform` query parameter. Refusal diagnostics retain that effective platform,
even when the caller supplied a different query platform.

In Rust, `BACKEND_REQUIRED` maps to the existing
`SdkError::UnsupportedBackendCapability`; its hint retains the exact registry
code and server message. `OS_VERSION_UNSUPPORTED` maps to
`SdkError::OsVersionUnsupported` with `model_id`, `platform`, `os`, `min_os`
and `message`. Both are permanent, nonretryable errors. Unknown codes,
malformed bodies and legacy 404 responses keep the existing `ModelNotFound`
behavior.

Foreign binding error shapes stay stable: backend refusals use
`UnsupportedBackendCapability`, and OS refusals use `ConfigError` containing
the full `OS_VERSION_UNSUPPORTED` diagnostic. Callers can display the reason
and required floor without relying on a new generated binding error variant.

The deployed [platform compatibility update](https://github.com/xybrid-ai/xybrid-platform/pull/308)
adds `ios-arm64` resolution of the existing Phonon GGUF with
`requires_backends: ["zzz-phonon-31m"]` and `min_os: ["ios-16.0"]`.
Native Swift/Flutter speech linkage remains unqualified. Of the standard
prebuilt SDK artifacts, only the Android AAR's arm64-v8a library ships the
speech engine; see [Phonon speech](../sdk/PHONON.md).

## Setting `binding` from a custom client

Most users never need to think about `binding` — the platform binding sets it for them. If you are embedding `xybrid-sdk` directly in a Rust app and want your own attribution string, set it at startup:

```rust
xybrid_sdk::set_binding("my-tool");
```

The first `set_binding` call wins — once a value is registered, subsequent calls are silent no-ops, matching the lifecycle of "one process, one binding." The init builder does the same:

```rust
xybrid_sdk::init().binding("my-tool").run();
```

`SdkConfig::with_binding` is deprecated: nothing reads a config's binding, so it never reaches a request.

If your value contains anything outside `[a-z0-9_-]`, the SDK substitutes `rust` before placing the value into the header. There is no way to inject a custom field name; the format is fixed.

## Verification

Run the CLI command above, or instrument an HTTP proxy in front of the registry and inspect the header directly. The header value is identical across the three metadata calls within a process.

## Related documentation

- [Platform telemetry exporter](../sdk/telemetry.md) — the opt-in exporter for inference events
- [Resource telemetry](../sdk/resource-telemetry.md) — per-inference resource summaries
- [API reference](../sdk/API_REFERENCE.md) — full SDK API surface
- [Feature matrix](../FEATURE_MATRIX.md) — which backend features compile together
