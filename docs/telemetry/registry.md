# Registry Call Telemetry

The Xybrid SDK attaches a single header — `X-Xybrid-Client` — to every metadata request it makes against the model registry. The header reports the binding that originated the call, the SDK and core versions, the build platform, and the active backend feature set. It is the only telemetry surface on registry traffic; outcome reporting (`/v1/anon` and friends) is intentionally not part of this path.

This page documents exactly what is sent, why it exists, and how to opt out.

## Opting out

Set the environment variable before any registry call is made:

```sh
export XYBRID_TELEMETRY_OPTOUT=1
```

When the variable is set to any of `1`, `true`, or `yes` (case-insensitive, surrounding whitespace ignored), the SDK omits `X-Xybrid-Client` from every registry request. The opt-out also disables the platform telemetry exporter described in [telemetry.md](../sdk/telemetry.md).

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
X-Xybrid-Client: binding=flutter; sdk_version=0.1.0-beta12; core_version=0.1.0-beta12; platform=ios-arm64; backends=candle-metal,llm-llamacpp,llm-mlx,ort-coreml,ort-download
```

| Field | Type | Description |
|-------|------|-------------|
| `binding` | enum string | The platform binding that made the call. One of `rust`, `flutter`, `kotlin`, `swift`, `unity`. Defaults to `rust` when no binding is registered. |
| `sdk_version` | semver-ish string | `xybrid-sdk` package version, baked in at compile time via `CARGO_PKG_VERSION`. |
| `core_version` | semver-ish string | `xybrid-core` package version, baked in via the same mechanism. Usually equal to `sdk_version` but reported independently so version skews surface. |
| `platform` | enum string | Compile-time target triple summary. See the table below. |
| `backends` | comma-separated list | Alphabetical list of enabled runtime feature flags from `xybrid_core::features::enabled()`. May be empty (`backends=`) if no backends are compiled in. |

### `binding` values

| Value | Source | Set by |
|-------|--------|--------|
| `rust` | Default — used when no platform binding registers itself. | `xybrid_sdk::DEFAULT_BINDING` |
| `flutter` | Flutter plugin via `flutter_rust_bridge`. | `XybridSdkClient` (`bindings/flutter/rust/src/api/sdk_client.rs`) |
| `kotlin` | Android library via UniFFI. | `Xybrid.init(context)` (`bindings/kotlin/src/main/kotlin/ai/xybrid/Xybrid.kt`) |
| `swift` | iOS / macOS Swift package via UniFFI. | `Xybrid.initialize()` (`bindings/apple/Sources/Xybrid/Xybrid.swift`) |
| `unity` | Unity / C# binding via the C FFI. | `XybridClient.Initialize()` (`bindings/unity/Runtime/Api/XybridClient.cs`) |

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
| `unknown` | Any target outside the table above |

Source: `xybrid_sdk::current_platform()` (`crates/xybrid-sdk/src/platform.rs`). The string is decided at compile time; switching architectures requires rebuilding.

### `backends` values

| Value | Cargo feature | Notes |
|-------|---------------|-------|
| `candle-cuda` | `candle-cuda` | Candle backend with CUDA acceleration |
| `candle-metal` | `candle-metal` | Candle backend with Metal acceleration |
| `espeak` | `espeak` | espeak-ng phonemizer (multi-language TTS) |
| `llm-llamacpp` | `llm-llamacpp` | llama.cpp backend (universal LLM runtime) |
| `llm-mistral` | `llm-mistral` | mistral.rs backend |
| `llm-mlx` | `llm-mlx` | MLX skeleton (Apple Silicon, no forward pass) |
| `llm-mlx-runtime` | `llm-mlx-runtime` | MLX with the real forward pass |
| `ort-coreml` | `ort-coreml` | ONNX Runtime with CoreML execution provider |
| `ort-cuda` | `ort-cuda` | ONNX Runtime with CUDA execution provider |
| `ort-download` | `ort-download` | ONNX Runtime resolved via prebuilt downloads |
| `ort-dynamic` | `ort-dynamic` | ONNX Runtime loaded dynamically at runtime (Android) |

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

The platform-ingest exporter described in [telemetry.md](../sdk/telemetry.md) is a separate, opt-in surface that does collect device context for inference performance analysis. Registry-call telemetry — the subject of this document — is the smaller and always-present surface used for fleet attribution.

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

Bundle download requests are intentionally **not** instrumented in this version. They are downloads, not metadata calls, and a per-byte header is not interesting in aggregate.

## Setting `binding` from a custom client

Most users never need to think about `binding` — the platform binding sets it for them. If you are embedding `xybrid-sdk` directly in a Rust app and want your own attribution string, set it at startup:

```rust
xybrid_sdk::set_binding("my-tool");
```

The first `set_binding` call wins — once a value is registered, subsequent calls are silent no-ops, matching the lifecycle of "one process, one binding." A second route is the per-config field on [`SdkConfig::with_binding`](../sdk/API_REFERENCE.md#9-configuration-types):

```rust
use xybrid_sdk::{SdkConfig, DEFAULT_BINDING};

let config = SdkConfig::default().with_binding("my-tool");
assert_eq!(config.binding(), "my-tool");
```

If your value contains anything outside `[a-z0-9_-]`, the SDK substitutes `rust` before placing the value into the header. There is no way to inject a custom field name; the format is fixed.

## Verification

Run the CLI command above, or instrument an HTTP proxy in front of the registry and inspect the header directly. The header value is identical across the three metadata calls within a process.

## Related documentation

- [Platform telemetry exporter](../sdk/telemetry.md) — the opt-in exporter for inference events
- [Resource telemetry](../sdk/resource-telemetry.md) — per-inference resource summaries
- [API reference](../sdk/API_REFERENCE.md) — full SDK API surface, including `SdkConfig.binding`
- [Feature matrix](../FEATURE_MATRIX.md) — which backend features compile together
