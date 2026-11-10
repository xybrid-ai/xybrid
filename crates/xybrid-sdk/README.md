# Xybrid SDK

Developer-facing API for hybrid cloud-edge AI inference with declarative routing annotations.

## Overview

The Xybrid SDK provides high-level abstractions and macros for building hybrid inference pipelines. It allows developers to annotate functions with `#[hybrid::route]` to enable automatic orchestrator-based routing between local and cloud execution.

## Initialization

Configure the SDK in one call with the `init()` builder. The SDK is
**local-first** — inference runs on-device whether or not you authenticate.
Passing an API key authenticates the cloud gateway and registry. Observability
export is enabled explicitly and uses OpenTelemetry:

```rust
use xybrid_sdk::observability::ObservabilityConfig;

// Anonymous local inference.
xybrid_sdk::init().run();

// Cloud authentication with explicit trace export to Xybrid.
xybrid_sdk::init()
    .api_key("sk_live_...")
    .observability(ObservabilityConfig::xybrid("sk_live_..."))
    .run();
```

An API key alone does not enable export. Without observability configuration,
the first inference logs a one-shot setup hint (suppress with `XYBRID_QUIET=1`).
Get a key at <https://console.xybrid.dev>. The builder also accepts
`.cache_dir(...)`, `.gateway_url(...)`, and `.resource_telemetry(...)`.
Use the fallible `init_observability` API below when startup errors must be
handled by your application; the convenience builder logs configuration errors.

## Usage

### Basic Example

```rust
use xybrid_sdk::hybrid;

#[hybrid::route]
fn asr_stage(input: String) -> String {
    // This function will be routed by the orchestrator
    // based on policy, metrics, and availability
    format!("asr_output: {}", input)
}
```

### Using Common Types

```rust
use xybrid_sdk::prelude::*;

// Use orchestrator directly
let mut orchestrator = Orchestrator::new();
let envelope = Envelope { kind: "AudioRaw".to_string() };
// ...
```

## Macro Status

The `#[hybrid::route]` macro is currently a **placeholder**. It:
- ✅ Compiles and can be used on functions
- ✅ Preserves the function signature and behavior
- ⏳ Does not yet transform functions to use the orchestrator

### Future Implementation

Future versions of the macro will:
1. Extract function metadata (name, parameters, return type)
2. Generate orchestrator calls automatically
3. Wrap function body with `orchestrator.execute_stage()` calls
4. Handle input/output envelope conversion
5. Inject `DeviceMetrics` and `LocalAvailability` handling

## Project Structure

- **`xybrid-sdk`**: Main SDK crate that re-exports `xybrid-core` and provides the macro
- **`xybrid-macros`**: Procedural macro crate implementing `#[hybrid::route]`

## Observability and local telemetry

`xybrid_sdk::observability` records inference, attempts, model loads, workflows,
and stages as OpenTelemetry spans, with optional metrics and log events.
`TelemetryEvent` and `register_telemetry_sender` provide a separate in-process
stream for host applications. Publishing a local event does not export it or
create an OpenTelemetry span.

### Explicit configuration and lifecycle

```rust
use std::time::Duration;
use xybrid_sdk::observability::{
    self, ContentPolicy, ObservabilityConfig, OtlpConfig,
};

// Send traces, metrics and logs to your own OpenTelemetry Collector.
observability::init_observability(
    ObservabilityConfig::otlp(OtlpConfig::collector("http://localhost:4318"))
        .service("my-app", Some("1.0.0".to_owned()))
        .content(ContentPolicy::metadata_only()),
)?;

// Load models and run inference through the SDK here.

// Flush without stopping export; use shutdown when the application exits.
observability::flush_observability(Duration::from_secs(5))?;
observability::shutdown_observability(Duration::from_secs(5))?;
Ok::<(), xybrid_sdk::SdkError>(())
```

For the Xybrid dashboard use `ObservabilityConfig::xybrid(api_key)`, which sends
bearer-authenticated **traces only** to `https://ingest.xybrid.dev/v1/traces`.
`ObservabilityConfig::xybrid_at(full_traces_endpoint, api_key)` selects a custom
trace endpoint. Metrics and logs require their own configured destinations.

Initialization replaces the previous configuration and shuts down its packaged
pipeline. Flush and shutdown have bounded timeouts; shutdown is idempotent and
disables the observer. `observability::stats()` exposes delivery counters while
a packaged pipeline is installed. With `ObservabilityConfig::with_observer`,
the application owns provider flush/shutdown.

Once observability is enabled, prompts and outputs are captured by default.
Select `ContentPolicy::metadata_only()` to retain metadata without content, or
configure a content policy with redaction. Per-run `ObservationOptions` on
`RunOptions` sets a parent trace, conversation id or trace suppression; successful
inference results expose an `observation_ref()` for feedback and evaluations.

### Resource attributes and delivery

The SDK attaches its version and binding to the OpenTelemetry resource. Device
attributes include detected OS, architecture, chip and RAM by default; hostnames
are never sent. Set `.device_attributes(false)` to disable hardware attributes.
Use `.service(...)` for application identity and
`OtlpConfig::with_resource_attribute(...)` for application-specific attributes.
Resource sampling around runs is off by default; enable it with
`.resource_telemetry(ResourceTelemetryMode::summary())`.

`OtlpConfig::with_batch(...)` and `with_retry(...)` configure bounded queues,
batches and retries. Export failures and dropped items are reflected in delivery
counters.

### Environment variables

Environment configuration is explicit too:

```rust
use xybrid_sdk::observability::{self, ObservabilityConfig};

if let Some(config) = ObservabilityConfig::from_env() {
    observability::init_observability(config)?;
}
Ok::<(), xybrid_sdk::SdkError>(())
```

| Variable | Purpose |
|----------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector base URL; enables all three signals in `from_env()` |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | Full per-signal destination URLs |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` (default) or `http/json`; per-signal overrides are supported |
| `OTEL_EXPORTER_OTLP_HEADERS` | Export headers; per-signal overrides are supported |
| `OTEL_SDK_DISABLED=true` | Makes `from_env()` return no configuration |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` | Content-capture policy when no policy is explicitly provided |
| `XYBRID_RESOURCE_TELEMETRY` | Overrides resource sampling: `off`, `boundary`, `summary[:ms]`, `debug_local[:ms]` |

`XYBRID_API_KEY` configures authentication separately; it does not automatically
start observability in the SDK. The examples explicitly use it as a dashboard
export preset when no OTLP endpoint is configured.

See [the observability crate](../xybrid-observability/README.md) for the signal
contract, configuration, and provider ownership.

## Examples

See `examples/macro_demo.rs` for a complete example of using the macro.

```bash
cargo run --example macro_demo
```
