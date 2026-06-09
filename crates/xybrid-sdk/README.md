# Xybrid SDK

Developer-facing API for hybrid cloud-edge AI inference with declarative routing annotations.

## Overview

The Xybrid SDK provides high-level abstractions and macros for building hybrid inference pipelines. It allows developers to annotate functions with `#[hybrid::route]` to enable automatic orchestrator-based routing between local and cloud execution.

## Initialization

Configure the SDK in one call with the `init()` builder. The SDK is
**local-first** — inference runs on-device whether or not you authenticate.
Passing an API key adds the **platform layer** on top of the same runtime: it
authenticates the cloud gateway (for local→cloud routing) and starts the
telemetry exporter so your runs show up on the dashboard. The local path is
unchanged either way.

```rust
// Anonymous — local inference, telemetry disabled
xybrid_sdk::init().run();

// Authenticated — telemetry exporter starts automatically
xybrid_sdk::init()
    .api_key("xy_live_...")
    .run();
```

Without a key, the first inference logs a one-shot hint pointing at the
dashboard (suppress with `XYBRID_QUIET=1`). Get a free key at
<https://dashboard.xybrid.dev>. The builder also takes `.cache_dir(...)`,
`.gateway_url(...)`, `.ingest_url(...)` (self-hosted dashboards), and
`.resource_telemetry(...)`.

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

## Telemetry

The SDK includes built-in telemetry that exports events to the Xybrid Platform.
For the common case you don't need anything here — just pass `.api_key(...)` to
[`init()`](#initialization) and the exporter starts with sensible defaults
(production ingest URL, batching, retry). The APIs below are the **advanced**
path for callers that need explicit control over batching, device context, or
the exporter lifecycle.

### Advanced configuration

```rust
use xybrid_sdk::telemetry::{TelemetryConfig, HttpTelemetryExporter};

// From environment variables
let exporter = HttpTelemetryExporter::from_env()?;

// Manual configuration
let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_batch_size(50)
    .with_flush_interval(Duration::from_secs(5));
let exporter = HttpTelemetryExporter::new(config);
```

### Device context

Telemetry auto-detects chip family, RAM, OS and kernel versions, CPU architecture,
and execution provider hints by default.

```rust
use xybrid_sdk::telemetry::{HttpTelemetryExporter, TelemetryConfig};

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_app_version("mirage-vault/0.0.1")
    .with_device_label("Sami's MacBook Pro")
    .with_hardware_chip("Apple M4 Max");

let exporter = HttpTelemetryExporter::new(config);
```

See [telemetry.md](../../docs/sdk/telemetry.md) for full privacy posture,
opt-out, and wire format.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `XYBRID_INGEST_URL` | Telemetry ingest endpoint | `https://ingest.xybrid.dev` |
| `XYBRID_API_KEY` | API key for authentication | (required) |
| `XYBRID_PLATFORM_URL` | Legacy fallback URL | - |

### Features

- **Circuit breaker**: Prevents hammering failing endpoints
- **Automatic retry**: Exponential backoff with jitter
- **Batching**: Configurable batch size and flush interval
- **Background queue**: Retries failed events automatically

## Examples

See `examples/macro_demo.rs` for a complete example of using the macro.

```bash
cargo run --example macro_demo
```
