# Telemetry

## Overview

The Rust SDK telemetry exporter sends pipeline events, a device profile, and the session and trace identifiers needed to group related events. Pipeline events describe inference lifecycle activity such as starts, completions, per-stage timings, and errors. Device context is attached so latency and throughput numbers can be interpreted against the machine that produced them.

## Automatic device detection

`TelemetryConfig` runs hardware detection by default when the HTTP exporter is created. Detection is best-effort: fields that cannot be determined are omitted from the `device` object rather than guessed.

### What's collected by default

| field | example | source |
|---|---|---|
| `chip_family` | `Apple M4 Max` | `sysinfo` CPU brand |
| `ram_gb` | `64` | `sysinfo` total memory, rounded to GB |
| `os` | `macOS` | `sysinfo` OS name |
| `os_version` | `14.5.0` | `sysinfo` OS version |
| `kernel_version` | `Darwin 23.5.0` | `sysinfo` kernel version |
| `arch` | `arm64` | `std::env::consts::ARCH` |

The hardware profile is resolved once per exporter. It is reused for all events from that exporter, so event emission does not repeatedly probe the machine.

### What's never collected

The SDK does not collect the following fields because they are personally identifying, provide high-entropy fingerprinting signals, or are not required to interpret inference performance:

- Username
- MAC address
- Serial number
- IP address in the telemetry payload
- Installed applications
- Running processes
- Attached peripherals

The HTTP transport still reveals the client IP address to the configured ingest endpoint, because that is part of making an HTTP request. The SDK does not add the IP address to the event payload.

### Opt-in fields

Hostname capture is off by default. Hostnames such as `Samis-MacBook` effectively identify a person, so the SDK only includes `device.hostname` when the application opts in.

```rust
use xybrid_sdk::telemetry::TelemetryConfig;

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_hostname_capture(true);
```

## Configuring TelemetryConfig

### Zero-effort setup

With the default configuration, the exporter auto-detects the hardware profile and attaches it to emitted events.

```rust
use xybrid_sdk::telemetry::{HttpTelemetryExporter, TelemetryConfig};

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_app_version("mirage-vault/0.0.1");

let exporter = HttpTelemetryExporter::new(config);
```

### Adding a human-friendly label

`device_id` is meant to be stable and machine-readable. `device_label` is the optional label shown to humans reviewing traces.

```rust
use xybrid_sdk::telemetry::TelemetryConfig;

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_app_version("mirage-vault/0.0.1")
    .with_device_label("Sami's MacBook Pro");
```

### Overriding hardware fields

Use `with_hardware` when the app wants to supply the complete profile itself. This disables automatic hardware detection for the profile.

```rust
use std::collections::BTreeMap;
use xybrid_sdk::telemetry::{DeviceProfile, TelemetryConfig};

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_app_version("mirage-vault/0.0.1")
    .with_hardware(DeviceProfile {
        chip_family: Some("Apple M4 Max".to_string()),
        ram_gb: Some(64),
        os: Some("macOS".to_string()),
        os_version: Some("14.5.0".to_string()),
        kernel_version: Some("Darwin 23.5.0".to_string()),
        arch: Some("arm64".to_string()),
        hostname: None,
        custom: BTreeMap::new(),
    });
```

For partial overrides, use the field-specific builders. These values are merged onto the automatically detected profile.

```rust
use xybrid_sdk::telemetry::TelemetryConfig;

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_app_version("mirage-vault/0.0.1")
    .with_hardware_chip("Apple M4 Max")
    .with_hardware_ram_gb(64);
```

`with_hardware_os` and `with_hardware_arch` follow the same pattern:

```rust
use xybrid_sdk::telemetry::TelemetryConfig;

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_hardware_os("macOS", "14.5.0")
    .with_hardware_arch("arm64");
```

### Adding custom attributes

Custom attributes are app-provided strings placed under `device.custom`. Use them for deployment metadata that helps group events without exposing user data.

```rust
use xybrid_sdk::telemetry::TelemetryConfig;

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_device_attribute("tailnet", "production");
```

### Opting out entirely

Disable automatic hardware detection when the app does not want the SDK to probe the local machine.

```rust
use xybrid_sdk::telemetry::TelemetryConfig;

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_auto_hardware_detection(false);
```

When `with_auto_hardware_detection(false)` is the only device-context call on the builder — no explicit hardware fields, no custom attributes, no hostname capture, no `with_device_label` — the SDK emits **neither** a `device` object **nor** an auto-generated `device_id` on events. The emitted event has those keys absent entirely.

Apps that still want a stable identifier under full opt-out should call `with_device(id, platform)` explicitly to supply one.

Explicit fields are honored even under opt-out: adding `with_hardware_chip("Apple M4 Max")` or `with_device_attribute("tailnet", "production")` puts the SDK back in the "has device context" mode, so those fields appear on events and an auto-generated `device_id` is created if the app hasn't supplied one.

## The default `device_id`

When the app does not call `with_device(…)` and is emitting any device context, the SDK generates a random UUID-based identifier on first run and persists it to `~/.xybrid/device_id`. On Unix the directory is set to mode `0700` and the file to mode `0600`, so other users on the same machine cannot read it.

The ID is intentionally **not** derived from hardware. Hardware-hashed identifiers collide on identical VMs (CI fleets, cloned sandboxes) and rename the same machine whenever RAM or OS changes. A random UUID per-install avoids both problems.

On ephemeral or read-only homes (containers, sandboxes with no writable `$HOME`) the SDK falls back to a process-local random ID that only lives for the exporter's lifetime. Telemetry grouping will be weaker in those environments — that's the honest behavior rather than silently merging unrelated runs into one "device".

## Migrating from `with_device(id, platform)`

The existing `with_device(id, platform)` call still works. It sets the legacy `device_id` and `platform` fields. The new API adds an independent hardware profile, plus an optional human-readable label.

Before:

```rust
use xybrid_sdk::telemetry::TelemetryConfig;

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_device("mirage-vault", "macos")
    .with_app_version("0.0.1");
```

After:

```rust
use xybrid_sdk::telemetry::TelemetryConfig;

let config = TelemetryConfig::new("https://ingest.xybrid.dev", "sk_live_xxx")
    .with_device("mirage-vault", "macos")
    .with_device_label("Sami's MacBook Pro")
    .with_app_version("0.0.1");
```

No migration is required if the app is already setting `device_id` and `platform` and is comfortable with automatic hardware detection. Add the new builders only when you need labels, overrides, custom attributes, hostname capture, or opt-out behavior.

## Privacy posture

The SDK avoids collecting direct identifiers and high-entropy fingerprinting data in the telemetry payload. By default it sends hardware and OS context that is needed to interpret local inference performance: chip family, RAM size, OS and kernel versions, architecture, and execution provider hints. Hostname is excluded unless the app explicitly calls `with_hostname_capture(true)`, because hostnames often contain a person's name. The SDK sends events only to the endpoint configured in `TelemetryConfig`; it does not add a third-party telemetry service. The HTTP request necessarily discloses the client's IP address to that endpoint, but the SDK does not include the IP address as a payload field. Server-side retention is controlled by the Xybrid Platform deployment that receives the event.

## Wire format

The device profile is encoded as a typed `device` substructure. Platform ingest builds older than this SDK ignore unknown top-level fields by default, so `device_label` and `device` are forward compatible.

```json
{
  "session_id": "...",
  "event_type": "...",
  "payload": {},
  "device_id": "dev_8a...",
  "device_label": "Sami's MacBook Pro",
  "platform": "macos",
  "app_version": "0.0.1",
  "device": {
    "chip_family": "Apple M4 Max",
    "ram_gb": 64,
    "os": "macOS",
    "os_version": "14.5.0",
    "kernel_version": "Darwin 23.5.0",
    "arch": "arm64",
    "custom": {
      "tailnet": "production"
    }
  },
  "timestamp": "..."
}
```

Older SDKs continue to send the legacy top-level fields. Newer platform deployments should treat `device` as optional.

## Verification

Run the SDK telemetry example from the repository root and inspect the JSON printed or received by your local ingest endpoint:

```sh
cargo run -p xybrid-sdk --example telemetry_test
```

The example lives at [crates/xybrid-sdk/examples/telemetry_test.rs](../../crates/xybrid-sdk/examples/telemetry_test.rs). Use it as the copy point for initializing telemetry, publishing one event, flushing, and shutting the exporter down.
