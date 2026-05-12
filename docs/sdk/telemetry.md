# Telemetry

## Overview

The Rust SDK telemetry exporter sends pipeline events, a device profile, and the session and trace identifiers needed to group related events. Pipeline events describe inference lifecycle activity such as starts, completions, per-stage timings, and errors. Device context is attached so latency and throughput numbers can be interpreted against the machine that produced them.

For the contract that determines **which events share a `trace_id` and how they collapse to a single Traces row on the dashboard**, see [`trace-model.md`](trace-model.md).

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

## Cost-attribution fields

Inference events (`ModelComplete`, `PipelineComplete`) carry per-call attribution scalars on the payload top level so the platform can compute cost without descending into the span tree. All fields are optional and absent when unknown — consumers must tolerate missing keys.

| field | type | events | values | source |
|---|---|---|---|---|
| `backend` | string | inference | `llamacpp` \| `mlx` \| `mistralrs` \| `ort` \| `candle` \| `cloud` | `ExecutionTemplate` variant + `metadata.backend` hint (GGUF requires the hint; SafeTensors defaults to `candle` and accepts `mlx` to override on Apple Silicon); `cloud` for the cloud adapter |
| `provider` | string | inference (cloud only) | `openai` \| `anthropic` \| `google` \| `elevenlabs` \| `openrouter` \| `custom` | Cloud `IntegrationProvider` resolved from envelope metadata |
| `task` | string | inference | `chat` \| `vlm` \| `asr` \| `tts` \| `embedding` \| `image-gen` \| `ocr` \| `rerank` \| `classify` (open string for forward-compat) | `ModelMetadata.metadata["task"]` from `model_metadata.json` |
| `quantization` | string | inference | `q4_0` \| `q4_k_m` \| `q5_k_m` \| `q8_0` \| `fp16` \| `fp32` (open string — common GGUF labels) | `ModelMetadata.metadata["quantization"]` first; falls back to GGUF filename inference; absent (not empty) when unknown |
| `execution_provider` | string | inference (local only) | `coreml` \| `cpu` \| `metal` \| `cuda` \| `mlx-metal` \| `ane` (open string) | ORT path: harvested from per-session profiling JSON after the first inference (ORT exposes no session-level resolved-EP getter, so we read `args.provider` from the Chrome-trace output and pick the EP that ran the most ops). LLM path: build-flag-derived label keyed on the backend name. Cloud paths omit — `provider` carries attribution. |
| `prompt_cached_tokens` | u64 | inference (local LLM, llama.cpp only) | — | Count of prompt tokens served from the backend's KV cache on this call (longest common prefix with the previous turn). Local mirror of cloud's `cache_read_input_tokens`. Absent on first turns and for backends that don't track prefix reuse (cloud, mistralrs, mock). Only emitted when positive — `0` looks indistinguishable from "no cache" so the field stays absent rather than reporting a misleading zero. |
| `tokens_in` | u64 | inference | — | LLM span (`prompt_tokens` for OpenAI; synthesized total for Anthropic) |
| `tokens_out` | u64 | inference | — | LLM span (`completion_tokens`) |
| `cache_read_input_tokens` | u64 | inference | — | Anthropic-canonical; OpenAI's nested `prompt_tokens_details.cached_tokens` maps here |
| `cache_creation_input_tokens` | u64 | inference | — | Anthropic-only |

The closed set for `backend` is intentionally narrow — values outside it are not emitted (the field stays absent) so the analytics column can pin a closed enum without rejecting future runtimes mid-flight. Forward-declared backends (e.g. `mlx`) are added to the set only when a runtime adapter for them lands.

For local LLM events `provider` is always absent; for cloud events it is always present alongside `backend = "cloud"`.

`execution_provider` is the diagnostic complement to `backend`: `backend` says *which engine we asked for*, `execution_provider` says *what actually ran*. The two diverge most often on the ORT path (CoreML can silently fall back to CPU per-op when an op isn't supported) — the field is the analytics signal that explains "why is this run slow on this chip?" The field is absent for cloud events because cloud `provider` already attributes execution end-to-end.

`prompt_cached_tokens` is the local-LLM analogue of cloud's `cache_read_input_tokens`. Multi-turn workloads with a stable system prompt and conversation prefix routinely see 70-90% of the prompt served from the backend's KV cache on every call after the first — that's the difference between a 7B model feeling responsive and feeling sluggish, and it's also a billing-correctness signal (prefill is the expensive part; cached tokens shouldn't count toward "tokens processed"). Stack with `cache_read_input_tokens` on the same dashboard axis to compare local vs cloud cache savings.

## `ModelDownload` event

Emitted exactly once per successful registry download, after the network transfer completes and (when applicable) SHA256 verification passes. Cache hits do **not** produce this event — the metric represents bytes-on-the-wire, not cache traffic.

```json
{
  "event_type": "ModelDownload",
  "session_id": "...",
  "payload": {
    "status": "success",
    "latency_ms": 5432,
    "data": {
      "model_id": "kokoro-82m",
      "bytes_downloaded": 132456789,
      "source": "huggingface",
      "duration_ms": 5432
    }
  },
  "timestamp": "..."
}
```

| field | type | description |
|---|---|---|
| `model_id` | string | Registry mask, e.g. `kokoro-82m` |
| `bytes_downloaded` | u64 | Final on-disk size of the model file or `.xyb` bundle. Differs from the registry-declared expected size when upstream changed between resolve and fetch. |
| `source` | string | Canonical download host: `r2` for Xybrid's R2 mirror, `huggingface` for direct HF pulls, `other` for any other host (forward-compat so a future provider doesn't lose attribution). |
| `duration_ms` | u32 | Wallclock time inside the network download, excluding SHA256 verification and bundle extraction. Mirrored onto the top-level `latency_ms` so the existing latency column lights up. |

The event respects the same opt-out as the registry call telemetry: when `XYBRID_TELEMETRY_OPTOUT=1` is set at process start, no `ModelDownload` event is emitted (the two leak the same kind of attribution surface — which model the user pulled — so they share one gate).

## Verification

Run the SDK telemetry example from the repository root and inspect the JSON printed or received by your local ingest endpoint:

```sh
cargo run -p xybrid-sdk --example telemetry_test
```

The example lives at [crates/xybrid-sdk/examples/telemetry_test.rs](../../crates/xybrid-sdk/examples/telemetry_test.rs). Use it as the copy point for initializing telemetry, publishing one event, flushing, and shutting the exporter down.
