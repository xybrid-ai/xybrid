# Resource telemetry

Resource sampling adds device-pressure context to observed inference without
changing execution results. The current exporter is
[`xybrid_sdk::observability`](telemetry.md), not the retired `TelemetryConfig` /
`HttpTelemetryExporter` event uploader. Sampling defaults to **Off**, including
when an authenticated Xybrid trace exporter is enabled.

## Configure sampling

```rust,no_run
use xybrid_core::device::ResourceTelemetryMode;
use xybrid_sdk::observability::{self, ContentPolicy, ObservabilityConfig, OtlpConfig};

observability::init_observability(
    ObservabilityConfig::otlp(OtlpConfig::collector("http://localhost:4318"))
        .content(ContentPolicy::metadata_only())
        .resource_telemetry(ResourceTelemetryMode::summary()),
)?;
# Ok::<(), xybrid_sdk::SdkError>(())
```

`XYBRID_RESOURCE_TELEMETRY=off|boundary|summary|debug_local` overrides the
configured sampling mode when observability is initialized. This does not turn
on export by itself. Resource sampling is independent of content capture and of
trace sampling/suppression.

The mode belongs to the session. An operation pinned to an
`ObservabilitySession` (`ObservationOptions::with_session`) samples with that
session's mode whether or not it is the default. Operations that select no
session use the process-wide mode. Installing a default session sets it to that
session's mode; `xybrid_sdk::init().resource_telemetry(...)` without an
observability configuration also sets it, and the most recent setting wins. An
operation fixes its mode when it starts, so replacing the default session does
not change sampling for work already under way.

| Mode | Per-run collection | Periodic sampler | Separate raw-sample output |
|---|---|---|---|
| `Off` | None | No | No |
| `Boundary` | Start/end aggregate | No | No |
| `Summary { interval_ms }` | Start/end and periodic aggregate | Per run | No |
| `DebugLocal { interval_ms }` | Same aggregate as Summary | Per run | Not implemented |

`ResourceTelemetryMode::summary()` uses a 1,000 ms interval. Intervals are
clamped to a minimum of 250 ms. A short sampled run can have only its two
bookend samples; that alone does not indicate a failure. Sampling failures or
unavailable signals produce partial observations, rather than failing inference.
The current core implementation treats DebugLocal like Summary with a distinct
mode label: it does not implement a separate raw-sample file writer. Periodic
samples are buffered in memory until aggregation; they are not uploaded.

## Field reference

The core [`ResourceSnapshot` and `ResourceUsageSummary`
types](../../crates/xybrid-core/src/device/resource/mod.rs) are richer than the
portable exported resource sample. They are not interchangeable wire schemas.

A snapshot contains global CPU percentage, current-process RSS, available/total
memory, derived memory pressure, thermal state, battery percentage and capture
time. Signal availability varies by platform; missing numeric readings are
optional. Memory pressure uses available/total memory: Normal at or above 15%,
Warn from 5% to below 15%, Critical below 5%, Unknown when unavailable.

A per-run summary retains average/peak CPU, peak process RSS, minimum available
memory, worst memory/thermal pressure, ending battery percentage, sample count,
mode and interval. `sample_count` includes start and end bookends plus any
periodic samples. These aggregate fields remain available to local diagnostics.

The SDK currently translates this subset to portable observations:

| Summary field | OTLP span attribute | Conversion |
|---|---|---|
| `cpu_avg_pct` | `xybrid.genai.resource.cpu.utilization` | Percentage divided by 100 |
| `process_rss_peak_mb` | `xybrid.genai.resource.process.peak_rss_bytes` | MiB multiplied by 1,048,576 |
| `thermal_state_peak` | `xybrid.genai.resource.thermal.state` | Normal/Warm/Hot/Critical → nominal/fair/serious/critical |
| `battery_pct_end` | `xybrid.genai.resource.battery.level` | Percentage divided by 100 |

These resource values are attached to spans, not emitted as separate resource
metric instruments by the current adapter. Available/minimum system memory, memory pressure, peak CPU,
sample count and sampling configuration are not currently mapped by the SDK to
portable resource attributes. Accelerator utilization/memory and charging state
are not populated by this adapter. Do not infer those fields from legacy
`event.data.resource_summary` examples.

## Live snapshots

The core `ResourceMonitor` retains one `sysinfo::System` and exposes
`current_snapshot(max_age: Duration)` for adaptive execution. A fresh cached
snapshot is reused; an expired one refreshes the retained monitor. Internal
callers use a 500 ms TTL; `Duration::ZERO` forces a refresh. This core Rust
primitive does not itself export telemetry.

The existing resource benchmark's design targets are under 100 µs for a cached
read, under 1 ms for a warm refresh/boundary collection, and under 1% throughput
impact for one-second summary sampling. They are performance targets, not a
promise for every platform. See `crates/xybrid-core/benches/resource_telemetry.rs`
for the benchmark; no historical measurement substitutes for testing a target
application/device.

## Privacy and local-event separation

Resource measurements contain operational state, not prompts, file paths,
model contents, hostnames or individual process lists. System-wide CPU/memory
and current-process RSS are distinct signals. Sampling may coexist with prompt
capture configured elsewhere; enabling metadata-only content capture does not
disable these resource measurements.

Detected OS/architecture/chip/RAM attributes are configured separately with
`ObservabilityConfig::device_attributes(false)`; the automatic OTLP resource
never includes a hostname. Application-supplied resource attributes remain the
application's responsibility.

Local `ModelComplete` / `PipelineComplete` event subscribers may still receive
`resource_summary` in event data. Those in-process events are not forwarded by
the OTLP exporter. Forwarding or persisting local events or snapshots
is application-owned and independent of OTLP privacy settings.
