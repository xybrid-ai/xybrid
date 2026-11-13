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

The SDK translates these summary fields to portable observations
([contract "Resource samples"](../../crates/xybrid-observability/CONTRACT.md#resource-samples)):

| Summary field | OTLP span attribute | Type | Conversion |
|---|---|---|---|
| `cpu_avg_pct` | `xybrid.genai.resource.cpu.utilization` | double, 0–1 | Percentage divided by 100 |
| `cpu_peak_pct` | `xybrid.genai.resource.cpu.peak_utilization` | double, 0–1 | Percentage divided by 100 |
| `available_mem_min_mb` | `xybrid.genai.resource.memory.min_available_bytes` | int, bytes | MiB multiplied by 1,048,576 |
| `memory_pressure_peak` | `xybrid.genai.resource.memory.pressure` | string | Normal/Warn/Critical → normal/warn/critical; Unknown → absent |
| `process_rss_peak_mb` | `xybrid.genai.resource.process.peak_rss_bytes` | int, bytes | MiB multiplied by 1,048,576 |
| `thermal_state_peak` | `xybrid.genai.resource.thermal.state` | string | Normal/Warm/Hot/Critical → nominal/fair/serious/critical; absent unless a platform reading supplied it |
| `battery_pct_end` | `xybrid.genai.resource.battery.level` | double, 0–1 | Percentage divided by 100 |

A summary field that was not measured stays absent; nothing is filled with zero,
`nominal` or `normal`. Peak and minimum values are extremes of the run's samples
(start and end bookends plus any periodic samples), not of continuous time, so a
short Boundary run reports the worse of its two readings. The core keeps memory
in whole MiB rounded down, so `memory.min_available_bytes` and
`process.peak_rss_bytes` are multiples of 1,048,576 and can understate the
reading by less than 1 MiB. A reading of 0 MiB available exports as `0`.
`memory.pressure` uses the same thresholds as the core classification above
(Critical below 5% of total memory available, Warn below 15%), evaluated per
sample; Unknown means no sample had both available and total memory.

CPU readings are not confined to the run. Each one is sysinfo's system-wide
utilization since the previous CPU refresh of the process-wide monitor,
whatever triggered it: SDK prewarm, an earlier or concurrent run, a live
snapshot. The start bookend's reading therefore describes the time before the
run. sysinfo also leaves CPU usage unchanged when two refreshes are less than
`sysinfo::MINIMUM_CPU_UPDATE_INTERVAL` apart (200 ms on Linux, Android, Apple
platforms and Windows), so the end bookend of a shorter run repeats the start reading.
`cpu.utilization` and `cpu.peak_utilization` can thus carry load from just
before the run, and a short Boundary run can carry nothing else. Read them as
system load around the run, not load the run caused. Memory, pressure, RSS,
thermal and battery readings are instantaneous and have no such lag.

These resource values are attached to spans, not emitted as separate resource
metric instruments, and are never metric dimensions. Sample count and sampling
configuration stay local. System memory in use (`memory.used_bytes`) is not
exported because the summary carries no device total to derive it from.
Accelerator utilization/memory, current process RSS and charging state are not
populated by this adapter. Do not infer those fields from legacy
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
