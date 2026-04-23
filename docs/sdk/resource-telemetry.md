# Resource Telemetry

> **Status**: Draft spec, Slice 1 of [INF-23](https://linear.app/xybrid/issue/INF-23).
> **Owner**: Inference team. **Last updated**: 2026-04-23.

Resource telemetry attaches CPU, memory, process RSS, memory pressure, thermal,
and battery context to each Xybrid inference trace so developers can see whether
slow or failed inference was caused by the model or by device pressure.

The same primitive also exposes a synchronous live snapshot consumed internally
by Xybrid's runtime selector and throttling decisions, so runtime choices adapt
to current device state instead of frozen init-time values.

## Contents

- [Goals and non-goals](#goals-and-non-goals)
- [Modes](#modes)
- [Field reference](#field-reference)
- [Live-snapshot surface](#live-snapshot-surface)
- [Trace payload placement](#trace-payload-placement)
- [Privacy guarantees](#privacy-guarantees)
- [Defaults and tuning](#defaults-and-tuning)
- [Compatibility](#compatibility)

## Goals and non-goals

**Goals**

- One shape (`ResourceSnapshot`) consumed by two producers:
  - a synchronous TTL-cached live-snapshot read, used by Xybrid's own runtime
    decisions on the inference hot path;
  - an asynchronous sampler, scoped to one `XybridModel::run*` or
    `Pipeline::run*` call, that produces a single `ResourceUsageSummary` and
    attaches it to the `ModelComplete` / `PipelineComplete` telemetry event.
- Single retained `sysinfo::System` instance, pre-warmed at platform-telemetry
  init so the first inference does not pay the `~100 ms` cold-refresh cost.
- Sampling that never fails the inference. A sampler error produces a partial
  summary rather than propagating up.
- Wire format additive to existing telemetry: `resource_summary` is an optional
  field on `event.data`; consumers that don't know about it ignore it.

**Non-goals**

- Raw per-sample time series in uploaded telemetry. Raw samples stay local
  unless `debug_local` mode is explicitly selected.
- Per-token resource sampling. Streaming callbacks annotate the existing trace;
  they do not spawn new monitors.
- GPU / ANE / NPU utilization (separate PRD).
- Prompting users for new permissions (mobile in particular — only
  permission-free signals are collected).

## Modes

`ResourceTelemetryMode` is the caller-facing switch. Set on `TelemetryConfig`
via `.with_resource_telemetry(...)`.

| Mode | Sampler thread | Collected per inference | Default for |
|---|---|---|---|
| `Off` | No | Nothing | Anonymous telemetry |
| `Boundary` | No | Start + end `ResourceSnapshot` only | — |
| `Summary { interval_ms }` | Yes (one per process) | Start snapshot + periodic samples + end snapshot, aggregated into `ResourceUsageSummary` | Authenticated telemetry, default `interval_ms: 1000` |
| `DebugLocal { interval_ms }` | Yes | Same as `Summary`, plus raw samples retained locally (never uploaded) | Manual opt-in for local debugging |

`Boundary` is cheap: no background task, two synchronous `sysinfo::System`
refreshes around the inference. `Summary` spawns one per-process async task
that samples on the configured interval; aggregation happens in place and the
task shuts down when the last in-flight run completes.

Env override, takes precedence over the config value:
`XYBRID_RESOURCE_TELEMETRY=off|boundary|summary|debug_local`.

## Field reference

### `ResourceSnapshot` (producer output)

Single point-in-time observation.

| Field | Type | Unit | Nullable | Notes |
|---|---|---|---|---|
| `cpu_pct` | `f32` | % [0.0 – 100.0] | yes | Global CPU usage (sysinfo `global_cpu_usage`). First-call value may be `None` on some platforms; sampler discards it. |
| `process_rss_mb` | `u32` | MiB | yes | Resident set size for the current process. Platform-dependent availability. |
| `available_mem_mb` | `u32` | MiB | yes | System-wide available memory. |
| `total_mem_mb` | `u32` | MiB | yes | Cached after first read; stable for the process lifetime. |
| `memory_pressure` | `MemoryPressure` | enum | no | Derived from `available_mem_mb / total_mem_mb`. `Unknown` when either field is `None`. |
| `thermal_state` | `ThermalState` | enum | no | Defaults to `Normal` on platforms without native thermal signals. Enriched by mobile native bridges (INF-26). |
| `battery_pct` | `u8` | % [0 – 100] | yes | Permission-free only. `None` on desktop and on mobile when not available. |
| `captured_at_ms` | `u64` | ms since epoch | no | For sampler debugging + deterministic tests. |

### `MemoryPressure`

Derived, not sampled. Purely a function of the available/total ratio.

| Variant | Threshold (`available / total`) |
|---|---|
| `Normal` | `>= 15 %` |
| `Warn` | `[5 %, 15 %)` |
| `Critical` | `< 5 %` |
| `Unknown` | either `available_mem_mb` or `total_mem_mb` is `None` |

Thresholds are a first-cut heuristic. Tuning is tracked against benchmark data
in INF-32; changes must be rolled with a config version bump and corresponding
dashboard legend update.

### `ResourceUsageSummary` (sampler output)

One per `ModelComplete` / `PipelineComplete`. All `_peak` / `_min` values are
taken across samples; `cpu_avg_pct` is a simple mean.

| Field | Type | Unit | Aggregation |
|---|---|---|---|
| `cpu_avg_pct` | `f32?` | % | mean across samples, `None` when zero samples had a CPU reading |
| `cpu_peak_pct` | `f32?` | % | max |
| `process_rss_peak_mb` | `u32?` | MiB | max |
| `available_mem_min_mb` | `u32?` | MiB | min |
| `memory_pressure_peak` | `MemoryPressure` | enum | worst (Critical > Warn > Normal > Unknown) |
| `thermal_state_peak` | `ThermalState` | enum | worst (Critical > Hot > Warm > Normal) |
| `battery_pct_end` | `u8?` | % | value from final snapshot |
| `sample_count` | `u32` | count | number of snapshots aggregated |
| `sampling_mode` | `string` | label | `"off"` / `"boundary"` / `"summary"` / `"debug_local"` — the label of the mode that produced this summary. Flat string so Tinybird's `LowCardinality(String)` column extracts cleanly. |
| `sampling_interval_ms` | `u32?` | ms | configured interval for `summary` / `debug_local`; absent on `off` / `boundary` |

## Live-snapshot surface

Adaptive execution (INF-30) consumes live snapshots synchronously on the hot
path. The same primitive backs the sampler, so there is never more than one
retained `sysinfo::System` in the process.

```rust
pub fn current_snapshot(&self, max_age: Duration) -> ResourceSnapshot;
```

Semantics:

- Returns the cached snapshot if it is younger than `max_age`.
- Otherwise refreshes the retained `System`, produces a new snapshot, and
  returns it.
- **Cached read**: `< 100 µs` target on desktop-class hardware. Validated by
  benchmark in INF-32.
- **Cache-miss refresh**: `< 1 ms` target on a warm `System`. First call after
  process start is allowed to exceed this.

Default TTL for internal callers: `500 ms`. Callers may pass `Duration::ZERO`
to force a refresh.

The live-snapshot API is **internal Rust surface only**. It is not exposed
through any FFI binding in the MVP.

## Trace payload placement

### Wire shape

`resource_summary` is a sibling of the existing cache-token fields on the
publisher's `event.data` JSON. Example `ModelComplete` payload, abbreviated:

```jsonc
{
  "model_id": "qwen2.5-0.5b",
  "tokens_in": 42,
  "tokens_out": 128,
  "resource_summary": {
    "cpu_avg_pct": 34.1,
    "cpu_peak_pct": 62.5,
    "process_rss_peak_mb": 712,
    "available_mem_min_mb": 4180,
    "memory_pressure_peak": "normal",
    "thermal_state_peak": "normal",
    "battery_pct_end": 72,
    "sample_count": 4,
    "sampling_mode": "summary"
  }
}
```

### SDK hoist

`xybrid-sdk::telemetry::convert_to_platform_event` hoists `resource_summary` to
the platform-event payload top level (same mechanism as `tokens_in`,
`cache_read_input_tokens`). This lets Tinybird `json:$.resource_summary.*`
column extraction work without teaching the ingest service the nested shape.

### Storage

Tinybird `telemetry_events` datasource adds one typed column per summary field
(Slice 2 / INF-31). Legacy rows keep `NULL` for every new column; the dashboard
hides its Resource Usage card when all fields are `NULL`.

## Privacy guarantees

Resource telemetry is **device-level only**. It describes the machine's
operational state, not the user or their content.

### Collected

Only the fields listed in [`ResourceSnapshot`](#resourcesnapshot-producer-output)
and [`ResourceUsageSummary`](#resourceusagesummary-sampler-output), plus the
existing `DeviceProfile` fields that were already on the wire before this spec
(`chip_family`, `ram_gb`, `os`, `os_version`, `kernel_version`, `arch`).

### Never collected

- Prompts, completions, or any user content.
- File paths, document names, or the contents of loaded models.
- Hostnames (they stay behind the existing `with_capture_hostname` opt-in and
  are independent of this spec).
- Process lists, other applications' resource use, or anything outside the
  current process.
- Per-sample raw observations — **unless** the caller explicitly opts into
  `DebugLocal` mode, in which case the samples stay on disk locally and are
  never uploaded.
- GPS, IP address, or any network-identifying signal.

### Mode table

| Mode | Sends to ingest | Keeps locally |
|---|---|---|
| `Off` | nothing | nothing |
| `Boundary` | summary (2-sample aggregate) | nothing |
| `Summary` | summary (N-sample aggregate) | nothing |
| `DebugLocal` | summary (N-sample aggregate) | raw samples in `${XYBRID_CACHE}/resource-debug/` |

### Anonymous telemetry

The anonymous telemetry channel stays `Off` by default. Enabling resource
telemetry on an anonymous SDK instance requires an explicit opt-in on
`TelemetryConfig::anonymous(...).with_resource_telemetry(...)` (tracked under
the anonymous telemetry PRD, not this one).

## Defaults and tuning

- **Authenticated telemetry**: `Summary { interval_ms: 1000 }`.
- **Anonymous telemetry**: `Off`.
- **Minimum sample interval**: `250 ms`. Values below this are clamped up by
  the SDK at config time. The PRD pins this floor; intervals below `250 ms`
  were not validated for overhead.
- **Live-snapshot TTL**: `500 ms` for internal callers.

Benchmarks in INF-32 lock in the SLOs:

- `Boundary` mode: overhead per inference `< 1 ms`.
- `Summary { interval_ms: 1000 }`: throughput impact `< 1 %` on the
  `llm_streaming_metrics` fixture.
- Cached live-snapshot read: `< 100 µs`.
- Cache-miss refresh: `< 1 ms`.

If any of these regress in CI, the default flips to `Boundary` until the
regression is resolved.

## Compatibility

- **Old SDK clients** (no resource telemetry) produce events without a
  `resource_summary` field. Ingest accepts them; the dashboard's Resource
  Usage card hides.
- **Old backend / old dashboard** sees events that carry `resource_summary`
  but ignores it — the wire format is additive.
- **Rollback**: remove the `.with_resource_telemetry(...)` call on
  `TelemetryConfig`; sampling stops, summaries stop flowing, no schema
  migration required. Tinybird columns default to `NULL` and stay empty.

## Platform availability

Cross-platform availability of each signal is tracked in Slice 4 (INF-26) when
the native mobile bridges land. Until then, the desktop / server picture is:

| Signal | macOS | Linux | Windows |
|---|---|---|---|
| `cpu_pct` | yes | yes | yes |
| `process_rss_mb` | yes | yes | yes |
| `available_mem_mb` / `total_mem_mb` | yes | yes | yes |
| `memory_pressure` | derived | derived | derived |
| `thermal_state` | native (Slice 4 extends) | `Normal` (no native) | `Normal` (no native) |
| `battery_pct` | on portables | on portables | on portables |

Missing signals always degrade to `None` / `Unknown`; they never fail the
summary.

## References

- [INF-23 (epic)](https://linear.app/xybrid/issue/INF-23) — goal, acceptance criteria.
- [INF-24 (spec)](https://linear.app/xybrid/issue/INF-24) — this document.
- [INF-25 (core sampler)](https://linear.app/xybrid/issue/INF-25) — implementation ticket.
- [INF-27 (SDK config)](https://linear.app/xybrid/issue/INF-27) — public API surface.
- [INF-29 (trace attachment)](https://linear.app/xybrid/issue/INF-29) — wire-up in `Model::run*` / `Pipeline::run*`.
- [INF-30 (adaptive execution)](https://linear.app/xybrid/issue/INF-30) — live-snapshot consumer.
- [INF-31 (dashboard)](https://linear.app/xybrid/issue/INF-31) — Resource Usage card.
- [INF-32 (validation)](https://linear.app/xybrid/issue/INF-32) — SLO benchmarks.
- `crates/xybrid-core/src/device/resource/` — Rust source.
- `docs/sdk/telemetry.md` — sibling doc for the existing platform-telemetry surface.
