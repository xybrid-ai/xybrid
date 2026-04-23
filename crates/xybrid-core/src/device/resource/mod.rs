//! Resource telemetry primitive.
//!
//! Provides one retained [`sysinfo::System`] behind a process-wide
//! [`ResourceMonitor`]. Two producers sit on top:
//!
//! - [`ResourceMonitor::current_snapshot`] — synchronous TTL-cached read used
//!   on the inference hot path by adaptive execution.
//! - [`ResourceMonitor::begin_run`] — per-run sampler that produces one
//!   [`ResourceUsageSummary`] and attaches it to the outgoing telemetry
//!   event.
//!
//! The full contract lives in `docs/sdk/resource-telemetry.md`. Thresholds,
//! SLOs, and the privacy posture are defined there; this module implements
//! them.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};

use super::types::ThermalState;

pub mod pressure;

pub use pressure::MemoryPressure;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// How much resource data to collect per inference, and whether to keep raw
/// samples locally. Default is [`ResourceTelemetryMode::Off`] so existing
/// callers see no behavior change.
///
/// This enum is SDK-runtime state; the serialized wire form for the summary's
/// `sampling_mode` field is a flat label string (see [`Self::label`]) + a
/// separate `sampling_interval_ms` to match the dashboard's low-cardinality
/// column contract. This type itself doesn't participate in the wire payload.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ResourceTelemetryMode {
    /// Disabled. No samples, no summary.
    #[default]
    Off,
    /// Start + end snapshots only. No background task.
    Boundary,
    /// Active sampling during inference. One background task per process.
    Summary { interval_ms: u32 },
    /// Same as `Summary`, plus raw samples retained locally. Never uploaded.
    DebugLocal { interval_ms: u32 },
}

impl ResourceTelemetryMode {
    /// Default summary interval (1 s). Tuned by the resource-telemetry bench
    /// suite; if benchmarks argue for a different default, update this constant
    /// and the spec doc together.
    pub const DEFAULT_SUMMARY_INTERVAL_MS: u32 = 1000;
    /// Minimum allowed sample interval. The PRD pins the floor at 250 ms:
    /// anything below was not validated for overhead and would invalidate
    /// the bench assumptions. Values below are clamped up at config time.
    pub const MIN_SAMPLE_INTERVAL_MS: u32 = 250;

    /// Convenience constructor with the default interval.
    pub fn summary() -> Self {
        Self::Summary {
            interval_ms: Self::DEFAULT_SUMMARY_INTERVAL_MS,
        }
    }

    /// Clamp any configured interval to the [`MIN_SAMPLE_INTERVAL_MS`] floor.
    /// Off / Boundary pass through unchanged.
    pub fn normalized(self) -> Self {
        match self {
            Self::Summary { interval_ms } => Self::Summary {
                interval_ms: interval_ms.max(Self::MIN_SAMPLE_INTERVAL_MS),
            },
            Self::DebugLocal { interval_ms } => Self::DebugLocal {
                interval_ms: interval_ms.max(Self::MIN_SAMPLE_INTERVAL_MS),
            },
            other => other,
        }
    }

    pub fn is_off(&self) -> bool {
        matches!(self, Self::Off)
    }

    /// Does this mode need a background sampler task?
    pub fn needs_sampler(&self) -> bool {
        matches!(self, Self::Summary { .. } | Self::DebugLocal { .. })
    }

    /// `None` for modes without a periodic interval (Off / Boundary).
    pub fn interval_ms(&self) -> Option<u32> {
        match self {
            Self::Summary { interval_ms } | Self::DebugLocal { interval_ms } => Some(*interval_ms),
            _ => None,
        }
    }

    /// Flat string label for the dashboard's `sampling_mode` column and the
    /// on-wire `resource_summary.sampling_mode` field. Matches the variant
    /// names used in `docs/sdk/resource-telemetry.md` exactly.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Boundary => "boundary",
            Self::Summary { .. } => "summary",
            Self::DebugLocal { .. } => "debug_local",
        }
    }
}

/// Point-in-time observation of device resource state. See
/// `docs/sdk/resource-telemetry.md#field-reference` for field semantics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub cpu_pct: Option<f32>,
    pub process_rss_mb: Option<u32>,
    pub available_mem_mb: Option<u32>,
    pub total_mem_mb: Option<u32>,
    pub memory_pressure: MemoryPressure,
    pub thermal_state: ThermalState,
    pub battery_pct: Option<u8>,
    pub captured_at_ms: u64,
}

impl ResourceSnapshot {
    /// All-unknown snapshot. Returned when the monitor is misconfigured or
    /// a refresh fails; callers should never fail inference because a
    /// snapshot came back empty.
    pub fn unknown() -> Self {
        Self {
            cpu_pct: None,
            process_rss_mb: None,
            available_mem_mb: None,
            total_mem_mb: None,
            memory_pressure: MemoryPressure::Unknown,
            thermal_state: ThermalState::Normal,
            battery_pct: None,
            captured_at_ms: now_ms(),
        }
    }
}

/// Aggregate observation across a single `ModelComplete` / `PipelineComplete`
/// run. Attached to `event.data.resource_summary` and hoisted to the
/// platform-event payload top level by the SDK. The wire shape is flat
/// (no nested enum) so the analytics backend's low-cardinality string
/// column for `sampling_mode` can extract cleanly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceUsageSummary {
    pub cpu_avg_pct: Option<f32>,
    pub cpu_peak_pct: Option<f32>,
    pub process_rss_peak_mb: Option<u32>,
    pub available_mem_min_mb: Option<u32>,
    pub memory_pressure_peak: MemoryPressure,
    pub thermal_state_peak: ThermalState,
    pub battery_pct_end: Option<u8>,
    pub sample_count: u32,
    /// Label matching `ResourceTelemetryMode::label()`: `"off"`, `"boundary"`,
    /// `"summary"`, or `"debug_local"`. The dashboard's low-cardinality
    /// column stores this verbatim.
    pub sampling_mode: String,
    /// Configured sample interval for Summary / DebugLocal modes. `None`
    /// for Off / Boundary so the JSON payload stays compact on the 99 %
    /// cold path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_interval_ms: Option<u32>,
}

// ---------------------------------------------------------------------------
// Aggregator
// ---------------------------------------------------------------------------

/// Collapses a stream of [`ResourceSnapshot`]s into a [`ResourceUsageSummary`]
/// in place. Used by both the `Boundary` path (2 snapshots) and the `Summary`
/// sampler (N snapshots).
#[derive(Debug)]
struct Aggregator {
    cpu_sum: f64,
    cpu_samples: u32,
    cpu_peak: Option<f32>,
    rss_peak: Option<u32>,
    mem_avail_min: Option<u32>,
    pressure_peak: MemoryPressure,
    thermal_peak: ThermalState,
    latest_battery: Option<u8>,
    sample_count: u32,
}

impl Aggregator {
    fn new() -> Self {
        Self {
            cpu_sum: 0.0,
            cpu_samples: 0,
            cpu_peak: None,
            rss_peak: None,
            mem_avail_min: None,
            pressure_peak: MemoryPressure::Unknown,
            thermal_peak: ThermalState::Normal,
            latest_battery: None,
            sample_count: 0,
        }
    }

    fn observe(&mut self, s: &ResourceSnapshot) {
        self.sample_count = self.sample_count.saturating_add(1);

        if let Some(cpu) = s.cpu_pct {
            self.cpu_sum += cpu as f64;
            self.cpu_samples = self.cpu_samples.saturating_add(1);
            self.cpu_peak = Some(match self.cpu_peak {
                Some(peak) if peak >= cpu => peak,
                _ => cpu,
            });
        }
        if let Some(rss) = s.process_rss_mb {
            self.rss_peak = Some(match self.rss_peak {
                Some(peak) if peak >= rss => peak,
                _ => rss,
            });
        }
        if let Some(avail) = s.available_mem_mb {
            self.mem_avail_min = Some(match self.mem_avail_min {
                Some(min) if min <= avail => min,
                _ => avail,
            });
        }
        self.pressure_peak = self.pressure_peak.worse_of(s.memory_pressure);
        self.thermal_peak = thermal_worse_of(self.thermal_peak, s.thermal_state);
        if s.battery_pct.is_some() {
            self.latest_battery = s.battery_pct;
        }
    }

    fn finish(self, mode: ResourceTelemetryMode) -> ResourceUsageSummary {
        let cpu_avg_pct = if self.cpu_samples > 0 {
            Some((self.cpu_sum / self.cpu_samples as f64) as f32)
        } else {
            None
        };
        ResourceUsageSummary {
            cpu_avg_pct,
            cpu_peak_pct: self.cpu_peak,
            process_rss_peak_mb: self.rss_peak,
            available_mem_min_mb: self.mem_avail_min,
            memory_pressure_peak: self.pressure_peak,
            thermal_state_peak: self.thermal_peak,
            battery_pct_end: self.latest_battery,
            sample_count: self.sample_count,
            sampling_mode: mode.label().to_string(),
            sampling_interval_ms: mode.interval_ms(),
        }
    }
}

/// `ThermalState` doesn't implement Ord (variant order reads as severity but
/// that's not guaranteed long-term), so collapse explicitly.
fn thermal_worse_of(a: ThermalState, b: ThermalState) -> ThermalState {
    fn rank(t: ThermalState) -> u8 {
        match t {
            ThermalState::Normal => 0,
            ThermalState::Warm => 1,
            ThermalState::Hot => 2,
            ThermalState::Critical => 3,
        }
    }
    if rank(b) > rank(a) {
        b
    } else {
        a
    }
}

// ---------------------------------------------------------------------------
// ResourceMonitor
// ---------------------------------------------------------------------------

/// Process-wide holder for a single retained [`sysinfo::System`] + its
/// TTL-cached snapshot. Cheap to clone — it's an `Arc` internally.
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    inner: Arc<Mutex<Inner>>,
    /// Current process id, resolved once at construction. `None` means
    /// `sysinfo::get_current_pid()` refused to answer, in which case process
    /// RSS stays `None` for every snapshot.
    pid: Option<Pid>,
}

#[derive(Debug)]
struct Inner {
    system: System,
    cached: Option<ResourceSnapshot>,
    cached_at: Option<Instant>,
    /// Caches `total_memory` after the first refresh — it doesn't change for
    /// a process lifetime and avoids re-reading on every cache-miss.
    total_mem_mb: Option<u32>,
}

impl ResourceMonitor {
    /// Build a monitor without touching the global singleton. Useful in tests
    /// that want isolation. Production code should use [`ResourceMonitor::global`].
    pub fn new() -> Self {
        let system = System::new();
        let pid = sysinfo::get_current_pid().ok();
        Self {
            inner: Arc::new(Mutex::new(Inner {
                system,
                cached: None,
                cached_at: None,
                total_mem_mb: None,
            })),
            pid,
        }
    }

    /// Process-wide monitor. First caller pays the initialization cost; every
    /// later caller gets the same `Arc`.
    pub fn global() -> Arc<Self> {
        static MONITOR: OnceLock<Arc<ResourceMonitor>> = OnceLock::new();
        MONITOR.get_or_init(|| Arc::new(Self::new())).clone()
    }

    /// Pre-warm the monitor so the first inference doesn't pay the
    /// `sysinfo::System::refresh` cold-read cost. Safe to call repeatedly.
    pub fn prewarm(&self) {
        let _ = self.refresh_locked();
    }

    /// Return a snapshot no older than `max_age`. Pass `Duration::ZERO` to
    /// force a refresh. The cached read targets `< 100 µs`; a cache-miss
    /// refresh targets `< 1 ms` on a warm `System`.
    pub fn current_snapshot(&self, max_age: Duration) -> ResourceSnapshot {
        // Fast path: read the cached snapshot under the lock and return it
        // without calling into sysinfo.
        {
            let inner = match self.inner.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let (Some(snap), Some(at)) = (inner.cached, inner.cached_at) {
                if at.elapsed() <= max_age {
                    return snap;
                }
            }
        }
        // Slow path: refresh and cache.
        self.refresh_locked()
    }

    /// Start a per-run monitor scope. The returned [`RunGuard`] captures a
    /// start snapshot (for Boundary / Summary / DebugLocal modes) and begins
    /// sampling if the mode needs it. Dropping or `finish()`-ing the guard
    /// produces the final `ResourceUsageSummary`.
    pub fn begin_run(&self, mode: ResourceTelemetryMode) -> RunGuard {
        let mode = mode.normalized();
        if mode.is_off() {
            return RunGuard::disabled(mode);
        }
        let monitor = self.clone();
        let start = monitor.current_snapshot(Duration::ZERO);
        let mut aggregator = Aggregator::new();
        aggregator.observe(&start);

        let sampler = if mode.needs_sampler() {
            // Sampler wires up on `start_sampler`, which is only available
            // when tokio is available in the caller's context. We spawn
            // eagerly so the sample stream begins right away; aggregation
            // happens inside the guard on finish/drop.
            sampler::start(monitor.clone(), mode)
        } else {
            None
        };

        RunGuard {
            monitor: Some(monitor),
            mode,
            aggregator: Some(aggregator),
            sampler,
        }
    }

    // -- private helpers --

    fn refresh_locked(&self) -> ResourceSnapshot {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        // Cross-platform refreshes available on all sysinfo-supported targets.
        inner.system.refresh_memory();
        inner.system.refresh_cpu_all();
        if let Some(pid) = self.pid {
            // sysinfo 0.32 replaced `refresh_process(pid)` with the plural
            // form; the second bool prunes dead processes from the cache.
            inner
                .system
                .refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        }

        let total_bytes = inner.system.total_memory();
        let total_mb = bytes_to_mb(total_bytes);
        if total_mb.is_some() && inner.total_mem_mb.is_none() {
            inner.total_mem_mb = total_mb;
        }
        let available_mb = bytes_to_mb(inner.system.available_memory());
        let cpu = inner.system.global_cpu_usage();
        let cpu_pct = if cpu.is_finite() && cpu >= 0.0 {
            Some(cpu.min(100.0))
        } else {
            None
        };
        let process_rss_mb = self.pid.and_then(|pid| {
            inner
                .system
                .process(pid)
                .map(|p| bytes_to_mb_u64(p.memory()))
                .unwrap_or(None)
        });

        let snap = ResourceSnapshot {
            cpu_pct,
            process_rss_mb,
            available_mem_mb: available_mb,
            total_mem_mb: inner.total_mem_mb.or(total_mb),
            memory_pressure: MemoryPressure::derive(available_mb, inner.total_mem_mb.or(total_mb)),
            // Thermal + battery come from platform-specific bridges in a
            // later slice (iOS thermal state, Android power manager, etc.).
            // For now desktop/server report Normal thermal and no battery —
            // see the spec's availability table.
            thermal_state: ThermalState::Normal,
            battery_pct: None,
            captured_at_ms: now_ms(),
        };
        inner.cached = Some(snap);
        inner.cached_at = Some(Instant::now());
        snap
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RunGuard
// ---------------------------------------------------------------------------

/// Lifetime-scoped handle returned by [`ResourceMonitor::begin_run`]. Call
/// [`RunGuard::finish`] when the inference completes to collect the summary,
/// or let the guard drop — in which case any in-progress sampler is cancelled
/// and the summary is discarded.
pub struct RunGuard {
    monitor: Option<ResourceMonitor>,
    mode: ResourceTelemetryMode,
    aggregator: Option<Aggregator>,
    sampler: Option<sampler::Handle>,
}

impl RunGuard {
    /// Disabled guard — no snapshots, no summary.
    fn disabled(mode: ResourceTelemetryMode) -> Self {
        Self {
            monitor: None,
            mode,
            aggregator: None,
            sampler: None,
        }
    }

    /// Produce the final [`ResourceUsageSummary`]. Returns `None` when the
    /// guard was built with `Off` mode. Safe to call exactly once.
    pub fn finish(mut self) -> Option<ResourceUsageSummary> {
        let monitor = self.monitor.take()?;
        let mut aggregator = self.aggregator.take()?;

        // Drain any samples that the background task collected. `sampler` is
        // `None` in Boundary mode.
        if let Some(sampler) = self.sampler.take() {
            for snap in sampler.stop() {
                aggregator.observe(&snap);
            }
        }

        // End snapshot — always captured, including in Boundary mode.
        let end = monitor.current_snapshot(Duration::ZERO);
        aggregator.observe(&end);
        Some(aggregator.finish(self.mode))
    }
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        // Drop path: cancel the sampler without producing a summary. This
        // runs if the caller never called `finish()` — for example if the
        // surrounding inference panicked. We don't emit telemetry from here.
        if let Some(sampler) = self.sampler.take() {
            let _ = sampler.stop();
        }
    }
}

// ---------------------------------------------------------------------------
// Sampler (background thread)
// ---------------------------------------------------------------------------

mod sampler {
    //! Background sampling thread for Summary / DebugLocal modes.
    //!
    //! Uses `std::thread` rather than a tokio task so the sampler works
    //! identically whether the caller sits inside a tokio runtime or in a
    //! plain synchronous context — the sync `XybridModel::run` and
    //! `Pipeline::run` paths (the common case) would otherwise never see
    //! their sampler start. Stops on demand via an `AtomicBool`.
    //!
    //! The polling slice (25 ms) is shorter than the full sampling period
    //! so `stop()` is observed promptly even when a long interval is
    //! configured; inference that completes mid-interval is cancelled
    //! within at most one poll slice.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::{ResourceMonitor, ResourceSnapshot, ResourceTelemetryMode};

    /// How often the sampler thread wakes to check the stop flag. Short
    /// enough that cancellation is effectively immediate regardless of
    /// the configured sampling period.
    const POLL_SLICE_MS: u64 = 25;

    pub(super) struct Handle {
        shared: Arc<Shared>,
        thread: Option<thread::JoinHandle<()>>,
    }

    struct Shared {
        stop: AtomicBool,
        samples: Mutex<Vec<ResourceSnapshot>>,
    }

    pub(super) fn start(monitor: ResourceMonitor, mode: ResourceTelemetryMode) -> Option<Handle> {
        let interval_ms = mode.interval_ms()?;
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            samples: Mutex::new(Vec::new()),
        });
        let thread_shared = Arc::clone(&shared);
        let thread = thread::Builder::new()
            .name("xybrid-resource-sampler".to_string())
            .spawn(move || {
                let period = Duration::from_millis(interval_ms as u64);
                let poll_slice = Duration::from_millis(POLL_SLICE_MS);
                while !thread_shared.stop.load(Ordering::Relaxed) {
                    let deadline = Instant::now() + period;
                    // Sleep in short slices so `stop()` cancels within
                    // POLL_SLICE_MS regardless of how large `period` is.
                    while Instant::now() < deadline {
                        if thread_shared.stop.load(Ordering::Relaxed) {
                            return;
                        }
                        thread::sleep(poll_slice);
                    }
                    if thread_shared.stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let snap = monitor.current_snapshot(Duration::ZERO);
                    if let Ok(mut samples) = thread_shared.samples.lock() {
                        samples.push(snap);
                    }
                }
            })
            .ok()?;

        Some(Handle {
            shared,
            thread: Some(thread),
        })
    }

    impl Handle {
        /// Signal the thread to stop, join it, and take the buffered
        /// samples. Safe to call exactly once.
        pub(super) fn stop(mut self) -> Vec<ResourceSnapshot> {
            self.shared.stop.store(true, Ordering::Relaxed);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
            match self.shared.samples.lock() {
                Ok(mut g) => std::mem::take(&mut *g),
                Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn bytes_to_mb(bytes: u64) -> Option<u32> {
    if bytes == 0 {
        None
    } else {
        Some((bytes / (1024 * 1024)) as u32)
    }
}

fn bytes_to_mb_u64(bytes: u64) -> Option<u32> {
    bytes_to_mb(bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_normalizes_interval_floor() {
        let m = ResourceTelemetryMode::Summary { interval_ms: 10 }.normalized();
        assert_eq!(
            m,
            ResourceTelemetryMode::Summary {
                interval_ms: ResourceTelemetryMode::MIN_SAMPLE_INTERVAL_MS
            }
        );

        let boundary = ResourceTelemetryMode::Boundary.normalized();
        assert_eq!(boundary, ResourceTelemetryMode::Boundary);
    }

    #[test]
    fn aggregator_tracks_peak_avg_min() {
        let mut agg = Aggregator::new();
        agg.observe(&ResourceSnapshot {
            cpu_pct: Some(10.0),
            process_rss_mb: Some(100),
            available_mem_mb: Some(8000),
            total_mem_mb: Some(16000),
            memory_pressure: MemoryPressure::Normal,
            thermal_state: ThermalState::Normal,
            battery_pct: Some(80),
            captured_at_ms: 0,
        });
        agg.observe(&ResourceSnapshot {
            cpu_pct: Some(90.0),
            process_rss_mb: Some(500),
            available_mem_mb: Some(1000),
            total_mem_mb: Some(16000),
            memory_pressure: MemoryPressure::Warn,
            thermal_state: ThermalState::Hot,
            battery_pct: Some(78),
            captured_at_ms: 1_000,
        });
        let summary = agg.finish(ResourceTelemetryMode::Boundary);
        assert_eq!(summary.cpu_peak_pct, Some(90.0));
        assert!((summary.cpu_avg_pct.unwrap() - 50.0).abs() < 0.01);
        assert_eq!(summary.process_rss_peak_mb, Some(500));
        assert_eq!(summary.available_mem_min_mb, Some(1000));
        assert_eq!(summary.memory_pressure_peak, MemoryPressure::Warn);
        assert_eq!(summary.thermal_state_peak, ThermalState::Hot);
        assert_eq!(summary.battery_pct_end, Some(78));
        assert_eq!(summary.sample_count, 2);
    }

    #[test]
    fn aggregator_handles_all_missing_cpu() {
        let mut agg = Aggregator::new();
        agg.observe(&ResourceSnapshot::unknown());
        agg.observe(&ResourceSnapshot::unknown());
        let summary = agg.finish(ResourceTelemetryMode::Boundary);
        assert_eq!(summary.cpu_avg_pct, None);
        assert_eq!(summary.cpu_peak_pct, None);
        assert_eq!(summary.sample_count, 2);
    }

    #[test]
    fn monitor_current_snapshot_is_cached_within_max_age() {
        let monitor = ResourceMonitor::new();
        let first = monitor.current_snapshot(Duration::ZERO);
        // Any non-zero max_age larger than our refresh window should hit the
        // cache. Use a generous 10 s to keep the test deterministic.
        let second = monitor.current_snapshot(Duration::from_secs(10));
        assert_eq!(first.captured_at_ms, second.captured_at_ms);
    }

    #[test]
    fn monitor_current_snapshot_refreshes_when_ttl_expires() {
        let monitor = ResourceMonitor::new();
        let first = monitor.current_snapshot(Duration::ZERO);
        // Force a refresh with max_age == 0 (strict less-than-or-equal).
        std::thread::sleep(Duration::from_millis(2));
        let second = monitor.current_snapshot(Duration::ZERO);
        assert!(second.captured_at_ms >= first.captured_at_ms);
    }

    #[test]
    fn begin_run_off_returns_no_summary() {
        let monitor = ResourceMonitor::new();
        let guard = monitor.begin_run(ResourceTelemetryMode::Off);
        assert!(guard.finish().is_none());
    }

    #[test]
    fn begin_run_boundary_produces_two_sample_summary() {
        let monitor = ResourceMonitor::new();
        let guard = monitor.begin_run(ResourceTelemetryMode::Boundary);
        let summary = guard.finish().expect("Boundary mode produces a summary");
        assert_eq!(summary.sample_count, 2);
        assert_eq!(summary.sampling_mode, "boundary");
        assert_eq!(summary.sampling_interval_ms, None);
    }

    #[test]
    fn begin_run_summary_mode_collects_interval_samples() {
        // Sampler uses std::thread, so no tokio runtime is required. Sleep
        // on the thread directly. Interval is the normalized floor
        // (250 ms); 700 ms gives ~2 mid-samples even on slow schedulers.
        let monitor = ResourceMonitor::new();
        let guard = monitor.begin_run(ResourceTelemetryMode::Summary { interval_ms: 250 });
        std::thread::sleep(Duration::from_millis(700));
        let summary = guard.finish().expect("Summary mode produces a summary");
        assert!(
            summary.sample_count >= 3,
            "expected at least start + 1 mid + end samples, got {}",
            summary.sample_count
        );
        assert_eq!(summary.sampling_mode, "summary");
        assert_eq!(summary.sampling_interval_ms, Some(250));
    }

    #[test]
    fn dropping_guard_without_finish_does_not_emit() {
        let monitor = ResourceMonitor::new();
        {
            let _guard = monitor.begin_run(ResourceTelemetryMode::Boundary);
            // Drop without calling finish().
        }
        // The monitor itself still works after a guard drop — no deadlock
        // from the sampler cancellation path.
        let snap = monitor.current_snapshot(Duration::ZERO);
        assert!(snap.captured_at_ms > 0);
    }

    #[test]
    fn global_monitor_is_shared() {
        let a = ResourceMonitor::global();
        let b = ResourceMonitor::global();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn summary_run_shorter_than_sampler_tick_still_produces_summary() {
        // Sampler can't tick before a near-instant inference returns —
        // the guard must still produce a well-formed summary from the
        // start+end bookends rather than failing the run. This is the
        // graceful-degradation promise for Summary mode: even if zero
        // samples come from the background thread, sample_count >= 2.
        let monitor = ResourceMonitor::new();
        monitor.prewarm();
        let guard = monitor.begin_run(ResourceTelemetryMode::Summary { interval_ms: 1000 });
        let summary = guard
            .finish()
            .expect("Summary mode should produce a summary even without sampler ticks");
        assert!(summary.sample_count >= 2);
        assert_eq!(summary.sampling_mode, "summary");
        assert_eq!(summary.sampling_interval_ms, Some(1000));
    }

    #[test]
    fn run_guard_drop_without_finish_stops_sampler_cleanly() {
        // Abandoning a guard (e.g. surrounding inference panicked)
        // must cancel the sampler without panicking or leaking the
        // thread. We can't observe join() state externally, so verify
        // by running many short drops back-to-back — if the sampler
        // didn't stop, the thread count would grow without bound and
        // the test would eventually allocate-deadlock or panic.
        let monitor = ResourceMonitor::new();
        monitor.prewarm();
        for _ in 0..64 {
            let guard = monitor.begin_run(ResourceTelemetryMode::Summary { interval_ms: 250 });
            drop(guard);
        }
        // If we got here, the drop path works.
    }

    #[test]
    fn concurrent_snapshots_share_one_system() {
        // Several threads hammering `current_snapshot` should not panic on
        // poisoned locks or construct multiple `sysinfo::System`s. We can't
        // assert on the number of sysinfo instances directly, but we can
        // assert that all threads get plausible data and don't deadlock.
        let monitor = ResourceMonitor::global();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let m = monitor.clone();
                std::thread::spawn(move || {
                    for _ in 0..32 {
                        let s = m.current_snapshot(Duration::from_millis(100));
                        assert!(s.captured_at_ms > 0);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
