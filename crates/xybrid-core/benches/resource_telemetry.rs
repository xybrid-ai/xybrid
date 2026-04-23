//! Criterion bench for the resource-telemetry monitor (INF-32).
//!
//! SLO gates defended:
//!   * cached snapshot read:     < 100 µs
//!   * cache-miss refresh:       < 1 ms (warm `System`)
//!   * Boundary mode, per run:   < 1 ms
//!   * Summary @ 1000 ms:        < 1 % throughput hit vs Off
//!
//! Run with:
//!   cargo bench -p xybrid-core --bench resource_telemetry

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use xybrid_core::device::{ResourceMonitor, ResourceTelemetryMode};

fn bench_cached_snapshot(c: &mut Criterion) {
    let monitor = ResourceMonitor::new();
    monitor.prewarm();
    c.bench_function("resource_snapshot::cached_500ms", |b| {
        b.iter(|| {
            let _ = monitor.current_snapshot(Duration::from_millis(500));
        });
    });
}

fn bench_refresh_snapshot(c: &mut Criterion) {
    let monitor = ResourceMonitor::new();
    monitor.prewarm();
    c.bench_function("resource_snapshot::cache_miss_refresh", |b| {
        b.iter(|| {
            // max_age == 0 forces a refresh every call.
            let _ = monitor.current_snapshot(Duration::ZERO);
        });
    });
}

fn bench_begin_run_off(c: &mut Criterion) {
    let monitor = ResourceMonitor::new();
    c.bench_function("resource_run::off", |b| {
        b.iter(|| {
            let guard = monitor.begin_run(ResourceTelemetryMode::Off);
            let _ = guard.finish();
        });
    });
}

fn bench_begin_run_boundary(c: &mut Criterion) {
    let monitor = ResourceMonitor::new();
    monitor.prewarm();
    c.bench_function("resource_run::boundary", |b| {
        b.iter(|| {
            let guard = monitor.begin_run(ResourceTelemetryMode::Boundary);
            let _ = guard.finish();
        });
    });
}

criterion_group!(
    benches,
    bench_cached_snapshot,
    bench_refresh_snapshot,
    bench_begin_run_off,
    bench_begin_run_boundary,
);
criterion_main!(benches);
