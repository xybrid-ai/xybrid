//! Test seams for resource-driven routing and the cloud-fallback wrapper.
//!
//! These types are visible to in-tree tests and to downstream consumers that
//! enable the `dev-tools` Cargo feature on `xybrid-core`. They let demos and
//! integration tests inject deterministic [`ResourceSnapshot`] values into a
//! [`LocalAuthority`](super::LocalAuthority) without standing up a real
//! [`ResourceMonitor`](crate::device::ResourceMonitor).
//!
//! Both providers are read-only after construction; `StagedResourceProvider`
//! tracks call counts via an atomic so it remains `Send + Sync`.

use crate::device::{ResourceSnapshot, ResourceSnapshotProvider};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Returns the same [`ResourceSnapshot`] on every call.
///
/// Use to drive a single deterministic state (e.g. `MemoryPressure::Critical`
/// for the entire run) when exercising the routing engine or the cloud-fallback
/// wrapper.
#[derive(Debug)]
pub struct FixedResourceProvider(pub ResourceSnapshot);

impl FixedResourceProvider {
    /// Construct a provider that always returns `snapshot`.
    pub fn new(snapshot: ResourceSnapshot) -> Self {
        Self(snapshot)
    }
}

impl ResourceSnapshotProvider for FixedResourceProvider {
    fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
        self.0
    }
}

/// Returns `normal` for the first `normal_reads` calls and `stressed`
/// thereafter.
///
/// This is the canonical setup for the cloud-fallback demo: the device
/// streams a few healthy tokens, then trips into resource pressure, then
/// the SDK aborts and re-runs on cloud.
#[derive(Debug)]
pub struct StagedResourceProvider {
    normal: ResourceSnapshot,
    stressed: ResourceSnapshot,
    normal_reads: usize,
    reads_so_far: AtomicUsize,
}

impl StagedResourceProvider {
    /// Construct with the default unknown-state baseline as `normal`.
    pub fn new(normal_reads: usize, stressed: ResourceSnapshot) -> Self {
        Self::with_baseline(ResourceSnapshot::unknown(), normal_reads, stressed)
    }

    /// Construct with explicit `normal` and `stressed` snapshots.
    pub fn with_baseline(
        normal: ResourceSnapshot,
        normal_reads: usize,
        stressed: ResourceSnapshot,
    ) -> Self {
        Self {
            normal,
            stressed,
            normal_reads,
            reads_so_far: AtomicUsize::new(0),
        }
    }

    /// Total number of times `current_snapshot` has been called.
    pub fn read_count(&self) -> usize {
        self.reads_so_far.load(Ordering::SeqCst)
    }
}

impl ResourceSnapshotProvider for StagedResourceProvider {
    fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
        let n = self.reads_so_far.fetch_add(1, Ordering::SeqCst);
        if n < self.normal_reads {
            self.normal
        } else {
            self.stressed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MemoryPressure;

    fn critical_snapshot() -> ResourceSnapshot {
        let mut snap = ResourceSnapshot::unknown();
        snap.memory_pressure = MemoryPressure::Critical;
        snap
    }

    #[test]
    fn fixed_provider_returns_same_snapshot_every_call() {
        let provider = FixedResourceProvider::new(critical_snapshot());

        let a = provider.current_snapshot(Duration::from_millis(0));
        let b = provider.current_snapshot(Duration::from_millis(0));

        assert_eq!(a.memory_pressure, MemoryPressure::Critical);
        assert_eq!(b.memory_pressure, MemoryPressure::Critical);
    }

    #[test]
    fn staged_provider_flips_after_n_reads() {
        let provider = StagedResourceProvider::new(3, critical_snapshot());

        for _ in 0..3 {
            let snap = provider.current_snapshot(Duration::from_millis(0));
            assert_ne!(snap.memory_pressure, MemoryPressure::Critical);
        }

        for _ in 0..2 {
            let snap = provider.current_snapshot(Duration::from_millis(0));
            assert_eq!(snap.memory_pressure, MemoryPressure::Critical);
        }

        assert_eq!(provider.read_count(), 5);
    }

    #[test]
    fn staged_provider_with_zero_normal_reads_flips_immediately() {
        let provider = StagedResourceProvider::new(0, critical_snapshot());

        let snap = provider.current_snapshot(Duration::from_millis(0));

        assert_eq!(snap.memory_pressure, MemoryPressure::Critical);
    }

    #[test]
    fn staged_provider_with_baseline_returns_baseline_first() {
        let mut normal = ResourceSnapshot::unknown();
        normal.memory_pressure = MemoryPressure::Warn;
        let provider = StagedResourceProvider::with_baseline(normal, 2, critical_snapshot());

        let r1 = provider.current_snapshot(Duration::from_millis(0));
        let r2 = provider.current_snapshot(Duration::from_millis(0));
        let r3 = provider.current_snapshot(Duration::from_millis(0));

        assert_eq!(r1.memory_pressure, MemoryPressure::Warn);
        assert_eq!(r2.memory_pressure, MemoryPressure::Warn);
        assert_eq!(r3.memory_pressure, MemoryPressure::Critical);
    }
}
