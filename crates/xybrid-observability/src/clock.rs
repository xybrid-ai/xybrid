//! Time source. Operation readings are wall-clock anchored when the operation
//! starts and monotonic within it, so durations never go negative when the
//! wall clock is adjusted mid-operation; tests inject a manual clock.

use std::fmt;
#[cfg(feature = "test-util")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "test-util")]
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Observer time source.
///
/// The system clock reads `SystemTime::now()` for calls that belong to no
/// operation (evaluation and feedback timestamps). Every operation captures
/// the wall and monotonic time when it starts; its later default readings are
/// the start wall time plus the monotonic time elapsed since then.
#[derive(Clone)]
pub struct Clock(Inner);

#[derive(Clone)]
enum Inner {
    System,
    #[cfg(feature = "test-util")]
    Manual(Arc<ManualClock>),
    #[cfg(feature = "test-util")]
    ManualWall(Arc<ManualClock>),
}

impl Clock {
    /// The system clock.
    #[must_use]
    pub fn system() -> Self {
        Self(Inner::System)
    }

    /// A manually driven clock (feature `test-util`). Its readings are used
    /// verbatim, also within operations.
    #[cfg(feature = "test-util")]
    #[must_use]
    pub fn manual(clock: Arc<ManualClock>) -> Self {
        Self(Inner::Manual(clock))
    }

    /// The system clock with its wall time read from `wall` (feature
    /// `test-util`). Calls that belong to no operation read `wall`; each
    /// operation anchors on `wall` when it starts and adds the monotonic time
    /// elapsed since then, as with the system clock. Moving `wall` while an
    /// operation runs models a wall-clock step (an NTP correction) that the
    /// operation's readings do not follow.
    #[cfg(feature = "test-util")]
    #[must_use]
    pub fn manual_wall(wall: Arc<ManualClock>) -> Self {
        Self(Inner::ManualWall(wall))
    }

    /// Current wall-clock time (or the manual clock's time).
    #[must_use]
    pub fn now(&self) -> SystemTime {
        match &self.0 {
            Inner::System => SystemTime::now(),
            #[cfg(feature = "test-util")]
            Inner::Manual(clock) | Inner::ManualWall(clock) => clock.now(),
        }
    }

    /// Anchors a clock for one operation, now.
    pub(crate) fn anchor(&self) -> OperationClock {
        match &self.0 {
            Inner::System => OperationClock(OpInner::System {
                wall: SystemTime::now(),
                mono: Instant::now(),
            }),
            #[cfg(feature = "test-util")]
            Inner::Manual(clock) => OperationClock(OpInner::Manual(clock.clone())),
            #[cfg(feature = "test-util")]
            Inner::ManualWall(wall) => OperationClock(OpInner::System {
                wall: wall.now(),
                mono: Instant::now(),
            }),
        }
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::system()
    }
}

impl fmt::Debug for Clock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Inner::System => f.write_str("Clock::System"),
            #[cfg(feature = "test-util")]
            Inner::Manual(_) => f.write_str("Clock::Manual"),
            #[cfg(feature = "test-util")]
            Inner::ManualWall(_) => f.write_str("Clock::ManualWall"),
        }
    }
}

/// The clock of one operation: anchored at its start (CONTRACT.md "Timing").
pub(crate) struct OperationClock(OpInner);

enum OpInner {
    System {
        wall: SystemTime,
        mono: Instant,
    },
    #[cfg(feature = "test-util")]
    Manual(Arc<ManualClock>),
}

impl OperationClock {
    /// `wall_start + (monotonic_now − monotonic_start)`, or the manual time.
    pub(crate) fn now(&self) -> SystemTime {
        match &self.0 {
            OpInner::System { wall, mono } => *wall + mono.elapsed(),
            #[cfg(feature = "test-util")]
            OpInner::Manual(clock) => clock.now(),
        }
    }
}

/// A clock that only moves when told to (feature `test-util`).
#[cfg(feature = "test-util")]
#[derive(Debug)]
pub struct ManualClock {
    nanos_since_epoch: AtomicU64,
}

#[cfg(feature = "test-util")]
impl ManualClock {
    /// A clock frozen at `at`.
    #[must_use]
    pub fn new(at: SystemTime) -> Self {
        Self {
            nanos_since_epoch: AtomicU64::new(nanos(at)),
        }
    }

    /// Current time.
    #[must_use]
    pub fn now(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_nanos(self.nanos_since_epoch.load(Ordering::SeqCst))
    }

    /// Sets the time (it may move backwards).
    pub fn set(&self, at: SystemTime) {
        self.nanos_since_epoch.store(nanos(at), Ordering::SeqCst);
    }

    /// Moves the clock forward.
    pub fn advance(&self, by: Duration) {
        let by = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        self.nanos_since_epoch.fetch_add(by, Ordering::SeqCst);
    }
}

#[cfg(feature = "test-util")]
fn nanos(at: SystemTime) -> u64 {
    at.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Seconds between two instants, clamped at zero when `end` precedes `start`.
pub(crate) fn seconds_between(start: SystemTime, end: SystemTime) -> f64 {
    end.duration_since(start)
        .unwrap_or(Duration::ZERO)
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_readings_follow_wall_start_plus_monotonic_elapsed() {
        // R-CLOCK-2
        let before = SystemTime::now();
        let clock = Clock::system().anchor();
        let start = clock.now();
        let after = SystemTime::now();
        assert!(start >= before.checked_sub(Duration::from_secs(1)).unwrap());
        assert!(start <= after + Duration::from_secs(1));
        let measured = Instant::now();
        std::thread::sleep(Duration::from_millis(20));
        let slept = measured.elapsed();
        let end = clock.now();
        assert!(seconds_between(start, end) >= slept.as_secs_f64() - 1e-3);
    }

    #[test]
    fn negative_spans_clamp_to_zero() {
        let t = SystemTime::UNIX_EPOCH + Duration::from_secs(10);
        assert_eq!(seconds_between(t, t - Duration::from_secs(1)), 0.0);
    }
}
