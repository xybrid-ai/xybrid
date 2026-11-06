//! Time source. The system clock is monotonic-anchored so durations never go
//! negative when the wall clock is adjusted; tests inject a manual clock.

use std::fmt;
#[cfg(feature = "test-util")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "test-util")]
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Observer time source.
#[derive(Clone)]
pub struct Clock(Inner);

#[derive(Clone)]
enum Inner {
    System {
        wall: SystemTime,
        mono: Instant,
    },
    #[cfg(feature = "test-util")]
    Manual(Arc<ManualClock>),
}

impl Clock {
    /// The system clock, anchored now.
    #[must_use]
    pub fn system() -> Self {
        Self(Inner::System {
            wall: SystemTime::now(),
            mono: Instant::now(),
        })
    }

    /// A manually driven clock (feature `test-util`).
    #[cfg(feature = "test-util")]
    #[must_use]
    pub fn manual(clock: Arc<ManualClock>) -> Self {
        Self(Inner::Manual(clock))
    }

    /// Current time.
    #[must_use]
    pub fn now(&self) -> SystemTime {
        match &self.0 {
            Inner::System { wall, mono } => *wall + mono.elapsed(),
            #[cfg(feature = "test-util")]
            Inner::Manual(clock) => clock.now(),
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
            Inner::System { .. } => f.write_str("Clock::System"),
            #[cfg(feature = "test-util")]
            Inner::Manual(_) => f.write_str("Clock::Manual"),
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

    /// Sets the time.
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

/// Seconds between two instants, zero if `end` precedes `start`.
pub(crate) fn seconds_between(start: SystemTime, end: SystemTime) -> f64 {
    end.duration_since(start)
        .unwrap_or(Duration::ZERO)
        .as_secs_f64()
}
