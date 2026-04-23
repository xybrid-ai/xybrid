//! Memory-pressure derivation.
//!
//! `MemoryPressure` is derived from the available/total memory ratio; it is
//! not sampled directly. Thresholds are intentionally coarse — they exist to
//! classify device state, not to surface precise measurements. See
//! `docs/sdk/resource-telemetry.md` for the public semantics.

use serde::{Deserialize, Serialize};

/// Derived classification of system memory pressure.
///
/// Ordered so that [`MemoryPressure::worse_of`] can collapse a stream of
/// snapshots into the peak pressure observed across a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryPressure {
    #[default]
    Unknown,
    Normal,
    Warn,
    Critical,
}

impl MemoryPressure {
    /// Thresholds are `< 5 %` critical / `< 15 %` warn / else normal. `None`
    /// on either input returns `Unknown` so downstream code can distinguish
    /// "didn't measure" from "measured, pressure is fine".
    pub fn derive(available_mb: Option<u32>, total_mb: Option<u32>) -> Self {
        let (Some(avail), Some(total)) = (available_mb, total_mb) else {
            return Self::Unknown;
        };
        if total == 0 {
            return Self::Unknown;
        }
        let ratio = avail as f64 / total as f64;
        if ratio < 0.05 {
            Self::Critical
        } else if ratio < 0.15 {
            Self::Warn
        } else {
            Self::Normal
        }
    }

    /// Collapse two pressure readings to the worst of the two. `Unknown` is
    /// the baseline so a real reading always wins over a missing one.
    pub fn worse_of(self, other: Self) -> Self {
        // Can't derive Ord because the variant order is insertion order
        // (Unknown first) rather than severity order. Hand-rank here.
        fn rank(p: MemoryPressure) -> u8 {
            match p {
                MemoryPressure::Unknown => 0,
                MemoryPressure::Normal => 1,
                MemoryPressure::Warn => 2,
                MemoryPressure::Critical => 3,
            }
        }
        if rank(other) > rank(self) {
            other
        } else {
            self
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryPressure::Unknown => "unknown",
            MemoryPressure::Normal => "normal",
            MemoryPressure::Warn => "warn",
            MemoryPressure::Critical => "critical",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_unknown_when_either_none() {
        assert_eq!(MemoryPressure::derive(None, None), MemoryPressure::Unknown);
        assert_eq!(
            MemoryPressure::derive(Some(1024), None),
            MemoryPressure::Unknown
        );
        assert_eq!(
            MemoryPressure::derive(None, Some(8192)),
            MemoryPressure::Unknown
        );
    }

    #[test]
    fn derive_unknown_when_total_zero() {
        assert_eq!(
            MemoryPressure::derive(Some(0), Some(0)),
            MemoryPressure::Unknown
        );
    }

    #[test]
    fn derive_critical_under_5pct() {
        // 200 / 8192 = 2.44 %
        assert_eq!(
            MemoryPressure::derive(Some(200), Some(8192)),
            MemoryPressure::Critical
        );
        // exactly 5% — boundary goes to Warn, not Critical.
        assert_eq!(
            MemoryPressure::derive(Some(410), Some(8192)),
            MemoryPressure::Warn
        );
    }

    #[test]
    fn derive_warn_5_to_15pct() {
        // 10 %
        assert_eq!(
            MemoryPressure::derive(Some(800), Some(8000)),
            MemoryPressure::Warn
        );
        // exactly 15 % — boundary goes to Normal.
        assert_eq!(
            MemoryPressure::derive(Some(1200), Some(8000)),
            MemoryPressure::Normal
        );
    }

    #[test]
    fn derive_normal_above_15pct() {
        assert_eq!(
            MemoryPressure::derive(Some(4096), Some(8192)),
            MemoryPressure::Normal
        );
    }

    #[test]
    fn worse_of_picks_higher_severity() {
        assert_eq!(
            MemoryPressure::Normal.worse_of(MemoryPressure::Warn),
            MemoryPressure::Warn
        );
        assert_eq!(
            MemoryPressure::Warn.worse_of(MemoryPressure::Critical),
            MemoryPressure::Critical
        );
        assert_eq!(
            MemoryPressure::Critical.worse_of(MemoryPressure::Normal),
            MemoryPressure::Critical
        );
    }

    #[test]
    fn worse_of_prefers_real_reading_over_unknown() {
        assert_eq!(
            MemoryPressure::Unknown.worse_of(MemoryPressure::Normal),
            MemoryPressure::Normal
        );
        assert_eq!(
            MemoryPressure::Normal.worse_of(MemoryPressure::Unknown),
            MemoryPressure::Normal
        );
    }
}
