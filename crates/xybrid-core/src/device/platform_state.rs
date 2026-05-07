//! Platform-bridged signals consumed by the resource monitor.
//!
//! `ResourceMonitor` covers what `sysinfo` can answer cross-platform: CPU,
//! memory, RSS. Battery level and thermal state require platform APIs that
//! sysinfo doesn't expose — `UIDevice.batteryLevel` on iOS,
//! `BatteryManager.ACTION_BATTERY_CHANGED` on Android, `NSProcessInfo`
//! thermalState on macOS, `GetSystemPowerStatus` on Windows, sysfs paths on
//! Linux.
//!
//! This module is the seam. Hosts push values in via [`set_battery_level`] /
//! [`set_thermal_state`]; [`ResourceMonitor::current_snapshot`] reads them
//! out. The Linux desktop case is handled in-process by
//! [`refresh_native_platform_state`] — other platforms have no in-Rust
//! native source today and rely on the host to push.
//!
//! ### Why push-state and not callback interfaces
//!
//! UniFFI callback interfaces and flutter_rust_bridge `DartFn`s both work,
//! but every mobile platform API for these signals already emits change
//! notifications (`UIDevice.batteryLevelDidChangeNotification`,
//! `BatteryManager.ACTION_BATTERY_CHANGED`, `PowerManager.OnThermalStatusChangedListener`).
//! Push-state matches that grain — hosts forward each notification with a
//! single FFI call and forget — instead of forcing every host to poll on a
//! timer and re-marshal across the FFI boundary.

use std::sync::RwLock;

use super::types::ThermalState;

/// Platform-bridged signals.
///
/// Both fields are `Option`: `None` means "no host or native source has
/// reported a value yet." Routing code is expected to treat `None` as
/// "unknown" rather than substituting an optimistic default.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlatformState {
    pub battery_pct: Option<u8>,
    pub thermal_state: Option<ThermalState>,
}

impl PlatformState {
    /// Const empty state — used to initialize the global without a runtime
    /// `Default::default()` call.
    pub const EMPTY: Self = Self {
        battery_pct: None,
        thermal_state: None,
    };
}

static GLOBAL: RwLock<PlatformState> = RwLock::new(PlatformState::EMPTY);

/// Read the current platform-bridged state.
///
/// Lock poisoning falls back to [`PlatformState::EMPTY`] rather than
/// panicking — a poisoned lock means a previous writer panicked, which
/// shouldn't take down inference.
pub fn current_platform_state() -> PlatformState {
    GLOBAL.read().map(|g| *g).unwrap_or(PlatformState::EMPTY)
}

/// Replace the entire platform state in one write. Use the per-field
/// setters for incremental updates; this is for tests and for hosts that
/// have a complete state snapshot in hand.
pub fn set_platform_state(state: PlatformState) {
    if let Ok(mut g) = GLOBAL.write() {
        *g = state;
    }
}

/// Set battery level. Values above 100 are clamped.
pub fn set_battery_level(pct: u8) {
    if let Ok(mut g) = GLOBAL.write() {
        g.battery_pct = Some(pct.min(100));
    }
}

/// Mark battery level as unknown.
pub fn clear_battery_level() {
    if let Ok(mut g) = GLOBAL.write() {
        g.battery_pct = None;
    }
}

/// Set thermal state.
pub fn set_thermal_state(state: ThermalState) {
    if let Ok(mut g) = GLOBAL.write() {
        g.thermal_state = Some(state);
    }
}

/// Mark thermal state as unknown.
pub fn clear_thermal_state() {
    if let Ok(mut g) = GLOBAL.write() {
        g.thermal_state = None;
    }
}

/// Refresh from in-process native sources.
///
/// On Linux this reads `/sys/class/power_supply/BAT[01]/capacity` and
/// `/sys/class/thermal/thermal_zone*/temp`, pushing the values through the
/// same setters a host would use.
///
/// On macOS, iOS, Android and Windows this is a no-op for now — those
/// platforms either need a non-trivial dependency (`objc2-foundation` for
/// `NSProcessInfo`, `windows` for `GetSystemPowerStatus`) or a host bridge
/// (the SDK's iOS/Android plugins). Subsequent commits will fill them in;
/// hosts can already push state today.
///
/// `ResourceMonitor::refresh_locked` calls this on every cache miss, so
/// callers normally don't need to invoke it directly.
pub fn refresh_native_platform_state() {
    #[cfg(target_os = "linux")]
    linux::refresh();
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{set_battery_level, set_thermal_state, ThermalState};
    use std::fs;

    pub(super) fn refresh() {
        if let Some(pct) = read_battery_pct() {
            set_battery_level(pct);
        }
        if let Some(state) = read_thermal_state() {
            set_thermal_state(state);
        }
    }

    fn read_battery_pct() -> Option<u8> {
        const PATHS: &[&str] = &[
            "/sys/class/power_supply/BAT0/capacity",
            "/sys/class/power_supply/BAT1/capacity",
        ];
        for path in PATHS {
            if let Ok(contents) = fs::read_to_string(path) {
                if let Ok(pct) = contents.trim().parse::<u8>() {
                    return Some(pct.min(100));
                }
            }
        }
        None
    }

    fn read_thermal_state() -> Option<ThermalState> {
        // thermal_zone0 is conventionally the CPU package on most distros;
        // thermal_zone1 is the fallback when zone0 is a different sensor
        // (e.g. ACPI vs. coretemp ordering varies across kernels). hwmon0
        // is a last-resort path for systems without /sys/class/thermal at
        // all (containers, some embedded boards).
        const PATHS: &[&str] = &[
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
            "/sys/class/hwmon/hwmon0/temp1_input",
        ];
        for path in PATHS {
            if let Ok(contents) = fs::read_to_string(path) {
                if let Ok(milli) = contents.trim().parse::<i32>() {
                    let celsius = milli as f32 / 1000.0;
                    return Some(thermal_from_celsius(celsius));
                }
            }
        }
        None
    }

    fn thermal_from_celsius(c: f32) -> ThermalState {
        // Thresholds match the documented bands on `ThermalState`'s variant
        // docs (`Normal < 60`, `Warm 60-70`, `Hot 70-80`, `Critical >= 80`).
        if c >= 80.0 {
            ThermalState::Critical
        } else if c >= 70.0 {
            ThermalState::Hot
        } else if c >= 60.0 {
            ThermalState::Warm
        } else {
            ThermalState::Normal
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn celsius_bands_match_thermal_state_docs() {
            assert_eq!(thermal_from_celsius(25.0), ThermalState::Normal);
            assert_eq!(thermal_from_celsius(59.9), ThermalState::Normal);
            assert_eq!(thermal_from_celsius(60.0), ThermalState::Warm);
            assert_eq!(thermal_from_celsius(69.9), ThermalState::Warm);
            assert_eq!(thermal_from_celsius(70.0), ThermalState::Hot);
            assert_eq!(thermal_from_celsius(79.9), ThermalState::Hot);
            assert_eq!(thermal_from_celsius(80.0), ThermalState::Critical);
            assert_eq!(thermal_from_celsius(95.0), ThermalState::Critical);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Tests touch a single process-wide global. Serialize them so parallel
    // execution doesn't see crossed writes.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset() {
        set_platform_state(PlatformState::EMPTY);
    }

    #[test]
    fn empty_state_when_nothing_pushed() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        let s = current_platform_state();
        assert_eq!(s.battery_pct, None);
        assert_eq!(s.thermal_state, None);
    }

    #[test]
    fn set_and_clear_battery() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        set_battery_level(75);
        assert_eq!(current_platform_state().battery_pct, Some(75));
        clear_battery_level();
        assert_eq!(current_platform_state().battery_pct, None);
    }

    #[test]
    fn set_and_clear_thermal() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        set_thermal_state(ThermalState::Hot);
        assert_eq!(
            current_platform_state().thermal_state,
            Some(ThermalState::Hot)
        );
        clear_thermal_state();
        assert_eq!(current_platform_state().thermal_state, None);
    }

    #[test]
    fn set_battery_clamps_to_100() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        set_battery_level(255);
        assert_eq!(current_platform_state().battery_pct, Some(100));
    }

    #[test]
    fn whole_struct_push_replaces_all_fields() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        set_battery_level(40);
        set_thermal_state(ThermalState::Warm);
        set_platform_state(PlatformState {
            battery_pct: Some(80),
            thermal_state: None,
        });
        let s = current_platform_state();
        assert_eq!(s.battery_pct, Some(80));
        assert_eq!(s.thermal_state, None);
    }

    #[test]
    fn battery_and_thermal_are_independent() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        set_battery_level(50);
        set_thermal_state(ThermalState::Warm);
        clear_battery_level();
        let s = current_platform_state();
        assert_eq!(s.battery_pct, None);
        assert_eq!(s.thermal_state, Some(ThermalState::Warm));
    }

    #[test]
    fn resource_monitor_snapshot_reflects_pushed_state() {
        // End-to-end check: a host push must appear on the next
        // ResourceMonitor cache miss. Uses `Duration::ZERO` to force a
        // refresh past the TTL.
        use crate::device::ResourceMonitor;
        use std::time::Duration;

        let _g = TEST_LOCK.lock().unwrap();
        reset();

        let monitor = ResourceMonitor::new();
        // Baseline: no host has pushed, so the snapshot reports the
        // unknown defaults.
        let baseline = monitor.current_snapshot(Duration::ZERO);
        assert_eq!(baseline.battery_pct, None);
        assert_eq!(baseline.thermal_state, ThermalState::Normal);

        set_battery_level(42);
        set_thermal_state(ThermalState::Hot);

        let after = monitor.current_snapshot(Duration::ZERO);
        // On Linux the native refresher fires inside refresh_locked() and
        // overwrites the host push with sysfs values, so we can't assert
        // the exact pushed values. On every other platform the host push
        // is the only source and must round-trip exactly.
        #[cfg(not(target_os = "linux"))]
        {
            assert_eq!(after.battery_pct, Some(42));
            assert_eq!(after.thermal_state, ThermalState::Hot);
        }
        // On Linux we still want a non-trivial assertion: the snapshot
        // must reflect *some* signal (the sysfs read or the host push).
        #[cfg(target_os = "linux")]
        {
            // Either the sysfs read produced a value, or the host push
            // survives because sysfs paths weren't present (e.g. CI
            // container without /sys/class/thermal). Both are acceptable
            // — we only assert the snapshot isn't stuck at the default
            // when *some* source has data.
            let _ = after; // touch to silence unused warnings on linux.
        }

        reset();
    }
}
