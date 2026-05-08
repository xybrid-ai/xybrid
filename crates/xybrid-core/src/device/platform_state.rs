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
/// - **Linux**: reads `/sys/class/power_supply/BAT[01]/capacity` and
///   `/sys/class/thermal/thermal_zone*/temp`.
/// - **macOS**: reads `NSProcessInfo.thermalState` (no entitlement
///   required, fast Foundation call) and queries IOKit
///   `IOPSCopyPowerSourcesInfo` for battery charge. Both calls are
///   in-process and thread-safe per Apple's docs — no fork/exec, no
///   COM, safe on the runtime thread that
///   `Orchestrator::execute_stage_async` lands on.
/// - **Windows**: reads `GetSystemPowerStatus` (Win32, in-process)
///   for battery on the cache-miss path. Thermal is sourced from
///   `MSAcpi_ThermalZoneTemperature` over WMI (`ROOT\WMI`); COM
///   init and `IWbemServices::ExecQuery` would freeze the runtime
///   thread that `Orchestrator::execute_stage_async` lands on, so
///   the WMI loop runs on a dedicated background thread that pushes
///   results through the public setters.
/// - **iOS, Android**: no-op for now. Hosts push state via the public
///   setters from platform observers; native in-process pollers come
///   later.
///
/// All in-process refreshers go through the same setters a host would
/// use, so behaviour is uniform whether data comes from sysfs, IOKit, or
/// a UniFFI host.
///
/// `ResourceMonitor::refresh_locked` calls this on every cache miss, so
/// callers normally don't need to invoke it directly.
pub fn refresh_native_platform_state() {
    #[cfg(target_os = "linux")]
    linux::refresh();
    #[cfg(target_os = "macos")]
    macos::refresh();
    #[cfg(target_os = "windows")]
    windows::refresh();
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

#[cfg(target_os = "macos")]
mod macos {
    //! macOS native pollers.
    //!
    //! Thermal: `NSProcessInfo.thermalState` — direct Foundation call,
    //! no entitlement, microsecond-class.
    //!
    //! Battery: IOKit `IOPSCopyPowerSourcesInfo` →
    //! `IOPSCopyPowerSourcesList` → `IOPSGetPowerSourceDescription`,
    //! reading `kIOPSCurrentCapacityKey` / `kIOPSMaxCapacityKey` from
    //! the per-source dictionary. All three calls are in-process and
    //! documented thread-safe; no fork+exec (which is what a `pmset
    //! -g batt` shellout would cost on every cache miss).
    //!
    //! Both pollers are safe on the cache-miss hot path that
    //! `Orchestrator::execute_stage_async` invokes via
    //! `ResourceMonitor::current_snapshot`. Devices without a battery
    //! (Mac mini, Mac Studio, etc.) return an empty source list and
    //! we report `None`.

    use core::ffi::{c_void, CStr};

    use super::{set_battery_level, set_thermal_state, ThermalState};
    use objc2_core_foundation::{CFDictionary, CFNumber, CFString, CFType};
    use objc2_foundation::NSProcessInfo;
    use objc2_io_kit::{
        kIOPSCurrentCapacityKey, kIOPSMaxCapacityKey, IOPSCopyPowerSourcesInfo,
        IOPSCopyPowerSourcesList, IOPSGetPowerSourceDescription,
    };

    pub(super) fn refresh() {
        set_thermal_state(read_thermal_state());
        if let Some(pct) = read_battery_pct() {
            set_battery_level(pct);
        }
    }

    fn read_thermal_state() -> ThermalState {
        // `NSProcessInfo.thermalState` returns one of four discrete states
        // matching the documented API levels (Nominal, Fair, Serious,
        // Critical). The Foundation method is marked `unsafe` because
        // it's an Objective-C method invocation, but it is documented
        // thread-safe and never null on every macOS we support — there
        // is no precondition the caller can violate.
        let info = NSProcessInfo::processInfo();
        // SAFETY: see comment above — `thermalState` is thread-safe and
        // has no caller-side preconditions; the cast to i64 is widening
        // from NSInteger and lossless.
        let raw = unsafe { info.thermalState() }.0 as i64;
        thermal_from_nsprocessinfo(raw)
    }

    /// Map the raw `NSProcessInfoThermalState` integer to our
    /// `ThermalState`. Documented values:
    /// - 0 = Nominal   → Normal
    /// - 1 = Fair      → Warm
    /// - 2 = Serious   → Hot
    /// - 3 = Critical  → Critical
    ///
    /// Unexpected values fall back to Normal — Foundation has only ever
    /// shipped these four, but a future addition shouldn't crash the
    /// inference path.
    fn thermal_from_nsprocessinfo(raw: i64) -> ThermalState {
        match raw {
            0 => ThermalState::Normal,
            1 => ThermalState::Warm,
            2 => ThermalState::Hot,
            3 => ThermalState::Critical,
            _ => ThermalState::Normal,
        }
    }

    fn read_battery_pct() -> Option<u8> {
        // `IOPSCopyPowerSourcesInfo` is the documented entry point; per
        // Apple's docs it does no I/O of its own, just snapshots a
        // pre-aggregated registry blob. Returns `None` on systems where
        // power-source info is unavailable (sandboxed contexts, edge cases).
        let blob = IOPSCopyPowerSourcesInfo()?;

        // SAFETY: `blob` was just produced by IOPSCopyPowerSourcesInfo,
        // which is the documented input contract for IOPSCopyPowerSourcesList.
        let sources = unsafe { IOPSCopyPowerSourcesList(Some(&blob)) }?;

        let count = sources.count();
        if count == 0 {
            // No power sources — Mac mini, Mac Studio, Mac Pro, etc. The
            // routing engine treats `None` as "battery unknown / not
            // applicable" and falls back to other evidence.
            return None;
        }

        for i in 0..count {
            // SAFETY: `i` is in 0..count, and `sources` is the CFArray
            // returned by IOPSCopyPowerSourcesList — its elements are
            // the IOKit-owned power-source handles documented to be
            // valid for the lifetime of the array.
            let raw = unsafe { sources.value_at_index(i) };
            if raw.is_null() {
                continue;
            }
            // SAFETY: `raw` is a non-null pointer to a CFTypeRef owned
            // by `sources`; the borrow is bounded by `sources`'s scope.
            let ps: &CFType = unsafe { &*(raw as *const CFType) };

            // SAFETY: `blob` and `ps` came from the matching pair of
            // IOPSCopyPowerSourcesInfo / IOPSCopyPowerSourcesList calls
            // above — the documented preconditions for
            // IOPSGetPowerSourceDescription.
            let Some(desc) = (unsafe { IOPSGetPowerSourceDescription(Some(&blob), Some(ps)) })
            else {
                continue;
            };

            // Some power sources (UPS, keyboard battery, etc.) may omit
            // capacity keys. Skip rather than fail — the next source
            // might be the laptop's internal battery.
            let Some(current) = lookup_int(&desc, kIOPSCurrentCapacityKey) else {
                continue;
            };
            let Some(max) = lookup_int(&desc, kIOPSMaxCapacityKey) else {
                continue;
            };
            if let Some(pct) = compute_pct(current, max) {
                return Some(pct);
            }
        }
        None
    }

    fn lookup_int(dict: &CFDictionary, key_cstr: &CStr) -> Option<i64> {
        // IOKit defines its dictionary keys as C strings (e.g.
        // `kIOPSCurrentCapacityKey == "Current Capacity"`). The
        // dictionary itself stores CFString keys, so we wrap before
        // lookup. UTF-8 conversion never fails for these constants but
        // we propagate the Result for hygiene.
        let key_str = key_cstr.to_str().ok()?;
        let cf_key = CFString::from_str(key_str);
        let key_ptr: *const c_void = (&*cf_key as *const CFString).cast();

        // SAFETY: `key_ptr` points to a live CFString (held by
        // `cf_key`), and `dict` is a power-source description
        // dictionary with CFString keys — equality uses CFEqual.
        let raw = unsafe { dict.value(key_ptr) };
        if raw.is_null() {
            return None;
        }

        // SAFETY: `raw` is a non-null CFTypeRef value owned by `desc`
        // (which the caller holds alive); converting to `&CFType` for
        // a runtime type-check is the documented pattern.
        let cf: &CFType = unsafe { &*(raw as *const CFType) };
        let num = cf.downcast_ref::<CFNumber>()?;
        num.as_i64()
    }

    /// Map IOKit `(current, max)` capacities to a 0..=100 charge percent.
    ///
    /// Returns `None` if `max <= 0` (would divide by zero, indicates a
    /// sensor glitch or uninitialized source) or `current < 0`.
    /// Otherwise clamps to 0..=100 — some power sources briefly report
    /// `current > max` while recalibrating.
    fn compute_pct(current: i64, max: i64) -> Option<u8> {
        if max <= 0 || current < 0 {
            return None;
        }
        // saturating_mul guards against pathological values from a
        // misbehaving driver — the division by `max > 0` then yields
        // a sane, in-range number after the clamp.
        let raw = current.saturating_mul(100) / max;
        Some(raw.clamp(0, 100) as u8)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn thermal_mapping_matches_apple_constants() {
            assert_eq!(thermal_from_nsprocessinfo(0), ThermalState::Normal);
            assert_eq!(thermal_from_nsprocessinfo(1), ThermalState::Warm);
            assert_eq!(thermal_from_nsprocessinfo(2), ThermalState::Hot);
            assert_eq!(thermal_from_nsprocessinfo(3), ThermalState::Critical);
        }

        #[test]
        fn thermal_mapping_unknown_falls_back_to_normal() {
            // Apple has only ever shipped 0..=3 for NSProcessInfoThermalState.
            // A future addition shouldn't crash inference; Normal is the
            // safest default (won't trigger should_throttle).
            assert_eq!(thermal_from_nsprocessinfo(99), ThermalState::Normal);
            assert_eq!(thermal_from_nsprocessinfo(-1), ThermalState::Normal);
        }

        #[test]
        fn read_thermal_state_does_not_panic() {
            // Smoke test: NSProcessInfo always exists and thermalState
            // always returns a valid value on macOS. Just verify the FFI
            // call returns something well-formed.
            let state = read_thermal_state();
            // All four variants are valid; we just want to confirm the
            // call returned without panicking.
            let _ = state;
        }

        #[test]
        fn compute_pct_handles_normal_values() {
            assert_eq!(compute_pct(0, 100), Some(0));
            assert_eq!(compute_pct(50, 100), Some(50));
            assert_eq!(compute_pct(100, 100), Some(100));
            assert_eq!(compute_pct(75, 100), Some(75));
            // Real IOKit values (laptop battery, mAh-scale).
            assert_eq!(compute_pct(4_200, 5_000), Some(84));
        }

        #[test]
        fn compute_pct_zero_or_negative_max_is_none() {
            assert_eq!(compute_pct(50, 0), None);
            assert_eq!(compute_pct(50, -100), None);
        }

        #[test]
        fn compute_pct_negative_current_is_none() {
            // A negative `current` is a sensor glitch — don't fold that
            // through to should_throttle as an artificial 0%.
            assert_eq!(compute_pct(-1, 100), None);
        }

        #[test]
        fn compute_pct_clamps_above_max() {
            // Power sources can briefly report current > max during
            // calibration. Clamp rather than reject.
            assert_eq!(compute_pct(105, 100), Some(100));
            assert_eq!(compute_pct(200, 100), Some(100));
        }

        #[test]
        fn read_battery_pct_returns_none_or_valid_percent() {
            // Smoke test: don't assert exact values — laptops, desktops,
            // sandboxed test runners all behave differently. Just verify
            // the IOKit path returns a well-formed Option<u8>.
            if let Some(pct) = read_battery_pct() {
                assert!(pct <= 100, "battery percent out of range: {}", pct);
            }
        }
    }
}

#[cfg(target_os = "windows")]
mod windows {
    //! Windows native pollers.
    //!
    //! Battery via `GetSystemPowerStatus` — a single Win32 syscall, no
    //! fork, no COM, no WMI. Runs synchronously on the cache-miss path.
    //! Returns `SYSTEM_POWER_STATUS` whose `BatteryLifePercent` field
    //! carries the charge in 0..=100, with `BATTERY_PERCENTAGE_UNKNOWN`
    //! (255) signalling "no battery / unknown" on desktops.
    //!
    //! Thermal via WMI's `MSAcpi_ThermalZoneTemperature` (`ROOT\WMI`).
    //! Each query path — `CoInitializeEx`, `IWbemLocator::ConnectServer`,
    //! `IWbemServices::ExecQuery`, `IEnumWbemClassObject::Next` — costs
    //! milliseconds and would block whatever Tokio runtime thread the
    //! orchestrator's cache-miss happens to land on. Instead a dedicated
    //! background thread polls every [`THERMAL_POLL_INTERVAL`] and pushes
    //! results through [`super::set_thermal_state`]. The thread is
    //! spawned lazily on first refresh and lives for the process lifetime;
    //! transient WMI errors are logged and the loop continues.
    //!
    //! Devices without an ACPI thermal zone (some VMs, headless servers)
    //! return zero rows — we leave thermal state unset rather than
    //! lying with `Normal`.

    use std::sync::OnceLock;
    use std::thread;
    use std::time::Duration;

    use windows::core::{BSTR, PCWSTR};
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
    };
    use windows::Win32::System::Variant::{VariantClear, VARIANT, VT_I4, VT_UI4};
    use windows::Win32::System::Wmi::{
        IEnumWbemClassObject, IWbemClassObject, IWbemLocator, IWbemServices, WbemLocator,
        WBEM_FLAG_FORWARD_ONLY, WBEM_FLAG_RETURN_IMMEDIATELY, WBEM_INFINITE,
    };
    use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};

    use super::{set_battery_level, set_thermal_state, ThermalState};

    /// `SYSTEM_POWER_STATUS::BatteryLifePercent` sentinel for "unknown
    /// or no battery". Documented in the Win32 SDK; reproduced here so
    /// we don't depend on a constant that windows-sys may or may not
    /// re-export across versions.
    const BATTERY_PERCENTAGE_UNKNOWN: u8 = 255;

    /// How often the background thread re-queries WMI. Each query is
    /// a cross-apartment COM round-trip (single-digit milliseconds);
    /// the cache TTL inside `ResourceMonitor` is 500 ms so a few-second
    /// cadence keeps the thermal signal fresh enough for routing
    /// decisions without burning CPU on a tight loop. Tuned by hand
    /// against the same bands the Linux sysfs path uses.
    const THERMAL_POLL_INTERVAL: Duration = Duration::from_secs(3);

    /// VARENUM raw value for `VT_I4`. Compared against the variant tag
    /// returned by `IWbemClassObject::Get` for `CIM_UINT32` properties,
    /// which WMI marshals as a signed 32-bit integer.
    const VT_I4_RAW: u16 = VT_I4.0;
    /// VARENUM raw value for `VT_UI4`. Some WMI providers report
    /// `CIM_UINT32` values directly as `VT_UI4` instead of `VT_I4`;
    /// accept both rather than dropping the reading.
    const VT_UI4_RAW: u16 = VT_UI4.0;

    pub(super) fn refresh() {
        if let Some(pct) = read_battery_pct() {
            set_battery_level(pct);
        }
        ensure_thermal_poller();
    }

    fn read_battery_pct() -> Option<u8> {
        // SAFETY: GetSystemPowerStatus writes a SYSTEM_POWER_STATUS into
        // the provided pointer and returns BOOL. Zero-initializing the
        // struct first ensures every field is valid even on the
        // never-observed-but-documented case where the OS leaves a field
        // untouched. The call has no caller-side preconditions and is
        // thread-safe per Microsoft docs.
        let mut status: SYSTEM_POWER_STATUS = unsafe { std::mem::zeroed() };
        let ok = unsafe { GetSystemPowerStatus(&mut status) };
        if ok == 0 {
            return None;
        }
        battery_pct_from_status(status.BatteryLifePercent)
    }

    /// Map raw `BatteryLifePercent` to our 0..=100 representation.
    /// 255 (BATTERY_PERCENTAGE_UNKNOWN) and any out-of-range value
    /// surface as `None`; everything else clamps into u8.
    fn battery_pct_from_status(raw: u8) -> Option<u8> {
        if raw == BATTERY_PERCENTAGE_UNKNOWN || raw > 100 {
            None
        } else {
            Some(raw)
        }
    }

    /// Map deci-Kelvin (the unit `MSAcpi_ThermalZoneTemperature` reports)
    /// to a [`ThermalState`] using the same bands the Linux sysfs path
    /// uses. The conversion is `(dK / 10) - 273.15`.
    fn thermal_from_dk(deci_kelvin: u32) -> ThermalState {
        let celsius = (deci_kelvin as f32 / 10.0) - 273.15;
        if celsius >= 80.0 {
            ThermalState::Critical
        } else if celsius >= 70.0 {
            ThermalState::Hot
        } else if celsius >= 60.0 {
            ThermalState::Warm
        } else {
            ThermalState::Normal
        }
    }

    /// Spawn the WMI thermal poller exactly once per process. Subsequent
    /// calls are O(1) and do not touch COM. Spawn failure is logged and
    /// the routing engine continues with `thermal_state = None` — a
    /// degraded but non-fatal mode.
    fn ensure_thermal_poller() {
        static POLLER: OnceLock<()> = OnceLock::new();
        POLLER.get_or_init(|| {
            let spawn = thread::Builder::new()
                .name("xybrid-wmi-thermal".into())
                .spawn(thermal_poller_main);
            if let Err(err) = spawn {
                log::warn!("xybrid-wmi-thermal: failed to spawn poller thread: {err}");
            }
        });
    }

    fn thermal_poller_main() {
        // SAFETY: `CoInitializeEx` runs exactly once on this dedicated
        // thread, before any other COM call. `COINIT_MULTITHREADED`
        // matches the WMI client model — we never marshal proxies into
        // another apartment. `CoUninitialize` is intentionally not
        // called: the thread runs for the process lifetime and the
        // OS reclaims COM state at exit.
        let init = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if init.is_err() {
            // S_FALSE here would mean COM was already initialised on
            // this thread, which can't happen for a thread we just
            // spawned — anything other than S_OK is a real failure.
            log::warn!(
                "xybrid-wmi-thermal: CoInitializeEx failed ({init:?}), thermal poller exiting"
            );
            return;
        }

        loop {
            match poll_once() {
                Ok(Some(state)) => set_thermal_state(state),
                Ok(None) => {}
                Err(err) => {
                    log::debug!("xybrid-wmi-thermal: query failed (continuing): {err:?}");
                }
            }
            thread::sleep(THERMAL_POLL_INTERVAL);
        }
    }

    fn poll_once() -> windows::core::Result<Option<ThermalState>> {
        // SAFETY: `CoCreateInstance` is the documented entry point for
        // creating an `IWbemLocator`; the windows crate enforces that
        // `T::IID` matches the requested class.
        let locator: IWbemLocator =
            unsafe { CoCreateInstance(&WbemLocator, None, CLSCTX_INPROC_SERVER)? };

        // SAFETY: `ConnectServer` accepts BSTR arguments — empty BSTRs
        // are the documented way to request defaults for user, password,
        // locale, and authority on a local connection. `ROOT\WMI` is the
        // namespace where `MSAcpi_ThermalZoneTemperature` lives.
        let services: IWbemServices = unsafe {
            locator.ConnectServer(
                &BSTR::from("ROOT\\WMI"),
                &BSTR::new(),
                &BSTR::new(),
                &BSTR::new(),
                0,
                &BSTR::new(),
                None,
            )?
        };

        // SAFETY: `ExecQuery` is the canonical fast-forward enumeration
        // entry point; `WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY`
        // is the documented combination for read-only WQL queries.
        let enumerator: IEnumWbemClassObject = unsafe {
            services.ExecQuery(
                &BSTR::from("WQL"),
                &BSTR::from("SELECT CurrentTemperature FROM MSAcpi_ThermalZoneTemperature"),
                WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
                None,
            )?
        };

        let mut warmest_dk: Option<u32> = None;
        loop {
            let mut row: [Option<IWbemClassObject>; 1] = [None];
            let mut returned: u32 = 0;
            // SAFETY: `Next` writes up to `row.len()` objects and stores
            // the actual count into `returned`; both pointers reference
            // stack storage that outlives the call.
            let _hr = unsafe { enumerator.Next(WBEM_INFINITE, &mut row, &mut returned) };
            if returned == 0 {
                break;
            }
            if let Some(obj) = &row[0] {
                if let Some(dk) = read_current_temperature(obj) {
                    warmest_dk = Some(warmest_dk.map_or(dk, |w| w.max(dk)));
                }
            }
        }

        Ok(warmest_dk.map(thermal_from_dk))
    }

    /// Read the `CurrentTemperature` property from a single WMI row.
    /// Returns `None` if the property is missing, has an unexpected
    /// VARIANT type, or `Get` itself fails — any of which we treat as
    /// "skip this zone" rather than failing the whole poll.
    fn read_current_temperature(obj: &IWbemClassObject) -> Option<u32> {
        let name: [u16; 19] = [
            b'C' as u16,
            b'u' as u16,
            b'r' as u16,
            b'r' as u16,
            b'e' as u16,
            b'n' as u16,
            b't' as u16,
            b'T' as u16,
            b'e' as u16,
            b'm' as u16,
            b'p' as u16,
            b'e' as u16,
            b'r' as u16,
            b'a' as u16,
            b't' as u16,
            b'u' as u16,
            b'r' as u16,
            b'e' as u16,
            0,
        ];
        let mut value = VARIANT::default();
        // SAFETY: `name` is a UTF-16 null-terminated string with stable
        // backing storage for the duration of the call; `value` is a
        // freshly-zeroed VARIANT. `ptype`/`plflavor` are optional and
        // we don't need either.
        let res = unsafe { obj.Get(PCWSTR(name.as_ptr()), 0, &mut value, None, None) };
        if res.is_err() {
            return None;
        }
        // SAFETY: VARIANT layout is `vt` followed by a union of value
        // arms; we read the union arm matching the tag we just inspected.
        // CIM_UINT32 is documented to marshal as VT_I4 but some providers
        // return VT_UI4 — accept both.
        let extracted = unsafe {
            let inner = &value.Anonymous.Anonymous;
            match inner.vt.0 {
                VT_I4_RAW => Some(inner.Anonymous.lVal as u32),
                VT_UI4_RAW => Some(inner.Anonymous.ulVal),
                _ => None,
            }
        };
        // SAFETY: `value` is a VARIANT we own; `VariantClear` releases
        // any allocations the marshaller attached (BSTRs, IUnknowns).
        // Ignoring the result mirrors how the windows-rs samples
        // handle the cleanup path.
        let _ = unsafe { VariantClear(&mut value) };
        extracted
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn raw_battery_in_range_round_trips() {
            assert_eq!(battery_pct_from_status(0), Some(0));
            assert_eq!(battery_pct_from_status(50), Some(50));
            assert_eq!(battery_pct_from_status(100), Some(100));
        }

        #[test]
        fn unknown_sentinel_maps_to_none() {
            assert_eq!(battery_pct_from_status(BATTERY_PERCENTAGE_UNKNOWN), None);
        }

        #[test]
        fn out_of_range_maps_to_none() {
            // 101..=254 is not a documented value but Microsoft's
            // BatteryLifePercent field is u8 so the OS could theoretically
            // hand us anything. Treating these as "unknown" rather than
            // clamping prevents lying to should_throttle().
            assert_eq!(battery_pct_from_status(101), None);
            assert_eq!(battery_pct_from_status(200), None);
            assert_eq!(battery_pct_from_status(254), None);
        }

        #[test]
        fn read_battery_pct_does_not_panic() {
            // Smoke test: GetSystemPowerStatus is always callable on
            // every supported Windows version. We can't assert the
            // exact value (depends on host hardware) but we can verify
            // it returns a well-formed Option<u8>.
            if let Some(pct) = read_battery_pct() {
                assert!(pct <= 100, "battery percent out of range: {}", pct);
            }
        }

        #[test]
        fn deci_kelvin_bands_match_thermal_state_docs() {
            // Conversion: dK / 10 - 273.15 → °C. The band cutoffs are
            // 60/70/80 °C, which fall between integer dK values
            // (60.0 °C = 3331.5 dK), so the closest integer dK on
            // either side is the strongest boundary check available.
            assert_eq!(thermal_from_dk(2731), ThermalState::Normal); // 0.0 °C
            assert_eq!(thermal_from_dk(3231), ThermalState::Normal); // 50.0 °C
            assert_eq!(thermal_from_dk(3331), ThermalState::Normal); // 59.95 °C
            assert_eq!(thermal_from_dk(3332), ThermalState::Warm); // 60.05 °C
            assert_eq!(thermal_from_dk(3431), ThermalState::Warm); // 69.95 °C
            assert_eq!(thermal_from_dk(3432), ThermalState::Hot); // 70.05 °C
            assert_eq!(thermal_from_dk(3531), ThermalState::Hot); // 79.95 °C
            assert_eq!(thermal_from_dk(3532), ThermalState::Critical); // 80.05 °C
            assert_eq!(thermal_from_dk(3731), ThermalState::Critical); // 100.0 °C
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
        // End-to-end check: a host push appears on the next
        // ResourceMonitor cache miss. Uses `Duration::ZERO` to force a
        // refresh past the TTL.
        //
        // On platforms with an active native poller (Linux sysfs, macOS
        // NSProcessInfo + pmset), `refresh_locked` will overwrite host
        // pushes with the native readings, so the exact-value
        // assertions only run where no native source competes.
        use crate::device::ResourceMonitor;
        use std::time::Duration;

        let _g = TEST_LOCK.lock().unwrap();
        reset();

        let monitor = ResourceMonitor::new();

        set_battery_level(42);
        set_thermal_state(ThermalState::Hot);

        let after = monitor.current_snapshot(Duration::ZERO);

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            // No in-process native poller — the host push is the only
            // source and must round-trip exactly.
            assert_eq!(after.battery_pct, Some(42));
            assert_eq!(after.thermal_state, ThermalState::Hot);
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            // Native poller may have overwritten the host push with
            // real sysfs / Foundation readings. We can't assert exact
            // values without mocking the platform; assert the overlay
            // path executed without crashing and produced well-formed
            // values.
            assert!(
                after.battery_pct.map(|p| p <= 100).unwrap_or(true),
                "battery_pct out of range: {:?}",
                after.battery_pct
            );
            // thermal_state is an enum, any variant is well-formed; the
            // bind silences unused-variable warnings while still
            // exercising the overlay.
            let _ = after.thermal_state;
        }

        reset();
    }
}
