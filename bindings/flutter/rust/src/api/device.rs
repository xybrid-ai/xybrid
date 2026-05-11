//! Device platform-state push API for Flutter hosts.
//!
//! Mobile telemetry APIs (`UIDevice.batteryLevelDidChangeNotification` on
//! iOS, `BatteryManager.ACTION_BATTERY_CHANGED` on Android,
//! `PowerManager.OnThermalStatusChangedListener` on Android) are
//! notification-based and live in the host runtime — there is no clean
//! in-Rust path on those platforms. Flutter hosts subscribe via the
//! `battery_plus` / `flutter_thermal_status` packages (or platform
//! channels) and forward each value through these FFI calls. The Rust
//! side just stores into the same `RwLock<PlatformState>` the desktop
//! pollers feed, so routing decisions are uniform across platforms.
//!
//! Push direction is one-way (host → Rust) by design: a callback or
//! `DartFn` shape would re-enter Rust on every change, which is much
//! more surface for marginal benefit when the host already gets these
//! notifications routinely.
//!
//! The host can also *read* the routing-engine's current view of device
//! state via [`XybridDevice::current_snapshot`]. This is the one-way
//! mirror of the push surface: the host sees exactly what the routing
//! engine sees, including signals that came from the in-Rust platform
//! pollers (Linux sysfs, macOS / iOS NSProcessInfo) and from sysinfo
//! (CPU + memory). Useful for diagnostics screens that need to verify
//! signal coverage on a given platform without round-tripping through
//! a telemetry event.

use std::time::Duration;

use flutter_rust_bridge::frb;

use super::FLUTTER_BINDING;

/// Thermal pressure state forwarded by the host.
///
/// Maps directly to [`xybrid_sdk::ThermalState`] — mirrored as an FRB
/// enum so Dart consumers see `FfiThermalState.normal | warm | hot |
/// critical`. Variants carry the same Celsius bands as the desktop
/// pollers so host code can quantize the OS signal consistently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiThermalState {
    /// Normal operating temperature (< 60 °C).
    Normal,
    /// Warm — first throttling tier (60–70 °C).
    Warm,
    /// Hot — performance reduced (70–80 °C).
    Hot,
    /// Critical — heavy operations should pause (>= 80 °C).
    Critical,
}

impl From<FfiThermalState> for xybrid_sdk::ThermalState {
    fn from(value: FfiThermalState) -> Self {
        match value {
            FfiThermalState::Normal => xybrid_sdk::ThermalState::Normal,
            FfiThermalState::Warm => xybrid_sdk::ThermalState::Warm,
            FfiThermalState::Hot => xybrid_sdk::ThermalState::Hot,
            FfiThermalState::Critical => xybrid_sdk::ThermalState::Critical,
        }
    }
}

impl From<xybrid_sdk::ThermalState> for FfiThermalState {
    fn from(value: xybrid_sdk::ThermalState) -> Self {
        match value {
            xybrid_sdk::ThermalState::Normal => FfiThermalState::Normal,
            xybrid_sdk::ThermalState::Warm => FfiThermalState::Warm,
            xybrid_sdk::ThermalState::Hot => FfiThermalState::Hot,
            xybrid_sdk::ThermalState::Critical => FfiThermalState::Critical,
        }
    }
}

/// Derived memory-pressure classification mirrored onto the FFI surface.
///
/// Maps directly to [`xybrid_sdk::MemoryPressure`]. `Unknown` means the
/// snapshot couldn't be computed (sysinfo refused to answer, or the
/// host hasn't pushed an iOS / Android memory-warning observer yet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiMemoryPressure {
    Unknown,
    Normal,
    Warn,
    Critical,
}

impl From<xybrid_sdk::MemoryPressure> for FfiMemoryPressure {
    fn from(value: xybrid_sdk::MemoryPressure) -> Self {
        match value {
            xybrid_sdk::MemoryPressure::Unknown => FfiMemoryPressure::Unknown,
            xybrid_sdk::MemoryPressure::Normal => FfiMemoryPressure::Normal,
            xybrid_sdk::MemoryPressure::Warn => FfiMemoryPressure::Warn,
            xybrid_sdk::MemoryPressure::Critical => FfiMemoryPressure::Critical,
        }
    }
}

/// Snapshot of routing-engine signals as the runtime currently sees them.
///
/// Field-for-field mirror of [`xybrid_sdk::ResourceSnapshot`]. `Option`
/// fields are `None` when the underlying sensor isn't available on the
/// running platform — a `None` here is what the routing engine reads as
/// "no signal," which downstream gates treat as "do not penalize."
#[derive(Clone, Debug)]
pub struct FfiResourceSnapshot {
    pub cpu_pct: Option<f32>,
    pub process_rss_mb: Option<u32>,
    pub available_mem_mb: Option<u32>,
    pub total_mem_mb: Option<u32>,
    pub memory_pressure: FfiMemoryPressure,
    pub thermal_state: FfiThermalState,
    pub battery_pct: Option<u8>,
    pub captured_at_ms: u64,
}

impl From<xybrid_sdk::ResourceSnapshot> for FfiResourceSnapshot {
    fn from(value: xybrid_sdk::ResourceSnapshot) -> Self {
        Self {
            cpu_pct: value.cpu_pct,
            process_rss_mb: value.process_rss_mb,
            available_mem_mb: value.available_mem_mb,
            total_mem_mb: value.total_mem_mb,
            memory_pressure: value.memory_pressure.into(),
            thermal_state: value.thermal_state.into(),
            battery_pct: value.battery_pct,
            captured_at_ms: value.captured_at_ms,
        }
    }
}

/// Marker type that namespaces the push-state functions on the Dart
/// side. Mirrors the [`super::sdk_client::XybridSdkClient`] shape — no
/// instance is ever constructed; Dart sees `XybridDevice.setBatteryLevel(...)`.
#[frb(opaque)]
pub struct XybridDevice;

impl XybridDevice {
    /// Forward a battery charge percent (0..=100) from the host.
    ///
    /// Values above 100 are clamped by the underlying setter — pass
    /// through whatever the OS observer reports without rounding host
    /// side, so the SDK has the freshest possible signal.
    #[frb(sync)]
    pub fn set_battery_level(percent: u8) {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::set_battery_level(percent);
    }

    /// Mark the battery level as unknown.
    ///
    /// Hosts call this on observer teardown or when the OS reports an
    /// unknown / unavailable charge (desktop docks without battery
    /// sensors). The routing engine treats `None` as "no signal" rather
    /// than substituting an optimistic default.
    #[frb(sync)]
    pub fn clear_battery_level() {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::clear_battery_level();
    }

    /// Forward a thermal pressure reading from the host.
    #[frb(sync)]
    pub fn set_thermal_state(state: FfiThermalState) {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::set_thermal_state(state.into());
    }

    /// Mark the thermal state as unknown.
    #[frb(sync)]
    pub fn clear_thermal_state() {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::clear_thermal_state();
    }

    /// Read the routing-engine's current device snapshot.
    ///
    /// Returns whatever the global [`xybrid_sdk::ResourceMonitor`] last
    /// observed — battery + thermal from the platform pollers / host
    /// pushes, CPU + memory from sysinfo. Force-refreshes on every
    /// call (passes [`Duration::ZERO`]) so a diagnostics surface that
    /// polls at ~1 Hz sees fresh data each tick. The refresh cost is
    /// bounded; the contract is documented in `resource-telemetry.md`
    /// at `< 1 ms` on a warm monitor.
    ///
    /// Intended for app-side diagnostics views ("what does the routing
    /// engine see on this device right now?"). Production code should
    /// not poll this — the engine reads it internally on each routing
    /// decision.
    #[frb(sync)]
    pub fn current_snapshot() -> FfiResourceSnapshot {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::ResourceMonitor::global()
            .current_snapshot(Duration::ZERO)
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure conversion tests for FfiThermalState. The push setters
    // themselves write into a process-global RwLock that other tests
    // (and other crates' integration tests) also touch — covering the
    // mapping at the conversion layer keeps these tests deterministic
    // regardless of test ordering.
    #[test]
    fn thermal_state_round_trips_through_sdk_type() {
        for variant in [
            FfiThermalState::Normal,
            FfiThermalState::Warm,
            FfiThermalState::Hot,
            FfiThermalState::Critical,
        ] {
            let sdk: xybrid_sdk::ThermalState = variant.into();
            let back: FfiThermalState = sdk.into();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn memory_pressure_maps_through_sdk_type() {
        for variant in [
            xybrid_sdk::MemoryPressure::Unknown,
            xybrid_sdk::MemoryPressure::Normal,
            xybrid_sdk::MemoryPressure::Warn,
            xybrid_sdk::MemoryPressure::Critical,
        ] {
            let ffi: FfiMemoryPressure = variant.into();
            // Round-trip is intentional via the SDK enum — we don't
            // need a reverse `From` because the host never pushes
            // memory pressure (Rust derives it). Just assert the
            // mapping is total.
            match (variant, ffi) {
                (xybrid_sdk::MemoryPressure::Unknown, FfiMemoryPressure::Unknown) => {}
                (xybrid_sdk::MemoryPressure::Normal, FfiMemoryPressure::Normal) => {}
                (xybrid_sdk::MemoryPressure::Warn, FfiMemoryPressure::Warn) => {}
                (xybrid_sdk::MemoryPressure::Critical, FfiMemoryPressure::Critical) => {}
                other => panic!("unexpected mapping: {other:?}"),
            }
        }
    }

    #[test]
    fn current_snapshot_returns_finite_timestamp() {
        // The reader is a thin wrapper around the global ResourceMonitor;
        // we only need to assert it hands back a usable shape (timestamp
        // populated, no panic). Actual sysinfo coverage lives in the
        // xybrid-core resource tests.
        let snap = XybridDevice::current_snapshot();
        assert!(snap.captured_at_ms > 0, "captured_at_ms must be populated");
    }

    #[test]
    fn thermal_state_maps_to_documented_sdk_variants() {
        assert_eq!(
            xybrid_sdk::ThermalState::from(FfiThermalState::Normal),
            xybrid_sdk::ThermalState::Normal
        );
        assert_eq!(
            xybrid_sdk::ThermalState::from(FfiThermalState::Warm),
            xybrid_sdk::ThermalState::Warm
        );
        assert_eq!(
            xybrid_sdk::ThermalState::from(FfiThermalState::Hot),
            xybrid_sdk::ThermalState::Hot
        );
        assert_eq!(
            xybrid_sdk::ThermalState::from(FfiThermalState::Critical),
            xybrid_sdk::ThermalState::Critical
        );
    }
}
