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
