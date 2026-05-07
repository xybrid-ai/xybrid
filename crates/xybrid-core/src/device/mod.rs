//! Device module - Hardware capability detection and device-specific functionality.
//!
//! This module provides unified hardware capability detection across platforms,
//! including GPU acceleration, neural network APIs, battery level, and thermal state.
//!
//! ## Module Organization
//!
//! The device module is organized into focused submodules:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`types`] | Data types (HardwareCapabilities, enums) |
//! | [`common`] | Cross-platform detection (memory, CPU) |
//! | [`apple`] | Apple platform detection (Metal, CoreML, Neural Engine) |
//! | [`android`] | Android platform detection (NNAPI, API level) |
//! | [`capabilities`] | Main detection logic and re-exports |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::device::{detect_capabilities, HardwareCapabilities};
//!
//! let capabilities = detect_capabilities();
//! println!("GPU available: {}", capabilities.has_gpu());
//! println!("Memory: {} MB", capabilities.memory_total_mb());
//! ```

// Core types
pub mod types;

// Cross-platform detection (memory, CPU)
pub mod common;

// Platform-specific detection
pub mod android;
pub mod apple;

// Main detection logic
pub mod capabilities;

// Telemetry-facing device profile (chip, RAM, OS, for wire events)
pub mod profile;

// Per-inference resource monitor + sampler.
// See `docs/sdk/resource-telemetry.md` for the public contract.
pub mod resource;

// Platform-bridged signals (battery, thermal) — consumed by `resource`.
pub mod platform_state;

// Platform tests
#[cfg(test)]
mod tests;

// Re-exports for convenience
pub use capabilities::detect_capabilities;
pub use platform_state::{
    clear_battery_level, clear_thermal_state, current_platform_state,
    refresh_native_platform_state, set_battery_level, set_platform_state, set_thermal_state,
    PlatformState,
};
pub use profile::DeviceProfile;
pub use resource::{
    MemoryPressure, ResourceMonitor, ResourceSnapshot, ResourceSnapshotProvider,
    ResourceTelemetryMode, ResourceUsageSummary, RunGuard,
};
pub use types::{
    DetectionConfidence, DetectionSource, GpuType, HardwareCapabilities, NpuType, Platform,
    ThermalState,
};
