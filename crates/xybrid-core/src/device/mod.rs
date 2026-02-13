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
//! use xybrid_core::context::DeviceMetrics;
//!
//! let metrics = DeviceMetrics {
//!     network_rtt: 100,
//!     battery: 75,
//!     temperature: 25.0,
//! };
//!
//! let capabilities = detect_capabilities(&metrics);
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

// Platform tests
#[cfg(test)]
mod tests;

// Re-exports for convenience
pub use capabilities::detect_capabilities;
pub use types::{
    DetectionConfidence, DetectionSource, GpuType, HardwareCapabilities, NpuType, Platform,
    ThermalState,
};
