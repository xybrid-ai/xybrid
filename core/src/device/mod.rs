//! Device module - Hardware capability detection and device-specific functionality.
//!
//! This module provides unified hardware capability detection across platforms,
//! including GPU acceleration, neural network APIs, battery level, and thermal state.
//!
//! ## v0.0.7 Updates
//!
//! - Real memory detection using `sysinfo` crate
//! - CPU usage and core count detection
//! - Detection confidence indicators (High/Medium/Low)

pub mod capabilities;

pub use capabilities::{
    detect_capabilities, DetectionConfidence, DetectionSource, GpuType, HardwareCapabilities,
    NpuType, Platform, ThermalState,
};
