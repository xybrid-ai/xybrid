//! Hardware Capability Detection module.
//!
//! This module provides unified hardware capability detection across platforms,
//! including GPU acceleration (Metal, Vulkan), neural processing units (CoreML, NNAPI),
//! battery level, thermal state, and memory profiling.
//!
//! ## Module Organization
//!
//! The device module is organized into focused submodules:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`types`](super::types) | Data types (HardwareCapabilities, enums) |
//! | [`common`](super::common) | Cross-platform detection (memory, CPU) |
//! | [`apple`](super::apple) | Apple platform detection (Metal, CoreML) |
//! | [`android`](super::android) | Android platform detection (NNAPI) |
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::device::capabilities::{HardwareCapabilities, detect_capabilities};
//! use xybrid_core::context::DeviceMetrics;
//!
//! let metrics = DeviceMetrics {
//!     network_rtt: 100,
//!     battery: 75,
//!     temperature: 25.0,
//! };
//!
//! let capabilities = detect_capabilities(&metrics);
//! if capabilities.has_gpu() {
//!     println!("GPU acceleration available");
//! }
//! if capabilities.has_npu() {
//!     println!("NPU available: {:?}", capabilities.npu_type());
//! }
//! println!("Memory confidence: {:?}", capabilities.memory_confidence);
//! ```

use crate::context::DeviceMetrics;

// Re-export types from submodules
pub use super::types::{
    DetectionConfidence, DetectionSource, GpuType, HardwareCapabilities, NpuType, Platform,
    ThermalState,
};

// Import platform-specific detection
use super::common::{detect_cpu, detect_memory};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use super::apple::{detect_coreml_availability, detect_metal_availability};

#[cfg(target_os = "android")]
use super::android::detect_nnapi_availability;

/// Detects hardware capabilities for the current platform.
///
/// This function performs platform-specific detection and returns a
/// `HardwareCapabilities` struct with the detected capabilities.
///
/// - Uses `sysinfo` crate for accurate memory/CPU detection
/// - Includes detection confidence indicators
/// - Cross-platform CPU core count and usage
///
/// # Arguments
///
/// * `metrics` - Device metrics containing battery and temperature information
///
/// # Returns
///
/// A `HardwareCapabilities` struct with detected capabilities
pub fn detect_capabilities(metrics: &DeviceMetrics) -> HardwareCapabilities {
    let mut capabilities = HardwareCapabilities::new();

    // Set battery level from metrics
    capabilities.battery_level = metrics.battery;

    // Determine thermal state from temperature
    // These thresholds are approximate and may need tuning per device
    capabilities.thermal_state = if metrics.temperature > 80.0 {
        ThermalState::Critical
    } else if metrics.temperature > 70.0 {
        ThermalState::Hot
    } else if metrics.temperature > 60.0 {
        ThermalState::Warm
    } else {
        ThermalState::Normal
    };

    // Detect memory using sysinfo
    let memory_info = detect_memory();
    capabilities.memory_available_mb = memory_info.available_mb;
    capabilities.memory_total_mb = memory_info.total_mb;
    capabilities.memory_confidence = memory_info.confidence;

    // Detect CPU using sysinfo
    let cpu_info = detect_cpu();
    capabilities.cpu_usage_percent = cpu_info.usage_percent;
    capabilities.cpu_cores = cpu_info.cores;

    // Platform-specific detection
    #[cfg(target_os = "macos")]
    {
        capabilities.has_metal = detect_metal_availability();
        capabilities.has_gpu = capabilities.has_metal;
        capabilities.gpu_type = if capabilities.has_metal {
            GpuType::Metal
        } else {
            GpuType::None
        };
        // GPU confidence: Medium (compile-time check, not runtime validation)
        capabilities.gpu_confidence = DetectionConfidence::Medium;

        // CoreML Neural Engine detection (stub based on arch)
        capabilities.has_npu = detect_coreml_availability();
        capabilities.npu_type = if capabilities.has_npu {
            NpuType::CoreML
        } else {
            NpuType::None
        };
        // NPU confidence: Medium (arch-based, not actual detection)
        capabilities.npu_confidence = DetectionConfidence::Medium;
    }

    #[cfg(target_os = "ios")]
    {
        capabilities.has_metal = detect_metal_availability();
        capabilities.has_gpu = capabilities.has_metal;
        capabilities.gpu_type = if capabilities.has_metal {
            GpuType::Metal
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = DetectionConfidence::Medium;

        capabilities.has_npu = detect_coreml_availability();
        capabilities.npu_type = if capabilities.has_npu {
            NpuType::CoreML
        } else {
            NpuType::None
        };
        capabilities.npu_confidence = DetectionConfidence::Medium;
    }

    #[cfg(target_os = "android")]
    {
        capabilities.has_nnapi = detect_nnapi_availability();
        capabilities.has_gpu = detect_gpu_availability_stub();
        capabilities.gpu_type = if capabilities.has_gpu {
            GpuType::Vulkan
        } else {
            GpuType::None
        };
        // GPU confidence: Low (stubbed, always true)
        capabilities.gpu_confidence = DetectionConfidence::Low;

        // NNAPI can use NPU accelerators
        capabilities.has_npu = capabilities.has_nnapi;
        capabilities.npu_type = if capabilities.has_nnapi {
            NpuType::NNAPI
        } else {
            NpuType::None
        };
        // NPU confidence: Medium (checks API level from env vars)
        let api_info = super::android::detect_android_api_level();
        capabilities.npu_confidence = api_info.confidence;
    }

    #[cfg(target_os = "windows")]
    {
        capabilities.has_gpu = detect_gpu_availability_stub();
        capabilities.gpu_type = if capabilities.has_gpu {
            GpuType::DirectX
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = DetectionConfidence::Low;

        // DirectML NPU (stub)
        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Low;
    }

    #[cfg(target_os = "linux")]
    {
        capabilities.has_gpu = detect_gpu_availability_stub();
        capabilities.gpu_type = if capabilities.has_gpu {
            GpuType::Vulkan
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = DetectionConfidence::Low;

        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Low;
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows",
        target_os = "linux"
    )))]
    {
        capabilities.has_gpu = detect_gpu_availability_stub();
        capabilities.gpu_type = GpuType::None;
        capabilities.gpu_confidence = DetectionConfidence::Low;
        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Low;
    }

    capabilities
}

/// Stub GPU detection for platforms without specific detection.
#[allow(dead_code)]
fn detect_gpu_availability_stub() -> bool {
    // Stub: Always return true for now
    // TODO: Real implementation would check:
    // - Vulkan device availability
    // - GPU compute shader support
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_capabilities() {
        let caps = HardwareCapabilities::new();
        assert!(!caps.has_gpu());
        assert!(!caps.has_nnapi());
        assert!(!caps.has_metal());
        assert!(!caps.has_npu());
        assert_eq!(caps.battery_level(), 100);
        assert_eq!(caps.thermal_state(), ThermalState::Normal);
        assert_eq!(caps.gpu_type(), GpuType::None);
        assert_eq!(caps.npu_type(), NpuType::None);
        // Check new fields
        assert_eq!(caps.cpu_cores(), 1);
        assert_eq!(caps.cpu_usage_percent(), 0.0);
        assert_eq!(caps.memory_confidence, DetectionConfidence::Low);
        assert_eq!(caps.gpu_confidence, DetectionConfidence::Low);
        assert_eq!(caps.npu_confidence, DetectionConfidence::Low);
    }

    #[test]
    fn test_should_throttle_low_battery() {
        let mut caps = HardwareCapabilities::new();
        caps.battery_level = 15;
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_should_throttle_hot_device() {
        let mut caps = HardwareCapabilities::new();
        caps.thermal_state = ThermalState::Hot;
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_should_not_throttle_normal() {
        let caps = HardwareCapabilities::new();
        assert!(!caps.should_throttle());
    }

    #[test]
    fn test_should_prefer_gpu() {
        let mut caps = HardwareCapabilities::new();
        caps.has_gpu = true;
        caps.battery_level = 50;
        assert!(caps.should_prefer_gpu());
    }

    #[test]
    fn test_should_not_prefer_gpu_low_battery() {
        let mut caps = HardwareCapabilities::new();
        caps.has_gpu = true;
        caps.battery_level = 25;
        assert!(!caps.should_prefer_gpu());
    }

    #[test]
    fn test_should_not_prefer_gpu_critical_thermal() {
        let mut caps = HardwareCapabilities::new();
        caps.has_gpu = true;
        caps.battery_level = 50;
        caps.thermal_state = ThermalState::Critical;
        assert!(!caps.should_prefer_gpu());
    }

    #[test]
    fn test_should_prefer_nnapi() {
        let mut caps = HardwareCapabilities::new();
        caps.has_nnapi = true;
        caps.battery_level = 50;
        assert!(caps.should_prefer_nnapi());
    }

    #[test]
    fn test_should_not_prefer_nnapi_low_battery() {
        let mut caps = HardwareCapabilities::new();
        caps.has_nnapi = true;
        caps.battery_level = 15;
        assert!(!caps.should_prefer_nnapi());
    }

    #[test]
    fn test_should_prefer_metal() {
        let mut caps = HardwareCapabilities::new();
        caps.has_metal = true;
        caps.battery_level = 50;
        assert!(caps.should_prefer_metal());
    }

    #[test]
    fn test_should_prefer_npu() {
        let mut caps = HardwareCapabilities::new();
        caps.has_npu = true;
        caps.battery_level = 50;
        assert!(caps.should_prefer_npu());
    }

    #[test]
    fn test_should_not_prefer_npu_low_battery() {
        let mut caps = HardwareCapabilities::new();
        caps.has_npu = true;
        caps.battery_level = 15;
        assert!(!caps.should_prefer_npu());
    }

    #[test]
    fn test_can_load_model() {
        let mut caps = HardwareCapabilities::new();
        caps.memory_available_mb = 4096;

        // Model that fits with default 1.5x margin
        assert!(caps.can_load_model(2000, None)); // 2000 * 1.5 = 3000 < 4096

        // Model that doesn't fit
        assert!(!caps.can_load_model(3000, None)); // 3000 * 1.5 = 4500 > 4096

        // Model that fits with custom margin
        assert!(caps.can_load_model(3000, Some(1.2))); // 3000 * 1.2 = 3600 < 4096
    }

    #[test]
    fn test_detect_capabilities() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 25.0,
        };

        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.battery_level(), 75);
        assert_eq!(caps.thermal_state(), ThermalState::Normal);
        assert!(caps.memory_total_mb() > 0);
    }

    #[test]
    fn test_detect_capabilities_hot_device() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 75.0,
        };

        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Hot);
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_detect_capabilities_critical_device() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 85.0,
        };

        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Critical);
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_detect_capabilities_low_battery() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 15,
            temperature: 25.0,
        };

        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.battery_level(), 15);
        assert!(caps.should_throttle());
    }

    #[test]
    fn test_thermal_state_thresholds() {
        // Normal
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };
        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Normal);

        // Warm
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 65.0,
        };
        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Warm);

        // Hot
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 75.0,
        };
        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Hot);

        // Critical
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 85.0,
        };
        let caps = detect_capabilities(&metrics);
        assert_eq!(caps.thermal_state(), ThermalState::Critical);
    }

    #[test]
    fn test_json_serialization() {
        let mut caps = HardwareCapabilities::new();
        caps.has_gpu = true;
        caps.gpu_type = GpuType::Metal;
        caps.battery_level = 85;
        caps.memory_total_mb = 16384;
        caps.memory_available_mb = 8192;

        let json = caps.to_json();
        assert!(json.contains("\"has_gpu\":true"));
        assert!(json.contains("\"battery_level\":85"));

        let parsed = HardwareCapabilities::from_json(&json).unwrap();
        assert_eq!(parsed.has_gpu, true);
        assert_eq!(parsed.battery_level, 85);
    }

    #[test]
    fn test_platform_current() {
        let platform = Platform::current();
        #[cfg(target_os = "macos")]
        assert_eq!(platform, Platform::MacOS);
        #[cfg(target_os = "linux")]
        assert_eq!(platform, Platform::Linux);
    }

    #[test]
    fn test_enum_as_str() {
        assert_eq!(ThermalState::Normal.as_str(), "normal");
        assert_eq!(ThermalState::Critical.as_str(), "critical");
        assert_eq!(GpuType::Metal.as_str(), "metal");
        assert_eq!(NpuType::CoreML.as_str(), "coreml");
        assert_eq!(Platform::MacOS.as_str(), "macos");
    }

    #[test]
    fn test_detection_confidence_as_str() {
        assert_eq!(DetectionConfidence::High.as_str(), "high");
        assert_eq!(DetectionConfidence::Medium.as_str(), "medium");
        assert_eq!(DetectionConfidence::Low.as_str(), "low");
    }

    #[test]
    fn test_detection_confidence_default() {
        let confidence: DetectionConfidence = Default::default();
        assert_eq!(confidence, DetectionConfidence::Low);
    }

    #[test]
    fn test_sysinfo_memory_detection() {
        let info = detect_memory();
        // sysinfo should return real values on all platforms
        assert!(info.total_mb > 0, "Total memory should be > 0");
        // Available memory should be <= total
        assert!(
            info.available_mb <= info.total_mb,
            "Available should be <= total"
        );
        // Confidence should be High when sysinfo works
        assert_eq!(info.confidence, DetectionConfidence::High);
    }

    #[test]
    fn test_sysinfo_cpu_detection() {
        let info = detect_cpu();
        // Should have at least 1 core
        assert!(info.cores >= 1, "Should have at least 1 CPU core");
        // CPU usage should be in valid range
        assert!(
            info.usage_percent >= 0.0 && info.usage_percent <= 100.0,
            "CPU usage should be 0-100%"
        );
    }

    #[test]
    fn test_detect_capabilities_has_confidence() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 25.0,
        };

        let caps = detect_capabilities(&metrics);
        // Memory confidence should be High when using sysinfo
        assert_eq!(
            caps.memory_confidence,
            DetectionConfidence::High,
            "Memory detection should have High confidence with sysinfo"
        );
        // GPU/NPU confidence depends on platform but should be set
        assert!(
            caps.gpu_confidence == DetectionConfidence::High
                || caps.gpu_confidence == DetectionConfidence::Medium
                || caps.gpu_confidence == DetectionConfidence::Low
        );
    }

    #[test]
    fn test_detect_capabilities_has_cpu_info() {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 75,
            temperature: 25.0,
        };

        let caps = detect_capabilities(&metrics);
        // Should have CPU info
        assert!(caps.cpu_cores >= 1, "Should detect at least 1 CPU core");
        // Memory should be detected
        assert!(caps.memory_total_mb > 0, "Should detect total memory");
    }
}
