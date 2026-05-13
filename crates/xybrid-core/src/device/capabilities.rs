//! Hardware Capability Detection module.
//!
//! This module provides unified hardware capability detection across platforms,
//! including GPU acceleration (Metal, Vulkan), neural processing units (CoreML, NNAPI),
//! and memory/CPU profiling. Battery and thermal state default to safe values
//! (`100%`, `Normal`); a future platform-bridge slice will populate them on
//! mobile and Windows.
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
//! ```rust,ignore
//! use xybrid_core::device::capabilities::{HardwareCapabilities, detect_capabilities};
//!
//! let capabilities = detect_capabilities();
//! if capabilities.has_gpu() {
//!     println!("GPU acceleration available");
//! }
//! if capabilities.has_npu() {
//!     println!("NPU available: {:?}", capabilities.npu_type());
//! }
//! println!("Memory confidence: {:?}", capabilities.memory_confidence);
//! ```

// Re-export types from submodules
pub use super::types::{
    DetectionConfidence, DetectionSource, GpuType, HardwareCapabilities, NpuType, Platform,
    ThermalState,
};

// Import platform-specific detection
use super::common::{detect_cpu, detect_memory};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use super::apple::{detect_metal_with_confidence, detect_neural_engine_with_confidence};

#[cfg(target_os = "android")]
use super::android::detect_nnapi_availability;

/// Detects hardware capabilities for the current platform.
///
/// Returns a `HardwareCapabilities` struct populated from cross-platform
/// detection (memory, CPU cores) and platform-specific accelerator probes
/// (Metal/CoreML on Apple, NNAPI on Android, Vulkan/DirectX stubs elsewhere).
///
/// **Cache scope:** only static fields are cached after the first call —
/// accelerator presence (`has_gpu`, `has_metal`, `has_npu`, `has_nnapi`),
/// accelerator types, confidence levels, CPU core count, and total memory.
/// Dynamic fields (`memory_available_mb`, `cpu_usage_percent`) are refreshed
/// on every call because they vary over the process lifetime. The cache
/// exists to skip the ~1-second cold-init cost of `MLAllComputeDevices`
/// on first Core ML access — not to freeze live system state.
///
/// `battery_level` and `thermal_state` are NOT populated here — they live on
/// the live `ResourceSnapshot` and are overlaid onto a `HardwareCapabilities`
/// view at routing time via [`crate::context::DeviceMetrics::with_live_snapshot`].
pub fn detect_capabilities() -> HardwareCapabilities {
    use std::sync::OnceLock;
    static STATIC_CACHE: OnceLock<HardwareCapabilities> = OnceLock::new();
    let mut caps = STATIC_CACHE
        .get_or_init(detect_capabilities_uncached)
        .clone();
    // Refresh dynamic fields on each call — sysinfo handles these cheaply
    // (< 1ms) so there's no reason to cache stale values.
    let memory_info = detect_memory();
    caps.memory_available_mb = memory_info.available_mb;
    let cpu_info = detect_cpu();
    caps.cpu_usage_percent = cpu_info.usage_percent;
    caps
}

/// Prewarm the capability cache.
///
/// Triggers `detect_capabilities` (synchronously populating the OnceLock
/// static-fields cache) if it hasn't been called yet. Cheap no-op after
/// the first call. Call this from long-lived construction paths — e.g.
/// `LocalAuthority::new` — to keep the cold-init cost (~1s on first
/// `MLAllComputeDevices` invocation when Core ML lazy-loads on macOS/iOS)
/// out of latency-sensitive hot paths like routing decisions. The refresh
/// of dynamic memory/CPU fields on each `detect_capabilities` call is
/// unaffected by this prewarm.
pub fn prewarm() {
    let _ = detect_capabilities();
}

fn detect_capabilities_uncached() -> HardwareCapabilities {
    let mut capabilities = HardwareCapabilities::new();

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
        let (metal_present, metal_conf) = detect_metal_with_confidence();
        capabilities.has_metal = metal_present;
        capabilities.has_gpu = metal_present;
        capabilities.gpu_type = if metal_present {
            GpuType::Metal
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = metal_conf;

        let (ne_present, ne_conf) = detect_neural_engine_with_confidence();
        capabilities.has_npu = ne_present;
        capabilities.npu_type = if ne_present {
            NpuType::CoreML
        } else {
            NpuType::None
        };
        capabilities.npu_confidence = ne_conf;
    }

    #[cfg(target_os = "ios")]
    {
        let (metal_present, metal_conf) = detect_metal_with_confidence();
        capabilities.has_metal = metal_present;
        capabilities.has_gpu = metal_present;
        capabilities.gpu_type = if metal_present {
            GpuType::Metal
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = metal_conf;

        let (ne_present, ne_conf) = detect_neural_engine_with_confidence();
        capabilities.has_npu = ne_present;
        capabilities.npu_type = if ne_present {
            NpuType::CoreML
        } else {
            NpuType::None
        };
        capabilities.npu_confidence = ne_conf;
    }

    #[cfg(target_os = "android")]
    {
        capabilities.has_nnapi = detect_nnapi_availability();

        // Android GPU probe via the NDK Vulkan loader requires a JNI
        // bridge; deferred. Returning Unknown is more honest than the
        // silent `true` v1 had.
        capabilities.has_gpu = false;
        capabilities.gpu_type = GpuType::None;
        capabilities.gpu_confidence = DetectionConfidence::Unknown;

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
        let (gpu_present, gpu_conf) = super::windows::detect_gpu_with_confidence();
        capabilities.has_gpu = gpu_present;
        capabilities.gpu_type = if gpu_present {
            GpuType::DirectX
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = gpu_conf;

        // DirectML NPU detection not implemented; mark Unknown rather than
        // the v1 hardcoded false which read as a confident negative.
        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Unknown;
    }

    #[cfg(target_os = "linux")]
    {
        // Cheap probe: /dev/dri/renderD* files exist when the kernel has
        // exposed a DRM render node. Catches NVIDIA/AMD/Intel/Mesa — any
        // userspace-visible GPU surface. Does NOT catch WSL2 (/dev/dxg)
        // or sandboxed containers where /dev/dri is denied; those return
        // (false, Unknown).
        let has_render_node = std::fs::read_dir("/dev/dri/")
            .map(|entries| {
                entries
                    .flatten()
                    .any(|e| e.file_name().to_string_lossy().starts_with("renderD"))
            })
            .unwrap_or(false);

        capabilities.has_gpu = has_render_node;
        capabilities.gpu_type = if has_render_node {
            GpuType::Vulkan
        } else {
            GpuType::None
        };
        capabilities.gpu_confidence = if has_render_node {
            DetectionConfidence::Medium
        } else {
            DetectionConfidence::Unknown
        };

        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Unknown;
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows",
        target_os = "linux"
    )))]
    {
        // Unknown platform — nothing to probe.
        capabilities.has_gpu = false;
        capabilities.gpu_type = GpuType::None;
        capabilities.gpu_confidence = DetectionConfidence::Unknown;
        capabilities.has_npu = false;
        capabilities.npu_type = NpuType::None;
        capabilities.npu_confidence = DetectionConfidence::Unknown;
    }

    capabilities
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
        let caps = detect_capabilities();
        // battery_level and thermal_state default to safe values; the live
        // overlay (via DeviceMetrics::with_live_snapshot) populates them.
        assert_eq!(caps.battery_level(), 100);
        assert_eq!(caps.thermal_state(), ThermalState::Normal);
        assert!(caps.memory_total_mb() > 0);
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
        assert!(parsed.has_gpu);
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
        let caps = detect_capabilities();
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
                || caps.gpu_confidence == DetectionConfidence::Unknown,
            "gpu_confidence must be one of the four documented variants",
        );
    }

    #[test]
    fn test_detect_capabilities_has_cpu_info() {
        let caps = detect_capabilities();
        // Should have CPU info
        assert!(caps.cpu_cores >= 1, "Should detect at least 1 CPU core");
        // Memory should be detected
        assert!(caps.memory_total_mb > 0, "Should detect total memory");
    }

    #[test]
    fn test_detection_confidence_unknown_added() {
        assert_eq!(DetectionConfidence::Unknown.as_str(), "unknown");
        let default: DetectionConfidence = Default::default();
        assert_eq!(default, DetectionConfidence::Low);
    }

    #[test]
    fn test_detection_confidence_wire_format_stays_capitalized() {
        let json = serde_json::to_string(&DetectionConfidence::Unknown).unwrap();
        assert_eq!(
            json, "\"Unknown\"",
            "wire format must stay capitalized — no rename_all"
        );
        let parsed: DetectionConfidence = serde_json::from_str("\"High\"").unwrap();
        assert_eq!(parsed, DetectionConfidence::High);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn test_capabilities_apple_uses_real_probes() {
        let caps = detect_capabilities();
        assert_eq!(caps.gpu_confidence, DetectionConfidence::High);
        assert!(matches!(
            caps.npu_confidence,
            DetectionConfidence::High | DetectionConfidence::Medium,
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_gpu_no_silent_true() {
        let caps = detect_capabilities();
        if caps.has_gpu {
            assert!(
                matches!(
                    caps.gpu_confidence,
                    DetectionConfidence::Medium | DetectionConfidence::High
                ),
                "has_gpu=true must come with Medium+ confidence",
            );
            assert_eq!(caps.gpu_type, GpuType::Vulkan);
        } else {
            assert_eq!(caps.gpu_confidence, DetectionConfidence::Unknown);
            assert_eq!(caps.gpu_type, GpuType::None);
        }
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_android_gpu_unknown() {
        let caps = detect_capabilities();
        assert!(!caps.has_gpu);
        assert_eq!(caps.gpu_confidence, DetectionConfidence::Unknown);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_windows_real_dxgi_probe() {
        let caps = detect_capabilities();
        assert!(matches!(
            caps.gpu_confidence,
            DetectionConfidence::High | DetectionConfidence::Unknown,
        ));
    }
}
