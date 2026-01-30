//! Apple platform detection (macOS/iOS).
//!
//! This module handles Metal and CoreML Neural Engine detection
//! for Apple platforms.

use super::types::DetectionConfidence;

/// iOS/macOS device family detection result.
#[derive(Debug, Clone)]
pub struct AppleDeviceInfo {
    /// Device identifier (e.g., "iPhone12,1", "MacBookPro18,1")
    pub device_model: Option<String>,
    /// Whether Neural Engine is likely available
    pub has_neural_engine: bool,
    /// Detection confidence
    pub confidence: DetectionConfidence,
}

/// Detects Metal framework availability (macOS/iOS only).
pub fn detect_metal_availability() -> bool {
    // Stub: Always return true on Apple platforms
    // TODO: Real implementation would check:
    // - MTLCreateSystemDefaultDevice() != nil
    // - Device supports compute shaders
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        true
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}

/// Detect Apple device model from environment variables.
///
/// On iOS, the device model can be obtained from various sources:
/// 1. `DEVICE_MODEL` environment variable (may be set by Flutter/runtime)
/// 2. `SIMULATOR_MODEL_IDENTIFIER` for iOS Simulator
///
/// Returns device info with Neural Engine availability inference.
pub fn detect_apple_device() -> AppleDeviceInfo {
    // Try environment variables that might be set by the runtime
    let model = std::env::var("DEVICE_MODEL")
        .or_else(|_| std::env::var("SIMULATOR_MODEL_IDENTIFIER"))
        .or_else(|_| std::env::var("APPLE_DEVICE_MODEL"))
        .ok();

    if let Some(ref model_str) = model {
        let has_ne = has_neural_engine_by_model(model_str);
        return AppleDeviceInfo {
            device_model: Some(model_str.clone()),
            has_neural_engine: has_ne,
            confidence: DetectionConfidence::Medium,
        };
    }

    // No model detected - use architecture-based fallback
    #[cfg(target_arch = "aarch64")]
    {
        // ARM64 Apple devices generally have Neural Engine:
        // - All Apple Silicon Macs (M1/M2/M3/M4)
        // - All iPhones since iPhone 8/X (A11+)
        // - All iPads since iPad Pro 2018 (A12X+)
        // Conservative: assume available on ARM64
        AppleDeviceInfo {
            device_model: None,
            has_neural_engine: true,
            confidence: DetectionConfidence::Medium,
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Intel Macs don't have Neural Engine
        AppleDeviceInfo {
            device_model: None,
            has_neural_engine: false,
            confidence: DetectionConfidence::High, // We know Intel has no NE
        }
    }
}

/// Check if a device model has Neural Engine based on identifier.
///
/// Neural Engine was introduced with:
/// - iPhone: A11 Bionic (iPhone 8/X, 2017) - "iPhone10,x"
/// - iPad: A12X Bionic (iPad Pro 2018) - "iPad8,x"
/// - Mac: M1 (2020) - "MacBookPro17,x", "MacBookAir10,x", etc.
pub fn has_neural_engine_by_model(model: &str) -> bool {
    let model_lower = model.to_lowercase();

    // iPhone detection
    // iPhone10,x = iPhone 8/X (A11) - first with Neural Engine
    // All iPhones after iPhone 10,x have Neural Engine
    if model_lower.starts_with("iphone") {
        if let Some(version_str) = model
            .strip_prefix("iPhone")
            .or_else(|| model.strip_prefix("iphone"))
        {
            if let Some((major_str, _)) = version_str.split_once(',') {
                if let Ok(major) = major_str.parse::<u32>() {
                    // iPhone10+ has Neural Engine (iPhone 8/X and later)
                    return major >= 10;
                }
            }
        }
        // Unknown iPhone format - conservative: assume newer iPhone
        return true;
    }

    // iPad detection
    // iPad8,x = iPad Pro 2018 (A12X) - first iPad with Neural Engine
    // iPad mini 5 (iPad11,1) also has NE
    if model_lower.starts_with("ipad") {
        if let Some(version_str) = model
            .strip_prefix("iPad")
            .or_else(|| model.strip_prefix("ipad"))
        {
            if let Some((major_str, _)) = version_str.split_once(',') {
                if let Ok(major) = major_str.parse::<u32>() {
                    // iPad8+ has Neural Engine
                    return major >= 8;
                }
            }
        }
        // Unknown iPad format - conservative: assume no NE (safer)
        return false;
    }

    // Mac detection - All Apple Silicon Macs have Neural Engine
    // M1 Macs: MacBookPro17, MacBookAir10, Macmini9, iMac21, Mac13
    // M2+ Macs: MacBookPro18+, MacBookAir11+, etc.
    if model_lower.contains("mac") {
        // Check for known Apple Silicon identifiers
        // These are ARM64 Macs with Neural Engine
        let apple_silicon_patterns = [
            "macbookpro17",
            "macbookpro18",
            "macbookpro19",
            "macbookpro20",
            "macbookair10",
            "macbookair11",
            "macbookair12",
            "macmini9",
            "macmini10",
            "imac21",
            "imac22",
            "imac23",
            "imac24",
            "mac13",
            "mac14",
            "mac15", // Mac Studio, Mac Pro
        ];

        for pattern in &apple_silicon_patterns {
            if model_lower.contains(pattern) {
                return true;
            }
        }

        // Unknown Mac - check architecture at compile time
        #[cfg(target_arch = "aarch64")]
        return true; // ARM64 Mac = Apple Silicon
        #[cfg(not(target_arch = "aarch64"))]
        return false; // Intel Mac
    }

    // Apple TV - A10X Fusion (2017) and later have some ANE capability
    // But it's limited, so conservative: false
    if model_lower.starts_with("appletv") {
        return false;
    }

    // Apple Watch - S4 (2018) and later have NE, but we likely won't run on Watch
    if model_lower.starts_with("watch") {
        return false;
    }

    // Unknown device - conservative default
    false
}

/// Detects CoreML Neural Engine availability (macOS/iOS only).
///
/// - Detects device model from environment variables
/// - Infers Neural Engine availability by device family
/// - iPhone 8/X (A11+) and iPad Pro 2018 (A12X+) have Neural Engine
/// - All Apple Silicon Macs have Neural Engine
/// - Intel Macs do NOT have Neural Engine
pub fn detect_coreml_availability() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let device_info = detect_apple_device();
        device_info.has_neural_engine
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}
