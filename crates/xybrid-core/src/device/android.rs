//! Android platform detection.
//!
//! This module handles NNAPI and API level detection for Android devices.

use super::types::DetectionConfidence;

/// Android API level detection result.
#[derive(Debug, Clone)]
pub struct AndroidApiInfo {
    pub api_level: Option<u32>,
    pub confidence: DetectionConfidence,
}

/// Detect Android API level from environment variables or build properties.
///
/// This checks several sources:
/// 1. `ANDROID_SDK_VERSION` environment variable
/// 2. `ro.build.version.sdk` system property (if exposed)
///
/// Returns None if API level cannot be determined, triggering conservative defaults.
pub fn detect_android_api_level() -> AndroidApiInfo {
    // Try environment variable first (may be set by build system or runtime)
    if let Ok(sdk_str) = std::env::var("ANDROID_SDK_VERSION") {
        if let Ok(level) = sdk_str.parse::<u32>() {
            return AndroidApiInfo {
                api_level: Some(level),
                confidence: DetectionConfidence::Medium,
            };
        }
    }

    // Try alternative environment variable names
    for env_var in &["ANDROID_API_LEVEL", "SDK_INT", "TARGET_SDK_VERSION"] {
        if let Ok(sdk_str) = std::env::var(env_var) {
            if let Ok(level) = sdk_str.parse::<u32>() {
                return AndroidApiInfo {
                    api_level: Some(level),
                    confidence: DetectionConfidence::Medium,
                };
            }
        }
    }

    // Could not detect - return unknown
    AndroidApiInfo {
        api_level: None,
        confidence: DetectionConfidence::Low,
    }
}

/// Detects NNAPI (Android Neural Networks API) availability.
///
/// - Checks Android API level (NNAPI requires API 27+)
/// - Uses conservative defaults when API level unknown
///
/// NNAPI was introduced in Android 8.1 (API level 27).
/// For full accelerator support, API 29+ is recommended.
pub fn detect_nnapi_availability() -> bool {
    #[cfg(target_os = "android")]
    {
        let api_info = detect_android_api_level();

        match api_info.api_level {
            Some(level) => {
                // NNAPI requires API level 27 (Android 8.1) or higher
                level >= 27
            }
            None => {
                // Conservative default: assume NNAPI is NOT available
                // This prevents attempting to use unavailable APIs
                false
            }
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        false
    }
}
