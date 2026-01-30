//! Platform-specific tests for device capability detection.
//!
//! These tests are separated to keep platform-specific logic isolated.

#[cfg(test)]
mod apple_tests {
    use crate::device::apple::{detect_apple_device, has_neural_engine_by_model};
    use crate::device::types::DetectionConfidence;

    #[test]
    fn test_has_neural_engine_iphone_models() {
        // iPhone 8/X and later (iPhone10+) have Neural Engine
        assert!(has_neural_engine_by_model("iPhone10,1")); // iPhone 8
        assert!(has_neural_engine_by_model("iPhone10,4")); // iPhone 8
        assert!(has_neural_engine_by_model("iPhone10,3")); // iPhone X
        assert!(has_neural_engine_by_model("iPhone11,2")); // iPhone XS
        assert!(has_neural_engine_by_model("iPhone12,1")); // iPhone 11
        assert!(has_neural_engine_by_model("iPhone13,1")); // iPhone 12 mini
        assert!(has_neural_engine_by_model("iPhone14,5")); // iPhone 13
        assert!(has_neural_engine_by_model("iPhone15,2")); // iPhone 14 Pro
        assert!(has_neural_engine_by_model("iPhone16,1")); // iPhone 15 Pro

        // iPhone 7 and earlier (iPhone9 and below) do NOT have Neural Engine
        assert!(!has_neural_engine_by_model("iPhone9,1")); // iPhone 7
        assert!(!has_neural_engine_by_model("iPhone9,3")); // iPhone 7
        assert!(!has_neural_engine_by_model("iPhone8,1")); // iPhone 6s
        assert!(!has_neural_engine_by_model("iPhone7,2")); // iPhone 6
    }

    #[test]
    fn test_has_neural_engine_ipad_models() {
        // iPad Pro 2018 and later (iPad8+) have Neural Engine
        assert!(has_neural_engine_by_model("iPad8,1")); // iPad Pro 11" 2018
        assert!(has_neural_engine_by_model("iPad8,5")); // iPad Pro 12.9" 2018
        assert!(has_neural_engine_by_model("iPad11,1")); // iPad mini 5
        assert!(has_neural_engine_by_model("iPad13,1")); // iPad Air 4
        assert!(has_neural_engine_by_model("iPad14,1")); // iPad mini 6

        // Older iPads do NOT have Neural Engine
        assert!(!has_neural_engine_by_model("iPad7,5")); // iPad 6th gen
        assert!(!has_neural_engine_by_model("iPad6,11")); // iPad 5th gen
        assert!(!has_neural_engine_by_model("iPad5,3")); // iPad Air 2
    }

    #[test]
    fn test_has_neural_engine_mac_models() {
        // Apple Silicon Macs have Neural Engine
        assert!(has_neural_engine_by_model("MacBookPro17,1")); // M1 MacBook Pro 13"
        assert!(has_neural_engine_by_model("MacBookPro18,1")); // M1 Pro MacBook Pro 16"
        assert!(has_neural_engine_by_model("MacBookAir10,1")); // M1 MacBook Air
        assert!(has_neural_engine_by_model("Macmini9,1")); // M1 Mac mini
        assert!(has_neural_engine_by_model("iMac21,1")); // M1 iMac 24"
        assert!(has_neural_engine_by_model("Mac13,1")); // M1 Mac Studio

        // Note: Intel Mac detection depends on architecture at compile time
        // These tests verify the pattern matching works
    }

    #[test]
    fn test_has_neural_engine_unknown_devices() {
        // Unknown devices should return conservative defaults
        assert!(!has_neural_engine_by_model("AppleTV6,2")); // Apple TV 4K
        assert!(!has_neural_engine_by_model("Watch5,1")); // Apple Watch Series 5
        assert!(!has_neural_engine_by_model("UnknownDevice1,1"));
    }

    #[test]
    fn test_apple_device_detection_fallback() {
        // Without environment variables, should fall back to architecture check
        let info = detect_apple_device();

        // On ARM64 (Apple Silicon), should detect Neural Engine
        #[cfg(target_arch = "aarch64")]
        {
            assert!(info.has_neural_engine);
            assert_eq!(info.confidence, DetectionConfidence::Medium);
        }

        // On Intel, should NOT detect Neural Engine
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert!(!info.has_neural_engine);
            assert_eq!(info.confidence, DetectionConfidence::High);
        }
    }
}

#[cfg(test)]
mod android_tests {
    use crate::device::android::detect_android_api_level;
    use crate::device::types::DetectionConfidence;

    #[test]
    fn test_android_api_level_detection_no_env() {
        // Without environment variables set, should return None/Low confidence
        // Note: This test assumes no ANDROID_SDK_VERSION env var is set
        // In CI, this should be the case
        let info = detect_android_api_level();
        // If no env var is set, api_level should be None
        // (unless running in an Android environment)
        if info.api_level.is_none() {
            assert_eq!(info.confidence, DetectionConfidence::Low);
        }
    }
}
