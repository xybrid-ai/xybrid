//! Platform and target resolution for xybrid bundles.
//!
//! This module provides:
//! - Platform detection (iOS, macOS, Android, Linux, Windows, Web)
//! - Target format definitions (onnx, coreml, tflite, etc.)
//! - Target resolution logic with fallback chain

use std::fmt;

/// Supported target formats for model bundles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Target {
    /// ONNX Runtime (cross-platform, default fallback)
    #[default]
    Onnx,
    /// Apple CoreML (iOS, macOS)
    CoreML,
    /// TensorFlow Lite (Android, embedded)
    TFLite,
    /// Generic/platform-agnostic
    Generic,
    /// WebGPU shaders (future)
    WebGPU,
    /// Metal compute shaders (future)
    Metal,
    /// Vulkan compute shaders (future)
    Vulkan,
}

impl Target {
    /// Parse a target from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "onnx" => Some(Target::Onnx),
            "coreml" | "core_ml" | "mlmodel" => Some(Target::CoreML),
            "tflite" | "tf_lite" | "tensorflow_lite" | "tensorflowlite" => Some(Target::TFLite),
            "generic" | "universal" => Some(Target::Generic),
            "webgpu" | "web_gpu" => Some(Target::WebGPU),
            "metal" => Some(Target::Metal),
            "vulkan" => Some(Target::Vulkan),
            _ => None,
        }
    }

    /// Get the string representation of this target.
    pub fn as_str(&self) -> &'static str {
        match self {
            Target::Onnx => "onnx",
            Target::CoreML => "coreml",
            Target::TFLite => "tflite",
            Target::Generic => "generic",
            Target::WebGPU => "webgpu",
            Target::Metal => "metal",
            Target::Vulkan => "vulkan",
        }
    }

    /// Check if this target is currently supported for inference.
    pub fn is_supported(&self) -> bool {
        matches!(self, Target::Onnx | Target::Generic)
        // CoreML, TFLite, WebGPU, Metal, Vulkan are future targets
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Runtime platform where xybrid is executing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    /// iOS (iPhone, iPad)
    IOS,
    /// macOS (Mac desktop/laptop)
    MacOS,
    /// Android
    Android,
    /// Linux
    Linux,
    /// Windows
    Windows,
    /// Web (WASM)
    Web,
    /// Unknown/generic platform
    Generic,
}

impl Platform {
    /// Detect the current platform at compile time.
    pub fn detect() -> Self {
        #[cfg(target_os = "ios")]
        return Platform::IOS;

        #[cfg(target_os = "macos")]
        return Platform::MacOS;

        #[cfg(target_os = "android")]
        return Platform::Android;

        #[cfg(target_os = "linux")]
        return Platform::Linux;

        #[cfg(target_os = "windows")]
        return Platform::Windows;

        #[cfg(target_arch = "wasm32")]
        return Platform::Web;

        #[cfg(not(any(
            target_os = "ios",
            target_os = "macos",
            target_os = "android",
            target_os = "linux",
            target_os = "windows",
            target_arch = "wasm32"
        )))]
        return Platform::Generic;
    }

    /// Get the preferred target for this platform.
    pub fn preferred_target(&self) -> Target {
        match self {
            Platform::IOS | Platform::MacOS => Target::CoreML,
            Platform::Android => Target::TFLite,
            Platform::Linux | Platform::Windows => Target::Onnx,
            Platform::Web => Target::WebGPU,
            Platform::Generic => Target::Onnx,
        }
    }

    /// Get the string representation of this platform.
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::IOS => "ios",
            Platform::MacOS => "macos",
            Platform::Android => "android",
            Platform::Linux => "linux",
            Platform::Windows => "windows",
            Platform::Web => "web",
            Platform::Generic => "generic",
        }
    }
}

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Target resolution configuration.
#[derive(Debug, Clone)]
pub struct TargetResolver {
    /// Explicitly requested target (takes priority)
    pub requested: Option<Target>,
    /// Available targets for the model
    pub available: Vec<Target>,
    /// Current platform
    pub platform: Platform,
}

impl TargetResolver {
    /// Create a new target resolver.
    pub fn new() -> Self {
        Self {
            requested: None,
            available: Vec::new(),
            platform: Platform::detect(),
        }
    }

    /// Set the explicitly requested target.
    pub fn with_requested(mut self, target: Option<&str>) -> Self {
        self.requested = target.and_then(Target::from_str);
        self
    }

    /// Set the available targets.
    pub fn with_available(mut self, targets: Vec<String>) -> Self {
        self.available = targets.iter().filter_map(|s| Target::from_str(s)).collect();
        self
    }

    /// Override the platform (for testing or cross-compilation).
    pub fn with_platform(mut self, platform: Platform) -> Self {
        self.platform = platform;
        self
    }

    /// Resolve the best target to use.
    ///
    /// Resolution order:
    /// 1. Explicitly requested target (if available)
    /// 2. Platform-preferred target (if available)
    /// 3. ONNX (universal fallback, if available)
    /// 4. Generic (if available)
    /// 5. First available target
    /// 6. ONNX (default if nothing available)
    pub fn resolve(&self) -> Target {
        // 1. If explicitly requested and available, use it
        if let Some(requested) = self.requested {
            if self.available.is_empty() || self.available.contains(&requested) {
                return requested;
            }
        }

        // If no available targets specified, return preferred or default
        if self.available.is_empty() {
            return self
                .requested
                .unwrap_or_else(|| self.platform.preferred_target());
        }

        // 2. Platform-preferred target
        let preferred = self.platform.preferred_target();
        if self.available.contains(&preferred) {
            return preferred;
        }

        // 3. ONNX (universal fallback)
        if self.available.contains(&Target::Onnx) {
            return Target::Onnx;
        }

        // 4. Generic
        if self.available.contains(&Target::Generic) {
            return Target::Generic;
        }

        // 5. First available
        self.available.first().copied().unwrap_or(Target::Onnx)
    }

    /// Resolve with a list of available target strings.
    pub fn resolve_from_strings(&self, available: &[String]) -> String {
        let resolver = self.clone().with_available(available.to_vec());
        resolver.resolve().as_str().to_string()
    }
}

impl Default for TargetResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the preferred target for the current platform.
pub fn preferred_target() -> Target {
    Platform::detect().preferred_target()
}

/// Get the current platform.
pub fn current_platform() -> Platform {
    Platform::detect()
}

/// Resolve a target from a string, with fallback to platform default.
pub fn resolve_target(target: Option<&str>) -> Target {
    target
        .and_then(Target::from_str)
        .unwrap_or_else(preferred_target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_from_str() {
        assert_eq!(Target::from_str("onnx"), Some(Target::Onnx));
        assert_eq!(Target::from_str("ONNX"), Some(Target::Onnx));
        assert_eq!(Target::from_str("coreml"), Some(Target::CoreML));
        assert_eq!(Target::from_str("CoreML"), Some(Target::CoreML));
        assert_eq!(Target::from_str("tflite"), Some(Target::TFLite));
        assert_eq!(Target::from_str("tf_lite"), Some(Target::TFLite));
        assert_eq!(Target::from_str("generic"), Some(Target::Generic));
        assert_eq!(Target::from_str("unknown"), None);
    }

    #[test]
    fn test_target_resolver_explicit() {
        let resolver = TargetResolver::new()
            .with_requested(Some("coreml"))
            .with_available(vec!["onnx".to_string(), "coreml".to_string()]);

        assert_eq!(resolver.resolve(), Target::CoreML);
    }

    #[test]
    fn test_target_resolver_fallback() {
        let resolver = TargetResolver::new()
            .with_platform(Platform::Android)
            .with_available(vec!["onnx".to_string()]);

        // Android prefers tflite, but only onnx is available
        assert_eq!(resolver.resolve(), Target::Onnx);
    }

    #[test]
    fn test_target_resolver_preferred() {
        let resolver = TargetResolver::new()
            .with_platform(Platform::IOS)
            .with_available(vec!["onnx".to_string(), "coreml".to_string()]);

        // iOS prefers coreml
        assert_eq!(resolver.resolve(), Target::CoreML);
    }

    #[test]
    fn test_platform_detection() {
        let platform = Platform::detect();
        // Just verify it returns something valid
        assert!(!platform.as_str().is_empty());
    }
}
