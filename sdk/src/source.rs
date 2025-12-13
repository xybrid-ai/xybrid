//! Model source definitions for xybrid-sdk.
//!
//! This module defines `ModelSource`, which specifies where to load a model from:
//! - Registry: Download from HTTP registry
//! - Bundle: Load from local .xyb file
//! - Directory: Load from local model directory (development)

use std::path::PathBuf;

/// Source for loading a model.
///
/// Determines where the model files come from before loading.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Load from HTTP registry (downloads bundle if not cached).
    ///
    /// # Example
    /// ```ignore
    /// ModelSource::Registry {
    ///     url: "http://localhost:8080".to_string(),
    ///     model_id: "whisper-tiny".to_string(),
    ///     version: "1.0".to_string(),
    ///     platform: None, // Auto-detect
    /// }
    /// ```
    Registry {
        /// Registry base URL
        url: String,
        /// Model identifier
        model_id: String,
        /// Model version
        version: String,
        /// Target platform (auto-detected if None)
        platform: Option<String>,
    },

    /// Load from local .xyb bundle file.
    ///
    /// # Example
    /// ```ignore
    /// ModelSource::Bundle {
    ///     path: PathBuf::from("models/whisper-tiny.xyb"),
    /// }
    /// ```
    Bundle {
        /// Path to the .xyb bundle file
        path: PathBuf,
    },

    /// Load from local model directory (development mode).
    ///
    /// The directory must contain `model_metadata.json` and model files.
    ///
    /// # Example
    /// ```ignore
    /// ModelSource::Directory {
    ///     path: PathBuf::from("test_models/whisper-tiny"),
    /// }
    /// ```
    Directory {
        /// Path to model directory containing model_metadata.json
        path: PathBuf,
    },
}

impl ModelSource {
    /// Create a registry source with auto-detected platform.
    pub fn registry(url: impl Into<String>, model_id: impl Into<String>, version: impl Into<String>) -> Self {
        ModelSource::Registry {
            url: url.into(),
            model_id: model_id.into(),
            version: version.into(),
            platform: None,
        }
    }

    /// Create a registry source with explicit platform.
    pub fn registry_with_platform(
        url: impl Into<String>,
        model_id: impl Into<String>,
        version: impl Into<String>,
        platform: impl Into<String>,
    ) -> Self {
        ModelSource::Registry {
            url: url.into(),
            model_id: model_id.into(),
            version: version.into(),
            platform: Some(platform.into()),
        }
    }

    /// Create a bundle source.
    pub fn bundle(path: impl Into<PathBuf>) -> Self {
        ModelSource::Bundle { path: path.into() }
    }

    /// Create a directory source.
    pub fn directory(path: impl Into<PathBuf>) -> Self {
        ModelSource::Directory { path: path.into() }
    }

    /// Get the source type as a string.
    pub fn source_type(&self) -> &'static str {
        match self {
            ModelSource::Registry { .. } => "registry",
            ModelSource::Bundle { .. } => "bundle",
            ModelSource::Directory { .. } => "directory",
        }
    }

    /// Get the model ID (if available from source).
    pub fn model_id(&self) -> Option<&str> {
        match self {
            ModelSource::Registry { model_id, .. } => Some(model_id),
            _ => None,
        }
    }

    /// Get the version (if available from source).
    pub fn version(&self) -> Option<&str> {
        match self {
            ModelSource::Registry { version, .. } => Some(version),
            _ => None,
        }
    }
}

/// Detect the current platform for registry downloads.
pub fn detect_platform() -> String {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return "macos-arm64".to_string();

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    return "macos-x86_64".to_string();

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return "linux-x86_64".to_string();

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    return "linux-arm64".to_string();

    #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
    return "ios-arm64".to_string();

    #[cfg(all(target_os = "android", target_arch = "aarch64"))]
    return "android-arm64".to_string();

    #[cfg(all(target_os = "android", target_arch = "arm"))]
    return "android-arm".to_string();

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return "windows-x86_64".to_string();

    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "ios", target_arch = "aarch64"),
        all(target_os = "android", target_arch = "aarch64"),
        all(target_os = "android", target_arch = "arm"),
        all(target_os = "windows", target_arch = "x86_64"),
    )))]
    return "unknown".to_string();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_source() {
        let source = ModelSource::registry("http://localhost:8080", "whisper", "1.0");
        assert_eq!(source.source_type(), "registry");
        assert_eq!(source.model_id(), Some("whisper"));
        assert_eq!(source.version(), Some("1.0"));
    }

    #[test]
    fn test_bundle_source() {
        let source = ModelSource::bundle("models/test.xyb");
        assert_eq!(source.source_type(), "bundle");
        assert_eq!(source.model_id(), None);
    }

    #[test]
    fn test_directory_source() {
        let source = ModelSource::directory("test_models/whisper");
        assert_eq!(source.source_type(), "directory");
    }

    #[test]
    fn test_detect_platform() {
        let platform = detect_platform();
        assert!(!platform.is_empty());
    }
}
