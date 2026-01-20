//! SDK Cache Provider - Implements Core's CacheProvider trait.
//!
//! This module bridges the SDK's cache management with Core's abstract cache interface.
//! It allows the Orchestrator to check model availability through the SDK's cache system.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_sdk::SdkCacheProvider;
//! use xybrid_core::orchestrator::LocalAuthority;
//! use std::sync::Arc;
//!
//! // Create SDK cache provider
//! let provider = Arc::new(SdkCacheProvider::new()?);
//!
//! // Inject into LocalAuthority
//! let authority = LocalAuthority::with_cache_provider(provider);
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use xybrid_core::cache_provider::CacheProvider;

use super::cache_manager::CacheManager;
use crate::model::SdkError;
use crate::registry_client::RegistryClient;

/// SDK Cache Provider - Unified cache interface for the SDK.
///
/// Implements Core's `CacheProvider` trait, allowing the Orchestrator to
/// check model availability through the SDK's cache system without code duplication.
///
/// ## Features
///
/// - Fuzzy model ID matching (e.g., "kokoro-82m" matches "Kokoro-82M-v1.0-ONNX")
/// - Integrates with `CacheManager` for bundle management
/// - Optionally uses `RegistryClient` for online resolution (when available)
///
/// ## Example
///
/// ```rust,ignore
/// use xybrid_sdk::SdkCacheProvider;
///
/// let provider = SdkCacheProvider::new()?;
/// if provider.is_model_cached("kokoro-82m") {
///     println!("Model is available locally");
/// }
/// ```
pub struct SdkCacheProvider {
    cache: CacheManager,
    /// Optional registry client for online resolution
    registry_client: Option<Arc<RegistryClient>>,
}

impl SdkCacheProvider {
    /// Creates a new SDK cache provider with default cache location.
    pub fn new() -> Result<Self, SdkError> {
        let cache = CacheManager::new()?;
        Ok(Self {
            cache,
            registry_client: None,
        })
    }

    /// Creates an SDK cache provider with a custom cache directory.
    pub fn with_dir(cache_dir: PathBuf) -> Result<Self, SdkError> {
        let cache = CacheManager::with_dir(cache_dir)?;
        Ok(Self {
            cache,
            registry_client: None,
        })
    }

    /// Creates an SDK cache provider with an optional registry client.
    ///
    /// The registry client allows for more precise cache validation (SHA256 checks)
    /// but requires network access.
    pub fn with_registry(cache: CacheManager, registry_client: Arc<RegistryClient>) -> Self {
        Self {
            cache,
            registry_client: Some(registry_client),
        }
    }

    /// Find a model directory that matches the given model ID.
    ///
    /// Uses fuzzy matching to handle:
    /// - Case differences: "kokoro-82m" matches "Kokoro-82M-v1.0-ONNX"
    /// - Separator differences: "kokoro_82m" matches "kokoro-82m"
    /// - Version suffixes: "kokoro-82m" matches "kokoro-82m-v1.0"
    fn find_matching_dir(&self, model_id: &str) -> Option<PathBuf> {
        let model_id_lower = model_id.to_lowercase();
        let model_id_normalized = normalize_name(&model_id_lower);

        let cache_dir = self.cache.cache_dir();

        if !cache_dir.exists() {
            return None;
        }

        // Search directories in the cache
        if let Ok(entries) = std::fs::read_dir(cache_dir) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if !entry_path.is_dir() {
                    continue;
                }

                let dir_name = entry.file_name().to_string_lossy().to_lowercase();
                let dir_name_normalized = normalize_name(&dir_name);

                // Match if directory name contains the model_id (case-insensitive, normalized)
                let is_match = dir_name.contains(&model_id_lower)
                    || dir_name_normalized.contains(&model_id_normalized);

                if is_match && has_model_files(&entry_path) {
                    return Some(entry_path);
                }
            }
        }

        // Also check test_models directory for development
        let test_models = Path::new("test_models").join(model_id);
        if test_models.exists() && has_model_files(&test_models) {
            return Some(test_models);
        }

        None
    }
}

impl CacheProvider for SdkCacheProvider {
    fn is_model_cached(&self, model_id: &str) -> bool {
        // First, try the SDK's CacheManager (for id@version format)
        if self.cache.is_cached(model_id) {
            return true;
        }

        // Then, try fuzzy directory matching
        self.find_matching_dir(model_id).is_some()
    }

    fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
        // First, try the SDK's CacheManager
        if let Some(path) = self.cache.get_cached_path(model_id) {
            // Return the parent directory (bundle file is in a directory)
            return path.parent().map(|p| p.to_path_buf());
        }

        // Then, try fuzzy directory matching
        self.find_matching_dir(model_id)
    }

    fn cache_dir(&self) -> PathBuf {
        self.cache.cache_dir().to_path_buf()
    }

    fn name(&self) -> &'static str {
        "sdk"
    }
}

/// Normalize a name by removing separators for comparison.
fn normalize_name(name: &str) -> String {
    name.replace(['-', '_', '.'], "")
}

/// Check if a directory contains model files.
fn has_model_files(path: &Path) -> bool {
    // A valid model directory has either:
    // - universal.xyb (SDK bundle)
    // - model_metadata.json (extracted model)
    // - model.onnx (raw model file)
    path.join("universal.xyb").exists()
        || path.join("model_metadata.json").exists()
        || path.join("model.onnx").exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_normalize_name() {
        assert_eq!(normalize_name("kokoro-82m"), "kokoro82m");
        assert_eq!(normalize_name("kokoro_82m"), "kokoro82m");
        assert_eq!(normalize_name("kokoro.82m.v1.0"), "kokoro82mv10");
    }

    #[test]
    fn test_sdk_cache_provider_creation() {
        let temp_dir = TempDir::new().unwrap();
        let provider = SdkCacheProvider::with_dir(temp_dir.path().to_path_buf()).unwrap();
        assert_eq!(provider.name(), "sdk");
        assert_eq!(provider.cache_dir(), temp_dir.path());
    }

    #[test]
    fn test_is_model_cached_empty() {
        let temp_dir = TempDir::new().unwrap();
        let provider = SdkCacheProvider::with_dir(temp_dir.path().to_path_buf()).unwrap();
        assert!(!provider.is_model_cached("nonexistent-model"));
    }

    #[test]
    fn test_get_model_path_empty() {
        let temp_dir = TempDir::new().unwrap();
        let provider = SdkCacheProvider::with_dir(temp_dir.path().to_path_buf()).unwrap();
        assert!(provider.get_model_path("nonexistent-model").is_none());
    }

    #[test]
    fn test_fuzzy_matching_with_directory() {
        let temp_dir = TempDir::new().unwrap();

        // Create a directory that simulates cached model
        let model_dir = temp_dir.path().join("Kokoro-82M-v1.0-ONNX");
        std::fs::create_dir_all(&model_dir).unwrap();

        // Add a bundle file to make it valid
        std::fs::write(model_dir.join("universal.xyb"), b"fake bundle").unwrap();

        let provider = SdkCacheProvider::with_dir(temp_dir.path().to_path_buf()).unwrap();

        // Should find via fuzzy match
        assert!(provider.is_model_cached("kokoro-82m"));

        let path = provider.get_model_path("kokoro-82m");
        assert!(path.is_some());
        assert!(path.unwrap().to_string_lossy().contains("Kokoro-82M"));
    }
}
