//! Cache Provider Abstraction
//!
//! Defines an abstract interface for model cache access. This allows the Core
//! to check model availability without depending on SDK-specific implementations.
//!
//! The SDK implements this trait with `RegistryClient`/`CacheManager`, while Core
//! provides a simple filesystem-based default for standalone use.
//!
//! ## Design Rationale
//!
//! - Core cannot depend on SDK (that would be circular)
//! - Cache management lives in SDK (with registry integration, TTL, etc.)
//! - Core needs to check if models are available locally for routing decisions
//! - Solution: Abstract trait in Core, concrete impl in SDK
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::CacheProvider;
//!
//! // SDK provides this at bootstrap time
//! let provider: Arc<dyn CacheProvider> = Arc::new(SdkCacheProvider::new());
//! let orchestrator = Orchestrator::with_cache_provider(provider);
//!
//! // Core uses it for routing decisions
//! if provider.is_model_cached("kokoro-82m") {
//!     route_to_device();
//! }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Abstract interface for checking model cache availability.
///
/// This trait allows the orchestrator to make routing decisions based on
/// model availability without coupling to specific cache implementations.
pub trait CacheProvider: Send + Sync {
    /// Check if a model is cached locally.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier (e.g., "kokoro-82m")
    ///
    /// # Returns
    ///
    /// `true` if the model is available locally (cached or in test_models)
    fn is_model_cached(&self, model_id: &str) -> bool;

    /// Get the local path for a cached model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier
    ///
    /// # Returns
    ///
    /// `Some(PathBuf)` if the model is cached, `None` otherwise
    fn get_model_path(&self, model_id: &str) -> Option<PathBuf>;

    /// Get the cache directory path.
    fn cache_dir(&self) -> PathBuf;

    /// Provider name for logging/debugging.
    fn name(&self) -> &'static str;
}

/// Default filesystem-based cache provider.
///
/// This implementation searches the standard cache locations:
/// - `~/.xybrid/cache/models/{model_dir}/` (SDK cache)
/// - `test_models/{model_id}/` (development)
///
/// It handles the mismatch between model masks (e.g., "kokoro-82m") and
/// HuggingFace repo names (e.g., "Kokoro-82M-v1.0-ONNX") via fuzzy matching.
#[derive(Debug, Clone)]
pub struct FilesystemCacheProvider {
    cache_dir: PathBuf,
    test_models_dir: PathBuf,
}

impl Default for FilesystemCacheProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl FilesystemCacheProvider {
    /// Create a new filesystem cache provider with default paths.
    pub fn new() -> Self {
        let cache_dir = dirs::home_dir()
            .map(|h| h.join(".xybrid").join("cache").join("models"))
            .unwrap_or_else(|| PathBuf::from(".xybrid/cache/models"));

        Self {
            cache_dir,
            test_models_dir: PathBuf::from("test_models"),
        }
    }

    /// Create with custom paths (for testing).
    pub fn with_paths(cache_dir: PathBuf, test_models_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            test_models_dir,
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

        // Check test_models first (exact match only for simplicity)
        let test_path = self.test_models_dir.join(model_id);
        if test_path.exists() && has_model_files(&test_path) {
            return Some(test_path);
        }

        // Search SDK cache with fuzzy matching
        if !self.cache_dir.exists() {
            return None;
        }

        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if !entry_path.is_dir() {
                    continue;
                }

                let dir_name = entry.file_name().to_string_lossy().to_lowercase();
                let dir_name_normalized = normalize_name(&dir_name);

                // Match if:
                // 1. Directory contains the model_id (e.g., "kokoro-82m" in "kokoro-82m-v1.0-onnx")
                // 2. Normalized names match (handles separators)
                let is_match = dir_name.contains(&model_id_lower)
                    || dir_name_normalized.contains(&model_id_normalized);

                if is_match && has_model_files(&entry_path) {
                    return Some(entry_path);
                }
            }
        }

        None
    }
}

impl CacheProvider for FilesystemCacheProvider {
    fn is_model_cached(&self, model_id: &str) -> bool {
        self.find_matching_dir(model_id).is_some()
    }

    fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
        self.find_matching_dir(model_id)
    }

    fn cache_dir(&self) -> PathBuf {
        self.cache_dir.clone()
    }

    fn name(&self) -> &'static str {
        "filesystem"
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
    path.join("universal.xyb").exists() || path.join("model_metadata.json").exists()
}

/// A no-op cache provider that always returns "not cached".
///
/// Useful for testing or when running in cloud-only mode.
#[derive(Debug, Clone, Default)]
pub struct NoopCacheProvider;

impl CacheProvider for NoopCacheProvider {
    fn is_model_cached(&self, _model_id: &str) -> bool {
        false
    }

    fn get_model_path(&self, _model_id: &str) -> Option<PathBuf> {
        None
    }

    fn cache_dir(&self) -> PathBuf {
        PathBuf::new()
    }

    fn name(&self) -> &'static str {
        "noop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_name() {
        // normalize_name just removes separators, doesn't lowercase
        // (lowercasing is done by callers before passing to normalize_name)
        assert_eq!(normalize_name("kokoro-82m"), "kokoro82m");
        assert_eq!(normalize_name("kokoro_82m"), "kokoro82m");
        assert_eq!(normalize_name("kokoro.82m.v1.0"), "kokoro82mv10");
    }

    #[test]
    fn test_noop_provider() {
        let provider = NoopCacheProvider;
        assert!(!provider.is_model_cached("any-model"));
        assert!(provider.get_model_path("any-model").is_none());
        assert_eq!(provider.name(), "noop");
    }

    #[test]
    fn test_filesystem_provider_creation() {
        let provider = FilesystemCacheProvider::new();
        assert!(provider.cache_dir().to_string_lossy().contains(".xybrid"));
        assert_eq!(provider.name(), "filesystem");
    }

    #[test]
    fn test_filesystem_provider_test_models() {
        // This test will only work if test_models exists
        let provider = FilesystemCacheProvider::new();

        // If test_models/kokoro-82m exists, it should be found
        let path = provider.get_model_path("kokoro-82m");
        if let Some(p) = path {
            assert!(p.exists());
        }
    }
}
