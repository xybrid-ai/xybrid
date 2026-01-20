//! Model Cache Manager - Platform-specific bundle storage and cache management.
//!
//! This module provides cache management for `.xyb` bundles,
//! including platform-specific paths, decompression, manifest validation, and
//! cache policies (local models persist, cloud models have TTL).
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_sdk::CacheManager;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cache = CacheManager::new()?;
//! let status = cache.status()?;
//! println!("Cache has {} models", status.total_models);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use xybrid_core::bundler::XyBundle;

use crate::model::SdkError;

/// Cache status information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatus {
    /// Total number of cached models
    pub total_models: u32,
    /// Total cache size in bytes
    pub total_size_bytes: u64,
    /// Number of local models (persist indefinitely)
    pub local_models: u32,
    /// Number of cloud models (24h TTL)
    pub cloud_models: u32,
    /// Available models by ID
    pub available_models: Vec<String>,
}

/// Cache entry metadata.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Bundle ID
    id: String,
    /// Bundle version
    version: String,
    /// Cache type (local or cloud)
    cache_type: CacheType,
    /// Path to cached bundle
    path: PathBuf,
    /// Size in bytes
    size_bytes: u64,
    /// Timestamp when cached (for TTL calculation)
    cached_at: u64,
}

/// Cache type determines retention policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheType {
    /// Local models persist indefinitely
    Local,
    /// Cloud models have  weeks TTL
    Cloud,
}

/// Cloud model TTL in seconds (24 hours)
const CLOUD_TTL_SECONDS: u64 = 24 * 60 * 60;

/// Model Cache Manager.
///
/// Manages `.xyb` bundle storage with platform-specific paths and cache policies.
#[derive(Debug)]
pub struct CacheManager {
    /// Base cache directory
    cache_dir: PathBuf,
    /// Cache entries
    entries: HashMap<String, CacheEntry>,
}

impl CacheManager {
    /// Creates a new cache manager with platform-specific cache directory.
    ///
    /// # Platform Paths
    /// - iOS: `~/Library/Application Support/Xybrid/Models`
    /// - Android: Requires `init_sdk_cache_dir()` to be called first
    /// - Desktop: `~/.xybrid/cache/models`
    pub fn new() -> Result<Self, SdkError> {
        let cache_dir = Self::get_cache_dir()?;

        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            SdkError::CacheError(format!("Failed to create cache directory: {}", e))
        })?;

        let mut manager = Self {
            cache_dir,
            entries: HashMap::new(),
        };

        // Load existing cache entries
        manager.scan_cache()?;

        Ok(manager)
    }

    /// Creates a cache manager with a custom directory.
    pub fn with_dir(cache_dir: PathBuf) -> Result<Self, SdkError> {
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            SdkError::CacheError(format!("Failed to create cache directory: {}", e))
        })?;

        let mut manager = Self {
            cache_dir,
            entries: HashMap::new(),
        };

        manager.scan_cache()?;
        Ok(manager)
    }

    /// Gets the platform-specific cache directory.
    ///
    /// Priority:
    /// 1. Global SDK config (set via `init_sdk_cache_dir()`)
    /// 2. Platform-specific default (iOS/macOS/Linux/Windows)
    /// 3. On Android: REQUIRES SDK config - returns error if not set
    fn get_cache_dir() -> Result<PathBuf, SdkError> {
        // First, check if SDK config has a custom cache directory
        if let Some(cache_dir) = crate::get_sdk_cache_dir() {
            return Ok(cache_dir);
        }

        // Platform-specific defaults
        #[cfg(target_os = "ios")]
        {
            let home = std::env::var("HOME").map_err(|_| {
                SdkError::CacheError("HOME environment variable not set".to_string())
            })?;
            Ok(PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("Xybrid")
                .join("Models"))
        }

        #[cfg(target_os = "android")]
        {
            // Android apps cannot write to arbitrary paths - they MUST use
            // the app's sandbox directory provided by the platform.
            // The directory must be passed from Flutter using path_provider.
            Err(SdkError::CacheError(
                "Android requires cache directory to be configured. \
                Call init_sdk_cache_dir() with a path from path_provider before loading models. \
                Example: initSdkCacheDir('${appDir.path}/xybrid/models')"
                    .to_string(),
            ))
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            let home = dirs::home_dir()
                .ok_or_else(|| SdkError::CacheError("Home directory not found".to_string()))?;
            Ok(home.join(".xybrid").join("cache").join("models"))
        }
    }

    /// Returns the cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Scans the cache directory for existing bundles.
    fn scan_cache(&mut self) -> Result<(), SdkError> {
        if !self.cache_dir.exists() {
            return Ok(());
        }

        let entries = std::fs::read_dir(&self.cache_dir)
            .map_err(|e| SdkError::CacheError(format!("Failed to read cache directory: {}", e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| SdkError::CacheError(format!("Failed to read cache entry: {}", e)))?;

            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("xyb") {
                // Extract ID and version from filename (format: id@version.xyb)
                if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some((id, version)) = file_stem.split_once('@') {
                        let metadata = std::fs::metadata(&path).map_err(|e| {
                            SdkError::CacheError(format!("Failed to read metadata: {}", e))
                        })?;

                        let cached_at = metadata
                            .modified()
                            .or_else(|_| metadata.created())
                            .ok()
                            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0);

                        // Determine cache type from manifest (assume local for now)
                        let cache_type = CacheType::Local;

                        let cache_entry = CacheEntry {
                            id: id.to_string(),
                            version: version.to_string(),
                            cache_type,
                            path: path.clone(),
                            size_bytes: metadata.len(),
                            cached_at,
                        };

                        let key = format!("{}@{}", id, version);
                        self.entries.insert(key, cache_entry);
                    }
                }
            }
        }

        Ok(())
    }

    /// Gets cache status.
    ///
    /// Returns information about cached models, sizes, and availability.
    pub fn status(&self) -> Result<CacheStatus, SdkError> {
        // Filter out expired cloud models
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let valid_entries: Vec<_> = self
            .entries
            .values()
            .filter(|entry| match entry.cache_type {
                CacheType::Local => true,
                CacheType::Cloud => (now - entry.cached_at) < CLOUD_TTL_SECONDS,
            })
            .collect();

        let total_models = valid_entries.len() as u32;
        let total_size_bytes: u64 = valid_entries.iter().map(|e| e.size_bytes).sum();
        let local_models = valid_entries
            .iter()
            .filter(|e| e.cache_type == CacheType::Local)
            .count() as u32;
        let cloud_models = valid_entries
            .iter()
            .filter(|e| e.cache_type == CacheType::Cloud)
            .count() as u32;
        let available_models: Vec<String> = valid_entries
            .iter()
            .map(|e| format!("{}@{}", e.id, e.version))
            .collect();

        Ok(CacheStatus {
            total_models,
            total_size_bytes,
            local_models,
            cloud_models,
            available_models,
        })
    }

    /// Checks if a model is available in cache.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier (format: "id@version" or just "id")
    ///
    /// # Returns
    ///
    /// True if model is cached and available
    pub fn is_cached(&self, model_id: &str) -> bool {
        // Check exact match first
        if self.entries.contains_key(model_id) {
            return true;
        }

        // Check if any version of this model is cached
        self.entries
            .keys()
            .any(|key| key.starts_with(&format!("{}@", model_id)))
    }

    /// Gets the path to a cached bundle.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier (format: "id@version" or just "id")
    ///
    /// # Returns
    ///
    /// Path to cached bundle if available
    pub fn get_cached_path(&self, model_id: &str) -> Option<PathBuf> {
        // Try exact match first
        if let Some(entry) = self.entries.get(model_id) {
            return Some(entry.path.clone());
        }

        // Try to find latest version
        let prefix = format!("{}@", model_id);
        self.entries
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix))
            .max_by_key(|(key, _)| *key)
            .map(|(_, entry)| entry.path.clone())
    }

    /// Decompresses and validates a `.xyb` bundle.
    ///
    /// # Arguments
    ///
    /// * `bundle_path` - Path to the `.xyb` bundle file
    ///
    /// # Returns
    ///
    /// Path to decompressed bundle directory
    pub fn decompress_bundle(&self, bundle_path: &Path) -> Result<PathBuf, SdkError> {
        // Validate bundle exists
        if !bundle_path.exists() {
            return Err(SdkError::CacheError(format!(
                "Bundle not found: {}",
                bundle_path.display()
            )));
        }

        // Extract bundle ID and version from path
        let file_stem = bundle_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| SdkError::CacheError("Invalid bundle filename".to_string()))?;

        let (id, version) = file_stem.split_once('@').ok_or_else(|| {
            SdkError::CacheError("Bundle filename must be in format id@version.xyb".to_string())
        })?;

        // Decompressed bundle directory
        let decompressed_dir = self.cache_dir.join(format!("{}_{}", id, version));

        // Create decompressed directory
        std::fs::create_dir_all(&decompressed_dir).map_err(|e| {
            SdkError::CacheError(format!("Failed to create decompressed directory: {}", e))
        })?;

        // Load and extract bundle
        let bundle = XyBundle::load(bundle_path)
            .map_err(|e| SdkError::CacheError(format!("Failed to load bundle: {}", e)))?;

        // Extract bundle contents
        bundle
            .extract_to(&decompressed_dir)
            .map_err(|e| SdkError::CacheError(format!("Failed to extract bundle: {}", e)))?;

        // Write manifest.json
        let manifest_path = decompressed_dir.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(bundle.manifest())
            .map_err(|e| SdkError::CacheError(format!("Failed to serialize manifest: {}", e)))?;
        std::fs::write(&manifest_path, manifest_json)
            .map_err(|e| SdkError::CacheError(format!("Failed to write manifest: {}", e)))?;

        Ok(decompressed_dir)
    }

    /// Cleans expired cache entries.
    ///
    /// Removes cloud models that have exceeded their TTL.
    ///
    /// # Returns
    ///
    /// Number of entries removed
    pub fn clean_expired(&mut self) -> Result<u32, SdkError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut removed_count = 0;
        let mut to_remove = Vec::new();

        for (key, entry) in &self.entries {
            if entry.cache_type == CacheType::Cloud {
                if (now - entry.cached_at) >= CLOUD_TTL_SECONDS {
                    to_remove.push(key.clone());
                }
            }
        }

        for key in to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                // Remove bundle file
                if entry.path.exists() {
                    std::fs::remove_file(&entry.path).map_err(|e| {
                        SdkError::CacheError(format!("Failed to remove expired bundle: {}", e))
                    })?;
                }
                removed_count += 1;
            }
        }

        Ok(removed_count)
    }

    /// Clears all cached models.
    ///
    /// # Returns
    ///
    /// Number of entries removed
    pub fn clear(&mut self) -> Result<u32, SdkError> {
        let count = self.entries.len() as u32;

        for entry in self.entries.values() {
            if entry.path.exists() {
                let _ = std::fs::remove_file(&entry.path);
            }
        }

        self.entries.clear();
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_status_empty() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CacheManager::with_dir(temp_dir.path().to_path_buf()).unwrap();
        let status = manager.status().unwrap();
        assert_eq!(status.total_models, 0);
        assert_eq!(status.total_size_bytes, 0);
    }

    #[test]
    fn test_is_cached_empty() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CacheManager::with_dir(temp_dir.path().to_path_buf()).unwrap();
        assert!(!manager.is_cached("test-model"));
    }

    #[test]
    fn test_get_cached_path_empty() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CacheManager::with_dir(temp_dir.path().to_path_buf()).unwrap();
        assert!(manager.get_cached_path("test-model").is_none());
    }
}
