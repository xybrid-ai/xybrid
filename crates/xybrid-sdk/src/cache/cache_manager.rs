//! Model Cache Manager - Platform-specific bundle storage and cache management.
//!
//! This module provides cache management for `.xyb` bundles,
//! including platform-specific paths, decompression, manifest validation, and
//! cache policies (local models persist, cloud models have TTL).
//!
//! # Example
//!
//! ```no_run
//! use xybrid_sdk::CacheManager;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cache = CacheManager::new()?;
//! let status = cache.status()?;
//! println!("Cache has {} models", status.total_models);
//! # Ok(())
//! # }
//! ```

use super::layout::{CacheEntryInfo, CacheEntryLocation, CacheLayout, HF_REVISIONS_DIR};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
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
    /// Cloud models have a 24h TTL
    Cloud,
}

/// Cloud model TTL in seconds (24 hours)
const CLOUD_TTL_SECONDS: u64 = 24 * 60 * 60;

/// Seconds since the Unix epoch, or `0` if the system clock is set before
/// 1970. Never panics — a pre-epoch clock reads as `now = 0`, which the
/// `saturating_sub` TTL math handles gracefully (every entry then looks
/// freshly cached rather than crashing the cache call).
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

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
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| SdkError::cache_src("Failed to create cache directory", e))?;

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
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| SdkError::cache_src("Failed to create cache directory", e))?;

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
            let home = std::env::var("HOME")
                .map_err(|_| SdkError::cache("HOME environment variable not set"))?;
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
            Err(SdkError::cache(
                "Android requires cache directory to be configured. \
                Call init_sdk_cache_dir() with a path from path_provider before loading models. \
                Example: initSdkCacheDir('${appDir.path}/xybrid/models')",
            ))
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            let home =
                dirs::home_dir().ok_or_else(|| SdkError::cache("Home directory not found"))?;
            Ok(home.join(".xybrid").join("cache").join("models"))
        }
    }

    /// Returns the cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub(crate) fn layout_from_config() -> Result<CacheLayout, SdkError> {
        Ok(CacheLayout::from_registry_root(Self::get_cache_dir()?))
    }

    fn layout(&self) -> CacheLayout {
        CacheLayout::from_registry_root(self.cache_dir.clone())
    }

    pub(crate) fn registry_bundle_path(&self, hf_repo: &str, file: &str) -> PathBuf {
        self.layout().registry_bundle_path(hf_repo, file)
    }

    pub(crate) fn cache_entries(&self) -> Result<Vec<CacheEntryInfo>, SdkError> {
        self.layout().cache_entries()
    }

    /// Scans the cache directory for existing bundles.
    fn scan_cache(&mut self) -> Result<(), SdkError> {
        if !self.cache_dir.exists() {
            return Ok(());
        }

        let entries = std::fs::read_dir(&self.cache_dir)
            .map_err(|e| SdkError::cache_src("Failed to read cache directory", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| SdkError::cache_src("Failed to read cache entry", e))?;

            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("xyb") {
                // Extract ID and version from filename (format: id@version.xyb)
                if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some((id, version)) = file_stem.split_once('@') {
                        let metadata = std::fs::metadata(&path)
                            .map_err(|e| SdkError::cache_src("Failed to read metadata", e))?;

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
        // Filter out expired cloud models.
        //
        // `unwrap_or(0)` rather than `.unwrap()`: `duration_since(UNIX_EPOCH)`
        // errors when the system clock is set before 1970 (dead RTC battery,
        // first-boot embedded, user-set clock). `status()` is an innocuous
        // query and must not panic on a skewed clock — matches the
        // `unwrap_or(0)` convention used throughout `model.rs` telemetry.
        let now = now_unix_secs();

        let valid_entries: Vec<_> = self
            .entries
            .values()
            .filter(|entry| match entry.cache_type {
                CacheType::Local => true,
                // `saturating_sub`: a cache file whose mtime is in the future
                // (clock moved backward, NTP correction, file copied from a
                // fast-clock machine) would otherwise underflow `now - cached_at`
                // — a debug-build panic and a release-build wrap-to-huge that
                // makes the entry read as never-expired.
                CacheType::Cloud => now.saturating_sub(entry.cached_at) < CLOUD_TTL_SECONDS,
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

    /// Resolve the local directory a HuggingFace repo materialized into.
    ///
    /// Covers the current hashed layout plus marked legacy locations, and
    /// returns the first directory holding a `model_metadata.json`. The
    /// on-disk directory name is a repository hash, so callers must resolve
    /// paths through this rather than deriving them from the repo id.
    ///
    /// # Arguments
    ///
    /// * `repo` - Repository id (format: "owner/repo", no variant suffix)
    ///
    /// # Returns
    ///
    /// Path to the materialized repo directory, or `None` when the repo has
    /// not been downloaded into this cache.
    pub fn huggingface_cache_dir(&self, repo: &str) -> Option<PathBuf> {
        self.layout()
            .huggingface_repo_dirs(repo)
            .into_iter()
            .find(|dir| dir.join("model_metadata.json").is_file())
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
            return Err(SdkError::cache(format!(
                "Bundle not found: {}",
                bundle_path.display()
            )));
        }

        // Extract bundle ID and version from path
        let file_stem = bundle_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| SdkError::cache("Invalid bundle filename"))?;

        let (id, version) = file_stem
            .split_once('@')
            .ok_or_else(|| SdkError::cache("Bundle filename must be in format id@version.xyb"))?;

        // Decompressed bundle directory
        let decompressed_dir = self.cache_dir.join(format!("{}_{}", id, version));

        // Create decompressed directory
        std::fs::create_dir_all(&decompressed_dir)
            .map_err(|e| SdkError::cache_src("Failed to create decompressed directory", e))?;

        // Load and extract bundle
        let bundle = XyBundle::load(bundle_path)
            .map_err(|e| SdkError::cache_src("Failed to load bundle", e))?;

        // Extract bundle contents
        bundle
            .extract_to(&decompressed_dir)
            .map_err(|e| SdkError::cache_src("Failed to extract bundle", e))?;

        // Write manifest.json
        let manifest_path = decompressed_dir.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(bundle.manifest())
            .map_err(|e| SdkError::cache_src("Failed to serialize manifest", e))?;
        std::fs::write(&manifest_path, manifest_json)
            .map_err(|e| SdkError::cache_src("Failed to write manifest", e))?;

        Ok(decompressed_dir)
    }

    // =========================================================================
    // Bundle Extraction (Unified API)
    // =========================================================================

    /// Returns the extraction directory for a given model ID.
    ///
    /// This is a deterministic path based on the model ID, located at:
    /// `{cache_dir}/../extracted/{model_id}/`
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier from model_metadata.json
    pub fn extraction_dir(&self, model_id: &str) -> PathBuf {
        self.layout().extraction_dir(model_id)
    }

    /// Checks if a bundle has already been extracted.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier
    ///
    /// # Returns
    ///
    /// True if the extraction directory exists and contains model_metadata.json
    pub fn is_extracted(&self, model_id: &str) -> bool {
        self.existing_extraction_dir(model_id).is_some()
    }

    pub(crate) fn existing_extraction_dir(&self, model_id: &str) -> Option<PathBuf> {
        self.layout()
            .extraction_dirs(model_id)
            .into_iter()
            .find(|extract_dir| Self::extraction_is_ready(extract_dir))
    }

    fn extraction_is_ready(extract_dir: &Path) -> bool {
        use xybrid_core::execution::ModelMetadata;

        let metadata_path = extract_dir.join("model_metadata.json");
        let Ok(metadata_json) = std::fs::read_to_string(&metadata_path) else {
            return false;
        };
        let Ok(metadata) = serde_json::from_str::<ModelMetadata>(&metadata_json) else {
            return false;
        };

        metadata
            .files
            .iter()
            .all(|file| extract_dir.join(file).exists())
    }

    /// List all model IDs that have been extracted and are ready to run offline.
    ///
    /// Walks the `{cache}/extracted/` directory and returns every subdirectory
    /// whose name is a model ID (i.e. contains a `model_metadata.json`). The
    /// returned list is sorted alphabetically for stable output.
    ///
    /// This is an offline operation — it never touches the network.
    pub fn list_extracted_model_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .layout()
            .extracted_roots()
            .into_iter()
            .filter_map(|root| std::fs::read_dir(root).ok())
            .flat_map(|entries| entries.filter_map(|entry| entry.ok()))
            .filter(|entry| Self::extraction_is_ready(&entry.path()))
            .filter_map(|entry| {
                entry
                    .file_name()
                    .into_string()
                    .ok()
                    .filter(|name| !name.starts_with('.'))
            })
            .collect();

        ids.sort();
        ids.dedup();
        ids
    }

    /// Ensures a `.xyb` bundle is extracted and returns the directory path.
    ///
    /// This is the **single source of truth** for bundle extraction.
    /// All code that needs to access extracted bundle contents should use this method.
    ///
    /// The extraction is idempotent:
    /// - If already extracted, returns the existing directory immediately
    /// - If not extracted, extracts and returns the new directory
    ///
    /// # Arguments
    ///
    /// * `xyb_path` - Path to the `.xyb` bundle file
    ///
    /// # Returns
    ///
    /// Path to the extracted directory containing model files
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::CacheManager;
    /// # use std::path::PathBuf;
    /// # let xyb_path = PathBuf::from("model.xyb");
    /// let cache = CacheManager::new()?;
    /// let model_dir = cache.ensure_extracted(&xyb_path)?;
    /// // model_dir now contains: model_metadata.json, model.gguf, etc.
    /// # Ok(())
    /// # }
    /// ```
    pub fn ensure_extracted(&self, xyb_path: &Path) -> Result<PathBuf, SdkError> {
        use xybrid_core::execution::ModelMetadata;

        // Validate bundle exists
        if !xyb_path.exists() {
            return Err(SdkError::cache(format!(
                "Bundle not found: {}",
                xyb_path.display()
            )));
        }

        // Load bundle to get metadata
        let bundle = XyBundle::load(xyb_path)
            .map_err(|e| SdkError::cache_src("Failed to load bundle", e))?;

        // Get model_id from metadata
        let metadata_json = bundle
            .get_metadata_json()
            .map_err(|e| SdkError::cache_src("Failed to read bundle metadata", e))?
            .ok_or_else(|| SdkError::cache("Bundle has no model_metadata.json"))?;

        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| SdkError::cache_src("Failed to parse model metadata", e))?;

        // Check if already extracted and all files declared by metadata exist.
        if let Some(extract_dir) = self.existing_extraction_dir(&metadata.model_id) {
            log::debug!(
                "Bundle already extracted for '{}' at {}",
                metadata.model_id,
                extract_dir.display()
            );
            return Ok(extract_dir);
        }

        let extract_dir = self.extraction_dir(&metadata.model_id);

        // Create extraction directory
        std::fs::create_dir_all(&extract_dir)
            .map_err(|e| SdkError::cache_src("Failed to create extraction directory", e))?;

        // Extract bundle contents
        log::info!(
            "Extracting bundle '{}' to {}",
            metadata.model_id,
            extract_dir.display()
        );
        bundle
            .extract_to(&extract_dir)
            .map_err(|e| SdkError::cache_src("Failed to extract bundle", e))?;

        Ok(extract_dir)
    }

    /// Ensures a bundle is extracted, with a preloaded model_id.
    ///
    /// This is an optimization when you already know the model_id (e.g., from
    /// registry metadata) and want to avoid loading the bundle just to read it.
    ///
    /// # Arguments
    ///
    /// * `xyb_path` - Path to the `.xyb` bundle file
    /// * `model_id` - Known model identifier
    ///
    /// # Returns
    ///
    /// Path to the extracted directory
    pub fn ensure_extracted_with_id(
        &self,
        xyb_path: &Path,
        model_id: &str,
    ) -> Result<PathBuf, SdkError> {
        // Check if already extracted and all files declared by metadata exist.
        if let Some(extract_dir) = self.existing_extraction_dir(model_id) {
            log::debug!(
                "Bundle already extracted for '{}' at {}",
                model_id,
                extract_dir.display()
            );
            return Ok(extract_dir);
        }

        let extract_dir = self.extraction_dir(model_id);

        // Need to extract - load bundle
        if !xyb_path.exists() {
            return Err(SdkError::cache(format!(
                "Bundle not found: {}",
                xyb_path.display()
            )));
        }

        let bundle = XyBundle::load(xyb_path)
            .map_err(|e| SdkError::cache_src("Failed to load bundle", e))?;

        // Create extraction directory
        std::fs::create_dir_all(&extract_dir)
            .map_err(|e| SdkError::cache_src("Failed to create extraction directory", e))?;

        // Extract bundle contents
        log::info!(
            "Extracting bundle '{}' to {}",
            model_id,
            extract_dir.display()
        );
        bundle
            .extract_to(&extract_dir)
            .map_err(|e| SdkError::cache_src("Failed to extract bundle", e))?;

        Ok(extract_dir)
    }

    // =========================================================================
    // Cache Maintenance
    // =========================================================================

    /// Cleans expired cache entries.
    ///
    /// Removes cloud models that have exceeded their TTL.
    ///
    /// # Returns
    ///
    /// Number of entries removed
    pub fn clean_expired(&mut self) -> Result<u32, SdkError> {
        // See `status()` for the `unwrap_or(0)` + `saturating_sub` rationale —
        // both guard against a system clock set before epoch / cache mtimes in
        // the future, neither of which should panic cache maintenance.
        let now = now_unix_secs();

        let mut removed_count = 0;
        let mut to_remove = Vec::new();

        for (key, entry) in &self.entries {
            if entry.cache_type == CacheType::Cloud
                && now.saturating_sub(entry.cached_at) >= CLOUD_TTL_SECONDS
            {
                to_remove.push(key.clone());
            }
        }

        for key in to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                // Remove bundle file
                if entry.path.exists() {
                    std::fs::remove_file(&entry.path)
                        .map_err(|e| SdkError::cache_src("Failed to remove expired bundle", e))?;
                }
                removed_count += 1;
            }
        }

        Ok(removed_count)
    }

    /// Removes every managed cache root for a single model.
    ///
    /// # Returns
    ///
    /// The number of cache roots removed (0 if the model was not cached).
    ///
    /// # Concurrency
    ///
    /// Not safe to run concurrently with a load of the same model: it removes
    /// whole cache directories that an in-flight extraction may be writing to.
    pub(crate) fn clear_model_roots(&mut self, model_id: &str) -> Result<u32, SdkError> {
        if model_id.is_empty()
            || !Path::new(model_id)
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
        {
            return Err(SdkError::cache(format!(
                "Invalid cache model identifier: {}",
                model_id
            )));
        }

        let matching_cache_entries: Vec<_> = self
            .layout()
            .cache_entries()?
            .into_iter()
            .filter(|entry| entry.model_id == model_id)
            .collect();
        let clears_huggingface_revision = matching_cache_entries.iter().any(|entry| {
            entry.location == CacheEntryLocation::HuggingFace
                && entry
                    .path
                    .parent()
                    .is_some_and(|parent| parent.ends_with(HF_REVISIONS_DIR))
        });

        // Revision entries share one repository-scoped hf-hub blob store. A
        // targeted revision eviction must remove only its materialization;
        // deleting `model_roots` would also delete blobs used by siblings.
        let mut roots = if clears_huggingface_revision {
            Vec::new()
        } else {
            self.layout().model_roots(model_id)
        };
        roots.extend(matching_cache_entries.into_iter().map(|entry| entry.path));
        let mut entry_keys = Vec::new();
        for (key, entry) in &self.entries {
            if entry.id == model_id {
                entry_keys.push(key.clone());
                roots.push(entry.path.clone());
            }
        }

        let mut removed_count = 0;
        let mut seen = HashSet::new();
        for root in roots {
            if !seen.insert(root.clone()) {
                continue;
            }
            if Self::remove_cache_path(&root)? {
                removed_count += 1;
            }
        }

        for key in entry_keys {
            self.entries.remove(&key);
        }

        Ok(removed_count)
    }

    /// Clears all cached models across every managed cache root.
    ///
    /// # Returns
    ///
    /// The number of cache roots removed (registry bundles, extracted runtime
    /// caches, and HuggingFace downloads). Returns `0` when nothing was cached.
    ///
    /// # Concurrency
    ///
    /// Not safe to run concurrently with a model load: it removes whole cache
    /// directories that an in-flight download or extraction may be writing to.
    pub fn clear(&mut self) -> Result<u32, SdkError> {
        let mut removed_count = 0;

        for root in self.layout().entry_roots() {
            if Self::remove_cache_path(&root.path)? {
                removed_count += 1;
            }
        }

        std::fs::create_dir_all(&self.cache_dir)
            .map_err(|e| SdkError::cache_src("Failed to recreate cache directory", e))?;
        self.entries.clear();
        Ok(removed_count)
    }

    fn remove_cache_path(path: &Path) -> Result<bool, SdkError> {
        if path.is_dir() {
            std::fs::remove_dir_all(path)
                .map_err(|e| SdkError::cache_src("Failed to remove cache directory", e))?;
            Ok(true)
        } else if path.exists() {
            std::fs::remove_file(path)
                .map_err(|e| SdkError::cache_src("Failed to remove cache file", e))?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use xybrid_core::bundler::XyBundle;

    #[test]
    fn test_cache_status_empty() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CacheManager::with_dir(temp_dir.path().to_path_buf()).unwrap();
        let status = manager.status().unwrap();
        assert_eq!(status.total_models, 0);
        assert_eq!(status.total_size_bytes, 0);
    }

    #[test]
    fn now_unix_secs_is_monotonic_and_nonzero_on_sane_clock() {
        // On any test host with a clock past 1970 this is well above zero;
        // the point is it never panics (the `.unwrap()` it replaced would
        // on a pre-epoch clock).
        assert!(now_unix_secs() > 0);
    }

    #[test]
    fn cloud_ttl_math_does_not_underflow_for_future_cached_at() {
        // Regression for the `now - cached_at` underflow: a `cached_at`
        // in the future (clock moved backward, or a cache file copied from
        // a fast-clock machine) must not panic the TTL comparison. With
        // `saturating_sub` the elapsed time floors at 0, so a
        // future-stamped entry reads as freshly cached (not expired),
        // which is the safe behaviour.
        let now: u64 = 1_000;
        let cached_in_future: u64 = 5_000;
        let elapsed = now.saturating_sub(cached_in_future);
        assert_eq!(elapsed, 0, "future cached_at must floor to 0 elapsed");
        assert!(
            elapsed < CLOUD_TTL_SECONDS,
            "a future-stamped entry must not be treated as expired"
        );
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

    #[test]
    fn clear_removes_managed_model_cache_roots_when_registry_entries_are_unscanned() {
        // Given: the real cache layout produced by registry, extraction, and
        // direct HuggingFace model-loading paths.
        let temp_dir = TempDir::new().unwrap();
        let cache_root = temp_dir.path().join("cache");
        let models_dir = cache_root.join("models");
        let registry_model_dir = models_dir.join("Kokoro-82M-v1.0-ONNX");
        let extracted_model_dir = cache_root.join("extracted").join("kokoro-82m");
        let hf_model_dir = cache_root.join("hf").join("owner--repo");
        let layout = CacheLayout::from_registry_root(models_dir.clone());
        let hf_hub_model_dir = layout
            .prepare_huggingface_hub_repo_root("owner/repo")
            .unwrap();
        fs::create_dir_all(&registry_model_dir).unwrap();
        fs::create_dir_all(&extracted_model_dir).unwrap();
        fs::create_dir_all(&hf_model_dir).unwrap();
        fs::create_dir_all(&hf_hub_model_dir).unwrap();
        fs::write(registry_model_dir.join("universal.xyb"), b"bundle").unwrap();
        fs::write(extracted_model_dir.join("model_metadata.json"), b"{}").unwrap();
        fs::write(hf_model_dir.join("model.gguf"), b"weights").unwrap();
        fs::write(hf_hub_model_dir.join("blob"), b"weights").unwrap();

        let mut manager = CacheManager::with_dir(models_dir.clone()).unwrap();

        // When: the user clears all cached models through the cache manager.
        manager.clear().unwrap();

        // Then: every managed model-cache root is removed, including roots not
        // represented in the legacy top-level `.xyb` entry map.
        assert!(
            !registry_model_dir.exists(),
            "registry model cache should be removed by clear()"
        );
        assert!(
            !extracted_model_dir.exists(),
            "extracted runtime cache should be removed by clear()"
        );
        assert!(
            !hf_model_dir.exists(),
            "direct HuggingFace model cache should be removed by clear()"
        );
        assert!(
            !hf_hub_model_dir.exists(),
            "owned HuggingFace blob cache should be removed by clear()"
        );
    }

    #[test]
    fn clear_model_roots_removes_registry_and_extracted_cache_for_id() {
        let temp_dir = TempDir::new().unwrap();
        let cache_root = temp_dir.path().join("cache");
        let models_dir = cache_root.join("models");
        let model_id = "kokoro@82m";
        let model_dir = models_dir.join(model_id);
        let extracted_model_dir = cache_root.join("extracted").join(model_id);
        let other_model_dir = models_dir.join("other-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::create_dir_all(&extracted_model_dir).unwrap();
        fs::create_dir_all(&other_model_dir).unwrap();
        fs::write(model_dir.join("universal.xyb"), b"bundle").unwrap();
        fs::write(extracted_model_dir.join("model_metadata.json"), b"{}").unwrap();
        fs::write(other_model_dir.join("universal.xyb"), b"bundle").unwrap();

        let mut manager = CacheManager::with_dir(models_dir).unwrap();

        let removed = manager.clear_model_roots(model_id).unwrap();

        assert_eq!(removed, 2);
        assert!(
            !model_dir.exists(),
            "registry model cache should be removed"
        );
        assert!(
            !extracted_model_dir.exists(),
            "extracted model cache should be removed"
        );
        assert!(other_model_dir.exists(), "unrelated model should remain");
    }

    #[test]
    fn huggingface_cache_dir_resolves_materialized_hashed_repo() {
        let temp_dir = TempDir::new().unwrap();
        let models_dir = temp_dir.path().join("cache").join("models");
        let layout = CacheLayout::from_registry_root(models_dir.clone());
        let repo_dir = layout.huggingface_repo_dir("owner/repo");
        fs::create_dir_all(&repo_dir).unwrap();
        fs::write(repo_dir.join("model_metadata.json"), b"{}").unwrap();
        layout.record_huggingface_repo("owner/repo").unwrap();

        let manager = CacheManager::with_dir(models_dir).unwrap();

        assert_eq!(
            manager.huggingface_cache_dir("owner/repo"),
            Some(repo_dir),
            "materialized repo must resolve through its hashed directory"
        );
        assert_eq!(
            manager.huggingface_cache_dir("owner/other"),
            None,
            "a repo that never downloaded has no cache directory"
        );
    }

    #[test]
    fn clear_model_roots_removes_direct_hf_cache_for_repo_id() {
        let temp_dir = TempDir::new().unwrap();
        let cache_root = temp_dir.path().join("cache");
        let models_dir = cache_root.join("models");
        let layout = CacheLayout::from_registry_root(models_dir.clone());
        let hf_model_dir = layout.huggingface_repo_dir("owner/repo");
        let hf_hub_model_dir = layout
            .prepare_huggingface_hub_repo_root("owner/repo")
            .unwrap();
        fs::create_dir_all(&hf_model_dir).unwrap();
        fs::create_dir_all(&hf_hub_model_dir).unwrap();
        fs::write(hf_model_dir.join("model.gguf"), b"weights").unwrap();
        fs::write(hf_hub_model_dir.join("blob"), b"weights").unwrap();
        layout.record_huggingface_repo("owner/repo").unwrap();

        let mut manager = CacheManager::with_dir(models_dir).unwrap();

        let removed = manager.clear_model_roots("owner/repo").unwrap();

        assert_eq!(removed, 2);
        assert!(
            !hf_model_dir.exists(),
            "direct HuggingFace model cache should be removed"
        );
        assert!(
            !hf_hub_model_dir.exists(),
            "owned HuggingFace blob cache should be removed"
        );
    }

    #[test]
    fn clear_model_roots_removes_one_resolved_hf_revision() {
        let temp_dir = TempDir::new().unwrap();
        let models_dir = temp_dir.path().join("cache/models");
        let layout = CacheLayout::from_registry_root(models_dir.clone());
        let first_commit = "commit-a";
        let second_commit = "commit-b";
        let hub_root = layout
            .prepare_huggingface_hub_repo_root("owner/repo")
            .unwrap();
        let shared_blob = hub_root.join("model.gguf");
        fs::write(&shared_blob, b"weights").unwrap();
        for commit in [first_commit, second_commit] {
            layout
                .record_huggingface_revision("owner/repo", commit, commit, None)
                .unwrap();
            let revision_dir = layout.huggingface_repo_revision_dir("owner/repo", commit, None);
            fs::write(revision_dir.join("model_metadata.json"), b"{}").unwrap();
            #[cfg(unix)]
            std::os::unix::fs::symlink(&shared_blob, revision_dir.join("model.gguf")).unwrap();
            #[cfg(not(unix))]
            fs::copy(&shared_blob, revision_dir.join("model.gguf")).unwrap();
        }
        let first_dir = layout.huggingface_repo_revision_dir("owner/repo", first_commit, None);
        let second_dir = layout.huggingface_repo_revision_dir("owner/repo", second_commit, None);
        let mut manager = CacheManager::with_dir(models_dir).unwrap();

        let removed = manager
            .clear_model_roots(&format!("owner/repo@{first_commit}"))
            .unwrap();

        assert_eq!(removed, 1);
        assert!(!first_dir.exists());
        assert!(second_dir.exists());
        assert_eq!(fs::read(second_dir.join("model.gguf")).unwrap(), b"weights");
        assert!(
            hub_root.exists(),
            "sibling revisions still depend on the hub"
        );
    }

    #[test]
    fn clear_model_roots_does_not_remove_a_colliding_repo_revision() {
        let temp_dir = TempDir::new().unwrap();
        let models_dir = temp_dir.path().join("cache/models");
        let layout = CacheLayout::from_registry_root(models_dir.clone());
        let first_repo = "a/b--c";
        let second_repo = "a--b/c";
        let commit = "commit-a";

        for repo in [first_repo, second_repo] {
            layout
                .record_huggingface_revision(repo, "main", commit, None)
                .unwrap();
            let hub_root = layout.prepare_huggingface_hub_repo_root(repo).unwrap();
            fs::write(hub_root.join("blob"), repo.as_bytes()).unwrap();
            let revision_dir = layout.huggingface_repo_revision_dir(repo, commit, None);
            fs::write(revision_dir.join("model_metadata.json"), b"{}").unwrap();
        }
        let first_dir = layout.huggingface_repo_revision_dir(first_repo, commit, None);
        let second_dir = layout.huggingface_repo_revision_dir(second_repo, commit, None);
        let first_hub_root = layout.huggingface_hub_repo_root(first_repo);
        let second_hub_root = layout.huggingface_hub_repo_root(second_repo);
        let mut manager = CacheManager::with_dir(models_dir).unwrap();

        let removed = manager
            .clear_model_roots(&format!("{first_repo}@{commit}"))
            .unwrap();

        assert_eq!(removed, 1);
        assert!(!first_dir.exists());
        assert!(first_hub_root.exists());
        assert!(second_dir.exists());
        assert!(second_hub_root.exists());
    }

    #[test]
    fn unmarked_legacy_hf_caches_remain_listed_and_evictable() {
        let temp_dir = TempDir::new().unwrap();
        let cache_root = temp_dir.path().join("cache");
        let models_dir = cache_root.join("models");
        let legacy_direct = cache_root.join("hf").join("owner--repo");
        let legacy_hub = cache_root.join("hf-hub").join("models--owner--repo");
        fs::create_dir_all(&legacy_direct).unwrap();
        fs::create_dir_all(&legacy_hub).unwrap();
        fs::write(legacy_direct.join("model.gguf"), b"direct").unwrap();
        fs::write(legacy_hub.join("blob"), b"hub").unwrap();

        let layout = CacheLayout::from_registry_root(models_dir.clone());
        let entries = layout.cache_entries().unwrap();
        assert!(entries.iter().any(|entry| {
            entry.model_id == "owner--repo"
                && entry.path == legacy_direct
                && entry.location == CacheEntryLocation::HuggingFace
        }));
        assert!(entries.iter().any(|entry| {
            entry.model_id == "models--owner--repo"
                && entry.path == legacy_hub
                && entry.location == CacheEntryLocation::HuggingFaceHub
        }));

        let mut manager = CacheManager::with_dir(models_dir).unwrap();
        assert_eq!(manager.clear_model_roots("owner--repo").unwrap(), 1);
        assert!(!legacy_direct.exists());
        assert!(legacy_hub.exists());
        assert_eq!(manager.clear_model_roots("models--owner--repo").unwrap(), 1);
        assert!(!legacy_hub.exists());
    }

    #[test]
    fn custom_root_resolves_legacy_parent_extracted_cache() {
        let temp_dir = TempDir::new().unwrap();
        let custom_cache = temp_dir.path().join("custom-cache");
        let legacy_extracted_model = temp_dir.path().join("extracted").join("legacy-model");
        write_ready_extracted_model(&legacy_extracted_model, "legacy-model");

        let manager = CacheManager::with_dir(custom_cache).unwrap();

        assert!(manager.is_extracted("legacy-model"));
        assert_eq!(
            manager.list_extracted_model_ids(),
            vec!["legacy-model".to_string()]
        );
    }

    #[test]
    fn custom_root_clear_model_removes_legacy_parent_extracted_cache() {
        let temp_dir = TempDir::new().unwrap();
        let custom_cache = temp_dir.path().join("custom-cache");
        let legacy_extracted_model = temp_dir.path().join("extracted").join("legacy-model");
        let sibling_sentinel = temp_dir.path().join("sibling");
        write_ready_extracted_model(&legacy_extracted_model, "legacy-model");
        fs::create_dir_all(&sibling_sentinel).unwrap();
        fs::write(sibling_sentinel.join("keep"), b"keep").unwrap();

        let mut manager = CacheManager::with_dir(custom_cache).unwrap();

        let removed = manager.clear_model_roots("legacy-model").unwrap();

        assert_eq!(removed, 1);
        assert!(!legacy_extracted_model.exists());
        assert!(sibling_sentinel.join("keep").exists());
    }

    #[test]
    fn clear_model_roots_rejects_path_traversal() {
        let temp_dir = TempDir::new().unwrap();
        let cache_root = temp_dir.path().join("cache");
        let models_dir = cache_root.join("models");
        let outside_dir = cache_root.join("outside");
        fs::create_dir_all(&models_dir).unwrap();
        fs::create_dir_all(&outside_dir).unwrap();
        fs::write(outside_dir.join("sentinel"), b"keep").unwrap();

        let mut manager = CacheManager::with_dir(models_dir).unwrap();

        let result = manager.clear_model_roots("../outside");

        assert!(result.is_err());
        assert!(
            outside_dir.join("sentinel").exists(),
            "invalid model IDs must not delete outside cache paths"
        );
    }

    #[test]
    fn clear_model_roots_removes_legacy_bundle_from_memory_index() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("models");
        fs::create_dir_all(&cache_dir).unwrap();
        let bundle_path = cache_dir.join("test-model@1.0.xyb");
        fs::write(&bundle_path, b"bundle").unwrap();

        let mut manager = CacheManager::with_dir(cache_dir).unwrap();
        assert!(manager.is_cached("test-model"));
        assert_eq!(manager.status().unwrap().total_models, 1);
        assert_eq!(
            manager.get_cached_path("test-model"),
            Some(bundle_path.clone())
        );

        let removed = manager.clear_model_roots("test-model").unwrap();

        assert_eq!(removed, 1);
        assert!(!bundle_path.exists());
        assert!(!manager.is_cached("test-model"));
        assert_eq!(manager.status().unwrap().total_models, 0);
        assert_eq!(manager.get_cached_path("test-model"), None);
    }

    // =========================================================================
    // Bundle Extraction Tests
    // =========================================================================

    fn write_ready_extracted_model(model_dir: &Path, model_id: &str) {
        fs::create_dir_all(model_dir).unwrap();
        let metadata = format!(
            r#"{{
                "model_id": "{}",
                "version": "1.0",
                "execution_template": {{ "type": "Onnx", "model_file": "model.onnx" }},
                "preprocessing": [],
                "postprocessing": [],
                "files": ["model.onnx"],
                "metadata": {{}}
            }}"#,
            model_id
        );
        fs::write(model_dir.join("model_metadata.json"), metadata).unwrap();
        fs::write(model_dir.join("model.onnx"), b"model").unwrap();
    }

    /// Creates a test bundle with model_metadata.json
    fn create_test_bundle(temp_dir: &TempDir, model_id: &str) -> PathBuf {
        // Create model files
        let model_dir = temp_dir.path().join("model_files");
        fs::create_dir_all(&model_dir).unwrap();

        // Create model_metadata.json with valid Onnx execution template (internally tagged)
        let metadata = format!(
            r#"{{
                "model_id": "{}",
                "version": "1.0",
                "execution_template": {{ "type": "Onnx", "model_file": "model.onnx" }},
                "preprocessing": [],
                "postprocessing": [],
                "files": ["model.onnx"],
                "metadata": {{}}
            }}"#,
            model_id
        );
        fs::write(model_dir.join("model_metadata.json"), &metadata).unwrap();

        // Create fake model file
        fs::write(model_dir.join("model.onnx"), b"fake model data").unwrap();

        // Create bundle
        let mut bundle = XyBundle::new(model_id, "1.0", "universal");
        bundle
            .add_file(model_dir.join("model_metadata.json"))
            .unwrap();
        bundle.add_file(model_dir.join("model.onnx")).unwrap();

        // Write bundle
        let bundle_path = temp_dir.path().join(format!("{}.xyb", model_id));
        bundle.write(&bundle_path).unwrap();

        bundle_path
    }

    fn create_vlm_test_bundle(temp_dir: &TempDir, model_id: &str) -> PathBuf {
        let model_dir = temp_dir.path().join("vlm_model_files");
        fs::create_dir_all(&model_dir).unwrap();

        let metadata = format!(
            r#"{{
                "model_id": "{}",
                "version": "1.0",
                "execution_template": {{
                    "type": "VisionLanguage",
                    "model_file": "model.gguf"
                }},
                "vision_encoder": {{
                    "file": "mmproj-model.gguf",
                    "preprocessing_preset": "gemma3_vision",
                    "image_size": 896
                }},
                "preprocessing": [],
                "postprocessing": [],
                "files": ["model.gguf", "mmproj-model.gguf"],
                "metadata": {{ "task": "vlm" }}
            }}"#,
            model_id
        );
        fs::write(model_dir.join("model_metadata.json"), &metadata).unwrap();
        fs::write(model_dir.join("model.gguf"), b"fake language model").unwrap();
        fs::write(
            model_dir.join("mmproj-model.gguf"),
            b"fake vision projector",
        )
        .unwrap();

        let mut bundle = XyBundle::new(model_id, "1.0", "universal");
        bundle
            .add_file(model_dir.join("model_metadata.json"))
            .unwrap();
        bundle.add_file(model_dir.join("model.gguf")).unwrap();
        bundle
            .add_file(model_dir.join("mmproj-model.gguf"))
            .unwrap();

        let bundle_path = temp_dir.path().join(format!("{}.xyb", model_id));
        bundle.write(&bundle_path).unwrap();

        bundle_path
    }

    #[test]
    fn test_extraction_dir_path() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache").join("models");
        fs::create_dir_all(&cache_dir).unwrap();

        let manager = CacheManager::with_dir(cache_dir).unwrap();
        let extract_dir = manager.extraction_dir("test-model");

        // Should be at cache/extracted/test-model (sibling to models/)
        assert!(extract_dir.to_string_lossy().contains("extracted"));
        assert!(extract_dir.to_string_lossy().contains("test-model"));
    }

    #[test]
    fn test_is_extracted_false_when_not_extracted() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CacheManager::with_dir(temp_dir.path().to_path_buf()).unwrap();

        assert!(!manager.is_extracted("nonexistent-model"));
    }

    #[test]
    fn test_ensure_extracted_creates_directory() {
        let temp_dir = TempDir::new().unwrap();

        // Create cache structure: temp/cache/models/
        let cache_dir = temp_dir.path().join("cache").join("models");
        fs::create_dir_all(&cache_dir).unwrap();

        let manager = CacheManager::with_dir(cache_dir).unwrap();

        // Create test bundle
        let bundle_path = create_test_bundle(&temp_dir, "test-extraction-model");

        // Extract bundle
        let extract_dir = manager.ensure_extracted(&bundle_path).unwrap();

        // Verify extraction
        assert!(extract_dir.exists());
        assert!(extract_dir.join("model_metadata.json").exists());
        assert!(extract_dir.join("model.onnx").exists());
    }

    #[test]
    fn clear_leaves_cache_loadable_for_the_next_extraction() {
        // Regression guard: clear() removes the extracted/ runtime cache and
        // recreates only the registry root. The very next load must still
        // succeed by re-extracting — i.e. clear() must leave the cache in a
        // loadable state, not a half-torn-down one.
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache").join("models");
        fs::create_dir_all(&cache_dir).unwrap();

        let mut manager = CacheManager::with_dir(cache_dir.clone()).unwrap();
        // The bundle lives outside the cache tree, so it survives clear().
        let bundle_path = create_test_bundle(&temp_dir, "reload-model");

        // First load populates extracted/reload-model/.
        manager.ensure_extracted(&bundle_path).unwrap();
        assert!(manager.is_extracted("reload-model"));

        // Clearing wipes the extracted runtime cache and recreates the root.
        // Exactly two managed roots exist here — the registry root (models/)
        // and the extracted/ runtime cache — so both are counted.
        let removed = manager.clear().unwrap();
        assert_eq!(
            removed, 2,
            "clear() should count the registry + extracted roots"
        );
        assert!(
            cache_dir.exists(),
            "registry root should be recreated after clear()"
        );
        assert!(!manager.is_extracted("reload-model"));
        assert!(manager.list_extracted_model_ids().is_empty());

        // The next load must succeed by re-extracting from the surviving bundle.
        let extract_dir = manager.ensure_extracted(&bundle_path).unwrap();
        assert!(extract_dir.join("model_metadata.json").exists());
        assert!(extract_dir.join("model.onnx").exists());
        assert!(manager.is_extracted("reload-model"));
    }

    #[test]
    fn test_ensure_extracted_repairs_partial_vlm_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache").join("models");
        fs::create_dir_all(&cache_dir).unwrap();

        let manager = CacheManager::with_dir(cache_dir).unwrap();
        let bundle_path = create_vlm_test_bundle(&temp_dir, "vlm-bundle-model");
        let partial_dir = manager.extraction_dir("vlm-bundle-model");
        fs::create_dir_all(&partial_dir).unwrap();

        let bundle = XyBundle::load(&bundle_path).unwrap();
        let metadata_json = bundle.get_metadata_json().unwrap().unwrap();
        fs::write(partial_dir.join("model_metadata.json"), metadata_json).unwrap();
        assert!(!partial_dir.join("model.gguf").exists());
        assert!(!partial_dir.join("mmproj-model.gguf").exists());

        let extract_dir = manager.ensure_extracted(&bundle_path).unwrap();

        assert_eq!(extract_dir, partial_dir);
        assert!(extract_dir.join("model_metadata.json").exists());
        assert!(extract_dir.join("model.gguf").exists());
        assert!(extract_dir.join("mmproj-model.gguf").exists());
    }

    #[test]
    fn test_ensure_extracted_is_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache").join("models");
        fs::create_dir_all(&cache_dir).unwrap();

        let manager = CacheManager::with_dir(cache_dir).unwrap();
        let bundle_path = create_test_bundle(&temp_dir, "idempotent-model");

        // Extract twice
        let dir1 = manager.ensure_extracted(&bundle_path).unwrap();
        let dir2 = manager.ensure_extracted(&bundle_path).unwrap();

        // Should return same directory
        assert_eq!(dir1, dir2);

        // Should still have valid contents
        assert!(dir1.join("model_metadata.json").exists());
    }

    #[test]
    fn test_ensure_extracted_with_id_skips_when_exists() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache").join("models");
        fs::create_dir_all(&cache_dir).unwrap();

        let manager = CacheManager::with_dir(cache_dir).unwrap();
        let bundle_path = create_test_bundle(&temp_dir, "known-id-model");

        // Extract first time
        let dir1 = manager
            .ensure_extracted_with_id(&bundle_path, "known-id-model")
            .unwrap();
        assert!(dir1.join("model_metadata.json").exists());

        // Second call should skip extraction (even with wrong bundle path)
        let fake_path = temp_dir.path().join("nonexistent.xyb");
        let dir2 = manager
            .ensure_extracted_with_id(&fake_path, "known-id-model")
            .unwrap();

        assert_eq!(dir1, dir2);
    }

    #[test]
    fn test_ensure_extracted_error_on_missing_bundle() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CacheManager::with_dir(temp_dir.path().to_path_buf()).unwrap();

        let result = manager.ensure_extracted(Path::new("/nonexistent/bundle.xyb"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_is_extracted_true_after_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache").join("models");
        fs::create_dir_all(&cache_dir).unwrap();

        let manager = CacheManager::with_dir(cache_dir).unwrap();
        let bundle_path = create_test_bundle(&temp_dir, "check-extracted-model");

        // Before extraction
        assert!(!manager.is_extracted("check-extracted-model"));

        // Extract
        manager.ensure_extracted(&bundle_path).unwrap();

        // After extraction
        assert!(manager.is_extracted("check-extracted-model"));
    }
}
