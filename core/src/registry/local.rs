//! Local Registry module - Storage and retrieval of bundles from filesystem.
//!
//! The LocalRegistry provides a unified interface for storing and retrieving bundles
//! from the local filesystem. This is the primary storage backend used by both
//! standalone local registries and as a cache for remote registries.

use std::fs;
use std::path::{Path, PathBuf};

/// Error type for registry operations.
#[derive(Debug, Clone)]
pub enum RegistryError {
    BundleNotFound(String),
    InvalidPath(String),
    IOError(String),
    InvalidBundle(String),
    RemoteError(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::BundleNotFound(id) => {
                write!(f, "Bundle not found: {}", id)
            }
            RegistryError::InvalidPath(path) => {
                write!(f, "Invalid path: {}", path)
            }
            RegistryError::IOError(msg) => {
                write!(f, "IO error: {}", msg)
            }
            RegistryError::InvalidBundle(msg) => {
                write!(f, "Invalid bundle: {}", msg)
            }
            RegistryError::RemoteError(msg) => {
                write!(f, "Remote registry error: {}", msg)
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// Result type for registry operations.
pub type RegistryResult<T> = Result<T, RegistryError>;

/// Metadata about a bundle stored in the registry.
#[derive(Debug, Clone)]
pub struct BundleMetadata {
    /// Bundle identifier/name
    pub id: String,
    /// Bundle version
    pub version: String,
    /// Path to the bundle (implementation-specific)
    pub path: String,
    /// Bundle size in bytes
    pub size_bytes: u64,
}

/// Registry trait for storing and retrieving bundles.
///
/// Bundles can represent:
/// - Policy bundles (rules, configurations)
/// - Model bundles (model files, weights)
/// - Configuration bundles
pub trait Registry: Send + Sync {
    /// Store a bundle in the registry.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the bundle
    /// * `version` - Version string for the bundle
    /// * `bundle_data` - Bundle contents as bytes
    ///
    /// # Returns
    ///
    /// Metadata about the stored bundle
    fn store_bundle(
        &mut self,
        id: &str,
        version: &str,
        bundle_data: Vec<u8>,
    ) -> RegistryResult<BundleMetadata>;

    /// Retrieve a bundle from the registry.
    ///
    /// # Arguments
    ///
    /// * `id` - Bundle identifier
    /// * `version` - Optional version (if None, retrieves latest)
    ///
    /// # Returns
    ///
    /// Bundle contents as bytes
    fn get_bundle(&self, id: &str, version: Option<&str>) -> RegistryResult<Vec<u8>>;

    /// Get metadata for a bundle without loading its contents.
    ///
    /// # Arguments
    ///
    /// * `id` - Bundle identifier
    /// * `version` - Optional version (if None, gets latest)
    ///
    /// # Returns
    ///
    /// Bundle metadata
    fn get_metadata(&self, id: &str, version: Option<&str>) -> RegistryResult<BundleMetadata>;

    /// List all bundles available in the registry.
    ///
    /// # Returns
    ///
    /// Vector of bundle metadata
    fn list_bundles(&self) -> RegistryResult<Vec<BundleMetadata>>;

    /// Remove a bundle from the registry.
    ///
    /// # Arguments
    ///
    /// * `id` - Bundle identifier
    /// * `version` - Optional version (if None, removes all versions)
    ///
    /// # Returns
    ///
    /// Success indicator
    fn remove_bundle(&mut self, id: &str, version: Option<&str>) -> RegistryResult<()>;
}

/// Local file system registry implementation.
///
/// Stores bundles in a directory structure:
/// ```text
/// <base_path>/
///   <id>/
///     <version>/
///       <platform>/
///         <id>.xyb
/// ```
///
/// For MVP, this uses simple file paths. Future versions may support
/// signed bundles, compression, and metadata files.
pub struct LocalRegistry {
    base_path: PathBuf,
}

impl LocalRegistry {
    /// Create a new LocalRegistry with the given base path.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Directory path where bundles will be stored
    ///
    /// # Returns
    ///
    /// LocalRegistry instance or error if path is invalid
    pub fn new<P: AsRef<Path>>(base_path: P) -> RegistryResult<Self> {
        let path = base_path.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        if !path.exists() {
            fs::create_dir_all(&path).map_err(|e| {
                RegistryError::IOError(format!("Failed to create registry directory: {}", e))
            })?;
        }

        if !path.is_dir() {
            return Err(RegistryError::InvalidPath(format!(
                "Path is not a directory: {}",
                path.display()
            )));
        }

        Ok(Self { base_path: path })
    }

    /// Create a new LocalRegistry in a default location.
    ///
    /// Uses `~/.xybrid/registry` on Unix systems or `%APPDATA%/xybrid/registry` on Windows.
    pub fn default() -> RegistryResult<Self> {
        let mut path = dirs::home_dir().ok_or_else(|| {
            RegistryError::InvalidPath("Could not determine home directory".to_string())
        })?;

        path.push(".xybrid");
        path.push("registry");

        Self::new(path)
    }

    /// Get the file path for a bundle using current platform.
    /// Uses the new multi-target storage structure: {id}/{version}/{platform}/{id}.xyb
    fn bundle_path(&self, id: &str, version: &str) -> PathBuf {
        self.bundle_path_with_target(id, version, &Self::current_platform())
    }

    /// Get the file path for a bundle with a specific target.
    /// Storage structure: {base_path}/{id}/{version}/{target}/{id}.xyb
    fn bundle_path_with_target(&self, id: &str, version: &str, target: &str) -> PathBuf {
        let mut path = self.base_path.clone();
        path.push(id);
        path.push(version);
        path.push(target);
        path.push(format!("{}.xyb", id));
        path
    }

    /// Detect the current platform target string
    fn current_platform() -> String {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        return "macos-arm64".to_string();

        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        return "macos-x86_64".to_string();

        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        return "linux-x86_64".to_string();

        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        return "linux-arm64".to_string();

        #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
        return "ios-aarch64".to_string();

        #[cfg(all(target_os = "android", target_arch = "aarch64"))]
        return "android-arm64".to_string();

        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "macos", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
            all(target_os = "ios", target_arch = "aarch64"),
            all(target_os = "android", target_arch = "aarch64"),
        )))]
        return "unknown".to_string();
    }

    /// Get the version directory path for a bundle.
    fn version_dir(&self, id: &str, version: &str) -> PathBuf {
        let mut path = self.base_path.clone();
        path.push(id);
        path.push(version);
        path
    }

    /// Get the directory path for a bundle ID.
    fn bundle_dir(&self, id: &str) -> PathBuf {
        let mut path = self.base_path.clone();
        path.push(id);
        path
    }

    /// Find the latest version of a bundle.
    /// Supports both new structure ({id}/{version}/{target}.xyb) and legacy ({id}/{version}.bundle)
    fn find_latest_version(&self, id: &str) -> RegistryResult<String> {
        let bundle_dir = self.bundle_dir(id);

        if !bundle_dir.exists() {
            return Err(RegistryError::BundleNotFound(id.to_string()));
        }

        let entries = fs::read_dir(&bundle_dir).map_err(|e| {
            RegistryError::IOError(format!("Failed to read bundle directory: {}", e))
        })?;

        let mut versions = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| {
                RegistryError::IOError(format!("Failed to read directory entry: {}", e))
            })?;
            let path = entry.path();

            if path.is_dir() {
                // New structure: version directories containing platform subdirs with .xyb files
                if let Some(version) = path.file_name().and_then(|n| n.to_str()) {
                    // Check if this directory contains platform subdirs with .xyb files
                    if let Ok(platform_entries) = fs::read_dir(&path) {
                        'find_version: for platform_entry in platform_entries.flatten() {
                            let platform_path = platform_entry.path();
                            if platform_path.is_dir() {
                                // Look for .xyb files inside platform directory
                                if let Ok(bundle_entries) = fs::read_dir(&platform_path) {
                                    for bundle_entry in bundle_entries.flatten() {
                                        if bundle_entry
                                            .path()
                                            .extension()
                                            .map_or(false, |ext| ext == "xyb")
                                        {
                                            versions.push(version.to_string());
                                            break 'find_version;
                                        }
                                    }
                                }
                            } else if platform_path.extension().map_or(false, |ext| ext == "xyb") {
                                // Legacy: .xyb directly in version dir
                                versions.push(version.to_string());
                                break 'find_version;
                            }
                        }
                    }
                }
            } else if path.is_file() {
                // Legacy structure: {version}.bundle or {version}.xyb files
                if let Some(extension) = path.extension() {
                    if extension == "bundle" || extension == "xyb" {
                        if let Some(stem) = path.file_stem() {
                            if let Some(version) = stem.to_str() {
                                versions.push(version.to_string());
                            }
                        }
                    }
                }
            }
        }

        if versions.is_empty() {
            return Err(RegistryError::BundleNotFound(id.to_string()));
        }

        // Simple version comparison: sort and take latest
        // For MVP, this is a simple string sort (works for semantic versions)
        versions.sort();
        versions.dedup(); // Remove duplicates from both structures
        Ok(versions.last().unwrap().clone())
    }

    /// Find all available targets for a specific model version.
    pub fn find_targets(&self, id: &str, version: &str) -> RegistryResult<Vec<String>> {
        let version_dir = self.version_dir(id, version);

        if !version_dir.exists() {
            // Try legacy path
            let legacy_path = self.bundle_dir(id).join(format!("{}.bundle", version));
            if legacy_path.exists() {
                return Ok(vec!["onnx".to_string()]); // Legacy bundles treated as onnx
            }
            return Err(RegistryError::BundleNotFound(format!("{}@{}", id, version)));
        }

        let entries = fs::read_dir(&version_dir).map_err(|e| {
            RegistryError::IOError(format!("Failed to read version directory: {}", e))
        })?;

        let mut targets = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| {
                RegistryError::IOError(format!("Failed to read directory entry: {}", e))
            })?;
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "xyb" {
                        if let Some(stem) = path.file_stem() {
                            if let Some(target) = stem.to_str() {
                                targets.push(target.to_string());
                            }
                        }
                    }
                }
            }
        }

        if targets.is_empty() {
            return Err(RegistryError::BundleNotFound(format!("{}@{}", id, version)));
        }

        targets.sort();
        Ok(targets)
    }

    /// Migrate legacy .bundle files to new .xyb structure.
    /// Moves {id}/{version}.bundle to {id}/{version}/onnx.xyb
    pub fn migrate_legacy_bundles(&self) -> RegistryResult<usize> {
        if !self.base_path.exists() {
            return Ok(0);
        }

        let mut migrated = 0;

        // Iterate through model directories
        let model_entries = fs::read_dir(&self.base_path).map_err(|e| {
            RegistryError::IOError(format!("Failed to read registry directory: {}", e))
        })?;

        for model_entry in model_entries {
            let model_entry = model_entry.map_err(|e| {
                RegistryError::IOError(format!("Failed to read model entry: {}", e))
            })?;
            let model_path = model_entry.path();

            if !model_path.is_dir() {
                continue;
            }

            // Look for .bundle files in model directory
            let version_entries = fs::read_dir(&model_path).map_err(|e| {
                RegistryError::IOError(format!("Failed to read model directory: {}", e))
            })?;

            for version_entry in version_entries {
                let version_entry = version_entry.map_err(|e| {
                    RegistryError::IOError(format!("Failed to read version entry: {}", e))
                })?;
                let version_path = version_entry.path();

                if version_path.is_file() {
                    if let Some(extension) = version_path.extension() {
                        if extension == "bundle" {
                            // Migrate this file
                            if let Some(stem) = version_path.file_stem() {
                                if let Some(version) = stem.to_str() {
                                    let model_id = model_path
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("unknown");

                                    // Create new version directory
                                    let new_version_dir = self.version_dir(model_id, version);
                                    fs::create_dir_all(&new_version_dir).map_err(|e| {
                                        RegistryError::IOError(format!(
                                            "Failed to create version directory: {}",
                                            e
                                        ))
                                    })?;

                                    // Move file to new location
                                    let new_path = new_version_dir.join("onnx.xyb");
                                    fs::rename(&version_path, &new_path).map_err(|e| {
                                        RegistryError::IOError(format!(
                                            "Failed to migrate bundle: {}",
                                            e
                                        ))
                                    })?;

                                    migrated += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(migrated)
    }

    /// Store a bundle with a specific target.
    /// Storage structure: {base_path}/{id}/{version}/{target}/{id}.xyb
    pub fn store_bundle_with_target(
        &mut self,
        id: &str,
        version: &str,
        target: &str,
        bundle_data: Vec<u8>,
    ) -> RegistryResult<BundleMetadata> {
        let bundle_path = self.bundle_path_with_target(id, version, target);

        // Create all parent directories (including target directory)
        if let Some(parent) = bundle_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                RegistryError::IOError(format!("Failed to create bundle directory: {}", e))
            })?;
        }

        // Write bundle data to file
        fs::write(&bundle_path, &bundle_data)
            .map_err(|e| RegistryError::IOError(format!("Failed to write bundle: {}", e)))?;

        let size_bytes = bundle_data.len() as u64;

        Ok(BundleMetadata {
            id: id.to_string(),
            version: version.to_string(),
            path: bundle_path.to_string_lossy().to_string(),
            size_bytes,
        })
    }

    /// Get a bundle with a specific target.
    pub fn get_bundle_with_target(
        &self,
        id: &str,
        version: &str,
        target: &str,
    ) -> RegistryResult<Vec<u8>> {
        let bundle_path = self.bundle_path_with_target(id, version, target);

        if bundle_path.exists() {
            return fs::read(&bundle_path)
                .map_err(|e| RegistryError::IOError(format!("Failed to read bundle: {}", e)));
        }

        // Try legacy path for backward compatibility
        let legacy_path = self.bundle_dir(id).join(format!("{}.bundle", version));
        if legacy_path.exists() && target == "onnx" {
            return fs::read(&legacy_path)
                .map_err(|e| RegistryError::IOError(format!("Failed to read bundle: {}", e)));
        }

        Err(RegistryError::BundleNotFound(format!(
            "{}@{}:{}",
            id, version, target
        )))
    }
}

impl Registry for LocalRegistry {
    fn store_bundle(
        &mut self,
        id: &str,
        version: &str,
        bundle_data: Vec<u8>,
    ) -> RegistryResult<BundleMetadata> {
        // Use current platform as target
        self.store_bundle_with_target(id, version, &Self::current_platform(), bundle_data)
    }

    fn get_bundle(&self, id: &str, version: Option<&str>) -> RegistryResult<Vec<u8>> {
        let version = match version {
            Some(v) => v.to_string(),
            None => self.find_latest_version(id)?,
        };

        // Try new structure first: {id}/{version}/{target}.xyb
        let bundle_path = self.bundle_path(id, &version);
        if bundle_path.exists() {
            return fs::read(&bundle_path)
                .map_err(|e| RegistryError::IOError(format!("Failed to read bundle: {}", e)));
        }

        // Try legacy structure: {id}/{version}.bundle
        let legacy_path = self.bundle_dir(id).join(format!("{}.bundle", version));
        if legacy_path.exists() {
            return fs::read(&legacy_path)
                .map_err(|e| RegistryError::IOError(format!("Failed to read bundle: {}", e)));
        }

        // Try legacy .xyb (flat): {id}/{version}.xyb
        let legacy_xyb_path = self.bundle_dir(id).join(format!("{}.xyb", version));
        if legacy_xyb_path.exists() {
            return fs::read(&legacy_xyb_path)
                .map_err(|e| RegistryError::IOError(format!("Failed to read bundle: {}", e)));
        }

        Err(RegistryError::BundleNotFound(format!("{}@{}", id, version)))
    }

    fn get_metadata(&self, id: &str, version: Option<&str>) -> RegistryResult<BundleMetadata> {
        let version = match version {
            Some(v) => v.to_string(),
            None => self.find_latest_version(id)?,
        };

        // Try new structure first: {id}/{version}/{target}.xyb
        let bundle_path = self.bundle_path(id, &version);
        if bundle_path.exists() {
            let metadata = fs::metadata(&bundle_path).map_err(|e| {
                RegistryError::IOError(format!("Failed to read bundle metadata: {}", e))
            })?;
            return Ok(BundleMetadata {
                id: id.to_string(),
                version: version.clone(),
                path: bundle_path.to_string_lossy().to_string(),
                size_bytes: metadata.len(),
            });
        }

        // Try legacy structure: {id}/{version}.bundle
        let legacy_path = self.bundle_dir(id).join(format!("{}.bundle", version));
        if legacy_path.exists() {
            let metadata = fs::metadata(&legacy_path).map_err(|e| {
                RegistryError::IOError(format!("Failed to read bundle metadata: {}", e))
            })?;
            return Ok(BundleMetadata {
                id: id.to_string(),
                version: version.clone(),
                path: legacy_path.to_string_lossy().to_string(),
                size_bytes: metadata.len(),
            });
        }

        // Try legacy .xyb (flat): {id}/{version}.xyb
        let legacy_xyb_path = self.bundle_dir(id).join(format!("{}.xyb", version));
        if legacy_xyb_path.exists() {
            let metadata = fs::metadata(&legacy_xyb_path).map_err(|e| {
                RegistryError::IOError(format!("Failed to read bundle metadata: {}", e))
            })?;
            return Ok(BundleMetadata {
                id: id.to_string(),
                version: version.clone(),
                path: legacy_xyb_path.to_string_lossy().to_string(),
                size_bytes: metadata.len(),
            });
        }

        Err(RegistryError::BundleNotFound(format!("{}@{}", id, version)))
    }

    fn list_bundles(&self) -> RegistryResult<Vec<BundleMetadata>> {
        if !self.base_path.exists() {
            return Ok(Vec::new());
        }

        let entries = fs::read_dir(&self.base_path).map_err(|e| {
            RegistryError::IOError(format!("Failed to read registry directory: {}", e))
        })?;

        let mut bundles = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| {
                RegistryError::IOError(format!("Failed to read directory entry: {}", e))
            })?;
            let model_path = entry.path();

            if model_path.is_dir() {
                if let Some(id) = model_path.file_name().and_then(|n| n.to_str()) {
                    // Read contents of model directory
                    let model_entries = fs::read_dir(&model_path).map_err(|e| {
                        RegistryError::IOError(format!("Failed to read model directory: {}", e))
                    })?;

                    for version_entry in model_entries {
                        let version_entry = version_entry.map_err(|e| {
                            RegistryError::IOError(format!("Failed to read version entry: {}", e))
                        })?;
                        let version_path = version_entry.path();

                        if version_path.is_dir() {
                            // New structure: version directory containing platform subdirs with .xyb files
                            if let Some(version) = version_path.file_name().and_then(|n| n.to_str())
                            {
                                // Check for platform subdirectories containing .xyb files
                                if let Ok(platform_entries) = fs::read_dir(&version_path) {
                                    'platform_loop: for platform_entry in platform_entries.flatten()
                                    {
                                        let platform_path = platform_entry.path();
                                        if platform_path.is_dir() {
                                            // Look for .xyb files inside platform directory
                                            if let Ok(bundle_entries) =
                                                fs::read_dir(&platform_path)
                                            {
                                                for bundle_entry in bundle_entries.flatten() {
                                                    let bundle_path = bundle_entry.path();
                                                    if bundle_path
                                                        .extension()
                                                        .map_or(false, |ext| ext == "xyb")
                                                    {
                                                        if let Ok(metadata) =
                                                            self.get_metadata(id, Some(version))
                                                        {
                                                            bundles.push(metadata);
                                                            break 'platform_loop;
                                                            // Only add once per version
                                                        }
                                                    }
                                                }
                                            }
                                        } else if platform_path
                                            .extension()
                                            .map_or(false, |ext| ext == "xyb")
                                        {
                                            // Legacy: .xyb directly in version dir
                                            if let Ok(metadata) =
                                                self.get_metadata(id, Some(version))
                                            {
                                                bundles.push(metadata);
                                                break 'platform_loop;
                                            }
                                        }
                                    }
                                }
                            }
                        } else if version_path.is_file() {
                            // Legacy structure: {version}.bundle or {version}.xyb
                            if let Some(extension) = version_path.extension() {
                                if extension == "bundle" || extension == "xyb" {
                                    if let Some(version_stem) =
                                        version_path.file_stem().and_then(|s| s.to_str())
                                    {
                                        if let Ok(metadata) =
                                            self.get_metadata(id, Some(version_stem))
                                        {
                                            bundles.push(metadata);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(bundles)
    }

    fn remove_bundle(&mut self, id: &str, version: Option<&str>) -> RegistryResult<()> {
        match version {
            Some(v) => {
                // Remove specific version - try new structure first, then legacy
                let version_dir = self.version_dir(id, v);
                if version_dir.exists() {
                    fs::remove_dir_all(&version_dir).map_err(|e| {
                        RegistryError::IOError(format!("Failed to remove version directory: {}", e))
                    })?;
                    return Ok(());
                }

                // Try legacy .bundle
                let legacy_path = self.bundle_dir(id).join(format!("{}.bundle", v));
                if legacy_path.exists() {
                    fs::remove_file(&legacy_path).map_err(|e| {
                        RegistryError::IOError(format!("Failed to remove bundle: {}", e))
                    })?;
                    return Ok(());
                }

                // Try legacy .xyb (flat)
                let legacy_xyb_path = self.bundle_dir(id).join(format!("{}.xyb", v));
                if legacy_xyb_path.exists() {
                    fs::remove_file(&legacy_xyb_path).map_err(|e| {
                        RegistryError::IOError(format!("Failed to remove bundle: {}", e))
                    })?;
                    return Ok(());
                }

                Err(RegistryError::BundleNotFound(format!("{}@{}", id, v)))
            }
            None => {
                // Remove all versions (remove entire directory)
                let bundle_dir = self.bundle_dir(id);
                if bundle_dir.exists() {
                    fs::remove_dir_all(&bundle_dir).map_err(|e| {
                        RegistryError::IOError(format!("Failed to remove bundle directory: {}", e))
                    })?;
                    Ok(())
                } else {
                    Err(RegistryError::BundleNotFound(id.to_string()))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_local_registry_creation() {
        let temp_dir = TempDir::new().unwrap();
        let registry = LocalRegistry::new(temp_dir.path()).unwrap();
        let _ = registry.list_bundles().unwrap();
    }

    #[test]
    fn test_store_and_get_bundle() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = LocalRegistry::new(temp_dir.path()).unwrap();

        let bundle_data = b"test bundle data".to_vec();
        let metadata = registry
            .store_bundle("test-bundle", "1.0.0", bundle_data.clone())
            .unwrap();

        assert_eq!(metadata.id, "test-bundle");
        assert_eq!(metadata.version, "1.0.0");

        let retrieved = registry.get_bundle("test-bundle", Some("1.0.0")).unwrap();
        assert_eq!(retrieved, bundle_data);
    }

    #[test]
    fn test_get_bundle_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let registry = LocalRegistry::new(temp_dir.path()).unwrap();

        let result = registry.get_bundle("nonexistent", Some("1.0.0"));
        assert!(result.is_err());
        match result.unwrap_err() {
            RegistryError::BundleNotFound(_) => {}
            _ => panic!("Expected BundleNotFound error"),
        }
    }

    #[test]
    fn test_list_bundles() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = LocalRegistry::new(temp_dir.path()).unwrap();

        registry
            .store_bundle("bundle1", "1.0.0", b"data1".to_vec())
            .unwrap();
        registry
            .store_bundle("bundle2", "2.0.0", b"data2".to_vec())
            .unwrap();

        let bundles = registry.list_bundles().unwrap();
        assert_eq!(bundles.len(), 2);
    }

    #[test]
    fn test_remove_bundle() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = LocalRegistry::new(temp_dir.path()).unwrap();

        registry
            .store_bundle("test", "1.0.0", b"data".to_vec())
            .unwrap();
        assert!(registry.get_bundle("test", Some("1.0.0")).is_ok());

        registry.remove_bundle("test", Some("1.0.0")).unwrap();
        assert!(registry.get_bundle("test", Some("1.0.0")).is_err());
    }
}
