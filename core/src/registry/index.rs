//! Registry Index module - Maintains a JSON index of available bundles.
//!
//! The registry index provides a fast way to discover available bundles without
//! scanning the file system. It maintains a JSON file at ~/.xybrid/registry/index.json
//! that lists all bundles by model_id and version with their metadata.
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::registry::{RegistryIndex, IndexEntry};
//!
//! // Load or create index
//! let mut index = RegistryIndex::load_or_create()?;
//!
//! // Add a bundle entry
//! index.add_entry(IndexEntry {
//!     model_id: "whisper-tiny".to_string(),
//!     version: "1.2.0".to_string(),
//!     target: "x86_64-linux".to_string(),
//!     hash: "abc123...".to_string(),
//!     size_bytes: 1024000,
//!     path: "/path/to/bundle".to_string(),
//! })?;
//!
//! // Save index
//! index.save()?;
//!
//! // Query bundles
//! let entries = index.find_by_model("whisper-tiny")?;
//! ```

use super::config::{BundleDescriptor, BundleLocation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Error type for registry index operations.
#[derive(Error, Debug)]
pub enum IndexError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Invalid index path: {0}")]
    InvalidPath(String),
    #[error("Entry not found: {model_id}@{version}:{target}")]
    EntryNotFound {
        model_id: String,
        version: String,
        target: String,
    },
}

/// Result type for index operations.
pub type IndexResult<T> = Result<T, IndexError>;

/// Entry in the registry index representing a single bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexEntry {
    /// Model identifier
    pub model_id: String,
    /// Version string
    pub version: String,
    /// Target platform
    pub target: String,
    /// Bundle content hash (optional for backwards compatibility)
    #[serde(default)]
    pub hash: String,
    /// Bundle size in bytes
    pub size_bytes: u64,
    /// Path to the bundle file
    pub path: String,
}

impl IndexEntry {
    /// Creates a new index entry.
    pub fn new(
        model_id: impl Into<String>,
        version: impl Into<String>,
        target: impl Into<String>,
        hash: impl Into<String>,
        size_bytes: u64,
        path: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            version: version.into(),
            target: target.into(),
            hash: hash.into(),
            size_bytes,
            path: path.into(),
        }
    }

    /// Gets the index key for this entry.
    /// Format: "model_id@version:target"
    pub fn key(&self) -> String {
        format!("{}@{}:{}", self.model_id, self.version, self.target)
    }

    /// Gets the version key (without target) for this entry.
    /// Format: "model_id@version"
    pub fn version_key(&self) -> String {
        format!("{}@{}", self.model_id, self.version)
    }

    /// Parse a key string into (model_id, version, target).
    /// Supports both "model_id@version:target" and legacy "model_id@version" formats.
    pub fn parse_key(key: &str) -> Option<(String, String, String)> {
        // Try new format: model_id@version:target
        if let Some((model_version, target)) = key.rsplit_once(':') {
            if let Some((model_id, version)) = model_version.split_once('@') {
                return Some((model_id.to_string(), version.to_string(), target.to_string()));
            }
        }
        // Try legacy format: model_id@version (default to "onnx" target)
        if let Some((model_id, version)) = key.split_once('@') {
            return Some((model_id.to_string(), version.to_string(), "onnx".to_string()));
        }
        None
    }
}

impl From<&IndexEntry> for BundleDescriptor {
    fn from(entry: &IndexEntry) -> Self {
        BundleDescriptor {
            id: entry.model_id.clone(),
            version: entry.version.clone(),
            target: Some(entry.target.clone()),
            hash: Some(entry.hash.clone()),
            size_bytes: entry.size_bytes,
            location: BundleLocation::Local {
                path: entry.path.clone(),
            },
        }
    }
}

/// Registry index that maintains a JSON file listing all available bundles.
///
/// The index file is stored at ~/.xybrid/registry/index.json and contains
/// a map of bundle entries keyed by "model_id@version".
pub struct RegistryIndex {
    index_path: PathBuf,
    entries: HashMap<String, IndexEntry>,
}

impl RegistryIndex {
    /// Gets the default index file path.
    fn default_index_path() -> IndexResult<PathBuf> {
        let mut path = dirs::home_dir().ok_or_else(|| {
            IndexError::InvalidPath("Could not determine home directory".to_string())
        })?;
        path.push(".xybrid");
        path.push("registry");
        path.push("index.json");
        Ok(path)
    }

    /// Creates a new empty index.
    pub fn new() -> IndexResult<Self> {
        let index_path = Self::default_index_path()?;
        Ok(Self {
            index_path,
            entries: HashMap::new(),
        })
    }

    /// Creates a new index with a custom path.
    pub fn with_path(path: impl AsRef<Path>) -> IndexResult<Self> {
        let index_path = path.as_ref().to_path_buf();
        Ok(Self {
            index_path,
            entries: HashMap::new(),
        })
    }

    /// Loads an existing index from disk, or creates a new one if it doesn't exist.
    pub fn load_or_create() -> IndexResult<Self> {
        let index_path = Self::default_index_path()?;

        // Ensure registry directory exists
        if let Some(parent) = index_path.parent() {
            fs::create_dir_all(parent)?;
        }

        if !index_path.exists() {
            // Create empty index
            return Ok(Self {
                index_path,
                entries: HashMap::new(),
            });
        }

        // Load existing index
        let content = fs::read_to_string(&index_path)?;
        let entries: HashMap<String, IndexEntry> = if content.trim().is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(&content)?
        };

        Ok(Self {
            index_path,
            entries,
        })
    }

    /// Loads an index from a custom path, or creates a new one if it doesn't exist.
    pub fn load_or_create_at(path: impl AsRef<Path>) -> IndexResult<Self> {
        let index_path = path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = index_path.parent() {
            fs::create_dir_all(parent)?;
        }

        if !index_path.exists() {
            return Ok(Self {
                index_path,
                entries: HashMap::new(),
            });
        }

        let content = fs::read_to_string(&index_path)?;
        let entries: HashMap<String, IndexEntry> = if content.trim().is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(&content)?
        };

        Ok(Self {
            index_path,
            entries,
        })
    }

    /// Saves the index to disk.
    pub fn save(&self) -> IndexResult<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.index_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&self.entries)?;
        fs::write(&self.index_path, json)?;
        Ok(())
    }

    /// Gets the index file path.
    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    /// Adds or updates an entry in the index.
    /// Uses the new key format: "model_id@version:target"
    pub fn add_entry(&mut self, entry: IndexEntry) -> IndexResult<()> {
        let key = entry.key();
        self.entries.insert(key, entry);
        Ok(())
    }

    /// Removes an entry from the index by model_id, version, and target.
    pub fn remove_entry_with_target(
        &mut self,
        model_id: &str,
        version: &str,
        target: &str,
    ) -> IndexResult<()> {
        let key = format!("{}@{}:{}", model_id, version, target);
        self.entries
            .remove(&key)
            .ok_or_else(|| IndexError::EntryNotFound {
                model_id: model_id.to_string(),
                version: version.to_string(),
                target: target.to_string(),
            })
            .map(|_| ())
    }

    /// Removes an entry from the index (for backward compatibility, defaults to onnx target).
    pub fn remove_entry(&mut self, model_id: &str, version: &str) -> IndexResult<()> {
        // First try with new key format
        let new_key = format!("{}@{}:onnx", model_id, version);
        if self.entries.remove(&new_key).is_some() {
            return Ok(());
        }

        // Then try legacy key format
        let legacy_key = format!("{}@{}", model_id, version);
        self.entries
            .remove(&legacy_key)
            .ok_or_else(|| IndexError::EntryNotFound {
                model_id: model_id.to_string(),
                version: version.to_string(),
                target: "onnx".to_string(),
            })
            .map(|_| ())
    }

    /// Finds an entry by model_id, version, and target.
    pub fn find_entry_with_target(
        &self,
        model_id: &str,
        version: &str,
        target: &str,
    ) -> IndexResult<Option<&IndexEntry>> {
        let key = format!("{}@{}:{}", model_id, version, target);
        Ok(self.entries.get(&key))
    }

    /// Finds an entry by model_id and version (defaults to onnx target).
    /// Also checks legacy key format for backward compatibility.
    pub fn find_entry(&self, model_id: &str, version: &str) -> IndexResult<Option<&IndexEntry>> {
        // Try new key format with onnx target
        let new_key = format!("{}@{}:onnx", model_id, version);
        if let Some(entry) = self.entries.get(&new_key) {
            return Ok(Some(entry));
        }

        // Try legacy key format
        let legacy_key = format!("{}@{}", model_id, version);
        Ok(self.entries.get(&legacy_key))
    }

    /// Finds all entries for a specific model and version (all targets).
    pub fn find_all_targets(&self, model_id: &str, version: &str) -> IndexResult<Vec<&IndexEntry>> {
        Ok(self
            .entries
            .values()
            .filter(|entry| entry.model_id == model_id && entry.version == version)
            .collect())
    }

    /// Finds all entries for a given model_id.
    pub fn find_by_model(&self, model_id: &str) -> IndexResult<Vec<&IndexEntry>> {
        Ok(self
            .entries
            .values()
            .filter(|entry| entry.model_id == model_id)
            .collect())
    }

    /// Finds all entries for a given target platform.
    pub fn find_by_target(&self, target: &str) -> IndexResult<Vec<&IndexEntry>> {
        Ok(self
            .entries
            .values()
            .filter(|entry| entry.target == target)
            .collect())
    }

    /// Gets all entries in the index.
    pub fn all_entries(&self) -> Vec<&IndexEntry> {
        self.entries.values().collect()
    }

    /// Gets the latest version of a model.
    pub fn find_latest_version(&self, model_id: &str) -> IndexResult<Option<&IndexEntry>> {
        let entries: Vec<&IndexEntry> = self.find_by_model(model_id)?;

        if entries.is_empty() {
            return Ok(None);
        }

        // Simple version comparison (string sort works for semver-like versions)
        // In production, you'd want proper semver parsing
        Ok(entries.iter().max_by_key(|e| &e.version).copied())
    }

    /// Lists all unique model IDs in the index.
    pub fn list_models(&self) -> Vec<String> {
        let mut models: std::collections::HashSet<String> =
            self.entries.values().map(|e| e.model_id.clone()).collect();
        let mut sorted: Vec<String> = models.drain().collect();
        sorted.sort();
        sorted
    }

    /// Gets the total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Checks if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns bundle descriptors suitable for higher-level registry discovery.
    pub fn bundle_descriptors(&self) -> Vec<BundleDescriptor> {
        self.entries.values().map(BundleDescriptor::from).collect()
    }
}

impl Default for RegistryIndex {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback: create with empty path if home dir can't be determined
            Self {
                index_path: PathBuf::from("index.json"),
                entries: HashMap::new(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_index() -> IndexResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("index.json");

        let index = RegistryIndex::with_path(&index_path)?;
        assert!(index.is_empty());

        Ok(())
    }

    #[test]
    fn test_add_and_find_entry() -> IndexResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("index.json");

        let mut index = RegistryIndex::with_path(&index_path)?;

        // Use "onnx" target since find_entry() defaults to searching for "onnx"
        let entry = IndexEntry::new(
            "test-model",
            "1.0.0",
            "onnx",
            "hash123",
            1000,
            "/path/to/bundle",
        );

        index.add_entry(entry.clone())?;
        index.save()?;

        // Reload and verify
        let loaded = RegistryIndex::load_or_create_at(&index_path)?;
        let found =
            loaded
                .find_entry("test-model", "1.0.0")?
                .ok_or_else(|| IndexError::EntryNotFound {
                    model_id: "test-model".to_string(),
                    version: "1.0.0".to_string(),
                    target: "onnx".to_string(),
                })?;

        assert_eq!(found.model_id, entry.model_id);
        assert_eq!(found.version, entry.version);
        assert_eq!(found.hash, entry.hash);

        Ok(())
    }

    #[test]
    fn test_find_by_model() -> IndexResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("index.json");

        let mut index = RegistryIndex::with_path(&index_path)?;

        index.add_entry(IndexEntry::new(
            "model-a",
            "1.0.0",
            "x86_64-linux",
            "hash1",
            1000,
            "/path1",
        ))?;
        index.add_entry(IndexEntry::new(
            "model-a",
            "1.1.0",
            "x86_64-linux",
            "hash2",
            2000,
            "/path2",
        ))?;
        index.add_entry(IndexEntry::new(
            "model-b",
            "1.0.0",
            "x86_64-linux",
            "hash3",
            3000,
            "/path3",
        ))?;

        let entries = index.find_by_model("model-a")?;
        assert_eq!(entries.len(), 2);

        Ok(())
    }

    #[test]
    fn test_remove_entry() -> IndexResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("index.json");

        let mut index = RegistryIndex::with_path(&index_path)?;

        // Use "onnx" target since remove_entry() defaults to searching for "onnx"
        index.add_entry(IndexEntry::new(
            "test-model",
            "1.0.0",
            "onnx",
            "hash",
            1000,
            "/path",
        ))?;
        assert_eq!(index.len(), 1);

        index.remove_entry("test-model", "1.0.0")?;
        assert_eq!(index.len(), 0);

        assert!(index.remove_entry("test-model", "1.0.0").is_err());

        Ok(())
    }

    #[test]
    fn test_find_latest_version() -> IndexResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("index.json");

        let mut index = RegistryIndex::with_path(&index_path)?;

        index.add_entry(IndexEntry::new(
            "test-model",
            "1.0.0",
            "x86_64-linux",
            "hash1",
            1000,
            "/path1",
        ))?;
        index.add_entry(IndexEntry::new(
            "test-model",
            "1.1.0",
            "x86_64-linux",
            "hash2",
            2000,
            "/path2",
        ))?;
        index.add_entry(IndexEntry::new(
            "test-model",
            "2.0.0",
            "x86_64-linux",
            "hash3",
            3000,
            "/path3",
        ))?;

        let latest =
            index
                .find_latest_version("test-model")?
                .ok_or_else(|| IndexError::EntryNotFound {
                    model_id: "test-model".to_string(),
                    version: "latest".to_string(),
                    target: "onnx".to_string(),
                })?;

        assert_eq!(latest.version, "2.0.0");

        Ok(())
    }

    #[test]
    fn test_list_models() -> IndexResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("index.json");

        let mut index = RegistryIndex::with_path(&index_path)?;

        index.add_entry(IndexEntry::new(
            "model-a",
            "1.0.0",
            "x86_64-linux",
            "hash1",
            1000,
            "/path1",
        ))?;
        index.add_entry(IndexEntry::new(
            "model-b",
            "1.0.0",
            "x86_64-linux",
            "hash2",
            2000,
            "/path2",
        ))?;
        index.add_entry(IndexEntry::new(
            "model-c",
            "1.0.0",
            "x86_64-linux",
            "hash3",
            3000,
            "/path3",
        ))?;

        let models = index.list_models();
        assert_eq!(models.len(), 3);
        assert!(models.contains(&"model-a".to_string()));
        assert!(models.contains(&"model-b".to_string()));
        assert!(models.contains(&"model-c".to_string()));

        Ok(())
    }

    #[test]
    fn test_load_or_create() -> IndexResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("index.json");

        // Create new index
        let mut index = RegistryIndex::load_or_create_at(&index_path)?;
        assert!(index.is_empty());

        // Add entry and save
        index.add_entry(IndexEntry::new(
            "test",
            "1.0.0",
            "x86_64-linux",
            "hash",
            1000,
            "/path",
        ))?;
        index.save()?;

        // Load existing index
        let loaded = RegistryIndex::load_or_create_at(&index_path)?;
        assert_eq!(loaded.len(), 1);

        Ok(())
    }
}
