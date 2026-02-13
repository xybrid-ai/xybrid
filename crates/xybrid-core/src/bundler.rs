//! Bundler module - Creates and manages .xyb bundle files.
//!
//! The bundler creates compressed bundle archives (tar + zstd) containing model files,
//! metadata, and a manifest.json file. Bundles can be created programmatically,
//! written to disk, and loaded for validation or extraction.
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_core::bundler::XyBundle;
//!
//! // Create a new bundle
//! let mut bundle = XyBundle::new("my-model", "1.0.0", "x86_64-linux");
//!
//! // Add files to the bundle
//! bundle.add_file("model.onnx")?;
//! bundle.add_file("config.json")?;
//!
//! // Write the bundle to disk
//! bundle.write("my-model-1.0.0.xyb")?;
//!
//! // Load an existing bundle
//! let loaded = XyBundle::load("my-model-1.0.0.xyb")?;
//! println!("Model ID: {}", loaded.manifest().model_id);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use thiserror::Error;

/// Error type for bundle operations.
#[derive(Error, Debug)]
pub enum BundlerError {
    #[error("IO error: {0}")]
    IOError(#[from] io::Error),
    #[error("Archive error: {0}")]
    ArchiveError(String),
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Result type for bundle operations.
pub type BundlerResult<T> = Result<T, BundlerError>;

/// Bundle manifest containing metadata about the bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundleManifest {
    /// Model identifier
    pub model_id: String,
    /// Version string (e.g., "1.0.0")
    pub version: String,
    /// Target platform (e.g., "x86_64-linux", "arm64-darwin")
    pub target: String,
    /// SHA-256 hash of the bundle contents
    pub hash: String,
    /// List of files in the bundle with their relative paths
    #[serde(default)]
    pub files: Vec<String>,
    /// Whether this bundle includes model_metadata.json for metadata-driven execution
    #[serde(default)]
    pub has_metadata: bool,
}

/// XyBundle for creating and managing .xyb bundle files.
///
/// Bundles are compressed tar archives using zstd compression, containing:
/// - A manifest.json file with metadata
/// - Model files and other assets
pub struct XyBundle {
    manifest: BundleManifest,
    files: HashMap<String, Vec<u8>>,
}

impl XyBundle {
    /// Creates a new empty bundle.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier
    /// * `version` - Version string (e.g., "1.0.0")
    /// * `target` - Target platform (e.g., "x86_64-linux")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xybrid_core::bundler::XyBundle;
    ///
    /// let bundle = XyBundle::new("whisper-tiny", "1.2.0", "x86_64-linux");
    /// ```
    pub fn new(
        model_id: impl Into<String>,
        version: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            manifest: BundleManifest {
                model_id: model_id.into(),
                version: version.into(),
                target: target.into(),
                hash: String::new(), // Will be computed when writing
                files: Vec::new(),
                has_metadata: false,
            },
            files: HashMap::new(),
        }
    }

    /// Adds a file to the bundle.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to add (only filename is preserved, not directory structure)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut bundle = XyBundle::new("my-model", "1.0.0", "x86_64-linux");
    /// bundle.add_file("model.onnx")?;
    /// ```
    pub fn add_file(&mut self, path: impl AsRef<Path>) -> BundlerResult<()> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(BundlerError::FileNotFound(path.display().to_string()));
        }

        // Read file contents
        let contents = fs::read(path)?;

        // Store with just the filename (use just the last component)
        let rel_path = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                BundlerError::InvalidManifest(format!("Invalid file path: {}", path.display()))
            })?
            .to_string();

        // Check if file already exists
        if self.files.contains_key(&rel_path) {
            return Err(BundlerError::ArchiveError(format!(
                "File already exists in bundle: {}",
                rel_path
            )));
        }

        self.files.insert(rel_path.clone(), contents);

        // Update manifest files list
        if !self.manifest.files.contains(&rel_path) {
            self.manifest.files.push(rel_path.clone());
        }

        // Check if this is model_metadata.json
        if rel_path == "model_metadata.json" {
            self.manifest.has_metadata = true;
        }

        Ok(())
    }

    /// Adds a file to the bundle with a custom relative path.
    ///
    /// This method allows preserving subdirectory structure in the bundle.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to add
    /// * `rel_path` - Relative path to use in the bundle (e.g., "misaki/us_gold.json")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut bundle = XyBundle::new("my-model", "1.0.0", "x86_64-linux");
    /// bundle.add_file_with_relative_path("/full/path/to/misaki/us_gold.json", "misaki/us_gold.json")?;
    /// ```
    pub fn add_file_with_relative_path(
        &mut self,
        path: impl AsRef<Path>,
        rel_path: impl Into<String>,
    ) -> BundlerResult<()> {
        let path = path.as_ref();
        let rel_path = rel_path.into();

        if !path.exists() {
            return Err(BundlerError::FileNotFound(path.display().to_string()));
        }

        // Read file contents
        let contents = fs::read(path)?;

        // Check if file already exists
        if self.files.contains_key(&rel_path) {
            return Err(BundlerError::ArchiveError(format!(
                "File already exists in bundle: {}",
                rel_path
            )));
        }

        self.files.insert(rel_path.clone(), contents);

        // Update manifest files list
        if !self.manifest.files.contains(&rel_path) {
            self.manifest.files.push(rel_path.clone());
        }

        // Check if this is model_metadata.json
        if rel_path == "model_metadata.json" {
            self.manifest.has_metadata = true;
        }

        Ok(())
    }

    /// Gets a reference to the bundle manifest.
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    /// Computes the SHA-256 hash of the bundle contents.
    ///
    /// The hash includes all file contents and the manifest (excluding the hash field itself).
    fn compute_hash(&self) -> BundlerResult<String> {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();

        // Hash file paths and contents in sorted order for deterministic hashing
        let mut file_paths: Vec<_> = self.files.keys().collect();
        file_paths.sort();

        for path in file_paths {
            hasher.update(path.as_bytes());
            if let Some(contents) = self.files.get(path) {
                hasher.update(contents);
            }
        }

        // Hash manifest (excluding hash field)
        let mut manifest = self.manifest.clone();
        manifest.hash = String::new();
        let manifest_json = serde_json::to_vec(&manifest)?;
        hasher.update(&manifest_json);

        let hash_bytes = hasher.finalize();
        Ok(format!("{:x}", hash_bytes))
    }

    /// Writes the bundle to disk as a compressed .xyb file.
    ///
    /// # Arguments
    ///
    /// * `path` - Output path for the bundle file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut bundle = XyBundle::new("my-model", "1.0.0", "x86_64-linux");
    /// bundle.add_file("model.onnx")?;
    /// bundle.write("my-model-1.0.0.xyb")?;
    /// ```
    pub fn write(&mut self, path: impl AsRef<Path>) -> BundlerResult<()> {
        let path = path.as_ref();

        // Compute hash before writing
        self.manifest.hash = self.compute_hash()?;

        // Create temporary file for tar archive
        let temp_dir = std::env::temp_dir();
        let tar_path = temp_dir.join(format!("xybundle_{}.tar", uuid::Uuid::new_v4()));

        // Create tar archive
        let tar_file = fs::File::create(&tar_path)?;
        let mut tar = tar::Builder::new(tar_file);

        // Add manifest.json
        let manifest_json = serde_json::to_string_pretty(&self.manifest)
            .map_err(BundlerError::SerializationError)?;
        let mut manifest_header = tar::Header::new_gnu();
        manifest_header.set_path("manifest.json").map_err(|e| {
            BundlerError::ArchiveError(format!("Failed to set manifest path: {}", e))
        })?;
        manifest_header.set_size(manifest_json.len() as u64);
        manifest_header.set_cksum();
        tar.append(&manifest_header, manifest_json.as_bytes())
            .map_err(|e| BundlerError::ArchiveError(format!("Failed to append manifest: {}", e)))?;

        // Add all files
        for (file_path, contents) in &self.files {
            let mut header = tar::Header::new_gnu();
            header.set_path(file_path).map_err(|e| {
                BundlerError::ArchiveError(format!("Failed to set file path {}: {}", file_path, e))
            })?;
            header.set_size(contents.len() as u64);
            header.set_cksum();
            tar.append(&header, contents.as_slice()).map_err(|e| {
                BundlerError::ArchiveError(format!("Failed to append file {}: {}", file_path, e))
            })?;
        }

        tar.finish().map_err(|e| {
            BundlerError::ArchiveError(format!("Failed to finish tar archive: {}", e))
        })?;

        // Compress with zstd
        let tar_bytes = fs::read(&tar_path)?;
        let output_file = fs::File::create(path)?;
        let mut zstd_writer = zstd::Encoder::new(output_file, 3) // Compression level 3 (balanced)
            .map_err(|e| {
                BundlerError::ArchiveError(format!("Failed to create zstd encoder: {}", e))
            })?;

        zstd_writer
            .write_all(&tar_bytes)
            .map_err(BundlerError::IOError)?;
        zstd_writer.finish().map_err(|e| {
            BundlerError::ArchiveError(format!("Failed to finish zstd compression: {}", e))
        })?;

        // Clean up temporary tar file
        let _ = fs::remove_file(&tar_path);

        Ok(())
    }

    /// Loads a bundle from disk.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .xyb bundle file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let bundle = XyBundle::load("my-model-1.0.0.xyb")?;
    /// println!("Model: {}", bundle.manifest().model_id);
    /// ```
    pub fn load(path: impl AsRef<Path>) -> BundlerResult<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(BundlerError::FileNotFound(path.display().to_string()));
        }

        // Read and decompress zstd
        let file = fs::File::open(path)?;
        let mut zstd_reader = zstd::Decoder::new(file).map_err(|e| {
            BundlerError::ArchiveError(format!("Failed to create zstd decoder: {}", e))
        })?;

        let mut tar_bytes = Vec::new();
        zstd_reader
            .read_to_end(&mut tar_bytes)
            .map_err(BundlerError::IOError)?;

        // Extract tar archive
        let mut tar = tar::Archive::new(tar_bytes.as_slice());
        let mut manifest: Option<BundleManifest> = None;
        let mut files = HashMap::new();

        for entry in tar
            .entries()
            .map_err(|e| BundlerError::ArchiveError(format!("Failed to read tar entries: {}", e)))?
        {
            let mut entry = entry.map_err(|e| {
                BundlerError::ArchiveError(format!("Failed to read tar entry: {}", e))
            })?;

            let entry_path = entry
                .path()
                .map_err(|e| {
                    BundlerError::ArchiveError(format!("Failed to get entry path: {}", e))
                })?
                .to_string_lossy()
                .to_string();

            let mut contents = Vec::new();
            entry
                .read_to_end(&mut contents)
                .map_err(BundlerError::IOError)?;

            if entry_path == "manifest.json" {
                manifest = Some(serde_json::from_slice(&contents).map_err(|e| {
                    BundlerError::InvalidManifest(format!("Failed to parse manifest: {}", e))
                })?);
            } else {
                files.insert(entry_path, contents);
            }
        }

        let manifest = manifest.ok_or_else(|| {
            BundlerError::InvalidManifest("manifest.json not found in bundle".to_string())
        })?;

        // Verify manifest files list matches actual files
        let manifest_files: std::collections::HashSet<String> =
            manifest.files.iter().cloned().collect();
        let actual_files: std::collections::HashSet<String> = files.keys().cloned().collect();

        if manifest_files != actual_files {
            return Err(BundlerError::InvalidManifest(format!(
                "Manifest file list mismatch. Manifest: {:?}, Actual: {:?}",
                manifest_files, actual_files
            )));
        }

        Ok(Self { manifest, files })
    }

    /// Load a bundle from bytes (e.g., downloaded from HTTP).
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw bundle bytes (zstd compressed tar archive)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let bytes = download_bundle_bytes("http://registry.example.com/bundle.xyb");
    /// let bundle = XyBundle::load_from_bytes(&bytes)?;
    /// println!("Model: {}", bundle.manifest().model_id);
    /// ```
    pub fn load_from_bytes(bytes: &[u8]) -> BundlerResult<Self> {
        use std::io::Cursor;

        // Decompress zstd
        let mut zstd_reader = zstd::Decoder::new(Cursor::new(bytes)).map_err(|e| {
            BundlerError::ArchiveError(format!("Failed to create zstd decoder: {}", e))
        })?;

        let mut tar_bytes = Vec::new();
        zstd_reader
            .read_to_end(&mut tar_bytes)
            .map_err(BundlerError::IOError)?;

        // Extract tar archive
        let mut tar = tar::Archive::new(tar_bytes.as_slice());
        let mut manifest: Option<BundleManifest> = None;
        let mut files = HashMap::new();

        for entry in tar
            .entries()
            .map_err(|e| BundlerError::ArchiveError(format!("Failed to read tar entries: {}", e)))?
        {
            let mut entry = entry.map_err(|e| {
                BundlerError::ArchiveError(format!("Failed to read tar entry: {}", e))
            })?;

            let entry_path = entry
                .path()
                .map_err(|e| {
                    BundlerError::ArchiveError(format!("Failed to get entry path: {}", e))
                })?
                .to_string_lossy()
                .to_string();

            let mut contents = Vec::new();
            entry
                .read_to_end(&mut contents)
                .map_err(BundlerError::IOError)?;

            if entry_path == "manifest.json" {
                manifest = Some(serde_json::from_slice(&contents).map_err(|e| {
                    BundlerError::InvalidManifest(format!("Failed to parse manifest: {}", e))
                })?);
            } else {
                files.insert(entry_path, contents);
            }
        }

        let manifest = manifest.ok_or_else(|| {
            BundlerError::InvalidManifest("manifest.json not found in bundle".to_string())
        })?;

        // Verify manifest files list matches actual files
        let manifest_files: std::collections::HashSet<String> =
            manifest.files.iter().cloned().collect();
        let actual_files: std::collections::HashSet<String> = files.keys().cloned().collect();

        if manifest_files != actual_files {
            return Err(BundlerError::InvalidManifest(format!(
                "Manifest file list mismatch. Manifest: {:?}, Actual: {:?}",
                manifest_files, actual_files
            )));
        }

        Ok(Self { manifest, files })
    }

    /// Extracts all files from the bundle to a directory.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory to extract files to
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let bundle = XyBundle::load("my-model-1.0.0.xyb")?;
    /// bundle.extract_to("/tmp/extracted")?;
    /// ```
    pub fn extract_to(&self, output_dir: impl AsRef<Path>) -> BundlerResult<()> {
        let output_dir = output_dir.as_ref();

        fs::create_dir_all(output_dir).map_err(BundlerError::IOError)?;

        for (file_path, contents) in &self.files {
            let full_path = output_dir.join(file_path);

            // Create parent directories if needed
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent).map_err(BundlerError::IOError)?;
            }

            fs::write(&full_path, contents).map_err(BundlerError::IOError)?;
        }

        Ok(())
    }

    /// Gets the model_metadata.json content if present in the bundle.
    ///
    /// # Returns
    ///
    /// The metadata JSON string if present, None otherwise
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let bundle = XyBundle::load("my-model-1.0.0.xyb")?;
    /// if let Some(metadata_json) = bundle.get_metadata_json()? {
    ///     println!("Metadata: {}", metadata_json);
    /// }
    /// ```
    pub fn get_metadata_json(&self) -> BundlerResult<Option<String>> {
        if let Some(contents) = self.files.get("model_metadata.json") {
            let json_str = String::from_utf8(contents.clone()).map_err(|e| {
                BundlerError::InvalidManifest(format!("Invalid UTF-8 in metadata: {}", e))
            })?;
            Ok(Some(json_str))
        } else {
            Ok(None)
        }
    }

    /// Gets a specific file's contents from the bundle.
    ///
    /// # Arguments
    ///
    /// * `filename` - Name of the file to retrieve
    ///
    /// # Returns
    ///
    /// File contents if present, None otherwise
    pub fn get_file(&self, filename: &str) -> Option<&Vec<u8>> {
        self.files.get(filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_bundle_creation() {
        let bundle = XyBundle::new("test-model", "1.0.0", "x86_64-linux");
        assert_eq!(bundle.manifest().model_id, "test-model");
        assert_eq!(bundle.manifest().version, "1.0.0");
        assert_eq!(bundle.manifest().target, "x86_64-linux");
        assert!(bundle.manifest().hash.is_empty());
    }

    #[test]
    fn test_bundle_add_file() -> BundlerResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, b"test content").unwrap();

        let mut bundle = XyBundle::new("test-model", "1.0.0", "x86_64-linux");
        bundle.add_file(&test_file)?;

        assert_eq!(bundle.manifest().files.len(), 1);
        assert!(bundle.files.contains_key("test.txt"));

        Ok(())
    }

    #[test]
    fn test_bundle_write_and_load() -> BundlerResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("model.onnx");
        fs::write(&test_file, b"fake model data").unwrap();

        let bundle_path = temp_dir.path().join("test.xyb");

        // Create and write bundle
        let mut bundle = XyBundle::new("test-model", "1.0.0", "x86_64-linux");
        bundle.add_file(&test_file)?;
        bundle.write(&bundle_path)?;

        // Verify bundle was created
        assert!(bundle_path.exists());
        assert!(!bundle.manifest().hash.is_empty());

        // Load bundle
        let loaded = XyBundle::load(&bundle_path)?;
        assert_eq!(loaded.manifest().model_id, "test-model");
        assert_eq!(loaded.manifest().version, "1.0.0");
        assert_eq!(loaded.manifest().target, "x86_64-linux");
        assert_eq!(loaded.manifest().hash, bundle.manifest().hash);
        assert_eq!(loaded.manifest().files.len(), 1);

        // Verify file contents
        let model_data = loaded.files.get("model.onnx").unwrap();
        assert_eq!(model_data, b"fake model data");

        Ok(())
    }

    #[test]
    fn test_bundle_extract() -> BundlerResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("model.onnx");
        fs::write(&test_file, b"fake model data").unwrap();

        let bundle_path = temp_dir.path().join("test.xyb");
        let extract_dir = temp_dir.path().join("extracted");

        // Create bundle
        let mut bundle = XyBundle::new("test-model", "1.0.0", "x86_64-linux");
        bundle.add_file(&test_file)?;
        bundle.write(&bundle_path)?;

        // Load and extract
        let loaded = XyBundle::load(&bundle_path)?;
        loaded.extract_to(&extract_dir)?;

        // Verify extracted file
        let extracted_file = extract_dir.join("model.onnx");
        assert!(extracted_file.exists());
        let contents = fs::read(&extracted_file).unwrap();
        assert_eq!(contents, b"fake model data");

        Ok(())
    }

    #[test]
    fn test_bundle_hash_consistency() -> BundlerResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("model.onnx");
        fs::write(&test_file, b"test content").unwrap();

        let bundle_path = temp_dir.path().join("test.xyb");

        // Create bundle twice with same content
        let mut bundle1 = XyBundle::new("test-model", "1.0.0", "x86_64-linux");
        bundle1.add_file(&test_file)?;
        bundle1.write(&bundle_path)?;
        let hash1 = bundle1.manifest().hash.clone();

        fs::remove_file(&bundle_path).unwrap();

        let mut bundle2 = XyBundle::new("test-model", "1.0.0", "x86_64-linux");
        bundle2.add_file(&test_file)?;
        bundle2.write(&bundle_path)?;
        let hash2 = bundle2.manifest().hash.clone();

        // Hashes should be identical
        assert_eq!(hash1, hash2);

        Ok(())
    }
}
