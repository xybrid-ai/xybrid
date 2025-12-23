//! Registry client for fetching models from api.xybrid.dev.
//!
//! This module provides:
//! - `RegistryClient`: High-level API for model resolution and download
//! - Mask-based model lookup with platform resolution
//! - SHA256 hash verification
//! - Download progress callbacks
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_sdk::registry_client::RegistryClient;
//!
//! let client = RegistryClient::default_client()?;
//!
//! // List available models
//! let models = client.list_models()?;
//! for model in models {
//!     println!("{}: {} ({})", model.id, model.description, model.task);
//! }
//!
//! // Resolve a model for the current platform
//! let resolved = client.resolve("kokoro-82m", None)?;
//! println!("Download URL: {}", resolved.download_url);
//!
//! // Fetch and cache the bundle
//! let bundle_path = client.fetch("kokoro-82m", None, |progress| {
//!     println!("Downloaded: {:.1}%", progress * 100.0);
//! })?;
//! ```

use crate::cache::CacheManager;
use crate::model::SdkError;
use crate::source::detect_platform;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;

/// Default registry API URL.
pub const DEFAULT_REGISTRY_URL: &str = "https://api.xybrid.dev";

/// Registry client for model resolution and download.
#[derive(Debug)]
pub struct RegistryClient {
    /// Base URL for the registry API
    api_url: String,
    /// Cache manager for storing downloaded bundles
    cache: CacheManager,
}

impl RegistryClient {
    /// Create a new registry client with the specified API URL.
    pub fn new(api_url: impl Into<String>) -> Result<Self, SdkError> {
        Ok(Self {
            api_url: api_url.into(),
            cache: CacheManager::new()?,
        })
    }

    /// Create a registry client with default API URL.
    pub fn default_client() -> Result<Self, SdkError> {
        Self::new(DEFAULT_REGISTRY_URL)
    }

    /// Create a registry client from environment variable or default.
    ///
    /// Checks `XYBRID_REGISTRY_URL` environment variable first,
    /// then falls back to `XYBRID_PLATFORM_URL`, then default.
    pub fn from_env() -> Result<Self, SdkError> {
        let url = std::env::var("XYBRID_REGISTRY_URL")
            .or_else(|_| std::env::var("XYBRID_PLATFORM_URL"))
            .unwrap_or_else(|_| DEFAULT_REGISTRY_URL.to_string());
        Self::new(url)
    }

    /// List all available models in the registry.
    pub fn list_models(&self) -> Result<Vec<ModelSummary>, SdkError> {
        let url = format!("{}/v1/models/registry", self.api_url);

        let response = ureq::get(&url)
            .call()
            .map_err(|e| SdkError::NetworkError(format!("Failed to list models: {}", e)))?;

        if response.status() != 200 {
            return Err(SdkError::NetworkError(format!(
                "Registry returned status {}",
                response.status()
            )));
        }

        let list_response: ListModelsResponse = response
            .into_json()
            .map_err(|e| SdkError::NetworkError(format!("Failed to parse response: {}", e)))?;

        Ok(list_response.models)
    }

    /// Get detailed information about a specific model.
    pub fn get_model(&self, mask: &str) -> Result<ModelDetail, SdkError> {
        let url = format!("{}/v1/models/registry/{}", self.api_url, mask);

        let response = ureq::get(&url)
            .call()
            .map_err(|e| SdkError::NetworkError(format!("Failed to get model: {}", e)))?;

        if response.status() == 404 {
            return Err(SdkError::ModelNotFound(format!("Model '{}' not found", mask)));
        }

        if response.status() != 200 {
            return Err(SdkError::NetworkError(format!(
                "Registry returned status {}",
                response.status()
            )));
        }

        response
            .into_json()
            .map_err(|e| SdkError::NetworkError(format!("Failed to parse response: {}", e)))
    }

    /// Resolve a model mask to the best variant for the given platform.
    ///
    /// If platform is None, auto-detects the current platform.
    pub fn resolve(&self, mask: &str, platform: Option<&str>) -> Result<ResolvedVariant, SdkError> {
        let platform = platform
            .map(String::from)
            .unwrap_or_else(detect_platform);

        let url = format!(
            "{}/v1/models/registry/{}/resolve?platform={}",
            self.api_url, mask, platform
        );

        let response = ureq::get(&url)
            .call()
            .map_err(|e| SdkError::NetworkError(format!("Failed to resolve model: {}", e)))?;

        if response.status() == 404 {
            return Err(SdkError::ModelNotFound(format!(
                "Model '{}' not found or no compatible variant for platform '{}'",
                mask, platform
            )));
        }

        if response.status() != 200 {
            return Err(SdkError::NetworkError(format!(
                "Registry returned status {}",
                response.status()
            )));
        }

        let resolve_response: ResolveResponse = response
            .into_json()
            .map_err(|e| SdkError::NetworkError(format!("Failed to parse response: {}", e)))?;

        Ok(resolve_response.resolved)
    }

    /// Check if a model is cached locally.
    pub fn is_cached(&self, mask: &str, platform: Option<&str>) -> Result<bool, SdkError> {
        let resolved = self.resolve(mask, platform)?;
        let cache_path = self.get_cache_path(&resolved);

        if !cache_path.exists() {
            return Ok(false);
        }

        // Verify hash if available
        if !resolved.sha256.is_empty() {
            let hash = compute_sha256(&cache_path)?;
            Ok(hash == resolved.sha256)
        } else {
            Ok(true)
        }
    }

    /// Get the local cache path for a resolved variant.
    pub fn get_cache_path(&self, resolved: &ResolvedVariant) -> PathBuf {
        // Extract model name from hf_repo (e.g., "xybrid-ai/kokoro-82m" -> "kokoro-82m")
        let model_name = resolved
            .hf_repo
            .split('/')
            .last()
            .unwrap_or(&resolved.hf_repo);

        self.cache
            .cache_dir()
            .join(model_name)
            .join(&resolved.file)
    }

    /// Fetch a model bundle, downloading if not cached.
    ///
    /// Returns the path to the extracted bundle directory.
    ///
    /// # Arguments
    ///
    /// * `mask` - Model mask (e.g., "kokoro-82m")
    /// * `platform` - Target platform (None for auto-detect)
    /// * `progress_callback` - Optional callback for download progress (0.0 to 1.0)
    pub fn fetch<F>(
        &self,
        mask: &str,
        platform: Option<&str>,
        progress_callback: F,
    ) -> Result<PathBuf, SdkError>
    where
        F: Fn(f32),
    {
        let resolved = self.resolve(mask, platform)?;
        let cache_path = self.get_cache_path(&resolved);

        // Check if already cached with correct hash
        if cache_path.exists() && !resolved.sha256.is_empty() {
            let hash = compute_sha256(&cache_path)?;
            if hash == resolved.sha256 {
                // Already cached and verified
                return Ok(cache_path);
            }
            // Hash mismatch - re-download
            std::fs::remove_file(&cache_path).ok();
        }

        // Create cache directory
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Download from HuggingFace
        self.download_with_progress(&resolved.download_url, &cache_path, resolved.size_bytes, progress_callback)?;

        // Verify hash
        if !resolved.sha256.is_empty() {
            let hash = compute_sha256(&cache_path)?;
            if hash != resolved.sha256 {
                std::fs::remove_file(&cache_path).ok();
                return Err(SdkError::CacheError(format!(
                    "SHA256 mismatch: expected {}, got {}",
                    resolved.sha256, hash
                )));
            }
        }

        Ok(cache_path)
    }

    /// Download a file with progress tracking.
    fn download_with_progress<F>(
        &self,
        url: &str,
        dest: &PathBuf,
        total_size: u64,
        progress_callback: F,
    ) -> Result<(), SdkError>
    where
        F: Fn(f32),
    {
        let response = ureq::get(url)
            .call()
            .map_err(|e| SdkError::NetworkError(format!("Failed to download: {}", e)))?;

        if response.status() != 200 {
            return Err(SdkError::NetworkError(format!(
                "Download failed with status {}",
                response.status()
            )));
        }

        let mut file = File::create(dest)?;
        let mut reader = response.into_reader();
        let mut buffer = [0u8; 8192];
        let mut downloaded: u64 = 0;

        loop {
            let bytes_read = reader
                .read(&mut buffer)
                .map_err(|e| SdkError::NetworkError(format!("Read error: {}", e)))?;

            if bytes_read == 0 {
                break;
            }

            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;

            // Report progress
            if total_size > 0 {
                let progress = downloaded as f32 / total_size as f32;
                progress_callback(progress.min(1.0));
            }
        }

        progress_callback(1.0);
        Ok(())
    }

    /// Clear the local cache for a specific model.
    pub fn clear_cache(&self, mask: &str) -> Result<(), SdkError> {
        let model_dir = self.cache.cache_dir().join(mask);
        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir)?;
        }
        Ok(())
    }

    /// Clear the entire model cache.
    pub fn clear_all_cache(&mut self) -> Result<(), SdkError> {
        self.cache.clear().map_err(|e| SdkError::CacheError(e.to_string()))?;
        Ok(())
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> Result<CacheStats, SdkError> {
        let cache_dir = self.cache.cache_dir();
        let mut total_size: u64 = 0;
        let mut model_count: usize = 0;

        if cache_dir.exists() {
            for entry in std::fs::read_dir(cache_dir)? {
                let entry = entry?;
                if entry.path().is_dir() {
                    model_count += 1;
                    total_size += dir_size(&entry.path())?;
                }
            }
        }

        Ok(CacheStats {
            total_size_bytes: total_size,
            model_count,
            cache_path: cache_dir.to_path_buf(),
        })
    }
}

/// Compute SHA256 hash of a file.
fn compute_sha256(path: &PathBuf) -> Result<String, SdkError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Calculate total size of a directory.
fn dir_size(path: &PathBuf) -> Result<u64, SdkError> {
    let mut total: u64 = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            total += dir_size(&entry.path())?;
        }
    }
    Ok(total)
}

// ============================================================================
// API Response Types
// ============================================================================

/// Response from GET /v1/models/registry
#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    models: Vec<ModelSummary>,
}

/// Summary of a model in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Model mask ID (e.g., "kokoro-82m")
    pub id: String,
    /// Model family (e.g., "hexgrad", "openai")
    pub family: String,
    /// Task type (e.g., "text-to-speech", "speech-recognition")
    pub task: String,
    /// Number of parameters
    pub parameters: u64,
    /// Human-readable description
    pub description: String,
    /// Available variants (e.g., ["universal"])
    pub variants: Vec<String>,
}

/// Detailed model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetail {
    /// Model mask ID
    pub id: String,
    /// Model family
    pub family: String,
    /// Task type
    pub task: String,
    /// Number of parameters
    pub parameters: u64,
    /// Description
    pub description: String,
    /// Default variant name
    pub default_variant: Option<String>,
    /// Available variants with details
    pub variants: HashMap<String, VariantInfo>,
}

/// Information about a model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantInfo {
    /// Platform identifier
    pub platform: String,
    /// Model format (e.g., "onnx", "safetensors")
    pub format: String,
    /// Quantization level (e.g., "fp16", "fp32", "int8")
    pub quantization: String,
    /// Bundle size in bytes
    pub size_bytes: u64,
    /// HuggingFace repository
    pub hf_repo: String,
    /// Bundle filename
    pub file: String,
}

/// Response from GET /v1/models/registry/{mask}/resolve
#[derive(Debug, Deserialize)]
struct ResolveResponse {
    #[allow(dead_code)]
    mask: String,
    #[allow(dead_code)]
    platform: String,
    resolved: ResolvedVariant,
}

/// Resolved variant ready for download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedVariant {
    /// HuggingFace repository
    pub hf_repo: String,
    /// Bundle filename
    pub file: String,
    /// Direct download URL
    pub download_url: String,
    /// Model format
    pub format: String,
    /// Quantization level
    pub quantization: String,
    /// Bundle size in bytes
    pub size_bytes: u64,
    /// SHA256 hash for verification
    pub sha256: String,
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total size of cached bundles in bytes
    pub total_size_bytes: u64,
    /// Number of cached models
    pub model_count: usize,
    /// Path to cache directory
    pub cache_path: PathBuf,
}

impl CacheStats {
    /// Get human-readable size.
    pub fn total_size_human(&self) -> String {
        let bytes = self.total_size_bytes;
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_client() {
        let client = RegistryClient::default_client().unwrap();
        assert_eq!(client.api_url, DEFAULT_REGISTRY_URL);
    }

    #[test]
    fn test_cache_path() {
        let client = RegistryClient::default_client().unwrap();
        let resolved = ResolvedVariant {
            hf_repo: "xybrid-ai/kokoro-82m".to_string(),
            file: "universal.xyb".to_string(),
            download_url: "https://example.com/bundle.xyb".to_string(),
            format: "onnx".to_string(),
            quantization: "fp16".to_string(),
            size_bytes: 100000,
            sha256: "abc123".to_string(),
        };
        let path = client.get_cache_path(&resolved);
        assert!(path.to_string_lossy().contains("kokoro-82m"));
        assert!(path.to_string_lossy().contains("universal.xyb"));
    }
}
