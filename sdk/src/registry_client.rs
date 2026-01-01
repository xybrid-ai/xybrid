//! Registry client for fetching models from api.xybrid.dev.
//!
//! This module provides:
//! - `RegistryClient`: High-level API for model resolution and download
//! - Mask-based model lookup with platform resolution
//! - SHA256 hash verification
//! - Download progress callbacks
//! - Automatic retry with exponential backoff
//! - Circuit breaker for failing endpoints
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
use std::sync::Arc;
use std::time::Duration;
use xybrid_core::http::{CircuitBreaker, CircuitConfig, RetryPolicy, RetryableError};

/// Default registry API URL.
pub const DEFAULT_REGISTRY_URL: &str = "https://api.xybrid.dev";

/// Connection timeout in milliseconds.
const CONNECT_TIMEOUT_MS: u64 = 10000;

/// Request timeout in milliseconds.
const REQUEST_TIMEOUT_MS: u64 = 30000;

/// Registry client for model resolution and download.
pub struct RegistryClient {
    /// Base URL for the registry API
    api_url: String,
    /// Cache manager for storing downloaded bundles
    cache: CacheManager,
    /// HTTP agent with timeouts configured
    agent: ureq::Agent,
    /// Circuit breaker for the registry API
    circuit: Arc<CircuitBreaker>,
    /// Retry policy for API calls
    retry_policy: RetryPolicy,
}

impl RegistryClient {
    /// Create a new registry client with the specified API URL.
    pub fn new(api_url: impl Into<String>) -> Result<Self, SdkError> {
        // Create HTTP agent with timeouts
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(CONNECT_TIMEOUT_MS))
            .timeout(Duration::from_millis(REQUEST_TIMEOUT_MS))
            .build();

        Ok(Self {
            api_url: api_url.into(),
            cache: CacheManager::new()?,
            agent,
            circuit: Arc::new(CircuitBreaker::new(CircuitConfig::default())),
            retry_policy: RetryPolicy::default(),
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

    /// Check if the circuit breaker is open (blocking requests).
    pub fn is_circuit_open(&self) -> bool {
        self.circuit.is_open()
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset_circuit(&self) {
        self.circuit.reset();
    }

    /// List all available models in the registry.
    ///
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn list_models(&self) -> Result<Vec<ModelSummary>, SdkError> {
        let url = format!("{}/v1/models/registry", self.api_url);

        self.execute_with_retry(|| {
            let response = self.agent.get(&url).call();
            self.handle_response(response, "list models")
        })
        .and_then(|response| {
            let list_response: ListModelsResponse = response
                .into_json()
                .map_err(|e| SdkError::NetworkError(format!("Failed to parse response: {}", e)))?;
            Ok(list_response.models)
        })
    }

    /// Get detailed information about a specific model.
    ///
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn get_model(&self, mask: &str) -> Result<ModelDetail, SdkError> {
        let url = format!("{}/v1/models/registry/{}", self.api_url, mask);

        self.execute_with_retry(|| {
            let response = self.agent.get(&url).call();
            self.handle_response_with_404(response, "get model", || {
                SdkError::ModelNotFound(format!("Model '{}' not found", mask))
            })
        })
        .and_then(|response| {
            response
                .into_json()
                .map_err(|e| SdkError::NetworkError(format!("Failed to parse response: {}", e)))
        })
    }

    /// Resolve a model mask to the best variant for the given platform.
    ///
    /// If platform is None, auto-detects the current platform.
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn resolve(&self, mask: &str, platform: Option<&str>) -> Result<ResolvedVariant, SdkError> {
        let platform = platform
            .map(String::from)
            .unwrap_or_else(detect_platform);

        let url = format!(
            "{}/v1/models/registry/{}/resolve?platform={}",
            self.api_url, mask, platform
        );

        self.execute_with_retry(|| {
            let response = self.agent.get(&url).call();
            self.handle_response_with_404(response, "resolve model", || {
                SdkError::ModelNotFound(format!(
                    "Model '{}' not found or no compatible variant for platform '{}'",
                    mask, platform
                ))
            })
        })
        .and_then(|response| {
            let resolve_response: ResolveResponse = response
                .into_json()
                .map_err(|e| SdkError::NetworkError(format!("Failed to parse response: {}", e)))?;
            Ok(resolve_response.resolved)
        })
    }

    /// Execute an operation with retry and circuit breaker.
    fn execute_with_retry<T, F>(&self, mut operation: F) -> Result<T, SdkError>
    where
        F: FnMut() -> Result<T, SdkError>,
    {
        // Check circuit breaker before starting
        if !self.circuit.can_execute() {
            return Err(SdkError::CircuitOpen(
                "Registry API circuit breaker is open, try again later".to_string(),
            ));
        }

        let mut last_error: Option<SdkError> = None;

        for attempt in 0..self.retry_policy.max_attempts {
            // Calculate delay for this attempt
            let delay = if let Some(ref err) = last_error {
                err.retry_after()
                    .unwrap_or_else(|| self.retry_policy.delay_for_attempt(attempt))
            } else {
                self.retry_policy.delay_for_attempt(attempt)
            };

            if !delay.is_zero() {
                std::thread::sleep(delay);
            }

            // Check circuit breaker again (might have opened)
            if !self.circuit.can_execute() {
                return Err(SdkError::CircuitOpen(
                    "Registry API circuit breaker is open, try again later".to_string(),
                ));
            }

            match operation() {
                Ok(result) => {
                    self.circuit.record_success();
                    return Ok(result);
                }
                Err(err) => {
                    self.circuit.record_failure();

                    // Check for rate limit (opens circuit immediately)
                    if let SdkError::RateLimited { .. } = &err {
                        self.circuit.record_rate_limited();
                    }

                    // Don't retry non-retryable errors
                    if !err.is_retryable() {
                        return Err(err);
                    }

                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            SdkError::NetworkError("All retry attempts exhausted".to_string())
        }))
    }

    /// Handle HTTP response, converting errors appropriately.
    fn handle_response(
        &self,
        response: Result<ureq::Response, ureq::Error>,
        operation: &str,
    ) -> Result<ureq::Response, SdkError> {
        match response {
            Ok(resp) => {
                if resp.status() == 200 {
                    Ok(resp)
                } else {
                    Err(self.status_to_error(resp.status(), operation))
                }
            }
            Err(e) => Err(self.ureq_error_to_sdk_error(e, operation)),
        }
    }

    /// Handle HTTP response with special 404 handling.
    fn handle_response_with_404<F>(
        &self,
        response: Result<ureq::Response, ureq::Error>,
        operation: &str,
        not_found_err: F,
    ) -> Result<ureq::Response, SdkError>
    where
        F: FnOnce() -> SdkError,
    {
        match response {
            Ok(resp) => {
                if resp.status() == 200 {
                    Ok(resp)
                } else if resp.status() == 404 {
                    Err(not_found_err())
                } else {
                    Err(self.status_to_error(resp.status(), operation))
                }
            }
            Err(ureq::Error::Status(404, _)) => Err(not_found_err()),
            Err(e) => Err(self.ureq_error_to_sdk_error(e, operation)),
        }
    }

    /// Convert HTTP status code to SdkError.
    fn status_to_error(&self, status: u16, operation: &str) -> SdkError {
        match status {
            429 => {
                // TODO: Parse Retry-After header when available
                SdkError::RateLimited {
                    retry_after_secs: 60,
                }
            }
            502 | 503 | 504 => SdkError::NetworkError(format!(
                "Registry {} failed with status {} (server error)",
                operation, status
            )),
            400 | 401 | 403 | 422 => SdkError::ConfigError(format!(
                "Registry {} failed with status {} (client error)",
                operation, status
            )),
            _ => SdkError::NetworkError(format!(
                "Registry {} returned status {}",
                operation, status
            )),
        }
    }

    /// Convert ureq error to SdkError.
    fn ureq_error_to_sdk_error(&self, error: ureq::Error, operation: &str) -> SdkError {
        match error {
            ureq::Error::Status(status, _) => self.status_to_error(status, operation),
            ureq::Error::Transport(transport) => {
                let kind = transport.kind();
                match kind {
                    ureq::ErrorKind::Dns => SdkError::NetworkError(format!(
                        "Failed to {} (DNS resolution failed)",
                        operation
                    )),
                    ureq::ErrorKind::ConnectionFailed => SdkError::NetworkError(format!(
                        "Failed to {} (connection failed)",
                        operation
                    )),
                    ureq::ErrorKind::Io => SdkError::NetworkError(format!(
                        "Failed to {} (I/O error: {})",
                        operation,
                        transport.message().unwrap_or("unknown")
                    )),
                    _ => SdkError::NetworkError(format!("Failed to {}: {}", operation, transport)),
                }
            }
        }
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

    /// Download a file with progress tracking and retry on connection failures.
    ///
    /// Note: Downloads use a separate retry mechanism because:
    /// 1. HuggingFace is a different endpoint than the registry API
    /// 2. Large file downloads need longer timeouts
    /// 3. We don't want a failed HuggingFace download to trip the registry circuit breaker
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
        // Use a more conservative retry policy for downloads (longer delays)
        let download_policy = RetryPolicy::conservative();
        let mut last_error: Option<SdkError> = None;

        for attempt in 0..download_policy.max_attempts {
            // Calculate delay
            let delay = if let Some(ref err) = last_error {
                err.retry_after()
                    .unwrap_or_else(|| download_policy.delay_for_attempt(attempt))
            } else {
                download_policy.delay_for_attempt(attempt)
            };

            if !delay.is_zero() {
                std::thread::sleep(delay);
            }

            match self.try_download(url, dest, total_size, &progress_callback) {
                Ok(()) => return Ok(()),
                Err(err) => {
                    if !err.is_retryable() {
                        return Err(err);
                    }
                    // Clean up partial file before retry
                    std::fs::remove_file(dest).ok();
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            SdkError::NetworkError("Download failed after all retry attempts".to_string())
        }))
    }

    /// Attempt a single download.
    fn try_download<F>(
        &self,
        url: &str,
        dest: &PathBuf,
        total_size: u64,
        progress_callback: &F,
    ) -> Result<(), SdkError>
    where
        F: Fn(f32),
    {
        // Use a longer timeout for downloads (5 minutes for large models)
        let download_agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(CONNECT_TIMEOUT_MS))
            .timeout(Duration::from_secs(300)) // 5 minute timeout for downloads
            .build();

        let response = download_agent
            .get(url)
            .call()
            .map_err(|e| self.ureq_error_to_sdk_error(e, "download bundle"))?;

        if response.status() != 200 {
            return Err(self.status_to_error(response.status(), "download bundle"));
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
