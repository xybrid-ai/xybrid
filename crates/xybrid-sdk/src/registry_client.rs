//! Registry client for fetching models from registry.xybrid.dev.
//!
//! This module provides:
//! - `RegistryClient`: High-level API for model resolution and download
//! - Mask-based model lookup with platform resolution
//! - SHA256 hash verification
//! - Download progress callbacks
//! - Automatic retry with exponential backoff
//! - Circuit breaker for failing endpoints
//! - **Dual-endpoint failover** (primary: registry.xybrid.dev, fallback: r2.xybrid.dev)
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
use log::{debug, info};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use xybrid_core::http::{CircuitBreaker, CircuitConfig, RetryPolicy, RetryableError};

pub const DEFAULT_REGISTRY_URL: &str = "https://registry.xybrid.dev";
pub const FALLBACK_REGISTRY_URL: &str = "https://r2.xybrid.dev";

/// All registry URLs in priority order.
pub const REGISTRY_URLS: &[&str] = &[DEFAULT_REGISTRY_URL, FALLBACK_REGISTRY_URL];

/// Connection timeout in milliseconds.
const CONNECT_TIMEOUT_MS: u64 = 5000;

/// Request timeout in milliseconds.
const REQUEST_TIMEOUT_MS: u64 = 15000;

/// Registry client for model resolution and download.
pub struct RegistryClient {
    /// Registry URLs in priority order (primary first, then fallbacks)
    api_urls: Vec<String>,
    /// Cache manager for storing downloaded bundles
    cache: CacheManager,
    /// HTTP agent with timeouts configured
    agent: ureq::Agent,
    /// Circuit breakers for each registry URL
    circuits: Vec<Arc<CircuitBreaker>>,
    /// Retry policy for API calls
    retry_policy: RetryPolicy,
}

impl RegistryClient {
    /// Create a new registry client with the specified API URLs (primary first).
    pub fn new(api_urls: Vec<String>) -> Result<Self, SdkError> {
        if api_urls.is_empty() {
            return Err(SdkError::ConfigError(
                "No registry URLs provided".to_string(),
            ));
        }

        // Create HTTP agent with timeouts
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(CONNECT_TIMEOUT_MS))
            .timeout(Duration::from_millis(REQUEST_TIMEOUT_MS))
            .build();

        let cache = CacheManager::new()?;

        // Create circuit breakers for each URL
        let circuits: Vec<Arc<CircuitBreaker>> = api_urls
            .iter()
            .map(|_| Arc::new(CircuitBreaker::new(CircuitConfig::default())))
            .collect();

        debug!(
            "RegistryClient created with {} URLs, cache_dir={}",
            api_urls.len(),
            cache.cache_dir().display()
        );

        Ok(Self {
            api_urls,
            cache,
            agent,
            circuits,
            retry_policy: RetryPolicy::default(),
        })
    }

    /// Create a new registry client with a single API URL.
    pub fn with_url(api_url: impl Into<String>) -> Result<Self, SdkError> {
        Self::new(vec![api_url.into()])
    }

    /// Create a registry client with default URLs (primary + fallback).
    pub fn default_client() -> Result<Self, SdkError> {
        Self::new(REGISTRY_URLS.iter().map(|s| s.to_string()).collect())
    }

    /// Create a registry client from environment variable or defaults.
    ///
    /// Checks `XYBRID_REGISTRY_URL` environment variable first.
    /// If set, uses only that URL. Otherwise uses default URLs with fallback.
    pub fn from_env() -> Result<Self, SdkError> {
        if let Ok(url) = std::env::var("XYBRID_REGISTRY_URL") {
            // User specified a custom URL, use only that
            Self::with_url(url)
        } else {
            // Use default URLs with fallback
            Self::default_client()
        }
    }

    /// Get the primary API URL.
    pub fn primary_url(&self) -> &str {
        &self.api_urls[0]
    }

    /// Check if any circuit breaker is allowing requests.
    pub fn is_circuit_open(&self) -> bool {
        self.circuits.iter().all(|c| c.is_open())
    }

    /// Reset all circuit breakers to closed state.
    pub fn reset_circuit(&self) {
        for circuit in &self.circuits {
            circuit.reset();
        }
    }

    /// List all available models in the registry.
    ///
    /// Tries primary URL first, falls back to secondary on failure.
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn list_models(&self) -> Result<Vec<ModelSummary>, SdkError> {
        self.execute_with_fallback(|api_url| {
            let url = format!("{}/v1/models", api_url);
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
    /// Tries primary URL first, falls back to secondary on failure.
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn get_model(&self, mask: &str) -> Result<ModelDetail, SdkError> {
        self.execute_with_fallback(|api_url| {
            let url = format!("{}/v1/models/{}", api_url, mask);
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
    /// Tries primary URL first, falls back to secondary on failure.
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn resolve(&self, mask: &str, platform: Option<&str>) -> Result<ResolvedVariant, SdkError> {
        let platform = platform.map(String::from).unwrap_or_else(detect_platform);

        self.execute_with_fallback(|api_url| {
            let url = format!(
                "{}/v1/models/{}/resolve?platform={}",
                api_url, mask, platform
            );
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

    /// Execute an operation with fallback to secondary URLs.
    ///
    /// Tries each URL in order until one succeeds or all fail.
    fn execute_with_fallback<T, F>(&self, mut operation: F) -> Result<T, SdkError>
    where
        F: FnMut(&str) -> Result<T, SdkError>,
    {
        let mut last_error: Option<SdkError> = None;

        for (idx, api_url) in self.api_urls.iter().enumerate() {
            let circuit = &self.circuits[idx];

            // Skip if circuit is open
            if !circuit.can_execute() {
                debug!("Skipping {} (circuit open)", api_url);
                continue;
            }

            match self.execute_with_retry_for_url(api_url, circuit, &mut operation) {
                Ok(result) => {
                    if idx > 0 {
                        info!("Request succeeded using fallback URL: {}", api_url);
                    }
                    return Ok(result);
                }
                Err(err) => {
                    // Don't try fallback for non-retryable errors (like 404)
                    if !err.is_retryable() {
                        return Err(err);
                    }
                    debug!("URL {} failed: {}, trying next", api_url, err);
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            SdkError::NetworkError("All registry URLs failed or circuits open".to_string())
        }))
    }

    /// Execute an operation with retry for a specific URL.
    fn execute_with_retry_for_url<T, F>(
        &self,
        api_url: &str,
        circuit: &Arc<CircuitBreaker>,
        operation: &mut F,
    ) -> Result<T, SdkError>
    where
        F: FnMut(&str) -> Result<T, SdkError>,
    {
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
            if !circuit.can_execute() {
                return Err(SdkError::CircuitOpen(format!(
                    "Circuit breaker open for {}",
                    api_url
                )));
            }

            match operation(api_url) {
                Ok(result) => {
                    circuit.record_success();
                    return Ok(result);
                }
                Err(err) => {
                    circuit.record_failure();

                    // Check for rate limit (opens circuit immediately)
                    if let SdkError::RateLimited { .. } = &err {
                        circuit.record_rate_limited();
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
            SdkError::NetworkError(format!("All retry attempts exhausted for {}", api_url))
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
            502..=504 => SdkError::NetworkError(format!(
                "Registry {} failed with status {} (server error)",
                operation, status
            )),
            400 | 401 | 403 | 422 => SdkError::ConfigError(format!(
                "Registry {} failed with status {} (client error)",
                operation, status
            )),
            _ => {
                SdkError::NetworkError(format!("Registry {} returned status {}", operation, status))
            }
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
            .next_back()
            .unwrap_or(&resolved.hf_repo);

        self.cache.cache_dir().join(model_name).join(&resolved.file)
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

        debug!(
            "Cache check for '{}': path={}, exists={}, sha256_provided={}",
            mask,
            cache_path.display(),
            cache_path.exists(),
            !resolved.sha256.is_empty()
        );

        // Check if already cached with correct hash
        if cache_path.exists() && !resolved.sha256.is_empty() {
            // Try fast path: read cached hash from sidecar file
            let hash = match read_cached_hash(&cache_path) {
                Some(cached_hash) => {
                    debug!("Using cached hash for '{}'", mask);
                    cached_hash
                }
                None => {
                    // Fall back to computing hash (slow for large files)
                    debug!("Computing hash for '{}' (no cached hash found)", mask);
                    let computed = compute_sha256(&cache_path)?;
                    // Cache the hash for next time
                    write_cached_hash(&cache_path, &computed);
                    computed
                }
            };

            debug!(
                "Cache verification for '{}': expected={}, actual={}",
                mask, resolved.sha256, hash
            );
            if hash == resolved.sha256 {
                // Already cached and verified
                info!("Cache hit for '{}' at {}", mask, cache_path.display());
                return Ok(cache_path);
            }
            // Hash mismatch - re-download
            info!("Cache hash mismatch for '{}', re-downloading", mask);
            std::fs::remove_file(&cache_path).ok();
            remove_cached_hash(&cache_path);
        } else if cache_path.exists() {
            info!(
                "Cache exists for '{}' but no sha256 to verify, re-downloading",
                mask
            );
        } else {
            info!(
                "Cache miss for '{}', downloading to {}",
                mask,
                cache_path.display()
            );
        }

        // Create cache directory
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Download from HuggingFace
        info!("Downloading '{}' from {}", mask, resolved.download_url);
        self.download_with_progress(
            &resolved.download_url,
            &cache_path,
            resolved.size_bytes,
            progress_callback,
        )?;

        // Verify hash and cache it for fast future lookups
        if !resolved.sha256.is_empty() {
            let hash = compute_sha256(&cache_path)?;
            if hash != resolved.sha256 {
                std::fs::remove_file(&cache_path).ok();
                return Err(SdkError::CacheError(format!(
                    "SHA256 mismatch: expected {}, got {}",
                    resolved.sha256, hash
                )));
            }
            // Cache the verified hash for instant verification next time
            write_cached_hash(&cache_path, &hash);
            info!(
                "Download complete for '{}', SHA256 verified, cached at {}",
                mask,
                cache_path.display()
            );
        } else {
            info!(
                "Download complete for '{}' (no SHA256 verification), cached at {}",
                mask,
                cache_path.display()
            );
        }

        Ok(cache_path)
    }

    /// Fetch a model bundle and extract it, returning the extracted directory path.
    ///
    /// This is the **preferred method** for fetching models, as it returns a ready-to-use
    /// directory containing the model files and `model_metadata.json`.
    ///
    /// Extraction is idempotent: if the bundle was already extracted, returns immediately.
    ///
    /// # Arguments
    ///
    /// * `mask` - Model mask (e.g., "kokoro-82m")
    /// * `platform` - Target platform (None for auto-detect)
    /// * `progress_callback` - Optional callback for download progress (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Path to the extracted directory containing `model_metadata.json` and model files.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = RegistryClient::default_client()?;
    /// let model_dir = client.fetch_extracted("kokoro-82m", None, |p| {
    ///     println!("Downloaded: {:.1}%", p * 100.0);
    /// })?;
    ///
    /// // model_dir now contains model_metadata.json and all model files
    /// let metadata_path = model_dir.join("model_metadata.json");
    /// ```
    pub fn fetch_extracted<F>(
        &self,
        mask: &str,
        platform: Option<&str>,
        progress_callback: F,
    ) -> Result<PathBuf, SdkError>
    where
        F: Fn(f32),
    {
        // First, ensure the bundle is downloaded
        let xyb_path = self.fetch(mask, platform, progress_callback)?;

        // Then ensure it's extracted (idempotent - returns immediately if already done)
        self.cache.ensure_extracted(&xyb_path)
    }

    /// Check if a model is already extracted and ready to use.
    ///
    /// Returns true if the model has been fetched AND extracted.
    pub fn is_extracted(&self, model_id: &str) -> bool {
        self.cache.is_extracted(model_id)
    }

    /// Get the extraction directory for a model.
    ///
    /// Note: This returns the path even if not yet extracted. Use `is_extracted()` to check.
    pub fn extraction_dir(&self, model_id: &str) -> PathBuf {
        self.cache.extraction_dir(model_id)
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
        self.cache
            .clear()
            .map_err(|e| SdkError::CacheError(e.to_string()))?;
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

/// Get the path to the cached hash sidecar file.
fn hash_cache_path(bundle_path: &PathBuf) -> PathBuf {
    bundle_path.with_extension("xyb.sha256")
}

/// Read cached hash from sidecar file if it exists and is still valid.
///
/// Returns None if:
/// - Sidecar file doesn't exist
/// - Sidecar file is older than the bundle file (bundle was modified)
/// - Sidecar file can't be read
fn read_cached_hash(bundle_path: &PathBuf) -> Option<String> {
    let hash_path = hash_cache_path(bundle_path);

    // Check if sidecar exists
    if !hash_path.exists() {
        return None;
    }

    // Check if bundle is newer than sidecar (invalidates cache)
    let bundle_mtime = std::fs::metadata(bundle_path).ok()?.modified().ok()?;
    let hash_mtime = std::fs::metadata(&hash_path).ok()?.modified().ok()?;
    if bundle_mtime > hash_mtime {
        // Bundle was modified after hash was cached
        return None;
    }

    // Read and validate hash format (64 hex chars)
    let hash = std::fs::read_to_string(&hash_path).ok()?;
    let hash = hash.trim();
    if hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(hash.to_string())
    } else {
        None
    }
}

/// Write hash to sidecar file for fast future lookups.
fn write_cached_hash(bundle_path: &PathBuf, hash: &str) {
    let hash_path = hash_cache_path(bundle_path);
    // Ignore errors - this is just an optimization
    let _ = std::fs::write(&hash_path, hash);
}

/// Remove the cached hash sidecar file.
fn remove_cached_hash(bundle_path: &PathBuf) {
    let hash_path = hash_cache_path(bundle_path);
    let _ = std::fs::remove_file(&hash_path);
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
        assert_eq!(client.api_urls.len(), 2);
        assert_eq!(client.primary_url(), DEFAULT_REGISTRY_URL);
    }

    #[test]
    fn test_single_url_client() {
        let client = RegistryClient::with_url("https://custom.example.com").unwrap();
        assert_eq!(client.api_urls.len(), 1);
        assert_eq!(client.primary_url(), "https://custom.example.com");
    }

    #[test]
    fn test_registry_urls_constant() {
        assert_eq!(REGISTRY_URLS.len(), 2);
        assert_eq!(REGISTRY_URLS[0], DEFAULT_REGISTRY_URL);
        assert_eq!(REGISTRY_URLS[1], FALLBACK_REGISTRY_URL);
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

    #[test]
    fn test_extraction_dir() {
        let client = RegistryClient::default_client().unwrap();
        let dir = client.extraction_dir("test-model");
        assert!(dir.to_string_lossy().contains("extracted"));
        assert!(dir.to_string_lossy().contains("test-model"));
    }

    #[test]
    fn test_is_extracted_false_for_nonexistent() {
        let client = RegistryClient::default_client().unwrap();
        // A random model ID should not be extracted
        assert!(!client.is_extracted("nonexistent-model-12345"));
    }
}
