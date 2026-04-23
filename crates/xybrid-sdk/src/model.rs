//! Model loading and execution for xybrid-sdk.
//!
//! This module provides:
//! - `ModelLoader`: Preparatory step for loading models (from registry, bundle, or directory)
//! - `XybridModel`: Loaded model ready for inference
//! - `ModelHandle`: Internal state management for the loaded model
//! - `StreamEvent`: Events emitted during streaming inference

use crate::registry_client::RegistryClient;
use crate::result::{InferenceResult, OutputType};
use crate::source::{detect_platform, ModelSource};
use crate::stream::XybridStream;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tempfile::TempDir;
use tokio_stream::wrappers::ReceiverStream;
use xybrid_core::conversation::ConversationContext;
use xybrid_core::execution::{
    ExecutionTemplate, ModelMetadata, TemplateExecutor, VoiceConfig, VoiceInfo,
};
use xybrid_core::ir::Envelope;
use xybrid_core::runtime_adapter::types::GenerationConfig;
use xybrid_core::streaming::{StreamConfig as CoreStreamConfig, VadStreamConfig as CoreVadConfig};

/// A token generated during streaming inference.
///
/// This is the SDK's version of the core `PartialToken`, re-exported for convenience.
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// The generated token text
    pub token: String,
    /// The token ID (if available from the model)
    pub token_id: Option<i64>,
    /// Index of this token in the generation sequence
    pub index: usize,
    /// All text generated so far (cumulative)
    pub cumulative_text: String,
    /// Reason for stopping (only set on the final token)
    pub finish_reason: Option<String>,
}

/// Events emitted during streaming inference.
///
/// Use this with `run_stream()` to handle tokens as they're generated.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A token was generated (emitted for each token during LLM inference)
    Token(StreamToken),
    /// Inference completed successfully with final result
    Complete(InferenceResult),
    /// An error occurred during inference
    Error(String),
}

/// SDK-level error type.
#[derive(Debug, thiserror::Error)]
pub enum SdkError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),
    #[error("model_metadata.json not found in directory: {0}")]
    MetadataNotFound(String),
    #[error("model_metadata.json is invalid: {0}")]
    MetadataInvalid(String),
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Inference failed: {0}")]
    InferenceError(String),
    #[error("Streaming not supported by this model")]
    StreamingNotSupported,
    #[error("Model not loaded")]
    NotLoaded,
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    /// The registry could not be reached at all (DNS failure, connection refused,
    /// network unreachable, interface down). This is distinct from `NetworkError`
    /// because it represents *local* unreachability rather than a server-side problem,
    /// and the circuit breaker treats it differently — offline errors don't count
    /// toward the failure threshold so callers aren't punished for being offline.
    #[error("Registry unreachable: {0}")]
    Offline(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Pipeline error: {0}")]
    PipelineError(String),
    #[error("Circuit breaker open: {0}")]
    CircuitOpen(String),
    #[error("Rate limited, retry after {retry_after_secs} seconds")]
    RateLimited { retry_after_secs: u64 },
    #[error("Request timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
}

/// Result type for SDK operations.
pub type SdkResult<T> = Result<T, SdkError>;

impl xybrid_core::http::RetryableError for SdkError {
    fn is_retryable(&self) -> bool {
        match self {
            // Retryable errors (transient failures)
            SdkError::NetworkError(_) => true,
            SdkError::RateLimited { .. } => true,
            SdkError::Timeout { .. } => true,
            // Offline is "retryable" only across URLs — the fallback registry
            // may be reachable even when the primary isn't. Within a single URL
            // the retry loop short-circuits immediately (see registry_client).
            SdkError::Offline(_) => true,

            // Non-retryable errors (permanent failures)
            SdkError::ModelNotFound(_) => false,
            SdkError::DirectoryNotFound(_) => false,
            SdkError::MetadataNotFound(_) => false,
            SdkError::MetadataInvalid(_) => false,
            SdkError::LoadError(_) => false,
            SdkError::InferenceError(_) => false,
            SdkError::StreamingNotSupported => false,
            SdkError::NotLoaded => false,
            SdkError::ConfigError(_) => false,
            SdkError::IoError(_) => false,
            SdkError::CacheError(_) => false,
            SdkError::PipelineError(_) => false,
            SdkError::CircuitOpen(_) => false, // Don't retry when circuit is open
        }
    }

    fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            SdkError::RateLimited { retry_after_secs } => {
                Some(std::time::Duration::from_secs(*retry_after_secs))
            }
            _ => None,
        }
    }
}

/// Configuration for streaming ASR sessions.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Enable VAD (Voice Activity Detection) for smart chunking
    pub enable_vad: bool,
    /// VAD threshold (0.0-1.0)
    pub vad_threshold: f32,
    /// Language hint for ASR
    pub language: Option<String>,
    /// Path to VAD model (uses default if None)
    pub vad_model_dir: Option<String>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            enable_vad: false,
            vad_threshold: 0.5,
            language: Some("en".to_string()),
            vad_model_dir: None,
        }
    }
}

impl StreamConfig {
    /// Create config with VAD enabled.
    pub fn with_vad() -> Self {
        Self {
            enable_vad: true,
            ..Default::default()
        }
    }

    /// Set language hint.
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set VAD threshold.
    pub fn vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold;
        self
    }
}

/// Internal handle holding the loaded model state.
struct ModelHandle {
    /// Template executor for running inference
    executor: TemplateExecutor,
    /// Model metadata
    metadata: ModelMetadata,
    /// Model directory path (permanent extraction in cache)
    model_dir: PathBuf,
    /// Whether model is currently loaded
    loaded: bool,
}

/// Represents a model that can be loaded.
///
/// Created by `Xybrid::model()`, must call `.load()` to use.
/// This is a preparatory step that doesn't download or load anything.
///
/// # Example (Recommended - Registry-based)
///
/// ```ignore
/// // Load using registry (recommended - auto-resolves to best variant)
/// let loader = ModelLoader::from_registry("kokoro-82m");
/// let model = loader.load()?;
/// let result = model.run(&envelope)?;
/// ```
///
/// # Example (With progress callback)
///
/// ```ignore
/// let loader = ModelLoader::from_registry("kokoro-82m");
/// let model = loader.load_with_progress(|progress| {
///     println!("Download: {:.1}%", progress * 100.0);
/// })?;
/// ```
/// GGUF quantization preference order for automatic selection.
/// Q4_K_M is the default — best quality/size tradeoff for edge devices.
const GGUF_PREFERENCE_ORDER: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "BF16", "F32",
];

/// Select the best GGUF file from a list based on user preference or default ranking.
///
/// If `variant` is specified, finds a file containing that quantization string (case-insensitive).
/// Otherwise, selects the file matching the highest-priority quantization from `GGUF_PREFERENCE_ORDER`.
fn select_gguf_variant(gguf_files: &[&str], variant: Option<&str>) -> SdkResult<String> {
    if let Some(v) = variant {
        let v_upper = v.to_uppercase();
        // Find a file containing the variant string (case-insensitive)
        if let Some(found) = gguf_files
            .iter()
            .find(|f| f.to_uppercase().contains(&v_upper))
        {
            return Ok(found.to_string());
        }
        return Err(SdkError::LoadError(format!(
            "No GGUF file matching variant '{}'. Available: {}",
            v,
            gguf_files.join(", ")
        )));
    }

    // Auto-select: try each preferred quantization in order
    for pref in GGUF_PREFERENCE_ORDER {
        if let Some(found) = gguf_files.iter().find(|f| f.to_uppercase().contains(pref)) {
            return Ok(found.to_string());
        }
    }

    // Fallback: pick the smallest file (likely the most quantized)
    Ok(gguf_files
        .first()
        .ok_or_else(|| SdkError::LoadError("No GGUF files found".to_string()))?
        .to_string())
}

#[derive(Debug, Clone)]
pub struct ModelLoader {
    source: ModelSource,
    model_id: Option<String>,
    version: Option<String>,
}

impl ModelLoader {
    /// Create loader from registry (recommended).
    ///
    /// Uses the registry API to resolve the model ID to the best variant
    /// for the current platform, then downloads from HuggingFace with
    /// caching and SHA256 verification.
    ///
    /// # Example
    /// ```ignore
    /// let loader = ModelLoader::from_registry("kokoro-82m");
    /// let model = loader.load()?;
    /// ```
    pub fn from_registry(id: &str) -> Self {
        Self {
            source: ModelSource::registry(id),
            model_id: Some(id.to_string()),
            version: None, // Version is resolved by registry API
        }
    }

    /// Create loader from registry with explicit platform.
    ///
    /// # Example
    /// ```ignore
    /// let loader = ModelLoader::from_registry_with_platform("kokoro-82m", "macos-arm64");
    /// let model = loader.load()?;
    /// ```
    pub fn from_registry_with_platform(id: &str, platform: &str) -> Self {
        Self {
            source: ModelSource::registry_with_platform(id, platform),
            model_id: Some(id.to_string()),
            version: None,
        }
    }

    /// Create loader from legacy registry with direct URL.
    ///
    /// # Deprecated
    /// Use `from_registry()` instead for automatic platform resolution and caching.
    #[deprecated(since = "0.0.17", note = "Use ModelLoader::from_registry() instead")]
    #[allow(deprecated)]
    pub fn from_legacy_registry(url: &str, model_id: &str, version: &str) -> Self {
        Self {
            source: ModelSource::legacy_registry(url, model_id, version),
            model_id: Some(model_id.to_string()),
            version: Some(version.to_string()),
        }
    }

    /// Create loader from legacy registry with explicit platform.
    ///
    /// # Deprecated
    /// Use `from_registry_with_platform()` instead.
    #[deprecated(
        since = "0.0.17",
        note = "Use ModelLoader::from_registry_with_platform() instead"
    )]
    #[allow(deprecated)]
    pub fn from_legacy_registry_with_platform(
        url: &str,
        model_id: &str,
        version: &str,
        platform: &str,
    ) -> Self {
        Self {
            source: ModelSource::legacy_registry_with_platform(url, model_id, version, platform),
            model_id: Some(model_id.to_string()),
            version: Some(version.to_string()),
        }
    }

    /// Create loader from local bundle file.
    pub fn from_bundle(path: impl Into<PathBuf>) -> SdkResult<Self> {
        let path: PathBuf = path.into();
        if !path.exists() {
            return Err(SdkError::ModelNotFound(format!(
                "Bundle not found: {:?}",
                path
            )));
        }
        Ok(Self {
            source: ModelSource::bundle(path),
            model_id: None,
            version: None,
        })
    }

    /// Create loader from local model directory.
    ///
    /// The directory must exist and contain a valid `model_metadata.json` file.
    ///
    /// # Errors
    ///
    /// - `SdkError::DirectoryNotFound` if the path does not exist
    /// - `SdkError::MetadataNotFound` if `model_metadata.json` is missing
    /// - `SdkError::MetadataInvalid` if `model_metadata.json` contains invalid JSON
    pub fn from_directory(path: impl Into<PathBuf>) -> SdkResult<Self> {
        let path: PathBuf = path.into();
        if !path.exists() {
            return Err(SdkError::DirectoryNotFound(path.display().to_string()));
        }
        let metadata_path = path.join("model_metadata.json");
        if !metadata_path.exists() {
            return Err(SdkError::MetadataNotFound(path.display().to_string()));
        }
        // Validate that the metadata is valid JSON
        let metadata_str = std::fs::read_to_string(&metadata_path).map_err(|e| {
            SdkError::MetadataInvalid(format!("failed to read model_metadata.json: {}", e))
        })?;
        let _metadata: xybrid_core::execution::ModelMetadata = serde_json::from_str(&metadata_str)
            .map_err(|e| {
                SdkError::MetadataInvalid(format!("invalid model_metadata.json: {}", e))
            })?;
        Ok(Self {
            source: ModelSource::directory(path),
            model_id: None,
            version: None,
        })
    }

    /// Create loader from a HuggingFace Hub repository.
    ///
    /// Downloads model files from the HuggingFace Hub and caches them locally.
    /// Subsequent calls use the cached files. The repository must contain a
    /// `model_metadata.json` for the model to be loadable (auto-generation
    /// is planned for a future version).
    ///
    /// Requires the `huggingface` feature flag at load time.
    /// The constructor itself is always available, but `load()` will return
    /// `SdkError::ConfigError` if the feature is not enabled.
    ///
    /// # Example
    /// ```ignore
    /// let loader = ModelLoader::from_huggingface("xybrid-ai/kokoro-82m");
    /// let model = loader.load()?;
    /// ```
    pub fn from_huggingface(repo: &str) -> Self {
        Self {
            source: ModelSource::huggingface(repo),
            model_id: Some(repo.to_string()),
            version: None,
        }
    }

    /// Create loader from a HuggingFace Hub repository with explicit revision.
    ///
    /// # Example
    /// ```ignore
    /// let loader = ModelLoader::from_huggingface_with_revision("xybrid-ai/kokoro-82m", "v1.0")?;
    /// let model = loader.load()?;
    /// ```
    pub fn from_huggingface_with_revision(repo: &str, revision: &str) -> Self {
        Self {
            source: ModelSource::huggingface_with_revision(repo, revision),
            model_id: Some(repo.to_string()),
            version: Some(revision.to_string()),
        }
    }

    /// Create loader from a HuggingFace repo string, parsing optional variant suffix.
    ///
    /// Supports `"org/repo:Q8_0"` syntax to select a specific GGUF quantization.
    /// Without a variant, defaults to Q4_K_M for GGUF repos.
    ///
    /// # Example
    /// ```ignore
    /// let loader = ModelLoader::from_huggingface_parsed("LiquidAI/LFM2.5-350M-GGUF:Q8_0");
    /// let model = loader.load()?;
    /// ```
    pub fn from_huggingface_parsed(input: &str) -> Self {
        let source = ModelSource::parse_huggingface(input);
        let repo = source.model_id().unwrap_or(input).to_string();
        Self {
            source,
            model_id: Some(repo),
            version: None,
        }
    }

    /// Get the model ID (if known).
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Get the version (if known).
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    /// Get the source type.
    pub fn source_type(&self) -> &'static str {
        self.source.source_type()
    }

    /// Load the model into memory (synchronous).
    ///
    /// This will:
    /// - For registry: Resolve via registry API, download from HuggingFace (with caching)
    /// - For legacy_registry (deprecated): Download the bundle if not cached, extract it
    /// - For bundle: Extract the bundle to a temp directory
    /// - For directory: Load directly from the directory
    ///
    /// Returns a loaded `XybridModel` ready for inference.
    #[allow(deprecated)]
    pub fn load(&self) -> SdkResult<XybridModel> {
        self.load_with_progress(|_| {})
    }

    /// Load the model with a progress callback.
    ///
    /// The callback receives progress as a float from 0.0 to 1.0.
    /// Only applies to registry-based loading (downloads from HuggingFace).
    ///
    /// # Example
    /// ```ignore
    /// let model = loader.load_with_progress(|progress| {
    ///     println!("Download: {:.1}%", progress * 100.0);
    /// })?;
    /// ```
    #[allow(deprecated)]
    pub fn load_with_progress<F>(&self, progress_callback: F) -> SdkResult<XybridModel>
    where
        F: Fn(f32),
    {
        match &self.source {
            ModelSource::Registry { id, platform } => {
                self.load_from_registry_api(id, platform.as_deref(), progress_callback)
            }
            ModelSource::LegacyRegistry {
                url,
                model_id,
                version,
                platform,
            } => self.load_from_legacy_registry(url, model_id, version, platform.as_deref()),
            ModelSource::Bundle { path } => self.load_from_bundle(path),
            ModelSource::Directory { path } => self.load_from_directory(path),
            ModelSource::HuggingFace {
                repo,
                revision,
                variant,
            } => self.load_from_huggingface(
                repo,
                revision.as_deref(),
                variant.as_deref(),
                progress_callback,
            ),
        }
    }

    /// Load the model asynchronously.
    pub async fn load_async(&self) -> SdkResult<XybridModel> {
        // For now, wrap the sync version. Real async would use tokio::fs and async HTTP.
        let loader = self.clone();
        tokio::task::spawn_blocking(move || loader.load())
            .await
            .map_err(|e| SdkError::LoadError(format!("Task join error: {}", e)))?
    }

    /// Load model from registry using RegistryClient.
    ///
    /// This is the recommended loading method - it uses the registry API to resolve
    /// the model ID to the best variant for the platform, downloads from HuggingFace,
    /// and caches locally with SHA256 verification.
    fn load_from_registry_api<F>(
        &self,
        id: &str,
        platform: Option<&str>,
        progress_callback: F,
    ) -> SdkResult<XybridModel>
    where
        F: Fn(f32),
    {
        // Create registry client (uses default API or environment variable)
        let client = RegistryClient::from_env()?;

        // Fetch and extract model (handles both .xyb bundles and passthrough GGUF files)
        let model_dir = client.fetch_extracted(id, platform, progress_callback)?;

        // Load from extracted directory
        self.load_from_directory(&model_dir)
    }

    /// Load from legacy registry (deprecated - use load_from_registry_api instead).
    fn load_from_legacy_registry(
        &self,
        url: &str,
        model_id: &str,
        version: &str,
        platform: Option<&str>,
    ) -> SdkResult<XybridModel> {
        let platform = platform.map(String::from).unwrap_or_else(detect_platform);

        // Build bundle URL
        let bundle_url = format!(
            "{}/bundles/{}/{}/{}/{}.xyb",
            url.trim_end_matches('/'),
            model_id,
            version,
            platform,
            model_id
        );

        // Download bundle to temp file
        let temp_dir = TempDir::new().map_err(SdkError::IoError)?;
        let bundle_path = temp_dir.path().join(format!("{}.xyb", model_id));

        // Use blocking HTTP client
        let response = ureq::get(&bundle_url)
            .call()
            .map_err(|e| SdkError::NetworkError(format!("Failed to download bundle: {}", e)))?;

        if response.status() != 200 {
            return Err(SdkError::ModelNotFound(format!(
                "Bundle not found at registry: {} (status {})",
                bundle_url,
                response.status()
            )));
        }

        // Write bundle to temp file
        let mut file = std::fs::File::create(&bundle_path)?;
        std::io::copy(&mut response.into_reader(), &mut file)?;

        // Extract using CacheManager (extracts to permanent cache location)
        // The temp_dir will be dropped after this, but extracted files persist
        self.load_from_bundle(&bundle_path)
    }

    fn load_from_bundle(&self, path: &PathBuf) -> SdkResult<XybridModel> {
        // Use CacheManager for unified extraction (single source of truth)
        let cache = crate::cache::CacheManager::new()?;
        let extract_dir = cache.ensure_extracted(path)?;

        // Load from extracted directory (extraction is permanent in cache)
        let handle = Self::create_model_handle(&extract_dir)?;

        let model_id = handle.metadata.model_id.clone();
        let version = handle.metadata.version.clone();
        let supports_streaming = Self::check_streaming_support(&handle.metadata);
        let output_type = Self::infer_output_type(&handle.metadata);

        Ok(XybridModel {
            handle: Arc::new(RwLock::new(handle)),
            model_id,
            version,
            output_type,
            supports_streaming,
        })
    }

    /// Load a model from HuggingFace Hub.
    ///
    /// Only downloads the selected GGUF variant (defaults to Q4_K_M) plus essential
    /// supporting files (config, tokenizer, README). Avoids downloading all variants.
    #[cfg(feature = "huggingface")]
    fn load_from_huggingface<F>(
        &self,
        repo: &str,
        revision: Option<&str>,
        variant: Option<&str>,
        _progress_callback: F,
    ) -> SdkResult<XybridModel>
    where
        F: Fn(f32),
    {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        // Determine our cache directory
        let cache_dir = Self::hf_cache_dir(repo)?;

        // Check if we already have a cached copy with model_metadata.json
        let metadata_path = cache_dir.join("model_metadata.json");
        if metadata_path.exists() {
            log::info!(target: "xybrid_sdk", "Loading HuggingFace model from cache: {}", cache_dir.display());
            return self.load_from_directory(&cache_dir);
        }

        log::info!(target: "xybrid_sdk", "Downloading model from HuggingFace: {}", repo);

        // Create HF API client
        let api = Api::new().map_err(|e| {
            SdkError::NetworkError(format!("Failed to create HuggingFace API client: {}", e))
        })?;

        // Create repo reference with optional revision
        let hf_repo = if let Some(rev) = revision {
            Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string())
        } else {
            Repo::new(repo.to_string(), RepoType::Model)
        };
        let repo_api = api.repo(hf_repo);

        // Get repo info to list all files
        let repo_info = repo_api.info().map_err(|e| {
            SdkError::NetworkError(format!(
                "Failed to get HuggingFace repo info for '{}': {}",
                repo, e
            ))
        })?;

        let siblings = repo_info.siblings;
        if siblings.is_empty() {
            return Err(SdkError::LoadError(format!(
                "HuggingFace repo '{}' has no files",
                repo
            )));
        }

        // Classify files by type to enable smart filtering
        let all_filenames: Vec<&str> = siblings.iter().map(|s| s.rfilename.as_str()).collect();
        let gguf_files: Vec<&str> = all_filenames
            .iter()
            .filter(|f| f.ends_with(".gguf"))
            .copied()
            .collect();

        // If multiple GGUF files exist, select the best one instead of downloading all
        let selected_gguf = if gguf_files.len() > 1 {
            Some(select_gguf_variant(&gguf_files, variant)?)
        } else {
            None
        };

        if let Some(ref selected) = selected_gguf {
            log::info!(
                target: "xybrid_sdk",
                "Selected GGUF variant: {} (from {} available)",
                selected, gguf_files.len()
            );
        }

        // Create cache directory
        std::fs::create_dir_all(&cache_dir)?;

        // Filter to only files we need
        let files_to_download: Vec<&str> = all_filenames
            .iter()
            .filter(|filename| {
                // Skip hidden files and directories
                if filename.starts_with('.') || filename.ends_with('/') {
                    return false;
                }

                // If we have a selected GGUF, skip other GGUF files
                if let Some(ref selected) = selected_gguf {
                    if filename.ends_with(".gguf") && **filename != *selected {
                        return false;
                    }
                }

                // Skip non-essential files (LICENSE, subdirectories like leap/)
                let dominated_by_model = selected_gguf.is_some() || gguf_files.len() == 1;
                if dominated_by_model {
                    Self::is_essential_file(filename)
                } else {
                    true
                }
            })
            .copied()
            .collect();

        let total_files = files_to_download.len();
        for (i, filename) in files_to_download.iter().enumerate() {
            log::debug!(target: "xybrid_sdk", "Downloading [{}/{}]: {}", i + 1, total_files, filename);

            // Report approximate progress
            _progress_callback((i as f32) / (total_files as f32));

            // Download file (hf-hub caches internally)
            let cached_path = repo_api.get(filename).map_err(|e| {
                SdkError::NetworkError(format!(
                    "Failed to download '{}' from '{}': {}",
                    filename, repo, e
                ))
            })?;

            // Create target path in our cache directory
            let target_path = cache_dir.join(filename);

            // Create parent directories if the file is in a subdirectory
            if let Some(parent) = target_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // Skip if already exists in our cache
            if target_path.exists() {
                continue;
            }

            // Create symlink to hf-hub's cached file (avoids duplication)
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(&cached_path, &target_path).map_err(|e| {
                    SdkError::IoError(std::io::Error::other(format!(
                        "Failed to symlink {} -> {}: {}",
                        cached_path.display(),
                        target_path.display(),
                        e
                    )))
                })?;
            }

            // On Windows, copy the file instead
            #[cfg(not(unix))]
            {
                std::fs::copy(&cached_path, &target_path)?;
            }
        }

        // Report completion
        _progress_callback(1.0);

        // Auto-generate model_metadata.json if not provided by the repo
        if !metadata_path.exists() {
            log::info!(
                target: "xybrid_sdk",
                "No model_metadata.json in repo '{}', attempting auto-generation...",
                repo
            );
            match crate::metadata_gen::generate_metadata(&cache_dir, repo) {
                Ok((_metadata, _task_inference)) => {
                    log::info!(
                        target: "xybrid_sdk",
                        "Auto-generated model_metadata.json for '{}'. \
                         Review and adjust if inference results are unexpected.",
                        repo
                    );
                }
                Err(e) => {
                    return Err(SdkError::MetadataNotFound(format!(
                        "HuggingFace repo '{}' does not contain model_metadata.json and \
                         auto-generation failed: {}. \
                         Create one manually — see docs/sdk/MODEL_METADATA.md",
                        repo, e
                    )));
                }
            }
        }

        log::info!(target: "xybrid_sdk", "Model cached at: {}", cache_dir.display());
        self.load_from_directory(&cache_dir)
    }

    /// Not available without the `huggingface` feature.
    #[cfg(not(feature = "huggingface"))]
    fn load_from_huggingface<F>(
        &self,
        _repo: &str,
        _revision: Option<&str>,
        _variant: Option<&str>,
        _progress_callback: F,
    ) -> SdkResult<XybridModel>
    where
        F: Fn(f32),
    {
        Err(SdkError::ConfigError(
            "HuggingFace loading requires the 'huggingface' feature flag. \
             Enable it with: cargo build --features huggingface"
                .to_string(),
        ))
    }

    /// Get the cache directory for a HuggingFace repo.
    ///
    /// Returns `~/.xybrid/cache/hf/{sanitized_repo}/` or the SDK-configured cache path.
    fn hf_cache_dir(repo: &str) -> SdkResult<PathBuf> {
        let base_cache = if let Some(sdk_cache) = crate::get_sdk_cache_dir() {
            sdk_cache.join("hf")
        } else {
            let home = dirs::home_dir().ok_or_else(|| {
                SdkError::CacheError("Cannot determine home directory".to_string())
            })?;
            home.join(".xybrid").join("cache").join("hf")
        };

        // Sanitize repo name for filesystem (e.g., "xybrid-ai/kokoro-82m" -> "xybrid-ai--kokoro-82m")
        let sanitized = repo.replace('/', "--");
        Ok(base_cache.join(sanitized))
    }

    /// Check if a file is essential and should always be downloaded.
    ///
    /// Essential files are model files (.gguf, .onnx, .safetensors), metadata files
    /// (config.json, tokenizer, vocab), and the README (for model card parsing).
    /// Non-essential files (LICENSE, leap/, etc.) are skipped.
    fn is_essential_file(filename: &str) -> bool {
        // Model files
        if filename.ends_with(".gguf")
            || filename.ends_with(".onnx")
            || filename.ends_with(".safetensors")
        {
            return true;
        }

        // Metadata and supporting files
        let basename = filename.rsplit('/').next().unwrap_or(filename);
        matches!(
            basename,
            "model_metadata.json"
                | "config.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
                | "vocab.json"
                | "vocab.txt"
                | "tokens.txt"
                | "merges.txt"
                | "preprocessor_config.json"
                | "generation_config.json"
                | "README.md"
        )
    }

    fn load_from_directory(&self, path: &PathBuf) -> SdkResult<XybridModel> {
        let handle = Self::create_model_handle(path)?;

        let model_id = handle.metadata.model_id.clone();
        let version = handle.metadata.version.clone();
        let supports_streaming = Self::check_streaming_support(&handle.metadata);
        let output_type = Self::infer_output_type(&handle.metadata);

        Ok(XybridModel {
            handle: Arc::new(RwLock::new(handle)),
            model_id,
            version,
            output_type,
            supports_streaming,
        })
    }

    fn create_model_handle(model_dir: &PathBuf) -> SdkResult<ModelHandle> {
        // Load metadata
        let metadata_path = model_dir.join("model_metadata.json");
        let metadata_str = std::fs::read_to_string(&metadata_path).map_err(|e| {
            SdkError::LoadError(format!("Failed to read model_metadata.json: {}", e))
        })?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_str)
            .map_err(|e| SdkError::LoadError(format!("Failed to parse metadata: {}", e)))?;

        // Create executor with base path
        let executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap_or("."));

        Ok(ModelHandle {
            executor,
            metadata,
            model_dir: model_dir.clone(),
            loaded: true,
        })
    }

    fn check_streaming_support(metadata: &ModelMetadata) -> bool {
        // Check if this is an ASR model (supports streaming)
        // Look at metadata task or model type (metadata is HashMap<String, serde_json::Value>)
        if let Some(task) = metadata.metadata.get("task").and_then(|v| v.as_str()) {
            if task == "speech-recognition" || task == "asr" {
                return true;
            }
        }

        // Check execution template type
        match &metadata.execution_template {
            ExecutionTemplate::SafeTensors { architecture, .. } => {
                architecture.as_deref() == Some("whisper")
            }
            ExecutionTemplate::Onnx { .. } => {
                // Check if preprocessing includes AudioDecode (likely ASR)
                metadata.preprocessing.iter().any(|step| {
                    matches!(
                        step,
                        xybrid_core::execution_template::PreprocessingStep::AudioDecode { .. }
                    )
                })
            }
            _ => false,
        }
    }

    fn infer_output_type(metadata: &ModelMetadata) -> OutputType {
        // Check metadata hints (metadata is HashMap<String, serde_json::Value>)
        if let Some(task) = metadata.metadata.get("task").and_then(|v| v.as_str()) {
            match task {
                "speech-recognition" | "asr" | "transcription" => return OutputType::Text,
                "text-to-speech" | "tts" | "speech-synthesis" => return OutputType::Audio,
                "embedding" | "feature-extraction" => return OutputType::Embedding,
                _ => {}
            }
        }

        // Check postprocessing steps for hints
        for step in &metadata.postprocessing {
            match step {
                xybrid_core::execution_template::PostprocessingStep::CTCDecode { .. }
                | xybrid_core::execution_template::PostprocessingStep::WhisperDecode { .. } => {
                    return OutputType::Text
                }
                xybrid_core::execution_template::PostprocessingStep::TTSAudioEncode { .. } => {
                    return OutputType::Audio
                }
                _ => {}
            }
        }

        OutputType::Unknown
    }
}

/// Represents a loaded model ready for inference.
///
/// Created by `ModelLoader::load()`. Provides both batch and streaming inference.
///
/// # Example
///
/// ```ignore
/// let model = loader.load()?;
///
/// // Batch inference
/// let result = model.run(&audio_envelope)?;
/// println!("Transcription: {}", result.unwrap_text());
///
/// // Streaming inference (if supported)
/// if model.supports_streaming() {
///     let stream = model.stream(StreamConfig::with_vad())?;
///     stream.feed(&samples)?;
///     let transcript = stream.flush()?;
/// }
///
/// // Cleanup
/// model.unload()?;
/// ```
pub struct XybridModel {
    handle: Arc<RwLock<ModelHandle>>,
    model_id: String,
    version: String,
    output_type: OutputType,
    supports_streaming: bool,
}

impl XybridModel {
    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the model version.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Check if the model is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.handle.read().map(|h| h.loaded).unwrap_or(false)
    }

    /// Check if this model supports streaming.
    pub fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }

    /// Get the expected output type for this model.
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }

    /// Check if this is an LLM model (uses GGUF execution template).
    ///
    /// LLM models support multi-turn conversation contexts. Use this to
    /// determine if conversation history should be maintained.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = loader.load()?;
    /// if model.is_llm() {
    ///     // Create conversation context for multi-turn chat
    ///     let mut ctx = ConversationContext::new();
    ///     // ... manage conversation history
    /// }
    /// ```
    pub fn is_llm(&self) -> bool {
        self.handle
            .read()
            .ok()
            .map(|h| {
                matches!(
                    h.metadata.execution_template,
                    ExecutionTemplate::Gguf { .. }
                )
            })
            .unwrap_or(false)
    }

    // =========================================================================
    // Voice Discovery (TTS models only)
    // =========================================================================

    /// Get the voice configuration for this model, if available.
    ///
    /// Returns `None` for non-TTS models or TTS models without voice configuration.
    pub fn voice_config(&self) -> Option<VoiceConfig> {
        self.handle
            .read()
            .ok()
            .and_then(|h| h.metadata.voices.clone())
    }

    /// Get all available voices for this TTS model.
    ///
    /// Returns `None` for non-TTS models or TTS models without voice configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(voices) = model.voices() {
    ///     for voice in voices {
    ///         println!("{}: {} ({})", voice.id, voice.name, voice.language.unwrap_or_default());
    ///     }
    /// }
    /// ```
    pub fn voices(&self) -> Option<Vec<VoiceInfo>> {
        self.voice_config().map(|vc| vc.catalog)
    }

    /// Get the default voice for this TTS model.
    ///
    /// Returns `None` for non-TTS models or if no default is configured.
    pub fn default_voice(&self) -> Option<VoiceInfo> {
        self.voice_config().and_then(|vc| {
            let default_id = &vc.default;
            vc.catalog.into_iter().find(|v| &v.id == default_id)
        })
    }

    /// Check if this model has voice configuration.
    ///
    /// Returns `true` for TTS models with voice support.
    pub fn has_voices(&self) -> bool {
        self.voice_config().is_some()
    }

    /// Get a specific voice by ID.
    ///
    /// Returns `None` if the voice is not found or the model has no voice support.
    ///
    /// # Arguments
    ///
    /// * `voice_id` - The voice identifier (e.g., "af_bella")
    pub fn voice(&self, voice_id: &str) -> Option<VoiceInfo> {
        self.voice_config()
            .and_then(|vc| vc.catalog.into_iter().find(|v| v.id == voice_id))
    }

    // =========================================================================
    // Warmup Methods (for pre-loading models)
    // =========================================================================

    /// Warm up the model by running a minimal inference.
    ///
    /// This pre-loads the model into memory, ensuring that the first real inference
    /// is fast. For LLM models, this loads the model weights and creates the context.
    ///
    /// Call this at app startup or after `load()` to eliminate cold-start latency.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = loader.load()?;
    /// model.warmup()?;  // Pre-load model
    ///
    /// // First inference is now fast
    /// let result = model.run(&envelope)?;
    /// ```
    pub fn warmup(&self) -> SdkResult<()> {
        use xybrid_core::ir::EnvelopeKind;

        log::info!(target: "xybrid_sdk", "Warming up model: {}", self.model_id);

        // Create a minimal input based on expected input type
        let warmup_input = match self.output_type {
            // For TTS models, use a short text
            OutputType::Audio => Envelope {
                kind: EnvelopeKind::Text("Hi".to_string()),
                metadata: std::collections::HashMap::new(),
            },
            // For ASR models, use minimal audio (1 second of silence at 16kHz)
            OutputType::Text if self.supports_streaming => {
                // Create a minimal WAV file with silence
                let silence_samples = vec![0i16; 16000]; // 1 second at 16kHz
                let audio_bytes = Self::create_wav_bytes(&silence_samples, 16000);
                Envelope {
                    kind: EnvelopeKind::Audio(audio_bytes),
                    metadata: std::collections::HashMap::new(),
                }
            }
            // For LLM/text models, use a short prompt
            OutputType::Text | OutputType::Embedding | OutputType::Unknown => Envelope {
                kind: EnvelopeKind::Text("Hi".to_string()),
                metadata: std::collections::HashMap::new(),
            },
        };

        // Run inference (this loads the model)
        let start = Instant::now();
        let _ = self.run(&warmup_input, None)?;
        let elapsed = start.elapsed();

        log::info!(
            target: "xybrid_sdk",
            "Model {} warmed up in {:?}",
            self.model_id,
            elapsed
        );

        Ok(())
    }

    /// Warm up the model asynchronously.
    ///
    /// This is useful for background pre-loading at app startup without blocking the UI.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = loader.load()?;
    ///
    /// // Start warmup in background
    /// let warmup_handle = tokio::spawn(async move {
    ///     model.warmup_async().await
    /// });
    ///
    /// // Do other initialization...
    ///
    /// // Wait for warmup if needed
    /// warmup_handle.await??;
    /// ```
    pub async fn warmup_async(&self) -> SdkResult<()> {
        let handle = self.handle.clone();
        let model_id = self.model_id.clone();
        let output_type = self.output_type;
        let supports_streaming = self.supports_streaming;

        tokio::task::spawn_blocking(move || {
            use xybrid_core::ir::EnvelopeKind;

            log::info!(target: "xybrid_sdk", "Warming up model (async): {}", model_id);

            // Create a minimal input based on expected input type
            let warmup_input = match output_type {
                OutputType::Audio => Envelope {
                    kind: EnvelopeKind::Text("Hi".to_string()),
                    metadata: std::collections::HashMap::new(),
                },
                OutputType::Text if supports_streaming => {
                    let silence_samples = vec![0i16; 16000];
                    let audio_bytes = Self::create_wav_bytes(&silence_samples, 16000);
                    Envelope {
                        kind: EnvelopeKind::Audio(audio_bytes),
                        metadata: std::collections::HashMap::new(),
                    }
                }
                OutputType::Text | OutputType::Embedding | OutputType::Unknown => Envelope {
                    kind: EnvelopeKind::Text("Hi".to_string()),
                    metadata: std::collections::HashMap::new(),
                },
            };

            let start = Instant::now();

            // Run inference
            let mut guard = handle.write().unwrap_or_else(|e| e.into_inner());
            if !guard.loaded {
                return Err(SdkError::NotLoaded);
            }

            let metadata = guard.metadata.clone();
            let _ = guard
                .executor
                .execute(&metadata, &warmup_input, None)
                .map_err(|e| SdkError::InferenceError(format!("Warmup failed: {}", e)))?;

            let elapsed = start.elapsed();
            log::info!(
                target: "xybrid_sdk",
                "Model {} warmed up (async) in {:?}",
                model_id,
                elapsed
            );

            Ok(())
        })
        .await
        .map_err(|e| SdkError::InferenceError(format!("Task join error: {}", e)))?
    }

    /// Create a minimal WAV file bytes from samples for warmup.
    fn create_wav_bytes(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        let num_samples = samples.len();
        let data_size = (num_samples * 2) as u32;
        let file_size = 36 + data_size;

        // RIFF header
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&file_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        // fmt chunk
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes()); // Chunk size
        bytes.extend_from_slice(&1u16.to_le_bytes()); // Audio format (PCM)
        bytes.extend_from_slice(&1u16.to_le_bytes()); // Num channels
        bytes.extend_from_slice(&sample_rate.to_le_bytes()); // Sample rate
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // Byte rate
        bytes.extend_from_slice(&2u16.to_le_bytes()); // Block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample

        // data chunk
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        bytes
    }

    /// Run batch inference with an Envelope.
    ///
    /// # Arguments
    ///
    /// * `envelope` - Input data wrapped in an Envelope
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the output with convenient accessors.
    pub fn run(
        &self,
        envelope: &Envelope,
        config: Option<&GenerationConfig>,
    ) -> SdkResult<InferenceResult> {
        let start = Instant::now();
        // Begin a resource-telemetry scope for this run. When
        // `resource_telemetry_mode()` is `Off` the guard is a no-op; otherwise
        // it captures start snapshots (and launches a sampler for Summary /
        // DebugLocal). Summary is produced by
        // `publish_with_resource_summary` at the end of this function.
        let resource_guard = crate::telemetry::begin_resource_run();

        // Recover from poisoned RwLock to prevent permanent lock errors
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to avoid borrow conflict with executor
        let metadata = handle.metadata.clone();
        let output = handle
            .executor
            .execute(&metadata, envelope, config)
            .map_err(|e| SdkError::InferenceError(format!("Execution failed: {}", e)))?;

        let latency_ms = start.elapsed().as_millis() as u32;

        // Emit ModelComplete telemetry event
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(
                serde_json::json!({
                    "model_id": self.model_id,
                    "version": self.version,
                    "output_type": format!("{:?}", self.output_type),
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_with_resource_summary(event, resource_guard);

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Run inference with conversation context.
    ///
    /// This method passes the conversation history to the model, allowing it to
    /// generate context-aware responses. The model uses its chat template to
    /// format the conversation history into a prompt.
    ///
    /// **Important:** This method does not mutate the context. The caller is
    /// responsible for pushing the result to the context if desired.
    ///
    /// # Arguments
    ///
    /// * `envelope` - The current user input (should have `MessageRole::User`)
    /// * `context` - Conversation history (system prompt + previous turns)
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the assistant's response (tagged with `MessageRole::Assistant`).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use xybrid_sdk::{ModelLoader, ConversationContext, Envelope, EnvelopeKind, MessageRole};
    ///
    /// let model = ModelLoader::from_registry("gemma-3-1b")?.load()?;
    /// let mut ctx = ConversationContext::new();
    ///
    /// // Add user message to context
    /// let user_input = Envelope::new(EnvelopeKind::Text("Hello!".into()))
    ///     .with_role(MessageRole::User);
    /// ctx.push(user_input.clone());
    ///
    /// // Run with context (model sees the full history)
    /// let result = model.run_with_context(&user_input, &ctx)?;
    ///
    /// // Add assistant response to context
    /// ctx.push(result.envelope().clone());
    ///
    /// println!("{}", result.text().unwrap_or_default());
    /// ```
    pub fn run_with_context(
        &self,
        envelope: &Envelope,
        context: &ConversationContext,
        config: Option<&GenerationConfig>,
    ) -> SdkResult<InferenceResult> {
        let start = Instant::now();
        let resource_guard = crate::telemetry::begin_resource_run();

        // Recover from poisoned RwLock to prevent permanent lock errors
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to avoid borrow conflict with executor
        let metadata = handle.metadata.clone();
        let output = handle
            .executor
            .execute_with_context(&metadata, envelope, context, config)
            .map_err(|e| SdkError::InferenceError(format!("Execution failed: {}", e)))?;

        let latency_ms = start.elapsed().as_millis() as u32;

        // Emit ModelComplete telemetry event
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(
                serde_json::json!({
                    "model_id": self.model_id,
                    "version": self.version,
                    "output_type": format!("{:?}", self.output_type),
                    "context_messages": context.history().len(),
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_with_resource_summary(event, resource_guard);

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Run streaming inference with conversation context.
    ///
    /// Combines streaming output with multi-turn conversation memory.
    /// The model sees the full conversation history when generating responses.
    ///
    /// # Arguments
    ///
    /// * `envelope` - Current user input wrapped in an Envelope
    /// * `context` - Conversation history for multi-turn chat
    /// * `on_token` - Callback invoked for each token (LLM) or once (other models)
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the final output.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut ctx = ConversationContext::new();
    ///
    /// // Add user message and run with streaming
    /// let input = Envelope::new(EnvelopeKind::Text("Tell me a joke".into()))
    ///     .with_role(MessageRole::User);
    /// ctx.push(input.clone());
    ///
    /// let result = model.run_streaming_with_context(&input, &ctx, |token| {
    ///     print!("{}", token.token);
    ///     std::io::Write::flush(&mut std::io::stdout())?;
    ///     Ok(())
    /// })?;
    ///
    /// // Add assistant response to context
    /// ctx.push(result.envelope().clone());
    /// ```
    pub fn run_streaming_with_context<F>(
        &self,
        envelope: &Envelope,
        context: &ConversationContext,
        config: Option<&GenerationConfig>,
        mut on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        use xybrid_core::execution::ExecutionTemplate;
        use xybrid_core::runtime_adapter::types::PartialToken;

        let start = Instant::now();

        // Get write lock on handle
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to check execution template
        let metadata = handle.metadata.clone();

        // Check if this is an LLM model (GGUF template)
        let is_llm = matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. });

        let output = if is_llm {
            // True streaming with context for LLM models
            handle
                .executor
                .execute_streaming_with_context(
                    &metadata,
                    envelope,
                    context,
                    Box::new(&mut on_token),
                    config,
                )
                .map_err(|e| {
                    SdkError::InferenceError(format!("Streaming execution failed: {}", e))
                })?
        } else {
            // For non-LLM models: run with context and emit single "token" with full result
            let result = handle
                .executor
                .execute_with_context(&metadata, envelope, context, config)
                .map_err(|e| SdkError::InferenceError(format!("Execution failed: {}", e)))?;

            // Extract text from result (if any) and emit as single token
            if let xybrid_core::ir::EnvelopeKind::Text(text) = &result.kind {
                let token = PartialToken {
                    token: text.clone(),
                    token_id: None,
                    index: 0,
                    cumulative_text: text.clone(),
                    finish_reason: Some("stop".to_string()),
                };
                let _ = on_token(token);
            }

            result
        };

        let latency_ms = start.elapsed().as_millis() as u32;

        // Emit telemetry event
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(
                serde_json::json!({
                    "model_id": self.model_id,
                    "version": self.version,
                    "output_type": format!("{:?}", self.output_type),
                    "streaming": true,
                    "context_messages": context.history().len(),
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_telemetry_event(event);

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Run inference with streaming output.
    ///
    /// This method provides a unified streaming interface for all model types:
    /// - **LLM models (GGUF)**: True token-by-token streaming via the callback
    /// - **Other models (TTS, ASR, etc.)**: Single callback with the full result
    ///
    /// This "everything is a stream" pattern allows consumers to use the same
    /// API regardless of model type, while LLMs get the latency benefits of
    /// true streaming.
    ///
    /// # Arguments
    ///
    /// * `envelope` - Input data wrapped in an Envelope
    /// * `on_token` - Callback invoked for each token (LLM) or once (other models)
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the final output.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Works for both LLM and non-LLM models
    /// let result = model.run_streaming(&envelope, |token| {
    ///     print!("{}", token.token);
    ///     std::io::Write::flush(&mut std::io::stdout())?;
    ///     Ok(())
    /// })?;
    /// ```
    pub fn run_streaming<F>(
        &self,
        envelope: &Envelope,
        config: Option<&GenerationConfig>,
        mut on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        use xybrid_core::execution::ExecutionTemplate;
        use xybrid_core::runtime_adapter::types::PartialToken;

        let start = Instant::now();

        // Get write lock on handle
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to check execution template
        let metadata = handle.metadata.clone();

        // Check if this is an LLM model (GGUF template)
        let is_llm = matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. });

        let output = if is_llm {
            // True streaming for LLM models
            handle
                .executor
                .execute_streaming(&metadata, envelope, Box::new(&mut on_token), config)
                .map_err(|e| {
                    SdkError::InferenceError(format!("Streaming execution failed: {}", e))
                })?
        } else {
            // For non-LLM models: run batch and emit single "token" with full result
            let result = handle
                .executor
                .execute(&metadata, envelope, config)
                .map_err(|e| SdkError::InferenceError(format!("Execution failed: {}", e)))?;

            // Extract text from result (if any) and emit as single token
            if let xybrid_core::ir::EnvelopeKind::Text(text) = &result.kind {
                let token = PartialToken {
                    token: text.clone(),
                    token_id: None,
                    index: 0,
                    cumulative_text: text.clone(),
                    finish_reason: Some("stop".to_string()),
                };
                let _ = on_token(token); // Ignore callback errors for non-streaming
            }

            result
        };

        let latency_ms = start.elapsed().as_millis() as u32;

        // Emit telemetry event
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(
                serde_json::json!({
                    "model_id": self.model_id,
                    "version": self.version,
                    "output_type": format!("{:?}", self.output_type),
                    "streaming": true,
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_telemetry_event(event);

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Run inference returning a stream of events.
    ///
    /// This is the idiomatic Rust streaming API that returns a `Stream` instead of
    /// using callbacks. Events are emitted as they occur:
    /// - `StreamEvent::Token` - for each generated token (LLM models)
    /// - `StreamEvent::Complete` - when inference finishes successfully
    /// - `StreamEvent::Error` - if an error occurs
    ///
    /// For non-LLM models, a single `Token` event is emitted with the full result,
    /// followed by `Complete`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tokio_stream::StreamExt;
    ///
    /// let mut stream = model.run_stream(envelope);
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         StreamEvent::Token(token) => print!("{}", token.token),
    ///         StreamEvent::Complete(result) => println!("\nDone: {}ms", result.latency_ms()),
    ///         StreamEvent::Error(e) => eprintln!("Error: {}", e),
    ///     }
    /// }
    /// ```
    pub fn run_stream(
        &self,
        envelope: Envelope,
        config: Option<GenerationConfig>,
    ) -> Pin<Box<dyn tokio_stream::Stream<Item = StreamEvent> + Send + '_>> {
        use tokio::sync::mpsc;
        use xybrid_core::runtime_adapter::types::PartialToken;

        let (tx, rx) = mpsc::channel::<StreamEvent>(100);
        let handle = self.handle.clone();
        let model_id = self.model_id.clone();
        let version = self.version.clone();
        let output_type = self.output_type;

        // Clone tx for the completion event (before moving into spawn_blocking)
        let tx_completion = tx.clone();

        // Spawn blocking task to run inference
        tokio::task::spawn(async move {
            let result = tokio::task::spawn_blocking(move || {
                let start = Instant::now();

                // Get write lock on handle
                let mut guard = handle.write().unwrap_or_else(|e| e.into_inner());

                if !guard.loaded {
                    return Err(SdkError::NotLoaded);
                }

                let metadata = guard.metadata.clone();
                let is_llm = matches!(
                    metadata.execution_template,
                    xybrid_core::execution::ExecutionTemplate::Gguf { .. }
                );

                // Clone tx for the streaming callback (so we can use tx in the else branch)
                let tx_for_callback = tx.clone();

                let output = if is_llm {
                    // True streaming for LLM models
                    guard
                        .executor
                        .execute_streaming(
                            &metadata,
                            &envelope,
                            Box::new(move |token: PartialToken| {
                                let stream_token = StreamToken {
                                    token: token.token.clone(),
                                    token_id: token.token_id.map(|id| id),
                                    index: token.index,
                                    cumulative_text: token.cumulative_text.clone(),
                                    finish_reason: token.finish_reason.clone(),
                                };
                                // Ignore send errors (receiver dropped)
                                let _ =
                                    tx_for_callback.blocking_send(StreamEvent::Token(stream_token));
                                Ok(())
                            }),
                            config.as_ref(),
                        )
                        .map_err(|e| {
                            SdkError::InferenceError(format!("Streaming execution failed: {}", e))
                        })?
                } else {
                    // Non-LLM: batch execution, emit single token
                    let result = guard
                        .executor
                        .execute(&metadata, &envelope, config.as_ref())
                        .map_err(|e| {
                            SdkError::InferenceError(format!("Execution failed: {}", e))
                        })?;

                    // Emit single token with full result
                    if let xybrid_core::ir::EnvelopeKind::Text(text) = &result.kind {
                        let stream_token = StreamToken {
                            token: text.clone(),
                            token_id: None,
                            index: 0,
                            cumulative_text: text.clone(),
                            finish_reason: Some("stop".to_string()),
                        };
                        let _ = tx.blocking_send(StreamEvent::Token(stream_token));
                    }
                    result
                };

                let latency_ms = start.elapsed().as_millis() as u32;

                // Emit telemetry
                let event = crate::telemetry::TelemetryEvent {
                    event_type: "ModelComplete".to_string(),
                    stage_name: Some(model_id.clone()),
                    target: Some("local".to_string()),
                    latency_ms: Some(latency_ms),
                    error: None,
                    data: Some(
                        serde_json::json!({
                            "model_id": model_id,
                            "version": version,
                            "output_type": format!("{:?}", output_type),
                            "streaming": true,
                        })
                        .to_string(),
                    ),
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                };
                crate::telemetry::publish_telemetry_event(event);

                Ok(InferenceResult::new(output, &model_id, latency_ms))
            })
            .await;

            // Send completion or error event
            match result {
                Ok(Ok(inference_result)) => {
                    let _ = tx_completion
                        .send(StreamEvent::Complete(inference_result))
                        .await;
                }
                Ok(Err(e)) => {
                    let _ = tx_completion.send(StreamEvent::Error(e.to_string())).await;
                }
                Err(e) => {
                    let _ = tx_completion
                        .send(StreamEvent::Error(format!("Task failed: {}", e)))
                        .await;
                }
            }
        });

        Box::pin(ReceiverStream::new(rx))
    }

    /// Check if this model supports true token streaming.
    ///
    /// Returns `true` for LLM models (GGUF) when LLM features are enabled,
    /// `false` for other model types or when LLM features are disabled.
    /// Note: `run_streaming()` works for all models, but only LLM models
    /// get true token-by-token streaming; others emit a single result.
    pub fn supports_token_streaming(&self) -> bool {
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        {
            use xybrid_core::execution::ExecutionTemplate;

            self.handle
                .read()
                .ok()
                .map(|h| {
                    matches!(
                        h.metadata.execution_template,
                        ExecutionTemplate::Gguf { .. }
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
        {
            false
        }
    }

    /// Run batch inference asynchronously.
    pub async fn run_async(
        &self,
        envelope: &Envelope,
        config: Option<&GenerationConfig>,
    ) -> SdkResult<InferenceResult> {
        let handle = self.handle.clone();
        let model_id = self.model_id.clone();
        let version = self.version.clone();
        let output_type = self.output_type;
        let envelope = envelope.clone();
        let config = config.cloned();

        tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let resource_guard = crate::telemetry::begin_resource_run();

            // Recover from poisoned RwLock to prevent permanent lock errors
            let mut guard = handle.write().unwrap_or_else(|e| e.into_inner());

            if !guard.loaded {
                return Err(SdkError::NotLoaded);
            }

            // Clone metadata to avoid borrow conflict with executor
            let metadata = guard.metadata.clone();
            let output = guard
                .executor
                .execute(&metadata, &envelope, config.as_ref())
                .map_err(|e| SdkError::InferenceError(format!("Execution failed: {}", e)))?;

            let latency_ms = start.elapsed().as_millis() as u32;

            // Emit ModelComplete telemetry event
            let event = crate::telemetry::TelemetryEvent {
                event_type: "ModelComplete".to_string(),
                stage_name: Some(model_id.clone()),
                target: Some("local".to_string()),
                latency_ms: Some(latency_ms),
                error: None,
                data: Some(
                    serde_json::json!({
                        "model_id": model_id,
                        "version": version,
                        "output_type": format!("{:?}", output_type),
                    })
                    .to_string(),
                ),
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            };
            crate::telemetry::publish_with_resource_summary(event, resource_guard);

            Ok(InferenceResult::new(output, &model_id, latency_ms))
        })
        .await
        .map_err(|e| SdkError::InferenceError(format!("Task join error: {}", e)))?
    }

    /// Create a streaming session for real-time ASR.
    ///
    /// Returns an error if `!supports_streaming()`.
    ///
    /// # Arguments
    ///
    /// * `config` - Streaming configuration (VAD, language, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stream = model.stream(StreamConfig::with_vad())?;
    ///
    /// // Feed audio chunks
    /// stream.feed(&audio_samples)?;
    ///
    /// // Get partial results
    /// if let Some(partial) = stream.partial_result() {
    ///     println!("Partial: {}", partial.text);
    /// }
    ///
    /// // Get final transcript
    /// let transcript = stream.flush()?;
    /// ```
    pub fn stream(&self, config: StreamConfig) -> SdkResult<XybridStream> {
        if !self.supports_streaming {
            return Err(SdkError::StreamingNotSupported);
        }

        // Recover from poisoned RwLock to prevent permanent lock errors
        let handle = self.handle.read().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Convert to core StreamConfig
        let core_config = CoreStreamConfig {
            vad: CoreVadConfig {
                enabled: config.enable_vad,
                model_dir: config.vad_model_dir,
                threshold: config.vad_threshold,
                ..Default::default()
            },
            language: config.language,
            ..Default::default()
        };

        XybridStream::new(&handle.model_dir, core_config, &self.model_id)
    }

    /// Unload the model from memory.
    ///
    /// The model can be reloaded by creating a new ModelLoader.
    pub fn unload(&self) -> SdkResult<()> {
        // Recover from poisoned RwLock to prevent permanent lock errors
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        handle.loaded = false;
        // Clear the session cache (drop executor and recreate empty)
        handle.executor = TemplateExecutor::default();

        Ok(())
    }
}

// Make XybridModel cloneable (shares the handle)
impl Clone for XybridModel {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            model_id: self.model_id.clone(),
            version: self.version.clone(),
            output_type: self.output_type,
            supports_streaming: self.supports_streaming,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loader_from_registry() {
        let loader = ModelLoader::from_registry("kokoro-82m");
        assert_eq!(loader.model_id(), Some("kokoro-82m"));
        assert_eq!(loader.version(), None); // Version resolved by registry
        assert_eq!(loader.source_type(), "registry");
    }

    #[test]
    fn test_model_loader_from_registry_with_platform() {
        let loader = ModelLoader::from_registry_with_platform("whisper-tiny", "macos-arm64");
        assert_eq!(loader.model_id(), Some("whisper-tiny"));
        assert_eq!(loader.source_type(), "registry");
    }

    #[test]
    #[allow(deprecated)]
    fn test_model_loader_from_legacy_registry() {
        let loader = ModelLoader::from_legacy_registry("http://localhost:8080", "whisper", "1.0");
        assert_eq!(loader.model_id(), Some("whisper"));
        assert_eq!(loader.version(), Some("1.0"));
        assert_eq!(loader.source_type(), "legacy_registry");
    }

    #[test]
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();
        assert!(!config.enable_vad);
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_stream_config_with_vad() {
        let config = StreamConfig::with_vad().language("fr").vad_threshold(0.7);
        assert!(config.enable_vad);
        assert_eq!(config.language, Some("fr".to_string()));
        assert_eq!(config.vad_threshold, 0.7);
    }
}
