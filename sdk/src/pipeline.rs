//! Pipeline loading and execution for xybrid-sdk.
//!
//! This module provides a simple two-type API:
//! - `PipelineRef`: Lightweight reference from parsed YAML (no network)
//! - `Pipeline`: Loaded pipeline ready to preload models and run
//!
//! # Example (Simple - just run)
//!
//! ```rust,ignore
//! use xybrid_sdk::{PipelineRef, Envelope};
//!
//! // Load and run in a few lines
//! let pipeline = PipelineRef::from_yaml(yaml_content)?.load()?;
//! pipeline.load_models()?;  // Optional: explicit preloading
//! let result = pipeline.run(&Envelope::audio(audio_bytes))?;
//! println!("Pipeline completed in {}ms", result.total_latency_ms);
//! ```
//!
//! # Example (Staged - inspect and preload)
//!
//! ```rust,ignore
//! use xybrid_sdk::{PipelineRef, Envelope};
//!
//! // Step 1: Parse YAML (instant, no network)
//! let ref_ = PipelineRef::from_yaml(yaml_content)?;
//! println!("Stages: {:?}", ref_.stage_ids());
//!
//! // Step 2: Load pipeline (resolves models via registry)
//! let pipeline = ref_.load()?;
//! println!("Download size: {} bytes", pipeline.download_size());
//!
//! // Step 3: Preload models (optional - useful for app startup)
//! pipeline.load_models_with_progress(|progress| {
//!     println!("Downloading {}: {}%", progress.model_id, progress.percent);
//! })?;
//!
//! // Step 4: Run
//! let result = pipeline.run(&Envelope::audio(audio_bytes))?;
//! ```

use crate::model::SdkError;
use crate::registry_client::RegistryClient;
use crate::result::OutputType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use xybrid_core::context::StageDescriptor;
use xybrid_core::device_adapter::{DeviceAdapter, LocalDeviceAdapter};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::routing_engine::LocalAvailability;
use xybrid_core::orchestrator::{
    LocalAuthority, OrchestrationAuthority, Orchestrator, ResolvedTarget, StageContext,
    StageExecutionResult,
};
use xybrid_core::pipeline::{ExecutionTarget, IntegrationProvider, StageOptions};
use xybrid_core::pipeline_config::PipelineConfig;

/// Result type for pipeline operations.
pub type PipelineResult<T> = Result<T, SdkError>;

// ============================================================================
// PipelineRef - Lightweight reference from YAML
// ============================================================================

/// A lightweight reference to a pipeline from parsed YAML.
///
/// `PipelineRef` is created instantly from YAML without any network calls.
/// Use `load()` to create a `Pipeline` that can preload models and run inference.
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_sdk::PipelineRef;
///
/// let ref_ = PipelineRef::from_yaml(yaml)?;
/// println!("Pipeline: {:?}", ref_.name());
/// println!("Stages: {:?}", ref_.stage_ids());
///
/// let pipeline = ref_.load()?;
/// ```
#[derive(Debug, Clone)]
pub struct PipelineRef {
    yaml_content: String,
    config: PipelineConfig,
}

impl PipelineRef {
    /// Parse a pipeline from YAML content (instant, no network).
    pub fn from_yaml(yaml: &str) -> PipelineResult<Self> {
        let config: PipelineConfig = serde_yaml::from_str(yaml)
            .map_err(|e| SdkError::PipelineError(format!("Failed to parse YAML: {}", e)))?;

        Ok(Self {
            yaml_content: yaml.to_string(),
            config,
        })
    }

    /// Parse a pipeline from a YAML file.
    pub fn from_file(path: impl Into<PathBuf>) -> PipelineResult<Self> {
        let path = path.into();
        let content = std::fs::read_to_string(&path)
            .map_err(|e| SdkError::PipelineError(format!("Failed to read file: {}", e)))?;
        Self::from_yaml(&content)
    }

    /// Get the pipeline name (if specified).
    pub fn name(&self) -> Option<&str> {
        self.config.name.as_deref()
    }

    /// Get the stage IDs (stage names/identifiers).
    pub fn stage_ids(&self) -> Vec<String> {
        self.config.stages.iter().map(|s| s.stage_id()).collect()
    }

    /// Get the number of stages.
    pub fn stage_count(&self) -> usize {
        self.config.stages.len()
    }

    /// Load the pipeline (resolves models via registry).
    ///
    /// This creates a `Pipeline` that can preload models and run inference.
    pub fn load(&self) -> PipelineResult<Pipeline> {
        Pipeline::from_ref(self)
    }

    /// Load the pipeline asynchronously.
    pub async fn load_async(&self) -> PipelineResult<Pipeline> {
        // For now, delegate to sync version
        self.load()
    }
}

// ============================================================================
// Stage Types
// ============================================================================

/// Information about a stage in a loaded pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageInfo {
    /// Stage identifier
    pub id: String,
    /// Model ID (if this stage uses a model)
    pub model_id: Option<String>,
    /// Execution target
    pub target: StageTarget,
    /// Current status
    pub status: StageStatus,
    /// Download size in bytes (if needs download)
    pub download_bytes: Option<u64>,
}

/// Execution target for a stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageTarget {
    /// Runs on device using local model (explicit `target: device` in YAML)
    Device,
    /// Let authority decide (default when no target specified, or `target: auto`)
    Auto,
    /// Runs on cloud server (explicit `target: cloud` in YAML)
    Cloud,
    /// Runs via integration provider (e.g., OpenAI API)
    Integration { provider: String },
}

impl std::fmt::Display for StageTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageTarget::Device => write!(f, "device"),
            StageTarget::Auto => write!(f, "auto"),
            StageTarget::Cloud => write!(f, "cloud"),
            StageTarget::Integration { provider } => write!(f, "integration:{}", provider),
        }
    }
}

/// Status of a stage's model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageStatus {
    /// Model is cached locally
    Cached,
    /// Model needs to be downloaded
    NeedsDownload,
    /// Integration stage (no local model needed)
    Integration,
    /// Resolution failed
    Error(String),
}

impl std::fmt::Display for StageStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageStatus::Cached => write!(f, "cached"),
            StageStatus::NeedsDownload => write!(f, "needs_download"),
            StageStatus::Integration => write!(f, "integration"),
            StageStatus::Error(msg) => write!(f, "error: {}", msg),
        }
    }
}

/// Progress information during model loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Model currently being downloaded
    pub model_id: String,
    /// Download progress (0-100)
    pub percent: u32,
    /// Bytes downloaded so far
    pub bytes_downloaded: u64,
    /// Total bytes for this model
    pub bytes_total: u64,
    /// Current stage index (0-based)
    pub stage_index: usize,
    /// Total stages needing download
    pub total_stages: usize,
}

// ============================================================================
// Pipeline Configuration Types (internal)
// ============================================================================

// Use StageConfig from xybrid_core::pipeline_config
use xybrid_core::pipeline_config::StageConfig;

/// Input type for pipeline (public API).
/// This will be auto-inferred from model metadata in a future release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineInputType {
    Audio,
    Text,
    Embedding,
    Unknown,
}

impl PipelineInputType {
    pub fn is_audio(&self) -> bool {
        matches!(self, PipelineInputType::Audio)
    }

    pub fn is_text(&self) -> bool {
        matches!(self, PipelineInputType::Text)
    }
}

// ============================================================================
// Pipeline Execution Result Types
// ============================================================================

/// Timing information for a single pipeline stage.
#[derive(Debug, Clone, Serialize)]
pub struct StageTiming {
    pub name: String,
    pub latency_ms: u32,
    pub target: String,
    pub reason: String,
}

/// Result of pipeline execution.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineExecutionResult {
    pub name: Option<String>,
    pub stages: Vec<StageTiming>,
    pub total_latency_ms: u32,
    pub output_type: OutputType,
    pub output: Envelope,
}

impl PipelineExecutionResult {
    pub fn text(&self) -> Option<&str> {
        match &self.output.kind {
            EnvelopeKind::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn audio_bytes(&self) -> Option<&[u8]> {
        match &self.output.kind {
            EnvelopeKind::Audio(bytes) => Some(bytes),
            _ => None,
        }
    }

    pub fn embedding(&self) -> Option<&[f32]> {
        match &self.output.kind {
            EnvelopeKind::Embedding(e) => Some(e),
            _ => None,
        }
    }
}

// ============================================================================
// Internal Pipeline Handle
// ============================================================================

/// Internal state for the loaded pipeline.
struct PipelineHandle {
    stage_descriptors: Vec<StageDescriptor>,
    /// Availability map (updated when models are downloaded)
    availability_map: HashMap<String, bool>,
    /// Registry URL (for downloading models)
    registry_url: Option<String>,
    stage_configs: Vec<StageConfig>,
    /// Bundle paths for each stage (set after downloading)
    bundle_paths: HashMap<String, PathBuf>,
}

// ============================================================================
// Pipeline - Main Type
// ============================================================================

/// A loaded pipeline ready to preload models and run inference.
///
/// Created via `PipelineRef::load()`. This is the main type for running pipelines.
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_sdk::{PipelineRef, Envelope};
///
/// let pipeline = PipelineRef::from_yaml(yaml)?.load()?;
///
/// // Inspect the pipeline
/// println!("Name: {:?}", pipeline.name());
/// println!("Stages: {:?}", pipeline.stage_names());
/// println!("Download size: {} bytes", pipeline.download_size());
///
/// // Optional: Preload models (useful for app startup)
/// pipeline.load_models()?;
///
/// // Run inference
/// let result = pipeline.run(&Envelope::audio(audio_bytes))?;
/// ```
pub struct Pipeline {
    name: Option<String>,
    handle: Arc<RwLock<PipelineHandle>>,
    stages: Vec<StageInfo>,
    total_download_bytes: u64,
}

impl Pipeline {
    /// Create a Pipeline from a PipelineRef by resolving models.
    fn from_ref(ref_: &PipelineRef) -> PipelineResult<Self> {
        let config = ref_.config.clone();

        // Build stage descriptors (bundle_path will be set after downloading)
        let stage_descriptors: Vec<StageDescriptor> = config
            .stages
            .iter()
            .map(|stage_config| {
                let name = stage_config.model_id();
                let mut desc = StageDescriptor::new(name);

                if let Some(target_str) = stage_config.target() {
                    desc.target = Self::parse_target(&target_str);
                }

                if let Some(provider_str) = stage_config.provider() {
                    desc.provider = Self::parse_provider(&provider_str);
                    if desc.target.is_none() {
                        desc.target = Some(ExecutionTarget::Cloud);
                    }
                }

                // Use model_id() for descriptor's model field
                desc.model = Some(stage_config.model_id());

                let opts = stage_config.options();
                if !opts.is_empty() {
                    desc.options = Some(Self::convert_options(&opts));
                }

                desc
            })
            .collect();

        // Extract registry URL
        let registry_url = config.registry.clone();

        let stage_configs = config.stages.clone();

        // Auto-detect availability by checking cache
        let mut availability_map = HashMap::new();
        let client = if let Some(ref url) = registry_url {
            RegistryClient::with_url(url.clone()).ok()
        } else {
            RegistryClient::from_env().ok()
        };

        if let Some(ref client) = client {
            for stage_config in &stage_configs {
                let model_id = stage_config.model_id();
                let is_cached = client.is_cached(&model_id, None).unwrap_or(false);
                availability_map.insert(model_id, is_cached);
            }
        }

        let handle = PipelineHandle {
            stage_descriptors,
            availability_map,
            registry_url,
            stage_configs,
            bundle_paths: HashMap::new(),
        };

        let handle = Arc::new(RwLock::new(handle));

        // Resolve stages and compute download info
        let (stages, total_download_bytes) = Self::resolve_stages(&handle, &config)?;

        Ok(Self {
            name: config.name,
            handle,
            stages,
            total_download_bytes,
        })
    }

    fn parse_target(target: &str) -> Option<ExecutionTarget> {
        match target.to_lowercase().as_str() {
            "device" | "local" => Some(ExecutionTarget::Device),
            "server" => Some(ExecutionTarget::Server),
            "cloud" | "integration" | "api" => Some(ExecutionTarget::Cloud),
            "auto" => Some(ExecutionTarget::Auto),
            _ => None,
        }
    }

    fn parse_provider(provider: &str) -> Option<IntegrationProvider> {
        match provider.to_lowercase().as_str() {
            "openai" => Some(IntegrationProvider::OpenAI),
            "anthropic" | "claude" => Some(IntegrationProvider::Anthropic),
            "google" | "gemini" => Some(IntegrationProvider::Google),
            "elevenlabs" | "eleven" | "eleven_labs" => Some(IntegrationProvider::ElevenLabs),
            "openrouter" | "open_router" => Some(IntegrationProvider::OpenRouter),
            _ => Some(IntegrationProvider::Custom),
        }
    }

    fn convert_options(options: &HashMap<String, serde_json::Value>) -> StageOptions {
        let mut stage_options = StageOptions::new();
        for (key, value) in options {
            match value {
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        stage_options.set(key, f);
                    } else if let Some(i) = n.as_u64() {
                        stage_options.set(key, i as u32);
                    }
                }
                serde_json::Value::String(s) => {
                    stage_options.set(key, s.clone());
                }
                serde_json::Value::Bool(b) => {
                    stage_options.set(key, *b);
                }
                _ => {}
            }
        }
        stage_options
    }

    /// Resolve stage information from the registry.
    fn resolve_stages(
        handle: &Arc<RwLock<PipelineHandle>>,
        config: &PipelineConfig,
    ) -> PipelineResult<(Vec<StageInfo>, u64)> {
        let registry_url = config.registry.clone();

        let client = if let Some(url) = registry_url {
            RegistryClient::with_url(url)?
        } else {
            RegistryClient::from_env()?
        };

        let mut stages = Vec::new();
        let mut total_download_bytes: u64 = 0;

        let handle_read = handle
            .read()
            .map_err(|_| SdkError::PipelineError("Failed to read pipeline handle".to_string()))?;

        for stage_config in &config.stages {
            let stage_id = stage_config.stage_id();
            let target_str = stage_config.target();
            let provider = stage_config.provider();
            let model_name = Some(stage_config.model_id());

            let stage_target = if provider.is_some() || target_str == Some("integration") {
                StageTarget::Integration {
                    provider: provider.unwrap_or("unknown").to_string(),
                }
            } else if target_str == Some("cloud") || target_str == Some("server") {
                StageTarget::Cloud
            } else if target_str == Some("device")
                || target_str == Some("local")
                || target_str == Some("edge")
            {
                StageTarget::Device // Explicit local execution
            } else {
                StageTarget::Auto // Default: let authority decide (includes "auto" and None)
            };

            let (status, download_bytes) =
                if matches!(stage_target, StageTarget::Device | StageTarget::Auto) {
                    // For device/auto stages, check if model is cached (might run locally)
                    let model_id = stage_config.model_id();
                    match client.resolve(&model_id, None) {
                        Ok(resolved) => {
                            let is_cached = client.is_cached(&model_id, None).unwrap_or(false);
                            if is_cached {
                                (StageStatus::Cached, None)
                            } else {
                                total_download_bytes += resolved.size_bytes;
                                (StageStatus::NeedsDownload, Some(resolved.size_bytes))
                            }
                        }
                        Err(e) => {
                            // Check availability map as fallback
                            if handle_read
                                .availability_map
                                .get(&model_id)
                                .copied()
                                .unwrap_or(false)
                            {
                                (StageStatus::Cached, None)
                            } else {
                                (StageStatus::Error(e.to_string()), None)
                            }
                        }
                    }
                } else {
                    (StageStatus::Integration, None)
                };

            stages.push(StageInfo {
                id: stage_id,
                model_id: model_name,
                target: stage_target,
                status,
                download_bytes,
            });
        }

        Ok((stages, total_download_bytes))
    }

    /// Get the pipeline name (if specified).
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the stage names.
    pub fn stage_names(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.id.clone()).collect()
    }

    /// Get detailed information about all stages.
    pub fn stages(&self) -> &[StageInfo] {
        &self.stages
    }

    /// Get the number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get the expected input type for this pipeline.
    ///
    /// Note: In a future release, this will be auto-inferred from the first stage's
    /// model metadata preprocessing steps.
    pub fn input_type(&self) -> PipelineInputType {
        // TODO: Auto-infer from first stage's model_metadata.json preprocessing
        PipelineInputType::Unknown
    }

    /// Check if all device models are cached and ready.
    pub fn is_ready(&self) -> bool {
        self.stages
            .iter()
            .all(|s| matches!(s.status, StageStatus::Cached | StageStatus::Integration))
    }

    /// Get the total bytes that need to be downloaded.
    pub fn download_size(&self) -> u64 {
        self.total_download_bytes
    }

    /// Get stages that need models downloaded.
    pub fn stages_needing_download(&self) -> Vec<&StageInfo> {
        self.stages
            .iter()
            .filter(|s| matches!(s.status, StageStatus::NeedsDownload))
            .collect()
    }

    /// Preload models (downloads if needed).
    ///
    /// This method downloads any models that aren't already cached.
    /// Call this at app startup for smooth UX.
    pub fn load_models(&self) -> PipelineResult<()> {
        self.load_models_with_progress(|_| {})
    }

    /// Preload models with progress callback.
    ///
    /// **Routing-Aware Downloads**: Before downloading each model, this method
    /// consults the `OrchestrationAuthority` to determine if the stage will
    /// actually run locally. If the authority routes to cloud (e.g., low battery,
    /// model too large), the download is skipped.
    ///
    /// This ensures we don't waste bandwidth downloading models that won't be
    /// used locally.
    pub fn load_models_with_progress<F>(&self, progress_callback: F) -> PipelineResult<()>
    where
        F: Fn(DownloadProgress),
    {
        let registry_url = self
            .handle
            .read()
            .map_err(|_| SdkError::PipelineError("Failed to read handle".to_string()))?
            .registry_url
            .clone();

        let client = if let Some(url) = registry_url {
            RegistryClient::with_url(url)?
        } else {
            RegistryClient::from_env()?
        };

        // Create authority for routing decisions
        let authority = LocalAuthority::new();

        // Get current device metrics for routing decisions
        let device_adapter = LocalDeviceAdapter::new();
        let metrics = device_adapter.collect_metrics();

        let stages_to_fetch: Vec<_> = self
            .stages
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s.status, StageStatus::NeedsDownload))
            .filter_map(|(idx, s)| {
                s.model_id.as_ref().map(|m| {
                    (
                        idx,
                        s.id.clone(),
                        m.clone(),
                        s.download_bytes.unwrap_or(0),
                        s.target.clone(),
                    )
                })
            })
            .collect();

        let total_stages = stages_to_fetch.len();
        let mut skipped_count = 0;

        for (stage_idx, (_, stage_id, model_id, total_bytes, stage_target)) in
            stages_to_fetch.into_iter().enumerate()
        {
            // Convert StageTarget to ExecutionTarget for authority
            // - Device: user explicitly wants local, authority should respect it
            // - Auto: let authority decide based on device conditions
            // - Cloud/Integration: shouldn't reach here (filtered earlier), but handle anyway
            let explicit_target = match &stage_target {
                StageTarget::Device => Some(ExecutionTarget::Device),
                StageTarget::Auto => None, // Let authority decide
                StageTarget::Cloud => Some(ExecutionTarget::Cloud),
                StageTarget::Integration { .. } => Some(ExecutionTarget::Cloud),
            };

            // Consult authority before downloading
            let stage_context = StageContext {
                stage_id: stage_id.clone(),
                model_id: model_id.clone(),
                input_kind: EnvelopeKind::Text("".to_string()), // At preload time, we don't have actual input
                metrics: metrics.clone(),
                explicit_target,
            };

            let decision = authority.resolve_target(&stage_context);

            // Only download if authority routes to device
            match decision.result {
                ResolvedTarget::Device => {
                    // Authority says run locally - proceed with download
                    let progress_for_model = |download_progress: f32| {
                        let bytes_downloaded = (download_progress * total_bytes as f32) as u64;
                        progress_callback(DownloadProgress {
                            model_id: model_id.clone(),
                            percent: (download_progress * 100.0) as u32,
                            bytes_downloaded,
                            bytes_total: total_bytes,
                            stage_index: stage_idx,
                            total_stages,
                        });
                    };

                    // Fetch model and extract to permanent cache directory
                    // Uses CacheManager.ensure_extracted() as single source of truth
                    let model_dir = client.fetch_extracted(&model_id, None, progress_for_model)?;

                    {
                        // Recover from poisoned RwLock to prevent permanent lock errors
                        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

                        handle.availability_map.insert(model_id.clone(), true);
                        handle.availability_map.insert(stage_id.clone(), true);
                        // Store the extracted directory path for this stage
                        handle
                            .bundle_paths
                            .insert(stage_id.clone(), model_dir.clone());
                        handle.bundle_paths.insert(model_id.clone(), model_dir);
                    }
                }
                ResolvedTarget::Cloud { .. } | ResolvedTarget::Server { .. } => {
                    // Authority routes to cloud/server - skip download
                    skipped_count += 1;
                    // Log the skip decision (could use telemetry here)
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[pipeline] Skipping download for '{}': authority routed to {:?} ({})",
                        model_id, decision.result, decision.reason
                    );
                }
            }
        }

        if skipped_count > 0 {
            #[cfg(debug_assertions)]
            eprintln!(
                "[pipeline] Skipped {} downloads based on authority routing decisions",
                skipped_count
            );
        }

        Ok(())
    }

    // =========================================================================
    // Warmup Methods (for pre-loading models)
    // =========================================================================

    /// Warm up the pipeline by running a minimal inference through all stages.
    ///
    /// This pre-loads all models into memory, ensuring that the first real inference
    /// is fast. For LLM pipelines, this loads model weights and creates contexts.
    ///
    /// Call this after `load_models()` to eliminate cold-start latency.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = PipelineRef::from_yaml(yaml)?.load()?;
    /// pipeline.load_models()?;  // Download models
    /// pipeline.warmup()?;       // Pre-load into memory
    ///
    /// // First inference is now fast
    /// let result = pipeline.run(&Envelope::text("Hello"))?;
    /// ```
    pub fn warmup(&self) -> PipelineResult<()> {
        log::info!(target: "xybrid_sdk", "Warming up pipeline: {:?}", self.name);

        // Ensure models are downloaded first
        if !self.is_ready() {
            self.load_models()?;
        }

        // Create a minimal warmup input
        // Use text as it works for most model types
        let warmup_input = Envelope {
            kind: EnvelopeKind::Text("Hi".to_string()),
            metadata: std::collections::HashMap::new(),
        };

        let start = std::time::Instant::now();
        let _ = self.run(&warmup_input)?;
        let elapsed = start.elapsed();

        log::info!(
            target: "xybrid_sdk",
            "Pipeline {:?} warmed up in {:?}",
            self.name,
            elapsed
        );

        Ok(())
    }

    /// Warm up the pipeline asynchronously.
    ///
    /// This is useful for background pre-loading at app startup without blocking the UI.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = PipelineRef::from_yaml(yaml)?.load()?;
    /// pipeline.load_models()?;
    ///
    /// // Start warmup in background
    /// let pipeline_clone = pipeline.clone();
    /// let warmup_handle = tokio::spawn(async move {
    ///     pipeline_clone.warmup_async().await
    /// });
    ///
    /// // Do other initialization...
    ///
    /// // Wait for warmup if needed
    /// warmup_handle.await??;
    /// ```
    pub async fn warmup_async(&self) -> PipelineResult<()> {
        log::info!(target: "xybrid_sdk", "Warming up pipeline (async): {:?}", self.name);

        // Ensure models are downloaded first
        if !self.is_ready() {
            self.load_models()?;
        }

        // Create a minimal warmup input
        let warmup_input = Envelope {
            kind: EnvelopeKind::Text("Hi".to_string()),
            metadata: std::collections::HashMap::new(),
        };

        let start = std::time::Instant::now();
        let _ = self.run_async(&warmup_input).await?;
        let elapsed = start.elapsed();

        log::info!(
            target: "xybrid_sdk",
            "Pipeline {:?} warmed up (async) in {:?}",
            self.name,
            elapsed
        );

        Ok(())
    }

    /// Run inference on the pipeline.
    ///
    /// If models aren't loaded yet, this will automatically download them first.
    pub fn run(&self, envelope: &Envelope) -> PipelineResult<PipelineExecutionResult> {
        if !self.is_ready() {
            self.load_models()?;
        }

        let handle = self
            .handle
            .read()
            .map_err(|_| SdkError::PipelineError("Failed to acquire pipeline lock".to_string()))?;

        // Clone stage descriptors and set bundle_path on each
        let mut stage_descriptors = handle.stage_descriptors.clone();
        for desc in &mut stage_descriptors {
            if let Some(bundle_path) = handle.bundle_paths.get(&desc.name) {
                desc.bundle_path = Some(bundle_path.to_string_lossy().to_string());
            }
        }
        let availability_map = handle.availability_map.clone();
        drop(handle);

        // Collect runtime metrics from device
        let device_adapter = LocalDeviceAdapter::new();
        let metrics = device_adapter.collect_metrics();

        // Set telemetry context
        let trace_id = uuid::Uuid::new_v4();
        let pipeline_id = self
            .name
            .as_ref()
            .map(|n| uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, n.as_bytes()));
        crate::telemetry::set_telemetry_pipeline_context(pipeline_id, Some(trace_id));

        let mut orchestrator = Orchestrator::new();
        // No need to set registry config - executor uses bundle_path from stage descriptors

        let availability_fn = move |stage: &str| -> LocalAvailability {
            let exists = availability_map.get(stage).copied().unwrap_or(false);
            LocalAvailability::new(exists)
        };

        let start_time = std::time::Instant::now();
        let results: Vec<StageExecutionResult> = orchestrator
            .execute_pipeline(&stage_descriptors, envelope, &metrics, &availability_fn)
            .map_err(|e| {
                crate::telemetry::set_telemetry_pipeline_context(None, None);
                SdkError::PipelineError(format!("Pipeline execution failed: {}", e))
            })?;
        let total_latency_ms = start_time.elapsed().as_millis() as u32;

        crate::telemetry::set_telemetry_pipeline_context(None, None);

        let stages: Vec<StageTiming> = results
            .iter()
            .map(|result| StageTiming {
                name: result.stage.clone(),
                latency_ms: result.latency_ms,
                target: result.routing_decision.target.to_string(),
                reason: result.routing_decision.reason.clone(),
            })
            .collect();

        let (output_type, output) = if let Some(last) = results.last() {
            let output_type = match &last.output.kind {
                EnvelopeKind::Text(_) => OutputType::Text,
                EnvelopeKind::Audio(_) => OutputType::Audio,
                EnvelopeKind::Embedding(_) => OutputType::Embedding,
            };
            (output_type, last.output.clone())
        } else {
            (
                OutputType::Unknown,
                Envelope::new(EnvelopeKind::Text(String::new())),
            )
        };

        // Emit telemetry event
        let event = crate::telemetry::TelemetryEvent {
            event_type: "PipelineComplete".to_string(),
            stage_name: self.name.clone(),
            target: None,
            latency_ms: Some(total_latency_ms),
            error: None,
            data: Some(
                serde_json::json!({
                    "stages": stages.iter().map(|s| serde_json::json!({
                        "name": s.name,
                        "latency_ms": s.latency_ms,
                        "target": s.target,
                    })).collect::<Vec<_>>(),
                    "output_type": format!("{:?}", output_type),
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_telemetry_event(event);

        Ok(PipelineExecutionResult {
            name: self.name.clone(),
            stages,
            total_latency_ms,
            output_type,
            output,
        })
    }

    /// Run inference asynchronously.
    pub async fn run_async(&self, envelope: &Envelope) -> PipelineResult<PipelineExecutionResult> {
        if !self.is_ready() {
            self.load_models()?;
        }

        let (stage_descriptors, availability_map) = {
            // Recover from poisoned RwLock to prevent permanent lock errors
            let handle = self.handle.read().unwrap_or_else(|e| e.into_inner());

            // Clone stage descriptors and set bundle_path on each
            let mut descriptors = handle.stage_descriptors.clone();
            for desc in &mut descriptors {
                if let Some(bundle_path) = handle.bundle_paths.get(&desc.name) {
                    desc.bundle_path = Some(bundle_path.to_string_lossy().to_string());
                }
            }

            (descriptors, handle.availability_map.clone())
        };

        let envelope_clone = envelope.clone();
        let name = self.name.clone();

        tokio::task::spawn_blocking(move || {
            // Collect runtime metrics from device
            let device_adapter = LocalDeviceAdapter::new();
            let metrics = device_adapter.collect_metrics();

            let trace_id = uuid::Uuid::new_v4();
            let pipeline_id = name
                .as_ref()
                .map(|n| uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, n.as_bytes()));
            crate::telemetry::set_telemetry_pipeline_context(pipeline_id, Some(trace_id));

            let mut orchestrator = Orchestrator::new();
            // No need to set registry config - executor uses bundle_path from stage descriptors

            let availability_fn = move |stage: &str| -> LocalAvailability {
                let exists = availability_map.get(stage).copied().unwrap_or(false);
                LocalAvailability::new(exists)
            };

            let start_time = std::time::Instant::now();
            let results: Vec<StageExecutionResult> = orchestrator
                .execute_pipeline(
                    &stage_descriptors,
                    &envelope_clone,
                    &metrics,
                    &availability_fn,
                )
                .map_err(|e| {
                    crate::telemetry::set_telemetry_pipeline_context(None, None);
                    SdkError::PipelineError(format!("Pipeline execution failed: {}", e))
                })?;
            let total_latency_ms = start_time.elapsed().as_millis() as u32;

            crate::telemetry::set_telemetry_pipeline_context(None, None);

            let stages: Vec<StageTiming> = results
                .iter()
                .map(|result| StageTiming {
                    name: result.stage.clone(),
                    latency_ms: result.latency_ms,
                    target: result.routing_decision.target.to_string(),
                    reason: result.routing_decision.reason.clone(),
                })
                .collect();

            let (output_type, output) = if let Some(last) = results.last() {
                let output_type = match &last.output.kind {
                    EnvelopeKind::Text(_) => OutputType::Text,
                    EnvelopeKind::Audio(_) => OutputType::Audio,
                    EnvelopeKind::Embedding(_) => OutputType::Embedding,
                };
                (output_type, last.output.clone())
            } else {
                (
                    OutputType::Unknown,
                    Envelope::new(EnvelopeKind::Text(String::new())),
                )
            };

            Ok(PipelineExecutionResult {
                name,
                stages,
                total_latency_ms,
                output_type,
                output,
            })
        })
        .await
        .map_err(|e| SdkError::PipelineError(format!("Task join error: {}", e)))?
    }
}

// Make Pipeline cloneable (shares the handle via Arc)
impl Clone for Pipeline {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            handle: self.handle.clone(),
            stages: self.stages.clone(),
            total_download_bytes: self.total_download_bytes,
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Convenience struct for running pipelines in one call.
pub struct Xybrid;

impl Xybrid {
    /// Run a pipeline from YAML in one call.
    ///
    /// This is the simplest way to run a pipeline - it handles everything:
    /// parsing, model resolution, downloading, and execution.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xybrid_sdk::{Xybrid, Envelope};
    ///
    /// let result = Xybrid::run_pipeline(yaml_content, &Envelope::audio(audio_bytes))?;
    /// println!("Output: {:?}", result.text());
    /// ```
    pub fn run_pipeline(
        yaml: &str,
        envelope: &Envelope,
    ) -> PipelineResult<PipelineExecutionResult> {
        let pipeline = PipelineRef::from_yaml(yaml)?.load()?;
        pipeline.run(envelope)
    }

    /// Create a pipeline reference from YAML.
    ///
    /// This is equivalent to `PipelineRef::from_yaml()`.
    pub fn pipeline(yaml: &str) -> PipelineResult<PipelineRef> {
        PipelineRef::from_yaml(yaml)
    }

    /// Run a pipeline with streaming output for LLM stages.
    ///
    /// This method is similar to `run_pipeline` but calls the provided callback
    /// for each generated token during LLM inference. This enables real-time
    /// display of generated text.
    ///
    /// # Arguments
    ///
    /// * `yaml` - Pipeline YAML content
    /// * `envelope` - Input envelope
    /// * `on_token` - Callback invoked for each generated token
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xybrid_sdk::{Xybrid, Envelope, PartialToken};
    /// use std::io::Write;
    ///
    /// let result = Xybrid::run_pipeline_streaming(
    ///     yaml_content,
    ///     &Envelope::text("Hello, how are you?"),
    ///     Box::new(|token: PartialToken| {
    ///         print!("{}", token.token);
    ///         std::io::stdout().flush()?;
    ///         Ok(())
    ///     }),
    /// )?;
    /// ```
    ///
    /// # Note
    ///
    /// Streaming is only supported for LLM stages (GGUF models). For other
    /// model types, this behaves identically to `run_pipeline`.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    pub fn run_pipeline_streaming<'a>(
        yaml: &str,
        envelope: &Envelope,
        on_token: xybrid_core::runtime_adapter::llm::StreamingCallback<'a>,
    ) -> PipelineResult<PipelineExecutionResult> {
        use xybrid_core::execution::{ModelMetadata, TemplateExecutor};

        let pipeline_ref = PipelineRef::from_yaml(yaml)?;
        let pipeline = pipeline_ref.load()?;

        // Load models if needed
        if !pipeline.is_ready() {
            pipeline.load_models()?;
        }

        let handle = pipeline
            .handle
            .read()
            .map_err(|_| SdkError::PipelineError("Failed to acquire pipeline lock".to_string()))?;

        // For streaming, we need to identify if there's an LLM stage and execute it with streaming
        // For now, support single-stage LLM pipelines
        if handle.stage_descriptors.len() == 1 {
            let stage_name = handle.stage_descriptors[0].name.clone();
            if let Some(bundle_path) = handle.bundle_paths.get(&stage_name) {
                let bundle_path = bundle_path.clone();  // Clone to avoid borrow issues
                let metadata_path = bundle_path.join("model_metadata.json");
                if metadata_path.exists() {
                    let metadata_str = std::fs::read_to_string(&metadata_path)
                        .map_err(|e| SdkError::PipelineError(format!("Failed to read metadata: {}", e)))?;
                    let metadata: ModelMetadata = serde_json::from_str(&metadata_str)
                        .map_err(|e| SdkError::PipelineError(format!("Failed to parse metadata: {}", e)))?;

                    // Check if this is an LLM model
                    if matches!(
                        metadata.execution_template,
                        xybrid_core::execution::ExecutionTemplate::Gguf { .. }
                    ) {
                        drop(handle);  // Release lock before executor call

                        let mut executor = TemplateExecutor::with_base_path(
                            bundle_path.to_str().unwrap_or("")
                        );

                        let start_time = std::time::Instant::now();
                        let output = executor
                            .execute_streaming(&metadata, envelope, on_token)
                            .map_err(|e| SdkError::InferenceError(format!("{}", e)))?;
                        let total_latency_ms = start_time.elapsed().as_millis() as u32;

                        let output_type = match &output.kind {
                            EnvelopeKind::Text(_) => OutputType::Text,
                            EnvelopeKind::Audio(_) => OutputType::Audio,
                            EnvelopeKind::Embedding(_) => OutputType::Embedding,
                        };

                        return Ok(PipelineExecutionResult {
                            name: pipeline.name.clone(),
                            stages: vec![StageTiming {
                                name: pipeline_ref.config.stages[0].model_id(),
                                latency_ms: total_latency_ms,
                                target: "local".to_string(),
                                reason: "streaming".to_string(),
                            }],
                            total_latency_ms,
                            output_type,
                            output,
                        });
                    }
                }
            }
        }

        // Fall back to non-streaming execution for multi-stage or non-LLM pipelines
        drop(handle);
        pipeline.run(envelope)
    }

    /// Stub for when LLM features are disabled.
    ///
    /// Without LLM features, streaming is not available and this falls back
    /// to regular pipeline execution.
    #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
    pub fn run_pipeline_streaming<F>(
        yaml: &str,
        envelope: &Envelope,
        _on_token: F,
    ) -> PipelineResult<PipelineExecutionResult>
    where
        F: FnMut(()) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        // Without LLM features, just run normally
        Self::run_pipeline(yaml, envelope)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_ref_from_yaml() {
        let yaml = r#"
name: "Test Pipeline"
stages:
  - test-stage@1.0
"#;
        let ref_ = PipelineRef::from_yaml(yaml).unwrap();
        assert_eq!(ref_.name(), Some("Test Pipeline"));
        assert_eq!(ref_.stage_count(), 1);
    }

    #[test]
    fn test_pipeline_ref_stage_ids() {
        let yaml = r#"
name: "Multi-Stage"
stages:
  - id: asr
    model: wav2vec2-base-960h
    version: "1.0"
  - id: llm
    model: gpt-4o-mini
    provider: openai
  - id: tts
    model: kokoro-82m
"#;
        let ref_ = PipelineRef::from_yaml(yaml).unwrap();
        assert_eq!(ref_.stage_ids(), vec!["asr", "llm", "tts"]);
        assert_eq!(ref_.stage_count(), 3);
    }

    #[test]
    fn test_stage_target_display() {
        assert_eq!(StageTarget::Device.to_string(), "device");
        assert_eq!(StageTarget::Auto.to_string(), "auto");
        assert_eq!(StageTarget::Cloud.to_string(), "cloud");
        assert_eq!(
            StageTarget::Integration {
                provider: "openai".to_string()
            }
            .to_string(),
            "integration:openai"
        );
    }

    #[test]
    fn test_stage_status_display() {
        assert_eq!(StageStatus::Cached.to_string(), "cached");
        assert_eq!(StageStatus::NeedsDownload.to_string(), "needs_download");
        assert_eq!(StageStatus::Integration.to_string(), "integration");
        assert_eq!(
            StageStatus::Error("failed".to_string()).to_string(),
            "error: failed"
        );
    }

    #[test]
    fn test_download_progress_serialization() {
        let progress = DownloadProgress {
            model_id: "kokoro-82m".to_string(),
            percent: 75,
            bytes_downloaded: 150_000_000,
            bytes_total: 200_000_000,
            stage_index: 1,
            total_stages: 2,
        };

        let json = serde_json::to_string(&progress).unwrap();
        assert!(json.contains("\"model_id\":\"kokoro-82m\""));
        assert!(json.contains("\"percent\":75"));
    }

    #[test]
    fn test_stage_info_serialization() {
        let info = StageInfo {
            id: "asr".to_string(),
            model_id: Some("wav2vec2".to_string()),
            target: StageTarget::Device,
            status: StageStatus::Cached,
            download_bytes: None,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"id\":\"asr\""));
        assert!(json.contains("\"status\":\"Cached\""));
    }

    #[test]
    fn test_xybrid_pipeline_convenience() {
        let yaml = r#"
name: "Test"
stages:
  - test-stage@1.0
"#;
        let ref_ = Xybrid::pipeline(yaml).unwrap();
        assert_eq!(ref_.name(), Some("Test"));
    }

    #[test]
    fn test_pipeline_input_type_methods() {
        assert!(PipelineInputType::Audio.is_audio());
        assert!(!PipelineInputType::Audio.is_text());
        assert!(PipelineInputType::Text.is_text());
        assert!(!PipelineInputType::Text.is_audio());
    }

    #[test]
    fn test_stage_config_model_id() {
        // Test simple string format
        let stage: StageConfig = serde_yaml::from_str(r#""kokoro-82m""#).unwrap();
        assert_eq!(stage.model_id(), "kokoro-82m");

        // Test simple string with version (model_id strips version)
        let stage: StageConfig = serde_yaml::from_str(r#""wav2vec2@1.0""#).unwrap();
        assert_eq!(stage.model_id(), "wav2vec2");

        // Test object format with model only
        let stage: StageConfig = serde_yaml::from_str(
            r#"
model: gpt-4o-mini
"#,
        )
        .unwrap();
        assert_eq!(stage.model_id(), "gpt-4o-mini");

        // Test id fallback when no model specified
        let stage: StageConfig = serde_yaml::from_str(
            r#"
id: asr
"#,
        )
        .unwrap();
        assert_eq!(stage.stage_id(), "asr");
    }
}
