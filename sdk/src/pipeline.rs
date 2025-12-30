//! Pipeline loading and execution for xybrid-sdk.
//!
//! This module provides a simple two-type API:
//! - `PipelineRef`: Lightweight reference from parsed YAML (no network)
//! - `Pipeline`: Loaded pipeline ready to preload models and run
//!
//! # Example (Simple - just run)
//!
//! ```rust,no_run
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
//! ```rust,no_run
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
use xybrid_core::context::{DeviceMetrics, StageDescriptor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::{Orchestrator, StageExecutionResult};
use xybrid_core::pipeline::{ExecutionTarget, IntegrationProvider, StageOptions};
use xybrid_core::orchestrator::routing_engine::LocalAvailability;

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
/// ```rust,no_run
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
        self.config
            .stages
            .iter()
            .map(|s| s.get_id().unwrap_or_else(|| s.get_name()))
            .collect()
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
    /// Runs on device using local model
    Device,
    /// Runs on cloud server
    Cloud,
    /// Runs via integration provider (e.g., OpenAI API)
    Integration { provider: String },
}

impl std::fmt::Display for StageTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageTarget::Device => write!(f, "device"),
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


/// Stage configuration - can be a string (name) or an object with full configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum StageConfig {
    Simple(String),
    Object {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        version: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        target: Option<String>,
        #[serde(default)]
        provider: Option<String>,
        #[serde(default)]
        options: Option<HashMap<String, serde_json::Value>>,
    },
}

impl StageConfig {
    fn get_name(&self) -> String {
        match self {
            StageConfig::Simple(name) => name.clone(),
            StageConfig::Object { id, model, version, name, .. } => {
                if let Some(model_name) = model {
                    if let Some(ver) = version {
                        format!("{}@{}", model_name, ver)
                    } else {
                        model_name.clone()
                    }
                } else if let Some(n) = name {
                    n.clone()
                } else if let Some(stage_id) = id {
                    stage_id.clone()
                } else {
                    "unknown".to_string()
                }
            }
        }
    }

    fn get_id(&self) -> Option<String> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { id, .. } => id.clone(),
        }
    }

    fn get_target(&self) -> Option<String> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { target, .. } => target.clone(),
        }
    }

    fn get_provider(&self) -> Option<String> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { provider, .. } => provider.clone(),
        }
    }

    fn get_model(&self) -> Option<String> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { model, .. } => model.clone(),
        }
    }

    fn get_options(&self) -> Option<HashMap<String, serde_json::Value>> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { options, .. } => options.clone(),
        }
    }
}

/// Registry URL configuration - can be a string (URL) or an object with base_url.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum RegistryUrlConfig {
    Simple(String),
    Object { base_url: String },
}

impl RegistryUrlConfig {
    fn url(&self) -> &str {
        match self {
            RegistryUrlConfig::Simple(url) => url,
            RegistryUrlConfig::Object { base_url } => base_url,
        }
    }
}

/// Pipeline configuration loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PipelineConfig {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    registry: Option<RegistryUrlConfig>,
    stages: Vec<StageConfig>,
    #[serde(default)]
    input: Option<InputConfig>,
    #[serde(default)]
    metrics: Option<MetricsConfig>,
    #[serde(default)]
    availability: HashMap<String, bool>,
}

/// Input envelope configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InputConfig {
    #[serde(default, rename = "type")]
    input_type: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    channels: Option<u8>,
    #[serde(default)]
    data: Option<String>,
}

impl InputConfig {
    fn get_type(&self) -> Option<&str> {
        self.input_type.as_deref().or(self.kind.as_deref())
    }
}

/// Input type for pipeline (public API)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineInputType {
    Audio,
    Text,
    Embedding,
    Unknown,
}

impl PipelineInputType {
    fn from_kind(kind: &str) -> Self {
        match kind.to_lowercase().as_str() {
            "audio" | "audioraw" | "audio_raw" => PipelineInputType::Audio,
            "text" => PipelineInputType::Text,
            "embedding" => PipelineInputType::Embedding,
            _ => PipelineInputType::Unknown,
        }
    }

    pub fn is_audio(&self) -> bool {
        matches!(self, PipelineInputType::Audio)
    }

    pub fn is_text(&self) -> bool {
        matches!(self, PipelineInputType::Text)
    }
}

/// Device metrics configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricsConfig {
    #[serde(default)]
    network_rtt: u32,
    #[serde(default = "default_battery")]
    battery: u8,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

fn default_battery() -> u8 { 100 }
fn default_temperature() -> f32 { 25.0 }

impl Default for MetricsConfig {
    fn default() -> Self {
        MetricsConfig {
            network_rtt: 50,
            battery: 100,
            temperature: 25.0,
        }
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
    metrics: DeviceMetrics,
    availability_map: HashMap<String, bool>,
    /// Registry URL (for downloading models)
    registry_url: Option<String>,
    input_type: PipelineInputType,
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
/// ```rust,no_run
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
                let name = stage_config.get_name();
                let mut desc = StageDescriptor::new(name);

                if let Some(target_str) = stage_config.get_target() {
                    desc.target = Self::parse_target(&target_str);
                }

                if let Some(provider_str) = stage_config.get_provider() {
                    desc.provider = Self::parse_provider(&provider_str);
                    if desc.target.is_none() {
                        desc.target = Some(ExecutionTarget::Cloud);
                    }
                }

                desc.model = stage_config.get_model();

                if let Some(opts) = stage_config.get_options() {
                    desc.options = Some(Self::convert_options(&opts));
                }

                desc
            })
            .collect();

        // Extract registry URL (just the URL, not the full config)
        let registry_url = config.registry.as_ref().map(|r| r.url().to_string());

        let metrics_config = config.metrics.clone().unwrap_or_default();
        let metrics = DeviceMetrics {
            network_rtt: metrics_config.network_rtt,
            battery: metrics_config.battery,
            temperature: metrics_config.temperature,
        };

        let input_type = config
            .input
            .as_ref()
            .and_then(|i| i.get_type())
            .map(PipelineInputType::from_kind)
            .unwrap_or(PipelineInputType::Unknown);

        let stage_configs = config.stages.clone();

        let handle = PipelineHandle {
            stage_descriptors,
            metrics,
            availability_map: config.availability.clone(),
            registry_url,
            input_type,
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
        let registry_url = config.registry.as_ref().map(|r| r.url().to_string());

        let client = if let Some(url) = registry_url {
            RegistryClient::new(url)?
        } else {
            RegistryClient::from_env()?
        };

        let mut stages = Vec::new();
        let mut total_download_bytes: u64 = 0;

        let handle_read = handle.read().map_err(|_| {
            SdkError::PipelineError("Failed to read pipeline handle".to_string())
        })?;

        for stage_config in &config.stages {
            let stage_id = stage_config.get_id().unwrap_or_else(|| stage_config.get_name());
            let target_str = stage_config.get_target();
            let provider = stage_config.get_provider();
            let model_name = stage_config.get_model();

            let stage_target = if provider.is_some() || target_str.as_deref() == Some("integration") {
                StageTarget::Integration {
                    provider: provider.unwrap_or_else(|| "unknown".to_string()),
                }
            } else if target_str.as_deref() == Some("cloud") || target_str.as_deref() == Some("server") {
                StageTarget::Cloud
            } else {
                StageTarget::Device
            };

            let (status, download_bytes) = if matches!(stage_target, StageTarget::Device) {
                if let Some(ref model_id) = model_name {
                    match client.resolve(model_id, None) {
                        Ok(resolved) => {
                            let is_cached = client.is_cached(model_id, None).unwrap_or(false);
                            if is_cached {
                                (StageStatus::Cached, None)
                            } else {
                                total_download_bytes += resolved.size_bytes;
                                (StageStatus::NeedsDownload, Some(resolved.size_bytes))
                            }
                        }
                        Err(e) => (StageStatus::Error(e.to_string()), None),
                    }
                } else {
                    let name = stage_config.get_name();
                    if handle_read.availability_map.get(&name).copied().unwrap_or(false) {
                        (StageStatus::Cached, None)
                    } else {
                        (StageStatus::NeedsDownload, None)
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
    pub fn input_type(&self) -> PipelineInputType {
        self.handle
            .read()
            .ok()
            .map(|h| h.input_type)
            .unwrap_or(PipelineInputType::Unknown)
    }

    /// Check if all device models are cached and ready.
    pub fn is_ready(&self) -> bool {
        self.stages.iter().all(|s| {
            matches!(s.status, StageStatus::Cached | StageStatus::Integration)
        })
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
    pub fn load_models_with_progress<F>(&self, progress_callback: F) -> PipelineResult<()>
    where
        F: Fn(DownloadProgress),
    {
        let registry_url = self.handle.read()
            .map_err(|_| SdkError::PipelineError("Failed to read handle".to_string()))?
            .registry_url
            .clone();

        let client = if let Some(url) = registry_url {
            RegistryClient::new(url)?
        } else {
            RegistryClient::from_env()?
        };

        let stages_to_fetch: Vec<_> = self.stages
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s.status, StageStatus::NeedsDownload))
            .filter_map(|(idx, s)| {
                s.model_id.as_ref().map(|m| (idx, s.id.clone(), m.clone(), s.download_bytes.unwrap_or(0)))
            })
            .collect();

        let total_stages = stages_to_fetch.len();

        for (stage_idx, (_, stage_id, model_id, total_bytes)) in stages_to_fetch.into_iter().enumerate() {
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

            // Fetch model and get the bundle path
            let bundle_path = client.fetch(&model_id, None, progress_for_model)?;

            {
                let mut handle = self.handle.write().map_err(|_| {
                    SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
                })?;

                handle.availability_map.insert(model_id.clone(), true);
                handle.availability_map.insert(stage_id.clone(), true);
                // Store the bundle path for this stage
                handle.bundle_paths.insert(stage_id.clone(), bundle_path.clone());
                handle.bundle_paths.insert(model_id.clone(), bundle_path);
            }
        }

        Ok(())
    }

    /// Run inference on the pipeline.
    ///
    /// If models aren't loaded yet, this will automatically download them first.
    pub fn run(&self, envelope: &Envelope) -> PipelineResult<PipelineExecutionResult> {
        if !self.is_ready() {
            self.load_models()?;
        }

        let handle = self.handle.read().map_err(|_| {
            SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
        })?;

        // Clone stage descriptors and set bundle_path on each
        let mut stage_descriptors = handle.stage_descriptors.clone();
        for desc in &mut stage_descriptors {
            if let Some(bundle_path) = handle.bundle_paths.get(&desc.name) {
                desc.bundle_path = Some(bundle_path.to_string_lossy().to_string());
            }
        }
        let metrics = handle.metrics.clone();
        let availability_map = handle.availability_map.clone();
        drop(handle);

        // Set telemetry context
        let trace_id = uuid::Uuid::new_v4();
        let pipeline_id = self.name.as_ref().map(|n| {
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, n.as_bytes())
        });
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
            (OutputType::Unknown, Envelope::new(EnvelopeKind::Text(String::new())))
        };

        // Emit telemetry event
        let event = crate::telemetry::TelemetryEvent {
            event_type: "PipelineComplete".to_string(),
            stage_name: self.name.clone(),
            target: None,
            latency_ms: Some(total_latency_ms),
            error: None,
            data: Some(serde_json::json!({
                "stages": stages.iter().map(|s| serde_json::json!({
                    "name": s.name,
                    "latency_ms": s.latency_ms,
                    "target": s.target,
                })).collect::<Vec<_>>(),
                "output_type": format!("{:?}", output_type),
            }).to_string()),
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

        let (stage_descriptors, metrics, availability_map) = {
            let handle = self.handle.read().map_err(|_| {
                SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
            })?;

            // Clone stage descriptors and set bundle_path on each
            let mut descriptors = handle.stage_descriptors.clone();
            for desc in &mut descriptors {
                if let Some(bundle_path) = handle.bundle_paths.get(&desc.name) {
                    desc.bundle_path = Some(bundle_path.to_string_lossy().to_string());
                }
            }

            (
                descriptors,
                handle.metrics.clone(),
                handle.availability_map.clone(),
            )
        };

        let envelope_clone = envelope.clone();
        let name = self.name.clone();

        tokio::task::spawn_blocking(move || {
            let trace_id = uuid::Uuid::new_v4();
            let pipeline_id = name.as_ref().map(|n| {
                uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, n.as_bytes())
            });
            crate::telemetry::set_telemetry_pipeline_context(pipeline_id, Some(trace_id));

            let mut orchestrator = Orchestrator::new();
            // No need to set registry config - executor uses bundle_path from stage descriptors

            let availability_fn = move |stage: &str| -> LocalAvailability {
                let exists = availability_map.get(stage).copied().unwrap_or(false);
                LocalAvailability::new(exists)
            };

            let start_time = std::time::Instant::now();
            let results: Vec<StageExecutionResult> = orchestrator
                .execute_pipeline(&stage_descriptors, &envelope_clone, &metrics, &availability_fn)
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
                (OutputType::Unknown, Envelope::new(EnvelopeKind::Text(String::new())))
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
    /// ```rust,no_run
    /// use xybrid_sdk::{Xybrid, Envelope};
    ///
    /// let result = Xybrid::run_pipeline(yaml_content, &Envelope::audio(audio_bytes))?;
    /// println!("Output: {:?}", result.text());
    /// ```
    pub fn run_pipeline(yaml: &str, envelope: &Envelope) -> PipelineResult<PipelineExecutionResult> {
        let pipeline = PipelineRef::from_yaml(yaml)?.load()?;
        pipeline.run(envelope)
    }

    /// Create a pipeline reference from YAML.
    ///
    /// This is equivalent to `PipelineRef::from_yaml()`.
    pub fn pipeline(yaml: &str) -> PipelineResult<PipelineRef> {
        PipelineRef::from_yaml(yaml)
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
        assert_eq!(StageTarget::Cloud.to_string(), "cloud");
        assert_eq!(StageTarget::Integration { provider: "openai".to_string() }.to_string(), "integration:openai");
    }

    #[test]
    fn test_stage_status_display() {
        assert_eq!(StageStatus::Cached.to_string(), "cached");
        assert_eq!(StageStatus::NeedsDownload.to_string(), "needs_download");
        assert_eq!(StageStatus::Integration.to_string(), "integration");
        assert_eq!(StageStatus::Error("failed".to_string()).to_string(), "error: failed");
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
    fn test_input_type_from_kind() {
        assert_eq!(PipelineInputType::from_kind("audio"), PipelineInputType::Audio);
        assert_eq!(PipelineInputType::from_kind("Audio"), PipelineInputType::Audio);
        assert_eq!(PipelineInputType::from_kind("text"), PipelineInputType::Text);
        assert_eq!(PipelineInputType::from_kind("embedding"), PipelineInputType::Embedding);
        assert_eq!(PipelineInputType::from_kind("unknown"), PipelineInputType::Unknown);
    }

    #[test]
    fn test_pipeline_input_type_methods() {
        assert!(PipelineInputType::Audio.is_audio());
        assert!(!PipelineInputType::Audio.is_text());
        assert!(PipelineInputType::Text.is_text());
        assert!(!PipelineInputType::Text.is_audio());
    }

    #[test]
    fn test_stage_config_get_name_priority() {
        // Test model@version format
        let stage: StageConfig = serde_yaml::from_str(r#"
model: wav2vec2
version: "1.0"
"#).unwrap();
        assert_eq!(stage.get_name(), "wav2vec2@1.0");

        // Test model only format
        let stage: StageConfig = serde_yaml::from_str(r#"
model: gpt-4o-mini
"#).unwrap();
        assert_eq!(stage.get_name(), "gpt-4o-mini");

        // Test legacy name format
        let stage: StageConfig = serde_yaml::from_str(r#"
name: "legacy-model@1.0"
"#).unwrap();
        assert_eq!(stage.get_name(), "legacy-model@1.0");

        // Test id fallback
        let stage: StageConfig = serde_yaml::from_str(r#"
id: asr
"#).unwrap();
        assert_eq!(stage.get_name(), "asr");
    }
}
