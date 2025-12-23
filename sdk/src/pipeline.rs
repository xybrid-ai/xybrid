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
//!
//! # Backwards Compatibility
//!
//! The old `PipelineLoader` and `XybridPipeline` types are still available
//! as aliases for backwards compatibility.

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
use xybrid_core::registry_config::RegistryConfig;
use xybrid_core::routing_engine::LocalAvailability;

/// Result type for pipeline operations.
pub type PipelineResult<T> = Result<T, SdkError>;

// ============================================================================
// Pipeline Status and Execution Plan Types
// ============================================================================

// ============================================================================
// New API Types: PipelineRef and Pipeline
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
        // In the future, this could do async model resolution
        self.load()
    }
}

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
// Legacy Status Types (kept for backwards compatibility)
// ============================================================================

/// Current status of the pipeline lifecycle.
///
/// **Deprecated**: Use `Pipeline::is_ready()` instead.
///
/// A pipeline transitions through these states:
/// - `Created` → Initial state after parsing YAML
/// - `Prepared` → After `prepare()`: configuration validated
/// - `Planned` → After `plan()`: models resolved, download plan computed
/// - `Ready` → After `fetch()`: all models cached and ready
/// - `Running` → During `run()`: currently executing
/// - `Completed` → After successful `run()`
/// - `Error` → If any operation failed
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStatus {
    /// Initial state after creation
    Created,
    /// YAML parsed and validated, stages extracted
    Prepared,
    /// Models resolved, execution plan computed
    Planned,
    /// All required models fetched and cached
    Ready,
    /// Currently executing
    Running,
    /// Execution completed
    Completed,
    /// Error occurred (with message)
    Error(String),
}

impl Default for PipelineStatus {
    fn default() -> Self {
        PipelineStatus::Created
    }
}

impl std::fmt::Display for PipelineStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineStatus::Created => write!(f, "created"),
            PipelineStatus::Prepared => write!(f, "prepared"),
            PipelineStatus::Planned => write!(f, "planned"),
            PipelineStatus::Ready => write!(f, "ready"),
            PipelineStatus::Running => write!(f, "running"),
            PipelineStatus::Completed => write!(f, "completed"),
            PipelineStatus::Error(msg) => write!(f, "error: {}", msg),
        }
    }
}

/// Execution plan computed by `plan()`.
///
/// Contains information about each stage's model, cache status,
/// and total download requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Pipeline name
    pub name: Option<String>,
    /// Per-stage information
    pub stages: Vec<StagePlan>,
    /// Total bytes to download (for stages needing download)
    pub total_download_bytes: u64,
    /// Whether pipeline can run offline (all device models cached, no integration stages)
    pub offline_capable: bool,
    /// Stage IDs requiring network (integration targets)
    pub requires_network: Vec<String>,
}

/// Plan information for a single stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePlan {
    /// Stage identifier (from YAML `id` field or model name)
    pub id: String,
    /// Model ID (if this stage uses a model)
    pub model_id: Option<String>,
    /// Execution target
    pub target: StageTarget,
    /// Current status (cached, needs download, etc.)
    pub status: StageReadyStatus,
    /// Download size in bytes (if needs download)
    pub download_bytes: Option<u64>,
    /// Model format and quantization (e.g., "onnx-fp16")
    pub format: Option<String>,
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

/// Status of a stage's readiness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageReadyStatus {
    /// Model is cached and ready
    Cached,
    /// Model needs to be downloaded
    NeedsDownload,
    /// Integration stage (no local model needed)
    Integration,
    /// Model resolution failed
    ResolutionFailed(String),
}

impl std::fmt::Display for StageReadyStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageReadyStatus::Cached => write!(f, "cached"),
            StageReadyStatus::NeedsDownload => write!(f, "needs_download"),
            StageReadyStatus::Integration => write!(f, "integration"),
            StageReadyStatus::ResolutionFailed(msg) => write!(f, "error: {}", msg),
        }
    }
}

/// Progress information during model fetching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchProgress {
    /// Model currently being downloaded
    pub model_id: String,
    /// Download progress (0-100)
    pub percent: u32,
    /// Bytes downloaded so far for this model
    pub bytes_downloaded: u64,
    /// Total bytes for this model
    pub bytes_total: u64,
    /// Current stage index (0-based)
    pub stage_index: usize,
    /// Total stages needing download
    pub total_stages: usize,
}

/// Stage registry configuration - can be a string (URL) or RegistryConfig object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum StageRegistryConfig {
    /// Simple string format: "https://registry.example.com"
    Simple(String),
    /// Full RegistryConfig object
    Full(RegistryConfig),
}

/// Stage configuration - can be a string (name) or an object with full configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum StageConfig {
    /// Simple string format: "whisper-tiny@1.2"
    Simple(String),
    /// Full object format with all stage configuration options
    Object {
        /// Stage ID (new format) - e.g., "asr", "llm", "tts"
        #[serde(default)]
        id: Option<String>,
        /// Model name - e.g., "wav2vec2-base-960h", "gpt-4o-mini"
        #[serde(default)]
        model: Option<String>,
        /// Model version - e.g., "1.0"
        #[serde(default)]
        version: Option<String>,
        /// Legacy name field (for backwards compatibility) - e.g., "whisper-tiny@1.0"
        #[serde(default)]
        name: Option<String>,
        /// Execution target: "device", "integration", "server", "auto"
        #[serde(default)]
        target: Option<String>,
        /// Integration provider: "openai", "anthropic", "google", etc.
        #[serde(default)]
        provider: Option<String>,
        /// Stage-specific options (temperature, max_tokens, system_prompt, etc.)
        #[serde(default)]
        options: Option<HashMap<String, serde_json::Value>>,
        /// Stage-level registry configuration
        #[serde(default)]
        registry: Option<StageRegistryConfig>,
    },
}

impl StageConfig {
    /// Get the stage name/identifier.
    /// For new format: uses "model@version" or "model" or "id".
    /// For legacy format: uses "name" or the simple string.
    fn get_name(&self) -> String {
        match self {
            StageConfig::Simple(name) => name.clone(),
            StageConfig::Object { id, model, version, name, .. } => {
                // Priority: model@version > model > name > id
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

    /// Get the stage ID (for display/identification purposes).
    fn get_id(&self) -> Option<String> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { id, .. } => id.clone(),
        }
    }

    /// Get the execution target.
    fn get_target(&self) -> Option<String> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { target, .. } => target.clone(),
        }
    }

    /// Get the integration provider.
    fn get_provider(&self) -> Option<String> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { provider, .. } => provider.clone(),
        }
    }

    /// Get the model name.
    fn get_model(&self) -> Option<String> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { model, .. } => model.clone(),
        }
    }

    /// Get the stage options.
    fn get_options(&self) -> Option<HashMap<String, serde_json::Value>> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { options, .. } => options.clone(),
        }
    }

    /// Get the registry config.
    fn get_registry(&self) -> Option<StageRegistryConfig> {
        match self {
            StageConfig::Simple(_) => None,
            StageConfig::Object { registry, .. } => registry.clone(),
        }
    }
}

/// Registry configuration value - can be a string (URL) or an object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum RegistryConfigValue {
    /// Simple string format: "https://registry.example.com"
    Simple(String),
    /// Full RegistryConfig object
    Full(RegistryConfig),
}

/// Pipeline configuration loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PipelineConfig {
    /// Pipeline name
    #[serde(default)]
    name: Option<String>,
    /// Pipeline-level registry configuration
    #[serde(default)]
    registry: Option<RegistryConfigValue>,
    /// Stage configurations
    stages: Vec<StageConfig>,
    /// Input envelope configuration
    #[serde(default)]
    input: Option<InputConfig>,
    /// Device metrics configuration
    #[serde(default)]
    metrics: Option<MetricsConfig>,
    /// Model availability mapping
    #[serde(default)]
    availability: HashMap<String, bool>,
}

/// Input envelope configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InputConfig {
    /// Input type (new format): "audio", "text"
    #[serde(default, rename = "type")]
    input_type: Option<String>,
    /// Legacy kind field (for backwards compatibility): "AudioRaw", "Text"
    #[serde(default)]
    kind: Option<String>,
    /// Sample rate for audio input
    #[serde(default)]
    sample_rate: Option<u32>,
    /// Number of channels for audio input
    #[serde(default)]
    channels: Option<u8>,
    /// Data payload (for text/embedding input)
    #[serde(default)]
    data: Option<String>,
}

impl InputConfig {
    /// Get the input type string, preferring the new format over legacy
    fn get_type(&self) -> Option<&str> {
        self.input_type.as_deref().or(self.kind.as_deref())
    }
}

/// Input type for pipeline (public API)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineInputType {
    /// Audio input (raw PCM or WAV)
    Audio,
    /// Text input
    Text,
    /// Embedding input
    Embedding,
    /// Unknown input type
    Unknown,
}

impl PipelineInputType {
    /// Convert from YAML kind string
    fn from_kind(kind: &str) -> Self {
        match kind.to_lowercase().as_str() {
            "audio" | "audioraw" | "audio_raw" => PipelineInputType::Audio,
            "text" => PipelineInputType::Text,
            "embedding" => PipelineInputType::Embedding,
            _ => PipelineInputType::Unknown,
        }
    }

    /// Check if this is audio input
    pub fn is_audio(&self) -> bool {
        matches!(self, PipelineInputType::Audio)
    }

    /// Check if this is text input
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

fn default_battery() -> u8 {
    100
}
fn default_temperature() -> f32 {
    25.0
}

impl Default for MetricsConfig {
    fn default() -> Self {
        MetricsConfig {
            network_rtt: 50,
            battery: 100,
            temperature: 25.0,
        }
    }
}

/// Timing information for a single pipeline stage.
#[derive(Debug, Clone, Serialize)]
pub struct StageTiming {
    /// Stage name
    pub name: String,
    /// Stage latency in milliseconds
    pub latency_ms: u32,
    /// Routing target (local, cloud, or fallback)
    pub target: String,
    /// Routing decision reason
    pub reason: String,
}

/// Result of pipeline execution.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineExecutionResult {
    /// Pipeline name
    pub name: Option<String>,
    /// Stage timing information
    pub stages: Vec<StageTiming>,
    /// Total pipeline latency in milliseconds
    pub total_latency_ms: u32,
    /// Final output type
    pub output_type: OutputType,
    /// Final output envelope
    pub output: Envelope,
}

impl PipelineExecutionResult {
    /// Get the final output as text (if text output).
    pub fn text(&self) -> Option<&str> {
        match &self.output.kind {
            EnvelopeKind::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get the final output as audio bytes (if audio output).
    pub fn audio_bytes(&self) -> Option<&[u8]> {
        match &self.output.kind {
            EnvelopeKind::Audio(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Get the final output as embedding (if embedding output).
    pub fn embedding(&self) -> Option<&[f32]> {
        match &self.output.kind {
            EnvelopeKind::Embedding(e) => Some(e),
            _ => None,
        }
    }
}

/// Pipeline source for loading.
#[derive(Debug, Clone)]
pub enum PipelineSource {
    /// YAML content string
    Yaml(String),
    /// Path to YAML file
    File(PathBuf),
}

/// Loader for creating pipeline instances.
///
/// # Example
///
/// ```rust,no_run
/// use xybrid_sdk::PipelineLoader;
///
/// let loader = PipelineLoader::from_yaml(r#"
/// name: "ASR Pipeline"
/// stages:
///   - whisper-tiny@1.0
/// "#)?;
/// let pipeline = loader.load()?;
/// ```
pub struct PipelineLoader {
    source: PipelineSource,
    config: PipelineConfig,
}

impl PipelineLoader {
    /// Create a loader from YAML content string.
    pub fn from_yaml(yaml_content: &str) -> PipelineResult<Self> {
        let config: PipelineConfig = serde_yaml::from_str(yaml_content)
            .map_err(|e| SdkError::PipelineError(format!("Failed to parse YAML: {}", e)))?;

        Ok(PipelineLoader {
            source: PipelineSource::Yaml(yaml_content.to_string()),
            config,
        })
    }

    /// Create a loader from a YAML file path.
    pub fn from_file(path: impl Into<PathBuf>) -> PipelineResult<Self> {
        let path = path.into();
        let content = std::fs::read_to_string(&path)
            .map_err(|e| SdkError::PipelineError(format!("Failed to read file: {}", e)))?;

        let config: PipelineConfig = serde_yaml::from_str(&content)
            .map_err(|e| SdkError::PipelineError(format!("Failed to parse YAML: {}", e)))?;

        Ok(PipelineLoader {
            source: PipelineSource::File(path),
            config,
        })
    }

    /// Get the pipeline name (if specified in config).
    pub fn name(&self) -> Option<&str> {
        self.config.name.as_deref()
    }

    /// Get the stage names.
    pub fn stage_names(&self) -> Vec<String> {
        self.config
            .stages
            .iter()
            .map(|s| s.get_name())
            .collect()
    }

    /// Load the pipeline, preparing it for execution.
    pub fn load(self) -> PipelineResult<XybridPipeline> {
        XybridPipeline::from_config(self.config, self.source)
    }

    /// Load the pipeline asynchronously.
    pub async fn load_async(self) -> PipelineResult<XybridPipeline> {
        // For now, just delegate to sync version
        // In the future, this could prefetch models
        self.load()
    }
}

/// Internal state for the loaded pipeline.
struct PipelineHandle {
    stage_descriptors: Vec<StageDescriptor>,
    metrics: DeviceMetrics,
    availability_map: HashMap<String, bool>,
    registry_config: Option<RegistryConfig>,
    input_type: PipelineInputType,
    /// Current pipeline status
    status: PipelineStatus,
    /// Cached execution plan (computed by plan())
    execution_plan: Option<ExecutionPlan>,
    /// Original stage configs (for plan() to resolve models)
    stage_configs: Vec<StageConfig>,
}

/// A loaded pipeline ready for execution.
///
/// # Example
///
/// ```rust,no_run
/// use xybrid_sdk::{PipelineLoader, Envelope};
///
/// let pipeline = PipelineLoader::from_yaml(yaml)?.load()?;
/// let result = pipeline.run(&Envelope::audio(audio_bytes))?;
/// println!("Output: {:?}", result.text());
/// ```
pub struct XybridPipeline {
    name: Option<String>,
    handle: Arc<RwLock<PipelineHandle>>,
    source: PipelineSource,
}

impl XybridPipeline {
    /// Parse a target string into ExecutionTarget.
    fn parse_target(target: &str) -> Option<ExecutionTarget> {
        match target.to_lowercase().as_str() {
            "device" | "local" => Some(ExecutionTarget::Device),
            "server" | "cloud" => Some(ExecutionTarget::Server),
            "integration" => Some(ExecutionTarget::Integration),
            "auto" => Some(ExecutionTarget::Auto),
            _ => None,
        }
    }

    /// Parse a provider string into IntegrationProvider.
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

    /// Convert JSON options to StageOptions.
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

    fn from_config(config: PipelineConfig, source: PipelineSource) -> PipelineResult<Self> {
        // Build stage descriptors with full configuration support
        let stage_descriptors: Vec<StageDescriptor> = config
            .stages
            .iter()
            .map(|stage_config| {
                let name = stage_config.get_name();
                let mut desc = StageDescriptor::new(name);

                // Set registry config if specified
                if let Some(reg) = stage_config.get_registry() {
                    let registry_config = match reg {
                        StageRegistryConfig::Simple(url) => Self::url_to_registry_config(&url),
                        StageRegistryConfig::Full(config) => config,
                    };
                    desc.registry = Some(registry_config);
                }

                // Set target if specified
                if let Some(target_str) = stage_config.get_target() {
                    desc.target = Self::parse_target(&target_str);
                }

                // Set provider if specified (also sets target to Integration)
                if let Some(provider_str) = stage_config.get_provider() {
                    desc.provider = Self::parse_provider(&provider_str);
                    // Integration provider implies integration target
                    if desc.target.is_none() {
                        desc.target = Some(ExecutionTarget::Integration);
                    }
                }

                // Set model if specified
                desc.model = stage_config.get_model();

                // Set options if specified
                if let Some(opts) = stage_config.get_options() {
                    desc.options = Some(Self::convert_options(&opts));
                }

                desc
            })
            .collect();

        // Extract pipeline-level registry config (will be applied when running)
        let registry_config = config.registry.as_ref().map(|registry_value| {
            match registry_value {
                RegistryConfigValue::Simple(url) => Self::url_to_registry_config(url),
                RegistryConfigValue::Full(config) => config.clone(),
            }
        });

        // Create device metrics
        let metrics_config = config.metrics.unwrap_or_default();
        let metrics = DeviceMetrics {
            network_rtt: metrics_config.network_rtt,
            battery: metrics_config.battery,
            temperature: metrics_config.temperature,
        };

        // Extract input type from config (supports both new "type" and legacy "kind" fields)
        let input_type = config
            .input
            .as_ref()
            .and_then(|i| i.get_type())
            .map(PipelineInputType::from_kind)
            .unwrap_or(PipelineInputType::Unknown);

        // Store original stage configs for plan() to use
        let stage_configs = config.stages.clone();

        let handle = PipelineHandle {
            stage_descriptors,
            metrics,
            availability_map: config.availability.clone(),
            registry_config,
            input_type,
            status: PipelineStatus::Created,
            execution_plan: None,
            stage_configs,
        };

        Ok(XybridPipeline {
            name: config.name,
            handle: Arc::new(RwLock::new(handle)),
            source,
        })
    }

    fn url_to_registry_config(url: &str) -> RegistryConfig {
        if url.starts_with("file://") {
            RegistryConfig {
                local_path: Some(url.strip_prefix("file://").unwrap().to_string()),
                remote: None,
            }
        } else {
            RegistryConfig {
                local_path: None,
                remote: Some(xybrid_core::registry_config::RemoteRegistryConfig {
                    base_url: url.to_string(),
                    index_path: None,
                    bundle_path: None,
                    auth: xybrid_core::registry_config::RegistryAuth::None,
                    timeout_ms: Some(30000),
                    retry_attempts: Some(3),
                }),
            }
        }
    }

    /// Get the pipeline name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the stage names.
    pub fn stage_names(&self) -> Vec<String> {
        self.handle
            .read()
            .ok()
            .map(|h| h.stage_descriptors.iter().map(|s| s.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Get the expected input type for this pipeline.
    ///
    /// This is determined from the `input.kind` field in the YAML config.
    pub fn input_type(&self) -> PipelineInputType {
        self.handle
            .read()
            .ok()
            .map(|h| h.input_type)
            .unwrap_or(PipelineInputType::Unknown)
    }

    /// Get the number of stages in this pipeline.
    pub fn stage_count(&self) -> usize {
        self.handle
            .read()
            .ok()
            .map(|h| h.stage_descriptors.len())
            .unwrap_or(0)
    }

    /// Run the pipeline with the given input envelope.
    pub fn run(&self, envelope: &Envelope) -> PipelineResult<PipelineExecutionResult> {
        let handle = self.handle.read().map_err(|_| {
            SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
        })?;

        // Clone all needed data to avoid borrow issues
        let stage_descriptors = handle.stage_descriptors.clone();
        let metrics = handle.metrics.clone();
        let availability_map = handle.availability_map.clone();
        let registry_config = handle.registry_config.clone();
        drop(handle); // Release lock before execution

        // Automatically set telemetry pipeline context for this run
        // Generate a unique trace_id for this execution
        let trace_id = uuid::Uuid::new_v4();
        let pipeline_id = self.name.as_ref().map(|n| {
            // Create a deterministic UUID from the pipeline name
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, n.as_bytes())
        });
        crate::telemetry::set_telemetry_pipeline_context(pipeline_id, Some(trace_id));

        // Create a fresh orchestrator for this run
        let mut orchestrator = Orchestrator::new();
        if let Some(ref config) = registry_config {
            orchestrator.executor_mut().set_pipeline_registry(config.clone());
        }

        let availability_fn = move |stage: &str| -> LocalAvailability {
            let exists = availability_map.get(stage).copied().unwrap_or(false);
            LocalAvailability::new(exists)
        };

        let start_time = std::time::Instant::now();
        let results: Vec<StageExecutionResult> = orchestrator
            .execute_pipeline(
                &stage_descriptors,
                envelope,
                &metrics,
                &availability_fn,
            )
            .map_err(|e| {
                // Clear context on error
                crate::telemetry::set_telemetry_pipeline_context(None, None);
                SdkError::PipelineError(format!("Pipeline execution failed: {}", e))
            })?;
        let total_latency_ms = start_time.elapsed().as_millis() as u32;

        // Clear telemetry context after execution
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

        // Emit PipelineComplete telemetry event
        // This will include the span tree captured during execution
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

    /// Run the pipeline asynchronously.
    pub async fn run_async(&self, envelope: &Envelope) -> PipelineResult<PipelineExecutionResult> {
        // Extract all needed data from handle before spawning
        let (stage_descriptors, metrics, availability_map, registry_config) = {
            let handle = self.handle.read().map_err(|_| {
                SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
            })?;
            (
                handle.stage_descriptors.clone(),
                handle.metrics.clone(),
                handle.availability_map.clone(),
                handle.registry_config.clone(),
            )
        };

        let envelope_clone = envelope.clone();
        let name = self.name.clone();

        tokio::task::spawn_blocking(move || {
            // Automatically set telemetry pipeline context for this run
            let trace_id = uuid::Uuid::new_v4();
            let pipeline_id = name.as_ref().map(|n| {
                uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, n.as_bytes())
            });
            crate::telemetry::set_telemetry_pipeline_context(pipeline_id, Some(trace_id));

            // Create a fresh orchestrator inside the blocking task
            let mut orchestrator = Orchestrator::new();
            if let Some(ref config) = registry_config {
                orchestrator.executor_mut().set_pipeline_registry(config.clone());
            }

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

            // Emit PipelineComplete telemetry event
            let event = crate::telemetry::TelemetryEvent {
                event_type: "PipelineComplete".to_string(),
                stage_name: name.clone(),
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

            // Clear telemetry context after execution
            crate::telemetry::set_telemetry_pipeline_context(None, None);

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

    /// Update the device metrics used for routing decisions.
    pub fn set_metrics(&self, metrics: DeviceMetrics) -> PipelineResult<()> {
        let mut handle = self.handle.write().map_err(|_| {
            SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
        })?;
        handle.metrics = metrics;
        Ok(())
    }

    /// Update the availability map for routing decisions.
    pub fn set_availability(&self, availability: HashMap<String, bool>) -> PipelineResult<()> {
        let mut handle = self.handle.write().map_err(|_| {
            SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
        })?;
        handle.availability_map = availability;
        Ok(())
    }

    /// Unload/reset the pipeline.
    ///
    /// This clears the availability map and any cached state.
    /// After calling this, the pipeline will need models to be marked as available
    /// again before execution will succeed.
    pub fn unload(&self) -> PipelineResult<()> {
        let mut handle = self.handle.write().map_err(|_| {
            SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
        })?;
        handle.availability_map.clear();
        Ok(())
    }

    /// Check if the pipeline has any models marked as available.
    pub fn is_loaded(&self) -> bool {
        self.handle
            .read()
            .ok()
            .map(|h| h.availability_map.values().any(|&v| v))
            .unwrap_or(false)
    }

    // ========================================================================
    // New Lifecycle Methods: prepare(), plan(), fetch()
    // ========================================================================

    /// Get the current pipeline status.
    pub fn status(&self) -> PipelineStatus {
        self.handle
            .read()
            .ok()
            .map(|h| h.status.clone())
            .unwrap_or(PipelineStatus::Error("Failed to read status".to_string()))
    }

    /// Phase 1: Prepare - Validate stage configuration.
    ///
    /// This method validates the pipeline configuration:
    /// - Verifies all stage descriptors are valid
    /// - Initializes the availability map
    /// - Sets status to `Prepared`
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if validation passes, or an error with details.
    pub fn prepare(&self) -> PipelineResult<()> {
        let mut handle = self.handle.write().map_err(|_| {
            SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
        })?;

        // Validate stage descriptors
        if handle.stage_descriptors.is_empty() {
            handle.status = PipelineStatus::Error("No stages defined".to_string());
            return Err(SdkError::PipelineError("Pipeline has no stages".to_string()));
        }

        // Validate each stage has required fields
        for (idx, stage) in handle.stage_descriptors.iter().enumerate() {
            if stage.name.is_empty() {
                handle.status = PipelineStatus::Error(format!("Stage {} has no name", idx));
                return Err(SdkError::PipelineError(format!(
                    "Stage {} has no name or model specified",
                    idx
                )));
            }
        }

        handle.status = PipelineStatus::Prepared;
        Ok(())
    }

    /// Phase 2: Plan - Resolve models and compute execution plan.
    ///
    /// This method:
    /// - Resolves each device stage's model via the registry
    /// - Checks cache status for each model
    /// - Identifies integration stages (no download needed)
    /// - Computes total download size
    /// - Sets status to `Planned`
    ///
    /// # Returns
    ///
    /// Returns the `ExecutionPlan` with details about each stage.
    pub fn plan(&self) -> PipelineResult<ExecutionPlan> {
        let mut handle = self.handle.write().map_err(|_| {
            SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
        })?;

        // Ensure we're at least Created (prepare is optional before plan)
        if matches!(handle.status, PipelineStatus::Error(_)) {
            return Err(SdkError::PipelineError(
                "Cannot plan: pipeline is in error state".to_string(),
            ));
        }

        // Get registry URL from config
        let registry_url = handle
            .registry_config
            .as_ref()
            .and_then(|c| c.remote.as_ref())
            .map(|r| r.base_url.clone());

        // Create registry client (use default if no URL specified)
        let client = if let Some(url) = registry_url {
            RegistryClient::new(url)?
        } else {
            RegistryClient::from_env()?
        };

        let mut stage_plans = Vec::new();
        let mut total_download_bytes: u64 = 0;
        let mut requires_network = Vec::new();
        let mut all_device_models_cached = true;
        let mut has_integration_stages = false;

        // Process each stage
        for stage_config in &handle.stage_configs {
            let stage_id = stage_config.get_id().unwrap_or_else(|| stage_config.get_name());
            let target_str = stage_config.get_target();
            let provider = stage_config.get_provider();
            let model_name = stage_config.get_model();

            // Determine stage target
            let stage_target = if provider.is_some() || target_str.as_deref() == Some("integration") {
                has_integration_stages = true;
                requires_network.push(stage_id.clone());
                StageTarget::Integration {
                    provider: provider.unwrap_or_else(|| "unknown".to_string()),
                }
            } else if target_str.as_deref() == Some("cloud") || target_str.as_deref() == Some("server") {
                requires_network.push(stage_id.clone());
                StageTarget::Cloud
            } else {
                StageTarget::Device
            };

            // For device stages, check model resolution and cache
            let (status, download_bytes, format) = if matches!(stage_target, StageTarget::Device) {
                if let Some(ref model_id) = model_name {
                    // Try to resolve and check cache
                    match client.resolve(model_id, None) {
                        Ok(resolved) => {
                            let is_cached = client.is_cached(model_id, None).unwrap_or(false);
                            if is_cached {
                                (
                                    StageReadyStatus::Cached,
                                    None,
                                    Some(format!("{}-{}", resolved.format, resolved.quantization)),
                                )
                            } else {
                                all_device_models_cached = false;
                                total_download_bytes += resolved.size_bytes;
                                (
                                    StageReadyStatus::NeedsDownload,
                                    Some(resolved.size_bytes),
                                    Some(format!("{}-{}", resolved.format, resolved.quantization)),
                                )
                            }
                        }
                        Err(e) => {
                            all_device_models_cached = false;
                            (StageReadyStatus::ResolutionFailed(e.to_string()), None, None)
                        }
                    }
                } else {
                    // No model specified - might be using availability map
                    let name = stage_config.get_name();
                    if handle.availability_map.get(&name).copied().unwrap_or(false) {
                        (StageReadyStatus::Cached, None, None)
                    } else {
                        all_device_models_cached = false;
                        (StageReadyStatus::NeedsDownload, None, None)
                    }
                }
            } else {
                // Integration or cloud stage
                (StageReadyStatus::Integration, None, None)
            };

            stage_plans.push(StagePlan {
                id: stage_id,
                model_id: model_name,
                target: stage_target,
                status,
                download_bytes,
                format,
            });
        }

        // Compute offline capability
        let offline_capable = all_device_models_cached && !has_integration_stages;

        let plan = ExecutionPlan {
            name: self.name.clone(),
            stages: stage_plans,
            total_download_bytes,
            offline_capable,
            requires_network,
        };

        // Cache the plan and update status
        handle.execution_plan = Some(plan.clone());
        handle.status = PipelineStatus::Planned;

        Ok(plan)
    }

    /// Get the cached execution plan (if plan() has been called).
    pub fn execution_plan(&self) -> Option<ExecutionPlan> {
        self.handle
            .read()
            .ok()
            .and_then(|h| h.execution_plan.clone())
    }

    /// Phase 3: Fetch - Download all required models.
    ///
    /// This method downloads all models that need to be fetched (status = NeedsDownload).
    /// After successful completion, status is set to `Ready`.
    ///
    /// # Note
    ///
    /// You should call `plan()` before `fetch()` to compute what needs downloading.
    /// If `plan()` wasn't called, this method will call it automatically.
    pub fn fetch(&self) -> PipelineResult<()> {
        self.fetch_with_progress(|_| {})
    }

    /// Phase 3: Fetch with progress - Download all required models with progress callback.
    ///
    /// # Arguments
    ///
    /// * `progress_callback` - Called with progress updates during download
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// pipeline.fetch_with_progress(|progress| {
    ///     println!("Downloading {}: {}%", progress.model_id, progress.percent);
    /// })?;
    /// ```
    pub fn fetch_with_progress<F>(&self, progress_callback: F) -> PipelineResult<()>
    where
        F: Fn(FetchProgress),
    {
        // Ensure plan exists
        let plan = {
            let handle = self.handle.read().map_err(|_| {
                SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
            })?;

            if handle.execution_plan.is_none() {
                drop(handle);
                // Auto-call plan() if not done
                self.plan()?;
                self.handle
                    .read()
                    .ok()
                    .and_then(|h| h.execution_plan.clone())
            } else {
                handle.execution_plan.clone()
            }
        };

        let plan = plan.ok_or_else(|| {
            SdkError::PipelineError("Failed to compute execution plan".to_string())
        })?;

        // Get registry URL
        let registry_url = {
            let handle = self.handle.read().map_err(|_| {
                SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
            })?;
            handle
                .registry_config
                .as_ref()
                .and_then(|c| c.remote.as_ref())
                .map(|r| r.base_url.clone())
        };

        // Create registry client
        let client = if let Some(url) = registry_url {
            RegistryClient::new(url)?
        } else {
            RegistryClient::from_env()?
        };

        // Find stages that need download
        let stages_to_fetch: Vec<_> = plan
            .stages
            .iter()
            .filter(|s| matches!(s.status, StageReadyStatus::NeedsDownload))
            .filter_map(|s| s.model_id.as_ref().map(|m| (s.id.clone(), m.clone(), s.download_bytes.unwrap_or(0))))
            .collect();

        let total_stages = stages_to_fetch.len();

        // Download each model
        for (stage_idx, (stage_id, model_id, total_bytes)) in stages_to_fetch.into_iter().enumerate() {
            let progress_for_model = |download_progress: f32| {
                let bytes_downloaded = (download_progress * total_bytes as f32) as u64;
                progress_callback(FetchProgress {
                    model_id: model_id.clone(),
                    percent: (download_progress * 100.0) as u32,
                    bytes_downloaded,
                    bytes_total: total_bytes,
                    stage_index: stage_idx,
                    total_stages,
                });
            };

            // Fetch the model
            client.fetch(&model_id, None, progress_for_model)?;

            // Update availability map
            {
                let mut handle = self.handle.write().map_err(|_| {
                    SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
                })?;

                // Mark model as available using various key formats
                handle.availability_map.insert(model_id.clone(), true);
                handle.availability_map.insert(stage_id.clone(), true);

                // Also try model@version format if we have version info
                if let Some(stage_config) = handle.stage_configs.iter().find(|s| s.get_model().as_ref() == Some(&model_id)) {
                    let name = stage_config.get_name();
                    handle.availability_map.insert(name, true);
                }
            }
        }

        // Update status to Ready
        {
            let mut handle = self.handle.write().map_err(|_| {
                SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
            })?;
            handle.status = PipelineStatus::Ready;
        }

        Ok(())
    }

    /// Load the pipeline completely (prepare + plan + fetch in one call).
    ///
    /// This is a convenience method that performs all three phases.
    /// After this returns successfully, the pipeline is ready to run.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let pipeline = PipelineLoader::from_yaml(yaml)?.into_pipeline()?;
    /// pipeline.load_all()?;
    /// // Now ready to run
    /// let result = pipeline.run(&envelope)?;
    /// ```
    pub fn load_all(&self) -> PipelineResult<()> {
        self.prepare()?;
        self.plan()?;
        self.fetch()?;
        Ok(())
    }

    /// Load with progress callback (prepare + plan + fetch).
    pub fn load_all_with_progress<F>(&self, progress_callback: F) -> PipelineResult<()>
    where
        F: Fn(FetchProgress),
    {
        self.prepare()?;
        self.plan()?;
        self.fetch_with_progress(progress_callback)?;
        Ok(())
    }
}

/// Extension to PipelineLoader for creating unprepared pipelines.
impl PipelineLoader {
    /// Create a pipeline without loading models (stays in Created status).
    ///
    /// Use this when you want to control the prepare/plan/fetch lifecycle manually.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let pipeline = PipelineLoader::from_yaml(yaml)?.into_pipeline()?;
    /// assert_eq!(pipeline.status(), PipelineStatus::Created);
    ///
    /// pipeline.prepare()?;
    /// let plan = pipeline.plan()?;
    /// println!("Need to download {} bytes", plan.total_download_bytes);
    /// pipeline.fetch()?;
    /// ```
    pub fn into_pipeline(self) -> PipelineResult<XybridPipeline> {
        XybridPipeline::from_config(self.config, self.source)
    }
}

// ============================================================================
// New Pipeline API (PipelineRef → Pipeline)
// ============================================================================

/// A loaded pipeline ready to preload models and run inference.
///
/// Created via `PipelineRef::load()`. This is the main type for the new simplified API.
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
    inner: XybridPipeline,
    stages: Vec<StageInfo>,
    total_download_bytes: u64,
}

impl Pipeline {
    /// Create a Pipeline from a PipelineRef by resolving models.
    fn from_ref(ref_: &PipelineRef) -> PipelineResult<Self> {
        // Create the inner XybridPipeline
        let inner = XybridPipeline::from_config(
            ref_.config.clone(),
            PipelineSource::Yaml(ref_.yaml_content.clone()),
        )?;

        // Resolve stages and compute download info
        let (stages, total_download_bytes) = Self::resolve_stages(&inner, &ref_.config)?;

        Ok(Self {
            inner,
            stages,
            total_download_bytes,
        })
    }

    /// Resolve stage information from the registry.
    fn resolve_stages(
        inner: &XybridPipeline,
        config: &PipelineConfig,
    ) -> PipelineResult<(Vec<StageInfo>, u64)> {
        // Get registry URL from config
        let registry_url = config
            .registry
            .as_ref()
            .and_then(|r| match r {
                RegistryConfigValue::Simple(url) => Some(url.clone()),
                RegistryConfigValue::Full(cfg) => cfg.remote.as_ref().map(|r| r.base_url.clone()),
            });

        // Create registry client
        let client = if let Some(url) = registry_url {
            RegistryClient::new(url)?
        } else {
            RegistryClient::from_env()?
        };

        let mut stages = Vec::new();
        let mut total_download_bytes: u64 = 0;

        for stage_config in &config.stages {
            let stage_id = stage_config.get_id().unwrap_or_else(|| stage_config.get_name());
            let target_str = stage_config.get_target();
            let provider = stage_config.get_provider();
            let model_name = stage_config.get_model();

            // Determine stage target
            let stage_target = if provider.is_some() || target_str.as_deref() == Some("integration") {
                StageTarget::Integration {
                    provider: provider.unwrap_or_else(|| "unknown".to_string()),
                }
            } else if target_str.as_deref() == Some("cloud") || target_str.as_deref() == Some("server") {
                StageTarget::Cloud
            } else {
                StageTarget::Device
            };

            // For device stages, check model resolution and cache
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
                    // Check availability map for models without explicit model ID
                    let name = stage_config.get_name();
                    let handle = inner.handle.read().map_err(|_| {
                        SdkError::PipelineError("Failed to read pipeline handle".to_string())
                    })?;
                    if handle.availability_map.get(&name).copied().unwrap_or(false) {
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
        self.inner.name()
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
        self.inner.input_type()
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
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// // At app startup
    /// let pipeline = PipelineRef::from_yaml(yaml)?.load()?;
    /// pipeline.load_models()?;  // Download models in background
    ///
    /// // Later, when user triggers action
    /// let result = pipeline.run(&envelope)?;  // Fast - models already loaded
    /// ```
    pub fn load_models(&self) -> PipelineResult<()> {
        self.load_models_with_progress(|_| {})
    }

    /// Preload models with progress callback.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// pipeline.load_models_with_progress(|progress| {
    ///     println!("Downloading {}: {}%", progress.model_id, progress.percent);
    /// })?;
    /// ```
    pub fn load_models_with_progress<F>(&self, progress_callback: F) -> PipelineResult<()>
    where
        F: Fn(DownloadProgress),
    {
        // Get registry URL
        let registry_url = self.inner.handle.read()
            .map_err(|_| SdkError::PipelineError("Failed to read handle".to_string()))?
            .registry_config
            .as_ref()
            .and_then(|c| c.remote.as_ref())
            .map(|r| r.base_url.clone());

        let client = if let Some(url) = registry_url {
            RegistryClient::new(url)?
        } else {
            RegistryClient::from_env()?
        };

        // Find stages needing download
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

            // Fetch the model
            client.fetch(&model_id, None, progress_for_model)?;

            // Update availability map in inner pipeline
            {
                let mut handle = self.inner.handle.write().map_err(|_| {
                    SdkError::PipelineError("Failed to acquire pipeline lock".to_string())
                })?;

                handle.availability_map.insert(model_id.clone(), true);
                handle.availability_map.insert(stage_id.clone(), true);
            }
        }

        Ok(())
    }

    /// Run inference on the pipeline.
    ///
    /// If models aren't loaded yet, this will automatically download them first.
    pub fn run(&self, envelope: &Envelope) -> PipelineResult<PipelineExecutionResult> {
        // Auto-load models if not ready
        if !self.is_ready() {
            self.load_models()?;
        }

        self.inner.run(envelope)
    }

    /// Run inference asynchronously.
    pub async fn run_async(&self, envelope: &Envelope) -> PipelineResult<PipelineExecutionResult> {
        // Auto-load models if not ready
        if !self.is_ready() {
            self.load_models()?;
        }

        self.inner.run_async(envelope).await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_loader_from_yaml() {
        let yaml = r#"
name: "Test Pipeline"
stages:
  - test-stage@1.0
"#;
        let loader = PipelineLoader::from_yaml(yaml).unwrap();
        assert_eq!(loader.name(), Some("Test Pipeline"));
        assert_eq!(loader.stage_names(), vec!["test-stage@1.0"]);
    }

    #[test]
    fn test_pipeline_loader_complex_stages() {
        let yaml = r#"
name: "Complex Pipeline"
registry: "http://localhost:8080"
stages:
  - whisper-tiny@1.0
  - name: "llm-stage@1.0"
    registry: "http://other-registry:8080"
metrics:
  network_rtt: 100
  battery: 80
  temperature: 30.0
availability:
  "whisper-tiny@1.0": true
  "llm-stage@1.0": false
"#;
        let loader = PipelineLoader::from_yaml(yaml).unwrap();
        assert_eq!(loader.name(), Some("Complex Pipeline"));
        assert_eq!(
            loader.stage_names(),
            vec!["whisper-tiny@1.0", "llm-stage@1.0"]
        );
    }

    #[test]
    fn test_pipeline_loader_new_format_with_target_and_provider() {
        // This is the new format used by voice-assistant.yaml
        let yaml = r#"
name: "Voice Assistant (wav2vec2 -> openai -> kokoro)"
registry: "http://localhost:8080"

stages:
  - id: asr
    model: wav2vec2-base-960h
    version: "1.0"
    target: device

  - id: llm
    model: gpt-4o-mini
    target: integration
    provider: openai
    options:
      temperature: 0.7
      max_tokens: 500
      system_prompt: "You are a helpful voice assistant."

  - id: tts
    model: kokoro-82m
    version: "0.1"
    target: device

input:
  type: audio
  sample_rate: 16000
  channels: 1

metrics:
  network_rtt: 100
  battery: 80
  temperature: 25.0

availability:
  "wav2vec2-base-960h@1.0": true
  "kokoro-82m@0.1": true
"#;
        let loader = PipelineLoader::from_yaml(yaml).unwrap();
        assert_eq!(
            loader.name(),
            Some("Voice Assistant (wav2vec2 -> openai -> kokoro)")
        );
        assert_eq!(
            loader.stage_names(),
            vec!["wav2vec2-base-960h@1.0", "gpt-4o-mini", "kokoro-82m@0.1"]
        );

        // Load and verify descriptors
        let pipeline = loader.load().unwrap();
        assert_eq!(pipeline.stage_count(), 3);
        assert_eq!(pipeline.input_type(), PipelineInputType::Audio);
    }

    #[test]
    fn test_pipeline_loader_new_format_anthropic() {
        // Test with Anthropic provider
        let yaml = r#"
name: "Voice Assistant B"
registry: "http://localhost:8080"

stages:
  - id: asr
    model: wav2vec2-base-960h
    version: "1.0"
    target: device

  - id: llm
    model: claude-3-5-sonnet-20241022
    target: integration
    provider: anthropic
    options:
      temperature: 0.7
      max_tokens: 500
      system_prompt: "You are a helpful voice assistant."

  - id: tts
    model: kokoro-82m
    version: "0.1"
    target: device

input:
  type: audio
  sample_rate: 16000
  channels: 1
"#;
        let loader = PipelineLoader::from_yaml(yaml).unwrap();
        assert_eq!(loader.name(), Some("Voice Assistant B"));
        assert_eq!(loader.stage_names().len(), 3);

        let pipeline = loader.load().unwrap();
        assert_eq!(pipeline.stage_count(), 3);
    }

    #[test]
    fn test_pipeline_loader_minimal_asr_only() {
        // Minimal pipeline with just ASR
        let yaml = r#"
name: "Speech-to-Text Demo"
registry: "http://localhost:8080"

stages:
  - id: asr
    model: wav2vec2-base-960h
    version: "1.0"
    target: device

input:
  type: audio
  sample_rate: 16000
  channels: 1
"#;
        let loader = PipelineLoader::from_yaml(yaml).unwrap();
        assert_eq!(loader.name(), Some("Speech-to-Text Demo"));
        assert_eq!(loader.stage_names(), vec!["wav2vec2-base-960h@1.0"]);

        let pipeline = loader.load().unwrap();
        assert_eq!(pipeline.stage_count(), 1);
        assert_eq!(pipeline.input_type(), PipelineInputType::Audio);
    }

    #[test]
    fn test_input_type_from_kind() {
        assert_eq!(PipelineInputType::from_kind("audio"), PipelineInputType::Audio);
        assert_eq!(PipelineInputType::from_kind("text"), PipelineInputType::Text);
        assert_eq!(PipelineInputType::from_kind("embedding"), PipelineInputType::Embedding);
        assert_eq!(PipelineInputType::from_kind("AudioRaw"), PipelineInputType::Audio);
        assert_eq!(PipelineInputType::from_kind("unknown"), PipelineInputType::Unknown);
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

    // ========================================================================
    // New Lifecycle Tests
    // ========================================================================

    #[test]
    fn test_pipeline_status_default() {
        assert_eq!(PipelineStatus::default(), PipelineStatus::Created);
    }

    #[test]
    fn test_pipeline_status_display() {
        assert_eq!(PipelineStatus::Created.to_string(), "created");
        assert_eq!(PipelineStatus::Prepared.to_string(), "prepared");
        assert_eq!(PipelineStatus::Planned.to_string(), "planned");
        assert_eq!(PipelineStatus::Ready.to_string(), "ready");
        assert_eq!(PipelineStatus::Running.to_string(), "running");
        assert_eq!(PipelineStatus::Completed.to_string(), "completed");
        assert_eq!(PipelineStatus::Error("test".to_string()).to_string(), "error: test");
    }

    #[test]
    fn test_stage_target_display() {
        assert_eq!(StageTarget::Device.to_string(), "device");
        assert_eq!(StageTarget::Cloud.to_string(), "cloud");
        assert_eq!(StageTarget::Integration { provider: "openai".to_string() }.to_string(), "integration:openai");
    }

    #[test]
    fn test_stage_ready_status_display() {
        assert_eq!(StageReadyStatus::Cached.to_string(), "cached");
        assert_eq!(StageReadyStatus::NeedsDownload.to_string(), "needs_download");
        assert_eq!(StageReadyStatus::Integration.to_string(), "integration");
        assert_eq!(StageReadyStatus::ResolutionFailed("not found".to_string()).to_string(), "error: not found");
    }

    #[test]
    fn test_into_pipeline_status() {
        let yaml = r#"
name: "Test Pipeline"
stages:
  - test-stage@1.0
"#;
        let pipeline = PipelineLoader::from_yaml(yaml).unwrap().into_pipeline().unwrap();
        assert_eq!(pipeline.status(), PipelineStatus::Created);
    }

    #[test]
    fn test_prepare_sets_status() {
        let yaml = r#"
name: "Test Pipeline"
stages:
  - test-stage@1.0
"#;
        let pipeline = PipelineLoader::from_yaml(yaml).unwrap().into_pipeline().unwrap();
        assert_eq!(pipeline.status(), PipelineStatus::Created);

        pipeline.prepare().unwrap();
        assert_eq!(pipeline.status(), PipelineStatus::Prepared);
    }

    #[test]
    fn test_prepare_fails_with_no_stages() {
        let yaml = r#"
name: "Empty Pipeline"
stages: []
"#;
        let pipeline = PipelineLoader::from_yaml(yaml).unwrap().into_pipeline().unwrap();
        let result = pipeline.prepare();
        assert!(result.is_err());
        assert!(matches!(pipeline.status(), PipelineStatus::Error(_)));
    }

    #[test]
    fn test_execution_plan_serialization() {
        let plan = ExecutionPlan {
            name: Some("Test".to_string()),
            stages: vec![
                StagePlan {
                    id: "asr".to_string(),
                    model_id: Some("wav2vec2".to_string()),
                    target: StageTarget::Device,
                    status: StageReadyStatus::Cached,
                    download_bytes: None,
                    format: Some("onnx-fp16".to_string()),
                },
                StagePlan {
                    id: "llm".to_string(),
                    model_id: Some("gpt-4o-mini".to_string()),
                    target: StageTarget::Integration { provider: "openai".to_string() },
                    status: StageReadyStatus::Integration,
                    download_bytes: None,
                    format: None,
                },
            ],
            total_download_bytes: 0,
            offline_capable: false,
            requires_network: vec!["llm".to_string()],
        };

        // Should serialize without errors
        let json = serde_json::to_string(&plan).unwrap();
        assert!(json.contains("\"name\":\"Test\""));
        assert!(json.contains("\"offline_capable\":false"));
    }

    #[test]
    fn test_fetch_progress_serialization() {
        let progress = FetchProgress {
            model_id: "wav2vec2".to_string(),
            percent: 50,
            bytes_downloaded: 50_000_000,
            bytes_total: 100_000_000,
            stage_index: 0,
            total_stages: 2,
        };

        let json = serde_json::to_string(&progress).unwrap();
        assert!(json.contains("\"percent\":50"));
    }

    // ========================================================================
    // New API Tests (PipelineRef → Pipeline)
    // ========================================================================

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
}
