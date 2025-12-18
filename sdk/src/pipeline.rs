//! Pipeline loading and execution for xybrid-sdk.
//!
//! This module provides:
//! - `PipelineLoader`: Loads pipeline configuration from YAML
//! - `XybridPipeline`: Loaded pipeline ready for execution
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_sdk::{PipelineLoader, Envelope};
//!
//! // Load pipeline from YAML
//! let loader = PipelineLoader::from_yaml(yaml_content)?;
//! let pipeline = loader.load()?;
//!
//! // Run inference
//! let result = pipeline.run(&Envelope::audio(audio_bytes))?;
//! println!("Pipeline completed in {}ms", result.total_latency_ms);
//! ```

use crate::model::SdkError;
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

        let handle = PipelineHandle {
            stage_descriptors,
            metrics,
            availability_map: config.availability.clone(),
            registry_config,
            input_type,
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
}
