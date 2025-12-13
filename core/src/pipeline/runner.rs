//! Pipeline Runner - Integration layer between Pipeline DSL and Orchestrator.
//!
//! The PipelineRunner bridges the high-level Pipeline DSL configuration with
//! the low-level Orchestrator execution engine. It handles:
//!
//! - Converting `StageConfig` to `StageDescriptor`
//! - Resolving `ExecutionTarget` to `RouteTarget`
//! - Evaluating `when` clause conditions
//! - Tracking stage outputs across pipeline execution
//! - Returning typed `PipelineResult` values

use super::{
    ConditionEvaluator, ConditionResult, IntegrationProvider, PipelineConfig, ResolutionContext,
    StageConfig, StageOutputContext, TargetResolver,
};
use crate::context::{DeviceMetrics, StageDescriptor};
use crate::device::capabilities::HardwareCapabilities;
use crate::ir::{Envelope, EnvelopeKind};
use crate::orchestrator::{Orchestrator, OrchestratorError};
use crate::routing_engine::LocalAvailability;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;

/// Error types for pipeline runner operations.
#[derive(Debug, Clone)]
pub enum PipelineRunnerError {
    /// Pipeline configuration validation failed.
    ValidationFailed(String),
    /// Stage condition evaluation failed.
    ConditionFailed(String),
    /// Target resolution failed.
    ResolutionFailed(String),
    /// Stage execution failed.
    ExecutionFailed(String),
    /// Input conversion failed.
    InputConversionFailed(String),
    /// Output conversion failed.
    OutputConversionFailed(String),
    /// Orchestrator error wrapper.
    OrchestratorError(String),
}

impl std::fmt::Display for PipelineRunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineRunnerError::ValidationFailed(msg) => {
                write!(f, "Pipeline validation failed: {}", msg)
            }
            PipelineRunnerError::ConditionFailed(msg) => {
                write!(f, "Condition evaluation failed: {}", msg)
            }
            PipelineRunnerError::ResolutionFailed(msg) => {
                write!(f, "Target resolution failed: {}", msg)
            }
            PipelineRunnerError::ExecutionFailed(msg) => {
                write!(f, "Stage execution failed: {}", msg)
            }
            PipelineRunnerError::InputConversionFailed(msg) => {
                write!(f, "Input conversion failed: {}", msg)
            }
            PipelineRunnerError::OutputConversionFailed(msg) => {
                write!(f, "Output conversion failed: {}", msg)
            }
            PipelineRunnerError::OrchestratorError(msg) => {
                write!(f, "Orchestrator error: {}", msg)
            }
        }
    }
}

impl std::error::Error for PipelineRunnerError {}

impl From<OrchestratorError> for PipelineRunnerError {
    fn from(err: OrchestratorError) -> Self {
        PipelineRunnerError::OrchestratorError(err.to_string())
    }
}

/// Result of executing a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage ID.
    pub stage_id: String,
    /// Whether the stage was executed or skipped.
    pub executed: bool,
    /// Skip reason if stage was skipped.
    pub skip_reason: Option<String>,
    /// Execution target used (if executed).
    pub target: Option<String>,
    /// Output from the stage (if executed).
    pub output: Option<Value>,
    /// Execution latency in milliseconds.
    pub latency_ms: u32,
}

/// Typed pipeline execution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// Pipeline name (if specified).
    #[serde(default)]
    pub name: Option<String>,

    /// Whether the pipeline completed successfully.
    pub success: bool,

    /// Error message if pipeline failed.
    #[serde(default)]
    pub error: Option<String>,

    /// Results by stage ID.
    #[serde(default)]
    pub stages: HashMap<String, StageResultSummary>,

    /// Final output type.
    pub output_type: OutputResultType,

    /// Final output data.
    pub output: OutputResult,

    /// Total execution time in milliseconds.
    pub total_latency_ms: u32,

    /// Number of stages executed.
    pub stages_executed: usize,

    /// Number of stages skipped (due to conditions).
    pub stages_skipped: usize,
}

/// Summary of a stage execution for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResultSummary {
    /// Whether the stage was executed.
    pub executed: bool,
    /// Skip reason if stage was skipped.
    #[serde(default)]
    pub skip_reason: Option<String>,
    /// Execution target used.
    #[serde(default)]
    pub target: Option<String>,
    /// Latency in milliseconds.
    pub latency_ms: u32,
}

/// Type of output result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OutputResultType {
    Text,
    Audio,
    Embedding,
    Image,
    Json,
    None,
}

/// Typed output result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OutputResult {
    /// Text output (e.g., from ASR, LLM).
    Text(String),
    /// Audio output (base64 encoded for serialization).
    Audio { bytes: Vec<u8>, sample_rate: u32 },
    /// Embedding output.
    Embedding(Vec<f32>),
    /// Image output (base64 encoded).
    Image { bytes: Vec<u8>, format: String },
    /// Structured JSON output.
    Json(Value),
    /// No output.
    None,
}

impl Default for OutputResult {
    fn default() -> Self {
        OutputResult::None
    }
}

/// Configuration for the pipeline runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Device metrics for routing decisions.
    pub metrics: DeviceMetrics,
    /// Hardware capabilities for target resolution.
    pub capabilities: HardwareCapabilities,
    /// Map of local model availability by model ID.
    pub local_models: HashMap<String, bool>,
    /// Map of server model availability by model ID.
    pub server_models: HashMap<String, bool>,
    /// Map of integration provider availability.
    pub integrations: HashMap<IntegrationProvider, bool>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            metrics: DeviceMetrics {
                network_rtt: 100,
                battery: 100,
                temperature: 25.0,
            },
            capabilities: HardwareCapabilities::default(),
            local_models: HashMap::new(),
            server_models: HashMap::new(),
            integrations: HashMap::new(),
        }
    }
}

/// Pipeline Runner - executes Pipeline DSL configurations through the Orchestrator.
pub struct PipelineRunner {
    /// The underlying orchestrator.
    orchestrator: Orchestrator,
    /// Runner configuration.
    config: RunnerConfig,
    /// Stage output context for condition evaluation.
    output_context: StageOutputContext,
}

impl PipelineRunner {
    /// Create a new PipelineRunner with default configuration.
    pub fn new() -> Self {
        Self {
            orchestrator: Orchestrator::new(),
            config: RunnerConfig::default(),
            output_context: StageOutputContext::new(),
        }
    }

    /// Create a new PipelineRunner with custom configuration.
    pub fn with_config(config: RunnerConfig) -> Self {
        Self {
            orchestrator: Orchestrator::new(),
            config,
            output_context: StageOutputContext::new(),
        }
    }

    /// Create a new PipelineRunner with a custom orchestrator.
    pub fn with_orchestrator(orchestrator: Orchestrator, config: RunnerConfig) -> Self {
        Self {
            orchestrator,
            config,
            output_context: StageOutputContext::new(),
        }
    }

    /// Execute a pipeline from a YAML string.
    pub fn run_yaml(&mut self, yaml: &str, input: Envelope) -> Result<PipelineResult, PipelineRunnerError> {
        let pipeline = PipelineConfig::from_yaml(yaml)
            .map_err(PipelineRunnerError::ValidationFailed)?;
        self.run(&pipeline, input)
    }

    /// Execute a pipeline configuration.
    pub fn run(
        &mut self,
        pipeline: &PipelineConfig,
        input: Envelope,
    ) -> Result<PipelineResult, PipelineRunnerError> {
        let start_time = Instant::now();

        // Validate pipeline
        pipeline.validate().map_err(PipelineRunnerError::ValidationFailed)?;

        // Reset output context for new pipeline run
        self.output_context = StageOutputContext::new();

        // Track results
        let mut stage_results: Vec<StageResult> = Vec::new();
        let mut current_input = input;
        let mut stages_executed = 0;
        let mut stages_skipped = 0;

        // Execute each stage
        for stage_config in &pipeline.stages {
            let stage_result = self.execute_stage(stage_config, &current_input)?;

            if stage_result.executed {
                stages_executed += 1;

                // Update input for next stage
                if let Some(ref output) = stage_result.output {
                    current_input = self.value_to_envelope(output);

                    // Store output in context for condition evaluation
                    self.output_context.add_output(&stage_config.id, output.clone());
                }
            } else {
                stages_skipped += 1;
            }

            stage_results.push(stage_result);
        }

        // Build final result
        let total_latency_ms = start_time.elapsed().as_millis() as u32;

        // Get final output from last executed stage
        let (output_type, output) = self.extract_final_output(&current_input);

        // Build stage summaries
        let mut stages = HashMap::new();
        for result in &stage_results {
            stages.insert(
                result.stage_id.clone(),
                StageResultSummary {
                    executed: result.executed,
                    skip_reason: result.skip_reason.clone(),
                    target: result.target.clone(),
                    latency_ms: result.latency_ms,
                },
            );
        }

        Ok(PipelineResult {
            name: pipeline.name.clone(),
            success: true,
            error: None,
            stages,
            output_type,
            output,
            total_latency_ms,
            stages_executed,
            stages_skipped,
        })
    }

    /// Execute a single stage.
    fn execute_stage(
        &mut self,
        stage_config: &StageConfig,
        input: &Envelope,
    ) -> Result<StageResult, PipelineRunnerError> {
        let stage_start = Instant::now();

        // Check condition if present
        if let Some(ref condition) = stage_config.when {
            match ConditionEvaluator::evaluate(condition, &self.output_context) {
                ConditionResult::True => {
                    // Condition passed, continue execution
                }
                ConditionResult::False => {
                    // Condition failed, skip stage
                    return Ok(StageResult {
                        stage_id: stage_config.id.clone(),
                        executed: false,
                        skip_reason: Some(format!("Condition '{}' evaluated to false", condition)),
                        target: None,
                        output: None,
                        latency_ms: 0,
                    });
                }
                ConditionResult::Error(err) => {
                    return Err(PipelineRunnerError::ConditionFailed(format!(
                        "Stage '{}': {}",
                        stage_config.id, err
                    )));
                }
            }
        }

        // Build resolution context
        let resolution_context = self.build_resolution_context(stage_config);

        // Resolve execution target
        let resolved = TargetResolver::resolve(stage_config, &resolution_context)
            .map_err(|e| PipelineRunnerError::ResolutionFailed(format!("{:?}", e)))?;

        // Convert to orchestrator types
        let stage_descriptor = self.stage_config_to_descriptor(stage_config);
        let availability = LocalAvailability::new(
            self.config.local_models.get(&stage_config.model).copied().unwrap_or(true)
        );

        // Execute through orchestrator
        let exec_result = self.orchestrator.execute_stage(
            &stage_descriptor,
            input,
            &self.config.metrics,
            &availability,
        )?;

        // Convert output to Value for context tracking
        let output_value = self.envelope_to_value(&exec_result.output);

        let latency_ms = stage_start.elapsed().as_millis() as u32;

        Ok(StageResult {
            stage_id: stage_config.id.clone(),
            executed: true,
            skip_reason: None,
            target: Some(resolved.target.to_string()),
            output: Some(output_value),
            latency_ms,
        })
    }

    /// Build resolution context for target resolution.
    fn build_resolution_context(&self, stage_config: &StageConfig) -> ResolutionContext {
        ResolutionContext {
            metrics: self.config.metrics.clone(),
            local_available: self
                .config
                .local_models
                .get(&stage_config.model)
                .copied()
                .unwrap_or(false),
            server_available: self
                .config
                .server_models
                .get(&stage_config.model)
                .copied()
                .unwrap_or(false),
            integration_available: self.config.integrations.clone(),
            capabilities: self.config.capabilities.clone(),
        }
    }

    /// Convert StageConfig to StageDescriptor.
    fn stage_config_to_descriptor(&self, stage_config: &StageConfig) -> StageDescriptor {
        StageDescriptor {
            name: stage_config.model_identifier(),
            registry: None, // Registry handling deferred to orchestrator
            target: Some(stage_config.target.clone()),
            provider: stage_config.provider,
            model: Some(stage_config.model.clone()),
            options: Some(stage_config.options.clone()),
        }
    }

    /// Convert Envelope to serde_json::Value for condition evaluation.
    fn envelope_to_value(&self, envelope: &Envelope) -> Value {
        match &envelope.kind {
            EnvelopeKind::Text(text) => {
                serde_json::json!({
                    "type": "text",
                    "output": text
                })
            }
            EnvelopeKind::Audio(bytes) => {
                serde_json::json!({
                    "type": "audio",
                    "bytes_len": bytes.len()
                })
            }
            EnvelopeKind::Embedding(values) => {
                serde_json::json!({
                    "type": "embedding",
                    "dimensions": values.len(),
                    "output": values
                })
            }
        }
    }

    /// Convert serde_json::Value back to Envelope.
    fn value_to_envelope(&self, value: &Value) -> Envelope {
        // Try to extract the output field and type
        if let Some(obj) = value.as_object() {
            if let Some(type_str) = obj.get("type").and_then(|v| v.as_str()) {
                match type_str {
                    "text" => {
                        if let Some(text) = obj.get("output").and_then(|v| v.as_str()) {
                            return Envelope::new(EnvelopeKind::Text(text.to_string()));
                        }
                    }
                    "embedding" => {
                        if let Some(values) = obj.get("output").and_then(|v| v.as_array()) {
                            let floats: Vec<f32> = values
                                .iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect();
                            return Envelope::new(EnvelopeKind::Embedding(floats));
                        }
                    }
                    "audio" => {
                        // Audio data would need to be decoded from the value
                        // For now, return an empty audio envelope
                        return Envelope::new(EnvelopeKind::Audio(Vec::new()));
                    }
                    _ => {}
                }
            }
        }

        // Default to text envelope with JSON string representation
        Envelope::new(EnvelopeKind::Text(value.to_string()))
    }

    /// Extract final output type and result from envelope.
    fn extract_final_output(&self, envelope: &Envelope) -> (OutputResultType, OutputResult) {
        match &envelope.kind {
            EnvelopeKind::Text(text) => (OutputResultType::Text, OutputResult::Text(text.clone())),
            EnvelopeKind::Audio(bytes) => (
                OutputResultType::Audio,
                OutputResult::Audio {
                    bytes: bytes.clone(),
                    sample_rate: 16000, // Default, should come from metadata
                },
            ),
            EnvelopeKind::Embedding(values) => {
                (OutputResultType::Embedding, OutputResult::Embedding(values.clone()))
            }
        }
    }

    /// Get the current stage output context.
    pub fn output_context(&self) -> &StageOutputContext {
        &self.output_context
    }

    /// Get a mutable reference to the underlying orchestrator.
    pub fn orchestrator_mut(&mut self) -> &mut Orchestrator {
        &mut self.orchestrator
    }

    /// Get a reference to the runner configuration.
    pub fn config(&self) -> &RunnerConfig {
        &self.config
    }

    /// Update the runner configuration.
    pub fn set_config(&mut self, config: RunnerConfig) {
        self.config = config;
    }

    /// Update device metrics.
    pub fn set_metrics(&mut self, metrics: DeviceMetrics) {
        self.config.metrics = metrics;
    }

    /// Register a local model as available.
    pub fn register_local_model(&mut self, model_id: &str, available: bool) {
        self.config.local_models.insert(model_id.to_string(), available);
    }

    /// Register a server model as available.
    pub fn register_server_model(&mut self, model_id: &str, available: bool) {
        self.config.server_models.insert(model_id.to_string(), available);
    }

    /// Register an integration provider as available.
    pub fn register_integration(&mut self, provider: IntegrationProvider, available: bool) {
        self.config.integrations.insert(provider, available);
    }
}

impl Default for PipelineRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_envelope(value: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(value.to_string()))
    }

    fn audio_envelope(bytes: &[u8]) -> Envelope {
        Envelope::new(EnvelopeKind::Audio(bytes.to_vec()))
    }

    #[test]
    fn test_pipeline_runner_new() {
        let runner = PipelineRunner::new();
        assert_eq!(runner.config().metrics.battery, 100);
    }

    #[test]
    fn test_pipeline_runner_with_config() {
        let config = RunnerConfig {
            metrics: DeviceMetrics {
                network_rtt: 50,
                battery: 80,
                temperature: 30.0,
            },
            ..Default::default()
        };
        let runner = PipelineRunner::with_config(config);
        assert_eq!(runner.config().metrics.battery, 80);
    }

    #[test]
    fn test_run_simple_pipeline() {
        let yaml = r#"
name: "Test Pipeline"
version: "1.0"

input:
  type: text

stages:
  - id: process
    model: test-model
    target: device
"#;
        let mut runner = PipelineRunner::new();
        runner.register_local_model("test-model", true);

        let input = text_envelope("Hello, world!");
        let result = runner.run_yaml(yaml, input);

        assert!(result.is_ok());
        let pipeline_result = result.unwrap();
        assert!(pipeline_result.success);
        assert_eq!(pipeline_result.stages_executed, 1);
        assert_eq!(pipeline_result.stages_skipped, 0);
    }

    #[test]
    fn test_run_pipeline_with_condition_skip() {
        let yaml = r#"
name: "Conditional Pipeline"
version: "1.0"

input:
  type: text

stages:
  - id: first
    model: model-a
    target: device

  - id: second
    model: model-b
    target: device
    when: "first.output == 'trigger'"
"#;
        let mut runner = PipelineRunner::new();
        runner.register_local_model("model-a", true);
        runner.register_local_model("model-b", true);

        let input = text_envelope("Hello");
        let result = runner.run_yaml(yaml, input);

        assert!(result.is_ok());
        let pipeline_result = result.unwrap();
        assert!(pipeline_result.success);
        // Second stage should be skipped because condition doesn't match
        assert_eq!(pipeline_result.stages_executed, 1);
        assert_eq!(pipeline_result.stages_skipped, 1);
    }

    #[test]
    fn test_stage_result_tracking() {
        let yaml = r#"
name: "Multi-Stage Pipeline"
version: "1.0"

input:
  type: audio
  sample_rate: 16000
  channels: 1
  format: float32

stages:
  - id: asr
    model: wav2vec2
    target: device

  - id: process
    model: processor
    target: device
"#;
        let mut runner = PipelineRunner::new();
        runner.register_local_model("wav2vec2", true);
        runner.register_local_model("processor", true);

        let input = audio_envelope(&[0u8; 32000]);
        let result = runner.run_yaml(yaml, input);

        assert!(result.is_ok());
        let pipeline_result = result.unwrap();
        assert!(pipeline_result.stages.contains_key("asr"));
        assert!(pipeline_result.stages.contains_key("process"));
    }

    #[test]
    fn test_invalid_pipeline_yaml() {
        let yaml = r#"
name: "Invalid"
stages: []
"#;
        let mut runner = PipelineRunner::new();
        let input = text_envelope("test");
        let result = runner.run_yaml(yaml, input);

        assert!(result.is_err());
    }

    #[test]
    fn test_output_result_types() {
        // Test text output
        let envelope = text_envelope("Hello");
        let runner = PipelineRunner::new();
        let (output_type, _) = runner.extract_final_output(&envelope);
        assert_eq!(output_type, OutputResultType::Text);

        // Test embedding output
        let envelope = Envelope::new(EnvelopeKind::Embedding(vec![0.1, 0.2, 0.3]));
        let (output_type, _) = runner.extract_final_output(&envelope);
        assert_eq!(output_type, OutputResultType::Embedding);
    }

    #[test]
    fn test_model_registration() {
        let mut runner = PipelineRunner::new();

        runner.register_local_model("wav2vec2", true);
        runner.register_server_model("whisper-large", true);
        runner.register_integration(IntegrationProvider::OpenAI, true);

        assert!(runner.config().local_models.get("wav2vec2").copied().unwrap_or(false));
        assert!(runner.config().server_models.get("whisper-large").copied().unwrap_or(false));
        assert!(runner
            .config()
            .integrations
            .get(&IntegrationProvider::OpenAI)
            .copied()
            .unwrap_or(false));
    }

    #[test]
    fn test_pipeline_result_serialization() {
        let result = PipelineResult {
            name: Some("Test".to_string()),
            success: true,
            error: None,
            stages: HashMap::new(),
            output_type: OutputResultType::Text,
            output: OutputResult::Text("Hello".to_string()),
            total_latency_ms: 100,
            stages_executed: 1,
            stages_skipped: 0,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"output_type\":\"text\""));
    }
}
