//! Stage configuration for pipeline stages.
//!
//! Each stage in a pipeline has:
//! - A unique ID within the pipeline
//! - A model identifier
//! - An execution target (device, server, integration, auto)
//! - Optional provider config (for integration targets)
//! - Optional fallback chain

use super::provider::IntegrationProvider;
use super::target::ExecutionTarget;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for a single pipeline stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageConfig {
    /// Unique stage identifier within the pipeline.
    pub id: String,

    /// Model identifier (e.g., "wav2vec2-base-960h", "gpt-4o-mini").
    pub model: String,

    /// Model version (for device/server targets).
    #[serde(default)]
    pub version: Option<String>,

    /// Execution target (device, server, integration, auto).
    #[serde(default)]
    pub target: ExecutionTarget,

    /// Integration provider (required for integration target).
    #[serde(default)]
    pub provider: Option<IntegrationProvider>,

    /// Input source (defaults to previous stage output).
    /// Can reference a specific stage: "asr.output"
    #[serde(default)]
    pub input: Option<String>,

    /// Preferred target for auto resolution.
    #[serde(default)]
    pub prefer: Option<ExecutionTarget>,

    /// Fallback chain for when primary target fails.
    #[serde(default)]
    pub fallback: Vec<FallbackConfig>,

    /// Conditional execution expression.
    /// Example: "intent.output.intent == 'weather'"
    #[serde(default)]
    pub when: Option<String>,

    /// Whether this stage supports streaming.
    #[serde(default)]
    pub streaming: bool,

    /// Stage-specific options.
    #[serde(default)]
    pub options: StageOptions,
}

impl StageConfig {
    /// Create a new stage with minimal config.
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            version: None,
            target: ExecutionTarget::Auto,
            provider: None,
            input: None,
            prefer: None,
            fallback: Vec::new(),
            when: None,
            streaming: false,
            options: StageOptions::default(),
        }
    }

    /// Set the execution target.
    pub fn with_target(mut self, target: ExecutionTarget) -> Self {
        self.target = target;
        self
    }

    /// Set the integration provider.
    pub fn with_provider(mut self, provider: IntegrationProvider) -> Self {
        self.provider = Some(provider);
        self.target = ExecutionTarget::Integration;
        self
    }

    /// Set the model version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add a fallback configuration.
    pub fn with_fallback(mut self, fallback: FallbackConfig) -> Self {
        self.fallback.push(fallback);
        self
    }

    /// Set the conditional expression.
    pub fn with_condition(mut self, when: impl Into<String>) -> Self {
        self.when = Some(when.into());
        self
    }

    /// Check if this stage has a valid configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("Stage ID cannot be empty".to_string());
        }
        if self.model.is_empty() {
            return Err(format!("Stage '{}': model cannot be empty", self.id));
        }
        if self.target == ExecutionTarget::Integration && self.provider.is_none() {
            return Err(format!(
                "Stage '{}': integration target requires a provider",
                self.id
            ));
        }
        Ok(())
    }

    /// Get the full model identifier (model@version).
    pub fn model_identifier(&self) -> String {
        match &self.version {
            Some(v) => format!("{}@{}", self.model, v),
            None => self.model.clone(),
        }
    }
}

/// Fallback configuration for a stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FallbackConfig {
    /// Fallback execution target.
    pub target: ExecutionTarget,

    /// Fallback model (optional, uses original if not specified).
    #[serde(default)]
    pub model: Option<String>,

    /// Fallback model version.
    #[serde(default)]
    pub version: Option<String>,

    /// Fallback provider (for integration target).
    #[serde(default)]
    pub provider: Option<IntegrationProvider>,
}

impl FallbackConfig {
    /// Create a new fallback config.
    pub fn new(target: ExecutionTarget) -> Self {
        Self {
            target,
            model: None,
            version: None,
            provider: None,
        }
    }

    /// Set the fallback model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the fallback provider.
    pub fn with_provider(mut self, provider: IntegrationProvider) -> Self {
        self.provider = Some(provider);
        self
    }
}

/// Stage-specific options (provider options, inference options, etc.).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StageOptions {
    /// Provider-specific options (temperature, max_tokens, etc.).
    #[serde(flatten)]
    pub values: HashMap<String, serde_json::Value>,
}

impl StageOptions {
    /// Create new empty options.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Set an option value.
    pub fn set(&mut self, key: impl Into<String>, value: impl Serialize) {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.values.insert(key.into(), json_value);
        }
    }

    /// Get an option value.
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.values
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Check if an option exists.
    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Get temperature option (common for LLMs).
    pub fn temperature(&self) -> Option<f32> {
        self.get("temperature")
    }

    /// Get max_tokens option (common for LLMs).
    pub fn max_tokens(&self) -> Option<u32> {
        self.get("max_tokens")
    }

    /// Get system_prompt option (common for LLMs).
    pub fn system_prompt(&self) -> Option<String> {
        self.get("system_prompt")
    }

    /// Get timeout_ms option.
    pub fn timeout_ms(&self) -> Option<u32> {
        self.get("timeout_ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_config_new() {
        let stage = StageConfig::new("asr", "wav2vec2-base-960h")
            .with_target(ExecutionTarget::Device)
            .with_version("1.0");

        assert_eq!(stage.id, "asr");
        assert_eq!(stage.model, "wav2vec2-base-960h");
        assert_eq!(stage.target, ExecutionTarget::Device);
        assert_eq!(stage.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_stage_config_integration() {
        let stage = StageConfig::new("llm", "gpt-4o-mini")
            .with_provider(IntegrationProvider::OpenAI);

        assert_eq!(stage.target, ExecutionTarget::Integration);
        assert_eq!(stage.provider, Some(IntegrationProvider::OpenAI));
    }

    #[test]
    fn test_stage_config_validate() {
        let stage = StageConfig::new("", "model");
        assert!(stage.validate().is_err());

        let stage = StageConfig::new("id", "");
        assert!(stage.validate().is_err());

        let stage = StageConfig::new("id", "model").with_target(ExecutionTarget::Integration);
        assert!(stage.validate().is_err()); // Missing provider

        let stage = StageConfig::new("id", "model")
            .with_target(ExecutionTarget::Integration)
            .with_provider(IntegrationProvider::OpenAI);
        assert!(stage.validate().is_ok());
    }

    #[test]
    fn test_stage_config_model_identifier() {
        let stage = StageConfig::new("asr", "wav2vec2-base-960h");
        assert_eq!(stage.model_identifier(), "wav2vec2-base-960h");

        let stage = StageConfig::new("asr", "wav2vec2-base-960h").with_version("1.0");
        assert_eq!(stage.model_identifier(), "wav2vec2-base-960h@1.0");
    }

    #[test]
    fn test_stage_config_serde() {
        let yaml = r#"
id: asr
model: wav2vec2-base-960h
version: "1.0"
target: device
"#;
        let stage: StageConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(stage.id, "asr");
        assert_eq!(stage.target, ExecutionTarget::Device);
    }

    #[test]
    fn test_stage_config_with_fallback() {
        let yaml = r#"
id: asr
model: whisper-large-v3
target: auto
fallback:
  - target: server
    model: whisper-large-v3
  - target: device
    model: wav2vec2-base-960h
  - target: integration
    provider: openai
    model: whisper-1
"#;
        let stage: StageConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(stage.fallback.len(), 3);
        assert_eq!(stage.fallback[0].target, ExecutionTarget::Server);
        assert_eq!(stage.fallback[1].target, ExecutionTarget::Device);
        assert_eq!(stage.fallback[2].target, ExecutionTarget::Integration);
    }

    #[test]
    fn test_stage_options() {
        let mut options = StageOptions::new();
        options.set("temperature", 0.7f32);
        options.set("max_tokens", 1000u32);
        options.set("system_prompt", "You are a helpful assistant.");

        assert_eq!(options.temperature(), Some(0.7f32));
        assert_eq!(options.max_tokens(), Some(1000u32));
        assert_eq!(
            options.system_prompt(),
            Some("You are a helpful assistant.".to_string())
        );
    }
}
