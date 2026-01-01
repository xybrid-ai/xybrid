//! Pipeline configuration - the top-level pipeline definition.
//!
//! This is the main struct that represents a complete pipeline configuration
//! parsed from YAML.

use super::input::{InputConfig, OutputConfig};
use super::stage::StageConfig;
use serde::{Deserialize, Serialize};

/// Pipeline metadata (name, version, description).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineMetadata {
    /// Pipeline name.
    #[serde(default)]
    pub name: Option<String>,

    /// Pipeline version.
    #[serde(default)]
    pub version: Option<String>,

    /// Pipeline description.
    #[serde(default)]
    pub description: Option<String>,
}

impl Default for PipelineMetadata {
    fn default() -> Self {
        Self {
            name: None,
            version: None,
            description: None,
        }
    }
}

/// Complete pipeline configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name.
    #[serde(default)]
    pub name: Option<String>,

    /// Pipeline version.
    #[serde(default)]
    pub version: Option<String>,

    /// Pipeline description.
    #[serde(default)]
    pub description: Option<String>,

    /// Registry URL for device bundles.
    #[serde(default)]
    pub registry: Option<String>,

    /// Pipeline input configuration.
    pub input: InputConfig,

    /// Pipeline output configuration.
    #[serde(default)]
    pub output: OutputConfig,

    /// Pipeline stages (executed in order).
    pub stages: Vec<StageConfig>,
}

impl PipelineConfig {
    /// Create a new pipeline config.
    pub fn new(input: InputConfig, stages: Vec<StageConfig>) -> Self {
        Self {
            name: None,
            version: None,
            description: None,
            registry: None,
            input,
            output: OutputConfig::default(),
            stages,
        }
    }

    /// Set the pipeline name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the pipeline version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Set the registry URL.
    pub fn with_registry(mut self, registry: impl Into<String>) -> Self {
        self.registry = Some(registry.into());
        self
    }

    /// Set the output configuration.
    pub fn with_output(mut self, output: OutputConfig) -> Self {
        self.output = output;
        self
    }

    /// Add a stage to the pipeline.
    pub fn add_stage(mut self, stage: StageConfig) -> Self {
        self.stages.push(stage);
        self
    }

    /// Get the pipeline metadata.
    pub fn metadata(&self) -> PipelineMetadata {
        PipelineMetadata {
            name: self.name.clone(),
            version: self.version.clone(),
            description: self.description.clone(),
        }
    }

    /// Get the number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get stage IDs.
    pub fn stage_ids(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.id.clone()).collect()
    }

    /// Get a stage by ID.
    pub fn get_stage(&self, id: &str) -> Option<&StageConfig> {
        self.stages.iter().find(|s| s.id == id)
    }

    /// Validate the pipeline configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.stages.is_empty() {
            return Err("Pipeline must have at least one stage".to_string());
        }

        // Validate each stage
        for stage in &self.stages {
            stage.validate()?;
        }

        // Check for duplicate stage IDs
        let mut ids = std::collections::HashSet::new();
        for stage in &self.stages {
            if !ids.insert(&stage.id) {
                return Err(format!("Duplicate stage ID: '{}'", stage.id));
            }
        }

        Ok(())
    }

    /// Parse a pipeline from YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        serde_yaml::from_str(yaml).map_err(|e| format!("Failed to parse pipeline YAML: {}", e))
    }

    /// Serialize the pipeline to YAML.
    pub fn to_yaml(&self) -> Result<String, String> {
        serde_yaml::to_string(self).map_err(|e| format!("Failed to serialize pipeline: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{AudioInputConfig, ExecutionTarget, IntegrationProvider};

    #[test]
    fn test_pipeline_config_new() {
        let input = InputConfig::audio(AudioInputConfig::asr_default());
        let stages = vec![StageConfig::new("asr", "wav2vec2-base-960h")];

        let pipeline = PipelineConfig::new(input, stages)
            .with_name("Test Pipeline")
            .with_version("1.0");

        assert_eq!(pipeline.name, Some("Test Pipeline".to_string()));
        assert_eq!(pipeline.version, Some("1.0".to_string()));
        assert_eq!(pipeline.stage_count(), 1);
    }

    #[test]
    fn test_pipeline_config_validate() {
        let input = InputConfig::audio(AudioInputConfig::asr_default());

        // Empty stages
        let pipeline = PipelineConfig::new(input.clone(), vec![]);
        assert!(pipeline.validate().is_err());

        // Valid pipeline
        let pipeline = PipelineConfig::new(
            input.clone(),
            vec![StageConfig::new("asr", "wav2vec2-base-960h")],
        );
        assert!(pipeline.validate().is_ok());

        // Duplicate stage IDs
        let pipeline = PipelineConfig::new(
            input.clone(),
            vec![
                StageConfig::new("asr", "wav2vec2-base-960h"),
                StageConfig::new("asr", "whisper-tiny"),
            ],
        );
        assert!(pipeline.validate().is_err());
    }

    #[test]
    fn test_pipeline_config_from_yaml() {
        let yaml = r#"
name: "Voice Assistant Pipeline"
version: "1.0"
description: "ASR → LLM → TTS pipeline"

registry: "https://registry.xybrid.dev"

input:
  type: audio
  sample_rate: 16000
  channels: 1
  format: float32

output:
  type: audio
  sample_rate: 22050
  format: pcm16

stages:
  - id: asr
    model: wav2vec2-base-960h
    version: "1.0"
    target: device

  - id: llm
    model: gpt-4o-mini
    target: cloud
    provider: openai
    options:
      temperature: 0.7
      max_tokens: 150

  - id: tts
    model: piper-en-us
    version: "1.0"
    target: auto
    prefer: device
    fallback:
      - target: cloud
        provider: elevenlabs
"#;

        let pipeline = PipelineConfig::from_yaml(yaml).unwrap();

        assert_eq!(pipeline.name, Some("Voice Assistant Pipeline".to_string()));
        assert_eq!(pipeline.version, Some("1.0".to_string()));
        assert_eq!(
            pipeline.registry,
            Some("https://registry.xybrid.dev".to_string())
        );
        assert_eq!(pipeline.stage_count(), 3);

        // Check ASR stage
        let asr = pipeline.get_stage("asr").unwrap();
        assert_eq!(asr.model, "wav2vec2-base-960h");
        assert_eq!(asr.target, ExecutionTarget::Device);

        // Check LLM stage
        let llm = pipeline.get_stage("llm").unwrap();
        assert_eq!(llm.model, "gpt-4o-mini");
        assert_eq!(llm.target, ExecutionTarget::Cloud);
        assert_eq!(llm.provider, Some(IntegrationProvider::OpenAI));

        // Check TTS stage
        let tts = pipeline.get_stage("tts").unwrap();
        assert_eq!(tts.target, ExecutionTarget::Auto);
        assert_eq!(tts.prefer, Some(ExecutionTarget::Device));
        assert_eq!(tts.fallback.len(), 1);
    }

    #[test]
    fn test_pipeline_config_roundtrip() {
        let input = InputConfig::audio(AudioInputConfig::asr_default());
        let stages = vec![
            StageConfig::new("asr", "wav2vec2-base-960h").with_target(ExecutionTarget::Device),
        ];

        let pipeline = PipelineConfig::new(input, stages)
            .with_name("Test")
            .with_version("1.0");

        let yaml = pipeline.to_yaml().unwrap();
        let parsed = PipelineConfig::from_yaml(&yaml).unwrap();

        assert_eq!(pipeline.name, parsed.name);
        assert_eq!(pipeline.version, parsed.version);
        assert_eq!(pipeline.stage_count(), parsed.stage_count());
    }
}
