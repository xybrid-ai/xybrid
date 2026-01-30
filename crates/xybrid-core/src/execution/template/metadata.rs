//! Model metadata and execution template definitions.
//!
//! This module contains the core types that define how models are executed.

use super::steps::{PostprocessingStep, PreprocessingStep};
use super::voice::{VoiceConfig, VoiceInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Execution Templates
// ============================================================================

/// Main execution template enum - defines how a model should be executed.
///
/// Variants are named by **format**, not by runtime implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExecutionTemplate {
    /// ONNX model execution via ONNX Runtime
    Onnx {
        /// Path to the ONNX model file (relative to bundle root)
        model_file: String,
    },

    /// SafeTensors model execution via Candle runtime (pure Rust)
    SafeTensors {
        /// Path to the SafeTensors model file (relative to bundle root)
        model_file: String,

        /// Model architecture for routing to Rust implementation
        #[serde(default)]
        architecture: Option<String>,

        /// Path to model configuration JSON
        #[serde(default)]
        config_file: Option<String>,

        /// Path to tokenizer JSON
        #[serde(default)]
        tokenizer_file: Option<String>,
    },

    /// CoreML model execution (Apple platforms)
    CoreMl {
        /// Path to the CoreML model file
        model_file: String,
    },

    /// TensorFlow Lite model execution (mobile)
    TfLite {
        /// Path to the TFLite model file
        model_file: String,
    },

    /// Multi-model graph execution (DAG of models)
    ModelGraph {
        /// Sequence of execution stages
        stages: Vec<PipelineStage>,

        /// Model-specific configuration
        #[serde(default)]
        config: HashMap<String, serde_json::Value>,
    },

    /// GGUF model execution for local LLMs
    Gguf {
        /// Path to the GGUF model file
        model_file: String,

        /// Path to chat template JSON file
        #[serde(default)]
        chat_template: Option<String>,

        /// Maximum context length (tokens)
        #[serde(default = "default_context_length")]
        context_length: usize,
    },
}

// ============================================================================
// Pipeline Stages
// ============================================================================

/// A single stage in a pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage name (e.g., "encoder", "decoder", "vocoder")
    pub name: String,

    /// Path to ONNX model file for this stage
    pub model_file: String,

    /// Execution mode for this stage
    #[serde(default)]
    pub execution_mode: ExecutionMode,

    /// Input tensor names expected by this stage
    pub inputs: Vec<String>,

    /// Output tensor names produced by this stage
    pub outputs: Vec<String>,

    /// Optional stage-specific configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Execution Modes
// ============================================================================

/// Execution mode for a pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExecutionMode {
    /// Run the model once (default)
    SingleShot,

    /// Run the model in an autoregressive loop
    Autoregressive {
        max_tokens: usize,
        start_token_id: i64,
        end_token_id: i64,
        #[serde(default)]
        repetition_penalty: f32,
    },

    /// Whisper-specific decoder with KV cache and forced tokens
    WhisperDecoder {
        max_tokens: usize,
        start_token_id: i64,
        end_token_id: i64,
        language_token_id: i64,
        task_token_id: i64,
        no_timestamps_token_id: i64,
        #[serde(default)]
        suppress_tokens: Vec<i64>,
        #[serde(default = "default_repetition_penalty")]
        repetition_penalty: f32,
    },

    /// Run the model iteratively with refinement (diffusion)
    IterativeRefinement {
        num_steps: usize,
        #[serde(default)]
        schedule: RefinementSchedule,
    },
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::SingleShot
    }
}

/// Schedule for iterative refinement (diffusion models)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RefinementSchedule {
    Linear,
    Cosine,
    Custom { timesteps: Vec<f32> },
}

impl Default for RefinementSchedule {
    fn default() -> Self {
        RefinementSchedule::Linear
    }
}

// ============================================================================
// Model Metadata
// ============================================================================

/// Complete model metadata describing execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier
    pub model_id: String,

    /// Model version
    pub version: String,

    /// Execution template defining how to run the model
    pub execution_template: ExecutionTemplate,

    /// Preprocessing steps to apply to input data
    #[serde(default)]
    pub preprocessing: Vec<PreprocessingStep>,

    /// Postprocessing steps to apply to output data
    #[serde(default)]
    pub postprocessing: Vec<PostprocessingStep>,

    /// List of files included in the model bundle
    pub files: Vec<String>,

    /// Optional: Human-readable description
    #[serde(default)]
    pub description: Option<String>,

    /// Optional: Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Optional: Voice configuration for TTS models
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voices: Option<VoiceConfig>,
}

impl ModelMetadata {
    /// Create an ONNX model metadata
    pub fn onnx(
        model_id: impl Into<String>,
        version: impl Into<String>,
        model_file: impl Into<String>,
    ) -> Self {
        let model_file = model_file.into();
        Self {
            model_id: model_id.into(),
            version: version.into(),
            execution_template: ExecutionTemplate::Onnx {
                model_file: model_file.clone(),
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files: vec![model_file],
            description: None,
            metadata: HashMap::new(),
            voices: None,
        }
    }

    /// Create a SafeTensors model metadata (Candle runtime)
    pub fn safetensors(
        model_id: impl Into<String>,
        version: impl Into<String>,
        model_file: impl Into<String>,
        architecture: impl Into<String>,
    ) -> Self {
        let model_file = model_file.into();
        Self {
            model_id: model_id.into(),
            version: version.into(),
            execution_template: ExecutionTemplate::SafeTensors {
                model_file: model_file.clone(),
                architecture: Some(architecture.into()),
                config_file: None,
                tokenizer_file: None,
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files: vec![model_file],
            description: None,
            metadata: HashMap::new(),
            voices: None,
        }
    }

    /// Create a model graph metadata (multi-model DAG)
    pub fn model_graph(
        model_id: impl Into<String>,
        version: impl Into<String>,
        stages: Vec<PipelineStage>,
        files: Vec<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            version: version.into(),
            execution_template: ExecutionTemplate::ModelGraph {
                stages,
                config: HashMap::new(),
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files,
            description: None,
            metadata: HashMap::new(),
            voices: None,
        }
    }

    /// Add preprocessing step
    pub fn with_preprocessing(mut self, step: PreprocessingStep) -> Self {
        self.preprocessing.push(step);
        self
    }

    /// Add postprocessing step
    pub fn with_postprocessing(mut self, step: PostprocessingStep) -> Self {
        self.postprocessing.push(step);
        self
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Get the voice configuration if this is a TTS model with voices.
    pub fn voice_config(&self) -> Option<&VoiceConfig> {
        self.voices.as_ref()
    }

    /// Look up a voice by ID
    pub fn get_voice(&self, voice_id: &str) -> Option<&VoiceInfo> {
        self.voices
            .as_ref()?
            .catalog
            .iter()
            .find(|v| v.id == voice_id)
    }

    /// Get the default voice for this model.
    pub fn default_voice(&self) -> Option<&VoiceInfo> {
        let config = self.voices.as_ref()?;
        self.get_voice(&config.default)
    }

    /// List all available voices.
    pub fn list_voices(&self) -> Vec<&VoiceInfo> {
        self.voices
            .as_ref()
            .map(|c| c.catalog.iter().collect())
            .unwrap_or_default()
    }

    /// Check if this model has voice configuration.
    pub fn has_voices(&self) -> bool {
        self.voices.is_some()
    }
}

// ============================================================================
// Default Functions
// ============================================================================

fn default_repetition_penalty() -> f32 {
    1.1
}

fn default_context_length() -> usize {
    4096
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_serialization() {
        let metadata = ModelMetadata::onnx("mnist", "1.0", "mnist.onnx")
            .with_preprocessing(PreprocessingStep::Normalize {
                mean: vec![0.1307],
                std: vec![0.3081],
            })
            .with_postprocessing(PostprocessingStep::Argmax { dim: None });

        let json = serde_json::to_string_pretty(&metadata).unwrap();
        let parsed: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_id, "mnist");
        assert!(json.contains("\"type\": \"Onnx\""));
    }

    #[test]
    fn test_execution_modes() {
        let autoregressive = ExecutionMode::Autoregressive {
            max_tokens: 100,
            start_token_id: 0,
            end_token_id: 1,
            repetition_penalty: 0.8,
        };

        let json = serde_json::to_string(&autoregressive).unwrap();
        let parsed: ExecutionMode = serde_json::from_str(&json).unwrap();

        match parsed {
            ExecutionMode::Autoregressive { max_tokens, .. } => assert_eq!(max_tokens, 100),
            _ => panic!("Expected autoregressive mode"),
        }
    }

    #[test]
    fn test_model_metadata_with_voices() {
        let json = r#"{
            "model_id": "test-tts",
            "version": "1.0",
            "execution_template": {"type": "Onnx", "model_file": "model.onnx"},
            "voices": {
                "format": "embedded",
                "file": "voices.bin",
                "loader": "binary_f32_256",
                "default": "voice_1",
                "catalog": [{"id": "voice_1", "name": "Voice 1", "index": 0}]
            },
            "files": ["model.onnx"]
        }"#;

        let metadata: ModelMetadata = serde_json::from_str(json).unwrap();
        assert!(metadata.has_voices());
        assert_eq!(metadata.default_voice().unwrap().id, "voice_1");
    }
}
