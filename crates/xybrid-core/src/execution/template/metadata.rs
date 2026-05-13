//! Model metadata and execution template definitions.
//!
//! This module contains the core types that define how models are executed.

use super::steps::{PostprocessingStep, PreprocessingStep};
use super::voice::{VoiceConfig, VoiceInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "schema")]
use schemars::JsonSchema;

// ============================================================================
// Execution Templates
// ============================================================================

/// Main execution template enum - defines how a model should be executed.
///
/// Variants are named by **format**, not by runtime implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
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

        /// Per-model generation sampling parameters. When absent or when a
        /// field is absent, the consuming strategy supplies its own defaults.
        /// Used by codec TTS models (e.g. NeuTTS) that need specific sampling
        /// config for speech-token generation.
        #[serde(default)]
        generation_params: Option<GenerationParams>,
    },
}

/// Sampling parameters for GGUF generation. All fields optional so metadata
/// only needs to specify overrides; absent fields use strategy defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
pub struct GenerationParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
}

// ============================================================================
// Pipeline Stages
// ============================================================================

/// A single stage in a pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(tag = "type")]
#[derive(Default)]
pub enum ExecutionMode {
    /// Run the model once (default)
    #[default]
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

/// Schedule for iterative refinement (diffusion models)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(tag = "type")]
#[derive(Default)]
pub enum RefinementSchedule {
    #[default]
    Linear,
    Cosine,
    Custom {
        timesteps: Vec<f32>,
    },
}

// ============================================================================
// Model Metadata
// ============================================================================

/// Complete model metadata describing execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
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

    /// Optional: Maximum text characters per TTS chunk.
    ///
    /// Overrides the default chunking limit (350 chars) for models that need
    /// shorter sequences. Smaller models (e.g., KittenTTS nano with 15M params)
    /// produce better quality with shorter chunks (150-200 chars).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chunk_chars: Option<usize>,

    /// Optional: Number of trailing audio samples to trim per TTS chunk.
    ///
    /// Some TTS models produce trailing artifacts at the end of each chunk.
    /// KittenTTS trims 5000 samples (~208ms at 24kHz) per chunk.
    /// Set to 0 or omit to disable trimming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trim_trailing_samples: Option<usize>,
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
            max_chunk_chars: None,
            trim_trailing_samples: None,
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
            max_chunk_chars: None,
            trim_trailing_samples: None,
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
            max_chunk_chars: None,
            trim_trailing_samples: None,
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
// Swim-lane grouping helpers
// ============================================================================

/// Map a model's declared task (`model_metadata.json::metadata.task`) to the
/// swim-lane category the console uses to group spans into horizontal lanes.
///
/// Returns `None` when the task is either missing or doesn't fit one of the
/// recognised lanes — the span still renders, it just lands in the catch-all
/// lane at the bottom of the view.
pub fn stage_kind_from_task(task: &str) -> Option<&'static str> {
    match task {
        "speech-recognition" | "speech-to-text" | "asr" => Some("asr"),
        "text-to-speech" | "tts" => Some("tts"),
        "text-generation" | "chat" | "llm" => Some("llm"),
        "translation" => Some("translate"),
        "image-classification" | "image-to-text" | "vision" => Some("vision"),
        "embedding" | "sentence-embedding" => Some("embed"),
        "audio-classification" | "vad" => Some("audio"),
        _ => None,
    }
}

/// Normalise a GGUF `metadata.backend` hint to the canonical wire label.
///
/// Accepts both the canonical name and the historical `mistral` alias that
/// flows through older bundle files; returns `None` for unrecognised hints
/// so the caller can omit the field rather than emit a guessed value.
///
/// Used both by [`backend_label_from_template`] (outer `execute:` span)
/// and by the inner `llm_inference` span sites so the wire payload's
/// `backend` field is the same canonical string regardless of which span
/// the SDK telemetry hoist reads.
pub fn normalize_llm_backend_hint(hint: &str) -> Option<&'static str> {
    match hint {
        "llamacpp" => Some("llamacpp"),
        "mistral" | "mistralrs" => Some("mistralrs"),
        "mlx" => Some("mlx"),
        _ => None,
    }
}

/// Map a model's execution template (and optional `backend` hint from
/// `ModelMetadata.metadata`) to the canonical backend label used by cost
/// telemetry and the analytics ingest path. Values are aligned with the
/// closed set documented for the `backend` field on `PlatformEvent`:
/// `llamacpp` | `mlx` | `mistralrs` | `ort` | `candle` | `cloud`.
///
/// Returns `None` when the runtime is not yet covered by that closed set
/// (e.g. CoreML, TFLite, ModelGraph) — the contract is "additive, omit
/// when unknown" so unrecognised templates simply leave the field absent.
///
/// `cloud` is intentionally not represented here: the cloud adapter
/// emits the label from its own span site (see
/// `runtime_adapter::cloud::CloudRuntimeAdapter::execute`) where the
/// provider identity is also in scope.
pub fn backend_label_from_template(
    template: &ExecutionTemplate,
    hint: Option<&str>,
) -> Option<&'static str> {
    match template {
        ExecutionTemplate::Onnx { .. } => Some("ort"),
        // SafeTensors is Candle's native format; an `mlx` hint on an
        // Apple Silicon bundle overrides the default so the wire label
        // reflects the actual runtime that will execute the model.
        ExecutionTemplate::SafeTensors { .. } => {
            hint.and_then(normalize_llm_backend_hint).or(Some("candle"))
        }
        ExecutionTemplate::Gguf { .. } => hint.and_then(normalize_llm_backend_hint),
        ExecutionTemplate::CoreMl { .. }
        | ExecutionTemplate::TfLite { .. }
        | ExecutionTemplate::ModelGraph { .. } => None,
    }
}

/// Normalise a quantization label to the canonical wire form.
///
/// GGUF filenames typically encode quantization in lowercase
/// (`model-q4_k_m.gguf`); upstream tooling sometimes ships the label
/// in uppercase (`Q4_K_M`) or with the GGUF-internal `f16` / `f32`
/// instead of the platform-canonical `fp16` / `fp32`. This normaliser
/// lowercases and rewrites those two aliases so the analytics column
/// doesn't split a single quantization across multiple row keys.
fn normalize_quantization_label(label: &str) -> String {
    let lower = label.to_lowercase();
    match lower.as_str() {
        "f16" => "fp16".to_string(),
        "f32" => "fp32".to_string(),
        _ => lower,
    }
}

/// Infer a quantization label from a GGUF model filename.
///
/// GGUF tooling encodes quantization in the filename by convention
/// (`qwen2.5-0.5b-instruct-q4_k_m.gguf` → `q4_k_m`). Matched against
/// the documented closed set; longer tokens are checked first so
/// `q4_k_m` doesn't get clipped to `q4_0` by an earlier substring
/// match. Returns `None` when no recognised pattern is found —
/// callers must omit the field rather than emit a guess.
fn infer_quantization_from_gguf_filename(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();
    // Order matters: check longer / more-specific patterns first.
    for q in &[
        "q3_k_l", "q3_k_m", "q3_k_s", "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q2_k", "q4_0",
        "q4_1", "q5_0", "q5_1", "q6_k", "q8_0", "f16", "f32",
    ] {
        if lower.contains(q) {
            return Some(normalize_quantization_label(q));
        }
    }
    None
}

/// Resolve the canonical quantization label for a model.
///
/// Source priority (per INF-91):
///  1. Explicit `metadata.quantization` from `model_metadata.json` —
///     publish-time, stable, preferred.
///  2. GGUF filename inference (fallback) when the template is GGUF
///     and the bundle didn't pin the label.
///  3. `None` — when no signal is available the field stays absent
///     rather than emitting an empty string or a guessed value.
pub fn quantization_label_from_metadata(metadata: &ModelMetadata) -> Option<String> {
    if let Some(declared) = metadata
        .metadata
        .get("quantization")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        return Some(normalize_quantization_label(declared));
    }
    if let ExecutionTemplate::Gguf { model_file, .. } = &metadata.execution_template {
        return infer_quantization_from_gguf_filename(model_file);
    }
    None
}

/// Map a model's execution template to the `span_kind` colour hint used by
/// the swim-lane bar renderer (`gpu` / `cpu` / `io` / `tool`).
///
/// This is the *outer* `execute:<model_id>` span annotation — the LLM adapter
/// overrides this on its inner `llm_inference` span with more precise Metal-
/// vs-CPU information once it knows which kernel path ran.
pub fn span_kind_from_template(template: &ExecutionTemplate) -> &'static str {
    match template {
        ExecutionTemplate::CoreMl { .. } => "gpu",
        ExecutionTemplate::SafeTensors { .. } => {
            #[cfg(feature = "candle-metal")]
            {
                "gpu"
            }
            #[cfg(not(feature = "candle-metal"))]
            {
                "cpu"
            }
        }
        ExecutionTemplate::Gguf { .. } => {
            // llm.rs overrides this on the inner llm_inference span; here we
            // set the outer execute span to the same best-guess value so the
            // swim-lane bar colour is consistent whether we read it off the
            // outer or inner span.
            #[cfg(all(
                any(feature = "llm-mistral-metal", feature = "llm-llamacpp"),
                target_os = "macos"
            ))]
            {
                "gpu"
            }
            #[cfg(not(all(
                any(feature = "llm-mistral-metal", feature = "llm-llamacpp"),
                target_os = "macos"
            )))]
            {
                "cpu"
            }
        }
        ExecutionTemplate::Onnx { .. }
        | ExecutionTemplate::TfLite { .. }
        | ExecutionTemplate::ModelGraph { .. } => "cpu",
    }
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
    fn backend_label_covers_canonical_runtimes() {
        // ONNX → "ort" (analytics-canonical name for the ONNX Runtime path).
        let onnx = ExecutionTemplate::Onnx {
            model_file: "m.onnx".into(),
        };
        assert_eq!(backend_label_from_template(&onnx, None), Some("ort"));

        // SafeTensors defaults to Candle when no hint is set; an
        // `mlx` hint overrides for Apple-Silicon-targeted bundles
        // where the runtime selector picks MLX over Candle.
        let safe = ExecutionTemplate::SafeTensors {
            model_file: "m.safetensors".into(),
            architecture: None,
            config_file: None,
            tokenizer_file: None,
        };
        assert_eq!(backend_label_from_template(&safe, None), Some("candle"));
        assert_eq!(
            backend_label_from_template(&safe, Some("mlx")),
            Some("mlx"),
            "mlx hint must override the candle default for SafeTensors bundles"
        );

        // GGUF: hint required to disambiguate llama.cpp vs mistral.rs;
        // omit when the bundle didn't pin a backend so downstream can
        // tell "we don't know" from "we know it's X".
        let gguf = ExecutionTemplate::Gguf {
            model_file: "m.gguf".into(),
            chat_template: None,
            context_length: 2048,
            generation_params: None,
        };
        assert_eq!(backend_label_from_template(&gguf, None), None);
        assert_eq!(
            backend_label_from_template(&gguf, Some("llamacpp")),
            Some("llamacpp")
        );
        // Accept both the bundle-file alias and the canonical name.
        assert_eq!(
            backend_label_from_template(&gguf, Some("mistral")),
            Some("mistralrs")
        );
        assert_eq!(
            backend_label_from_template(&gguf, Some("mistralrs")),
            Some("mistralrs")
        );
        // GGUF + mlx hint: the MLX runtime can also consume converted
        // GGUFs, so the hint path must accept `"mlx"` here too.
        assert_eq!(backend_label_from_template(&gguf, Some("mlx")), Some("mlx"));
    }

    #[test]
    fn normalize_llm_backend_hint_canonicalises_aliases() {
        // The legacy `mistral` alias must normalise to the canonical
        // `mistralrs` so the inner `llm_inference` span (read by the SDK
        // hoist for `PlatformEvent.backend`) and the outer `execute:`
        // span agree on the closed-set wire value.
        assert_eq!(normalize_llm_backend_hint("mistral"), Some("mistralrs"));
        assert_eq!(normalize_llm_backend_hint("mistralrs"), Some("mistralrs"));
        assert_eq!(normalize_llm_backend_hint("llamacpp"), Some("llamacpp"));
        // MLX is the Apple-Silicon-only LLM runtime; the wire label is
        // already canonical so the mapping is identity.
        assert_eq!(normalize_llm_backend_hint("mlx"), Some("mlx"));
        // Unknown hints must return None so callers omit the field
        // rather than emit a guessed value.
        assert_eq!(normalize_llm_backend_hint("unknown"), None);
        assert_eq!(normalize_llm_backend_hint(""), None);
    }

    #[test]
    fn backend_label_omits_unknown_runtimes() {
        // Templates not yet covered by the closed-set contract must
        // return None so the wire field stays absent rather than
        // emitting a guessed value.
        let coreml = ExecutionTemplate::CoreMl {
            model_file: "m.mlmodel".into(),
        };
        assert!(backend_label_from_template(&coreml, None).is_none());

        let tflite = ExecutionTemplate::TfLite {
            model_file: "m.tflite".into(),
        };
        assert!(backend_label_from_template(&tflite, None).is_none());

        let graph = ExecutionTemplate::ModelGraph {
            stages: Vec::new(),
            config: HashMap::new(),
        };
        assert!(backend_label_from_template(&graph, None).is_none());
    }

    #[test]
    fn quantization_label_prefers_explicit_metadata_field() {
        // Source-priority rule (INF-91): explicit `metadata.quantization`
        // beats filename inference. Lets the bundle author override the
        // filename-derived guess for cases where the filename is wrong
        // or non-canonical (e.g. a re-uploaded model).
        let mut metadata = ModelMetadata {
            model_id: "qwen2.5-0.5b-instruct".into(),
            version: "1".into(),
            execution_template: ExecutionTemplate::Gguf {
                model_file: "qwen2.5-0.5b-instruct-q4_k_m.gguf".into(),
                chat_template: None,
                context_length: 2048,
                generation_params: None,
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files: Vec::new(),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };
        // Explicit declaration wins over the filename's `q4_k_m`.
        metadata
            .metadata
            .insert("quantization".into(), serde_json::json!("Q8_0"));
        assert_eq!(
            quantization_label_from_metadata(&metadata).as_deref(),
            Some("q8_0"),
            "explicit metadata.quantization must override filename inference and be lowercased"
        );
    }

    #[test]
    fn quantization_label_falls_back_to_gguf_filename() {
        // No explicit metadata field → filename inference for GGUF.
        // Must produce the lowercase canonical label, and `f16` /
        // `f32` must rewrite to `fp16` / `fp32` for analytics-column
        // stability.
        let metadata = ModelMetadata {
            model_id: "qwen2.5-0.5b".into(),
            version: "1".into(),
            execution_template: ExecutionTemplate::Gguf {
                model_file: "qwen2.5-0.5b-instruct-q4_k_m.gguf".into(),
                chat_template: None,
                context_length: 2048,
                generation_params: None,
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files: Vec::new(),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };
        assert_eq!(
            quantization_label_from_metadata(&metadata).as_deref(),
            Some("q4_k_m")
        );

        let fp16 = ModelMetadata {
            execution_template: ExecutionTemplate::Gguf {
                model_file: "tinyllama-1.1b-chat-f16.gguf".into(),
                chat_template: None,
                context_length: 2048,
                generation_params: None,
            },
            ..metadata.clone()
        };
        assert_eq!(
            quantization_label_from_metadata(&fp16).as_deref(),
            Some("fp16"),
            "GGUF `f16` must rewrite to canonical `fp16`"
        );
    }

    #[test]
    fn quantization_label_omits_when_unknown() {
        // Contract: the field stays absent (not empty string, not
        // "unknown") when no signal is available — analytics rows
        // with missing quantization must be distinguishable from
        // legitimate `fp32` rows.
        let onnx_no_meta = ModelMetadata {
            model_id: "wav2vec2".into(),
            version: "1".into(),
            execution_template: ExecutionTemplate::Onnx {
                model_file: "model.onnx".into(),
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files: Vec::new(),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };
        assert!(
            quantization_label_from_metadata(&onnx_no_meta).is_none(),
            "ONNX with no metadata.quantization must omit the label"
        );

        // GGUF whose filename doesn't carry a recognised pattern.
        let gguf_unknown = ModelMetadata {
            execution_template: ExecutionTemplate::Gguf {
                model_file: "custom-experimental.gguf".into(),
                chat_template: None,
                context_length: 2048,
                generation_params: None,
            },
            ..onnx_no_meta
        };
        assert!(
            quantization_label_from_metadata(&gguf_unknown).is_none(),
            "GGUF with no quantization marker in filename must omit"
        );
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
