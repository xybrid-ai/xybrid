//! Execution Template module - Metadata-driven model execution strategies.
//!
//! This module defines the execution templates that describe how to run ML models
//! without hard-coding model-specific logic. It supports a three-stage evolution:
//!
//! 1. **Simple Mode**: One-shot models (classifiers, embedders, basic encoders)
//! 2. **Pipeline Architecture**: Multi-step execution (preprocess → encode → decode → postprocess)
//! 3. **Full DAG** (future): Arbitrary computation graphs for complex models
//!
//! The pipeline architecture is the "90% solution" - easy to write, validate, and execute,
//! while covering real-world models like Whisper, GPT, TTS, and diffusion models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main execution template enum - defines how a model should be executed
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExecutionTemplate {
    /// Simple Mode: Single ONNX model, run once
    ///
    /// Use for: Image classifiers, embedders, simple encoders (BERT, ResNet, CLIP)
    ///
    /// Example:
    /// ```json
    /// {
    ///   "type": "SimpleMode",
    ///   "model_file": "mnist.onnx"
    /// }
    /// ```
    SimpleMode {
        /// Path to the ONNX model file (relative to bundle root)
        model_file: String,
    },

    /// Candle Model: SafeTensors model using Candle runtime (pure Rust)
    ///
    /// Use for: Whisper ASR, LLaMA, and other HuggingFace models with SafeTensors weights
    ///
    /// Example:
    /// ```json
    /// {
    ///   "type": "CandleModel",
    ///   "model_file": "model.safetensors",
    ///   "config_file": "config.json",
    ///   "tokenizer_file": "tokenizer.json",
    ///   "model_type": "whisper"
    /// }
    /// ```
    CandleModel {
        /// Path to the SafeTensors model file (relative to bundle root)
        model_file: String,

        /// Path to model configuration JSON (HuggingFace config.json format)
        #[serde(default)]
        config_file: Option<String>,

        /// Path to tokenizer JSON (HuggingFace tokenizer format)
        #[serde(default)]
        tokenizer_file: Option<String>,

        /// Model type hint for specialized execution (e.g., "whisper", "llama")
        #[serde(default)]
        model_type: Option<String>,
    },

    /// Pipeline Architecture: Structured multi-step execution
    ///
    /// Defines a sequence of stages that process data through multiple models
    /// with control flow (loops, conditionals, etc.)
    ///
    /// Use for: Whisper (encoder→decoder loop), GPT (autoregressive),
    ///          TTS (text→mel→vocoder), Diffusion (iterative denoising)
    ///
    /// Example:
    /// ```json
    /// {
    ///   "type": "Pipeline",
    ///   "stages": [
    ///     {
    ///       "name": "encoder",
    ///       "model_file": "encoder.onnx",
    ///       "inputs": ["mel_spectrogram"],
    ///       "outputs": ["cross_k", "cross_v"]
    ///     },
    ///     {
    ///       "name": "decoder",
    ///       "model_file": "decoder.onnx",
    ///       "execution_mode": {
    ///         "type": "Autoregressive",
    ///         "max_tokens": 448,
    ///         "start_token_id": 50258,
    ///         "end_token_id": 50256
    ///       },
    ///       "inputs": ["tokens", "kv_cache_k", "kv_cache_v", "cross_k", "cross_v", "offset"],
    ///       "outputs": ["logits", "updated_kv_k", "updated_kv_v"]
    ///     }
    ///   ],
    ///   "config": {
    ///     "kv_cache_shape": [4, 1, 448, 384]
    ///   }
    /// }
    /// ```
    Pipeline {
        /// Sequence of execution stages
        stages: Vec<PipelineStage>,

        /// Model-specific configuration (KV cache shapes, hidden dims, etc.)
        #[serde(default)]
        config: HashMap<String, serde_json::Value>,
    },
}

/// A single stage in a pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage name (e.g., "encoder", "decoder", "vocoder")
    pub name: String,

    /// Path to ONNX model file for this stage
    pub model_file: String,

    /// Execution mode for this stage (single-shot, autoregressive, iterative)
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

/// Execution mode for a pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExecutionMode {
    /// Run the model once (default)
    ///
    /// Use for: Encoders, embedders, single-pass transforms
    SingleShot,

    /// Run the model in an autoregressive loop
    ///
    /// Use for: Language models (GPT), TTS models
    ///
    /// The model generates one token at a time, feeding outputs back as inputs
    /// until an end condition is met (max tokens or end token generated)
    Autoregressive {
        /// Maximum number of tokens to generate
        max_tokens: usize,

        /// Token ID that starts generation (e.g., <|startoftext|>)
        start_token_id: i64,

        /// Token ID that ends generation (e.g., <|endoftext|>)
        end_token_id: i64,

        /// Optional: Repetition penalty to prevent loops (0.0 = disabled, 0.5 = moderate)
        #[serde(default)]
        repetition_penalty: f32,
    },

    /// Whisper-specific decoder with KV cache and forced tokens
    ///
    /// Use for: Whisper ASR models (encoder-decoder with cross-attention)
    ///
    /// Handles the HuggingFace ONNX format with separate decoder/encoder KV cache
    WhisperDecoder {
        /// Maximum number of tokens to generate
        max_tokens: usize,

        /// Token ID that starts generation (<|startoftranscript|> = 50258)
        start_token_id: i64,

        /// Token ID that ends generation (<|endoftext|> = 50257)
        end_token_id: i64,

        /// Language token ID (<|en|> = 50259 for English)
        language_token_id: i64,

        /// Task token ID (transcribe = 50359, translate = 50358)
        task_token_id: i64,

        /// No timestamps token ID (50363)
        no_timestamps_token_id: i64,

        /// Tokens to suppress during generation (avoid invalid tokens)
        #[serde(default)]
        suppress_tokens: Vec<i64>,

        /// Repetition penalty to prevent loops (1.0 = disabled, 1.1 = moderate)
        #[serde(default = "default_repetition_penalty")]
        repetition_penalty: f32,
    },

    /// Run the model iteratively with refinement
    ///
    /// Use for: Diffusion models (Stable Diffusion), iterative denoisers
    ///
    /// The model refines its output over N steps, typically with a schedule
    IterativeRefinement {
        /// Number of refinement steps (e.g., 50 for diffusion)
        num_steps: usize,

        /// Optional: Timestep schedule (linear, cosine, etc.)
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
    /// Linear schedule from 1.0 to 0.0
    Linear,

    /// Cosine schedule (DDPM-style)
    Cosine,

    /// Custom timesteps provided explicitly
    Custom {
        timesteps: Vec<f32>,
    },
}

impl Default for RefinementSchedule {
    fn default() -> Self {
        RefinementSchedule::Linear
    }
}

/// Preprocessing step applied before model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PreprocessingStep {
    /// Convert audio to mel spectrogram
    ///
    /// Use for: ASR models (Whisper), audio classifiers
    ///
    /// # Presets
    ///
    /// Instead of specifying individual parameters, you can use a preset:
    /// ```json
    /// { "type": "MelSpectrogram", "preset": "whisper" }
    /// ```
    ///
    /// Available presets: `whisper`, `whisper-large`
    MelSpectrogram {
        /// Optional: Use a preset configuration (overrides other fields)
        /// Available: "whisper", "whisper-large"
        #[serde(default)]
        preset: Option<String>,

        /// Number of mel frequency bins (default: 80)
        #[serde(default = "default_n_mels")]
        n_mels: usize,

        /// Audio sample rate (Hz) (default: 16000)
        #[serde(default = "default_sample_rate")]
        sample_rate: u32,

        /// FFT window size (default: 400)
        #[serde(default = "default_fft_size")]
        fft_size: usize,

        /// Hop length between frames (default: 160)
        #[serde(default = "default_hop_length")]
        hop_length: usize,

        /// Mel frequency scale (default: Slaney)
        #[serde(default)]
        mel_scale: MelScaleType,

        /// Maximum number of output frames (default: 3000 for 30 seconds)
        #[serde(default = "default_max_frames")]
        max_frames: Option<usize>,
    },

    /// Tokenize text using a vocabulary file
    ///
    /// Use for: Language models, text classifiers
    Tokenize {
        /// Path to vocabulary file (relative to bundle root)
        vocab_file: String,

        /// Type of tokenizer (BPE, WordPiece, SentencePiece)
        tokenizer_type: TokenizerType,

        /// Optional: Maximum sequence length
        #[serde(default)]
        max_length: Option<usize>,
    },

    /// Normalize tensor values
    ///
    /// Use for: Image models (ResNet, CLIP), any model with standardized inputs
    Normalize {
        /// Mean values for normalization (per channel)
        mean: Vec<f32>,

        /// Standard deviation values (per channel)
        std: Vec<f32>,
    },

    /// Resize image to target dimensions
    ///
    /// Use for: Image classifiers, vision models
    Resize {
        /// Target width
        width: usize,

        /// Target height
        height: usize,

        /// Interpolation method
        #[serde(default)]
        interpolation: InterpolationMethod,
    },

    /// Center crop image to target dimensions
    ///
    /// Use for: Image classifiers (ResNet, EfficientNet) that need centered crops
    CenterCrop {
        /// Target width
        width: usize,

        /// Target height
        height: usize,
    },

    /// Convert audio bytes to PCM samples
    ///
    /// Use for: Audio models that need raw PCM input
    AudioDecode {
        /// Target sample rate (Hz)
        sample_rate: u32,

        /// Number of channels (1 = mono, 2 = stereo)
        channels: usize,
    },

    /// Reshape tensor to target dimensions
    ///
    /// Use for: Converting flat embeddings to multi-dimensional tensors
    Reshape {
        /// Target shape (e.g., [1, 1, 28, 28] for MNIST)
        shape: Vec<usize>,
    },

    /// Phonemize text to IPA symbols for TTS
    ///
    /// Use for: TTS models (KittenTTS, Piper, Kokoro) that require phoneme input
    Phonemize {
        /// Path to tokens.txt or vocab file (maps IPA symbols to token IDs)
        tokens_file: String,

        /// Phonemization backend to use (default: CmuDictionary)
        #[serde(default)]
        backend: PhonemizerBackend,

        /// Path to dictionary file (cmudict.dict for CMU, not needed for EspeakNG)
        /// If not specified, will try to load from default locations (CMU only)
        #[serde(default)]
        dict_file: Option<String>,

        /// Language code for espeak-ng (e.g., "en-us", "en-gb")
        /// Only used when backend is EspeakNG
        #[serde(default)]
        language: Option<String>,

        /// Whether to add padding tokens at start/end (default: true)
        #[serde(default = "default_add_padding")]
        add_padding: bool,

        /// Whether to normalize text before phonemization (default: false)
        /// Applies text cleanup (quotes, abbreviations, etc.) before phonemization
        #[serde(default)]
        normalize_text: bool,
    },
}

/// Phonemizer backend for text-to-phoneme conversion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PhonemizerBackend {
    /// CMU Pronouncing Dictionary (ARPABET → IPA conversion)
    /// Best for: KittenTTS, models trained on CMU phonemes
    #[default]
    CmuDictionary,

    /// espeak-ng TTS engine (direct IPA output)
    /// Best for: Kokoro, Piper, models expecting espeak IPA
    /// Requires espeak-ng to be installed on the system (`brew install espeak-ng`)
    EspeakNG,

    /// Misaki dictionary-based phonemizer (IPA output, no system dependencies)
    /// Best for: Kokoro on mobile/embedded where espeak-ng isn't available
    /// Uses bundled JSON dictionaries (us_gold.json + us_silver.json)
    /// Falls back to basic letter-by-letter for OOD words
    MisakiDictionary,
}

/// Mel frequency scale type for mel spectrogram preprocessing.
///
/// Different models use different mel scale formulas. Using the wrong scale
/// will produce incorrect mel spectrograms and poor model performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MelScaleType {
    /// Slaney mel scale (used by Whisper, transformers.js, librosa default).
    ///
    /// Formula:
    /// - For freq < 1000 Hz: `mel = 3 × freq / 200`
    /// - For freq >= 1000 Hz: `mel = 15 + 27 × ln(freq / 1000) / ln(6.4)`
    #[default]
    Slaney,

    /// HTK mel scale (used by older implementations, mel-spec library).
    ///
    /// Formula: `mel = 2595 × log10(1 + freq / 700)`
    Htk,
}

fn default_add_padding() -> bool {
    true
}

fn default_repetition_penalty() -> f32 {
    1.1 // Moderate penalty for Whisper
}

fn default_n_mels() -> usize {
    80
}

fn default_sample_rate() -> u32 {
    16000
}

fn default_fft_size() -> usize {
    400
}

fn default_hop_length() -> usize {
    160
}

fn default_max_frames() -> Option<usize> {
    Some(3000) // 30 seconds @ 100fps
}

/// Tokenizer type for text preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenizerType {
    /// Byte-Pair Encoding (GPT-2, GPT-3, Whisper)
    BPE,

    /// WordPiece (BERT, DistilBERT)
    WordPiece,

    /// SentencePiece (T5, ALBERT, XLNet)
    SentencePiece,
}

/// Interpolation method for image resizing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    Nearest,
    Bilinear,
    Bicubic,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        InterpolationMethod::Bilinear
    }
}

/// Postprocessing step applied after model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PostprocessingStep {
    /// Decode BPE tokens to text
    ///
    /// Use for: Language models, ASR models (Whisper)
    BPEDecode {
        /// Path to vocabulary file (same as used in tokenization)
        vocab_file: String,
    },

    /// Apply argmax to get class index or token ID
    ///
    /// Use for: Classifiers, greedy decoding
    Argmax {
        /// Dimension to apply argmax over (default: last dimension)
        #[serde(default)]
        dim: Option<usize>,
    },

    /// Apply softmax to get probabilities
    ///
    /// Use for: Classifiers, probability distributions
    Softmax {
        /// Dimension to apply softmax over (default: last dimension)
        #[serde(default)]
        dim: Option<usize>,
    },

    /// Get top-K predictions with scores
    ///
    /// Use for: Multi-class classification, recommendation systems
    TopK {
        /// Number of top predictions to return
        k: usize,

        /// Dimension to apply top-k over (default: last dimension)
        #[serde(default)]
        dim: Option<usize>,
    },

    /// Apply threshold to convert probabilities to binary predictions
    ///
    /// Use for: Binary classification, object detection confidence filtering
    Threshold {
        /// Threshold value (0.0-1.0)
        threshold: f32,

        /// Whether to return indices (true) or binary mask (false)
        #[serde(default)]
        return_indices: bool,
    },

    /// Apply temperature sampling for token generation
    ///
    /// Use for: Creative text generation
    TemperatureSample {
        /// Temperature value (higher = more random, lower = more deterministic)
        temperature: f32,

        /// Optional: Top-k filtering
        #[serde(default)]
        top_k: Option<usize>,

        /// Optional: Top-p (nucleus) filtering
        #[serde(default)]
        top_p: Option<f32>,
    },

    /// Denormalize tensor values (inverse of Normalize)
    ///
    /// Use for: Image generation, reversing preprocessing
    Denormalize {
        /// Mean values used in normalization
        mean: Vec<f32>,

        /// Std values used in normalization
        std: Vec<f32>,
    },

    /// Mean pooling over token embeddings to get sentence embedding
    ///
    /// Use for: Sentence transformers (BERT, DistilBERT, all-MiniLM)
    MeanPool {
        /// Dimension to pool over (default: 1, the sequence dimension)
        #[serde(default = "default_pool_dim")]
        dim: usize,
    },

    /// CTC (Connectionist Temporal Classification) decoding for ASR
    ///
    /// Use for: Wav2Vec2, Hubert, other CTC-based ASR models
    CTCDecode {
        /// Path to vocabulary file (character or word-level)
        vocab_file: String,

        /// Blank token index (usually 0)
        #[serde(default)]
        blank_index: usize,
    },

    /// Convert TTS waveform output to audio bytes
    ///
    /// Use for: TTS models (KittenTTS, Piper) that output float32 waveforms
    TTSAudioEncode {
        /// Output sample rate in Hz
        sample_rate: u32,

        /// Whether to apply postprocessing (normalization, silence trimming)
        #[serde(default = "default_tts_postprocess")]
        apply_postprocessing: bool,
    },

    /// Decode Whisper token IDs to text using HuggingFace tokenizer
    ///
    /// Use for: Whisper ASR models with HuggingFace tokenizer format
    WhisperDecode {
        /// Path to tokenizer.json file (HuggingFace format)
        tokenizer_file: String,
    },
}

fn default_pool_dim() -> usize {
    1
}

fn default_tts_postprocess() -> bool {
    true
}

/// Complete model metadata describing execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier (e.g., "whisper-tiny", "gpt2", "stable-diffusion")
    pub model_id: String,

    /// Model version (semver recommended: "1.2.0")
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

    /// Optional: Additional metadata (tags, author, license, etc.)
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ModelMetadata {
    /// Create a simple one-shot model metadata
    pub fn simple(
        model_id: impl Into<String>,
        version: impl Into<String>,
        model_file: impl Into<String>,
    ) -> Self {
        let model_file = model_file.into();
        Self {
            model_id: model_id.into(),
            version: version.into(),
            execution_template: ExecutionTemplate::SimpleMode {
                model_file: model_file.clone(),
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files: vec![model_file],
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a pipeline-based model metadata
    pub fn pipeline(
        model_id: impl Into<String>,
        version: impl Into<String>,
        stages: Vec<PipelineStage>,
        files: Vec<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            version: version.into(),
            execution_template: ExecutionTemplate::Pipeline {
                stages,
                config: HashMap::new(),
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files,
            description: None,
            metadata: HashMap::new(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_mode_serialization() {
        let metadata = ModelMetadata::simple("mnist", "1.0", "mnist.onnx")
            .with_preprocessing(PreprocessingStep::Normalize {
                mean: vec![0.1307],
                std: vec![0.3081],
            })
            .with_postprocessing(PostprocessingStep::Argmax { dim: None });

        let json = serde_json::to_string_pretty(&metadata).unwrap();
        println!("{}", json);

        // Deserialize back
        let parsed: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_id, "mnist");
        assert_eq!(parsed.preprocessing.len(), 1);
        assert_eq!(parsed.postprocessing.len(), 1);
    }

    #[test]
    fn test_pipeline_mode_serialization() {
        let encoder_stage = PipelineStage {
            name: "encoder".to_string(),
            model_file: "encoder.onnx".to_string(),
            execution_mode: ExecutionMode::SingleShot,
            inputs: vec!["mel_spectrogram".to_string()],
            outputs: vec!["cross_k".to_string(), "cross_v".to_string()],
            config: HashMap::new(),
        };

        let decoder_stage = PipelineStage {
            name: "decoder".to_string(),
            model_file: "decoder.onnx".to_string(),
            execution_mode: ExecutionMode::Autoregressive {
                max_tokens: 448,
                start_token_id: 50258,
                end_token_id: 50256,
                repetition_penalty: 0.5,
            },
            inputs: vec![
                "tokens".to_string(),
                "kv_cache_k".to_string(),
                "kv_cache_v".to_string(),
                "cross_k".to_string(),
                "cross_v".to_string(),
                "offset".to_string(),
            ],
            outputs: vec![
                "logits".to_string(),
                "updated_kv_k".to_string(),
                "updated_kv_v".to_string(),
            ],
            config: HashMap::new(),
        };

        let metadata = ModelMetadata::pipeline(
            "whisper-tiny",
            "1.2",
            vec![encoder_stage, decoder_stage],
            vec![
                "encoder.onnx".to_string(),
                "decoder.onnx".to_string(),
                "tokens.txt".to_string(),
            ],
        )
        .with_preprocessing(PreprocessingStep::MelSpectrogram {
            preset: Some("whisper".to_string()),
            n_mels: 80,
            sample_rate: 16000,
            fft_size: 400,
            hop_length: 160,
            mel_scale: MelScaleType::Slaney,
            max_frames: Some(3000),
        })
        .with_postprocessing(PostprocessingStep::BPEDecode {
            vocab_file: "tokens.txt".to_string(),
        })
        .with_description("Whisper-tiny English ASR model");

        let json = serde_json::to_string_pretty(&metadata).unwrap();
        println!("{}", json);

        // Deserialize back
        let parsed: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_id, "whisper-tiny");

        match &parsed.execution_template {
            ExecutionTemplate::Pipeline { stages, .. } => {
                assert_eq!(stages.len(), 2);
                assert_eq!(stages[0].name, "encoder");
                assert_eq!(stages[1].name, "decoder");

                match &stages[1].execution_mode {
                    ExecutionMode::Autoregressive { max_tokens, .. } => {
                        assert_eq!(*max_tokens, 448);
                    }
                    _ => panic!("Expected autoregressive mode"),
                }
            }
            _ => panic!("Expected pipeline template"),
        }
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
            ExecutionMode::Autoregressive { max_tokens, .. } => {
                assert_eq!(max_tokens, 100);
            }
            _ => panic!("Expected autoregressive mode"),
        }
    }

    #[test]
    fn test_preprocessing_steps() {
        let steps = vec![
            PreprocessingStep::MelSpectrogram {
                preset: None,
                n_mels: 80,
                sample_rate: 16000,
                fft_size: 400,
                hop_length: 160,
                mel_scale: MelScaleType::Slaney,
                max_frames: Some(3000),
            },
            PreprocessingStep::Normalize {
                mean: vec![0.5, 0.5, 0.5],
                std: vec![0.5, 0.5, 0.5],
            },
        ];

        let json = serde_json::to_string_pretty(&steps).unwrap();
        println!("{}", json);

        let parsed: Vec<PreprocessingStep> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 2);
    }
}
