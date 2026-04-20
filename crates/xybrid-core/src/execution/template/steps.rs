//! Preprocessing and postprocessing step definitions.
//!
//! This module defines the steps that transform data before and after model execution.

use serde::{Deserialize, Serialize};

#[cfg(feature = "schema")]
use schemars::JsonSchema;

// ============================================================================
// Preprocessing Steps
// ============================================================================

/// Preprocessing step applied before model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
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

    /// Convert text to raw IPA phoneme strings without token-ID mapping
    ///
    /// Use for: LLM-based TTS models (NeuTTS) that inject phonemes into text prompts
    PhonemeRaw {
        /// Phonemization backend to use (default: MisakiDictionary)
        #[serde(default)]
        backend: PhonemizerBackend,

        /// Language code (e.g., "en-us")
        #[serde(default)]
        language: Option<String>,
    },

    /// Phonemize text to IPA symbols for TTS
    ///
    /// Use for: TTS models (KittenTTS, Piper, Kokoro) that require phoneme input
    Phonemize {
        /// Path to tokens.txt or vocab file (maps IPA symbols to token IDs)
        tokens_file: String,

        /// Phonemization backend to use (default: MisakiDictionary)
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

        /// Number of silence tokens to prepend before speech onset (default: None/0).
        /// Helps smooth plosive-initial words (e.g., "b", "p", "t" sounds).
        /// Uses token ID 30 (Kokoro silence token).
        #[serde(default)]
        silence_tokens: Option<u8>,
    },
}

impl PreprocessingStep {
    /// Get the name of this preprocessing step for tracing
    pub fn step_name(&self) -> &'static str {
        match self {
            PreprocessingStep::MelSpectrogram { .. } => "MelSpectrogram",
            PreprocessingStep::Tokenize { .. } => "Tokenize",
            PreprocessingStep::Normalize { .. } => "Normalize",
            PreprocessingStep::Resize { .. } => "Resize",
            PreprocessingStep::CenterCrop { .. } => "CenterCrop",
            PreprocessingStep::AudioDecode { .. } => "AudioDecode",
            PreprocessingStep::Reshape { .. } => "Reshape",
            PreprocessingStep::PhonemeRaw { .. } => "PhonemeRaw",
            PreprocessingStep::Phonemize { .. } => "Phonemize",
        }
    }
}

// ============================================================================
// Postprocessing Steps
// ============================================================================

/// Postprocessing step applied after model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
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

        /// Whether to trim trailing near-silence from the waveform (default: false)
        #[serde(default)]
        trim_trailing_silence: bool,
    },

    /// Decode Whisper token IDs to text using HuggingFace tokenizer
    ///
    /// Use for: Whisper ASR models with HuggingFace tokenizer format
    WhisperDecode {
        /// Path to tokenizer.json file (HuggingFace format)
        tokenizer_file: String,
    },

    /// Decode LLM speech token output to audio via an ONNX codec decoder
    ///
    /// Use for: Codec-based TTS models (NeuTTS, MARS5, F5-TTS)
    CodecDecode {
        /// Path to the ONNX codec decoder model (relative to bundle root)
        decoder_model: String,

        /// Output sample rate in Hz
        sample_rate: u32,

        /// Regex pattern to extract speech token IDs from LLM output text
        token_pattern: String,

        /// Whether to apply postprocessing (normalization, silence trimming)
        #[serde(default = "default_tts_postprocess")]
        apply_postprocessing: bool,
    },
}

impl PostprocessingStep {
    /// Get the name of this postprocessing step for tracing
    pub fn step_name(&self) -> &'static str {
        match self {
            PostprocessingStep::BPEDecode { .. } => "BPEDecode",
            PostprocessingStep::Argmax { .. } => "Argmax",
            PostprocessingStep::Softmax { .. } => "Softmax",
            PostprocessingStep::TopK { .. } => "TopK",
            PostprocessingStep::Threshold { .. } => "Threshold",
            PostprocessingStep::TemperatureSample { .. } => "TemperatureSample",
            PostprocessingStep::Denormalize { .. } => "Denormalize",
            PostprocessingStep::MeanPool { .. } => "MeanPool",
            PostprocessingStep::CTCDecode { .. } => "CTCDecode",
            PostprocessingStep::TTSAudioEncode { .. } => "TTSAudioEncode",
            PostprocessingStep::WhisperDecode { .. } => "WhisperDecode",
            PostprocessingStep::CodecDecode { .. } => "CodecDecode",
        }
    }
}

// ============================================================================
// Helper Types
// ============================================================================

/// Phonemizer backend for text-to-phoneme conversion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
pub enum PhonemizerBackend {
    /// Legacy. ARPABET-based. May produce lower quality for models trained on Misaki/espeak IPA.
    CmuDictionary,

    /// Multi-language. Requires espeak-ng system installation.
    /// Install with: `brew install espeak-ng` (macOS) or `apt-get install espeak-ng` (Linux)
    EspeakNG,

    /// Default. Pure Rust, no system dependencies. Recommended for all TTS models (Kokoro, KittenTTS).
    /// Uses bundled JSON dictionaries (us_gold.json + us_silver.json)
    /// Falls back to rule-based G2P for out-of-vocabulary words
    #[default]
    MisakiDictionary,

    /// Hybrid dictionary + ONNX neural G2P fallback. No system dependencies.
    /// Uses a ~274k word dictionary for fast lookup and a small ONNX model (~59MB)
    /// for unknown words via CTC decoding.
    /// Requires `open-phonemizer.onnx` and `dictionary.json` in the model directory.
    OpenPhonemizer,
}

impl PhonemizerBackend {
    /// Create a trait object for this backend variant.
    ///
    /// # Arguments
    /// - `base_path`: Model directory path (used by MisakiDictionary for dictionary files)
    /// - `dict_path`: Optional dictionary file path (used by CmuDictionary)
    /// - `language`: Optional language code (used by EspeakNG, e.g. "en-us")
    pub fn create(
        &self,
        base_path: &str,
        dict_path: Option<&str>,
        language: Option<&str>,
    ) -> Box<dyn crate::execution::preprocessing::backends::PhonemizerBackend> {
        use crate::execution::preprocessing::backends::{
            CmuDictionaryBackend, EspeakBackend, MisakiBackend, OpenPhonemizerBackend,
        };

        match self {
            PhonemizerBackend::CmuDictionary => {
                Box::new(CmuDictionaryBackend::new(dict_path.map(|s| s.to_string())))
            }
            PhonemizerBackend::MisakiDictionary => {
                Box::new(MisakiBackend::new(base_path.to_string()))
            }
            PhonemizerBackend::EspeakNG => {
                let lang = language.unwrap_or("en-us").to_string();
                Box::new(EspeakBackend::new(lang))
            }
            PhonemizerBackend::OpenPhonemizer => {
                Box::new(OpenPhonemizerBackend::new(base_path.to_string()))
            }
        }
    }
}

/// Mel frequency scale type for mel spectrogram preprocessing.
///
/// Different models use different mel scale formulas. Using the wrong scale
/// will produce incorrect mel spectrograms and poor model performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum MelScaleType {
    /// Slaney mel scale (used by Whisper, transformers.js, librosa default).
    ///
    /// Formula:
    /// - For freq < 1000 Hz: `mel = 3 * freq / 200`
    /// - For freq >= 1000 Hz: `mel = 15 + 27 * ln(freq / 1000) / ln(6.4)`
    #[default]
    Slaney,

    /// HTK mel scale (used by older implementations, mel-spec library).
    ///
    /// Formula: `mel = 2595 * log10(1 + freq / 700)`
    Htk,
}

/// Tokenizer type for text preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
pub enum TokenizerType {
    /// Byte-Pair Encoding (GPT-2, GPT-3, Whisper)
    BPE,

    /// WordPiece (BERT, DistilBERT)
    WordPiece,

    /// SentencePiece (T5, ALBERT, XLNet)
    SentencePiece,
}

/// Interpolation method for image resizing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
pub enum InterpolationMethod {
    Nearest,
    #[default]
    Bilinear,
    Bicubic,
}

// ============================================================================
// Default Functions
// ============================================================================

fn default_add_padding() -> bool {
    true
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

fn default_pool_dim() -> usize {
    1
}

fn default_tts_postprocess() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phoneme_raw_serde_round_trip() {
        let json = r#"{
            "type": "PhonemeRaw",
            "backend": "EspeakNG",
            "language": "en-us"
        }"#;

        let step: PreprocessingStep = serde_json::from_str(json).unwrap();
        match &step {
            PreprocessingStep::PhonemeRaw { backend, language } => {
                assert!(matches!(backend, PhonemizerBackend::EspeakNG));
                assert_eq!(language.as_deref(), Some("en-us"));
            }
            _ => panic!("Expected PhonemeRaw variant"),
        }

        let serialized = serde_json::to_string(&step).unwrap();
        let deserialized: PreprocessingStep = serde_json::from_str(&serialized).unwrap();
        let reserialized = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(serialized, reserialized);
    }

    #[test]
    fn phoneme_raw_defaults() {
        let json = r#"{"type": "PhonemeRaw"}"#;

        let step: PreprocessingStep = serde_json::from_str(json).unwrap();
        match &step {
            PreprocessingStep::PhonemeRaw { backend, language } => {
                assert!(matches!(backend, PhonemizerBackend::MisakiDictionary));
                assert!(language.is_none());
            }
            _ => panic!("Expected PhonemeRaw variant"),
        }
    }

    #[test]
    fn codec_decode_serde_round_trip() {
        let json = r#"{
            "type": "CodecDecode",
            "decoder_model": "neucodec_mini_decoder.onnx",
            "sample_rate": 24000,
            "token_pattern": "<\\|speech_(\\d+)\\|>",
            "apply_postprocessing": true
        }"#;

        let step: PostprocessingStep = serde_json::from_str(json).unwrap();
        match &step {
            PostprocessingStep::CodecDecode {
                decoder_model,
                sample_rate,
                token_pattern,
                apply_postprocessing,
            } => {
                assert_eq!(decoder_model, "neucodec_mini_decoder.onnx");
                assert_eq!(*sample_rate, 24000);
                assert_eq!(token_pattern, r"<\|speech_(\d+)\|>");
                assert!(*apply_postprocessing);
            }
            _ => panic!("Expected CodecDecode variant"),
        }

        let serialized = serde_json::to_string(&step).unwrap();
        let deserialized: PostprocessingStep = serde_json::from_str(&serialized).unwrap();
        let reserialized = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(serialized, reserialized);
    }

    #[test]
    fn codec_decode_default_apply_postprocessing() {
        let json = r#"{
            "type": "CodecDecode",
            "decoder_model": "decoder.onnx",
            "sample_rate": 24000,
            "token_pattern": "<\\|speech_(\\d+)\\|>"
        }"#;

        let step: PostprocessingStep = serde_json::from_str(json).unwrap();
        match &step {
            PostprocessingStep::CodecDecode {
                apply_postprocessing,
                ..
            } => {
                assert!(*apply_postprocessing);
            }
            _ => panic!("Expected CodecDecode variant"),
        }
    }
}
