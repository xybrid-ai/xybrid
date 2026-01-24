//! Preprocessing and postprocessing step definitions.
//!
//! This module defines the steps that transform data before and after model execution.

use serde::{Deserialize, Serialize};

// ============================================================================
// Preprocessing Steps
// ============================================================================

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
            PreprocessingStep::Phonemize { .. } => "Phonemize",
        }
    }
}

// ============================================================================
// Postprocessing Steps
// ============================================================================

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
        }
    }
}

// ============================================================================
// Helper Types
// ============================================================================

/// Phonemizer backend for text-to-phoneme conversion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PhonemizerBackend {
    /// CMU Pronouncing Dictionary (ARPABET -> IPA conversion)
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
