//! TTS engine implementation.

use super::error::TtsError;
use super::request::{SynthesisRequest, Voice};
use super::response::AudioOutput;
use crate::phonemizer::{load_tokens_map, postprocess_tts_audio, Phonemizer};
use ndarray::Array;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Default sample rate for KittenTTS output
const SAMPLE_RATE: u32 = 24000;

/// Voice embedding dimension
const EMBEDDING_DIM: usize = 256;

/// TTS configuration.
#[derive(Debug, Clone)]
pub struct TtsConfig {
    /// Path to the TTS model directory
    pub model_dir: PathBuf,

    /// Path to CMU dictionary (optional, will search default locations)
    pub dict_path: Option<PathBuf>,

    /// Whether to use FP16 model (default: true)
    pub use_fp16: bool,
}

impl TtsConfig {
    /// Create a new config with model directory.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            dict_path: None,
            use_fp16: true,
        }
    }

    /// Set custom dictionary path.
    pub fn with_dict_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.dict_path = Some(path.into());
        self
    }

    /// Use FP32 model instead of FP16.
    pub fn with_fp32(mut self) -> Self {
        self.use_fp16 = false;
        self
    }

    /// Get the model file path.
    fn model_path(&self) -> PathBuf {
        if self.use_fp16 {
            self.model_dir.join("model.fp16.onnx")
        } else {
            self.model_dir.join("model.onnx")
        }
    }
}

/// Text-to-Speech engine wrapping phonemizer and KittenTTS.
///
/// ## Example
///
/// ```rust,ignore
/// use xybrid_core::tts::{Tts, SynthesisRequest, Voice};
///
/// let tts = Tts::new("path/to/model")?;
///
/// // Simple synthesis
/// let audio = tts.synthesize("Hello, world!")?;
/// audio.save_wav("output.wav")?;
///
/// // With options
/// let audio = tts.synthesize_with(
///     SynthesisRequest::new("Hello!")
///         .with_voice(Voice::Male1)
///         .with_speed(1.2)
/// )?;
/// ```
pub struct Tts {
    config: TtsConfig,
    session: Session,
    phonemizer: Phonemizer,
    tokens_map: HashMap<char, i64>,
    voice_embeddings: Vec<Vec<f32>>,
}

impl Tts {
    /// Create a new TTS engine with default configuration.
    ///
    /// # Arguments
    /// * `model_dir` - Path to KittenTTS model directory containing:
    ///   - model.fp16.onnx (or model.onnx)
    ///   - tokens.txt
    ///   - voices.bin
    pub fn new(model_dir: impl Into<PathBuf>) -> Result<Self, TtsError> {
        let config = TtsConfig::new(model_dir);
        Self::with_config(config)
    }

    /// Create a new TTS engine with custom configuration.
    pub fn with_config(config: TtsConfig) -> Result<Self, TtsError> {
        // Validate model directory exists
        if !config.model_dir.exists() {
            return Err(TtsError::ModelNotFound(
                config.model_dir.display().to_string(),
            ));
        }

        // Validate model file exists
        let model_path = config.model_path();
        if !model_path.exists() {
            return Err(TtsError::ModelNotFound(model_path.display().to_string()));
        }

        // Load phonemizer
        let phonemizer = if let Some(ref dict_path) = config.dict_path {
            if !dict_path.exists() {
                return Err(TtsError::DictionaryNotFound(
                    dict_path.display().to_string(),
                ));
            }
            Phonemizer::new(dict_path).map_err(|e| TtsError::PhonemizationError(e.to_string()))?
        } else {
            Phonemizer::from_default_location()
                .map_err(|e| TtsError::DictionaryNotFound(e.to_string()))?
        };

        // Load tokens map
        let tokens_path = config.model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(TtsError::ConfigError(format!(
                "tokens.txt not found in model directory: {}",
                config.model_dir.display()
            )));
        }
        let tokens_content = std::fs::read_to_string(&tokens_path)?;
        let tokens_map = load_tokens_map(&tokens_content);

        // Load voice embeddings
        let voices_path = config.model_dir.join("voices.bin");
        if !voices_path.exists() {
            return Err(TtsError::ConfigError(format!(
                "voices.bin not found in model directory: {}",
                config.model_dir.display()
            )));
        }
        let voice_embeddings = load_voice_embeddings(&voices_path)?;

        // Initialize ONNX runtime
        ort::init()
            .commit()
            .map_err(|e| TtsError::InferenceError(format!("Failed to init ONNX runtime: {}", e)))?;

        // Load model
        let session = Session::builder()
            .map_err(|e| TtsError::InferenceError(format!("Failed to create session: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TtsError::InferenceError(format!("Failed to set optimization: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| TtsError::InferenceError(format!("Failed to load model: {}", e)))?;

        Ok(Self {
            config,
            session,
            phonemizer,
            tokens_map,
            voice_embeddings,
        })
    }

    /// Synthesize speech from text using default options.
    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        let request = SynthesisRequest::new(text);
        self.synthesize_with(request)
    }

    /// Synthesize speech with custom options.
    pub fn synthesize_with(&mut self, request: SynthesisRequest) -> Result<AudioOutput, TtsError> {
        // Validate request
        request.validate()?;

        let start = Instant::now();

        // Phonemize text
        let token_ids = self
            .phonemizer
            .text_to_token_ids(&request.text, &self.tokens_map, true);

        if token_ids.is_empty() {
            return Err(TtsError::PhonemizationError(
                "No tokens generated from text".to_string(),
            ));
        }

        // Get voice embedding
        let voice_embedding = self
            .voice_embeddings
            .get(request.voice.index())
            .ok_or_else(|| TtsError::InvalidVoice(format!("{}", request.voice)))?
            .clone();

        // Prepare input tensors
        let seq_len = token_ids.len();
        let input_ids: Array<i64, _> =
            Array::from_shape_vec((1, seq_len), token_ids).map_err(|e| {
                TtsError::InferenceError(format!("Failed to create input tensor: {}", e))
            })?;

        let style: Array<f32, _> =
            Array::from_shape_vec((1, EMBEDDING_DIM), voice_embedding).map_err(|e| {
                TtsError::InferenceError(format!("Failed to create style tensor: {}", e))
            })?;

        let speed: Array<f32, _> =
            Array::from_shape_vec((1,), vec![request.speed]).map_err(|e| {
                TtsError::InferenceError(format!("Failed to create speed tensor: {}", e))
            })?;

        // Create ONNX tensors
        let input_ids_tensor = ort::value::Tensor::from_array(input_ids)
            .map_err(|e| TtsError::InferenceError(format!("Failed to create tensor: {}", e)))?;
        let style_tensor = ort::value::Tensor::from_array(style)
            .map_err(|e| TtsError::InferenceError(format!("Failed to create tensor: {}", e)))?;
        let speed_tensor = ort::value::Tensor::from_array(speed)
            .map_err(|e| TtsError::InferenceError(format!("Failed to create tensor: {}", e)))?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "style" => style_tensor,
                "speed" => speed_tensor
            ])
            .map_err(|e| TtsError::InferenceError(format!("Inference failed: {}", e)))?;

        // Extract waveform
        let waveform = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::InferenceError(format!("Failed to extract output: {}", e)))?;

        let (_, audio_data) = waveform;
        let raw_samples: Vec<f32> = audio_data.iter().cloned().collect();

        // Apply postprocessing if enabled
        let samples = if request.postprocess {
            postprocess_tts_audio(&raw_samples, SAMPLE_RATE)
        } else {
            raw_samples
        };

        let latency_ms = start.elapsed().as_millis() as u32;

        Ok(AudioOutput::new(samples, SAMPLE_RATE).with_latency(latency_ms))
    }

    /// Get available voices.
    pub fn voices(&self) -> &'static [Voice] {
        Voice::all()
    }

    /// Get the number of loaded voice embeddings.
    pub fn num_voices(&self) -> usize {
        self.voice_embeddings.len()
    }

    /// Get the model configuration.
    pub fn config(&self) -> &TtsConfig {
        &self.config
    }

    /// Get the sample rate of output audio.
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
}

/// Load voice embeddings from voices.bin file.
fn load_voice_embeddings(path: &Path) -> Result<Vec<Vec<f32>>, TtsError> {
    let data = std::fs::read(path)?;

    // Each embedding is EMBEDDING_DIM * 4 bytes (f32)
    let embedding_bytes = EMBEDDING_DIM * 4;
    let num_voices = data.len() / embedding_bytes;

    let mut embeddings = Vec::with_capacity(num_voices);

    for voice_idx in 0..num_voices {
        let offset = voice_idx * embedding_bytes;
        let embedding: Vec<f32> = data[offset..offset + embedding_bytes]
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        embeddings.push(embedding);
    }

    if embeddings.is_empty() {
        return Err(TtsError::ConfigError(
            "No voice embeddings found in voices.bin".to_string(),
        ));
    }

    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_model_path() {
        let config = TtsConfig::new("/path/to/model");
        assert!(config.model_path().to_string_lossy().contains("fp16"));

        let config = config.with_fp32();
        assert!(!config.model_path().to_string_lossy().contains("fp16"));
    }

    #[test]
    fn test_config_dict_path() {
        let config = TtsConfig::new("/model").with_dict_path("/custom/dict.txt");
        assert_eq!(
            config.dict_path,
            Some(PathBuf::from("/custom/dict.txt"))
        );
    }

    // Integration tests require model files
    #[test]
    #[ignore = "Requires KittenTTS model"]
    fn test_tts_synthesis() {
        let model_dir = "test_models/kitten-tts/kitten-nano-en-v0_1-fp16";
        let mut tts = Tts::new(model_dir).unwrap();

        let audio = tts.synthesize("Hello, world!").unwrap();
        assert!(!audio.samples.is_empty());
        assert_eq!(audio.sample_rate, SAMPLE_RATE);
    }
}
