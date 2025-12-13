//! Whisper model wrapper for Candle.
//!
//! This module provides a high-level interface for running Whisper ASR
//! using the candle-transformers implementation.

use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use std::path::Path;
use tokenizers::Tokenizer;

/// Whisper model configuration
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Model size variant
    pub model_size: WhisperSize,
    /// Language for transcription (None for auto-detect)
    pub language: Option<String>,
    /// Task: transcribe or translate
    pub task: Task,
    /// Enable timestamps in output
    pub timestamps: bool,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_size: WhisperSize::Tiny,
            language: Some("en".to_string()),
            task: Task::Transcribe,
            timestamps: false,
        }
    }
}

/// Whisper model size variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
}

impl WhisperSize {
    pub fn as_str(&self) -> &'static str {
        match self {
            WhisperSize::Tiny => "tiny",
            WhisperSize::Base => "base",
            WhisperSize::Small => "small",
            WhisperSize::Medium => "medium",
            WhisperSize::Large => "large",
            WhisperSize::LargeV2 => "large-v2",
            WhisperSize::LargeV3 => "large-v3",
            WhisperSize::LargeV3Turbo => "large-v3-turbo",
        }
    }

    /// Get HuggingFace model ID
    pub fn hf_model_id(&self) -> &'static str {
        match self {
            WhisperSize::Tiny => "openai/whisper-tiny",
            WhisperSize::Base => "openai/whisper-base",
            WhisperSize::Small => "openai/whisper-small",
            WhisperSize::Medium => "openai/whisper-medium",
            WhisperSize::Large => "openai/whisper-large",
            WhisperSize::LargeV2 => "openai/whisper-large-v2",
            WhisperSize::LargeV3 => "openai/whisper-large-v3",
            WhisperSize::LargeV3Turbo => "openai/whisper-large-v3-turbo",
        }
    }
}

/// Whisper task type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    #[default]
    Transcribe,
    Translate,
}

/// Whisper model wrapper
pub struct WhisperModel {
    /// The underlying Whisper model
    model: m::model::Whisper,
    /// Tokenizer for decoding
    tokenizer: Tokenizer,
    /// Model configuration
    config: Config,
    /// Device for inference
    device: Device,
    /// Mel filter bank for audio preprocessing
    mel_filters: Vec<f32>,
    /// Special token IDs
    sot_token: u32,
    eot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
    /// User configuration
    user_config: WhisperConfig,
}

impl WhisperModel {
    /// Load a Whisper model from a local directory.
    ///
    /// The directory should contain:
    /// - `model.safetensors` - Model weights
    /// - `config.json` - Model configuration
    /// - `tokenizer.json` - Tokenizer configuration
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to model directory
    /// * `device` - Device for inference
    pub fn load(model_dir: &Path, device: &Device) -> anyhow::Result<Self> {
        Self::load_with_config(model_dir, device, WhisperConfig::default())
    }

    /// Load a Whisper model with custom configuration.
    pub fn load_with_config(
        model_dir: &Path,
        device: &Device,
        user_config: WhisperConfig,
    ) -> anyhow::Result<Self> {
        // Load configuration
        let config_path = model_dir.join("config.json");
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load mel filters
        let mel_filters_path = model_dir.join("melfilters.bytes");
        let mel_filters = if mel_filters_path.exists() {
            let mel_bytes = std::fs::read(&mel_filters_path)?;
            let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
            LittleEndian::read_f32_into(&mel_bytes, &mut mel_filters);
            mel_filters
        } else {
            // Use embedded filters based on num_mel_bins
            match config.num_mel_bins {
                80 => {
                    // Standard Whisper mel filters (80 bins)
                    anyhow::bail!("melfilters.bytes not found at {:?}. Please download from Candle examples.", mel_filters_path)
                }
                128 => {
                    anyhow::bail!("melfilters128.bytes not found at {:?}. Please download from Candle examples.", mel_filters_path)
                }
                n => anyhow::bail!("Unsupported num_mel_bins: {}", n),
            }
        };

        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, device)?
        };

        let model = m::model::Whisper::load(&vb, config.clone())?;

        // Get special token IDs
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

        // Get language token if specified
        let language_token = if let Some(ref lang) = user_config.language {
            let lang_token = format!("<|{}|>", lang);
            token_id(&tokenizer, &lang_token).ok()
        } else {
            None
        };

        Ok(Self {
            model,
            tokenizer,
            config,
            device: device.clone(),
            mel_filters,
            sot_token,
            eot_token,
            transcribe_token,
            translate_token,
            no_timestamps_token,
            language_token,
            user_config,
        })
    }

    /// Download and load a Whisper model from HuggingFace.
    ///
    /// # Arguments
    ///
    /// * `size` - Whisper model size
    /// * `device` - Device for inference
    #[cfg(feature = "hf-hub")]
    pub fn from_hf(size: WhisperSize, device: &Device) -> anyhow::Result<Self> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let api = Api::new()?;
        let repo = api.repo(Repo::new(size.hf_model_id().to_string(), RepoType::Model));

        // Download required files
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        // Create a temporary directory structure
        let model_dir = weights_path.parent().unwrap();

        Self::load_with_config(
            model_dir,
            device,
            WhisperConfig {
                model_size: size,
                ..Default::default()
            },
        )
    }

    /// Run encoder on mel spectrogram.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel spectrogram tensor [1, n_mels, n_frames]
    ///
    /// # Returns
    ///
    /// Encoder output tensor
    pub fn encode(&mut self, mel: &Tensor) -> candle_core::Result<Tensor> {
        self.model.encoder.forward(mel, true)
    }

    /// Transcribe audio from mel spectrogram.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel spectrogram tensor [1, n_mels, n_frames]
    ///
    /// # Returns
    ///
    /// Transcribed text
    pub fn transcribe(&mut self, mel: &Tensor) -> anyhow::Result<String> {
        // Run encoder
        let audio_features = self.model.encoder.forward(mel, true)?;

        // Initialize decoder tokens
        let mut tokens = vec![self.sot_token];
        if let Some(lang_token) = self.language_token {
            tokens.push(lang_token);
        }
        match self.user_config.task {
            Task::Transcribe => tokens.push(self.transcribe_token),
            Task::Translate => tokens.push(self.translate_token),
        }
        if !self.user_config.timestamps {
            tokens.push(self.no_timestamps_token);
        }

        // Autoregressive decoding
        let sample_len = self.config.max_target_positions / 2;
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?;
            let tokens_t = tokens_t.unsqueeze(0)?;

            let ys = self.model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            // Get logits for last position
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self.model.decoder.final_linear(
                &ys.i((.., seq_len - 1.., ..))?
            )?.i(0)?.i(0)?;

            // Greedy decoding: take argmax
            let next_token = logits
                .argmax(candle_core::D::Minus1)?
                .to_scalar::<u32>()?;

            if next_token == self.eot_token || tokens.len() > self.config.max_target_positions {
                break;
            }

            tokens.push(next_token);
        }

        // Decode tokens to text
        let text = self.tokenizer.decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;

        Ok(text)
    }

    /// Get model configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get mel filters.
    pub fn mel_filters(&self) -> &[f32] {
        &self.mel_filters
    }

    /// Convert PCM audio samples to mel spectrogram tensor.
    ///
    /// # Arguments
    ///
    /// * `pcm_data` - Audio samples (16kHz, mono, f32)
    ///
    /// # Returns
    ///
    /// Mel spectrogram tensor [1, n_mels, n_frames]
    ///
    /// Note: Audio is automatically truncated to 30 seconds (480,000 samples @ 16kHz)
    /// to fit within Whisper's encoder context window (1500 mel frames).
    pub fn pcm_to_mel_tensor(&self, pcm_data: &[f32]) -> anyhow::Result<Tensor> {
        // Whisper's encoder has a fixed context of 1500 mel frames (30 seconds @ 16kHz)
        // Truncate audio to 30 seconds max to avoid encoder overflow
        const MAX_SAMPLES_30S: usize = 16000 * 30; // 480,000 samples
        let pcm_data = if pcm_data.len() > MAX_SAMPLES_30S {
            &pcm_data[..MAX_SAMPLES_30S]
        } else {
            pcm_data
        };

        let mel = audio::pcm_to_mel(&self.config, pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let n_mels = self.config.num_mel_bins;

        Tensor::from_vec(mel, (1, n_mels, mel_len / n_mels), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create mel tensor: {}", e))
    }

    /// Transcribe audio from PCM samples.
    ///
    /// # Arguments
    ///
    /// * `pcm_data` - Audio samples (16kHz, mono, f32)
    ///
    /// # Returns
    ///
    /// Transcribed text
    pub fn transcribe_pcm(&mut self, pcm_data: &[f32]) -> anyhow::Result<String> {
        let mel = self.pcm_to_mel_tensor(pcm_data)?;
        self.transcribe(&mel)
    }
}

/// Helper to get token ID from tokenizer
fn token_id(tokenizer: &Tokenizer, token: &str) -> anyhow::Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow::anyhow!("Token '{}' not found in vocabulary", token))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_size_as_str() {
        assert_eq!(WhisperSize::Tiny.as_str(), "tiny");
        assert_eq!(WhisperSize::LargeV3.as_str(), "large-v3");
    }

    #[test]
    fn test_whisper_config_default() {
        let config = WhisperConfig::default();
        assert_eq!(config.model_size, WhisperSize::Tiny);
        assert_eq!(config.language, Some("en".to_string()));
        assert_eq!(config.task, Task::Transcribe);
        assert!(!config.timestamps);
    }
}
