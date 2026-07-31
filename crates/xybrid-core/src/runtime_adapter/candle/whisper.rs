//! Whisper model wrapper for Candle.
//!
//! This module provides a high-level interface for running Whisper ASR
//! using the candle-transformers implementation.

use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer;

/// Errors that can occur during Whisper model operations.
#[derive(Error, Debug)]
pub enum WhisperError {
    /// Failed to load model configuration
    #[error("Config error: {0}")]
    Config(String),

    /// Failed to load tokenizer
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Failed to load mel filters
    #[error("Mel filters error: {0}")]
    MelFilters(String),

    /// Failed to load model weights
    #[error("Model weights error: {0}")]
    Weights(String),

    /// Token not found in vocabulary
    #[error("Token '{0}' not found in vocabulary")]
    TokenNotFound(String),

    /// Requested transcription language has no token in this model's vocabulary
    ///
    /// Distinct from [`WhisperError::TokenNotFound`] on purpose: a missing
    /// `<|xx|>` token for a *caller-supplied* language is bad input, not a
    /// broken model, and callers map it to an invalid-input error rather than
    /// an inference failure.
    #[error("Unsupported language '{0}': the model vocabulary has no '<|{0}|>' token")]
    UnsupportedLanguage(String),

    /// Candle tensor/model error
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for Whisper operations.
pub type WhisperResult<T> = Result<T, WhisperError>;

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

/// Per-request transcription parameters.
///
/// These are *call arguments*, never stored on [`WhisperModel`]: the runtime
/// caches one loaded model per directory and serves concurrent requests from
/// it, so a request that asks for French must not leave the cached model
/// speaking French for the next request.
///
/// [`Default`] means "no per-request override": `language: None` falls back to
/// the language resolved at load time from [`WhisperConfig::language`], and
/// `task` is [`Task::Transcribe`]. [`WhisperModel::transcribe`] and
/// [`WhisperModel::transcribe_pcm`] additionally carry the load-time task
/// forward, so existing callers see no behavior change.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TranscribeOptions {
    /// Language code for this request (e.g. `"en"`, `"fr"`), or `None` to use
    /// the model's load-time language. Matched case-insensitively against the
    /// vocabulary's `<|xx|>` tokens; an unknown code is an error, never a
    /// silent fallback.
    pub language: Option<String>,
    /// Transcribe (keep the source language) or translate (into English).
    pub task: Task,
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
    /// Language token resolved at load time from [`WhisperConfig::language`].
    /// Used only when a request does not name its own language — see
    /// [`TranscribeOptions`].
    language_token: Option<u32>,
    /// Load-time configuration. Its `language`/`task` are fallbacks for
    /// requests that don't specify their own; `timestamps` is not a
    /// per-request parameter and always comes from here.
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
    pub fn load(model_dir: &Path, device: &Device) -> WhisperResult<Self> {
        Self::load_with_config(model_dir, device, WhisperConfig::default())
    }

    /// Load a Whisper model with custom configuration.
    pub fn load_with_config(
        model_dir: &Path,
        device: &Device,
        user_config: WhisperConfig,
    ) -> WhisperResult<Self> {
        // Load configuration
        let config_path = model_dir.join("config.json");
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| WhisperError::Tokenizer(format!("Failed to load tokenizer: {}", e)))?;

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
                    return Err(WhisperError::MelFilters(format!(
                        "melfilters.bytes not found at {:?}. Please download from Candle examples.",
                        mel_filters_path
                    )));
                }
                128 => {
                    return Err(WhisperError::MelFilters(format!(
                        "melfilters128.bytes not found at {:?}. Please download from Candle examples.",
                        mel_filters_path
                    )));
                }
                n => {
                    return Err(WhisperError::MelFilters(format!(
                        "Unsupported num_mel_bins: {}",
                        n
                    )));
                }
            }
        };

        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        // SAFETY: `from_mmaped_safetensors` memory-maps the weights file, and
        // the resulting borrow is sound only while the file is not mutated
        // underneath the mapping. `weights_path` is inside xybrid's
        // app-controlled model cache (written once at download, read-only
        // thereafter), so no concurrent writer aliases the mapping for the
        // lifetime of the returned `VarBuilder`.
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
    #[cfg(feature = "candle-hub")]
    pub fn from_hf(size: WhisperSize, device: &Device) -> WhisperResult<Self> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let api = Api::new().map_err(|e| WhisperError::Config(format!("HF API error: {}", e)))?;
        let repo = api.repo(Repo::new(size.hf_model_id().to_string(), RepoType::Model));

        // Download required files
        let _config_path = repo
            .get("config.json")
            .map_err(|e| WhisperError::Config(format!("Failed to download config: {}", e)))?;
        let _tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| WhisperError::Tokenizer(format!("Failed to download tokenizer: {}", e)))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| WhisperError::Weights(format!("Failed to download weights: {}", e)))?;

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
    /// * `mel` - Mel spectrogram tensor [1, n_mels, n_frames], where `n_frames`
    ///   is at most [`m::N_FRAMES`]. Longer mels must be sliced into windows
    ///   first — see [`mel_windows`] — because the encoder's positional
    ///   embedding table only has `n_audio_ctx` (`N_FRAMES / 2`) rows.
    ///
    /// # Returns
    ///
    /// Encoder output tensor
    pub fn encode(&mut self, mel: &Tensor) -> candle_core::Result<Tensor> {
        self.model.encoder.forward(mel, true)
    }

    /// Transcribe audio from mel spectrogram using the model's load-time
    /// language and task.
    ///
    /// Equivalent to [`WhisperModel::transcribe_with_options`] with the
    /// load-time fallbacks, so callers that predate per-request options keep
    /// their existing behavior.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel spectrogram tensor [1, n_mels, n_frames]
    ///
    /// # Returns
    ///
    /// Transcribed text (empty when the mel has no frames)
    pub fn transcribe(&mut self, mel: &Tensor) -> WhisperResult<String> {
        let options = self.load_time_options();
        self.transcribe_with_options(mel, &options)
    }

    /// Transcribe audio from mel spectrogram with per-request options.
    ///
    /// The mel is sliced into encoder-sized windows by [`mel_windows`] and each
    /// window is encoded and decoded independently, then the per-window texts
    /// are joined with a single space. Audio of any length is accepted; nothing
    /// is truncated.
    ///
    /// `options` is resolved into a forced-token prefix once, before the window
    /// loop, and nothing from it is written back to `self` — two calls with
    /// different languages on the same loaded model are independent.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel spectrogram tensor [1, n_mels, n_frames]
    /// * `options` - Per-request language and task
    ///
    /// # Errors
    ///
    /// [`WhisperError::UnsupportedLanguage`] when `options.language` names a
    /// language this model's vocabulary has no token for.
    ///
    /// # Returns
    ///
    /// Transcribed text (empty when the mel has no frames)
    pub(crate) fn transcribe_with_options(
        &mut self,
        mel: &Tensor,
        options: &TranscribeOptions,
    ) -> WhisperResult<String> {
        let prefix = self.decode_prefix(options)?;
        let (_, _, content_frames) = mel.dims3()?;

        let mut segments: Vec<String> = Vec::new();
        for (start, len) in mel_windows(content_frames) {
            let mel_segment = mel.narrow(2, start, len)?;

            // The encoder's blocks are self-attention only (`xa: None`), so the
            // flush flag is inert there and recomputed per call; `true` matches
            // candle's own whisper example.
            let audio_features = self.model.encoder.forward(&mel_segment, true)?;

            let text = self.decode_segment(&audio_features, &prefix)?;
            let text = text.trim();
            if !text.is_empty() {
                segments.push(text.to_string());
            }
        }

        Ok(segments.join(" "))
    }

    /// Options standing in for "no per-request override": both language and
    /// task fall back to what the model was loaded with.
    fn load_time_options(&self) -> TranscribeOptions {
        TranscribeOptions {
            language: None,
            task: self.user_config.task,
        }
    }

    /// Resolve `options` into the forced-token prefix every window starts from.
    fn decode_prefix(&self, options: &TranscribeOptions) -> WhisperResult<Vec<u32>> {
        let language_token = match options.language.as_deref() {
            Some(language) => Some(self.resolve_language_token(language)?),
            // No per-request language: use whatever was resolved at load time
            // (`None` there means auto-detect, i.e. no forced language token).
            None => self.language_token,
        };

        Ok(build_decode_prefix(
            self.sot_token,
            language_token,
            options.task,
            self.transcribe_token,
            self.translate_token,
            self.no_timestamps_token,
            self.user_config.timestamps,
        ))
    }

    /// Look up the `<|xx|>` token for a caller-supplied language code.
    ///
    /// Whisper spells these tokens in lowercase, so the code is lowercased
    /// before lookup; `"FR"` and `"fr"` both resolve to `<|fr|>`. An
    /// unrecognized code is rejected rather than silently dropped — decoding
    /// without a language token makes Whisper auto-detect, which would look
    /// like the request succeeded while ignoring what it asked for.
    fn resolve_language_token(&self, language: &str) -> WhisperResult<u32> {
        let token = format!("<|{}|>", language.trim().to_ascii_lowercase());
        self.tokenizer
            .token_to_id(&token)
            .ok_or_else(|| WhisperError::UnsupportedLanguage(language.to_string()))
    }

    /// Greedily decode one encoder window into text.
    ///
    /// Starts from `prefix`, the same forced-token prefix for every window, so
    /// each window decodes independently.
    fn decode_segment(&mut self, audio_features: &Tensor, prefix: &[u32]) -> WhisperResult<String> {
        let mut tokens = prefix.to_vec();

        // Autoregressive decoding
        let sample_len = self.config.max_target_positions / 2;
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?;
            let tokens_t = tokens_t.unsqueeze(0)?;

            // The decoder caches the cross-attention keys/values derived from
            // `audio_features`. Those differ per window, so the first step of
            // every window must flush the cache — otherwise later windows would
            // attend to the first window's audio.
            let ys = self
                .model
                .decoder
                .forward(&tokens_t, audio_features, i == 0)?;

            // Get logits for last position
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((.., seq_len - 1.., ..))?)?
                .i(0)?
                .i(0)?;

            // Greedy decoding: take argmax
            let next_token = logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;

            if next_token == self.eot_token || tokens.len() > self.config.max_target_positions {
                break;
            }

            tokens.push(next_token);
        }

        // Decode tokens to text
        self.tokenizer
            .decode(&tokens, true)
            .map_err(|e| WhisperError::Tokenizer(format!("Tokenizer decode error: {}", e)))
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
    /// Note: audio of any length is accepted and nothing is truncated. The
    /// resulting mel may exceed the encoder's [`m::N_FRAMES`] context; callers
    /// slice it into encoder windows with [`mel_windows`], which is what
    /// [`WhisperModel::transcribe`] does.
    pub fn pcm_to_mel_tensor(&self, pcm_data: &[f32]) -> WhisperResult<Tensor> {
        let mel = audio::pcm_to_mel(&self.config, pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let n_mels = self.config.num_mel_bins;

        Tensor::from_vec(mel, (1, n_mels, mel_len / n_mels), &self.device)
            .map_err(WhisperError::from)
    }

    /// Transcribe audio from PCM samples using the model's load-time language
    /// and task.
    ///
    /// # Arguments
    ///
    /// * `pcm_data` - Audio samples (16kHz, mono, f32)
    ///
    /// # Returns
    ///
    /// Transcribed text
    pub fn transcribe_pcm(&mut self, pcm_data: &[f32]) -> WhisperResult<String> {
        let options = self.load_time_options();
        self.transcribe_pcm_with_options(pcm_data, &options)
    }

    /// Transcribe audio from PCM samples with per-request options.
    ///
    /// # Arguments
    ///
    /// * `pcm_data` - Audio samples (16kHz, mono, f32)
    /// * `options` - Per-request language and task
    ///
    /// # Errors
    ///
    /// [`WhisperError::UnsupportedLanguage`] when `options.language` names a
    /// language this model's vocabulary has no token for.
    ///
    /// # Returns
    ///
    /// Transcribed text
    pub(crate) fn transcribe_pcm_with_options(
        &mut self,
        pcm_data: &[f32],
        options: &TranscribeOptions,
    ) -> WhisperResult<String> {
        let mel = self.pcm_to_mel_tensor(pcm_data)?;
        self.transcribe_with_options(&mel, options)
    }
}

/// The forced-token prefix Whisper's decoder starts every window from.
///
/// Order is fixed by the model's training: start-of-transcript, then the
/// language token (absent means auto-detect), then the task token, then
/// `<|notimestamps|>` when timestamps are off. Kept as a free function over
/// plain token ids so the ordering is testable without weights or a tokenizer
/// — the prefix is the entire mechanism by which `language` and `task` take
/// effect, and getting it wrong fails silently as a wrong-language transcript
/// rather than as an error.
fn build_decode_prefix(
    sot: u32,
    language: Option<u32>,
    task: Task,
    transcribe: u32,
    translate: u32,
    no_timestamps: u32,
    timestamps: bool,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(4);
    tokens.push(sot);
    if let Some(language) = language {
        tokens.push(language);
    }
    tokens.push(match task {
        Task::Transcribe => transcribe,
        Task::Translate => translate,
    });
    if !timestamps {
        tokens.push(no_timestamps);
    }
    tokens
}

/// Offsets and lengths of the encoder windows covering `content_frames` mel frames.
///
/// Whisper's audio encoder halves the frame count in `conv2` and then indexes a
/// positional-embedding table with exactly `n_audio_ctx` (= [`m::N_FRAMES`] / 2)
/// rows, so a single forward pass accepts at most `N_FRAMES` (3000) mel frames.
/// candle's `pcm_to_mel` deliberately over-pads beyond that — it rounds the
/// frame count up to a multiple of 1500 and then appends one more 1500-frame
/// chunk unconditionally — because it expects the caller to slice. This is that
/// slicing loop; candle's own whisper example runs the equivalent between the
/// mel step and the greedy decode.
///
/// Returned `(start, len)` windows tile `[0, content_frames)` contiguously: no
/// gaps, no overlap, every `len` in `1..=N_FRAMES`, and `sum(len) ==
/// content_frames`, so no audio is silently dropped.
///
/// `content_frames == 0` yields no windows, so a caller transcribes nothing and
/// returns an empty transcript rather than encoding an empty tensor. A mel built
/// from real PCM never hits this: `pcm_to_mel` always pads to at least one full
/// 1500-frame chunk, even for zero samples. It is reachable only when a mel
/// tensor is handed in directly.
pub(crate) fn mel_windows(content_frames: usize) -> Vec<(usize, usize)> {
    let mut windows = Vec::with_capacity(content_frames.div_ceil(m::N_FRAMES));
    let mut seek = 0;
    while seek < content_frames {
        let len = usize::min(content_frames - seek, m::N_FRAMES);
        windows.push((seek, len));
        seek += len;
    }
    windows
}

/// Helper to get token ID from tokenizer
fn token_id(tokenizer: &Tokenizer, token: &str) -> WhisperResult<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| WhisperError::TokenNotFound(token.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_size_as_str() {
        assert_eq!(WhisperSize::Tiny.as_str(), "tiny");
        assert_eq!(WhisperSize::LargeV3.as_str(), "large-v3");
    }

    /// `WhisperConfig` is now the *load-time fallback*, not the per-request
    /// truth: since [`TranscribeOptions`] exists, `language: Some("en")` only
    /// decides what a request that names no language of its own gets. This
    /// test still pins those load-time values (dropping them would silently
    /// change what an unspecified request decodes as), but it deliberately no
    /// longer stands for "Whisper always decodes English" — the companion
    /// assertion below is what keeps that reading from creeping back.
    #[test]
    fn test_whisper_config_default_is_load_time_fallback() {
        let config = WhisperConfig::default();
        assert_eq!(config.model_size, WhisperSize::Tiny);
        assert_eq!(config.language, Some("en".to_string()));
        assert_eq!(config.task, Task::Transcribe);
        assert!(!config.timestamps);

        // Per-request default is "no override", which is not the same value.
        let options = TranscribeOptions::default();
        assert_eq!(options.language, None);
        assert_eq!(options.task, Task::Transcribe);
    }

    // Synthetic token ids for the prefix tests. Real Whisper ids are ~50257+
    // and adjacent, which makes an off-by-one ordering bug invisible in a
    // failure diff; small distinct numbers make the *shape* of the prefix the
    // thing under test.
    const SOT: u32 = 1;
    const EN: u32 = 2;
    const FR: u32 = 3;
    const TRANSCRIBE: u32 = 4;
    const TRANSLATE: u32 = 5;
    const NO_TIMESTAMPS: u32 = 6;

    /// `build_decode_prefix` with this file's synthetic ids.
    fn prefix(language: Option<u32>, task: Task, timestamps: bool) -> Vec<u32> {
        build_decode_prefix(
            SOT,
            language,
            task,
            TRANSCRIBE,
            TRANSLATE,
            NO_TIMESTAMPS,
            timestamps,
        )
    }

    #[test]
    fn test_decode_prefix_default_is_language_then_transcribe() {
        assert_eq!(
            prefix(Some(EN), Task::Transcribe, false),
            vec![SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
        );
    }

    #[test]
    fn test_decode_prefix_language_fr() {
        // The D3 regression: before per-request options this stayed <|en|>,
        // and French audio came back decoded as English.
        assert_eq!(
            prefix(Some(FR), Task::Transcribe, false),
            vec![SOT, FR, TRANSCRIBE, NO_TIMESTAMPS]
        );
    }

    #[test]
    fn test_decode_prefix_task_translate() {
        assert_eq!(
            prefix(Some(EN), Task::Translate, false),
            vec![SOT, EN, TRANSLATE, NO_TIMESTAMPS]
        );
    }

    #[test]
    fn test_decode_prefix_language_fr_and_task_translate() {
        assert_eq!(
            prefix(Some(FR), Task::Translate, false),
            vec![SOT, FR, TRANSLATE, NO_TIMESTAMPS]
        );
    }

    #[test]
    fn test_decode_prefix_without_language_omits_language_token() {
        // No language token at all is how Whisper is asked to auto-detect;
        // it must not degrade into some default id sitting in the slot.
        assert_eq!(
            prefix(None, Task::Transcribe, false),
            vec![SOT, TRANSCRIBE, NO_TIMESTAMPS]
        );
    }

    #[test]
    fn test_decode_prefix_with_timestamps_omits_no_timestamps_token() {
        assert_eq!(
            prefix(Some(EN), Task::Transcribe, true),
            vec![SOT, EN, TRANSCRIBE]
        );
    }

    #[test]
    fn test_decode_prefix_alternates_without_carrying_state() {
        // The cached-model hazard: one loaded model serves many requests, so
        // en → fr → en → fr must produce alternating prefixes, and each must
        // equal what that language produces in isolation. A builder that read
        // or wrote model state would drift after the first switch.
        let en_alone = prefix(Some(EN), Task::Transcribe, false);
        let fr_alone = prefix(Some(FR), Task::Translate, false);
        assert_ne!(en_alone, fr_alone);

        for _ in 0..4 {
            assert_eq!(prefix(Some(EN), Task::Transcribe, false), en_alone);
            assert_eq!(prefix(Some(FR), Task::Translate, false), fr_alone);
        }
    }

    /// Mel frame count candle produces for `samples` PCM samples at 16 kHz.
    ///
    /// Mirrors `log_mel_spectrogram_` in candle-transformers 0.8.4
    /// (`src/models/whisper/audio.rs`, the `n_len` block): `n_len = samples /
    /// HOP_LENGTH`, rounded up to a multiple of `100 * CHUNK_LENGTH / 2` (1500)
    /// when it is not already one, plus one unconditional extra 1500-frame
    /// chunk. The mel vector is `n_len * n_mel` long, so `n_len` is the frame
    /// dimension `transcribe` windows over.
    ///
    /// Duplicated here on purpose: if candle changes its padding, this helper
    /// stops matching reality and the boundary table below is where the drift
    /// becomes visible.
    fn candle_mel_frames(samples: usize) -> usize {
        const PAD: usize = 100 * m::CHUNK_LENGTH / 2; // 1500
        let n_len = samples / m::HOP_LENGTH;
        // candle spells the round-up as `if n_len % pad != 0 { (n_len / pad + 1) * pad }`.
        let n_len = if n_len.is_multiple_of(PAD) {
            n_len
        } else {
            (n_len / PAD + 1) * PAD
        };
        n_len + PAD
    }

    /// Contiguity, bounds, exact coverage. Violating this *is* the D1 defect.
    fn assert_window_invariants(content_frames: usize, windows: &[(usize, usize)]) {
        let mut expected_start = 0usize;
        for &(start, len) in windows {
            assert_eq!(
                start, expected_start,
                "gap or overlap at window start for {content_frames} frames: {windows:?}"
            );
            assert!(
                (1..=m::N_FRAMES).contains(&len),
                "window len {len} out of 1..={} for {content_frames} frames",
                m::N_FRAMES
            );
            assert!(
                start < content_frames,
                "window starts at {start}, past end {content_frames}"
            );
            expected_start = start + len;
        }
        assert_eq!(
            expected_start, content_frames,
            "windows must cover exactly [0, {content_frames})"
        );
    }

    #[test]
    fn test_mel_windows_boundary_table() {
        // (samples, expected mel frames, expected window count).
        // The 240_159 / 240_160 pair is the one-sample cliff observed in prod:
        // 3000 frames fit the encoder, 4500 did not and hard-errored with
        // "narrow invalid args start + len > dim_len".
        let cases = [
            (0usize, 1_500usize, 1usize), // no samples: still one padded chunk
            (1, 1_500, 1),                // 62.5 us
            (240_159, 3_000, 1),          // 15.00994 s
            (240_160, 4_500, 2),          // 15.01000 s
            (480_000, 4_500, 2),          // 30.00 s
            (9_600_000, 61_500, 21),      // 10 min
        ];

        for (samples, expected_frames, expected_windows) in cases {
            let frames = candle_mel_frames(samples);
            assert_eq!(
                frames, expected_frames,
                "mel frames for {samples} samples changed"
            );

            let windows = mel_windows(frames);
            assert_eq!(
                windows.len(),
                expected_windows,
                "window count for {samples} samples ({frames} frames)"
            );
            assert_window_invariants(frames, &windows);
        }
    }

    #[test]
    fn test_mel_windows_invariants_over_sample_sweep() {
        // Irregular steps so the sweep lands on and between chunk boundaries
        // rather than only on multiples of 1500 frames.
        const TEN_MINUTES_SAMPLES: usize = 16_000 * 600;
        let steps = [1usize, 7, 159, 160, 161, 1_601, 24_001, 240_159, 480_001];

        let mut samples = 0usize;
        let mut i = 0usize;
        while samples <= TEN_MINUTES_SAMPLES {
            let frames = candle_mel_frames(samples);
            let windows = mel_windows(frames);

            assert!(
                !windows.is_empty(),
                "real audio always yields at least one window ({samples} samples)"
            );
            assert_window_invariants(frames, &windows);

            samples += steps[i % steps.len()];
            i += 1;
        }
    }

    #[test]
    fn test_mel_windows_invariants_over_frame_sweep() {
        // Frame counts candle's padding never produces (including 0) still have
        // to tile correctly, because `transcribe` takes whatever mel it is given.
        for frames in 0..=(m::N_FRAMES * 2 + 1) {
            assert_window_invariants(frames, &mel_windows(frames));
        }
    }

    #[test]
    fn test_mel_windows_empty_input_yields_no_windows() {
        assert!(mel_windows(0).is_empty());
    }

    #[test]
    fn test_mel_windows_no_silent_loss() {
        // Guards D2: the deleted 30 s PCM clamp must not come back as truncation
        // anywhere in the windowing. Total covered frames == content frames.
        let frame_counts = [0usize, 1, 1_500, 2_999, 3_000, 3_001, 4_500, 61_500];
        for frames in frame_counts {
            let covered: usize = mel_windows(frames).iter().map(|&(_, len)| len).sum();
            assert_eq!(covered, frames, "windows dropped frames for {frames}");
        }

        let sample_counts = [0usize, 1, 240_159, 240_160, 480_000, 1_234_567, 9_600_000];
        for samples in sample_counts {
            let frames = candle_mel_frames(samples);
            let covered: usize = mel_windows(frames).iter().map(|&(_, len)| len).sum();
            assert_eq!(
                covered, frames,
                "windows dropped frames for {samples} samples"
            );
        }
    }
}
