//! Whisper model wrapper for Candle.
//!
//! This module provides a high-level interface for running Whisper ASR
//! using the candle-transformers implementation.

use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as m, audio, Config};
use serde::Deserialize;
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

/// Decoding parameters shipped in `config.json` that candle's [`Config`] drops.
///
/// candle keeps `suppress_tokens` but not `begin_suppress_tokens`, and both are
/// properties of the trained model rather than caller-tunable settings, so they
/// are read from the same file rather than reinvented as defaults. A model
/// whose config omits the key suppresses nothing extra.
#[derive(Debug, Default, Deserialize)]
struct DecodeConfig {
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
}

/// One decoded window: its text and the two scores that decide whether to keep it.
struct SegmentDecode {
    text: String,
    /// Probability the model put on "this window is not speech", read at the
    /// first decoded position. NaN when the model has no no-speech token.
    no_speech_prob: f64,
    /// Mean log-probability of the decoded tokens — how confident the decode
    /// was, which is low when the model is fabricating.
    avg_logprob: f64,
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
    /// `<|nocaptions|>` / `<|nospeech|>`, whichever this vocabulary spells.
    /// `None` for a model that has neither, which only disables the no-speech
    /// guard — see [`is_silent_segment`].
    no_speech_token: Option<u32>,
    /// Additive `[vocab_size]` logit mask (`0.0` / `-inf`) applied at every
    /// decode step, from the model's own `suppress_tokens`.
    suppress_mask: Tensor,
    /// The same shape, from `begin_suppress_tokens`, applied only on a
    /// segment's first sampled step.
    begin_suppress_mask: Tensor,
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
        // Load configuration. Parsed twice on purpose: candle's `Config` keeps
        // `suppress_tokens` but drops `begin_suppress_tokens`, and both are
        // decoding parameters shipped with the weights rather than something a
        // caller gets to choose.
        let config_path = model_dir.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_json)?;
        let decode_config: DecodeConfig = serde_json::from_str(&config_json)?;

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

        // Whichever spelling this vocabulary uses, or neither. A model without
        // one is not broken — it just cannot be asked "was that speech?", so
        // the guard downstream stays inert instead of failing the load.
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| tokenizer.token_to_id(token));

        // Whisper's own logit filters, built once: without them greedy decoding
        // is free to emit the non-speech annotation tokens ("[", "(", "♪") the
        // model was trained to produce for music and silence, which on a
        // mostly-padding window is exactly what it does.
        let suppress_mask = suppress_mask_tensor(
            config.vocab_size,
            &config.suppress_tokens,
            // Timestamps mode has to suppress the token that turns it off.
            user_config.timestamps.then_some(no_timestamps_token),
            device,
        )?;
        let begin_suppress_mask = suppress_mask_tensor(
            config.vocab_size,
            &decode_config.begin_suppress_tokens,
            None,
            device,
        )?;

        // Resolve the load-time language now, and fail if it is not in the
        // vocabulary. Swallowing the lookup here would turn a misconfigured
        // language into silent auto-detect: every later request that names no
        // language of its own would decode as whatever Whisper guessed, with
        // nothing in the response saying the configured language was dropped.
        let language_token = user_config
            .language
            .as_deref()
            .map(|language| language_token_id(&tokenizer, language))
            .transpose()?;

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
            no_speech_token,
            suppress_mask,
            begin_suppress_mask,
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
    ///   is at most [`m::N_FRAMES`]. Longer mels must be sliced into
    ///   `N_FRAMES`-sized windows first — as [`WhisperModel::transcribe`]
    ///   does — because the encoder's positional embedding table only has
    ///   `n_audio_ctx` (`N_FRAMES / 2`) rows.
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
    /// The mel is taken at face value: every frame it carries is windowed and
    /// decoded, padding included. Prefer [`WhisperModel::transcribe_pcm`],
    /// which trims `pcm_to_mel`'s padded tail before windowing; hand a mel in
    /// directly only if it is already cut to the frames that carry audio.
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
        self.transcribe_mel(mel, &prefix)
    }

    /// Window, encode and decode `mel` against an already-resolved prefix.
    ///
    /// Split out from [`WhisperModel::transcribe_with_options`] so callers that
    /// have to resolve the prefix early — [`WhisperModel::transcribe_pcm_with_options`]
    /// validates the request's language before computing a mel that may be
    /// minutes long — do not resolve it twice.
    ///
    /// Every frame handed in is transcribed: the caller decides what counts as
    /// content. From PCM that decision is [`trimmed_mel_frames`]; a mel passed
    /// in directly is taken at face value.
    ///
    /// A window the model reports as speechless and low-confidence is dropped
    /// rather than transcribed — see [`is_silent_segment`]. Every window can be
    /// dropped, which is an empty transcript, not an error.
    fn transcribe_mel(&mut self, mel: &Tensor, prefix: &[u32]) -> WhisperResult<String> {
        let (_, _, content_frames) = mel.dims3()?;

        let mut segments: Vec<String> = Vec::new();
        for (start, len) in mel_windows(content_frames) {
            let mel_segment = mel.narrow(2, start, len)?;

            // The encoder's blocks are self-attention only (`xa: None`), so the
            // flush flag is inert there and recomputed per call; `true` matches
            // candle's own whisper example.
            let audio_features = self.model.encoder.forward(&mel_segment, true)?;

            let decoded = self.decode_segment(&audio_features, prefix)?;
            if is_silent_segment(decoded.no_speech_prob, decoded.avg_logprob) {
                continue;
            }

            let text = decoded.text.trim();
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
            None => fallback_language_token(self.language_token, options.task),
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
    fn resolve_language_token(&self, language: &str) -> WhisperResult<u32> {
        language_token_id(&self.tokenizer, language)
    }

    /// Greedily decode one encoder window into text and its two quality scores.
    ///
    /// Starts from `prefix`, the same forced-token prefix for every window, so
    /// each window decodes independently. Mirrors the decode loop in candle's
    /// whisper example: same logit masking, same point at which the no-speech
    /// probability is read, same average-log-probability arithmetic — so
    /// [`m::NO_SPEECH_THRESHOLD`] and [`m::LOGPROB_THRESHOLD`] mean here what
    /// they mean there.
    fn decode_segment(
        &mut self,
        audio_features: &Tensor,
        prefix: &[u32],
    ) -> WhisperResult<SegmentDecode> {
        let mut tokens = prefix.to_vec();
        let mut sum_logprob = 0f64;
        // NaN when this model has no no-speech token: every comparison against
        // it is false, so the guard stays out of the way.
        let mut no_speech_prob = f64::NAN;

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

            // "Did this window contain speech at all?" is asked once, of the
            // *first* position of the first step — the slot right after
            // `<|startoftranscript|>`, which is where Whisper was trained to
            // put `<|nocaptions|>` for silence. Read from the raw logits: the
            // suppress mask below sets that token to -inf, and masking it
            // before measuring it would answer "no speech" with probability 0
            // every time.
            if i == 0 {
                if let Some(no_speech_token) = self.no_speech_token {
                    let first_logits = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                    no_speech_prob = f64::from(
                        softmax(&first_logits, 0)?
                            .i(no_speech_token as usize)?
                            .to_scalar::<f32>()?,
                    );
                }
            }

            // Get logits for last position
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((.., seq_len - 1.., ..))?)?
                .i(0)?
                .i(0)?;

            let logits = logits.broadcast_add(&self.suppress_mask)?;
            // `begin_suppress_tokens` (a leading space, and end-of-text) is a
            // rule about how a transcript may *start*, so it applies to the
            // first sampled token only. A window with nothing to say still ends
            // up empty — it decodes one token and then stops — but the no-speech
            // guard, not an immediate end-of-text, is what discards it.
            let logits = if i == 0 {
                logits.broadcast_add(&self.begin_suppress_mask)?
            } else {
                logits
            };

            // Greedy decoding: take argmax
            let next_token = logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            let prob = f64::from(
                softmax(&logits, candle_core::D::Minus1)?
                    .i(next_token as usize)?
                    .to_scalar::<f32>()?,
            );

            // The end-of-text token is pushed before the loop breaks, and the
            // average below divides by the whole token vector, prefix included.
            // Both match candle's example: the thresholds are calibrated
            // against that denominator, and a tighter one would shift what
            // counts as a low-confidence segment.
            tokens.push(next_token);
            if next_token == self.eot_token || tokens.len() > self.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| WhisperError::Tokenizer(format!("Tokenizer decode error: {}", e)))?;

        Ok(SegmentDecode {
            text,
            no_speech_prob,
            avg_logprob: sum_logprob / tokens.len() as f64,
        })
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
    /// returned mel is candle's, padding and all: it may exceed the encoder's
    /// [`m::N_FRAMES`] context, and its tail may be padding past the last real
    /// frame. [`WhisperModel::transcribe`] takes a mel at face value — it
    /// windows and decodes every frame it is given, padding included, and a
    /// window that is entirely padding decodes to fabricated text. Prefer
    /// [`WhisperModel::transcribe_pcm`], which trims the padded tail before
    /// windowing; feed a mel back into `transcribe` directly only if it is
    /// already cut to the frames that carry audio.
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
    /// The mel is trimmed to [`trimmed_mel_frames`] before windowing, so no
    /// encoder window is made entirely of `pcm_to_mel`'s padding — see that
    /// function for why an all-padding window is not merely wasted work.
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
    /// Transcribed text (empty when the audio is shorter than one mel frame)
    pub(crate) fn transcribe_pcm_with_options(
        &mut self,
        pcm_data: &[f32],
        options: &TranscribeOptions,
    ) -> WhisperResult<String> {
        // Resolve the prefix first: it is what validates `options.language`,
        // and a rejected request should not first pay for the mel of a file
        // that may be minutes long.
        let prefix = self.decode_prefix(options)?;

        // candle's own floor: `pcm_to_mel` derives its pre-padding frame count
        // the same way, so this is the number of frames that carry audio.
        let content_frames = pcm_data.len() / m::HOP_LENGTH;
        if content_frames == 0 {
            // Under one hop of audio: the mel would be padding end to end, and
            // decoding padding invents text. Nothing to transcribe.
            return Ok(String::new());
        }

        let mel = self.pcm_to_mel_tensor(pcm_data)?;
        let (_, _, mel_frames) = mel.dims3()?;
        let trimmed = trimmed_mel_frames(content_frames, mel_frames);
        let mel = if trimmed < mel_frames {
            mel.narrow(2, 0, trimmed)?
        } else {
            mel
        };

        self.transcribe_mel(&mel, &prefix)
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

/// Additive logit mask values: `0.0` for every token, `-inf` for suppressed ones.
///
/// Adding this to a step's logits before the argmax is how Whisper's reference
/// decoder forbids a token: `-inf` survives the softmax as probability 0, so
/// greedy sampling can never pick it and the remaining distribution is
/// unchanged. `extra` is folded in for the one suppression that depends on
/// runtime configuration rather than on the model file (timestamps mode has to
/// suppress `<|notimestamps|>`).
///
/// Ids at or past `vocab_size` are ignored rather than an error: the mask has
/// to be exactly as wide as the logits, and a config listing an out-of-range id
/// is the config's problem, not something to fail a model load over.
fn suppress_mask_values(vocab_size: usize, suppressed: &[u32], extra: Option<u32>) -> Vec<f32> {
    let mut mask = vec![0f32; vocab_size];
    for &token in suppressed.iter().chain(extra.iter()) {
        if let Some(slot) = mask.get_mut(token as usize) {
            *slot = f32::NEG_INFINITY;
        }
    }
    mask
}

/// [`suppress_mask_values`] as a `[vocab_size]` tensor on `device`.
fn suppress_mask_tensor(
    vocab_size: usize,
    suppressed: &[u32],
    extra: Option<u32>,
    device: &Device,
) -> WhisperResult<Tensor> {
    let mask = suppress_mask_values(vocab_size, suppressed, extra);
    Tensor::new(mask.as_slice(), device).map_err(WhisperError::from)
}

/// Whether a decoded window is silence the model talked over, and so is dropped.
///
/// Both halves are required, which is what keeps it from eating real speech: a
/// window has to *both* look like non-speech to the model and have decoded with
/// low confidence. Real speech that happens to score one of the two — a
/// confident decode of a quiet passage, or a hesitant decode of clear speech —
/// is kept.
///
/// A NaN `no_speech_prob` (this model has no no-speech token) compares false,
/// so the guard is simply inert rather than dropping everything.
///
/// Thresholds are candle's own constants, which are OpenAI's: not tuned here,
/// so they stay comparable to every other Whisper implementation.
fn is_silent_segment(no_speech_prob: f64, avg_logprob: f64) -> bool {
    no_speech_prob > m::NO_SPEECH_THRESHOLD && avg_logprob < m::LOGPROB_THRESHOLD
}

/// The language token to force when a request names no language of its own.
///
/// Transcribe keeps the load-time language: a request that says nothing wants
/// the model's configured default, and naming the source language is what makes
/// Whisper transcribe it rather than guess.
///
/// Translate drops it. The task is "render this audio in English", and the
/// language token names the *source*, not the target — forcing the load-time
/// `<|en|>` onto translate would assert the audio is already English, which is
/// exactly what a translate request says it is not. The OpenAI-compatible
/// `/audio/translations` endpoint in front of this runtime rejects `language`
/// outright, so *every* request through it arrives here with `None` and would
/// otherwise be told its French audio is English. `None` means auto-detect,
/// which is what translation without a declared source language means.
///
/// Deliberate trade: this also drops a load-time language that was configured
/// *explicitly* (e.g. `WhisperConfig { language: Some("fr"), task: Translate }`
/// falls back to auto-detect rather than forcing `<|fr|>`). The load-time
/// config cannot distinguish "defaulted to en" from "explicitly en", and
/// forcing the default onto translate is the worse failure; a translate caller
/// that knows the source language states it per request.
fn fallback_language_token(load_time: Option<u32>, task: Task) -> Option<u32> {
    match task {
        Task::Transcribe => load_time,
        Task::Translate => None,
    }
}

/// How many of a `pcm_to_mel` output's frames to keep before windowing.
///
/// `content_frames` is the pre-padding frame count — `pcm_samples /
/// HOP_LENGTH`, candle's own floor — and `mel_frames` is what `pcm_to_mel`
/// actually returned. The result rounds the content up to a whole number of
/// [`m::N_FRAMES`] encoder windows, capped by what exists.
///
/// The padding contract makes this necessary rather than merely tidy.
/// `pcm_to_mel` rounds the frame count up to a multiple of 1500 and then
/// appends one more 1500-frame chunk unconditionally, so whenever that multiple
/// is odd the 3000-frame tiling ends on a window that is 100% zero padding.
/// Whisper is trained on 30 s windows whose *tail* is padded, so a padded tail
/// is in-distribution — but a window with no real frame at all is not, and
/// greedy decoding does not return empty for it: it fabricates text
/// (whisper-tiny emits bracketed non-speech annotations like "[Opening]"). Half
/// of all durations land there — [15.01 s, 30.00 s], [45.01, 60.00], and so on
/// — so this is a routine input, not an edge case.
///
/// Every kept window therefore starts before `content_frames` and contains real
/// audio. Rounding *up* rather than down is what keeps that from costing
/// anything: no real frame is ever dropped.
///
/// Holding a real frame is necessary but not sufficient: a window that is 99%
/// padding (0.3 s of speech in a 30 s window) can still read as non-speech and
/// fabricate. Trimming is only the first of three defenses — the suppress mask
/// takes the annotation tokens off the table entirely, and [`is_silent_segment`]
/// discards what the model itself reports as speechless.
///
/// Zero content frames yields zero (`div_ceil(0, N) * N == 0`), i.e. audio
/// shorter than one hop keeps nothing and transcribes to nothing.
pub(crate) fn trimmed_mel_frames(content_frames: usize, mel_frames: usize) -> usize {
    usize::min(
        content_frames.div_ceil(m::N_FRAMES) * m::N_FRAMES,
        mel_frames,
    )
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
/// returns an empty transcript rather than encoding an empty tensor. The PCM
/// path never gets here with zero: [`WhisperModel::transcribe_pcm_with_options`]
/// returns early on sub-hop audio, and anything longer trims to at least one
/// full window. It is reachable only when a mel tensor is handed in directly.
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

/// Look up the `<|xx|>` token for a language code.
///
/// Whisper spells these tokens in lowercase, so the code is lowercased before
/// lookup; `"FR"` and `"fr"` both resolve to `<|fr|>`. An unrecognized code is
/// rejected rather than silently dropped — decoding without a language token
/// makes Whisper auto-detect, which would look like the request succeeded while
/// ignoring what it asked for.
fn language_token_id(tokenizer: &Tokenizer, language: &str) -> WhisperResult<u32> {
    let token = format!("<|{}|>", language.trim().to_ascii_lowercase());
    tokenizer
        .token_to_id(&token)
        .ok_or_else(|| WhisperError::UnsupportedLanguage(language.to_string()))
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

    #[test]
    fn test_fallback_language_transcribe_uses_load_time_language() {
        // Nothing named per request: transcribe means "in the configured
        // language", and a model loaded without one still auto-detects.
        assert_eq!(
            fallback_language_token(Some(EN), Task::Transcribe),
            Some(EN)
        );
        assert_eq!(fallback_language_token(None, Task::Transcribe), None);
    }

    #[test]
    fn test_fallback_language_translate_auto_detects_source() {
        // The F2 regression. The language token names the *source* audio, so
        // falling back to the load-time <|en|> told Whisper that French audio
        // was English — and /audio/translations rejects `language`, so every
        // request through it arrives with None and hit exactly this path.
        assert_eq!(fallback_language_token(Some(EN), Task::Translate), None);
        assert_eq!(fallback_language_token(None, Task::Translate), None);
    }

    #[test]
    fn test_suppress_mask_blocks_only_the_listed_tokens() {
        let mask = suppress_mask_values(8, &[1, 5], None);
        assert_eq!(mask.len(), 8, "the mask has to be as wide as the logits");
        for (token, value) in mask.iter().enumerate() {
            if token == 1 || token == 5 {
                assert_eq!(
                    *value,
                    f32::NEG_INFINITY,
                    "token {token} should be suppressed"
                );
            } else {
                // Exactly zero, not merely small: the mask is *added* to the
                // logits, so any other value would reweight allowed tokens.
                assert_eq!(*value, 0.0, "token {token} should be untouched");
            }
        }
    }

    #[test]
    fn test_suppress_mask_folds_in_the_extra_token() {
        let mask = suppress_mask_values(4, &[0], Some(3));
        assert_eq!(mask, vec![f32::NEG_INFINITY, 0.0, 0.0, f32::NEG_INFINITY]);

        // No extra: the same list, nothing else touched.
        let mask = suppress_mask_values(4, &[0], None);
        assert_eq!(mask, vec![f32::NEG_INFINITY, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_suppress_mask_ignores_out_of_range_ids() {
        // A config listing an id past the vocabulary must not panic or widen
        // the mask — the tensor has to line up with the logits either way.
        let mask = suppress_mask_values(3, &[1, 3, 99], None);
        assert_eq!(mask, vec![0.0, f32::NEG_INFINITY, 0.0]);

        assert!(suppress_mask_values(0, &[0], Some(1)).is_empty());
    }

    #[test]
    fn test_is_silent_segment_needs_both_halves() {
        // (no_speech_prob, avg_logprob, dropped?) around candle's 0.6 / -1.0.
        let cases = [
            (0.9, -2.0, true),   // no speech and unconfident: fabricated text
            (0.9, -0.5, false),  // reads as non-speech but decoded confidently
            (0.3, -2.0, false),  // unconfident, but the model heard speech
            (0.3, -0.5, false),  // ordinary speech
            (0.6, -2.0, false),  // exactly at the threshold is not past it
            (0.61, -1.0, false), // ditto for the log-probability bound
            (0.61, -1.01, true), // just past both
        ];

        for (no_speech_prob, avg_logprob, expected) in cases {
            assert_eq!(
                is_silent_segment(no_speech_prob, avg_logprob),
                expected,
                "no_speech_prob {no_speech_prob}, avg_logprob {avg_logprob}"
            );
        }
    }

    #[test]
    fn test_is_silent_segment_is_inert_without_a_no_speech_token() {
        // NaN stands for "this model cannot tell us" — it must not drop
        // segments, however low the confidence.
        assert!(!is_silent_segment(f64::NAN, -5.0));
        assert!(!is_silent_segment(f64::NAN, 0.0));
    }

    #[test]
    fn test_decode_config_defaults_to_no_begin_suppression() {
        // Not every config.json carries the key; missing means "suppress
        // nothing extra", not a failed load.
        let config: DecodeConfig = serde_json::from_str("{}").expect("empty config parses");
        assert!(config.begin_suppress_tokens.is_empty());

        let config: DecodeConfig =
            serde_json::from_str(r#"{"begin_suppress_tokens": [220, 50257]}"#)
                .expect("whisper-tiny's value parses");
        assert_eq!(config.begin_suppress_tokens, vec![220, 50257]);
    }

    #[test]
    fn test_trimmed_mel_frames_is_capped_by_the_mel_it_is_given() {
        // Rounding up never over-reads: a mel shorter than the round-up (not
        // something candle's padding produces, but `narrow` would error on it)
        // is kept whole.
        assert_eq!(trimmed_mel_frames(2_000, 2_500), 2_500);
        assert_eq!(trimmed_mel_frames(2_000, 3_000), 3_000);
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
    /// This is a local reimplementation checked against local constants, so it
    /// cannot detect candle changing its own padding — it would drift in
    /// lockstep with nothing. What it pins is *this file's* understanding of
    /// candle 0.8.4's arithmetic, which is what [`trimmed_mel_frames`] is
    /// designed against. Real drift is caught by the integration tests, which
    /// call the actual `pcm_to_mel`.
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
    fn test_mel_trimming_boundary_table() {
        // (samples, padded frames from candle, trimmed frames, window count).
        //
        // Each row derived from: content = samples / 160 (HOP_LENGTH);
        // padded = roundup1500(content) + 1500; trimmed = min(roundup3000(content), padded).
        //
        // The 240_159 / 240_160 pair is the one-sample cliff observed in prod:
        // 3000 frames fit the encoder, 4500 did not and hard-errored with
        // "narrow invalid args start + len > dim_len".
        //
        // The trimmed column is the F1 fix. Four of these rows used to window
        // over the padded count and hand a 100%-padding window to the decoder,
        // which fabricated text from it.
        let cases = [
            // content 0 -> padded 0+1500; trimmed min(0, 1500) = 0: nothing to decode.
            (0usize, 1_500usize, 0usize, 0usize),
            // content 0 (0.0625 frames floors to 0) -> same as above.
            (1, 1_500, 0, 0),
            // content 1 -> padded 1500+1500; trimmed min(3000, 3000) = 3000.
            (160, 3_000, 3_000, 1),
            // content 1500 (15.00994 s) -> padded 1500+1500; trimmed min(3000, 3000).
            (240_159, 3_000, 3_000, 1),
            // content 1501 (15.01 s) -> padded 3000+1500 = 4500; trimmed min(3000, 4500)
            // = 3000. Was 2 windows, the second one all padding.
            (240_160, 4_500, 3_000, 1),
            // content 3000 (30.00 s) -> padded 3000+1500 = 4500; trimmed 3000.
            // Was 2 windows, the second one all padding.
            (480_000, 4_500, 3_000, 1),
            // content 6600 (66 s) -> padded 7500+1500 = 9000; trimmed min(9000, 9000).
            (1_056_000, 9_000, 9_000, 3),
            // content 60000 (10 min) -> padded 60000+1500 = 61500; trimmed min(60000, 61500)
            // = 60000. Was 21 windows, the 21st all padding.
            (9_600_000, 61_500, 60_000, 20),
        ];

        for (samples, expected_padded, expected_trimmed, expected_windows) in cases {
            let content_frames = samples / m::HOP_LENGTH;
            let padded = candle_mel_frames(samples);
            assert_eq!(
                padded, expected_padded,
                "mel frames for {samples} samples changed"
            );

            let trimmed = trimmed_mel_frames(content_frames, padded);
            assert_eq!(
                trimmed, expected_trimmed,
                "trimmed frames for {samples} samples ({padded} padded)"
            );

            let windows = mel_windows(trimmed);
            assert_eq!(
                windows.len(),
                expected_windows,
                "window count for {samples} samples ({trimmed} trimmed frames)"
            );
            assert_window_invariants(trimmed, &windows);
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
            let content_frames = samples / m::HOP_LENGTH;
            let padded = candle_mel_frames(samples);
            let trimmed = trimmed_mel_frames(content_frames, padded);
            let windows = mel_windows(trimmed);

            assert!(
                trimmed <= padded,
                "trimmed {trimmed} exceeds what the mel holds ({padded}) for {samples} samples"
            );
            // Rounding content *up* to a window multiple can only ever be
            // capped away by `padded`, and roundup3000(f) <= roundup1500(f) +
            // 1500 = padded always holds, so the cap never bites and no real
            // frame is dropped. This is the no-silent-truncation half of the
            // fix; the window-start assertion below is the no-hallucination
            // half.
            assert!(
                trimmed >= content_frames,
                "trimmed {trimmed} drops real content ({content_frames} frames) for {samples} samples"
            );

            if content_frames == 0 {
                assert!(
                    windows.is_empty(),
                    "sub-hop audio has no real frames to decode ({samples} samples)"
                );
            } else {
                assert!(
                    !windows.is_empty(),
                    "real audio always yields at least one window ({samples} samples)"
                );
                for &(start, _) in &windows {
                    assert!(
                        start < content_frames,
                        "window at {start} begins past the last real frame \
                         ({content_frames}) for {samples} samples: it would be all padding"
                    );
                }
            }
            assert_window_invariants(trimmed, &windows);

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

        // From PCM the same has to hold after trimming: the windows cover every
        // trimmed frame, and the trim itself keeps every real one.
        let sample_counts = [0usize, 1, 240_159, 240_160, 480_000, 1_234_567, 9_600_000];
        for samples in sample_counts {
            let content_frames = samples / m::HOP_LENGTH;
            let trimmed = trimmed_mel_frames(content_frames, candle_mel_frames(samples));
            let covered: usize = mel_windows(trimmed).iter().map(|&(_, len)| len).sum();
            assert_eq!(
                covered, trimmed,
                "windows dropped frames for {samples} samples"
            );
            assert!(
                covered >= content_frames,
                "{covered} covered frames lose real audio ({content_frames} content frames) \
                 for {samples} samples"
            );
        }
    }
}
