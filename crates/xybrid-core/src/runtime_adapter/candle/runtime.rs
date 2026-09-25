use super::device::{select_device, DeviceSelection};
use super::whisper::{Task, TranscribeOptions, WhisperError, WhisperModel};
use crate::audio::decode_wav_audio;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, AdapterResult, ModelRuntime};
use std::collections::HashMap;
use std::path::Path;

/// Candle-based model runtime implementation.
///
/// Manages Candle models (currently Whisper) and executes inference.
pub struct CandleRuntime {
    /// Cache of loaded models (key: model directory path)
    models: HashMap<String, WhisperModel>,
    /// Active model key (most recently loaded or used)
    active_model: Option<String>,
}

impl Default for CandleRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl CandleRuntime {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            active_model: None,
        }
    }

    fn resolve_model(&mut self, input: &Envelope) -> AdapterResult<&mut WhisperModel> {
        // If envelope specifies a model_id/path in metadata, try to use it
        // Otherwise use active_model

        let model_key = if let Some(id) = input.metadata.get("model_id") {
            // For now assuming model_id might map to our cache keys
            Some(id.clone())
        } else {
            self.active_model.clone()
        };

        let key = model_key.ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model selected and no active model".to_string())
        })?;

        // If key matches a loaded model, return it
        // Note: The key in our cache is currently the full path string
        // We might need better logical ID mapping later

        // Find best match in cache (exact match or suffix match)
        // Since we don't have the full path from just an ID sometimes
        let match_key = self
            .models
            .keys()
            .find(|k| k.ends_with(&key) || k == &&key)
            .cloned();

        if let Some(real_key) = match_key {
            self.models.get_mut(&real_key).ok_or_else(|| {
                AdapterError::ModelNotLoaded(format!("Model not found in cache: {}", real_key))
            })
        } else {
            Err(AdapterError::ModelNotLoaded(format!(
                "Model '{}' not loaded",
                key
            )))
        }
    }
}

impl ModelRuntime for CandleRuntime {
    fn name(&self) -> &str {
        "candle"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supported_formats(&self) -> Vec<&str> {
        vec!["safetensors"]
    }

    fn is_loaded(&self, model_path: &Path) -> bool {
        // Same dir normalization as `load`, so the two always agree.
        let model_dir = if model_path.is_file() {
            model_path.parent().unwrap_or(model_path)
        } else {
            model_path
        };
        self.models
            .contains_key(model_dir.to_string_lossy().as_ref())
    }

    fn load(&mut self, model_path: &Path) -> AdapterResult<()> {
        let _path_str = model_path.to_string_lossy().to_string();

        // If path is a file (e.g. model.safetensors), use parent dir
        let model_dir = if model_path.is_file() {
            model_path.parent().unwrap_or(model_path)
        } else {
            model_path
        };
        let dir_str = model_dir.to_string_lossy().to_string();

        if self.models.contains_key(&dir_str) {
            self.active_model = Some(dir_str);
            return Ok(());
        }

        // Load model
        // Determine configuration (can infer from path or default)
        // For now, default to Tiny/English like TemplateExecutor, or read config
        //
        // iOS quirk: candle-metal's IOGPUMetalBuffer path uses
        // `MTLStorageModeManaged` (macOS-only) when wrapping the model's
        // weight tensors, which trips an `Invalid storageMode 1` assertion on
        // physical iOS devices (reproducible on iPadOS 26.4.2 / iOS 26.5).
        // Until candle-metal is fixed upstream, prefer CPU on iOS — Whisper
        // Tiny still runs comfortably in real-time on Apple Silicon CPU.
        let preference = if cfg!(target_os = "ios") {
            DeviceSelection::Cpu
        } else {
            DeviceSelection::Auto
        };
        let device = select_device(preference)
            .map_err(|e| AdapterError::RuntimeError(format!("Device selection failed: {}", e)))?;

        // Try to load
        let model = WhisperModel::load(model_dir, &device).map_err(|e| {
            AdapterError::RuntimeError(format!("Failed to load Candle model: {}", e))
        })?;

        self.models.insert(dir_str.clone(), model);
        self.active_model = Some(dir_str);

        Ok(())
    }

    fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope> {
        // Need mutable access to model (which is in self.models)
        // We can't borrow self.models mutably AND call methods on self easily if we aren't careful
        // But resolve_model takes &mut self and returns &mut WhisperModel.
        // It borrows self.models, so we can't use other parts of self.

        // Resolve model first
        // If we don't have a specific model targeting mechanism in Envelope yet, we rely on active_model
        // But active_model is stored in self.

        // To simplify, just get the active model:
        let key = self
            .active_model
            .as_ref()
            .ok_or_else(|| AdapterError::ModelNotLoaded("No active model loaded".to_string()))?
            .clone();

        let model = self.models.get_mut(&key).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Active model '{}' missing from cache", key))
        })?;

        match &input.kind {
            EnvelopeKind::Audio(bytes) => {
                let options = parse_transcribe_options(&input.metadata)?;

                // Decode audio
                // Whisper expects 16kHz mono
                let samples = decode_wav_audio(bytes, 16000, 1).map_err(|e| {
                    AdapterError::InvalidInput(format!("Audio decode failed: {}", e))
                })?;

                // Transcribe
                let text = model
                    .transcribe_pcm_with_options(&samples, &options)
                    .map_err(transcription_error)?;

                Ok(Envelope::new(EnvelopeKind::Text(text)))
            }
            EnvelopeKind::Embedding(_mel) => {
                // Assume Mel spectrogram input [1, n_mels, n_frames] flattened
                // We need to reconstruct Tensor from Vec<f32>
                // Use helper from pcm_to_mel (but that does conversion)
                // Need transcribe() method on model.
                let _tensor = model
                    .pcm_to_mel_tensor(&[]) // Hack/Stub? No, we need direct mel tensor creation
                    .map_err(|e| {
                        AdapterError::RuntimeError(format!(
                            "Failed to create tensor context: {}",
                            e
                        ))
                    })?;

                // Actually we can't easily create a tensor from generic slice without device context which is inside model.
                // We should add a method to WhisperModel to transcribe_from_mel_slice.
                // For now, fail or implement if critical. TemplateExecutor mostly does Audio -> Text.
                Err(AdapterError::InvalidInput(
                    "Direct Mel spectrogram input not fully supported in CandleRuntime yet"
                        .to_string(),
                ))
            }
            _ => Err(AdapterError::InvalidInput(
                "Candle runtime expects Audio input".to_string(),
            )),
        }
    }
}

/// Reads a metadata value, treating a missing key and a blank value alike.
///
/// The OpenAI-compatible gateway in front of this runtime stringifies its
/// request fields, so an option the client left unset arrives as `""` rather
/// than as an absent key. Blank therefore means "not specified" for every key
/// below — it is never a request for something we then have to honor or
/// reject.
fn metadata_value<'a>(metadata: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    metadata
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
}

/// Parse per-request Whisper parameters out of envelope metadata.
///
/// Honored: `language` (any code the model's vocabulary has a `<|xx|>` token
/// for — validated later, at prefix construction) and `task`
/// (`"transcribe"` / `"translate"`, case-insensitive).
///
/// Rejected with [`AdapterError::InvalidInput`], never accepted-and-ignored,
/// because each one would change the transcript a caller gets back and
/// silently not doing so is indistinguishable from working: `prompt`,
/// non-zero `temperature`, and `timestamp_granularities`.
///
/// Ignored: everything else, including the gateway's `response_format`,
/// `format`, `filename` and `model_id`, plus the bookkeeping keys
/// [`Envelope`] adds to its own metadata. These do not affect decoding.
fn parse_transcribe_options(
    metadata: &HashMap<String, String>,
) -> AdapterResult<TranscribeOptions> {
    if let Some(prompt) = metadata_value(metadata, "prompt") {
        return Err(AdapterError::InvalidInput(format!(
            "'prompt' is not supported by the Candle Whisper runtime: decoding starts from a \
             fixed forced-token prefix, so the prompt would be dropped without affecting the \
             transcript (received a non-empty value of {} bytes)",
            prompt.len()
        )));
    }

    if let Some(temperature) = metadata_value(metadata, "temperature") {
        let value: f32 = temperature.parse().map_err(|_| {
            AdapterError::InvalidInput(format!(
                "'temperature' must be a number, got {temperature:?}"
            ))
        })?;
        if value != 0.0 {
            return Err(AdapterError::InvalidInput(format!(
                "'temperature' must be 0 for the Candle Whisper runtime: decoding is greedy \
                 argmax, so a non-zero temperature is never sampled with (got {temperature:?})"
            )));
        }
    }

    if let Some(granularities) = metadata_value(metadata, "timestamp_granularities") {
        // Comma-joined by the gateway, so "," and ", " are still empty lists.
        if granularities
            .split(',')
            .any(|granularity| !granularity.trim().is_empty())
        {
            return Err(AdapterError::InvalidInput(format!(
                "'timestamp_granularities' is not supported by the Candle Whisper runtime: it \
                 decodes with <|notimestamps|> and returns plain text (got {granularities:?})"
            )));
        }
    }

    let task = match metadata_value(metadata, "task") {
        None => Task::default(),
        Some(task) => match task.to_ascii_lowercase().as_str() {
            "transcribe" => Task::Transcribe,
            "translate" => Task::Translate,
            _ => {
                return Err(AdapterError::InvalidInput(format!(
                    "'task' must be \"transcribe\" or \"translate\", got {task:?}"
                )))
            }
        },
    };

    Ok(TranscribeOptions {
        language: metadata_value(metadata, "language").map(str::to_string),
        task,
    })
}

/// Map a transcription failure onto the adapter's error surface.
///
/// An unsupported `language` is the caller's input being wrong, so it has to
/// stay [`AdapterError::InvalidInput`] all the way up — folding it into
/// [`AdapterError::InferenceFailed`] would surface upstream as a server error
/// and invite a retry that cannot succeed.
fn transcription_error(error: WhisperError) -> AdapterError {
    match error {
        WhisperError::UnsupportedLanguage { .. } => AdapterError::InvalidInput(error.to_string()),
        other => AdapterError::InferenceFailed(format!("Transcription failed: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Envelope metadata from a list of pairs.
    fn metadata(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|&(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    /// Unwrap the message of an `InvalidInput`, failing the test otherwise.
    fn invalid_input_message(result: AdapterResult<TranscribeOptions>) -> String {
        match result {
            Err(AdapterError::InvalidInput(message)) => message,
            Err(other) => panic!("expected InvalidInput, got {other:?}"),
            Ok(options) => panic!("expected InvalidInput, got Ok({options:?})"),
        }
    }

    #[test]
    fn test_parse_options_empty_metadata_uses_defaults() {
        let options = parse_transcribe_options(&metadata(&[])).expect("empty metadata is valid");
        assert_eq!(options, TranscribeOptions::default());
    }

    #[test]
    fn test_parse_options_accepts_language() {
        let options = parse_transcribe_options(&metadata(&[("language", "fr")]))
            .expect("'fr' is a valid language");
        assert_eq!(options.language.as_deref(), Some("fr"));
        assert_eq!(options.task, Task::Transcribe);
    }

    #[test]
    fn test_parse_options_translate_task_selects_translate() {
        let options = parse_transcribe_options(&metadata(&[("task", "translate")]))
            .expect("'translate' is a valid task");
        assert_eq!(options.task, Task::Translate);
    }

    #[test]
    fn test_parse_options_task_is_case_insensitive() {
        for value in ["Translate", "TRANSLATE", "  translate  "] {
            let options = parse_transcribe_options(&metadata(&[("task", value)]))
                .unwrap_or_else(|e| panic!("{value:?} should parse: {e}"));
            assert_eq!(options.task, Task::Translate, "for {value:?}");
        }
    }

    #[test]
    fn test_parse_options_language_and_task_together() {
        let options =
            parse_transcribe_options(&metadata(&[("language", "fr"), ("task", "translate")]))
                .expect("fr + translate is valid");
        assert_eq!(options.language.as_deref(), Some("fr"));
        assert_eq!(options.task, Task::Translate);
    }

    #[test]
    fn test_parse_options_rejects_unknown_task() {
        let message = invalid_input_message(parse_transcribe_options(&metadata(&[(
            "task",
            "summarize",
        )])));
        assert!(
            message.contains("task") && message.contains("summarize"),
            "error should name the parameter and the rejected value: {message}"
        );
    }

    #[test]
    fn test_parse_options_rejects_prompt() {
        let private_prompt = "PRIVATE_PROMPT_SENTINEL_7c1f";
        let message = invalid_input_message(parse_transcribe_options(&metadata(&[(
            "prompt",
            private_prompt,
        )])));
        assert!(
            message.contains("prompt"),
            "error should name the parameter: {message}"
        );
        assert!(
            message.contains(&private_prompt.len().to_string()),
            "error should report the rejected prompt's byte length: {message}"
        );
        assert!(
            !message.contains(private_prompt),
            "error must not expose the rejected prompt: {message}"
        );
    }

    #[test]
    fn test_parse_options_accepts_zero_temperature() {
        // 0 is what the gateway sends for deterministic decoding, which is
        // exactly what greedy argmax already does — nothing to reject.
        for value in ["0", "0.0", "-0.0"] {
            let options = parse_transcribe_options(&metadata(&[("temperature", value)]))
                .unwrap_or_else(|e| panic!("temperature {value:?} should be accepted: {e}"));
            assert_eq!(options, TranscribeOptions::default(), "for {value:?}");
        }
    }

    #[test]
    fn test_parse_options_rejects_non_zero_temperature() {
        let message = invalid_input_message(parse_transcribe_options(&metadata(&[(
            "temperature",
            "0.7",
        )])));
        assert!(
            message.contains("temperature") && message.contains("0.7"),
            "error should name the parameter and the rejected value: {message}"
        );
    }

    #[test]
    fn test_parse_options_rejects_unparseable_temperature() {
        let message = invalid_input_message(parse_transcribe_options(&metadata(&[(
            "temperature",
            "abc",
        )])));
        assert!(
            message.contains("temperature"),
            "error should name the parameter: {message}"
        );
    }

    #[test]
    fn test_parse_options_rejects_timestamp_granularities() {
        for value in ["word", "segment", "word,segment"] {
            let message = invalid_input_message(parse_transcribe_options(&metadata(&[(
                "timestamp_granularities",
                value,
            )])));
            assert!(
                message.contains("timestamp_granularities"),
                "error should name the parameter for {value:?}: {message}"
            );
        }
    }

    #[test]
    fn test_parse_options_blank_values_mean_unspecified() {
        // The gateway stringifies unset options into empty values; a blank
        // `prompt` is not a prompt and a lone separator is not a granularity.
        let options = parse_transcribe_options(&metadata(&[
            ("language", ""),
            ("task", "   "),
            ("prompt", ""),
            ("temperature", ""),
            ("timestamp_granularities", ","),
        ]))
        .expect("blank values are not requests for anything");
        assert_eq!(options, TranscribeOptions::default());
    }

    #[test]
    fn test_parse_options_ignores_benign_keys() {
        // These reach the runtime on every gateway request and none of them
        // changes decoding, so they must not trip the reject path.
        let options = parse_transcribe_options(&metadata(&[
            ("response_format", "json"),
            ("format", "wav"),
            ("filename", "meeting.wav"),
            ("model_id", "whisper-tiny"),
            ("xybrid.local_id", "abc-123"),
        ]))
        .expect("benign keys are ignored");
        assert_eq!(options, TranscribeOptions::default());
    }

    #[test]
    fn test_unsupported_language_maps_to_invalid_input() {
        let error = transcription_error(WhisperError::UnsupportedLanguage { byte_len: 2 });
        match error {
            AdapterError::InvalidInput(message) => assert!(
                message.contains("language") && message.contains("2 bytes"),
                "error should name the parameter and report its byte length: {message}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_other_whisper_errors_map_to_inference_failed() {
        let error = transcription_error(WhisperError::Tokenizer("boom".to_string()));
        assert!(
            matches!(error, AdapterError::InferenceFailed(_)),
            "expected InferenceFailed, got {error:?}"
        );
    }
}
