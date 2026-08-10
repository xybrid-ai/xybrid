//! The [`ModelRuntime`] implementation backed by whisper.cpp.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use xybrid_whisper::{
    Segment, Task, TranscribeParams, WhisperError as NativeError, WhisperModel, SAMPLE_RATE,
};

use crate::audio::decode_wav_audio;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, AdapterResult, ModelRuntime};

/// Default compute threads when the caller does not say.
///
/// Whisper is memory-bandwidth-bound, not compute-bound: measured on a Pixel 8,
/// going from 1 to 8 threads buys only 1.5x on the encoder and makes total time
/// *worse*. Four is the measured knee on mobile and a safe floor on desktop, so
/// it is the default here for the same reason `candle::device` caps its pool.
const DEFAULT_THREADS: u32 = 4;

/// whisper.cpp-backed model runtime.
///
/// Holds at most one loaded model. Unlike [`crate::runtime_adapter::candle`]'s
/// runtime there is no multi-model cache: a GGML whisper context reserves
/// ~150 MB of compute buffers, so keeping several resident is a memory problem
/// rather than a latency win.
pub struct WhisperCppRuntime {
    /// The loaded context.
    ///
    /// `Mutex` is here to satisfy [`ModelRuntime`]'s `Sync` bound, not to
    /// arbitrate real contention: whisper.cpp documents `whisper_full` as unsafe
    /// to call concurrently on one context, so [`WhisperModel`] is `Send` but
    /// deliberately not `Sync`. Every method here takes `&mut self`, so the lock
    /// is uncontended in practice.
    model: Mutex<Option<WhisperModel>>,
    /// Path of the loaded model, for diagnostics and reload short-circuiting.
    loaded_path: Option<String>,
    /// Defaults from the model bundle's `GgmlWhisper` template. Per-request
    /// envelope metadata overrides these.
    defaults: TranscribeParams,
}

impl Default for WhisperCppRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperCppRuntime {
    /// Create an empty runtime with the conservative defaults: auto-detect
    /// language, no encoder truncation, four threads.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: Mutex::new(None),
            loaded_path: None,
            defaults: TranscribeParams {
                n_threads: DEFAULT_THREADS,
                ..Default::default()
            },
        }
    }

    /// Apply the defaults a model bundle declared in its `GgmlWhisper`
    /// template.
    ///
    /// Notably `audio_ctx`: the executor passes the bundle's value through
    /// rather than picking one, because truncating the encoder too far makes
    /// the decoder emit repetition loops and the safe threshold differs per
    /// model (English-only models tolerate more truncation than multilingual
    /// quantized ones).
    #[must_use]
    pub fn with_defaults(
        mut self,
        language: Option<String>,
        audio_ctx: u32,
        translate: bool,
    ) -> Self {
        self.defaults.language = language;
        self.defaults.audio_ctx = audio_ctx;
        self.defaults.task = if translate {
            Task::Translate
        } else {
            Task::Transcribe
        };
        self
    }

    /// Transcribe 16 kHz mono PCM, returning the joined transcript.
    ///
    /// # Errors
    ///
    /// Propagates load and inference failures as [`AdapterError`].
    pub fn transcribe_pcm(
        &mut self,
        pcm: &[f32],
        params: &TranscribeParams,
    ) -> AdapterResult<String> {
        let segments = self.transcribe_pcm_segments(pcm, params)?;
        Ok(Segment::join(&segments))
    }

    /// Transcribe and keep the per-segment spans, which the streaming
    /// reconciler needs to decide whether a new window supersedes an older one.
    ///
    /// # Errors
    ///
    /// Propagates load and inference failures as [`AdapterError`].
    pub fn transcribe_pcm_segments(
        &mut self,
        pcm: &[f32],
        params: &TranscribeParams,
    ) -> AdapterResult<Vec<Segment>> {
        let mut guard = lock(&self.model)?;
        let model = guard.as_mut().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No whisper.cpp model loaded".to_string())
        })?;
        model.transcribe(pcm, params).map_err(native_error)
    }

    /// The parameters this runtime would use for a request carrying
    /// `metadata`, starting from the bundle's declared defaults.
    fn params_for(&self, metadata: &HashMap<String, String>) -> AdapterResult<TranscribeParams> {
        let mut params = self.defaults.clone();

        if let Some(language) = metadata_value(metadata, "language") {
            params.language = Some(language.to_string());
        }

        if let Some(audio_ctx) = metadata_value(metadata, "audio_ctx") {
            params.audio_ctx = audio_ctx.parse().map_err(|_| {
                AdapterError::InvalidInput(format!(
                    "'audio_ctx' must be a non-negative integer of mel frames, got {audio_ctx:?}"
                ))
            })?;
        }

        match metadata_value(metadata, "task") {
            None => {}
            Some("transcribe") => params.task = Task::Transcribe,
            Some("translate") => params.task = Task::Translate,
            Some(other) => {
                return Err(AdapterError::InvalidInput(format!(
                    "'task' must be 'transcribe' or 'translate', got {other:?}"
                )))
            }
        }

        // whisper.cpp supports an initial prompt, but xybrid has not exposed
        // or validated that capability yet. Reject it rather than silently
        // claiming to honour caller input; do not echo the prompt because it
        // may contain private user text.
        if metadata_value(metadata, "prompt").is_some() {
            return Err(AdapterError::InvalidInput(
                "'prompt' is not supported by the whisper.cpp runtime".to_string(),
            ));
        }

        // Rejected rather than ignored: greedy decoding with temperature
        // fallback disabled never samples, so silently accepting a temperature
        // would misreport what ran. Mirrors the Candle runtime's contract.
        if let Some(temperature) = metadata_value(metadata, "temperature") {
            let value: f32 = temperature.parse().map_err(|_| {
                AdapterError::InvalidInput(format!(
                    "'temperature' must be a number, got {temperature:?}"
                ))
            })?;
            if value != 0.0 {
                return Err(AdapterError::InvalidInput(format!(
                    "'temperature' must be 0 for the whisper.cpp runtime: decoding is greedy \
                     with temperature fallback disabled, so a non-zero temperature is never \
                     sampled with (got {temperature:?})"
                )));
            }
        }

        Ok(params)
    }
}

impl ModelRuntime for WhisperCppRuntime {
    fn name(&self) -> &str {
        "whispercpp"
    }

    fn supported_formats(&self) -> Vec<&str> {
        vec!["bin"]
    }

    fn is_loaded(&self, model_path: &Path) -> bool {
        // Same key as `load`, so the two always agree.
        let Ok(guard) = self.model.lock() else {
            // A poisoned lock means the context is unusable and the caller will
            // have to reload, so "not loaded" is the honest answer.
            return false;
        };
        self.loaded_path.as_deref() == Some(model_path.display().to_string().as_str())
            && guard.is_some()
    }

    fn load(&mut self, model_path: &Path) -> AdapterResult<()> {
        let key = model_path.display().to_string();
        let mut guard = lock(&self.model)?;
        if self.loaded_path.as_deref() == Some(key.as_str()) && guard.is_some() {
            return Ok(());
        }

        // Drop the previous context BEFORE loading the next one. A whisper
        // context holds ~150 MB of compute buffers; holding two across a model
        // swap is a peak-memory spike on mobile for no benefit.
        *guard = None;
        self.loaded_path = None;

        *guard = Some(WhisperModel::load(model_path).map_err(native_error)?);
        drop(guard);
        self.loaded_path = Some(key);
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope> {
        let EnvelopeKind::Audio(bytes) = &input.kind else {
            return Err(AdapterError::InvalidInput(
                "whisper.cpp runtime expects Audio input".to_string(),
            ));
        };

        let params = self.params_for(&input.metadata)?;

        let samples = decode_wav_audio(bytes, SAMPLE_RATE, 1)
            .map_err(|e| AdapterError::InvalidInput(format!("Audio decode failed: {e}")))?;

        let text = self.transcribe_pcm(&samples, &params)?;
        Ok(Envelope::new(EnvelopeKind::Text(text)))
    }
}

/// Take the model lock, converting poisoning into a runtime error.
///
/// A poisoned lock means a previous transcription panicked while holding the
/// context. Recovering the guard would hand back a context whose internal state
/// is whatever the panic left behind, so this reports instead — the caller can
/// reload the model to get a clean one.
fn lock(
    model: &Mutex<Option<WhisperModel>>,
) -> AdapterResult<std::sync::MutexGuard<'_, Option<WhisperModel>>> {
    model.lock().map_err(|_| {
        AdapterError::RuntimeError(
            "whisper.cpp model lock was poisoned by an earlier panic; reload the model".to_string(),
        )
    })
}

/// Read a metadata value, treating a missing key and a blank value alike.
///
/// The OpenAI-compatible gateway in front of this runtime stringifies absent
/// fields as empty strings, so `Some("")` must behave as "not supplied" rather
/// than as an explicit empty language code.
fn metadata_value<'a>(metadata: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    metadata
        .get(key)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// Map the safe wrapper's errors onto the adapter surface.
///
/// The split matters for callers: `InvalidInput` means the request can be fixed
/// and retried with different arguments, while `RuntimeError` means the model
/// or environment is at fault.
fn native_error(err: NativeError) -> AdapterError {
    match err {
        NativeError::InvalidInput(_)
        | NativeError::UnsupportedLanguage { .. }
        | NativeError::AudioTooShort { .. } => AdapterError::InvalidInput(err.to_string()),
        NativeError::LoadFailed(_) | NativeError::NotCompiledIn => {
            AdapterError::ModelNotLoaded(err.to_string())
        }
        NativeError::InferenceFailed { .. } | NativeError::InvalidUtf8 { .. } => {
            AdapterError::RuntimeError(err.to_string())
        }
        // `WhisperError` is `#[non_exhaustive]`: a future variant must map to
        // something rather than break the build, and a runtime failure is the
        // safe assumption.
        other => AdapterError::RuntimeError(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn bundle_defaults_are_used_when_metadata_is_silent() {
        let runtime = WhisperCppRuntime::new().with_defaults(Some("en".into()), 500, false);
        let params = runtime.params_for(&meta(&[])).expect("no overrides");
        assert_eq!(params.language.as_deref(), Some("en"));
        assert_eq!(params.normalized_audio_ctx(), 500);
        assert_eq!(params.task, Task::Transcribe);
    }

    #[test]
    fn request_metadata_overrides_the_bundle_language() {
        let runtime = WhisperCppRuntime::new().with_defaults(Some("en".into()), 0, false);
        let params = runtime
            .params_for(&meta(&[("language", "fr")]))
            .expect("valid override");
        assert_eq!(params.language.as_deref(), Some("fr"));
    }

    #[test]
    fn a_blank_language_is_treated_as_absent_not_as_an_empty_code() {
        // The OpenAI-compatible gateway stringifies unset fields as "".
        let runtime = WhisperCppRuntime::new().with_defaults(Some("en".into()), 0, false);
        let params = runtime
            .params_for(&meta(&[("language", "   ")]))
            .expect("blank is not an override");
        assert_eq!(params.language.as_deref(), Some("en"));
    }

    #[test]
    fn translate_task_is_accepted() {
        let runtime = WhisperCppRuntime::new();
        let params = runtime
            .params_for(&meta(&[("task", "translate")]))
            .expect("valid task");
        assert_eq!(params.task, Task::Translate);
    }

    #[test]
    fn an_unknown_task_is_rejected_rather_than_ignored() {
        let runtime = WhisperCppRuntime::new();
        let err = runtime
            .params_for(&meta(&[("task", "summarize")]))
            .expect_err("unknown task");
        assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn zero_temperature_is_accepted_and_nonzero_is_rejected() {
        let runtime = WhisperCppRuntime::new();
        assert!(runtime.params_for(&meta(&[("temperature", "0")])).is_ok());
        let err = runtime
            .params_for(&meta(&[("temperature", "0.7")]))
            .expect_err("greedy decoding never samples");
        assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn a_non_numeric_temperature_is_a_clear_input_error() {
        let runtime = WhisperCppRuntime::new();
        let err = runtime
            .params_for(&meta(&[("temperature", "hot")]))
            .expect_err("not a number");
        assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn prompt_is_rejected_without_echoing_its_contents() {
        let runtime = WhisperCppRuntime::new();
        let private_prompt = "PRIVATE_PROMPT_SENTINEL_7c1f";
        let err = runtime
            .params_for(&meta(&[("prompt", private_prompt)]))
            .expect_err("prompt handling is not implemented");

        assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
        let message = err.to_string();
        assert!(message.contains("prompt"), "{message}");
        assert!(!message.contains(private_prompt), "{message}");
    }

    #[test]
    fn executing_without_a_loaded_model_reports_model_not_loaded() {
        let mut runtime = WhisperCppRuntime::new();
        let err = runtime
            .transcribe_pcm(&[0.0; 16_000], &TranscribeParams::default())
            .expect_err("nothing loaded");
        assert!(matches!(err, AdapterError::ModelNotLoaded(_)), "{err:?}");
    }

    #[test]
    fn non_audio_input_is_rejected() {
        let mut runtime = WhisperCppRuntime::new();
        let err = runtime
            .execute(&Envelope::new(EnvelopeKind::Text("hello".into())))
            .expect_err("text is not audio");
        assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
    }
}
