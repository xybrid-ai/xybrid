//! The [`ModelRuntime`] implementation for the pinned zzz CPU engine
//! (Kitten TTS 2 profile).
//!
//! Mirrors the crate layering of the whisper.cpp and llama.cpp integrations:
//! `xybrid-zzz-sys` owns the FFI; this module is the thin adapter glue. The
//! engine is a prebuilt, privately staged slice — model weights and voice
//! JSONs stay separate files, resolved relative to the model bundle root
//! and declared in the bundle's `ZzzEmbed` template.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use xybrid_zzz_sys::{
    EmbedResult, KittenSession, KittenSettings, SYNTHESIS_CHANNELS, SYNTHESIS_SAMPLE_RATE,
    ZZZ_EMBED_ABI_VERSION,
};

use crate::audio::samples_to_wav;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, AdapterResult, ModelRuntime};

/// Default per-chunk character budget before text is split.
///
/// Mirrors the executor's ONNX TTS default. Kitten's language-model window
/// is 1024 positions, so a larger chunk risks the engine's documented
/// LIMIT_REACHED partial-audio case on ordinary sentences.
const DEFAULT_MAX_TTS_CHARS: usize = 350;

/// The bundle-declared assets a `ZzzEmbed` template resolves to: full paths
/// (base_path joined because the [`ModelRuntime`] surface has no base-path
/// access) plus the request knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct ZzzDefaults {
    /// Primary language (speech-token) GGUF.
    pub language_model: PathBuf,
    /// S3 waveform-decoder GGUF.
    pub decoder_model: PathBuf,
    /// Language-model voice JSON.
    pub language_voice: PathBuf,
    /// S3 decoder voice JSON (`zzz.kitten_s3.voice.v1`).
    pub decoder_voice: PathBuf,
    /// Forced language; the engine's PoC accepts prepared English.
    pub language: Option<String>,
    /// Threads, token cap (0 selects engine defaults) and waveform seed.
    pub settings: KittenSettings,
}

/// zzz-engine-backed runtime (Kitten TTS 2).
///
/// Holds at most one loaded session. The engine forbids concurrent
/// synthesis on one session, so the session sits in a `Mutex` that is
/// uncontended in practice: every public method takes `&mut self`, matching
/// [`crate::runtime_adapter::WhisperCppRuntime`]'s discipline.
pub struct ZzzKittenRuntime {
    session: Mutex<Option<KittenSession>>,
    /// Identity tuple backing the open session: the resolved defaults.
    loaded_identity: Mutex<Option<ZzzDefaults>>,
    /// The dispatcher-installed defaults, consumed by the next `load`.
    pending: Mutex<Option<ZzzDefaults>>,
}

impl Default for ZzzKittenRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ZzzKittenRuntime {
    /// Empty runtime awaiting a dispatcher's defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            session: Mutex::new(None),
            loaded_identity: Mutex::new(None),
            pending: Mutex::new(None),
        }
    }

    /// Install the full-path defaults the dispatcher resolved; the next
    /// `load` opens a session with them. Interior-mutability by design so a
    /// `&` downcast (the executor carries `Box<dyn ModelRuntime>`) can
    /// configure the runtime before its `&mut` trait calls.
    pub fn apply_defaults(&self, defaults: ZzzDefaults) {
        *self.pending.lock().unwrap() = Some(defaults);
    }

    /// Open (or reuse) the session, keyed on the resolved defaults.
    ///
    /// Reopening per path change keeps single ownership over the engine
    /// session: it holds GGUF mappings and activation buffers, not reusable
    /// graphs, so a real path change has nothing to preserve.
    fn open(&mut self, defaults: &ZzzDefaults, primary: &Path) -> AdapterResult<()> {
        if self.loaded_identity.lock().unwrap().as_ref() == Some(defaults)
            && self.session.lock().unwrap().is_some()
        {
            return Ok(());
        }
        let session = KittenSession::open_kitten(
            &defaults.language_model,
            &defaults.decoder_model,
            &defaults.language_voice,
            &defaults.decoder_voice,
            defaults.settings,
        )
        .map_err(|error| AdapterError::RuntimeError(error.to_string()))?;
        *self.session.lock().unwrap() = Some(session);
        *self.loaded_identity.lock().unwrap() = Some(defaults.clone());
        let _ = primary;
        Ok(())
    }
}

impl ModelRuntime for ZzzKittenRuntime {
    fn name(&self) -> &str {
        "zzz"
    }

    fn supported_formats(&self) -> Vec<&str> {
        vec!["gguf"]
    }

    /// Install the pending defaults and open the engine session for
    /// `model_path` (the bundle's primary language GGUF).
    fn load(&mut self, model_path: &Path) -> AdapterResult<()> {
        let Some(mut defaults) = self.pending.lock().unwrap().take() else {
            return Err(AdapterError::InvalidInput(
                "zzz execution requires the dispatcher to apply the bundle's \
                 ZzzEmbed defaults before load"
                    .to_string(),
            ));
        };
        if defaults.language_model.as_os_str().is_empty() {
            // `load` receives the primary file the dispatcher resolved as
            // the single-model path; keep identity in the defaults so the
            // session keying stays one tuple.
            defaults.language_model = model_path.to_path_buf();
        }
        self.open(&defaults, model_path)
    }

    fn is_loaded(&self, model_path: &Path) -> bool {
        self.loaded_identity
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|defaults| defaults.language_model == model_path)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Synthesize `EnvelopeKind::Text` into WAV audio.
    ///
    /// Long text is split with the executor's TTS chunker and the chunk
    /// audio is concatenated; the engine resets its waveform seed per
    /// utterance, keeping chunks reproducible within a build. The engine's
    /// token-limit case (`limited_chunks`) is carried on the envelope as
    /// `zzz_limited_chunks` metadata rather than dropped silently.
    fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope> {
        let text = match &input.kind {
            EnvelopeKind::Text(text) => text.clone(),
            _ => {
                return Err(AdapterError::InvalidInput(
                    "zzz TTS requires text input".to_string(),
                ))
            }
        };

        let defaults = self
            .loaded_identity
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| {
                AdapterError::RuntimeError("zzz engine session is not loaded".to_string())
            })?;
        let mut guard = self.session.lock().unwrap();
        let Some(session) = guard.as_mut() else {
            return Err(AdapterError::RuntimeError(
                "zzz engine session is not loaded".to_string(),
            ));
        };
        let max_chars = input
            .metadata
            .get("max_chunk_chars")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_TTS_CHARS);

        let chunks = if text.chars().count() <= max_chars {
            vec![text]
        } else {
            crate::execution::text_chunking::chunk_text_for_tts(&text, max_chars)
        };
        let mut all = Vec::new();
        let mut profile = EmbedResult::default();
        for chunk in chunks {
            let outcome = session
                .synthesize(&chunk, defaults.language.as_deref())
                .map_err(|error| AdapterError::RuntimeError(error.to_string()))?;
            profile = outcome.result;
            if outcome.samples.is_empty() {
                // Matching the ONNX chunking invariant: the degenerate
                // empty case never enters concatenation.
                continue;
            }
            all.extend(outcome.samples);
        }

        let wav = samples_to_wav(&all, SYNTHESIS_SAMPLE_RATE);
        let mut result = Envelope::new(EnvelopeKind::Audio(wav));
        result
            .metadata
            .insert("sample_rate".to_string(), SYNTHESIS_SAMPLE_RATE.to_string());
        result
            .metadata
            .insert("channels".to_string(), SYNTHESIS_CHANNELS.to_string());
        if profile.limited_chunks > 0 {
            result.metadata.insert(
                "zzz_limited_chunks".to_string(),
                profile.limited_chunks.to_string(),
            );
        }
        Ok(result)
    }
}

/// Guard against engine ABI drift: the pinned slice must still declare the
/// same ABI version these bindings compile against.
pub fn assert_engine_abi() -> Result<(), AdapterError> {
    let linked = xybrid_zzz_sys::abi_version();
    if linked != ZZZ_EMBED_ABI_VERSION {
        return Err(AdapterError::RuntimeError(format!(
            "linked zzz engine ABI {linked} does not match the pinned ABI {ZZZ_EMBED_ABI_VERSION}"
        )));
    }
    Ok(())
}

#[cfg(all(test, feature = "tts-zzz"))]
mod tests {
    use super::*;
    use crate::ir::Envelope;

    fn defaults_with(primary: &str) -> ZzzDefaults {
        ZzzDefaults {
            language_model: PathBuf::from(primary),
            decoder_model: PathBuf::from("decoder.gguf"),
            language_voice: PathBuf::from("lm-voice.json"),
            decoder_voice: PathBuf::from("s3-voice.json"),
            language: Some("en".to_string()),
            settings: KittenSettings::default(),
        }
    }

    fn text_envelope(text: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(text.to_string()))
    }

    #[test]
    fn load_requires_dispatcher_defaults() {
        let mut runtime = ZzzKittenRuntime::new();
        let error = runtime
            .load(Path::new("/tmp/language.gguf"))
            .expect_err("load without applied defaults must fail");
        assert!(matches!(error, AdapterError::InvalidInput(_)), "{error:?}");
    }

    #[test]
    fn execute_before_load_fails_plain_text() {
        let mut runtime = ZzzKittenRuntime::new();
        let error = runtime
            .execute(&text_envelope("hello"))
            .expect_err("execute before load must fail");
        assert!(matches!(error, AdapterError::RuntimeError(_)), "{error:?}");
    }

    #[test]
    fn execute_rejects_non_text_envelopes() {
        let mut runtime = ZzzKittenRuntime::new();
        // No session, so the text check must fire before any engine call.
        let envelope = Envelope::new(EnvelopeKind::Embedding(vec![0.0]));
        let error = runtime.execute(&envelope).expect_err("non-text");
        assert!(matches!(error, AdapterError::InvalidInput(_)), "{error:?}");
    }

    #[test]
    fn is_loaded_is_false_until_a_successful_load() {
        let mut runtime = ZzzKittenRuntime::new();
        runtime.apply_defaults(defaults_with("/tmp/nonexistent-bundle/language.gguf"));
        assert!(!runtime.is_loaded(Path::new("/tmp/nonexistent-bundle/language.gguf")));
        // A failed open (missing bundle) must not leave a stale identity.
        let error = runtime
            .load(Path::new("/tmp/nonexistent-bundle/language.gguf"))
            .expect_err("missing bundle must fail");
        assert!(matches!(error, AdapterError::RuntimeError(_)), "{error:?}");
        assert!(!runtime.is_loaded(Path::new("/tmp/nonexistent-bundle/language.gguf")));
    }
}
