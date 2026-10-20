//! Resident Phonon sessions for GGUF bundles and existing TTS delivery APIs.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use xybrid_zzz_sys::{PhononSession, PhononSettings};

use crate::execution::{TtsAudioChunk, TtsStreamResult};
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, AdapterResult, ModelRuntime};

/// Resolved model and request settings for one Phonon run.
#[derive(Clone, Debug, PartialEq)]
pub struct PhononDefaults {
    /// GGUF kept unchanged while its native session is resident.
    pub model: PathBuf,
    /// Voice from that GGUF, or the engine-selected default.
    pub voice: Option<String>,
    /// Per-utterance language; changing it does not reopen the session.
    pub language: Option<String>,
    /// Native open settings, including bundle temperature selection.
    pub settings: PhononSettings,
}

impl PhononDefaults {
    fn same_session(&self, other: &Self) -> bool {
        self.model == other.model && self.voice == other.voice && self.settings == other.settings
    }
}

#[derive(Default)]
struct State {
    pending: Option<PhononDefaults>,
    loaded: Option<PhononDefaults>,
    session: Option<PhononSession>,
    #[cfg(test)]
    opens: usize,
}

/// One reusable Phonon session, serialized independently of its cancellation handle.
#[derive(Default)]
pub struct ZzzPhononRuntime {
    state: Mutex<State>,
}

impl ZzzPhononRuntime {
    /// Create a runtime awaiting resolved bundle settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Install settings for the next load without retaining request data afterward.
    pub fn apply_defaults(&self, defaults: PhononDefaults) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).pending = Some(defaults);
    }

    /// Deliver owned audio packets while permitting concurrent cancellation.
    pub fn execute_stream(
        &self,
        input: &Envelope,
        cancelled: &(dyn Fn() -> bool + Sync),
        on_chunk: &mut dyn FnMut(TtsAudioChunk) -> bool,
    ) -> AdapterResult<TtsStreamResult> {
        if !matches!(input.kind, EnvelopeKind::Text(_)) {
            return Err(AdapterError::InvalidInput(
                "Phonon requires text input".into(),
            ));
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let language = state
            .loaded
            .as_ref()
            .and_then(|value| value.language.clone());
        let session = state
            .session
            .as_mut()
            .ok_or_else(|| AdapterError::RuntimeError("Phonon session is not loaded".into()))?;
        super::stream::stream_session(
            input,
            cancelled,
            on_chunk,
            session.cancellation_handle(),
            |text, callback| session.synthesize_stream_outcome(text, language.as_deref(), callback),
        )
    }
}

impl ModelRuntime for ZzzPhononRuntime {
    fn name(&self) -> &str {
        "zzz-phonon"
    }

    fn supported_formats(&self) -> Vec<&str> {
        vec!["gguf"]
    }

    fn load(&mut self, model_path: &Path) -> AdapterResult<()> {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        let defaults = state.pending.take().ok_or_else(|| {
            AdapterError::InvalidInput("Phonon load requires resolved bundle settings".into())
        })?;
        if defaults.model != model_path {
            return Err(AdapterError::InvalidInput(
                "Phonon model identity mismatch".into(),
            ));
        }
        defaults
            .settings
            .validate()
            .map_err(|error| AdapterError::InvalidInput(error.to_string()))?;
        if state.session.is_none()
            || !state
                .loaded
                .as_ref()
                .is_some_and(|old| old.same_session(&defaults))
        {
            // Open transactionally: a failed override leaves the prior session
            // and identity intact, ready for the caller's next valid request.
            let session = PhononSession::open_phonon(
                &defaults.model,
                defaults.voice.as_deref(),
                defaults.settings,
            )
            .map_err(|error| AdapterError::RuntimeError(error.to_string()))?;
            state.session = Some(session);
            #[cfg(test)]
            {
                state.opens += 1;
            }
        }
        state.loaded = Some(defaults);
        Ok(())
    }

    fn is_loaded(&self, model_path: &Path) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .loaded
            .as_ref()
            .is_some_and(|value| value.model == model_path)
    }

    fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope> {
        let mut pcm = Vec::new();
        let summary = self.execute_stream(input, &|| false, &mut |packet| {
            pcm.extend(packet.pcm);
            true
        })?;
        let mut output = summary.into_envelope();
        output.kind = EnvelopeKind::Audio(pcm);
        Ok(output)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zzz_phonon_requires_settings_and_text_without_touching_assets() {
        let mut runtime = ZzzPhononRuntime::new();
        assert!(matches!(
            runtime.load(Path::new("missing.gguf")),
            Err(AdapterError::InvalidInput(_))
        ));
        assert!(matches!(
            runtime.execute(&Envelope::new(EnvelopeKind::Embedding(vec![]))),
            Err(AdapterError::InvalidInput(_))
        ));
        assert!(runtime
            .execute(&Envelope::new(EnvelopeKind::Text("hello".into())))
            .is_err());
        assert!(!runtime.is_loaded(Path::new("missing.gguf")));
    }

    #[test]
    fn zzz_phonon_reuses_open_state_and_failed_override_is_transactional() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let path = PathBuf::from(home).join(".zzz/models/Phonon-7e71a02d-200.gguf");
        if !path.is_file() {
            assert_ne!(
                std::env::var("ZZZ_REQUIRE_FIXTURES").as_deref(),
                Ok("1"),
                "required Phonon runtime fixture missing"
            );
            eprintln!("skipping Phonon resident-session test: model fixture unavailable");
            return;
        }
        let mut runtime = ZzzPhononRuntime::new();
        let mut defaults = PhononDefaults {
            model: path.clone(),
            voice: None,
            language: None,
            settings: PhononSettings::default(),
        };
        runtime.apply_defaults(defaults.clone());
        runtime.load(&path).unwrap();
        defaults.language = Some("fr".into());
        runtime.apply_defaults(defaults.clone());
        runtime.load(&path).unwrap();
        assert_eq!(runtime.state.lock().unwrap().opens, 1);
        defaults.settings.seed = 0;
        runtime.apply_defaults(defaults.clone());
        runtime.load(&path).unwrap();
        assert_eq!(runtime.state.lock().unwrap().opens, 2);
        let valid = defaults.clone();
        defaults.voice = Some("voice-that-is-not-bundled".into());
        runtime.apply_defaults(defaults);
        assert!(runtime.load(&path).is_err());
        assert_eq!(runtime.state.lock().unwrap().loaded.as_ref(), Some(&valid));
        runtime.apply_defaults(valid);
        runtime.load(&path).unwrap();
        assert_eq!(runtime.state.lock().unwrap().opens, 2);
    }
}
