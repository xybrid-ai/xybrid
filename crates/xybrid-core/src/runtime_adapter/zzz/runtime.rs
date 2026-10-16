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
    KittenSession, KittenSettings, SYNTHESIS_CHANNELS, SYNTHESIS_SAMPLE_RATE, ZZZ_EMBED_ABI_VERSION,
};

use crate::audio::f32_to_pcm16;
use crate::execution::{TtsAudioChunk, TtsStatus, TtsStreamResult};
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, AdapterResult, ModelRuntime};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

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
/// synthesis on one session, so a `Mutex` serializes synthesis. Cancellation
/// uses a separate lifetime-safe handle without taking that mutex.
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
        *self.pending.lock().unwrap_or_else(|e| e.into_inner()) = Some(defaults);
    }

    /// Open (or reuse) the session, keyed on the resolved defaults.
    ///
    /// Reopening per path change keeps single ownership over the engine
    /// session: it holds GGUF mappings and activation buffers, not reusable
    /// graphs, so a real path change has nothing to preserve.
    fn open(&mut self, defaults: &ZzzDefaults, primary: &Path) -> AdapterResult<()> {
        if self
            .loaded_identity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            == Some(defaults)
            && self
                .session
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
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
        *self.session.lock().unwrap_or_else(|e| e.into_inner()) = Some(session);
        *self
            .loaded_identity
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(defaults.clone());
        let _ = primary;
        Ok(())
    }
}

impl ZzzKittenRuntime {
    /// Deliver packets during synthesis. The engine owns all text splitting.
    pub fn execute_stream(
        &self,
        input: &Envelope,
        cancelled: &(dyn Fn() -> bool + Sync),
        on_chunk: &mut dyn FnMut(TtsAudioChunk) -> bool,
    ) -> AdapterResult<TtsStreamResult> {
        let EnvelopeKind::Text(text) = &input.kind else {
            return Err(AdapterError::InvalidInput(
                "zzz TTS requires text input".into(),
            ));
        };
        let defaults = self
            .loaded_identity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| AdapterError::RuntimeError("zzz engine session is not loaded".into()))?;
        let mut guard = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let session = guard
            .as_mut()
            .ok_or_else(|| AdapterError::RuntimeError("zzz engine session is not loaded".into()))?;
        let mut summary = TtsStreamResult {
            status: TtsStatus::Cancelled,
            sample_rate: SYNTHESIS_SAMPLE_RATE,
            channels: SYNTHESIS_CHANNELS,
            samples: 0,
            chunks: 0,
            limited_chunks: 0,
        };
        if cancelled() {
            return Ok(summary);
        }
        let cancel = session.cancellation_handle();
        let finished = AtomicBool::new(false);
        // The watcher never takes the session/model lock. Repeated cancel covers
        // native's idle-cancel/start race; the scoped join precedes session close.
        let mut delivered_end = 0;
        let outcome = std::thread::scope(|scope| {
            scope.spawn(|| {
                while !finished.load(Ordering::Acquire) {
                    if cancelled() {
                        cancel.cancel();
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            });
            struct FinishOnDrop<'a>(&'a AtomicBool);
            impl Drop for FinishOnDrop<'_> {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Release);
                }
            }
            let finish_guard = FinishOnDrop(&finished);
            let result = session.synthesize_stream_outcome(
                text,
                defaults.language.as_deref(),
                &mut |samples, info| {
                    if cancelled() {
                        cancel.cancel();
                        return false;
                    }
                    // Copy/convert before the native borrowed buffer expires. No
                    // trim, normalization or fade is applied to delivery packets.
                    summary.sample_rate = info.sample_rate;
                    summary.channels = info.channels;
                    delivered_end =
                        info.first_sample + samples.len() as u64 / u64::from(info.channels.max(1));
                    let pcm = f32_to_pcm16(samples);
                    let keep_going = on_chunk(TtsAudioChunk {
                        pcm,
                        sample_rate: info.sample_rate,
                        channels: info.channels,
                        first_sample: info.first_sample,
                    });
                    if !keep_going || cancelled() {
                        cancel.cancel();
                        return false;
                    }
                    true
                },
            );
            drop(finish_guard);
            result
        })
        .map_err(|error| AdapterError::RuntimeError(error.to_string()))?;
        // Native return is authoritative. A new predicate poll here could
        // turn completed/limited speech into cancellation after synthesis ends.
        summary.status = match outcome.status {
            xybrid_zzz_sys::SynthesisStatus::Completed => TtsStatus::Completed,
            xybrid_zzz_sys::SynthesisStatus::Cancelled => TtsStatus::Cancelled,
            xybrid_zzz_sys::SynthesisStatus::Limited => TtsStatus::Limited,
        };
        if outcome.result.sample_rate != 0 {
            summary.sample_rate = outcome.result.sample_rate;
        }
        if outcome.result.channels != 0 {
            summary.channels = outcome.result.channels;
        }
        summary.samples = outcome.result.samples.max(delivered_end);
        summary.chunks = outcome.result.chunks;
        summary.limited_chunks = outcome.result.limited_chunks;
        Ok(summary)
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
        let Some(mut defaults) = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        else {
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
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|defaults| defaults.language_model == model_path)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Collect native packets once, retaining explicit terminal metadata.
    fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope> {
        let mut pcm = Vec::new();
        let summary = self.execute_stream(input, &|| false, &mut |packet| {
            pcm.extend(packet.pcm);
            true
        })?;
        let mut result = summary.into_envelope();
        result.kind = EnvelopeKind::Audio(pcm);
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
