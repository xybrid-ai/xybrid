//! RAII handle to a loaded whisper.cpp model, and the transcription entry
//! point. Every `unsafe` block in the crate lives here, each with a `# Safety`
//! note.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use xybrid_whisper_sys::bindings as sys;

use crate::coreml;
use crate::error::{WhisperError, WhisperResult};
use crate::params::{Task, TranscribeParams, SAMPLE_RATE};
use crate::segment::Segment;

/// Shortest PCM buffer that produces a real mel frame rather than pure
/// padding. whisper's STFT window is 400 samples; below that the encoder sees
/// only padding, and decoding padding fabricates text.
const MIN_SAMPLES: usize = 400;

/// Samples in Whisper's full 30-second encoder window.
const FULL_WINDOW_SAMPLES: usize = SAMPLE_RATE as usize * 30;

/// Whether new contexts request GPU offload on this target.
///
/// Metal was 2.7–7.8x faster in end-to-end ASR benchmarks on an M1 Pro, with
/// no word-level regressions in the release cases. Other targets remain on the
/// CPU path until they are measured independently.
const DEFAULT_USE_GPU: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

fn default_context_params() -> sys::whisper_context_params {
    // SAFETY: `whisper_context_default_params` is a pure value constructor
    // with no preconditions.
    let mut params = unsafe { sys::whisper_context_default_params() };
    params.use_gpu = DEFAULT_USE_GPU;
    params
}

/// An owning handle to a whisper.cpp context and its inference state.
///
/// Dropping frees both. Transcription takes `&mut self` because whisper.cpp is
/// *not thread safe for the same context*; the borrow checker enforces that.
pub struct WhisperModel {
    ctx: NonNull<sys::whisper_context>,
    /// `None` after a Core ML failure until a new state is created.
    state: Option<NonNull<sys::whisper_state>>,
    /// The model path, kept to recreate the state (and re-find its encoder).
    path: String,
    multilingual: bool,
    last_detected_language: Option<String>,
    /// Full-window `audio_ctx` when Core ML runs the encoder, else `None`.
    coreml_audio_ctx: Option<u32>,
}

// SAFETY: the context and state own only heap memory with no thread affinity;
// on iOS the state also holds an `MLModel`, which has none either, and the
// bridge's per-thread data is not part of the handle. `Sync` is deliberately
// not implemented: `whisper_full` is not safe to call concurrently on one
// context, so `transcribe` takes `&mut self`.
unsafe impl Send for WhisperModel {}

impl WhisperModel {
    /// Load a GGML-format whisper model from disk.
    ///
    /// # Errors
    ///
    /// - [`WhisperError::InvalidInput`] if the path is not representable as a C
    ///   string (interior null byte) or is not valid UTF-8.
    /// - [`WhisperError::LoadFailed`] if whisper.cpp could not read the file —
    ///   missing, unreadable, or not a GGML whisper model.
    pub fn load(path: impl AsRef<Path>) -> WhisperResult<Self> {
        let path = path.as_ref();
        let path_str = path.to_str().ok_or_else(|| {
            WhisperError::InvalidInput(format!("model path is not valid UTF-8: {}", path.display()))
        })?;
        let c_path = CString::new(path_str)
            .map_err(|_| WhisperError::InvalidInput("model path contains a null byte".into()))?;

        let cparams = default_context_params();

        // SAFETY: `c_path` is a valid null-terminated string that outlives the
        // call, and `cparams` is a by-value POD struct. A null return is the
        // documented failure signal and is checked immediately below.
        let raw =
            unsafe { sys::whisper_init_from_file_with_params_no_state(c_path.as_ptr(), cparams) };
        let ctx = NonNull::new(raw)
            .ok_or_else(|| WhisperError::LoadFailed(path.display().to_string()))?;

        let Some((state, coreml_audio_ctx)) = new_state(ctx, path_str) else {
            // SAFETY: `ctx` is live and owned here; nothing else references it.
            unsafe { sys::whisper_free(ctx.as_ptr()) };
            return Err(WhisperError::LoadFailed(path.display().to_string()));
        };

        // SAFETY: `ctx` was just checked non-null and is a live context.
        let multilingual = unsafe { sys::whisper_is_multilingual(ctx.as_ptr()) } != 0;

        Ok(Self {
            ctx,
            state: Some(state),
            path: path_str.to_owned(),
            multilingual,
            last_detected_language: None,
            coreml_audio_ctx,
        })
    }

    /// Whether this model's vocabulary carries language tokens beyond English.
    #[must_use]
    pub fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// Return the language auto-detected by the most recent successful transcription.
    ///
    /// Returns `None` before the first transcription, when the caller supplied
    /// an explicit language, or when whisper.cpp did not report a valid
    /// language code.
    #[must_use]
    pub fn detected_language(&self) -> Option<&str> {
        self.last_detected_language.as_deref()
    }

    /// Transcribe 16 kHz mono PCM in `[-1.0, 1.0]`.
    ///
    /// Returns one [`Segment`] per text span whisper emitted. Callers wanting
    /// the plain string can [`Segment::join`] them. When `params.language` is
    /// absent, [`WhisperModel::detected_language`] exposes the selected code
    /// after this call succeeds.
    ///
    /// # Errors
    ///
    /// - [`WhisperError::AudioTooShort`] if `pcm` is shorter than one STFT
    ///   window. Decoding pure padding fabricates text, so this is rejected
    ///   rather than passed through.
    /// - [`WhisperError::UnsupportedLanguage`] if `params.language` names a
    ///   language this model has no token for.
    /// - [`WhisperError::InvalidInput`] if the language string contains a null
    ///   byte.
    /// - [`WhisperError::CoreMlAudioCtx`] if the encoder runs on Core ML and
    ///   `params.audio_ctx` asks for less than the full window.
    /// - [`WhisperError::InferenceFailed`] if `whisper_full` returned non-zero.
    /// - [`WhisperError::CoreMlEncoderFailed`] if the Core ML encoder failed on
    ///   any window. The transcript is discarded and the state rebuilt.
    /// - [`WhisperError::LoadFailed`] if a state lost to such a failure still
    ///   can't be recreated; every later call tries again.
    /// - [`WhisperError::InvalidUtf8`] if a segment's bytes were not UTF-8.
    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        params: &TranscribeParams,
    ) -> WhisperResult<Vec<Segment>> {
        // A failed or explicitly-languaged request must not expose a stale
        // auto-detection from an earlier transcription.
        self.last_detected_language = None;

        if pcm.len() < MIN_SAMPLES {
            return Err(WhisperError::AudioTooShort {
                samples: pcm.len(),
                minimum: MIN_SAMPLES,
            });
        }

        // Recreates the state if a Core ML failure left none.
        let state = self.ensure_state()?;

        // A Core ML encoder only takes the full window; refuse anything shorter.
        if let Some(supported) = self.coreml_audio_ctx {
            let requested = params.normalized_audio_ctx();
            if requested != supported {
                return Err(WhisperError::CoreMlAudioCtx {
                    requested,
                    supported,
                });
            }
        }

        // Keep the CString alive for the whole call: `whisper_full_params`
        // borrows the language pointer rather than copying it.
        let language = self.resolve_language(params.language.as_deref())?;

        // SAFETY: pure value constructor, no preconditions.
        let mut wparams = unsafe {
            sys::whisper_full_default_params(sys::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY)
        };

        wparams.n_threads = i32::try_from(params.n_threads).unwrap_or(i32::MAX).max(1);
        wparams.audio_ctx = i32::try_from(params.normalized_audio_ctx()).unwrap_or(i32::MAX);
        // Timestamp tokens let whisper.cpp advance to the true end of speech
        // inside each 30 s window. They are optional for one-window live
        // partials, but required for longer input: without them, the final
        // short window is padded to 30 s and tiny can fill that padding with a
        // long repetition loop. Segment spans remain an internal part of our
        // return type either way, so enabling them here does not change the IO
        // shape.
        wparams.no_timestamps = !params.timestamps && pcm.len() <= FULL_WINDOW_SAMPLES;
        wparams.no_context = !params.use_context;
        wparams.translate = params.task == Task::Translate;
        wparams.print_progress = false;
        wparams.print_realtime = false;
        wparams.print_special = false;
        wparams.print_timestamps = false;
        // Keep Whisper's non-speech vocabulary out of user-facing text. The
        // upstream default is false, which lets bracketed annotations such as
        // "[BLANK_AUDIO]" and "[Speaking in French]" surface as transcripts.
        // Candle applies the equivalent suppress-token mask, so enabling
        // whisper.cpp's built-in mask preserves that behavior across backends.
        wparams.suppress_nst = true;
        // Disable temperature fallback. A fallback re-runs the whole window at
        // a higher temperature, which on a live path turns one slow window into
        // several — the measured 2.7 s outlier behind an otherwise 0.7 s
        // cadence. Better to surface a poor partial than to blow the budget.
        wparams.temperature_inc = 0.0;
        wparams.language = language.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        // The encoder runs on this thread, so the counter's growth is this call's.
        let failures_before = coreml::failure_count();
        // SAFETY: `self.ctx` and its `state` are live; `pcm`'s length is passed
        // with it; `language` outlives `wparams`; `&mut self` gives the exclusive
        // access whisper.cpp requires per context.
        let code = unsafe {
            sys::whisper_full_with_state(
                self.ctx.as_ptr(),
                state.as_ptr(),
                wparams,
                pcm.as_ptr(),
                i32::try_from(pcm.len()).unwrap_or(i32::MAX),
            )
        };
        let coreml_failures = coreml::failure_count().wrapping_sub(failures_before);
        // A Core ML failure explains any accompanying error code, so it wins.
        if coreml_failures > 0 {
            // The state's text context now holds a decode of zeros and can't be
            // cleared, so replace it. Free it first: two states at once is a
            // memory spike when memory may be what failed.
            self.release_state();
            if self.ensure_state().is_err() {
                tracing::warn!(
                    model = %self.path,
                    "could not recreate the whisper state after a Core ML failure; retrying on the next call"
                );
            }
            return Err(WhisperError::CoreMlEncoderFailed {
                failures: u32::try_from(coreml_failures).unwrap_or(u32::MAX),
            });
        }
        if code != 0 {
            return Err(WhisperError::InferenceFailed { code });
        }

        let segments = Self::collect_segments(state)?;
        if params.language.is_none() {
            self.last_detected_language = Self::read_detected_language(state);
        }
        Ok(segments)
    }

    /// The live state, created (with its Core ML encoder) if none is left.
    fn ensure_state(&mut self) -> WhisperResult<NonNull<sys::whisper_state>> {
        if let Some(state) = self.state {
            return Ok(state);
        }
        let (state, coreml_audio_ctx) = new_state(self.ctx, &self.path)
            .ok_or_else(|| WhisperError::LoadFailed(self.path.clone()))?;
        self.state = Some(state);
        self.coreml_audio_ctx = coreml_audio_ctx;
        Ok(state)
    }

    /// Free the inference state, if any.
    fn release_state(&mut self) {
        if let Some(state) = self.state.take() {
            // SAFETY: the state belongs to `self.ctx`, nothing borrows it
            // (`&mut self` or Drop), and `take` prevents any reuse.
            unsafe { sys::whisper_free_state(state.as_ptr()) };
        }
        self.coreml_audio_ctx = None;
    }

    /// Read the language selected by the preceding auto-detecting inference.
    fn read_detected_language(state: NonNull<sys::whisper_state>) -> Option<String> {
        // SAFETY: `state` is live; this only reads the last transcription's result.
        let id = unsafe { sys::whisper_full_lang_id_from_state(state.as_ptr()) };
        if id < 0 {
            return None;
        }

        // SAFETY: `whisper_lang_str` returns either null or a pointer to a
        // process-lifetime, null-terminated language-code string owned by
        // whisper.cpp. We check for null before reading and copy the result.
        let language_ptr = unsafe { sys::whisper_lang_str(id) };
        if language_ptr.is_null() {
            return None;
        }
        // SAFETY: checked non-null above; whisper.cpp guarantees null
        // termination for its static language-code strings.
        unsafe { CStr::from_ptr(language_ptr) }
            .to_str()
            .ok()
            .map(str::to_owned)
    }

    /// Map a caller-supplied language code to a C string whisper accepts,
    /// rejecting codes this model has no token for.
    fn resolve_language(&self, requested: Option<&str>) -> WhisperResult<Option<CString>> {
        let Some(lang) = requested else {
            return Ok(None);
        };

        let c_lang = CString::new(lang)
            .map_err(|_| WhisperError::InvalidInput("language contains a null byte".into()))?;

        // SAFETY: `c_lang` is a valid null-terminated string. `whisper_lang_id`
        // reads it and returns -1 for an unknown code — it does not retain the
        // pointer.
        let id = unsafe { sys::whisper_lang_id(c_lang.as_ptr()) };
        if id < 0 {
            return Err(WhisperError::UnsupportedLanguage {
                requested: lang.to_string(),
            });
        }
        // An English-only model has exactly one language token; asking it for
        // anything else is a configuration error the caller can fix, and
        // whisper.cpp would otherwise silently ignore the request.
        if !self.multilingual && !lang.eq_ignore_ascii_case("en") {
            return Err(WhisperError::UnsupportedLanguage {
                requested: lang.to_string(),
            });
        }

        Ok(Some(c_lang))
    }

    /// Read back the segments produced by the last `whisper_full` call.
    fn collect_segments(state: NonNull<sys::whisper_state>) -> WhisperResult<Vec<Segment>> {
        // SAFETY: `state` is live; this reads a counter the last run set.
        let n = unsafe { sys::whisper_full_n_segments_from_state(state.as_ptr()) };
        let n = usize::try_from(n).unwrap_or(0);

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let idx = i32::try_from(i).unwrap_or(i32::MAX);

            // SAFETY: `idx` is in `0..n_segments`, so whisper returns a
            // null-terminated buffer valid until the next run; we copy it first.
            let text_ptr =
                unsafe { sys::whisper_full_get_segment_text_from_state(state.as_ptr(), idx) };
            if text_ptr.is_null() {
                return Err(WhisperError::InvalidUtf8 { index: i });
            }
            // SAFETY: checked non-null above; whisper guarantees null
            // termination.
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_str()
                .map_err(|_| WhisperError::InvalidUtf8 { index: i })?
                .to_string();

            // SAFETY: same index bounds as the text call above.
            let (t0, t1) = unsafe {
                (
                    sys::whisper_full_get_segment_t0_from_state(state.as_ptr(), idx),
                    sys::whisper_full_get_segment_t1_from_state(state.as_ptr(), idx),
                )
            };

            out.push(Segment {
                text,
                // whisper reports centiseconds; xybrid's cross-boundary time
                // unit is milliseconds (see the "no datetime crosses FFI" rule).
                start_ms: t0.saturating_mul(10),
                end_ms: t1.saturating_mul(10),
            });
        }
        Ok(out)
    }
}

/// Creates a state for `ctx`, vetting any Core ML encoder beside `model`, and
/// returns it with the encoder's full-window `audio_ctx` if Core ML runs it.
fn new_state(
    ctx: NonNull<sys::whisper_context>,
    model: &str,
) -> Option<(NonNull<sys::whisper_state>, Option<u32>)> {
    // whisper.cpp loads the encoder inside this init, on this thread.
    let coreml = coreml::LoadScope::begin(ctx, model);
    // SAFETY: `ctx` is live; a null return becomes `None` for the caller.
    let raw = unsafe { sys::whisper_init_state(ctx.as_ptr()) };
    let coreml_audio_ctx = coreml.finish();
    NonNull::new(raw).map(|state| (state, coreml_audio_ctx))
}

impl Drop for WhisperModel {
    fn drop(&mut self) {
        // The state goes first: it belongs to this context.
        self.release_state();
        // SAFETY: `self.ctx` is unfreed and solely owned (`WhisperModel` is not `Clone`).
        unsafe { sys::whisper_free(self.ctx.as_ptr()) };
    }
}

impl std::fmt::Debug for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately does not print the raw pointer: it is not actionable and
        // leaks address-space layout into logs.
        f.debug_struct("WhisperModel")
            .field("multilingual", &self.multilingual)
            .field("last_detected_language", &self.last_detected_language)
            .field("sample_rate", &SAMPLE_RATE)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::default_context_params;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_macos_requests_gpu_by_default() {
        assert!(default_context_params().use_gpu);
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn other_targets_keep_gpu_disabled_by_default() {
        assert!(!default_context_params().use_gpu);
    }
}
