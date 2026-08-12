//! RAII handle to a loaded whisper.cpp model, and the transcription entry
//! point. Every `unsafe` block in the crate lives here, each with a `# Safety`
//! note.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use xybrid_whisper_sys::bindings as sys;

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

/// An owning handle to a whisper.cpp context.
///
/// Dropping frees the native context. Transcription takes `&mut self` because
/// whisper.cpp documents `whisper_full` as *not thread safe for the same
/// context* — the borrow checker enforces that for us without a lock.
pub struct WhisperModel {
    ctx: NonNull<sys::whisper_context>,
    multilingual: bool,
}

// SAFETY: a `whisper_context` owns only heap allocations reachable through the
// pointer; nothing in it is bound to the creating thread (no TLS, no thread
// handles). Moving the handle between threads is therefore sound. `Sync` is
// deliberately NOT implemented: `whisper_full` mutates context-internal state,
// and upstream documents it as unsafe to call concurrently on one context. The
// `&mut self` receiver on [`WhisperModel::transcribe`] plus the absence of
// `Sync` means shared concurrent use cannot be spelled.
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
        let raw = unsafe { sys::whisper_init_from_file_with_params(c_path.as_ptr(), cparams) };

        let ctx = NonNull::new(raw)
            .ok_or_else(|| WhisperError::LoadFailed(path.display().to_string()))?;

        // SAFETY: `ctx` was just checked non-null and is a live context.
        let multilingual = unsafe { sys::whisper_is_multilingual(ctx.as_ptr()) } != 0;

        Ok(Self { ctx, multilingual })
    }

    /// Whether this model's vocabulary carries language tokens beyond English.
    #[must_use]
    pub fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// Transcribe 16 kHz mono PCM in `[-1.0, 1.0]`.
    ///
    /// Returns one [`Segment`] per text span whisper emitted. Callers wanting
    /// the plain string can [`Segment::join`] them.
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
    /// - [`WhisperError::InferenceFailed`] if `whisper_full` returned non-zero.
    /// - [`WhisperError::InvalidUtf8`] if a segment's bytes were not UTF-8.
    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        params: &TranscribeParams,
    ) -> WhisperResult<Vec<Segment>> {
        if pcm.len() < MIN_SAMPLES {
            return Err(WhisperError::AudioTooShort {
                samples: pcm.len(),
                minimum: MIN_SAMPLES,
            });
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

        // SAFETY: `self.ctx` is live for the duration; `pcm` is a valid slice
        // whose length is passed alongside it; `wparams` is a by-value POD
        // whose `language` pointer is kept alive by `language` above. Exclusive
        // access is guaranteed by the `&mut self` receiver, which is what makes
        // this sound given upstream's "not thread safe for same context".
        let code = unsafe {
            sys::whisper_full(
                self.ctx.as_ptr(),
                wparams,
                pcm.as_ptr(),
                i32::try_from(pcm.len()).unwrap_or(i32::MAX),
            )
        };
        if code != 0 {
            return Err(WhisperError::InferenceFailed { code });
        }

        self.collect_segments()
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
    fn collect_segments(&self) -> WhisperResult<Vec<Segment>> {
        // SAFETY: `self.ctx` is live; the call reads a counter set by the
        // preceding `whisper_full`.
        let n = unsafe { sys::whisper_full_n_segments(self.ctx.as_ptr()) };
        let n = usize::try_from(n).unwrap_or(0);

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let idx = i32::try_from(i).unwrap_or(i32::MAX);

            // SAFETY: `idx` is in `0..n_segments`, so whisper returns a pointer
            // to a null-terminated buffer it owns. The buffer stays valid until
            // the next `whisper_full` on this context; we copy out of it before
            // returning, and `&self` prevents an intervening call.
            let text_ptr = unsafe { sys::whisper_full_get_segment_text(self.ctx.as_ptr(), idx) };
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
                    sys::whisper_full_get_segment_t0(self.ctx.as_ptr(), idx),
                    sys::whisper_full_get_segment_t1(self.ctx.as_ptr(), idx),
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

impl Drop for WhisperModel {
    fn drop(&mut self) {
        // SAFETY: `self.ctx` was produced by `whisper_init_*` and has not been
        // freed — `WhisperModel` is the sole owner and is not `Clone`.
        unsafe { sys::whisper_free(self.ctx.as_ptr()) };
    }
}

impl std::fmt::Debug for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately does not print the raw pointer: it is not actionable and
        // leaks address-space layout into logs.
        f.debug_struct("WhisperModel")
            .field("multilingual", &self.multilingual)
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
