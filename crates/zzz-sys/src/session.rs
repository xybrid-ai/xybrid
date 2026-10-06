//! Safe session surface over the C ABI.
//!
//! Every `unsafe` block in this module carries a `# Safety` comment tied to
//! the header's ownership contract: the library owns all session memory and
//! releases it on close; one synthesis at a time per session; borrowed audio
//! is valid only until callback return; paths need only live through open.

use std::ffi::CString;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;

use crate::{
    AudioCallback, EmbedError, EmbedOptions, EmbedResult, KittenOptions, ZZZ_EMBED_CANCELLED,
    ZZZ_EMBED_INVALID_ARGUMENT, ZZZ_EMBED_LIMIT_REACHED, ZZZ_EMBED_LOAD_FAILED,
    ZZZ_EMBED_MODEL_NOT_COMPILED, ZZZ_EMBED_OK, ZZZ_EMBED_OUT_OF_MEMORY,
    ZZZ_EMBED_SYNTHESIS_FAILED,
};

/// Output format of the pinned slices, per `zzz_embed.h` ("currently 24 kHz
/// mono"). The ABI also carries both values per call; these constants exist
/// so callers can pre-size buffers without a completed call.
pub const SYNTHESIS_SAMPLE_RATE: u32 = 24_000;
pub const SYNTHESIS_CHANNELS: u32 = 1;
/// Engine default when `KittenSettings.max_tokens` is zero.
pub const DEFAULT_MAX_TOKENS: u32 = 128;

/// Typed error surface mapped from the ABI's integer codes, carrying the
/// engine's privacy-screened message.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ZzzError {
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },
    #[error("model is not compiled into this engine slice: {message}")]
    ModelNotCompiled { message: String },
    #[error("engine load failed: {message}")]
    LoadFailed { message: String },
    #[error("engine ran out of memory: {message}")]
    OutOfMemory { message: String },
    #[error("synthesis was cancelled: {message}")]
    Cancelled { message: String },
    #[error("output limit reached, partial audio delivered: {message}")]
    LimitReached { message: String },
    #[error("synthesis failed: {message}")]
    SynthesisFailed { message: String },
    #[error("engine call failed: {message}")]
    Failed { message: String },
}

#[must_use]
fn message_of(error: &EmbedError, fallback: &str) -> ZzzError {
    let spoken = error.message();
    let message = String::from(if spoken.is_empty() { fallback } else { spoken });
    match error.code {
        ZZZ_EMBED_INVALID_ARGUMENT => ZzzError::InvalidArgument { message },
        ZZZ_EMBED_MODEL_NOT_COMPILED => ZzzError::ModelNotCompiled { message },
        ZZZ_EMBED_LOAD_FAILED => ZzzError::LoadFailed { message },
        ZZZ_EMBED_OUT_OF_MEMORY => ZzzError::OutOfMemory { message },
        ZZZ_EMBED_CANCELLED => ZzzError::Cancelled { message },
        ZZZ_EMBED_LIMIT_REACHED => ZzzError::LimitReached { message },
        ZZZ_EMBED_SYNTHESIS_FAILED => ZzzError::SynthesisFailed { message },
        code => ZzzError::Failed {
            message: format!("{message} (code {code})"),
        },
    }
}

/// The linked engine's documented defaults (4 frame/chunk budget, 256-token
/// chunk budget, temperature 0.4), read from the engine rather than carried
/// as a parallel constant table.
#[must_use]
pub fn default_options() -> EmbedOptions {
    let mut options = EmbedOptions {
        struct_size: core::mem::size_of::<EmbedOptions>() as u32,
        ..EmbedOptions::default()
    };
    // # Safety: writes exactly the one `EmbedOptions` allocated above.
    unsafe { zzz_embed_default_options(ptr::addr_of_mut!(options)) };
    options
}

extern "C" {
    fn zzz_embed_abi_version() -> u32;
    fn zzz_embed_supports_model(model: *const std::os::raw::c_char) -> i32;
    fn zzz_embed_default_options(options: *mut crate::EmbedOptions);
    fn zzz_embed_open_kitten(
        options: *const KittenOptions,
        out_session: *mut *mut core::ffi::c_void,
        error: *mut EmbedError,
    ) -> i32;
    fn zzz_embed_synthesize(
        session: *mut core::ffi::c_void,
        text: *const std::os::raw::c_char,
        language: *const std::os::raw::c_char,
        callback: Option<AudioCallback>,
        user_data: *mut core::ffi::c_void,
        result: *mut EmbedResult,
        error: *mut EmbedError,
    ) -> i32;
    fn zzz_embed_close(session: *mut core::ffi::c_void);
}

/// Runtime ABI version of the linked engine slice.
///
/// The verified slice comes from the same pinned release as these
/// declarations, so a mismatch means duplicate-engine drift. Callers may
/// `assert_eq!(crate::session::abi_version(), crate::ZZZ_EMBED_ABI_VERSION)`.
#[must_use]
pub fn abi_version() -> u32 {
    // # Safety: a plain function returning an integer; no shared state.
    unsafe { zzz_embed_abi_version() }
}

/// Whether the linked slice compiled support for `model` (e.g.
/// `"kitten-tts2"`).
#[must_use]
pub fn supports_model(model: &str) -> bool {
    let name = match CString::new(model) {
        Ok(name) => name,
        // An interior NUL can never match an implementation name.
        Err(_) => return false,
    };
    // # Safety: the C side only spans the input string and returns an int.
    unsafe { zzz_embed_supports_model(name.as_ptr()) != 0 }
}

/// The C-layout options the header's documented defaults select (4 threads;
/// 128-token cap) plus the waveform seed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KittenSettings {
    /// 1..=64; 0 selects the engine default of 4.
    pub threads: u32,
    /// 1..=1023; 0 selects the engine default of 128.
    pub max_tokens: u32,
    /// Waveform-stage seed; 0 is a valid seed, not a sentinel.
    pub seed: u64,
}

/// One loaded Kitten session. `Drop` closes it exactly once.
///
/// Not `Sync`: the header documents one synthesis at a time per session,
/// and calls must not be made re-entrantly from the audio callback.
pub struct KittenSession {
    /// Engine-owned session handle, released by `Drop` via
    /// `zzz_embed_close`. After the header's guarantees, success always
    /// yields non-null — hence `NonNull`.
    session: ptr::NonNull<core::ffi::c_void>,
}

// The engine forbids concurrent calls on one session, which callers enforce
// externally, but a session moving between threads across sequential calls
// is fine, so `Send` holds.
unsafe impl Send for KittenSession {}

impl core::fmt::Debug for KittenSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Printed address only; engine-owned state stays private.
        f.debug_struct("KittenSession")
            .field("session", &(self.session.as_ptr() as usize))
            .finish()
    }
}

/// Owned C strings backing one `open` call: per the header, paths need only
/// live through open, so they are dropped when the call returns.
struct AssetPaths {
    language_model: CString,
    decoder_model: CString,
    language_voice: CString,
    decoder_voice: CString,
}

/// NUL-terminated UTF-8 C string for the engine; an interior NUL byte would
/// truncate the path in C, so it is rejected here.
fn c_string(argument: &str) -> Result<CString, ZzzError> {
    CString::new(argument.as_bytes()).map_err(|_| ZzzError::InvalidArgument {
        message: "string contains an interior NUL byte".to_string(),
    })
}

impl KittenSession {
    /// Open a Kitten TTS 2 session from its four asset paths.
    ///
    /// The language-model voice JSON (`reference_text`, `reference_tokens`,
    /// `speaker`) and the S3 decoder voice JSON must describe the same
    /// speaker. Independent sessions can run concurrently.
    ///
    /// # Errors
    ///
    /// Propagates the engine's typed errors ([`ZzzError::LoadFailed`],
    /// [`ZzzError::InvalidArgument`], ...) with the engine's own message
    /// when it has one.
    pub fn open_kitten(
        language_model: &Path,
        decoder_model: &Path,
        language_voice: &Path,
        decoder_voice: &Path,
        settings: KittenSettings,
    ) -> Result<Self, ZzzError> {
        let as_c_path = |path: &Path| -> Result<CString, ZzzError> {
            let string = path.to_str().ok_or_else(|| ZzzError::InvalidArgument {
                message: "asset path is not valid UTF-8".to_string(),
            })?;
            c_string(string)
        };
        let paths = AssetPaths {
            language_model: as_c_path(language_model)?,
            decoder_model: as_c_path(decoder_model)?,
            language_voice: as_c_path(language_voice)?,
            decoder_voice: as_c_path(decoder_voice)?,
        };
        let options = KittenOptions {
            struct_size: core::mem::size_of::<KittenOptions>() as u32,
            threads: settings.threads,
            max_tokens: settings.max_tokens,
            reserved: 0,
            seed: settings.seed,
            // Byte pointers into the CStrings above; each stays valid
            // through the call that consumes them, per the header's note
            // that paths need only live through open.
            language_model_path: paths.language_model.as_ptr().cast(),
            decoder_model_path: paths.decoder_model.as_ptr().cast(),
            language_voice_path: paths.language_voice.as_ptr().cast(),
            decoder_voice_path: paths.decoder_voice.as_ptr().cast(),
        };

        let mut error = EmbedError::zeroed();
        let mut session: *mut core::ffi::c_void = ptr::null_mut();
        // # Safety: every argument stays valid for the duration of the call
        // (the CStrings above outlive it). `out_session` receives a
        // caller-owned handle, null on failure per the header; on success
        // the handle is wrapped in `Self`, whose `Drop` owns the single
        // close call.
        let code =
            unsafe { zzz_embed_open_kitten(ptr::addr_of!(options), &mut session, &mut error) };
        if code != ZZZ_EMBED_OK {
            return Err(message_of(&error, "opening the Kitten TTS 2 engine failed"));
        }
        let session = ptr::NonNull::new(session).ok_or_else(|| ZzzError::Failed {
            message: "engine reported success without a session".to_string(),
        })?;
        Ok(Self { session })
    }

    /// Synthesize `text`, delivering audio through a streaming callback.
    ///
    /// The callback receives borrowed sample chunks plus per-chunk delivery
    /// metadata — copy or encode them to retain. Returning `false` from the
    /// callback cancels the remaining delivery; audio delivered before the
    /// cancellation is final (the header cannot retract it) and the call
    /// reports [`ZzzError::Cancelled`]. One call at a time per session;
    /// never call session methods from inside the callback.
    ///
    /// # Errors
    ///
    /// Propagates the engine's typed errors. [`ZzzError::LimitReached`]
    /// carries partial audio already delivered; the session remains usable
    /// after cancellation or any other error.
    pub fn synthesize_stream(
        &mut self,
        text: &str,
        language: Option<&str>,
        callback: &mut dyn FnMut(&[f32], ChunkInfo) -> bool,
    ) -> Result<EmbedResult, ZzzError> {
        let text = c_string(text)?;
        let language = match language {
            Some(language) => Some(c_string(language)?),
            None => None,
        };

        // Boxed so the trampoline holds one stable heap address for the
        // synchronous call, with no lifetime the C side could outlive.
        let mut boxed: Box<dyn FnMut(&[f32], ChunkInfo) -> bool> =
            Box::new(move |samples, info| callback(samples, info));
        let mut error = EmbedError::zeroed();
        let mut result = EmbedResult {
            sample_rate: SYNTHESIS_SAMPLE_RATE,
            channels: SYNTHESIS_CHANNELS,
            ..EmbedResult::default()
        };
        // # Safety: `self.session` is engine-owned and valid for the struct's
        // lifetime; the boxed closure address is valid until the engine
        // returns; the call is synchronous and single-threaded by the
        // header's contract, which keeps the callback's borrowed samples
        // slice valid until callback return; `text` and `language` are
        // NUL-terminated strings alive for the whole call.
        let code = unsafe {
            zzz_embed_synthesize(
                self.session.as_ptr(),
                text.as_ptr().cast(),
                language.map_or(ptr::null(), |lang| lang.as_ptr().cast()),
                Some(sample_callback),
                ptr::addr_of_mut!(boxed).cast(),
                ptr::addr_of_mut!(result),
                ptr::addr_of_mut!(error),
            )
        };
        drop(boxed);
        if code != ZZZ_EMBED_OK {
            return Err(message_of(&error, "synthesis failed"));
        }
        Ok(result)
    }

    /// Synthesize, collecting every delivered sample.
    ///
    /// # Errors
    ///
    /// Cancellation in collect mode returns [`ZzzError::Cancelled`] and
    /// discards the partial audio already delivered; use
    /// [`KittenSession::synthesize_stream`] when partial audio on
    /// cancellation matters (the xybrid adapter wraps the streaming form).
    /// [`ZzzError::LimitReached`] likewise discards partial audio here.
    pub fn synthesize(
        &mut self,
        text: &str,
        language: Option<&str>,
    ) -> Result<SynthesisOutcome, ZzzError> {
        let mut samples = Vec::new();
        self.synthesize_stream(text, language, &mut |chunk, _info| {
            samples.extend_from_slice(chunk);
            true
        })
        .map(|result| SynthesisOutcome { samples, result })
    }
}

impl Drop for KittenSession {
    fn drop(&mut self) {
        // # Safety: closing exactly once is owned by Drop after the header's
        // rules — no calls from the callback and no concurrent calls. The
        // engine releases all session memory here.
        unsafe { zzz_embed_close(self.session.as_ptr()) }
    }
}

/// Per-chunk delivery metadata, mirroring the callback's C arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkInfo {
    pub sample_rate: u32,
    pub channels: u32,
    pub first_sample: u64,
}

/// What one collected synthesis produced: the delivered samples plus the
/// engine's result record (`sample_rate`, `channels`, chunk counts, and the
/// `limited_chunks` census for the partial-audio case).
#[derive(Clone, Debug, PartialEq)]
pub struct SynthesisOutcome {
    pub samples: Vec<f32>,
    pub result: EmbedResult,
}

/// # Safety (trampoline)
///
/// Runs on the synthesis thread, inside the engine's call. Only ever
/// dereferences `user_data`, which points at the boxed closure owned
/// locally by the running synthesize call, and borrows the samples slice,
/// which the header promises is valid until callback return. Returning
/// nonzero cancels the remaining delivery per the header.
unsafe extern "C" fn sample_callback(
    samples: *const f32,
    sample_count: usize,
    sample_rate: u32,
    channels: u32,
    first_sample: u64,
    user_data: *mut core::ffi::c_void,
) -> i32 {
    if samples.is_null() || user_data.is_null() {
        // A null buffer or context is an engine contract violation; cancel
        // the delivery rather than dereference.
        return 1;
    }
    let boxed: *mut Box<dyn FnMut(&[f32], ChunkInfo) -> bool> = user_data.cast();
    if boxed.is_null() {
        return 1;
    }
    // # Safety: the pointer was produced by `ptr::addr_of_mut!(callback)` in
    // the synchronous call above and is still owned by that call frame.
    let callback: &mut dyn FnMut(&[f32], ChunkInfo) -> bool = unsafe { &mut *boxed };
    let audio = std::slice::from_raw_parts(samples, sample_count);
    let info = ChunkInfo {
        sample_rate,
        channels,
        first_sample,
    };
    match catch_unwind(AssertUnwindSafe(|| callback(audio, info))) {
        Ok(true) => ZZZ_EMBED_OK,
        Ok(false) => 1, // delivery cancelled by user request
        Err(_) => 1,    // unwinding across this C boundary must not happen
    }
}
