//! The safe-wrapper error surface. No `unsafe`, no FFI imports — this module
//! compiles on every target, so downstream code can spell error variants even
//! in no-bindings builds.
//!
//! Mirrors `xybrid_llama::LlamaError`'s flat `#[non_exhaustive]` shape so the
//! two native backends present the same texture to `xybrid-core`.

/// Errors produced by the safe whisper.cpp wrappers.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WhisperError {
    /// Caller passed input that can't be turned into a valid C-side payload —
    /// typically a path or language string containing an interior null byte
    /// that `CString::new` rejects.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// `whisper_init_from_file_with_params` returned a null pointer. The string
    /// captures the model path the caller passed.
    #[error("failed to load whisper model from {0}")]
    LoadFailed(String),

    /// The caller asked for a language the model's vocabulary has no token for.
    ///
    /// Distinct from [`WhisperError::InvalidInput`] on purpose: requesting
    /// Japanese from an English-only model is a *configuration* error the
    /// caller can fix, not a malformed string. Mirrors the same distinction the
    /// Candle Whisper adapter draws.
    #[error("unsupported language {requested:?}: the model vocabulary has no matching token")]
    UnsupportedLanguage {
        /// The language code the caller asked for.
        requested: String,
    },

    /// Audio was too short to produce even one mel frame. whisper.cpp would
    /// decode pure padding, and decoding padding invents text — so this is
    /// rejected rather than passed through.
    #[error("audio too short: {samples} samples ({:.3} s), need at least {minimum}", *samples as f64 / 16_000.0)]
    AudioTooShort {
        /// Sample count the caller supplied (16 kHz mono assumed).
        samples: usize,
        /// Minimum sample count this build requires.
        minimum: usize,
    },

    /// `whisper_full` returned a non-zero code.
    #[error("whisper inference failed (code {code})")]
    InferenceFailed {
        /// Raw return value from `whisper_full`.
        code: i32,
    },

    /// A segment's text was not valid UTF-8. whisper.cpp emits UTF-8, so this
    /// indicates a truncated multi-byte sequence at a segment boundary rather
    /// than caller error.
    #[error("segment {index} contained invalid UTF-8")]
    InvalidUtf8 {
        /// Zero-based segment index.
        index: usize,
    },

    /// The crate was built without the `bindings` feature, so no native
    /// whisper.cpp is linked.
    #[error("whisper.cpp support not compiled in (enable the `bindings` feature)")]
    NotCompiledIn,
}

/// Result type for the safe whisper.cpp wrappers.
pub type WhisperResult<T> = Result<T, WhisperError>;
