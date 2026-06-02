//! The safe-wrapper error surface. No `unsafe`, no FFI imports — this module
//! compiles on every target, so downstream code can spell error variants
//! even in no-bindings builds.
//!
//! Mirrors `xybrid-mlx::MlxError`'s flat `#[non_exhaustive]` shape so the
//! two backends present the same surface texture to downstream callers.

use std::error::Error;

/// Errors produced by the safe llama.cpp wrappers.
///
/// All llama-returning APIs in this crate yield [`LlamaResult`]. Variants
/// here map 1:1 onto the failure modes that `xybrid-core` previously
/// surfaced through `runtime_adapter::AdapterError::{InvalidInput,
/// RuntimeError, ModelNotLoaded}` from inside the llama path; the
/// `From<LlamaError> for AdapterError` impl in `xybrid-core` preserves that
/// outward shape.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LlamaError {
    /// Caller passed input that can't be transformed into a valid C-side
    /// payload — typically a string containing an interior null byte that
    /// `CString::new` rejects, or an empty token slice into a generation
    /// call that requires at least one prefill token.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// `llama_load_model_from_file_c` returned a null pointer for the given
    /// path. The string captures the model path the caller passed.
    #[error("failed to load model from {0}")]
    LoadFailed(String),

    /// `llama_new_context_with_model_c` returned a null pointer. The caller
    /// passed combination of `n_ctx` / `n_threads` / `n_batch` / `flash_attn`
    /// was rejected by the backend (commonly: insufficient memory, or
    /// `n_batch < n_ctx` for some quantization paths).
    #[error("failed to create llama context: {0}")]
    ContextCreationFailed(String),

    /// Tokenization (`llama_tokenize_c`) returned a negative count after the
    /// sizing-probe round-trip. Almost always a vocab / text encoding
    /// mismatch.
    #[error("tokenization failed")]
    TokenizationFailed,

    /// A decode / generation call returned a hard error code from the C
    /// wrapper. `code` preserves the raw return value (see
    /// `llama_wrapper.cpp` for the mapping: -1 invalid args, -2 sampler
    /// chain failed, -3 `llama_decode` failed, -4 input exceeded context
    /// window). `n_past_in` carries the prefix-reuse position the wrapper
    /// was called with — essential context for the KV-cache state-mismatch
    /// class of bugs (INF-162). `detail` is the human-readable mapping
    /// produced by the safe wrapper.
    #[error("llama generation failed (code {code}; n_past_in={n_past_in}): {detail}")]
    DecodeFailed {
        /// Raw return code from the C wrapper.
        code: i32,
        /// Prefix-reuse position the call was made with (0 if no prefix
        /// reuse).
        n_past_in: usize,
        /// Human-readable mapping of `code` to root cause.
        detail: String,
    },

    /// Streaming generation's per-token closure aborted by returning
    /// `Err(_)`. The boxed error preserves the original `dyn Error` so
    /// downstream callers can downcast it back — in particular
    /// `xybrid-core::abort::CloudFallbackAbort` survives the trampoline
    /// round-trip and is recovered on the other side via
    /// `cloud_fallback_reason_from_error`.
    #[error("streaming callback aborted: {0}")]
    StreamingCallbackAborted(Box<dyn Error + Send + Sync>),

    /// The model embeds a chat template, but llama.cpp could not render it.
    /// This is intentionally distinct from "no embedded template": callers
    /// may fallback when no template exists, but a present template that fails
    /// should surface as a real runtime error instead of silently switching
    /// prompt families.
    #[error("chat template render failed: {detail}")]
    ChatTemplateFailed {
        /// Human-readable failure detail from the safe wrapper.
        detail: String,
    },

    /// Catch-all for other non-zero C return codes. Carries whatever
    /// contextual string the safe wrapper could gather.
    #[error("internal llama.cpp error: {0}")]
    Internal(String),
}

/// Result alias used throughout the safe wrappers.
pub type LlamaResult<T> = Result<T, LlamaError>;
