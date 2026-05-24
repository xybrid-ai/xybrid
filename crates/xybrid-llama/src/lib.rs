//! Safe Rust wrappers over [`llama-cpp-sys`].
//!
//! Owns the FFI boundary for llama.cpp: RAII handles, typed errors, the
//! streaming trampoline. Downstream code (the `xybrid-core` adapter, Phase
//! 2's consumers, and any future backend that wants llama-cpp without the
//! `xybrid-core` surface) only touches the safe types in this crate.
//!
//! # Activation
//!
//! The real implementation lives behind the `bindings` cargo feature. A
//! default build — `cargo build -p xybrid-llama` — compiles this crate to
//! an empty shell on every target, which keeps `cargo clippy --workspace`
//! on Linux CI runners green even without a C++ toolchain.
//!
//! # Public surface
//!
//! - [`LlamaModel`] — owning handle to a loaded GGUF model
//! - [`LlamaContext`] — owning handle to a llama context, with KV-cache
//!   manipulation methods
//! - [`SamplingParams`] — data-only sampling configuration
//! - [`StreamingCallback`] — closure type alias for streaming generation
//! - [`generate_streaming`] / [`generate_with_stops`] — the autoregressive
//!   loops, including the prefix-reuse `n_past_in` knob
//! - [`set_verbosity`] / [`get_verbosity`] — llama.cpp log-level control
//! - [`LlamaError`] / [`LlamaResult`] — error surface
//!
//! Zero `unsafe` appears on the public surface. Every `unsafe` block lives
//! in the [`mod@ffi`] module behind `pub(crate)` with `# Safety` doc
//! comments, mirroring `xybrid-mlx::ffi`'s discipline.

// Unconditional: callers can spell error variants and stub-call
// `backend_init` even in a no-bindings build.
mod error;
pub use error::{LlamaError, LlamaResult};

/// Initialize the llama.cpp backend. Re-export from `llama_cpp_sys` so
/// callers can spell `xybrid_llama::backend_init()` rather than reaching
/// into the `-sys` crate directly.
pub use llama_cpp_sys::backend_init;

#[cfg(feature = "bindings")]
pub(crate) mod ffi;

#[cfg(feature = "bindings")]
mod context;
#[cfg(feature = "bindings")]
mod generation;
#[cfg(feature = "bindings")]
mod log_control;
#[cfg(feature = "bindings")]
mod model;
#[cfg(feature = "bindings")]
mod sampling;

#[cfg(feature = "bindings")]
pub use context::LlamaContext;
#[cfg(feature = "bindings")]
pub use generation::{
    format_chat, generate_streaming, generate_with_stops, ChatMessageView, StreamingCallback,
};
#[cfg(feature = "bindings")]
pub use log_control::{get_verbosity, set_verbosity};
#[cfg(feature = "bindings")]
pub use model::LlamaModel;
#[cfg(feature = "bindings")]
pub use sampling::SamplingParams;

/// Internal hooks exposed for integration tests in `tests/`. Not part of
/// the stable public surface — opting in requires the `bindings` feature
/// and these symbols may move at any time. The `#[doc(hidden)]` keeps
/// them out of rustdoc and IDE autocomplete.
#[cfg(feature = "bindings")]
#[doc(hidden)]
pub mod __test_hooks {
    pub use crate::generation::{streaming_trampoline, StreamingContext};
}

// =========================================================================
// No-bindings stubs
// =========================================================================
//
// These let xybrid-core's `mod sys` re-export `xybrid_llama::*` and still
// compile on a default build (no `llm-llamacpp` feature). Calls bubble up
// `LlamaError::Internal` so the runtime adapter can map them to the
// pre-refactor `AdapterError::RuntimeError("llm-llamacpp feature not
// enabled")`.

/// Stub returned when the `bindings` feature is disabled.
#[cfg(not(feature = "bindings"))]
pub struct LlamaModel;

/// Stub returned when the `bindings` feature is disabled.
#[cfg(not(feature = "bindings"))]
pub struct LlamaContext;

/// Stub sampling params for no-bindings builds.
#[cfg(not(feature = "bindings"))]
#[derive(Clone, Default)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
}

#[cfg(not(feature = "bindings"))]
pub fn set_verbosity(_level: i32) {}

#[cfg(not(feature = "bindings"))]
pub fn get_verbosity() -> i32 {
    0
}
