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

/// Initialize the llama.cpp backend and apply Xybrid's log policy once.
///
/// The `-sys` crate owns only native backend initialization. This wrapper
/// keeps the Xybrid-specific `XYBRID_LLAMACPP_VERBOSITY` env-var contract
/// in the safe wrapper crate while preserving the historical one-time
/// init timing: the env var is read during the same `Once` closure as
/// `llama_backend_init_c()`.
pub fn backend_init() {
    llama_cpp_sys::backend_init_with_configure(configure_verbosity_from_env);
}

#[cfg(feature = "bindings")]
fn configure_verbosity_from_env() {
    if let Ok(level) = std::env::var("XYBRID_LLAMACPP_VERBOSITY") {
        if let Ok(v) = level.parse::<i32>() {
            crate::log_control::set_verbosity(v);
        }
    }
}

#[cfg(not(feature = "bindings"))]
fn configure_verbosity_from_env() {}

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
#[cfg(all(feature = "bindings", feature = "vision"))]
mod vision;

#[cfg(feature = "bindings")]
pub use context::LlamaContext;
#[cfg(all(feature = "bindings", feature = "vision"))]
pub use generation::generate_from_current_logits_streaming;
#[cfg(feature = "bindings")]
pub use generation::{format_chat, generate_streaming, generate_with_stops, StreamingCallback};
#[cfg(feature = "bindings")]
pub use log_control::{get_verbosity, set_verbosity};
#[cfg(feature = "bindings")]
pub use model::LlamaModel;
#[cfg(all(feature = "bindings", feature = "vision"))]
pub use vision::{
    mtmd_helper_eval_chunks, MtmdBitmap, MtmdChunksSummary, MtmdContext, MtmdInputChunks,
};

// =========================================================================
// No-bindings stubs
// =========================================================================
//
// These keep the crate's public type surface present on a default build
// (no `bindings` feature), so `cargo build -p xybrid-llama` and
// `cargo clippy --workspace` stay green on toolchain-free CI runners.

/// Stub returned when the `bindings` feature is disabled.
#[cfg(not(feature = "bindings"))]
pub struct LlamaModel;

/// Stub returned when the `bindings` feature is disabled.
#[cfg(not(feature = "bindings"))]
pub struct LlamaContext;

#[cfg(not(feature = "bindings"))]
pub fn set_verbosity(_level: i32) {}

#[cfg(not(feature = "bindings"))]
pub fn get_verbosity() -> i32 {
    0
}
