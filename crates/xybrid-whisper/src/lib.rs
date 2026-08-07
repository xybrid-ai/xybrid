//! Safe Rust wrappers over [`xybrid-whisper-sys`].
//!
//! Owns the FFI boundary for whisper.cpp: an RAII context handle, typed errors,
//! and validated transcription parameters. Downstream code — the `xybrid-core`
//! adapter, and anything else wanting whisper.cpp without the `xybrid-core`
//! surface — only touches the safe types here.
//!
//! # Activation
//!
//! The real implementation lives behind the `bindings` cargo feature. A default
//! build compiles this crate to an empty shell on every target, which keeps
//! `cargo clippy --workspace` green on CI runners without a C++ toolchain.
//!
//! Because whisper.cpp links the ggml that llama.cpp builds (see the
//! `xybrid-whisper-sys` crate docs for why there must only ever be one),
//! enabling `bindings` transitively enables the llama.cpp native build.
//!
//! # Public surface
//!
//! - [`WhisperModel`] — owning handle to a loaded GGML whisper model
//! - [`TranscribeParams`] / [`Task`] — validated inference parameters
//! - [`Segment`] — one span of transcript with millisecond offsets
//! - [`WhisperError`] / [`WhisperResult`] — error surface
//!
//! Zero `unsafe` appears on the public surface. Every `unsafe` block lives in
//! [`mod@model`] with a `# Safety` comment, mirroring `xybrid-llama`'s
//! discipline.
//!
//! # Examples
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "bindings")] {
//! use xybrid_whisper::{Segment, TranscribeParams, WhisperModel};
//!
//! let mut model = WhisperModel::load("models/ggml-base.en-q5_1.bin")?;
//!
//! // 16 kHz mono PCM in [-1.0, 1.0].
//! let pcm: Vec<f32> = vec![0.0; 16_000 * 5];
//!
//! let params = TranscribeParams {
//!     language: Some("en".to_string()),
//!     n_threads: 4,
//!     ..Default::default()
//! };
//!
//! let segments = model.transcribe(&pcm, &params)?;
//! println!("{}", Segment::join(&segments));
//! # }
//! # Ok(())
//! # }
//! ```

// Unconditional: callers can spell error variants, build parameters, and
// pattern-match segments even in a no-bindings build.
mod error;
mod params;
mod segment;

pub use error::{WhisperError, WhisperResult};
pub use params::{Task, TranscribeParams, FULL_AUDIO_CTX, SAMPLE_RATE};
pub use segment::Segment;

#[cfg(feature = "bindings")]
mod model;
#[cfg(feature = "bindings")]
pub use model::WhisperModel;
