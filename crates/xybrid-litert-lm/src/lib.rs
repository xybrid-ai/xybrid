//! Safe Rust bindings to Google's [LiteRT-LM] runtime through its C API.
//!
//! LiteRT-LM runs `.litertlm` language models on CPU and GPU. This crate
//! wraps the part of its C API that chat needs: reading what a model file
//! supports, loading an [`Engine`], and streaming a [`Conversation`] turn.
//!
//! # Nothing is linked at build time
//!
//! The runtime is a shared library opened at run time ([`Library`]). A build
//! that includes this crate starts normally on a device without the library;
//! [`Library::open_default`] then returns an error for which
//! [`Error::is_runtime_unavailable`] is true. The library carries its own
//! copies of abseil, protobuf and a Rust runtime, which is why it is never
//! linked statically into Xybrid.
//!
//! # Pinned interface
//!
//! The declarations in [`sys`] are written by hand against C API 1.0.0, which
//! ships with LiteRT-LM v0.19.1. [`Library`] refuses a library that reports
//! another major version. The C API version says nothing about the runtime
//! release behind it; qualifying a release is the embedder's job.
//!
//! # Threading
//!
//! Upstream's rule is that calls on one handle must not overlap. Here that is
//! the borrow checker's job: everything that uses a conversation takes
//! `&mut self`, and the engine handle is taken under an internal lock. Chunks
//! arrive on runtime threads and are handed over through a queue, so no user
//! code ever runs on them ([`Stream`]).
//!
//! # Testing without the runtime
//!
//! The `test-util` feature adds `testing::FakeRuntime`, a scripted runtime
//! that lives in the calling process. It needs no model and no native
//! library, and it can play what the real runtime rarely does: a stream that
//! never ends, a wait that fails, a callback after everything was freed.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), xybrid_litert_lm::Error> {
//! use std::time::Duration;
//! use xybrid_litert_lm::{
//!     ConversationConfig, Engine, EngineSettings, Library, Sampling, TurnOptions,
//! };
//!
//! let library = Library::open_default()?;
//! let engine = Engine::load(library, &EngineSettings::new("model.litertlm"))?;
//! let mut conversation =
//!     engine.create_conversation(&ConversationConfig::new(Sampling::greedy()))?;
//!
//! let mut stream = conversation.send_stream(
//!     r#"{"role":"user","content":"Hello"}"#,
//!     &TurnOptions::default(),
//! )?;
//! while let Some(chunk) = stream.next_chunk(Duration::from_secs(60)) {
//!     if let Some(json) = chunk.text {
//!         println!("{json}");
//!     }
//! }
//! stream.finish(Duration::from_secs(30))?;
//! # Ok(())
//! # }
//! ```
//!
//! [LiteRT-LM]: https://github.com/google-ai-edge/LiteRT-LM

mod conversation;
mod engine;
mod error;
mod library;
mod stream;
pub mod sys;
#[cfg(feature = "test-util")]
pub mod testing;

pub use conversation::{
    Conversation, ConversationConfig, RepetitionPenalty, Sampling, TurnOptions, Usage,
};
pub use engine::{ActivationPrecision, Backend, Engine, EngineSettings, ModelInfo};
pub use error::{Error, Result, StatusCode};
pub use library::{default_library_name, Library, LogLevel, C_API_MAJOR, LIBRARY_ENV};
pub use stream::{Chunk, Stream, StreamEnd, DEFAULT_GRACE};
