//! whisper.cpp speech-recognition backend.
//!
//! Runs GGML-format Whisper models through the same ggml the local LLM backend
//! already links — see the `xybrid-whisper-sys` crate docs for why there must
//! only ever be one ggml in the binary. Selected by
//! [`ExecutionTemplate::GgmlWhisper`], which is why this backend needs no
//! path-sniffing heuristics: the model bundle declares it.
//!
//! # When to use this instead of Candle
//!
//! Both this and [`crate::runtime_adapter::candle`] transcribe Whisper. They
//! differ in weight format and therefore in cost. Candle runs F32 safetensors
//! and always feeds the encoder a full 30 s window; this runs quantized GGML
//! weights and can truncate the encoder to the audio actually present. On a
//! Pixel 8 that is the difference between ~3.5 s and ~0.75 s for a 5 s live
//! window. On a bandwidth-rich desktop the gap narrows to roughly 1.7x, because
//! the win is memory-bandwidth-driven rather than compute-driven.
//!
//! This backend is also available where Candle is not: the `platform-desktop`
//! preset does not enable Candle, so on Linux and Windows this is the only
//! Whisper path.
//!
//! [`ExecutionTemplate::GgmlWhisper`]: crate::execution::template::ExecutionTemplate::GgmlWhisper

mod runtime;

pub use runtime::WhisperCppRuntime;
