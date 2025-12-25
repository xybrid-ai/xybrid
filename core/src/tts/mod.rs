//! TTS (Text-to-Speech) helper utilities.
//!
//! This module provides utilities for TTS model execution via `TemplateExecutor`.
//! TTS models are executed through the standard metadata-driven pipeline:
//!
//! ```text
//! model_metadata.json → TemplateExecutor → ONNX Runtime
//!       ↓                      ↓
//!   Phonemize preprocess    OnnxRuntime
//!   TTSAudioEncode postprocess
//! ```
//!
//! ## Voice Embedding Loading
//!
//! The `VoiceEmbeddingLoader` utility handles loading voice embeddings from
//! different formats used by TTS models:
//!
//! - **Raw binary** (KittenTTS): `voices.bin` - contiguous f32 arrays
//! - **NPZ format** (Kokoro): `voices.npz` - NumPy ZIP archives
//!
//! ## Usage
//!
//! TTS models should be executed via `TemplateExecutor`:
//!
//! ```rust,ignore
//! use xybrid_core::template_executor::TemplateExecutor;
//! use xybrid_core::execution_template::ModelMetadata;
//! use xybrid_core::ir::{Envelope, EnvelopeKind};
//!
//! let metadata: ModelMetadata = serde_json::from_str(
//!     &std::fs::read_to_string("model_metadata.json")?
//! )?;
//! let mut executor = TemplateExecutor::with_base_path("models/kitten-tts");
//! let input = Envelope::new(EnvelopeKind::Text("Hello, world!".to_string()));
//! let output = executor.execute(&metadata, &input)?;
//! ```

pub mod voice_embedding;

pub use voice_embedding::{VoiceEmbeddingLoader, VoiceError, VoiceFormat, DEFAULT_EMBEDDING_DIM};
