//! Text-to-Speech (TTS) module.
//!
//! This module provides a unified interface for text-to-speech synthesis,
//! supporting both local on-device models (KittenTTS) and cloud providers.
//!
//! ## Features
//!
//! - **Local inference**: On-device TTS with KittenTTS (private, low-latency)
//! - **Multiple voices**: 8 voices (4 male, 4 female)
//! - **Audio postprocessing**: Loudness normalization, silence trimming, high-pass filter
//! - **Phonemization**: CMU dictionary-based English phonemization
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::tts::{Tts, SynthesisRequest, Voice};
//!
//! // Create TTS engine
//! let tts = Tts::new()?;
//!
//! // Simple synthesis
//! let audio = tts.synthesize("Hello, world!")?;
//!
//! // With options
//! let audio = tts.synthesize_with(
//!     SynthesisRequest::new("Hello!")
//!         .with_voice(Voice::Female1)
//!         .with_speed(1.0)
//! )?;
//!
//! // Save to file
//! audio.save_wav("output.wav")?;
//! ```

mod engine;
mod error;
mod request;
mod response;
pub mod voice_embedding;

pub use engine::{Tts, TtsConfig};
pub use error::TtsError;
pub use request::{SynthesisRequest, Voice};
pub use response::{AudioOutput, AudioFormat};
pub use voice_embedding::{VoiceEmbeddingLoader, VoiceError, VoiceFormat, DEFAULT_EMBEDDING_DIM};
