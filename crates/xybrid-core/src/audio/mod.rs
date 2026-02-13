//! Audio processing module for xybrid-core (v0.0.7).
//!
//! This module provides type-safe audio handling with format detection,
//! conversion utilities, and integration with the Envelope IR.
//!
//! ## Features
//!
//! - `AudioFormat` enum for type-safe format specification
//! - `AudioEnvelope` struct with rich audio metadata
//! - WAV file parsing and validation
//! - Audio format conversion (PCM normalization, resampling)
//! - Integration with pipeline Envelope system
//!
//! ## Example
//!
//! ```rust,ignore
//! use xybrid_core::audio::{AudioEnvelope, AudioFormat};
//!
//! // Create from WAV bytes
//! let wav_bytes = std::fs::read("audio.wav").unwrap();
//! let audio = AudioEnvelope::from_wav(&wav_bytes).unwrap();
//! println!("Duration: {}ms, Sample Rate: {}Hz", audio.duration_ms(), audio.sample_rate);
//!
//! // Convert to pipeline Envelope
//! let envelope = audio.to_envelope();
//! ```
//!
//! ## Implementation Notes
//!
//! The WAV parser is a lightweight custom implementation to minimize dependencies.
//! For broader format support (24-bit PCM, extensible format, metadata chunks),
//! consider the [`hound`](https://crates.io/crates/hound) crate as a future enhancement.

mod convert;
mod envelope;
mod format;
pub mod mel;
pub mod vad;

pub use convert::{
    decode_wav_audio, f32_to_pcm16, normalize_pcm16_to_f32, prepare_audio_samples, resample_audio,
    samples_to_wav, ResampleMethod,
};
pub use envelope::{AudioEnvelope, AudioEnvelopeError};
pub use format::{AudioFormat, AudioFormatError};

// New unified mel spectrogram API
pub use mel::{compute_mel_spectrogram, MelConfig, MelScale, PaddingMode};

// Legacy re-exports for backwards compatibility
pub use mel::whisper::{compute_whisper_mel, WhisperMelConfig};

// Voice Activity Detection
pub use vad::{
    SimpleVad, SpeechSegment, VadConfig, VadError, VadFrame, VadResult, VadSampleRate, VadSession,
    VadSessionState,
};
