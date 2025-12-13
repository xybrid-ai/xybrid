//! Preprocessing steps for the execution pipeline.
//!
//! This module provides thin step wrappers that integrate core audio/image
//! processing implementations with the execution pipeline. Each step handles
//! the envelope interface and delegates to the core implementation.
//!
//! ## Architecture
//!
//! ```text
//! template_executor
//!     └── preprocessing step (this module)
//!             └── core implementation (audio/, image/, etc.)
//! ```
//!
//! ## Available Steps
//!
//! - `mel_spectrogram`: Audio → Mel spectrogram conversion for ASR models

pub mod mel_spectrogram;

pub use mel_spectrogram::{
    MelSpectrogramStep,
    MelSpectrogramConfig,
    audio_to_whisper_mel,
    audio_bytes_to_whisper_mel,
};
