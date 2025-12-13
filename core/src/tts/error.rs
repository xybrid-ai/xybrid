//! Error types for TTS operations.

use thiserror::Error;

/// Errors that can occur during TTS operations.
#[derive(Debug, Error)]
pub enum TtsError {
    /// Model not found or not loaded.
    #[error("TTS model not found: {0}")]
    ModelNotFound(String),

    /// Dictionary not found (required for phonemization).
    #[error("Phoneme dictionary not found: {0}")]
    DictionaryNotFound(String),

    /// Phonemization failed.
    #[error("Phonemization failed: {0}")]
    PhonemizationError(String),

    /// Inference failed.
    #[error("Inference failed: {0}")]
    InferenceError(String),

    /// Invalid voice ID.
    #[error("Invalid voice: {0}")]
    InvalidVoice(String),

    /// Audio encoding/decoding error.
    #[error("Audio error: {0}")]
    AudioError(String),

    /// File I/O error.
    #[error("File error: {0}")]
    FileError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Text too long for synthesis.
    #[error("Text too long: {length} characters (max: {max})")]
    TextTooLong { length: usize, max: usize },

    /// Empty text provided.
    #[error("Empty text provided")]
    EmptyText,
}

impl From<std::io::Error> for TtsError {
    fn from(err: std::io::Error) -> Self {
        TtsError::FileError(err.to_string())
    }
}

impl From<crate::phonemizer::PhonemizeError> for TtsError {
    fn from(err: crate::phonemizer::PhonemizeError) -> Self {
        TtsError::PhonemizationError(err.to_string())
    }
}
