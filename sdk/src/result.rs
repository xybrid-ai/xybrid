//! Inference result types for xybrid-sdk.
//!
//! This module provides `InferenceResult` - the output of model inference
//! with convenient accessors for different output types.

use serde::{Deserialize, Serialize};
use xybrid_core::ir::{Envelope, EnvelopeKind};

/// Output type enumeration for model inference results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Text output (ASR transcription, NLP results)
    Text,
    /// Audio output (TTS synthesis, audio processing)
    Audio,
    /// Embedding output (vector representation)
    Embedding,
    /// Unknown or custom output type
    Unknown,
}

impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputType::Text => write!(f, "text"),
            OutputType::Audio => write!(f, "audio"),
            OutputType::Embedding => write!(f, "embedding"),
            OutputType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Result from model.run() or pipeline.run().
///
/// Provides type-safe accessors for different output types with both
/// safe (Option-returning) and panicking (unwrap) variants.
///
/// # Example
///
/// ```ignore
/// let result = model.run(&envelope)?;
///
/// // Check output type
/// match result.output_type() {
///     OutputType::Text => println!("Text: {}", result.unwrap_text()),
///     OutputType::Audio => println!("Audio: {} bytes", result.unwrap_audio().len()),
///     OutputType::Embedding => println!("Embedding: {} dims", result.unwrap_embedding().len()),
///     OutputType::Unknown => println!("Unknown output"),
/// }
///
/// // Or use safe accessors
/// if let Some(text) = result.text() {
///     println!("Transcription: {}", text);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The underlying envelope containing the result
    envelope: Envelope,
    /// Inferred output type
    output_type: OutputType,
    /// Inference latency in milliseconds
    latency_ms: u32,
    /// Model ID that produced this result
    model_id: String,
}

impl InferenceResult {
    /// Create a new inference result from an envelope.
    pub fn new(envelope: Envelope, model_id: impl Into<String>, latency_ms: u32) -> Self {
        let output_type = match &envelope.kind {
            EnvelopeKind::Text(_) => OutputType::Text,
            EnvelopeKind::Audio(_) => OutputType::Audio,
            EnvelopeKind::Embedding(_) => OutputType::Embedding,
        };

        Self {
            envelope,
            output_type,
            latency_ms,
            model_id: model_id.into(),
        }
    }

    /// Create from envelope with pre-computed output type.
    pub fn with_output_type(
        envelope: Envelope,
        output_type: OutputType,
        model_id: impl Into<String>,
        latency_ms: u32,
    ) -> Self {
        Self {
            envelope,
            output_type,
            latency_ms,
            model_id: model_id.into(),
        }
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /// Get the output type of this result.
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }

    /// Get the inference latency in milliseconds.
    pub fn latency_ms(&self) -> u32 {
        self.latency_ms
    }

    /// Get the model ID that produced this result.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get a reference to the underlying envelope.
    pub fn envelope(&self) -> &Envelope {
        &self.envelope
    }

    /// Consume self and return the underlying envelope.
    pub fn into_envelope(self) -> Envelope {
        self.envelope
    }

    // ========================================================================
    // Safe Accessors (return Option)
    // ========================================================================

    /// Get text output if available.
    ///
    /// Returns `None` if the output is not text.
    pub fn text(&self) -> Option<&str> {
        match &self.envelope.kind {
            EnvelopeKind::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Get audio bytes if available.
    ///
    /// Returns `None` if the output is not audio.
    pub fn audio_bytes(&self) -> Option<&[u8]> {
        match &self.envelope.kind {
            EnvelopeKind::Audio(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Get embedding vector if available.
    ///
    /// Returns `None` if the output is not an embedding.
    pub fn embedding(&self) -> Option<&[f32]> {
        match &self.envelope.kind {
            EnvelopeKind::Embedding(vec) => Some(vec),
            _ => None,
        }
    }

    // ========================================================================
    // Unwrap Accessors (panic on wrong type)
    // ========================================================================

    /// Get text output, panicking if not text.
    ///
    /// # Panics
    ///
    /// Panics if the output type is not `Text`.
    pub fn unwrap_text(&self) -> &str {
        self.text().expect("InferenceResult is not Text type")
    }

    /// Get audio bytes, panicking if not audio.
    ///
    /// # Panics
    ///
    /// Panics if the output type is not `Audio`.
    pub fn unwrap_audio(&self) -> &[u8] {
        self.audio_bytes()
            .expect("InferenceResult is not Audio type")
    }

    /// Get embedding vector, panicking if not embedding.
    ///
    /// # Panics
    ///
    /// Panics if the output type is not `Embedding`.
    pub fn unwrap_embedding(&self) -> &[f32] {
        self.embedding()
            .expect("InferenceResult is not Embedding type")
    }

    // ========================================================================
    // Convenience Methods
    // ========================================================================

    /// Check if this result contains text.
    pub fn is_text(&self) -> bool {
        self.output_type == OutputType::Text
    }

    /// Check if this result contains audio.
    pub fn is_audio(&self) -> bool {
        self.output_type == OutputType::Audio
    }

    /// Check if this result contains an embedding.
    pub fn is_embedding(&self) -> bool {
        self.output_type == OutputType::Embedding
    }

    /// Get metadata value from the envelope.
    pub fn metadata(&self, key: &str) -> Option<&String> {
        self.envelope.metadata.get(key)
    }

    /// Get all metadata.
    pub fn all_metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.envelope.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_text_result() {
        let envelope = Envelope {
            kind: EnvelopeKind::Text("hello world".to_string()),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "test-model", 100);

        assert_eq!(result.output_type(), OutputType::Text);
        assert!(result.is_text());
        assert!(!result.is_audio());
        assert_eq!(result.text(), Some("hello world"));
        assert_eq!(result.unwrap_text(), "hello world");
        assert_eq!(result.audio_bytes(), None);
        assert_eq!(result.latency_ms(), 100);
        assert_eq!(result.model_id(), "test-model");
    }

    #[test]
    fn test_audio_result() {
        let envelope = Envelope {
            kind: EnvelopeKind::Audio(vec![1, 2, 3, 4]),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "tts-model", 50);

        assert_eq!(result.output_type(), OutputType::Audio);
        assert!(result.is_audio());
        assert!(!result.is_text());
        assert_eq!(result.audio_bytes(), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(result.text(), None);
    }

    #[test]
    fn test_embedding_result() {
        let envelope = Envelope {
            kind: EnvelopeKind::Embedding(vec![0.1, 0.2, 0.3]),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "embed-model", 25);

        assert_eq!(result.output_type(), OutputType::Embedding);
        assert!(result.is_embedding());
        assert_eq!(result.embedding(), Some(&[0.1f32, 0.2, 0.3][..]));
        assert_eq!(result.unwrap_embedding().len(), 3);
    }

    #[test]
    #[should_panic(expected = "InferenceResult is not Text type")]
    fn test_unwrap_wrong_type() {
        let envelope = Envelope {
            kind: EnvelopeKind::Audio(vec![1, 2, 3]),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "model", 0);
        result.unwrap_text(); // Should panic
    }

    #[test]
    fn test_into_envelope() {
        let envelope = Envelope {
            kind: EnvelopeKind::Text("test".to_string()),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "model", 0);
        let recovered = result.into_envelope();

        match recovered.kind {
            EnvelopeKind::Text(text) => assert_eq!(text, "test"),
            _ => panic!("Expected Text"),
        }
    }
}
