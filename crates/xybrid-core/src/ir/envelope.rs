//! Envelope IR - Typed payload container for pipeline data flow.
//!
//! The Envelope is the Intermediate Representation (IR) that defines how data
//! flows between pipeline stages. It encapsulates typed payloads such as audio,
//! text, or embeddings, along with metadata for routing and telemetry.
//!
//! # Serialization
//!
//! Envelopes are serialized using `bincode` for efficient binary encoding.
//! They can be stored or streamed between local processes or over HTTP to
//! cloud endpoints, maintaining consistent encoding regardless of runtime backend.
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::ir::{Envelope, EnvelopeKind};
//! use std::collections::HashMap;
//!
//! // Create an audio envelope
//! let mut metadata = HashMap::new();
//! metadata.insert("sample_rate".to_string(), "16000".to_string());
//! let envelope = Envelope {
//!     kind: EnvelopeKind::Audio(vec![0u8; 1024]),
//!     metadata,
//! };
//!
//! // Serialize to bytes
//! let bytes = envelope.to_bytes().unwrap();
//!
//! // Deserialize from bytes
//! let deserialized = Envelope::from_bytes(&bytes).unwrap();
//! ```

use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

/// Typed payload variants for envelope data.
///
/// Each variant represents a different data type that can flow through
/// the pipeline stages.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EnvelopeKind {
    /// Raw audio data (PCM samples, WAV bytes, etc.)
    Audio(Vec<u8>),
    /// Text data (transcriptions, LLM outputs, etc.)
    Text(String),
    /// Embedding vectors (feature vectors, embeddings, etc.)
    Embedding(Vec<f32>),
}

impl EnvelopeKind {
    /// Returns a string representation of the envelope kind.
    ///
    /// # Returns
    ///
    /// A string describing the variant (e.g., "Audio", "Text", "Embedding")
    pub fn as_str(&self) -> &'static str {
        match self {
            EnvelopeKind::Audio(_) => "Audio",
            EnvelopeKind::Text(_) => "Text",
            EnvelopeKind::Embedding(_) => "Embedding",
        }
    }

    /// Returns the size of the payload in bytes (approximate).
    ///
    /// For Audio, returns the length of the byte vector.
    /// For Text, returns the byte length of the string.
    /// For Embedding, returns the byte length of the float vector.
    pub fn payload_size(&self) -> usize {
        match self {
            EnvelopeKind::Audio(data) => data.len(),
            EnvelopeKind::Text(data) => data.len(),
            EnvelopeKind::Embedding(data) => data.len() * std::mem::size_of::<f32>(),
        }
    }
}

/// Data payload envelope containing inference inputs/outputs.
///
/// Envelopes are the primary data structure for passing data between
/// pipeline stages. They encapsulate typed payloads and metadata for
/// routing, telemetry, and processing hints.
///
/// # Serialization
///
/// Envelopes can be serialized to binary format using `bincode` for
/// efficient transmission and storage.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Envelope {
    /// The typed payload data
    pub kind: EnvelopeKind,
    /// Metadata key-value pairs for routing, telemetry, and processing hints
    pub metadata: HashMap<String, String>,
}

impl Envelope {
    /// Metadata key for storing the local unique ID.
    pub const LOCAL_ID_METADATA_KEY: &'static str = "xybrid.local_id";

    /// Creates a new envelope with the specified kind and empty metadata.
    ///
    /// A unique local ID is automatically generated for tracking and
    /// duplicate detection.
    ///
    /// # Arguments
    ///
    /// * `kind` - The envelope kind (Audio, Text, or Embedding)
    ///
    /// # Returns
    ///
    /// A new `Envelope` instance with a unique local ID
    pub fn new(kind: EnvelopeKind) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert(
            Self::LOCAL_ID_METADATA_KEY.to_string(),
            Uuid::new_v4().to_string(),
        );
        Self { kind, metadata }
    }

    /// Creates a new envelope with the specified kind and metadata.
    ///
    /// If the metadata does not contain a local ID, one is automatically generated.
    ///
    /// # Arguments
    ///
    /// * `kind` - The envelope kind (Audio, Text, or Embedding)
    /// * `metadata` - Metadata key-value pairs
    ///
    /// # Returns
    ///
    /// A new `Envelope` instance with a unique local ID
    pub fn with_metadata(kind: EnvelopeKind, mut metadata: HashMap<String, String>) -> Self {
        // Ensure a local ID exists
        if !metadata.contains_key(Self::LOCAL_ID_METADATA_KEY) {
            metadata.insert(
                Self::LOCAL_ID_METADATA_KEY.to_string(),
                Uuid::new_v4().to_string(),
            );
        }
        Self { kind, metadata }
    }

    /// Returns the unique local ID of this envelope.
    ///
    /// Each envelope gets a UUID on creation for tracking and duplicate detection.
    ///
    /// # Returns
    ///
    /// The local ID string, or an empty string if somehow missing
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind};
    ///
    /// let e1 = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
    /// let e2 = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
    ///
    /// // Each envelope has a unique ID even with identical content
    /// assert_ne!(e1.local_id(), e2.local_id());
    /// ```
    pub fn local_id(&self) -> &str {
        self.metadata
            .get(Self::LOCAL_ID_METADATA_KEY)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Sets a custom local ID for this envelope (builder pattern).
    ///
    /// Useful for testing or when resuming from serialized state.
    ///
    /// # Arguments
    ///
    /// * `id` - The custom local ID
    ///
    /// # Returns
    ///
    /// Self with the custom local ID set
    pub fn with_local_id(mut self, id: impl Into<String>) -> Self {
        self.metadata
            .insert(Self::LOCAL_ID_METADATA_KEY.to_string(), id.into());
        self
    }

    /// Adds a metadata key-value pair.
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key
    /// * `value` - Metadata value
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets a metadata value by key.
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key
    ///
    /// # Returns
    ///
    /// `Some(value)` if the key exists, `None` otherwise
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    // =========================================================================
    // Message Role Helpers (for conversation/chat contexts)
    // =========================================================================

    /// Metadata key for storing the message role.
    pub const ROLE_METADATA_KEY: &'static str = "xybrid.role";

    /// Sets the message role for this envelope and returns self (builder pattern).
    ///
    /// Stores the role under the `xybrid.role` metadata key.
    ///
    /// # Arguments
    ///
    /// * `role` - The message role (System, User, or Assistant)
    ///
    /// # Returns
    ///
    /// Self with the role set
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let envelope = Envelope::new(EnvelopeKind::Text("Hello".to_string()))
    ///     .with_role(MessageRole::User);
    /// assert_eq!(envelope.role(), Some(MessageRole::User));
    /// ```
    pub fn with_role(mut self, role: super::MessageRole) -> Self {
        self.metadata.insert(
            Self::ROLE_METADATA_KEY.to_string(),
            role.as_str().to_string(),
        );
        self
    }

    /// Gets the message role of this envelope.
    ///
    /// Reads the role from the `xybrid.role` metadata key.
    ///
    /// # Returns
    ///
    /// `Some(MessageRole)` if a valid role is set, `None` otherwise
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let envelope = Envelope::new(EnvelopeKind::Text("Hello".to_string()))
    ///     .with_role(MessageRole::User);
    /// assert_eq!(envelope.role(), Some(MessageRole::User));
    ///
    /// // Envelopes without a role return None
    /// let plain = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
    /// assert_eq!(plain.role(), None);
    /// ```
    pub fn role(&self) -> Option<super::MessageRole> {
        self.metadata
            .get(Self::ROLE_METADATA_KEY)
            .and_then(|s| match s.as_str() {
                "system" => Some(super::MessageRole::System),
                "user" => Some(super::MessageRole::User),
                "assistant" => Some(super::MessageRole::Assistant),
                _ => None,
            })
    }

    /// Returns `true` if this envelope has the User message role.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let user_msg = Envelope::new(EnvelopeKind::Text("Hi".to_string()))
    ///     .with_role(MessageRole::User);
    /// assert!(user_msg.is_user_message());
    /// ```
    pub fn is_user_message(&self) -> bool {
        self.role() == Some(super::MessageRole::User)
    }

    /// Returns `true` if this envelope has the Assistant message role.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let assistant_msg = Envelope::new(EnvelopeKind::Text("Hello!".to_string()))
    ///     .with_role(MessageRole::Assistant);
    /// assert!(assistant_msg.is_assistant_message());
    /// ```
    pub fn is_assistant_message(&self) -> bool {
        self.role() == Some(super::MessageRole::Assistant)
    }

    /// Returns `true` if this envelope has the System message role.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let system_msg = Envelope::new(EnvelopeKind::Text("You are helpful.".to_string()))
    ///     .with_role(MessageRole::System);
    /// assert!(system_msg.is_system_message());
    /// ```
    pub fn is_system_message(&self) -> bool {
        self.role() == Some(super::MessageRole::System)
    }

    /// Returns a string representation of the envelope kind.
    ///
    /// # Returns
    ///
    /// A string describing the variant (e.g., "Audio", "Text", "Embedding")
    pub fn kind_str(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Returns the approximate size of the envelope payload in bytes.
    ///
    /// # Returns
    ///
    /// The size of the payload data
    pub fn payload_size(&self) -> usize {
        self.kind.payload_size()
    }

    /// Serializes the envelope to a byte vector using bincode.
    ///
    /// # Returns
    ///
    /// A `Result` containing the serialized bytes or an error
    pub fn to_bytes(&self) -> Result<Vec<u8>, EnvelopeError> {
        bincode::serialize(self).map_err(|e| {
            EnvelopeError::SerializationError(format!("Failed to serialize envelope: {}", e))
        })
    }

    /// Deserializes an envelope from a byte vector using bincode.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The serialized envelope bytes
    ///
    /// # Returns
    ///
    /// A `Result` containing the deserialized envelope or an error
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EnvelopeError> {
        bincode::deserialize(bytes).map_err(|e| {
            EnvelopeError::DeserializationError(format!("Failed to deserialize envelope: {}", e))
        })
    }

    /// Serializes the envelope to JSON format (for debugging/telemetry).
    ///
    /// # Returns
    ///
    /// A `Result` containing the JSON string or an error
    pub fn to_json(&self) -> Result<String, EnvelopeError> {
        serde_json::to_string_pretty(self).map_err(|e| {
            EnvelopeError::SerializationError(format!(
                "Failed to serialize envelope to JSON: {}",
                e
            ))
        })
    }

    /// Deserializes an envelope from JSON format.
    ///
    /// # Arguments
    ///
    /// * `json` - The JSON string
    ///
    /// # Returns
    ///
    /// A `Result` containing the deserialized envelope or an error
    pub fn from_json(json: &str) -> Result<Self, EnvelopeError> {
        serde_json::from_str(json).map_err(|e| {
            EnvelopeError::DeserializationError(format!(
                "Failed to deserialize envelope from JSON: {}",
                e
            ))
        })
    }

    /// Extracts audio samples from the envelope based on the `format` metadata.
    ///
    /// Supports the following formats (via `format` metadata):
    /// - `"float32"`: Pre-decoded float32 samples (from AudioEnvelope)
    /// - `"pcm16"`: Raw 16-bit PCM bytes
    /// - `"wav"` or unset: WAV file bytes (caller should use WAV decoder)
    ///
    /// # Returns
    ///
    /// - `Ok(Some(samples))` if audio was successfully extracted
    /// - `Ok(None)` if format is WAV or unknown (caller should decode)
    /// - `Err` if envelope is not Audio type
    ///
    /// # Audio Metadata
    ///
    /// The following metadata keys are used:
    /// - `format`: Audio format ("float32", "pcm16", "wav")
    /// - `sample_rate`: Sample rate in Hz
    /// - `channels`: Number of channels
    pub fn to_audio_samples(&self) -> Result<Option<AudioSamples>, EnvelopeError> {
        let audio_bytes = match &self.kind {
            EnvelopeKind::Audio(bytes) => bytes,
            _ => {
                return Err(EnvelopeError::DeserializationError(
                    "Envelope is not Audio type".to_string(),
                ))
            }
        };

        let format = self
            .get_metadata("format")
            .map(|s| s.as_str())
            .unwrap_or("wav");

        let sample_rate: u32 = self
            .get_metadata("sample_rate")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16000);

        let channels: u32 = self
            .get_metadata("channels")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        match format {
            "float32" => {
                // Pre-decoded float32 samples from AudioEnvelope
                let num_samples = audio_bytes.len() / 4;
                let mut samples = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let idx = i * 4;
                    if idx + 3 < audio_bytes.len() {
                        let sample = f32::from_le_bytes([
                            audio_bytes[idx],
                            audio_bytes[idx + 1],
                            audio_bytes[idx + 2],
                            audio_bytes[idx + 3],
                        ]);
                        samples.push(sample);
                    }
                }
                Ok(Some(AudioSamples {
                    samples,
                    sample_rate,
                    channels,
                }))
            }
            "pcm16" => {
                // Raw 16-bit PCM bytes
                let num_samples = audio_bytes.len() / 2;
                let mut samples = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let idx = i * 2;
                    if idx + 1 < audio_bytes.len() {
                        let sample_i16 =
                            i16::from_le_bytes([audio_bytes[idx], audio_bytes[idx + 1]]);
                        samples.push(sample_i16 as f32 / 32768.0);
                    }
                }
                Ok(Some(AudioSamples {
                    samples,
                    sample_rate,
                    channels,
                }))
            }
            _ => {
                // WAV or unknown format - caller should use WAV decoder
                Ok(None)
            }
        }
    }

    /// Returns the raw audio bytes if this is an Audio envelope.
    pub fn audio_bytes(&self) -> Option<&[u8]> {
        match &self.kind {
            EnvelopeKind::Audio(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Returns the audio format from metadata.
    pub fn audio_format(&self) -> Option<&str> {
        self.get_metadata("format").map(|s| s.as_str())
    }
}

/// Extracted audio samples with metadata.
#[derive(Debug, Clone)]
pub struct AudioSamples {
    /// Normalized float32 samples (-1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
}

impl AudioSamples {
    /// Convert to mono by averaging channels.
    pub fn to_mono(&self) -> Self {
        if self.channels <= 1 {
            return self.clone();
        }

        let channels = self.channels as usize;
        let mono_samples: Vec<f32> = self
            .samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();

        Self {
            samples: mono_samples,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Resample to target sample rate using linear interpolation.
    pub fn resample(&self, target_rate: u32) -> Self {
        if self.sample_rate == target_rate {
            return self.clone();
        }

        let ratio = target_rate as f32 / self.sample_rate as f32;
        let target_len = (self.samples.len() as f32 * ratio) as usize;

        let resampled: Vec<f32> = (0..target_len)
            .map(|i| {
                let source_idx = (i as f32 / ratio) as usize;
                self.samples.get(source_idx).copied().unwrap_or(0.0)
            })
            .collect();

        Self {
            samples: resampled,
            sample_rate: target_rate,
            channels: self.channels,
        }
    }

    /// Prepare for ASR (convert to mono 16kHz).
    pub fn prepare_for_asr(&self) -> Self {
        self.to_mono().resample(16000)
    }
}

/// Error type for envelope operations.
#[derive(Error, Debug)]
pub enum EnvelopeError {
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
}

/// Result type for envelope operations.
pub type EnvelopeResult<T> = Result<T, EnvelopeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_kind_as_str() {
        assert_eq!(EnvelopeKind::Audio(vec![]).as_str(), "Audio");
        assert_eq!(EnvelopeKind::Text(String::new()).as_str(), "Text");
        assert_eq!(EnvelopeKind::Embedding(vec![]).as_str(), "Embedding");
    }

    #[test]
    fn test_envelope_kind_payload_size() {
        let audio = EnvelopeKind::Audio(vec![0u8; 100]);
        assert_eq!(audio.payload_size(), 100);

        let text = EnvelopeKind::Text("hello".to_string());
        assert_eq!(text.payload_size(), 5);

        let embedding = EnvelopeKind::Embedding(vec![0.0f32; 10]);
        assert_eq!(embedding.payload_size(), 10 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_envelope_new() {
        let envelope = Envelope::new(EnvelopeKind::Text("test".to_string()));
        assert_eq!(envelope.kind, EnvelopeKind::Text("test".to_string()));
        // New envelopes have a local_id automatically generated
        assert!(!envelope.local_id().is_empty());
        assert_eq!(envelope.local_id().len(), 36); // UUID format
    }

    #[test]
    fn test_envelope_unique_local_ids() {
        let e1 = Envelope::new(EnvelopeKind::Text("same text".to_string()));
        let e2 = Envelope::new(EnvelopeKind::Text("same text".to_string()));

        // Each envelope has a unique local ID even with identical content
        assert_ne!(e1.local_id(), e2.local_id());
    }

    #[test]
    fn test_envelope_with_local_id() {
        let envelope =
            Envelope::new(EnvelopeKind::Text("test".to_string())).with_local_id("custom-id-123");

        assert_eq!(envelope.local_id(), "custom-id-123");
    }

    #[test]
    fn test_envelope_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        let envelope =
            Envelope::with_metadata(EnvelopeKind::Audio(vec![1, 2, 3]), metadata.clone());

        // with_metadata preserves provided metadata AND adds local_id
        assert_eq!(envelope.get_metadata("key1"), Some(&"value1".to_string()));
        assert!(!envelope.local_id().is_empty());
    }

    #[test]
    fn test_envelope_with_metadata_preserves_local_id() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert(
            Envelope::LOCAL_ID_METADATA_KEY.to_string(),
            "my-custom-id".to_string(),
        );
        let envelope = Envelope::with_metadata(EnvelopeKind::Audio(vec![1, 2, 3]), metadata);

        // Custom local_id in metadata is preserved
        assert_eq!(envelope.local_id(), "my-custom-id");
    }

    #[test]
    fn test_envelope_metadata_operations() {
        let mut envelope = Envelope::new(EnvelopeKind::Text("test".to_string()));

        envelope.set_metadata("key1".to_string(), "value1".to_string());
        assert_eq!(envelope.get_metadata("key1"), Some(&"value1".to_string()));
        assert_eq!(envelope.get_metadata("nonexistent"), None);
    }

    #[test]
    fn test_envelope_kind_str() {
        let envelope = Envelope::new(EnvelopeKind::Audio(vec![]));
        assert_eq!(envelope.kind_str(), "Audio");
    }

    #[test]
    fn test_envelope_serialization() -> Result<(), EnvelopeError> {
        let mut envelope = Envelope::new(EnvelopeKind::Text("hello world".to_string()));
        envelope.set_metadata("stage".to_string(), "asr".to_string());

        // Serialize to bytes
        let bytes = envelope.to_bytes()?;
        assert!(!bytes.is_empty());

        // Deserialize from bytes
        let deserialized = Envelope::from_bytes(&bytes)?;
        assert_eq!(deserialized.kind, envelope.kind);
        assert_eq!(deserialized.metadata, envelope.metadata);

        Ok(())
    }

    #[test]
    fn test_envelope_json_serialization() -> Result<(), EnvelopeError> {
        let mut envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));
        envelope.set_metadata("key".to_string(), "value".to_string());

        // Serialize to JSON
        let json = envelope.to_json()?;
        assert!(json.contains("hello"));
        assert!(json.contains("key"));

        // Deserialize from JSON
        let deserialized = Envelope::from_json(&json)?;
        assert_eq!(deserialized.kind, envelope.kind);
        assert_eq!(deserialized.metadata, envelope.metadata);

        Ok(())
    }

    #[test]
    fn test_envelope_audio_roundtrip() -> Result<(), EnvelopeError> {
        let audio_data = vec![0u8, 1u8, 2u8, 3u8, 4u8];
        let envelope = Envelope::new(EnvelopeKind::Audio(audio_data.clone()));

        let bytes = envelope.to_bytes()?;
        let deserialized = Envelope::from_bytes(&bytes)?;

        match deserialized.kind {
            EnvelopeKind::Audio(data) => assert_eq!(data, audio_data),
            _ => panic!("Expected Audio variant"),
        }

        Ok(())
    }

    #[test]
    fn test_envelope_embedding_roundtrip() -> Result<(), EnvelopeError> {
        let embedding_data = vec![1.0f32, 2.0f32, 3.0f32];
        let envelope = Envelope::new(EnvelopeKind::Embedding(embedding_data.clone()));

        let bytes = envelope.to_bytes()?;
        let deserialized = Envelope::from_bytes(&bytes)?;

        match deserialized.kind {
            EnvelopeKind::Embedding(data) => assert_eq!(data, embedding_data),
            _ => panic!("Expected Embedding variant"),
        }

        Ok(())
    }

    // =========================================================================
    // Message Role Tests
    // =========================================================================

    #[test]
    fn test_envelope_with_role_user() {
        use super::super::MessageRole;

        let envelope =
            Envelope::new(EnvelopeKind::Text("Hello".to_string())).with_role(MessageRole::User);

        assert_eq!(envelope.role(), Some(MessageRole::User));
        assert!(envelope.is_user_message());
        assert!(!envelope.is_assistant_message());
        assert!(!envelope.is_system_message());
    }

    #[test]
    fn test_envelope_with_role_assistant() {
        use super::super::MessageRole;

        let envelope = Envelope::new(EnvelopeKind::Text("Hi there!".to_string()))
            .with_role(MessageRole::Assistant);

        assert_eq!(envelope.role(), Some(MessageRole::Assistant));
        assert!(!envelope.is_user_message());
        assert!(envelope.is_assistant_message());
        assert!(!envelope.is_system_message());
    }

    #[test]
    fn test_envelope_with_role_system() {
        use super::super::MessageRole;

        let envelope = Envelope::new(EnvelopeKind::Text("You are helpful.".to_string()))
            .with_role(MessageRole::System);

        assert_eq!(envelope.role(), Some(MessageRole::System));
        assert!(!envelope.is_user_message());
        assert!(!envelope.is_assistant_message());
        assert!(envelope.is_system_message());
    }

    #[test]
    fn test_envelope_without_role() {
        let envelope = Envelope::new(EnvelopeKind::Text("Plain message".to_string()));

        // Envelopes without a role return None (backwards compatible)
        assert_eq!(envelope.role(), None);
        assert!(!envelope.is_user_message());
        assert!(!envelope.is_assistant_message());
        assert!(!envelope.is_system_message());
    }

    #[test]
    fn test_envelope_role_roundtrip() {
        use super::super::MessageRole;

        // Test round-trip: with_role -> role() returns correct value
        for role in [
            MessageRole::System,
            MessageRole::User,
            MessageRole::Assistant,
        ] {
            let envelope = Envelope::new(EnvelopeKind::Text("test".to_string())).with_role(role);
            assert_eq!(
                envelope.role(),
                Some(role),
                "Round-trip failed for {:?}",
                role
            );
        }
    }

    #[test]
    fn test_envelope_role_metadata_key() {
        use super::super::MessageRole;

        let envelope =
            Envelope::new(EnvelopeKind::Text("test".to_string())).with_role(MessageRole::User);

        // Verify the metadata key is correctly set
        assert_eq!(
            envelope.get_metadata(Envelope::ROLE_METADATA_KEY),
            Some(&"user".to_string())
        );
    }

    #[test]
    fn test_envelope_role_serialization_roundtrip() -> Result<(), EnvelopeError> {
        use super::super::MessageRole;

        let envelope =
            Envelope::new(EnvelopeKind::Text("Hello".to_string())).with_role(MessageRole::User);

        // Binary roundtrip
        let bytes = envelope.to_bytes()?;
        let deserialized = Envelope::from_bytes(&bytes)?;
        assert_eq!(deserialized.role(), Some(MessageRole::User));

        // JSON roundtrip
        let json = envelope.to_json()?;
        let from_json = Envelope::from_json(&json)?;
        assert_eq!(from_json.role(), Some(MessageRole::User));

        Ok(())
    }
}
