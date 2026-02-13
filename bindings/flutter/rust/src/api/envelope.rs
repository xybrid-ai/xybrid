//! Envelope FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::collections::HashMap;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};

use super::context::FfiMessageRole;

/// FFI wrapper for input envelopes.
#[frb(opaque)]
pub struct FfiEnvelope(pub(crate) Envelope);

impl FfiEnvelope {
    /// Create audio envelope with raw bytes and format metadata.
    #[frb(sync)]
    pub fn audio(bytes: Vec<u8>, sample_rate: u32, channels: u32) -> FfiEnvelope {
        let mut metadata = HashMap::new();
        metadata.insert("sample_rate".to_string(), sample_rate.to_string());
        metadata.insert("channels".to_string(), channels.to_string());
        FfiEnvelope(Envelope::with_metadata(
            EnvelopeKind::Audio(bytes),
            metadata,
        ))
    }

    /// Create text envelope for TTS with optional voice and speed.
    #[frb(sync)]
    pub fn text(text: String, voice_id: Option<String>, speed: Option<f64>) -> FfiEnvelope {
        let mut metadata = HashMap::new();
        if let Some(v) = voice_id {
            metadata.insert("voice_id".to_string(), v);
        }
        if let Some(s) = speed {
            metadata.insert("speed".to_string(), s.to_string());
        }
        FfiEnvelope(Envelope::with_metadata(EnvelopeKind::Text(text), metadata))
    }

    /// Create embedding envelope from float vector.
    #[frb(sync)]
    pub fn embedding(data: Vec<f32>) -> FfiEnvelope {
        FfiEnvelope(Envelope::new(EnvelopeKind::Embedding(data)))
    }

    /// Create a text envelope with a specific message role.
    ///
    /// This is useful for building conversation context.
    #[frb(sync)]
    pub fn text_with_role(text: String, role: FfiMessageRole) -> FfiEnvelope {
        let envelope = Envelope::new(EnvelopeKind::Text(text)).with_role(role.into());
        FfiEnvelope(envelope)
    }

    /// Set the message role on this envelope.
    ///
    /// Returns a new envelope with the role set.
    #[frb(sync)]
    pub fn with_role(&self, role: FfiMessageRole) -> FfiEnvelope {
        FfiEnvelope(self.0.clone().with_role(role.into()))
    }

    /// Get the message role of this envelope, if set.
    #[frb(sync)]
    pub fn role(&self) -> Option<FfiMessageRole> {
        self.0.role().map(|r| r.into())
    }

    /// Get the unique local ID of this envelope.
    ///
    /// Each envelope has a UUID generated on creation for tracking
    /// and duplicate detection.
    #[frb(sync)]
    pub fn local_id(&self) -> String {
        self.0.local_id().to_string()
    }

    /// Convert to inner Envelope for SDK calls.
    pub(crate) fn into_envelope(self) -> Envelope {
        self.0
    }

    /// Clone the inner envelope (for context operations).
    #[allow(dead_code)]
    pub(crate) fn clone_envelope(&self) -> Envelope {
        self.0.clone()
    }
}
