//! Envelope FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::collections::HashMap;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};

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
        FfiEnvelope(Envelope::with_metadata(EnvelopeKind::Audio(bytes), metadata))
    }

    /// Create text envelope for TTS with optional voice and speed.
    #[frb(sync)]
    pub fn text(text: String, voice_id: Option<String>, speed: Option<f64>) -> FfiEnvelope {
        let mut metadata = HashMap::new();
        if let Some(v) = voice_id { metadata.insert("voice_id".to_string(), v); }
        if let Some(s) = speed { metadata.insert("speed".to_string(), s.to_string()); }
        FfiEnvelope(Envelope::with_metadata(EnvelopeKind::Text(text), metadata))
    }

    /// Create embedding envelope from float vector.
    #[frb(sync)]
    pub fn embedding(data: Vec<f32>) -> FfiEnvelope {
        FfiEnvelope(Envelope::new(EnvelopeKind::Embedding(data)))
    }

    /// Convert to inner Envelope for SDK calls.
    pub(crate) fn into_envelope(self) -> Envelope { self.0 }
}
