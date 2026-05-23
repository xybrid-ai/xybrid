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

    /// Create an encoded image envelope.
    #[frb(sync)]
    pub fn image(bytes: Vec<u8>, format: String) -> Result<FfiEnvelope, String> {
        Envelope::image(bytes, format)
            .map(FfiEnvelope)
            .map_err(|err| err.to_string())
    }

    /// Create a user-role multi-part envelope with image attachments.
    #[frb(sync)]
    pub fn user_message(text: String, images: Vec<FfiEnvelope>) -> Result<FfiEnvelope, String> {
        let images = images.into_iter().map(|image| image.0).collect();
        Envelope::user_message(text, images)
            .map(FfiEnvelope)
            .map_err(|err| err.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_sdk::ir::MessageRole;

    #[test]
    fn image_rejects_unsupported_format() {
        let error = match FfiEnvelope::image(vec![1, 2, 3], "heic".to_string()) {
            Ok(_) => panic!("expected unsupported image format error"),
            Err(error) => error,
        };

        assert!(error.contains("Unsupported image format 'heic'"));
    }

    #[test]
    fn image_rejects_corrupt_bytes_with_redacted_error() {
        let error = match FfiEnvelope::image(vec![42, 42, 42, 42], "jpeg".to_string()) {
            Ok(_) => panic!("expected corrupt image bytes error"),
            Err(error) => error,
        };

        assert!(error.contains("invalid or corrupt jpeg image bytes"));
        assert!(!error.contains("[42"));
        assert!(!error.contains("42, 42"));
    }

    #[test]
    fn image_rejects_oversized_encoded_payload() {
        let bytes = vec![0; xybrid_sdk::ir::envelope::DEFAULT_MAX_ENCODED_IMAGE_BYTES + 1];
        let error = match FfiEnvelope::image(bytes, "png".to_string()) {
            Ok(_) => panic!("expected oversized image payload error"),
            Err(error) => error,
        };

        assert!(error.contains("Image payload too large"));
        assert!(!error.contains("[0"));
    }

    #[test]
    fn user_message_sets_user_role_and_multipart_shape() {
        let envelope = FfiEnvelope::user_message("Describe this image".to_string(), Vec::new())
            .expect("empty image list still produces a user multipart envelope");

        assert_eq!(envelope.0.role(), Some(MessageRole::User));
        match envelope.0.kind {
            EnvelopeKind::MultiPart(parts) => {
                assert_eq!(parts.len(), 1);
                assert_eq!(parts[0].as_text(), Some("Describe this image"));
            }
            other => panic!("expected multipart envelope, got {other:?}"),
        }
    }
}
