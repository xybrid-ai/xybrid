//! Voice configuration types for TTS models.
//!
//! This module defines how voices are stored, loaded, and selected in TTS models.

use serde::{Deserialize, Serialize};

/// Voice format describing how voices are stored and loaded.
///
/// Different TTS models store voice embeddings in different formats:
/// - Kokoro/KittenTTS: Embedded binary file with multiple voices
/// - Piper: Each voice is a separate model file
/// - Chatterbox: Voice cloning from reference audio
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "format")]
pub enum VoiceFormat {
    /// Voices embedded in single file (Kokoro, KittenTTS)
    ///
    /// Example:
    /// ```json
    /// {
    ///   "format": "embedded",
    ///   "file": "voices.bin",
    ///   "loader": "binary_f32_256"
    /// }
    /// ```
    #[serde(rename = "embedded")]
    Embedded {
        /// Path to voice embedding file (relative to bundle root)
        file: String,
        /// Loader type for parsing the file
        loader: VoiceLoader,
    },

    /// Each voice is a separate model file (Piper)
    ///
    /// Example:
    /// ```json
    /// {
    ///   "format": "per_model",
    ///   "voice_dir": "voices/",
    ///   "pattern": "{voice_id}.onnx"
    /// }
    /// ```
    #[serde(rename = "per_model")]
    PerModel {
        /// Directory containing voice model files
        voice_dir: String,
        /// File pattern (e.g., "{voice_id}.onnx")
        pattern: String,
    },

    /// Voice cloning from reference audio (Chatterbox, future)
    ///
    /// Example:
    /// ```json
    /// {
    ///   "format": "cloning",
    ///   "encoder_model": "voice_encoder.onnx",
    ///   "min_audio_seconds": 3.0,
    ///   "max_audio_seconds": 30.0
    /// }
    /// ```
    #[serde(rename = "cloning")]
    Cloning {
        /// Voice encoder model file
        encoder_model: String,
        /// Minimum reference audio duration (seconds)
        min_audio_seconds: f32,
        /// Maximum reference audio duration (seconds)
        max_audio_seconds: f32,
    },
}

/// Voice embedding loader type.
///
/// Specifies how to parse the voice embedding file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum VoiceLoader {
    /// Binary f32 embeddings, 256 dimensions
    ///
    /// Used by: KittenTTS v0.1, Kokoro
    /// Format: Contiguous f32 little-endian arrays, 1024 bytes per voice
    #[default]
    #[serde(rename = "binary_f32_256")]
    BinaryF32_256,

    /// NumPy .npz archive
    ///
    /// Used by: KittenTTS v0.2
    /// Format: ZIP archive with NumPy arrays
    #[serde(rename = "numpy_npz")]
    NumpyNpz,

    /// JSON with base64-encoded embeddings (future)
    #[serde(rename = "json_base64")]
    JsonBase64,
}

/// Voice catalog entry with rich metadata.
///
/// Each voice in a TTS model has an entry describing its properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Unique identifier (used in API calls, e.g., "af_bella")
    pub id: String,

    /// Human-readable display name (e.g., "Bella")
    pub name: String,

    /// Gender: "male", "female", "neutral"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,

    /// BCP-47 language tag (e.g., "en-US", "ja-JP")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Optional style descriptor (e.g., "neutral", "cheerful", "professional")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,

    /// Index in embedding file (for embedded format)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,

    /// Preview audio URL (optional, for UI)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview_url: Option<String>,
}

/// Strategy for selecting voice embeddings at inference time.
///
/// Different TTS models use different voice selection mechanisms:
/// - KittenTTS: Select by fixed catalog index
/// - Kokoro: Select from voicepack by phoneme token count
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum VoiceSelectionStrategy {
    /// Select voice by catalog index (default).
    ///
    /// Used by: KittenTTS and other models with simple voice catalogs.
    /// The voice embedding is loaded at the catalog entry's fixed index.
    #[default]
    FixedIndex,

    /// Select voice embedding slice by phoneme token count.
    ///
    /// Used by: Kokoro-style voicepacks (3D arrays, shape `[510, 1, 256]`).
    /// The style vector is indexed by: `pack[min(max(token_count - 2, 0), 509)]`.
    TokenLength,
}

/// Complete voice configuration for a TTS model.
///
/// This struct describes all available voices for a TTS model,
/// including how they are stored and which one is the default.
///
/// Example:
/// ```json
/// {
///   "format": "embedded",
///   "file": "voices.bin",
///   "loader": "binary_f32_256",
///   "default": "af_bella",
///   "selection_strategy": "FixedIndex",
///   "catalog": [
///     {"id": "af_bella", "name": "Bella", "gender": "female", "language": "en-US", "index": 0}
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Voice storage format and location
    #[serde(flatten)]
    pub format: VoiceFormat,

    /// Default voice ID (used when no voice specified)
    pub default: String,

    /// Voice catalog with metadata
    pub catalog: Vec<VoiceInfo>,

    /// How voice embeddings are selected at inference time.
    ///
    /// Defaults to `FixedIndex` if not specified (backwards compatible).
    #[serde(default, skip_serializing_if = "is_default_strategy")]
    pub selection_strategy: VoiceSelectionStrategy,
}

fn is_default_strategy(strategy: &VoiceSelectionStrategy) -> bool {
    *strategy == VoiceSelectionStrategy::FixedIndex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_config_serialization() {
        let config = VoiceConfig {
            format: VoiceFormat::Embedded {
                file: "voices.bin".to_string(),
                loader: VoiceLoader::BinaryF32_256,
            },
            default: "af_bella".to_string(),
            catalog: vec![
                VoiceInfo {
                    id: "af_bella".to_string(),
                    name: "Bella".to_string(),
                    gender: Some("female".to_string()),
                    language: Some("en-US".to_string()),
                    style: None,
                    index: Some(0),
                    preview_url: None,
                },
                VoiceInfo {
                    id: "am_adam".to_string(),
                    name: "Adam".to_string(),
                    gender: Some("male".to_string()),
                    language: Some("en-US".to_string()),
                    style: Some("neutral".to_string()),
                    index: Some(1),
                    preview_url: None,
                },
            ],
            selection_strategy: VoiceSelectionStrategy::default(),
        };

        let json = serde_json::to_string_pretty(&config).unwrap();

        // Verify JSON structure
        assert!(json.contains("\"format\": \"embedded\""));
        assert!(json.contains("\"file\": \"voices.bin\""));
        assert!(json.contains("\"default\": \"af_bella\""));

        // Deserialize back
        let parsed: VoiceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.default, "af_bella");
        assert_eq!(parsed.catalog.len(), 2);
    }

    #[test]
    fn test_voice_format_per_model() {
        let json = r#"{
            "format": "per_model",
            "voice_dir": "voices/",
            "pattern": "{voice_id}.onnx",
            "default": "en-us-libritts",
            "catalog": [
                {"id": "en-us-libritts", "name": "LibriTTS", "language": "en-US"}
            ]
        }"#;

        let config: VoiceConfig = serde_json::from_str(json).unwrap();
        match &config.format {
            VoiceFormat::PerModel { voice_dir, pattern } => {
                assert_eq!(voice_dir, "voices/");
                assert_eq!(pattern, "{voice_id}.onnx");
            }
            _ => panic!("Expected PerModel format"),
        }
    }

    #[test]
    fn test_voice_format_cloning() {
        let json = r#"{
            "format": "cloning",
            "encoder_model": "voice_encoder.onnx",
            "min_audio_seconds": 3.0,
            "max_audio_seconds": 30.0,
            "default": "cloned",
            "catalog": []
        }"#;

        let config: VoiceConfig = serde_json::from_str(json).unwrap();
        match &config.format {
            VoiceFormat::Cloning {
                encoder_model,
                min_audio_seconds,
                max_audio_seconds,
            } => {
                assert_eq!(encoder_model, "voice_encoder.onnx");
                assert_eq!(*min_audio_seconds, 3.0);
                assert_eq!(*max_audio_seconds, 30.0);
            }
            _ => panic!("Expected Cloning format"),
        }
    }
}
