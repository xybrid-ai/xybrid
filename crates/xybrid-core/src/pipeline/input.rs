//! Input and output configuration types for pipeline stages.
//!
//! Defines the various input/output types supported by pipelines:
//! - Audio (ASR, TTS input)
//! - Text (LLM, embeddings)
//! - Image (vision models)
//! - Embedding (vector operations)

use serde::{Deserialize, Serialize};

/// Input type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    Audio,
    Text,
    Image,
    Embedding,
}

impl InputType {
    pub fn as_str(&self) -> &'static str {
        match self {
            InputType::Audio => "audio",
            InputType::Text => "text",
            InputType::Image => "image",
            InputType::Embedding => "embedding",
        }
    }
}

/// Output type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    Audio,
    Text,
    Image,
    Embedding,
    Structured,
}

impl OutputType {
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputType::Audio => "audio",
            OutputType::Text => "text",
            OutputType::Image => "image",
            OutputType::Embedding => "embedding",
            OutputType::Structured => "structured",
        }
    }
}

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioSampleFormat {
    Pcm16,
    Pcm32,
    Float32,
}

impl Default for AudioSampleFormat {
    fn default() -> Self {
        AudioSampleFormat::Float32
    }
}

/// Image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    Rgb,
    Bgr,
    Grayscale,
}

impl Default for ImageFormat {
    fn default() -> Self {
        ImageFormat::Rgb
    }
}

/// Audio input configuration for ASR and audio processing stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioInputConfig {
    /// Sample rate in Hz (e.g., 16000, 44100).
    pub sample_rate: u32,

    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u32,

    /// Audio sample format.
    #[serde(default)]
    pub format: AudioSampleFormat,

    /// Whether streaming input is supported.
    #[serde(default)]
    pub streaming: bool,
}

impl AudioInputConfig {
    /// Standard ASR configuration (16kHz mono float32).
    pub fn asr_default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            format: AudioSampleFormat::Float32,
            streaming: false,
        }
    }

    /// CD quality configuration (44.1kHz stereo PCM16).
    pub fn cd_quality() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            format: AudioSampleFormat::Pcm16,
            streaming: false,
        }
    }
}

impl Default for AudioInputConfig {
    fn default() -> Self {
        Self::asr_default()
    }
}

/// Text input configuration for LLM and NLP stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextInputConfig {
    /// Maximum text length in characters/tokens.
    #[serde(default)]
    pub max_length: Option<u32>,

    /// Text encoding (default: utf8).
    #[serde(default = "default_encoding")]
    pub encoding: String,
}

fn default_encoding() -> String {
    "utf8".to_string()
}

impl Default for TextInputConfig {
    fn default() -> Self {
        Self {
            max_length: None,
            encoding: "utf8".to_string(),
        }
    }
}

/// Image input configuration for vision models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInputConfig {
    /// Image width in pixels.
    pub width: u32,

    /// Image height in pixels.
    pub height: u32,

    /// Number of channels (1 = grayscale, 3 = RGB).
    #[serde(default = "default_image_channels")]
    pub channels: u32,

    /// Image format.
    #[serde(default)]
    pub format: ImageFormat,
}

fn default_image_channels() -> u32 {
    3
}

impl Default for ImageInputConfig {
    fn default() -> Self {
        Self {
            width: 224,
            height: 224,
            channels: 3,
            format: ImageFormat::Rgb,
        }
    }
}

/// Embedding input configuration for vector operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingInputConfig {
    /// Embedding dimensions.
    pub dimensions: u32,
}

impl Default for EmbeddingInputConfig {
    fn default() -> Self {
        Self { dimensions: 384 }
    }
}

/// Unified input configuration with type tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum InputConfig {
    Audio {
        #[serde(flatten)]
        config: AudioInputConfig,
    },
    Text {
        #[serde(flatten)]
        config: TextInputConfig,
    },
    Image {
        #[serde(flatten)]
        config: ImageInputConfig,
    },
    Embedding {
        #[serde(flatten)]
        config: EmbeddingInputConfig,
    },
}

impl InputConfig {
    /// Get the input type.
    pub fn input_type(&self) -> InputType {
        match self {
            InputConfig::Audio { .. } => InputType::Audio,
            InputConfig::Text { .. } => InputType::Text,
            InputConfig::Image { .. } => InputType::Image,
            InputConfig::Embedding { .. } => InputType::Embedding,
        }
    }

    /// Create audio input config.
    pub fn audio(config: AudioInputConfig) -> Self {
        InputConfig::Audio { config }
    }

    /// Create text input config.
    pub fn text(config: TextInputConfig) -> Self {
        InputConfig::Text { config }
    }

    /// Create image input config.
    pub fn image(config: ImageInputConfig) -> Self {
        InputConfig::Image { config }
    }

    /// Create embedding input config.
    pub fn embedding(config: EmbeddingInputConfig) -> Self {
        InputConfig::Embedding { config }
    }
}

/// Audio output configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioOutputConfig {
    /// Sample rate in Hz.
    #[serde(default = "default_output_sample_rate")]
    pub sample_rate: u32,

    /// Audio sample format.
    #[serde(default)]
    pub format: AudioSampleFormat,
}

fn default_output_sample_rate() -> u32 {
    22050
}

impl Default for AudioOutputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            format: AudioSampleFormat::Pcm16,
        }
    }
}

/// Embedding output configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingOutputConfig {
    /// Embedding dimensions.
    pub dimensions: u32,
}

/// Structured output schema (for classification, NER, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputConfig {
    /// JSON schema for the output.
    #[serde(default)]
    pub schema: serde_json::Value,
}

impl Default for StructuredOutputConfig {
    fn default() -> Self {
        Self {
            schema: serde_json::Value::Object(serde_json::Map::new()),
        }
    }
}

/// Unified output configuration with type tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum OutputConfig {
    Audio {
        #[serde(flatten)]
        config: AudioOutputConfig,
    },
    Text,
    Image,
    Embedding {
        #[serde(flatten)]
        config: EmbeddingOutputConfig,
    },
    Structured {
        #[serde(flatten)]
        config: StructuredOutputConfig,
    },
}

impl OutputConfig {
    /// Get the output type.
    pub fn output_type(&self) -> OutputType {
        match self {
            OutputConfig::Audio { .. } => OutputType::Audio,
            OutputConfig::Text => OutputType::Text,
            OutputConfig::Image => OutputType::Image,
            OutputConfig::Embedding { .. } => OutputType::Embedding,
            OutputConfig::Structured { .. } => OutputType::Structured,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig::Text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_input_config_serde() {
        let config = InputConfig::Audio {
            config: AudioInputConfig {
                sample_rate: 16000,
                channels: 1,
                format: AudioSampleFormat::Float32,
                streaming: false,
            },
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("type: audio"));
        assert!(yaml.contains("sample_rate: 16000"));

        let parsed: InputConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_text_input_config_serde() {
        let config = InputConfig::Text {
            config: TextInputConfig {
                max_length: Some(4096),
                encoding: "utf8".to_string(),
            },
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: InputConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_output_config_text() {
        let yaml = "type: text";
        let config: OutputConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.output_type(), OutputType::Text);
    }

    #[test]
    fn test_output_config_audio() {
        let yaml = r#"
type: audio
sample_rate: 22050
format: pcm16
"#;
        let config: OutputConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.output_type(), OutputType::Audio);
    }

    #[test]
    fn test_input_type_as_str() {
        assert_eq!(InputType::Audio.as_str(), "audio");
        assert_eq!(InputType::Text.as_str(), "text");
        assert_eq!(InputType::Image.as_str(), "image");
        assert_eq!(InputType::Embedding.as_str(), "embedding");
    }
}
