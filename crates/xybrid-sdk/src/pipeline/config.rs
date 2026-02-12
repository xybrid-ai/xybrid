//! Pipeline configuration types for FFI bindings.
//!
//! These types are designed to be imported by platform SDKs (Flutter, Kotlin, Swift)
//! to avoid duplicating type definitions across bindings.
//!
//! # Example
//!
//! ```rust
//! use xybrid_sdk::pipeline::config::{PipelineSource, InputType, OutputType, AudioSampleFormat};
//!
//! let source = PipelineSource::Bundle;
//! let input = InputType::Audio;
//! let output = OutputType::Text;
//! let format = AudioSampleFormat::Float32;
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// Pipeline Source
// ============================================================================

/// How the pipeline was loaded.
///
/// This enum indicates the origin of a pipeline configuration:
/// - `Yaml`: From a .yaml/.yml pipeline config file
/// - `Bundle`: From a local .xyb bundle file
/// - `Registry`: From the HTTP model registry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineSource {
    /// From .yaml/.yml pipeline config
    Yaml,
    /// From local .xyb bundle
    Bundle,
    /// From HTTP registry
    Registry,
}

// ============================================================================
// Input/Output Types
// ============================================================================

/// Input type for the pipeline.
///
/// Represents the type of data that can be fed into a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputType {
    /// Audio data (WAV, PCM samples)
    Audio,
    /// Text string
    Text,
    /// Image data
    Image,
    /// Pre-computed embedding vector
    Embedding,
}

/// Output type from the pipeline.
///
/// Represents the type of data produced by a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Text string (e.g., transcription, generated text)
    Text,
    /// Embedding vector
    Embedding,
    /// Audio data (e.g., TTS output)
    Audio,
    /// Image data
    Image,
    /// Structured JSON data
    Json,
}

// ============================================================================
// Audio Configuration
// ============================================================================

/// Audio sample format.
///
/// Specifies how audio samples are encoded in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioSampleFormat {
    /// 16-bit signed integer PCM
    Pcm16,
    /// 32-bit signed integer PCM
    Pcm32,
    /// 32-bit floating point
    Float32,
}

// ============================================================================
// Audio Configuration
// ============================================================================

/// Configuration for audio input to a pipeline.
///
/// This struct specifies how audio data should be formatted when feeding it
/// to a pipeline. Most ASR (Automatic Speech Recognition) models expect
/// 16kHz mono audio in float32 format.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::pipeline::config::AudioInputConfig;
///
/// // Use defaults (16kHz mono float32)
/// let config = AudioInputConfig::default();
/// assert_eq!(config.sample_rate, 16000);
///
/// // Create custom config
/// let config = AudioInputConfig {
///     sample_rate: 48000,
///     channels: 2,
///     format: xybrid_sdk::pipeline::config::AudioSampleFormat::Pcm16,
///     streaming: true,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInputConfig {
    /// Sample rate in Hz (e.g., 16000 for 16kHz)
    pub sample_rate: u32,
    /// Number of channels (1=mono, 2=stereo)
    pub channels: u32,
    /// Sample format
    pub format: AudioSampleFormat,
    /// Whether streaming input is supported
    pub streaming: bool,
}

impl Default for AudioInputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            format: AudioSampleFormat::Float32,
            streaming: false,
        }
    }
}

impl AudioInputConfig {
    /// Create a new AudioInputConfig with custom settings.
    pub fn new(
        sample_rate: u32,
        channels: u32,
        format: AudioSampleFormat,
        streaming: bool,
    ) -> Self {
        Self {
            sample_rate,
            channels,
            format,
            streaming,
        }
    }

    /// Standard ASR configuration (16kHz mono float32, non-streaming).
    ///
    /// This is the most common configuration for speech recognition models
    /// like Whisper, Wav2Vec2, etc.
    pub fn asr_default() -> Self {
        Self::default()
    }

    /// Configuration for streaming ASR (16kHz mono float32, streaming enabled).
    pub fn asr_streaming() -> Self {
        Self {
            streaming: true,
            ..Self::default()
        }
    }
}

// ============================================================================
// Text Configuration
// ============================================================================

/// Configuration for text input to a pipeline.
///
/// This struct specifies constraints and encoding for text input.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::pipeline::config::TextInputConfig;
///
/// // Use defaults (UTF-8, no length limit)
/// let config = TextInputConfig::default();
/// assert_eq!(config.encoding, "utf8");
///
/// // Create config with max length
/// let config = TextInputConfig {
///     max_length: Some(4096),
///     encoding: "utf8".to_string(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextInputConfig {
    /// Maximum text length in characters (None = unlimited)
    pub max_length: Option<u32>,
    /// Text encoding (typically "utf8")
    pub encoding: String,
}

impl Default for TextInputConfig {
    fn default() -> Self {
        Self {
            max_length: None,
            encoding: "utf8".to_string(),
        }
    }
}

impl TextInputConfig {
    /// Create a new TextInputConfig with custom settings.
    pub fn new(max_length: Option<u32>, encoding: impl Into<String>) -> Self {
        Self {
            max_length,
            encoding: encoding.into(),
        }
    }

    /// Configuration for LLM input with a token limit approximation.
    ///
    /// The max_length is a character estimate (4 chars ≈ 1 token for English).
    pub fn llm_default(max_tokens: u32) -> Self {
        Self {
            max_length: Some(max_tokens * 4), // Rough char-to-token conversion
            encoding: "utf8".to_string(),
        }
    }
}

// ============================================================================
// Input Configuration Union
// ============================================================================

/// Pipeline input configuration (union type).
///
/// This enum wraps the different input configuration types, allowing
/// pipelines to accept different input modalities.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::pipeline::config::{InputConfig, AudioInputConfig, TextInputConfig, InputType};
///
/// let audio_config = InputConfig::Audio(AudioInputConfig::asr_default());
/// assert_eq!(audio_config.input_type(), InputType::Audio);
///
/// let text_config = InputConfig::Text(TextInputConfig::default());
/// assert_eq!(text_config.input_type(), InputType::Text);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputConfig {
    /// Audio input configuration
    Audio(AudioInputConfig),
    /// Text input configuration
    Text(TextInputConfig),
    // Image and Embedding configs to be added in future releases
}

impl InputConfig {
    /// Get the input type for this configuration.
    pub fn input_type(&self) -> InputType {
        match self {
            InputConfig::Audio(_) => InputType::Audio,
            InputConfig::Text(_) => InputType::Text,
        }
    }

    /// Get the audio configuration, if this is an audio input.
    pub fn as_audio(&self) -> Option<&AudioInputConfig> {
        match self {
            InputConfig::Audio(config) => Some(config),
            _ => None,
        }
    }

    /// Get the text configuration, if this is a text input.
    pub fn as_text(&self) -> Option<&TextInputConfig> {
        match self {
            InputConfig::Text(config) => Some(config),
            _ => None,
        }
    }
}

impl Default for InputConfig {
    /// Default to audio input (most common for on-device ML).
    fn default() -> Self {
        InputConfig::Audio(AudioInputConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_source_serialization() {
        let source = PipelineSource::Bundle;
        let json = serde_json::to_string(&source).unwrap();
        assert_eq!(json, "\"Bundle\"");

        let parsed: PipelineSource = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, PipelineSource::Bundle);
    }

    #[test]
    fn test_input_type_serialization() {
        let input = InputType::Audio;
        let json = serde_json::to_string(&input).unwrap();
        assert_eq!(json, "\"Audio\"");

        let parsed: InputType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, InputType::Audio);
    }

    #[test]
    fn test_output_type_serialization() {
        let output = OutputType::Text;
        let json = serde_json::to_string(&output).unwrap();
        assert_eq!(json, "\"Text\"");

        let parsed: OutputType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, OutputType::Text);
    }

    #[test]
    fn test_audio_sample_format_serialization() {
        let format = AudioSampleFormat::Float32;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"Float32\"");

        let parsed: AudioSampleFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, AudioSampleFormat::Float32);
    }

    #[test]
    fn test_all_pipeline_sources() {
        assert_eq!(
            serde_json::to_string(&PipelineSource::Yaml).unwrap(),
            "\"Yaml\""
        );
        assert_eq!(
            serde_json::to_string(&PipelineSource::Bundle).unwrap(),
            "\"Bundle\""
        );
        assert_eq!(
            serde_json::to_string(&PipelineSource::Registry).unwrap(),
            "\"Registry\""
        );
    }

    #[test]
    fn test_all_input_types() {
        assert_eq!(
            serde_json::to_string(&InputType::Audio).unwrap(),
            "\"Audio\""
        );
        assert_eq!(serde_json::to_string(&InputType::Text).unwrap(), "\"Text\"");
        assert_eq!(
            serde_json::to_string(&InputType::Image).unwrap(),
            "\"Image\""
        );
        assert_eq!(
            serde_json::to_string(&InputType::Embedding).unwrap(),
            "\"Embedding\""
        );
    }

    #[test]
    fn test_all_output_types() {
        assert_eq!(
            serde_json::to_string(&OutputType::Text).unwrap(),
            "\"Text\""
        );
        assert_eq!(
            serde_json::to_string(&OutputType::Embedding).unwrap(),
            "\"Embedding\""
        );
        assert_eq!(
            serde_json::to_string(&OutputType::Audio).unwrap(),
            "\"Audio\""
        );
        assert_eq!(
            serde_json::to_string(&OutputType::Image).unwrap(),
            "\"Image\""
        );
        assert_eq!(
            serde_json::to_string(&OutputType::Json).unwrap(),
            "\"Json\""
        );
    }

    #[test]
    fn test_all_audio_formats() {
        assert_eq!(
            serde_json::to_string(&AudioSampleFormat::Pcm16).unwrap(),
            "\"Pcm16\""
        );
        assert_eq!(
            serde_json::to_string(&AudioSampleFormat::Pcm32).unwrap(),
            "\"Pcm32\""
        );
        assert_eq!(
            serde_json::to_string(&AudioSampleFormat::Float32).unwrap(),
            "\"Float32\""
        );
    }

    // ========================================================================
    // AudioInputConfig Tests
    // ========================================================================

    #[test]
    fn test_audio_input_config_default() {
        let config = AudioInputConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.format, AudioSampleFormat::Float32);
        assert!(!config.streaming);
    }

    #[test]
    fn test_audio_input_config_asr_default() {
        let config = AudioInputConfig::asr_default();
        // ASR default should be same as Default
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.format, AudioSampleFormat::Float32);
        assert!(!config.streaming);
    }

    #[test]
    fn test_audio_input_config_asr_streaming() {
        let config = AudioInputConfig::asr_streaming();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.format, AudioSampleFormat::Float32);
        assert!(config.streaming); // streaming enabled
    }

    #[test]
    fn test_audio_input_config_new() {
        let config = AudioInputConfig::new(48000, 2, AudioSampleFormat::Pcm16, true);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.format, AudioSampleFormat::Pcm16);
        assert!(config.streaming);
    }

    #[test]
    fn test_audio_input_config_serialization() {
        let config = AudioInputConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"sample_rate\":16000"));
        assert!(json.contains("\"channels\":1"));
        assert!(json.contains("\"format\":\"Float32\""));
        assert!(json.contains("\"streaming\":false"));

        let parsed: AudioInputConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.sample_rate, config.sample_rate);
        assert_eq!(parsed.channels, config.channels);
        assert_eq!(parsed.format, config.format);
        assert_eq!(parsed.streaming, config.streaming);
    }

    // ========================================================================
    // TextInputConfig Tests
    // ========================================================================

    #[test]
    fn test_text_input_config_default() {
        let config = TextInputConfig::default();
        assert_eq!(config.max_length, None);
        assert_eq!(config.encoding, "utf8");
    }

    #[test]
    fn test_text_input_config_new() {
        let config = TextInputConfig::new(Some(1000), "utf8");
        assert_eq!(config.max_length, Some(1000));
        assert_eq!(config.encoding, "utf8");
    }

    #[test]
    fn test_text_input_config_llm_default() {
        let config = TextInputConfig::llm_default(1024); // 1024 tokens
        assert_eq!(config.max_length, Some(4096)); // 1024 * 4 chars
        assert_eq!(config.encoding, "utf8");
    }

    #[test]
    fn test_text_input_config_serialization() {
        let config = TextInputConfig {
            max_length: Some(4096),
            encoding: "utf8".to_string(),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"max_length\":4096"));
        assert!(json.contains("\"encoding\":\"utf8\""));

        let parsed: TextInputConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_length, config.max_length);
        assert_eq!(parsed.encoding, config.encoding);
    }

    #[test]
    fn test_text_input_config_serialization_null_max_length() {
        let config = TextInputConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"max_length\":null"));

        let parsed: TextInputConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_length, None);
    }

    // ========================================================================
    // InputConfig Tests
    // ========================================================================

    #[test]
    fn test_input_config_audio() {
        let config = InputConfig::Audio(AudioInputConfig::asr_default());
        assert_eq!(config.input_type(), InputType::Audio);
        assert!(config.as_audio().is_some());
        assert!(config.as_text().is_none());
    }

    #[test]
    fn test_input_config_text() {
        let config = InputConfig::Text(TextInputConfig::default());
        assert_eq!(config.input_type(), InputType::Text);
        assert!(config.as_text().is_some());
        assert!(config.as_audio().is_none());
    }

    #[test]
    fn test_input_config_default() {
        let config = InputConfig::default();
        // Default is Audio
        assert_eq!(config.input_type(), InputType::Audio);
        assert!(config.as_audio().is_some());
    }

    #[test]
    fn test_input_config_serialization_audio() {
        let config = InputConfig::Audio(AudioInputConfig::default());
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"Audio\""));
        assert!(json.contains("\"sample_rate\":16000"));

        let parsed: InputConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.input_type(), InputType::Audio);
    }

    #[test]
    fn test_input_config_serialization_text() {
        let config = InputConfig::Text(TextInputConfig::default());
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"Text\""));
        assert!(json.contains("\"encoding\":\"utf8\""));

        let parsed: InputConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.input_type(), InputType::Text);
    }
}
