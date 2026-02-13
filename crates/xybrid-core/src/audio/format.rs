//! Audio format definitions and detection.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Audio format specification with type-safe parameters.
///
/// This enum represents the various audio formats that can be processed,
/// including raw PCM formats and container formats like WAV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    /// 16-bit signed PCM (little-endian)
    Pcm16 { sample_rate: u32, channels: u32 },
    /// 32-bit signed PCM (little-endian)
    Pcm32 { sample_rate: u32, channels: u32 },
    /// 32-bit floating point PCM
    Float32 { sample_rate: u32, channels: u32 },
    /// WAV container format (format details extracted from header)
    Wav,
}

impl AudioFormat {
    /// Returns the sample rate for this format.
    ///
    /// For WAV format, returns None as the sample rate is in the header.
    pub fn sample_rate(&self) -> Option<u32> {
        match self {
            AudioFormat::Pcm16 { sample_rate, .. } => Some(*sample_rate),
            AudioFormat::Pcm32 { sample_rate, .. } => Some(*sample_rate),
            AudioFormat::Float32 { sample_rate, .. } => Some(*sample_rate),
            AudioFormat::Wav => None,
        }
    }

    /// Returns the number of channels for this format.
    ///
    /// For WAV format, returns None as the channels are in the header.
    pub fn channels(&self) -> Option<u32> {
        match self {
            AudioFormat::Pcm16 { channels, .. } => Some(*channels),
            AudioFormat::Pcm32 { channels, .. } => Some(*channels),
            AudioFormat::Float32 { channels, .. } => Some(*channels),
            AudioFormat::Wav => None,
        }
    }

    /// Returns the bytes per sample for this format.
    pub fn bytes_per_sample(&self) -> Option<u32> {
        match self {
            AudioFormat::Pcm16 { .. } => Some(2),
            AudioFormat::Pcm32 { .. } => Some(4),
            AudioFormat::Float32 { .. } => Some(4),
            AudioFormat::Wav => None,
        }
    }

    /// Returns a string representation of the format.
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioFormat::Pcm16 { .. } => "pcm16",
            AudioFormat::Pcm32 { .. } => "pcm32",
            AudioFormat::Float32 { .. } => "float32",
            AudioFormat::Wav => "wav",
        }
    }

    /// Creates a PCM16 format with given parameters.
    pub fn pcm16(sample_rate: u32, channels: u32) -> Self {
        AudioFormat::Pcm16 {
            sample_rate,
            channels,
        }
    }

    /// Creates a Float32 format with given parameters.
    pub fn float32(sample_rate: u32, channels: u32) -> Self {
        AudioFormat::Float32 {
            sample_rate,
            channels,
        }
    }

    /// Default format for ASR models (16kHz mono PCM16).
    pub fn asr_default() -> Self {
        AudioFormat::Pcm16 {
            sample_rate: 16000,
            channels: 1,
        }
    }
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::asr_default()
    }
}

/// Detects the audio format from file bytes.
///
/// Currently supports:
/// - WAV files (RIFF header detection)
///
/// # Arguments
///
/// * `data` - The audio file bytes
///
/// # Returns
///
/// The detected `AudioFormat` or an error if format cannot be determined.
pub fn detect_format(data: &[u8]) -> Result<AudioFormat, AudioFormatError> {
    // Check for WAV format (RIFF header)
    if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        return Ok(AudioFormat::Wav);
    }

    // TODO: Add detection for other formats (MP3, OGG, FLAC, etc.)

    Err(AudioFormatError::UnknownFormat(
        "Could not detect audio format from header".to_string(),
    ))
}

/// Error type for audio format operations.
#[derive(Error, Debug)]
pub enum AudioFormatError {
    #[error("Unknown audio format: {0}")]
    UnknownFormat(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid format parameters: {0}")]
    InvalidParameters(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_pcm16() {
        let format = AudioFormat::pcm16(16000, 1);
        assert_eq!(format.sample_rate(), Some(16000));
        assert_eq!(format.channels(), Some(1));
        assert_eq!(format.bytes_per_sample(), Some(2));
        assert_eq!(format.as_str(), "pcm16");
    }

    #[test]
    fn test_audio_format_float32() {
        let format = AudioFormat::float32(44100, 2);
        assert_eq!(format.sample_rate(), Some(44100));
        assert_eq!(format.channels(), Some(2));
        assert_eq!(format.bytes_per_sample(), Some(4));
        assert_eq!(format.as_str(), "float32");
    }

    #[test]
    fn test_audio_format_wav() {
        let format = AudioFormat::Wav;
        assert_eq!(format.sample_rate(), None);
        assert_eq!(format.channels(), None);
        assert_eq!(format.as_str(), "wav");
    }

    #[test]
    fn test_audio_format_default() {
        let format = AudioFormat::default();
        assert_eq!(format.sample_rate(), Some(16000));
        assert_eq!(format.channels(), Some(1));
    }

    #[test]
    fn test_detect_format_wav() {
        // RIFF....WAVEfmt  header
        let wav_header = b"RIFF\x00\x00\x00\x00WAVEfmt ";
        let result = detect_format(wav_header);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), AudioFormat::Wav);
    }

    #[test]
    fn test_detect_format_unknown() {
        let unknown_data = b"unknown_format_data";
        let result = detect_format(unknown_data);
        assert!(result.is_err());
    }
}
