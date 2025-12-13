//! TTS response types.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Audio output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// WAV format (16-bit PCM)
    Wav,
    /// Raw PCM samples (f32)
    RawF32,
    /// Raw PCM samples (i16)
    RawI16,
}

impl Default for AudioFormat {
    fn default() -> Self {
        AudioFormat::Wav
    }
}

/// Audio output from TTS synthesis.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    /// Raw audio samples (float32, normalized to -1.0 to 1.0)
    pub samples: Vec<f32>,

    /// Sample rate in Hz (typically 24000 for KittenTTS)
    pub sample_rate: u32,

    /// Number of audio channels (always 1 for TTS)
    pub channels: u16,

    /// Inference latency in milliseconds (if measured)
    pub latency_ms: Option<u32>,
}

impl AudioOutput {
    /// Create new audio output from samples.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
            latency_ms: None,
        }
    }

    /// Set the latency measurement.
    pub fn with_latency(mut self, latency_ms: u32) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }

    /// Get duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Get duration in milliseconds.
    pub fn duration_ms(&self) -> u32 {
        (self.duration_secs() * 1000.0) as u32
    }

    /// Convert samples to 16-bit PCM.
    pub fn to_i16(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }

    /// Convert samples to bytes (16-bit PCM, little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let i16_samples = self.to_i16();
        let mut bytes = Vec::with_capacity(i16_samples.len() * 2);
        for sample in i16_samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    /// Encode as WAV file bytes.
    pub fn to_wav(&self) -> Vec<u8> {
        let num_samples = self.samples.len() as u32;
        let bits_per_sample: u16 = 16;
        let byte_rate =
            self.sample_rate * self.channels as u32 * bits_per_sample as u32 / 8;
        let block_align = self.channels * bits_per_sample / 8;
        let data_size = num_samples * self.channels as u32 * bits_per_sample as u32 / 8;
        let file_size = 36 + data_size;

        let mut wav = Vec::with_capacity(44 + data_size as usize);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav.extend_from_slice(&self.channels.to_le_bytes());
        wav.extend_from_slice(&self.sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());

        // Audio data
        wav.extend_from_slice(&self.to_bytes());

        wav
    }

    /// Save audio to a WAV file.
    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<(), super::TtsError> {
        let wav_data = self.to_wav();
        std::fs::write(path, wav_data)?;
        Ok(())
    }

    /// Save audio as raw PCM bytes to a file.
    pub fn save_raw(&self, path: impl AsRef<Path>) -> Result<(), super::TtsError> {
        let bytes = self.to_bytes();
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_output_duration() {
        let samples = vec![0.0; 24000]; // 1 second at 24kHz
        let audio = AudioOutput::new(samples, 24000);

        assert_eq!(audio.duration_secs(), 1.0);
        assert_eq!(audio.duration_ms(), 1000);
    }

    #[test]
    fn test_to_i16() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let audio = AudioOutput::new(samples, 24000);
        let i16_samples = audio.to_i16();

        assert_eq!(i16_samples[0], 0);
        assert_eq!(i16_samples[1], 16383);
        assert_eq!(i16_samples[2], -16383);
        assert_eq!(i16_samples[3], 32767);
        assert_eq!(i16_samples[4], -32767);
    }

    #[test]
    fn test_wav_header() {
        let samples = vec![0.0; 100];
        let audio = AudioOutput::new(samples, 24000);
        let wav = audio.to_wav();

        // Check RIFF header
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
    }

    #[test]
    fn test_with_latency() {
        let audio = AudioOutput::new(vec![0.0], 24000).with_latency(150);
        assert_eq!(audio.latency_ms, Some(150));
    }
}
