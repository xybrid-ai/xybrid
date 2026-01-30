//! Mel spectrogram preprocessing step for ASR models.
//!
//! This module provides a thin step wrapper around the core mel spectrogram
//! implementation in `audio::mel::whisper`. It handles:
//!
//! - WAV file parsing and format detection
//! - Sample rate resampling to 16kHz
//! - Integration with the pipeline error types
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::preprocessing::mel_spectrogram::{audio_bytes_to_whisper_mel};
//!
//! let wav_bytes = std::fs::read("audio.wav").unwrap();
//! let mel_tensor = audio_bytes_to_whisper_mel(&wav_bytes)?;
//! // mel_tensor has shape [1, 80, 3000]
//! ```

use crate::audio::mel::whisper::{compute_whisper_mel, WhisperMelConfig};
use crate::runtime_adapter::{AdapterError, AdapterResult};
use ndarray::ArrayD;

/// Configuration for the mel spectrogram preprocessing step.
#[derive(Debug, Clone)]
pub struct MelSpectrogramConfig {
    /// Target sample rate (default: 16000 Hz)
    pub target_sample_rate: u32,
    /// Whisper mel configuration
    pub mel_config: WhisperMelConfig,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16000,
            mel_config: WhisperMelConfig::default(),
        }
    }
}

/// Mel spectrogram preprocessing step.
///
/// This step converts audio input to mel spectrogram tensors suitable for
/// Whisper and other ASR models.
pub struct MelSpectrogramStep {
    config: MelSpectrogramConfig,
}

impl MelSpectrogramStep {
    /// Create a new mel spectrogram step with default configuration.
    pub fn new() -> Self {
        Self {
            config: MelSpectrogramConfig::default(),
        }
    }

    /// Create a new mel spectrogram step with custom configuration.
    pub fn with_config(config: MelSpectrogramConfig) -> Self {
        Self { config }
    }

    /// Process audio samples to mel spectrogram.
    pub fn process(&self, samples: &[f32]) -> AdapterResult<ArrayD<f32>> {
        compute_whisper_mel(samples, &self.config.mel_config)
            .map_err(|e| AdapterError::InvalidInput(e))
    }

    /// Process audio bytes (WAV or raw PCM) to mel spectrogram.
    pub fn process_bytes(&self, audio_bytes: &[u8]) -> AdapterResult<ArrayD<f32>> {
        audio_bytes_to_whisper_mel(audio_bytes)
    }
}

impl Default for MelSpectrogramStep {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Convert PCM audio samples to Whisper-compatible mel spectrogram.
///
/// # Arguments
/// * `audio_samples` - PCM audio samples (f32, normalized to [-1.0, 1.0])
///
/// # Returns
/// Mel spectrogram tensor with shape [1, 80, 3000]
pub fn audio_to_whisper_mel(audio_samples: &[f32]) -> AdapterResult<ArrayD<f32>> {
    if audio_samples.is_empty() {
        return Err(AdapterError::InvalidInput(
            "Cannot compute mel spectrogram from empty audio".to_string(),
        ));
    }

    let config = WhisperMelConfig::default();
    compute_whisper_mel(audio_samples, &config).map_err(|e| AdapterError::InvalidInput(e))
}

/// Convert audio bytes (WAV or raw PCM) to Whisper-compatible mel spectrogram.
///
/// Handles both:
/// - WAV files (parses header, extracts PCM, resamples if needed)
/// - Raw PCM bytes (16-bit signed little-endian, assumes 16kHz)
///
/// # Arguments
/// * `audio_bytes` - WAV file bytes or raw PCM bytes
///
/// # Returns
/// Mel spectrogram tensor with shape [1, 80, 3000]
pub fn audio_bytes_to_whisper_mel(audio_bytes: &[u8]) -> AdapterResult<ArrayD<f32>> {
    // Check if this is a WAV file
    let (samples, actual_sample_rate) = if audio_bytes.len() >= 44
        && &audio_bytes[0..4] == b"RIFF"
        && &audio_bytes[8..12] == b"WAVE"
    {
        parse_wav_to_samples(audio_bytes)?
    } else {
        // Treat as raw PCM
        if audio_bytes.len() % 2 != 0 {
            return Err(AdapterError::InvalidInput(
                "Audio data length must be even for 16-bit PCM".to_string(),
            ));
        }

        let samples: Vec<f32> = audio_bytes
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        (samples, 16000) // Assume 16kHz for raw PCM
    };

    // Resample if necessary (Whisper expects 16kHz)
    let resampled = if actual_sample_rate != 16000 {
        resample_linear(&samples, actual_sample_rate, 16000)
    } else {
        samples
    };

    audio_to_whisper_mel(&resampled)
}

// ============================================================================
// WAV Parsing
// ============================================================================

/// Parse WAV file to f32 samples.
fn parse_wav_to_samples(wav_bytes: &[u8]) -> AdapterResult<(Vec<f32>, u32)> {
    // RIFF header: "RIFF" + file_size + "WAVE"
    if wav_bytes.len() < 44 {
        return Err(AdapterError::InvalidInput("WAV file too short".to_string()));
    }

    // Find "fmt " chunk
    let mut pos = 12; // Skip RIFF header
    let mut sample_rate = 0u32;
    let mut num_channels = 0u16;
    let mut bits_per_sample = 0u16;

    while pos + 8 <= wav_bytes.len() {
        let chunk_id = &wav_bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            wav_bytes[pos + 4],
            wav_bytes[pos + 5],
            wav_bytes[pos + 6],
            wav_bytes[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            if pos + 8 + 16 > wav_bytes.len() {
                return Err(AdapterError::InvalidInput("Invalid fmt chunk".to_string()));
            }

            let fmt_start = pos + 8;
            num_channels = u16::from_le_bytes([wav_bytes[fmt_start + 2], wav_bytes[fmt_start + 3]]);
            sample_rate = u32::from_le_bytes([
                wav_bytes[fmt_start + 4],
                wav_bytes[fmt_start + 5],
                wav_bytes[fmt_start + 6],
                wav_bytes[fmt_start + 7],
            ]);
            bits_per_sample =
                u16::from_le_bytes([wav_bytes[fmt_start + 14], wav_bytes[fmt_start + 15]]);
        } else if chunk_id == b"data" {
            // Found data chunk
            let data_start = pos + 8;
            let data_end = (data_start + chunk_size).min(wav_bytes.len());
            let audio_data = &wav_bytes[data_start..data_end];

            // Convert to f32 samples
            let samples = convert_pcm_to_f32(audio_data, bits_per_sample, num_channels)?;
            return Ok((samples, sample_rate));
        }

        pos += 8 + chunk_size;
        // Align to word boundary
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }

    Err(AdapterError::InvalidInput(
        "No data chunk found in WAV file".to_string(),
    ))
}

/// Convert PCM bytes to f32 samples (mono, normalized to [-1.0, 1.0]).
fn convert_pcm_to_f32(
    audio_data: &[u8],
    bits_per_sample: u16,
    num_channels: u16,
) -> AdapterResult<Vec<f32>> {
    let mut samples = Vec::new();

    match bits_per_sample {
        16 => {
            let bytes_per_frame = 2 * num_channels as usize;
            for frame in audio_data.chunks_exact(bytes_per_frame) {
                // Mix down to mono by averaging channels
                let mut sum = 0i32;
                for ch in 0..num_channels as usize {
                    let sample = i16::from_le_bytes([frame[ch * 2], frame[ch * 2 + 1]]) as i32;
                    sum += sample;
                }
                let mono = (sum / num_channels as i32) as f32 / 32768.0;
                samples.push(mono);
            }
        }
        8 => {
            let bytes_per_frame = num_channels as usize;
            for frame in audio_data.chunks_exact(bytes_per_frame) {
                // 8-bit PCM is unsigned, centered at 128
                let mut sum = 0i32;
                for ch in 0..num_channels as usize {
                    let sample = (frame[ch] as i32 - 128) as i32;
                    sum += sample;
                }
                let mono = (sum / num_channels as i32) as f32 / 128.0;
                samples.push(mono);
            }
        }
        _ => {
            return Err(AdapterError::InvalidInput(format!(
                "Unsupported bits per sample: {}",
                bits_per_sample
            )));
        }
    }

    Ok(samples)
}

// ============================================================================
// Resampling
// ============================================================================

/// Simple linear interpolation resampling.
fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = src_idx - idx0 as f64;

        let sample = samples[idx0] * (1.0 - frac as f32) + samples[idx1] * frac as f32;
        resampled.push(sample);
    }

    resampled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_creation() {
        let step = MelSpectrogramStep::new();
        assert_eq!(step.config.target_sample_rate, 16000);
    }

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let resampled = resample_linear(&samples, 16000, 16000);
        assert_eq!(samples.len(), resampled.len());
    }

    #[test]
    fn test_resample_downsample() {
        let samples: Vec<f32> = (0..32000).map(|i| (i as f32) / 32000.0).collect();
        let resampled = resample_linear(&samples, 32000, 16000);
        // Should be approximately half the length
        assert!(resampled.len() < samples.len());
        assert!(resampled.len() > samples.len() / 3);
    }
}
