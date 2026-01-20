//! Mel Spectrogram preprocessing (DEPRECATED).
//!
//! **DEPRECATED**: This module is deprecated and will be removed in a future version.
//!
//! Use the canonical implementation instead:
//! - For direct mel spectrogram computation: `audio::mel::compute_mel_spectrogram`
//! - For pipeline preprocessing: `preprocessing::mel_spectrogram`
//!
//! This module now re-exports from `audio::mel` for backward compatibility.

#![deprecated(
    since = "0.1.0",
    note = "Use `audio::mel::compute_mel_spectrogram` or `preprocessing::mel_spectrogram` instead"
)]

use crate::audio::mel::{compute_mel_spectrogram, MelConfig, MelScale};
use crate::runtime_adapter::{AdapterError, AdapterResult};
use ndarray::ArrayD;

/// Convert PCM audio to mel spectrogram.
///
/// **DEPRECATED**: Use `audio::mel::compute_mel_spectrogram` instead.
///
/// # Arguments
/// * `audio_samples` - PCM audio samples (f32, normalized to [-1.0, 1.0])
/// * `sample_rate` - Sample rate in Hz (e.g., 16000)
/// * `n_mels` - Number of mel filter banks (80 for Whisper)
/// * `hop_length` - Hop length in samples (default: 160 for 16kHz)
/// * `n_fft` - FFT window size (default: 400 for 16kHz)
///
/// # Returns
/// Mel spectrogram tensor with shape [batch, n_mels, time_frames]
#[deprecated(
    since = "0.1.0",
    note = "Use `audio::mel::compute_mel_spectrogram` with `MelConfig` instead"
)]
pub fn audio_to_mel_spectrogram(
    audio_samples: &[f32],
    sample_rate: u32,
    n_mels: usize,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
) -> AdapterResult<ArrayD<f32>> {
    let config = MelConfig {
        n_mels,
        sample_rate,
        hop_length: hop_length.unwrap_or(160),
        n_fft: n_fft.unwrap_or(400),
        mel_scale: MelScale::Slaney,
        normalize: true,
        max_frames: None, // Don't pad/truncate
        ..Default::default()
    };

    compute_mel_spectrogram(audio_samples, &config).map_err(|e| AdapterError::InvalidInput(e))
}

/// Convert PCM audio bytes to mel spectrogram.
///
/// **DEPRECATED**: Use `preprocessing::mel_spectrogram::audio_bytes_to_whisper_mel` instead.
///
/// Convenience function that handles PCM → f32 conversion first.
#[deprecated(
    since = "0.1.0",
    note = "Use `preprocessing::mel_spectrogram::audio_bytes_to_whisper_mel` instead"
)]
pub fn audio_bytes_to_mel_spectrogram(
    audio_bytes: &[u8],
    sample_rate: u32,
    n_mels: usize,
) -> AdapterResult<ArrayD<f32>> {
    // Convert PCM bytes to f32 samples
    if audio_bytes.len() % 2 != 0 {
        return Err(AdapterError::InvalidInput(
            "Audio data length must be even for 16-bit PCM".to_string(),
        ));
    }

    let samples: Vec<f32> = audio_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0 // Normalize to [-1.0, 1.0]
        })
        .collect();

    #[allow(deprecated)]
    audio_to_mel_spectrogram(&samples, sample_rate, n_mels, None, None)
}

#[cfg(test)]
mod tests {
    #[allow(deprecated)]
    use super::*;

    #[test]
    fn test_mel_spectrogram_basic() {
        // Generate 1 second of silence at 16kHz
        let samples = vec![0.0f32; 16000];
        #[allow(deprecated)]
        let result = audio_to_mel_spectrogram(&samples, 16000, 80, None, None);
        assert!(result.is_ok());
        let mel = result.unwrap();
        assert_eq!(mel.shape()[0], 1); // batch size
        assert_eq!(mel.shape()[1], 80); // n_mels
    }
}
