//! Mel Spectrogram preprocessing using mel-spec library (legacy).
//!
//! This module provides mel spectrogram computation using the `mel-spec` crate.
//! This is a generic implementation that may produce slightly different results
//! than model-specific implementations.
//!
//! For Whisper-compatible mel spectrograms, use `preprocessing::mel_spectrogram`
//! which implements the exact Slaney mel scale and reflect padding.

use crate::runtime_adapter::{AdapterError, AdapterResult};
use mel_spec::prelude::*;
use ndarray::{ArrayD, IxDyn};

/// Convert PCM audio to mel spectrogram using mel-spec library
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
pub fn audio_to_mel_spectrogram(
    audio_samples: &[f32],
    sample_rate: u32,
    n_mels: usize,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
) -> AdapterResult<ArrayD<f32>> {
    let hop_length = hop_length.unwrap_or(160); // 10ms at 16kHz
    let n_fft = n_fft.unwrap_or(400); // 25ms at 16kHz

    // Create STFT processor (converts audio to FFT frames)
    let mut stft = Spectrogram::new(n_fft, hop_length);

    // Create mel spectrogram processor (converts FFT frames to mel spectrogram)
    let mut mel_spec = MelSpectrogram::new(n_fft, sample_rate as f64, n_mels);

    // Process audio samples and collect mel frames
    let mut mel_frames = Vec::new();

    // Process audio in chunks
    for chunk in audio_samples.chunks(hop_length) {
        // Get FFT frame from STFT (returns Option<Array1<Complex<f64>>>)
        if let Some(fft_frame) = stft.add(chunk) {
            // Apply mel filter bank to FFT frame
            // mel_spec.add() returns Array2<f64> with shape [n_mels, time_frames_in_frame]
            let mel_frame = mel_spec.add(&fft_frame);

            // mel_frame is [n_mels, frames], extract each frame
            let n_frames_in_batch = mel_frame.shape()[1];
            for frame_idx in 0..n_frames_in_batch {
                let mut frame_data = Vec::with_capacity(n_mels);
                for mel_idx in 0..n_mels {
                    frame_data.push(mel_frame[[mel_idx, frame_idx]] as f32);
                }
                mel_frames.push(frame_data);
            }
        }
    }

    if mel_frames.is_empty() {
        return Err(AdapterError::InvalidInput(
            "No mel spectrogram frames generated".to_string(),
        ));
    }

    // Combine all frames into a single array
    // Each frame is [n_mels], we want [n_mels, time_frames]
    let time_frames = mel_frames.len();
    let mut mel_data = Vec::with_capacity(n_mels * time_frames);

    // Apply log scaling (log10 with small epsilon to avoid log(0))
    let epsilon = 1e-10;

    for frame in &mel_frames {
        for &value in frame {
            let log_value = (value.max(epsilon)).log10();
            mel_data.push(log_value);
        }
    }

    // Apply Whisper-specific normalization (dynamic range compression + scaling)
    normalize_mel_spectrogram_whisper(&mut mel_data);

    // Create tensor with shape [batch, n_mels, time_frames]
    let batch_size = 1;
    let mel_shape = vec![batch_size, n_mels, time_frames];
    let mel_tensor = ArrayD::<f32>::from_shape_vec(IxDyn(&mel_shape), mel_data)
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to create mel tensor: {:?}", e)))?;

    Ok(mel_tensor)
}

/// Apply Whisper-specific normalization to log-mel spectrogram
///
/// Whisper uses a specific normalization:
/// 1. Dynamic range compression: max(log_spec, max_val - 8.0)
/// 2. Normalize: (log_spec + 4.0) / 4.0
///
/// This gives values roughly in the range [0, 1] or slightly beyond
fn normalize_mel_spectrogram_whisper(mel_data: &mut [f32]) {
    if mel_data.is_empty() {
        return;
    }

    // Find maximum value for dynamic range compression
    let max_val = mel_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_clamp = max_val - 8.0; // Dynamic range of 8.0 (in log10 scale)

    // Apply Whisper normalization: clamp to dynamic range, then scale
    for value in mel_data.iter_mut() {
        // Clamp to dynamic range
        *value = value.max(min_clamp);
        // Normalize: (log_spec + 4.0) / 4.0
        *value = (*value + 4.0) / 4.0;
    }
}

/// Convert PCM audio bytes to mel spectrogram
///
/// Convenience function that handles PCM → f32 conversion first
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

    audio_to_mel_spectrogram(&samples, sample_rate, n_mels, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_spectrogram_basic() {
        // Generate 1 second of silence at 16kHz
        let samples = vec![0.0f32; 16000];
        let result = audio_to_mel_spectrogram(&samples, 16000, 80, None, None);
        assert!(result.is_ok());
        let mel = result.unwrap();
        assert_eq!(mel.shape()[0], 1); // batch size
        assert_eq!(mel.shape()[1], 80); // n_mels
    }

    #[test]
    fn test_normalization() {
        let mut data = vec![-5.0, -3.0, -1.0, 0.0, 1.0];
        normalize_mel_spectrogram_whisper(&mut data);
        // Max is 1.0, so min_clamp is -7.0
        // After normalization: (val + 4) / 4
        // Smallest valid value: (-7 + 4) / 4 = -0.75
        // Largest value: (1 + 4) / 4 = 1.25
        for val in &data {
            assert!(*val >= -0.75 && *val <= 1.25, "Value {} out of expected range", val);
        }
    }
}
