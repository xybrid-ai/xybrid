//! Common utilities shared across mel spectrogram implementations.
//!
//! This module provides shared functionality for STFT, windowing, and padding
//! that is used by all mel scale implementations.

use rustfft::{num_complex::Complex, FftPlanner};

use super::config::PaddingMode;

// ============================================================================
// Padding Functions
// ============================================================================

/// Apply padding to audio signal based on padding mode.
pub fn apply_padding(samples: &[f32], pad_size: usize, mode: PaddingMode) -> Vec<f32> {
    match mode {
        PaddingMode::Reflect => pad_reflect(samples, pad_size, pad_size),
        PaddingMode::Zero => pad_zero(samples, pad_size, pad_size),
        PaddingMode::None => samples.to_vec(),
    }
}

/// Calculate reflect offset using transformers.js formula.
fn calculate_reflect_offset(i: i32, w: i32) -> usize {
    ((i + w) % (2 * w) - w).unsigned_abs() as usize
}

/// Apply reflect padding to audio signal.
fn pad_reflect(samples: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let n = samples.len();

    // Handle empty input - return zeros
    if n == 0 {
        return vec![0.0f32; pad_left + pad_right];
    }

    let total_len = pad_left + n + pad_right;
    let mut padded = vec![0.0f32; total_len];
    let w = (n - 1) as i32;

    // Copy original samples
    for i in 0..n {
        padded[pad_left + i] = samples[i];
    }

    // Only apply padding if we have at least 2 samples to reflect
    if n > 1 {
        for i in 1..=pad_left {
            padded[pad_left - i] = samples[calculate_reflect_offset(i as i32, w)];
        }

        for i in 1..=pad_right {
            padded[w as usize + pad_left + i] = samples[calculate_reflect_offset(w - i as i32, w)];
        }
    } else {
        // For single sample, just replicate
        let val = samples[0];
        for i in 0..pad_left {
            padded[i] = val;
        }
        for i in 0..pad_right {
            padded[pad_left + 1 + i] = val;
        }
    }

    padded
}

/// Apply zero padding to audio signal.
fn pad_zero(samples: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; pad_left + samples.len() + pad_right];
    padded[pad_left..pad_left + samples.len()].copy_from_slice(samples);
    padded
}

// ============================================================================
// Window Functions
// ============================================================================

/// Create periodic Hann window.
///
/// The periodic variant is used for STFT (as opposed to symmetric for filter design).
pub fn create_hann_window(size: usize) -> Vec<f64> {
    let factor = 2.0 * std::f64::consts::PI / size as f64;
    (0..size)
        .map(|i| 0.5 - 0.5 * (i as f64 * factor).cos())
        .collect()
}

// ============================================================================
// STFT
// ============================================================================

/// Compute STFT power spectrogram.
///
/// Returns a vector of frames, where each frame is a vector of power values
/// for each frequency bin (n_fft/2 + 1 bins).
pub fn compute_stft_power(samples: &[f32], n_fft: usize, hop_length: usize) -> Vec<Vec<f64>> {
    let window = create_hann_window(n_fft);
    let n_freqs = n_fft / 2 + 1;

    let n_frames = (samples.len().saturating_sub(n_fft)) / hop_length + 1;
    if n_frames == 0 {
        return vec![];
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut power_spec = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = start + n_fft;

        if end > samples.len() {
            break;
        }

        let mut fft_input: Vec<Complex<f64>> = (0..n_fft)
            .map(|i| {
                let sample = samples[start + i] as f64;
                Complex::new(sample * window[i], 0.0)
            })
            .collect();

        fft.process(&mut fft_input);

        let frame_power: Vec<f64> = fft_input[..n_freqs].iter().map(|c| c.norm_sqr()).collect();

        power_spec.push(frame_power);
    }

    power_spec
}

// ============================================================================
// Normalization
// ============================================================================

/// Apply Whisper-style log normalization to mel spectrogram.
///
/// Steps:
/// 1. Apply log10 with epsilon floor
/// 2. Clamp to dynamic range (max - 8.0)
/// 3. Normalize: (val + 4.0) / 4.0
pub fn apply_log_normalization(mel_spec: &mut [Vec<f64>], max_frames: Option<usize>) -> Vec<f32> {
    const EPSILON: f64 = 1e-10;

    // Apply log scaling with floor
    for mel_row in mel_spec.iter_mut() {
        for val in mel_row.iter_mut() {
            *val = (*val).max(EPSILON).log10();
        }
    }

    // Find max value for dynamic range compression
    let max_val = mel_spec
        .iter()
        .flat_map(|row| row.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let min_clamp = max_val - 8.0;

    let n_mels = mel_spec.len();
    let n_frames = mel_spec.first().map(|r| r.len()).unwrap_or(0);
    let output_frames = max_frames.unwrap_or(n_frames);

    let mut mel_data = Vec::with_capacity(n_mels * output_frames);

    for mel_idx in 0..n_mels {
        for frame_idx in 0..output_frames {
            let val = if frame_idx < n_frames {
                mel_spec[mel_idx][frame_idx]
            } else {
                min_clamp // Pad with min value
            };

            let clamped = val.max(min_clamp);
            let normalized = ((clamped + 4.0) / 4.0) as f32;
            mel_data.push(normalized);
        }
    }

    mel_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = create_hann_window(400);
        assert_eq!(window.len(), 400);
        assert!((window[0] - 0.0).abs() < 0.001);
        assert!((window[200] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pad_reflect_basic() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = pad_reflect(&samples, 2, 2);
        // Reflect: [3, 2, 1, 2, 3, 4, 5, 4, 3]
        assert_eq!(padded.len(), 9);
        assert_eq!(padded[2], 1.0); // Original start
        assert_eq!(padded[6], 5.0); // Original end
    }

    #[test]
    fn test_pad_zero() {
        let samples = vec![1.0, 2.0, 3.0];
        let padded = pad_zero(&samples, 2, 2);
        assert_eq!(padded, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_stft_power_shape() {
        // 1 second of sine wave at 16kHz
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let power = compute_stft_power(&samples, 400, 160);
        assert!(!power.is_empty());
        assert_eq!(power[0].len(), 201); // n_fft/2 + 1
    }
}
