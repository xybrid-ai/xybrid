//! Whisper-compatible mel spectrogram computation.
//!
//! This implementation exactly matches transformers.js / OpenAI Whisper:
//!
//! ## Key differences from standard mel spectrograms:
//!
//! 1. **Slaney mel scale** (not HTK):
//!    - For freq < 1000Hz: `mel = 3.0 * freq / 200.0`
//!    - For freq >= 1000Hz: `mel = 15.0 + 27.0 * log(freq / 1000) / log(6.4)`
//!
//! 2. **Slaney normalization** for mel filter bank:
//!    - Each filter is normalized by: `2.0 / (upper_freq - lower_freq)`
//!
//! 3. **Reflect padding** before FFT:
//!    - Pads 200 samples on each side using mirror reflection
//!
//! 4. **Frequency range**: 0Hz to 8000Hz (not full spectrum)
//!
//! 5. **Hann window**: Periodic variant for STFT
//!
//! ## Whisper parameters:
//! - n_fft = 400 (25ms window at 16kHz)
//! - hop_length = 160 (10ms hop at 16kHz)
//! - n_mels = 80
//! - sample_rate = 16000 Hz
//! - max_frames = 3000 (30 seconds @ 100fps)

use ndarray::{ArrayD, IxDyn};
use rustfft::{FftPlanner, num_complex::Complex};

/// Configuration for Whisper mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct WhisperMelConfig {
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: u32,
    pub max_frames: usize,
    pub f_min: f64,
    pub f_max: f64,
}

impl Default for WhisperMelConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            sample_rate: 16000,
            max_frames: 3000,  // 30 seconds @ 100fps
            f_min: 0.0,
            f_max: 8000.0,     // Nyquist / 2 for 16kHz
        }
    }
}

/// Compute Whisper-compatible mel spectrogram from audio samples.
///
/// # Arguments
/// * `audio_samples` - PCM audio samples (f32, normalized to [-1.0, 1.0])
/// * `config` - Mel spectrogram configuration
///
/// # Returns
/// Mel spectrogram tensor with shape [1, n_mels, max_frames]
pub fn compute_whisper_mel(
    audio_samples: &[f32],
    config: &WhisperMelConfig,
) -> Result<ArrayD<f32>, String> {
    // Calculate padding: half_window = floor((n_fft - 1) / 2) + 1
    let pad_size = (config.n_fft - 1) / 2 + 1;

    // Apply reflect padding
    let padded_samples = pad_reflect(audio_samples, pad_size, pad_size);

    // Create Slaney mel filter bank
    let mel_filters = create_mel_filter_bank_slaney(
        config.n_mels,
        config.n_fft,
        config.sample_rate,
        config.f_min,
        config.f_max,
    );

    // Compute STFT power spectrogram
    let power_spec = stft_whisper(&padded_samples, config.n_fft, config.hop_length);

    if power_spec.is_empty() {
        return Err("No STFT frames generated - audio too short".to_string());
    }

    let n_frames = power_spec.len();
    let n_freqs = config.n_fft / 2 + 1;

    // Apply mel filter bank
    let mut mel_spec = vec![vec![0.0f64; n_frames]; config.n_mels];

    for (frame_idx, frame_power) in power_spec.iter().enumerate() {
        for mel_idx in 0..config.n_mels {
            let mut mel_value = 0.0;
            for freq_idx in 0..n_freqs {
                mel_value += mel_filters[mel_idx][freq_idx] * frame_power[freq_idx];
            }
            mel_spec[mel_idx][frame_idx] = mel_value;
        }
    }

    // Apply log scaling with floor
    const EPSILON: f64 = 1e-10;
    for mel_row in mel_spec.iter_mut() {
        for val in mel_row.iter_mut() {
            *val = (*val).max(EPSILON).log10();
        }
    }

    // Find max value for dynamic range compression
    let max_val = mel_spec.iter()
        .flat_map(|row| row.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let min_clamp = max_val - 8.0;

    // Apply Whisper normalization and pad/truncate to max_frames
    let mut mel_data = Vec::with_capacity(config.n_mels * config.max_frames);

    for mel_idx in 0..config.n_mels {
        for frame_idx in 0..config.max_frames {
            let val = if frame_idx < n_frames {
                mel_spec[mel_idx][frame_idx]
            } else {
                min_clamp
            };

            let clamped = val.max(min_clamp);
            let normalized = ((clamped + 4.0) / 4.0) as f32;
            mel_data.push(normalized);
        }
    }

    // Create tensor with shape [batch, n_mels, time_frames]
    let mel_shape = vec![1, config.n_mels, config.max_frames];
    ArrayD::<f32>::from_shape_vec(IxDyn(&mel_shape), mel_data)
        .map_err(|e| format!("Failed to create mel tensor: {:?}", e))
}

// ============================================================================
// Slaney Mel Scale
// ============================================================================

/// Convert frequency in Hz to mel scale using Slaney formula.
fn hz_to_mel_slaney(freq: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = 0.06875177742094912; // log(6.4) / 27

    if freq < MIN_LOG_HZ {
        freq / F_SP
    } else {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

/// Convert mel scale to frequency in Hz using Slaney formula.
fn mel_to_hz_slaney(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = 0.06875177742094912;

    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOGSTEP).exp()
    }
}

/// Create Slaney-normalized mel filter bank.
fn create_mel_filter_bank_slaney(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    f_min: f64,
    f_max: f64,
) -> Vec<Vec<f64>> {
    let n_freqs = n_fft / 2 + 1;

    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| (i as f64 * sample_rate as f64) / n_fft as f64)
        .collect();

    let mel_min = hz_to_mel_slaney(f_min);
    let mel_max = hz_to_mel_slaney(f_max);

    let n_mel_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_mel_points)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f64) / ((n_mel_points - 1) as f64))
        .collect();

    let freq_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    let mut filter_bank = vec![vec![0.0; n_freqs]; n_mels];

    for i in 0..n_mels {
        let f_lower = freq_points[i];
        let f_center = freq_points[i + 1];
        let f_upper = freq_points[i + 2];
        let enorm = 2.0 / (f_upper - f_lower);

        for (j, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_lower && freq <= f_center {
                filter_bank[i][j] = enorm * (freq - f_lower) / (f_center - f_lower);
            } else if freq > f_center && freq <= f_upper {
                filter_bank[i][j] = enorm * (f_upper - freq) / (f_upper - f_center);
            }
        }
    }

    filter_bank
}

// ============================================================================
// Reflect Padding
// ============================================================================

/// Calculate reflect offset using transformers.js formula.
fn calculate_reflect_offset(i: i32, w: i32) -> usize {
    (((i + w) % (2 * w) - w).abs()) as usize
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

// ============================================================================
// STFT with Hann Window
// ============================================================================

/// Create periodic Hann window.
fn create_hann_window(size: usize) -> Vec<f64> {
    let factor = 2.0 * std::f64::consts::PI / size as f64;
    (0..size)
        .map(|i| 0.5 - 0.5 * (i as f64 * factor).cos())
        .collect()
}

/// Compute STFT power spectrogram.
fn stft_whisper(samples: &[f32], n_fft: usize, hop_length: usize) -> Vec<Vec<f64>> {
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

        let frame_power: Vec<f64> = fft_input[..n_freqs]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        power_spec.push(frame_power);
    }

    power_spec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slaney_mel_scale_roundtrip() {
        for freq in [100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel_slaney(freq);
            let freq_back = mel_to_hz_slaney(mel);
            assert!((freq - freq_back).abs() < 0.001, "Failed for freq {}", freq);
        }
    }

    #[test]
    fn test_hann_window() {
        let window = create_hann_window(400);
        assert_eq!(window.len(), 400);
        assert!((window[0] - 0.0).abs() < 0.001);
        assert!((window[200] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mel_filter_bank_shape() {
        let filters = create_mel_filter_bank_slaney(80, 400, 16000, 0.0, 8000.0);
        assert_eq!(filters.len(), 80);
        assert_eq!(filters[0].len(), 201); // n_fft/2 + 1
    }
}
