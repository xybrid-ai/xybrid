//! Mel spectrogram computation.
//!
//! This module provides mel spectrogram computation with support for different
//! mel scales. The main entry point is [`compute_mel_spectrogram`].
//!
//! ## Architecture
//!
//! ```text
//! compute_mel_spectrogram(samples, config)
//!           │
//!           ├── config.mel_scale == Slaney
//!           │         └── slaney::create_filter_bank()
//!           │
//!           └── config.mel_scale == Htk
//!                     └── htk::create_filter_bank()
//!
//! common::apply_padding()
//! common::compute_stft_power()
//! common::apply_log_normalization()
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::audio::mel::{compute_mel_spectrogram, MelConfig};
//!
//! // Using preset
//! let config = MelConfig::whisper();
//! let mel = compute_mel_spectrogram(&samples, &config)?;
//!
//! // Or with explicit parameters
//! let config = MelConfig {
//!     n_mels: 80,
//!     mel_scale: MelScale::Slaney,
//!     ..Default::default()
//! };
//! let mel = compute_mel_spectrogram(&samples, &config)?;
//! ```

pub mod common;
pub mod config;
pub mod htk;
pub mod slaney;

// Legacy module for backwards compatibility
#[allow(clippy::module_inception)]
pub mod whisper;

// Re-export main types
pub use config::{MelConfig, MelScale, PaddingMode};

// Legacy re-exports for backwards compatibility
pub use whisper::{compute_whisper_mel, WhisperMelConfig};

use common::{apply_log_normalization, apply_padding, compute_stft_power};
use ndarray::{ArrayD, IxDyn};

/// Compute mel spectrogram from audio samples.
///
/// This is the main entry point for mel spectrogram computation. It routes
/// to the appropriate implementation based on the mel scale in the config.
///
/// # Arguments
/// * `audio_samples` - PCM audio samples (f32, normalized to [-1.0, 1.0])
/// * `config` - Mel spectrogram configuration
///
/// # Returns
/// Mel spectrogram tensor with shape [1, n_mels, frames]
pub fn compute_mel_spectrogram(
    audio_samples: &[f32],
    config: &MelConfig,
) -> Result<ArrayD<f32>, String> {
    if audio_samples.is_empty() {
        return Err("Cannot compute mel spectrogram from empty audio".to_string());
    }

    // Apply padding
    let padded = apply_padding(audio_samples, config.pad_size(), config.padding);

    // Create mel filter bank based on mel scale
    let mel_filters = match config.mel_scale {
        MelScale::Slaney => slaney::create_filter_bank(
            config.n_mels,
            config.n_fft,
            config.sample_rate,
            config.f_min,
            config.effective_f_max(),
        ),
        MelScale::Htk => htk::create_filter_bank(
            config.n_mels,
            config.n_fft,
            config.sample_rate,
            config.f_min,
            config.effective_f_max(),
        ),
    };

    // Compute STFT power spectrogram
    let power_spec = compute_stft_power(&padded, config.n_fft, config.hop_length);

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

    // Apply normalization
    let mel_data = if config.normalize {
        apply_log_normalization(&mut mel_spec, config.max_frames)
    } else {
        // Just flatten without normalization
        let output_frames = config.max_frames.unwrap_or(n_frames);
        let mut data = Vec::with_capacity(config.n_mels * output_frames);
        for mel_idx in 0..config.n_mels {
            for frame_idx in 0..output_frames {
                let val = if frame_idx < n_frames {
                    mel_spec[mel_idx][frame_idx] as f32
                } else {
                    0.0
                };
                data.push(val);
            }
        }
        data
    };

    // Create tensor with shape [batch, n_mels, time_frames]
    let output_frames = config.max_frames.unwrap_or(n_frames);
    let mel_shape = vec![1, config.n_mels, output_frames];
    ArrayD::<f32>::from_shape_vec(IxDyn(&mel_shape), mel_data)
        .map_err(|e| format!("Failed to create mel tensor: {:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mel_spectrogram_whisper() {
        // 1 second of silence at 16kHz
        let samples = vec![0.0f32; 16000];
        let config = MelConfig::whisper();

        let result = compute_mel_spectrogram(&samples, &config);
        assert!(result.is_ok());

        let mel = result.unwrap();
        assert_eq!(mel.shape(), &[1, 80, 3000]);
    }

    #[test]
    fn test_compute_mel_spectrogram_htk() {
        let samples = vec![0.0f32; 16000];
        let config = MelConfig {
            mel_scale: MelScale::Htk,
            max_frames: Some(100), // Smaller for test
            ..Default::default()
        };

        let result = compute_mel_spectrogram(&samples, &config);
        assert!(result.is_ok());

        let mel = result.unwrap();
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], 80);
    }

    #[test]
    fn test_compute_mel_spectrogram_empty() {
        let samples: Vec<f32> = vec![];
        let config = MelConfig::whisper();

        let result = compute_mel_spectrogram(&samples, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_different_scales_produce_different_results() {
        // Generate a simple sine wave
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let slaney_config = MelConfig {
            mel_scale: MelScale::Slaney,
            max_frames: Some(100),
            ..Default::default()
        };

        let htk_config = MelConfig {
            mel_scale: MelScale::Htk,
            max_frames: Some(100),
            ..Default::default()
        };

        let slaney_result = compute_mel_spectrogram(&samples, &slaney_config).unwrap();
        let htk_result = compute_mel_spectrogram(&samples, &htk_config).unwrap();

        // Results should be different
        let slaney_sum: f32 = slaney_result.iter().sum();
        let htk_sum: f32 = htk_result.iter().sum();

        assert!(
            (slaney_sum - htk_sum).abs() > 0.1,
            "Slaney and HTK should produce different results"
        );
    }
}
