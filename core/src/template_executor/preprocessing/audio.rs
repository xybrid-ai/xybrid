//! Audio preprocessing operations.
//!
//! This module provides:
//! - `mel_spectrogram_step`: Convert audio samples to mel spectrogram
//! - `decode_audio_step`: Decode audio from various formats to float32 samples

use crate::audio::mel::{compute_mel_spectrogram, MelConfig, MelScale, PaddingMode};
use crate::audio::{decode_wav_audio, prepare_audio_samples};
use crate::execution_template::MelScaleType;
use crate::ir::Envelope;
use crate::preprocessing::mel_spectrogram::audio_bytes_to_whisper_mel;
use crate::runtime_adapter::AdapterError;
use super::super::types::{ExecutorResult, PreprocessedData};

/// Convert audio samples to mel spectrogram.
///
/// # Arguments
/// - `data`: Input data (AudioSamples or AudioBytes)
/// - `preset`: Optional preset name (e.g., "whisper", "whisper-large")
/// - `n_mels`: Number of mel frequency bins
/// - `sample_rate`: Target sample rate in Hz
/// - `fft_size`: FFT window size
/// - `hop_length`: Hop length between frames
/// - `mel_scale`: Mel frequency scale (Slaney or HTK)
/// - `max_frames`: Maximum number of output frames
///
/// # Presets
/// If a preset is specified, it overrides the individual parameters.
/// Available presets: "whisper", "whisper-large"
pub fn mel_spectrogram_step(
    data: PreprocessedData,
    preset: Option<&str>,
    n_mels: usize,
    sample_rate: u32,
    fft_size: usize,
    hop_length: usize,
    mel_scale: MelScaleType,
    max_frames: Option<usize>,
) -> ExecutorResult<PreprocessedData> {
    // Build MelConfig from preset or individual parameters
    let config = if let Some(preset_name) = preset {
        MelConfig::from_preset(preset_name).unwrap_or_else(|| {
            eprintln!(
                "[WARN] Unknown mel spectrogram preset '{}', using parameters",
                preset_name
            );
            build_mel_config(n_mels, sample_rate, fft_size, hop_length, mel_scale, max_frames)
        })
    } else {
        build_mel_config(n_mels, sample_rate, fft_size, hop_length, mel_scale, max_frames)
    };

    match data {
        PreprocessedData::AudioSamples(samples) => {
            let mel = compute_mel_spectrogram(&samples, &config)
                .map_err(|e| AdapterError::InvalidInput(e))?;
            Ok(PreprocessedData::Tensor(mel))
        }
        PreprocessedData::AudioBytes(bytes) => {
            // For audio bytes, we need to parse WAV first
            // Use the preprocessing module's audio_bytes_to_whisper_mel for now
            let mel = audio_bytes_to_whisper_mel(&bytes)?;
            Ok(PreprocessedData::Tensor(mel))
        }
        _ => Err(AdapterError::InvalidInput(
            "MelSpectrogram requires audio samples or bytes input".to_string(),
        )),
    }
}

/// Build MelConfig from individual parameters.
fn build_mel_config(
    n_mels: usize,
    sample_rate: u32,
    fft_size: usize,
    hop_length: usize,
    mel_scale: MelScaleType,
    max_frames: Option<usize>,
) -> MelConfig {
    MelConfig {
        n_mels,
        n_fft: fft_size,
        hop_length,
        sample_rate,
        mel_scale: match mel_scale {
            MelScaleType::Slaney => MelScale::Slaney,
            MelScaleType::Htk => MelScale::Htk,
        },
        f_min: 0.0,
        f_max: (sample_rate / 2) as f64, // Nyquist
        padding: PaddingMode::Reflect,
        max_frames,
        normalize: true,
    }
}

/// Decode audio from various formats to float32 samples.
///
/// Supports:
/// - Pre-decoded float32 samples (from AudioEnvelope, format="float32")
/// - Raw PCM16 bytes (format="pcm16")
/// - WAV files (format="wav" or unspecified)
///
/// After decoding, converts to target sample rate and channel count.
pub fn decode_audio_step(
    data: PreprocessedData,
    input_envelope: &Envelope,
    target_sample_rate: u32,
    target_channels: usize,
) -> ExecutorResult<PreprocessedData> {
    // First, try to use Envelope's format-aware extraction
    if let Ok(Some(audio_samples)) = input_envelope.to_audio_samples() {
        // Audio was already decoded (float32 or pcm16 format)
        let prepared = prepare_audio_samples(
            audio_samples.samples,
            audio_samples.sample_rate,
            audio_samples.channels as usize,
            target_sample_rate,
            target_channels,
        );
        return Ok(PreprocessedData::AudioSamples(prepared));
    }

    // Fallback: decode from raw bytes (WAV format)
    let audio_bytes = match data {
        PreprocessedData::AudioBytes(bytes) => bytes,
        _ => {
            return Err(AdapterError::InvalidInput(
                "AudioDecode requires audio bytes input".to_string(),
            ))
        }
    };

    let samples = decode_wav_audio(&audio_bytes, target_sample_rate, target_channels)
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to decode audio: {}", e)))?;

    Ok(PreprocessedData::AudioSamples(samples))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_spectrogram_step_basic() {
        // Create a simple sine wave as audio input
        let sample_rate = 16000;
        let duration_secs = 0.5;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let frequency = 440.0; // A4 note

        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let data = PreprocessedData::AudioSamples(samples);
        let result = mel_spectrogram_step(
            data,
            Some("whisper"),
            80,
            16000,
            400,
            160,
            MelScaleType::Slaney,
            Some(3000),
        );

        assert!(result.is_ok());
        match result.unwrap() {
            PreprocessedData::Tensor(tensor) => {
                let shape = tensor.shape();
                assert_eq!(shape.len(), 3);
                assert_eq!(shape[0], 1); // batch
                assert_eq!(shape[1], 80); // n_mels
                assert!(shape[2] > 0); // time frames
            }
            _ => panic!("Expected Tensor output from mel_spectrogram_step"),
        }
    }

    #[test]
    fn test_mel_spectrogram_step_invalid_input() {
        let data = PreprocessedData::Text("invalid".to_string());
        let result = mel_spectrogram_step(
            data,
            None,
            80,
            16000,
            400,
            160,
            MelScaleType::Slaney,
            Some(3000),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_mel_spectrogram_step_empty_audio() {
        let data = PreprocessedData::AudioSamples(vec![]);
        let result = mel_spectrogram_step(
            data,
            None,
            80,
            16000,
            400,
            160,
            MelScaleType::Slaney,
            Some(3000),
        );

        // Empty audio produces an error (cannot compute spectrogram without samples)
        assert!(result.is_err());
    }
}
