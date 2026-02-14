//! Audio postprocessing operations.
//!
//! This module provides:
//! - `tts_audio_encode_step`: Convert TTS waveform tensor to audio bytes

use super::super::types::{ExecutorResult, RawOutputs};
use crate::runtime_adapter::AdapterError;

/// Convert TTS waveform tensor to audio bytes.
///
/// # Arguments
/// - `data`: Input data (TensorMap with waveform tensor)
/// - `sample_rate`: Output sample rate
/// - `apply_postprocessing`: Whether to apply audio postprocessing (normalization, etc.)
/// - `trim_trailing_silence`: Whether to trim trailing near-silence from the waveform
pub fn tts_audio_encode_step(
    data: RawOutputs,
    sample_rate: u32,
    apply_postprocessing: bool,
    trim_trailing_silence: bool,
) -> ExecutorResult<RawOutputs> {
    let tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "TTSAudioEncode requires tensor map".to_string(),
            ))
        }
    };

    // Get the waveform tensor (usually named "waveform" or first output)
    let waveform = tensor_map
        .get("waveform")
        .or_else(|| tensor_map.values().next())
        .ok_or_else(|| AdapterError::InvalidInput("No waveform output for TTS".to_string()))?;

    // Convert tensor to f32 samples
    let samples: Vec<f32> = waveform
        .as_slice()
        .ok_or_else(|| AdapterError::InvalidInput("Waveform tensor not contiguous".to_string()))?
        .to_vec();

    // Apply postprocessing if enabled
    let mut processed_samples = if apply_postprocessing {
        use crate::phonemizer::postprocess_tts_audio;
        postprocess_tts_audio(&samples, sample_rate)
    } else {
        samples
    };

    // Trim trailing silence if enabled
    if trim_trailing_silence {
        processed_samples = trim_trailing_near_silence(&processed_samples, sample_rate);
    }

    // Convert f32 samples to 16-bit PCM bytes
    let audio_bytes = samples_to_pcm16(&processed_samples);

    Ok(RawOutputs::AudioBytes(audio_bytes))
}

/// Trim trailing near-silence from the end of a waveform.
///
/// Scans from the end and finds where sustained silence ends. Silence is defined
/// as absolute amplitude below a threshold for a sustained period (>50ms).
fn trim_trailing_near_silence(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    const SILENCE_THRESHOLD: f32 = 0.01;
    // Minimum sustained silence duration to consider for trimming: 50ms
    let min_silence_samples = (sample_rate as usize * 50) / 1000;

    if samples.len() <= min_silence_samples {
        return samples.to_vec();
    }

    // Scan from the end to find where non-silence begins
    let mut last_non_silent = samples.len();
    let mut silence_run = 0;

    for i in (0..samples.len()).rev() {
        if samples[i].abs() < SILENCE_THRESHOLD {
            silence_run += 1;
        } else {
            if silence_run >= min_silence_samples {
                // Keep a small buffer after the last non-silent sample (~10ms)
                let fade_buffer = (sample_rate as usize * 10) / 1000;
                last_non_silent = (i + 1 + fade_buffer).min(samples.len());
            }
            break;
        }
    }

    if silence_run >= min_silence_samples {
        samples[..last_non_silent].to_vec()
    } else {
        samples.to_vec()
    }
}

/// Convert f32 audio samples to 16-bit PCM bytes.
fn samples_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);

    for &sample in samples {
        // Clamp to [-1.0, 1.0] and convert to i16
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm16 = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&pcm16.to_le_bytes());
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_trailing_silence_removes_silence() {
        let sample_rate = 24000;
        // 1 second of audio + 100ms of silence (2400 samples at 24kHz)
        let mut samples = vec![0.5f32; sample_rate as usize];
        samples.extend(vec![0.0f32; 2400]);

        let trimmed = trim_trailing_near_silence(&samples, sample_rate);
        assert!(
            trimmed.len() < samples.len(),
            "Trailing silence should be trimmed: original={}, trimmed={}",
            samples.len(),
            trimmed.len()
        );
    }

    #[test]
    fn test_trim_trailing_silence_preserves_short_silence() {
        let sample_rate = 24000;
        // Audio + 20ms of silence (480 samples, below 50ms threshold)
        let mut samples = vec![0.5f32; sample_rate as usize];
        samples.extend(vec![0.0f32; 480]);

        let trimmed = trim_trailing_near_silence(&samples, sample_rate);
        assert_eq!(
            trimmed.len(),
            samples.len(),
            "Short trailing silence should not be trimmed"
        );
    }

    #[test]
    fn test_trim_trailing_silence_no_change_when_no_silence() {
        let sample_rate = 24000;
        let samples = vec![0.5f32; sample_rate as usize];

        let trimmed = trim_trailing_near_silence(&samples, sample_rate);
        assert_eq!(trimmed.len(), samples.len());
    }
}
