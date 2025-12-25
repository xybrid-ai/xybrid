//! Audio postprocessing operations.
//!
//! This module provides:
//! - `tts_audio_encode_step`: Convert TTS waveform tensor to audio bytes

use crate::runtime_adapter::AdapterError;
use super::super::types::{ExecutorResult, RawOutputs};

/// Convert TTS waveform tensor to audio bytes.
///
/// # Arguments
/// - `data`: Input data (TensorMap with waveform tensor)
/// - `sample_rate`: Output sample rate
/// - `apply_postprocessing`: Whether to apply audio postprocessing (normalization, etc.)
pub fn tts_audio_encode_step(
    data: RawOutputs,
    sample_rate: u32,
    apply_postprocessing: bool,
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
    let processed_samples = if apply_postprocessing {
        use crate::phonemizer::postprocess_tts_audio;
        postprocess_tts_audio(&samples, sample_rate)
    } else {
        samples
    };

    // Convert f32 samples to 16-bit PCM bytes
    let audio_bytes = samples_to_pcm16(&processed_samples);

    Ok(RawOutputs::AudioBytes(audio_bytes))
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
