//! Audio postprocessing operations.
//!
//! This module provides:
//! - `tts_audio_encode_step`: Convert TTS waveform tensor to audio bytes
//! - `crossfade_audio_chunks`: Concatenate audio chunks with linear crossfading

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

/// Concatenate audio chunks with linear crossfading at boundaries.
///
/// Applies a linear crossfade of `crossfade_len` samples between adjacent chunks.
/// Single-chunk input is returned as-is. Chunks shorter than `2 * crossfade_len`
/// skip crossfading for that boundary (safety guard).
pub(crate) fn crossfade_audio_chunks(chunks: &[Vec<f32>], crossfade_len: usize) -> Vec<f32> {
    if chunks.is_empty() {
        return Vec::new();
    }
    if chunks.len() == 1 {
        return chunks[0].clone();
    }

    // Start with the first chunk
    let mut result = chunks[0].clone();

    for chunk in &chunks[1..] {
        // Skip crossfading if either the current result tail or the new chunk head
        // is too short for the crossfade
        if result.len() < 2 * crossfade_len || chunk.len() < 2 * crossfade_len {
            result.extend(chunk);
            continue;
        }

        let overlap_start = result.len() - crossfade_len;

        // Apply crossfade in the overlap region
        for i in 0..crossfade_len {
            let t = (i + 1) as f32 / (crossfade_len + 1) as f32;
            let fade_out = 1.0 - t;
            let fade_in = t;
            result[overlap_start + i] = result[overlap_start + i] * fade_out + chunk[i] * fade_in;
        }

        // Append the rest of the new chunk (after the overlap region)
        result.extend_from_slice(&chunk[crossfade_len..]);
    }

    result
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

    #[test]
    fn test_crossfade_empty_chunks() {
        let chunks: Vec<Vec<f32>> = vec![];
        let result = crossfade_audio_chunks(&chunks, 480);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crossfade_single_chunk_unchanged() {
        let chunk = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = crossfade_audio_chunks(std::slice::from_ref(&chunk), 480);
        assert_eq!(result, chunk);
    }

    #[test]
    fn test_crossfade_two_chunks() {
        let crossfade_len = 4;
        // Chunk A: 10 samples of 1.0
        let chunk_a = vec![1.0; 10];
        // Chunk B: 10 samples of 0.0
        let chunk_b = vec![0.0; 10];

        let result = crossfade_audio_chunks(&[chunk_a, chunk_b], crossfade_len);

        // Result length: 10 + 10 - 4 (overlap) = 16
        assert_eq!(result.len(), 16);

        // First 6 samples: unchanged from chunk_a (before overlap)
        for &v in &result[..6] {
            assert!((v - 1.0).abs() < 1e-6);
        }

        // Overlap region (4 samples): linear blend from 1.0 to 0.0
        // t = (i+1) / (crossfade_len+1), fade_out = 1-t, fade_in = t
        // result[6+i] = 1.0 * (1-t) + 0.0 * t = 1-t
        for i in 0..crossfade_len {
            let t = (i + 1) as f32 / (crossfade_len + 1) as f32;
            let expected = 1.0 - t;
            assert!(
                (result[6 + i] - expected).abs() < 1e-6,
                "at overlap index {i}: got {}, expected {expected}",
                result[6 + i]
            );
        }

        // Last 6 samples: from chunk_b after overlap
        for &v in &result[10..] {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_crossfade_three_chunks() {
        let crossfade_len = 2;
        let chunk_a = vec![1.0; 8];
        let chunk_b = vec![0.5; 8];
        let chunk_c = vec![0.0; 8];

        let result = crossfade_audio_chunks(&[chunk_a, chunk_b, chunk_c], crossfade_len);

        // Length: 8 + (8 - 2) + (8 - 2) = 20
        assert_eq!(result.len(), 20);
    }

    #[test]
    fn test_crossfade_short_chunk_skips_crossfade() {
        let crossfade_len = 4;
        // Chunk too short (len < 2 * crossfade_len = 8)
        let chunk_a = vec![1.0; 10];
        let chunk_b = vec![0.5; 6]; // Too short for crossfade

        let result = crossfade_audio_chunks(&[chunk_a, chunk_b], crossfade_len);

        // Should be simple concatenation (no crossfade)
        assert_eq!(result.len(), 16);
        assert!((result[9] - 1.0).abs() < 1e-6);
        assert!((result[10] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_preserves_total_energy() {
        // When both chunks have the same constant value, crossfade should preserve it
        let crossfade_len = 4;
        let chunk_a = vec![0.5; 10];
        let chunk_b = vec![0.5; 10];

        let result = crossfade_audio_chunks(&[chunk_a, chunk_b], crossfade_len);

        // In the overlap region: 0.5 * fade_out + 0.5 * fade_in = 0.5 * (fade_out + fade_in) = 0.5
        // since fade_out + fade_in = 1.0
        for &v in &result {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "expected 0.5, got {v} — crossfade should preserve constant signal"
            );
        }
    }
}
