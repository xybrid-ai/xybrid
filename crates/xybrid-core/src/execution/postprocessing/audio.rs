//! Audio postprocessing operations.
//!
//! This module provides:
//! - `tts_audio_encode_step`: Convert TTS waveform tensor to audio bytes
//! - `crossfade_audio_chunks`: Concatenate audio chunks with linear crossfading

use super::super::modes::tts_waveform_output;
use super::super::types::{ExecutorResult, RawOutputs};
use crate::runtime_adapter::AdapterError;

/// Convert TTS waveform tensor to audio bytes.
///
/// Returns [`RawOutputs::TtsAudio`]: the 16-bit PCM, plus the range of the
/// waveform's samples it holds once silence trimming is done (`0..len` when
/// nothing is trimmed). PCM sample `i` is waveform sample `kept.start + i`.
///
/// # Arguments
/// - `data`: Input data (TensorMap with waveform tensor)
/// - `sample_rate`: Output sample rate
/// - `apply_postprocessing`: Whether to apply audio postprocessing (normalization, etc.)
/// - `trim_trailing_silence`: Whether to trim trailing near-silence from the waveform
///
/// # Errors
///
/// Returns [`AdapterError::InvalidInput`] if `data` is not a tensor map, the
/// map is empty, it holds several tensors and none is named as a waveform
/// (see [`tts_waveform_output`]), or the waveform is not contiguous.
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
    if tensor_map.is_empty() {
        return Err(AdapterError::InvalidInput(
            "No waveform output for TTS".to_string(),
        ));
    }

    // Get the waveform tensor: the only one, or the one named as a waveform.
    // Never an arbitrary entry, which could be per-phoneme durations.
    let mut names: Vec<String> = tensor_map.keys().cloned().collect();
    names.sort();
    let waveform_name = tts_waveform_output(&names)?;
    let waveform = tensor_map
        .get(waveform_name)
        .ok_or_else(|| AdapterError::InvalidInput("No waveform output for TTS".to_string()))?;

    // Convert tensor to f32 samples
    let samples: Vec<f32> = waveform
        .as_slice()
        .ok_or_else(|| AdapterError::InvalidInput("Waveform tensor not contiguous".to_string()))?
        .to_vec();

    // Apply postprocessing if enabled
    let (mut processed_samples, mut kept) = if apply_postprocessing {
        use crate::phonemizer::postprocess_tts_audio_with_range;
        postprocess_tts_audio_with_range(&samples, sample_rate)
    } else {
        let len = samples.len();
        (samples, 0..len)
    };

    // Trim trailing silence if enabled
    if trim_trailing_silence {
        let end = trailing_near_silence_end(&processed_samples, sample_rate);
        processed_samples.truncate(end);
        kept.end = kept.start + end;
    }

    // Convert f32 samples to 16-bit PCM bytes
    let audio_bytes = samples_to_pcm16(&processed_samples);

    Ok(RawOutputs::TtsAudio {
        pcm: audio_bytes,
        kept,
    })
}

/// Trim trailing near-silence from the end of a waveform.
///
/// Scans from the end and finds where sustained silence ends. Silence is defined
/// as absolute amplitude below a threshold for a sustained period (>50ms).
fn trim_trailing_near_silence(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    samples[..trailing_near_silence_end(samples, sample_rate)].to_vec()
}

/// How many leading samples [`trim_trailing_near_silence`] keeps: the
/// waveform's length when there is no sustained trailing silence to trim.
fn trailing_near_silence_end(samples: &[f32], sample_rate: u32) -> usize {
    const SILENCE_THRESHOLD: f32 = 0.01;
    // Minimum sustained silence duration to consider for trimming: 50ms
    let min_silence_samples = (sample_rate as usize * 50) / 1000;

    if samples.len() <= min_silence_samples {
        return samples.len();
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
        last_non_silent
    } else {
        samples.len()
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
    crossfade_audio_chunks_with_starts(chunks, crossfade_len).0
}

/// [`crossfade_audio_chunks`], also returning where each chunk starts in the
/// joined audio (one entry per chunk).
///
/// The first chunk starts at 0. A later chunk starts `crossfade_len` samples
/// before the end of the audio joined so far when it is crossfaded in (both
/// that audio and the chunk are at least `2 * crossfade_len` long), and at the
/// end of it otherwise.
pub(crate) fn crossfade_audio_chunks_with_starts(
    chunks: &[Vec<f32>],
    crossfade_len: usize,
) -> (Vec<f32>, Vec<usize>) {
    let Some((first, rest)) = chunks.split_first() else {
        return (Vec::new(), Vec::new());
    };

    // Start with the first chunk
    let mut result = first.clone();
    let mut starts = Vec::with_capacity(chunks.len());
    starts.push(0);

    for chunk in rest {
        // Skip crossfading if either the current result tail or the new chunk head
        // is too short for the crossfade
        if result.len() < 2 * crossfade_len || chunk.len() < 2 * crossfade_len {
            starts.push(result.len());
            result.extend(chunk);
            continue;
        }

        let overlap_start = result.len() - crossfade_len;
        starts.push(overlap_start);

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

    (result, starts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    use std::collections::HashMap;
    use std::ops::Range;

    /// A 220 Hz tone at 24 kHz between `lead` and `tail` silent samples.
    fn padded_sine(lead: usize, voiced: usize, tail: usize) -> Vec<f32> {
        let step = 2.0 * std::f32::consts::PI * 220.0 / 24_000.0;
        let mut samples = vec![0.0; lead];
        samples.extend((0..voiced).map(|i| 0.5 * (i as f32 * step).sin()));
        samples.extend(vec![0.0; tail]);
        samples
    }

    fn tensor_map(entries: &[(&str, &[f32])]) -> RawOutputs {
        RawOutputs::TensorMap(
            entries
                .iter()
                .map(|(name, values)| {
                    let array = ndarray::Array1::from_vec(values.to_vec()).into_dyn();
                    (name.to_string(), array)
                })
                .collect::<HashMap<String, ArrayD<f32>>>(),
        )
    }

    fn encode(data: RawOutputs, postprocess: bool, trim_tail: bool) -> (Vec<u8>, Range<usize>) {
        match tts_audio_encode_step(data, 24_000, postprocess, trim_tail).unwrap() {
            RawOutputs::TtsAudio { pcm, kept } => (pcm, kept),
            other => panic!("expected TtsAudio, got {other:?}"),
        }
    }

    #[test]
    fn encode_output_is_byte_identical_to_the_previous_pipeline() {
        use crate::phonemizer::postprocess_tts_audio;
        let samples = padded_sine(4800, 12_000, 7200);
        for (postprocess, trim_tail) in [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut expected = if postprocess {
                postprocess_tts_audio(&samples, 24_000)
            } else {
                samples.clone()
            };
            if trim_tail {
                expected = trim_trailing_near_silence(&expected, 24_000);
            }

            let (pcm, kept) = encode(tensor_map(&[("audio", &samples)]), postprocess, trim_tail);
            assert_eq!(
                pcm,
                samples_to_pcm16(&expected),
                "{postprocess} {trim_tail}"
            );
            assert_eq!(pcm.len(), kept.len() * 2, "{postprocess} {trim_tail}");
        }
    }

    #[test]
    fn encode_without_trimming_keeps_every_sample() {
        let samples = padded_sine(4800, 12_000, 7200);
        let (_, kept) = encode(tensor_map(&[("waveform", &samples)]), false, false);
        assert_eq!(kept, 0..samples.len());
    }

    #[test]
    fn encode_reports_the_kept_range_after_head_and_tail_trim() {
        let samples = padded_sine(4800, 12_000, 7200);

        let (_, head_and_tail) = encode(tensor_map(&[("audio", &samples)]), true, false);
        assert!(head_and_tail.start > 0, "{head_and_tail:?}");
        assert!(head_and_tail.end < samples.len(), "{head_and_tail:?}");

        // Trailing trimming only ever shortens what postprocessing kept.
        let (_, both) = encode(tensor_map(&[("audio", &samples)]), true, true);
        assert_eq!(both.start, head_and_tail.start);
        assert!(both.end <= head_and_tail.end);

        // Without postprocessing, only the tail goes.
        let (_, tail_only) = encode(tensor_map(&[("audio", &samples)]), false, true);
        assert_eq!(tail_only.start, 0);
        assert!(tail_only.end < samples.len(), "{tail_only:?}");
        assert!(tail_only.end >= 4800 + 12_000, "{tail_only:?}");
    }

    #[test]
    fn encode_picks_the_waveform_by_name_among_several_outputs() {
        let samples = padded_sine(0, 2400, 0);
        let durations = [3.0, 4.0, 5.0];
        for waveform_name in ["audio", "waveform"] {
            let (pcm, kept) = encode(
                tensor_map(&[("duration", &durations), (waveform_name, &samples)]),
                false,
                false,
            );
            assert_eq!(pcm, samples_to_pcm16(&samples), "{waveform_name}");
            assert_eq!(kept, 0..samples.len());
        }
    }

    #[test]
    fn encode_refuses_to_guess_the_waveform() {
        let ambiguous = tensor_map(&[("duration", &[3.0, 4.0]), ("pitch", &[0.1, 0.2])]);
        assert!(matches!(
            tts_audio_encode_step(ambiguous, 24_000, false, false),
            Err(AdapterError::InvalidInput(_))
        ));
        assert!(matches!(
            tts_audio_encode_step(tensor_map(&[]), 24_000, false, false),
            Err(AdapterError::InvalidInput(_))
        ));
    }

    #[test]
    fn trailing_near_silence_end_matches_the_trim() {
        let mut samples = vec![0.5f32; 24_000];
        samples.extend(vec![0.0f32; 2400]);
        for candidate in [samples.clone(), vec![0.5f32; 24_000], vec![0.0f32; 100]] {
            let end = trailing_near_silence_end(&candidate, 24_000);
            assert_eq!(
                candidate[..end].to_vec(),
                trim_trailing_near_silence(&candidate, 24_000)
            );
        }
        // 1 s of tone keeps a 10 ms (240-sample) buffer of the silence.
        assert_eq!(trailing_near_silence_end(&samples, 24_000), 24_240);
    }

    #[test]
    fn crossfade_reports_chunk_starts_including_short_chunk_skip() {
        let crossfade_len = 4;
        let chunks = vec![
            vec![1.0; 10],
            vec![0.5; 6], // shorter than 2 * crossfade_len: appended, no crossfade
            vec![0.25; 10],
            vec![0.0; 10],
        ];

        let (joined, starts) = crossfade_audio_chunks_with_starts(&chunks, crossfade_len);

        assert_eq!(joined, crossfade_audio_chunks(&chunks, crossfade_len));
        // 0; then the end of the first chunk (10); then 4 before the end of
        // 16 samples (12); then 4 before the end of 22 samples (18).
        assert_eq!(starts, vec![0, 10, 12, 18]);
        assert_eq!(joined.len(), 28);
        // Outside the crossfades, sample `start + i` is the chunk's sample `i`.
        let constant = |range: Range<usize>, value: f32| {
            for at in range {
                assert_eq!(joined[at], value, "sample {at}");
            }
        };
        constant(0..10, 1.0); // the whole first chunk: no crossfade into chunk 2
        constant(10..12, 0.5); // chunk 2 until chunk 3 fades in at 12
        constant(16..18, 0.25); // chunk 3 between its two crossfades
        constant(22..28, 0.0); // chunk 4 after its crossfade
    }

    #[test]
    fn crossfade_starts_for_no_chunks_and_one_chunk() {
        let (joined, starts) = crossfade_audio_chunks_with_starts(&[], 480);
        assert!(joined.is_empty() && starts.is_empty());

        let chunk = vec![1.0, 2.0, 3.0];
        let (joined, starts) =
            crossfade_audio_chunks_with_starts(std::slice::from_ref(&chunk), 480);
        assert_eq!(joined, chunk);
        assert_eq!(starts, vec![0]);
    }

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
