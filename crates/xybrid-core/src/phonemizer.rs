//! TTS audio postprocessing and token mapping utilities.
//!
//! This module provides:
//! - `load_tokens_map`: Parse tokens.txt files mapping IPA characters to token IDs
//! - `postprocess_tts_audio`: Full audio postprocessing pipeline (high-pass → trim → normalize)
//! - Individual audio utilities: `normalize_loudness`, `trim_silence`, `high_pass_filter`
//!
//! The `_range` variants (`trim_silence_range`, `postprocess_tts_audio_with_range`)
//! also report which input samples survived trimming, so timings measured on
//! the input can be moved onto the output.

use std::collections::HashMap;
use std::ops::Range;

// ============================================================================
// Audio Postprocessing Utilities
// ============================================================================

/// Normalize audio to target loudness (simple RMS-based normalization)
///
/// # Arguments
/// * `samples` - Audio samples (float32, -1.0 to 1.0)
/// * `target_rms` - Target RMS level (e.g., 0.1 for speech)
///
/// # Returns
/// Normalized audio samples
pub fn normalize_loudness(samples: &[f32], target_rms: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    // Calculate current RMS
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    let current_rms = (sum_sq / samples.len() as f32).sqrt();

    if current_rms < 1e-10 {
        // Silence - return as-is
        return samples.to_vec();
    }

    // Calculate gain
    let gain = target_rms / current_rms;

    // Apply gain with soft clipping to avoid harsh distortion
    samples
        .iter()
        .map(|s| {
            let amplified = s * gain;
            // Soft clip using tanh for values approaching limits
            if amplified.abs() > 0.95 {
                amplified.signum() * (0.95 + 0.05 * (amplified.abs() - 0.95).tanh())
            } else {
                amplified
            }
        })
        .collect()
}

/// Trim silence from the beginning and end of audio
///
/// # Arguments
/// * `samples` - Audio samples
/// * `threshold_db` - Silence threshold in dB (e.g., -40.0)
/// * `min_silence_samples` - Minimum silence duration to trim (in samples)
///
/// # Returns
/// Trimmed audio samples: `samples[trim_silence_range(..)]`
pub fn trim_silence(samples: &[f32], threshold_db: f32, min_silence_samples: usize) -> Vec<f32> {
    samples[trim_silence_range(samples, threshold_db, min_silence_samples)].to_vec()
}

/// The range of `samples` that [`trim_silence`] keeps.
///
/// Leading and trailing samples at or below `threshold_db` are cut, keeping
/// `min_silence_samples / 4` samples of lead-in and tail. All-silent audio is
/// kept whole (`0..len`), and empty audio gives `0..0`.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `threshold_db` - Silence threshold in dB (e.g., -40.0)
/// * `min_silence_samples` - Minimum silence duration to trim (in samples)
pub fn trim_silence_range(
    samples: &[f32],
    threshold_db: f32,
    min_silence_samples: usize,
) -> Range<usize> {
    if samples.is_empty() {
        return 0..0;
    }

    // Convert dB threshold to linear amplitude
    let threshold = 10.0_f32.powf(threshold_db / 20.0);

    // Find first non-silent sample
    let mut start = 0;
    for (i, &sample) in samples.iter().enumerate() {
        if sample.abs() > threshold {
            start = i.saturating_sub(min_silence_samples / 4); // Keep a bit of lead-in
            break;
        }
    }

    // Find last non-silent sample
    let mut end = samples.len();
    for (i, &sample) in samples.iter().enumerate().rev() {
        if sample.abs() > threshold {
            end = (i + min_silence_samples / 4).min(samples.len()); // Keep a bit of tail
            break;
        }
    }

    if start >= end {
        // All silence or invalid range
        return 0..samples.len();
    }

    start..end
}

/// Apply a simple high-pass filter to remove low-frequency rumble
///
/// # Arguments
/// * `samples` - Audio samples
/// * `cutoff_hz` - Cutoff frequency in Hz
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Filtered audio samples
pub fn high_pass_filter(samples: &[f32], cutoff_hz: f32, sample_rate: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    // Simple 1st-order high-pass filter (RC filter approximation)
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let dt = 1.0 / sample_rate;
    let alpha = rc / (rc + dt);

    let mut output = Vec::with_capacity(samples.len());
    let mut prev_input = samples[0];
    let mut prev_output = 0.0_f32;

    for &sample in samples.iter() {
        let filtered = alpha * (prev_output + sample - prev_input);
        output.push(filtered);
        prev_input = sample;
        prev_output = filtered;
    }

    output
}

/// Full audio postprocessing pipeline for TTS output
///
/// Applies: high-pass filter → silence trim → loudness normalization
pub fn postprocess_tts_audio(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    postprocess_tts_audio_with_range(samples, sample_rate).0
}

/// [`postprocess_tts_audio`], also returning the range of `samples` the
/// output holds.
///
/// The filter and normalization keep every sample; only the silence trim
/// drops some, so output sample `i` is input sample `range.start + i`.
pub fn postprocess_tts_audio_with_range(
    samples: &[f32],
    sample_rate: u32,
) -> (Vec<f32>, Range<usize>) {
    // 1. High-pass filter to remove rumble (80 Hz cutoff)
    let filtered = high_pass_filter(samples, 80.0, sample_rate as f32);

    // 2. Trim silence (-40 dB threshold, keep 50ms padding)
    let min_silence = (sample_rate as f32 * 0.05) as usize; // 50ms
    let kept = trim_silence_range(&filtered, -40.0, min_silence);

    // 3. Normalize loudness (target RMS: 0.1, good for speech)
    let normalized = normalize_loudness(&filtered[kept.clone()], 0.1);
    (normalized, kept)
}

// ============================================================================
// Token Mapping
// ============================================================================

/// Load token mapping from a tokens.txt file
///
/// Format: Each line is "CHARACTER ID" where CHARACTER is a single Unicode character
/// (including space, combining marks, etc.) followed by whitespace and the integer ID.
pub fn load_tokens_map(tokens_content: &str) -> HashMap<char, i64> {
    let mut map = HashMap::new();

    for line in tokens_content.lines() {
        if line.is_empty() {
            continue;
        }
        // Parse as: first char = token, remaining (trimmed) = ID.
        // This correctly handles the space character entry ("  16")
        // and combining characters like U+0303 ("̃ 17").
        let mut chars = line.chars();
        if let Some(token_char) = chars.next() {
            let remaining: String = chars.collect();
            let id_str = remaining.trim();
            if let Ok(id) = id_str.parse::<i64>() {
                map.insert(token_char, id);
            }
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tokens_map() {
        let tokens_content = "$ 0\n; 1\na 43\nb 44\n";
        let map = load_tokens_map(tokens_content);
        assert_eq!(map.get(&'$'), Some(&0));
        assert_eq!(map.get(&';'), Some(&1));
        assert_eq!(map.get(&'a'), Some(&43));
        assert_eq!(map.get(&'b'), Some(&44));
    }

    #[test]
    fn test_load_tokens_map_handles_space_character() {
        // Space entry: " 16" (space char followed by space and ID)
        let tokens_content = "; 1\n 16\nA 24\n";
        let map = load_tokens_map(tokens_content);
        assert_eq!(map.get(&' '), Some(&16), "Space character should map to 16");
        assert_eq!(map.get(&';'), Some(&1));
        assert_eq!(map.get(&'A'), Some(&24));
    }

    /// `lead` silent samples, `voiced` samples that are all well above the
    /// -40 dB threshold, then `tail` silent samples.
    fn padded_tone(lead: usize, voiced: usize, tail: usize) -> Vec<f32> {
        let mut samples = vec![0.0; lead];
        samples.extend((0..voiced).map(|i| 0.5 + 0.2 * (i as f32 * 0.3).sin()));
        samples.extend(vec![0.0; tail]);
        samples
    }

    /// A 220 Hz tone at 24 kHz (passes the high-pass filter) between
    /// `lead` and `tail` silent samples.
    fn padded_sine(lead: usize, voiced: usize, tail: usize) -> Vec<f32> {
        let step = 2.0 * std::f32::consts::PI * 220.0 / 24_000.0;
        let mut samples = vec![0.0; lead];
        samples.extend((0..voiced).map(|i| 0.5 * (i as f32 * step).sin()));
        samples.extend(vec![0.0; tail]);
        samples
    }

    #[test]
    fn trim_silence_range_matches_trim_silence() {
        let cases: Vec<Vec<f32>> = vec![
            padded_tone(2000, 4000, 3000),
            padded_tone(0, 4000, 0),
            padded_tone(100, 50, 100),
            vec![0.0; 500], // all silence
            vec![0.3; 7],
            Vec::new(),
        ];
        for samples in &cases {
            for min_silence in [0, 4, 1200] {
                let range = trim_silence_range(samples, -40.0, min_silence);
                assert!(range.start <= range.end && range.end <= samples.len());
                assert_eq!(
                    samples[range.clone()].to_vec(),
                    trim_silence(samples, -40.0, min_silence),
                    "len={} min_silence={min_silence}",
                    samples.len()
                );
            }
        }
    }

    #[test]
    fn trim_silence_range_keeps_a_lead_in_and_tail() {
        let samples = padded_tone(2000, 4000, 3000);
        // 1200 / 4 = 300 samples kept on each side of the voiced part.
        assert_eq!(trim_silence_range(&samples, -40.0, 1200), 1700..6299);
        assert_eq!(trim_silence_range(&[0.0; 500], -40.0, 1200), 0..500);
        assert_eq!(trim_silence_range(&[], -40.0, 1200), 0..0);
    }

    #[test]
    fn postprocess_with_range_matches_postprocess_and_reports_the_trim() {
        let samples = padded_sine(4800, 12_000, 7200);
        let (processed, kept) = postprocess_tts_audio_with_range(&samples, 24_000);

        assert_eq!(processed, postprocess_tts_audio(&samples, 24_000));
        assert_eq!(processed.len(), kept.len());
        assert!(kept.start > 0 && kept.end < samples.len(), "{kept:?}");
    }

    #[test]
    fn test_load_tokens_map_handles_empty_lines() {
        let tokens_content = "a 1\n\nb 2\n";
        let map = load_tokens_map(tokens_content);
        assert_eq!(map.get(&'a'), Some(&1));
        assert_eq!(map.get(&'b'), Some(&2));
    }
}
