//! TTS audio postprocessing and token mapping utilities.
//!
//! This module provides:
//! - `load_tokens_map`: Parse tokens.txt files mapping IPA characters to token IDs
//! - `postprocess_tts_audio`: Full audio postprocessing pipeline (high-pass → trim → normalize)
//! - Individual audio utilities: `normalize_loudness`, `trim_silence`, `high_pass_filter`

use std::collections::HashMap;

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
/// Trimmed audio samples
pub fn trim_silence(samples: &[f32], threshold_db: f32, min_silence_samples: usize) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
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
        return samples.to_vec();
    }

    samples[start..end].to_vec()
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
    // 1. High-pass filter to remove rumble (80 Hz cutoff)
    let filtered = high_pass_filter(samples, 80.0, sample_rate as f32);

    // 2. Trim silence (-40 dB threshold, keep 50ms padding)
    let min_silence = (sample_rate as f32 * 0.05) as usize; // 50ms
    let trimmed = trim_silence(&filtered, -40.0, min_silence);

    // 3. Normalize loudness (target RMS: 0.1, good for speech)
    normalize_loudness(&trimmed, 0.1)
}

// ============================================================================
// Token Mapping
// ============================================================================

/// Load token mapping from a tokens.txt file
///
/// Format: Each line is "TOKEN ID" (space-separated)
pub fn load_tokens_map(tokens_content: &str) -> HashMap<char, i64> {
    let mut map = HashMap::new();

    for line in tokens_content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let token = parts[0];
            if let Ok(id) = parts[1].parse::<i64>() {
                // Handle single character tokens
                if let Some(c) = token.chars().next() {
                    if token.chars().count() == 1 {
                        map.insert(c, id);
                    }
                }
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
}
