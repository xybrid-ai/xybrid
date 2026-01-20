//! Phonemizer module for text-to-speech preprocessing.
//!
//! This module provides phoneme conversion from English text to IPA symbols
//! using the CMU Pronouncing Dictionary. It's designed for TTS models like
//! KittenTTS, Piper, and others that require phoneme input.

use cmudict_fast::{Cmudict, Rule};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

lazy_static! {
    /// ARPABET to IPA symbol mapping
    static ref ARPABET_TO_IPA: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        // Vowels
        m.insert("AA", "ɑ");
        m.insert("AE", "æ");
        m.insert("AH", "ʌ");
        m.insert("AO", "ɔ");
        m.insert("AW", "aʊ");
        m.insert("AX", "ə");
        m.insert("AXR", "ɚ");
        m.insert("AY", "aɪ");
        m.insert("EH", "ɛ");
        m.insert("ER", "ɝ");
        m.insert("EY", "eɪ");
        m.insert("IH", "ɪ");
        m.insert("IX", "ɨ");
        m.insert("IY", "i");
        m.insert("OW", "oʊ");
        m.insert("OY", "ɔɪ");
        m.insert("UH", "ʊ");
        m.insert("UW", "u");
        m.insert("UX", "ʉ");
        // Consonants
        m.insert("B", "b");
        m.insert("CH", "tʃ");
        m.insert("D", "d");
        m.insert("DH", "ð");
        m.insert("DX", "ɾ");
        m.insert("EL", "l̩");
        m.insert("EM", "m̩");
        m.insert("EN", "n̩");
        m.insert("F", "f");
        m.insert("G", "ɡ");
        m.insert("HH", "h");
        m.insert("JH", "dʒ");
        m.insert("K", "k");
        m.insert("L", "l");
        m.insert("M", "m");
        m.insert("N", "n");
        m.insert("NG", "ŋ");
        m.insert("NX", "ɾ̃");
        m.insert("P", "p");
        m.insert("Q", "ʔ");
        m.insert("R", "ɹ");
        m.insert("S", "s");
        m.insert("SH", "ʃ");
        m.insert("T", "t");
        m.insert("TH", "θ");
        m.insert("V", "v");
        m.insert("W", "w");
        m.insert("WH", "ʍ");
        m.insert("Y", "j");
        m.insert("Z", "z");
        m.insert("ZH", "ʒ");
        m
    };
}

/// Phonemizer that converts English text to IPA phonemes using CMU dictionary
pub struct Phonemizer {
    cmudict: Cmudict,
}

impl Phonemizer {
    /// Create a new Phonemizer by loading the CMU dictionary from a file
    ///
    /// # Arguments
    /// * `dict_path` - Path to the CMU dictionary file (cmudict.dict)
    pub fn new(dict_path: impl AsRef<Path>) -> Result<Self, PhonemizeError> {
        let cmudict = Cmudict::new(dict_path)
            .map_err(|e| PhonemizeError::DictionaryLoadError(e.to_string()))?;
        Ok(Self { cmudict })
    }

    /// Create a Phonemizer from the default system dictionary location
    ///
    /// Looks for the dictionary in standard locations:
    /// - ~/.xybrid/cmudict.dict
    /// - /usr/share/cmudict/cmudict.dict
    /// - bundled with the model
    pub fn from_default_location() -> Result<Self, PhonemizeError> {
        // Try home directory first
        if let Some(home) = dirs::home_dir() {
            let home_dict = home.join(".xybrid").join("cmudict.dict");
            if home_dict.exists() {
                return Self::new(&home_dict);
            }
        }

        // Try system location
        let system_dict = Path::new("/usr/share/cmudict/cmudict.dict");
        if system_dict.exists() {
            return Self::new(system_dict);
        }

        Err(PhonemizeError::DictionaryLoadError(
            "CMU dictionary not found. Please download it to ~/.xybrid/cmudict.dict".to_string()
        ))
    }

    /// Convert text to IPA phonemes
    ///
    /// Returns a string of IPA symbols. Unknown words are passed through as-is.
    /// Stress markers (0, 1, 2) are converted to IPA stress marks.
    pub fn phonemize(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result = Vec::new();

        for word in words {
            let phonemes = self.phonemize_word(word);
            result.push(phonemes);
        }

        result.join(" ")
    }

    /// Convert a single word to IPA phonemes
    fn phonemize_word(&self, word: &str) -> String {
        let clean_word = word.to_lowercase();

        // Handle punctuation
        let (prefix_punct, word_part, suffix_punct) = extract_punctuation(&clean_word);

        if word_part.is_empty() {
            return clean_word;
        }

        // Look up in CMU dictionary
        if let Some(rules) = self.cmudict.get(&word_part) {
            if let Some(rule) = rules.first() {
                let ipa = self.arpabet_to_ipa(rule);
                return format!("{}{}{}", prefix_punct, ipa, suffix_punct);
            }
        }

        // Unknown word - return as-is (fallback for OOV words)
        clean_word
    }

    /// Convert ARPABET phonemes to IPA
    fn arpabet_to_ipa(&self, rule: &Rule) -> String {
        let mut ipa = String::new();

        for phoneme in rule.pronunciation() {
            let phoneme_str = phoneme.to_string();

            // Extract stress marker if present (0, 1, 2 at the end)
            let (base_phoneme, stress) = if phoneme_str.ends_with('0')
                || phoneme_str.ends_with('1')
                || phoneme_str.ends_with('2')
            {
                let stress_char = phoneme_str.chars().last().unwrap();
                let base = &phoneme_str[..phoneme_str.len() - 1];
                (base, Some(stress_char))
            } else {
                (phoneme_str.as_str(), None)
            };

            // Add stress marker before the phoneme if primary stress
            if stress == Some('1') {
                ipa.push('ˈ');
            } else if stress == Some('2') {
                ipa.push('ˌ');
            }

            // Convert to IPA
            if let Some(ipa_sym) = ARPABET_TO_IPA.get(base_phoneme) {
                ipa.push_str(ipa_sym);
            } else {
                // Unknown phoneme - use as-is
                ipa.push_str(base_phoneme);
            }
        }

        ipa
    }

    /// Convert text to phoneme token IDs using a token mapping
    ///
    /// The tokens_map maps IPA characters to their integer IDs.
    /// Returns a vector of token IDs with optional padding tokens at start and end.
    ///
    /// Note: Phonemes are normalized to NFC Unicode form before token lookup
    /// to handle combining diacritics (e.g., syllabic markers).
    pub fn text_to_token_ids(
        &self,
        text: &str,
        tokens_map: &HashMap<char, i64>,
        add_padding: bool,
    ) -> Vec<i64> {
        let phonemes = self.phonemize(text);
        // Normalize to NFC to match tokens.txt encoding
        let normalized: String = phonemes.nfc().collect();

        let mut ids = Vec::with_capacity(normalized.len() + 2);

        if add_padding {
            ids.push(0); // Start padding token
        }

        for c in normalized.chars() {
            if let Some(&id) = tokens_map.get(&c) {
                ids.push(id);
            } else if c == ' ' {
                // Space character - check if it has a mapping
                if let Some(&id) = tokens_map.get(&' ') {
                    ids.push(id);
                }
            }
            // Skip unknown characters silently
        }

        if add_padding {
            ids.push(0); // End padding token
        }

        ids
    }
}

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
    samples.iter().map(|s| {
        let amplified = s * gain;
        // Soft clip using tanh for values approaching limits
        if amplified.abs() > 0.95 {
            amplified.signum() * (0.95 + 0.05 * (amplified.abs() - 0.95).tanh())
        } else {
            amplified
        }
    }).collect()
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
    // let mut silence_count = 0;
    for (i, &sample) in samples.iter().enumerate() {
        if sample.abs() > threshold {
            // silence_count = 0;
            start = i.saturating_sub(min_silence_samples / 4); // Keep a bit of lead-in
            break;
        }
        // silence_count += 1;
    }

    // Find last non-silent sample
    let mut end = samples.len();
    // silence_count = 0;
    for (i, &sample) in samples.iter().enumerate().rev() {
        if sample.abs() > threshold {
            // silence_count = 0;
            end = (i + min_silence_samples / 4).min(samples.len()); // Keep a bit of tail
            break;
        }
        // silence_count += 1;
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
pub fn postprocess_tts_audio(
    samples: &[f32],
    sample_rate: u32,
) -> Vec<f32> {
    // 1. High-pass filter to remove rumble (80 Hz cutoff)
    let filtered = high_pass_filter(samples, 80.0, sample_rate as f32);

    // 2. Trim silence (-40 dB threshold, keep 50ms padding)
    let min_silence = (sample_rate as f32 * 0.05) as usize; // 50ms
    let trimmed = trim_silence(&filtered, -40.0, min_silence);

    // 3. Normalize loudness (target RMS: 0.1, good for speech)
    normalize_loudness(&trimmed, 0.1)
}

/// Extract leading and trailing punctuation from a word
fn extract_punctuation(word: &str) -> (String, String, String) {
    let mut prefix = String::new();
    let mut suffix = String::new();
    let mut word_chars: Vec<char> = word.chars().collect();

    // Extract leading punctuation
    while !word_chars.is_empty() && !word_chars[0].is_alphanumeric() {
        prefix.push(word_chars.remove(0));
    }

    // Extract trailing punctuation
    while !word_chars.is_empty() && !word_chars.last().unwrap().is_alphanumeric() {
        suffix.insert(0, word_chars.pop().unwrap());
    }

    let word_part: String = word_chars.into_iter().collect();
    (prefix, word_part, suffix)
}

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

/// Errors that can occur during phonemization
#[derive(Debug, thiserror::Error)]
pub enum PhonemizeError {
    #[error("Failed to load CMU dictionary: {0}")]
    DictionaryLoadError(String),

    #[error("Failed to phonemize text: {0}")]
    PhonemeConversionError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_dict_path() -> Option<std::path::PathBuf> {
        // Try to find the dictionary in common locations
        if let Some(home) = dirs::home_dir() {
            let home_dict = home.join(".xybrid").join("cmudict.dict");
            if home_dict.exists() {
                return Some(home_dict);
            }
        }
        None
    }

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
    fn test_extract_punctuation() {
        let (prefix, word, suffix) = extract_punctuation("hello,");
        assert_eq!(prefix, "");
        assert_eq!(word, "hello");
        assert_eq!(suffix, ",");

        let (prefix, word, suffix) = extract_punctuation("\"hello!\"");
        assert_eq!(prefix, "\"");
        assert_eq!(word, "hello");
        assert_eq!(suffix, "!\"");
    }

    #[test]
    #[ignore = "Requires CMU dictionary to be installed"]
    fn test_phonemize_hello() {
        if let Some(dict_path) = get_test_dict_path() {
            let phonemizer = Phonemizer::new(&dict_path).unwrap();
            let result = phonemizer.phonemize("hello");
            assert!(!result.is_empty());
            println!("hello -> {}", result);
        }
    }
}
