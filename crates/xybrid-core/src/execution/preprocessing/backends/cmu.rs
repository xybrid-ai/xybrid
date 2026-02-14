//! CMU Dictionary-based phonemizer backend.
//!
//! Uses the CMU Pronouncing Dictionary to convert English text to IPA phonemes
//! via ARPABET-to-IPA mapping.

use std::collections::HashMap;
use std::path::Path;

use cmudict_fast::{Cmudict, Rule};
use lazy_static::lazy_static;

use crate::execution::types::ExecutorResult;
use crate::runtime_adapter::AdapterError;

use super::PhonemizerBackend;

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

/// Errors that can occur during CMU phonemization
#[derive(Debug, thiserror::Error)]
enum CmuPhonemizeError {
    #[error("Failed to load CMU dictionary: {0}")]
    DictionaryLoadError(String),
}

/// Internal CMU dictionary phonemizer.
///
/// Converts English text to IPA phonemes using the CMU Pronouncing Dictionary.
struct CmuPhonemizer {
    cmudict: Cmudict,
}

impl CmuPhonemizer {
    fn new(dict_path: impl AsRef<Path>) -> Result<Self, CmuPhonemizeError> {
        let cmudict = Cmudict::new(dict_path)
            .map_err(|e| CmuPhonemizeError::DictionaryLoadError(e.to_string()))?;
        Ok(Self { cmudict })
    }

    fn from_default_location() -> Result<Self, CmuPhonemizeError> {
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

        Err(CmuPhonemizeError::DictionaryLoadError(
            "CMU dictionary not found. Please download it to ~/.xybrid/cmudict.dict".to_string(),
        ))
    }

    fn phonemize(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result = Vec::new();

        for word in words {
            let phonemes = self.phonemize_word(word);
            result.push(phonemes);
        }

        result.join(" ")
    }

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

        // Unknown word - return as-is
        clean_word
    }

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
                ipa.push_str(base_phoneme);
            }
        }

        ipa
    }
}

/// CMU Dictionary-based phonemizer backend.
///
/// Loads the CMU Pronouncing Dictionary from a file path or default system
/// location and converts text to IPA via ARPABET mapping.
pub struct CmuDictionaryBackend {
    dict_path: Option<String>,
}

impl CmuDictionaryBackend {
    /// Create a new CmuDictionaryBackend.
    ///
    /// # Arguments
    /// - `dict_path`: Optional path to the CMU dictionary file (cmudict.dict).
    ///   If `None`, the default system location is used.
    pub fn new(dict_path: Option<String>) -> Self {
        Self { dict_path }
    }
}

impl PhonemizerBackend for CmuDictionaryBackend {
    fn phonemize(&self, text: &str, _tokens_map: &HashMap<char, i64>) -> ExecutorResult<String> {
        let phonemizer = if let Some(ref path) = self.dict_path {
            CmuPhonemizer::new(path).map_err(|e| {
                AdapterError::InvalidInput(format!(
                    "Failed to load CMU dictionary from {}: {}",
                    path, e
                ))
            })?
        } else {
            CmuPhonemizer::from_default_location().map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to initialize phonemizer: {}", e))
            })?
        };

        Ok(phonemizer.phonemize(text))
    }

    fn name(&self) -> &'static str {
        "CmuDictionary"
    }
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
