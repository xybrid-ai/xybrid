//! CMU Dictionary-based phonemizer backend.
//!
//! Uses the CMU Pronouncing Dictionary to convert English text to IPA phonemes
//! via ARPABET-to-IPA mapping.

use std::collections::HashMap;

use crate::execution::types::ExecutorResult;
use crate::runtime_adapter::AdapterError;

use super::PhonemizerBackend;

/// CMU Dictionary-based phonemizer.
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
        use crate::phonemizer::Phonemizer;

        let phonemizer = if let Some(ref path) = self.dict_path {
            Phonemizer::new(path).map_err(|e| {
                AdapterError::InvalidInput(format!(
                    "Failed to load CMU dictionary from {}: {}",
                    path, e
                ))
            })?
        } else {
            Phonemizer::from_default_location().map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to initialize phonemizer: {}", e))
            })?
        };

        Ok(phonemizer.phonemize(text))
    }

    fn name(&self) -> &'static str {
        "CmuDictionary"
    }
}
