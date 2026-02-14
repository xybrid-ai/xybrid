//! Misaki dictionary-based phonemizer backend.
//!
//! Uses bundled JSON dictionaries (us_gold.json, us_silver.json) to convert
//! text to IPA phonemes without any system dependencies.

use std::collections::HashMap;

use crate::execution::types::ExecutorResult;

use super::PhonemizerBackend;

/// Misaki dictionary-based phonemizer.
///
/// Loads gold and silver dictionaries from a `misaki/` subdirectory under
/// the provided base path. Implements the same pipeline as the official
/// Python Misaki package: dictionary lookup with case variant generation,
/// morphological stemming, and post-lookup ɾ→T / ʔ→t replacements.
pub struct MisakiBackend {
    base_path: String,
}

impl MisakiBackend {
    /// Create a new MisakiBackend.
    ///
    /// # Arguments
    /// - `base_path`: Path to the model directory containing a `misaki/` subdirectory
    ///   with `us_gold.json` and `us_silver.json`.
    pub fn new(base_path: String) -> Self {
        Self { base_path }
    }
}

impl PhonemizerBackend for MisakiBackend {
    fn phonemize(&self, text: &str, vocab: &HashMap<char, i64>) -> ExecutorResult<String> {
        // Delegate to the existing implementation in text.rs
        super::super::text::phonemize_with_misaki(text, &self.base_path, vocab)
    }

    fn name(&self) -> &'static str {
        "MisakiDictionary"
    }
}
