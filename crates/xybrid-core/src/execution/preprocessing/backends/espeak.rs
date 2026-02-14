//! espeak-ng phonemizer backend.
//!
//! Calls the `espeak-ng` system command to convert text to IPA phonemes.
//! Requires espeak-ng to be installed on the system.

use std::collections::HashMap;

use crate::execution::types::ExecutorResult;

use super::PhonemizerBackend;

/// espeak-ng phonemizer backend.
///
/// Uses the system-installed `espeak-ng` command to convert text to IPA
/// phonemes. Supports multiple languages via the language code parameter.
///
/// Requires espeak-ng to be installed:
/// - macOS: `brew install espeak-ng`
/// - Linux: `apt-get install espeak-ng`
pub struct EspeakBackend {
    language: String,
}

impl EspeakBackend {
    /// Create a new EspeakBackend.
    ///
    /// # Arguments
    /// - `language`: Language code for espeak-ng (e.g., "en-us", "en-gb").
    pub fn new(language: String) -> Self {
        Self { language }
    }
}

impl PhonemizerBackend for EspeakBackend {
    fn phonemize(&self, text: &str, tokens_map: &HashMap<char, i64>) -> ExecutorResult<String> {
        // Delegate to the existing implementation in text.rs
        super::super::text::phonemize_with_espeak(text, &self.language, tokens_map)
    }

    fn name(&self) -> &'static str {
        "EspeakNG"
    }
}
