//! espeak-ng phonemizer backend.
//!
//! Calls the `espeak-ng` system command to convert text to IPA phonemes.
//! Requires espeak-ng to be installed on the system.

use std::collections::HashMap;
use std::process::Command;

use crate::execution::types::ExecutorResult;
use crate::runtime_adapter::AdapterError;

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

/// Punctuation marks preserved by `phonemize_raw` (mirrors the default
/// punctuation set of the Python `phonemizer` library's preserve_punctuation
/// feature). Stays inline in the IPA output so downstream LLMs see the
/// sentence/clause boundaries the model was trained on.
const PRESERVED_PUNCTUATION: &[char] = &[',', '.', ';', ':', '!', '?'];

impl EspeakBackend {
    /// Run espeak-ng once and collapse internal whitespace to single spaces.
    /// The CLI emits one line per sentence; we want a single flat IPA string.
    fn run_espeak(&self, text: &str) -> ExecutorResult<String> {
        let output = Command::new("espeak-ng")
            .args(["--ipa", "-q", "-v", &self.language])
            .arg(text)
            .output()
            .map_err(|e| {
                AdapterError::InvalidInput(format!(
                    "Failed to run espeak-ng. Is it installed? Error: {}. \
                    Install with: brew install espeak-ng (macOS) or apt-get install espeak-ng (Linux)",
                    e
                ))
            })?;

        if !output.status.success() {
            return Err(AdapterError::InvalidInput(format!(
                "espeak-ng failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let phonemes = String::from_utf8_lossy(&output.stdout);
        Ok(phonemes.split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

impl PhonemizerBackend for EspeakBackend {
    fn phonemize(&self, text: &str, tokens_map: &HashMap<char, i64>) -> ExecutorResult<String> {
        let phonemes = self.run_espeak(text)?;

        // Filter to only characters in vocabulary
        let filtered: String = phonemes
            .chars()
            .filter(|c| tokens_map.contains_key(c))
            .collect();

        Ok(filtered.trim().to_string())
    }

    /// Preserve sentence/clause punctuation by phonemizing each non-punct segment
    /// separately and reassembling with the original punctuation in place.
    ///
    /// Matches the Python `phonemizer` library's `preserve_punctuation=True`
    /// behavior used by the official NeuTTS implementation. Without this, the
    /// espeak-ng CLI strips `.,;:!?` from its IPA output, which loses the
    /// sentence boundaries the LLM was trained on and causes generation drift.
    fn phonemize_raw(&self, text: &str) -> ExecutorResult<String> {
        let mut result = String::new();
        let mut segment = String::new();

        let flush = |seg: &mut String, out: &mut String| -> ExecutorResult<()> {
            let trimmed = seg.trim();
            if !trimmed.is_empty() {
                let phones = self.run_espeak(trimmed)?;
                if !phones.is_empty() {
                    if !out.is_empty() && !out.ends_with(' ') {
                        out.push(' ');
                    }
                    out.push_str(&phones);
                }
            }
            seg.clear();
            Ok(())
        };

        for c in text.chars() {
            if PRESERVED_PUNCTUATION.contains(&c) {
                flush(&mut segment, &mut result)?;
                result.push(c);
            } else {
                segment.push(c);
            }
        }
        flush(&mut segment, &mut result)?;

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "EspeakNG"
    }
}
