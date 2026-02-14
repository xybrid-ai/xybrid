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

impl PhonemizerBackend for EspeakBackend {
    fn phonemize(&self, text: &str, tokens_map: &HashMap<char, i64>) -> ExecutorResult<String> {
        // Call espeak-ng with IPA output
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
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AdapterError::InvalidInput(format!(
                "espeak-ng failed: {}",
                stderr
            )));
        }

        let phonemes = String::from_utf8_lossy(&output.stdout);

        // Filter to only characters in vocabulary
        let filtered: String = phonemes
            .chars()
            .filter(|c| tokens_map.contains_key(c))
            .collect();

        Ok(filtered.trim().to_string())
    }

    fn name(&self) -> &'static str {
        "EspeakNG"
    }
}
