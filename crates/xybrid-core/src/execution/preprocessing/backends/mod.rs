//! Pluggable phonemizer backends.
//!
//! Each backend implements the [`PhonemizerBackend`] trait, allowing
//! `model_metadata.json` to drive backend selection without modifying
//! dispatch code.

pub mod cmu;
pub mod espeak;
pub mod misaki;
pub mod openphonemizer;

pub use cmu::CmuDictionaryBackend;
pub use espeak::EspeakBackend;
pub use misaki::MisakiBackend;
pub use openphonemizer::OpenPhonemizerBackend;

use std::collections::HashMap;

use crate::execution::types::ExecutorResult;

/// Trait for phonemizer backends that convert text to IPA phoneme strings.
///
/// Implementations receive the normalized text and the tokens vocabulary map,
/// and return an IPA phoneme string whose characters can be mapped to token IDs.
pub trait PhonemizerBackend: Send + Sync {
    /// Convert text to IPA phonemes.
    ///
    /// # Arguments
    /// - `text`: Input text (already normalized if `normalize_text` was enabled)
    /// - `tokens_map`: Mapping from IPA characters to token IDs (used for vocab filtering)
    fn phonemize(&self, text: &str, tokens_map: &HashMap<char, i64>) -> ExecutorResult<String>;

    /// Convert text to raw IPA phonemes without vocab filtering.
    ///
    /// Used by codec TTS strategies that need unfiltered phoneme output.
    /// Default implementation calls `phonemize()` with a permissive map
    /// covering Latin, IPA Extensions, and common Unicode ranges.
    fn phonemize_raw(&self, text: &str) -> ExecutorResult<String> {
        let mut universal_map = HashMap::new();
        for i in 0x20u32..0x2000 {
            if let Some(c) = char::from_u32(i) {
                universal_map.insert(c, i as i64);
            }
        }
        self.phonemize(text, &universal_map)
    }

    /// Human-readable name of this backend.
    fn name(&self) -> &'static str;
}
