//! Text preprocessing operations.
//!
//! This module provides:
//! - `tokenize_step`: Tokenize text for NLP models
//! - `phonemize_step`: Convert text to phonemes for TTS models

use super::super::types::{ExecutorResult, PreprocessedData};
use crate::execution::template::{PhonemizerBackend, TokenizerType};
use crate::runtime_adapter::AdapterError;

/// Tokenize text input for NLP models.
///
/// # Arguments
/// - `data`: Input data (Text)
/// - `tokenizer_path`: Path to tokenizer.json file
/// - `tokenizer_type`: Type of tokenizer (WordPiece, BPE, SentencePiece)
/// - `max_length`: Optional maximum sequence length
pub fn tokenize_step(
    data: PreprocessedData,
    tokenizer_path: &str,
    tokenizer_type: &TokenizerType,
    max_length: Option<usize>,
) -> ExecutorResult<PreprocessedData> {
    use tokenizers::Tokenizer;

    let text = match data {
        PreprocessedData::Text(text) => text,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Tokenize requires text input".to_string(),
            ))
        }
    };

    let tokenizer = match tokenizer_type {
        TokenizerType::WordPiece | TokenizerType::BPE => Tokenizer::from_file(tokenizer_path)
            .map_err(|e| {
                AdapterError::InvalidInput(format!(
                    "Failed to load tokenizer from {}: {}",
                    tokenizer_path, e
                ))
            })?,
        TokenizerType::SentencePiece => {
            return Err(AdapterError::InvalidInput(
                "SentencePiece tokenizer not yet implemented".to_string(),
            ));
        }
    };

    let encoding = tokenizer
        .encode(text.clone(), false)
        .map_err(|e| AdapterError::InvalidInput(format!("Tokenization failed: {}", e)))?;

    let mut ids: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
    let mut attention_mask: Vec<usize> = encoding
        .get_attention_mask()
        .iter()
        .map(|&mask| mask as usize)
        .collect();
    let mut token_type_ids: Vec<usize> = encoding
        .get_type_ids()
        .iter()
        .map(|&type_id| type_id as usize)
        .collect();

    if let Some(max_len) = max_length {
        if ids.len() > max_len {
            ids.truncate(max_len);
            attention_mask.truncate(max_len);
            token_type_ids.truncate(max_len);
        }
    }

    Ok(PreprocessedData::TokenIds {
        ids,
        attention_mask,
        token_type_ids,
        vocab_file: tokenizer_path.to_string(),
        original_text: text,
    })
}

/// Phonemize text input for TTS models.
///
/// Converts English text to IPA phonemes using either CMU Dictionary or espeak-ng,
/// then maps phonemes to token IDs using the provided tokens file.
///
/// # Arguments
/// - `data`: Input data (Text)
/// - `tokens_path`: Path to tokens.txt file (maps IPA symbols to token IDs)
/// - `backend`: Which phonemization backend to use
/// - `dict_path`: Optional path to CMU dictionary file (CMU backend only)
/// - `language`: Language code for espeak-ng (e.g., "en-us", "en-gb")
/// - `add_padding`: Whether to add padding tokens (0) at start and end
/// - `normalize_text`: Whether to normalize text before phonemization
pub fn phonemize_step(
    data: PreprocessedData,
    tokens_path: &str,
    backend: &PhonemizerBackend,
    dict_path: Option<&str>,
    language: Option<&str>,
    add_padding: bool,
    normalize_text: bool,
) -> ExecutorResult<PreprocessedData> {
    use crate::phonemizer::load_tokens_map;

    let text = match data {
        PreprocessedData::Text(text) => text,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Phonemize requires text input".to_string(),
            ))
        }
    };

    // Load tokens mapping
    let tokens_content = std::fs::read_to_string(tokens_path).map_err(|e| {
        AdapterError::InvalidInput(format!("Failed to read tokens file {}: {}", tokens_path, e))
    })?;
    let tokens_map = load_tokens_map(&tokens_content);

    // Optionally normalize text
    let processed_text = if normalize_text {
        normalize_text_for_tts(&text)
    } else {
        text.clone()
    };

    // Derive base_path from tokens_path (go up one directory from tokens.txt)
    let base_path = std::path::Path::new(tokens_path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_str()
        .unwrap_or(".");

    // Create the appropriate backend trait object and phonemize
    let backend_impl = backend.create(base_path, dict_path, language);
    let phonemes = backend_impl.phonemize(&processed_text, &tokens_map)?;

    // Convert phonemes to token IDs
    let mut ids: Vec<i64> = Vec::new();

    if add_padding {
        ids.push(0); // Start padding token
    }

    for c in phonemes.chars() {
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

    // Return as PhonemeIds for use by TTS models
    Ok(PreprocessedData::PhonemeIds {
        ids,
        phonemes,
        original_text: text,
    })
}

/// Normalize text for TTS processing.
///
/// Applies common text transformations:
/// - Parse `[word](/phonemes/)` markdown syntax for explicit pronunciations
/// - Normalize quotes and special characters
/// - Expand common abbreviations (Dr., Mr., etc.)
/// - Expand numbers to words (e.g. "82" → "eighty two")
/// - Normalize ASCII ellipsis (...) to Unicode ellipsis (…)
/// - Fix spaced punctuation ("you ?" → "you?")
/// - Ensure space after punctuation before words ("look…Lost" → "look… Lost")
/// - Clean up whitespace
pub fn normalize_text_for_tts(text: &str) -> String {
    let mut result = text.to_string();

    // Parse [word](/phonemes/) markdown syntax — extract explicit pronunciations.
    // The phoneme content is preserved as-is (wrapped in markers) for the phonemizer
    // to use directly instead of dictionary lookup.
    // e.g. "[Kokoro](/kˈOkəɹO/)" → "\x01kˈOkəɹO\x02"
    result = parse_phoneme_links(&result);

    // Normalize quotes
    result = result.replace(['\u{2018}', '\u{2019}'], "'");
    result = result.replace(['\u{201C}', '\u{201D}'], "\"");

    // Expand common abbreviations (before punctuation normalization)
    result = result.replace("Dr.", "Doctor");
    result = result.replace("Mr.", "Mister");
    result = result.replace("Mrs.", "Missus");
    result = result.replace("Ms.", "Miss");
    result = result.replace("etc.", "etcetera");

    // Expand currency ($X.XX → "X dollars and XX cents")
    result = expand_currency(&result);

    // Expand percentages (X% → "X percent")
    result = expand_percentage(&result);

    // Expand numbers to words (basic support)
    result = expand_numbers(&result);

    // Normalize ASCII ellipsis to Unicode ellipsis (token 10 in Kokoro)
    result = result.replace("...", "\u{2026}");

    // Fix spaced punctuation: "you ?" → "you?"
    result = result.replace(" .", ".");
    result = result.replace(" ?", "?");
    result = result.replace(" !", "!");
    result = result.replace(" ,", ",");
    result = result.replace(" ;", ";");

    // Ensure space after sentence punctuation when directly followed by a word character.
    // e.g. "look…Lost" → "look… Lost", "Hello.World" → "Hello. World"
    let chars: Vec<char> = result.chars().collect();
    let mut spaced = String::with_capacity(result.len() + 16);
    for i in 0..chars.len() {
        spaced.push(chars[i]);
        if i + 1 < chars.len()
            && matches!(chars[i], '.' | '!' | '?' | '\u{2026}')
            && chars[i + 1].is_alphanumeric()
        {
            spaced.push(' ');
        }
    }
    result = spaced;

    // Normalize whitespace
    let mut prev_space = false;
    result = result
        .chars()
        .filter_map(|c| {
            if c.is_whitespace() {
                if prev_space {
                    None
                } else {
                    prev_space = true;
                    Some(' ')
                }
            } else {
                prev_space = false;
                Some(c)
            }
        })
        .collect();

    result.trim().to_string()
}

/// Sentinel characters used to wrap explicit phoneme overrides from `[word](/phonemes/)` syntax.
/// These are control characters unlikely to appear in normal text.
pub(crate) const PHONEME_LINK_START: char = '\x01';
pub(crate) const PHONEME_LINK_END: char = '\x02';

/// Parse `[text](/phonemes/)` markdown-like syntax for explicit pronunciation overrides.
///
/// The official Kokoro pipeline supports this syntax to provide IPA phonemes directly:
/// - `[Kokoro](/kˈOkəɹO/)` → uses `kˈOkəɹO` as the phoneme output instead of dictionary lookup
///
/// The phonemes are wrapped in sentinel characters (\x01...\x02) so the phonemizer can
/// detect and pass them through without dictionary lookup.
fn parse_phoneme_links(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some(bracket_start) = remaining.find('[') {
        // Add everything before the bracket
        result.push_str(&remaining[..bracket_start]);

        let after_bracket = &remaining[bracket_start + 1..];
        if let Some(bracket_end) = after_bracket.find(']') {
            let link_text = &after_bracket[..bracket_end];
            let after_close = &after_bracket[bracket_end + 1..];

            // Check for (/.../)-style phoneme link
            if after_close.starts_with("(/") {
                if let Some(paren_end) = after_close.find("/)") {
                    let phonemes = &after_close[2..paren_end];
                    // Wrap phonemes in sentinels for the phonemizer
                    result.push(PHONEME_LINK_START);
                    result.push_str(phonemes);
                    result.push(PHONEME_LINK_END);
                    remaining = &after_close[paren_end + 2..];
                    continue;
                }
            }

            // Check for numeric stress override: [word](+1), [word](-1), etc.
            if after_close.starts_with('(') {
                if let Some(paren_end) = after_close.find(')') {
                    let _feature = &after_close[1..paren_end];
                    // For now, just emit the link text without the feature
                    result.push_str(link_text);
                    remaining = &after_close[paren_end + 1..];
                    continue;
                }
            }

            // Not a phoneme link — emit the bracket and text literally
            result.push('[');
            result.push_str(link_text);
            result.push(']');
            remaining = after_close;
        } else {
            // No closing bracket — emit literally
            result.push('[');
            remaining = after_bracket;
        }
    }

    result.push_str(remaining);
    result
}

/// Expand currency symbols for TTS.
///
/// Handles `$X.XX` → "X dollars and XX cents", `$X` → "X dollars".
fn expand_currency(text: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            // Collect the number after $
            let start = i + 1;
            let mut end = start;
            let mut has_dot = false;
            while end < chars.len()
                && (chars[end].is_ascii_digit() || (chars[end] == '.' && !has_dot))
            {
                if chars[end] == '.' {
                    has_dot = true;
                }
                end += 1;
            }
            let num_str: String = chars[start..end].iter().collect();
            if has_dot {
                let parts: Vec<&str> = num_str.split('.').collect();
                if parts.len() == 2 {
                    let dollars = parts[0];
                    let cents = parts[1];
                    if cents == "00" {
                        result.push_str(&format!("{} dollars", dollars));
                    } else {
                        result.push_str(&format!("{} dollars and {} cents", dollars, cents));
                    }
                } else {
                    result.push_str(&format!("{} dollars", num_str));
                }
            } else {
                result.push_str(&format!("{} dollars", num_str));
            }
            i = end;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Expand percentage symbols for TTS.
///
/// Handles `X%` → "X percent".
fn expand_percentage(text: &str) -> String {
    let mut result = String::new();
    let words: Vec<&str> = text.split(' ').collect();

    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        if let Some(num_part) = word.strip_suffix('%') {
            if !num_part.is_empty() && num_part.chars().all(|c| c.is_ascii_digit() || c == '.') {
                result.push_str(num_part);
                result.push_str(" percent");
            } else {
                result.push_str(word);
            }
        } else {
            result.push_str(word);
        }
    }

    result
}

/// Expand numbers to words for TTS.
///
/// Handles common number patterns:
/// - Cardinal numbers: "82" → "eighty two"
/// - Numbers with units: "82 million" → "eighty two million"
/// - Ordinals: "1st", "2nd", "3rd" are kept as-is (handled by dictionary)
fn expand_numbers(text: &str) -> String {
    let mut result = String::new();
    let words: Vec<&str> = text.split(' ').collect();

    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }

        // Check if this word is a pure number
        let clean = word.trim_end_matches(|c: char| !c.is_ascii_digit());
        let suffix = &word[clean.len()..];

        if !clean.is_empty() && clean.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = clean.parse::<u64>() {
                if suffix.is_empty() {
                    result.push_str(&number_to_words(n));
                } else {
                    // Has trailing non-digit chars (like ordinal suffixes)
                    result.push_str(word);
                }
            } else {
                result.push_str(word);
            }
        } else {
            result.push_str(word);
        }
    }

    result
}

/// Convert a number to English words.
fn number_to_words(n: u64) -> String {
    if n == 0 {
        return "zero".to_string();
    }

    let ones = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    let tens = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    fn convert(n: u64, ones: &[&str], tens: &[&str]) -> String {
        if n == 0 {
            return String::new();
        }
        if n < 20 {
            return ones[n as usize].to_string();
        }
        if n < 100 {
            let t = tens[(n / 10) as usize].to_string();
            let o = convert(n % 10, ones, tens);
            return if o.is_empty() {
                t
            } else {
                format!("{} {}", t, o)
            };
        }
        if n < 1000 {
            let h = format!("{} hundred", ones[(n / 100) as usize]);
            let r = convert(n % 100, ones, tens);
            return if r.is_empty() {
                h
            } else {
                format!("{} {}", h, r)
            };
        }

        let scales: &[(u64, &str)] = &[
            (1_000_000_000_000, "trillion"),
            (1_000_000_000, "billion"),
            (1_000_000, "million"),
            (1_000, "thousand"),
        ];

        for &(scale, name) in scales {
            if n >= scale {
                let high = convert(n / scale, ones, tens);
                let low = convert(n % scale, ones, tens);
                return if low.is_empty() {
                    format!("{} {}", high, name)
                } else {
                    format!("{} {} {}", high, name, low)
                };
            }
        }

        String::new()
    }

    convert(n, &ones, &tens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::execution::preprocessing::backends::MisakiBackend;
    use crate::execution::preprocessing::backends::PhonemizerBackend as PhonemizerBackendTrait;
    use crate::execution::template::PhonemizerBackend;

    // ============================================================================
    // Text Normalization Tests
    // ============================================================================

    #[test]
    fn test_normalize_ellipsis_to_unicode() {
        let result = normalize_text_for_tts("You look...Lost");
        assert!(
            result.contains('\u{2026}'),
            "Expected Unicode ellipsis (…), got: {}",
            result
        );
        assert!(!result.contains("..."), "ASCII ellipsis should be replaced");
    }

    #[test]
    fn test_normalize_spaced_question_mark() {
        assert_eq!(normalize_text_for_tts("you ?"), "you?");
    }

    #[test]
    fn test_normalize_spaced_period() {
        assert_eq!(normalize_text_for_tts("hello ."), "hello.");
    }

    #[test]
    fn test_normalize_spaced_exclamation() {
        assert_eq!(normalize_text_for_tts("wow !"), "wow!");
    }

    #[test]
    fn test_normalize_spaced_comma() {
        assert_eq!(normalize_text_for_tts("Beer , Ale"), "Beer, Ale");
    }

    #[test]
    fn test_normalize_space_after_punct_before_word() {
        let result = normalize_text_for_tts("look...Lost");
        // "..." → "…", then "…L" → "… L"
        assert_eq!(result, "look\u{2026} Lost");
    }

    #[test]
    fn test_normalize_period_joined_words() {
        let result = normalize_text_for_tts("Hello.World");
        assert_eq!(result, "Hello. World");
    }

    #[test]
    fn test_normalize_preserves_normal_punctuation() {
        let result = normalize_text_for_tts("Hello. How are you?");
        assert_eq!(result, "Hello. How are you?");
    }

    #[test]
    fn test_normalize_full_example() {
        let input = "Hello there. You look...Lost. Can I help you ? we have Beer, Ale and an extensive menu.";
        let result = normalize_text_for_tts(input);
        assert_eq!(
            result,
            "Hello there. You look\u{2026} Lost. Can I help you? we have Beer, Ale and an extensive menu."
        );
    }

    #[test]
    fn test_normalize_abbreviation_expansion() {
        let result = normalize_text_for_tts("Dr. Smith and Mr. Jones");
        assert_eq!(result, "Doctor Smith and Mister Jones");
    }

    // ============================================================================
    // Misaki Phonemizer Punctuation Preservation Tests
    // (Requires fixture dictionaries)
    // ============================================================================

    /// Path to the Kokoro model fixtures directory.
    fn kokoro_fixture_path() -> String {
        // Try workspace-relative path first (running from workspace root)
        let workspace_path = "repos/xybrid/integration-tests/fixtures/models/kokoro-82m";
        if std::path::Path::new(workspace_path).join("misaki").exists() {
            return workspace_path.to_string();
        }
        // Try crate-relative path (running from within xybrid-core)
        let crate_path = "../../integration-tests/fixtures/models/kokoro-82m";
        if std::path::Path::new(crate_path).join("misaki").exists() {
            return crate_path.to_string();
        }
        // Try absolute path as last resort
        env!("CARGO_MANIFEST_DIR").to_string()
            + "/../../integration-tests/fixtures/models/kokoro-82m"
    }

    /// Load the Kokoro tokens vocabulary map.
    fn kokoro_vocab() -> HashMap<char, i64> {
        use crate::phonemizer::load_tokens_map;
        let base = kokoro_fixture_path();
        let tokens_path = std::path::Path::new(&base).join("tokens.txt");
        let tokens_content = std::fs::read_to_string(&tokens_path)
            .unwrap_or_else(|e| panic!("Failed to read tokens.txt at {:?}: {}", tokens_path, e));
        load_tokens_map(&tokens_content)
    }

    /// Check if Kokoro fixtures are available (skip tests gracefully if not).
    fn has_kokoro_fixtures() -> bool {
        let base = kokoro_fixture_path();
        std::path::Path::new(&base)
            .join("misaki")
            .join("us_gold.json")
            .exists()
    }

    #[test]
    fn test_misaki_preserves_period() {
        if !has_kokoro_fixtures() {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        }
        let vocab = kokoro_vocab();
        let base = kokoro_fixture_path();
        let backend = MisakiBackend::new(base);
        let result = backend.phonemize("Lost. Can", &vocab).unwrap();
        assert!(
            result.contains('.'),
            "Period should be preserved in phoneme output, got: {}",
            result
        );
    }

    #[test]
    fn test_misaki_preserves_comma() {
        if !has_kokoro_fixtures() {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        }
        let vocab = kokoro_vocab();
        let base = kokoro_fixture_path();
        let backend = MisakiBackend::new(base);
        let result = backend.phonemize("Beer, Ale", &vocab).unwrap();
        assert!(
            result.contains(','),
            "Comma should be preserved in phoneme output, got: {}",
            result
        );
    }

    #[test]
    fn test_misaki_preserves_question_mark() {
        if !has_kokoro_fixtures() {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        }
        let vocab = kokoro_vocab();
        let base = kokoro_fixture_path();
        let backend = MisakiBackend::new(base);
        let result = backend.phonemize("help you?", &vocab).unwrap();
        assert!(
            result.contains('?'),
            "Question mark should be preserved in phoneme output, got: {}",
            result
        );
    }

    #[test]
    fn test_misaki_preserves_ellipsis() {
        if !has_kokoro_fixtures() {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        }
        let vocab = kokoro_vocab();
        let base = kokoro_fixture_path();
        let backend = MisakiBackend::new(base);
        // After normalization, "..." becomes "…" (token 10)
        let result = backend.phonemize("look\u{2026} Lost", &vocab).unwrap();
        assert!(
            result.contains('\u{2026}'),
            "Unicode ellipsis should be preserved in phoneme output, got: {}",
            result
        );
    }

    #[test]
    fn test_misaki_standalone_punctuation() {
        if !has_kokoro_fixtures() {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        }
        let vocab = kokoro_vocab();
        let base = kokoro_fixture_path();
        let backend = MisakiBackend::new(base);
        // Standalone "?" should still produce the ? token
        let result = backend.phonemize("yes ?", &vocab).unwrap();
        assert!(
            result.contains('?'),
            "Standalone question mark should be preserved, got: {}",
            result
        );
    }

    // ============================================================================
    // Full Pipeline Token ID Tests (normalize → phonemize → tokenize)
    // ============================================================================

    /// Helper: run the full phonemize_step pipeline and return token IDs.
    fn run_phonemize_pipeline(text: &str) -> Option<Vec<i64>> {
        if !has_kokoro_fixtures() {
            return None;
        }
        let base = kokoro_fixture_path();
        let tokens_path = std::path::Path::new(&base)
            .join("tokens.txt")
            .to_string_lossy()
            .to_string();

        let result = phonemize_step(
            PreprocessedData::Text(text.to_string()),
            &tokens_path,
            &PhonemizerBackend::MisakiDictionary,
            None,
            None,
            true, // add_padding
            true, // normalize_text
        )
        .unwrap();

        match result {
            PreprocessedData::PhonemeIds { ids, .. } => Some(ids),
            _ => panic!("Expected PhonemeIds"),
        }
    }

    // Token ID reference from tokens.txt:
    // ; = 1, : = 2, , = 3, . = 4, ! = 5, ? = 6, … = 10

    #[test]
    fn test_pipeline_period_token_present() {
        let Some(ids) = run_phonemize_pipeline("Hello there.") else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        assert!(
            ids.contains(&4),
            "Token IDs should contain period (4), got: {:?}",
            ids
        );
    }

    #[test]
    fn test_pipeline_question_mark_token_present() {
        let Some(ids) = run_phonemize_pipeline("Can I help you?") else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        assert!(
            ids.contains(&6),
            "Token IDs should contain question mark (6), got: {:?}",
            ids
        );
    }

    #[test]
    fn test_pipeline_comma_token_present() {
        let Some(ids) = run_phonemize_pipeline("Beer, Ale and menu.") else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        assert!(
            ids.contains(&3),
            "Token IDs should contain comma (3), got: {:?}",
            ids
        );
        assert!(
            ids.contains(&4),
            "Token IDs should contain period (4), got: {:?}",
            ids
        );
    }

    #[test]
    fn test_pipeline_ellipsis_token_present() {
        let Some(ids) = run_phonemize_pipeline("You look...Lost.") else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        assert!(
            ids.contains(&10),
            "Token IDs should contain ellipsis (10), got: {:?}",
            ids
        );
        assert!(
            ids.contains(&4),
            "Token IDs should contain period (4), got: {:?}",
            ids
        );
    }

    #[test]
    fn test_pipeline_spaced_question_mark_fixed() {
        // "you ?" should be normalized to "you?" and produce token 6
        let Some(ids) = run_phonemize_pipeline("Can I help you ?") else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        assert!(
            ids.contains(&6),
            "Spaced '?' should still produce question mark token (6), got: {:?}",
            ids
        );
    }

    #[test]
    fn test_pipeline_full_example_all_punctuation() {
        let input = "Hello there. You look...Lost. Can I help you ? we have Beer, Ale and an extensive menu.";
        let Some(ids) = run_phonemize_pipeline(input) else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };

        // Count punctuation tokens
        let period_count = ids.iter().filter(|&&id| id == 4).count();
        let comma_count = ids.iter().filter(|&&id| id == 3).count();
        let question_count = ids.iter().filter(|&&id| id == 6).count();
        let ellipsis_count = ids.iter().filter(|&&id| id == 10).count();

        assert!(
            period_count >= 3,
            "Expected at least 3 periods (after 'there', 'Lost', 'menu'), got {}. IDs: {:?}",
            period_count,
            ids
        );
        assert!(
            comma_count >= 1,
            "Expected at least 1 comma (after 'Beer'), got {}. IDs: {:?}",
            comma_count,
            ids
        );
        assert!(
            question_count >= 1,
            "Expected at least 1 question mark (after 'you'), got {}. IDs: {:?}",
            question_count,
            ids
        );
        assert!(
            ellipsis_count >= 1,
            "Expected at least 1 ellipsis (after 'look'), got {}. IDs: {:?}",
            ellipsis_count,
            ids
        );

        // Verify padding tokens at start and end
        assert_eq!(ids[0], 0, "Should start with padding token");
        assert_eq!(ids[ids.len() - 1], 0, "Should end with padding token");
    }

    #[test]
    fn test_pipeline_no_punctuation_regression() {
        // Simple text without punctuation should still work
        let Some(ids) = run_phonemize_pipeline("hello world") else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        assert!(ids.len() > 2, "Should produce phoneme tokens");
        assert_eq!(ids[0], 0, "Should start with padding");
        assert_eq!(ids[ids.len() - 1], 0, "Should end with padding");
    }

    // ============================================================================
    // Backend-Agnostic Text Normalization Tests (US-011)
    // ============================================================================

    #[test]
    fn test_normalize_expands_number_82() {
        let result = normalize_text_for_tts("I have 82 items");
        assert!(
            result.contains("eighty two"),
            "Expected '82' to be expanded to 'eighty two', got: {}",
            result
        );
        assert!(
            !result.contains("82"),
            "Original '82' should be replaced, got: {}",
            result
        );
    }

    #[test]
    fn test_normalize_expands_various_numbers() {
        assert_eq!(normalize_text_for_tts("0"), "zero");
        assert!(normalize_text_for_tts("15 cats").contains("fifteen"));
        assert!(normalize_text_for_tts("100 percent").contains("one hundred"));
    }

    /// Helper: run phonemize_step with a specific backend and normalize_text flag.
    /// Returns (phonemes, ids) if fixtures are available, None otherwise.
    fn run_phonemize_step_with_backend(
        text: &str,
        backend: &PhonemizerBackend,
        normalize_text: bool,
    ) -> Option<(String, Vec<i64>)> {
        if !has_kokoro_fixtures() {
            return None;
        }
        let base = kokoro_fixture_path();
        let tokens_path = std::path::Path::new(&base)
            .join("tokens.txt")
            .to_string_lossy()
            .to_string();

        let dict_path = match backend {
            PhonemizerBackend::CmuDictionary => None,
            _ => None,
        };

        let result = phonemize_step(
            PreprocessedData::Text(text.to_string()),
            &tokens_path,
            backend,
            dict_path,
            None,
            true,
            normalize_text,
        )
        .unwrap();

        match result {
            PreprocessedData::PhonemeIds { ids, phonemes, .. } => Some((phonemes, ids)),
            _ => panic!("Expected PhonemeIds"),
        }
    }

    #[test]
    fn test_phonemize_step_cmu_normalize_true_expands_numbers() {
        let Some((phonemes, ids)) =
            run_phonemize_step_with_backend("82", &PhonemizerBackend::CmuDictionary, true)
        else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        // With normalize_text=true, "82" → "eighty two" → phonemes
        // The phoneme output should be non-trivial (real words produce phonemes)
        assert!(
            !phonemes.is_empty(),
            "CmuDictionary with normalize_text=true should produce phonemes for '82' (expanded to 'eighty two'), got empty"
        );
        // Should produce token IDs beyond just padding
        assert!(
            ids.len() > 2,
            "CmuDictionary with normalize_text=true should produce token IDs for '82', got: {:?}",
            ids
        );
    }

    #[test]
    fn test_phonemize_step_misaki_normalize_true_expands_numbers() {
        let Some((phonemes, ids)) =
            run_phonemize_step_with_backend("82", &PhonemizerBackend::MisakiDictionary, true)
        else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        // With normalize_text=true, "82" → "eighty two" → phonemes
        assert!(
            !phonemes.is_empty(),
            "MisakiDictionary with normalize_text=true should produce phonemes for '82' (expanded to 'eighty two'), got empty"
        );
        assert!(
            ids.len() > 2,
            "MisakiDictionary with normalize_text=true should produce token IDs for '82', got: {:?}",
            ids
        );
    }

    #[test]
    fn test_phonemize_step_normalize_false_does_not_expand_numbers() {
        let Some((phonemes_no_norm, ids_no_norm)) =
            run_phonemize_step_with_backend("82", &PhonemizerBackend::MisakiDictionary, false)
        else {
            eprintln!("Skipping: Kokoro fixtures not found");
            return;
        };
        let Some((phonemes_norm, ids_norm)) =
            run_phonemize_step_with_backend("82", &PhonemizerBackend::MisakiDictionary, true)
        else {
            return;
        };

        // With normalize_text=false, "82" is passed as-is to the phonemizer.
        // It should either produce no phonemes (not a dictionary word) or
        // produce different/fewer phonemes than the normalized version.
        assert_ne!(
            phonemes_no_norm, phonemes_norm,
            "normalize_text=false should produce different phonemes than normalize_text=true for '82'. \
             no_norm='{}', norm='{}'",
            phonemes_no_norm, phonemes_norm
        );

        // The normalized version should produce more token IDs (real words have more phonemes)
        assert!(
            ids_norm.len() > ids_no_norm.len(),
            "Normalized '82' (→ 'eighty two') should produce more tokens than raw '82'. \
             norm_ids={:?}, no_norm_ids={:?}",
            ids_norm,
            ids_no_norm
        );
    }

    // ============================================================================
    // Text Normalization Edge Case Tests (US-003)
    // ============================================================================

    #[test]
    fn test_normalize_currency_dollar() {
        let result = normalize_text_for_tts("$3.50");
        assert!(
            !result.contains('$'),
            "Dollar sign should be removed after normalization, got: {}",
            result
        );
        assert!(
            result.contains("three") || result.contains("3"),
            "Currency amount should be expanded, got: {}",
            result
        );
    }

    #[test]
    fn test_normalize_percentage() {
        let result = normalize_text_for_tts("100%");
        assert!(
            !result.contains('%'),
            "Percent sign should be removed after normalization, got: {}",
            result
        );
        assert!(
            result.to_lowercase().contains("percent"),
            "Percentage should be expanded to 'percent', got: {}",
            result
        );
    }

    #[test]
    fn test_normalize_usa_abbreviation() {
        // Should not crash and should produce some output
        let result = normalize_text_for_tts("U.S.A.");
        assert!(
            !result.is_empty(),
            "U.S.A. normalization should not produce empty string"
        );
    }

    #[test]
    fn test_normalize_dr_expansion() {
        let result = normalize_text_for_tts("Dr.");
        assert!(
            result.contains("Doctor") || result.contains("doctor"),
            "Dr. should expand to Doctor/doctor, got: {}",
            result
        );
    }

    #[test]
    fn test_normalize_smart_quotes() {
        let input = "\u{201C}Hello\u{201D}";
        let result = normalize_text_for_tts(input);
        assert!(
            result.contains('"'),
            "Smart quotes should normalize to ASCII quotes, got: {}",
            result
        );
        assert!(
            !result.contains('\u{201C}') && !result.contains('\u{201D}'),
            "Smart quotes should be replaced, got: {}",
            result
        );
    }

    // ============================================================================
    // KittenTTS Token Mapping Validation (US-002)
    // Verifies Misaki-produced phonemes map correctly to KittenTTS token IDs
    // ============================================================================

    /// Path to the KittenTTS model fixtures directory.
    fn kittentts_fixture_path() -> String {
        let workspace_path = "repos/xybrid/integration-tests/fixtures/models/kitten-tts-nano-0.2";
        if std::path::Path::new(workspace_path).join("misaki").exists() {
            return workspace_path.to_string();
        }
        let crate_path = "../../integration-tests/fixtures/models/kitten-tts-nano-0.2";
        if std::path::Path::new(crate_path).join("misaki").exists() {
            return crate_path.to_string();
        }
        env!("CARGO_MANIFEST_DIR").to_string()
            + "/../../integration-tests/fixtures/models/kitten-tts-nano-0.2"
    }

    /// Check if KittenTTS fixtures are available.
    fn has_kittentts_fixtures() -> bool {
        let base = kittentts_fixture_path();
        let p = std::path::Path::new(&base);
        p.join("tokens.txt").exists() && p.join("misaki").join("us_gold.json").exists()
    }

    /// Load the KittenTTS tokens vocabulary map.
    fn kittentts_vocab() -> HashMap<char, i64> {
        use crate::phonemizer::load_tokens_map;
        let base = kittentts_fixture_path();
        let tokens_path = std::path::Path::new(&base).join("tokens.txt");
        let tokens_content = std::fs::read_to_string(&tokens_path)
            .unwrap_or_else(|e| panic!("Failed to read tokens.txt at {:?}: {}", tokens_path, e));
        load_tokens_map(&tokens_content)
    }

    #[test]
    #[ignore]
    fn test_kittentts_misaki_token_mapping_validation() {
        if !has_kittentts_fixtures() {
            eprintln!("Skipping: KittenTTS fixtures not found");
            return;
        }

        let vocab = kittentts_vocab();
        let base = kittentts_fixture_path();
        let backend = MisakiBackend::new(base);

        let test_phrases = [
            "Hello world",
            "The year was 1984",
            "Dr. Smith has 3 cats",
            "Good morning everyone",
            "This costs five dollars",
        ];

        let mut all_unmapped: Vec<(String, Vec<char>)> = Vec::new();

        for phrase in &test_phrases {
            // Normalize text first (as the pipeline would)
            let normalized = normalize_text_for_tts(phrase);
            let phonemes = backend
                .phonemize(&normalized, &vocab)
                .unwrap_or_else(|e| panic!("Phonemization failed for '{}': {}", phrase, e));

            assert!(
                !phonemes.is_empty(),
                "Phonemization of '{}' (normalized: '{}') produced empty output",
                phrase,
                normalized
            );

            // Check every character in phoneme output maps to a valid token ID
            let unmapped: Vec<char> = phonemes
                .chars()
                .filter(|c| !vocab.contains_key(c) && *c != ' ')
                .collect();

            if !unmapped.is_empty() {
                all_unmapped.push((phrase.to_string(), unmapped));
            }
        }

        if !all_unmapped.is_empty() {
            let details: Vec<String> = all_unmapped
                .iter()
                .map(|(phrase, chars)| {
                    let char_details: Vec<String> = chars
                        .iter()
                        .map(|c| format!("'{}' (U+{:04X})", c, *c as u32))
                        .collect();
                    format!(
                        "  '{}': unmapped chars: [{}]",
                        phrase,
                        char_details.join(", ")
                    )
                })
                .collect();
            panic!(
                "Token mapping validation failed — phoneme characters not in tokens.txt:\n{}",
                details.join("\n")
            );
        }
    }
}
