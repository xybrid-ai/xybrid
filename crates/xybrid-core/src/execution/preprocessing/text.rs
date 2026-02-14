//! Text preprocessing operations.
//!
//! This module provides:
//! - `tokenize_step`: Tokenize text for NLP models
//! - `phonemize_step`: Convert text to phonemes for TTS models

use super::super::types::{ExecutorResult, PreprocessedData};
use crate::execution::template::{PhonemizerBackend, TokenizerType};
use crate::runtime_adapter::AdapterError;
use std::collections::HashMap;

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

    // Convert text to IPA phonemes based on backend
    let phonemes = match backend {
        PhonemizerBackend::CmuDictionary => {
            use crate::phonemizer::Phonemizer;

            let phonemizer = if let Some(path) = dict_path {
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

            phonemizer.phonemize(&processed_text)
        }
        PhonemizerBackend::EspeakNG => {
            phonemize_with_espeak(&processed_text, language.unwrap_or("en-us"), &tokens_map)?
        }
        PhonemizerBackend::MisakiDictionary => {
            // Derive base_path from tokens_path (go up one directory from tokens.txt)
            let tokens_dir = std::path::Path::new(tokens_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            phonemize_with_misaki(
                &processed_text,
                tokens_dir.to_str().unwrap_or("."),
                &tokens_map,
            )?
        }
    };

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
const PHONEME_LINK_START: char = '\x01';
const PHONEME_LINK_END: char = '\x02';

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

/// Phonemize text using espeak-ng backend.
///
/// This function calls espeak-ng as an external command to convert text to IPA phonemes,
/// then filters the output to only include characters in the vocabulary.
///
/// Requires espeak-ng to be installed on the system:
/// - macOS: `brew install espeak-ng`
/// - Linux: `apt-get install espeak-ng`
fn phonemize_with_espeak(
    text: &str,
    language: &str,
    vocab: &HashMap<char, i64>,
) -> ExecutorResult<String> {
    use std::process::Command;

    // Call espeak-ng with IPA output
    let output = Command::new("espeak-ng")
        .args(["--ipa", "-q", "-v", language])
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
    let filtered: String = phonemes.chars().filter(|c| vocab.contains_key(c)).collect();

    Ok(filtered.trim().to_string())
}

/// Phonemize text using Misaki dictionary-based backend.
///
/// This function uses bundled JSON dictionaries (us_gold.json, us_silver.json) to convert
/// text to IPA phonemes without any system dependencies.
///
/// Key behaviors matching the official Python Misaki pipeline:
/// - Dictionary lookup with case variant generation (grow pattern)
/// - Morphological stemming for -s, -ed, -ing suffixes
/// - Post-lookup replacement of ɾ→T and ʔ→t (the model was trained with these substitutions)
/// - Support for explicit phoneme overrides via `[word](/phonemes/)` syntax
fn phonemize_with_misaki(
    text: &str,
    base_path: &str,
    vocab: &HashMap<char, i64>,
) -> ExecutorResult<String> {
    // Load dictionaries
    let misaki_dir = std::path::Path::new(base_path).join("misaki");
    let gold_path = misaki_dir.join("us_gold.json");
    let silver_path = misaki_dir.join("us_silver.json");

    // Parse dictionaries
    let gold_dict: HashMap<String, serde_json::Value> = if gold_path.exists() {
        let content = std::fs::read_to_string(&gold_path).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to read misaki gold dictionary: {}", e))
        })?;
        serde_json::from_str(&content).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to parse misaki gold dictionary: {}", e))
        })?
    } else {
        HashMap::new()
    };

    let silver_dict: HashMap<String, serde_json::Value> = if silver_path.exists() {
        let content = std::fs::read_to_string(&silver_path).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to read misaki silver dictionary: {}", e))
        })?;
        serde_json::from_str(&content).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to parse misaki silver dictionary: {}", e))
        })?
    } else {
        HashMap::new()
    };

    if gold_dict.is_empty() && silver_dict.is_empty() {
        return Err(AdapterError::InvalidInput(
            "No misaki dictionaries found. Expected us_gold.json and us_silver.json in misaki/ directory".to_string()
        ));
    }

    // Apply the "grow dictionary" pattern from the official pipeline:
    // For lowercase entries, also generate a Capitalized variant and vice versa.
    let gold_grown = grow_dictionary(&gold_dict);
    let silver_grown = grow_dictionary(&silver_dict);

    // Tokenize by whitespace and process each word, preserving punctuation as prosody cues.
    let mut result = String::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        // Check if this word is a phoneme link override (wrapped in sentinel chars)
        if word.starts_with(PHONEME_LINK_START) && word.ends_with(PHONEME_LINK_END) {
            let phonemes =
                &word[PHONEME_LINK_START.len_utf8()..word.len() - PHONEME_LINK_END.len_utf8()];
            result.push_str(phonemes);
            if i < words.len() - 1 {
                result.push(' ');
            }
            continue;
        }

        // Also handle phoneme links that might be attached to punctuation
        if word.contains(PHONEME_LINK_START) {
            for segment in word.split([PHONEME_LINK_START, PHONEME_LINK_END]) {
                if segment.is_empty() {
                    continue;
                }
                // If this segment was between sentinels, it's a phoneme override
                // We can detect this by checking the original positions
                result.push_str(segment);
            }
            if i < words.len() - 1 {
                result.push(' ');
            }
            continue;
        }

        // Extract leading and trailing punctuation, keeping the alphabetic core.
        let trimmed_start = word.trim_start_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        let leading_punct = &word[..word.len() - trimmed_start.len()];
        let clean_word =
            trimmed_start.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        let trailing_punct = &trimmed_start[clean_word.len()..];

        // Emit leading punctuation (e.g. opening quotes) if in vocab
        for c in leading_punct.chars() {
            if vocab.contains_key(&c) {
                result.push(c);
            }
        }

        // Phonemize the word itself
        if !clean_word.is_empty() {
            // Handle hyphenated compounds by splitting and phonemizing each part
            if clean_word.contains('-') {
                for (j, part) in clean_word.split('-').enumerate() {
                    if j > 0 && !part.is_empty() {
                        // No explicit separator — parts are concatenated naturally
                    }
                    if !part.is_empty() {
                        result.push_str(&phonemize_single_word(part, &gold_grown, &silver_grown));
                    }
                }
            } else {
                result.push_str(&phonemize_single_word(
                    clean_word,
                    &gold_grown,
                    &silver_grown,
                ));
            }
        }

        // Emit trailing punctuation (periods, commas, question marks, etc.) as prosody tokens
        for c in trailing_punct.chars() {
            if vocab.contains_key(&c) {
                result.push(c);
            }
        }

        // Add space between words
        if i < words.len() - 1 {
            result.push(' ');
        }
    }

    // Apply post-phonemization replacements matching the official pipeline.
    // The Kokoro model was trained with these substitutions (misaki/en.py line 733-736):
    //   ɾ (IPA flap T, token 125) → T (uppercase letter, token 36)
    //   ʔ (IPA glottal stop, token 148) → t (lowercase letter, token 62)
    let result = result.replace('ɾ', "T").replace('ʔ', "t");

    // Filter to only characters in vocabulary
    let filtered: String = result.chars().filter(|c| vocab.contains_key(c)).collect();

    Ok(filtered.trim().to_string())
}

/// Apply the "grow dictionary" pattern from the official Misaki pipeline.
///
/// For each entry:
/// - If lowercase (e.g., "hello"), also create a Capitalized variant ("Hello")
/// - If Capitalized (e.g., "Hello"), also create a lowercase variant ("hello")
///
/// The original entries take priority over generated ones.
fn grow_dictionary(
    dict: &HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    let mut grown: HashMap<String, serde_json::Value> = HashMap::new();

    for (k, v) in dict {
        if k.len() < 2 {
            continue;
        }
        let lower = k.to_lowercase();
        if k == &lower {
            // Lowercase entry — generate Capitalized variant
            let mut chars = k.chars();
            if let Some(first) = chars.next() {
                let capitalized: String = first.to_uppercase().chain(chars).collect();
                if capitalized != *k {
                    grown.entry(capitalized).or_insert_with(|| v.clone());
                }
            }
        } else {
            // Check if it's Capitalized (first char upper, rest lower)
            let mut chars = k.chars();
            let first = chars.next().unwrap();
            let rest: String = chars.collect();
            if first.is_uppercase() && rest == rest.to_lowercase() {
                grown.entry(lower).or_insert_with(|| v.clone());
            }
        }
    }

    // Original entries take priority
    for (k, v) in dict {
        grown.insert(k.clone(), v.clone());
    }
    grown
}

/// Try morphological stemming to find a word's pronunciation.
///
/// Handles English inflected forms by stripping suffixes and looking up the stem:
/// - `-s` / `-es` / `-ies`: plural/third person (e.g., "parameters" → "parameter" + "z")
/// - `-ed`: past tense (e.g., "deployed" → "deploy" + "d")
/// - `-ing`: progressive (e.g., "delivering" → "deliver" + "ɪŋ")
fn stem_and_lookup(
    word: &str,
    gold: &HashMap<String, serde_json::Value>,
    silver: &HashMap<String, serde_json::Value>,
) -> Option<String> {
    // Try -s stemming
    if let Some(ps) = stem_s(word, gold, silver) {
        return Some(ps);
    }
    // Try -ed stemming
    if let Some(ps) = stem_ed(word, gold, silver) {
        return Some(ps);
    }
    // Try -ing stemming
    if let Some(ps) = stem_ing(word, gold, silver) {
        return Some(ps);
    }
    None
}

/// Stem -s/-es/-ies suffix and apply phonemic suffix rule.
fn stem_s(
    word: &str,
    gold: &HashMap<String, serde_json::Value>,
    silver: &HashMap<String, serde_json::Value>,
) -> Option<String> {
    if word.len() < 3 || !word.ends_with('s') {
        return None;
    }

    let stem = if !word.ends_with("ss") {
        // Try removing just -s
        let candidate = &word[..word.len() - 1];
        if lookup_word_phonemes(candidate, gold).is_some()
            || lookup_word_phonemes(candidate, silver).is_some()
        {
            Some(candidate.to_string())
        } else if word.len() > 4 && word.ends_with("es") && !word.ends_with("ies") {
            // Try removing -es
            let candidate = &word[..word.len() - 2];
            if lookup_word_phonemes(candidate, gold).is_some()
                || lookup_word_phonemes(candidate, silver).is_some()
            {
                Some(candidate.to_string())
            } else {
                None
            }
        } else if word.len() > 4 && word.ends_with("ies") {
            // -ies → -y (e.g., "countries" → "country")
            let candidate = format!("{}y", &word[..word.len() - 3]);
            if lookup_word_phonemes(&candidate, gold).is_some()
                || lookup_word_phonemes(&candidate, silver).is_some()
            {
                Some(candidate)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    stem.and_then(|s| {
        let stem_ps =
            lookup_word_phonemes(&s, gold).or_else(|| lookup_word_phonemes(&s, silver))?;
        // Apply -s suffix phoneme rule (matching official Misaki _s method)
        let last = stem_ps.chars().last()?;
        let suffix = if "ptkfθ".contains(last) {
            "s"
        } else if "szʃʒʧʤ".contains(last) {
            "ᵻz"
        } else {
            "z"
        };
        Some(format!("{}{}", stem_ps, suffix))
    })
}

/// Stem -ed suffix and apply phonemic suffix rule.
fn stem_ed(
    word: &str,
    gold: &HashMap<String, serde_json::Value>,
    silver: &HashMap<String, serde_json::Value>,
) -> Option<String> {
    if word.len() < 4 || !word.ends_with('d') {
        return None;
    }

    let stem = if !word.ends_with("dd") {
        let candidate = &word[..word.len() - 1]; // remove -d
        if lookup_word_phonemes(candidate, gold).is_some()
            || lookup_word_phonemes(candidate, silver).is_some()
        {
            Some(candidate.to_string())
        } else if word.len() > 4 && word.ends_with("ed") && !word.ends_with("eed") {
            let candidate = &word[..word.len() - 2]; // remove -ed
            if lookup_word_phonemes(candidate, gold).is_some()
                || lookup_word_phonemes(candidate, silver).is_some()
            {
                Some(candidate.to_string())
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    stem.and_then(|s| {
        let stem_ps =
            lookup_word_phonemes(&s, gold).or_else(|| lookup_word_phonemes(&s, silver))?;
        let last = stem_ps.chars().last()?;
        // Apply -ed suffix phoneme rule (matching official Misaki _ed method)
        let suffix = if "pkfθʃsʧ".contains(last) {
            "t"
        } else if last == 'd' || last == 't' {
            "ᵻd"
        } else {
            "d"
        };
        Some(format!("{}{}", stem_ps, suffix))
    })
}

/// Stem -ing suffix and apply phonemic suffix rule.
fn stem_ing(
    word: &str,
    gold: &HashMap<String, serde_json::Value>,
    silver: &HashMap<String, serde_json::Value>,
) -> Option<String> {
    if word.len() < 5 || !word.ends_with("ing") {
        return None;
    }

    // Try removing -ing
    let stem_candidate = &word[..word.len() - 3];
    let stem = if word.len() > 5
        && (lookup_word_phonemes(stem_candidate, gold).is_some()
            || lookup_word_phonemes(stem_candidate, silver).is_some())
    {
        Some(stem_candidate.to_string())
    } else {
        // Try removing -ing and adding -e (e.g., "making" → "make")
        let with_e = format!("{}e", stem_candidate);
        if lookup_word_phonemes(&with_e, gold).is_some()
            || lookup_word_phonemes(&with_e, silver).is_some()
        {
            Some(with_e)
        } else {
            None
        }
    };

    stem.and_then(|s| {
        let stem_ps =
            lookup_word_phonemes(&s, gold).or_else(|| lookup_word_phonemes(&s, silver))?;
        Some(format!("{}ɪŋ", stem_ps))
    })
}

/// Look up a word's phonemes in a misaki dictionary.
fn lookup_word_phonemes(word: &str, dict: &HashMap<String, serde_json::Value>) -> Option<String> {
    dict.get(word).and_then(|v| match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(obj) => {
            // Has POS-specific pronunciations, use DEFAULT
            obj.get("DEFAULT")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string())
        }
        _ => None,
    })
}

/// Phonemize a single word using dictionary lookup, stemming, acronym detection, or G2P rules.
///
/// Resolution order:
/// 1. Dictionary lookup (gold, then silver, lowercase then original case)
/// 2. Morphological stemming (-s, -ed, -ing)
/// 3. Acronym spelling (all-uppercase words like "TTS" → "tˈiː tˈiː ˈɛs")
/// 4. Rule-based grapheme-to-phoneme conversion
fn phonemize_single_word(
    word: &str,
    gold: &HashMap<String, serde_json::Value>,
    silver: &HashMap<String, serde_json::Value>,
) -> String {
    let lower = word.to_lowercase();

    // 1. Dictionary lookup
    if let Some(ps) = lookup_word_phonemes(&lower, gold)
        .or_else(|| lookup_word_phonemes(&lower, silver))
        .or_else(|| lookup_word_phonemes(word, gold))
        .or_else(|| lookup_word_phonemes(word, silver))
    {
        return ps;
    }

    // 2. Morphological stemming
    if let Some(ps) = stem_and_lookup(&lower, gold, silver) {
        return ps;
    }

    // 3. Acronym detection: all-uppercase ASCII letters, 2+ chars
    if word.len() >= 2 && word.chars().all(|c| c.is_ascii_uppercase()) {
        return spell_as_letters(word);
    }

    // 4. Rule-based G2P fallback
    rule_based_g2p(&lower)
}

/// Spell a word as individual letter names for acronyms.
///
/// "TTS" → "tˈiː tˈiː ˈɛs"
/// "API" → "ˈeɪ pˈiː ˈaɪ"
fn spell_as_letters(word: &str) -> String {
    let mut parts = Vec::new();
    for c in word.chars() {
        let phoneme = match c.to_ascii_uppercase() {
            'A' => "ˈeɪ",
            'B' => "bˈiː",
            'C' => "sˈiː",
            'D' => "dˈiː",
            'E' => "ˈiː",
            'F' => "ˈɛf",
            'G' => "ʤˈiː",
            'H' => "ˈeɪʧ",
            'I' => "ˈaɪ",
            'J' => "ʤˈeɪ",
            'K' => "kˈeɪ",
            'L' => "ˈɛl",
            'M' => "ˈɛm",
            'N' => "ˈɛn",
            'O' => "ˈoʊ",
            'P' => "pˈiː",
            'Q' => "kjˈuː",
            'R' => "ˈɑːɹ",
            'S' => "ˈɛs",
            'T' => "tˈiː",
            'U' => "jˈuː",
            'V' => "vˈiː",
            'W' => "dˈʌbəljˌuː",
            'X' => "ˈɛks",
            'Y' => "wˈaɪ",
            'Z' => "zˈiː",
            _ => "",
        };
        if !phoneme.is_empty() {
            parts.push(phoneme);
        }
    }
    parts.join(" ")
}

/// Rule-based grapheme-to-phoneme conversion for out-of-vocabulary English words.
///
/// Uses a greedy left-to-right matching approach with multi-character grapheme rules.
/// Handles consonant digraphs (th, sh, ch, ph), vowel digraphs (ee, ea, ai, ou),
/// silent combinations (gh after vowel, kn, wr), context-sensitive consonants
/// (soft c/g before e/i/y), R-controlled vowels, and common suffixes (-tion, -ight).
fn rule_based_g2p(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    let mut result = String::new();
    let mut i = 0;

    while i < n {
        let remaining = n - i;

        // Try 4-char patterns
        if remaining >= 4 {
            let matched = match (chars[i], chars[i + 1], chars[i + 2], chars[i + 3]) {
                ('i', 'g', 'h', 't') => {
                    result.push_str("aɪt");
                    true
                }
                ('t', 'i', 'o', 'n') => {
                    result.push_str("ʃən");
                    true
                }
                ('s', 'i', 'o', 'n') => {
                    result.push_str("ʒən");
                    true
                }
                ('o', 'u', 'g', 'h') => {
                    result.push_str("ɔː");
                    true
                }
                _ => false,
            };
            if matched {
                i += 4;
                continue;
            }
        }

        // Try 3-char patterns
        if remaining >= 3 {
            let matched = match (chars[i], chars[i + 1], chars[i + 2]) {
                ('t', 'c', 'h') => {
                    result.push('ʧ');
                    true
                }
                ('d', 'g', 'e') => {
                    result.push('ʤ');
                    true
                }
                _ => false,
            };
            if matched {
                i += 3;
                continue;
            }
        }

        // Try 2-char patterns
        if remaining >= 2 {
            let next = chars[i + 1];
            let matched = match (chars[i], next) {
                // Consonant digraphs
                ('t', 'h') => {
                    result.push('θ');
                    true
                }
                ('s', 'h') => {
                    result.push('ʃ');
                    true
                }
                ('c', 'h') => {
                    result.push('ʧ');
                    true
                }
                ('c', 'k') => {
                    result.push('k');
                    true
                }
                ('p', 'h') => {
                    result.push('f');
                    true
                }
                ('w', 'h') => {
                    result.push('w');
                    true
                }
                ('w', 'r') => {
                    result.push('ɹ');
                    true
                }
                ('k', 'n') if i == 0 => {
                    result.push('n');
                    true
                }
                ('g', 'n') if i == 0 => {
                    result.push('n');
                    true
                }
                ('n', 'g') => {
                    result.push('ŋ');
                    true
                }
                ('q', 'u') => {
                    result.push_str("kw");
                    true
                }
                // Silent gh after vowel
                ('g', 'h') if i > 0 => true,
                // Vowel digraphs
                ('e', 'e') => {
                    result.push_str("iː");
                    true
                }
                ('e', 'a') => {
                    result.push_str("iː");
                    true
                }
                ('a', 'i') => {
                    result.push_str("eɪ");
                    true
                }
                ('a', 'y') => {
                    result.push_str("eɪ");
                    true
                }
                ('e', 'i') => {
                    result.push_str("eɪ");
                    true
                }
                ('e', 'y') => {
                    result.push_str("eɪ");
                    true
                }
                ('o', 'a') => {
                    result.push_str("oʊ");
                    true
                }
                ('o', 'o') => {
                    result.push_str("uː");
                    true
                }
                ('o', 'u') => {
                    result.push_str("aʊ");
                    true
                }
                ('o', 'w') => {
                    result.push_str("oʊ");
                    true
                }
                ('o', 'i') => {
                    result.push_str("ɔɪ");
                    true
                }
                ('o', 'y') => {
                    result.push_str("ɔɪ");
                    true
                }
                ('u', 'e') => {
                    result.push_str("uː");
                    true
                }
                ('e', 'w') => {
                    result.push_str("uː");
                    true
                }
                ('a', 'u') => {
                    result.push_str("ɔː");
                    true
                }
                ('a', 'w') => {
                    result.push_str("ɔː");
                    true
                }
                ('i', 'e') => {
                    result.push_str("iː");
                    true
                }
                // R-controlled vowels
                ('a', 'r') => {
                    result.push_str("ɑːɹ");
                    true
                }
                ('e', 'r') => {
                    result.push('ɚ');
                    true
                }
                ('i', 'r') => {
                    result.push('ɝ');
                    true
                }
                ('o', 'r') => {
                    result.push_str("ɔːɹ");
                    true
                }
                ('u', 'r') => {
                    result.push('ɝ');
                    true
                }
                _ => false,
            };
            if matched {
                i += 2;
                continue;
            }
        }

        // Single character with context awareness
        let c = chars[i];
        let next_ch = chars.get(i + 1);

        match c {
            'a' => result.push('æ'),
            'e' => result.push('ɛ'),
            'i' => result.push('ɪ'),
            'o' => result.push('ɑ'),
            'u' => result.push('ʌ'),
            'y' => {
                if i == 0 {
                    result.push('j');
                } else {
                    result.push('ɪ');
                }
            }
            // Soft c before e, i, y
            'c' => {
                if next_ch.is_some_and(|&nc| "eiy".contains(nc)) {
                    result.push('s');
                } else {
                    result.push('k');
                }
            }
            // Soft g before e, i, y
            'g' => {
                if next_ch.is_some_and(|&nc| "eiy".contains(nc)) {
                    result.push('ʤ');
                } else {
                    result.push('ɡ');
                }
            }
            'b' => result.push('b'),
            'd' => result.push('d'),
            'f' => result.push('f'),
            'h' => result.push('h'),
            'j' => result.push('ʤ'),
            'k' => result.push('k'),
            'l' => result.push('l'),
            'm' => result.push('m'),
            'n' => result.push('n'),
            'p' => result.push('p'),
            'r' => result.push('ɹ'),
            's' => result.push('s'),
            't' => result.push('t'),
            'v' => result.push('v'),
            'w' => result.push('w'),
            'x' => result.push_str("ks"),
            'z' => result.push('z'),
            _ => {} // Skip non-alphabetic characters
        }
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let result = phonemize_with_misaki("Lost. Can", &base, &vocab).unwrap();
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
        let result = phonemize_with_misaki("Beer, Ale", &base, &vocab).unwrap();
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
        let result = phonemize_with_misaki("help you?", &base, &vocab).unwrap();
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
        // After normalization, "..." becomes "…" (token 10)
        let result = phonemize_with_misaki("look\u{2026} Lost", &base, &vocab).unwrap();
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
        // Standalone "?" should still produce the ? token
        let result = phonemize_with_misaki("yes ?", &base, &vocab).unwrap();
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
}
