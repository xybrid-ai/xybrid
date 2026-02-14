//! Misaki dictionary-based phonemizer backend.
//!
//! Uses bundled JSON dictionaries (us_gold.json, us_silver.json) to convert
//! text to IPA phonemes without any system dependencies.
//!
//! Key behaviors matching the official Python Misaki pipeline:
//! - Dictionary lookup with case variant generation (grow pattern)
//! - Morphological stemming for -s, -ed, -ing suffixes
//! - Post-lookup replacement of ɾ→T and ʔ→t (the model was trained with these substitutions)
//! - Support for explicit phoneme overrides via `[word](/phonemes/)` syntax

use std::collections::HashMap;

use crate::execution::preprocessing::text::{PHONEME_LINK_END, PHONEME_LINK_START};
use crate::execution::types::ExecutorResult;
use crate::runtime_adapter::AdapterError;

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
        // Load dictionaries
        let misaki_dir = std::path::Path::new(&self.base_path).join("misaki");
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
                AdapterError::InvalidInput(format!(
                    "Failed to read misaki silver dictionary: {}",
                    e
                ))
            })?;
            serde_json::from_str(&content).map_err(|e| {
                AdapterError::InvalidInput(format!(
                    "Failed to parse misaki silver dictionary: {}",
                    e
                ))
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
                    result.push_str(segment);
                }
                if i < words.len() - 1 {
                    result.push(' ');
                }
                continue;
            }

            // Extract leading and trailing punctuation, keeping the alphabetic core.
            let trimmed_start =
                word.trim_start_matches(|c: char| !c.is_alphanumeric() && c != '\'');
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
                            result.push_str(&phonemize_single_word(
                                part,
                                &gold_grown,
                                &silver_grown,
                            ));
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

    fn name(&self) -> &'static str {
        "MisakiDictionary"
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

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
    if let Some(ps) = stem_s(word, gold, silver) {
        return Some(ps);
    }
    if let Some(ps) = stem_ed(word, gold, silver) {
        return Some(ps);
    }
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
            'c' => {
                if next_ch.is_some_and(|&nc| "eiy".contains(nc)) {
                    result.push('s');
                } else {
                    result.push('k');
                }
            }
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
            _ => {}
        }
        i += 1;
    }
    result
}
