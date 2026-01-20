//! Text preprocessing operations.
//!
//! This module provides:
//! - `tokenize_step`: Tokenize text for NLP models
//! - `phonemize_step`: Convert text to phonemes for TTS models

use super::super::types::{ExecutorResult, PreprocessedData};
use crate::execution_template::{PhonemizerBackend, TokenizerType};
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
/// - Normalize quotes and special characters
/// - Expand common abbreviations (Dr., Mr., etc.)
/// - Clean up whitespace
pub fn normalize_text_for_tts(text: &str) -> String {
    let mut result = text.to_string();

    // Normalize quotes
    result = result.replace('\u{2018}', "'").replace('\u{2019}', "'");
    result = result.replace('\u{201C}', "\"").replace('\u{201D}', "\"");

    // Expand common abbreviations
    result = result.replace("Dr.", "Doctor");
    result = result.replace("Mr.", "Mister");
    result = result.replace("Mrs.", "Missus");
    result = result.replace("Ms.", "Miss");
    result = result.replace("etc.", "etcetera");

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

    // Simple tokenization: split by whitespace and punctuation
    let mut result = String::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        // Clean punctuation from word edges
        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        let lower_word = clean_word.to_lowercase();

        // Try gold dict first, then silver
        let phonemes = lookup_word_phonemes(&lower_word, &gold_dict)
            .or_else(|| lookup_word_phonemes(&lower_word, &silver_dict))
            .or_else(|| lookup_word_phonemes(clean_word, &gold_dict))
            .or_else(|| lookup_word_phonemes(clean_word, &silver_dict))
            .unwrap_or_else(|| fallback_phonemize(&lower_word));

        result.push_str(&phonemes);

        // Add space between words (will be filtered if not in vocab)
        if i < words.len() - 1 {
            result.push(' ');
        }
    }

    // Filter to only characters in vocabulary
    let filtered: String = result.chars().filter(|c| vocab.contains_key(c)).collect();

    Ok(filtered.trim().to_string())
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

/// Fallback phonemization for out-of-vocabulary words.
/// Uses simple letter-to-phoneme mapping for English.
fn fallback_phonemize(word: &str) -> String {
    let mut result = String::new();
    for c in word.chars() {
        match c.to_ascii_lowercase() {
            'a' => result.push_str("æ"),
            'e' => result.push_str("ɛ"),
            'i' => result.push_str("ɪ"),
            'o' => result.push_str("ɑ"),
            'u' => result.push_str("ʌ"),
            'b' => result.push_str("b"),
            'c' => result.push_str("k"),
            'd' => result.push_str("d"),
            'f' => result.push_str("f"),
            'g' => result.push_str("ɡ"),
            'h' => result.push_str("h"),
            'j' => result.push_str("ʤ"),
            'k' => result.push_str("k"),
            'l' => result.push_str("l"),
            'm' => result.push_str("m"),
            'n' => result.push_str("n"),
            'p' => result.push_str("p"),
            'q' => result.push_str("k"),
            'r' => result.push_str("ɹ"),
            's' => result.push_str("s"),
            't' => result.push_str("t"),
            'v' => result.push_str("v"),
            'w' => result.push_str("w"),
            'x' => result.push_str("ks"),
            'y' => result.push_str("j"),
            'z' => result.push_str("z"),
            _ => {} // Skip non-alphabetic characters
        }
    }
    result
}
