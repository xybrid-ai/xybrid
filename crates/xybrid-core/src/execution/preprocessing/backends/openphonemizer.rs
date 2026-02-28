//! OpenPhonemizer backend — dictionary + ONNX neural fallback.
//!
//! Provides grapheme-to-phoneme conversion without system dependencies.
//! Uses a ~274k word dictionary for fast lookup and a small ONNX model (~59MB)
//! for unknown words via CTC decoding.
//!
//! Ported from the KittenTTS-rust implementation, originally based on
//! the maise project's Kotlin OpenPhonemizer for Kokoro TTS.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use regex::Regex;

use crate::execution::types::ExecutorResult;
use crate::runtime_adapter::AdapterError;

use super::PhonemizerBackend;

// CTC decoding constants
const BLANK_ID: i32 = 0; // "_" — blank/pad
const END_ID: i32 = 2; // "<end>"

/// Number of times each character is repeated in the input tensor.
const CHAR_REPEATS: usize = 3;

/// Maximum input tensor length for the ONNX model.
const MAX_INPUT_LEN: usize = 64;

/// Punctuation that attaches to the preceding word (no leading space).
const PUNCTUATION_BEFORE: &[&str] = &[
    ".", ",", "!", "?", ";", ":", ")", "]", "}", "\u{00BB}", "\u{201D}",
];

/// Punctuation that attaches to the following word (no trailing space).
const PUNCTUATION_AFTER: &[&str] = &["(", "[", "{", "\u{00AB}", "\u{201C}"];

/// Input tokenizer: maps characters to token IDs for the ONNX model input.
/// 53 symbols: _ (0), a-z (3-28), A-Z (29-54).
/// IDs 1 (<en_us>) and 2 (<end>) are multi-char tokens handled separately.
fn build_input_symbol_table() -> HashMap<char, i32> {
    let mut map = HashMap::with_capacity(53);
    map.insert('_', 0);
    for (i, c) in ('a'..='z').enumerate() {
        map.insert(c, (i as i32) + 3);
    }
    for (i, c) in ('A'..='Z').enumerate() {
        map.insert(c, (i as i32) + 29);
    }
    map
}

/// Output tokenizer: maps ONNX model output token IDs to IPA phoneme strings.
/// 64 symbols (IDs 0-63) produced by the open-phonemizer model.
fn build_output_phoneme_table() -> HashMap<i32, &'static str> {
    HashMap::from([
        (0, "_"),
        (1, "<en_us>"),
        (2, "<end>"),
        (3, "a"),
        (4, "b"),
        (5, "d"),
        (6, "e"),
        (7, "f"),
        (8, "g"),
        (9, "h"),
        (10, "i"),
        (11, "j"),
        (12, "k"),
        (13, "l"),
        (14, "m"),
        (15, "n"),
        (16, "o"),
        (17, "p"),
        (18, "r"),
        (19, "s"),
        (20, "t"),
        (21, "u"),
        (22, "v"),
        (23, "w"),
        (24, "x"),
        (25, "y"),
        (26, "z"),
        (27, "\u{00E6}"), // æ
        (28, "\u{00E7}"), // ç
        (29, "\u{00F0}"), // ð
        (30, "\u{00F8}"), // ø
        (31, "\u{014B}"), // ŋ
        (32, "\u{0153}"), // œ
        (33, "\u{0250}"), // ɐ
        (34, "\u{0251}"), // ɑ
        (35, "\u{0254}"), // ɔ
        (36, "\u{0259}"), // ə
        (37, "\u{025B}"), // ɛ
        (38, "\u{025C}"), // ɜ
        (39, "\u{025D}"), // ɝ
        (40, "\u{0279}"), // ɹ
        (41, "\u{025A}"), // ɚ
        (42, "\u{0261}"), // ɡ
        (43, "\u{026A}"), // ɪ
        (44, "\u{0281}"), // ʁ
        (45, "\u{0283}"), // ʃ
        (46, "\u{028A}"), // ʊ
        (47, "\u{028C}"), // ʌ
        (48, "\u{028F}"), // ʏ
        (49, "\u{0292}"), // ʒ
        (50, "\u{0294}"), // ʔ
        (51, "\u{02C8}"), // ˈ
        (52, "\u{02CC}"), // ˌ
        (53, "\u{02D0}"), // ː
        (54, "\u{0303}"), // ̃  (combining tilde)
        (55, "\u{030D}"), // ̍  (combining vertical line above)
        (56, "\u{0325}"), // ̥  (combining ring below)
        (57, "\u{0329}"), // ̩  (combining vertical line below)
        (58, "\u{032F}"), // ̯  (combining inverted breve below)
        (59, "\u{0361}"), // ͡  (combining double inverted breve)
        (60, "\u{03B8}"), // θ
        (61, "'"),
        (62, "\u{027E}"), // ɾ
        (63, "\u{1D7B}"), // ᵻ
    ])
}

/// Internal state loaded lazily on first phonemize call.
struct OpenPhonemizerState {
    dictionary: HashMap<String, String>,
    session: Mutex<Session>,
    input_symbols: HashMap<char, i32>,
    output_phonemes: HashMap<i32, &'static str>,
    word_regex: Regex,
}

/// OpenPhonemizer backend — hybrid dictionary + ONNX neural G2P.
///
/// Requires `open-phonemizer.onnx` (~59MB) and `dictionary.json` (~10MB) in
/// the model directory. No system dependencies needed.
///
/// On first call, lazily loads the dictionary and ONNX session. The ONNX
/// session is wrapped in a `Mutex` to satisfy the `&self` trait requirement
/// while allowing `Session::run(&mut self)`.
pub struct OpenPhonemizerBackend {
    base_path: String,
    state: OnceLock<Result<OpenPhonemizerState, String>>,
}

impl OpenPhonemizerBackend {
    /// Create a new OpenPhonemizerBackend.
    ///
    /// # Arguments
    /// - `base_path`: Path to the model directory containing
    ///   `open-phonemizer.onnx` and `dictionary.json`.
    pub fn new(base_path: String) -> Self {
        Self {
            base_path,
            state: OnceLock::new(),
        }
    }

    /// Initialize the backend state (called lazily on first phonemize).
    fn init_state(base_path: &str) -> Result<OpenPhonemizerState, String> {
        let dir = Path::new(base_path);
        let model_path = dir.join("open-phonemizer.onnx");
        let dict_path = dir.join("dictionary.json");

        if !model_path.exists() {
            return Err(format!(
                "OpenPhonemizer ONNX model not found at {}. \
                 Ensure open-phonemizer.onnx is in the model directory.",
                model_path.display()
            ));
        }
        if !dict_path.exists() {
            return Err(format!(
                "OpenPhonemizer dictionary not found at {}. \
                 Ensure dictionary.json is in the model directory.",
                dict_path.display()
            ));
        }

        let dictionary = Self::load_dictionary(&dict_path)?;
        let session = Session::builder()
            .map_err(|e| format!("Failed to create ONNX session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("Failed to load open-phonemizer.onnx: {e}"))?;

        Ok(OpenPhonemizerState {
            dictionary,
            session: Mutex::new(session),
            input_symbols: build_input_symbol_table(),
            output_phonemes: build_output_phoneme_table(),
            word_regex: Regex::new(r"[\w']+|[^\w\s]").expect("valid regex"),
        })
    }

    /// Load the en_us word→phonemes dictionary from dictionary.json.
    /// Structure: `{"en_us": {"word": "IPA phonemes", ...}}`
    fn load_dictionary(path: &Path) -> Result<HashMap<String, String>, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

        let root: serde_json::Value =
            serde_json::from_str(&data).map_err(|e| format!("Failed to parse dictionary: {e}"))?;

        let en_us = root
            .get("en_us")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "Missing 'en_us' key in dictionary.json".to_string())?;

        let mut map = HashMap::with_capacity(en_us.len());
        for (word, phonemes) in en_us {
            if let Some(p) = phonemes.as_str() {
                map.insert(word.clone(), p.to_string());
            }
        }
        Ok(map)
    }

    /// Run the ONNX model on a single unknown word to produce IPA phonemes.
    fn phonemize_word(state: &OpenPhonemizerState, word: &str) -> Result<String, String> {
        let cleaned = word.to_lowercase().replace(' ', "_");

        let mut encoded = Vec::with_capacity(cleaned.len() * CHAR_REPEATS + 2);

        // Start token: <en_us> = 1
        encoded.push(1i64);

        for c in cleaned.chars() {
            if let Some(&id) = state.input_symbols.get(&c) {
                for _ in 0..CHAR_REPEATS {
                    encoded.push(id as i64);
                }
            }
        }

        // End token: <end> = 2
        encoded.push(2i64);

        // Pad to MAX_INPUT_LEN
        let mut padded = vec![0i64; MAX_INPUT_LEN];
        let copy_len = encoded.len().min(MAX_INPUT_LEN);
        padded[..copy_len].copy_from_slice(&encoded[..copy_len]);

        // Create input tensor [1, 64]
        let input_tensor = Tensor::from_array(
            ndarray::Array2::from_shape_vec((1, MAX_INPUT_LEN), padded)
                .map_err(|e| format!("Failed to create input array: {e}"))?,
        )
        .map_err(|e| format!("Failed to create ONNX tensor: {e}"))?;

        // Lock session for mutable access
        let mut session = state
            .session
            .lock()
            .map_err(|e| format!("Failed to lock ONNX session: {e}"))?;

        let outputs = session
            .run(ort::inputs!["text" => input_tensor])
            .map_err(|e| format!("ONNX inference failed: {e}"))?;

        // Output shape: [1, 64, 64] — (batch=1, seq_len=64, vocab=64)
        let output_array = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Failed to extract output: {e}"))?;

        let shape = output_array.shape();
        let seq_len = shape[1];
        let vocab_size = shape[2];

        // Argmax across vocab dimension for each position
        let mut argmax = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let mut best_id = 0;
            let mut best_val = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let val = output_array[[0, pos, v]];
                if val > best_val {
                    best_val = val;
                    best_id = v as i32;
                }
            }
            argmax.push(best_id);
        }

        // CTC decode: collapse consecutive duplicates, remove blanks, stop at <end>
        let mut result = String::new();
        let mut prev = -1i32;
        for &id in &argmax {
            if id == prev {
                continue;
            }
            prev = id;
            if id == BLANK_ID {
                continue;
            }
            if id == END_ID {
                break;
            }
            if let Some(sym) = state.output_phonemes.get(&id) {
                if !sym.starts_with('<') {
                    result.push_str(sym);
                }
            }
        }

        Ok(result)
    }
}

impl PhonemizerBackend for OpenPhonemizerBackend {
    fn phonemize(&self, text: &str, tokens_map: &HashMap<char, i64>) -> ExecutorResult<String> {
        let state = self.state.get_or_init(|| Self::init_state(&self.base_path));

        let state = state.as_ref().map_err(|e| {
            AdapterError::InvalidInput(format!("OpenPhonemizer initialization failed: {e}"))
        })?;

        let tokens: Vec<&str> = state
            .word_regex
            .find_iter(text)
            .map(|m| m.as_str())
            .collect();

        let mut result = String::with_capacity(text.len() * 2);

        for token in &tokens {
            let is_punct = !token.chars().any(|c| c.is_alphanumeric() || c == '\'');

            if is_punct {
                if PUNCTUATION_BEFORE.contains(token) {
                    if result.ends_with(' ') {
                        result.pop();
                    }
                    result.push_str(token);
                } else if PUNCTUATION_AFTER.contains(token) {
                    result.push_str(token);
                } else {
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                    result.push_str(token);
                }
            } else {
                if !result.is_empty()
                    && !result.ends_with(' ')
                    && !PUNCTUATION_AFTER.iter().any(|p| result.ends_with(p))
                {
                    result.push(' ');
                }

                let word = token.to_lowercase();
                if let Some(phonemes) = state.dictionary.get(&word) {
                    result.push_str(phonemes);
                } else {
                    let phonemes = Self::phonemize_word(state, &word).map_err(|e| {
                        AdapterError::InferenceFailed(format!(
                            "OpenPhonemizer neural fallback failed for '{word}': {e}"
                        ))
                    })?;
                    result.push_str(&phonemes);
                }
            }
        }

        let trimmed = result.trim().to_string();

        // Filter to only characters in vocabulary (matches espeak/misaki pattern)
        let filtered: String = trimmed
            .chars()
            .filter(|c| tokens_map.contains_key(c))
            .collect();

        Ok(filtered)
    }

    fn name(&self) -> &'static str {
        "OpenPhonemizer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_symbol_table() {
        let table = build_input_symbol_table();
        assert_eq!(table[&'_'], 0);
        assert_eq!(table[&'a'], 3);
        assert_eq!(table[&'z'], 28);
        assert_eq!(table[&'A'], 29);
        assert_eq!(table[&'Z'], 54);
        assert_eq!(table.len(), 53); // _ + 26 lowercase + 26 uppercase
    }

    #[test]
    fn test_output_phoneme_table() {
        let table = build_output_phoneme_table();
        assert_eq!(table[&0], "_");
        assert_eq!(table[&1], "<en_us>");
        assert_eq!(table[&2], "<end>");
        assert_eq!(table[&27], "\u{00E6}"); // æ
        assert_eq!(table[&60], "\u{03B8}"); // θ
        assert_eq!(table.len(), 64);
    }
}
