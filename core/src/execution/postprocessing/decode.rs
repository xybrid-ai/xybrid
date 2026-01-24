//! Decoding postprocessing operations.
//!
//! This module provides:
//! - `ctc_decode_step`: CTC decoding for ASR models (Wav2Vec2)
//! - `bpe_decode_step`: BPE token decoding
//! - `whisper_decode_step`: Whisper token decoding using HuggingFace tokenizer

use super::super::types::{ExecutorResult, RawOutputs};
use crate::runtime_adapter::AdapterError;
use ndarray::IxDyn;

/// CTC decoding for Wav2Vec2-style models.
///
/// # Arguments
/// - `data`: Input data (TensorMap with logits)
/// - `vocab_path`: Path to vocabulary file
/// - `blank_index`: Index of the blank token in vocabulary
pub fn ctc_decode_step(
    data: RawOutputs,
    vocab_path: &str,
    blank_index: usize,
) -> ExecutorResult<RawOutputs> {
    let tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "CTCDecode requires tensor map".to_string(),
            ))
        }
    };

    // Get logits tensor (usually "logits" output)
    let logits = tensor_map
        .values()
        .next()
        .ok_or_else(|| AdapterError::InvalidInput("No outputs for CTCDecode".to_string()))?;

    let shape = logits.shape();
    // Expected shape: [batch, time_steps, vocab_size]
    if shape.len() != 3 {
        return Err(AdapterError::InvalidInput(format!(
            "CTCDecode expects 3D tensor [batch, time, vocab], got {:?}",
            shape
        )));
    }

    let _batch_size = shape[0];
    let time_steps = shape[1];
    let vocab_size = shape[2];

    // Simple greedy CTC decoding:
    // 1. Take argmax over vocab dimension for each timestep
    // 2. Remove consecutive duplicates
    // 3. Remove blank tokens

    let mut token_ids = Vec::new();
    let mut prev_id: Option<usize> = None;

    for t in 0..time_steps {
        // Get argmax over vocab dimension
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;

        for v in 0..vocab_size {
            let val = logits[IxDyn(&[0, t, v])]; // batch=0 for simplicity
            if val > max_val {
                max_val = val;
                max_idx = v;
            }
        }

        // Skip blank tokens and consecutive duplicates
        if max_idx != blank_index && Some(max_idx) != prev_id {
            token_ids.push(max_idx);
        }

        prev_id = Some(max_idx);
    }

    // Load vocabulary and decode
    let text = decode_ctc_tokens(&token_ids, vocab_path)?;

    Ok(RawOutputs::Text(text))
}

/// BPE token decoding.
///
/// # Arguments
/// - `data`: Input data (TokenIds)
/// - `vocab_path`: Path to vocabulary file
pub fn bpe_decode_step(data: RawOutputs, vocab_path: &str) -> ExecutorResult<RawOutputs> {
    let token_ids = match data {
        RawOutputs::TokenIds(ids) => ids,
        _ => {
            return Err(AdapterError::InvalidInput(
                "BPEDecode requires token IDs".to_string(),
            ))
        }
    };

    let text = decode_bpe_tokens(&token_ids, vocab_path)?;

    Ok(RawOutputs::Text(text))
}

/// Whisper token decoding using HuggingFace tokenizer.
///
/// # Arguments
/// - `data`: Input data (TokenIds)
/// - `tokenizer_path`: Path to tokenizer.json file
pub fn whisper_decode_step(data: RawOutputs, tokenizer_path: &str) -> ExecutorResult<RawOutputs> {
    let token_ids = match data {
        RawOutputs::TokenIds(ids) => ids,
        _ => {
            return Err(AdapterError::InvalidInput(
                "WhisperDecode requires token IDs".to_string(),
            ))
        }
    };

    let text = decode_whisper_tokens(&token_ids, tokenizer_path)?;

    Ok(RawOutputs::Text(text))
}

/// Decode CTC tokens to text (for Wav2Vec2-style models).
fn decode_ctc_tokens(token_ids: &[usize], vocab_path: &str) -> ExecutorResult<String> {
    // Load vocabulary
    let content = std::fs::read_to_string(vocab_path)
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to read vocab file: {}", e)))?;

    // Try to parse as JSON first (Wav2Vec2 format: {"char": id, ...})
    let json_vocab = if content.trim().starts_with('{') {
        // Parse JSON vocab: {"'": 27, "A": 7, "B": 24, ...}
        let json_vocab = serde_json::from_str::<std::collections::HashMap<String, usize>>(&content)
            .map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to parse vocab JSON: {}", e))
            })?;

        // Create reverse mapping: id -> char
        let max_id = json_vocab.values().max().copied().unwrap_or(0);
        let mut id_to_char = vec![String::new(); max_id + 1];

        for (char_str, id) in json_vocab {
            if id < id_to_char.len() {
                id_to_char[id] = char_str;
            }
        }
        Some(id_to_char)
    } else {
        None
    };

    let vocab: Vec<String> = if let Some(jv) = json_vocab {
        jv
    } else {
        // Plain text format: one token per line
        content
            .lines()
            .map(|line| line.trim().to_string())
            .collect()
    };

    // Build text from token IDs
    let mut text = String::new();
    for &id in token_ids {
        if id < vocab.len() {
            let token = &vocab[id];
            // Handle special Wav2Vec2 tokens
            if token == "|" {
                text.push(' '); // Word boundary
            } else if !token.starts_with('<') && !token.ends_with('>') {
                // Regular character token
                text.push_str(token);
            }
        }
    }

    // Clean up extra spaces
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Decode BPE tokens to text.
fn decode_bpe_tokens(token_ids: &[usize], vocab_path: &str) -> ExecutorResult<String> {
    use base64::{engine::general_purpose, Engine as _};

    // Load vocabulary
    let content = std::fs::read_to_string(vocab_path)
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to read vocab file: {}", e)))?;

    let tokens: Vec<String> = content
        .lines()
        .map(|line| line.trim().to_string())
        .collect();

    // Decode tokens
    let mut decoded_bytes = Vec::new();

    for &id in token_ids {
        if id < tokens.len() {
            let token_line = &tokens[id];

            // Skip special tokens
            if token_line.starts_with("<|") && token_line.ends_with("|>") {
                continue;
            }

            // Token format: "BASE64 ID"
            let base64_part = if let Some(space_idx) = token_line.find(' ') {
                &token_line[..space_idx]
            } else {
                token_line
            };

            // Decode base64 to bytes
            if let Ok(bytes) = general_purpose::STANDARD.decode(base64_part) {
                decoded_bytes.extend_from_slice(&bytes);
            }
        }
    }

    Ok(String::from_utf8_lossy(&decoded_bytes).to_string())
}

/// Decode Whisper tokens using HuggingFace tokenizer.json.
fn decode_whisper_tokens(token_ids: &[usize], tokenizer_path: &str) -> ExecutorResult<String> {
    use tokenizers::Tokenizer;

    // Load the HuggingFace tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to load tokenizer: {}", e)))?;

    // Convert token IDs to u32 (tokenizers crate uses u32)
    let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();

    // Filter out special tokens:
    // - Whisper special tokens are in range 50257-50364
    // - Keep only normal text tokens
    let filtered_ids: Vec<u32> = ids.into_iter().filter(|&id| id < 50257).collect();

    // Decode using the tokenizer
    let text = tokenizer
        .decode(&filtered_ids, true) // skip_special_tokens=true
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to decode tokens: {}", e)))?;

    // Clean up whitespace
    Ok(text.trim().to_string())
}
