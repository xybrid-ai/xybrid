//! Whisper decoder execution mode.
//!
//! This module handles Whisper-specific decoding with:
//! - Encoder hidden states for cross-attention
//! - Separate decoder and encoder KV caches
//! - Forced decoder IDs (language, task, no_timestamps)

use super::super::types::ExecutorResult;
use super::{parse_kv_cache_name, parse_present_name_full};
use crate::execution_template::PipelineStage;
use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{Array2, ArrayD, IxDyn};
use ort::value::Value;
use std::collections::HashMap;

/// Execute Whisper decoder stage (HuggingFace ONNX format).
///
/// This handles the onnx-community/whisper-* format where:
/// - Encoder outputs `last_hidden_state` [batch, 1500, hidden_size]
/// - Decoder takes `input_ids` + 16 past_key_values tensors
/// - Decoder outputs `logits` + 8 present tensors (decoder KV only)
#[allow(clippy::too_many_arguments)]
pub fn execute_whisper_decoder_stage(
    _stage: &PipelineStage,
    stage_outputs: &HashMap<String, HashMap<String, ArrayD<f32>>>,
    config: &HashMap<String, serde_json::Value>,
    max_tokens: usize,
    start_token_id: i64,
    end_token_id: i64,
    language_token_id: i64,
    task_token_id: i64,
    no_timestamps_token_id: i64,
    suppress_tokens: &[i64],
    repetition_penalty: f32,
    session: &ONNXSession,
) -> ExecutorResult<Vec<usize>> {
    // Get config values
    let num_layers = config
        .get("num_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;
    let num_heads = config
        .get("num_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(6) as usize;
    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(64) as usize;
    let encoder_seq_len = config
        .get("encoder_seq_len")
        .and_then(|v| v.as_u64())
        .unwrap_or(1500) as usize;

    // Get encoder hidden states from previous stage
    let encoder_outputs = stage_outputs
        .get("encoder")
        .ok_or_else(|| AdapterError::InvalidInput("No encoder outputs found".to_string()))?;

    let encoder_hidden_states = encoder_outputs
        .get("last_hidden_state")
        .or_else(|| encoder_outputs.values().next())
        .ok_or_else(|| AdapterError::InvalidInput("No encoder hidden states".to_string()))?;

    // Clone encoder hidden states shape for computing encoder KV cache
    let enc_shape = encoder_hidden_states.shape();
    let batch_size = enc_shape[0];

    // Get input/output names
    let input_names = session.input_names();

    // Initialize token sequence with forced decoder IDs
    // Whisper expects: <|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]
    let forced_tokens: Vec<i64> = vec![
        start_token_id,         // <|startoftranscript|> = 50258
        language_token_id,      // <|en|> = 50259
        task_token_id,          // <|transcribe|> = 50359
        no_timestamps_token_id, // <|notimestamps|> = 50363
    ];
    let num_forced = forced_tokens.len();

    // Initialize decoder KV cache (starts empty, grows each step)
    // Shape: [batch, num_heads, 0, head_dim] → grows to [batch, num_heads, seq_len, head_dim]
    let mut decoder_kv_cache: Vec<ArrayD<f32>> = Vec::new();
    for _ in 0..(num_layers * 2) {
        // key + value for each layer
        let kv = ArrayD::<f32>::zeros(IxDyn(&[batch_size, num_heads, 0, head_dim]));
        decoder_kv_cache.push(kv);
    }

    // Encoder KV cache is computed once and reused
    let mut encoder_kv_cache: Vec<ArrayD<f32>> = (0..(num_layers * 2))
        .map(|_| ArrayD::<f32>::zeros(IxDyn(&[batch_size, num_heads, encoder_seq_len, head_dim])))
        .collect();

    // Track generated tokens (starts with forced tokens)
    let mut generated_tokens: Vec<usize> = forced_tokens.iter().map(|&t| t as usize).collect();

    // Convert suppress_tokens to a HashSet for fast lookup
    let suppress_set: std::collections::HashSet<i64> = suppress_tokens.iter().copied().collect();

    // Autoregressive loop
    for step in 0..max_tokens {
        // Get the token to process
        let current_token = if step < num_forced {
            forced_tokens[step]
        } else {
            *generated_tokens.last().unwrap() as i64
        };

        // Create input_ids tensor [batch, 1]
        let input_ids =
            Array2::<i64>::from_shape_vec((batch_size, 1), vec![current_token; batch_size])
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create input_ids: {}", e))
                })?;

        let input_ids_value: Value = Value::from_array(input_ids)
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert input_ids: {}", e)))?
            .into();

        // Build inputs map
        let mut inputs: HashMap<String, Value> = HashMap::new();

        // Add input_ids
        if let Some(name) = input_names.iter().find(|n| n.contains("input_ids")) {
            inputs.insert(name.clone(), input_ids_value);
        } else if !input_names.is_empty() {
            inputs.insert(input_names[0].clone(), input_ids_value);
        }

        // Add encoder hidden states (required for cross-attention)
        let enc_hidden_value: Value = Value::from_array(encoder_hidden_states.clone())
            .map_err(|e| {
                AdapterError::InvalidInput(format!(
                    "Failed to convert encoder hidden states: {}",
                    e
                ))
            })?
            .into();

        // Find and set encoder KV cache inputs
        for name in input_names.iter() {
            if name.contains("past_key_values") {
                // Parse layer index and type from name
                if let Some(captures) = parse_kv_cache_name(name) {
                    let (layer, is_encoder, is_key) = captures;

                    if is_encoder {
                        // Encoder KV cache (fixed size, computed from cross-attention)
                        let kv_idx = layer * 2 + if is_key { 0 } else { 1 };
                        if kv_idx < encoder_kv_cache.len() {
                            let kv_value: Value =
                                Value::from_array(encoder_kv_cache[kv_idx].clone())
                                    .map_err(|e| {
                                        AdapterError::InvalidInput(format!(
                                            "Failed to convert encoder KV: {}",
                                            e
                                        ))
                                    })?
                                    .into();
                            inputs.insert(name.clone(), kv_value);
                        }
                    } else {
                        // Decoder KV cache (grows with each step)
                        let kv_idx = layer * 2 + if is_key { 0 } else { 1 };
                        if kv_idx < decoder_kv_cache.len() {
                            let kv_value: Value =
                                Value::from_array(decoder_kv_cache[kv_idx].clone())
                                    .map_err(|e| {
                                        AdapterError::InvalidInput(format!(
                                            "Failed to convert decoder KV: {}",
                                            e
                                        ))
                                    })?
                                    .into();
                            inputs.insert(name.clone(), kv_value);
                        }
                    }
                }
            }
        }

        // Check if we have encoder_hidden_states input
        if let Some(name) = input_names
            .iter()
            .find(|n| n.contains("encoder_hidden_states"))
        {
            inputs.insert(name.clone(), enc_hidden_value);
        }

        // Run decoder
        let outputs = session
            .run_with_values(inputs)
            .map_err(|e| AdapterError::InferenceFailed(format!("Whisper decoder failed: {}", e)))?;

        // Get logits from outputs
        let logits = outputs
            .get("logits")
            .or_else(|| outputs.values().next())
            .ok_or_else(|| AdapterError::InvalidInput("No logits output".to_string()))?;

        // Update KV cache from present outputs
        for (name, tensor) in &outputs {
            if name.starts_with("present.") {
                if let Some((layer, is_encoder, is_key)) = parse_present_name_full(name) {
                    let kv_idx = layer * 2 + if is_key { 0 } else { 1 };
                    if is_encoder {
                        // Encoder KV cache - update only on first step
                        if step == 0 && kv_idx < encoder_kv_cache.len() {
                            encoder_kv_cache[kv_idx] = tensor.clone();
                        }
                    } else {
                        // Decoder KV cache - grows each step
                        if kv_idx < decoder_kv_cache.len() {
                            decoder_kv_cache[kv_idx] = tensor.clone();
                        }
                    }
                }
            }
        }

        // Skip forced tokens during generation (don't process logits yet)
        if step < num_forced - 1 {
            continue;
        }

        // Apply token suppression and repetition penalty
        let mut logits_vec = logits
            .as_slice()
            .ok_or_else(|| AdapterError::InvalidInput("Logits not contiguous".to_string()))?
            .to_vec();

        // Suppress tokens
        for &token in &suppress_set {
            if (token as usize) < logits_vec.len() {
                logits_vec[token as usize] = f32::NEG_INFINITY;
            }
        }

        // Apply repetition penalty
        if repetition_penalty != 1.0 && generated_tokens.len() > 4 {
            let recent: std::collections::HashSet<usize> =
                generated_tokens.iter().rev().take(10).copied().collect();

            for token in &recent {
                if *token < logits_vec.len() {
                    let score = logits_vec[*token];
                    logits_vec[*token] = if score > 0.0 {
                        score / repetition_penalty
                    } else {
                        score * repetition_penalty
                    };
                }
            }
        }

        // Get next token (argmax)
        let next_token = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(end_token_id as usize);

        // Check for end token
        if next_token == end_token_id as usize {
            break;
        }

        // Check for repetition loop (same token 5+ times)
        if generated_tokens.len() >= 5 {
            let last_five: Vec<usize> = generated_tokens.iter().rev().take(5).copied().collect();
            if last_five.iter().all(|&id| id == next_token) {
                break;
            }
        }

        generated_tokens.push(next_token);
    }

    Ok(generated_tokens)
}
