//! Autoregressive execution mode.
//!
//! This module handles token-by-token generation with KV cache management
//! for decoder-only language models.

use super::super::types::ExecutorResult;
use crate::execution::template::PipelineStage;
use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{ArrayD, IxDyn};
use ort::value::Value;
use std::collections::HashMap;

/// Execute an autoregressive stage (token generation loop).
///
/// # Arguments
/// - `stage`: The pipeline stage configuration
/// - `stage_outputs`: Previous stage outputs (encoder outputs)
/// - `config`: Pipeline configuration values
/// - `max_tokens`: Maximum number of tokens to generate
/// - `start_token_id`: Token ID to start generation with
/// - `end_token_id`: Token ID that signals end of generation
/// - `repetition_penalty`: Penalty for repeating tokens (0.0 = disabled)
/// - `session`: The ONNX session to use
pub fn execute_autoregressive_stage(
    _stage: &PipelineStage,
    stage_outputs: &HashMap<String, HashMap<String, ArrayD<f32>>>,
    config: &HashMap<String, serde_json::Value>,
    max_tokens: usize,
    start_token_id: i64,
    end_token_id: i64,
    repetition_penalty: f32,
    session: &ONNXSession,
) -> ExecutorResult<Vec<usize>> {
    // Extract KV cache shape from config
    let kv_cache_shape = if let Some(shape_value) = config.get("kv_cache_shape") {
        shape_value
            .as_array()
            .ok_or_else(|| {
                AdapterError::InvalidInput("kv_cache_shape must be an array".to_string())
            })?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| {
                        AdapterError::InvalidInput(
                            "kv_cache_shape values must be numbers".to_string(),
                        )
                    })
                    .map(|n| n as usize)
            })
            .collect::<Result<Vec<usize>, _>>()?
    } else {
        return Err(AdapterError::InvalidInput(
            "Autoregressive stage requires kv_cache_shape in config".to_string(),
        ));
    };

    // Initialize KV caches with zeros
    let kv_cache_size: usize = kv_cache_shape.iter().product();
    let kv_cache_data = vec![0.0f32; kv_cache_size];

    let mut kv_cache_k =
        ArrayD::<f32>::from_shape_vec(IxDyn(&kv_cache_shape), kv_cache_data.clone()).map_err(
            |e| AdapterError::InvalidInput(format!("Failed to create KV cache K: {:?}", e)),
        )?;
    let mut kv_cache_v = ArrayD::<f32>::from_shape_vec(IxDyn(&kv_cache_shape), kv_cache_data)
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to create KV cache V: {:?}", e)))?;

    // Get encoder outputs (cross-attention keys/values)
    let encoder_outputs = stage_outputs
        .values()
        .next()
        .ok_or_else(|| AdapterError::InvalidInput("No encoder outputs found".to_string()))?;

    // Extract cross-attention keys/values
    let (cross_k, cross_v) = extract_encoder_cross_attention(encoder_outputs)?;
    let cross_k = cross_k.clone();
    let cross_v = cross_v.clone();

    // Autoregressive loop
    let mut token_ids = vec![start_token_id as usize];
    let mut offset = 0i64;

    for _ in 0..max_tokens {
        // Create tokens tensor [batch=1, seq_len=1]
        let current_token_id = *token_ids.last().unwrap() as i64;
        let tokens_shape = vec![1, 1];
        let tokens_data = vec![current_token_id];
        let tokens_i64 =
            ArrayD::<i64>::from_shape_vec(IxDyn(&tokens_shape), tokens_data).map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to create tokens tensor: {:?}", e))
            })?;
        let tokens_value: Value = Value::from_array(tokens_i64)
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert tokens: {:?}", e)))?
            .into();

        // Convert caches to Values
        let kv_cache_k_value: Value = Value::from_array(kv_cache_k.clone())
            .map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to convert KV cache K: {:?}", e))
            })?
            .into();
        let kv_cache_v_value: Value = Value::from_array(kv_cache_v.clone())
            .map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to convert KV cache V: {:?}", e))
            })?
            .into();

        let cross_k_value: Value = Value::from_array(cross_k.clone())
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert cross_k: {:?}", e)))?
            .into();
        let cross_v_value: Value = Value::from_array(cross_v.clone())
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert cross_v: {:?}", e)))?
            .into();

        // Offset tensor
        let offset_shape = vec![1];
        let offset_data = vec![offset];
        let offset_i64 =
            ArrayD::<i64>::from_shape_vec(IxDyn(&offset_shape), offset_data).map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to create offset tensor: {:?}", e))
            })?;
        let offset_value: Value = Value::from_array(offset_i64)
            .map_err(|e| AdapterError::InvalidInput(format!("Failed to convert offset: {:?}", e)))?
            .into();

        // Query ONNX session for actual input/output names
        let actual_input_names = session.input_names();
        let actual_output_names = session.output_names();

        if actual_input_names.len() < 6 {
            return Err(AdapterError::InvalidInput(format!(
                "Decoder model expected 6 inputs, found {}",
                actual_input_names.len()
            )));
        }

        if actual_output_names.len() < 3 {
            return Err(AdapterError::InvalidInput(format!(
                "Decoder model expected 3 outputs, found {}",
                actual_output_names.len()
            )));
        }

        // Build decoder inputs using actual ONNX input names (in order)
        // Order: [tokens, in_kv_k, in_kv_v, cross_k, cross_v, offset]
        let mut decoder_inputs = HashMap::new();
        decoder_inputs.insert(actual_input_names[0].clone(), tokens_value);
        decoder_inputs.insert(actual_input_names[1].clone(), kv_cache_k_value);
        decoder_inputs.insert(actual_input_names[2].clone(), kv_cache_v_value);
        decoder_inputs.insert(actual_input_names[3].clone(), cross_k_value);
        decoder_inputs.insert(actual_input_names[4].clone(), cross_v_value);
        decoder_inputs.insert(actual_input_names[5].clone(), offset_value);

        // Run decoder
        let decoder_outputs = session.run_with_values(decoder_inputs).map_err(|e| {
            AdapterError::InvalidInput(format!("Decoder inference failed: {:?}", e))
        })?;

        // Extract logits and updated KV caches using actual output names
        // Order: [logits, out_kv_k, out_kv_v]
        let logits = decoder_outputs
            .get(&actual_output_names[0])
            .ok_or_else(|| AdapterError::InvalidInput("Missing logits output".to_string()))?
            .clone();

        if let Some(updated_k) = decoder_outputs.get(&actual_output_names[1]) {
            kv_cache_k = updated_k.clone();
        }
        if let Some(updated_v) = decoder_outputs.get(&actual_output_names[2]) {
            kv_cache_v = updated_v.clone();
        }

        // Apply repetition penalty if enabled
        let mut logits = logits;
        if repetition_penalty > 0.0 && token_ids.len() > 1 {
            let recent_tokens: std::collections::HashSet<usize> =
                token_ids.iter().rev().take(10).copied().collect();

            if let Some(logits_slice) = logits.as_slice_mut() {
                for token_id in &recent_tokens {
                    if *token_id < logits_slice.len() {
                        logits_slice[*token_id] *= repetition_penalty;
                    }
                }
            }
        }

        // Get next token (argmax)
        let next_token_id = argmax_token(&logits)?;

        // Check for end token
        if next_token_id == end_token_id as usize {
            break;
        }

        // Check for repetition (stop if same token appears 5+ times in a row)
        if token_ids.len() >= 5 {
            let last_five: Vec<usize> = token_ids.iter().rev().take(5).copied().collect();
            if last_five.iter().all(|&id| id == next_token_id) {
                break;
            }
        }

        token_ids.push(next_token_id);
        offset += 1;
    }

    Ok(token_ids)
}

/// Extract encoder cross-attention keys/values from encoder outputs.
fn extract_encoder_cross_attention(
    encoder_outputs: &HashMap<String, ArrayD<f32>>,
) -> ExecutorResult<(&ArrayD<f32>, &ArrayD<f32>)> {
    // Try named lookup first (preferred if metadata is accurate)
    if let (Some(k), Some(v)) = (
        encoder_outputs.get("n_layer_cross_k"),
        encoder_outputs.get("n_layer_cross_v"),
    ) {
        return Ok((k, v));
    }

    // Fallback: use first two outputs by position
    if encoder_outputs.len() < 2 {
        return Err(AdapterError::InvalidInput(format!(
            "Encoder must produce at least 2 outputs (cross_k, cross_v), found {}",
            encoder_outputs.len()
        )));
    }

    let mut values = encoder_outputs.values();
    let k = values.next().unwrap();
    let v = values.next().unwrap();

    Ok((k, v))
}

/// Apply argmax to logits to get token ID.
fn argmax_token(logits: &ArrayD<f32>) -> ExecutorResult<usize> {
    let shape = logits.shape();
    let data = logits
        .as_slice()
        .ok_or_else(|| AdapterError::InvalidInput("Logits tensor is not contiguous".to_string()))?;

    // Handle 3D logits [batch, seq_len, vocab_size]
    if shape.len() == 3 {
        let vocab_size = shape[2];
        let start_idx = 0;
        let end_idx = start_idx + vocab_size;

        let slice = &data[start_idx..end_idx];
        let max_idx = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(max_idx)
    } else {
        Err(AdapterError::InvalidInput(format!(
            "Unexpected logits shape: {:?}",
            shape
        )))
    }
}
