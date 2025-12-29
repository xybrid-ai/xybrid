//! BERT-style execution mode.
//!
//! This module handles execution of BERT-style models that require
//! integer token inputs (input_ids, attention_mask, token_type_ids).

use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{Array2, ArrayD};
use ort::value::Value;
use std::collections::HashMap;

use super::super::types::ExecutorResult;

/// Execute BERT-style inference with input_ids, attention_mask, and token_type_ids.
///
/// This is for models like all-MiniLM-L6-v2, DistilBERT, etc. that expect
/// integer token inputs rather than float tensors.
///
/// # Arguments
/// - `session`: The ONNX session to run
/// - `ids`: Token IDs from tokenization
/// - `attention_mask`: Attention mask (1 for real tokens, 0 for padding)
/// - `token_type_ids`: Token type IDs (for sentence pairs)
///
/// # Returns
/// A HashMap of output tensors from the model
pub fn execute_bert_inference(
    session: &ONNXSession,
    ids: &[usize],
    attention_mask: &[usize],
    token_type_ids: &[usize],
) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
    let input_names = session.input_names();

    let batch_size = 1;
    let seq_len = ids.len();

    // Convert usize to i64 for ONNX
    let ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
    let type_ids_i64: Vec<i64> = token_type_ids.iter().map(|&x| x as i64).collect();

    let mut value_inputs: HashMap<String, Value> = HashMap::new();

    for input_name in input_names.iter() {
        let arr = if input_name == "input_ids" || input_name.contains("input_ids") {
            Array2::<i64>::from_shape_vec((batch_size, seq_len), ids_i64.clone())
        } else if input_name == "attention_mask" || input_name.contains("attention_mask") {
            Array2::<i64>::from_shape_vec((batch_size, seq_len), mask_i64.clone())
        } else if input_name == "token_type_ids" || input_name.contains("token_type_ids") {
            Array2::<i64>::from_shape_vec((batch_size, seq_len), type_ids_i64.clone())
        } else {
            return Err(AdapterError::InvalidInput(format!(
                "Unknown BERT input: {}",
                input_name
            )));
        };

        let arr = arr.map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to create {} array: {}", input_name, e))
        })?;

        let val: Value = Value::from_array(arr)
            .map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to create {} value: {}", input_name, e))
            })?
            .into();
        value_inputs.insert(input_name.clone(), val);
    }

    // Verify we mapped all inputs
    if value_inputs.len() != input_names.len() {
        return Err(AdapterError::InvalidInput(format!(
            "BERT model input mismatch. Expected {} inputs ({:?}), mapped {}",
            input_names.len(),
            input_names,
            value_inputs.len()
        )));
    }

    // Run inference
    session.run_with_values(value_inputs)
}
