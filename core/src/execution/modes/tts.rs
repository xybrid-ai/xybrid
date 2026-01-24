//! TTS execution mode.
//!
//! This module handles execution of TTS models that require phoneme IDs,
//! voice embeddings, and speed parameters.

use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{Array1, Array2, ArrayD};
use ort::value::Value;
use std::collections::HashMap;

use super::super::types::ExecutorResult;

/// Execute TTS inference with phoneme IDs, voice embedding, and speed.
///
/// This is for models like KittenTTS, Kokoro, etc. that expect:
/// - Token/phoneme IDs as int64 [1, seq_len]
/// - Voice/style embedding as float32 [1, 256]
/// - Speed multiplier as float32 [1]
///
/// # Arguments
/// - `session`: The ONNX session to run
/// - `phoneme_ids`: Phoneme token IDs from phonemization
/// - `voice_embedding`: Voice style embedding (256-dim)
///
/// # Returns
/// A HashMap of output tensors (typically contains audio waveform)
pub fn execute_tts_inference(
    session: &ONNXSession,
    phoneme_ids: &[i64],
    voice_embedding: Vec<f32>,
) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
    // Get model input names
    let input_names = session.input_names();

    let batch_size = 1;
    let seq_len = phoneme_ids.len();
    let embedding_len = voice_embedding.len();

    // Build inputs
    let mut value_inputs: HashMap<String, Value> = HashMap::new();

    for input_name in input_names.iter() {
        // Token/phoneme IDs input - KittenTTS uses "input_ids", Kokoro uses "tokens"
        if input_name.contains("input_ids")
            || input_name == "input_ids"
            || input_name.contains("tokens")
            || input_name == "tokens"
        {
            let arr = Array2::<i64>::from_shape_vec((batch_size, seq_len), phoneme_ids.to_vec())
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create input_ids array: {}", e))
                })?;
            let val: Value = Value::from_array(arr)
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create input_ids value: {}", e))
                })?
                .into();
            value_inputs.insert(input_name.clone(), val);
        } else if input_name.contains("style") || input_name == "style" {
            let arr = Array2::<f32>::from_shape_vec((1, embedding_len), voice_embedding.clone())
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create style array: {}", e))
                })?;
            let val: Value = Value::from_array(arr)
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create style value: {}", e))
                })?
                .into();
            value_inputs.insert(input_name.clone(), val);
        } else if input_name.contains("speed") || input_name == "speed" {
            let arr = Array1::<f32>::from_vec(vec![1.0]);
            let val: Value = Value::from_array(arr)
                .map_err(|e| {
                    AdapterError::InvalidInput(format!("Failed to create speed value: {}", e))
                })?
                .into();
            value_inputs.insert(input_name.clone(), val);
        }
    }

    // Verify we mapped all inputs
    if value_inputs.len() != input_names.len() {
        return Err(AdapterError::InvalidInput(format!(
            "TTS model input mismatch. Expected {} inputs ({:?}), mapped {}",
            input_names.len(),
            input_names,
            value_inputs.len()
        )));
    }

    // Run inference
    session.run_with_values(value_inputs)
}
