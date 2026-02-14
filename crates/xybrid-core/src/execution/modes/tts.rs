//! TTS execution mode.
//!
//! This module handles execution of TTS models that require phoneme IDs,
//! voice embeddings, and speed parameters.
//!
//! Input mapping is based on ONNX metadata (dtype + shape), not input names,
//! so any TTS model with the standard input signature works without code changes.

use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{Array1, Array2, ArrayD};
use ort::tensor::TensorElementType;
use ort::value::Value;
use std::collections::HashMap;

use super::super::types::ExecutorResult;

/// Execute TTS inference with phoneme IDs, voice embedding, and speed.
///
/// Inputs are mapped by dtype and shape pattern, not by name:
/// - int64 input with shape [1, N] (dynamic) → token/phoneme IDs
/// - f32 input with shape [1, 256] → voice/style embedding
/// - f32 input with shape [1] → speed multiplier
///
/// This makes the function model-agnostic: KittenTTS (input_ids, style, speed)
/// and Kokoro (tokens, style, speed) both work without name-specific code.
pub fn execute_tts_inference(
    session: &ONNXSession,
    phoneme_ids: &[i64],
    voice_embedding: Vec<f32>,
) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
    let input_names = session.input_names();
    let input_shapes = session.input_shapes();
    let input_dtypes = session.input_dtypes();

    let batch_size = 1;
    let seq_len = phoneme_ids.len();
    let embedding_len = voice_embedding.len();

    let mut value_inputs: HashMap<String, Value> = HashMap::new();

    for (i, input_name) in input_names.iter().enumerate() {
        let dtype = input_dtypes.get(i).and_then(|d| *d);
        let shape = input_shapes.get(i).map(|s| s.as_slice()).unwrap_or(&[]);

        match classify_tts_input(dtype, shape) {
            TtsInputKind::Tokens => {
                let arr =
                    Array2::<i64>::from_shape_vec((batch_size, seq_len), phoneme_ids.to_vec())
                        .map_err(|e| {
                            AdapterError::InvalidInput(format!(
                                "Failed to create token array for '{}': {}",
                                input_name, e
                            ))
                        })?;
                let val: Value = Value::from_array(arr)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create token value for '{}': {}",
                            input_name, e
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::VoiceEmbedding => {
                let arr =
                    Array2::<f32>::from_shape_vec((1, embedding_len), voice_embedding.clone())
                        .map_err(|e| {
                            AdapterError::InvalidInput(format!(
                                "Failed to create voice embedding array for '{}': {}",
                                input_name, e
                            ))
                        })?;
                let val: Value = Value::from_array(arr)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create voice embedding value for '{}': {}",
                            input_name, e
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::Speed => {
                let arr = Array1::<f32>::from_vec(vec![1.0]);
                let val: Value = Value::from_array(arr)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create speed value for '{}': {}",
                            input_name, e
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::Unknown => {
                // Not classified — will trigger the mismatch error below
            }
        }
    }

    // Verify we mapped all inputs
    if value_inputs.len() != input_names.len() {
        let found: Vec<String> = input_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let dtype = input_dtypes
                    .get(i)
                    .and_then(|d| *d)
                    .map_or("unknown".to_string(), |d| format!("{:?}", d));
                let shape = input_shapes
                    .get(i)
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_default();
                format!("'{}' (dtype={}, shape={})", name, dtype, shape)
            })
            .collect();

        return Err(AdapterError::InvalidInput(format!(
            "TTS model has unexpected inputs. Expected patterns: \
             int64 [1, N] (tokens), f32 [1, 256] (voice embedding), f32 [1] (speed). \
             Found: [{}]",
            found.join(", ")
        )));
    }

    session.run_with_values(value_inputs)
}

/// Classification of a TTS model input based on dtype and shape.
enum TtsInputKind {
    /// int64 [1, N] — phoneme/token IDs
    Tokens,
    /// f32 [1, 256] — voice/style embedding
    VoiceEmbedding,
    /// f32 [1] — speed multiplier
    Speed,
    /// Unrecognized input pattern
    Unknown,
}

/// Classify a TTS input by its element type and shape dimensions.
///
/// Rules:
/// - int64 with 2D shape [1, N] (N fixed or dynamic) → Tokens
/// - f32 with 2D shape [1, 256] → VoiceEmbedding
/// - f32 with 1D shape [1] (or dynamic [N]) → Speed
fn classify_tts_input(dtype: Option<TensorElementType>, shape: &[i64]) -> TtsInputKind {
    match dtype {
        Some(TensorElementType::Int64) => {
            // int64 with 2 dims where first is 1 (or dynamic) → token input
            if shape.len() == 2 && (shape[0] == 1 || shape[0] == -1) {
                return TtsInputKind::Tokens;
            }
        }
        Some(TensorElementType::Float32) => {
            if shape.len() == 2 && (shape[0] == 1 || shape[0] == -1) {
                // f32 [1, 256] → voice embedding
                // Distinguish from tokens: embedding dim is typically 256 (fixed)
                if shape[1] > 1 {
                    return TtsInputKind::VoiceEmbedding;
                }
            }
            if shape.len() == 1 {
                // f32 [1] or f32 [-1] → speed
                return TtsInputKind::Speed;
            }
        }
        _ => {}
    }
    TtsInputKind::Unknown
}
