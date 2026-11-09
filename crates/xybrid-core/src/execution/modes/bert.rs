//! BERT-style execution mode.
//!
//! This module handles execution of BERT-style models that require
//! integer token inputs (input_ids, attention_mask, token_type_ids).
//!
//! Multimodal embedding models exported for ONNX, such as EmbeddingGemma 2,
//! also declare `image_features`, `video_features` and `audio_features`
//! inputs. A text run has no such features, so each gets an empty
//! `[0, width]` float tensor.

use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{Array2, ArrayD};
use ort::tensor::TensorElementType;
use ort::value::Value;
use std::collections::HashMap;

use super::super::types::ExecutorResult;

/// Inputs through which a multimodal embedding model takes encoder features,
/// one row per placeholder token in the prompt. Text alone fills none of them.
const MODALITY_FEATURE_INPUTS: [&str; 3] = ["image_features", "video_features", "audio_features"];

/// What [`execute_bert_inference`] feeds one model input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BertInput {
    /// Token IDs from tokenization.
    InputIds,
    /// 1 for real tokens, 0 for padding.
    AttentionMask,
    /// Segment IDs for sentence pairs.
    TokenTypeIds,
    /// A modality-feature input, given zero rows of `width` floats.
    NoFeatures { width: usize },
}

/// Decides what to feed the model input `name`, given its declared element
/// type and shape (`-1` for a dynamic dimension).
///
/// # Errors
///
/// [`AdapterError::InvalidInput`] for an input this mode cannot fill: an
/// unknown name, or a modality-feature input that is not float32
/// `[rows, width]` with dynamic rows.
fn classify_input(
    name: &str,
    dtype: Option<TensorElementType>,
    shape: &[i64],
) -> Result<BertInput, AdapterError> {
    if name.contains("input_ids") {
        return Ok(BertInput::InputIds);
    }
    if name.contains("attention_mask") {
        return Ok(BertInput::AttentionMask);
    }
    if name.contains("token_type_ids") {
        return Ok(BertInput::TokenTypeIds);
    }
    if MODALITY_FEATURE_INPUTS.contains(&name) {
        let width = match (dtype, shape) {
            (Some(TensorElementType::Float32), &[rows, width]) if rows < 0 => {
                usize::try_from(width).ok().filter(|&width| width > 0)
            }
            _ => None,
        };
        return width
            .map(|width| BertInput::NoFeatures { width })
            .ok_or_else(|| {
                AdapterError::InvalidInput(format!(
                    "Input '{name}' must be float32 [rows, width] with dynamic rows to run \
                     text-only, but the model declares {dtype:?} {shape:?}"
                ))
            });
    }
    Err(AdapterError::InvalidInput(format!(
        "Unknown BERT input: {name}"
    )))
}

/// Builds the `[batch_size, seq_len]` int64 tensor for a token input.
fn token_tensor(
    name: &str,
    values: &[i64],
    batch_size: usize,
    seq_len: usize,
) -> ExecutorResult<Value> {
    let arr =
        Array2::<i64>::from_shape_vec((batch_size, seq_len), values.to_vec()).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to create {} array: {}", name, e))
        })?;
    Ok(Value::from_array(arr)
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to create {} value: {}", name, e)))?
        .into())
}

/// Execute BERT-style inference with input_ids, attention_mask, and token_type_ids.
///
/// This is for models like all-MiniLM-L6-v2, DistilBERT, etc. that expect
/// integer token inputs rather than float tensors. Modality-feature inputs
/// (`image_features`, `video_features`, `audio_features`) receive zero rows.
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
    let input_dtypes = session.input_dtypes();
    let input_shapes = session.input_shapes();

    let batch_size = 1;
    let seq_len = ids.len();

    // Convert usize to i64 for ONNX
    let ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
    let type_ids_i64: Vec<i64> = token_type_ids.iter().map(|&x| x as i64).collect();

    let mut value_inputs: HashMap<String, Value> = HashMap::new();

    for (index, input_name) in input_names.iter().enumerate() {
        let dtype = input_dtypes.get(index).copied().flatten();
        let shape = input_shapes.get(index).map_or(&[][..], Vec::as_slice);

        let val = match classify_input(input_name, dtype, shape)? {
            BertInput::InputIds => token_tensor(input_name, &ids_i64, batch_size, seq_len)?,
            BertInput::AttentionMask => token_tensor(input_name, &mask_i64, batch_size, seq_len)?,
            BertInput::TokenTypeIds => {
                token_tensor(input_name, &type_ids_i64, batch_size, seq_len)?
            }
            BertInput::NoFeatures { width } => Value::from_array(Array2::<f32>::zeros((0, width)))
                .map_err(|e| {
                    AdapterError::InvalidInput(format!(
                        "Failed to create {} value: {}",
                        input_name, e
                    ))
                })?
                .into(),
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    const DYNAMIC: i64 = -1;

    #[test]
    fn token_inputs_are_recognized_by_name() {
        let int64 = Some(TensorElementType::Int64);
        let shape = [DYNAMIC, DYNAMIC];

        assert_eq!(
            classify_input("input_ids", int64, &shape).unwrap(),
            BertInput::InputIds
        );
        assert_eq!(
            classify_input("attention_mask", int64, &shape).unwrap(),
            BertInput::AttentionMask
        );
        assert_eq!(
            classify_input("token_type_ids", int64, &shape).unwrap(),
            BertInput::TokenTypeIds
        );
    }

    #[test]
    fn modality_feature_inputs_get_zero_rows_of_their_width() {
        let float32 = Some(TensorElementType::Float32);

        for name in MODALITY_FEATURE_INPUTS {
            assert_eq!(
                classify_input(name, float32, &[DYNAMIC, 512]).unwrap(),
                BertInput::NoFeatures { width: 512 },
                "{name}"
            );
        }
    }

    #[test]
    fn modality_feature_input_with_fixed_rows_is_rejected() {
        let error = classify_input(
            "image_features",
            Some(TensorElementType::Float32),
            &[1, 512],
        )
        .unwrap_err();

        assert!(error.to_string().contains("image_features"), "{error}");
    }

    #[test]
    fn modality_feature_input_needs_float32_and_a_known_width() {
        let wrong_type = classify_input(
            "audio_features",
            Some(TensorElementType::Int64),
            &[DYNAMIC, 512],
        );
        let unknown_width = classify_input(
            "audio_features",
            Some(TensorElementType::Float32),
            &[DYNAMIC, DYNAMIC],
        );
        let wrong_rank = classify_input(
            "audio_features",
            Some(TensorElementType::Float32),
            &[DYNAMIC, DYNAMIC, 512],
        );

        assert!(wrong_type.is_err());
        assert!(unknown_width.is_err());
        assert!(wrong_rank.is_err());
    }

    #[test]
    fn unknown_input_is_still_rejected() {
        let error = classify_input(
            "pixel_values",
            Some(TensorElementType::Float32),
            &[DYNAMIC, 768],
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Unknown BERT input: pixel_values"),
            "{error}"
        );
    }
}
