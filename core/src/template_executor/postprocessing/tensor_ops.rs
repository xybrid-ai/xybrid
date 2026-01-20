//! Tensor operation postprocessing.
//!
//! This module provides:
//! - `argmax_step`: Get class ID with highest probability
//! - `softmax_step`: Apply softmax normalization
//! - `topk_step`: Get top-K predictions
//! - `threshold_step`: Apply threshold to probabilities
//! - `meanpool_step`: Mean pooling over sequence dimension

use super::super::types::{ExecutorResult, RawOutputs};
use crate::runtime_adapter::AdapterError;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Apply argmax to get class ID with highest probability.
///
/// # Arguments
/// - `data`: Input data (TensorMap)
/// - `dim`: Dimension to apply argmax (ignored, uses last dimension)
pub fn argmax_step(data: RawOutputs, _dim: Option<usize>) -> ExecutorResult<RawOutputs> {
    let tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Argmax requires tensor map".to_string(),
            ))
        }
    };

    // Get the first output tensor
    let tensor = tensor_map
        .values()
        .next()
        .ok_or_else(|| AdapterError::InvalidInput("No outputs to apply argmax".to_string()))?;

    let class_id = argmax_token(tensor)?;

    Ok(RawOutputs::ClassId(class_id))
}

/// Apply softmax normalization to tensor outputs.
///
/// # Arguments
/// - `data`: Input data (TensorMap)
/// - `dim`: Dimension to apply softmax (default: last dimension)
pub fn softmax_step(data: RawOutputs, dim: Option<usize>) -> ExecutorResult<RawOutputs> {
    let mut tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Softmax requires tensor map".to_string(),
            ))
        }
    };

    // Apply softmax to each tensor in the map
    for (_name, tensor) in tensor_map.iter_mut() {
        apply_softmax(tensor, dim)?;
    }

    Ok(RawOutputs::TensorMap(tensor_map))
}

/// Get top-K predictions with scores.
///
/// # Arguments
/// - `data`: Input data (TensorMap)
/// - `k`: Number of top predictions to return
/// - `dim`: Dimension to apply topk (default: last dimension)
pub fn topk_step(data: RawOutputs, k: usize, dim: Option<usize>) -> ExecutorResult<RawOutputs> {
    let tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "TopK requires tensor map".to_string(),
            ))
        }
    };

    // Get the first output tensor
    let tensor = tensor_map
        .values()
        .next()
        .ok_or_else(|| AdapterError::InvalidInput("No outputs for TopK".to_string()))?;

    // Apply top-k
    let top_k_results = top_k_predictions(tensor, k, dim)?;

    // Return as tensor map with flattened [index1, score1, index2, score2, ...]
    let mut flattened = Vec::with_capacity(k * 2);
    for (idx, score) in top_k_results {
        flattened.push(idx as f32);
        flattened.push(score);
    }

    // Create a 1D tensor from the flattened results
    let topk_tensor = ArrayD::from_shape_vec(IxDyn(&[k * 2]), flattened).map_err(|e| {
        AdapterError::InvalidInput(format!("Failed to create TopK tensor: {:?}", e))
    })?;

    let mut result_map = HashMap::new();
    result_map.insert("topk".to_string(), topk_tensor);

    Ok(RawOutputs::TensorMap(result_map))
}

/// Apply threshold to convert probabilities to binary predictions.
///
/// # Arguments
/// - `data`: Input data (TensorMap)
/// - `threshold`: Probability threshold
/// - `return_indices`: If true, return indices where value > threshold; otherwise return binary mask
pub fn threshold_step(
    data: RawOutputs,
    threshold: f32,
    return_indices: bool,
) -> ExecutorResult<RawOutputs> {
    let tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Threshold requires tensor map".to_string(),
            ))
        }
    };

    // Get the first output tensor
    let tensor = tensor_map
        .values()
        .next()
        .ok_or_else(|| AdapterError::InvalidInput("No outputs for Threshold".to_string()))?;

    let values = tensor.as_slice().ok_or_else(|| {
        AdapterError::InvalidInput("Tensor is not contiguous for Threshold".to_string())
    })?;

    if return_indices {
        // Return indices where value > threshold
        let indices: Vec<f32> = values
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| {
                if val > threshold {
                    Some(idx as f32)
                } else {
                    None
                }
            })
            .collect();

        let result_tensor =
            ArrayD::from_shape_vec(IxDyn(&[indices.len()]), indices).map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to create threshold tensor: {:?}", e))
            })?;

        let mut result_map = HashMap::new();
        result_map.insert("threshold_indices".to_string(), result_tensor);
        Ok(RawOutputs::TensorMap(result_map))
    } else {
        // Return binary mask (0 or 1)
        let binary: Vec<f32> = values
            .iter()
            .map(|&val| if val > threshold { 1.0 } else { 0.0 })
            .collect();

        let result_tensor = ArrayD::from_shape_vec(IxDyn(tensor.shape()), binary).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to create threshold mask: {:?}", e))
        })?;

        let mut result_map = HashMap::new();
        result_map.insert("threshold_mask".to_string(), result_tensor);
        Ok(RawOutputs::TensorMap(result_map))
    }
}

/// Apply mean pooling over token embeddings.
///
/// # Arguments
/// - `data`: Input data (TensorMap with 3D tensor [batch, seq_len, hidden_size])
/// - `dim`: Dimension to pool over (must be 1 for sequence dimension)
pub fn meanpool_step(data: RawOutputs, dim: usize) -> ExecutorResult<RawOutputs> {
    let tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "MeanPool requires tensor map".to_string(),
            ))
        }
    };

    // Get the first output tensor (usually "last_hidden_state" or similar)
    let tensor = tensor_map
        .values()
        .next()
        .ok_or_else(|| AdapterError::InvalidInput("No outputs for MeanPool".to_string()))?;

    let shape = tensor.shape();

    // Expected shape: [batch, sequence_length, hidden_size]
    if shape.len() != 3 {
        return Err(AdapterError::InvalidInput(format!(
            "MeanPool expects 3D tensor [batch, seq_len, hidden_size], got {:?}",
            shape
        )));
    }

    let batch_size = shape[0];
    let seq_len = shape[1];
    let hidden_size = shape[2];

    // Pool over the sequence dimension (dim=1 by default)
    if dim != 1 {
        return Err(AdapterError::InvalidInput(format!(
            "MeanPool only supports pooling over dim=1 (sequence), got dim={}",
            dim
        )));
    }

    // Create output tensor [batch, hidden_size]
    let mut pooled = ArrayD::<f32>::zeros(IxDyn(&[batch_size, hidden_size]));

    // Compute mean over sequence length for each batch and hidden dimension
    for b in 0..batch_size {
        for h in 0..hidden_size {
            let mut sum = 0.0;
            for s in 0..seq_len {
                sum += tensor[IxDyn(&[b, s, h])];
            }
            pooled[IxDyn(&[b, h])] = sum / (seq_len as f32);
        }
    }

    // Return pooled embedding
    let mut result_map = HashMap::new();
    result_map.insert("sentence_embedding".to_string(), pooled);

    Ok(RawOutputs::TensorMap(result_map))
}

// ============================================================================
// Helper functions
// ============================================================================

/// Apply argmax to logits to get token ID.
pub fn argmax_token(logits: &ArrayD<f32>) -> ExecutorResult<usize> {
    let shape = logits.shape();
    let data = logits
        .as_slice()
        .ok_or_else(|| AdapterError::InvalidInput("Logits tensor is not contiguous".to_string()))?;

    // Handle 3D logits [batch, seq_len, vocab_size]
    if shape.len() == 3 {
        let vocab_size = shape[2];
        let start_idx = 0; // First batch, first position
        let end_idx = start_idx + vocab_size;

        let slice = &data[start_idx..end_idx];
        let max_idx = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(max_idx)
    } else if shape.len() == 2 {
        // 2D logits [batch, vocab_size]
        let vocab_size = shape[1];
        let slice = &data[0..vocab_size];
        let max_idx = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(max_idx)
    } else if shape.len() == 1 {
        // 1D logits [vocab_size]
        let max_idx = data
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

/// Apply softmax to a tensor along a dimension.
fn apply_softmax(tensor: &mut ArrayD<f32>, dim: Option<usize>) -> ExecutorResult<()> {
    let shape = tensor.shape().to_vec(); // Clone shape to avoid borrow conflicts

    // Default to last dimension if not specified
    let dim = dim.unwrap_or(shape.len() - 1);

    if dim >= shape.len() {
        return Err(AdapterError::InvalidInput(format!(
            "Softmax dimension {} out of bounds for tensor with {} dimensions",
            dim,
            shape.len()
        )));
    }

    // For simplicity, only handle the common case of 2D tensors (batch, classes)
    // or 1D tensors (classes)
    if let Some(slice) = tensor.as_slice_mut() {
        if shape.len() == 1 {
            // 1D tensor: apply softmax directly
            softmax_1d(slice);
        } else if shape.len() == 2 && dim == 1 {
            // 2D tensor: apply softmax along last dimension
            let batch_size = shape[0];
            let class_size = shape[1];

            for batch in 0..batch_size {
                let start = batch * class_size;
                let end = start + class_size;
                softmax_1d(&mut slice[start..end]);
            }
        } else {
            return Err(AdapterError::InvalidInput(format!(
                "Softmax only supports 1D or 2D tensors, got shape {:?}",
                shape
            )));
        }
    } else {
        return Err(AdapterError::InvalidInput(
            "Tensor is not contiguous, cannot apply softmax".to_string(),
        ));
    }

    Ok(())
}

/// Apply softmax to a 1D slice.
fn softmax_1d(slice: &mut [f32]) {
    // Find max for numerical stability
    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for val in slice.iter_mut() {
        *val = (*val - max).exp();
        sum += *val;
    }

    // Normalize
    for val in slice.iter_mut() {
        *val /= sum;
    }
}

/// Get top-K predictions from a tensor.
/// Returns Vec of (class_index, score) tuples.
fn top_k_predictions(
    tensor: &ArrayD<f32>,
    k: usize,
    dim: Option<usize>,
) -> ExecutorResult<Vec<(usize, f32)>> {
    let shape = tensor.shape();

    // Default to last dimension
    let _dim = dim.unwrap_or(shape.len() - 1);

    // Get values as slice
    let values = tensor.as_slice().ok_or_else(|| {
        AdapterError::InvalidInput("Tensor is not contiguous for TopK".to_string())
    })?;

    // For simplicity, handle the common case: 1D (classes) or 2D (batch=1, classes)
    let class_scores: &[f32] = if shape.len() == 1 {
        values
    } else if shape.len() == 2 && shape[0] == 1 {
        // Batch size 1, get the first batch
        &values[0..shape[1]]
    } else {
        return Err(AdapterError::InvalidInput(format!(
            "TopK only supports 1D or 2D (batch=1) tensors, got shape {:?}",
            shape
        )));
    };

    // Create (index, score) pairs and sort by score descending
    let mut indexed_scores: Vec<(usize, f32)> = class_scores
        .iter()
        .enumerate()
        .map(|(idx, &score)| (idx, score))
        .collect();

    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top K
    let top_k: Vec<(usize, f32)> = indexed_scores.into_iter().take(k).collect();

    Ok(top_k)
}
