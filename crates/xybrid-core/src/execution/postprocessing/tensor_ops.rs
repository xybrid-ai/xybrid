//! Tensor operation postprocessing.
//!
//! This module provides:
//! - `argmax_step`: Get class ID with highest probability
//! - `softmax_step`: Apply softmax normalization
//! - `topk_step`: Get top-K predictions
//! - `temperature_sample_step`: Sample a class from filtered logits
//! - `threshold_step`: Apply threshold to probabilities
//! - `meanpool_step`: Mean pooling over sequence dimension
//! - `denormalize_step`: Denormalize tensor values (inverse of Normalize preprocessing)

use super::super::types::{ExecutorResult, RawOutputs};
use crate::runtime_adapter::AdapterError;
use ndarray::{ArrayD, IxDyn};
use rand::Rng;
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
    for tensor in tensor_map.values_mut() {
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

/// Sample a class ID from temperature-scaled, optionally filtered logits.
///
/// `temperature = 0` and `top_k = 1` use exact argmax. Positive temperatures
/// use a production RNG for a weighted draw after top-k and top-p filters.
///
/// # Errors
/// Returns [`AdapterError::InvalidInput`] for invalid parameters or logits,
/// unsupported/empty tensors, and inputs other than a tensor map.
pub fn temperature_sample_step(
    data: RawOutputs,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> ExecutorResult<RawOutputs> {
    let mut rng = rand::rng();
    temperature_sample_step_with_rng(data, temperature, top_k, top_p, &mut rng)
}

fn temperature_sample_step_with_rng<R: Rng + ?Sized>(
    data: RawOutputs,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut R,
) -> ExecutorResult<RawOutputs> {
    validate_sampling_params(temperature, top_k, top_p)?;
    let tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "TemperatureSample requires tensor map".to_string(),
            ))
        }
    };
    let tensor = select_logits_output(&tensor_map)?;
    let logits = sampling_logits(tensor)?;

    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(AdapterError::InvalidInput(
            "TemperatureSample requires finite logits".to_string(),
        ));
    }
    if temperature == 0.0 || top_k == Some(1) {
        return Ok(RawOutputs::ClassId(argmax_token(tensor)?));
    }

    let candidates = filtered_sampling_weights(logits, temperature, top_k, top_p);
    Ok(RawOutputs::ClassId(sample_weighted_index(
        &candidates,
        rng,
    )?))
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

/// Denormalize tensor values (inverse of Normalize preprocessing).
///
/// Applies `output = (input * std) + mean` element-wise, cycling through
/// `mean`/`std` by flat index modulo their length. A length-1 slice broadcasts
/// the single value across all elements; a longer slice applies per-channel.
///
/// # Arguments
/// - `data`: Input data (TensorMap)
/// - `mean`: Per-channel mean values used during normalization
/// - `std`: Per-channel standard deviation values used during normalization
///
/// # Errors
/// Returns an error if `mean` and `std` have different lengths, if either is
/// empty, or if the input is not a TensorMap.
pub fn denormalize_step(data: RawOutputs, mean: &[f32], std: &[f32]) -> ExecutorResult<RawOutputs> {
    if mean.len() != std.len() {
        return Err(AdapterError::InvalidInput(format!(
            "Denormalize mean length ({}) must match std length ({})",
            mean.len(),
            std.len()
        )));
    }
    if mean.is_empty() {
        return Err(AdapterError::InvalidInput(
            "Denormalize requires at least one mean/std value".to_string(),
        ));
    }

    let mut tensor_map = match data {
        RawOutputs::TensorMap(map) => map,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Denormalize requires tensor map input".to_string(),
            ))
        }
    };

    for (name, tensor) in tensor_map.iter_mut() {
        let tensor_slice = tensor.as_slice_mut().ok_or_else(|| {
            AdapterError::InvalidInput(format!(
                "Denormalize requires a contiguous tensor (output \"{}\" is non-contiguous)",
                name
            ))
        })?;

        for (i, val) in tensor_slice.iter_mut().enumerate() {
            let channel = i % mean.len();
            *val = (*val * std[channel]) + mean[channel];
        }
    }

    Ok(RawOutputs::TensorMap(tensor_map))
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
        if shape.contains(&0) {
            return Err(AdapterError::InvalidInput(format!(
                "Unexpected empty logits shape: {:?}",
                shape
            )));
        }
        let vocab_size = shape[2];
        let start_idx = (shape[1] - 1) * vocab_size; // First batch, final position
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

fn select_logits_output(tensor_map: &HashMap<String, ArrayD<f32>>) -> ExecutorResult<&ArrayD<f32>> {
    if let Some(logits) = tensor_map.get("logits") {
        return Ok(logits);
    }

    match tensor_map.len() {
        0 => Err(AdapterError::InvalidInput(
            "No outputs for TemperatureSample".to_string(),
        )),
        1 => tensor_map.values().next().ok_or_else(|| {
            AdapterError::InvalidInput("No outputs for TemperatureSample".to_string())
        }),
        _ => Err(AdapterError::InvalidInput(
            "TemperatureSample requires a named 'logits' output when multiple outputs are present"
                .to_string(),
        )),
    }
}

fn validate_sampling_params(
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> ExecutorResult<()> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(AdapterError::InvalidInput(format!(
            "TemperatureSample temperature must be finite and non-negative, got {}",
            temperature
        )));
    }
    if top_k == Some(0) {
        return Err(AdapterError::InvalidInput(
            "TemperatureSample top_k must be greater than zero".to_string(),
        ));
    }
    if let Some(top_p) = top_p {
        if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
            return Err(AdapterError::InvalidInput(format!(
                "TemperatureSample top_p must be finite and in (0, 1], got {}",
                top_p
            )));
        }
    }
    Ok(())
}

fn sampling_logits(tensor: &ArrayD<f32>) -> ExecutorResult<&[f32]> {
    let shape = tensor.shape();
    if !(1..=3).contains(&shape.len()) || shape.contains(&0) {
        return Err(AdapterError::InvalidInput(format!(
            "TemperatureSample expects a non-empty 1D, 2D, or 3D logits tensor, got {:?}",
            shape
        )));
    }

    let data = tensor.as_slice().ok_or_else(|| {
        AdapterError::InvalidInput(
            "TemperatureSample requires a contiguous logits tensor".to_string(),
        )
    })?;
    let vocab_size = *shape.last().ok_or_else(|| {
        AdapterError::InvalidInput("TemperatureSample received an empty tensor shape".to_string())
    })?;
    let start_idx = if shape.len() == 3 {
        (shape[1] - 1) * vocab_size
    } else {
        0
    };
    Ok(&data[start_idx..start_idx + vocab_size])
}

fn filtered_sampling_weights(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> Vec<(usize, f64)> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut candidates: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| {
            let scaled = (f64::from(logit) - f64::from(max_logit)) / f64::from(temperature);
            (index, scaled.exp())
        })
        .collect();

    if top_k.is_some() || top_p.is_some_and(|top_p| top_p < 1.0) {
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    }
    if let Some(top_k) = top_k {
        candidates.truncate(top_k.min(candidates.len()));
    }
    if let Some(top_p) = top_p.filter(|top_p| *top_p < 1.0) {
        let total_weight: f64 = candidates.iter().map(|(_, weight)| weight).sum();
        let cutoff = total_weight * f64::from(top_p);
        let mut cumulative = 0.0;
        let keep = candidates
            .iter()
            .position(|(_, weight)| {
                cumulative += weight;
                cumulative >= cutoff
            })
            .map_or(candidates.len(), |position| position + 1)
            .max(1);
        candidates.truncate(keep);
    }
    candidates
}

fn sample_weighted_index<R: Rng + ?Sized>(
    candidates: &[(usize, f64)],
    rng: &mut R,
) -> ExecutorResult<usize> {
    let total_weight: f64 = candidates.iter().map(|(_, weight)| weight).sum();
    if candidates.is_empty() || !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(AdapterError::InvalidInput(
            "TemperatureSample could not construct a probability distribution".to_string(),
        ));
    }

    let draw = rng.random_range(0.0..total_weight);
    let mut cumulative = 0.0;
    for &(index, weight) in candidates {
        cumulative += weight;
        if draw < cumulative {
            return Ok(index);
        }
    }
    candidates.last().map(|(index, _)| *index).ok_or_else(|| {
        AdapterError::InvalidInput(
            "TemperatureSample could not select from an empty distribution".to_string(),
        )
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::preprocessing::tensor::normalize_step;
    use crate::execution::types::PreprocessedData;
    use ndarray::Array2;
    use rand::{rngs::StdRng, SeedableRng};

    fn logits_data(logits: &[f32]) -> RawOutputs {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[logits.len()]), logits.to_vec())
            .expect("valid logits shape");
        RawOutputs::TensorMap(HashMap::from([("logits".to_string(), tensor)]))
    }

    fn sampled_class(result: RawOutputs) -> usize {
        match result {
            RawOutputs::ClassId(class_id) => class_id,
            other => panic!("expected ClassId, got {other:?}"),
        }
    }

    fn sample_with_seed(
        data: RawOutputs,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> ExecutorResult<RawOutputs> {
        let mut rng = StdRng::seed_from_u64(42);
        temperature_sample_step_with_rng(data, temperature, top_k, top_p, &mut rng)
    }

    fn assert_invalid(result: ExecutorResult<RawOutputs>) {
        assert!(matches!(result, Err(AdapterError::InvalidInput(_))));
    }

    #[test]
    fn temperature_zero_returns_exact_argmax() {
        let result = sample_with_seed(logits_data(&[0.5, 3.0, 1.0]), 0.0, None, None).unwrap();

        assert_eq!(sampled_class(result), 1);
    }

    #[test]
    fn top_k_one_returns_exact_argmax() {
        let result =
            sample_with_seed(logits_data(&[0.5, 3.0, 1.0]), 100.0, Some(1), Some(0.1)).unwrap();

        assert_eq!(sampled_class(result), 1);
    }

    #[test]
    fn temperature_sample_argmax_uses_final_sequence_position() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 3]),
            vec![1_000.0, -1_000.0, -1_000.0, -1_000.0, 1_000.0, -1_000.0],
        )
        .expect("valid logits shape");
        let data = RawOutputs::TensorMap(HashMap::from([("logits".to_string(), tensor)]));

        let result = sample_with_seed(data, 0.0, None, None).unwrap();

        assert_eq!(sampled_class(result), 1);
    }

    #[test]
    fn temperature_sample_weighted_draw_uses_final_sequence_position() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 3]),
            vec![1_000.0, -1_000.0, -1_000.0, -1_000.0, 1_000.0, -1_000.0],
        )
        .expect("valid logits shape");
        let data = RawOutputs::TensorMap(HashMap::from([("logits".to_string(), tensor)]));

        let result = sample_with_seed(data, 1.0, None, None).unwrap();

        assert_eq!(sampled_class(result), 1);
    }

    #[test]
    fn temperature_sample_prefers_named_logits_output() {
        let logits =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 10.0]).expect("valid logits shape");
        let cache =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 0.0]).expect("valid cache shape");
        let data = RawOutputs::TensorMap(HashMap::from([
            ("logits".to_string(), logits),
            ("present.0.key".to_string(), cache),
        ]));

        let result = sample_with_seed(data, 0.0, None, None).unwrap();

        assert_eq!(sampled_class(result), 1);
    }

    #[test]
    fn temperature_sample_rejects_multiple_unnamed_outputs() {
        let first =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).expect("valid output shape");
        let second =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 0.0]).expect("valid output shape");
        let data = RawOutputs::TensorMap(HashMap::from([
            ("output_a".to_string(), first),
            ("output_b".to_string(), second),
        ]));

        let result = sample_with_seed(data, 0.0, None, None);

        assert_invalid(result);
    }

    #[test]
    fn temperature_sample_accepts_single_unnamed_output() {
        let tensor =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).expect("valid output shape");
        let data = RawOutputs::TensorMap(HashMap::from([("scores".to_string(), tensor)]));

        let result = sample_with_seed(data, 0.0, None, None).unwrap();

        assert_eq!(sampled_class(result), 1);
    }

    #[test]
    fn fixed_seed_repeats_the_same_sample_sequence() {
        let mut first_rng = StdRng::seed_from_u64(7);
        let mut second_rng = StdRng::seed_from_u64(7);

        let first: Vec<_> = (0..32)
            .map(|_| {
                let result = temperature_sample_step_with_rng(
                    logits_data(&[0.0, 0.5, 1.0]),
                    1.0,
                    None,
                    None,
                    &mut first_rng,
                )
                .unwrap();
                sampled_class(result)
            })
            .collect();
        let second: Vec<_> = (0..32)
            .map(|_| {
                let result = temperature_sample_step_with_rng(
                    logits_data(&[0.0, 0.5, 1.0]),
                    1.0,
                    None,
                    None,
                    &mut second_rng,
                )
                .unwrap();
                sampled_class(result)
            })
            .collect();

        assert_eq!(first, second);
    }

    #[test]
    fn top_k_and_top_p_filters_compose() {
        let candidates = filtered_sampling_weights(&[4.0, 3.0, 2.0, 1.0], 1.0, Some(3), Some(0.8));
        let indices: Vec<_> = candidates.iter().map(|(index, _)| *index).collect();

        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn top_p_always_retains_one_token() {
        let candidates = filtered_sampling_weights(&[2.0, 1.0, 0.0], 1.0, None, Some(f32::EPSILON));

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, 0);
    }

    #[test]
    fn top_p_one_retains_all_tokens() {
        let candidates = filtered_sampling_weights(&[3.0, 2.0, 1.0], 0.001, None, Some(1.0));
        let indices: Vec<_> = candidates.iter().map(|(index, _)| *index).collect();

        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn high_temperature_with_full_top_p_is_nearly_uniform() {
        const SAMPLES: usize = 20_000;
        const EXPECTED_PER_CLASS: usize = SAMPLES / 4;
        const TOLERANCE: usize = 400;

        let mut rng = StdRng::seed_from_u64(2026);
        let mut counts = [0usize; 4];
        for _ in 0..SAMPLES {
            let result = temperature_sample_step_with_rng(
                logits_data(&[0.0, 1.0, 2.0, 3.0]),
                1_000_000.0,
                None,
                Some(1.0),
                &mut rng,
            )
            .unwrap();
            counts[sampled_class(result)] += 1;
        }

        for count in counts {
            assert!(
                count.abs_diff(EXPECTED_PER_CLASS) <= TOLERANCE,
                "expected near-uniform counts, got {counts:?}"
            );
        }
    }

    #[test]
    fn temperature_sample_rejects_invalid_parameters() {
        for (temperature, top_k, top_p) in [
            (-0.1, None, None),
            (f32::NAN, None, None),
            (f32::INFINITY, None, None),
            (1.0, Some(0), None),
            (1.0, None, Some(0.0)),
            (1.0, None, Some(-0.1)),
            (1.0, None, Some(1.1)),
            (1.0, None, Some(f32::NAN)),
            (1.0, None, Some(f32::INFINITY)),
        ] {
            assert_invalid(sample_with_seed(
                logits_data(&[0.0, 1.0]),
                temperature,
                top_k,
                top_p,
            ));
        }
    }

    #[test]
    fn temperature_sample_rejects_invalid_inputs() {
        assert_invalid(sample_with_seed(
            RawOutputs::Text("not logits".to_string()),
            1.0,
            None,
            None,
        ));
        assert_invalid(sample_with_seed(
            RawOutputs::TensorMap(HashMap::new()),
            1.0,
            None,
            None,
        ));
        assert_invalid(sample_with_seed(logits_data(&[]), 1.0, None, None));
        assert_invalid(sample_with_seed(
            logits_data(&[0.0, f32::NAN]),
            1.0,
            None,
            None,
        ));

        let non_contiguous = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0])
            .expect("valid shape")
            .reversed_axes()
            .into_dyn();
        let data = RawOutputs::TensorMap(HashMap::from([("logits".to_string(), non_contiguous)]));
        assert_invalid(sample_with_seed(data, 1.0, None, None));
    }

    #[test]
    fn test_denormalize_step_round_trip() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let mean = vec![2.5f32];
        let std_vals = vec![1.0f32];

        let orig_tensor =
            ArrayD::from_shape_vec(IxDyn(&[4]), original.clone()).expect("valid shape");
        let norm_data =
            normalize_step(PreprocessedData::Tensor(orig_tensor), &mean, &std_vals).unwrap();

        let norm_tensor = match norm_data {
            PreprocessedData::Tensor(t) => t,
            _ => panic!("Expected Tensor"),
        };

        let mut map = HashMap::new();
        map.insert("output".to_string(), norm_tensor);

        let result = denormalize_step(RawOutputs::TensorMap(map), &mean, &std_vals).unwrap();

        match result {
            RawOutputs::TensorMap(out_map) => {
                let out = out_map.values().next().unwrap();
                for (actual, expected) in out.iter().zip(original.iter()) {
                    assert!(
                        (actual - expected).abs() < 1e-5,
                        "round-trip failed: expected {}, got {}",
                        expected,
                        actual
                    );
                }
            }
            _ => panic!("Expected TensorMap output"),
        }
    }

    #[test]
    fn test_denormalize_step_per_channel() {
        // Flat slice with 3-element cycling: indices 0,1,2 use channels 0,1,2; then repeat.
        // Pre-normalized values: (-1 * std[c]) + mean[c] should recover the original.
        let mean = vec![1.0f32, 2.0, 3.0];
        let std_vals = vec![1.0f32, 1.0, 1.0];

        // Normalized values corresponding to original [0,1,2,3,4,5]:
        // (0-1)/1=-1, (1-2)/1=-1, (2-3)/1=-1, (3-1)/1=2, (4-2)/1=2, (5-3)/1=2
        let normalized = vec![-1.0f32, -1.0, -1.0, 2.0, 2.0, 2.0];
        let expected = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];

        let tensor = ArrayD::from_shape_vec(IxDyn(&[6]), normalized).expect("valid shape");
        let mut map = HashMap::new();
        map.insert("output".to_string(), tensor);

        let result = denormalize_step(RawOutputs::TensorMap(map), &mean, &std_vals).unwrap();

        match result {
            RawOutputs::TensorMap(out_map) => {
                let out = out_map.values().next().unwrap();
                for (actual, exp) in out.iter().zip(expected.iter()) {
                    assert!(
                        (actual - exp).abs() < 1e-5,
                        "per-channel failed: expected {}, got {}",
                        exp,
                        actual
                    );
                }
            }
            _ => panic!("Expected TensorMap output"),
        }
    }

    #[test]
    fn test_denormalize_step_scalar_broadcast() {
        // Single mean/std broadcasts to all elements.
        let mean = vec![0.5f32];
        let std_vals = vec![2.0f32];

        // Normalized zeros: (0.0 * 2.0) + 0.5 = 0.5 for every element.
        let tensor = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0f32; 4]).expect("valid shape");
        let mut map = HashMap::new();
        map.insert("output".to_string(), tensor);

        let result = denormalize_step(RawOutputs::TensorMap(map), &mean, &std_vals).unwrap();

        match result {
            RawOutputs::TensorMap(out_map) => {
                let out = out_map.values().next().unwrap();
                for &val in out.iter() {
                    assert!(
                        (val - 0.5f32).abs() < 1e-5,
                        "scalar broadcast failed: expected 0.5, got {}",
                        val
                    );
                }
            }
            _ => panic!("Expected TensorMap output"),
        }
    }

    #[test]
    fn test_denormalize_step_shape_mismatch() {
        // mean and std have different lengths — must error, not panic.
        let mean = vec![0.5f32, 0.5];
        let std_vals = vec![1.0f32]; // length mismatch

        let tensor = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0f32; 4]).expect("valid shape");
        let mut map = HashMap::new();
        map.insert("output".to_string(), tensor);

        let result = denormalize_step(RawOutputs::TensorMap(map), &mean, &std_vals);
        assert!(result.is_err(), "expected error on shape mismatch");
    }
}
