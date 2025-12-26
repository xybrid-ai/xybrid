//! Tensor preprocessing operations.
//!
//! This module provides:
//! - `normalize_step`: Normalize tensor values using mean and standard deviation
//! - `reshape_step`: Reshape tensor to target dimensions

use crate::runtime_adapter::AdapterError;
use ndarray::IxDyn;
use super::super::types::{ExecutorResult, PreprocessedData};

/// Normalize tensor values using mean and standard deviation.
///
/// # Arguments
/// - `data`: Input data (Tensor)
/// - `mean`: Per-channel mean values
/// - `std`: Per-channel standard deviation values
pub fn normalize_step(
    data: PreprocessedData,
    mean: &[f32],
    std: &[f32],
) -> ExecutorResult<PreprocessedData> {
    let mut tensor = match data {
        PreprocessedData::Tensor(t) => t,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Normalize requires tensor input".to_string(),
            ))
        }
    };

    if let Some(tensor_slice) = tensor.as_slice_mut() {
        for (i, val) in tensor_slice.iter_mut().enumerate() {
            let channel = i % mean.len();
            *val = (*val - mean[channel]) / std[channel];
        }
    }

    Ok(PreprocessedData::Tensor(tensor))
}

/// Reshape tensor to target dimensions.
///
/// # Arguments
/// - `data`: Input data (Tensor)
/// - `shape`: Target shape dimensions
pub fn reshape_step(data: PreprocessedData, shape: &[usize]) -> ExecutorResult<PreprocessedData> {
    let tensor = match data {
        PreprocessedData::Tensor(t) => t,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Reshape requires tensor input".to_string(),
            ))
        }
    };

    let total_elements: usize = shape.iter().product();
    let tensor_elements = tensor.len();

    if total_elements != tensor_elements {
        return Err(AdapterError::InvalidInput(format!(
            "Cannot reshape tensor: shape {:?} requires {} elements, but tensor has {}",
            shape, total_elements, tensor_elements
        )));
    }

    #[allow(deprecated)]
    let reshaped = tensor
        .into_shape(IxDyn(shape))
        .map_err(|e| AdapterError::InvalidInput(format!("Failed to reshape tensor: {:?}", e)))?;

    Ok(PreprocessedData::Tensor(reshaped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_normalize_step_basic() {
        let data = Array1::from_vec(vec![0.0, 10.0, 20.0, 30.0]).into_dyn();
        let input = PreprocessedData::Tensor(data);

        let mean = vec![10.0];
        let std = vec![10.0];

        let result = normalize_step(input, &mean, &std);

        assert!(result.is_ok());
        match result.unwrap() {
            PreprocessedData::Tensor(tensor) => {
                let values: Vec<f32> = tensor.iter().cloned().collect();
                // (0-10)/10 = -1, (10-10)/10 = 0, (20-10)/10 = 1, (30-10)/10 = 2
                assert!((values[0] - (-1.0)).abs() < 0.001);
                assert!((values[1] - 0.0).abs() < 0.001);
                assert!((values[2] - 1.0).abs() < 0.001);
                assert!((values[3] - 2.0).abs() < 0.001);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_normalize_step_multichannel() {
        let data = ndarray::Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();
        let input = PreprocessedData::Tensor(data);

        let mean = vec![1.0, 2.0, 3.0];
        let std = vec![1.0, 1.0, 1.0];

        let result = normalize_step(input, &mean, &std);

        assert!(result.is_ok());
    }

    #[test]
    fn test_normalize_step_invalid_input() {
        let data = PreprocessedData::Text("text".to_string());
        let result = normalize_step(data, &[0.0], &[1.0]);

        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_step_basic() {
        let data = Array1::from_vec((0..12).map(|i| i as f32).collect()).into_dyn();
        let input = PreprocessedData::Tensor(data);

        let result = reshape_step(input, &[3, 4]);

        assert!(result.is_ok());
        match result.unwrap() {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[3, 4]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_reshape_step_multidimensional() {
        let data = Array1::from_vec((0..24).map(|i| i as f32).collect()).into_dyn();
        let input = PreprocessedData::Tensor(data);

        let result = reshape_step(input, &[2, 4, 3]);

        assert!(result.is_ok());
        match result.unwrap() {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[2, 4, 3]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_reshape_step_invalid_shape() {
        let data = Array1::from_vec((0..10).map(|i| i as f32).collect()).into_dyn();
        let input = PreprocessedData::Tensor(data);

        // 10 cannot be reshaped to [3, 4] (12 elements)
        let result = reshape_step(input, &[3, 4]);

        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_step_invalid_input() {
        let data = PreprocessedData::Text("text".to_string());
        let result = reshape_step(data, &[2, 2]);

        assert!(result.is_err());
    }
}
