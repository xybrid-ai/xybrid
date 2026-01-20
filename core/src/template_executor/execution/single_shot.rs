//! Single-shot execution mode.
//!
//! This module handles single forward pass execution for pipeline stages
//! that don't require iterative processing.

use super::super::types::{ExecutorResult, PreprocessedData, RawOutputs};
use crate::execution_template::PipelineStage;
use crate::runtime_adapter::{AdapterError, ModelRuntime};
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

/// Execute a single-shot stage (run once).
///
/// # Arguments
/// - `stage`: The pipeline stage configuration
/// - `current_data`: Input data for this stage
/// - `_stage_outputs`: Previous stage outputs (unused for single-shot)
/// - `runtime`: The model runtime to use
/// - `base_path`: Base path for resolving model files
pub fn execute_single_shot_stage(
    stage: &PipelineStage,
    current_data: &PreprocessedData,
    _stage_outputs: &HashMap<String, HashMap<String, ArrayD<f32>>>,
    runtime: &mut dyn ModelRuntime,
    base_path: &str,
) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
    // Resolution of path
    let model_full_path = Path::new(base_path).join(&stage.model_file);
    runtime
        .load(&model_full_path)
        .map_err(|e| AdapterError::RuntimeError(format!("Stage load failed: {}", e)))?;

    // Convert input
    let input_envelope = current_data.to_envelope()?;

    // Execute
    let output_envelope = runtime.execute(&input_envelope)?;

    // Convert output to TensorMap
    let raw_outputs = RawOutputs::from_envelope(&output_envelope)?;
    if let RawOutputs::TensorMap(map) = raw_outputs {
        Ok(map)
    } else {
        // Adapt other outputs to TensorMap if possible, or error
        Err(AdapterError::RuntimeError(
            "Stage execution did not return tensors".to_string(),
        ))
    }
}
