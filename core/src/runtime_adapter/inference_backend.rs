//! Inference Backend Trait - Low-level runtime abstraction for model execution.
//!
//! This module provides a unified interface for different inference runtimes
//! (ONNX Runtime, Burn, CoreML, etc.) at the tensor level.
//!
//! Unlike the high-level RuntimeAdapter trait which works with Envelopes,
//! InferenceBackend works directly with tensors, allowing the TemplateExecutor
//! to remain runtime-agnostic.

use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Error type for inference backend operations
#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Failed to load model: {0}")]
    LoadFailed(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

pub type BackendResult<T> = Result<T, BackendError>;

/// Runtime type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeType {
    /// ONNX Runtime (ort crate)
    Onnx,

    /// Burn framework (pure Rust)
    Burn,

    /// CoreML (Apple platforms)
    CoreML,

    /// Candle framework (HuggingFace, pure Rust)
    #[cfg(feature = "candle")]
    Candle,
}

impl RuntimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RuntimeType::Onnx => "onnx",
            RuntimeType::Burn => "burn",
            RuntimeType::CoreML => "coreml",
            #[cfg(feature = "candle")]
            RuntimeType::Candle => "candle",
        }
    }
}

/// Low-level inference backend trait
///
/// This trait provides a runtime-agnostic interface for tensor-based inference.
/// Implementations handle model loading and execution for specific runtimes.
///
/// # Thread Safety
///
/// Implementations must be Send + Sync to support concurrent execution.
pub trait InferenceBackend: Send + Sync {
    /// Get the runtime type identifier
    fn runtime_type(&self) -> RuntimeType;

    /// Load a model from disk
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file (e.g., .onnx, .mpk.gz)
    /// * `config_path` - Optional path to config file (e.g., .cfg for Burn)
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or BackendError on failure
    fn load_model(
        &mut self,
        model_path: &Path,
        config_path: Option<&Path>,
    ) -> BackendResult<()>;

    /// Run inference with tensor inputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input name -> tensor (f32 arrays)
    ///
    /// # Returns
    ///
    /// Map of output name -> tensor, or BackendError on failure
    fn run_inference(
        &self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> BackendResult<HashMap<String, ArrayD<f32>>>;

    /// Check if a model is currently loaded
    fn is_loaded(&self) -> bool;

    /// Get input names expected by the loaded model
    fn input_names(&self) -> BackendResult<Vec<String>>;

    /// Get output names produced by the loaded model
    fn output_names(&self) -> BackendResult<Vec<String>>;
}
