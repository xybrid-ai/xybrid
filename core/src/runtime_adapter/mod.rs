//! Runtime Adapter module - Interface for model runtime backends.
//!
//! The RuntimeAdapter trait provides a unified interface for executing inference
//! across different model runtime backends (ONNX, CoreML, Candle, etc.).
//!
//! Runtime adapters are responsible for:
//! - Loading model files from .xyb bundles
//! - Executing inference on input envelopes
//! - Returning output envelopes
//! - Managing runtime-specific resources (session pools, memory, etc.)
//!
//! # Module Organization
//!
//! Each runtime backend is organized in its own subdirectory:
//! - `onnx/` - ONNX Runtime backend (cross-platform)
//! - `coreml/` - CoreML backend (iOS/macOS)
//! - `candle/` - Candle backend (pure Rust, feature-gated) [planned]
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::runtime_adapter::{RuntimeAdapter, OnnxRuntimeAdapter};
//! use xybrid_core::ir::{Envelope, EnvelopeKind};
//!
//! // Create adapter
//! let mut adapter = OnnxRuntimeAdapter::new();
//!
//! // Load model from bundle
//! adapter.load_model("/path/to/model.onnx")?;
//!
//! // Run inference
//! let input = Envelope::new(EnvelopeKind::Text("hello world".to_string()));
//! let output = adapter.execute(&input)?;
//! ```

use crate::ir::Envelope;
use std::collections::HashMap;
use thiserror::Error;

// Shared utilities (stay at root level)
pub mod inference_backend;
pub mod mel_spectrogram;
pub mod metadata_driven;
pub mod tensor_utils;
pub mod traits;

// Runtime backends (organized in subdirectories)
pub mod onnx;

#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub mod coreml;

// Candle backend (feature-gated, pure Rust ML framework)
#[cfg(feature = "candle")]
pub mod candle;

// Re-exports from runtime backends
pub use onnx::OnnxBackend;
pub use onnx::OnnxRuntimeAdapter;
pub use onnx::ONNXSession;

#[cfg(any(target_os = "android", test))]
pub use onnx::ONNXMobileRuntimeAdapter;

#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub use coreml::CoreMLRuntimeAdapter;

#[cfg(feature = "candle")]
pub use candle::{CandleBackend, CandleRuntimeAdapter};

// Re-export inference backend types
pub use inference_backend::{BackendError, BackendResult, InferenceBackend, RuntimeType};
pub use traits::ModelRuntime;

/// Error type for runtime adapter operations.
#[derive(Error, Debug)]
pub enum AdapterError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

/// Result type for runtime adapter operations.
pub type AdapterResult<T> = Result<T, AdapterError>;

/// Metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model identifier
    pub model_id: String,
    /// Model version
    pub version: String,
    /// Runtime type (e.g., "onnx", "coreml", "candle")
    pub runtime_type: String,
    /// Model file path or location
    pub model_path: String,
    /// Input shapes/names (runtime-specific)
    pub input_schema: HashMap<String, Vec<u64>>,
    /// Output shapes/names (runtime-specific)
    pub output_schema: HashMap<String, Vec<u64>>,
}

/// Trait for model runtime adapters.
///
/// Runtime adapters abstract over different inference backends,
/// allowing the orchestrator to execute models without knowing
/// the underlying runtime implementation.
///
/// Adapters must be thread-safe (Send + Sync) to support concurrent
/// execution in the orchestrator.
pub trait RuntimeAdapter: Send + Sync {
    /// Returns the name of this adapter (e.g., "onnx", "coreml", "candle").
    fn name(&self) -> &str;

    /// Returns a list of file formats supported by this adapter.
    ///
    /// Examples: ["onnx", "onnx.gz"], ["mlpackage"], ["safetensors"]
    fn supported_formats(&self) -> Vec<&'static str>;

    /// Loads a model from the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the model file (ONNX, CoreML, SafeTensors, etc.)
    ///
    /// # Returns
    ///
    /// Unit on success, or an error if loading fails
    fn load_model(&mut self, path: &str) -> AdapterResult<()>;

    /// Executes inference on the currently loaded model.
    ///
    /// # Arguments
    ///
    /// * `input` - Input envelope containing the inference data
    ///
    /// # Returns
    ///
    /// Output envelope with inference results
    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope>;
}

/// Extension trait for runtime adapters that support multiple models.
///
/// This provides additional functionality for adapters that need to manage
/// multiple loaded models simultaneously.
pub trait RuntimeAdapterExt {
    /// Checks if a model is loaded.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier to check
    ///
    /// # Returns
    ///
    /// True if the model is loaded, false otherwise
    fn is_loaded(&self, model_id: &str) -> bool;

    /// Gets metadata for a loaded model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier
    ///
    /// # Returns
    ///
    /// ModelMetadata if the model is loaded
    fn get_metadata(&self, model_id: &str) -> AdapterResult<&ModelMetadata>;

    /// Runs inference on the specified model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier
    /// * `input` - Input envelope containing the inference data
    ///
    /// # Returns
    ///
    /// Output envelope with inference results
    fn infer(&self, model_id: &str, input: &Envelope) -> AdapterResult<Envelope>;

    /// Unloads a model, freeing its resources.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier to unload
    fn unload_model(&mut self, model_id: &str) -> AdapterResult<()>;

    /// Lists all currently loaded models.
    ///
    /// # Returns
    ///
    /// Vector of model identifiers
    fn list_loaded_models(&self) -> Vec<String>;
}
