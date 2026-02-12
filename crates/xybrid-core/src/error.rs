//! Unified error types for xybrid-core public API.
//!
//! This module provides a canonical error type hierarchy for all public API methods.
//! Internal modules may use their own error types, but should convert to `XybridError`
//! at module boundaries.
//!
//! # Error Hierarchy
//!
//! ```text
//! XybridError
//! ├── Inference(InferenceError)  -- Model execution failures
//! ├── Pipeline(PipelineError)    -- Pipeline orchestration failures
//! ├── NotFound(String)           -- Resource/model not found
//! ├── Config(String)             -- Configuration errors
//! ├── Io(std::io::Error)         -- I/O errors
//! └── Serialization(String)      -- JSON/YAML parsing errors
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_core::error::{XybridError, XybridResult};
//!
//! fn load_and_run() -> XybridResult<String> {
//!     // Operations that may fail return XybridResult
//!     Ok("success".to_string())
//! }
//! ```

use thiserror::Error;

/// The canonical error type for xybrid-core public API.
///
/// All public API methods return `Result<T, XybridError>`.
#[derive(Error, Debug)]
pub enum XybridError {
    /// Model inference failed
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    /// Pipeline execution failed
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Resource not found (model, file, etc.)
    #[error("Not found: {0}")]
    NotFound(String),

    /// Invalid configuration
    #[error("Configuration error: {0}")]
    Config(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error (JSON, YAML, etc.)
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Errors during model inference.
#[derive(Error, Debug)]
pub enum InferenceError {
    /// Model not loaded before inference
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    /// Invalid input data format or shape
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Backend-specific error (ONNX, CoreML, Candle, etc.)
    #[error("Backend error: {0}")]
    Backend(String),

    /// Preprocessing step failed
    #[error("Preprocessing failed: {0}")]
    Preprocessing(String),

    /// Postprocessing step failed
    #[error("Postprocessing failed: {0}")]
    Postprocessing(String),
}

/// Errors during pipeline execution.
#[derive(Error, Debug)]
pub enum PipelineError {
    /// A pipeline stage failed
    #[error("Stage '{stage}' failed: {reason}")]
    StageFailed {
        /// Name of the failed stage
        stage: String,
        /// Reason for failure
        reason: String,
    },

    /// Invalid execution target specified
    #[error("Invalid target: {0}")]
    InvalidTarget(String),

    /// Provider/backend error
    #[error("Provider error: {0}")]
    Provider(String),

    /// Policy denied execution
    #[error("Policy denied: {0}")]
    PolicyDenied(String),

    /// Pipeline resolution/parsing error
    #[error("Resolution error: {0}")]
    Resolution(String),
}

/// Result type alias for xybrid-core.
pub type XybridResult<T> = Result<T, XybridError>;

// ─────────────────────────────────────────────────────────────────────────────
// Conversions from internal errors
// ─────────────────────────────────────────────────────────────────────────────

impl From<crate::runtime_adapter::AdapterError> for XybridError {
    fn from(e: crate::runtime_adapter::AdapterError) -> Self {
        use crate::runtime_adapter::AdapterError;
        match e {
            AdapterError::ModelNotFound(s) => XybridError::NotFound(s),
            AdapterError::ModelNotLoaded(s) => {
                XybridError::Inference(InferenceError::ModelNotLoaded(s))
            }
            AdapterError::InvalidInput(s) => {
                XybridError::Inference(InferenceError::InvalidInput(s))
            }
            AdapterError::InferenceFailed(s) => XybridError::Inference(InferenceError::Backend(s)),
            AdapterError::IOError(e) => XybridError::Io(e),
            AdapterError::SerializationError(s) => XybridError::Serialization(s),
            AdapterError::RuntimeError(s) => XybridError::Inference(InferenceError::Backend(s)),
        }
    }
}

impl From<serde_json::Error> for XybridError {
    fn from(e: serde_json::Error) -> Self {
        XybridError::Serialization(e.to_string())
    }
}

impl From<serde_yaml::Error> for XybridError {
    fn from(e: serde_yaml::Error) -> Self {
        XybridError::Serialization(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

impl XybridError {
    /// Create a "not found" error.
    pub fn not_found(msg: impl Into<String>) -> Self {
        XybridError::NotFound(msg.into())
    }

    /// Create a configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        XybridError::Config(msg.into())
    }

    /// Create a serialization error.
    pub fn serialization(msg: impl Into<String>) -> Self {
        XybridError::Serialization(msg.into())
    }
}

impl InferenceError {
    /// Create a "model not loaded" error.
    pub fn model_not_loaded(msg: impl Into<String>) -> Self {
        InferenceError::ModelNotLoaded(msg.into())
    }

    /// Create an "invalid input" error.
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        InferenceError::InvalidInput(msg.into())
    }

    /// Create a backend error.
    pub fn backend(msg: impl Into<String>) -> Self {
        InferenceError::Backend(msg.into())
    }

    /// Create a preprocessing error.
    pub fn preprocessing(msg: impl Into<String>) -> Self {
        InferenceError::Preprocessing(msg.into())
    }

    /// Create a postprocessing error.
    pub fn postprocessing(msg: impl Into<String>) -> Self {
        InferenceError::Postprocessing(msg.into())
    }
}

impl PipelineError {
    /// Create a stage failed error.
    pub fn stage_failed(stage: impl Into<String>, reason: impl Into<String>) -> Self {
        PipelineError::StageFailed {
            stage: stage.into(),
            reason: reason.into(),
        }
    }

    /// Create an invalid target error.
    pub fn invalid_target(msg: impl Into<String>) -> Self {
        PipelineError::InvalidTarget(msg.into())
    }

    /// Create a provider error.
    pub fn provider(msg: impl Into<String>) -> Self {
        PipelineError::Provider(msg.into())
    }

    /// Create a policy denied error.
    pub fn policy_denied(msg: impl Into<String>) -> Self {
        PipelineError::PolicyDenied(msg.into())
    }

    /// Create a resolution error.
    pub fn resolution(msg: impl Into<String>) -> Self {
        PipelineError::Resolution(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xybrid_error_display() {
        let err = XybridError::NotFound("model.onnx".to_string());
        assert_eq!(err.to_string(), "Not found: model.onnx");
    }

    #[test]
    fn test_inference_error_display() {
        let err = InferenceError::Backend("ONNX runtime failed".to_string());
        assert_eq!(err.to_string(), "Backend error: ONNX runtime failed");
    }

    #[test]
    fn test_pipeline_error_display() {
        let err = PipelineError::StageFailed {
            stage: "tts".to_string(),
            reason: "voice not found".to_string(),
        };
        assert_eq!(err.to_string(), "Stage 'tts' failed: voice not found");
    }

    #[test]
    fn test_adapter_error_conversion() {
        use crate::runtime_adapter::AdapterError;

        let adapter_err = AdapterError::ModelNotFound("whisper.onnx".to_string());
        let xybrid_err: XybridError = adapter_err.into();
        assert!(matches!(xybrid_err, XybridError::NotFound(_)));

        let adapter_err = AdapterError::InferenceFailed("ORT error".to_string());
        let xybrid_err: XybridError = adapter_err.into();
        assert!(matches!(
            xybrid_err,
            XybridError::Inference(InferenceError::Backend(_))
        ));
    }

    #[test]
    fn test_convenience_constructors() {
        let err = XybridError::not_found("test.onnx");
        assert!(matches!(err, XybridError::NotFound(_)));

        let err = InferenceError::backend("runtime crash");
        assert!(matches!(err, InferenceError::Backend(_)));

        let err = PipelineError::stage_failed("asr", "timeout");
        assert!(matches!(err, PipelineError::StageFailed { .. }));
    }

    #[test]
    fn test_json_error_conversion() {
        let json_str = "invalid json {";
        let result: Result<serde_json::Value, _> = serde_json::from_str(json_str);
        let xybrid_err: XybridError = result.unwrap_err().into();
        assert!(matches!(xybrid_err, XybridError::Serialization(_)));
    }
}
