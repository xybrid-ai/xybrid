//! ONNX Runtime Inference Backend
//!
//! This module provides an ONNX Runtime-based inference backend.
//! It wraps the existing ONNXSession implementation to conform to the
//! InferenceBackend trait.

use crate::runtime_adapter::inference_backend::{BackendError, BackendResult, InferenceBackend, RuntimeType};
use super::session::ONNXSession;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

/// ONNX Runtime inference backend
///
/// This backend uses ONNX Runtime (via the ort crate) for model execution.
/// It provides hardware-accelerated inference on various platforms:
/// - macOS/iOS: Metal backend
/// - Android: NNAPI backend
/// - Desktop: CPU backend with optional CUDA
pub struct OnnxBackend {
    /// ONNX Runtime session
    session: Option<ONNXSession>,
}

impl OnnxBackend {
    /// Create a new ONNX backend
    pub fn new() -> Self {
        Self { session: None }
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for OnnxBackend {
    fn runtime_type(&self) -> RuntimeType {
        RuntimeType::Onnx
    }

    fn load_model(
        &mut self,
        model_path: &Path,
        _config_path: Option<&Path>,
    ) -> BackendResult<()> {
        // Load ONNX model using existing ONNXSession
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| BackendError::LoadFailed("Invalid model path".to_string()))?;

        // Create session with hardware acceleration hints (auto-detected by ort)
        let session = ONNXSession::new(model_path_str, false, false).map_err(|e| {
            BackendError::LoadFailed(format!("Failed to load ONNX model: {}", e))
        })?;

        self.session = Some(session);
        Ok(())
    }

    fn run_inference(
        &self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> BackendResult<HashMap<String, ArrayD<f32>>> {
        let session = self
            .session
            .as_ref()
            .ok_or(BackendError::ModelNotLoaded)?;

        // ONNXSession.run() already takes HashMap<String, ArrayD<f32>>
        // and returns HashMap<String, ArrayD<f32>>, so we just pass through
        session.run(inputs).map_err(|e| {
            BackendError::InferenceFailed(format!("ONNX inference failed: {}", e))
        })
    }

    fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    fn input_names(&self) -> BackendResult<Vec<String>> {
        let session = self
            .session
            .as_ref()
            .ok_or(BackendError::ModelNotLoaded)?;

        Ok(session.input_names().to_vec())
    }

    fn output_names(&self) -> BackendResult<Vec<String>> {
        let session = self
            .session
            .as_ref()
            .ok_or(BackendError::ModelNotLoaded)?;

        Ok(session.output_names().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_backend_creation() {
        let backend = OnnxBackend::new();
        assert_eq!(backend.runtime_type(), RuntimeType::Onnx);
        assert!(!backend.is_loaded());
    }
}
