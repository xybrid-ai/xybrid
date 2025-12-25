//! ONNX Runtime Adapter implementation.
//!
//! This module provides real ONNX Runtime inference for desktop platforms.
//! It uses the ort crate for ONNX Runtime bindings and supports CPU execution
//! with optional Metal acceleration on macOS/iOS.
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::runtime_adapter::onnx::OnnxRuntimeAdapter;
//! use xybrid_core::runtime_adapter::RuntimeAdapter;
//!
//! let mut adapter = OnnxRuntimeAdapter::new();
//! adapter.load_model("/path/to/model.onnx")?;
//! ```

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{
    AdapterError, AdapterResult, ModelMetadata, RuntimeAdapter, RuntimeAdapterExt,
};
use super::session::ONNXSession;
use crate::runtime_adapter::tensor_utils::{envelope_to_tensors, tensors_to_envelope};
use std::collections::HashMap;
use std::path::Path;

/// ONNX Runtime Adapter for desktop platforms.
///
/// This adapter provides real ONNX Runtime inference using the ort crate.
/// It supports CPU execution with optional Metal acceleration on macOS/iOS.
///
/// # Behavior
///
/// - `load_model()`: Loads ONNX model using ONNX Runtime and stores session
/// - `execute()`: Runs real inference using ONNX Runtime
/// - Supports multiple models loaded simultaneously via `RuntimeAdapterExt`
pub struct OnnxRuntimeAdapter {
    /// Map of loaded models (model_id -> metadata)
    models: HashMap<String, ModelMetadata>,
    /// Map of ONNX Runtime sessions (model_id -> session)
    sessions: HashMap<String, ONNXSession>,
    /// Currently active model (for simple single-model execution)
    current_model: Option<String>,
}

impl OnnxRuntimeAdapter {
    /// Creates a new ONNX Runtime Adapter instance.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            sessions: HashMap::new(),
            current_model: None,
        }
    }

    /// Validates that a model file exists and is accessible.
    fn validate_model_file(&self, model_path: &str) -> AdapterResult<()> {
        let path = Path::new(model_path);

        if !path.exists() {
            return Err(AdapterError::ModelNotFound(format!(
                "Model file not found: {}",
                model_path
            )));
        }

        if !path.is_file() {
            return Err(AdapterError::InvalidInput(format!(
                "Path is not a file: {}",
                model_path
            )));
        }

        // Check if it's an ONNX file (basic validation)
        if let Some(ext) = path.extension() {
            if ext != "onnx" && ext != "ONNX" {
                // Warn but don't fail (some models might have different extensions)
            }
        }

        Ok(())
    }

    /// Extracts model ID from file path (for internal tracking).
    fn extract_model_id(&self, path: &str) -> String {
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Runs real ONNX Runtime inference.
    ///
    /// # Arguments
    ///
    /// * `session` - ONNX Runtime session for the model
    /// * `input` - Input envelope
    ///
    /// # Returns
    ///
    /// Output envelope with inference results
    fn real_inference(&self, session: &ONNXSession, input: &Envelope) -> AdapterResult<Envelope> {
        // Convert Envelope to tensors
        let input_shapes: Vec<Vec<i64>> = session.input_shapes().to_vec();
        let input_names: Vec<String> = session.input_names().to_vec();

        let input_tensors = envelope_to_tensors(input, &input_shapes, &input_names)?;

        // Run inference (session uses RefCell for interior mutability)
        let output_tensors = session
            .run(input_tensors)
            .map_err(|e| AdapterError::InferenceFailed(format!("ONNX Runtime inference failed: {e}")))?;

        // DEBUG: Log raw output tensor info before conversion
        eprintln!("🔵 DEBUG: Raw ONNX Output Tensors");
        eprintln!("   Number of outputs: {}", output_tensors.len());
        for (name, tensor) in &output_tensors {
            eprintln!("   Output '{}': shape {:?}, size {}", name, tensor.shape(), tensor.len());
        }

        let output_names: Vec<String> = session.output_names().to_vec();
        let output = tensors_to_envelope(&output_tensors, &output_names)?;

        Ok(output)
    }
    /// Get session for a loaded model path
    pub fn get_session(&self, model_path: &str) -> AdapterResult<&ONNXSession> {
        // Extract ID just like load_model does
        let model_id = self.extract_model_id(model_path);
        
        self.sessions.get(&model_id).ok_or_else(|| {
             AdapterError::ModelNotLoaded(format!("Session for model '{}' (path: {}) not found", model_id, model_path))
        })
    }
}

impl Default for OnnxRuntimeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for OnnxRuntimeAdapter {
    fn name(&self) -> &str {
        "onnx"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["onnx", "onnx.gz"]
    }

    fn load_model(&mut self, path: &str) -> AdapterResult<()> {
        // Validate model file exists
        self.validate_model_file(path)?;

        // Extract model ID from path
        let model_id = self.extract_model_id(path);

        // Check if model is already loaded
        if self.models.contains_key(&model_id) {
            return Err(AdapterError::RuntimeError(format!(
                "Model '{}' is already loaded",
                model_id
            )));
        }

        // Create ONNX Runtime session
        // Use Metal on macOS/iOS if available, otherwise CPU
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let use_metal = true;
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let use_metal = false;

        let use_nnapi = false; // NNAPI is Android-specific

        let session = ONNXSession::new(path, use_nnapi, use_metal)?;

        // Extract real input/output shapes from session
        let input_shapes = session.input_shapes();
        let output_shapes = session.output_shapes();
        let input_names = session.input_names();
        let output_names = session.output_names();

        // Create metadata with real shapes from session
        let mut input_schema = HashMap::new();
        for (i, name) in input_names.iter().enumerate() {
            if let Some(shape) = input_shapes.get(i) {
                input_schema.insert(name.clone(), shape.iter().map(|&s| s as u64).collect());
            }
        }

        let mut output_schema = HashMap::new();
        for (i, name) in output_names.iter().enumerate() {
            if let Some(shape) = output_shapes.get(i) {
                output_schema.insert(name.clone(), shape.iter().map(|&s| s as u64).collect());
            }
        }

        let metadata = ModelMetadata {
            model_id: model_id.clone(),
            version: "1.0.0".to_string(), // Default version
            runtime_type: "onnx".to_string(),
            model_path: path.to_string(),
            input_schema,
            output_schema,
        };

        // Store session and metadata
        self.sessions.insert(model_id.clone(), session);
        self.models.insert(model_id.clone(), metadata);
        self.current_model = Some(model_id);

        Ok(())
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        // Check if a model is loaded
        let model_id = self.current_model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load_model() first.".to_string())
        })?;

        // Get session for current model
        let session = self.sessions.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Session for model '{}' not found", model_id))
        })?;

        // Run real inference
        self.real_inference(session, input)
    }
}

impl RuntimeAdapterExt for OnnxRuntimeAdapter {
    fn is_loaded(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    fn get_metadata(&self, model_id: &str) -> AdapterResult<&ModelMetadata> {
        self.models.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Model '{}' is not loaded", model_id))
        })
    }

    fn infer(&self, model_id: &str, input: &Envelope) -> AdapterResult<Envelope> {
        // Check if model is loaded
        if !self.is_loaded(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{}' is not loaded. Call load_model() first.",
                model_id
            )));
        }

        // Get session for model
        let session = self.sessions.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Session for model '{}' not found", model_id))
        })?;

        // Run real inference
        self.real_inference(session, input)
    }

    fn unload_model(&mut self, model_id: &str) -> AdapterResult<()> {
        if !self.models.contains_key(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{}' is not loaded",
                model_id
            )));
        }

        // Remove session (will be dropped automatically, freeing resources)
        self.sessions.remove(model_id);

        // Remove metadata
        self.models.remove(model_id);

        // Clear current model if it was the one being unloaded
        if self.current_model.as_ref() == Some(&model_id.to_string()) {
            self.current_model = None;
        }

        Ok(())
    }

    fn list_loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_mock_onnx_file() -> (TempDir, String) {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");
        fs::write(&model_path, b"fake onnx model data").unwrap();
        (temp_dir, model_path.to_string_lossy().to_string())
    }

    #[test]
    fn test_create_adapter() {
        let adapter = OnnxRuntimeAdapter::new();
        assert!(adapter.list_loaded_models().is_empty());
    }

    #[test]
    fn test_adapter_name() {
        let adapter = OnnxRuntimeAdapter::new();
        assert_eq!(adapter.name(), "onnx");
    }

    #[test]
    fn test_supported_formats() {
        let adapter = OnnxRuntimeAdapter::new();
        let formats = adapter.supported_formats();
        assert!(formats.contains(&"onnx"));
        assert!(formats.contains(&"onnx.gz"));
    }

    #[test]
    fn test_load_model_not_found() {
        let mut adapter = OnnxRuntimeAdapter::new();
        let result = adapter.load_model("/nonexistent/model.onnx");
        assert!(matches!(result, Err(AdapterError::ModelNotFound(_))));
    }

    #[test]
    fn test_execute_no_model_loaded() {
        let adapter = OnnxRuntimeAdapter::new();
        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = adapter.execute(&input);
        assert!(matches!(result, Err(AdapterError::ModelNotLoaded(_))));
    }

    #[test]
    fn test_infer_model_not_loaded() {
        let adapter = OnnxRuntimeAdapter::new();
        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = adapter.infer("nonexistent-model", &input);
        assert!(matches!(result, Err(AdapterError::ModelNotLoaded(_))));
    }
}
