//! CoreML Runtime Adapter implementation.
//!
//! This module provides a stub implementation of RuntimeAdapter for CoreML models.
//! For MVP, it simulates CoreML inference without requiring the actual CoreML framework.
//!
//! Future versions will integrate with coreml-rs or similar crates to provide
//! real inference capabilities with Metal acceleration.
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_core::runtime_adapter::coreml::CoreMLRuntimeAdapter;
//! use xybrid_core::runtime_adapter::RuntimeAdapter;
//!
//! let mut adapter = CoreMLRuntimeAdapter::new();
//! adapter.load_model("/path/to/model.mlpackage")?;
//! ```

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{
    AdapterError, AdapterResult, ModelMetadata, RuntimeAdapter, RuntimeAdapterExt,
};
use std::collections::HashMap;
use std::path::Path;

/// Mock CoreML Runtime Adapter.
///
/// This adapter simulates CoreML model loading and inference without requiring
/// the actual CoreML framework. It's designed for MVP/testing and will be
/// replaced with a real implementation in future versions.
///
/// # Behavior
///
/// - `load_model()`: Simulates loading by checking file/directory existence and storing metadata
/// - `execute()`: Returns a mock output envelope based on input kind
/// - Supports multiple models loaded simultaneously via `RuntimeAdapterExt`
/// - Metal acceleration detection (stub: returns true for now)
pub struct CoreMLRuntimeAdapter {
    /// Map of loaded models (model_id -> metadata)
    models: HashMap<String, ModelMetadata>,
    /// Currently active model (for simple single-model execution)
    current_model: Option<String>,
    /// Metal acceleration availability (stub: always true for now)
    metal_available: bool,
}

impl CoreMLRuntimeAdapter {
    /// Creates a new CoreML Runtime Adapter instance.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            current_model: None,
            metal_available: Self::detect_metal_availability(),
        }
    }

    /// Detects Metal acceleration availability.
    ///
    /// For MVP, this is a stub that returns true.
    /// Real implementation would check:
    /// - Metal device availability on macOS/iOS
    /// - GPU compute capability
    /// - Memory constraints
    fn detect_metal_availability() -> bool {
        // Stub: Always return true for now
        // TODO: Real implementation would check:
        // - MTLCreateSystemDefaultDevice() != nil
        // - Device supports compute shaders
        // - Sufficient memory available
        true
    }

    /// Returns whether Metal acceleration is available.
    pub fn has_metal(&self) -> bool {
        self.metal_available
    }

    /// Validates that a model file/directory exists and is accessible.
    ///
    /// CoreML models can be:
    /// - `.mlpackage` (directory/bundle)
    /// - `.mlmodel` (single file, deprecated but still supported)
    fn validate_model_file(&self, model_path: &str) -> AdapterResult<()> {
        let path = Path::new(model_path);

        if !path.exists() {
            return Err(AdapterError::ModelNotFound(format!(
                "Model file/directory not found: {}",
                model_path
            )));
        }

        // Check if it's a .mlpackage directory or .mlmodel file
        if path.is_dir() {
            // .mlpackage is a directory bundle
            // Check for required structure (stub: just verify it exists)
            if path.extension().is_none_or(|ext| ext != "mlpackage") {
                // Directory exists but might not be a valid .mlpackage
                // For stub, we'll allow it
            }
        } else if path.is_file() {
            // .mlmodel is a single file
            if let Some(ext) = path.extension() {
                if ext != "mlmodel" && ext != "mlmodelc" {
                    // Warn but don't fail (some models might have different extensions)
                }
            }
        } else {
            return Err(AdapterError::InvalidInput(format!(
                "Path is neither a file nor directory: {}",
                model_path
            )));
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

    /// Simulates inference execution.
    ///
    /// For MVP, this generates mock outputs based on input kind.
    /// Real implementation would:
    /// 1. Convert Envelope to CoreML input format (MLMultiArray, etc.)
    /// 2. Run inference via CoreML framework (with Metal acceleration if available)
    /// 3. Convert CoreML output back to Envelope
    fn simulate_inference(&self, input: &Envelope) -> Envelope {
        // Mock inference: transform input kind to output kind
        // CoreML is commonly used for:
        // - Vision models (image -> classification)
        // - NLP models (text -> embeddings/predictions)
        // - Audio models (audio -> features)
        match &input.kind {
            EnvelopeKind::Audio(_) => {
                // ASR or audio classification
                Envelope::new(EnvelopeKind::Text("coreml-transcribed text".to_string()))
            }
            EnvelopeKind::Text(text) => {
                // NLP model processing
                Envelope::new(EnvelopeKind::Text(format!("coreml-{}-output", text)))
            }
            EnvelopeKind::Embedding(_) => {
                // Embedding similarity or classification
                Envelope::new(EnvelopeKind::Text("coreml-similarity result".to_string()))
            }
        }
    }
}

impl Default for CoreMLRuntimeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for CoreMLRuntimeAdapter {
    fn name(&self) -> &str {
        "coreml"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["mlpackage", "mlmodel", "mlmodelc"]
    }

    fn load_model(&mut self, path: &str) -> AdapterResult<()> {
        // Validate model file/directory exists
        self.validate_model_file(path)?;

        // Extract model ID from path
        let model_id = self.extract_model_id(path);

        // Check if model is already loaded - just log and continue
        if self.models.contains_key(&model_id) {
            log::warn!("Model '{}' is already loaded, skipping reload", model_id);
            return Ok(());
        }

        // Create metadata (stub: in real implementation, would parse CoreML model)
        let metadata = ModelMetadata {
            model_id: model_id.clone(),
            version: "1.0.0".to_string(), // Default version
            runtime_type: "coreml".to_string(),
            model_path: path.to_string(),
            // Mock input/output schemas (real implementation would parse from CoreML)
            input_schema: {
                let mut schema = HashMap::new();
                schema.insert("input".to_string(), vec![1, 1]); // Batch, sequence length
                schema
            },
            output_schema: {
                let mut schema = HashMap::new();
                schema.insert("output".to_string(), vec![1, 1]); // Batch, sequence length
                schema
            },
        };

        self.models.insert(model_id.clone(), metadata);
        self.current_model = Some(model_id);

        Ok(())
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        // Check if a model is loaded
        if self.current_model.is_none() {
            return Err(AdapterError::ModelNotLoaded(
                "No model loaded. Call load_model() first.".to_string(),
            ));
        }

        // Simulate inference execution
        // TODO: In real implementation:
        // 1. Convert Envelope to CoreML input format (MLMultiArray, CVPixelBuffer, etc.)
        // 2. Create MLModel and MLPredictionOptions
        // 3. Run inference: model.prediction(from: input) -> output
        // 4. Convert CoreML output back to Envelope
        // 5. Handle errors from CoreML framework
        // 6. Use Metal acceleration if available (via MLModelConfiguration)

        let output = self.simulate_inference(input);

        Ok(output)
    }
}

impl RuntimeAdapterExt for CoreMLRuntimeAdapter {
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

        // Simulate inference execution
        let output = self.simulate_inference(input);

        Ok(output)
    }

    fn unload_model(&mut self, model_id: &str) -> AdapterResult<()> {
        if !self.models.contains_key(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{}' is not loaded",
                model_id
            )));
        }

        // In real implementation, would:
        // 1. Release MLModel instance
        // 2. Free Metal resources (buffers, command queues)
        // 3. Clean up any cached predictions

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

    fn create_mock_mlpackage() -> (TempDir, String) {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.mlpackage");
        fs::create_dir_all(&model_path).unwrap();
        // Create a minimal .mlpackage structure (stub)
        let manifest_path = model_path.join("manifest.json");
        fs::write(&manifest_path, b"{}").unwrap();
        (temp_dir, model_path.to_string_lossy().to_string())
    }

    fn create_mock_mlmodel() -> (TempDir, String) {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.mlmodel");
        fs::write(&model_path, b"fake coreml model data").unwrap();
        (temp_dir, model_path.to_string_lossy().to_string())
    }

    #[test]
    fn test_create_adapter() {
        let adapter = CoreMLRuntimeAdapter::new();
        assert!(adapter.list_loaded_models().is_empty());
        assert!(adapter.has_metal()); // Stub always returns true
    }

    #[test]
    fn test_adapter_name() {
        let adapter = CoreMLRuntimeAdapter::new();
        assert_eq!(adapter.name(), "coreml");
    }

    #[test]
    fn test_supported_formats() {
        let adapter = CoreMLRuntimeAdapter::new();
        let formats = adapter.supported_formats();
        assert!(formats.contains(&"mlpackage"));
        assert!(formats.contains(&"mlmodel"));
        assert!(formats.contains(&"mlmodelc"));
    }

    #[test]
    fn test_load_mlpackage() -> AdapterResult<()> {
        let (_temp_dir, model_path) = create_mock_mlpackage();
        let mut adapter = CoreMLRuntimeAdapter::new();

        adapter.load_model(&model_path)?;

        let model_id = adapter.extract_model_id(&model_path);
        assert!(adapter.is_loaded(&model_id));

        Ok(())
    }

    #[test]
    fn test_load_mlmodel() -> AdapterResult<()> {
        let (_temp_dir, model_path) = create_mock_mlmodel();
        let mut adapter = CoreMLRuntimeAdapter::new();

        adapter.load_model(&model_path)?;

        let model_id = adapter.extract_model_id(&model_path);
        assert!(adapter.is_loaded(&model_id));

        Ok(())
    }

    #[test]
    fn test_load_model_not_found() {
        let mut adapter = CoreMLRuntimeAdapter::new();
        let result = adapter.load_model("/nonexistent/model.mlpackage");
        assert!(matches!(result, Err(AdapterError::ModelNotFound(_))));
    }

    #[test]
    fn test_execute() -> AdapterResult<()> {
        let (_temp_dir, model_path) = create_mock_mlpackage();
        let mut adapter = CoreMLRuntimeAdapter::new();
        adapter.load_model(&model_path)?;

        let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));

        let output = adapter.execute(&input)?;
        assert_eq!(output.kind_str(), "Text"); // ASR converts audio to text
                                               // Check that output contains "coreml" prefix in the text content
        if let EnvelopeKind::Text(text) = &output.kind {
            assert!(text.contains("coreml"));
        } else {
            panic!("Expected Text output");
        }

        Ok(())
    }

    #[test]
    fn test_execute_no_model_loaded() {
        let adapter = CoreMLRuntimeAdapter::new();
        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = adapter.execute(&input);
        assert!(matches!(result, Err(AdapterError::ModelNotLoaded(_))));
    }

    #[test]
    fn test_infer() -> AdapterResult<()> {
        let (_temp_dir, model_path) = create_mock_mlpackage();
        let mut adapter = CoreMLRuntimeAdapter::new();
        adapter.load_model(&model_path)?;

        let model_id = adapter.extract_model_id(&model_path);
        let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024]));

        let output = adapter.infer(&model_id, &input)?;
        assert_eq!(output.kind_str(), "Text");
        // Check that output contains "coreml" prefix in the text content
        if let EnvelopeKind::Text(text) = &output.kind {
            assert!(text.contains("coreml"));
        } else {
            panic!("Expected Text output");
        }

        Ok(())
    }

    #[test]
    fn test_infer_model_not_loaded() {
        let adapter = CoreMLRuntimeAdapter::new();
        let input = Envelope::new(EnvelopeKind::Text("test".to_string()));

        let result = adapter.infer("nonexistent-model", &input);
        assert!(matches!(result, Err(AdapterError::ModelNotLoaded(_))));
    }

    #[test]
    fn test_unload_model() -> AdapterResult<()> {
        let (_temp_dir, model_path) = create_mock_mlpackage();
        let mut adapter = CoreMLRuntimeAdapter::new();
        adapter.load_model(&model_path)?;

        let model_id = adapter.extract_model_id(&model_path);
        assert!(adapter.is_loaded(&model_id));
        adapter.unload_model(&model_id)?;
        assert!(!adapter.is_loaded(&model_id));

        Ok(())
    }

    #[test]
    fn test_list_loaded_models() -> AdapterResult<()> {
        let temp_dir1 = TempDir::new().unwrap();
        let path1 = temp_dir1.path().join("model1.mlpackage");
        fs::create_dir_all(&path1).unwrap();
        let path1_str = path1.to_string_lossy().to_string();

        let temp_dir2 = TempDir::new().unwrap();
        let path2 = temp_dir2.path().join("model2.mlpackage");
        fs::create_dir_all(&path2).unwrap();
        let path2_str = path2.to_string_lossy().to_string();

        let mut adapter = CoreMLRuntimeAdapter::new();

        adapter.load_model(&path1_str)?;
        adapter.load_model(&path2_str)?;

        let loaded = adapter.list_loaded_models();
        assert_eq!(loaded.len(), 2);
        let id1 = adapter.extract_model_id(&path1_str);
        let id2 = adapter.extract_model_id(&path2_str);
        assert!(loaded.contains(&id1));
        assert!(loaded.contains(&id2));

        Ok(())
    }

    #[test]
    fn test_get_metadata() -> AdapterResult<()> {
        let (_temp_dir, model_path) = create_mock_mlpackage();
        let mut adapter = CoreMLRuntimeAdapter::new();
        adapter.load_model(&model_path)?;

        let model_id = adapter.extract_model_id(&model_path);
        let metadata = adapter.get_metadata(&model_id)?;
        assert_eq!(metadata.model_id, model_id);
        assert_eq!(metadata.runtime_type, "coreml");

        Ok(())
    }

    #[test]
    fn test_double_load_succeeds() -> AdapterResult<()> {
        let (_temp_dir, model_path) = create_mock_mlpackage();
        let mut adapter = CoreMLRuntimeAdapter::new();
        adapter.load_model(&model_path)?;

        // Double load should succeed (idempotent) - just logs a warning
        let result = adapter.load_model(&model_path);
        assert!(result.is_ok());

        Ok(())
    }

    #[test]
    fn test_metal_detection() {
        let adapter = CoreMLRuntimeAdapter::new();
        // Stub always returns true
        assert!(adapter.has_metal());
    }
}
