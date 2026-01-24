//! Metadata-driven execution integration for runtime adapters.
//!
//! This module provides functionality to execute models based on ModelMetadata
//! (from execution_template) rather than hard-coded logic.

use crate::execution::{ModelMetadata as ExecutionModelMetadata, TemplateExecutor};
use crate::ir::Envelope;
use crate::runtime_adapter::{AdapterError, AdapterResult};
use std::path::{Path, PathBuf};

/// Metadata-driven executor wrapper for runtime adapters.
///
/// This wrapper allows runtime adapters to execute models using metadata
/// from JSON files rather than hard-coded execution logic.
pub struct MetadataDrivenAdapter {
    /// Template executor for metadata interpretation
    executor: TemplateExecutor,

    /// Currently loaded metadata (if any)
    current_metadata: Option<ExecutionModelMetadata>,

    /// Base path for resolving model files
    base_path: PathBuf,
}

impl MetadataDrivenAdapter {
    /// Create a new metadata-driven adapter
    pub fn new() -> Self {
        Self {
            executor: TemplateExecutor::new("."),
            current_metadata: None,
            base_path: PathBuf::from("."),
        }
    }

    /// Create a new metadata-driven adapter with a base path
    pub fn with_base_path(base_path: impl Into<PathBuf>) -> Self {
        let base_path_buf = base_path.into();
        Self {
            executor: TemplateExecutor::with_base_path(&base_path_buf.to_string_lossy()),
            current_metadata: None,
            base_path: base_path_buf,
        }
    }

    /// Load model metadata from a JSON file
    ///
    /// # Arguments
    ///
    /// * `metadata_path` - Path to model_metadata.json file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use xybrid_core::runtime_adapter::metadata_driven::MetadataDrivenAdapter;
    /// let mut adapter = MetadataDrivenAdapter::new();
    /// adapter.load_metadata("models/whisper-tiny/model_metadata.json")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn load_metadata(&mut self, metadata_path: impl AsRef<Path>) -> AdapterResult<()> {
        let path = metadata_path.as_ref();

        // Read metadata file
        let metadata_json = std::fs::read_to_string(path).map_err(|e| {
            AdapterError::ModelNotFound(format!(
                "Failed to read metadata file '{}': {}",
                path.display(),
                e
            ))
        })?;

        // Parse metadata
        let metadata: ExecutionModelMetadata =
            serde_json::from_str(&metadata_json).map_err(|e| {
                AdapterError::InvalidInput(format!("Failed to parse metadata JSON: {}", e))
            })?;

        // Update base path to metadata file's directory
        if let Some(parent) = path.parent() {
            self.base_path = parent.to_path_buf();
            self.executor = TemplateExecutor::with_base_path(&parent.to_string_lossy());
        }

        self.current_metadata = Some(metadata);

        Ok(())
    }

    /// Load model metadata from a JSON string
    pub fn load_metadata_from_json(&mut self, json: &str) -> AdapterResult<()> {
        let metadata: ExecutionModelMetadata = serde_json::from_str(json).map_err(|e| {
            AdapterError::InvalidInput(format!("Failed to parse metadata JSON: {}", e))
        })?;

        self.current_metadata = Some(metadata);

        Ok(())
    }

    /// Execute model using loaded metadata
    ///
    /// # Arguments
    ///
    /// * `input` - Input envelope
    ///
    /// # Returns
    ///
    /// Output envelope from model execution
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No metadata is loaded
    /// - Model files cannot be found
    /// - Inference fails
    pub fn execute(&mut self, input: &Envelope) -> AdapterResult<Envelope> {
        let metadata = self.current_metadata.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded(
                "No metadata loaded. Call load_metadata() first.".to_string(),
            )
        })?;

        self.executor.execute(metadata, input)
    }

    /// Get the currently loaded metadata (if any)
    pub fn current_metadata(&self) -> Option<&ExecutionModelMetadata> {
        self.current_metadata.as_ref()
    }

    /// Check if metadata is loaded
    pub fn is_loaded(&self) -> bool {
        self.current_metadata.is_some()
    }

    /// Get the base path for model file resolution
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }
}

impl Default for MetadataDrivenAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = MetadataDrivenAdapter::new();
        assert!(!adapter.is_loaded());
    }

    #[test]
    fn test_adapter_with_base_path() {
        let adapter = MetadataDrivenAdapter::with_base_path("/models");
        assert_eq!(adapter.base_path(), Path::new("/models"));
    }

    #[test]
    fn test_load_metadata_from_json() {
        let mut adapter = MetadataDrivenAdapter::new();

        let json = r#"{
            "model_id": "test-model",
            "version": "1.0",
            "execution_template": {
                "type": "Onnx",
                "model_file": "model.onnx"
            },
            "preprocessing": [],
            "postprocessing": [],
            "files": ["model.onnx"]
        }"#;

        adapter.load_metadata_from_json(json).unwrap();
        assert!(adapter.is_loaded());

        let metadata = adapter.current_metadata().unwrap();
        assert_eq!(metadata.model_id, "test-model");
        assert_eq!(metadata.version, "1.0");
    }
}
