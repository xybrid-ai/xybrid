//! Pipeline FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::sync::Arc;
use xybrid_sdk::{Pipeline, PipelineRef};

use super::envelope::FfiEnvelope;
use super::result::FfiResult;

/// FFI wrapper for a loaded Pipeline ready for execution.
#[frb(opaque)]
pub struct FfiPipeline(Arc<Pipeline>);

impl FfiPipeline {
    /// Load a pipeline from a YAML string.
    ///
    /// Parses the YAML and resolves all model references via the registry.
    #[frb(sync)]
    pub fn from_yaml(yaml: String) -> Result<FfiPipeline, String> {
        let pipeline_ref = PipelineRef::from_yaml(&yaml).map_err(|e| e.to_string())?;
        let pipeline = pipeline_ref.load().map_err(|e| e.to_string())?;
        Ok(FfiPipeline(Arc::new(pipeline)))
    }

    /// Load a pipeline from a YAML file path.
    ///
    /// Reads the file, parses the YAML, and resolves all model references.
    #[frb(sync)]
    pub fn from_file(path: String) -> Result<FfiPipeline, String> {
        let pipeline_ref = PipelineRef::from_file(&path).map_err(|e| e.to_string())?;
        let pipeline = pipeline_ref.load().map_err(|e| e.to_string())?;
        Ok(FfiPipeline(Arc::new(pipeline)))
    }

    /// Load a pipeline from a bundle path.
    ///
    /// Note: Currently delegates to from_file as bundles are YAML-based.
    /// In future, this may support additional bundle formats (e.g., .xpkg).
    #[frb(sync)]
    pub fn from_bundle(path: String) -> Result<FfiPipeline, String> {
        // Bundle format is currently just a YAML file
        Self::from_file(path)
    }

    /// Execute the pipeline with the given input envelope.
    ///
    /// Returns the inference result from the final stage.
    pub fn run(&self, envelope: FfiEnvelope) -> Result<FfiResult, String> {
        let input = envelope.into_envelope();
        let result = self.0.run(&input).map_err(|e| e.to_string())?;

        // Convert PipelineExecutionResult to FfiResult
        Ok(FfiResult {
            success: true,
            text: result.text().map(|s| s.to_string()),
            audio_bytes: result.audio_bytes().map(|b| b.to_vec()),
            embedding: result.embedding().map(|e| e.to_vec()),
            latency_ms: result.total_latency_ms,
        })
    }

    /// Get the pipeline name (if specified in YAML).
    #[frb(sync)]
    pub fn name(&self) -> Option<String> {
        self.0.name().map(|s| s.to_string())
    }

    /// Get the stage names/identifiers in execution order.
    #[frb(sync)]
    pub fn stage_names(&self) -> Vec<String> {
        self.0.stage_names()
    }

    /// Get the number of stages in the pipeline.
    #[frb(sync)]
    pub fn stage_count(&self) -> usize {
        self.0.stage_count()
    }
}
