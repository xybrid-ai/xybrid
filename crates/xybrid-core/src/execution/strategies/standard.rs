//! Standard execution strategy for generic models.
//!
//! Handles single-model execution for ONNX, SafeTensors, CoreML, and TFLite models
//! that don't require specialized execution (like TTS or LLM).

use log::{debug, info};

use super::{ExecutionContext, ExecutionStrategy};
use crate::execution::modes::execute_bert_inference;
use crate::execution::template::{ExecutionTemplate, ModelMetadata, PreprocessingStep};
use crate::execution::types::{ExecutorResult, PreprocessedData, RawOutputs};
use crate::execution::{postprocessing, preprocessing};
use crate::ir::Envelope;
use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use crate::tracing as xybrid_trace;

/// Standard execution strategy for generic models.
///
/// Handles:
/// - ONNX models via OnnxRuntime
/// - SafeTensors models via Candle
/// - CoreML/TFLite models (when runtimes available)
/// - BERT-style token-based inference
pub struct StandardStrategy;

impl StandardStrategy {
    /// Create a new standard strategy.
    pub fn new() -> Self {
        Self
    }

    /// Check if this is a TTS model (should be handled by TtsStrategy).
    fn is_tts_model(metadata: &ModelMetadata) -> bool {
        metadata
            .preprocessing
            .iter()
            .any(|step| matches!(step, PreprocessingStep::Phonemize { .. }))
    }

    /// Check if this is an LLM model (should be handled by LlmStrategy).
    fn is_llm_model(metadata: &ModelMetadata) -> bool {
        matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. })
    }

    /// Check if this is a model graph (should be handled by PipelineStrategy).
    fn is_model_graph(metadata: &ModelMetadata) -> bool {
        matches!(
            metadata.execution_template,
            ExecutionTemplate::ModelGraph { .. }
        )
    }

    /// Get the runtime type and model file from metadata.
    fn get_runtime_info(metadata: &ModelMetadata) -> ExecutorResult<(&'static str, &str)> {
        match &metadata.execution_template {
            ExecutionTemplate::Onnx { model_file } => Ok(("onnx", model_file)),
            ExecutionTemplate::SafeTensors { model_file, .. } => Ok(("candle", model_file)),
            ExecutionTemplate::CoreMl { model_file } => Ok(("coreml", model_file)),
            ExecutionTemplate::TfLite { model_file } => Ok(("tflite", model_file)),
            ExecutionTemplate::ModelGraph { .. } => Err(AdapterError::InvalidInput(
                "ModelGraph should use PipelineStrategy".to_string(),
            )),
            ExecutionTemplate::Gguf { .. } => Err(AdapterError::InvalidInput(
                "GGUF models should use LlmStrategy".to_string(),
            )),
        }
    }

    /// Run preprocessing steps from metadata.
    fn run_preprocessing(
        &self,
        ctx: &ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<PreprocessedData> {
        if metadata.preprocessing.is_empty() {
            debug!(target: "xybrid_core", "No preprocessing steps configured");
            return PreprocessedData::from_envelope(input);
        }

        info!(
            target: "xybrid_core",
            "Running {} preprocessing step(s)",
            metadata.preprocessing.len()
        );

        let _preprocess_span = xybrid_trace::SpanGuard::new("preprocessing");
        xybrid_trace::add_metadata("steps", &metadata.preprocessing.len().to_string());

        let mut data = PreprocessedData::from_envelope(input)?;

        for step in &metadata.preprocessing {
            let step_name = step.step_name();
            debug!(target: "xybrid_core", "Applying preprocessing: {}", step_name);

            let _step_span = xybrid_trace::SpanGuard::new(&format!("preprocessing:{}", step_name));

            data = preprocessing::apply_preprocessing_step(step, data, input, ctx.base_path)?;
        }

        debug!(target: "xybrid_core", "Preprocessing complete");
        Ok(data)
    }

    /// Run postprocessing steps from metadata.
    fn run_postprocessing(
        &self,
        ctx: &ExecutionContext<'_>,
        metadata: &ModelMetadata,
        outputs: RawOutputs,
    ) -> ExecutorResult<Envelope> {
        if metadata.postprocessing.is_empty() {
            debug!(target: "xybrid_core", "No postprocessing steps configured");
            return outputs.to_envelope();
        }

        info!(
            target: "xybrid_core",
            "Running {} postprocessing step(s)",
            metadata.postprocessing.len()
        );

        let _postprocess_span = xybrid_trace::SpanGuard::new("postprocessing");
        xybrid_trace::add_metadata("steps", &metadata.postprocessing.len().to_string());

        let mut data = outputs;

        for step in &metadata.postprocessing {
            let step_name = step.step_name();
            debug!(target: "xybrid_core", "Applying postprocessing: {}", step_name);

            let _step_span = xybrid_trace::SpanGuard::new(&format!("postprocessing:{}", step_name));

            data = postprocessing::apply_postprocessing_step(step, data, ctx.base_path)?;
        }

        debug!(target: "xybrid_core", "Postprocessing complete");
        data.to_envelope()
    }

    /// Execute BERT-style inference with token IDs.
    fn execute_bert(
        &self,
        ctx: &ExecutionContext<'_>,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedData,
        model_path: &std::path::Path,
    ) -> ExecutorResult<Envelope> {
        debug!(target: "xybrid_core", "Detected BERT-style inference (token IDs)");

        let (ids, attention_mask, token_type_ids) = preprocessed
            .as_token_ids()
            .ok_or_else(|| AdapterError::InvalidInput("Expected token IDs".to_string()))?;

        // Create and run BERT session directly
        let session = ONNXSession::new(model_path.to_str().unwrap(), false, false)?;
        let raw_outputs = execute_bert_inference(&session, ids, attention_mask, token_type_ids)?;

        // Convert outputs to envelope
        crate::runtime_adapter::tensor_utils::tensors_to_envelope(
            &raw_outputs,
            session.output_names(),
        )
    }

    /// Execute standard runtime inference.
    fn execute_runtime(
        &self,
        ctx: &mut ExecutionContext<'_>,
        metadata: &ModelMetadata,
        preprocessed: PreprocessedData,
        runtime_type: &str,
        model_path: &std::path::Path,
    ) -> ExecutorResult<Envelope> {
        debug!(target: "xybrid_core", "Using standard execution path");

        let runtime_input = preprocessed.to_envelope()?;

        // Get Runtime & Execute
        let runtime = ctx.get_runtime(runtime_type).ok_or_else(|| {
            AdapterError::RuntimeError(format!("Runtime '{}' not configured", runtime_type))
        })?;

        // Ensure model is loaded (runtime handles caching)
        debug!(target: "xybrid_core", "Loading model: {:?}", model_path);
        runtime
            .load(model_path)
            .map_err(|e| AdapterError::RuntimeError(format!("Load failed: {}", e)))?;

        debug!(target: "xybrid_core", "Running inference");
        runtime.execute(&runtime_input)
    }
}

impl Default for StandardStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionStrategy for StandardStrategy {
    fn can_handle(&self, metadata: &ModelMetadata) -> bool {
        // Don't handle TTS, LLM, or ModelGraph - those have specialized strategies
        !Self::is_tts_model(metadata)
            && !Self::is_llm_model(metadata)
            && !Self::is_model_graph(metadata)
    }

    fn execute(
        &self,
        ctx: &mut ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope> {
        let _span = xybrid_trace::SpanGuard::new("standard_execution");

        let (runtime_type, model_file) = Self::get_runtime_info(metadata)?;
        let model_path = ctx.resolve_path(model_file);

        debug!(
            target: "xybrid_core",
            "Using {} runtime with model: {}",
            runtime_type,
            model_file
        );

        // Run preprocessing
        let preprocessed = self.run_preprocessing(ctx, metadata, input)?;

        // Check if this is BERT-style inference with token IDs
        let result_envelope = if preprocessed.is_token_ids() {
            self.execute_bert(ctx, metadata, &preprocessed, &model_path)?
        } else {
            self.execute_runtime(ctx, metadata, preprocessed, runtime_type, &model_path)?
        };

        // Run postprocessing
        let raw_outputs = RawOutputs::from_envelope(&result_envelope)?;
        let result = self.run_postprocessing(ctx, metadata, raw_outputs)?;

        info!(
            target: "xybrid_core",
            "Model execution complete: {} -> {}",
            metadata.model_id,
            result.kind_str()
        );

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "standard"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_handle_onnx() {
        let strategy = StandardStrategy::new();
        let metadata = ModelMetadata::onnx("test-model", "1.0", "model.onnx");

        assert!(strategy.can_handle(&metadata));
    }

    #[test]
    fn test_cannot_handle_tts() {
        let strategy = StandardStrategy::new();
        let metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx").with_preprocessing(
            PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
            },
        );

        assert!(!strategy.can_handle(&metadata));
    }

    #[test]
    fn test_get_runtime_info_onnx() {
        let metadata = ModelMetadata::onnx("test", "1.0", "model.onnx");
        let (runtime, file) = StandardStrategy::get_runtime_info(&metadata).unwrap();

        assert_eq!(runtime, "onnx");
        assert_eq!(file, "model.onnx");
    }

    #[test]
    fn test_get_runtime_info_safetensors() {
        let metadata = ModelMetadata::safetensors("test", "1.0", "model.safetensors", "whisper");
        let (runtime, file) = StandardStrategy::get_runtime_info(&metadata).unwrap();

        assert_eq!(runtime, "candle");
        assert_eq!(file, "model.safetensors");
    }
}
