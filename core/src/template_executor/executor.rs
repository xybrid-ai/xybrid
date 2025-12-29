//! TemplateExecutor - Main executor implementation.
//!
//! This module contains the `TemplateExecutor` struct and its core execution logic.
//! Preprocessing, postprocessing, and execution mode implementations are delegated
//! to their respective submodules.

use crate::execution_template::{ExecutionMode, ExecutionTemplate, ModelMetadata, PipelineStage};
use crate::ir::Envelope;
use crate::runtime_adapter::onnx::{ONNXSession, OnnxRuntime};
use crate::runtime_adapter::{AdapterError, ModelRuntime};
use crate::tracing as trace;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "candle")]
use crate::runtime_adapter::candle::CandleRuntime;

use super::execution::{
    execute_autoregressive_stage, execute_bert_inference, execute_single_shot_stage,
    execute_tts_inference, execute_whisper_decoder_stage,
};
use super::postprocessing;
use super::preprocessing;
use super::types::{ExecutorResult, PreprocessedData, RawOutputs};

/// Template Executor implementation.
///
/// Handles execution of models via pluggable runtimes.
pub struct TemplateExecutor {
    /// Configured runtimes (e.g., "onnx", "candle")
    runtimes: HashMap<String, Box<dyn ModelRuntime>>,
    /// Base path for resolving relative model paths
    base_path: String,
}

impl TemplateExecutor {
    /// Create a new TemplateExecutor with a base path for resolving relative model paths.
    pub fn new(base_path: &str) -> Self {
        let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
        runtimes.insert("onnx".to_string(), Box::new(OnnxRuntime::new()));
        #[cfg(feature = "candle")]
        runtimes.insert("candle".to_string(), Box::new(CandleRuntime::new()));

        Self {
            runtimes,
            base_path: base_path.into(),
        }
    }

    /// Alias for `new` - creates executor with specified base path.
    pub fn with_base_path(base_path: &str) -> Self {
        Self::new(base_path)
    }

    /// Execute a model based on its metadata.
    pub fn execute(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope> {
        // Start execution span
        let _exec_span = trace::SpanGuard::new(format!("execute:{}", metadata.model_id));
        trace::add_metadata("model_id", &metadata.model_id);
        trace::add_metadata("version", &metadata.version);

        // Step 1: Handling Pipelines
        if let ExecutionTemplate::Pipeline { stages, config } = &metadata.execution_template {
            let _span = trace::SpanGuard::new("pipeline_inference");
            trace::add_metadata("stages", &stages.len().to_string());

            // Run preprocessing
            let preprocessed = self.run_preprocessing(metadata, input)?;

            let raw_outputs = self.execute_pipeline(stages, config, preprocessed, metadata)?;
            return self.run_postprocessing(metadata, raw_outputs);
        }

        // Step 2: Single Model Execution
        let (runtime_type, model_file) = match &metadata.execution_template {
            ExecutionTemplate::CandleModel { model_file, .. } => ("candle", model_file.clone()),
            ExecutionTemplate::SimpleMode { model_file } => ("onnx", model_file.clone()),
            ExecutionTemplate::Pipeline { .. } => {
                return Err(AdapterError::RuntimeError(
                    "Pipeline execution should not reach single model path".to_string(),
                ));
            }
        };

        // Run Preprocessing
        let preprocessed = self.run_preprocessing(metadata, input)?;

        let model_full_path = Path::new(&self.base_path).join(&model_file);

        // Check if this is TTS with phoneme IDs - needs special handling
        let result_envelope = if preprocessed.is_phoneme_ids() {
            // TTS models need phoneme IDs + voice embedding + speed
            let phoneme_ids = preprocessed.as_phoneme_ids().ok_or_else(|| {
                AdapterError::InvalidInput("Expected phoneme IDs".to_string())
            })?;

            // Load voice embedding from voices.bin
            let voices_path = Path::new(&self.base_path).join("voices.bin");
            let voice_embedding = if voices_path.exists() {
                let loader = crate::tts::voice_embedding::VoiceEmbeddingLoader::new(256);
                loader.load(&voices_path, 0).map_err(|e| {
                    AdapterError::RuntimeError(format!("Failed to load voice embedding: {}", e))
                })?
            } else {
                // Default voice embedding (zeros)
                vec![0.0f32; 256]
            };

            // Create and run TTS session directly
            let session = ONNXSession::new(model_full_path.to_str().unwrap(), false, false)?;
            let raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding)?;

            // Convert outputs to envelope
            crate::runtime_adapter::tensor_utils::tensors_to_envelope(&raw_outputs, session.output_names())?
        } else if preprocessed.is_token_ids() {
            // BERT-style models need input_ids, attention_mask, and token_type_ids as int64
            let (ids, attention_mask, token_type_ids) = preprocessed.as_token_ids().ok_or_else(|| {
                AdapterError::InvalidInput("Expected token IDs".to_string())
            })?;

            // Create and run BERT session directly
            let session = ONNXSession::new(model_full_path.to_str().unwrap(), false, false)?;
            let raw_outputs = execute_bert_inference(&session, ids, attention_mask, token_type_ids)?;

            // Convert outputs to envelope
            crate::runtime_adapter::tensor_utils::tensors_to_envelope(&raw_outputs, session.output_names())?
        } else {
            // Standard execution path
            let runtime_input = preprocessed.to_envelope()?;

            // Get Runtime & Execute
            let runtime = self.runtimes.get_mut(runtime_type).ok_or_else(|| {
                AdapterError::RuntimeError(format!("Runtime '{}' not configured", runtime_type))
            })?;

            // Ensure model is loaded (runtime handles caching)
            runtime
                .load(&model_full_path)
                .map_err(|e| AdapterError::RuntimeError(format!("Load failed: {}", e)))?;

            runtime.execute(&runtime_input)?
        };

        // Run Postprocessing
        let raw_outputs = RawOutputs::from_envelope(&result_envelope)?;
        self.run_postprocessing(metadata, raw_outputs)
    }

    /// Run preprocessing steps from metadata.
    fn run_preprocessing(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<PreprocessedData> {
        if metadata.preprocessing.is_empty() {
            return PreprocessedData::from_envelope(input);
        }

        let _preprocess_span = trace::SpanGuard::new("preprocessing");
        trace::add_metadata("steps", &metadata.preprocessing.len().to_string());

        let mut data = PreprocessedData::from_envelope(input)?;

        for step in &metadata.preprocessing {
            let step_name = format!("preprocessing:{}", step.step_name());
            let _step_span = trace::SpanGuard::new(&step_name);

            data = preprocessing::apply_preprocessing_step(step, data, input, &self.base_path)?;
        }

        Ok(data)
    }

    /// Execute Pipeline: multi-stage execution with control flow.
    fn execute_pipeline(
        &mut self,
        stages: &[PipelineStage],
        config: &HashMap<String, serde_json::Value>,
        initial_input: PreprocessedData,
        metadata: &ModelMetadata,
    ) -> ExecutorResult<RawOutputs> {
        let mut stage_outputs: HashMap<String, HashMap<String, ArrayD<f32>>> = HashMap::new();
        let mut current_data = initial_input;

        for stage in stages {
            match &stage.execution_mode {
                ExecutionMode::SingleShot => {
                    let runtime = self.runtimes.get_mut("onnx").ok_or_else(|| {
                        AdapterError::RuntimeError("ONNX runtime not configured".to_string())
                    })?;

                    let outputs = execute_single_shot_stage(
                        stage,
                        &current_data,
                        &stage_outputs,
                        runtime.as_mut(),
                        &self.base_path,
                    )?;
                    stage_outputs.insert(stage.name.clone(), outputs.clone());

                    if let Some(first_output) = outputs.values().next() {
                        current_data = PreprocessedData::Tensor(first_output.clone());
                    }
                }

                ExecutionMode::Autoregressive {
                    max_tokens,
                    start_token_id,
                    end_token_id,
                    repetition_penalty,
                } => {
                    let session = self.get_or_load_session(&stage.model_file)?;
                    let token_ids = execute_autoregressive_stage(
                        stage,
                        &stage_outputs,
                        config,
                        *max_tokens,
                        *start_token_id,
                        *end_token_id,
                        *repetition_penalty,
                        session,
                    )?;

                    return Ok(RawOutputs::TokenIds(token_ids));
                }

                ExecutionMode::IterativeRefinement { num_steps, .. } => {
                    return Err(AdapterError::InvalidInput(format!(
                        "IterativeRefinement not yet implemented (needs {} steps)",
                        num_steps
                    )));
                }

                ExecutionMode::WhisperDecoder {
                    max_tokens,
                    start_token_id,
                    end_token_id,
                    language_token_id,
                    task_token_id,
                    no_timestamps_token_id,
                    suppress_tokens,
                    repetition_penalty,
                } => {
                    let session = self.get_or_load_session(&stage.model_file)?;
                    let token_ids = execute_whisper_decoder_stage(
                        stage,
                        &stage_outputs,
                        config,
                        *max_tokens,
                        *start_token_id,
                        *end_token_id,
                        *language_token_id,
                        *task_token_id,
                        *no_timestamps_token_id,
                        suppress_tokens,
                        *repetition_penalty,
                        session,
                    )?;

                    return Ok(RawOutputs::TokenIds(token_ids));
                }
            }
        }

        // Return the last stage's outputs
        if let Some((_, outputs)) = stage_outputs.iter().last() {
            Ok(RawOutputs::TensorMap(outputs.clone()))
        } else {
            Err(AdapterError::InvalidInput(
                "Pipeline produced no outputs".to_string(),
            ))
        }
    }

    /// Run postprocessing steps from metadata.
    fn run_postprocessing(
        &mut self,
        metadata: &ModelMetadata,
        outputs: RawOutputs,
    ) -> ExecutorResult<Envelope> {
        if metadata.postprocessing.is_empty() {
            return outputs.to_envelope();
        }

        let _postprocess_span = trace::SpanGuard::new("postprocessing");
        trace::add_metadata("steps", &metadata.postprocessing.len().to_string());

        let mut data = outputs;

        for step in &metadata.postprocessing {
            let step_name = format!("postprocessing:{}", step.step_name());
            let _step_span = trace::SpanGuard::new(&step_name);

            data = postprocessing::apply_postprocessing_step(step, data, &self.base_path)?;
        }

        data.to_envelope()
    }

    /// Get or load an ONNX session.
    fn get_or_load_session(&mut self, model_file: &str) -> ExecutorResult<&ONNXSession> {
        let model_full_path = Path::new(&self.base_path).join(model_file);

        // Load the model
        {
            let runtime = self.runtimes.get_mut("onnx").ok_or_else(|| {
                AdapterError::RuntimeError("ONNX runtime not configured".to_string())
            })?;
            runtime
                .load(&model_full_path)
                .map_err(|e| AdapterError::RuntimeError(format!("Failed to load session: {}", e)))?;
        }

        // Get session (immutable borrow)
        let runtime = self.runtimes.get("onnx").unwrap();
        if let Some(onnx_rt) = runtime.as_any().downcast_ref::<OnnxRuntime>() {
            let path_str = model_full_path.to_string_lossy();
            onnx_rt.get_session(&path_str)
        } else {
            Err(AdapterError::RuntimeError(
                "Runtime 'onnx' is not OnnxRuntime".to_string(),
            ))
        }
    }

    /// Resolve a file path relative to base_path.
    pub fn resolve_file_path(&self, file: &str) -> String {
        if self.base_path.is_empty() {
            file.to_string()
        } else {
            Path::new(&self.base_path)
                .join(file)
                .to_string_lossy()
                .to_string()
        }
    }
}

impl Default for TemplateExecutor {
    fn default() -> Self {
        Self::new("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = TemplateExecutor::default();
        assert_eq!(executor.base_path, "");
    }

    #[test]
    fn test_executor_with_base_path() {
        let executor = TemplateExecutor::with_base_path("/path/to/models");
        assert_eq!(executor.base_path, "/path/to/models");
    }

    #[test]
    fn test_resolve_file_path() {
        let executor = TemplateExecutor::with_base_path("/models");
        let resolved = executor.resolve_file_path("encoder.onnx");
        assert!(resolved.contains("encoder.onnx"));
    }
}
