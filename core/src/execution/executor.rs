//! TemplateExecutor - Main executor implementation.
//!
//! This module contains the `TemplateExecutor` struct and its core execution logic.
//! Preprocessing, postprocessing, and execution mode implementations are delegated
//! to their respective submodules.

use log::{debug, info};

use super::template::{ExecutionMode, ExecutionTemplate, ModelMetadata, PipelineStage};
use crate::ir::Envelope;
use crate::runtime_adapter::onnx::{ONNXSession, OnnxRuntime};
use crate::runtime_adapter::{AdapterError, ModelRuntime};
use crate::tracing as xybrid_trace;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "candle")]
use crate::runtime_adapter::candle::CandleRuntime;

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::runtime_adapter::llm::{LlmConfig, LlmRuntimeAdapter};
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::runtime_adapter::RuntimeAdapter;

use super::modes::{
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
        info!(
            target: "xybrid_core",
            "Executing model: {} v{}",
            metadata.model_id,
            metadata.version
        );
        debug!(
            target: "xybrid_core",
            "Input envelope kind: {}",
            input.kind_str()
        );

        // Start execution span
        let _exec_span = xybrid_trace::SpanGuard::new(format!("execute:{}", metadata.model_id));
        xybrid_trace::add_metadata("model_id", &metadata.model_id);
        xybrid_trace::add_metadata("version", &metadata.version);

        // Step 1: Handling ModelGraph (multi-model DAG)
        if let ExecutionTemplate::ModelGraph { stages, config } = &metadata.execution_template {
            info!(
                target: "xybrid_core",
                "Executing model graph with {} stages",
                stages.len()
            );
            let _span = xybrid_trace::SpanGuard::new("model_graph_inference");
            xybrid_trace::add_metadata("stages", &stages.len().to_string());

            // Run preprocessing
            let preprocessed = self.run_preprocessing(metadata, input)?;

            let raw_outputs = self.execute_pipeline(stages, config, preprocessed, metadata)?;
            return self.run_postprocessing(metadata, raw_outputs);
        }

        // Step 2: Single Model Execution
        let (runtime_type, model_file) = match &metadata.execution_template {
            ExecutionTemplate::SafeTensors { model_file, .. } => ("candle", model_file.clone()),
            ExecutionTemplate::Onnx { model_file } => ("onnx", model_file.clone()),
            ExecutionTemplate::CoreMl { model_file } => ("coreml", model_file.clone()),
            ExecutionTemplate::TfLite { model_file } => ("tflite", model_file.clone()),
            ExecutionTemplate::ModelGraph { .. } => {
                return Err(AdapterError::RuntimeError(
                    "ModelGraph execution should not reach single model path".to_string(),
                ));
            }
            #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
            ExecutionTemplate::Gguf {
                model_file,
                chat_template,
                context_length,
            } => {
                // Extract backend hint from metadata (e.g., "llamacpp" for Gemma 3)
                let backend_hint = metadata
                    .metadata
                    .get("backend")
                    .and_then(|v| v.as_str());

                // LLM execution via LlmRuntimeAdapter
                return self.execute_llm(
                    model_file,
                    chat_template.as_deref(),
                    *context_length,
                    input,
                    backend_hint,
                );
            }
            #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
            ExecutionTemplate::Gguf { .. } => {
                return Err(AdapterError::RuntimeError(
                    "GGUF/LLM execution requires the 'llm-mistral' or 'llm-llamacpp' feature".to_string(),
                ));
            }
        };

        debug!(
            target: "xybrid_core",
            "Using {} runtime with model: {}",
            runtime_type,
            model_file
        );

        // Run Preprocessing
        let preprocessed = self.run_preprocessing(metadata, input)?;

        let model_full_path = Path::new(&self.base_path).join(&model_file);

        // Check if this is TTS with phoneme IDs - needs special handling
        let result_envelope = if preprocessed.is_phoneme_ids() {
            debug!(target: "xybrid_core", "Detected TTS inference (phoneme IDs)");
            // TTS models need phoneme IDs + voice embedding + speed
            let phoneme_ids = preprocessed
                .as_phoneme_ids()
                .ok_or_else(|| AdapterError::InvalidInput("Expected phoneme IDs".to_string()))?;

            // Load voice embedding based on metadata and envelope
            let voice_embedding = self.load_voice_embedding(metadata, input)?;

            // Create and run TTS session directly
            let session = ONNXSession::new(model_full_path.to_str().unwrap(), false, false)?;
            let raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding)?;

            // Convert outputs to envelope
            crate::runtime_adapter::tensor_utils::tensors_to_envelope(
                &raw_outputs,
                session.output_names(),
            )?
        } else if preprocessed.is_token_ids() {
            debug!(target: "xybrid_core", "Detected BERT-style inference (token IDs)");
            // BERT-style models need input_ids, attention_mask, and token_type_ids as int64
            let (ids, attention_mask, token_type_ids) = preprocessed
                .as_token_ids()
                .ok_or_else(|| AdapterError::InvalidInput("Expected token IDs".to_string()))?;

            // Create and run BERT session directly
            let session = ONNXSession::new(model_full_path.to_str().unwrap(), false, false)?;
            let raw_outputs =
                execute_bert_inference(&session, ids, attention_mask, token_type_ids)?;

            // Convert outputs to envelope
            crate::runtime_adapter::tensor_utils::tensors_to_envelope(
                &raw_outputs,
                session.output_names(),
            )?
        } else {
            // Standard execution path
            debug!(target: "xybrid_core", "Using standard execution path");
            let runtime_input = preprocessed.to_envelope()?;

            // Get Runtime & Execute
            let runtime = self.runtimes.get_mut(runtime_type).ok_or_else(|| {
                AdapterError::RuntimeError(format!("Runtime '{}' not configured", runtime_type))
            })?;

            // Ensure model is loaded (runtime handles caching)
            debug!(target: "xybrid_core", "Loading model: {:?}", model_full_path);
            runtime
                .load(&model_full_path)
                .map_err(|e| AdapterError::RuntimeError(format!("Load failed: {}", e)))?;

            debug!(target: "xybrid_core", "Running inference");
            runtime.execute(&runtime_input)?
        };

        // Run Postprocessing
        let raw_outputs = RawOutputs::from_envelope(&result_envelope)?;
        let result = self.run_postprocessing(metadata, raw_outputs)?;

        info!(
            target: "xybrid_core",
            "Model execution complete: {} -> {}",
            metadata.model_id,
            result.kind_str()
        );

        Ok(result)
    }

    /// Execute LLM inference via LlmRuntimeAdapter.
    ///
    /// This is a separate execution path for GGUF-based LLMs that bypasses
    /// the standard preprocessing/inference/postprocessing pipeline.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    fn execute_llm(
        &self,
        model_file: &str,
        chat_template: Option<&str>,
        context_length: usize,
        input: &Envelope,
        backend_hint: Option<&str>,
    ) -> ExecutorResult<Envelope> {
        info!(
            target: "xybrid_core",
            "Executing LLM inference: {} (backend: {:?})",
            model_file,
            backend_hint.unwrap_or("default")
        );

        let _llm_span = xybrid_trace::SpanGuard::new("llm_inference");
        xybrid_trace::add_metadata("model", model_file);
        if let Some(hint) = backend_hint {
            xybrid_trace::add_metadata("backend", hint);
        }

        // Build full model path
        let model_path = Path::new(&self.base_path).join(model_file);

        // Create LLM config
        let mut config = LlmConfig::new(model_path.to_string_lossy().to_string())
            .with_context_length(context_length);

        if let Some(template) = chat_template {
            let template_path = Path::new(&self.base_path).join(template);
            config = config.with_chat_template(template_path.to_string_lossy().to_string());
        }

        // Create adapter with the appropriate backend based on hint
        let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
        adapter.load_model(&config.model_path)?;

        // Execute inference
        let result = adapter.execute(input)?;

        info!(
            target: "xybrid_core",
            "LLM inference complete"
        );

        Ok(result)
    }

    /// Run preprocessing steps from metadata.
    fn run_preprocessing(
        &mut self,
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

            data = preprocessing::apply_preprocessing_step(step, data, input, &self.base_path)?;
        }

        debug!(target: "xybrid_core", "Preprocessing complete");
        Ok(data)
    }

    /// Execute Pipeline: multi-stage execution with control flow.
    fn execute_pipeline(
        &mut self,
        stages: &[PipelineStage],
        config: &HashMap<String, serde_json::Value>,
        initial_input: PreprocessedData,
        _metadata: &ModelMetadata,
    ) -> ExecutorResult<RawOutputs> {
        let mut stage_outputs: HashMap<String, HashMap<String, ArrayD<f32>>> = HashMap::new();
        let mut current_data = initial_input;

        for (idx, stage) in stages.iter().enumerate() {
            debug!(
                target: "xybrid_core",
                "Executing pipeline stage {}/{}: {} ({:?})",
                idx + 1,
                stages.len(),
                stage.name,
                stage.execution_mode
            );

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

            data = postprocessing::apply_postprocessing_step(step, data, &self.base_path)?;
        }

        debug!(target: "xybrid_core", "Postprocessing complete");
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
            runtime.load(&model_full_path).map_err(|e| {
                AdapterError::RuntimeError(format!("Failed to load session: {}", e))
            })?;
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

    /// Load voice embedding based on metadata and input envelope.
    ///
    /// Resolution order:
    /// 1. `voice_id` from Envelope.metadata (if present)
    /// 2. Default voice from ModelMetadata.voices.default
    /// 3. Index 0 (legacy fallback for models without voice config)
    fn load_voice_embedding(
        &self,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Vec<f32>> {
        let loader = crate::tts::voice_embedding::VoiceEmbeddingLoader::new(256);

        // Determine voice file path based on voice config or legacy detection
        let voice_path = if let Some(voice_config) = &metadata.voices {
            // Use path from voice config
            match &voice_config.format {
                super::template::VoiceFormat::Embedded { file, .. } => {
                    Path::new(&self.base_path).join(file)
                }
                _ => {
                    // For non-embedded formats, return error for now
                    return Err(AdapterError::InvalidInput(
                        "Only embedded voice format is currently supported".to_string(),
                    ));
                }
            }
        } else {
            // Legacy: auto-detect voices.bin or voices.npz
            let voices_bin_path = Path::new(&self.base_path).join("voices.bin");
            let voices_npz_path = Path::new(&self.base_path).join("voices.npz");

            if voices_bin_path.exists() {
                voices_bin_path
            } else if voices_npz_path.exists() {
                voices_npz_path
            } else {
                // No voice file - return default embedding
                debug!(target: "xybrid_core", "No voice file found, using zero embedding");
                return Ok(vec![0.0f32; 256]);
            }
        };

        // Check if voice file exists
        if !voice_path.exists() {
            debug!(target: "xybrid_core", "Voice file not found: {:?}, using zero embedding", voice_path);
            return Ok(vec![0.0f32; 256]);
        }

        // Get voice_id from envelope metadata (priority 1)
        let voice_id = input.metadata.get("voice_id");

        // If we have structured voice config, use it for resolution
        if let Some(voice_config) = &metadata.voices {
            let voice_info = if let Some(vid) = voice_id {
                // Look up requested voice
                metadata.get_voice(vid).ok_or_else(|| {
                    let available: Vec<_> = voice_config.catalog.iter().map(|v| v.id.as_str()).collect();
                    AdapterError::InvalidInput(format!(
                        "Voice '{}' not found. Available voices: {:?}",
                        vid, available
                    ))
                })?
            } else {
                // Use default voice
                metadata.default_voice().ok_or_else(|| {
                    AdapterError::RuntimeError(format!(
                        "Default voice '{}' not found in catalog",
                        voice_config.default
                    ))
                })?
            };

            debug!(
                target: "xybrid_core",
                "Loading voice: {} (index: {:?})",
                voice_info.id,
                voice_info.index
            );

            // Determine loader type from voice config
            let is_npz_format = matches!(
                &voice_config.format,
                super::template::VoiceFormat::Embedded {
                    loader: super::template::VoiceLoader::NumpyNpz,
                    ..
                }
            );

            if is_npz_format {
                // For NPZ format, always load by voice ID (name) for reliability.
                // The index field is for documentation/UI; NPZ arrays may not be stored
                // in the same order as catalog indices.
                loader
                    .load_npz_by_name(&voice_path, &voice_info.id, None)
                    .map_err(|e| {
                        AdapterError::RuntimeError(format!(
                            "Failed to load voice '{}' by name: {}",
                            voice_info.id, e
                        ))
                    })
            } else if let Some(index) = voice_info.index {
                // For raw binary format, use index
                loader.load(&voice_path, index).map_err(|e| {
                    AdapterError::RuntimeError(format!(
                        "Failed to load voice '{}' (index {}): {}",
                        voice_info.id, index, e
                    ))
                })
            } else {
                // Fallback: try loading by name (for edge cases)
                loader
                    .load_npz_by_name(&voice_path, &voice_info.id, None)
                    .map_err(|e| {
                        AdapterError::RuntimeError(format!(
                            "Failed to load voice '{}' by name: {}",
                            voice_info.id, e
                        ))
                    })
            }
        } else {
            // Legacy path: no structured voice config
            if let Some(vid) = voice_id {
                // Try parsing as index first
                if let Ok(index) = vid.parse::<usize>() {
                    debug!(target: "xybrid_core", "Loading voice by index: {}", index);
                    loader.load(&voice_path, index)
                } else {
                    // Try as name (NPZ only)
                    debug!(target: "xybrid_core", "Loading voice by name: {}", vid);
                    loader.load_npz_by_name(&voice_path, vid, None)
                }
            } else {
                // Default to index 0
                debug!(target: "xybrid_core", "Loading default voice (index 0)");
                loader.load(&voice_path, 0)
            }
            .map_err(|e| {
                AdapterError::RuntimeError(format!("Failed to load voice embedding: {}", e))
            })
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
