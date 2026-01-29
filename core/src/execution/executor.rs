//! TemplateExecutor - Main executor implementation.
//!
//! This module contains the `TemplateExecutor` struct and its core execution logic.
//! Preprocessing, postprocessing, and execution mode implementations are delegated
//! to their respective submodules.
//!
//! # Runtime Injection
//!
//! The executor supports dependency injection for testability:
//!
//! ```rust,ignore
//! // Default: uses ONNX (and Candle if feature-enabled)
//! let executor = TemplateExecutor::new("models/");
//!
//! // Custom runtime injection for testing:
//! let runtimes = HashMap::new();
//! runtimes.insert("mock".to_string(), Box::new(MockRuntime::new()));
//! let executor = TemplateExecutor::with_runtimes("models/", runtimes);
//! ```

use log::{debug, info};

use super::template::{ExecutionMode, ExecutionTemplate, ModelMetadata, PipelineStage};
use crate::ir::Envelope;
use crate::runtime_adapter::{AdapterError, ModelRuntime};
use crate::tracing as xybrid_trace;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

// Internal: ONNX-specific types needed for optimized execution paths
// These are implementation details, not part of the public API
use crate::runtime_adapter::onnx::{ONNXSession, OnnxRuntime};

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
use super::voice_loader::TtsVoiceLoader;

/// Template Executor implementation.
///
/// Handles execution of models via pluggable runtimes.
///
/// # Runtime Configuration
///
/// The executor can be created with default runtimes or with custom injected runtimes:
///
/// - [`new()`](Self::new) / [`with_base_path()`](Self::with_base_path) - Uses default runtimes (ONNX, Candle if enabled)
/// - [`with_runtimes()`](Self::with_runtimes) - Inject custom runtimes for testing or custom backends
///
/// # Example
///
/// ```rust,ignore
/// // Default usage (recommended)
/// let mut executor = TemplateExecutor::new("models/whisper");
/// let output = executor.execute(&metadata, &input)?;
///
/// // Testing with mock runtime
/// use std::collections::HashMap;
/// let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
/// runtimes.insert("mock".to_string(), Box::new(MockRuntime::new()));
/// let executor = TemplateExecutor::with_runtimes("models/", runtimes);
/// ```
pub struct TemplateExecutor {
    /// Configured runtimes (e.g., "onnx", "candle")
    runtimes: HashMap<String, Box<dyn ModelRuntime>>,
    /// Base path for resolving relative model paths
    base_path: String,
    /// Cached LLM adapter to avoid reloading models between executions.
    /// Stores (model_path, adapter) tuple - reused if model_path matches.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    llm_adapter_cache: Option<(String, LlmRuntimeAdapter)>,
}

impl TemplateExecutor {
    /// Create a new TemplateExecutor with default runtimes.
    ///
    /// Default runtimes:
    /// - `"onnx"` - ONNX Runtime (always available)
    /// - `"candle"` - Candle runtime (when `candle` feature is enabled)
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path for resolving relative model file paths
    pub fn new(base_path: &str) -> Self {
        Self::with_runtimes(base_path, Self::default_runtimes())
    }

    /// Alias for `new` - creates executor with specified base path.
    pub fn with_base_path(base_path: &str) -> Self {
        Self::new(base_path)
    }

    /// Create a TemplateExecutor with custom runtimes.
    ///
    /// Use this for dependency injection in tests or to provide custom runtime implementations.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path for resolving relative model file paths
    /// * `runtimes` - Map of runtime name to runtime implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::collections::HashMap;
    /// use xybrid_core::runtime_adapter::ModelRuntime;
    ///
    /// // Inject a mock runtime for testing
    /// let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
    /// runtimes.insert("onnx".to_string(), Box::new(MockRuntime::new()));
    /// let executor = TemplateExecutor::with_runtimes("models/", runtimes);
    /// ```
    pub fn with_runtimes(
        base_path: &str,
        runtimes: HashMap<String, Box<dyn ModelRuntime>>,
    ) -> Self {
        Self {
            runtimes,
            base_path: base_path.into(),
            #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
            llm_adapter_cache: None,
        }
    }

    /// Create the default set of runtimes based on enabled features.
    ///
    /// This is used by [`new()`](Self::new) and can be called directly
    /// if you want to extend the defaults with additional runtimes.
    pub fn default_runtimes() -> HashMap<String, Box<dyn ModelRuntime>> {
        let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
        runtimes.insert("onnx".to_string(), Box::new(OnnxRuntime::new()));
        #[cfg(feature = "candle")]
        runtimes.insert("candle".to_string(), Box::new(CandleRuntime::new()));
        runtimes
    }

    /// Register an additional runtime.
    ///
    /// Use this to add custom runtimes after construction.
    ///
    /// # Arguments
    ///
    /// * `name` - Runtime identifier (e.g., "custom", "mock")
    /// * `runtime` - Runtime implementation
    pub fn register_runtime(&mut self, name: impl Into<String>, runtime: Box<dyn ModelRuntime>) {
        self.runtimes.insert(name.into(), runtime);
    }

    /// Get a reference to a registered runtime.
    pub fn get_runtime(&self, name: &str) -> Option<&dyn ModelRuntime> {
        self.runtimes.get(name).map(|r| r.as_ref())
    }

    /// List registered runtime names.
    pub fn list_runtimes(&self) -> Vec<&str> {
        self.runtimes.keys().map(|s| s.as_str()).collect()
    }

    /// Execute a model based on its metadata.
    pub fn execute(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope> {
        debug!(
            target: "xybrid_core",
            "TemplateExecutor.execute START: model_id={}, template={:?}",
            metadata.model_id,
            std::mem::discriminant(&metadata.execution_template)
        );
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
                debug!(
                    target: "xybrid_core",
                    "Detected GGUF template, routing to execute_llm()"
                );
                debug!(
                    target: "xybrid_core",
                    "GGUF model_file: {}, chat_template: {:?}, context_length: {}",
                    model_file,
                    chat_template,
                    context_length
                );

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

        let model_full_path = Path::new(&self.base_path).join(&model_file);

        // Check if this is a TTS model - use chunked execution for long text
        let is_tts = Self::is_tts_model(metadata);
        debug!(
            target: "xybrid_core",
            "Checking TTS: is_tts_model={}, preprocessing steps: {:?}",
            is_tts,
            metadata.preprocessing.iter().map(|s| s.step_name()).collect::<Vec<_>>()
        );
        if is_tts {
            debug!(target: "xybrid_core", "TTS detected, calling execute_tts_chunked");
            return self.execute_tts_chunked(metadata, input, &model_full_path);
        }

        // Run Preprocessing for non-TTS models
        let preprocessed = self.run_preprocessing(metadata, input)?;

        // Check if this is BERT-style inference with token IDs
        let result_envelope = if preprocessed.is_token_ids() {
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

    /// Execute a model with streaming support.
    ///
    /// This is similar to `execute()` but calls the provided callback for each
    /// generated token during LLM inference. For non-LLM models, falls back to
    /// regular execution without streaming.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Model metadata with execution configuration
    /// * `input` - Input envelope
    /// * `on_token` - Callback invoked for each generated token
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// executor.execute_streaming(&metadata, &input, Box::new(|token| {
    ///     print!("{}", token.token);
    ///     std::io::stdout().flush()?;
    ///     Ok(())
    /// }))?;
    /// ```
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    pub fn execute_streaming(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        on_token: crate::runtime_adapter::llm::StreamingCallback<'_>,
    ) -> ExecutorResult<Envelope> {
        // Only GGUF (LLM) templates support streaming
        if let super::template::ExecutionTemplate::Gguf {
            model_file,
            chat_template,
            context_length,
        } = &metadata.execution_template
        {
            let backend_hint = metadata
                .metadata
                .get("backend")
                .and_then(|v| v.as_str());

            return self.execute_llm_streaming(
                model_file,
                chat_template.as_deref(),
                *context_length,
                input,
                backend_hint,
                on_token,
            );
        }

        // Non-LLM models: fall back to regular execution
        debug!(
            target: "xybrid_core",
            "execute_streaming: Non-LLM model, falling back to regular execute()"
        );
        self.execute(metadata, input)
    }

    /// Execute a model with streaming support (stub when LLM features disabled).
    ///
    /// Without LLM features, streaming is not available. This stub falls back
    /// to regular execution.
    #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
    pub fn execute_streaming<F>(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        _on_token: F,
    ) -> ExecutorResult<Envelope>
    where
        F: FnMut(()) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        // No LLM support - just use regular execution
        self.execute(metadata, input)
    }

    /// Execute LLM inference with streaming.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    fn execute_llm_streaming(
        &mut self,
        model_file: &str,
        chat_template: Option<&str>,
        context_length: usize,
        input: &Envelope,
        backend_hint: Option<&str>,
        on_token: crate::runtime_adapter::llm::StreamingCallback<'_>,
    ) -> ExecutorResult<Envelope> {
        use crate::ir::EnvelopeKind;
        use crate::runtime_adapter::llm::{ChatMessage, GenerationConfig};

        info!(
            target: "xybrid_core",
            "Executing LLM inference with streaming: {} (backend: {:?})",
            model_file,
            backend_hint.unwrap_or("default")
        );

        let _llm_span = xybrid_trace::SpanGuard::new("llm_inference_streaming");
        xybrid_trace::add_metadata("model", model_file);
        xybrid_trace::add_metadata("streaming", "true");

        // Build full model path
        let model_path = Path::new(&self.base_path).join(model_file);
        let model_path_str = model_path.to_string_lossy().to_string();

        // Check if we have a cached adapter for this model path
        let need_load = match &self.llm_adapter_cache {
            Some((cached_path, _)) if cached_path == &model_path_str => false,
            _ => true,
        };

        // Load model if needed
        if need_load {
            let mut config = LlmConfig::new(model_path_str.clone())
                .with_context_length(context_length);

            if let Some(template) = chat_template {
                let template_path = Path::new(&self.base_path).join(template);
                config = config.with_chat_template(template_path.to_string_lossy().to_string());
            }

            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            adapter.load_model(&config.model_path)?;
            self.llm_adapter_cache = Some((model_path_str.clone(), adapter));
        }

        // Extract prompt from input
        let prompt = match &input.kind {
            EnvelopeKind::Text(text) => text.clone(),
            _ => {
                return Err(AdapterError::InvalidInput(
                    "LLM streaming requires text input".to_string(),
                ))
            }
        };

        // Build messages
        let system_prompt = input.metadata.get("system_prompt").map(|s| s.as_str());
        let mut messages = Vec::new();
        if let Some(sys) = system_prompt {
            messages.push(ChatMessage::system(sys));
        }
        messages.push(ChatMessage::user(&prompt));

        // Parse generation config from metadata
        let mut gen_config = GenerationConfig::default();
        if let Some(max_tokens) = input.metadata.get("max_tokens").and_then(|s| s.parse().ok()) {
            gen_config.max_tokens = max_tokens;
        }
        if let Some(temperature) = input.metadata.get("temperature").and_then(|s| s.parse().ok()) {
            gen_config.temperature = temperature;
        }

        // Execute with streaming
        let output = if let Some((_, adapter)) = &self.llm_adapter_cache {
            adapter.backend().generate_streaming(&messages, &gen_config, on_token)?
        } else {
            return Err(AdapterError::RuntimeError(
                "LLM adapter cache unexpectedly empty".to_string(),
            ));
        };

        // Build response envelope
        let mut response_metadata = std::collections::HashMap::new();
        response_metadata.insert(
            "tokens_generated".to_string(),
            output.tokens_generated.to_string(),
        );
        response_metadata.insert(
            "generation_time_ms".to_string(),
            output.generation_time_ms.to_string(),
        );
        response_metadata.insert(
            "tokens_per_second".to_string(),
            format!("{:.2}", output.tokens_per_second),
        );
        response_metadata.insert("finish_reason".to_string(), output.finish_reason);

        Ok(Envelope {
            kind: EnvelopeKind::Text(output.text),
            metadata: response_metadata,
        })
    }

    /// Execute LLM inference via LlmRuntimeAdapter.
    ///
    /// This is a separate execution path for GGUF-based LLMs that bypasses
    /// the standard preprocessing/inference/postprocessing pipeline.
    ///
    /// The adapter is cached to avoid reloading the model on subsequent calls
    /// with the same model path. This provides significant speedup for REPL
    /// and interactive use cases.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    fn execute_llm(
        &mut self,
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
        let model_path_str = model_path.to_string_lossy().to_string();

        // Check if we have a cached adapter for this model path
        let need_load = match &self.llm_adapter_cache {
            Some((cached_path, _)) if cached_path == &model_path_str => {
                info!(target: "xybrid_core", "Reusing cached LLM adapter for: {}", model_path_str);
                false
            }
            Some((cached_path, _)) => {
                info!(
                    target: "xybrid_core",
                    "Model path changed ({} -> {}), loading new model",
                    cached_path,
                    model_path_str
                );
                true
            }
            None => {
                info!(target: "xybrid_core", "No cached adapter, loading model: {}", model_path_str);
                true
            }
        };

        // Load model if needed (cache miss or different model)
        if need_load {
            // Create LLM config
            let mut config = LlmConfig::new(model_path_str.clone())
                .with_context_length(context_length);

            if let Some(template) = chat_template {
                let template_path = Path::new(&self.base_path).join(template);
                config = config.with_chat_template(template_path.to_string_lossy().to_string());
            }

            // Create adapter with the appropriate backend based on hint
            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            adapter.load_model(&config.model_path)?;

            // Cache the adapter
            self.llm_adapter_cache = Some((model_path_str.clone(), adapter));
        }

        // Execute inference using cached adapter
        let result = if let Some((_, adapter)) = &self.llm_adapter_cache {
            adapter.execute(input)?
        } else {
            // This should never happen, but handle gracefully
            return Err(AdapterError::RuntimeError("LLM adapter cache unexpectedly empty".to_string()));
        };

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

    /// Split text into chunks at sentence boundaries for TTS.
    ///
    /// This ensures each chunk is within the model's token limit while
    /// preserving natural speech breaks.
    fn chunk_text_for_tts(text: &str, max_chars: usize) -> Vec<String> {
        if text.len() <= max_chars {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        // Split into sentences (keep delimiter)
        let sentences: Vec<&str> = text
            .split_inclusive(|c| c == '.' || c == '!' || c == '?')
            .collect();

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            // If single sentence is too long, split at commas or spaces
            if sentence.len() > max_chars {
                // Flush current chunk first
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk = String::new();
                }

                // Split long sentence at commas or spaces
                let mut remaining = sentence;
                while remaining.len() > max_chars {
                    let split_at = remaining[..max_chars]
                        .rfind(|c: char| c == ',' || c.is_whitespace())
                        .unwrap_or(max_chars);
                    chunks.push(remaining[..split_at].trim().to_string());
                    remaining = remaining[split_at..].trim_start_matches(',').trim();
                }
                if !remaining.is_empty() {
                    current_chunk = remaining.to_string();
                }
            } else if current_chunk.len() + sentence.len() + 1 > max_chars {
                // Current chunk would exceed limit, start new chunk
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                }
                current_chunk = sentence.to_string();
            } else {
                // Add to current chunk
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(sentence);
            }
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        chunks
    }

    /// Execute TTS with automatic chunking for long text.
    ///
    /// Splits input text into chunks, processes each through preprocessing + TTS,
    /// and concatenates the audio output.
    fn execute_tts_chunked(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        model_path: &Path,
    ) -> ExecutorResult<Envelope> {
        use crate::ir::EnvelopeKind;

        // Maximum chars per chunk (Kokoro's BERT encoder has ~512 token limit)
        const MAX_TTS_CHARS: usize = 350;

        let text = match &input.kind {
            EnvelopeKind::Text(t) => t.clone(),
            _ => {
                return Err(AdapterError::InvalidInput(
                    "TTS requires text input".to_string(),
                ))
            }
        };

        debug!(
            target: "xybrid_core",
            "TTS Chunked: Input text length: {} chars (MAX_TTS_CHARS={})",
            text.len(),
            MAX_TTS_CHARS
        );

        // Check if chunking is needed
        if text.len() <= MAX_TTS_CHARS {
            debug!(target: "xybrid_core", "TTS: Text is short enough, using single execution");
            // Single chunk - use normal path
            return self.execute_tts_single(metadata, input, model_path);
        }

        debug!(
            target: "xybrid_core",
            "TTS: Text too long ({} chars), splitting into chunks",
            text.len()
        );

        // Split text into chunks
        let chunks = Self::chunk_text_for_tts(&text, MAX_TTS_CHARS);
        debug!(target: "xybrid_core", "TTS: Split into {} chunks", chunks.len());

        // Process each chunk and collect audio
        let mut all_audio: Vec<f32> = Vec::new();
        let session = ONNXSession::new(model_path.to_str().unwrap(), false, false)?;

        for (i, chunk) in chunks.iter().enumerate() {
            debug!(target: "xybrid_core", "TTS: Processing chunk {}/{}: {} chars", i + 1, chunks.len(), chunk.len());

            // Create envelope for this chunk
            let chunk_input = Envelope {
                kind: EnvelopeKind::Text(chunk.clone()),
                metadata: input.metadata.clone(),
            };

            // Run preprocessing on chunk
            let preprocessed = self.run_preprocessing(metadata, &chunk_input)?;

            // Get phoneme IDs
            let phoneme_ids = preprocessed
                .as_phoneme_ids()
                .ok_or_else(|| AdapterError::InvalidInput("Expected phoneme IDs".to_string()))?;

            debug!(target: "xybrid_core", "TTS: Chunk {} has {} phoneme IDs", i + 1, phoneme_ids.len());

            // Load voice embedding (same for all chunks)
            let voice_loader = TtsVoiceLoader::new(&self.base_path);
            let voice_embedding = voice_loader.load(metadata, input)?;

            // Run TTS inference
            let raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding)?;

            // Extract audio from outputs
            if let Some(audio_tensor) = raw_outputs.values().next() {
                let chunk_audio: Vec<f32> = audio_tensor.iter().cloned().collect();
                all_audio.extend(chunk_audio);
            }
        }

        debug!(target: "xybrid_core", "TTS: Total audio samples: {}", all_audio.len());

        // Convert concatenated audio to envelope
        // The postprocessing will handle conversion to bytes
        let output_names = session.output_names();
        let output_name = output_names.first().map(|s| s.as_str()).unwrap_or("audio");

        let mut combined_outputs: HashMap<String, ArrayD<f32>> = HashMap::new();
        let audio_array = ndarray::Array1::from_vec(all_audio).into_dyn();
        combined_outputs.insert(output_name.to_string(), audio_array);

        // Run postprocessing on combined audio
        self.run_postprocessing(metadata, RawOutputs::TensorMap(combined_outputs))
    }

    /// Execute TTS for a single (short) text input.
    fn execute_tts_single(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        model_path: &Path,
    ) -> ExecutorResult<Envelope> {
        // Run preprocessing
        let preprocessed = self.run_preprocessing(metadata, input)?;

        let phoneme_ids = preprocessed
            .as_phoneme_ids()
            .ok_or_else(|| AdapterError::InvalidInput("Expected phoneme IDs".to_string()))?;

        debug!(
            target: "xybrid_core",
            "TTS Single: Input text length: {} chars, first 100: {:?}",
            match &input.kind {
                crate::ir::EnvelopeKind::Text(t) => t.len(),
                _ => 0,
            },
            match &input.kind {
                crate::ir::EnvelopeKind::Text(t) => t.chars().take(100).collect::<String>(),
                _ => "(not text)".to_string(),
            }
        );
        debug!(
            target: "xybrid_core",
            "TTS: Phoneme IDs count: {}, first 20: {:?}",
            phoneme_ids.len(),
            &phoneme_ids[..phoneme_ids.len().min(20)]
        );

        // Load voice embedding
        let voice_loader = TtsVoiceLoader::new(&self.base_path);
        let voice_embedding = voice_loader.load(metadata, input)?;

        // Create and run TTS session
        let session = ONNXSession::new(model_path.to_str().unwrap(), false, false)?;
        let raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding)?;

        // Run postprocessing
        self.run_postprocessing(metadata, RawOutputs::TensorMap(raw_outputs))
    }

    /// Check if this model is a TTS model (has Phonemize preprocessing).
    fn is_tts_model(metadata: &ModelMetadata) -> bool {
        use super::template::PreprocessingStep;
        metadata
            .preprocessing
            .iter()
            .any(|step| matches!(step, PreprocessingStep::Phonemize { .. }))
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
    use super::super::template::PreprocessingStep;

    // ============================================================================
    // Constructor Tests
    // ============================================================================

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

    #[test]
    fn test_resolve_file_path_empty_base() {
        let executor = TemplateExecutor::with_base_path("");
        let resolved = executor.resolve_file_path("encoder.onnx");
        assert_eq!(resolved, "encoder.onnx");
    }

    #[test]
    fn test_default_runtimes_contains_onnx() {
        let runtimes = TemplateExecutor::default_runtimes();
        assert!(runtimes.contains_key("onnx"));
    }

    #[test]
    fn test_with_runtimes_custom_injection() {
        // Create executor with empty runtimes (for testing)
        let runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
        let executor = TemplateExecutor::with_runtimes("/test", runtimes);
        assert_eq!(executor.base_path, "/test");
        assert!(executor.list_runtimes().is_empty());
    }

    #[test]
    fn test_register_runtime() {
        let mut executor = TemplateExecutor::with_runtimes("/test", HashMap::new());
        assert!(executor.list_runtimes().is_empty());

        // Register a runtime
        executor.register_runtime("onnx", Box::new(OnnxRuntime::new()));
        assert!(executor.list_runtimes().contains(&"onnx"));
        assert!(executor.get_runtime("onnx").is_some());
    }

    #[test]
    fn test_list_runtimes() {
        let executor = TemplateExecutor::new("/test");
        let runtimes = executor.list_runtimes();
        assert!(runtimes.contains(&"onnx"));
    }

    #[test]
    fn test_get_runtime_not_found() {
        let executor = TemplateExecutor::new("/test");
        assert!(executor.get_runtime("nonexistent").is_none());
    }

    // ============================================================================
    // chunk_text_for_tts Tests
    // ============================================================================

    #[test]
    fn test_chunk_text_short_input_unchanged() {
        let text = "Hello world.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 350);
        assert_eq!(chunks, vec!["Hello world."]);
    }

    #[test]
    fn test_chunk_text_exactly_at_limit() {
        let text = "A".repeat(350);
        let chunks = TemplateExecutor::chunk_text_for_tts(&text, 350);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 350);
    }

    #[test]
    fn test_chunk_text_splits_at_sentence_boundaries() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 20);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First sentence.");
        assert_eq!(chunks[1], "Second sentence.");
        assert_eq!(chunks[2], "Third sentence.");
    }

    #[test]
    fn test_chunk_text_combines_short_sentences() {
        let text = "Hi. Hello. Hey there.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 50);
        // All sentences fit in one chunk
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hi. Hello. Hey there.");
    }

    #[test]
    fn test_chunk_text_handles_exclamation_and_question() {
        let text = "What? Really! Yes.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 10);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "What?");
        assert_eq!(chunks[1], "Really!");
        assert_eq!(chunks[2], "Yes.");
    }

    #[test]
    fn test_chunk_text_splits_long_sentence_at_comma() {
        let text = "This is a very long sentence, with a comma in the middle, that exceeds the limit.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 40);
        // Should split at commas when sentence exceeds limit
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.len() <= 40));
    }

    #[test]
    fn test_chunk_text_splits_long_sentence_at_space() {
        let text = "Thisisaverylongwordwithoutspaces but then some normal words follow here.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 30);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunk_text_empty_input() {
        let chunks = TemplateExecutor::chunk_text_for_tts("", 350);
        assert!(chunks.is_empty() || chunks == vec![""]);
    }

    #[test]
    fn test_chunk_text_whitespace_only() {
        let chunks = TemplateExecutor::chunk_text_for_tts("   ", 350);
        // Should handle gracefully - either empty or trimmed
        assert!(chunks.is_empty() || chunks.iter().all(|c| c.trim().is_empty()));
    }

    #[test]
    fn test_chunk_text_preserves_content() {
        let text = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 50);
        let rejoined: String = chunks.join(" ");
        // All words should be preserved (though spacing might differ slightly)
        assert!(rejoined.contains("quick"));
        assert!(rejoined.contains("fox"));
        assert!(rejoined.contains("liquor"));
    }

    // ============================================================================
    // is_tts_model Tests
    // ============================================================================

    #[test]
    fn test_is_tts_model_with_phonemize_step() {
        let metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx")
            .with_preprocessing(PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
            });
        assert!(TemplateExecutor::is_tts_model(&metadata));
    }

    #[test]
    fn test_is_tts_model_without_phonemize() {
        let metadata = ModelMetadata::onnx("test-asr", "1.0", "model.onnx")
            .with_preprocessing(PreprocessingStep::AudioDecode {
                sample_rate: 16000,
                channels: 1,
            });
        assert!(!TemplateExecutor::is_tts_model(&metadata));
    }

    #[test]
    fn test_is_tts_model_no_preprocessing() {
        let metadata = ModelMetadata::onnx("test-model", "1.0", "model.onnx");
        assert!(!TemplateExecutor::is_tts_model(&metadata));
    }

    #[test]
    fn test_is_tts_model_phonemize_among_other_steps() {
        let metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx")
            .with_preprocessing(PreprocessingStep::Normalize {
                mean: vec![0.0],
                std: vec![1.0],
            })
            .with_preprocessing(PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
            });
        assert!(TemplateExecutor::is_tts_model(&metadata));
    }

    #[test]
    fn test_is_tts_model_with_mel_spectrogram_is_not_tts() {
        let metadata = ModelMetadata::onnx("test-asr", "1.0", "model.onnx")
            .with_preprocessing(PreprocessingStep::MelSpectrogram {
                preset: Some("whisper".to_string()),
                n_mels: 80,
                sample_rate: 16000,
                fft_size: 400,
                hop_length: 160,
                mel_scale: Default::default(),
                max_frames: Some(3000),
            });
        assert!(!TemplateExecutor::is_tts_model(&metadata));
    }
}
