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

use log::{debug, info, warn};

use super::template::{ExecutionMode, ExecutionTemplate, ModelMetadata, PipelineStage};
use crate::conversation::ConversationContext;
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::ir::EnvelopeKind;
use crate::ir::{Envelope, MessageRole};
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

// Always-available LLM types (defined in runtime_adapter/types.rs)
use crate::runtime_adapter::types::StreamingCallback;
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::runtime_adapter::types::{ChatMessage, GenerationConfig, LlmConfig};

// LLM adapter implementation (only available with LLM features)
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::runtime_adapter::llm::LlmRuntimeAdapter;
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
    /// This field always exists but is only populated when LLM features are enabled.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    llm_adapter_cache: Option<(String, LlmRuntimeAdapter)>,
    /// Placeholder for llm_adapter_cache when LLM features are disabled.
    /// This ensures the struct has consistent fields regardless of features.
    #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
    llm_adapter_cache: Option<()>,
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
            xybrid_trace::add_metadata("stages", stages.len().to_string());

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
                let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());

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
                    "GGUF/LLM execution requires the 'llm-mistral' or 'llm-llamacpp' feature"
                        .to_string(),
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

    /// Execute a model with conversation context.
    ///
    /// For LLM models (GGUF), this builds the full prompt from the conversation
    /// context plus the current input envelope using the appropriate chat template.
    /// The result envelope is automatically tagged with `MessageRole::Assistant`.
    ///
    /// For non-LLM models, the context is passed through transparently and the
    /// model receives its normal input (the context is available but not consumed
    /// by the model execution).
    ///
    /// # Arguments
    ///
    /// * `metadata` - Model metadata with execution configuration
    /// * `input` - Current input envelope (typically a user message)
    /// * `context` - Conversation context containing history and optional system prompt
    ///
    /// # Important: Context Update Pattern
    ///
    /// **Do NOT push the input to context before calling this method!**
    /// The input is automatically appended to the context when building the prompt.
    /// Push both input and result to context **after** execution for the next turn:
    ///
    /// ```rust,ignore
    /// // CORRECT: Push AFTER execution
    /// let input = Envelope::new(EnvelopeKind::Text("Hello".into()))
    ///     .with_role(MessageRole::User);
    /// let result = executor.execute_with_context(&metadata, &input, &ctx)?;
    /// ctx.push(input);   // Push for next turn
    /// ctx.push(result);  // Push for next turn
    ///
    /// // WRONG: Pushing before causes duplicate messages!
    /// ctx.push(input.clone());  // DON'T DO THIS
    /// let result = executor.execute_with_context(&metadata, &input, &ctx)?;
    /// ```
    ///
    /// A runtime warning is logged if the input is already in context (detected by local_id).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xybrid_core::execution::TemplateExecutor;
    /// use xybrid_core::conversation::ConversationContext;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let mut ctx = ConversationContext::new()
    ///     .with_system(Envelope::new(EnvelopeKind::Text("You are helpful.".into()))
    ///         .with_role(MessageRole::System));
    ///
    /// // Previous conversation turns (already happened)
    /// ctx.push(Envelope::new(EnvelopeKind::Text("Hello".into()))
    ///     .with_role(MessageRole::User));
    /// ctx.push(Envelope::new(EnvelopeKind::Text("Hi there!".into()))
    ///     .with_role(MessageRole::Assistant));
    ///
    /// // Current turn - don't push input before execution
    /// let input = Envelope::new(EnvelopeKind::Text("How are you?".into()))
    ///     .with_role(MessageRole::User);
    ///
    /// let result = executor.execute_with_context(&metadata, &input, &ctx)?;
    /// assert!(result.is_assistant_message());
    ///
    /// // Update context for next turn
    /// ctx.push(input);
    /// ctx.push(result);
    /// ```
    pub fn execute_with_context(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        context: &ConversationContext,
    ) -> ExecutorResult<Envelope> {
        debug!(
            target: "xybrid_core",
            "TemplateExecutor.execute_with_context START: model_id={}, context_id={}",
            metadata.model_id,
            context.id()
        );

        // Warn if input was already pushed to context (common mistake)
        // This causes the input message to appear twice in the prompt
        if let Some(last) = context.history().last() {
            if last.local_id() == input.local_id() {
                warn!(
                    target: "xybrid_core",
                    "Input envelope was already pushed to context (local_id={}). \
                     This will cause the message to appear twice in the prompt. \
                     Push input to context AFTER execute_with_context, not before.",
                    input.local_id()
                );
            }
        }

        // Check if this is a GGUF (LLM) model
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        if let ExecutionTemplate::Gguf {
            model_file,
            context_length,
            ..
        } = &metadata.execution_template
        {
            debug!(
                target: "xybrid_core",
                "LLM model detected, converting context to ChatMessages"
            );

            // Convert ConversationContext + input to ChatMessages directly.
            // This avoids double-formatting — we let the LLM backend (llama.cpp)
            // apply its native chat template once, rather than formatting in Rust
            // and then having the backend format again.
            let mut chat_messages: Vec<ChatMessage> = Vec::new();

            // Add context messages (system + history)
            for envelope in context.context_for_llm() {
                if let EnvelopeKind::Text(text) = &envelope.kind {
                    let role = envelope.role().unwrap_or(MessageRole::User);
                    chat_messages.push(ChatMessage {
                        role,
                        content: text.clone(),
                    });
                }
            }

            // Add current input
            if let EnvelopeKind::Text(text) = &input.kind {
                let role = input.role().unwrap_or(MessageRole::User);
                chat_messages.push(ChatMessage {
                    role,
                    content: text.clone(),
                });
            }

            debug!(
                target: "xybrid_core",
                "Converted {} messages for LLM",
                chat_messages.len()
            );

            let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());

            let mut result = self.execute_llm_with_messages(
                model_file,
                *context_length,
                &chat_messages,
                backend_hint,
            )?;

            // Tag the result as an assistant message
            result = result.with_role(MessageRole::Assistant);

            return Ok(result);
        }

        // For non-LLM models, execute normally (context is available but not consumed)
        debug!(
            target: "xybrid_core",
            "Non-LLM model, executing without context transformation"
        );
        let mut result = self.execute(metadata, input)?;

        // Tag the result as an assistant message
        result = result.with_role(MessageRole::Assistant);

        Ok(result)
    }

    /// Execute a model with streaming support.
    ///
    /// This is similar to `execute()` but calls the provided callback for each
    /// generated token during LLM inference. For non-LLM models, falls back to
    /// regular execution without streaming.
    ///
    /// **Note**: This method signature is always available, but streaming only
    /// works when the `llm-mistral` or `llm-llamacpp` feature is enabled.
    /// Without these features, the callback is ignored and regular execution
    /// is used.
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
    pub fn execute_streaming(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        #[allow(unused_variables)] on_token: StreamingCallback<'_>,
    ) -> ExecutorResult<Envelope> {
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        {
            // Only GGUF (LLM) templates support streaming
            if let super::template::ExecutionTemplate::Gguf {
                model_file,
                chat_template,
                context_length,
            } = &metadata.execution_template
            {
                let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());

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
        }

        #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
        {
            debug!(
                target: "xybrid_core",
                "execute_streaming: LLM features not enabled, falling back to regular execute()"
            );
        }

        self.execute(metadata, input)
    }

    /// Execute a model with streaming and conversation context.
    ///
    /// Combines streaming execution with conversation history management.
    /// The context provides previous messages which are formatted into the prompt
    /// before streaming inference begins.
    ///
    /// **Note**: This method signature is always available, but streaming only
    /// works when the `llm-mistral` or `llm-llamacpp` feature is enabled.
    /// Without these features, the callback is ignored and regular execution
    /// with context is used.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Model metadata with execution configuration
    /// * `input` - Current user input envelope
    /// * `context` - Conversation history for multi-turn chat
    /// * `on_token` - Callback invoked for each generated token
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut ctx = ConversationContext::new();
    /// ctx.push(Envelope::new(EnvelopeKind::Text("Hello!".into())).with_role(MessageRole::User));
    ///
    /// executor.execute_streaming_with_context(&metadata, &input, &ctx, Box::new(|token| {
    ///     print!("{}", token.token);
    ///     std::io::stdout().flush()?;
    ///     Ok(())
    /// }))?;
    /// ```
    #[allow(unused_variables)]
    pub fn execute_streaming_with_context(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        context: &ConversationContext,
        on_token: StreamingCallback<'_>,
    ) -> ExecutorResult<Envelope> {
        debug!(
            target: "xybrid_core",
            "TemplateExecutor.execute_streaming_with_context START: model_id={}, context_id={}",
            metadata.model_id,
            context.id()
        );

        // Warn if input was already pushed to context (common mistake)
        if let Some(last) = context.history().last() {
            if last.local_id() == input.local_id() {
                warn!(
                    target: "xybrid_core",
                    "Input envelope was already pushed to context (local_id={}). \
                     This will cause the message to appear twice in the prompt. \
                     Push input to context AFTER execute_streaming_with_context, not before.",
                    input.local_id()
                );
            }
        }

        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        {
            // Check if this is a GGUF (LLM) model
            if let ExecutionTemplate::Gguf {
                model_file,
                context_length,
                ..
            } = &metadata.execution_template
            {
                debug!(
                    target: "xybrid_core",
                    "LLM model detected, converting context to ChatMessages for streaming"
                );

                // Convert ConversationContext + input to ChatMessages
                // This avoids double-formatting - we let llama.cpp apply its native template
                let mut chat_messages: Vec<ChatMessage> = Vec::new();

                // Add context messages (system + history)
                for envelope in context.context_for_llm() {
                    if let EnvelopeKind::Text(text) = &envelope.kind {
                        let role = envelope.role().unwrap_or(MessageRole::User);
                        chat_messages.push(ChatMessage {
                            role,
                            content: text.clone(),
                        });
                    }
                }

                // Add current input
                if let EnvelopeKind::Text(text) = &input.kind {
                    let role = input.role().unwrap_or(MessageRole::User);
                    chat_messages.push(ChatMessage {
                        role,
                        content: text.clone(),
                    });
                }

                debug!(
                    target: "xybrid_core",
                    "Converted {} messages for LLM",
                    chat_messages.len()
                );

                // Execute streaming with ChatMessages directly
                let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());

                let result = self.execute_llm_streaming_with_messages(
                    model_file,
                    *context_length,
                    &chat_messages,
                    backend_hint,
                    on_token,
                )?;

                // Tag the result as an assistant message
                let result = result.with_role(MessageRole::Assistant);

                return Ok(result);
            }

            // For non-LLM models, execute normally with streaming fallback
            debug!(
                target: "xybrid_core",
                "Non-LLM model, executing streaming without context transformation"
            );
            let mut result = self.execute_streaming(metadata, input, on_token)?;
            result = result.with_role(MessageRole::Assistant);

            Ok(result)
        }

        #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
        {
            debug!(
                target: "xybrid_core",
                "execute_streaming_with_context: LLM features not enabled, using execute_with_context()"
            );
            // No LLM support - just use regular execution with context
            self.execute_with_context(metadata, input, context)
        }
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
        on_token: StreamingCallback<'_>,
    ) -> ExecutorResult<Envelope> {
        // ChatMessage, GenerationConfig, LlmConfig are imported at module level from types

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
            let mut config =
                LlmConfig::new(model_path_str.clone()).with_context_length(context_length);

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
        if let Some(max_tokens) = input
            .metadata
            .get("max_tokens")
            .and_then(|s| s.parse().ok())
        {
            gen_config.max_tokens = max_tokens;
        }
        if let Some(temperature) = input
            .metadata
            .get("temperature")
            .and_then(|s| s.parse().ok())
        {
            gen_config.temperature = temperature;
        }

        // Execute with streaming
        let output = if let Some((_, adapter)) = &self.llm_adapter_cache {
            adapter
                .backend()
                .generate_streaming(&messages, &gen_config, on_token)?
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

    /// Execute LLM inference with pre-built ChatMessages (non-streaming).
    ///
    /// This function takes ChatMessages directly, avoiding double-formatting
    /// that would occur if we pre-formatted the prompt ourselves. The LLM
    /// backend (llama.cpp) applies its native chat template to the messages.
    ///
    /// Used by `execute_with_context` to pass conversation history
    /// to the LLM without our custom template formatting.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    fn execute_llm_with_messages(
        &mut self,
        model_file: &str,
        context_length: usize,
        messages: &[ChatMessage],
        backend_hint: Option<&str>,
    ) -> ExecutorResult<Envelope> {
        info!(
            target: "xybrid_core",
            "Executing LLM with {} ChatMessages: {} (backend: {:?})",
            messages.len(),
            model_file,
            backend_hint.unwrap_or("default")
        );

        let _llm_span = xybrid_trace::SpanGuard::new("llm_inference_with_messages");
        xybrid_trace::add_metadata("model", model_file);
        xybrid_trace::add_metadata("message_count", messages.len().to_string());

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
            let config = LlmConfig::new(model_path_str.clone()).with_context_length(context_length);

            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            adapter.load_model(&config.model_path)?;
            self.llm_adapter_cache = Some((model_path_str.clone(), adapter));
        }

        // Use default generation config
        let gen_config = GenerationConfig::default();

        // Execute with ChatMessages directly — backend applies template once
        let output = if let Some((_, adapter)) = &self.llm_adapter_cache {
            adapter.backend().generate(messages, &gen_config)?
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

    /// Execute LLM streaming with pre-built ChatMessages.
    ///
    /// This function takes ChatMessages directly, avoiding double-formatting
    /// that would occur if we pre-formatted the prompt ourselves. The LLM
    /// backend (llama.cpp) applies its native chat template to the messages.
    ///
    /// Used by `execute_streaming_with_context` to pass conversation history
    /// to the LLM without our custom template formatting.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    fn execute_llm_streaming_with_messages(
        &mut self,
        model_file: &str,
        context_length: usize,
        messages: &[ChatMessage],
        backend_hint: Option<&str>,
        on_token: StreamingCallback<'_>,
    ) -> ExecutorResult<Envelope> {
        // GenerationConfig, LlmConfig are imported at module level from types

        info!(
            target: "xybrid_core",
            "Executing LLM streaming with {} ChatMessages: {} (backend: {:?})",
            messages.len(),
            model_file,
            backend_hint.unwrap_or("default")
        );

        let _llm_span = xybrid_trace::SpanGuard::new("llm_inference_streaming_with_messages");
        xybrid_trace::add_metadata("model", model_file);
        xybrid_trace::add_metadata("message_count", messages.len().to_string());

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
            let config = LlmConfig::new(model_path_str.clone()).with_context_length(context_length);

            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            adapter.load_model(&config.model_path)?;
            self.llm_adapter_cache = Some((model_path_str.clone(), adapter));
        }

        // Use default generation config
        let gen_config = GenerationConfig::default();

        // Execute with streaming - pass ChatMessages directly to backend
        let output = if let Some((_, adapter)) = &self.llm_adapter_cache {
            adapter
                .backend()
                .generate_streaming(messages, &gen_config, on_token)?
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
            let mut config =
                LlmConfig::new(model_path_str.clone()).with_context_length(context_length);

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
            return Err(AdapterError::RuntimeError(
                "LLM adapter cache unexpectedly empty".to_string(),
            ));
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
        xybrid_trace::add_metadata("steps", metadata.preprocessing.len().to_string());

        let mut data = PreprocessedData::from_envelope(input)?;

        for step in &metadata.preprocessing {
            let step_name = step.step_name();
            debug!(target: "xybrid_core", "Applying preprocessing: {}", step_name);

            let _step_span = xybrid_trace::SpanGuard::new(format!("preprocessing:{}", step_name));

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
        xybrid_trace::add_metadata("steps", metadata.postprocessing.len().to_string());

        let mut data = outputs;

        for step in &metadata.postprocessing {
            let step_name = step.step_name();
            debug!(target: "xybrid_core", "Applying postprocessing: {}", step_name);

            let _step_span = xybrid_trace::SpanGuard::new(format!("postprocessing:{}", step_name));

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

    /// Break words used as secondary split points for center-break chunking.
    const BREAK_WORDS: &'static [&'static str] = &[
        "and", "or", "but", "because", "if", "however", "which", "when", "where", "while",
        "although", "since", "unless", "after", "before", "that",
    ];

    /// Split text into chunks at sentence boundaries for TTS.
    ///
    /// Uses a center-break algorithm for oversized sentences: splits at the
    /// natural break point nearest to the center of the text, with priority
    /// (1) comma, (2) break word, (3) whitespace. Recursive splitting handles
    /// chunks that remain too long (max depth 3).
    ///
    /// A post-pass migrates trailing break words from the end of one chunk
    /// to the start of the next chunk for more natural prosody.
    fn chunk_text_for_tts(text: &str, max_chars: usize) -> Vec<String> {
        if text.len() <= max_chars {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        // Split into sentences (keep delimiter)
        let sentences: Vec<&str> = text.split_inclusive(['.', '!', '?']).collect();

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            if sentence.len() > max_chars {
                // Flush current chunk first
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk = String::new();
                }

                // Center-break split with recursive depth
                let mut sub_chunks = Vec::new();
                Self::center_break_split(sentence, max_chars, 0, &mut sub_chunks);

                // Add all sub-chunks except the last to output, keep last as current
                if let Some(last) = sub_chunks.pop() {
                    for sc in sub_chunks {
                        if !sc.is_empty() {
                            chunks.push(sc);
                        }
                    }
                    current_chunk = last;
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

        // Post-pass: migrate trailing break words
        Self::migrate_trailing_break_words(&mut chunks);

        chunks
    }

    /// Recursively split text at the natural break point nearest to center.
    ///
    /// Priority: (1) comma nearest center, (2) break word nearest center,
    /// (3) whitespace nearest center.
    fn center_break_split(text: &str, max_chars: usize, depth: usize, out: &mut Vec<String>) {
        const MAX_DEPTH: usize = 3;

        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }

        if trimmed.len() <= max_chars || depth >= MAX_DEPTH {
            out.push(trimmed.to_string());
            return;
        }

        let center = trimmed.len() / 2;

        // Priority 1: comma nearest center
        if let Some(pos) = Self::find_nearest(trimmed, center, |i, _| {
            trimmed.as_bytes().get(i) == Some(&b',')
        }) {
            let left = trimmed[..=pos].trim();
            let right = trimmed[pos + 1..].trim();
            Self::center_break_split(left, max_chars, depth + 1, out);
            Self::center_break_split(right, max_chars, depth + 1, out);
            return;
        }

        // Priority 2: break word nearest center (match at word boundary)
        if let Some((word_start, word_len)) = Self::find_nearest_break_word(trimmed, center) {
            let left = trimmed[..word_start].trim();
            let right = trimmed[word_start + word_len..].trim();
            // Include the break word with the right chunk (post-pass may move it)
            let break_word = &trimmed[word_start..word_start + word_len];
            Self::center_break_split(left, max_chars, depth + 1, out);
            let right_with_word = format!("{} {}", break_word, right);
            Self::center_break_split(right_with_word.trim(), max_chars, depth + 1, out);
            return;
        }

        // Priority 3: whitespace nearest center
        if let Some(pos) = Self::find_nearest(trimmed, center, |i, _| {
            trimmed
                .as_bytes()
                .get(i)
                .is_some_and(|b| b.is_ascii_whitespace())
        }) {
            let left = trimmed[..pos].trim();
            let right = trimmed[pos + 1..].trim();
            Self::center_break_split(left, max_chars, depth + 1, out);
            Self::center_break_split(right, max_chars, depth + 1, out);
            return;
        }

        // No split point found — push as-is
        out.push(trimmed.to_string());
    }

    /// Find the position nearest to `center` where the predicate matches.
    /// Searches outward from center in both directions simultaneously.
    fn find_nearest<F>(text: &str, center: usize, pred: F) -> Option<usize>
    where
        F: Fn(usize, char) -> bool,
    {
        let len = text.len();
        for offset in 0..len {
            // Check right of center
            let right = center + offset;
            if right < len {
                if let Some(ch) = text[right..].chars().next() {
                    if pred(right, ch) {
                        return Some(right);
                    }
                }
            }
            // Check left of center
            if offset > 0 && offset <= center {
                let left = center - offset;
                if let Some(ch) = text[left..].chars().next() {
                    if pred(left, ch) {
                        return Some(left);
                    }
                }
            }
        }
        None
    }

    /// Find the break word nearest to center, returning (start_byte, word_byte_len).
    fn find_nearest_break_word(text: &str, center: usize) -> Option<(usize, usize)> {
        let lower = text.to_lowercase();
        let mut best: Option<(usize, usize, usize)> = None; // (start, len, distance)

        for word in Self::BREAK_WORDS {
            let pattern = format!(" {} ", word);
            let mut search_start = 0;
            while let Some(pos) = lower[search_start..].find(&pattern) {
                let abs_pos = search_start + pos + 1; // +1 to skip leading space
                let dist = abs_pos.abs_diff(center);
                if best.is_none() || dist < best.unwrap().2 {
                    best = Some((abs_pos, word.len(), dist));
                }
                search_start = search_start + pos + 1;
            }
        }

        best.map(|(start, len, _)| (start, len))
    }

    /// Post-pass: if a chunk ends with a break word, move it to the start of the next chunk.
    fn migrate_trailing_break_words(chunks: &mut [String]) {
        let mut i = 0;
        while i + 1 < chunks.len() {
            let ends_with_break = Self::BREAK_WORDS.iter().any(|w| {
                let chunk = &chunks[i];
                let lower = chunk.to_lowercase();
                lower.ends_with(&format!(" {}", w)) || lower == *w
            });

            if ends_with_break {
                // Find the break word at the end
                let chunk = chunks[i].clone();
                if let Some(last_space) = chunk.rfind(' ') {
                    let word = &chunk[last_space + 1..];
                    let lower_word = word.to_lowercase();
                    if Self::BREAK_WORDS.contains(&lower_word.as_str()) {
                        chunks[i] = chunk[..last_space].trim().to_string();
                        chunks[i + 1] = format!("{} {}", word, chunks[i + 1]);
                    }
                }
            }
            i += 1;
        }
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
        // Crossfade length: 480 samples (~20ms at 24kHz)
        const CROSSFADE_SAMPLES: usize = 480;

        let mut audio_chunks: Vec<Vec<f32>> = Vec::new();
        let session = ONNXSession::new(model_path.to_str().unwrap(), false, false)?;
        let speed = extract_tts_speed(input);

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
            let raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding, speed)?;

            // Extract audio from outputs
            if let Some(audio_tensor) = raw_outputs.values().next() {
                let chunk_audio: Vec<f32> = audio_tensor.iter().cloned().collect();
                audio_chunks.push(chunk_audio);
            }
        }

        // Concatenate chunks with crossfading
        let all_audio = crossfade_audio_chunks(&audio_chunks, CROSSFADE_SAMPLES);

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
        let speed = extract_tts_speed(input);
        let raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding, speed)?;

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

/// Extract TTS speed from envelope metadata, clamped to [0.5, 2.0].
///
/// Reads the "speed" key from `envelope.metadata`. Returns 1.0 if absent or
/// unparseable. Logs a warning if the value is outside the valid range.
pub(crate) fn extract_tts_speed(envelope: &Envelope) -> f32 {
    let speed = envelope
        .metadata
        .get("speed")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);

    if !(0.5..=2.0).contains(&speed) {
        warn!(
            "TTS speed {:.2} is outside valid range [0.5, 2.0], clamping",
            speed
        );
        return speed.clamp(0.5, 2.0);
    }

    speed
}

/// Concatenate audio chunks with linear crossfading at boundaries.
///
/// Applies a linear crossfade of `crossfade_len` samples between adjacent chunks.
/// Single-chunk input is returned as-is. Chunks shorter than `2 * crossfade_len`
/// skip crossfading for that boundary (safety guard).
fn crossfade_audio_chunks(chunks: &[Vec<f32>], crossfade_len: usize) -> Vec<f32> {
    if chunks.is_empty() {
        return Vec::new();
    }
    if chunks.len() == 1 {
        return chunks[0].clone();
    }

    // Start with the first chunk
    let mut result = chunks[0].clone();

    for chunk in &chunks[1..] {
        // Skip crossfading if either the current result tail or the new chunk head
        // is too short for the crossfade
        if result.len() < 2 * crossfade_len || chunk.len() < 2 * crossfade_len {
            result.extend(chunk);
            continue;
        }

        let overlap_start = result.len() - crossfade_len;

        // Apply crossfade in the overlap region
        for i in 0..crossfade_len {
            let t = (i + 1) as f32 / (crossfade_len + 1) as f32;
            let fade_out = 1.0 - t;
            let fade_in = t;
            result[overlap_start + i] = result[overlap_start + i] * fade_out + chunk[i] * fade_in;
        }

        // Append the rest of the new chunk (after the overlap region)
        result.extend_from_slice(&chunk[crossfade_len..]);
    }

    result
}

impl Default for TemplateExecutor {
    fn default() -> Self {
        Self::new("")
    }
}

#[cfg(test)]
mod tests {
    use super::super::template::PreprocessingStep;
    use super::*;

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
        // (a) text under 350 chars returns single chunk
        let text = "Hello world, this is a short sentence that is well under the limit.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 350);
        assert_eq!(chunks, vec![text]);
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
    fn test_chunk_text_center_break_comma() {
        // (b) 350+ char text with comma splits at comma nearest center
        // Build a long sentence with a comma near the center
        let left = "The quick brown fox jumped over the lazy dog and ran across the wide green meadow towards the old wooden fence in the distance while the birds sang their morning songs above the tall oak trees lining the path";
        let right = " and then it stopped to rest under the shade of a willow tree by the river where the water flowed gently over smooth stones and the fish swam lazily in the warm afternoon sun as clouds drifted slowly overhead";
        let text = format!("{},{}", left, right);
        assert!(
            text.len() > 350,
            "Test text must exceed 350 chars, got {}",
            text.len()
        );

        let chunks = TemplateExecutor::chunk_text_for_tts(&text, 350);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");
        // The first chunk should end with a comma (split at comma nearest center)
        assert!(
            chunks[0].ends_with(','),
            "First chunk should end at comma, got: '{}'",
            chunks[0]
        );
    }

    #[test]
    fn test_chunk_text_center_break_word() {
        // (c) 350+ char text with break word and no comma splits at break word nearest center
        let text = "The quick brown fox jumped over the lazy dog running across the wide green meadow towards the old wooden fence in the distance while birds sang their morning songs above the tall oak trees however the gentle breeze carried the sweet scent of wildflowers across the rolling hills and through the valleys where deer grazed peacefully in the golden light of the setting sun painting everything in warm hues";
        assert!(
            text.len() > 350,
            "Test text must exceed 350 chars, got {}",
            text.len()
        );
        // Verify no commas in the text
        assert!(!text.contains(','), "Test text should have no commas");

        let chunks = TemplateExecutor::chunk_text_for_tts(text, 350);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");
        // After post-pass migration, chunks[1] should start with the break word "however"
        let second_lower = chunks[1].to_lowercase();
        let starts_with_break = TemplateExecutor::BREAK_WORDS
            .iter()
            .any(|w| second_lower.starts_with(w));
        assert!(
            starts_with_break,
            "Second chunk should start with a break word after post-pass, got: '{}'",
            &chunks[1][..chunks[1].len().min(40)]
        );
    }

    #[test]
    fn test_chunk_text_center_break_whitespace() {
        // (d) 350+ char text with only whitespace splits at space nearest center
        // No commas, no break words — only plain words
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo pppp qqqq rrrr ssss tttt uuuu vvvv wwww xxxx yyyy zzzz aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo pppp qqqq rrrr ssss tttt uuuu vvvv wwww xxxx yyyy zzzz aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo pppp qqqq rrrr ssss tttt uuuu";
        assert!(
            text.len() > 350,
            "Test text must exceed 350 chars, got {}",
            text.len()
        );
        assert!(!text.contains(','), "No commas");

        let chunks = TemplateExecutor::chunk_text_for_tts(text, 350);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");
        // Each chunk should be trimmed and non-empty
        for chunk in &chunks {
            assert!(!chunk.is_empty(), "Chunks should not be empty");
            assert_eq!(chunk.as_str(), chunk.trim(), "Chunks should be trimmed");
        }
    }

    #[test]
    fn test_chunk_text_multi_sentence_long_first() {
        // (e) multi-sentence text where first sentence >350 chars gets center-break split
        //     while short second sentence stays intact
        let long_sentence = "The magnificent cathedral stood tall against the stormy sky its ancient stone walls bearing witness to centuries of history while gargoyles perched on every corner watched over the bustling city below where merchants sold their wares in the cobblestone market square filled with the aroma of freshly baked bread and exotic spices brought by traders from distant lands across vast oceans and treacherous mountain passes.";
        let short_sentence = " A bird sang nearby.";
        let text = format!("{}{}", long_sentence, short_sentence);
        assert!(
            long_sentence.len() > 350,
            "First sentence must exceed 350 chars"
        );

        let chunks = TemplateExecutor::chunk_text_for_tts(&text, 350);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");
        // The last chunk should contain the short sentence
        let last = chunks.last().unwrap();
        assert!(
            last.contains("A bird sang nearby"),
            "Short sentence should be intact in last chunk, got: '{}'",
            last
        );
    }

    #[test]
    fn test_chunk_text_empty_input() {
        let chunks = TemplateExecutor::chunk_text_for_tts("", 350);
        assert!(chunks.is_empty() || chunks == vec![""]);
    }

    #[test]
    fn test_chunk_text_whitespace_only() {
        let chunks = TemplateExecutor::chunk_text_for_tts("   ", 350);
        assert!(chunks.is_empty() || chunks.iter().all(|c| c.trim().is_empty()));
    }

    #[test]
    fn test_chunk_text_preserves_content() {
        let text =
            "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 50);
        let rejoined: String = chunks.join(" ");
        assert!(rejoined.contains("quick"));
        assert!(rejoined.contains("fox"));
        assert!(rejoined.contains("liquor"));
    }

    #[test]
    fn test_chunk_text_post_pass_break_word_migration() {
        // Verify the post-pass migration: if chunk ends with break word, it moves to next chunk
        // Use a shorter max to make behavior predictable
        let text =
            "The fox ran fast and the dog chased it quickly through the woods and over the hill.";
        let chunks = TemplateExecutor::chunk_text_for_tts(text, 30);
        // No chunk should end with a standalone break word (post-pass moves them)
        for (i, chunk) in chunks.iter().enumerate() {
            if i + 1 < chunks.len() {
                let lower = chunk.to_lowercase();
                for w in TemplateExecutor::BREAK_WORDS {
                    assert!(
                        !lower.ends_with(&format!(" {}", w)),
                        "Chunk {} should not end with break word '{}': '{}'",
                        i,
                        w,
                        chunk
                    );
                }
            }
        }
    }

    // ============================================================================
    // is_tts_model Tests
    // ============================================================================

    #[test]
    fn test_is_tts_model_with_phonemize_step() {
        let metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx").with_preprocessing(
            PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
                silence_tokens: None,
            },
        );
        assert!(TemplateExecutor::is_tts_model(&metadata));
    }

    #[test]
    fn test_is_tts_model_without_phonemize() {
        let metadata = ModelMetadata::onnx("test-asr", "1.0", "model.onnx").with_preprocessing(
            PreprocessingStep::AudioDecode {
                sample_rate: 16000,
                channels: 1,
            },
        );
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
                silence_tokens: None,
            });
        assert!(TemplateExecutor::is_tts_model(&metadata));
    }

    #[test]
    fn test_is_tts_model_with_mel_spectrogram_is_not_tts() {
        let metadata = ModelMetadata::onnx("test-asr", "1.0", "model.onnx").with_preprocessing(
            PreprocessingStep::MelSpectrogram {
                preset: Some("whisper".to_string()),
                n_mels: 80,
                sample_rate: 16000,
                fft_size: 400,
                hop_length: 160,
                mel_scale: Default::default(),
                max_frames: Some(3000),
            },
        );
        assert!(!TemplateExecutor::is_tts_model(&metadata));
    }

    // ============================================================================
    // execute_with_context Tests
    // ============================================================================

    #[test]
    fn test_execute_with_context_builds_message_list() {
        // Test that the method correctly builds the message list from context + input
        // This is a unit test that verifies the logic without actual model execution

        use crate::conversation::ConversationContext;

        // Create a context with system and history
        let mut ctx = ConversationContext::new().with_system(
            Envelope::new(EnvelopeKind::Text("You are helpful.".to_string()))
                .with_role(MessageRole::System),
        );

        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hello!".to_string())).with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hi there!".to_string()))
                .with_role(MessageRole::Assistant),
        );

        // Verify context_for_llm returns correct structure
        let messages = ctx.context_for_llm();
        assert_eq!(messages.len(), 3);
        assert!(messages[0].is_system_message());
        assert!(messages[1].is_user_message());
        assert!(messages[2].is_assistant_message());

        // The input would be the next user message
        let input = Envelope::new(EnvelopeKind::Text("How are you?".to_string()))
            .with_role(MessageRole::User);

        // Verify we can append input to messages
        let mut all_messages = messages.clone();
        all_messages.push(&input);
        assert_eq!(all_messages.len(), 4);
    }

    #[test]
    fn test_execute_with_context_uses_chat_template_formatter() {
        // Test that ChatTemplateFormatter correctly formats context + input

        use super::super::chat_template::{ChatTemplateFormat, ChatTemplateFormatter};
        use crate::conversation::ConversationContext;

        let mut ctx = ConversationContext::new().with_system(
            Envelope::new(EnvelopeKind::Text("You are helpful.".to_string()))
                .with_role(MessageRole::System),
        );

        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hello!".to_string())).with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hi there!".to_string()))
                .with_role(MessageRole::Assistant),
        );

        let input = Envelope::new(EnvelopeKind::Text("How are you?".to_string()))
            .with_role(MessageRole::User);

        // Build messages as execute_with_context would
        let mut messages: Vec<&Envelope> = ctx.context_for_llm();
        messages.push(&input);

        // Format with ChatML
        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);

        // Verify the prompt contains all messages in order
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello!<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant\nHi there!<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHow are you?<|im_end|>"));
        // Should end with assistant start marker
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_execute_with_context_result_tagged_as_assistant() {
        // Verify that the result envelope role tagging works correctly

        let envelope = Envelope::new(EnvelopeKind::Text("I'm doing great!".to_string()));
        assert!(envelope.role().is_none());

        let tagged = envelope.with_role(MessageRole::Assistant);
        assert!(tagged.is_assistant_message());
        assert_eq!(tagged.role(), Some(MessageRole::Assistant));
    }

    #[test]
    fn test_execute_with_context_preserves_input_content() {
        // Verify that the input content is included in the formatted prompt

        use super::super::chat_template::{ChatTemplateFormat, ChatTemplateFormatter};
        use crate::conversation::ConversationContext;

        let ctx = ConversationContext::new();
        let input = Envelope::new(EnvelopeKind::Text("What is 2+2?".to_string()))
            .with_role(MessageRole::User);

        let mut messages: Vec<&Envelope> = ctx.context_for_llm();
        messages.push(&input);

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);

        // The input content should be in the formatted prompt
        assert!(prompt.contains("What is 2+2?"));
    }

    #[test]
    fn test_execute_with_context_with_empty_context() {
        // Test behavior with empty context (no system, no history)

        use super::super::chat_template::{ChatTemplateFormat, ChatTemplateFormatter};
        use crate::conversation::ConversationContext;

        let ctx = ConversationContext::new();
        let input =
            Envelope::new(EnvelopeKind::Text("Hello!".to_string())).with_role(MessageRole::User);

        let mut messages: Vec<&Envelope> = ctx.context_for_llm();
        messages.push(&input);

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);

        // With empty context, should just have the input message
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_execute_with_context_llama_format() {
        // Test with Llama format instead of ChatML

        use super::super::chat_template::{ChatTemplateFormat, ChatTemplateFormatter};
        use crate::conversation::ConversationContext;

        let mut ctx = ConversationContext::new().with_system(
            Envelope::new(EnvelopeKind::Text("Be concise.".to_string()))
                .with_role(MessageRole::System),
        );

        ctx.push(Envelope::new(EnvelopeKind::Text("Hi!".to_string())).with_role(MessageRole::User));
        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hello!".to_string()))
                .with_role(MessageRole::Assistant),
        );

        let input =
            Envelope::new(EnvelopeKind::Text("Bye!".to_string())).with_role(MessageRole::User);

        let mut messages: Vec<&Envelope> = ctx.context_for_llm();
        messages.push(&input);

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Llama);

        // Llama format should contain system in <<SYS>> tags
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("Be concise."));
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("[/INST]"));
    }

    #[test]
    fn test_chat_template_format_from_str() {
        // Test that chat_template field parsing works

        use super::super::chat_template::ChatTemplateFormat;

        assert_eq!(
            ChatTemplateFormat::from_str("chatml"),
            Some(ChatTemplateFormat::ChatML)
        );
        assert_eq!(
            ChatTemplateFormat::from_str("llama"),
            Some(ChatTemplateFormat::Llama)
        );
        assert_eq!(
            ChatTemplateFormat::from_str("llama2"),
            Some(ChatTemplateFormat::Llama)
        );
        assert_eq!(ChatTemplateFormat::from_str("unknown"), None);

        // Default should be ChatML when None
        let default: ChatTemplateFormat = Default::default();
        assert_eq!(default, ChatTemplateFormat::ChatML);
    }

    // ============================================================================
    // Crossfade Tests
    // ============================================================================

    #[test]
    fn test_crossfade_empty_chunks() {
        let chunks: Vec<Vec<f32>> = vec![];
        let result = crossfade_audio_chunks(&chunks, 480);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crossfade_single_chunk_unchanged() {
        let chunk = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = crossfade_audio_chunks(std::slice::from_ref(&chunk), 480);
        assert_eq!(result, chunk);
    }

    #[test]
    fn test_crossfade_two_chunks() {
        let crossfade_len = 4;
        // Chunk A: 10 samples of 1.0
        let chunk_a = vec![1.0; 10];
        // Chunk B: 10 samples of 0.0
        let chunk_b = vec![0.0; 10];

        let result = crossfade_audio_chunks(&[chunk_a, chunk_b], crossfade_len);

        // Result length: 10 + 10 - 4 (overlap) = 16
        assert_eq!(result.len(), 16);

        // First 6 samples: unchanged from chunk_a (before overlap)
        for &v in &result[..6] {
            assert!((v - 1.0).abs() < 1e-6);
        }

        // Overlap region (4 samples): linear blend from 1.0 to 0.0
        // t = (i+1) / (crossfade_len+1), fade_out = 1-t, fade_in = t
        // result[6+i] = 1.0 * (1-t) + 0.0 * t = 1-t
        for i in 0..crossfade_len {
            let t = (i + 1) as f32 / (crossfade_len + 1) as f32;
            let expected = 1.0 - t;
            assert!(
                (result[6 + i] - expected).abs() < 1e-6,
                "at overlap index {i}: got {}, expected {expected}",
                result[6 + i]
            );
        }

        // Last 6 samples: from chunk_b after overlap
        for &v in &result[10..] {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_crossfade_three_chunks() {
        let crossfade_len = 2;
        let chunk_a = vec![1.0; 8];
        let chunk_b = vec![0.5; 8];
        let chunk_c = vec![0.0; 8];

        let result = crossfade_audio_chunks(&[chunk_a, chunk_b, chunk_c], crossfade_len);

        // Length: 8 + (8 - 2) + (8 - 2) = 20
        assert_eq!(result.len(), 20);
    }

    #[test]
    fn test_crossfade_short_chunk_skips_crossfade() {
        let crossfade_len = 4;
        // Chunk too short (len < 2 * crossfade_len = 8)
        let chunk_a = vec![1.0; 10];
        let chunk_b = vec![0.5; 6]; // Too short for crossfade

        let result = crossfade_audio_chunks(&[chunk_a, chunk_b], crossfade_len);

        // Should be simple concatenation (no crossfade)
        assert_eq!(result.len(), 16);
        assert!((result[9] - 1.0).abs() < 1e-6);
        assert!((result[10] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_preserves_total_energy() {
        // When both chunks have the same constant value, crossfade should preserve it
        let crossfade_len = 4;
        let chunk_a = vec![0.5; 10];
        let chunk_b = vec![0.5; 10];

        let result = crossfade_audio_chunks(&[chunk_a, chunk_b], crossfade_len);

        // In the overlap region: 0.5 * fade_out + 0.5 * fade_in = 0.5 * (fade_out + fade_in) = 0.5
        // since fade_out + fade_in = 1.0
        for &v in &result {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "expected 0.5, got {v} — crossfade should preserve constant signal"
            );
        }
    }
}
