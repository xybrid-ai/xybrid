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
//! ```no_run
//! # fn _example() {
//! use std::collections::HashMap;
//! use xybrid_core::execution::TemplateExecutor;
//! use xybrid_core::runtime_adapter::ModelRuntime;
//! use xybrid_core::testing::mocks::MockRuntime;
//!
//! // Default: uses ONNX (and Candle if feature-enabled)
//! let executor = TemplateExecutor::new("models/");
//!
//! // Custom runtime injection for testing:
//! let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
//! runtimes.insert("mock".to_string(), Box::new(MockRuntime::with_text("hi")));
//! let executor = TemplateExecutor::with_runtimes("models/", runtimes);
//! # }
//! ```

use log::{debug, info, warn};

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use super::template::PostprocessingStep;
use super::template::{
    backend_label_from_template, quantization_label_from_metadata, span_kind_from_template,
    stage_kind_from_task, ExecutionMode, ExecutionTemplate, ModelMetadata, PipelineStage,
};
use crate::conversation::ConversationContext;
#[cfg(any(feature = "vision", feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::ir::EnvelopeKind;
use crate::ir::{Envelope, MessageRole};
use crate::runtime_adapter::{AdapterError, ModelRuntime};
use crate::tracing as xybrid_trace;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::listener::ExecutionGuard;

fn mark_execution_terminal(guard: &ExecutionGuard, error: &AdapterError) {
    if error.cloud_fallback_abort_reason().is_some() {
        guard.set_controlled_abort();
    } else {
        guard.set_failed(error.to_string());
    }
}

/// Stamp cost-attribution metadata (`backend`, `quantization`) onto the
/// currently-open span.
///
/// The outer `execute:<model_id>` span set up by `execute_impl` already
/// carries these — but the chat-context entry points
/// (`execute_with_context_impl`, `execute_streaming_with_context_impl`)
/// dispatch directly to the inner `execute_llm*` methods without ever
/// opening that outer span, so the inner LLM spans are the only place the
/// SDK telemetry hoist can read these fields from on a chat-context call.
/// Call this from each inner LLM span site immediately after `SpanGuard`
/// so both the non-context and chat-context flows produce the same wire
/// shape.
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn stamp_llm_span_cost_attribution(metadata: &ModelMetadata) {
    // Reuse the same resolver the outer `execute:<model_id>` span uses
    // so both spans agree on the canonical wire label — including the
    // GGUF-defaults-to-llamacpp behaviour that lights up unannotated
    // bundles in the registry.
    let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());
    if let Some(label) = backend_label_from_template(&metadata.execution_template, backend_hint) {
        xybrid_trace::add_metadata("backend", label);
    }
    if let Some(quant) = quantization_label_from_metadata(metadata) {
        xybrid_trace::add_metadata("quantization", quant);
    }
}

/// Overwrite the currently-open span's `backend` cost-attribution
/// metadata with the actual runtime's wire label.
///
/// `stamp_llm_span_cost_attribution` stamps the *template-derived*
/// default first, but the runtime is selected by cargo feature (see
/// `LlmRuntimeAdapter::new` precedence), so the template-derived
/// label can disagree with the runtime that actually executes — for
/// example, an `llm-mistral`-only build loading an unannotated GGUF
/// bundle runs on mistral.rs but the template default says `llamacpp`.
/// Calling this after the adapter is resolved replaces the default
/// with ground truth (via [`LlmRuntimeAdapter::wire_label`]). Spans
/// without a wire label (mock/test backends) leave the
/// template-derived stamp in place.
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn stamp_llm_runtime_backend(adapter: &LlmRuntimeAdapter) {
    if let Some(label) = adapter.wire_label() {
        xybrid_trace::add_metadata("backend", label);
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct LlmAdapterCacheKey {
    model_path: String,
    chat_template_path: Option<String>,
    context_length: usize,
    backend_hint: Option<String>,
    vision_encoder_path: Option<String>,
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
impl LlmAdapterCacheKey {
    fn new(
        model_path: String,
        chat_template_path: Option<String>,
        context_length: usize,
        backend_hint: Option<&str>,
        vision_encoder_path: Option<String>,
    ) -> Self {
        Self {
            model_path,
            chat_template_path,
            context_length,
            backend_hint: backend_hint.map(ToOwned::to_owned),
            vision_encoder_path,
        }
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn resolve_optional_model_path(base_path: &str, path: Option<&str>) -> Option<String> {
    path.map(|p| Path::new(base_path).join(p).to_string_lossy().to_string())
}

#[cfg(all(
    feature = "vision",
    any(feature = "llm-mistral", feature = "llm-llamacpp")
))]
fn reject_text_only_model_image_input(
    metadata: &ModelMetadata,
    input: &Envelope,
) -> Result<(), AdapterError> {
    if matches!(
        input.kind,
        EnvelopeKind::Image { .. } | EnvelopeKind::MultiPart(_)
    ) {
        return Err(AdapterError::UnsupportedModelCapability {
            model_id: metadata.model_id.clone(),
            capability: "image input".to_string(),
            hint: "use a VisionLanguage model with a vision_encoder for multimodal requests"
                .to_string(),
        });
    }

    Ok(())
}

#[cfg(all(
    feature = "vision",
    any(feature = "llm-mistral", feature = "llm-llamacpp")
))]
fn unsupported_backend_vision_error(metadata: &ModelMetadata, backend_name: &str) -> AdapterError {
    AdapterError::UnsupportedBackendCapability {
        model_id: metadata.model_id.clone(),
        backend: backend_name.to_string(),
        capability: "vision input".to_string(),
        hint: "enable llm-llamacpp-vision or select a vision-capable backend".to_string(),
    }
}

#[cfg(all(
    feature = "vision",
    any(feature = "llm-mistral", feature = "llm-llamacpp")
))]
fn ensure_backend_supports_vision(
    metadata: &ModelMetadata,
    adapter: &LlmRuntimeAdapter,
) -> Result<(), AdapterError> {
    let backend = adapter.backend();
    if backend.supports_vision() {
        Ok(())
    } else {
        Err(unsupported_backend_vision_error(metadata, backend.name()))
    }
}

#[cfg(feature = "vision")]
fn elapsed_millis_floor_one(start: std::time::Instant) -> u32 {
    start.elapsed().as_millis().max(1).min(u32::MAX as u128) as u32
}

// Internal: ONNX-specific types needed for optimized execution paths
// These are implementation details, not part of the public API
use crate::execution::session_factory::OnnxSessionFactory;
use crate::runtime_adapter::onnx::{
    ExecutionProviderKind, ONNXSession, OnnxRuntime, SessionOptions,
};

#[cfg(feature = "candle")]
use crate::runtime_adapter::candle::CandleRuntime;

// Always-available LLM types (defined in runtime_adapter/types.rs)
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::runtime_adapter::types::{ChatMessage, LlmConfig};
use crate::runtime_adapter::types::{GenerationConfig, StreamingCallback};
#[cfg(feature = "vision")]
use crate::runtime_adapter::{MultimodalChatMessage, MultimodalMessagePart, VisionEncoder};

// LLM adapter implementation (only available with LLM features)
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::runtime_adapter::llm::LlmRuntimeAdapter;

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
/// ```no_run
/// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
/// use std::collections::HashMap;
/// use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
/// use xybrid_core::ir::{Envelope, EnvelopeKind};
/// use xybrid_core::runtime_adapter::ModelRuntime;
/// use xybrid_core::testing::mocks::MockRuntime;
///
/// # let metadata: ModelMetadata = unimplemented!();
/// # let input = Envelope::new(EnvelopeKind::Text("hi".into()));
/// // Default usage (recommended)
/// let mut executor = TemplateExecutor::new("models/whisper");
/// let output = executor.execute(&metadata, &input, None)?;
///
/// // Testing with mock runtime
/// let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
/// runtimes.insert("mock".to_string(), Box::new(MockRuntime::with_text("hi")));
/// let executor = TemplateExecutor::with_runtimes("models/", runtimes);
/// # let _ = (output, executor);
/// # Ok(())
/// # }
/// ```
/// A cached ONNX session for the TTS path, reused across runs to avoid the
/// per-call graph load + Level-3 optimization that otherwise dominates short-TTS
/// latency. Keyed by the model file path; the captured `(len, mtime)` identity
/// self-invalidates the cache if the file at that path is replaced in place.
struct TtsSessionCache {
    model_path: PathBuf,
    /// `(len, modified)` of the model file at build time. `None` when the file
    /// metadata couldn't be read — treated as "unverifiable", forcing a rebuild
    /// rather than trusting a possibly-stale session.
    file_identity: Option<(u64, std::time::SystemTime)>,
    session: Arc<ONNXSession>,
}

pub struct TemplateExecutor {
    /// Configured runtimes (e.g., "onnx", "candle")
    runtimes: HashMap<String, Box<dyn ModelRuntime>>,
    /// Base path for resolving relative model paths
    base_path: String,
    /// Cached LLM adapter to avoid reloading models between executions.
    /// Stores (cache key, adapter) tuple - reused only when all load-relevant
    /// config matches. The key includes the model path, context window, chat
    /// template, backend hint, and optional vision encoder/mmproj artifact.
    /// This field always exists but is only populated when LLM features are enabled.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    llm_adapter_cache: Option<(LlmAdapterCacheKey, LlmRuntimeAdapter)>,
    /// Placeholder for llm_adapter_cache when LLM features are disabled.
    /// This ensures the struct has consistent fields regardless of features.
    #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
    llm_adapter_cache: Option<()>,
    /// Cached ONNX session for the TTS path (see [`TtsSessionCache`]). Lives for
    /// the executor's lifetime — i.e. the TTS model's load — and drops on unload,
    /// exactly like `llm_adapter_cache`; no separate eviction needed.
    tts_session_cache: Option<TtsSessionCache>,
    /// Optional embedding-style vision encoders keyed by metadata `vision_encoder.file`.
    ///
    /// llama.cpp VLMs do not use this registry: they consume raw ordered
    /// multimodal messages through their backend-owned mtmd path. This registry
    /// exists for backends that expose a separate image-encoder tensor seam.
    #[cfg(feature = "vision")]
    vision_encoders: HashMap<String, Box<dyn VisionEncoder>>,
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
    /// ```no_run
    /// # fn _example() {
    /// use std::collections::HashMap;
    /// use xybrid_core::execution::TemplateExecutor;
    /// use xybrid_core::runtime_adapter::ModelRuntime;
    /// use xybrid_core::testing::mocks::MockRuntime;
    ///
    /// // Inject a mock runtime for testing
    /// let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
    /// runtimes.insert("onnx".to_string(), Box::new(MockRuntime::with_text("hi")));
    /// let executor = TemplateExecutor::with_runtimes("models/", runtimes);
    /// # }
    /// ```
    pub fn with_runtimes(
        base_path: &str,
        runtimes: HashMap<String, Box<dyn ModelRuntime>>,
    ) -> Self {
        Self {
            runtimes,
            base_path: base_path.into(),
            llm_adapter_cache: None,
            tts_session_cache: None,
            #[cfg(feature = "vision")]
            vision_encoders: HashMap::new(),
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

    /// Register an embedding-style vision encoder for a sibling encoder file.
    ///
    /// The key should match `ModelMetadata::vision_encoder.file`. Backends that
    /// own raw multimodal planning internally, such as llama.cpp mtmd, should
    /// leave this registry empty and consume `MultimodalChatMessage` directly.
    #[cfg(feature = "vision")]
    pub fn register_vision_encoder(
        &mut self,
        encoder_file: impl Into<String>,
        encoder: Box<dyn VisionEncoder>,
    ) {
        self.vision_encoders.insert(encoder_file.into(), encoder);
    }

    /// Get a reference to a registered runtime.
    pub fn get_runtime(&self, name: &str) -> Option<&dyn ModelRuntime> {
        self.runtimes.get(name).map(|r| r.as_ref())
    }

    /// List registered runtime names.
    pub fn list_runtimes(&self) -> Vec<&str> {
        self.runtimes.keys().map(|s| s.as_str()).collect()
    }

    #[cfg(feature = "vision")]
    fn multimodal_messages_with_context(
        input: &Envelope,
        context: &ConversationContext,
    ) -> ExecutorResult<Vec<MultimodalChatMessage>> {
        let mut messages = MultimodalChatMessage::from_context(context)?;
        messages.push(MultimodalChatMessage::from_envelope(input)?);
        Ok(messages)
    }

    #[cfg(feature = "vision")]
    fn encode_registered_vision_inputs(
        &mut self,
        metadata: &ModelMetadata,
        messages: &[MultimodalChatMessage],
    ) -> ExecutorResult<Option<u32>> {
        let Some(config) = metadata.vision_encoder.as_ref() else {
            return Ok(None);
        };
        if !self.vision_encoders.contains_key(&config.file) {
            return Ok(None);
        }

        let steps = config.preprocessing_steps();
        let image_preprocess_started = std::time::Instant::now();
        let image_tensors = Self::preprocess_multimodal_images(&self.base_path, &steps, messages)?;
        if image_tensors.is_empty() {
            return Ok(None);
        }
        let image_preprocess_ms = elapsed_millis_floor_one(image_preprocess_started);

        let _span = xybrid_trace::SpanGuard::new("vision_encoder");
        xybrid_trace::add_metadata("encoder_file", &config.file);
        xybrid_trace::add_metadata("image_count", image_tensors.len().to_string());
        xybrid_trace::add_metadata("image_preprocess_ms", image_preprocess_ms.to_string());

        let encoder = self.vision_encoders.get_mut(&config.file).ok_or_else(|| {
            AdapterError::RuntimeError(format!(
                "Vision encoder '{}' disappeared during execution",
                config.file
            ))
        })?;

        let mut placeholder_tokens = 0usize;
        for tensor in image_tensors {
            let embeddings = encoder.encode(tensor).map_err(|err| {
                AdapterError::RuntimeError(format!(
                    "Vision encoder '{}' failed: {}",
                    config.file, err
                ))
            })?;
            placeholder_tokens += embeddings.placeholder_tokens.len();
        }

        xybrid_trace::add_metadata("placeholder_tokens", placeholder_tokens.to_string());
        Ok(Some(image_preprocess_ms))
    }

    #[cfg(feature = "vision")]
    fn preprocess_multimodal_images(
        base_path: &str,
        steps: &[super::template::PreprocessingStep],
        messages: &[MultimodalChatMessage],
    ) -> ExecutorResult<Vec<ArrayD<f32>>> {
        let mut tensors = Vec::new();

        for message in messages {
            for part in &message.parts {
                let MultimodalMessagePart::Image(image) = part else {
                    continue;
                };

                let image_input = Envelope::new(EnvelopeKind::Image {
                    source: image.source.clone(),
                })
                .with_local_id(image.local_id.clone());
                let mut data = PreprocessedData::from_envelope(&image_input)?;

                for step in steps {
                    data = preprocessing::apply_preprocessing_step(
                        step,
                        data,
                        &image_input,
                        base_path,
                    )?;
                }

                tensors.push(data.to_tensor()?);
            }
        }

        Ok(tensors)
    }

    /// Execute a model based on its metadata.
    pub fn execute(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        // Silent guard: the SDK's `XybridModel::run` / `run_async` wrappers
        // around this method emit `ModelComplete` with full attribution
        // (model_id, backend, latency, etc.). Emitting an outer
        // `Started` / `Completed` pair from here would surface as
        // duplicate noise rows on the Traces dashboard. Failed-path
        // emission via `mark_execution_terminal` is preserved.
        let guard = ExecutionGuard::new_silent(&metadata.model_id, "execute");
        let result = self.execute_impl(metadata, input, config);
        if let Err(e) = &result {
            mark_execution_terminal(&guard, e);
        }
        result
    }

    /// Internal implementation of execute — no telemetry listener events.
    fn execute_impl(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        #[allow(unused_variables)] config: Option<&GenerationConfig>,
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
        // Grouping + colour hints for the swim-lanes renderer. `stage_kind`
        // drives the lane label (ASR / LLM / TTS / …); `span_kind` drives
        // the bar colour (gpu / cpu / io / tool). LLM adapters override
        // `span_kind` on their inner `llm_inference` span with more precise
        // information (Metal vs CPU kernels).
        if let Some(task) = metadata.metadata.get("task").and_then(|v| v.as_str()) {
            if let Some(kind) = stage_kind_from_task(task) {
                xybrid_trace::add_metadata("stage_kind", kind);
            }
            // Semantic task label for cost-attribution telemetry
            // (per `PlatformEvent.task`). Echoes the raw value from
            // `model_metadata.json` so the dashboard can filter
            // chat/vlm/asr/tts/embedding/etc. without joining against
            // the registry at render time. Omitted when the model
            // bundle didn't declare a task.
            xybrid_trace::add_metadata("task", task);
        }
        xybrid_trace::add_metadata(
            "span_kind",
            span_kind_from_template(&metadata.execution_template),
        );

        // Cost-accounting backend label (per `PlatformEvent.backend`).
        // The bundle's `metadata.backend` hint disambiguates GGUF runtimes
        // (llama.cpp vs mistral.rs); for ONNX/SafeTensors the template
        // itself fixes the label. Omitted when the runtime isn't part of
        // the closed set yet (CoreML / TFLite / ModelGraph) so analytics
        // sees "absent" not "guessed".
        let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());
        if let Some(label) = backend_label_from_template(&metadata.execution_template, backend_hint)
        {
            xybrid_trace::add_metadata("backend", label);
        }

        // Quantization label for cost-attribution telemetry
        // (per `PlatformEvent.quantization`). Two runs of "the same
        // model" can differ by 4× in size / speed / quality based on
        // quantization, so the dashboard needs this to avoid
        // collapsing `kokoro-82m@q4_0` and `kokoro-82m@fp16` into one
        // row. Source priority: bundle metadata > GGUF filename >
        // omitted. See `quantization_label_from_metadata`.
        if let Some(quant) = quantization_label_from_metadata(metadata) {
            xybrid_trace::add_metadata("quantization", quant);
        }

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

        // Step 1.5: Codec TTS dispatch (GGUF backbone + CodecDecode postprocessing).
        // Must come before the plain-GGUF fast-path below — CodecTtsStrategy orchestrates
        // PhonemeRaw preprocessing, voice codes, LLM generation, and ONNX codec decoding
        // in a single call. Plain-GGUF models fall through to execute_llm() as before.
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        if matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. })
            && metadata
                .postprocessing
                .iter()
                .any(|s| matches!(s, PostprocessingStep::CodecDecode { .. }))
        {
            use super::strategies::{CodecTtsStrategy, ExecutionContext, ExecutionStrategy};
            debug!(
                target: "xybrid_core",
                "Detected codec TTS metadata, dispatching to CodecTtsStrategy"
            );
            let strategy = CodecTtsStrategy::new();
            let mut ctx = ExecutionContext {
                base_path: &self.base_path,
                runtimes: &mut self.runtimes,
            };
            return strategy.execute(&mut ctx, metadata, input);
        }

        #[cfg(all(
            feature = "vision",
            any(feature = "llm-mistral", feature = "llm-llamacpp")
        ))]
        if let ExecutionTemplate::VisionLanguage {
            model_file,
            chat_template,
            context_length,
            ..
        } = &metadata.execution_template
        {
            let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());
            return self.execute_vision_language(
                metadata,
                model_file,
                chat_template.as_deref(),
                *context_length,
                input,
                backend_hint,
                config,
            );
        }

        #[cfg(all(
            feature = "vision",
            not(any(feature = "llm-mistral", feature = "llm-llamacpp"))
        ))]
        if matches!(
            metadata.execution_template,
            ExecutionTemplate::VisionLanguage { .. }
        ) {
            return Err(AdapterError::RuntimeError(
                "VisionLanguage execution requires the 'llm-mistral' or 'llm-llamacpp' feature"
                    .to_string(),
            ));
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
                ..
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
                    metadata,
                    model_file,
                    chat_template.as_deref(),
                    *context_length,
                    input,
                    backend_hint,
                    config,
                );
            }
            #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
            ExecutionTemplate::Gguf { .. } => {
                return Err(AdapterError::RuntimeError(
                    "GGUF/LLM execution requires the 'llm-mistral' or 'llm-llamacpp' feature"
                        .to_string(),
                ));
            }
            #[cfg(feature = "vision")]
            ExecutionTemplate::VisionLanguage { .. } => {
                return Err(AdapterError::RuntimeError(
                    "VisionLanguage execution should dispatch before the single-model path"
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

            // Create and run BERT session through the shared factory entry.
            let session = OnnxSessionFactory::create_session(
                &model_full_path,
                ExecutionProviderKind::Cpu,
                SessionOptions::default(),
            )?;
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
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
    /// # use xybrid_core::conversation::ConversationContext;
    /// # use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    /// # let metadata: ModelMetadata = unimplemented!();
    /// # let mut executor = TemplateExecutor::new("models/");
    /// # let mut ctx = ConversationContext::new();
    /// // CORRECT: Push AFTER execution
    /// let input = Envelope::new(EnvelopeKind::Text("Hello".into()))
    ///     .with_role(MessageRole::User);
    /// let result = executor.execute_with_context(&metadata, &input, &ctx, None)?;
    /// ctx.push(input.clone());   // Push for next turn
    /// ctx.push(result);  // Push for next turn
    ///
    /// // WRONG: Pushing before causes duplicate messages!
    /// ctx.push(input.clone());  // DON'T DO THIS
    /// let result = executor.execute_with_context(&metadata, &input, &ctx, None)?;
    /// # let _ = result;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// A runtime warning is logged if the input is already in context (detected by local_id).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
    /// use xybrid_core::conversation::ConversationContext;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// # let metadata: ModelMetadata = unimplemented!();
    /// # let mut executor = TemplateExecutor::new("models/");
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
    /// let result = executor.execute_with_context(&metadata, &input, &ctx, None)?;
    /// assert!(result.is_assistant_message());
    ///
    /// // Update context for next turn
    /// ctx.push(input);
    /// ctx.push(result);
    /// # Ok(())
    /// # }
    /// ```
    pub fn execute_with_context(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        context: &ConversationContext,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        // Silent guard for the same reason as `execute_streaming_with_context`:
        // the user-facing telemetry for a chat-context turn is the SDK's
        // `ModelComplete` event from `XybridModel::run_with_context`. The
        // outer executor span is an implementation detail and emitting
        // `Started` / `Completed` from here surfaces as separate noise
        // rows in the Traces dashboard. Error reporting is preserved.
        let guard = ExecutionGuard::new_silent(&metadata.model_id, "execute_with_context");
        let result = self.execute_with_context_impl(metadata, input, context, config);
        if let Err(e) = &result {
            mark_execution_terminal(&guard, e);
        }
        result
    }

    /// Internal implementation of execute_with_context — no telemetry listener events.
    fn execute_with_context_impl(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        context: &ConversationContext,
        config: Option<&GenerationConfig>,
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

        #[cfg(all(
            feature = "vision",
            any(feature = "llm-mistral", feature = "llm-llamacpp")
        ))]
        if let ExecutionTemplate::VisionLanguage {
            model_file,
            chat_template,
            context_length,
            ..
        } = &metadata.execution_template
        {
            debug!(
                target: "xybrid_core",
                "VisionLanguage model detected, converting context to multimodal messages"
            );

            let messages = Self::multimodal_messages_with_context(input, context)?;
            let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());

            let mut result = self.execute_llm_multimodal_messages(
                metadata,
                model_file,
                chat_template.as_deref(),
                *context_length,
                &messages,
                backend_hint,
                config,
            )?;

            result = result.with_role(MessageRole::Assistant);
            return Ok(result);
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

            #[cfg(feature = "vision")]
            reject_text_only_model_image_input(metadata, input)?;

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
                metadata,
                model_file,
                *context_length,
                &chat_messages,
                backend_hint,
                config,
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
        let mut result = self.execute_impl(metadata, input, config)?;

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
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use std::io::Write;
    /// # use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
    /// # use xybrid_core::ir::{Envelope, EnvelopeKind};
    /// # let metadata: ModelMetadata = unimplemented!();
    /// # let input = Envelope::new(EnvelopeKind::Text("hi".into()));
    /// # let mut executor = TemplateExecutor::new("models/");
    /// executor.execute_streaming(&metadata, &input, Box::new(|token| {
    ///     print!("{}", token.token);
    ///     std::io::stdout().flush().ok();
    ///     Ok(())
    /// }), None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn execute_streaming(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        on_token: StreamingCallback<'_>,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        // Silent guard: the SDK's `XybridModel::run_streaming` wrapper
        // around this method emits `ModelComplete` with full attribution.
        // The outer `Started` / `Completed` pair would surface as
        // duplicate noise rows on the Traces dashboard. Failed-path
        // emission via `mark_execution_terminal` is preserved.
        let guard = ExecutionGuard::new_silent(&metadata.model_id, "execute_streaming");
        let result = self.execute_streaming_impl(metadata, input, on_token, config);
        if let Err(e) = &result {
            mark_execution_terminal(&guard, e);
        }
        result
    }

    /// Internal implementation of execute_streaming — no telemetry listener events.
    fn execute_streaming_impl(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        #[allow(unused_variables)] on_token: StreamingCallback<'_>,
        #[allow(unused_variables)] config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        {
            #[cfg(feature = "vision")]
            if let super::template::ExecutionTemplate::VisionLanguage {
                model_file,
                chat_template,
                context_length,
                ..
            } = &metadata.execution_template
            {
                let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());

                return self.execute_vision_language_streaming(
                    metadata,
                    model_file,
                    chat_template.as_deref(),
                    *context_length,
                    input,
                    backend_hint,
                    on_token,
                    config,
                );
            }

            // Only GGUF (LLM) templates support streaming
            if let super::template::ExecutionTemplate::Gguf {
                model_file,
                chat_template,
                context_length,
                ..
            } = &metadata.execution_template
            {
                let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());

                return self.execute_llm_streaming(
                    metadata,
                    model_file,
                    chat_template.as_deref(),
                    *context_length,
                    input,
                    backend_hint,
                    on_token,
                    config,
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

        self.execute_impl(metadata, input, config)
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
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use std::io::Write;
    /// # use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
    /// # use xybrid_core::conversation::ConversationContext;
    /// # use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    /// # let metadata: ModelMetadata = unimplemented!();
    /// # let input = Envelope::new(EnvelopeKind::Text("hi".into()));
    /// # let mut executor = TemplateExecutor::new("models/");
    /// let mut ctx = ConversationContext::new();
    /// ctx.push(Envelope::new(EnvelopeKind::Text("Hello!".into())).with_role(MessageRole::User));
    ///
    /// executor.execute_streaming_with_context(&metadata, &input, &ctx, Box::new(|token| {
    ///     print!("{}", token.token);
    ///     std::io::stdout().flush().ok();
    ///     Ok(())
    /// }), None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn execute_streaming_with_context(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        context: &ConversationContext,
        on_token: StreamingCallback<'_>,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        // Silent guard: the user-facing telemetry for a chat-context
        // turn is the SDK's `ModelComplete` event from
        // `XybridModel::run_streaming_with_context`. Emitting an outer
        // `Started`/`Completed` pair here surfaces as separate noise
        // rows in the Traces dashboard with the executor-internal span
        // name. Error reporting is preserved — `mark_execution_terminal`
        // still flips the guard to emit `Failed` on the error path.
        let guard =
            ExecutionGuard::new_silent(&metadata.model_id, "execute_streaming_with_context");
        let result =
            self.execute_streaming_with_context_impl(metadata, input, context, on_token, config);
        if let Err(e) = &result {
            mark_execution_terminal(&guard, e);
        }
        result
    }

    /// Internal implementation of execute_streaming_with_context — no telemetry listener events.
    #[allow(unused_variables)]
    fn execute_streaming_with_context_impl(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        context: &ConversationContext,
        on_token: StreamingCallback<'_>,
        config: Option<&GenerationConfig>,
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
            #[cfg(feature = "vision")]
            if let ExecutionTemplate::VisionLanguage {
                model_file,
                chat_template,
                context_length,
                ..
            } = &metadata.execution_template
            {
                debug!(
                    target: "xybrid_core",
                    "VisionLanguage model detected, converting context to multimodal messages for streaming"
                );

                let messages = Self::multimodal_messages_with_context(input, context)?;
                let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());

                let result = self.execute_llm_multimodal_streaming_messages(
                    metadata,
                    model_file,
                    chat_template.as_deref(),
                    *context_length,
                    &messages,
                    backend_hint,
                    on_token,
                    config,
                )?;

                return Ok(result.with_role(MessageRole::Assistant));
            }

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

                #[cfg(feature = "vision")]
                reject_text_only_model_image_input(metadata, input)?;

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
                    metadata,
                    model_file,
                    *context_length,
                    &chat_messages,
                    backend_hint,
                    on_token,
                    config,
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
            let mut result = self.execute_streaming_impl(metadata, input, on_token, config)?;
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
            self.execute_with_context_impl(metadata, input, context, config)
        }
    }

    /// Execute LLM inference with streaming.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    fn execute_llm_streaming(
        &mut self,
        metadata: &ModelMetadata,
        model_file: &str,
        chat_template: Option<&str>,
        context_length: usize,
        input: &Envelope,
        backend_hint: Option<&str>,
        on_token: StreamingCallback<'_>,
        config: Option<&GenerationConfig>,
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
        stamp_llm_span_cost_attribution(metadata);

        #[cfg(feature = "vision")]
        reject_text_only_model_image_input(metadata, input)?;

        // Build full model path
        let model_path = Path::new(&self.base_path).join(model_file);
        let model_path_str = model_path.to_string_lossy().to_string();
        let chat_template_path = resolve_optional_model_path(&self.base_path, chat_template);
        let cache_key = LlmAdapterCacheKey::new(
            model_path_str.clone(),
            chat_template_path.clone(),
            context_length,
            backend_hint,
            None,
        );

        // Check if we have a cached adapter for this exact load config.
        let need_load = match &self.llm_adapter_cache {
            Some((cached_key, _)) if cached_key == &cache_key => false,
            _ => true,
        };

        // Load model if needed
        if need_load {
            let mut config =
                LlmConfig::new(model_path_str.clone()).with_context_length(context_length);

            if let Some(template_path) = chat_template_path {
                config = config.with_chat_template(template_path);
            }

            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            adapter.load_model_with_config(&config)?;
            self.llm_adapter_cache = Some((cache_key.clone(), adapter));
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

        // Build generation config: explicit config wins, then envelope metadata, then defaults
        let gen_config = if let Some(cfg) = config {
            cfg.clone()
        } else {
            let mut cfg = GenerationConfig::default();
            if let Some(max_tokens) = input
                .metadata
                .get("max_tokens")
                .and_then(|s| s.parse().ok())
            {
                cfg.max_tokens = max_tokens;
            }
            if let Some(temperature) = input
                .metadata
                .get("temperature")
                .and_then(|s| s.parse().ok())
            {
                cfg.temperature = temperature;
            }
            cfg
        };

        // Execute with streaming. Capture the backend name + the prefix
        // length the backend reused from its KV cache so the metric
        // mirror can attach the resolved execution provider and the
        // local-cache-hit count to the wire payload.
        let (output, backend_name, cached_prefix) =
            if let Some((_, adapter)) = &self.llm_adapter_cache {
                // Overwrite the template-derived `backend` stamp with
                // the runtime that actually executes (mistral.rs vs
                // llama.cpp, selected by cargo feature, not by the
                // bundle metadata).
                stamp_llm_runtime_backend(adapter);
                let backend = adapter.backend();
                let out = backend.generate_streaming(&messages, &gen_config, on_token)?;
                let name = backend.name().to_string();
                let cached = backend.last_cached_prefix_len();
                (out, name, cached)
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
        response_metadata.insert("finish_reason".to_string(), output.finish_reason.clone());
        insert_llm_streaming_metrics(&mut response_metadata, &output);
        mirror_llm_metrics_to_span(&output, &backend_name, cached_prefix);

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
        metadata: &ModelMetadata,
        model_file: &str,
        context_length: usize,
        messages: &[ChatMessage],
        backend_hint: Option<&str>,
        config: Option<&GenerationConfig>,
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
        stamp_llm_span_cost_attribution(metadata);

        // Build full model path
        let model_path = Path::new(&self.base_path).join(model_file);
        let model_path_str = model_path.to_string_lossy().to_string();
        let cache_key = LlmAdapterCacheKey::new(
            model_path_str.clone(),
            None,
            context_length,
            backend_hint,
            None,
        );

        // Check if we have a cached adapter for this exact load config.
        let need_load = match &self.llm_adapter_cache {
            Some((cached_key, _)) if cached_key == &cache_key => false,
            _ => true,
        };

        // Load model if needed
        if need_load {
            let config = LlmConfig::new(model_path_str.clone()).with_context_length(context_length);

            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            adapter.load_model_with_config(&config)?;
            self.llm_adapter_cache = Some((cache_key.clone(), adapter));
        }

        // Use explicit config or fall back to defaults
        let gen_config = config.cloned().unwrap_or_default();

        // Execute with ChatMessages directly — backend applies template once.
        // Capture backend name + cached-prefix length for the metric mirror.
        let (output, backend_name, cached_prefix) =
            if let Some((_, adapter)) = &self.llm_adapter_cache {
                // Overwrite the template-derived `backend` stamp with
                // the runtime that actually executes (mistral.rs vs
                // llama.cpp, selected by cargo feature, not by the
                // bundle metadata).
                stamp_llm_runtime_backend(adapter);
                let backend = adapter.backend();
                let out = backend.generate(messages, &gen_config)?;
                let name = backend.name().to_string();
                let cached = backend.last_cached_prefix_len();
                (out, name, cached)
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
        response_metadata.insert("finish_reason".to_string(), output.finish_reason.clone());
        insert_llm_streaming_metrics(&mut response_metadata, &output);
        mirror_llm_metrics_to_span(&output, &backend_name, cached_prefix);

        Ok(Envelope {
            kind: EnvelopeKind::Text(output.text),
            metadata: response_metadata,
        })
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    fn execute_vision_language(
        &mut self,
        metadata: &ModelMetadata,
        model_file: &str,
        chat_template: Option<&str>,
        context_length: usize,
        input: &Envelope,
        backend_hint: Option<&str>,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        let messages = vec![MultimodalChatMessage::from_envelope(input)?];
        self.execute_llm_multimodal_messages(
            metadata,
            model_file,
            chat_template,
            context_length,
            &messages,
            backend_hint,
            config,
        )
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    fn vision_language_llm_config(
        &self,
        metadata: &ModelMetadata,
        model_path_str: String,
        chat_template: Option<&str>,
        context_length: usize,
    ) -> LlmConfig {
        let mut llm_config = LlmConfig::new(model_path_str).with_context_length(context_length);
        if let Some(template) = chat_template {
            let template_path = Path::new(&self.base_path).join(template);
            llm_config = llm_config.with_chat_template(template_path.to_string_lossy().to_string());
        }
        if let Some(vision_encoder) = metadata.vision_encoder.as_ref() {
            let vision_encoder_path = Path::new(&self.base_path).join(&vision_encoder.file);
            llm_config =
                llm_config.with_vision_encoder(vision_encoder_path.to_string_lossy().to_string());
        }
        llm_config
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    fn execute_llm_multimodal_messages(
        &mut self,
        metadata: &ModelMetadata,
        model_file: &str,
        chat_template: Option<&str>,
        context_length: usize,
        messages: &[MultimodalChatMessage],
        backend_hint: Option<&str>,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        info!(
            target: "xybrid_core",
            "Executing vision-language LLM with {} multimodal message(s): {} (backend: {:?})",
            messages.len(),
            model_file,
            backend_hint.unwrap_or("default")
        );

        let _llm_span = xybrid_trace::SpanGuard::new("vlm_inference_with_messages");
        xybrid_trace::add_metadata("model", model_file);
        xybrid_trace::add_metadata("message_count", messages.len().to_string());
        let image_count: usize = messages
            .iter()
            .map(MultimodalChatMessage::image_count)
            .sum();
        xybrid_trace::add_metadata("image_count", image_count.to_string());
        stamp_llm_span_cost_attribution(metadata);

        let model_path = Path::new(&self.base_path).join(model_file);
        let model_path_str = model_path.to_string_lossy().to_string();
        let chat_template_path = resolve_optional_model_path(&self.base_path, chat_template);
        let vision_encoder_path = metadata.vision_encoder.as_ref().map(|vision_encoder| {
            Path::new(&self.base_path)
                .join(&vision_encoder.file)
                .to_string_lossy()
                .to_string()
        });
        let cache_key = LlmAdapterCacheKey::new(
            model_path_str.clone(),
            chat_template_path,
            context_length,
            backend_hint,
            vision_encoder_path,
        );

        let gen_config = config.cloned().unwrap_or_default();

        let need_load = match &self.llm_adapter_cache {
            Some((cached_key, _)) if cached_key == &cache_key => false,
            _ => true,
        };

        if need_load {
            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            stamp_llm_runtime_backend(&adapter);
            ensure_backend_supports_vision(metadata, &adapter)?;

            let llm_config = self.vision_language_llm_config(
                metadata,
                model_path_str.clone(),
                chat_template,
                context_length,
            );

            adapter.load_model_with_config(&llm_config)?;
            self.llm_adapter_cache = Some((cache_key.clone(), adapter));
        }

        if let Some((_, adapter)) = &self.llm_adapter_cache {
            ensure_backend_supports_vision(metadata, adapter)?;
        }

        let registered_image_preprocess_ms =
            self.encode_registered_vision_inputs(metadata, messages)?;

        let (output, backend_name, cached_prefix) =
            if let Some((_, adapter)) = &self.llm_adapter_cache {
                stamp_llm_runtime_backend(adapter);
                let backend = adapter.backend();
                let out = backend.generate_multimodal(messages, &gen_config)?;
                let name = backend.name().to_string();
                let cached = backend.last_cached_prefix_len();
                (out, name, cached)
            } else {
                return Err(AdapterError::RuntimeError(
                    "LLM adapter cache unexpectedly empty".to_string(),
                ));
            };

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
        response_metadata.insert("finish_reason".to_string(), output.finish_reason.clone());
        insert_llm_streaming_metrics(&mut response_metadata, &output);
        insert_image_preprocess_metric(&mut response_metadata, registered_image_preprocess_ms);
        mirror_llm_metrics_to_span(&output, &backend_name, cached_prefix);

        Ok(Envelope {
            kind: EnvelopeKind::Text(output.text),
            metadata: response_metadata,
        })
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    fn execute_vision_language_streaming(
        &mut self,
        metadata: &ModelMetadata,
        model_file: &str,
        chat_template: Option<&str>,
        context_length: usize,
        input: &Envelope,
        backend_hint: Option<&str>,
        on_token: StreamingCallback<'_>,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        let messages = vec![MultimodalChatMessage::from_envelope(input)?];
        self.execute_llm_multimodal_streaming_messages(
            metadata,
            model_file,
            chat_template,
            context_length,
            &messages,
            backend_hint,
            on_token,
            config,
        )
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    fn execute_llm_multimodal_streaming_messages(
        &mut self,
        metadata: &ModelMetadata,
        model_file: &str,
        chat_template: Option<&str>,
        context_length: usize,
        messages: &[MultimodalChatMessage],
        backend_hint: Option<&str>,
        on_token: StreamingCallback<'_>,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        info!(
            target: "xybrid_core",
            "Executing vision-language streaming LLM with {} multimodal message(s): {} (backend: {:?})",
            messages.len(),
            model_file,
            backend_hint.unwrap_or("default")
        );

        let _llm_span = xybrid_trace::SpanGuard::new("vlm_inference_streaming_with_messages");
        xybrid_trace::add_metadata("model", model_file);
        xybrid_trace::add_metadata("message_count", messages.len().to_string());
        let image_count: usize = messages
            .iter()
            .map(MultimodalChatMessage::image_count)
            .sum();
        xybrid_trace::add_metadata("image_count", image_count.to_string());
        stamp_llm_span_cost_attribution(metadata);

        let model_path = Path::new(&self.base_path).join(model_file);
        let model_path_str = model_path.to_string_lossy().to_string();
        let chat_template_path = resolve_optional_model_path(&self.base_path, chat_template);
        let vision_encoder_path = metadata.vision_encoder.as_ref().map(|vision_encoder| {
            Path::new(&self.base_path)
                .join(&vision_encoder.file)
                .to_string_lossy()
                .to_string()
        });
        let cache_key = LlmAdapterCacheKey::new(
            model_path_str.clone(),
            chat_template_path,
            context_length,
            backend_hint,
            vision_encoder_path,
        );

        let gen_config = config.cloned().unwrap_or_default();

        let need_load = match &self.llm_adapter_cache {
            Some((cached_key, _)) if cached_key == &cache_key => false,
            _ => true,
        };

        if need_load {
            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            stamp_llm_runtime_backend(&adapter);
            ensure_backend_supports_vision(metadata, &adapter)?;

            let llm_config = self.vision_language_llm_config(
                metadata,
                model_path_str.clone(),
                chat_template,
                context_length,
            );

            adapter.load_model_with_config(&llm_config)?;
            self.llm_adapter_cache = Some((cache_key.clone(), adapter));
        }

        if let Some((_, adapter)) = &self.llm_adapter_cache {
            ensure_backend_supports_vision(metadata, adapter)?;
        }

        let registered_image_preprocess_ms =
            self.encode_registered_vision_inputs(metadata, messages)?;

        let (output, backend_name, cached_prefix) =
            if let Some((_, adapter)) = &self.llm_adapter_cache {
                stamp_llm_runtime_backend(adapter);
                let backend = adapter.backend();
                let out = backend.generate_multimodal_streaming(messages, &gen_config, on_token)?;
                let name = backend.name().to_string();
                let cached = backend.last_cached_prefix_len();
                (out, name, cached)
            } else {
                return Err(AdapterError::RuntimeError(
                    "LLM adapter cache unexpectedly empty".to_string(),
                ));
            };

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
        response_metadata.insert("finish_reason".to_string(), output.finish_reason.clone());
        insert_llm_streaming_metrics(&mut response_metadata, &output);
        insert_image_preprocess_metric(&mut response_metadata, registered_image_preprocess_ms);
        mirror_llm_metrics_to_span(&output, &backend_name, cached_prefix);

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
        metadata: &ModelMetadata,
        model_file: &str,
        context_length: usize,
        messages: &[ChatMessage],
        backend_hint: Option<&str>,
        on_token: StreamingCallback<'_>,
        config: Option<&GenerationConfig>,
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
        stamp_llm_span_cost_attribution(metadata);

        // Build full model path
        let model_path = Path::new(&self.base_path).join(model_file);
        let model_path_str = model_path.to_string_lossy().to_string();
        let cache_key = LlmAdapterCacheKey::new(
            model_path_str.clone(),
            None,
            context_length,
            backend_hint,
            None,
        );

        // Check if we have a cached adapter for this exact load config.
        let need_load = match &self.llm_adapter_cache {
            Some((cached_key, _)) if cached_key == &cache_key => false,
            _ => true,
        };

        // Load model if needed
        if need_load {
            let config = LlmConfig::new(model_path_str.clone()).with_context_length(context_length);

            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            adapter.load_model_with_config(&config)?;
            self.llm_adapter_cache = Some((cache_key.clone(), adapter));
        }

        // Use explicit config or fall back to defaults
        let gen_config = config.cloned().unwrap_or_default();

        // Execute with streaming - pass ChatMessages directly to backend.
        // Capture backend name + cached-prefix length for the metric mirror.
        let (output, backend_name, cached_prefix) =
            if let Some((_, adapter)) = &self.llm_adapter_cache {
                // Overwrite the template-derived `backend` stamp with
                // the runtime that actually executes (mistral.rs vs
                // llama.cpp, selected by cargo feature, not by the
                // bundle metadata).
                stamp_llm_runtime_backend(adapter);
                let backend = adapter.backend();
                let out = backend.generate_streaming(messages, &gen_config, on_token)?;
                let name = backend.name().to_string();
                let cached = backend.last_cached_prefix_len();
                (out, name, cached)
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
        response_metadata.insert("finish_reason".to_string(), output.finish_reason.clone());
        insert_llm_streaming_metrics(&mut response_metadata, &output);
        mirror_llm_metrics_to_span(&output, &backend_name, cached_prefix);

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
        metadata: &ModelMetadata,
        model_file: &str,
        chat_template: Option<&str>,
        context_length: usize,
        input: &Envelope,
        backend_hint: Option<&str>,
        config: Option<&GenerationConfig>,
    ) -> ExecutorResult<Envelope> {
        info!(
            target: "xybrid_core",
            "Executing LLM inference: {} (backend: {:?})",
            model_file,
            backend_hint.unwrap_or("default")
        );

        let _llm_span = xybrid_trace::SpanGuard::new("llm_inference");
        xybrid_trace::add_metadata("model", model_file);
        // Stamp the canonical `backend` + `quantization` labels onto the
        // inner LLM span. The SDK telemetry hoist reads from any span in
        // the trace; the outer `execute:<model_id>` span set up by
        // `execute_impl` also carries these, but the chat-context
        // dispatch path bypasses that outer span entirely, so stamping
        // here is what makes the wire shape consistent across both
        // entry points.
        stamp_llm_span_cost_attribution(metadata);

        #[cfg(feature = "vision")]
        reject_text_only_model_image_input(metadata, input)?;

        // Build full model path
        let model_path = Path::new(&self.base_path).join(model_file);
        let model_path_str = model_path.to_string_lossy().to_string();
        let chat_template_path = resolve_optional_model_path(&self.base_path, chat_template);
        let cache_key = LlmAdapterCacheKey::new(
            model_path_str.clone(),
            chat_template_path.clone(),
            context_length,
            backend_hint,
            None,
        );

        // Check if we have a cached adapter for this exact load config.
        let need_load = match &self.llm_adapter_cache {
            Some((cached_key, _)) if cached_key == &cache_key => {
                info!(target: "xybrid_core", "Reusing cached LLM adapter for: {}", model_path_str);
                false
            }
            Some((cached_key, _)) => {
                info!(
                    target: "xybrid_core",
                    "LLM adapter cache key changed (cached model {}, requested model {}), loading new model",
                    cached_key.model_path,
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

            if let Some(template_path) = chat_template_path {
                config = config.with_chat_template(template_path);
            }

            // Create adapter with the appropriate backend based on hint
            let mut adapter = LlmRuntimeAdapter::with_backend_hint(backend_hint)?;
            adapter.load_model_with_config(&config)?;

            // Cache the adapter
            self.llm_adapter_cache = Some((cache_key.clone(), adapter));
        }

        // Build generation config: explicit config wins, then envelope metadata, then defaults
        let gen_config = if let Some(cfg) = config {
            cfg.clone()
        } else {
            let mut cfg = GenerationConfig::default();
            if let Some(max_tokens) = input
                .metadata
                .get("max_tokens")
                .and_then(|s| s.parse().ok())
            {
                cfg.max_tokens = max_tokens;
            }
            if let Some(temperature) = input
                .metadata
                .get("temperature")
                .and_then(|s| s.parse().ok())
            {
                cfg.temperature = temperature;
            }
            cfg
        };

        // Build messages from input
        let prompt = match &input.kind {
            EnvelopeKind::Text(text) => text.clone(),
            _ => {
                return Err(AdapterError::InvalidInput(
                    "LLM requires text input".to_string(),
                ))
            }
        };

        let system_prompt = input.metadata.get("system_prompt").map(|s| s.as_str());
        let mut messages = Vec::new();
        if let Some(sys) = system_prompt {
            messages.push(ChatMessage::system(sys));
        }
        messages.push(ChatMessage::user(&prompt));

        // Execute inference using cached adapter's backend directly.
        // Capture backend name + cached-prefix length for the metric mirror.
        let (output, backend_name, cached_prefix) =
            if let Some((_, adapter)) = &self.llm_adapter_cache {
                // Overwrite the template-derived `backend` stamp with
                // the runtime that actually executes (mistral.rs vs
                // llama.cpp, selected by cargo feature, not by the
                // bundle metadata).
                stamp_llm_runtime_backend(adapter);
                let backend = adapter.backend();
                let out = backend.generate(&messages, &gen_config)?;
                let name = backend.name().to_string();
                let cached = backend.last_cached_prefix_len();
                (out, name, cached)
            } else {
                return Err(AdapterError::RuntimeError(
                    "LLM adapter cache unexpectedly empty".to_string(),
                ));
            };

        info!(
            target: "xybrid_core",
            "LLM inference complete"
        );

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
        response_metadata.insert("finish_reason".to_string(), output.finish_reason.clone());
        insert_llm_streaming_metrics(&mut response_metadata, &output);
        mirror_llm_metrics_to_span(&output, &backend_name, cached_prefix);

        Ok(Envelope {
            kind: EnvelopeKind::Text(output.text),
            metadata: response_metadata,
        })
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
            // Audio decode is I/O-dominant (WAV parse → PCM samples); the
            // rest (Tokenize / Phonemize / MelSpectrogram / …) is pure CPU.
            xybrid_trace::add_metadata(
                "span_kind",
                if step_name.eq_ignore_ascii_case("audiodecode") {
                    "io"
                } else {
                    "cpu"
                },
            );

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
            // TTSAudioEncode writes raw PCM → WAV (I/O). Everything else
            // (CTCDecode / ArgMax / Softmax / …) is pure CPU.
            xybrid_trace::add_metadata(
                "span_kind",
                if step_name.eq_ignore_ascii_case("ttsaudioencode") {
                    "io"
                } else {
                    "cpu"
                },
            );

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
    /// Returns the cached TTS ONNX session for `model_path`, building and caching
    /// it on a miss. Caching skips the per-call `commit_from_file` + Level-3 graph
    /// optimization — the dominant latency for short TTS. The cache invalidates if
    /// the model file's `(len, mtime)` changes. TTS always runs on CPU with
    /// default options, so path + file identity is a sufficient key.
    fn tts_session(&mut self, model_path: &Path) -> ExecutorResult<Arc<ONNXSession>> {
        let identity = tts_file_identity(model_path);
        if let Some(cache) = &self.tts_session_cache {
            // Reuse only when the path matches AND the file identity is both
            // readable and unchanged — an unreadable identity (`None`) forces a
            // rebuild rather than trusting a possibly-stale session.
            if cache.model_path == model_path
                && identity.is_some()
                && cache.file_identity == identity
            {
                return Ok(Arc::clone(&cache.session));
            }
        }
        let session = Arc::new(OnnxSessionFactory::create_session(
            model_path,
            ExecutionProviderKind::Cpu,
            SessionOptions::default(),
        )?);
        self.tts_session_cache = Some(TtsSessionCache {
            model_path: model_path.to_path_buf(),
            file_identity: identity,
            session: Arc::clone(&session),
        });
        Ok(session)
    }

    fn execute_tts_chunked(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        model_path: &Path,
    ) -> ExecutorResult<Envelope> {
        use crate::ir::EnvelopeKind;

        // Use model-specific chunk limit if set, otherwise default
        const DEFAULT_MAX_TTS_CHARS: usize = 350;
        let max_tts_chars = metadata.max_chunk_chars.unwrap_or(DEFAULT_MAX_TTS_CHARS);

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
            "TTS Chunked: Input text length: {} chars (max_chunk_chars={})",
            text.len(),
            max_tts_chars
        );

        // Check if chunking is needed
        if text.len() <= max_tts_chars {
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
        let chunks = Self::chunk_text_for_tts(&text, max_tts_chars);
        debug!(target: "xybrid_core", "TTS: Split into {} chunks", chunks.len());

        // Process each chunk and collect audio
        // Crossfade length: 480 samples (~20ms at 24kHz)
        const CROSSFADE_SAMPLES: usize = 480;

        let mut audio_chunks: Vec<Vec<f32>> = Vec::new();
        // Reuse the cached session across runs (and across chunks within this
        // run); the graph load is paid once, not per call.
        let session = self.tts_session(model_path)?;
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
                let mut chunk_audio: Vec<f32> = audio_tensor.iter().cloned().collect();

                // Trim trailing samples per chunk to remove artifacts
                // (KittenTTS upstream trims 5000 samples ≈ 208ms at 24kHz per chunk)
                let trim_count = metadata.trim_trailing_samples.unwrap_or(0);
                if trim_count > 0 && chunk_audio.len() > trim_count {
                    chunk_audio.truncate(chunk_audio.len() - trim_count);
                }

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

        // Reuse the cached TTS session — the first run builds it, later runs skip
        // the graph load + Level-3 optimization (the dominant short-TTS latency).
        let session = self.tts_session(model_path)?;
        let speed = extract_tts_speed(input);
        let mut raw_outputs = execute_tts_inference(&session, phoneme_ids, voice_embedding, speed)?;

        // Trim trailing samples to remove artifacts (same logic as chunked path)
        let trim_count = metadata.trim_trailing_samples.unwrap_or(0);
        if trim_count > 0 {
            for audio in raw_outputs.values_mut() {
                let len = audio.len();
                if len > trim_count {
                    audio.slice_collapse(ndarray::s![..len - trim_count]);
                }
            }
        }

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

    /// Resolve the on-disk model file path for a single-model (TTS) template,
    /// the same way `execute()` does (`base_path` + the template's `model_file`).
    fn tts_model_path(&self, metadata: &ModelMetadata) -> ExecutorResult<PathBuf> {
        let model_file = match &metadata.execution_template {
            ExecutionTemplate::Onnx { model_file } => model_file.clone(),
            ExecutionTemplate::CoreMl { model_file } => model_file.clone(),
            ExecutionTemplate::TfLite { model_file } => model_file.clone(),
            ExecutionTemplate::SafeTensors { model_file, .. } => model_file.clone(),
            _ => {
                return Err(AdapterError::InvalidInput(
                    "Streaming TTS requires a single-model execution template".to_string(),
                ))
            }
        };
        Ok(Path::new(&self.base_path).join(model_file))
    }

    /// Output sample rate of the model's `TTSAudioEncode` step, or 24000 Hz when
    /// absent. Streamed chunks carry it so the consumer wraps each chunk's PCM in
    /// a correctly-headed WAV.
    fn tts_output_sample_rate(metadata: &ModelMetadata) -> u32 {
        use super::template::PostprocessingStep;
        metadata
            .postprocessing
            .iter()
            .find_map(|step| match step {
                PostprocessingStep::TTSAudioEncode { sample_rate, .. } => Some(*sample_rate),
                _ => None,
            })
            .unwrap_or(24000)
    }

    /// Streaming TTS: synthesize the text sentence-chunk by sentence-chunk and
    /// hand each chunk's PCM to `on_chunk` as it's produced, instead of
    /// concatenating the whole utterance (see `execute_tts_chunked`). For long
    /// text this lets playback start after the first sentence.
    ///
    /// `on_chunk(pcm, sample_rate)` returns `false` to stop early (barge-in /
    /// sink closed) — the next chunk's inference is skipped. Postprocessing runs
    /// per chunk: `TTSAudioEncode` normalizes each chunk to the same target RMS,
    /// so chunks stay at a consistent level (no inter-sentence drift); a short
    /// edge fade masks the per-chunk high-pass startup transient and the seam
    /// where consecutive chunks are spliced at playback. The crossfade used by
    /// the batch path is dropped (there is no in-Rust concatenation here).
    pub fn execute_tts_streaming(
        &mut self,
        metadata: &ModelMetadata,
        input: &Envelope,
        on_chunk: &mut dyn FnMut(Vec<u8>, u32) -> bool,
    ) -> ExecutorResult<()> {
        use crate::ir::EnvelopeKind;

        const DEFAULT_MAX_TTS_CHARS: usize = 350;
        let max_tts_chars = metadata.max_chunk_chars.unwrap_or(DEFAULT_MAX_TTS_CHARS);

        let text = match &input.kind {
            EnvelopeKind::Text(t) => t.clone(),
            _ => {
                return Err(AdapterError::InvalidInput(
                    "TTS requires text input".to_string(),
                ))
            }
        };
        if text.trim().is_empty() {
            return Ok(());
        }

        let model_path = self.tts_model_path(metadata)?;
        let sample_rate = Self::tts_output_sample_rate(metadata);
        let fade_samples = (sample_rate as usize * 5) / 1000; // ~5ms edge fade
        let chunks = Self::chunk_text_for_tts(&text, max_tts_chars);
        let session = self.tts_session(&model_path)?;
        let speed = extract_tts_speed(input);
        let output_name = session
            .output_names()
            .first()
            .cloned()
            .unwrap_or_else(|| "audio".to_string());

        for (i, chunk) in chunks.iter().enumerate() {
            debug!(target: "xybrid_core", "TTS stream: chunk {}/{} ({} chars)", i + 1, chunks.len(), chunk.len());

            let chunk_input = Envelope {
                kind: EnvelopeKind::Text(chunk.clone()),
                metadata: input.metadata.clone(),
            };
            let preprocessed = self.run_preprocessing(metadata, &chunk_input)?;
            let phoneme_ids = preprocessed
                .as_phoneme_ids()
                .ok_or_else(|| AdapterError::InvalidInput("Expected phoneme IDs".to_string()))?;
            // Voice embedding is the same for every chunk; load it per chunk to
            // mirror the batch path (cheap read from voices.bin).
            let voice_loader = TtsVoiceLoader::new(&self.base_path);
            let voice_embedding = voice_loader.load(metadata, input)?;
            let raw_outputs =
                execute_tts_inference(&session, phoneme_ids, voice_embedding, speed)?;

            let Some(audio_tensor) = raw_outputs.values().next() else {
                continue;
            };
            let mut chunk_audio: Vec<f32> = audio_tensor.iter().cloned().collect();
            let trim_count = metadata.trim_trailing_samples.unwrap_or(0);
            if trim_count > 0 && chunk_audio.len() > trim_count {
                chunk_audio.truncate(chunk_audio.len() - trim_count);
            }

            // Per-chunk postprocessing → PCM bytes (mirrors the batch path, but
            // applied to this chunk rather than the concatenated buffer).
            let mut outputs: HashMap<String, ArrayD<f32>> = HashMap::new();
            outputs.insert(output_name.clone(), ndarray::Array1::from_vec(chunk_audio).into_dyn());
            let env = self.run_postprocessing(metadata, RawOutputs::TensorMap(outputs))?;
            let mut pcm = match env.kind {
                EnvelopeKind::Audio(bytes) => bytes,
                _ => {
                    return Err(AdapterError::InvalidInput(
                        "TTS postprocessing did not yield audio".to_string(),
                    ))
                }
            };
            fade_pcm16_edges(&mut pcm, fade_samples);

            if !on_chunk(pcm, sample_rate) {
                debug!(target: "xybrid_core", "TTS stream: stopped early at chunk {}", i + 1);
                break;
            }
        }
        Ok(())
    }
}

/// Apply a short linear fade-in and fade-out over `fade_samples` samples at each
/// edge of a 16-bit little-endian PCM buffer. In streaming TTS this masks the
/// per-chunk high-pass-filter startup transient and the seam where consecutive
/// chunks are spliced gaplessly at playback. No-op for an empty buffer or zero
/// fade; the fade is clamped so the two ramps never overlap.
fn fade_pcm16_edges(pcm: &mut [u8], fade_samples: usize) {
    let total = pcm.len() / 2;
    let n = fade_samples.min(total / 2);
    if n == 0 {
        return;
    }
    let scale = |pcm: &mut [u8], idx: usize, gain: f32| {
        let o = idx * 2;
        let s = i16::from_le_bytes([pcm[o], pcm[o + 1]]);
        let v = (s as f32 * gain).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        let b = v.to_le_bytes();
        pcm[o] = b[0];
        pcm[o + 1] = b[1];
    };
    for i in 0..n {
        let gain = (i as f32 + 0.5) / n as f32;
        scale(pcm, i, gain); // fade-in at the head
        scale(pcm, total - 1 - i, gain); // fade-out at the tail
    }
}

/// `(len, modified)` of the file at `path`, or `None` if the metadata can't be
/// read. Used as the TTS session-cache identity so a model file replaced in
/// place invalidates the cached session.
fn tts_file_identity(path: &Path) -> Option<(u64, std::time::SystemTime)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.len(), meta.modified().ok()?))
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

// =============================================================================
// LLM streaming-telemetry helpers
// =============================================================================
//
// The executor's four LLM execution paths (`execute_llm`,
// `execute_llm_streaming`, `execute_llm_with_messages`,
// `execute_llm_streaming_with_messages`) all call
// `adapter.backend().generate(...)` directly, bypassing
// `runtime_adapter::llm::LlmRuntimeAdapter::execute()` where the telemetry
// extraction originally lived. These helpers re-apply the same contract at
// the executor layer so all four paths surface the 9 LLM scalars the
// consuming analytics backend expects on every LLM span.

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn insert_llm_streaming_metrics(
    response_metadata: &mut HashMap<String, String>,
    output: &crate::runtime_adapter::llm::GenerationOutput,
) {
    if let Some(v) = output.ttft_ms {
        response_metadata.insert("ttft_ms".to_string(), v.to_string());
    }
    if let Some(v) = output.mean_itl_ms {
        response_metadata.insert("mean_itl_ms".to_string(), format!("{:.4}", v));
    }
    if let Some(v) = output.p95_itl_ms {
        response_metadata.insert("p95_itl_ms".to_string(), v.to_string());
    }
    if let Some(v) = output.emitted_chunks {
        response_metadata.insert("emitted_chunks".to_string(), v.to_string());
    }
    if let Some(v) = output.decode_tps {
        response_metadata.insert("decode_tps".to_string(), format!("{:.4}", v));
    }
    if let Some(v) = output.prefill_tps {
        response_metadata.insert("prefill_tps".to_string(), format!("{:.4}", v));
    }
    insert_image_preprocess_metric(response_metadata, output.image_preprocess_ms);
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn insert_image_preprocess_metric(
    response_metadata: &mut HashMap<String, String>,
    image_preprocess_ms: Option<u32>,
) {
    if let Some(v) = image_preprocess_ms {
        response_metadata.insert("image_preprocess_ms".to_string(), v.to_string());
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
fn mirror_llm_metrics_to_span(
    output: &crate::runtime_adapter::llm::GenerationOutput,
    backend_name: &str,
    cached_prefix_tokens: Option<usize>,
) {
    // Always-present scalars. These reach the platform via
    // `PlatformEvent.stages[].spans[].metadata` (populated by
    // `xybrid_core::tracing::add_metadata` on the currently active span).
    xybrid_trace::add_metadata("tokens_generated", output.tokens_generated.to_string());
    xybrid_trace::add_metadata("generation_time_ms", output.generation_time_ms.to_string());
    xybrid_trace::add_metadata(
        "tokens_per_second",
        format!("{:.2}", output.tokens_per_second),
    );
    xybrid_trace::add_metadata("finish_reason", &output.finish_reason);

    // Resolved execution provider: which on-device engine path actually
    // ran. Cost-attribution telemetry uses this to explain latency
    // variance on the same chip + model. Sourced from build flags via
    // `local_execution_provider` because backend selection is compile-
    // time. Cloud LLMs go through a different adapter and get
    // attribution from the `provider` field instead.
    xybrid_trace::add_metadata(
        "execution_provider",
        crate::runtime_adapter::llm::local_execution_provider(backend_name),
    );

    // Local KV cache hits: how many prompt tokens this call reused from
    // the cache the previous turn left behind. Only emit when positive
    // — `Some(0)` means a first turn or a totally divergent prompt
    // (telemetry should look like a non-cached call), and `None` means
    // the backend doesn't track prefix reuse at all (cloud, mistralrs,
    // mock test backends). The local mirror of cloud's
    // `cache_read_input_tokens` so analytics can stack them on the
    // same axis.
    if let Some(n) = cached_prefix_tokens {
        if n > 0 {
            xybrid_trace::add_metadata("prompt_cached_tokens", n.to_string());
        }
    }

    // Streaming-derived scalars. Only mirror when the backend reported them;
    // the `Option<_>` + `nonzero` filter in mistral keeps misleading zeros
    // out of the dashboard.
    if let Some(v) = output.ttft_ms {
        xybrid_trace::add_metadata("ttft_ms", v.to_string());
    }
    if let Some(v) = output.mean_itl_ms {
        xybrid_trace::add_metadata("mean_itl_ms", format!("{:.4}", v));
    }
    if let Some(v) = output.p95_itl_ms {
        xybrid_trace::add_metadata("p95_itl_ms", v.to_string());
    }
    if let Some(v) = output.emitted_chunks {
        xybrid_trace::add_metadata("emitted_chunks", v.to_string());
    }
    if let Some(v) = output.decode_tps {
        xybrid_trace::add_metadata("decode_tps", format!("{:.4}", v));
    }
    if let Some(v) = output.prefill_tps {
        xybrid_trace::add_metadata("prefill_tps", format!("{:.4}", v));
    }
}

#[cfg(test)]
mod tests {
    use super::super::template::PreprocessingStep;
    use super::*;
    use crate::ir::EnvelopeKind;

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    struct TextOnlyVisionBoundaryBackend;

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    impl crate::runtime_adapter::LlmBackend for TextOnlyVisionBoundaryBackend {
        fn name(&self) -> &str {
            "text-only"
        }

        fn supported_formats(&self) -> Vec<&'static str> {
            vec!["gguf"]
        }

        fn load(&mut self, _config: &crate::runtime_adapter::LlmConfig) -> ExecutorResult<()> {
            Ok(())
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn unload(&mut self) -> ExecutorResult<()> {
            Ok(())
        }

        fn generate(
            &self,
            _messages: &[crate::runtime_adapter::ChatMessage],
            _config: &GenerationConfig,
        ) -> ExecutorResult<crate::runtime_adapter::GenerationOutput> {
            unreachable!("VisionLanguage should use the multimodal backend path")
        }

        fn generate_raw(
            &self,
            _prompt: &str,
            _config: &GenerationConfig,
        ) -> ExecutorResult<crate::runtime_adapter::GenerationOutput> {
            unreachable!("VisionLanguage should use the multimodal backend path")
        }

        fn generate_multimodal(
            &self,
            _messages: &[crate::runtime_adapter::MultimodalChatMessage],
            _config: &GenerationConfig,
        ) -> ExecutorResult<crate::runtime_adapter::GenerationOutput> {
            unreachable!("non-vision backends must fail capability checks before generation")
        }

        fn generate_multimodal_streaming(
            &self,
            _messages: &[crate::runtime_adapter::MultimodalChatMessage],
            _config: &GenerationConfig,
            _on_token: StreamingCallback<'_>,
        ) -> ExecutorResult<crate::runtime_adapter::GenerationOutput> {
            unreachable!("non-vision backends must fail capability checks before streaming")
        }
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    fn cache_text_only_boundary_backend(
        executor: &mut TemplateExecutor,
        model_path: &str,
        vision_encoder_path: Option<&str>,
    ) {
        executor.llm_adapter_cache = Some((
            LlmAdapterCacheKey::new(
                model_path.to_string(),
                None,
                4096,
                None,
                vision_encoder_path.map(ToOwned::to_owned),
            ),
            crate::runtime_adapter::LlmRuntimeAdapter::with_backend(Box::new(
                TextOnlyVisionBoundaryBackend,
            )),
        ));
    }

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

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    #[test]
    fn vision_language_execute_reaches_multimodal_backend_boundary() {
        use crate::execution::template::{VisionEncoderConfig, VisionPreprocessingPreset};

        let metadata = ModelMetadata {
            model_id: "gemma-3n-e2b".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::VisionLanguage {
                model_file: "missing-model.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["missing-model.gguf".to_string(), "mmproj.gguf".to_string()],
            vision_encoder: Some(VisionEncoderConfig {
                file: "mmproj.gguf".to_string(),
                preprocessing_preset: VisionPreprocessingPreset::Gemma3Vision,
                image_size: 896,
                patch_size: Some(14),
            }),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };

        let image_bytes = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                2,
                2,
                image::Rgb([17, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        };
        let image = Envelope::image(image_bytes, "png").unwrap();
        let input = Envelope::user_message("Describe this image", vec![image]).unwrap();

        let mut executor = TemplateExecutor::with_base_path("/models");
        cache_text_only_boundary_backend(
            &mut executor,
            "/models/missing-model.gguf",
            Some("/models/mmproj.gguf"),
        );
        let err = executor.execute(&metadata, &input, None).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("does not support vision input"));
        assert!(!message.contains("not wired"));
        assert!(!message.contains("missing-model.gguf"));
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp-vision")
    ))]
    #[test]
    fn vision_language_does_not_reuse_text_only_cache_for_same_model_path() {
        use crate::execution::template::{VisionEncoderConfig, VisionPreprocessingPreset};

        let metadata = ModelMetadata {
            model_id: "lfm2-vl-450m".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::VisionLanguage {
                model_file: "missing-model.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["missing-model.gguf".to_string(), "mmproj.gguf".to_string()],
            vision_encoder: Some(VisionEncoderConfig {
                file: "mmproj.gguf".to_string(),
                preprocessing_preset: VisionPreprocessingPreset::SigLip,
                image_size: 512,
                patch_size: Some(16),
            }),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };

        let image_bytes = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                1,
                1,
                image::Rgb([17, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        };
        let image = Envelope::image(image_bytes, "png").unwrap();
        let input = Envelope::user_message("Describe this image", vec![image]).unwrap();

        let mut executor = TemplateExecutor::with_base_path("/models");
        cache_text_only_boundary_backend(&mut executor, "/models/missing-model.gguf", None);

        let err = executor.execute(&metadata, &input, None).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("/models/missing-model.gguf"),
            "VisionLanguage must reload with its mmproj-aware config instead of reusing a stale text-only cache: {message}"
        );
        assert!(
            !message.contains("does not support vision input"),
            "stale text-only cache was reused for a VisionLanguage call: {message}"
        );
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    #[test]
    fn vision_language_text_only_backend_reports_backend_capability_before_generation() {
        use crate::execution::template::{VisionEncoderConfig, VisionPreprocessingPreset};

        let metadata = ModelMetadata {
            model_id: "vlm-on-text-backend".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::VisionLanguage {
                model_file: "model.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["model.gguf".to_string(), "mmproj.gguf".to_string()],
            vision_encoder: Some(VisionEncoderConfig {
                file: "mmproj.gguf".to_string(),
                preprocessing_preset: VisionPreprocessingPreset::SigLip,
                image_size: 512,
                patch_size: Some(16),
            }),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };

        let image_bytes = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                1,
                1,
                image::Rgb([17, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        };
        let image = Envelope::image(image_bytes, "png").unwrap();
        let input = Envelope::user_message("Describe this image", vec![image]).unwrap();

        let mut executor = TemplateExecutor::with_base_path("/models");
        cache_text_only_boundary_backend(
            &mut executor,
            "/models/model.gguf",
            Some("/models/mmproj.gguf"),
        );

        match executor.execute(&metadata, &input, None).unwrap_err() {
            AdapterError::UnsupportedBackendCapability {
                model_id,
                backend,
                capability,
                hint,
            } => {
                assert_eq!(model_id, "vlm-on-text-backend");
                assert_eq!(backend, "text-only");
                assert_eq!(capability, "vision input");
                assert!(hint.contains("llm-llamacpp-vision"));
            }
            other => panic!("expected UnsupportedBackendCapability, got {other:?}"),
        }
    }

    /// Streaming variant of the text-only-backend rejection test. The
    /// streaming `execute_streaming_impl` has its own
    /// `ensure_backend_supports_vision()` call site separate from the
    /// batch path — without symmetric coverage, regressions in just
    /// one of the two paths slip through. Confirms that a Studio-style
    /// streaming VLM turn against a non-vision backend produces the
    /// same typed error as the batch path, before any tokens stream.
    ///
    /// **Scope**: this test exercises the cache-hit streaming guard at
    /// `execute_vision_language_streaming`'s cached-adapter branch.
    /// The cache-miss guard (model load path) and the
    /// `execute_streaming_with_context` multimodal entrance are not
    /// covered here; they are separate call sites that share the
    /// underlying `ensure_backend_supports_vision()` helper and would
    /// benefit from their own dedicated tests as the surface stabilises.
    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    #[test]
    fn vision_language_streaming_text_only_backend_reports_backend_capability_before_tokens() {
        use crate::execution::template::{VisionEncoderConfig, VisionPreprocessingPreset};

        let metadata = ModelMetadata {
            model_id: "vlm-on-text-backend-streaming".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::VisionLanguage {
                model_file: "model.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["model.gguf".to_string(), "mmproj.gguf".to_string()],
            vision_encoder: Some(VisionEncoderConfig {
                file: "mmproj.gguf".to_string(),
                preprocessing_preset: VisionPreprocessingPreset::SigLip,
                image_size: 512,
                patch_size: Some(16),
            }),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };

        let image_bytes = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                1,
                1,
                image::Rgb([17, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        };
        let image = Envelope::image(image_bytes, "png").unwrap();
        let input = Envelope::user_message("Describe this image", vec![image]).unwrap();

        let mut executor = TemplateExecutor::with_base_path("/models");
        cache_text_only_boundary_backend(
            &mut executor,
            "/models/model.gguf",
            Some("/models/mmproj.gguf"),
        );

        // Token sink that fails the test if any token reaches the user
        // — the capability gate must fire BEFORE the streaming loop.
        let tokens_seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let tokens_seen_clone = tokens_seen.clone();
        let on_token: StreamingCallback<'_> = Box::new(move |token| {
            tokens_seen_clone
                .lock()
                .expect("token sink lock")
                .push(token.token);
            Ok(())
        });

        match executor
            .execute_streaming(&metadata, &input, on_token, None)
            .unwrap_err()
        {
            AdapterError::UnsupportedBackendCapability {
                model_id,
                backend,
                capability,
                hint,
            } => {
                assert_eq!(model_id, "vlm-on-text-backend-streaming");
                assert_eq!(backend, "text-only");
                assert_eq!(capability, "vision input");
                assert!(hint.contains("llm-llamacpp-vision"));
            }
            other => panic!("expected UnsupportedBackendCapability, got {other:?}"),
        }
        assert!(
            tokens_seen.lock().unwrap().is_empty(),
            "no tokens may be emitted before the capability gate rejects the turn"
        );
    }

    #[cfg(all(feature = "vision", feature = "llm-llamacpp-vision"))]
    #[test]
    fn vision_language_missing_mmproj_reports_missing_artifact_before_model_parse() {
        use crate::execution::template::{VisionEncoderConfig, VisionPreprocessingPreset};

        let tempdir = tempfile::tempdir().unwrap();
        std::fs::write(tempdir.path().join("model.gguf"), b"not a real gguf").unwrap();

        let mut bundle_metadata = HashMap::new();
        bundle_metadata.insert("backend".to_string(), serde_json::json!("llamacpp"));
        let metadata = ModelMetadata {
            model_id: "vlm-missing-mmproj".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::VisionLanguage {
                model_file: "model.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["model.gguf".to_string(), "missing-mmproj.gguf".to_string()],
            vision_encoder: Some(VisionEncoderConfig {
                file: "missing-mmproj.gguf".to_string(),
                preprocessing_preset: VisionPreprocessingPreset::SigLip,
                image_size: 512,
                patch_size: Some(16),
            }),
            description: None,
            metadata: bundle_metadata,
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };

        let image_bytes = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                1,
                1,
                image::Rgb([17, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        };
        let image = Envelope::image(image_bytes, "png").unwrap();
        let input = Envelope::user_message("Describe this image", vec![image]).unwrap();

        let mut executor = TemplateExecutor::with_base_path(tempdir.path().to_str().unwrap());
        let err = executor.execute(&metadata, &input, None).unwrap_err();

        match err {
            AdapterError::MissingArtifact { artifact, path } => {
                assert_eq!(artifact, "vision_encoder");
                assert!(path.contains("missing-mmproj.gguf"));
            }
            other => panic!("expected MissingArtifact for missing mmproj, got {other:?}"),
        }
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    #[test]
    fn gguf_image_input_returns_text_only_model_error_before_load() {
        let metadata = ModelMetadata {
            model_id: "text-only-gguf".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::Gguf {
                model_file: "missing-text-only.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["missing-text-only.gguf".to_string()],
            vision_encoder: None,
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };

        let image_bytes = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                1,
                1,
                image::Rgb([17, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        };
        let image = Envelope::image(image_bytes, "png").unwrap();
        let input = Envelope::user_message("Describe this image", vec![image]).unwrap();

        let mut executor = TemplateExecutor::with_base_path("/models");
        match executor.execute(&metadata, &input, None).unwrap_err() {
            AdapterError::UnsupportedModelCapability {
                model_id,
                capability,
                hint,
            } => {
                assert_eq!(model_id, "text-only-gguf");
                assert_eq!(capability, "image input");
                assert!(hint.contains("VisionLanguage"));
                assert!(
                    !hint.contains("missing-text-only.gguf"),
                    "guard must not depend on the missing GGUF path: {hint}"
                );
            }
            other => panic!("expected UnsupportedModelCapability, got {other:?}"),
        }
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    #[test]
    fn vision_language_streaming_uses_multimodal_streaming_span() {
        use crate::execution::template::{VisionEncoderConfig, VisionPreprocessingPreset};
        let _trace_lock = crate::tracing::test_lock();

        let metadata = ModelMetadata {
            model_id: "gemma-3n-e2b".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::VisionLanguage {
                model_file: "missing-model.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["missing-model.gguf".to_string(), "mmproj.gguf".to_string()],
            vision_encoder: Some(VisionEncoderConfig {
                file: "mmproj.gguf".to_string(),
                preprocessing_preset: VisionPreprocessingPreset::Gemma3Vision,
                image_size: 896,
                patch_size: Some(14),
            }),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };

        let image_bytes = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                2,
                2,
                image::Rgb([17, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        };
        let image = Envelope::image(image_bytes, "png").unwrap();
        let input = Envelope::user_message("Describe this image", vec![image]).unwrap();

        crate::tracing::init_tracing(true);
        crate::tracing::reset_tracing();

        let mut executor = TemplateExecutor::with_base_path("/models");
        cache_text_only_boundary_backend(
            &mut executor,
            "/models/missing-model.gguf",
            Some("/models/mmproj.gguf"),
        );
        let err = executor
            .execute_streaming(&metadata, &input, Box::new(|_| Ok(())), None)
            .unwrap_err();
        assert!(err.to_string().contains("does not support vision input"));

        let json = crate::tracing::get_stages_json();
        crate::tracing::reset_tracing();
        let spans = json["spans"].as_array().unwrap();
        assert!(
            spans
                .iter()
                .any(|span| span["name"].as_str() == Some("vlm_inference_streaming_with_messages")),
            "VisionLanguage streaming must use the multimodal streaming path; spans={spans:?}"
        );
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    #[test]
    fn vision_language_registered_encoder_preprocesses_image_before_llm() {
        use crate::execution::template::{VisionEncoderConfig, VisionPreprocessingPreset};
        use crate::runtime_adapter::{
            BackendResult, GenerationOutput, LlmBackend, LlmConfig, LlmRuntimeAdapter,
            VisionEmbeddings, VisionEncoder,
        };
        use ndarray::{ArrayD, IxDyn};
        use std::sync::{Arc, Mutex};
        let _trace_lock = crate::tracing::test_lock();

        struct SpyVisionEncoder {
            events: Arc<Mutex<Vec<&'static str>>>,
            observed_shape: Arc<Mutex<Option<Vec<usize>>>>,
        }

        impl VisionEncoder for SpyVisionEncoder {
            fn encode(&mut self, image_tensor: ArrayD<f32>) -> BackendResult<VisionEmbeddings> {
                self.events.lock().unwrap().push("encoder");
                *self.observed_shape.lock().unwrap() = Some(image_tensor.shape().to_vec());
                Ok(VisionEmbeddings {
                    placeholder_tokens: vec![32000, 32001],
                    embeddings: ArrayD::zeros(IxDyn(&[2, 4])),
                })
            }
        }

        struct StubVisionBackend {
            events: Arc<Mutex<Vec<&'static str>>>,
            image_count: Arc<Mutex<Option<usize>>>,
        }

        impl LlmBackend for StubVisionBackend {
            fn name(&self) -> &str {
                "stub-vision"
            }

            fn supported_formats(&self) -> Vec<&'static str> {
                vec!["gguf"]
            }

            fn load(&mut self, _config: &LlmConfig) -> crate::runtime_adapter::LlmResult<()> {
                Ok(())
            }

            fn is_loaded(&self) -> bool {
                true
            }

            fn unload(&mut self) -> crate::runtime_adapter::LlmResult<()> {
                Ok(())
            }

            fn generate(
                &self,
                _messages: &[crate::runtime_adapter::ChatMessage],
                _config: &GenerationConfig,
            ) -> crate::runtime_adapter::LlmResult<GenerationOutput> {
                unreachable!("VisionLanguage should use the multimodal backend path")
            }

            fn generate_raw(
                &self,
                _prompt: &str,
                _config: &GenerationConfig,
            ) -> crate::runtime_adapter::LlmResult<GenerationOutput> {
                unreachable!("VisionLanguage should use the multimodal backend path")
            }

            fn supports_vision(&self) -> bool {
                true
            }

            fn generate_multimodal(
                &self,
                messages: &[crate::runtime_adapter::MultimodalChatMessage],
                _config: &GenerationConfig,
            ) -> crate::runtime_adapter::LlmResult<GenerationOutput> {
                self.events.lock().unwrap().push("llm");
                *self.image_count.lock().unwrap() =
                    Some(messages.iter().map(|message| message.image_count()).sum());
                Ok(GenerationOutput {
                    text: "vision stub".to_string(),
                    tokens_generated: 2,
                    generation_time_ms: 1,
                    tokens_per_second: 2.0,
                    finish_reason: "stop".to_string(),
                    ttft_ms: None,
                    mean_itl_ms: None,
                    p95_itl_ms: None,
                    emitted_chunks: None,
                    inter_chunk_ms: Vec::new(),
                    decode_tps: None,
                    prefill_tps: None,
                    image_preprocess_ms: None,
                })
            }
        }

        let metadata = ModelMetadata {
            model_id: "embedding-style-vlm".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::VisionLanguage {
                model_file: "model.gguf".to_string(),
                chat_template: None,
                context_length: 4096,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["model.gguf".to_string(), "mmproj.gguf".to_string()],
            vision_encoder: Some(VisionEncoderConfig {
                file: "mmproj.gguf".to_string(),
                preprocessing_preset: VisionPreprocessingPreset::SigLip,
                image_size: 2,
                patch_size: Some(1),
            }),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };

        let image_bytes = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                2,
                2,
                image::Rgb([127, 127, 127]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            encoded.into_inner()
        };
        let image = Envelope::image(image_bytes, "png").unwrap();
        let input = Envelope::user_message("Describe this image", vec![image]).unwrap();

        crate::tracing::init_tracing(true);
        crate::tracing::reset_tracing();

        let events = Arc::new(Mutex::new(Vec::new()));
        let observed_shape = Arc::new(Mutex::new(None));
        let image_count = Arc::new(Mutex::new(None));
        let mut executor = TemplateExecutor::with_base_path("/models");
        executor.register_vision_encoder(
            "mmproj.gguf",
            Box::new(SpyVisionEncoder {
                events: events.clone(),
                observed_shape: observed_shape.clone(),
            }),
        );
        executor.llm_adapter_cache = Some((
            LlmAdapterCacheKey::new(
                "/models/model.gguf".to_string(),
                None,
                4096,
                None,
                Some("/models/mmproj.gguf".to_string()),
            ),
            LlmRuntimeAdapter::with_backend(Box::new(StubVisionBackend {
                events: events.clone(),
                image_count: image_count.clone(),
            })),
        ));

        let output = executor.execute(&metadata, &input, None).unwrap();

        assert_eq!(output.kind, EnvelopeKind::Text("vision stub".to_string()));
        let image_preprocess_ms = output
            .metadata
            .get("image_preprocess_ms")
            .and_then(|value| value.parse::<u32>().ok());
        assert!(
            image_preprocess_ms.is_some_and(|value| value > 0),
            "vision response metadata must carry positive image_preprocess_ms, got {:?}",
            output.metadata
        );
        assert_eq!(*observed_shape.lock().unwrap(), Some(vec![1, 3, 2, 2]));
        assert_eq!(*image_count.lock().unwrap(), Some(1));
        assert_eq!(*events.lock().unwrap(), vec!["encoder", "llm"]);

        let spans = crate::tracing::get_stages_json();
        crate::tracing::reset_tracing();
        let vision_encoder_metadata = spans["spans"]
            .as_array()
            .and_then(|spans| {
                spans
                    .iter()
                    .find(|span| span["name"].as_str() == Some("vision_encoder"))
            })
            .and_then(|span| span["metadata"].as_object())
            .expect("vision encoder span should carry metadata");
        let image_preprocess_ms = vision_encoder_metadata
            .get("image_preprocess_ms")
            .and_then(|value| value.as_str())
            .and_then(|value| value.parse::<u64>().ok());
        assert!(
            image_preprocess_ms.is_some_and(|value| value > 0),
            "vision encoder span must carry positive image_preprocess_ms, got {vision_encoder_metadata:?}"
        );
    }

    #[cfg(feature = "vision")]
    #[test]
    fn vision_language_context_planning_replays_history_then_current_input() {
        use crate::runtime_adapter::MultimodalMessagePart;

        fn png_image() -> Envelope {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                2,
                2,
                image::Rgb([17, 34, 51]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image encodes");
            Envelope::image(encoded.into_inner(), "png").unwrap()
        }

        let mut context = ConversationContext::new().with_system(
            Envelope::new(EnvelopeKind::Text("You describe images.".to_string()))
                .with_role(MessageRole::System),
        );
        context.push(Envelope::user_message("Earlier image", vec![png_image()]).unwrap());
        context.push(
            Envelope::new(EnvelopeKind::Text("Earlier answer".to_string()))
                .with_role(MessageRole::Assistant),
        );

        let current = Envelope::user_message("Current image", vec![png_image()]).unwrap();
        let messages = TemplateExecutor::multimodal_messages_with_context(&current, &context)
            .expect("multimodal planning succeeds");

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[1].role, MessageRole::User);
        assert_eq!(messages[2].role, MessageRole::Assistant);
        assert_eq!(messages[3].role, MessageRole::User);

        assert_eq!(
            messages[3].parts[0],
            MultimodalMessagePart::Text("Current image".to_string())
        );
        assert!(matches!(
            messages[3].parts[1],
            MultimodalMessagePart::Image(_)
        ));
    }

    #[cfg(all(
        feature = "vision",
        any(feature = "llm-mistral", feature = "llm-llamacpp")
    ))]
    #[test]
    fn vision_language_llm_config_resolves_sibling_mmproj_path() {
        use crate::execution::template::{VisionEncoderConfig, VisionPreprocessingPreset};

        let metadata = ModelMetadata {
            model_id: "gemma-3n-e2b".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::VisionLanguage {
                model_file: "model.gguf".to_string(),
                chat_template: Some("chat_template.json".to_string()),
                context_length: 8192,
                generation_params: None,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec![
                "model.gguf".to_string(),
                "chat_template.json".to_string(),
                "mmproj.gguf".to_string(),
            ],
            vision_encoder: Some(VisionEncoderConfig {
                file: "mmproj.gguf".to_string(),
                preprocessing_preset: VisionPreprocessingPreset::Gemma3Vision,
                image_size: 896,
                patch_size: Some(14),
            }),
            description: None,
            metadata: HashMap::new(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        };
        let executor = TemplateExecutor::with_base_path("/models");

        let config = executor.vision_language_llm_config(
            &metadata,
            "/models/model.gguf".to_string(),
            Some("chat_template.json"),
            8192,
        );

        assert_eq!(config.model_path, "/models/model.gguf");
        assert_eq!(config.context_length, 8192);
        assert_eq!(
            config.chat_template.as_deref(),
            Some("/models/chat_template.json")
        );
        assert_eq!(
            config.vision_encoder_path.as_deref(),
            Some("/models/mmproj.gguf")
        );
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
    fn test_fade_pcm16_edges_ramps_both_ends() {
        const S: i16 = 1000;
        let mut pcm: Vec<u8> = (0..20).flat_map(|_| S.to_le_bytes()).collect();
        fade_pcm16_edges(&mut pcm, 4);
        let read = |i: usize| i16::from_le_bytes([pcm[i * 2], pcm[i * 2 + 1]]);
        // Head ramps up.
        assert!(read(0).abs() < S, "head attenuated");
        assert!(read(0).abs() < read(3).abs(), "head ramps up");
        // Middle untouched.
        assert_eq!(read(10), S);
        // Tail ramps down.
        assert!(read(19).abs() < S, "tail attenuated");
        assert!(read(19).abs() < read(16).abs(), "tail ramps down");
    }

    #[test]
    fn test_fade_pcm16_edges_noop_for_tiny_buffer() {
        // One sample (2 bytes): n clamps to total/2 = 0 → no change.
        let mut pcm = 1234i16.to_le_bytes().to_vec();
        fade_pcm16_edges(&mut pcm, 4);
        assert_eq!(i16::from_le_bytes([pcm[0], pcm[1]]), 1234);
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

    // ============================================================================
    // stamp_llm_span_cost_attribution Tests
    // ============================================================================
    //
    // These cases pin the chat-context cost-attribution path: the wire `backend`
    // label that lands on the currently-open span when
    // `execute_with_context_impl` / `execute_streaming_with_context_impl` bypass
    // the outer `execute:<model_id>` span. They share state with the global
    // span collector, so a process-wide mutex serialises them — `cargo test`
    // parallelises within a binary and these would otherwise stomp each other's
    // reset/measure window.

    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    mod stamp_llm_span {
        use super::*;
        use crate::tracing;

        fn gguf_metadata(backend_hint: Option<&str>) -> ModelMetadata {
            let mut bundle_metadata = HashMap::new();
            if let Some(hint) = backend_hint {
                bundle_metadata.insert("backend".to_string(), serde_json::json!(hint));
            }
            ModelMetadata {
                model_id: "test-gguf".into(),
                version: "1".into(),
                execution_template: ExecutionTemplate::Gguf {
                    model_file: "test.gguf".into(),
                    chat_template: None,
                    context_length: 2048,
                    generation_params: None,
                },
                preprocessing: Vec::new(),
                postprocessing: Vec::new(),
                files: Vec::new(),
                #[cfg(feature = "vision")]
                vision_encoder: None,
                description: None,
                metadata: bundle_metadata,
                voices: None,
                max_chunk_chars: None,
                trim_trailing_samples: None,
            }
        }

        fn capture_span_metadata(
            span_name: &str,
            metadata: &ModelMetadata,
        ) -> HashMap<String, String> {
            tracing::init_tracing(true);
            tracing::reset_tracing();
            {
                let _guard = tracing::SpanGuard::new(span_name);
                stamp_llm_span_cost_attribution(metadata);
            }
            let json = tracing::get_stages_json();
            tracing::reset_tracing();

            let span = json["spans"]
                .as_array()
                .and_then(|spans| spans.iter().find(|s| s["name"].as_str() == Some(span_name)))
                .expect("span recorded by SpanGuard must be present in stages json");
            span["metadata"]
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                        .collect()
                })
                .unwrap_or_default()
        }

        #[test]
        fn unannotated_gguf_stamps_llamacpp_default() {
            let _lock = tracing::test_lock();
            let captured = capture_span_metadata("execute:test", &gguf_metadata(None));
            assert_eq!(
                captured.get("backend").map(String::as_str),
                Some("llamacpp"),
                "chat-context flow must default unannotated GGUF bundles to llamacpp so PlatformEvent.backend is non-empty"
            );
        }

        #[test]
        fn mistralrs_hint_wins_on_gguf() {
            let _lock = tracing::test_lock();
            let captured = capture_span_metadata("execute:test", &gguf_metadata(Some("mistralrs")));
            assert_eq!(
                captured.get("backend").map(String::as_str),
                Some("mistralrs")
            );
        }

        #[test]
        fn legacy_mistral_alias_normalises_to_mistralrs() {
            let _lock = tracing::test_lock();
            let captured = capture_span_metadata("execute:test", &gguf_metadata(Some("mistral")));
            assert_eq!(
                captured.get("backend").map(String::as_str),
                Some("mistralrs"),
                "the legacy `mistral` bundle alias must canonicalise to the wire label"
            );
        }

        #[test]
        fn quantization_stamped_from_gguf_filename() {
            let _lock = tracing::test_lock();
            let mut metadata = gguf_metadata(None);
            metadata.execution_template = ExecutionTemplate::Gguf {
                model_file: "tinyllama-1.1b-chat-q4_k_m.gguf".into(),
                chat_template: None,
                context_length: 2048,
                generation_params: None,
            };
            let captured = capture_span_metadata("execute:test", &metadata);
            assert_eq!(
                captured.get("quantization").map(String::as_str),
                Some("q4_k_m"),
                "stamp must surface the filename-inferred quantization alongside backend"
            );
        }

        // A backend stub that lets a test pick the wire label so we can
        // verify `stamp_llm_runtime_backend` overwrites the
        // template-derived default with the runtime's identity. Needed
        // because the runtime is chosen by cargo feature, so the
        // template-derived label can disagree with the runtime that
        // actually executes (e.g. mistralrs on an `llm-mistral`-only
        // build loading an unannotated GGUF bundle).
        struct WireLabelStub(Option<&'static str>);

        impl crate::runtime_adapter::llm::LlmBackend for WireLabelStub {
            fn name(&self) -> &str {
                "wire-label-stub"
            }
            fn wire_label(&self) -> Option<&'static str> {
                self.0
            }
            fn supported_formats(&self) -> Vec<&'static str> {
                vec!["gguf"]
            }
            fn load(
                &mut self,
                _config: &crate::runtime_adapter::llm::LlmConfig,
            ) -> crate::runtime_adapter::llm::LlmResult<()> {
                Ok(())
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn unload(&mut self) -> crate::runtime_adapter::llm::LlmResult<()> {
                Ok(())
            }
            fn generate(
                &self,
                _messages: &[crate::runtime_adapter::llm::ChatMessage],
                _config: &crate::runtime_adapter::llm::GenerationConfig,
            ) -> crate::runtime_adapter::llm::LlmResult<crate::runtime_adapter::llm::GenerationOutput>
            {
                unreachable!("stub backend should not be invoked for inference in this test")
            }
            fn generate_raw(
                &self,
                _prompt: &str,
                _config: &crate::runtime_adapter::llm::GenerationConfig,
            ) -> crate::runtime_adapter::llm::LlmResult<crate::runtime_adapter::llm::GenerationOutput>
            {
                unreachable!("stub backend should not be invoked for inference in this test")
            }
        }

        fn capture_with_runtime_overwrite(
            metadata: &ModelMetadata,
            wire_label: Option<&'static str>,
        ) -> HashMap<String, String> {
            let adapter = crate::runtime_adapter::llm::LlmRuntimeAdapter::with_backend(Box::new(
                WireLabelStub(wire_label),
            ));
            tracing::init_tracing(true);
            tracing::reset_tracing();
            {
                let _guard = tracing::SpanGuard::new("execute:test");
                stamp_llm_span_cost_attribution(metadata);
                stamp_llm_runtime_backend(&adapter);
            }
            let json = tracing::get_stages_json();
            tracing::reset_tracing();

            let span = json["spans"]
                .as_array()
                .and_then(|spans| {
                    spans
                        .iter()
                        .find(|s| s["name"].as_str() == Some("execute:test"))
                })
                .expect("span recorded by SpanGuard must be present in stages json");
            span["metadata"]
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                        .collect()
                })
                .unwrap_or_default()
        }

        #[test]
        fn runtime_wire_label_overwrites_template_default() {
            let _lock = tracing::test_lock();
            // Template default for unannotated GGUF is `llamacpp`, but
            // the runtime selected by cargo feature is mistral.rs. The
            // overwrite must flip the stamp to ground truth so the
            // dashboard reflects the runtime that actually executed.
            let captured = capture_with_runtime_overwrite(&gguf_metadata(None), Some("mistralrs"));
            assert_eq!(
                captured.get("backend").map(String::as_str),
                Some("mistralrs"),
                "runtime wire label must overwrite the template-derived default"
            );
        }

        #[test]
        fn runtime_overwrite_preserves_template_default_when_label_absent() {
            let _lock = tracing::test_lock();
            // Mock/test backends return None from wire_label so the
            // template-derived stamp survives — anything else would
            // erase ground truth that the template *did* carry.
            let captured = capture_with_runtime_overwrite(&gguf_metadata(None), None);
            assert_eq!(
                captured.get("backend").map(String::as_str),
                Some("llamacpp"),
                "stub backends without a wire label must not erase the template-derived stamp"
            );
        }
    }
}
