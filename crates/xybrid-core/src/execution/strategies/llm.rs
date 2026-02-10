//! LLM execution strategy for GGUF models.
//!
//! Handles local LLM inference via GGUF models with:
//! - Pluggable backend abstraction (mockable for tests)
//! - Chat template support
//! - Generation config parsing from envelope metadata
//!
//! This module is feature-gated and requires either `llm-mistral` or `llm-llamacpp`.

use log::{debug, info};
use std::path::Path;

use super::{ExecutionContext, ExecutionStrategy};
use crate::execution::template::{ExecutionTemplate, ModelMetadata};
use crate::execution::types::ExecutorResult;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::AdapterError;
use crate::tracing as xybrid_trace;

// ============================================================================
// LLM Inference Trait (for mockability)
// ============================================================================

/// Configuration for LLM model loading.
#[derive(Debug, Clone)]
pub struct LlmModelConfig {
    /// Path to the GGUF model file
    pub model_path: String,
    /// Optional chat template path
    pub chat_template: Option<String>,
    /// Maximum context length
    pub context_length: usize,
    /// Backend hint ("mistral", "llamacpp", or None for default)
    pub backend_hint: Option<String>,
}

impl LlmModelConfig {
    /// Create a new config with required fields.
    pub fn new(model_path: impl Into<String>, context_length: usize) -> Self {
        Self {
            model_path: model_path.into(),
            chat_template: None,
            context_length,
            backend_hint: None,
        }
    }

    /// Set the chat template path.
    pub fn with_chat_template(mut self, path: impl Into<String>) -> Self {
        self.chat_template = Some(path.into());
        self
    }

    /// Set the backend hint.
    pub fn with_backend_hint(mut self, hint: impl Into<String>) -> Self {
        self.backend_hint = Some(hint.into());
        self
    }
}

/// Default stop sequences for ChatML format (Qwen, Phi, etc.)
pub const CHATML_STOP_SEQUENCES: &[&str] = &["<|im_end|>", "<|im_start|>"];

/// Default stop sequences for Llama format
pub const LLAMA_STOP_SEQUENCES: &[&str] = &["</s>", "[/INST]"];

/// Generation parameters for LLM inference.
#[derive(Debug, Clone)]
pub struct LlmGenerationParams {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// System prompt (optional)
    pub system_prompt: Option<String>,
    /// Stop sequences - generation stops when any of these are encountered
    pub stop_sequences: Vec<String>,
}

impl Default for LlmGenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            system_prompt: None,
            stop_sequences: Vec::new(),
        }
    }
}

impl LlmGenerationParams {
    /// Create params with ChatML stop sequences (for Qwen, Phi, etc.)
    pub fn with_chatml_stops() -> Self {
        Self {
            stop_sequences: CHATML_STOP_SEQUENCES.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    /// Create params with Llama stop sequences
    pub fn with_llama_stops() -> Self {
        Self {
            stop_sequences: LLAMA_STOP_SEQUENCES.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    /// Add stop sequences to existing params
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }

    /// Detect appropriate stop sequences based on model name.
    ///
    /// Returns ChatML stops for Qwen/Phi models, Llama stops for Llama/Mistral models.
    pub fn default_stops_for_model(model_id: &str) -> Vec<String> {
        let model_lower = model_id.to_lowercase();

        // ChatML format models
        if model_lower.contains("qwen")
            || model_lower.contains("phi")
            || model_lower.contains("yi-")
            || model_lower.contains("deepseek")
        {
            return CHATML_STOP_SEQUENCES.iter().map(|s| s.to_string()).collect();
        }

        // Llama format models
        if model_lower.contains("llama")
            || model_lower.contains("mistral")
            || model_lower.contains("mixtral")
            || model_lower.contains("gemma")
        {
            return LLAMA_STOP_SEQUENCES.iter().map(|s| s.to_string()).collect();
        }

        // Default: use ChatML as it's most common
        CHATML_STOP_SEQUENCES.iter().map(|s| s.to_string()).collect()
    }
}

impl LlmGenerationParams {
    /// Parse generation params from envelope metadata.
    ///
    /// Supports parsing:
    /// - `max_tokens`: Maximum tokens to generate
    /// - `temperature`: Sampling temperature
    /// - `top_p`: Nucleus sampling threshold
    /// - `top_k`: Top-k sampling
    /// - `system_prompt`: System prompt text
    /// - `stop_sequences`: Comma-separated list of stop sequences
    /// - `model_id`: Used to auto-detect stop sequences if not explicitly provided
    pub fn from_envelope_metadata(metadata: &std::collections::HashMap<String, String>) -> Self {
        let mut params = Self::default();

        if let Some(val) = metadata.get("max_tokens").and_then(|s| s.parse().ok()) {
            params.max_tokens = val;
        }
        if let Some(val) = metadata.get("temperature").and_then(|s| s.parse().ok()) {
            params.temperature = val;
        }
        if let Some(val) = metadata.get("top_p").and_then(|s| s.parse().ok()) {
            params.top_p = val;
        }
        if let Some(val) = metadata.get("top_k").and_then(|s| s.parse().ok()) {
            params.top_k = val;
        }
        if let Some(val) = metadata.get("system_prompt") {
            params.system_prompt = Some(val.clone());
        }

        // Parse stop sequences from comma-separated string
        if let Some(val) = metadata.get("stop_sequences") {
            params.stop_sequences = val
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        params
    }

    /// Parse generation params with auto-detected stop sequences based on model ID.
    ///
    /// If no explicit stop sequences are provided in metadata, auto-detects
    /// appropriate stops based on the model ID (ChatML for Qwen/Phi, Llama for others).
    pub fn from_envelope_metadata_with_model(
        metadata: &std::collections::HashMap<String, String>,
        model_id: &str,
    ) -> Self {
        let mut params = Self::from_envelope_metadata(metadata);

        // If no stop sequences were explicitly provided, auto-detect from model
        if params.stop_sequences.is_empty() {
            params.stop_sequences = Self::default_stops_for_model(model_id);
        }

        params
    }
}

/// Trait for LLM inference - enables mocking in tests.
///
/// This trait abstracts the LLM backend, allowing the strategy to be
/// tested without loading actual models.
pub trait LlmInference: Send + Sync {
    /// Load a model with the given configuration.
    fn load_model(&mut self, config: &LlmModelConfig) -> ExecutorResult<()>;

    /// Generate text from a prompt.
    fn generate(&self, prompt: &str, params: &LlmGenerationParams) -> ExecutorResult<String>;

    /// Check if a model is currently loaded.
    fn is_loaded(&self) -> bool;

    /// Get the name of this backend for logging.
    fn backend_name(&self) -> &str;
}

// ============================================================================
// Default Implementation (wraps LlmRuntimeAdapter)
// ============================================================================

/// Default LLM inference implementation using LlmRuntimeAdapter.
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
pub struct DefaultLlmInference {
    adapter: Option<crate::runtime_adapter::llm::LlmRuntimeAdapter>,
    backend_hint: Option<String>,
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
impl DefaultLlmInference {
    /// Create a new default inference backend.
    pub fn new() -> Self {
        Self {
            adapter: None,
            backend_hint: None,
        }
    }

    /// Create with a specific backend hint.
    pub fn with_backend_hint(hint: Option<&str>) -> Self {
        Self {
            adapter: None,
            backend_hint: hint.map(String::from),
        }
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
impl Default for DefaultLlmInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
impl LlmInference for DefaultLlmInference {
    fn load_model(&mut self, config: &LlmModelConfig) -> ExecutorResult<()> {
        use crate::runtime_adapter::llm::{LlmConfig, LlmRuntimeAdapter};
        use crate::runtime_adapter::RuntimeAdapter;

        // Create LLM config
        let mut llm_config = LlmConfig::new(&config.model_path)
            .with_context_length(config.context_length);

        if let Some(template) = &config.chat_template {
            llm_config = llm_config.with_chat_template(template);
        }

        // Determine backend hint
        let hint = config
            .backend_hint
            .as_deref()
            .or(self.backend_hint.as_deref());

        // Create adapter with backend hint
        let mut adapter = LlmRuntimeAdapter::with_backend_hint(hint)?;
        adapter.load_model(&config.model_path)?;

        self.adapter = Some(adapter);
        Ok(())
    }

    fn generate(&self, prompt: &str, params: &LlmGenerationParams) -> ExecutorResult<String> {
        use crate::runtime_adapter::llm::GenerationConfig;

        let adapter = self.adapter.as_ref().ok_or_else(|| {
            AdapterError::RuntimeError("No model loaded".to_string())
        })?;

        let gen_config = GenerationConfig {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: 1.1,
            stop_sequences: params.stop_sequences.clone(),
            ..Default::default()
        };

        debug!(
            target: "xybrid_core",
            "LLM generation with {} stop sequences: {:?}",
            gen_config.stop_sequences.len(),
            gen_config.stop_sequences
        );

        let output = adapter.generate_with_config(
            prompt,
            params.system_prompt.as_deref(),
            &gen_config,
        )?;

        Ok(output.text)
    }

    fn is_loaded(&self) -> bool {
        self.adapter.is_some()
    }

    fn backend_name(&self) -> &str {
        self.backend_hint.as_deref().unwrap_or("default")
    }
}

// ============================================================================
// LLM Strategy
// ============================================================================

/// LLM execution strategy for GGUF models.
///
/// This strategy handles models with `ExecutionTemplate::Gguf` and provides:
/// - Configurable backend selection (mistral, llamacpp)
/// - Chat template support
/// - Generation parameter parsing from envelope metadata
pub struct LlmStrategy<I: LlmInference = DefaultLlmInferenceType> {
    inference: std::sync::Mutex<I>,
}

// Type alias for the default inference type based on features
#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
type DefaultLlmInferenceType = DefaultLlmInference;

#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
type DefaultLlmInferenceType = NoOpLlmInference;

/// No-op inference for when LLM features are disabled.
#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
pub struct NoOpLlmInference;

#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
impl LlmInference for NoOpLlmInference {
    fn load_model(&mut self, _config: &LlmModelConfig) -> ExecutorResult<()> {
        Err(AdapterError::RuntimeError(
            "LLM features not enabled. Enable 'llm-mistral' or 'llm-llamacpp' feature.".to_string(),
        ))
    }

    fn generate(&self, _prompt: &str, _params: &LlmGenerationParams) -> ExecutorResult<String> {
        Err(AdapterError::RuntimeError(
            "LLM features not enabled".to_string(),
        ))
    }

    fn is_loaded(&self) -> bool {
        false
    }

    fn backend_name(&self) -> &str {
        "none"
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
impl LlmStrategy<DefaultLlmInference> {
    /// Create a new LLM strategy with default inference backend.
    pub fn new() -> Self {
        Self {
            inference: std::sync::Mutex::new(DefaultLlmInference::new()),
        }
    }

    /// Create with a specific backend hint.
    pub fn with_backend_hint(hint: Option<&str>) -> Self {
        Self {
            inference: std::sync::Mutex::new(DefaultLlmInference::with_backend_hint(hint)),
        }
    }
}

#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
impl LlmStrategy<NoOpLlmInference> {
    /// Create a new LLM strategy (no-op when features disabled).
    pub fn new() -> Self {
        Self {
            inference: std::sync::Mutex::new(NoOpLlmInference),
        }
    }
}

impl<I: LlmInference> LlmStrategy<I> {
    /// Create with a custom inference backend (for testing).
    pub fn with_inference(inference: I) -> Self {
        Self {
            inference: std::sync::Mutex::new(inference),
        }
    }

    /// Check if this is an LLM model (GGUF template).
    fn is_llm_model(metadata: &ModelMetadata) -> bool {
        matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. })
    }

    /// Extract GGUF config from metadata.
    fn extract_gguf_config(
        metadata: &ModelMetadata,
        base_path: &str,
    ) -> ExecutorResult<LlmModelConfig> {
        match &metadata.execution_template {
            ExecutionTemplate::Gguf {
                model_file,
                chat_template,
                context_length,
            } => {
                let model_path = Path::new(base_path).join(model_file);

                let mut config = LlmModelConfig::new(
                    model_path.to_string_lossy().to_string(),
                    *context_length,
                );

                if let Some(template) = chat_template {
                    let template_path = Path::new(base_path).join(template);
                    config = config.with_chat_template(template_path.to_string_lossy().to_string());
                }

                // Extract backend hint from metadata
                if let Some(hint) = metadata.metadata.get("backend").and_then(|v| v.as_str()) {
                    config = config.with_backend_hint(hint);
                }

                Ok(config)
            }
            _ => Err(AdapterError::InvalidInput(
                "Expected GGUF execution template".to_string(),
            )),
        }
    }

    /// Extract prompt from input envelope.
    fn extract_prompt(input: &Envelope) -> ExecutorResult<String> {
        match &input.kind {
            EnvelopeKind::Text(text) => Ok(text.clone()),
            _ => Err(AdapterError::InvalidInput(
                "LLM requires text input".to_string(),
            )),
        }
    }
}

impl<I: LlmInference + 'static> ExecutionStrategy for LlmStrategy<I> {
    fn can_handle(&self, metadata: &ModelMetadata) -> bool {
        Self::is_llm_model(metadata)
    }

    fn execute(
        &self,
        ctx: &mut ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope> {
        let _span = xybrid_trace::SpanGuard::new("llm_execution");

        // Extract configuration
        let config = Self::extract_gguf_config(metadata, ctx.base_path)?;

        info!(
            target: "xybrid_core",
            "Executing LLM inference: {} (backend: {:?})",
            config.model_path,
            config.backend_hint.as_deref().unwrap_or("default")
        );

        xybrid_trace::add_metadata("model", &config.model_path);
        if let Some(hint) = &config.backend_hint {
            xybrid_trace::add_metadata("backend", hint);
        }

        // Load model if needed
        let mut inference = self.inference.lock().map_err(|e| {
            AdapterError::RuntimeError(format!("Failed to acquire lock: {}", e))
        })?;

        if !inference.is_loaded() {
            debug!(target: "xybrid_core", "Loading LLM model: {}", config.model_path);
            inference.load_model(&config)?;
        }

        // Extract prompt
        let prompt = Self::extract_prompt(input)?;

        // Parse generation params from envelope metadata with auto-detected stop sequences
        let params = LlmGenerationParams::from_envelope_metadata_with_model(
            &input.metadata,
            &metadata.model_id,
        );

        debug!(
            target: "xybrid_core",
            "LLM generation: max_tokens={}, temp={}, prompt_len={}, stop_sequences={:?}",
            params.max_tokens,
            params.temperature,
            prompt.len(),
            params.stop_sequences
        );

        // Generate
        let result = inference.generate(&prompt, &params)?;

        info!(
            target: "xybrid_core",
            "LLM inference complete: {} chars generated",
            result.len()
        );

        Ok(Envelope::new(EnvelopeKind::Text(result)))
    }

    fn name(&self) -> &'static str {
        "llm"
    }
}

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
impl Default for LlmStrategy<DefaultLlmInference> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    // ========================================================================
    // Mock LLM Inference for Testing
    // ========================================================================

    /// Mock inference that tracks calls and returns configurable responses.
    pub struct MockLlmInference {
        pub loaded: AtomicBool,
        pub load_count: AtomicUsize,
        pub generate_count: AtomicUsize,
        pub response: String,
        pub should_fail_load: bool,
        pub should_fail_generate: bool,
        pub last_prompt: std::sync::Mutex<Option<String>>,
        pub last_params: std::sync::Mutex<Option<LlmGenerationParams>>,
    }

    impl MockLlmInference {
        pub fn new() -> Self {
            Self {
                loaded: AtomicBool::new(false),
                load_count: AtomicUsize::new(0),
                generate_count: AtomicUsize::new(0),
                response: "Mock response".to_string(),
                should_fail_load: false,
                should_fail_generate: false,
                last_prompt: std::sync::Mutex::new(None),
                last_params: std::sync::Mutex::new(None),
            }
        }

        pub fn with_response(mut self, response: impl Into<String>) -> Self {
            self.response = response.into();
            self
        }

        pub fn failing_load(mut self) -> Self {
            self.should_fail_load = true;
            self
        }

        pub fn failing_generate(mut self) -> Self {
            self.should_fail_generate = true;
            self
        }
    }

    impl LlmInference for MockLlmInference {
        fn load_model(&mut self, _config: &LlmModelConfig) -> ExecutorResult<()> {
            self.load_count.fetch_add(1, Ordering::SeqCst);

            if self.should_fail_load {
                return Err(AdapterError::RuntimeError("Mock load failure".to_string()));
            }

            self.loaded.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn generate(&self, prompt: &str, params: &LlmGenerationParams) -> ExecutorResult<String> {
            self.generate_count.fetch_add(1, Ordering::SeqCst);

            // Store last call params
            *self.last_prompt.lock().unwrap() = Some(prompt.to_string());
            *self.last_params.lock().unwrap() = Some(params.clone());

            if self.should_fail_generate {
                return Err(AdapterError::RuntimeError("Mock generate failure".to_string()));
            }

            Ok(self.response.clone())
        }

        fn is_loaded(&self) -> bool {
            self.loaded.load(Ordering::SeqCst)
        }

        fn backend_name(&self) -> &str {
            "mock"
        }
    }

    // ========================================================================
    // LlmModelConfig Tests
    // ========================================================================

    #[test]
    fn test_llm_model_config_new() {
        let config = LlmModelConfig::new("/path/to/model.gguf", 4096);

        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.context_length, 4096);
        assert!(config.chat_template.is_none());
        assert!(config.backend_hint.is_none());
    }

    #[test]
    fn test_llm_model_config_with_options() {
        let config = LlmModelConfig::new("/model.gguf", 2048)
            .with_chat_template("/template.json")
            .with_backend_hint("llamacpp");

        assert_eq!(config.model_path, "/model.gguf");
        assert_eq!(config.context_length, 2048);
        assert_eq!(config.chat_template, Some("/template.json".to_string()));
        assert_eq!(config.backend_hint, Some("llamacpp".to_string()));
    }

    // ========================================================================
    // LlmGenerationParams Tests
    // ========================================================================

    #[test]
    fn test_generation_params_default() {
        let params = LlmGenerationParams::default();

        assert_eq!(params.max_tokens, 256);
        assert!((params.temperature - 0.7).abs() < 0.001);
        assert!((params.top_p - 0.9).abs() < 0.001);
        assert_eq!(params.top_k, 40);
        assert!(params.system_prompt.is_none());
        assert!(params.stop_sequences.is_empty());
    }

    // ========================================================================
    // Stop Sequences Tests
    // ========================================================================

    #[test]
    fn test_chatml_stop_sequences() {
        let params = LlmGenerationParams::with_chatml_stops();

        assert!(params.stop_sequences.contains(&"<|im_end|>".to_string()));
        assert!(params.stop_sequences.contains(&"<|im_start|>".to_string()));
    }

    #[test]
    fn test_llama_stop_sequences() {
        let params = LlmGenerationParams::with_llama_stops();

        assert!(params.stop_sequences.contains(&"</s>".to_string()));
        assert!(params.stop_sequences.contains(&"[/INST]".to_string()));
    }

    #[test]
    fn test_with_stop_sequences() {
        let params = LlmGenerationParams::default()
            .with_stop_sequences(vec!["STOP".to_string(), "END".to_string()]);

        assert_eq!(params.stop_sequences.len(), 2);
        assert!(params.stop_sequences.contains(&"STOP".to_string()));
        assert!(params.stop_sequences.contains(&"END".to_string()));
    }

    #[test]
    fn test_default_stops_for_qwen() {
        let stops = LlmGenerationParams::default_stops_for_model("qwen2.5-0.5b-instruct");

        assert!(stops.contains(&"<|im_end|>".to_string()));
        assert!(stops.contains(&"<|im_start|>".to_string()));
    }

    #[test]
    fn test_default_stops_for_phi() {
        let stops = LlmGenerationParams::default_stops_for_model("phi-3-mini-4k");

        assert!(stops.contains(&"<|im_end|>".to_string()));
    }

    #[test]
    fn test_default_stops_for_llama() {
        let stops = LlmGenerationParams::default_stops_for_model("llama-3.2-1b");

        assert!(stops.contains(&"</s>".to_string()));
        assert!(stops.contains(&"[/INST]".to_string()));
    }

    #[test]
    fn test_default_stops_for_mistral() {
        let stops = LlmGenerationParams::default_stops_for_model("mistral-7b");

        assert!(stops.contains(&"</s>".to_string()));
    }

    #[test]
    fn test_default_stops_for_unknown_model() {
        // Unknown models should default to ChatML (most common)
        let stops = LlmGenerationParams::default_stops_for_model("some-unknown-model");

        assert!(stops.contains(&"<|im_end|>".to_string()));
    }

    #[test]
    fn test_default_stops_case_insensitive() {
        let stops_lower = LlmGenerationParams::default_stops_for_model("qwen2.5");
        let stops_upper = LlmGenerationParams::default_stops_for_model("QWEN2.5");

        assert_eq!(stops_lower, stops_upper);
    }

    #[test]
    fn test_parse_stop_sequences_from_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("stop_sequences".to_string(), "<|im_end|>,<|im_start|>".to_string());

        let params = LlmGenerationParams::from_envelope_metadata(&metadata);

        assert_eq!(params.stop_sequences.len(), 2);
        assert!(params.stop_sequences.contains(&"<|im_end|>".to_string()));
        assert!(params.stop_sequences.contains(&"<|im_start|>".to_string()));
    }

    #[test]
    fn test_parse_stop_sequences_with_spaces() {
        let mut metadata = HashMap::new();
        metadata.insert("stop_sequences".to_string(), " STOP , END , HALT ".to_string());

        let params = LlmGenerationParams::from_envelope_metadata(&metadata);

        assert_eq!(params.stop_sequences.len(), 3);
        assert!(params.stop_sequences.contains(&"STOP".to_string()));
        assert!(params.stop_sequences.contains(&"END".to_string()));
        assert!(params.stop_sequences.contains(&"HALT".to_string()));
    }

    #[test]
    fn test_parse_empty_stop_sequences() {
        let mut metadata = HashMap::new();
        metadata.insert("stop_sequences".to_string(), "".to_string());

        let params = LlmGenerationParams::from_envelope_metadata(&metadata);

        assert!(params.stop_sequences.is_empty());
    }

    #[test]
    fn test_auto_detect_stops_for_qwen_model() {
        let metadata = HashMap::new();

        let params = LlmGenerationParams::from_envelope_metadata_with_model(
            &metadata,
            "qwen2.5-0.5b-instruct",
        );

        assert!(!params.stop_sequences.is_empty());
        assert!(params.stop_sequences.contains(&"<|im_end|>".to_string()));
    }

    #[test]
    fn test_explicit_stops_override_auto_detect() {
        let mut metadata = HashMap::new();
        metadata.insert("stop_sequences".to_string(), "CUSTOM_STOP".to_string());

        let params = LlmGenerationParams::from_envelope_metadata_with_model(
            &metadata,
            "qwen2.5-0.5b-instruct",
        );

        // Should use explicit stops, not auto-detected
        assert_eq!(params.stop_sequences.len(), 1);
        assert!(params.stop_sequences.contains(&"CUSTOM_STOP".to_string()));
        assert!(!params.stop_sequences.contains(&"<|im_end|>".to_string()));
    }

    #[test]
    fn test_generation_params_from_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("max_tokens".to_string(), "512".to_string());
        metadata.insert("temperature".to_string(), "0.5".to_string());
        metadata.insert("top_p".to_string(), "0.8".to_string());
        metadata.insert("top_k".to_string(), "20".to_string());
        metadata.insert("system_prompt".to_string(), "You are helpful.".to_string());

        let params = LlmGenerationParams::from_envelope_metadata(&metadata);

        assert_eq!(params.max_tokens, 512);
        assert!((params.temperature - 0.5).abs() < 0.001);
        assert!((params.top_p - 0.8).abs() < 0.001);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.system_prompt, Some("You are helpful.".to_string()));
    }

    #[test]
    fn test_generation_params_partial_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("max_tokens".to_string(), "1024".to_string());
        // Other fields not specified

        let params = LlmGenerationParams::from_envelope_metadata(&metadata);

        assert_eq!(params.max_tokens, 1024);
        // Defaults should be used for unspecified fields
        assert!((params.temperature - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_generation_params_invalid_values_ignored() {
        let mut metadata = HashMap::new();
        metadata.insert("max_tokens".to_string(), "not_a_number".to_string());
        metadata.insert("temperature".to_string(), "invalid".to_string());

        let params = LlmGenerationParams::from_envelope_metadata(&metadata);

        // Defaults should be used when parsing fails
        assert_eq!(params.max_tokens, 256);
        assert!((params.temperature - 0.7).abs() < 0.001);
    }

    // ========================================================================
    // LlmStrategy Tests
    // ========================================================================

    fn create_gguf_metadata() -> ModelMetadata {
        ModelMetadata {
            model_id: "test-llm".to_string(),
            version: "1.0".to_string(),
            execution_template: ExecutionTemplate::Gguf {
                model_file: "model.gguf".to_string(),
                chat_template: Some("template.json".to_string()),
                context_length: 4096,
            },
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec!["model.gguf".to_string()],
            description: None,
            metadata: HashMap::new(),
            voices: None,
        }
    }

    fn create_onnx_metadata() -> ModelMetadata {
        ModelMetadata::onnx("test-model", "1.0", "model.onnx")
    }

    #[test]
    fn test_is_llm_model_true() {
        let metadata = create_gguf_metadata();
        assert!(LlmStrategy::<MockLlmInference>::is_llm_model(&metadata));
    }

    #[test]
    fn test_is_llm_model_false() {
        let metadata = create_onnx_metadata();
        assert!(!LlmStrategy::<MockLlmInference>::is_llm_model(&metadata));
    }

    #[test]
    fn test_can_handle_gguf() {
        let strategy = LlmStrategy::with_inference(MockLlmInference::new());
        let metadata = create_gguf_metadata();

        assert!(strategy.can_handle(&metadata));
    }

    #[test]
    fn test_cannot_handle_onnx() {
        let strategy = LlmStrategy::with_inference(MockLlmInference::new());
        let metadata = create_onnx_metadata();

        assert!(!strategy.can_handle(&metadata));
    }

    #[test]
    fn test_extract_gguf_config() {
        let metadata = create_gguf_metadata();
        let config = LlmStrategy::<MockLlmInference>::extract_gguf_config(&metadata, "/base")
            .unwrap();

        assert!(config.model_path.contains("model.gguf"));
        assert!(config.chat_template.is_some());
        assert_eq!(config.context_length, 4096);
    }

    #[test]
    fn test_extract_gguf_config_with_backend_hint() {
        let mut metadata = create_gguf_metadata();
        metadata.metadata.insert(
            "backend".to_string(),
            serde_json::Value::String("llamacpp".to_string()),
        );

        let config = LlmStrategy::<MockLlmInference>::extract_gguf_config(&metadata, "/base")
            .unwrap();

        assert_eq!(config.backend_hint, Some("llamacpp".to_string()));
    }

    #[test]
    fn test_extract_prompt_text() {
        let input = Envelope::new(EnvelopeKind::Text("Hello, world!".to_string()));
        let prompt = LlmStrategy::<MockLlmInference>::extract_prompt(&input).unwrap();

        assert_eq!(prompt, "Hello, world!");
    }

    #[test]
    fn test_extract_prompt_non_text_fails() {
        let input = Envelope::new(EnvelopeKind::Audio(vec![1, 2, 3]));
        let result = LlmStrategy::<MockLlmInference>::extract_prompt(&input);

        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_name() {
        let strategy = LlmStrategy::with_inference(MockLlmInference::new());
        assert_eq!(strategy.name(), "llm");
    }

    // ========================================================================
    // Mock Inference Behavior Tests
    // ========================================================================

    #[test]
    fn test_mock_inference_load() {
        let mut mock = MockLlmInference::new();
        let config = LlmModelConfig::new("/model.gguf", 4096);

        assert!(!mock.is_loaded());
        mock.load_model(&config).unwrap();
        assert!(mock.is_loaded());
        assert_eq!(mock.load_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_mock_inference_generate() {
        let mut mock = MockLlmInference::new().with_response("Generated text");
        let config = LlmModelConfig::new("/model.gguf", 4096);
        let params = LlmGenerationParams::default();

        mock.load_model(&config).unwrap();
        let result = mock.generate("Test prompt", &params).unwrap();

        assert_eq!(result, "Generated text");
        assert_eq!(mock.generate_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            mock.last_prompt.lock().unwrap().as_ref().unwrap(),
            "Test prompt"
        );
    }

    #[test]
    fn test_mock_inference_load_failure() {
        let mut mock = MockLlmInference::new().failing_load();
        let config = LlmModelConfig::new("/model.gguf", 4096);

        let result = mock.load_model(&config);
        assert!(result.is_err());
        assert!(!mock.is_loaded());
    }

    #[test]
    fn test_mock_inference_generate_failure() {
        let mut mock = MockLlmInference::new().failing_generate();
        let config = LlmModelConfig::new("/model.gguf", 4096);
        let params = LlmGenerationParams::default();

        mock.load_model(&config).unwrap();
        let result = mock.generate("Test", &params);

        assert!(result.is_err());
    }

    // ========================================================================
    // Integration-style Tests (with Mock)
    // ========================================================================

    #[test]
    fn test_strategy_execute_with_mock() {
        use crate::runtime_adapter::ModelRuntime;

        let mock = MockLlmInference::new().with_response("LLM response");
        let strategy = LlmStrategy::with_inference(mock);

        let metadata = create_gguf_metadata();
        let input = Envelope::new(EnvelopeKind::Text("What is 2+2?".to_string()));

        let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
        let mut ctx = ExecutionContext {
            base_path: "/models",
            runtimes: &mut runtimes,
        };

        let result = strategy.execute(&mut ctx, &metadata, &input).unwrap();

        match result.kind {
            EnvelopeKind::Text(text) => assert_eq!(text, "LLM response"),
            _ => panic!("Expected text output"),
        }
    }

    #[test]
    fn test_strategy_passes_generation_params() {
        use crate::runtime_adapter::ModelRuntime;

        let mock = MockLlmInference::new();
        let strategy = LlmStrategy::with_inference(mock);

        let metadata = create_gguf_metadata();

        let mut input_metadata = HashMap::new();
        input_metadata.insert("max_tokens".to_string(), "100".to_string());
        input_metadata.insert("temperature".to_string(), "0.3".to_string());

        let input = Envelope::with_metadata(
            EnvelopeKind::Text("Test prompt".to_string()),
            input_metadata,
        );

        let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
        let mut ctx = ExecutionContext {
            base_path: "/models",
            runtimes: &mut runtimes,
        };

        let _ = strategy.execute(&mut ctx, &metadata, &input);

        // Get the mock to verify params were passed correctly
        let inference = strategy.inference.lock().unwrap();
        let last_params = inference.last_params.lock().unwrap();
        let params = last_params.as_ref().unwrap();

        assert_eq!(params.max_tokens, 100);
        assert!((params.temperature - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_strategy_auto_detects_stop_sequences() {
        use crate::runtime_adapter::ModelRuntime;

        let mock = MockLlmInference::new();
        let strategy = LlmStrategy::with_inference(mock);

        // Create metadata with a Qwen model ID
        let mut metadata = create_gguf_metadata();
        metadata.model_id = "qwen2.5-0.5b-instruct".to_string();

        let input = Envelope::new(EnvelopeKind::Text("Test prompt".to_string()));

        let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
        let mut ctx = ExecutionContext {
            base_path: "/models",
            runtimes: &mut runtimes,
        };

        let _ = strategy.execute(&mut ctx, &metadata, &input);

        // Verify stop sequences were auto-detected
        let inference = strategy.inference.lock().unwrap();
        let last_params = inference.last_params.lock().unwrap();
        let params = last_params.as_ref().unwrap();

        assert!(!params.stop_sequences.is_empty(), "Stop sequences should be auto-detected for Qwen");
        assert!(
            params.stop_sequences.contains(&"<|im_end|>".to_string()),
            "Should contain ChatML stop token"
        );
    }

    #[test]
    fn test_strategy_uses_explicit_stop_sequences() {
        use crate::runtime_adapter::ModelRuntime;

        let mock = MockLlmInference::new();
        let strategy = LlmStrategy::with_inference(mock);

        let metadata = create_gguf_metadata();

        // Provide explicit stop sequences
        let mut input_metadata = HashMap::new();
        input_metadata.insert("stop_sequences".to_string(), "STOP,END".to_string());

        let input = Envelope::with_metadata(
            EnvelopeKind::Text("Test prompt".to_string()),
            input_metadata,
        );

        let mut runtimes: HashMap<String, Box<dyn ModelRuntime>> = HashMap::new();
        let mut ctx = ExecutionContext {
            base_path: "/models",
            runtimes: &mut runtimes,
        };

        let _ = strategy.execute(&mut ctx, &metadata, &input);

        // Verify explicit stop sequences were used
        let inference = strategy.inference.lock().unwrap();
        let last_params = inference.last_params.lock().unwrap();
        let params = last_params.as_ref().unwrap();

        assert_eq!(params.stop_sequences.len(), 2);
        assert!(params.stop_sequences.contains(&"STOP".to_string()));
        assert!(params.stop_sequences.contains(&"END".to_string()));
    }
}
