//! LLM inference infrastructure for GGUF models.
//!
//! Provides the building blocks consumed by [`CodecTtsStrategy`](super::CodecTtsStrategy):
//! - The [`LlmInference`] backend abstraction (mockable for tests)
//! - [`LlmModelConfig`] and [`LlmGenerationParams`] (incl. stop-sequence handling)
//! - The default [`DefaultLlmInference`] backend (feature-gated) plus a no-op stub
//!
//! The real backend is feature-gated and requires either `llm-mistral` or
//! `llm-llamacpp`; without them a no-op stub keeps the module compiling.

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use log::debug;

use crate::execution::types::ExecutorResult;
use crate::gateway::Tool;
use crate::runtime_adapter::AdapterError;
use crate::runtime_adapter::MultimodalChatMessage;
use crate::runtime_adapter::{parse_stop_sequences, STOP_SEQUENCES_METADATA_KEY};

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
    /// Optional sibling vision encoder / mmproj path.
    pub vision_encoder_path: Option<String>,
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
            vision_encoder_path: None,
            context_length,
            backend_hint: None,
        }
    }

    /// Set the chat template path.
    pub fn with_chat_template(mut self, path: impl Into<String>) -> Self {
        self.chat_template = Some(path.into());
        self
    }

    /// Set the sibling vision encoder / mmproj artifact path.
    pub fn with_vision_encoder(mut self, path: impl Into<String>) -> Self {
        self.vision_encoder_path = Some(path.into());
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
    /// Repetition penalty (1.0 = disabled). Harmful for codec TTS where
    /// speech tokens legitimately repeat — keep at 1.0 for raw generation.
    pub repetition_penalty: f32,
    /// System prompt (optional)
    pub system_prompt: Option<String>,
    /// Stop sequences - generation stops when any of these are encountered
    pub stop_sequences: Vec<String>,
    /// Tools (functions) offered to the model, mirrored into the rebuilt
    /// `GenerationConfig`. Passive mirror only: the executor path is the
    /// tool-calling surface (it parses emitted calls back out); this
    /// strategy path renders definitions but does not parse responses, so
    /// there is deliberately no envelope-metadata entry point for it.
    pub tools: Vec<Tool>,
}

impl Default for LlmGenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            system_prompt: None,
            stop_sequences: Vec::new(),
            tools: Vec::new(),
        }
    }
}

impl LlmGenerationParams {
    /// Create params with ChatML stop sequences (for Qwen, Phi, etc.)
    pub fn with_chatml_stops() -> Self {
        Self {
            stop_sequences: CHATML_STOP_SEQUENCES
                .iter()
                .map(|s| s.to_string())
                .collect(),
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
            return CHATML_STOP_SEQUENCES
                .iter()
                .map(|s| s.to_string())
                .collect();
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
        CHATML_STOP_SEQUENCES
            .iter()
            .map(|s| s.to_string())
            .collect()
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
    /// - `stop_sequences`: Stop sequences — see [`parse_stop_sequences`]
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

        if let Some(val) = metadata.get(STOP_SEQUENCES_METADATA_KEY) {
            params.stop_sequences = parse_stop_sequences(val);
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

    /// Generate text from a prompt. The prompt is wrapped in a chat template
    /// (as a user message) before being sent to the backend.
    fn generate(&self, prompt: &str, params: &LlmGenerationParams) -> ExecutorResult<String>;

    /// Generate text from a raw prompt WITHOUT applying any chat template.
    ///
    /// Used by codec TTS (NeuTTS) and any caller that has already formatted
    /// the prompt with model-specific control tokens. `params.system_prompt`
    /// is ignored for raw generation — include the system prompt in the
    /// raw prompt text if you need it.
    ///
    /// Default implementation falls back to `generate()` for mocks; the real
    /// `DefaultLlmInference` overrides this to call the backend's raw path.
    fn generate_raw(&self, prompt: &str, params: &LlmGenerationParams) -> ExecutorResult<String> {
        self.generate(prompt, params)
    }

    /// Generate text from backend-neutral multimodal messages.
    fn generate_multimodal(
        &self,
        _messages: &[MultimodalChatMessage],
        _params: &LlmGenerationParams,
    ) -> ExecutorResult<String> {
        Err(AdapterError::InvalidInput(format!(
            "LLM backend '{}' does not support vision input",
            self.backend_name()
        )))
    }

    /// Check if a model is currently loaded.
    fn is_loaded(&self) -> bool;

    /// Get the name of this backend for logging.
    fn backend_name(&self) -> &str;

    /// Canonical wire label of the actual runtime that will execute
    /// inference (`Some("llamacpp")` / `Some("mistralrs")`), or `None`
    /// for mock/test backends that shouldn't claim a real identity.
    ///
    /// Used by the inner LLM span site to overwrite the
    /// template-derived `backend` stamp with the runtime that was
    /// actually selected by cargo feature — see
    /// `runtime_adapter::llm::LlmBackend::wire_label`.
    fn wire_label(&self) -> Option<&'static str> {
        None
    }
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

        // Determine backend hint
        let hint = config
            .backend_hint
            .as_deref()
            .or(self.backend_hint.as_deref());

        // Build the rich LlmConfig so chat_template and context_length
        // reach the backend instead of being dropped by the trait
        // `load_model(path)` wrapper.
        let mut llm_config = LlmConfig::new(&config.model_path);
        llm_config.context_length = config.context_length;
        if let Some(template_path) = config.chat_template.as_ref() {
            llm_config = llm_config.with_chat_template(template_path.clone());
        }
        if let Some(vision_encoder_path) = config.vision_encoder_path.as_ref() {
            llm_config = llm_config.with_vision_encoder(vision_encoder_path.clone());
        }

        // Create adapter with backend hint
        let mut adapter = LlmRuntimeAdapter::with_backend_hint(hint)?;
        adapter.load_model_with_config(&llm_config)?;

        self.adapter = Some(adapter);
        Ok(())
    }

    fn generate(&self, prompt: &str, params: &LlmGenerationParams) -> ExecutorResult<String> {
        use crate::runtime_adapter::llm::GenerationConfig;

        let adapter = self
            .adapter
            .as_ref()
            .ok_or_else(|| AdapterError::RuntimeError("No model loaded".to_string()))?;

        let gen_config = GenerationConfig {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty,
            stop_sequences: params.stop_sequences.clone(),
            tools: params.tools.clone(),
            ..Default::default()
        };

        debug!(
            target: "xybrid_core",
            "LLM generation with {} stop sequences: {:?}",
            gen_config.stop_sequences.len(),
            gen_config.stop_sequences
        );

        let output =
            adapter.generate_with_config(prompt, params.system_prompt.as_deref(), &gen_config)?;

        Ok(output.text)
    }

    fn generate_raw(&self, prompt: &str, params: &LlmGenerationParams) -> ExecutorResult<String> {
        use crate::runtime_adapter::llm::GenerationConfig;

        let adapter = self
            .adapter
            .as_ref()
            .ok_or_else(|| AdapterError::RuntimeError("No model loaded".to_string()))?;

        let gen_config = GenerationConfig {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty,
            stop_sequences: params.stop_sequences.clone(),
            tools: params.tools.clone(),
            ..Default::default()
        };

        debug!(
            target: "xybrid_core",
            "LLM raw generation (no chat template) with {} stop sequences: {:?}",
            gen_config.stop_sequences.len(),
            gen_config.stop_sequences
        );

        let output = adapter
            .backend()
            .generate_raw(prompt, &gen_config)
            .map_err(|e| AdapterError::RuntimeError(format!("LLM raw generation failed: {}", e)))?;

        Ok(output.text)
    }

    fn generate_multimodal(
        &self,
        messages: &[MultimodalChatMessage],
        params: &LlmGenerationParams,
    ) -> ExecutorResult<String> {
        use crate::runtime_adapter::llm::GenerationConfig;

        let adapter = self
            .adapter
            .as_ref()
            .ok_or_else(|| AdapterError::RuntimeError("No model loaded".to_string()))?;

        let gen_config = GenerationConfig {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty,
            stop_sequences: params.stop_sequences.clone(),
            tools: params.tools.clone(),
            ..Default::default()
        };

        let output = adapter
            .backend()
            .generate_multimodal(messages, &gen_config)?;

        Ok(output.text)
    }

    fn is_loaded(&self) -> bool {
        self.adapter.is_some()
    }

    fn backend_name(&self) -> &str {
        self.backend_hint.as_deref().unwrap_or("default")
    }

    fn wire_label(&self) -> Option<&'static str> {
        self.adapter.as_ref().and_then(|a| a.wire_label())
    }
}

// ============================================================================
// No-op Implementation (LLM features disabled)
// ============================================================================

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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

        assert_eq!(params.max_tokens, 2048);
        assert!((params.temperature - 0.7).abs() < 0.001);
        assert!((params.top_p - 0.9).abs() < 0.001);
        assert_eq!(params.top_k, 40);
        assert!(params.system_prompt.is_none());
        assert!(params.stop_sequences.is_empty());
    }

    #[test]
    fn test_generation_params_default_has_empty_tools() {
        let params = LlmGenerationParams::default();

        assert!(params.tools.is_empty());
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
        metadata.insert(
            "stop_sequences".to_string(),
            "<|im_end|>,<|im_start|>".to_string(),
        );

        let params = LlmGenerationParams::from_envelope_metadata(&metadata);

        assert_eq!(params.stop_sequences.len(), 2);
        assert!(params.stop_sequences.contains(&"<|im_end|>".to_string()));
        assert!(params.stop_sequences.contains(&"<|im_start|>".to_string()));
    }

    #[test]
    fn test_parse_stop_sequences_with_spaces() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "stop_sequences".to_string(),
            " STOP , END , HALT ".to_string(),
        );

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
        assert_eq!(params.max_tokens, 2048);
        assert!((params.temperature - 0.7).abs() < 0.001);
    }
}
