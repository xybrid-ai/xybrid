//! LLM types and adapter - shared abstractions for local LLM inference.
//!
//! This module provides:
//! - `LlmBackend` trait - interface for LLM backends (mistral, llama_cpp)
//! - `LlmConfig` / `GenerationConfig` - configuration types
//! - `LlmRuntimeAdapter` - RuntimeAdapter implementation that uses backends
//!
//! # Architecture
//!
//! ```text
//! LlmRuntimeAdapter (RuntimeAdapter impl)
//!     │
//!     └── LlmBackend (trait - swappable)
//!             │
//!             ├── MistralBackend (mistral.rs - desktop)
//!             └── LlamaCppBackend (llama.cpp - Android + fallback)
//! ```

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{
    AdapterError, AdapterResult, ModelMetadata, RuntimeAdapter, RuntimeAdapterExt,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Result Type
// =============================================================================

/// Result type for LLM backend operations.
pub type LlmResult<T> = Result<T, AdapterError>;

// =============================================================================
// Chat Message
// =============================================================================

/// Chat message for multi-turn conversations.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

// =============================================================================
// Generation Output
// =============================================================================

/// Generation output from LLM inference.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f32,
    /// Finish reason: "stop", "length", "error"
    pub finish_reason: String,
}

// =============================================================================
// LLM Configuration
// =============================================================================

/// Configuration for loading a local LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Path to the model file (GGUF format).
    pub model_path: String,

    /// Path to chat template file (optional).
    pub chat_template: Option<String>,

    /// Maximum context length (tokens). Default: 4096
    #[serde(default = "default_context_length")]
    pub context_length: usize,

    /// Number of GPU layers to offload. 0 = CPU only, -1 = all layers on GPU.
    #[serde(default)]
    pub gpu_layers: i32,

    /// Enable paged attention for memory efficiency. Default: true
    #[serde(default = "default_paged_attention")]
    pub paged_attention: bool,

    /// Enable logging during inference. Default: false
    #[serde(default)]
    pub logging: bool,
}

fn default_context_length() -> usize {
    4096
}

fn default_paged_attention() -> bool {
    true
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            chat_template: None,
            context_length: default_context_length(),
            gpu_layers: 0,
            paged_attention: default_paged_attention(),
            logging: false,
        }
    }
}

impl LlmConfig {
    /// Create a new config with the model path.
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    /// Set the chat template path.
    pub fn with_chat_template(mut self, path: impl Into<String>) -> Self {
        self.chat_template = Some(path.into());
        self
    }

    /// Set the context length.
    pub fn with_context_length(mut self, length: usize) -> Self {
        self.context_length = length;
        self
    }

    /// Set GPU layers to offload.
    pub fn with_gpu_layers(mut self, layers: i32) -> Self {
        self.gpu_layers = layers;
        self
    }

    /// Enable or disable paged attention.
    pub fn with_paged_attention(mut self, enabled: bool) -> Self {
        self.paged_attention = enabled;
        self
    }

    /// Enable or disable logging.
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.logging = enabled;
        self
    }
}

/// Generation parameters for LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate. Default: 256
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for sampling. Default: 0.7
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold. Default: 0.9
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling (0 = disabled). Default: 40
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Repetition penalty (1.0 = disabled). Default: 1.1
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Stop sequences.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

fn default_top_k() -> usize {
    40
}

fn default_repetition_penalty() -> f32 {
    1.1
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            repetition_penalty: default_repetition_penalty(),
            stop_sequences: Vec::new(),
        }
    }
}

impl GenerationConfig {
    /// Create config for greedy decoding (deterministic).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        }
    }

    /// Create config for creative generation.
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Add stop sequence.
    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }
}

// =============================================================================
// LLM Backend Trait
// =============================================================================

/// Trait for LLM inference backends.
///
/// This abstraction allows the LlmRuntimeAdapter to work with different
/// LLM engines (mistral.rs, llama.cpp, etc.) through a common interface.
pub trait LlmBackend: Send + Sync {
    /// Returns the name of this backend (e.g., "mistral", "llama-cpp").
    fn name(&self) -> &str;

    /// Returns supported model file formats.
    fn supported_formats(&self) -> Vec<&'static str>;

    /// Load a model from the specified configuration.
    fn load(&mut self, config: &LlmConfig) -> LlmResult<()>;

    /// Check if a model is currently loaded.
    fn is_loaded(&self) -> bool;

    /// Unload the current model, freeing resources.
    fn unload(&mut self) -> LlmResult<()>;

    /// Generate text from a list of chat messages.
    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput>;

    /// Generate text from a raw prompt (no chat template).
    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput>;

    /// Get approximate memory usage in bytes.
    fn memory_usage(&self) -> Option<u64> {
        None
    }

    /// Get the maximum context length for the loaded model.
    fn context_length(&self) -> Option<usize> {
        None
    }
}

/// Factory function type for creating LLM backends.
pub type BackendFactory = fn() -> LlmResult<Box<dyn LlmBackend>>;

// =============================================================================
// LLM Runtime Adapter
// =============================================================================

/// LLM Runtime Adapter.
///
/// Provides local LLM inference through the RuntimeAdapter interface.
/// Uses a pluggable backend for actual inference.
pub struct LlmRuntimeAdapter {
    /// The underlying LLM backend
    backend: Box<dyn LlmBackend>,
    /// Model metadata
    metadata: Option<ModelMetadata>,
    /// Current model path
    current_model_path: Option<String>,
    /// Default generation config
    default_generation_config: GenerationConfig,
}

impl LlmRuntimeAdapter {
    /// Create a new LLM Runtime Adapter with the default backend.
    ///
    /// The default backend depends on feature flags:
    /// - `local-llm`: Uses MistralBackend (desktop, not Android)
    /// - `local-llm-llamacpp`: Uses LlamaCppBackend (Android compatible)
    ///
    /// If both are enabled, prefers MistralBackend on non-Android platforms
    /// and LlamaCppBackend on Android.
    #[cfg(feature = "local-llm")]
    pub fn new() -> AdapterResult<Self> {
        use crate::runtime_adapter::mistral::MistralBackend;
        let backend = MistralBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with LlamaCppBackend.
    #[cfg(all(feature = "local-llm-llamacpp", not(feature = "local-llm")))]
    pub fn new() -> AdapterResult<Self> {
        use crate::runtime_adapter::llama_cpp::LlamaCppBackend;
        let backend = LlamaCppBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with MistralBackend explicitly.
    ///
    /// Use this when you need to force MistralBackend regardless of metadata hints.
    #[cfg(feature = "local-llm")]
    pub fn with_mistral() -> AdapterResult<Self> {
        use crate::runtime_adapter::mistral::MistralBackend;
        let backend = MistralBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with LlamaCppBackend explicitly.
    ///
    /// Use this when you need to force LlamaCppBackend (e.g., for models that
    /// require it like Gemma 3 which isn't supported by mistral.rs GGUF).
    #[cfg(feature = "local-llm-llamacpp")]
    pub fn with_llamacpp() -> AdapterResult<Self> {
        use crate::runtime_adapter::llama_cpp::LlamaCppBackend;
        let backend = LlamaCppBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter based on a backend hint.
    ///
    /// If the hint is "llamacpp" and the feature is available, uses LlamaCppBackend.
    /// Otherwise falls back to the default backend for the platform.
    #[cfg(any(feature = "local-llm", feature = "local-llm-llamacpp"))]
    pub fn with_backend_hint(hint: Option<&str>) -> AdapterResult<Self> {
        match hint {
            #[cfg(feature = "local-llm-llamacpp")]
            Some("llamacpp") => Self::with_llamacpp(),
            #[cfg(feature = "local-llm")]
            Some("mistral") => Self::with_mistral(),
            _ => Self::new(),
        }
    }

    /// Create a new adapter with a custom backend.
    pub fn with_backend(backend: Box<dyn LlmBackend>) -> Self {
        Self {
            backend,
            metadata: None,
            current_model_path: None,
            default_generation_config: GenerationConfig::default(),
        }
    }

    /// Set the default generation configuration.
    pub fn set_generation_config(&mut self, config: GenerationConfig) {
        self.default_generation_config = config;
    }

    /// Get the current generation configuration.
    pub fn generation_config(&self) -> &GenerationConfig {
        &self.default_generation_config
    }

    /// Get memory usage in bytes (if available).
    pub fn memory_usage(&self) -> Option<u64> {
        self.backend.memory_usage()
    }

    /// Get the context length of the loaded model.
    pub fn context_length(&self) -> Option<usize> {
        self.backend.context_length()
    }

    /// Generate with custom configuration.
    pub fn generate_with_config(
        &self,
        prompt: &str,
        system: Option<&str>,
        config: &GenerationConfig,
    ) -> AdapterResult<GenerationOutput> {
        let mut messages = Vec::new();

        if let Some(sys) = system {
            messages.push(ChatMessage::system(sys));
        }
        messages.push(ChatMessage::user(prompt));

        self.backend.generate(&messages, config)
    }

    /// Extract model ID from path.
    fn extract_model_id(&self, path: &str) -> String {
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Parse generation config from envelope metadata.
    fn parse_generation_config(&self, metadata: &HashMap<String, String>) -> GenerationConfig {
        let mut config = self.default_generation_config.clone();

        if let Some(max_tokens) = metadata.get("max_tokens").and_then(|s| s.parse().ok()) {
            config.max_tokens = max_tokens;
        }
        if let Some(temperature) = metadata.get("temperature").and_then(|s| s.parse().ok()) {
            config.temperature = temperature;
        }
        if let Some(top_p) = metadata.get("top_p").and_then(|s| s.parse().ok()) {
            config.top_p = top_p;
        }
        if let Some(top_k) = metadata.get("top_k").and_then(|s| s.parse().ok()) {
            config.top_k = top_k;
        }

        config
    }
}

impl RuntimeAdapter for LlmRuntimeAdapter {
    fn name(&self) -> &str {
        "llm"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        self.backend.supported_formats()
    }

    fn load_model(&mut self, path: &str) -> AdapterResult<()> {
        let model_path = Path::new(path);

        if !model_path.exists() {
            return Err(AdapterError::ModelNotFound(path.to_string()));
        }

        let config = LlmConfig::new(path);
        self.backend.load(&config)?;

        let model_id = self.extract_model_id(path);

        self.metadata = Some(ModelMetadata {
            model_id: model_id.clone(),
            version: "1.0.0".to_string(),
            runtime_type: self.backend.name().to_string(),
            model_path: path.to_string(),
            input_schema: {
                let mut schema = HashMap::new();
                schema.insert("text".to_string(), vec![1]);
                schema
            },
            output_schema: {
                let mut schema = HashMap::new();
                schema.insert("text".to_string(), vec![1]);
                schema
            },
        });

        self.current_model_path = Some(path.to_string());
        Ok(())
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        if !self.backend.is_loaded() {
            return Err(AdapterError::ModelNotLoaded(
                "No model loaded. Call load_model() first.".to_string(),
            ));
        }

        match &input.kind {
            EnvelopeKind::Text(prompt) => {
                let system = input.metadata.get("system_prompt").map(|s| s.as_str());
                let config = self.parse_generation_config(&input.metadata);

                let mut messages = Vec::new();
                if let Some(sys) = system {
                    messages.push(ChatMessage::system(sys));
                }
                messages.push(ChatMessage::user(prompt));

                let output = self.backend.generate(&messages, &config)?;

                let mut response_metadata = HashMap::new();
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
            EnvelopeKind::Audio(_) => Err(AdapterError::InvalidInput(
                "LLM adapter expects Text input, not Audio".to_string(),
            )),
            EnvelopeKind::Embedding(_) => Err(AdapterError::InvalidInput(
                "LLM adapter expects Text input, not Embedding".to_string(),
            )),
        }
    }
}

impl RuntimeAdapterExt for LlmRuntimeAdapter {
    fn is_loaded(&self, _model_id: &str) -> bool {
        self.backend.is_loaded()
    }

    fn get_metadata(&self, _model_id: &str) -> AdapterResult<&ModelMetadata> {
        self.metadata
            .as_ref()
            .ok_or_else(|| AdapterError::ModelNotLoaded("No model loaded".to_string()))
    }

    fn infer(&self, _model_id: &str, input: &Envelope) -> AdapterResult<Envelope> {
        self.execute(input)
    }

    fn unload_model(&mut self, _model_id: &str) -> AdapterResult<()> {
        self.backend.unload()?;
        self.metadata = None;
        self.current_model_path = None;
        Ok(())
    }

    fn list_loaded_models(&self) -> Vec<String> {
        if self.backend.is_loaded() {
            self.metadata
                .as_ref()
                .map(|m| vec![m.model_id.clone()])
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}
