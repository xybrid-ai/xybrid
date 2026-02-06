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

use crate::ir::{Envelope, EnvelopeKind, MessageRole};
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
///
/// This is the unified ChatMessage type used by the LLM runtime adapter.
/// It uses `MessageRole` from `xybrid_core::ir` to ensure type-safe role handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender (system, user, or assistant)
    pub role: MessageRole,
    /// Message content
    pub content: String,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
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
// Streaming Types
// =============================================================================

/// Partial token emitted during streaming generation.
///
/// This is passed to the callback function during `generate_streaming()` calls.
#[derive(Debug, Clone)]
pub struct PartialToken {
    /// The decoded token text (may be partial UTF-8 for some tokenizers)
    pub token: String,
    /// Raw token ID if available (backend-specific)
    pub token_id: Option<i64>,
    /// Zero-based index of this token in the generation sequence
    pub index: usize,
    /// Cumulative text generated so far (all tokens concatenated)
    pub cumulative_text: String,
    /// Finish reason if this is the final token, None otherwise.
    /// Values: "stop" (hit stop sequence/EOS), "length" (hit max_tokens)
    pub finish_reason: Option<String>,
}

impl PartialToken {
    /// Create a new partial token.
    pub fn new(token: String, index: usize, cumulative_text: String) -> Self {
        Self {
            token,
            token_id: None,
            index,
            cumulative_text,
            finish_reason: None,
        }
    }

    /// Set the token ID.
    pub fn with_token_id(mut self, id: i64) -> Self {
        self.token_id = Some(id);
        self
    }

    /// Mark this as the final token with the given finish reason.
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Check if this is the final token.
    pub fn is_final(&self) -> bool {
        self.finish_reason.is_some()
    }
}

/// Error type for streaming callback failures.
pub type StreamingError = Box<dyn std::error::Error + Send + Sync>;

/// Callback type for streaming token generation.
///
/// This is a boxed function that receives partial tokens during streaming generation.
/// Return `Ok(())` to continue generation, or `Err(...)` to stop.
pub type StreamingCallback<'a> = Box<dyn FnMut(PartialToken) -> Result<(), StreamingError> + Send + 'a>;

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

    /// Number of GPU layers to offload. 0 = CPU only, 99 = all layers on GPU (default).
    /// Use 99 as default to enable GPU acceleration when available.
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,

    /// Enable paged attention for memory efficiency. Default: true
    #[serde(default = "default_paged_attention")]
    pub paged_attention: bool,

    /// Enable logging during inference. Default: false
    #[serde(default)]
    pub logging: bool,

    /// Number of threads for inference. 0 = auto-detect (uses all available cores).
    #[serde(default)]
    pub n_threads: usize,

    /// Batch size for prompt processing. 0 = default (512).
    #[serde(default)]
    pub n_batch: usize,
}

fn default_context_length() -> usize {
    4096
}

fn default_paged_attention() -> bool {
    true
}

fn default_gpu_layers() -> i32 {
    // Default to 99 layers on GPU for maximum performance
    // llama.cpp will automatically use fewer if the model has fewer layers
    // or fall back to CPU if no GPU is available
    99
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            chat_template: None,
            context_length: default_context_length(),
            gpu_layers: default_gpu_layers(),
            paged_attention: default_paged_attention(),
            logging: false,
            n_threads: 0, // 0 = auto-detect
            n_batch: 0,   // 0 = default (512)
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

    /// Set the number of threads for inference. 0 = auto-detect.
    pub fn with_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// Set the batch size for prompt processing. 0 = default (512).
    pub fn with_batch_size(mut self, n_batch: usize) -> Self {
        self.n_batch = n_batch;
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

    /// Generate text with streaming, calling the callback for each token.
    ///
    /// The callback receives a `PartialToken` for each generated token.
    /// If the callback returns an error, generation stops and the error is propagated.
    ///
    /// # Default Implementation
    ///
    /// The default implementation falls back to non-streaming `generate()` and
    /// emits all tokens at once. Override this for true streaming support.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// backend.generate_streaming(&messages, &config, Box::new(|token| {
    ///     print!("{}", token.token);
    ///     std::io::stdout().flush()?;
    ///     Ok(())
    /// }))?;
    /// ```
    fn generate_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        on_token: StreamingCallback<'_>,
    ) -> LlmResult<GenerationOutput> {
        // Default implementation: fall back to non-streaming and emit all at once
        let output = self.generate(messages, config)?;

        // Emit the entire output as a single "token"
        let partial = PartialToken {
            token: output.text.clone(),
            token_id: None,
            index: 0,
            cumulative_text: output.text.clone(),
            finish_reason: Some(output.finish_reason.clone()),
        };

        let mut callback = on_token;
        callback(partial).map_err(|e| {
            AdapterError::RuntimeError(format!("Streaming callback error: {}", e))
        })?;

        Ok(output)
    }

    /// Check if this backend supports true streaming.
    ///
    /// Backends that override `generate_streaming` should return `true`.
    fn supports_streaming(&self) -> bool {
        false
    }

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
    /// - `llm-mistral`: Uses MistralBackend (desktop, not Android)
    /// - `llm-llamacpp`: Uses LlamaCppBackend (Android compatible)
    ///
    /// If both are enabled, prefers MistralBackend on non-Android platforms
    /// and LlamaCppBackend on Android.
    #[cfg(feature = "llm-mistral")]
    pub fn new() -> AdapterResult<Self> {
        use crate::runtime_adapter::mistral::MistralBackend;
        let backend = MistralBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with LlamaCppBackend.
    #[cfg(all(feature = "llm-llamacpp", not(feature = "llm-mistral")))]
    pub fn new() -> AdapterResult<Self> {
        use crate::runtime_adapter::llama_cpp::LlamaCppBackend;
        let backend = LlamaCppBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with MistralBackend explicitly.
    ///
    /// Use this when you need to force MistralBackend regardless of metadata hints.
    #[cfg(feature = "llm-mistral")]
    pub fn with_mistral() -> AdapterResult<Self> {
        use crate::runtime_adapter::mistral::MistralBackend;
        let backend = MistralBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with LlamaCppBackend explicitly.
    ///
    /// Use this when you need to force LlamaCppBackend (e.g., for models that
    /// require it like Gemma 3 which isn't supported by mistral.rs GGUF).
    #[cfg(feature = "llm-llamacpp")]
    pub fn with_llamacpp() -> AdapterResult<Self> {
        use crate::runtime_adapter::llama_cpp::LlamaCppBackend;
        let backend = LlamaCppBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter based on a backend hint.
    ///
    /// If the hint is "llamacpp" and the feature is available, uses LlamaCppBackend.
    /// Otherwise falls back to the default backend for the platform.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    pub fn with_backend_hint(hint: Option<&str>) -> AdapterResult<Self> {
        match hint {
            #[cfg(feature = "llm-llamacpp")]
            Some("llamacpp") => Self::with_llamacpp(),
            #[cfg(feature = "llm-mistral")]
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

    /// Get a reference to the underlying backend.
    ///
    /// This is useful for accessing backend-specific features like streaming.
    pub fn backend(&self) -> &dyn LlmBackend {
        self.backend.as_ref()
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_token_new() {
        let token = PartialToken::new("hello".to_string(), 0, "hello".to_string());
        assert_eq!(token.token, "hello");
        assert_eq!(token.index, 0);
        assert_eq!(token.cumulative_text, "hello");
        assert!(token.token_id.is_none());
        assert!(token.finish_reason.is_none());
        assert!(!token.is_final());
    }

    #[test]
    fn test_partial_token_with_token_id() {
        let token = PartialToken::new("world".to_string(), 1, "hello world".to_string())
            .with_token_id(42);
        assert_eq!(token.token_id, Some(42));
    }

    #[test]
    fn test_partial_token_with_finish_reason() {
        let token = PartialToken::new("".to_string(), 5, "final text".to_string())
            .with_finish_reason("stop");
        assert_eq!(token.finish_reason, Some("stop".to_string()));
        assert!(token.is_final());
    }

    #[test]
    fn test_partial_token_chained_builders() {
        let token = PartialToken::new("token".to_string(), 3, "all tokens".to_string())
            .with_token_id(100)
            .with_finish_reason("length");
        assert_eq!(token.token, "token");
        assert_eq!(token.index, 3);
        assert_eq!(token.token_id, Some(100));
        assert_eq!(token.finish_reason, Some("length".to_string()));
        assert!(token.is_final());
    }

    #[test]
    fn test_chat_message_constructors() {
        let user = ChatMessage::user("hello");
        assert_eq!(user.role, MessageRole::User);
        assert_eq!(user.content, "hello");

        let system = ChatMessage::system("you are helpful");
        assert_eq!(system.role, MessageRole::System);
        assert_eq!(system.content, "you are helpful");

        let assistant = ChatMessage::assistant("hi there");
        assert_eq!(assistant.role, MessageRole::Assistant);
        assert_eq!(assistant.content, "hi there");
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage::user("test");
        let json = serde_json::to_string(&msg).unwrap();
        // Role should serialize to lowercase
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"test\""));

        // Deserialize back
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, MessageRole::User);
        assert_eq!(parsed.content, "test");
    }

    #[test]
    fn test_chat_message_role_as_str() {
        let system = ChatMessage::system("sys");
        let user = ChatMessage::user("usr");
        let assistant = ChatMessage::assistant("ast");

        assert_eq!(system.role.as_str(), "system");
        assert_eq!(user.role.as_str(), "user");
        assert_eq!(assistant.role.as_str(), "assistant");
    }

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert!((config.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 40);
        assert!((config.repetition_penalty - 1.1).abs() < f32::EPSILON);
        assert!(config.stop_sequences.is_empty());
    }

    #[test]
    fn test_generation_config_with_max_tokens() {
        let config = GenerationConfig::default().with_max_tokens(1024);
        assert_eq!(config.max_tokens, 1024);
    }

    #[test]
    fn test_generation_config_with_stop_sequences() {
        let config = GenerationConfig::default()
            .with_stop("<|end|>")
            .with_stop("STOP");
        assert_eq!(config.stop_sequences.len(), 2);
        assert_eq!(config.stop_sequences[0], "<|end|>");
        assert_eq!(config.stop_sequences[1], "STOP");
    }

    #[test]
    fn test_llm_config_defaults() {
        let config = LlmConfig::new("/path/to/model.gguf");
        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.context_length, 4096);
        assert_eq!(config.gpu_layers, 99);
        assert!(config.chat_template.is_none());
        assert!(!config.logging);
        assert!(config.paged_attention); // Default is true for better memory efficiency
    }

    #[test]
    fn test_llm_config_with_context_length() {
        let config = LlmConfig::new("/path/to/model.gguf").with_context_length(8192);
        assert_eq!(config.context_length, 8192);
    }

    #[test]
    fn test_llm_config_with_chat_template() {
        let config = LlmConfig::new("/path/to/model.gguf")
            .with_chat_template("chatml".to_string());
        assert_eq!(config.chat_template, Some("chatml".to_string()));
    }

    #[test]
    fn test_generation_output_structure() {
        let output = GenerationOutput {
            text: "Hello world".to_string(),
            tokens_generated: 3,
            generation_time_ms: 100,
            tokens_per_second: 30.0,
            finish_reason: "stop".to_string(),
        };
        assert_eq!(output.text, "Hello world");
        assert_eq!(output.tokens_generated, 3);
        assert_eq!(output.generation_time_ms, 100);
        assert!((output.tokens_per_second - 30.0).abs() < f32::EPSILON);
        assert_eq!(output.finish_reason, "stop");
    }

    /// Test that the default streaming implementation emits all text as one token
    #[test]
    fn test_default_streaming_implementation() {
        // Create a mock backend that implements LlmBackend with default streaming
        struct MockBackend;

        impl LlmBackend for MockBackend {
            fn name(&self) -> &str { "mock" }
            fn supported_formats(&self) -> Vec<&'static str> { vec!["test"] }
            fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> { Ok(()) }
            fn is_loaded(&self) -> bool { true }
            fn unload(&mut self) -> LlmResult<()> { Ok(()) }
            fn generate(&self, _messages: &[ChatMessage], _config: &GenerationConfig) -> LlmResult<GenerationOutput> {
                Ok(GenerationOutput {
                    text: "Test response".to_string(),
                    tokens_generated: 2,
                    generation_time_ms: 50,
                    tokens_per_second: 40.0,
                    finish_reason: "stop".to_string(),
                })
            }
            fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
                self.generate(&[ChatMessage::user(prompt)], config)
            }
            // Uses default generate_streaming implementation
        }

        let backend = MockBackend;
        assert!(!backend.supports_streaming()); // Default is false

        let messages = vec![ChatMessage::user("test")];
        let config = GenerationConfig::default();

        let mut received_tokens: Vec<PartialToken> = Vec::new();
        let result = backend.generate_streaming(
            &messages,
            &config,
            Box::new(|token| {
                received_tokens.push(token);
                Ok(())
            }),
        );

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.text, "Test response");

        // Default implementation emits entire text as single "token"
        assert_eq!(received_tokens.len(), 1);
        assert_eq!(received_tokens[0].token, "Test response");
        assert_eq!(received_tokens[0].cumulative_text, "Test response");
        assert_eq!(received_tokens[0].finish_reason, Some("stop".to_string()));
        assert!(received_tokens[0].is_final());
    }

    /// Test that callback errors propagate correctly in default streaming
    #[test]
    fn test_default_streaming_callback_error() {
        struct MockBackend;

        impl LlmBackend for MockBackend {
            fn name(&self) -> &str { "mock" }
            fn supported_formats(&self) -> Vec<&'static str> { vec!["test"] }
            fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> { Ok(()) }
            fn is_loaded(&self) -> bool { true }
            fn unload(&mut self) -> LlmResult<()> { Ok(()) }
            fn generate(&self, _messages: &[ChatMessage], _config: &GenerationConfig) -> LlmResult<GenerationOutput> {
                Ok(GenerationOutput {
                    text: "Test".to_string(),
                    tokens_generated: 1,
                    generation_time_ms: 10,
                    tokens_per_second: 100.0,
                    finish_reason: "stop".to_string(),
                })
            }
            fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
                self.generate(&[ChatMessage::user(prompt)], config)
            }
        }

        let backend = MockBackend;
        let messages = vec![ChatMessage::user("test")];
        let config = GenerationConfig::default();

        // Callback that returns an error
        let result = backend.generate_streaming(
            &messages,
            &config,
            Box::new(|_token| {
                Err("User cancelled".into())
            }),
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("callback error") || err_msg.contains("User cancelled"));
    }
}
