//! LLM Backend trait - abstraction over LLM inference engines.
//!
//! This trait allows swapping the underlying LLM engine (mistral.rs, llama.cpp, etc.)
//! without changing the rest of the codebase.

use super::config::{GenerationConfig, LlmConfig};
use crate::runtime_adapter::AdapterError;

/// Result type for LLM backend operations.
pub type LlmResult<T> = Result<T, AdapterError>;

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

/// Trait for LLM inference backends.
///
/// This abstraction allows the LlmRuntimeAdapter to work with different
/// LLM engines (mistral.rs, llama-cpp-rs, etc.) through a common interface.
///
/// # Implementation Notes
///
/// Backends should be stateful - they load a model and keep it in memory
/// for repeated inference calls. This avoids the overhead of loading the
/// model on every request.
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::runtime_adapter::llm::{LlmBackend, LlmConfig, ChatMessage};
///
/// let mut backend = MistralBackend::new()?;
/// backend.load(&LlmConfig::new("model.gguf"))?;
///
/// let messages = vec![ChatMessage::user("Hello!")];
/// let output = backend.generate(&messages, &GenerationConfig::default())?;
/// println!("Response: {}", output.text);
/// ```
pub trait LlmBackend: Send + Sync {
    /// Returns the name of this backend (e.g., "mistral", "llama-cpp").
    fn name(&self) -> &str;

    /// Returns supported model file formats.
    fn supported_formats(&self) -> Vec<&'static str>;

    /// Load a model from the specified configuration.
    ///
    /// After loading, the backend is ready for inference calls.
    fn load(&mut self, config: &LlmConfig) -> LlmResult<()>;

    /// Check if a model is currently loaded.
    fn is_loaded(&self) -> bool;

    /// Unload the current model, freeing resources.
    fn unload(&mut self) -> LlmResult<()>;

    /// Generate text from a list of chat messages.
    ///
    /// This is the primary inference method. The backend applies the chat
    /// template to format the messages, then generates a response.
    ///
    /// # Arguments
    ///
    /// * `messages` - Chat history (system, user, assistant messages)
    /// * `config` - Generation parameters (temperature, max_tokens, etc.)
    ///
    /// # Returns
    ///
    /// Generated output including text and metadata.
    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput>;

    /// Generate text from a raw prompt (no chat template).
    ///
    /// Use this for completion-style inference where you provide the
    /// exact prompt without chat formatting.
    fn generate_raw(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput>;

    /// Get approximate memory usage in bytes.
    ///
    /// Returns None if memory tracking is not supported.
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
