//! Always-available streaming and chat types for LLM inference.
//!
//! This module contains types that are used by FFI/binding code and need to be
//! available regardless of which LLM backend (if any) is enabled.
//!
//! Types here:
//! - `PartialToken` - Token emitted during streaming generation
//! - `StreamingCallback` - Callback type for streaming token generation
//! - `StreamingError` - Error type for streaming callback failures
//! - `ChatMessage` - Chat message for multi-turn conversations
//! - `GenerationConfig` - Generation parameters for LLM inference
//! - `LlmConfig` - Configuration for loading a local LLM

use crate::ir::MessageRole;
use serde::{Deserialize, Serialize};

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
// Generation Configuration
// =============================================================================

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

    /// Min-p sampling threshold. Default: 0.05
    ///
    /// Prunes tokens with probability below `min_p * max_probability`.
    /// This is more adaptive than top_p: for confident predictions it
    /// aggressively prunes, for uncertain ones it keeps more candidates.
    /// Set to 0.0 to disable.
    #[serde(default = "default_min_p")]
    pub min_p: f32,

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

fn default_min_p() -> f32 {
    0.05
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
            min_p: default_min_p(),
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
        assert!((config.min_p - 0.05).abs() < f32::EPSILON);
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
}
