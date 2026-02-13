//! LLM configuration types for the Xybrid SDK.
//!
//! This module provides configuration types for LLM (Large Language Model) clients.
//! These types are used by platform bindings (Flutter, Kotlin, Swift) to configure
//! LLM requests through either the Xybrid Gateway or direct API calls.
//!
//! # Backend Options
//!
//! - **Gateway** (default, recommended): Routes requests through Xybrid's managed gateway,
//!   which handles provider API keys server-side. You only need a Xybrid API key.
//!
//! - **Direct** (development only): Direct API calls to providers. This requires
//!   passing the provider's API key directly, which exposes it to the client.
//!
//! # Example
//!
//! ```rust
//! use xybrid_sdk::llm::{LlmBackend, LlmClientConfig};
//!
//! // Gateway backend (recommended)
//! let config = LlmClientConfig {
//!     backend: LlmBackend::Gateway,
//!     api_key: Some("your-xybrid-api-key".to_string()),
//!     ..Default::default()
//! };
//!
//! // Direct backend (development only)
//! let direct_config = LlmClientConfig {
//!     backend: LlmBackend::Direct,
//!     direct_provider: Some("openai".to_string()),
//!     api_key: Some("sk-...".to_string()),
//!     ..Default::default()
//! };
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// LLM Backend Configuration
// ============================================================================

/// Backend routing for LLM requests.
///
/// Determines how LLM requests are routed - either through the Xybrid Gateway
/// (recommended for production) or directly to provider APIs (development only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LlmBackend {
    /// Route through Xybrid Gateway (default, recommended).
    ///
    /// The gateway provides:
    /// - Secure API key management (keys never exposed to client)
    /// - Rate limiting and usage tracking
    /// - Multiple provider support (OpenAI, Anthropic, etc.)
    #[default]
    Gateway,

    /// Direct API calls (development only - exposes API keys).
    ///
    /// **Warning**: Never ship production apps with API keys embedded in code.
    /// Use the Gateway backend for production, or fetch keys from a secure backend.
    Direct,
}

// ============================================================================
// LLM Client Configuration
// ============================================================================

/// Configuration for LLM client.
///
/// ## API Key Requirements
///
/// - **Gateway backend**: Requires a Xybrid API key (`api_key` field).
///   Get one from <https://xybrid.ai>
///
/// - **Direct backend**: Requires the provider's API key (`api_key` field).
///   For OpenAI, this is your `sk-...` key.
///   For Anthropic, this is your `sk-ant-...` key.
///
/// ## Security Note
///
/// Never embed API keys directly in your app code for production releases.
/// Instead:
/// - Use the Gateway backend (keys are managed server-side)
/// - Or fetch keys from your own secure backend at runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmClientConfig {
    /// Backend to use for requests.
    pub backend: LlmBackend,

    /// Gateway URL (for Gateway backend).
    ///
    /// Defaults to the result of `default_gateway_url()`, which checks
    /// environment variables before falling back to the production URL.
    pub gateway_url: String,

    /// API key for authentication.
    ///
    /// - For Gateway: Your Xybrid API key
    /// - For Direct: The provider's API key (e.g., OpenAI's `sk-...`)
    pub api_key: Option<String>,

    /// Default model to use if not specified in request.
    ///
    /// Examples: `"gpt-4o-mini"`, `"claude-3-5-sonnet"`, `"llama-3.2-1b"`
    pub default_model: Option<String>,

    /// Request timeout in milliseconds.
    ///
    /// Default: 60000 (60 seconds)
    pub timeout_ms: u32,

    /// Provider for Direct backend.
    ///
    /// Required when `backend` is `LlmBackend::Direct`.
    /// Examples: `"openai"`, `"anthropic"`, `"groq"`
    pub direct_provider: Option<String>,

    /// Enable debug logging.
    pub debug: bool,
}

impl Default for LlmClientConfig {
    fn default() -> Self {
        Self {
            backend: LlmBackend::Gateway,
            gateway_url: default_gateway_url(),
            api_key: None,
            default_model: Some("gpt-4o-mini".to_string()),
            timeout_ms: 60000,
            direct_provider: None,
            debug: false,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get default gateway URL from environment or fallback to production URL.
///
/// Priority:
/// 1. `XYBRID_GATEWAY_URL` env var (explicit override, should include `/v1`)
/// 2. `XYBRID_PLATFORM_URL` env var + `/v1` suffix (shared with telemetry)
/// 3. Default production URL (`https://api.xybrid.dev/v1`)
///
/// # Note
///
/// The `/v1` prefix is required for OpenAI-compatible API endpoints.
/// The client appends `/chat/completions`, so the full path becomes `/v1/chat/completions`.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::llm::default_gateway_url;
///
/// // Returns URL based on environment or default
/// let url = default_gateway_url();
/// assert!(url.ends_with("/v1"));
/// ```
pub fn default_gateway_url() -> String {
    if let Ok(url) = std::env::var("XYBRID_GATEWAY_URL") {
        return url;
    }
    if let Ok(url) = std::env::var("XYBRID_PLATFORM_URL") {
        // Platform URL needs /v1 suffix for gateway endpoints
        return format!("{}/v1", url.trim_end_matches('/'));
    }
    "https://api.xybrid.dev/v1".to_string()
}

// ============================================================================
// Message Types
// ============================================================================

// Re-export MessageRole from xybrid-core (single source of truth)
pub use xybrid_core::ir::MessageRole;

/// A message in the conversation.
///
/// Used in multi-turn chat completions to provide conversation history.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::llm::{ChatMessage, MessageRole};
///
/// let system = ChatMessage::system("You are a helpful assistant.".to_string());
/// let user = ChatMessage::user("What is Rust?".to_string());
/// let assistant = ChatMessage::assistant("Rust is a systems programming language.".to_string());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message sender
    pub role: MessageRole,
    /// The content of the message
    pub content: String,
}

impl ChatMessage {
    /// Create a system message.
    ///
    /// System messages set the behavior and context for the assistant.
    pub fn system(content: String) -> Self {
        Self {
            role: MessageRole::System,
            content,
        }
    }

    /// Create a user message.
    ///
    /// User messages represent the human's input in the conversation.
    pub fn user(content: String) -> Self {
        Self {
            role: MessageRole::User,
            content,
        }
    }

    /// Create an assistant message.
    ///
    /// Assistant messages represent previous AI responses in the conversation history.
    pub fn assistant(content: String) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
        }
    }
}

// ============================================================================
// Request Types
// ============================================================================

/// Request for LLM completion.
///
/// Supports both single-turn (prompt) and multi-turn (messages) completions.
///
/// # Single-turn Completion
///
/// ```rust
/// use xybrid_sdk::llm::CompletionRequest;
///
/// let request = CompletionRequest::new("What is the capital of France?".to_string())
///     .with_max_tokens(100)
///     .with_temperature(0.7);
/// ```
///
/// # Multi-turn Chat
///
/// ```rust
/// use xybrid_sdk::llm::{CompletionRequest, ChatMessage};
///
/// let messages = vec![
///     ChatMessage::system("You are a helpful assistant.".to_string()),
///     ChatMessage::user("What is Rust?".to_string()),
/// ];
/// let request = CompletionRequest::chat(messages)
///     .with_model("gpt-4o-mini".to_string());
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model to use (optional - uses default if not specified)
    pub model: Option<String>,
    /// Simple prompt (for single-turn completion)
    pub prompt: Option<String>,
    /// Conversation messages (for multi-turn chat)
    pub messages: Option<Vec<ChatMessage>>,
    /// System prompt
    pub system: Option<String>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0.0 - 2.0)
    pub temperature: Option<f32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
}

impl CompletionRequest {
    /// Create a new completion request with a prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt text for single-turn completion
    pub fn new(prompt: String) -> Self {
        Self {
            prompt: Some(prompt),
            ..Default::default()
        }
    }

    /// Create a chat completion request with messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages: Some(messages),
            ..Default::default()
        }
    }

    /// Set the model.
    ///
    /// # Arguments
    ///
    /// * `model` - Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet")
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the system prompt.
    ///
    /// # Arguments
    ///
    /// * `system` - System message to set context/behavior
    pub fn with_system(mut self, system: String) -> Self {
        self.system = Some(system);
        self
    }

    /// Set max tokens.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum number of tokens to generate
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Sampling temperature (0.0 = deterministic, 2.0 = creative)
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p sampling.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Nucleus sampling probability (0.0 - 1.0)
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences.
    ///
    /// # Arguments
    ///
    /// * `stop` - Sequences that trigger completion stop
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Token usage statistics from an LLM completion.
///
/// Reports the number of tokens consumed by the prompt and generated
/// by the completion. Useful for cost tracking and rate limit management.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::llm::TokenUsage;
///
/// let usage = TokenUsage {
///     prompt_tokens: 10,
///     completion_tokens: 50,
///     total_tokens: 60,
/// };
/// println!("Total tokens used: {}", usage.total_tokens);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens generated in the completion
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion)
    pub total_tokens: u32,
}

/// Response from an LLM completion request.
///
/// Contains the generated text along with metadata about the completion,
/// including token usage, latency, and finish reason.
///
/// Use the [`success()`](CompletionResponse::success) and [`error()`](CompletionResponse::error)
/// constructors to create responses.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::llm::{CompletionResponse, TokenUsage};
///
/// // Success response
/// let response = CompletionResponse::success(
///     "Paris is the capital of France.".to_string(),
///     "gpt-4".to_string(),
///     Some("stop".to_string()),
///     Some(TokenUsage { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 }),
///     Some(150),
///     Some("gateway".to_string()),
/// );
/// assert!(response.success);
///
/// // Error response
/// let error = CompletionResponse::error("Rate limit exceeded".to_string());
/// assert!(!error.success);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Whether the request succeeded
    pub success: bool,
    /// Error message if the request failed
    pub error: Option<String>,
    /// Generated text (empty string on error)
    pub text: String,
    /// Model that was used (empty string on error)
    pub model: String,
    /// Reason why generation stopped (e.g., "stop", "length", "content_filter")
    pub finish_reason: Option<String>,
    /// Token usage statistics
    pub usage: Option<TokenUsage>,
    /// Request latency in milliseconds
    pub latency_ms: Option<u32>,
    /// Backend used (e.g., "gateway", "direct:openai")
    pub backend: Option<String>,
}

impl CompletionResponse {
    /// Create a successful completion response.
    ///
    /// # Arguments
    ///
    /// * `text` - The generated text
    /// * `model` - The model that was used
    /// * `finish_reason` - Why generation stopped (e.g., "stop", "length")
    /// * `usage` - Token usage statistics
    /// * `latency_ms` - Request latency in milliseconds
    /// * `backend` - Backend identifier (e.g., "gateway", "direct:openai")
    pub fn success(
        text: String,
        model: String,
        finish_reason: Option<String>,
        usage: Option<TokenUsage>,
        latency_ms: Option<u32>,
        backend: Option<String>,
    ) -> Self {
        Self {
            success: true,
            error: None,
            text,
            model,
            finish_reason,
            usage,
            latency_ms,
            backend,
        }
    }

    /// Create an error response.
    ///
    /// # Arguments
    ///
    /// * `message` - Description of the error
    pub fn error(message: String) -> Self {
        Self {
            success: false,
            error: Some(message),
            text: String::new(),
            model: String::new(),
            finish_reason: None,
            usage: None,
            latency_ms: None,
            backend: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_backend_default() {
        assert_eq!(LlmBackend::default(), LlmBackend::Gateway);
    }

    #[test]
    fn test_llm_client_config_default() {
        let config = LlmClientConfig::default();
        assert_eq!(config.backend, LlmBackend::Gateway);
        assert_eq!(config.timeout_ms, 60000);
        assert_eq!(config.default_model, Some("gpt-4o-mini".to_string()));
        assert!(config.api_key.is_none());
        assert!(config.direct_provider.is_none());
        assert!(!config.debug);
    }

    #[test]
    fn test_default_gateway_url_fallback() {
        // When no env vars are set, should return production URL
        // Note: This test may fail if XYBRID_GATEWAY_URL or XYBRID_PLATFORM_URL is set
        // in the test environment, so we just check it ends with /v1
        let url = default_gateway_url();
        assert!(
            url.ends_with("/v1"),
            "gateway_url should end with '/v1', got: {}",
            url
        );
    }

    #[test]
    fn test_llm_backend_serialization() {
        let gateway = LlmBackend::Gateway;
        let direct = LlmBackend::Direct;

        let gateway_json = serde_json::to_string(&gateway).unwrap();
        let direct_json = serde_json::to_string(&direct).unwrap();

        assert_eq!(gateway_json, "\"Gateway\"");
        assert_eq!(direct_json, "\"Direct\"");

        // Deserialize back
        let gateway_parsed: LlmBackend = serde_json::from_str(&gateway_json).unwrap();
        let direct_parsed: LlmBackend = serde_json::from_str(&direct_json).unwrap();

        assert_eq!(gateway_parsed, LlmBackend::Gateway);
        assert_eq!(direct_parsed, LlmBackend::Direct);
    }

    #[test]
    fn test_llm_client_config_serialization() {
        let config = LlmClientConfig {
            backend: LlmBackend::Direct,
            gateway_url: "https://test.example.com/v1".to_string(),
            api_key: Some("test-key".to_string()),
            default_model: Some("gpt-4".to_string()),
            timeout_ms: 30000,
            direct_provider: Some("openai".to_string()),
            debug: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: LlmClientConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.backend, LlmBackend::Direct);
        assert_eq!(parsed.gateway_url, "https://test.example.com/v1");
        assert_eq!(parsed.api_key, Some("test-key".to_string()));
        assert_eq!(parsed.default_model, Some("gpt-4".to_string()));
        assert_eq!(parsed.timeout_ms, 30000);
        assert_eq!(parsed.direct_provider, Some("openai".to_string()));
        assert!(parsed.debug);
    }

    // ========================================================================
    // Message Type Tests
    // ========================================================================

    #[test]
    fn test_chat_message_system() {
        let msg = ChatMessage::system("You are helpful".to_string());
        assert_eq!(msg.role, MessageRole::System);
        assert_eq!(msg.content, "You are helpful");
    }

    #[test]
    fn test_chat_message_user() {
        let msg = ChatMessage::user("Hello!".to_string());
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn test_chat_message_assistant() {
        let msg = ChatMessage::assistant("Hi there!".to_string());
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.content, "Hi there!");
    }

    #[test]
    fn test_message_role_serialization() {
        let system = MessageRole::System;
        let user = MessageRole::User;
        let assistant = MessageRole::Assistant;

        let system_json = serde_json::to_string(&system).unwrap();
        let user_json = serde_json::to_string(&user).unwrap();
        let assistant_json = serde_json::to_string(&assistant).unwrap();

        // MessageRole serializes to lowercase (from xybrid-core)
        assert_eq!(system_json, "\"system\"");
        assert_eq!(user_json, "\"user\"");
        assert_eq!(assistant_json, "\"assistant\"");

        // Deserialize back
        let system_parsed: MessageRole = serde_json::from_str(&system_json).unwrap();
        let user_parsed: MessageRole = serde_json::from_str(&user_json).unwrap();
        let assistant_parsed: MessageRole = serde_json::from_str(&assistant_json).unwrap();

        assert_eq!(system_parsed, MessageRole::System);
        assert_eq!(user_parsed, MessageRole::User);
        assert_eq!(assistant_parsed, MessageRole::Assistant);
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage::user("Hello!".to_string());
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.role, MessageRole::User);
        assert_eq!(parsed.content, "Hello!");
    }

    // ========================================================================
    // CompletionRequest Tests
    // ========================================================================

    #[test]
    fn test_completion_request_new() {
        let request = CompletionRequest::new("Hello".to_string());

        assert_eq!(request.prompt, Some("Hello".to_string()));
        assert!(request.messages.is_none());
        assert!(request.model.is_none());
    }

    #[test]
    fn test_completion_request_chat() {
        let messages = vec![
            ChatMessage::system("Be helpful".to_string()),
            ChatMessage::user("Hello".to_string()),
        ];
        let request = CompletionRequest::chat(messages);

        assert!(request.prompt.is_none());
        assert!(request.messages.is_some());
        let msgs = request.messages.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::System);
        assert_eq!(msgs[1].role, MessageRole::User);
    }

    #[test]
    fn test_completion_request_builder() {
        let request = CompletionRequest::new("Hello".to_string())
            .with_model("gpt-4".to_string())
            .with_system("Be helpful".to_string())
            .with_max_tokens(100)
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_stop(vec!["END".to_string()]);

        assert_eq!(request.prompt, Some("Hello".to_string()));
        assert_eq!(request.model, Some("gpt-4".to_string()));
        assert_eq!(request.system, Some("Be helpful".to_string()));
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.top_p, Some(0.9));
        assert_eq!(request.stop, Some(vec!["END".to_string()]));
    }

    #[test]
    fn test_completion_request_serialization() {
        let request = CompletionRequest::new("Hello".to_string())
            .with_model("gpt-4".to_string())
            .with_max_tokens(100);

        let json = serde_json::to_string(&request).unwrap();
        let parsed: CompletionRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.prompt, Some("Hello".to_string()));
        assert_eq!(parsed.model, Some("gpt-4".to_string()));
        assert_eq!(parsed.max_tokens, Some(100));
    }

    #[test]
    fn test_completion_request_with_messages_serialization() {
        let messages = vec![
            ChatMessage::system("System message".to_string()),
            ChatMessage::user("User message".to_string()),
            ChatMessage::assistant("Assistant message".to_string()),
        ];
        let request = CompletionRequest::chat(messages);

        let json = serde_json::to_string(&request).unwrap();
        let parsed: CompletionRequest = serde_json::from_str(&json).unwrap();

        let msgs = parsed.messages.unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, MessageRole::System);
        assert_eq!(msgs[0].content, "System message");
        assert_eq!(msgs[1].role, MessageRole::User);
        assert_eq!(msgs[1].content, "User message");
        assert_eq!(msgs[2].role, MessageRole::Assistant);
        assert_eq!(msgs[2].content, "Assistant message");
    }

    // ========================================================================
    // TokenUsage Tests
    // ========================================================================

    #[test]
    fn test_token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_token_usage_serialization() {
        let usage = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };

        let json = serde_json::to_string(&usage).unwrap();
        let parsed: TokenUsage = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.prompt_tokens, 10);
        assert_eq!(parsed.completion_tokens, 20);
        assert_eq!(parsed.total_tokens, 30);
    }

    // ========================================================================
    // CompletionResponse Tests
    // ========================================================================

    #[test]
    fn test_completion_response_success() {
        let usage = TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 15,
            total_tokens: 20,
        };
        let response = CompletionResponse::success(
            "Hello, world!".to_string(),
            "gpt-4".to_string(),
            Some("stop".to_string()),
            Some(usage),
            Some(150),
            Some("gateway".to_string()),
        );

        assert!(response.success);
        assert!(response.error.is_none());
        assert_eq!(response.text, "Hello, world!");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.finish_reason, Some("stop".to_string()));
        assert_eq!(response.latency_ms, Some(150));
        assert_eq!(response.backend, Some("gateway".to_string()));
        assert!(response.usage.is_some());
        let u = response.usage.unwrap();
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 15);
        assert_eq!(u.total_tokens, 20);
    }

    #[test]
    fn test_completion_response_error() {
        let response = CompletionResponse::error("Connection timeout".to_string());

        assert!(!response.success);
        assert_eq!(response.error, Some("Connection timeout".to_string()));
        assert_eq!(response.text, "");
        assert_eq!(response.model, "");
        assert!(response.finish_reason.is_none());
        assert!(response.usage.is_none());
        assert!(response.latency_ms.is_none());
        assert!(response.backend.is_none());
    }

    #[test]
    fn test_completion_response_serialization() {
        let response = CompletionResponse::success(
            "Test response".to_string(),
            "claude-3".to_string(),
            Some("length".to_string()),
            None,
            Some(200),
            None,
        );

        let json = serde_json::to_string(&response).unwrap();
        let parsed: CompletionResponse = serde_json::from_str(&json).unwrap();

        assert!(parsed.success);
        assert_eq!(parsed.text, "Test response");
        assert_eq!(parsed.model, "claude-3");
        assert_eq!(parsed.finish_reason, Some("length".to_string()));
        assert_eq!(parsed.latency_ms, Some(200));
    }

    #[test]
    fn test_completion_response_error_serialization() {
        let response = CompletionResponse::error("API rate limit exceeded".to_string());

        let json = serde_json::to_string(&response).unwrap();
        let parsed: CompletionResponse = serde_json::from_str(&json).unwrap();

        assert!(!parsed.success);
        assert_eq!(parsed.error, Some("API rate limit exceeded".to_string()));
        assert_eq!(parsed.text, "");
    }
}
