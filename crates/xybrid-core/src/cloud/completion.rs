//! Completion request and response types.
//!
//! These types are provider-agnostic and map to the underlying
//! provider formats (OpenAI, Anthropic, local models, etc.)

use serde::{Deserialize, Serialize};

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum Role {
    /// System message (sets behavior/context).
    System,
    /// User message.
    #[default]
    User,
    /// Assistant response.
    Assistant,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    pub role: Role,
    /// Content of the message.
    pub content: String,
}

impl Message {
    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }
}

/// Token usage statistics.
///
/// Cache fields follow Anthropic-flavored canonical names because Anthropic
/// is the only major provider that reports both buckets; flattening down
/// to a hit/miss pair would lose the cache-creation premium tier.
///
/// Per-provider mapping (wire field names live in each adapter's doc,
/// not here, so this file stays canonical-only):
/// - **DeepSeek**: hit count → `cache_read_input_tokens`; no cache
///   creation concept → `cache_creation_input_tokens` is `None`.
/// - **Anthropic**: both fields direct. `prompt_tokens` is canonical
///   `input_tokens + cache_read + cache_creation`, not raw `input_tokens`.
/// - **OpenAI**: `prompt_tokens_details.cached_tokens` →
///   `cache_read_input_tokens`; no cache creation reported.
/// - **Gemini** (future): `cached_content_token_count` →
///   `cache_read_input_tokens`.
///
/// Derived (not stored): `uncached_input_tokens = max(0, prompt_tokens -
/// cache_read - cache_creation)`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Total effective input tokens across ALL buckets (uncached +
    /// cache-read + cache-creation). For Anthropic, this is synthesized at
    /// parse time to preserve the sum semantics; for other providers the
    /// raw `prompt_tokens` already has this shape.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens used.
    pub total_tokens: u32,
    /// Prompt tokens served from the provider's prefix cache. Discounted
    /// tier across every provider that reports caching. `None` when the
    /// provider doesn't report cache metrics (Groq, and OpenAI when a
    /// request is too short to be auto-cached).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    /// Prompt tokens that ESTABLISHED new cache entries on this request.
    /// Premium tier (~1.25× input rate on the only provider that reports
    /// it today, Anthropic). `None` / `Some(0)` on every other provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
}

/// Whether a reasoning-capable model may think before it answers.
///
/// DeepSeek enables thinking by default and returns the reasoning in a separate
/// `reasoning_content` field; hidden reasoning still consumes the output token
/// budget, so a small `max_tokens` can be exhausted before any answer appears.
/// [`ThinkingMode::Disabled`] requests answer-only generation.
///
/// On the OpenAI-compatible wire the mode is serialized as
/// `"thinking": {"type": "enabled" | "disabled"}`. A `None` on
/// [`CompletionRequest::thinking`] omits the field so the provider default
/// applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    /// The model may emit hidden reasoning before its answer.
    Enabled,
    /// Answer-only generation; no reasoning budget is spent.
    Disabled,
}

impl ThinkingMode {
    /// Wire value: `"enabled"` or `"disabled"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkingMode::Enabled => "enabled",
            ThinkingMode::Disabled => "disabled",
        }
    }
}

impl std::fmt::Display for ThinkingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ThinkingMode {
    type Err = String;

    /// Parse `enabled` / `disabled`, case-insensitively and ignoring
    /// surrounding whitespace. Any other value is an error.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "enabled" => Ok(ThinkingMode::Enabled),
            "disabled" => Ok(ThinkingMode::Disabled),
            _ => Err(format!(
                "invalid thinking mode '{}': expected 'enabled' or 'disabled'",
                s.trim()
            )),
        }
    }
}

/// Request for cloud completion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier (optional - uses default if not specified).
    /// Examples: "gpt-4o-mini", "claude-3-5-sonnet", "llama-3.2-1b"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Simple prompt (for single-turn completion).
    /// Either `prompt` or `messages` should be provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// Conversation messages (for multi-turn chat).
    /// Takes precedence over `prompt` if both are provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,

    /// System prompt (prepended as system message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Stream the response (token-by-token).
    #[serde(default)]
    pub stream: bool,

    /// Thinking mode for reasoning-capable providers (DeepSeek).
    ///
    /// `None` omits the field from the request body so the provider default
    /// applies. Honored by the OpenAI-compatible transport; the native direct
    /// clients in `cloud_llm` have no equivalent and ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingMode>,
}

impl CompletionRequest {
    /// Create a new completion request with a simple prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: Some(prompt.into()),
            ..Default::default()
        }
    }

    /// Create a chat request with messages.
    pub fn chat(messages: Vec<Message>) -> Self {
        Self {
            messages: Some(messages),
            ..Default::default()
        }
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p (nucleus) sampling.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Enable streaming.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set the thinking mode (DeepSeek `"thinking": {"type": ...}`).
    pub fn with_thinking(mut self, mode: ThinkingMode) -> Self {
        self.thinking = Some(mode);
        self
    }

    /// Convert to messages format.
    pub fn to_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        if let Some(ref system) = self.system {
            messages.push(Message::system(system.clone()));
        }

        if let Some(ref msgs) = self.messages {
            messages.extend(msgs.clone());
        } else if let Some(ref prompt) = self.prompt {
            messages.push(Message::user(prompt.clone()));
        }

        messages
    }
}

/// Response from cloud completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Generated text content.
    pub text: String,

    /// Model used for generation.
    pub model: String,

    /// Finish reason (e.g., "stop", "length", "content_filter").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// Response ID (provider-specific).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u32>,

    /// Backend that served the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

impl CompletionResponse {
    /// Create a new response.
    pub fn new(text: String, model: String) -> Self {
        Self {
            text,
            model,
            finish_reason: None,
            usage: None,
            id: None,
            latency_ms: None,
            backend: None,
        }
    }

    /// Check if generation stopped due to max tokens.
    pub fn truncated(&self) -> bool {
        matches!(
            self.finish_reason.as_deref(),
            Some("length") | Some("max_tokens")
        )
    }

    /// Check if generation was blocked by content filter.
    pub fn blocked(&self) -> bool {
        matches!(self.finish_reason.as_deref(), Some("content_filter"))
    }
}

// Conversions from cloud_llm types
impl From<crate::cloud_llm::LlmResponse> for CompletionResponse {
    fn from(resp: crate::cloud_llm::LlmResponse) -> Self {
        Self {
            text: resp.text,
            model: resp.model,
            finish_reason: resp.finish_reason,
            usage: resp.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                cache_read_input_tokens: u.cache_read_input_tokens,
                cache_creation_input_tokens: u.cache_creation_input_tokens,
            }),
            id: resp.id,
            latency_ms: None,
            backend: Some("direct".to_string()),
        }
    }
}

impl From<CompletionRequest> for crate::cloud_llm::LlmRequest {
    fn from(req: CompletionRequest) -> Self {
        let mut llm_req = if let Some(ref prompt) = req.prompt {
            crate::cloud_llm::LlmRequest::prompt(prompt)
        } else if let Some(ref messages) = req.messages {
            let msgs: Vec<crate::cloud_llm::Message> = messages
                .iter()
                .map(|m| crate::cloud_llm::Message {
                    role: match m.role {
                        Role::System => crate::cloud_llm::Role::System,
                        Role::User => crate::cloud_llm::Role::User,
                        Role::Assistant => crate::cloud_llm::Role::Assistant,
                    },
                    content: m.content.clone(),
                })
                .collect();
            crate::cloud_llm::LlmRequest::chat(msgs)
        } else {
            crate::cloud_llm::LlmRequest::default()
        };

        if let Some(model) = req.model {
            llm_req = llm_req.with_model(model);
        }
        if let Some(system) = req.system {
            llm_req = llm_req.with_system(system);
        }
        if let Some(max_tokens) = req.max_tokens {
            llm_req = llm_req.with_max_tokens(max_tokens);
        }
        if let Some(temperature) = req.temperature {
            llm_req = llm_req.with_temperature(temperature);
        }
        if let Some(stop) = req.stop {
            llm_req = llm_req.with_stop(stop);
        }
        // `thinking` has no `LlmRequest` equivalent: the native direct clients
        // are not reasoning-mode aware, and the adapter only accepts the option
        // for DeepSeek, which rides the OpenAI-compatible transport instead.

        llm_req
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_request_new() {
        let req = CompletionRequest::new("Hello");
        assert_eq!(req.prompt, Some("Hello".to_string()));
    }

    #[test]
    fn test_completion_request_builder() {
        let req = CompletionRequest::new("Test")
            .with_model("gpt-4o-mini")
            .with_system("Be concise")
            .with_max_tokens(100)
            .with_temperature(0.5);

        assert_eq!(req.model, Some("gpt-4o-mini".to_string()));
        assert_eq!(req.system, Some("Be concise".to_string()));
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.temperature, Some(0.5));
    }

    #[test]
    fn test_to_messages() {
        let req = CompletionRequest::new("Hello").with_system("Be helpful");
        let messages = req.to_messages();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
    }

    #[test]
    fn test_response_truncated() {
        let mut resp = CompletionResponse::new("test".into(), "gpt-4".into());
        assert!(!resp.truncated());

        resp.finish_reason = Some("length".into());
        assert!(resp.truncated());
    }

    #[test]
    fn thinking_defaults_to_none_and_is_omitted_from_json() {
        let req = CompletionRequest::new("Hello");
        assert_eq!(req.thinking, None);

        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("thinking").is_none(), "got {json}");
    }

    #[test]
    fn with_thinking_sets_the_mode() {
        let req = CompletionRequest::new("Hello").with_thinking(ThinkingMode::Disabled);
        assert_eq!(req.thinking, Some(ThinkingMode::Disabled));
        assert_eq!(ThinkingMode::Disabled.as_str(), "disabled");
        assert_eq!(ThinkingMode::Enabled.to_string(), "enabled");
    }

    #[test]
    fn thinking_mode_parses_case_insensitively_and_rejects_other_values() {
        assert_eq!(
            "disabled".parse::<ThinkingMode>(),
            Ok(ThinkingMode::Disabled)
        );
        assert_eq!(
            " Enabled ".parse::<ThinkingMode>(),
            Ok(ThinkingMode::Enabled)
        );
        assert_eq!(
            "DISABLED".parse::<ThinkingMode>(),
            Ok(ThinkingMode::Disabled)
        );

        let err = "maybe".parse::<ThinkingMode>().unwrap_err();
        assert!(err.contains("maybe"), "got {err}");
        assert!(err.contains("'enabled' or 'disabled'"), "got {err}");
        assert!("".parse::<ThinkingMode>().is_err());
        assert!("true".parse::<ThinkingMode>().is_err());
    }
}
