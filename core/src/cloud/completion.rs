//! Completion request and response types.
//!
//! These types are provider-agnostic and map to the underlying
//! provider formats (OpenAI, Anthropic, local models, etc.)

use serde::{Deserialize, Serialize};

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message (sets behavior/context).
    System,
    /// User message.
    User,
    /// Assistant response.
    Assistant,
}

impl Default for Role {
    fn default() -> Self {
        Role::User
    }
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens used.
    pub total_tokens: u32,
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
}
