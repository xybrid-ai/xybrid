//! Response types for LLM API calls.

use serde::{Deserialize, Serialize};

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

/// Response from an LLM API call.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Generated text content.
    pub text: String,

    /// Model used for generation.
    pub model: String,

    /// Finish reason (e.g., "stop", "length", "content_filter").
    pub finish_reason: Option<String>,

    /// Token usage statistics.
    pub usage: Option<Usage>,

    /// Response ID (provider-specific).
    pub id: Option<String>,
}

impl LlmResponse {
    /// Create a new response.
    pub fn new(text: String, model: String) -> Self {
        Self {
            text,
            model,
            finish_reason: None,
            usage: None,
            id: None,
        }
    }

    /// Check if generation stopped due to max tokens.
    pub fn truncated(&self) -> bool {
        matches!(self.finish_reason.as_deref(), Some("length") | Some("max_tokens"))
    }

    /// Check if generation was blocked by content filter.
    pub fn blocked(&self) -> bool {
        matches!(self.finish_reason.as_deref(), Some("content_filter"))
    }
}

// ============================================================================
// OpenAI API Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIChoice {
    pub message: OpenAIMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIError {
    pub error: OpenAIErrorDetail,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    pub code: Option<String>,
}

// ============================================================================
// Anthropic API Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<AnthropicContent>,
    pub stop_reason: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ============================================================================
// Conversion implementations
// ============================================================================

impl From<OpenAIResponse> for LlmResponse {
    fn from(resp: OpenAIResponse) -> Self {
        let choice = resp.choices.first();
        let text = choice
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();
        let finish_reason = choice.and_then(|c| c.finish_reason.clone());

        LlmResponse {
            text,
            model: resp.model,
            finish_reason,
            usage: resp.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
            id: Some(resp.id),
        }
    }
}

impl From<AnthropicResponse> for LlmResponse {
    fn from(resp: AnthropicResponse) -> Self {
        let text = resp
            .content
            .iter()
            .filter_map(|c| {
                if c.content_type == "text" {
                    c.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        LlmResponse {
            text,
            model: resp.model,
            finish_reason: resp.stop_reason,
            usage: resp.usage.map(|u| Usage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            }),
            id: Some(resp.id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_truncated() {
        let mut resp = LlmResponse::new("test".into(), "gpt-4".into());
        assert!(!resp.truncated());

        resp.finish_reason = Some("length".into());
        assert!(resp.truncated());

        resp.finish_reason = Some("max_tokens".into());
        assert!(resp.truncated());
    }

    #[test]
    fn test_response_blocked() {
        let mut resp = LlmResponse::new("test".into(), "gpt-4".into());
        assert!(!resp.blocked());

        resp.finish_reason = Some("content_filter".into());
        assert!(resp.blocked());
    }

    #[test]
    fn test_openai_response_conversion() {
        let openai = OpenAIResponse {
            id: "chatcmpl-123".into(),
            model: "gpt-4o-mini".into(),
            choices: vec![OpenAIChoice {
                message: OpenAIMessage {
                    role: "assistant".into(),
                    content: Some("Hello!".into()),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };

        let resp: LlmResponse = openai.into();
        assert_eq!(resp.text, "Hello!");
        assert_eq!(resp.model, "gpt-4o-mini");
        assert_eq!(resp.finish_reason, Some("stop".into()));
        assert!(resp.usage.is_some());
    }

    #[test]
    fn test_anthropic_response_conversion() {
        let anthropic = AnthropicResponse {
            id: "msg_123".into(),
            model: "claude-3-5-sonnet-20241022".into(),
            content: vec![AnthropicContent {
                content_type: "text".into(),
                text: Some("Hello!".into()),
            }],
            stop_reason: Some("end_turn".into()),
            usage: Some(AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            }),
        };

        let resp: LlmResponse = anthropic.into();
        assert_eq!(resp.text, "Hello!");
        assert_eq!(resp.model, "claude-3-5-sonnet-20241022");
        assert!(resp.usage.is_some());
        let usage = resp.usage.unwrap();
        assert_eq!(usage.total_tokens, 15);
    }
}
