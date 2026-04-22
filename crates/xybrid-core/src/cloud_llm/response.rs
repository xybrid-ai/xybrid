//! Response types for LLM API calls.

use serde::{Deserialize, Serialize};

/// Token usage statistics. Carrier struct: per-provider fields get
/// normalized into this on parse, and this in turn feeds
/// `cloud::completion::Usage` downstream. See that type's docs for the
/// canonical semantics of the cache fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Effective input tokens across all buckets. For Anthropic this is
    /// synthesized as `input_tokens + cache_read + cache_creation`; for
    /// OpenAI it's the provider's reported `prompt_tokens` unchanged.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens used.
    pub total_tokens: u32,
    /// Prompt tokens served from the provider's prefix cache. See
    /// `cloud::completion::Usage` for provider-specific mapping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    /// Prompt tokens that ESTABLISHED new cache entries. Anthropic-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
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
    /// OpenAI nests cached-prompt-token count under `prompt_tokens_details`;
    /// it's a subset of `prompt_tokens`, not a disjoint bucket.
    #[serde(default)]
    pub prompt_tokens_details: Option<OpenAIPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIPromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
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
    /// Raw input tokens NOT served from cache and NOT establishing new
    /// cache entries. To produce the canonical `prompt_tokens`, sum this
    /// with the two cache fields at conversion time.
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Prompt tokens served from cache. Discounted tier.
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
    /// Prompt tokens that established new cache entries on this request.
    /// Premium tier (~1.25× input). v1 collapses 5m/1h TTL buckets into
    /// one field; if the response carries a nested `cache_creation`
    /// object keyed by TTL, serde will ignore it here.
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
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
            usage: resp.usage.map(|u| {
                let cache_read_input_tokens = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
                Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                    cache_read_input_tokens,
                    // OpenAI doesn't report cache creation; caching is
                    // implicit and billed as a read discount only.
                    cache_creation_input_tokens: None,
                }
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
            usage: resp.usage.map(|u| {
                let cache_read = u.cache_read_input_tokens.unwrap_or(0);
                let cache_creation = u.cache_creation_input_tokens.unwrap_or(0);
                // Anthropic reports `input_tokens` as the uncached-and-
                // not-establishing-cache bucket only. Re-synthesize
                // canonical `prompt_tokens` as the sum of all three
                // buckets so downstream derivation
                // `uncached = prompt_tokens - cache_read - cache_creation`
                // returns raw `input_tokens`.
                let prompt_tokens = u.input_tokens + cache_read + cache_creation;
                Usage {
                    prompt_tokens,
                    completion_tokens: u.output_tokens,
                    total_tokens: prompt_tokens + u.output_tokens,
                    cache_read_input_tokens: u.cache_read_input_tokens,
                    cache_creation_input_tokens: u.cache_creation_input_tokens,
                }
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
                prompt_tokens_details: None,
            }),
        };

        let resp: LlmResponse = openai.into();
        assert_eq!(resp.text, "Hello!");
        assert_eq!(resp.model, "gpt-4o-mini");
        assert_eq!(resp.finish_reason, Some("stop".into()));
        assert!(resp.usage.is_some());
        let usage = resp.usage.unwrap();
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn test_openai_response_with_cached_tokens() {
        let openai = OpenAIResponse {
            id: "chatcmpl-cache".into(),
            model: "gpt-4o".into(),
            choices: vec![OpenAIChoice {
                message: OpenAIMessage {
                    role: "assistant".into(),
                    content: Some("cached".into()),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(OpenAIUsage {
                prompt_tokens: 1000,
                completion_tokens: 50,
                total_tokens: 1050,
                prompt_tokens_details: Some(OpenAIPromptTokensDetails {
                    cached_tokens: Some(400),
                }),
            }),
        };
        let resp: LlmResponse = openai.into();
        let usage = resp.usage.expect("usage present");
        // OpenAI `cached_tokens` is a SUBSET of prompt_tokens — unchanged.
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(usage.cache_read_input_tokens, Some(400));
        assert_eq!(usage.cache_creation_input_tokens, None);
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
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            }),
        };

        let resp: LlmResponse = anthropic.into();
        assert_eq!(resp.text, "Hello!");
        assert_eq!(resp.model, "claude-3-5-sonnet-20241022");
        assert!(resp.usage.is_some());
        let usage = resp.usage.unwrap();
        // With no cache, prompt_tokens == input_tokens.
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn test_anthropic_response_with_cache_buckets() {
        let anthropic = AnthropicResponse {
            id: "msg_cache".into(),
            model: "claude-3-5-sonnet".into(),
            content: vec![AnthropicContent {
                content_type: "text".into(),
                text: Some("reply".into()),
            }],
            stop_reason: Some("end_turn".into()),
            usage: Some(AnthropicUsage {
                input_tokens: 100,
                output_tokens: 20,
                cache_read_input_tokens: Some(500),
                cache_creation_input_tokens: Some(300),
            }),
        };
        let resp: LlmResponse = anthropic.into();
        let usage = resp.usage.expect("usage present");
        // Canonical prompt_tokens = raw input + read + creation.
        assert_eq!(usage.prompt_tokens, 900);
        assert_eq!(usage.cache_read_input_tokens, Some(500));
        assert_eq!(usage.cache_creation_input_tokens, Some(300));
        // Derived uncached = prompt_tokens - read - creation matches raw input.
        let derived_uncached = usage
            .prompt_tokens
            .saturating_sub(usage.cache_read_input_tokens.unwrap_or(0))
            .saturating_sub(usage.cache_creation_input_tokens.unwrap_or(0));
        assert_eq!(derived_uncached, 100);
    }
}
