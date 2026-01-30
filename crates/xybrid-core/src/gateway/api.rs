//! Gateway API types - OpenAI-compatible request/response structures.

use serde::{Deserialize, Serialize};

/// Message role in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
    Function,
}

/// A message in the chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message author.
    pub role: MessageRole,

    /// The content of the message.
    pub content: Option<String>,

    /// Name of the author (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls made by the assistant (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID when role is "tool" (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Tool call made by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call.
    pub id: String,

    /// Type of tool (currently only "function").
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function call details.
    pub function: FunctionCall,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call.
    pub name: String,

    /// Arguments to pass to the function (JSON string).
    pub arguments: String,
}

/// Tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Type of tool (currently only "function").
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function definition.
    pub function: FunctionDefinition,
}

/// Function definition for tool use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Name of the function.
    pub name: String,

    /// Description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for function parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Chat completion request (OpenAI-compatible).
///
/// ## Example
/// ```json
/// {
///   "model": "gpt-4o-mini",
///   "messages": [
///     {"role": "system", "content": "You are a helpful assistant."},
///     {"role": "user", "content": "Hello!"}
///   ],
///   "temperature": 0.7,
///   "max_tokens": 100
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID to use for completion.
    /// Examples: "gpt-4o-mini", "claude-3-sonnet", "llama-3.1-70b"
    pub model: String,

    /// Messages in the conversation.
    pub messages: Vec<ChatMessage>,

    /// Sampling temperature (0.0 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Number of completions to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to stream responses.
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Presence penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Modify likelihood of specific tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,

    /// User identifier for abuse tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Tools (functions) available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// How to select tools: "auto", "none", or specific tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,

    /// Response format specification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Seed for deterministic sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

/// Response format specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// Type of response format: "text" or "json_object".
    #[serde(rename = "type")]
    pub format_type: String,
}

/// Chat completion response (OpenAI-compatible).
///
/// ## Example
/// ```json
/// {
///   "id": "chatcmpl-123",
///   "object": "chat.completion",
///   "created": 1677652288,
///   "model": "gpt-4o-mini",
///   "choices": [{
///     "index": 0,
///     "message": {"role": "assistant", "content": "Hello!"},
///     "finish_reason": "stop"
///   }],
///   "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique ID for this completion.
    pub id: String,

    /// Object type (always "chat.completion").
    pub object: String,

    /// Unix timestamp of creation.
    pub created: i64,

    /// Model used for completion.
    pub model: String,

    /// List of completion choices.
    pub choices: Vec<ChatChoice>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// System fingerprint (for caching/debugging).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A completion choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Index of this choice.
    pub index: u32,

    /// The generated message.
    pub message: ChatMessage,

    /// Reason why the model stopped generating.
    /// Values: "stop", "length", "tool_calls", "content_filter"
    pub finish_reason: Option<String>,

    /// Log probabilities (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,

    /// Tokens in the completion.
    pub completion_tokens: u32,

    /// Total tokens used.
    pub total_tokens: u32,
}

/// Streaming chat completion chunk (SSE).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique ID for this completion.
    pub id: String,

    /// Object type (always "chat.completion.chunk").
    pub object: String,

    /// Unix timestamp of creation.
    pub created: i64,

    /// Model used for completion.
    pub model: String,

    /// System fingerprint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    /// List of delta choices.
    pub choices: Vec<ChatChunkChoice>,
}

/// A streaming choice delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    /// Index of this choice.
    pub index: u32,

    /// The delta content.
    pub delta: ChatDelta,

    /// Reason why the model stopped (only in final chunk).
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Role (only in first chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,

    /// Content fragment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls (streamed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Tool call delta in streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Index of this tool call.
    pub index: u32,

    /// Tool call ID (first chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Type (first chunk only).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,

    /// Function details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Function call delta in streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name (first chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Arguments fragment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Error response (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Error message.
    pub message: String,

    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,

    /// Parameter that caused the error (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,

    /// Error code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_constructors() {
        let system = ChatMessage::system("You are helpful.");
        assert_eq!(system.role, MessageRole::System);
        assert_eq!(system.content, Some("You are helpful.".to_string()));

        let user = ChatMessage::user("Hello!");
        assert_eq!(user.role, MessageRole::User);

        let assistant = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant.role, MessageRole::Assistant);
    }

    #[test]
    fn test_request_serialization() {
        let request = ChatCompletionRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![
                ChatMessage::system("Be helpful"),
                ChatMessage::user("Hello"),
            ],
            temperature: Some(0.7),
            top_p: None,
            n: None,
            stream: false,
            stop: None,
            max_tokens: Some(100),
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            seed: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("gpt-4o-mini"));
        assert!(json.contains("system"));
        assert!(json.contains("user"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.choices[0].message.role, MessageRole::Assistant);
        assert_eq!(response.usage.unwrap().total_tokens, 21);
    }
}
