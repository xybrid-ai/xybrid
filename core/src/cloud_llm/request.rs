//! Request types for LLM API calls.

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

/// Request for LLM completion/chat.
#[derive(Debug, Clone, Default)]
pub struct LlmRequest {
    /// Model ID to use (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022").
    /// If not specified, uses provider's default model.
    pub model: Option<String>,

    /// Simple prompt (for single-turn completion).
    /// Either `prompt` or `messages` should be provided.
    pub prompt: Option<String>,

    /// Conversation messages (for multi-turn chat).
    /// Takes precedence over `prompt` if both are provided.
    pub messages: Option<Vec<Message>>,

    /// System prompt (prepended as system message).
    pub system: Option<String>,

    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 - 2.0).
    /// Lower = more deterministic, higher = more creative.
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling.
    pub top_p: Option<f32>,

    /// Top-k sampling (Anthropic only).
    pub top_k: Option<u32>,

    /// Stop sequences (generation stops when any is encountered).
    pub stop: Option<Vec<String>>,

    /// Frequency penalty (-2.0 - 2.0, OpenAI only).
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 - 2.0, OpenAI only).
    pub presence_penalty: Option<f32>,
}

impl LlmRequest {
    /// Create a simple prompt request.
    pub fn prompt(text: impl Into<String>) -> Self {
        Self {
            prompt: Some(text.into()),
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

    /// Convert to messages format (handles prompt vs messages).
    pub fn to_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(ref system) = self.system {
            messages.push(Message::system(system.clone()));
        }

        // Use messages if provided, otherwise convert prompt
        if let Some(ref msgs) = self.messages {
            messages.extend(msgs.clone());
        } else if let Some(ref prompt) = self.prompt {
            messages.push(Message::user(prompt.clone()));
        }

        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_constructors() {
        let user = Message::user("Hello");
        assert_eq!(user.role, Role::User);
        assert_eq!(user.content, "Hello");

        let assistant = Message::assistant("Hi there!");
        assert_eq!(assistant.role, Role::Assistant);

        let system = Message::system("Be helpful");
        assert_eq!(system.role, Role::System);
    }

    #[test]
    fn test_request_prompt() {
        let request = LlmRequest::prompt("Tell me a joke");
        assert_eq!(request.prompt, Some("Tell me a joke".to_string()));
        assert!(request.messages.is_none());
    }

    #[test]
    fn test_request_chat() {
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi!"),
            Message::user("How are you?"),
        ];
        let request = LlmRequest::chat(messages.clone());
        assert_eq!(request.messages.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_request_builder() {
        let request = LlmRequest::prompt("Test")
            .with_model("gpt-4o-mini")
            .with_system("Be concise")
            .with_max_tokens(100)
            .with_temperature(0.5);

        assert_eq!(request.model, Some("gpt-4o-mini".to_string()));
        assert_eq!(request.system, Some("Be concise".to_string()));
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.5));
    }

    #[test]
    fn test_to_messages_with_system() {
        let request = LlmRequest::prompt("Hello")
            .with_system("Be helpful");

        let messages = request.to_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
    }
}
