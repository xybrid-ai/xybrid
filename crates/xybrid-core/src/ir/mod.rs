//! Intermediate Representation (IR) module - Data serialization layer.
//!
//! This module defines the Envelope IR, which is the serialization layer that
//! defines how data flows between pipeline stages. Envelopes encapsulate typed
//! payloads (audio, text, embeddings) and can be serialized for storage or
//! transmission between local processes or over HTTP to cloud endpoints.

pub mod envelope;

pub use envelope::{AudioSamples, Envelope, EnvelopeKind};
// Note: MessageRole is defined below in this file and is public by default

/// Message role in a conversation.
///
/// Defines the role of each message in a chat completion or conversation context.
/// This is the single source of truth for message roles across xybrid-core and xybrid-sdk.
///
/// # Serialization
///
/// Serializes to lowercase strings for JSON compatibility:
/// - `System` → `"system"`
/// - `User` → `"user"`
/// - `Assistant` → `"assistant"`
///
/// # Example
///
/// ```rust
/// use xybrid_core::ir::MessageRole;
///
/// let role = MessageRole::User;
/// assert_eq!(serde_json::to_string(&role).unwrap(), "\"user\"");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message - sets the behavior and context for the assistant
    System,
    /// User message - the human's input
    User,
    /// Assistant message - the AI's response
    Assistant,
}

impl MessageRole {
    /// Returns the role as a lowercase string.
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_role_serialization_lowercase() {
        assert_eq!(serde_json::to_string(&MessageRole::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&MessageRole::User).unwrap(), "\"user\"");
        assert_eq!(serde_json::to_string(&MessageRole::Assistant).unwrap(), "\"assistant\"");
    }

    #[test]
    fn test_message_role_deserialization_lowercase() {
        assert_eq!(serde_json::from_str::<MessageRole>("\"system\"").unwrap(), MessageRole::System);
        assert_eq!(serde_json::from_str::<MessageRole>("\"user\"").unwrap(), MessageRole::User);
        assert_eq!(serde_json::from_str::<MessageRole>("\"assistant\"").unwrap(), MessageRole::Assistant);
    }

    #[test]
    fn test_message_role_as_str() {
        assert_eq!(MessageRole::System.as_str(), "system");
        assert_eq!(MessageRole::User.as_str(), "user");
        assert_eq!(MessageRole::Assistant.as_str(), "assistant");
    }

    #[test]
    fn test_message_role_display() {
        assert_eq!(format!("{}", MessageRole::System), "system");
        assert_eq!(format!("{}", MessageRole::User), "user");
        assert_eq!(format!("{}", MessageRole::Assistant), "assistant");
    }
}
