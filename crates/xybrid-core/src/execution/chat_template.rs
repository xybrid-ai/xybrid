//! Chat template formatting for LLM prompts.
//!
//! This module provides formatters that convert conversation history (Vec<&Envelope>)
//! into formatted prompt strings suitable for different LLM chat formats.
//!
//! # Supported Formats
//!
//! - **ChatML**: The default format used by many models (e.g., OpenAI, Qwen, Yi)
//! - **Llama**: The format used by Meta's Llama 2 models
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_core::execution::chat_template::{ChatTemplateFormat, ChatTemplateFormatter};
//! use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
//!
//! let system = Envelope::new(EnvelopeKind::Text("You are helpful.".into()))
//!     .with_role(MessageRole::System);
//! let user = Envelope::new(EnvelopeKind::Text("Hello!".into()))
//!     .with_role(MessageRole::User);
//!
//! let messages = vec![&system, &user];
//! let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);
//! ```

use crate::ir::{Envelope, EnvelopeKind, MessageRole};

/// Supported chat template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateFormat {
    /// ChatML format: `<|im_start|>role\ncontent<|im_end|>\n`
    /// Used by many models including OpenAI, Qwen, Yi, and others.
    #[default]
    ChatML,

    /// Llama 2 format: `[INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant`
    /// Used by Meta's Llama 2 family of models.
    Llama,

    /// Gemma format: `<start_of_turn>role\ncontent<end_of_turn>\n`
    /// Used by Google's Gemma family of models.
    /// Note: Gemma uses "model" instead of "assistant" for the assistant role.
    Gemma,
}

impl ChatTemplateFormat {
    /// Parse a chat template format from a string.
    ///
    /// Returns `None` for unrecognized formats, allowing the caller to fall back to default.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "chatml" | "chat_ml" | "chat-ml" => Some(Self::ChatML),
            "llama" | "llama2" | "llama-2" => Some(Self::Llama),
            "gemma" | "gemma2" | "gemma3" | "gemma-2" | "gemma-3" => Some(Self::Gemma),
            _ => None,
        }
    }
}

/// Formatter for converting conversation envelopes to LLM prompt strings.
pub struct ChatTemplateFormatter;

impl ChatTemplateFormatter {
    /// Format a conversation into a prompt string using the specified template format.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of envelope references representing the conversation
    /// * `format` - The chat template format to use
    ///
    /// # Returns
    ///
    /// A formatted prompt string suitable for the target LLM.
    ///
    /// # Notes
    ///
    /// - Envelopes without a role are treated as user messages
    /// - Non-text envelopes are skipped
    /// - For Llama format, the system message must be the first message
    pub fn format(messages: &[&Envelope], format: ChatTemplateFormat) -> String {
        match format {
            ChatTemplateFormat::ChatML => Self::format_chatml(messages),
            ChatTemplateFormat::Llama => Self::format_llama(messages),
            ChatTemplateFormat::Gemma => Self::format_gemma(messages),
        }
    }

    /// Format using ChatML template.
    ///
    /// Format: `<|im_start|>role\ncontent<|im_end|>\n`
    fn format_chatml(messages: &[&Envelope]) -> String {
        let mut prompt = String::new();

        for envelope in messages {
            let content = match &envelope.kind {
                EnvelopeKind::Text(text) => text,
                _ => continue, // Skip non-text envelopes
            };

            let role = envelope.role().unwrap_or(MessageRole::User);
            let role_str = role.as_str();

            prompt.push_str("<|im_start|>");
            prompt.push_str(role_str);
            prompt.push('\n');
            prompt.push_str(content);
            prompt.push_str("<|im_end|>\n");
        }

        // Add the assistant start marker for generation
        prompt.push_str("<|im_start|>assistant\n");

        prompt
    }

    /// Format using Llama 2 template.
    ///
    /// Format for system + user: `[INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] `
    /// Format for subsequent turns: `user [INST] user [/INST] assistant`
    fn format_llama(messages: &[&Envelope]) -> String {
        let mut prompt = String::new();
        let mut system_content: Option<&str> = None;
        let mut turns: Vec<(&str, MessageRole)> = Vec::new();

        // First pass: collect messages and extract system prompt
        for envelope in messages {
            let content = match &envelope.kind {
                EnvelopeKind::Text(text) => text.as_str(),
                _ => continue,
            };

            let role = envelope.role().unwrap_or(MessageRole::User);

            if role == MessageRole::System {
                system_content = Some(content);
            } else {
                turns.push((content, role));
            }
        }

        // Build the prompt
        // Llama 2 format groups user/assistant pairs into [INST]...[/INST] blocks
        let mut i = 0;
        while i < turns.len() {
            let (user_content, user_role) = turns[i];

            // Start of a new instruction block
            prompt.push_str("[INST] ");

            // Include system prompt only in the first block
            if i == 0 {
                if let Some(sys) = system_content {
                    prompt.push_str("<<SYS>>\n");
                    prompt.push_str(sys);
                    prompt.push_str("\n<</SYS>>\n\n");
                }
            }

            // Add user content (or treat assistant content as user if misaligned)
            if user_role == MessageRole::User {
                prompt.push_str(user_content);
            } else {
                // This is an assistant message without a preceding user message
                // Treat it as context
                prompt.push_str(user_content);
            }

            prompt.push_str(" [/INST] ");

            // Check if there's an assistant response following
            if i + 1 < turns.len() {
                let (next_content, next_role) = turns[i + 1];
                if next_role == MessageRole::Assistant {
                    prompt.push_str(next_content);
                    i += 2;

                    // Add separator for next turn if there are more messages
                    if i < turns.len() {
                        prompt.push_str(" </s><s>");
                    }
                    continue;
                }
            }

            // No assistant response yet (this is where generation should happen)
            i += 1;
        }

        // Handle edge case: only system message provided
        if turns.is_empty() && system_content.is_some() {
            prompt.push_str("[INST] ");
            prompt.push_str("<<SYS>>\n");
            prompt.push_str(system_content.unwrap());
            prompt.push_str("\n<</SYS>>\n\n");
            prompt.push_str(" [/INST] ");
        }

        prompt
    }

    /// Format using Gemma template.
    ///
    /// Format: `<start_of_turn>role\ncontent<end_of_turn>\n`
    /// Note: Gemma uses "model" instead of "assistant" for the assistant role.
    fn format_gemma(messages: &[&Envelope]) -> String {
        let mut prompt = String::new();

        for envelope in messages {
            let content = match &envelope.kind {
                EnvelopeKind::Text(text) => text,
                _ => continue, // Skip non-text envelopes
            };

            let role = envelope.role().unwrap_or(MessageRole::User);
            // Gemma uses "model" instead of "assistant"
            let role_str = match role {
                MessageRole::System => "user", // Gemma doesn't have a system role, prepend to first user message
                MessageRole::User => "user",
                MessageRole::Assistant => "model",
            };

            // For system messages, we format them as a user turn with special prefix
            if role == MessageRole::System {
                prompt.push_str("<start_of_turn>user\n");
                prompt.push_str("[System Instructions]\n");
                prompt.push_str(content);
                prompt.push_str("<end_of_turn>\n");
            } else {
                prompt.push_str("<start_of_turn>");
                prompt.push_str(role_str);
                prompt.push('\n');
                prompt.push_str(content);
                prompt.push_str("<end_of_turn>\n");
            }
        }

        // Add the model start marker for generation
        prompt.push_str("<start_of_turn>model\n");

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn text_envelope(text: &str, role: MessageRole) -> Envelope {
        Envelope::new(EnvelopeKind::Text(text.to_string())).with_role(role)
    }

    // =========================================================================
    // ChatML Format Tests
    // =========================================================================

    #[test]
    fn test_chatml_single_user_message() {
        let user = text_envelope("Hello!", MessageRole::User);
        let messages = vec![&user];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);

        assert_eq!(
            prompt,
            "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_chatml_system_and_user() {
        let system = text_envelope("You are a helpful assistant.", MessageRole::System);
        let user = text_envelope("What is 2+2?", MessageRole::User);
        let messages = vec![&system, &user];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);

        let expected = "<|im_start|>system\n\
                        You are a helpful assistant.<|im_end|>\n\
                        <|im_start|>user\n\
                        What is 2+2?<|im_end|>\n\
                        <|im_start|>assistant\n";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_chatml_multi_turn_conversation() {
        let system = text_envelope("You are a helpful assistant.", MessageRole::System);
        let user1 = text_envelope("Hello!", MessageRole::User);
        let assistant1 = text_envelope("Hi there! How can I help?", MessageRole::Assistant);
        let user2 = text_envelope("What's the weather?", MessageRole::User);
        let assistant2 = text_envelope("I don't have weather data.", MessageRole::Assistant);
        let user3 = text_envelope("Thanks anyway!", MessageRole::User);

        let messages = vec![&system, &user1, &assistant1, &user2, &assistant2, &user3];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);

        let expected = "<|im_start|>system\n\
                        You are a helpful assistant.<|im_end|>\n\
                        <|im_start|>user\n\
                        Hello!<|im_end|>\n\
                        <|im_start|>assistant\n\
                        Hi there! How can I help?<|im_end|>\n\
                        <|im_start|>user\n\
                        What's the weather?<|im_end|>\n\
                        <|im_start|>assistant\n\
                        I don't have weather data.<|im_end|>\n\
                        <|im_start|>user\n\
                        Thanks anyway!<|im_end|>\n\
                        <|im_start|>assistant\n";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_chatml_envelope_without_role_defaults_to_user() {
        let envelope = Envelope::new(EnvelopeKind::Text("No role set".to_string()));
        let messages = vec![&envelope];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);

        assert!(prompt.contains("<|im_start|>user\nNo role set<|im_end|>"));
    }

    #[test]
    fn test_chatml_skips_non_text_envelopes() {
        let text = text_envelope("Hello", MessageRole::User);
        let audio = Envelope::new(EnvelopeKind::Audio(vec![1, 2, 3])).with_role(MessageRole::User);
        let messages = vec![&audio, &text];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::ChatML);

        // Audio envelope should be skipped
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    // =========================================================================
    // Llama Format Tests
    // =========================================================================

    #[test]
    fn test_llama_single_user_message() {
        let user = text_envelope("Hello!", MessageRole::User);
        let messages = vec![&user];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Llama);

        assert_eq!(prompt, "[INST] Hello! [/INST] ");
    }

    #[test]
    fn test_llama_system_and_user() {
        let system = text_envelope("You are a helpful assistant.", MessageRole::System);
        let user = text_envelope("What is 2+2?", MessageRole::User);
        let messages = vec![&system, &user];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Llama);

        let expected = "[INST] <<SYS>>\n\
                        You are a helpful assistant.\n\
                        <</SYS>>\n\n\
                        What is 2+2? [/INST] ";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_llama_multi_turn_conversation() {
        let system = text_envelope("You are a helpful assistant.", MessageRole::System);
        let user1 = text_envelope("Hello!", MessageRole::User);
        let assistant1 = text_envelope("Hi there!", MessageRole::Assistant);
        let user2 = text_envelope("How are you?", MessageRole::User);
        let assistant2 = text_envelope("I'm doing well!", MessageRole::Assistant);
        let user3 = text_envelope("Great!", MessageRole::User);

        let messages = vec![&system, &user1, &assistant1, &user2, &assistant2, &user3];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Llama);

        // Llama format: system in first [INST], then user/assistant pairs
        let expected = "[INST] <<SYS>>\n\
                        You are a helpful assistant.\n\
                        <</SYS>>\n\n\
                        Hello! [/INST] Hi there! </s><s>\
                        [INST] How are you? [/INST] I'm doing well! </s><s>\
                        [INST] Great! [/INST] ";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_llama_only_system_message() {
        let system = text_envelope("You are a helpful assistant.", MessageRole::System);
        let messages = vec![&system];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Llama);

        // When only a system message is provided, we still generate the instruction block
        // The space after the system block is intentional (user content would go there)
        let expected = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n [/INST] ";
        assert_eq!(prompt, expected);
    }

    // =========================================================================
    // Gemma Format Tests
    // =========================================================================

    #[test]
    fn test_gemma_single_user_message() {
        let user = text_envelope("Hello!", MessageRole::User);
        let messages = vec![&user];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Gemma);

        assert_eq!(
            prompt,
            "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn test_gemma_system_and_user() {
        let system = text_envelope("You are a helpful assistant.", MessageRole::System);
        let user = text_envelope("What is 2+2?", MessageRole::User);
        let messages = vec![&system, &user];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Gemma);

        // System is formatted as a special user turn with [System Instructions] prefix
        let expected = "<start_of_turn>user\n\
                        [System Instructions]\n\
                        You are a helpful assistant.<end_of_turn>\n\
                        <start_of_turn>user\n\
                        What is 2+2?<end_of_turn>\n\
                        <start_of_turn>model\n";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_gemma_multi_turn_conversation() {
        let system = text_envelope("You are helpful.", MessageRole::System);
        let user1 = text_envelope("Hello!", MessageRole::User);
        let assistant1 = text_envelope("Hi there!", MessageRole::Assistant);
        let user2 = text_envelope("How are you?", MessageRole::User);

        let messages = vec![&system, &user1, &assistant1, &user2];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Gemma);

        let expected = "<start_of_turn>user\n\
                        [System Instructions]\n\
                        You are helpful.<end_of_turn>\n\
                        <start_of_turn>user\n\
                        Hello!<end_of_turn>\n\
                        <start_of_turn>model\n\
                        Hi there!<end_of_turn>\n\
                        <start_of_turn>user\n\
                        How are you?<end_of_turn>\n\
                        <start_of_turn>model\n";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_gemma_uses_model_for_assistant() {
        let assistant = text_envelope("I am an assistant.", MessageRole::Assistant);
        let messages = vec![&assistant];

        let prompt = ChatTemplateFormatter::format(&messages, ChatTemplateFormat::Gemma);

        // Gemma uses "model" instead of "assistant"
        assert!(prompt.contains("<start_of_turn>model\nI am an assistant.<end_of_turn>"));
    }

    // =========================================================================
    // Format Parsing Tests
    // =========================================================================

    #[test]
    fn test_format_from_str() {
        assert_eq!(ChatTemplateFormat::from_str("chatml"), Some(ChatTemplateFormat::ChatML));
        assert_eq!(ChatTemplateFormat::from_str("ChatML"), Some(ChatTemplateFormat::ChatML));
        assert_eq!(ChatTemplateFormat::from_str("chat_ml"), Some(ChatTemplateFormat::ChatML));
        assert_eq!(ChatTemplateFormat::from_str("chat-ml"), Some(ChatTemplateFormat::ChatML));

        assert_eq!(ChatTemplateFormat::from_str("llama"), Some(ChatTemplateFormat::Llama));
        assert_eq!(ChatTemplateFormat::from_str("Llama"), Some(ChatTemplateFormat::Llama));
        assert_eq!(ChatTemplateFormat::from_str("llama2"), Some(ChatTemplateFormat::Llama));
        assert_eq!(ChatTemplateFormat::from_str("llama-2"), Some(ChatTemplateFormat::Llama));

        assert_eq!(ChatTemplateFormat::from_str("gemma"), Some(ChatTemplateFormat::Gemma));
        assert_eq!(ChatTemplateFormat::from_str("Gemma"), Some(ChatTemplateFormat::Gemma));
        assert_eq!(ChatTemplateFormat::from_str("gemma2"), Some(ChatTemplateFormat::Gemma));
        assert_eq!(ChatTemplateFormat::from_str("gemma3"), Some(ChatTemplateFormat::Gemma));
        assert_eq!(ChatTemplateFormat::from_str("gemma-3"), Some(ChatTemplateFormat::Gemma));

        assert_eq!(ChatTemplateFormat::from_str("unknown"), None);
        assert_eq!(ChatTemplateFormat::from_str(""), None);
    }

    #[test]
    fn test_format_default_is_chatml() {
        let format: ChatTemplateFormat = Default::default();
        assert_eq!(format, ChatTemplateFormat::ChatML);
    }
}
