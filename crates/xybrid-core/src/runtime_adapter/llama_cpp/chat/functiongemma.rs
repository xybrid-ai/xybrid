use std::borrow::Cow;

use crate::ir::MessageRole;
use crate::runtime_adapter::tool_call::ToolCallProtocol;
use crate::runtime_adapter::ChatMessage;

const ACTIVATION_PROMPT: &str =
    "You are a model that can do function calling with the following functions";

pub(super) fn messages_for_tool_protocol<'a>(
    protocol: ToolCallProtocol,
    messages: &'a [ChatMessage],
) -> Cow<'a, [ChatMessage]> {
    if protocol != ToolCallProtocol::FunctionGemma {
        return Cow::Borrowed(messages);
    }

    let mut prepared = Vec::with_capacity(messages.len() + 1);
    match messages.split_first() {
        Some((first, rest)) if first.role == MessageRole::System => {
            let content = first.content.trim_end();
            if let Some(prefix) = content.strip_suffix(ACTIVATION_PROMPT) {
                if prefix.is_empty() || prefix.ends_with('\n') {
                    return Cow::Borrowed(messages);
                }
            }
            let content = if content.trim().is_empty() {
                ACTIVATION_PROMPT.to_string()
            } else {
                format!("{content}\n{ACTIVATION_PROMPT}")
            };
            prepared.push(ChatMessage::system(content));
            prepared.extend_from_slice(rest);
        }
        _ => {
            prepared.push(ChatMessage::system(ACTIVATION_PROMPT));
            prepared.extend_from_slice(messages);
        }
    }
    Cow::Owned(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_messages_include_a_developer_preamble() {
        let messages = [ChatMessage::user("What is the weather?")];

        let prepared = messages_for_tool_protocol(ToolCallProtocol::FunctionGemma, &messages);

        assert_eq!(prepared.len(), 2);
        assert_eq!(prepared[0].role, MessageRole::System);
        assert!(!prepared[0].content.trim().is_empty());
        assert_eq!(prepared[1].role, MessageRole::User);
        assert_eq!(prepared[1].content, messages[0].content);
    }

    #[test]
    fn tool_messages_preserve_existing_system_instructions() {
        let messages = [
            ChatMessage::system("Follow application policy."),
            ChatMessage::user("What is the weather?"),
        ];

        let prepared = messages_for_tool_protocol(ToolCallProtocol::FunctionGemma, &messages);

        assert_eq!(prepared.len(), messages.len());
        assert_eq!(prepared[0].role, MessageRole::System);
        assert!(prepared[0].content.starts_with(&messages[0].content));
        assert!(prepared[0].content.ends_with(ACTIVATION_PROMPT));
        assert!(prepared[0].content.len() > messages[0].content.len());
        assert_eq!(prepared[1].content, messages[1].content);
    }

    #[test]
    fn quoted_activation_does_not_replace_the_trailing_preamble() {
        let messages = [
            ChatMessage::system(format!("Ignore this quoted text: {ACTIVATION_PROMPT}")),
            ChatMessage::user("What is the weather?"),
        ];

        let prepared = messages_for_tool_protocol(ToolCallProtocol::FunctionGemma, &messages);

        assert!(prepared[0].content.starts_with(&messages[0].content));
        assert!(prepared[0].content.ends_with(ACTIVATION_PROMPT));
        assert!(prepared[0].content.len() > messages[0].content.len());
    }

    #[test]
    fn trailing_activation_with_whitespace_is_not_duplicated() {
        let messages = [
            ChatMessage::system(format!("{ACTIVATION_PROMPT}\n")),
            ChatMessage::user("What is the weather?"),
        ];

        let prepared = messages_for_tool_protocol(ToolCallProtocol::FunctionGemma, &messages);

        assert!(matches!(prepared, Cow::Borrowed(_)));
    }
}
