//! Chat prompt formatting policy for the llama.cpp adapter.

use crate::runtime_adapter::llm::LlmResult;
use crate::runtime_adapter::ChatMessage;

pub(super) fn format_chat_prompt(
    model: &xybrid_llama::LlamaModel,
    messages: &[ChatMessage],
) -> LlmResult<String> {
    let roles: Vec<&str> = messages
        .iter()
        .map(|message| message.role.as_str())
        .collect();
    let contents: Vec<&str> = messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();

    if let Some(prompt) = xybrid_llama::format_chat(model, &roles, &contents)? {
        return Ok(prompt);
    }

    Ok(format_chat_chatml(messages))
}

fn format_chat_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content))
            }
            "user" => prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content)),
            "assistant" => prompt.push_str(&format!(
                "<|im_start|>assistant\n{}<|im_end|>\n",
                msg.content
            )),
            _ => prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content)),
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
