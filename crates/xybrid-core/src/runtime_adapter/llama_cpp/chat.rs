//! Chat prompt formatting policy for the llama.cpp adapter.

use crate::runtime_adapter::llm::LlmResult;
use crate::runtime_adapter::ChatMessage;

/// The reasoning-channel opener primed onto the assistant turn for
/// thinking models. The model continues from here, emitting its
/// chain-of-thought and a closing `</think>` before the final answer.
///
/// Kept in sync with [`THINK_OPEN`](crate::runtime_adapter::streaming_postprocess)
/// semantics: the backend reconstructs the opening tag around the output so
/// the standard `<think>...</think>` capture path applies.
pub(super) const THINK_PRIME: &str = "<think>\n";

pub(super) fn format_chat_prompt(
    model: &xybrid_llama::LlamaModel,
    messages: &[ChatMessage],
    reasoning: bool,
) -> LlmResult<String> {
    let roles: Vec<&str> = messages
        .iter()
        .map(|message| message.role.as_str())
        .collect();
    let contents: Vec<&str> = messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();

    let mut prompt = match xybrid_llama::format_chat(model, &roles, &contents)? {
        Some(prompt) => prompt,
        None => format_chat_chatml(messages),
    };

    // Thinking models (metadata `reasoning: true`) gate their `<think>` block
    // behind the chat template's thinking flag, which llama.cpp's legacy
    // `llama_chat_apply_template` does not render. Prime the channel ourselves so
    // the model emits its chain-of-thought; the backend reconstructs the opening
    // tag for capture. See `.context/reasoning-content-surfacing.md`.
    if reasoning {
        prompt.push_str(THINK_PRIME);
    }

    Ok(prompt)
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
