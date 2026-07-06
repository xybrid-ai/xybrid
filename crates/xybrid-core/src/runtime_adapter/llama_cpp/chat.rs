//! Chat prompt formatting policy for the llama.cpp adapter.

use super::jinja_template::{JinjaChatTemplate, RenderOptions};
use crate::runtime_adapter::llm::LlmResult;
use crate::runtime_adapter::{AdapterError, ChatMessage};
use xybrid_llama::LlamaError;

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
    let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
    let contents: Vec<&str> = messages.iter().map(|m| m.content.as_str()).collect();

    // Base prompt: the model's native template via llama.cpp, the built-in
    // ChatML fallback when the model has no template, or a minijinja render of
    // the model's own embedded template when llama.cpp's non-Jinja matcher
    // rejects it (e.g. gemma-4 returns code -1).
    let mut prompt = match xybrid_llama::format_chat(model, &roles, &contents) {
        Ok(Some(prompt)) => prompt,
        Ok(None) => format_chat_chatml(messages),
        Err(err @ LlamaError::ChatTemplateFailed { .. }) => {
            render_embedded_fallback(model, messages, err)?
        }
        Err(err) => return Err(err.into()),
    };

    // Thinking models (metadata `reasoning: true`) gate their `<think>` block
    // behind the chat template's thinking flag, which llama.cpp's legacy
    // `llama_chat_apply_template` does not render. Prime the channel ourselves so
    // the model emits its chain-of-thought; the backend reconstructs the opening
    // tag for capture. See `.context/reasoning-content-surfacing.md`.
    //
    // Guard against a doubled tag: if the template (or a future llama.cpp / the
    // minijinja fallback with thinking enabled) already rendered an opening
    // `<think>`, don't prime a second one.
    if reasoning && !prompt.trim_end().ends_with("<think>") {
        prompt.push_str(THINK_PRIME);
    }

    Ok(prompt)
}

/// Render the model's own embedded chat template through minijinja and return
/// it as a raw prompt. This is the fallback for when llama.cpp's non-Jinja
/// matcher rejects an otherwise-valid embedded template (e.g. gemma-4, which
/// returns code -1) — the same engine the MLX path uses.
///
/// Policy: never substitute a different prompt family. If minijinja also fails,
/// surface an error chaining both causes (the original llama rejection and the
/// minijinja failure) rather than silently swapping templates.
///
/// `model.chat_template()` is re-read here and is expected to be `Some`:
/// `format_chat` only returns `ChatTemplateFailed` when a template is present
/// (it returns `Ok(None)` otherwise). The `None` guard is defensive — if that
/// cross-crate invariant ever changed it re-surfaces the original `err` rather
/// than panicking.
fn render_embedded_fallback(
    model: &xybrid_llama::LlamaModel,
    messages: &[ChatMessage],
    err: LlamaError,
) -> LlmResult<String> {
    let Some(template) = model.chat_template() else {
        return Err(err.into());
    };

    // BOS is deliberately suppressed (passed as `None`): `tokenize_chat_prompt`
    // tokenizes this prompt with `add_special = true`, so llama.cpp prepends the
    // BOS token itself. HF Jinja templates (gemma-4's included) emit a literal
    // `{{ bos_token }}` which `parse_special` would turn into a *second* BOS —
    // and gemma degrades badly on a doubled BOS. llama.cpp's own native gemma
    // template likewise emits no leading BOS and relies on `add_special`; we
    // match that contract. This assumes the model adds BOS at tokenization,
    // which holds for every template that reaches this chat path.
    //
    // EOS is passed through for templates that interpolate `{{ eos_token }}` to
    // terminate prior assistant turns; where it appears it is mid-prompt, so
    // `add_special` does not duplicate it. (gemma-4 itself uses the literal
    // `<turn|>` marker and never references `eos_token`, so this is a no-op for
    // the motivating model but keeps other templates correct.)
    let tpl = JinjaChatTemplate::new(template, None, detokenize_special(model, model.token_eos()));

    // `enable_thinking = false`: xybrid's llama.cpp backend is a raw
    // tokenize → generate → detokenize loop with no parser for gemma-4's
    // `<|channel>thought … <channel|>` reasoning blocks (unlike llama.cpp's own
    // chat server). Requesting a reasoning channel would leak its markers and
    // reasoning text into the user-facing output, which the post-processing
    // stop/strip patterns don't yet understand. Asking for a direct answer makes
    // gemma-4's template pre-fill an empty thought and emit the response
    // straight away.
    let options = RenderOptions {
        add_generation_prompt: true,
        enable_thinking: false,
    };

    match tpl.render(messages, &options) {
        Ok(prompt) => Ok(prompt),
        Err(render_err) => {
            tracing::warn!(
                target: "xybrid_core",
                "minijinja chat template fallback failed: {render_err}"
            );
            Err(AdapterError::RuntimeError(format!(
                "llama.cpp did not recognize the embedded chat template ({err}); minijinja fallback also failed ({render_err})"
            )))
        }
    }
}

/// Detokenize a single special token (the EOS token) to its text form for the
/// Jinja context. Never panics: a negative/invalid token id or a detokenize
/// error yields `None`, so the template sees an empty `eos_token`.
fn detokenize_special(model: &xybrid_llama::LlamaModel, token: i32) -> Option<String> {
    if token < 0 {
        return None;
    }
    match model.detokenize(&[token]) {
        Ok(s) if !s.is_empty() => Some(s),
        _ => None,
    }
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
