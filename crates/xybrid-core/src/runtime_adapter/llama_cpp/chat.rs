//! Chat prompt formatting policy for the llama.cpp adapter.
//!
//! Tool-less prompts prefer llama.cpp's native `llama_chat_apply_template`,
//! falling back to built-in ChatML when the model has no template, or to a
//! minijinja render of the model's own embedded template when llama.cpp's
//! non-Jinja matcher rejects it. Tool-bearing prompts skip the native path
//! entirely — `llama_chat_apply_template` takes only role/content pairs and
//! cannot carry tool definitions — and render the embedded template directly
//! with the definitions in the Jinja context. In every case the policy is
//! the same: never silently substitute a different prompt family.

mod functiongemma;

use super::jinja_template::{JinjaChatTemplate, JinjaTemplateError, RenderOptions};
use crate::gateway::Tool;
use crate::runtime_adapter::llm::LlmResult;
use crate::runtime_adapter::tool_call::ToolCallProtocol;
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

/// Format `messages` (plus any tool definitions) into the raw prompt string
/// this model expects.
///
/// Tool-less requests take the native-template → ChatML → minijinja-fallback
/// chain, unchanged. Tool-bearing requests render through
/// [`render_with_tools`] instead, because the native path physically cannot
/// carry tool definitions. Both branches share the `<think>` priming
/// post-processing for reasoning models.
pub(super) fn format_chat_prompt(
    model: &xybrid_llama::LlamaModel,
    messages: &[ChatMessage],
    reasoning: bool,
    tools: &[Tool],
) -> LlmResult<String> {
    let mut prompt = if tools.is_empty() {
        let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
        let contents: Vec<&str> = messages.iter().map(|m| m.content.as_str()).collect();

        // Base prompt: the model's native template via llama.cpp, the built-in
        // ChatML fallback when the model has no template, or a minijinja render of
        // the model's own embedded template when llama.cpp's non-Jinja matcher
        // rejects it (e.g. gemma-4 returns code -1).
        match xybrid_llama::format_chat(model, &roles, &contents) {
            Ok(Some(prompt)) => prompt,
            Ok(None) => format_chat_chatml(messages),
            Err(err @ LlamaError::ChatTemplateFailed { .. }) => {
                render_embedded_fallback(model, messages, err)?
            }
            Err(err) => return Err(err.into()),
        }
    } else {
        render_with_tools(model, messages, tools)?
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

    match render_embedded(model, &template, messages, None) {
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

/// Render a tool-bearing prompt through the model's own embedded chat
/// template, exposing the definitions as the `tools` context array.
///
/// The native path is skipped by design — `llama_chat_apply_template` takes
/// only role/content pairs, so tool definitions can only reach the model
/// through a real Jinja render. Same no-substitution policy as
/// [`render_embedded_fallback`]: a model without an embedded template is an
/// invalid-input error (we will not guess a prompt family and paste tools
/// into it), and a render failure surfaces as an error.
fn render_with_tools(
    model: &xybrid_llama::LlamaModel,
    messages: &[ChatMessage],
    tools: &[Tool],
) -> LlmResult<String> {
    let Some(template) = model.chat_template() else {
        return Err(AdapterError::InvalidInput(
            "model has no embedded chat template, so tool definitions cannot be rendered \
             into the prompt; load a GGUF whose chat template supports tools or drop \
             `tools` from the request"
                .to_string(),
        ));
    };

    // The tool shape is template-dependent, keyed on the template's own
    // protocol markers:
    //
    // - Structured Gemma templates index `tool['function']['name']` — they
    //   need the FULL OpenAI wrapper
    //   ({"type": "function", "function": {...}}).
    // - LFM2-family (and HF-generic `tojson`-the-entry) templates get the
    //   BARE function objects ({name, description, parameters}): they inline
    //   each entry verbatim into their system-prompt tool list, and the
    //   wrapper shape is not what those models were trained on (verified
    //   against the LFM2.5-230M/350M GGUF templates).
    let protocol = ToolCallProtocol::detect_from_template(&template);
    let entries = match protocol {
        ToolCallProtocol::Gemma | ToolCallProtocol::FunctionGemma => tools
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>(),
        ToolCallProtocol::Lfm2 => tools
            .iter()
            .map(|tool| serde_json::to_value(&tool.function))
            .collect::<Result<Vec<_>, _>>(),
    };
    let entries = entries.map_err(|e| {
        AdapterError::InvalidInput(format!("tool definition failed to serialize: {e}"))
    })?;
    let messages = functiongemma::messages_for_tool_protocol(protocol, messages);

    let prompt = render_embedded(
        model,
        &template,
        messages.as_ref(),
        Some(serde_json::Value::Array(entries)),
    )
    .map_err(|render_err| {
        AdapterError::RuntimeError(format!(
            "embedded chat template failed to render tool definitions ({render_err})"
        ))
    })?;
    ensure_tool_names_rendered(&prompt, tools)?;
    Ok(prompt)
}

fn ensure_tool_names_rendered(prompt: &str, tools: &[Tool]) -> LlmResult<()> {
    let mut names = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .filter(|name| !name.is_empty());
    let all_names_rendered = match names.next() {
        Some(first_name) => prompt.contains(first_name) && names.all(|name| prompt.contains(name)),
        None => false,
    };

    if all_names_rendered {
        Ok(())
    } else {
        Err(AdapterError::InvalidInput(
            "model's embedded chat template does not render tool definitions, so tool-bearing \
             requests cannot be honored; load a tool-capable gguf or drop `tools` from the request"
                .to_string(),
        ))
    }
}

/// Shared minijinja plumbing for both embedded-template entry points
/// ([`render_embedded_fallback`] and [`render_with_tools`]), so their
/// BOS/EOS handling and thinking policy cannot drift.
///
/// BOS is deliberately suppressed (passed as `None`): `tokenize_chat_prompt`
/// tokenizes this prompt with `add_special = true`, so llama.cpp prepends the
/// BOS token itself. HF Jinja templates (gemma-4's included) emit a literal
/// `{{ bos_token }}` which `parse_special` would turn into a *second* BOS —
/// and gemma degrades badly on a doubled BOS. llama.cpp's own native gemma
/// template likewise emits no leading BOS and relies on `add_special`; we
/// match that contract. This assumes the model adds BOS at tokenization,
/// which holds for every template that reaches this chat path.
///
/// EOS is passed through for templates that interpolate `{{ eos_token }}` to
/// terminate prior assistant turns; where it appears it is mid-prompt, so
/// `add_special` does not duplicate it. (gemma-4 itself uses the literal
/// `<turn|>` marker and never references `eos_token`, so this is a no-op for
/// the motivating model but keeps other templates correct.)
///
/// `enable_thinking = false`: xybrid's llama.cpp backend is a raw
/// tokenize → generate → detokenize loop with no parser for gemma-4's
/// `<|channel>thought … <channel|>` reasoning blocks (unlike llama.cpp's own
/// chat server). Requesting a reasoning channel would leak its markers and
/// reasoning text into the user-facing output, which the post-processing
/// stop/strip patterns don't yet understand. Asking for a direct answer makes
/// gemma-4's template pre-fill an empty thought and emit the response
/// straight away.
fn render_embedded(
    model: &xybrid_llama::LlamaModel,
    template: &str,
    messages: &[ChatMessage],
    tools: Option<serde_json::Value>,
) -> Result<String, JinjaTemplateError> {
    let tpl = JinjaChatTemplate::new(template, None, detokenize_special(model, model.token_eos()));
    let options = RenderOptions {
        add_generation_prompt: true,
        enable_thinking: false,
        tools,
    };
    tpl.render(messages, &options)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> Tool {
        Tool::function(
            name,
            "test tool",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        )
    }

    #[test]
    fn ensure_tool_names_rendered_returns_invalid_input_when_template_ignores_tools() {
        let tools = vec![tool("get_weather")];

        let err = ensure_tool_names_rendered("<|im_start|>user\nhello<|im_end|>", &tools)
            .expect_err("missing tool name should fail closed");

        match err {
            AdapterError::InvalidInput(message) => {
                assert!(message.contains("does not render tool definitions"));
                assert!(message.contains("drop `tools`"));
            }
            other => panic!("expected invalid input, got {other:?}"),
        }
    }

    #[test]
    fn ensure_tool_names_rendered_succeeds_when_template_echoes_tool_name() {
        let tools = vec![tool("get_weather")];

        ensure_tool_names_rendered("available tools: get_weather", &tools)
            .expect("rendered tool name should be accepted");
    }

    #[test]
    fn ensure_tool_names_rendered_skips_empty_tool_names() {
        let tools = vec![tool(""), tool("get_weather")];

        let err = ensure_tool_names_rendered("available tools:", &tools)
            .expect_err("missing non-empty tool name should fail closed");

        assert!(matches!(err, AdapterError::InvalidInput(_)));
    }

    #[test]
    fn ensure_tool_names_rendered_rejects_only_empty_tool_names() {
        let tools = vec![tool("")];

        let err = ensure_tool_names_rendered("available tools:", &tools)
            .expect_err("empty tool name should not prove tools rendered");

        assert!(matches!(err, AdapterError::InvalidInput(_)));
    }
}
