//! Minijinja fallback renderer for a GGUF's embedded chat template.
//!
//! llama.cpp ships a hardcoded, *non-Jinja* matcher for chat templates: it
//! recognises a fixed set of known families by sniffing for literal markers
//! (`<start_of_turn>`, `<|im_start|>`, …) in the embedded template string.
//! When a GGUF carries a real Jinja template that the matcher can't classify
//! — e.g. gemma-4's 16 KB tool-calling template, which contains no literal
//! `<start_of_turn>` — `llama_chat_apply_template` returns error code `-1`
//! and native formatting hard-fails.
//!
//! This module renders that same embedded template through [`minijinja`], a
//! real Jinja engine, so the resulting raw prompt can be fed back into
//! llama.cpp directly. It is the exact engine and context shape the MLX
//! adapter uses for HuggingFace templates. It is used *only* as a fallback by
//! [`super::chat::format_chat_prompt`] when native formatting reports a
//! template is present but unrenderable; if minijinja also fails, the caller
//! surfaces an error chaining both the original llama rejection and this
//! render failure.
//!
//! There is no file I/O here: the template string and BOS/EOS tokens are
//! supplied by the caller, which already holds the loaded [`LlamaModel`].

use minijinja::{context, Environment, Value};
use minijinja_contrib::pycompat::unknown_method_callback;

use crate::runtime_adapter::ChatMessage;

/// Error from compiling or rendering a GGUF's embedded Jinja chat template.
#[derive(Debug, thiserror::Error)]
pub(super) enum JinjaTemplateError {
    /// Template failed to compile (Jinja2 syntax error) or render (undefined
    /// variable, type mismatch, etc.). Wraps the minijinja error message.
    #[error("chat template render failed: {0}")]
    Render(String),
}

/// Options the caller passes at render time. Kept tiny on purpose —
/// expanding it is cheaper than inventing a second rendering path.
#[derive(Debug, Clone)]
pub(super) struct RenderOptions {
    /// Whether to append the assistant-turn scaffolding so the LLM continues
    /// mid-turn. Almost always `true` when the caller plans to call
    /// `generate` on the rendered prompt.
    pub add_generation_prompt: bool,
    /// Whether reasoning-model templates should enter their thinking mode.
    /// Templates that honor this switch skip reasoning-channel scaffolding
    /// for direct user-facing answers.
    pub enable_thinking: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
        }
    }
}

/// A GGUF's embedded Jinja chat template, ready to render message lists into
/// raw prompt text.
#[derive(Debug, Clone)]
pub(super) struct JinjaChatTemplate {
    /// The raw Jinja template string, as embedded in the GGUF.
    template: String,
    /// Optional BOS token (e.g. `<s>`) passed into the Jinja context as
    /// `bos_token` for templates that interpolate it.
    bos_token: Option<String>,
    /// Optional EOS token (e.g. `</s>`) — same passthrough.
    eos_token: Option<String>,
}

impl JinjaChatTemplate {
    /// Construct a template from a raw Jinja string plus optional BOS/EOS
    /// tokens. The tokens flow into the render context; pass `None` when the
    /// model exposes no special-token text.
    pub(super) fn new(
        template: impl Into<String>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        Self {
            template: template.into(),
            bos_token,
            eos_token,
        }
    }

    /// Render a message list to a prompt string. The Jinja context mirrors
    /// the HuggingFace transformers tokenizer's `apply_chat_template`, so a
    /// template copied verbatim from a GGUF renders without modification.
    pub(super) fn render(
        &self,
        messages: &[ChatMessage],
        options: &RenderOptions,
    ) -> Result<String, JinjaTemplateError> {
        // Build an Environment per-call. minijinja's envs are cheap to
        // construct (~1us) and the alternative — caching a compiled template
        // on the struct — would require a self-referential lifetime or
        // `Arc<Environment<'static>>` with a leaked template string. The
        // per-call cost is invisible relative to the forward pass.
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_template("chat", &self.template)
            .map_err(|e| JinjaTemplateError::Render(format!("compile: {e}")))?;

        // Messages flow into the template as a list of `{role, content}` maps —
        // the shape every HF template expects. `context!` builds each map as a
        // minijinja `Value` directly, without an intermediate `serde_json::Value`.
        let messages_value: Vec<Value> = messages
            .iter()
            .map(|m| {
                context! {
                    role => m.role.as_str(),
                    content => m.content.as_str(),
                }
            })
            .collect();

        let ctx = context! {
            messages => messages_value,
            add_generation_prompt => options.add_generation_prompt,
            enable_thinking => options.enable_thinking,
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone().unwrap_or_default(),
        };

        let tmpl = env
            .get_template("chat")
            .map_err(|e| JinjaTemplateError::Render(format!("lookup: {e}")))?;
        tmpl.render(ctx)
            .map_err(|e| JinjaTemplateError::Render(format!("render: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_supports_macro_templates() {
        let tmpl = JinjaChatTemplate::new(
            concat!(
                "{%- macro turn(message) -%}",
                "<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n",
                "{%- endmacro -%}",
                "{%- for message in messages -%}",
                "{{ turn(message) }}",
                "{%- endfor -%}",
                "{%- if add_generation_prompt -%}<start_of_turn>assistant\n{%- endif -%}"
            ),
            None,
            None,
        );
        let out = tmpl
            .render(&[ChatMessage::user("hello")], &RenderOptions::default())
            .unwrap();
        assert!(out.contains("<start_of_turn>user\nhello<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>assistant"));
    }

    #[test]
    fn render_applies_trim_filter() {
        let tmpl = JinjaChatTemplate::new("{{ messages[0].content | trim }}", None, None);
        let out = tmpl
            .render(&[ChatMessage::user(" hi ")], &RenderOptions::default())
            .unwrap();
        assert_eq!(out, "hi");
    }

    #[test]
    fn bos_eos_flow_through_to_context() {
        let tmpl = JinjaChatTemplate::new(
            "{{ bos_token }}[{{ messages[0].content }}]{{ eos_token }}",
            Some("<s>".into()),
            Some("</s>".into()),
        );
        let out = tmpl
            .render(&[ChatMessage::user("x")], &RenderOptions::default())
            .unwrap();
        assert_eq!(out, "<s>[x]</s>");
    }

    #[test]
    fn add_generation_prompt_false_skips_scaffold() {
        let tmpl = JinjaChatTemplate::new(
            concat!(
                "{%- macro turn(message) -%}",
                "<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n",
                "{%- endmacro -%}",
                "{%- for message in messages -%}",
                "{{ turn(message) }}",
                "{%- endfor -%}",
                "{%- if add_generation_prompt -%}<start_of_turn>assistant\n{%- endif -%}"
            ),
            None,
            None,
        );
        let opts = RenderOptions {
            add_generation_prompt: false,
            enable_thinking: true,
        };
        let out = tmpl.render(&[ChatMessage::user("hello")], &opts).unwrap();
        assert!(!out.contains("<start_of_turn>assistant"));
    }

    /// Mirrors gemma-4's `enable_thinking` gating: with thinking off the
    /// template must NOT inject a `<|think|>` reasoning block and instead
    /// pre-fills an empty thought, steering the model to answer directly.
    /// This is the shape the llama.cpp fallback requests (it cannot parse
    /// reasoning channels in the raw-decode path).
    #[test]
    fn enable_thinking_false_suppresses_reasoning_block() {
        let gemma4_shaped = concat!(
            "{%- if enable_thinking -%}<|turn>system\n<|think|>\n<turn|>\n{%- endif -%}",
            "{%- for message in messages -%}",
            "<|turn>{{ message.role }}\n{{ message.content }}<turn|>\n",
            "{%- endfor -%}",
            "{%- if add_generation_prompt -%}<|turn>model\n",
            "{%- if not enable_thinking -%}<|channel>thought\n<channel|>{%- endif -%}",
            "{%- endif -%}"
        );
        let tmpl = JinjaChatTemplate::new(gemma4_shaped, None, None);

        let off = tmpl
            .render(
                &[ChatMessage::user("Hello")],
                &RenderOptions {
                    add_generation_prompt: true,
                    enable_thinking: false,
                },
            )
            .unwrap();
        assert!(
            !off.contains("<|think|>"),
            "thinking-off must not inject a reasoning block: {off}"
        );
        assert!(off.contains("<|turn>user\nHello<turn|>"));
        // thinking-off pre-fills an empty thought channel after the model turn,
        // steering gemma-4 to answer directly.
        assert!(off.contains("<|turn>model"));
        assert!(off.contains("<|channel>thought") && off.contains("<channel|>"));

        // Contrast: thinking-on injects the reasoning block and does NOT
        // pre-fill the empty-thought suppressor (proves both gates work).
        let on = tmpl
            .render(
                &[ChatMessage::user("Hello")],
                &RenderOptions {
                    add_generation_prompt: true,
                    enable_thinking: true,
                },
            )
            .unwrap();
        assert!(on.contains("<|think|>"));
        assert!(!on.contains("<|channel>thought"));
    }

    /// A syntactically broken template must surface as `Err`, never panic —
    /// the caller relies on this to surface an error chaining the original
    /// llama rejection and this render failure.
    #[test]
    fn invalid_template_returns_err_not_panic() {
        let tmpl = JinjaChatTemplate::new("{% for x in %}", None, None);
        let result = tmpl.render(&[ChatMessage::user("hi")], &RenderOptions::default());
        assert!(matches!(result, Err(JinjaTemplateError::Render(_))));
    }
}
