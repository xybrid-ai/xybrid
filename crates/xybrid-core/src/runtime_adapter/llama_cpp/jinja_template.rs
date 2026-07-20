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
//! adapter uses for HuggingFace templates. [`super::chat::format_chat_prompt`]
//! reaches for it in two cases: as a fallback when native formatting reports
//! a template is present but unrenderable, and as the *primary* path for
//! tool-bearing prompts — llama.cpp's `llama_chat_apply_template` takes only
//! role/content pairs, so tool definitions can only reach the model through a
//! real Jinja render. If minijinja also fails, the caller surfaces an error
//! rather than substituting a different prompt family.
//!
//! There is no file I/O here: the template string and BOS/EOS tokens are
//! supplied by the caller, which already holds the loaded [`LlamaModel`].

use minijinja::value::Kwargs;
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
    /// Tool definitions to expose in the template context, pre-serialized as
    /// a JSON array (bare function objects for LFM2-family templates). When
    /// `None` the `tools` key is omitted from the context entirely so
    /// `{%- if tools -%}` template guards see an undefined (falsy) value —
    /// an empty array would also be falsy, but omission matches the
    /// HuggingFace `apply_chat_template` contract.
    pub tools: Option<serde_json::Value>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            tools: None,
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
    ///
    /// HuggingFace `{% generation %}` / `{% endgeneration %}` statement tags
    /// are rewritten to comments up front — see
    /// [`neutralize_generation_tags`].
    pub(super) fn new(
        template: impl Into<String>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        Self {
            template: neutralize_generation_tags(&template.into()),
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
        // `tojson` is what chat templates use to inline tool definitions
        // (LFM2: `{{ tool | tojson }}`; Llama-3.1: `tojson(indent=4)`), but
        // minijinja gates its builtin behind the `json` cargo feature — off
        // in this crate — and that builtin HTML-escapes `<`, `>`, `&`, `'`
        // to `\u003c`-style sequences, which is wrong for LLM prompts.
        // HuggingFace's `apply_chat_template` overrides Jinja2's builtin
        // with plain `json.dumps` for the same reason; we register the
        // plain-JSON equivalent so prompts match what models trained on.
        env.add_filter("tojson", tojson_plain);
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

        let base = context! {
            messages => messages_value,
            add_generation_prompt => options.add_generation_prompt,
            enable_thinking => options.enable_thinking,
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone().unwrap_or_default(),
        };
        // The `tools` key is merged in only when the caller supplied
        // definitions: HuggingFace's `apply_chat_template` omits the key
        // entirely for tool-less calls, so `{%- if tools -%}` guards see an
        // undefined (falsy) value and `tools is defined` checks stay
        // accurate. Passing null or `[]` instead would flip the latter.
        let ctx = match &options.tools {
            Some(tools) => context! { tools => Value::from_serialize(tools), ..base },
            None => base,
        };

        let tmpl = env
            .get_template("chat")
            .map_err(|e| JinjaTemplateError::Render(format!("lookup: {e}")))?;
        tmpl.render(ctx)
            .map_err(|e| JinjaTemplateError::Render(format!("render: {e}")))
    }
}

/// Rewrites HuggingFace `{% generation %}` / `{% endgeneration %}` statement
/// tags into Jinja comments, preserving their whitespace-control dashes.
///
/// These tags mark assistant-token spans for training-time loss masks and are
/// rendering no-ops, but minijinja — unlike HF's Jinja2 extension or
/// llama.cpp's minja — rejects them as unknown statements. LFM2.5-230M's
/// embedded template uses them (its 350M sibling does not). Rewriting
/// `{%- generation -%}` to `{#- generation -#}` keeps the surrounding
/// whitespace-trim semantics byte-identical, which plain removal would not.
///
/// Only the bare tags are rewritten; a statement with extra tokens (e.g.
/// `{% generation foo %}`) has unknown semantics and is left to fail
/// compilation loudly.
fn neutralize_generation_tags(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;

    while let Some(start) = rest.find("{%") {
        let after_open = &rest[start + 2..];
        let (leading_dash, body) = match after_open.strip_prefix('-') {
            Some(b) => (true, b),
            None => (false, after_open),
        };
        let body = body.trim_start();
        let keyword = ["endgeneration", "generation"]
            .iter()
            .find(|kw| body.starts_with(**kw));

        let rewritten = keyword.and_then(|kw| {
            let after_kw = body[kw.len()..].trim_start();
            let (trailing_dash, after_kw) = match after_kw.strip_prefix('-') {
                Some(b) => (true, b),
                None => (false, after_kw),
            };
            after_kw
                .strip_prefix("%}")
                .map(|tail| (kw, trailing_dash, tail))
        });

        match rewritten {
            Some((kw, trailing_dash, tail)) => {
                out.push_str(&rest[..start]);
                out.push_str("{#");
                if leading_dash {
                    out.push('-');
                }
                out.push(' ');
                out.push_str(kw);
                out.push(' ');
                if trailing_dash {
                    out.push('-');
                }
                out.push_str("#}");
                rest = tail;
            }
            None => {
                out.push_str(&rest[..start + 2]);
                rest = after_open;
            }
        }
    }

    out.push_str(rest);
    out
}

/// Plain-JSON `tojson` filter, registered in place of minijinja's builtin.
///
/// The builtin (cargo feature `json`, not enabled here) escapes `<`, `>`,
/// `&`, and `'` to unicode sequences for HTML safety. Chat prompts are not
/// HTML: a tool description like "true when a < b" must reach the model as
/// the raw text it was trained on, which is why HuggingFace transformers
/// replaces Jinja2's builtin with plain `json.dumps` in
/// `apply_chat_template` (llama.cpp's own minja engine does the same).
/// Argument handling mirrors the builtin — `indent` positional or keyword,
/// integer width or `true` for 2 spaces — so templates written for either
/// engine (LFM2's bare `tojson`, Llama-3.1's `tojson(indent=4)`) render.
fn tojson_plain(
    value: Value,
    indent: Option<Value>,
    kwargs: Kwargs,
) -> Result<String, minijinja::Error> {
    fn invalid_op(err: impl std::error::Error + Send + Sync + 'static) -> minijinja::Error {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "could not serialize to json",
        )
        .with_source(err)
    }

    let indent = match indent {
        Some(indent) => Some(indent),
        None => kwargs.get("indent")?,
    };
    kwargs.assert_all_used()?;
    let indent_width = match indent {
        None => None,
        Some(ref val) => match bool::try_from(val.clone()).ok() {
            Some(true) => Some(2),
            Some(false) => None,
            None => Some(usize::try_from(val.clone())?),
        },
    };

    match indent_width {
        Some(width) => {
            let indentation = " ".repeat(width);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(indentation.as_bytes());
            let mut out = Vec::new();
            let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
            serde::Serialize::serialize(&value, &mut ser).map_err(invalid_op)?;
            String::from_utf8(out).map_err(invalid_op)
        }
        None => serde_json::to_string(&value).map_err(invalid_op),
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
            tools: None,
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
                    tools: None,
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
                    tools: None,
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

    /// The tool-list section of the LFM2 GGUF's embedded template, reduced
    /// to its load-bearing shape: an `{%- if tools -%}` guard around a loop
    /// that `tojson`s each entry.
    const LFM2_TOOLS_FRAGMENT: &str = concat!(
        "{%- if tools -%}",
        "List of tools: [",
        "{%- for tool in tools -%}",
        "{{ tool | tojson }}",
        "{%- if not loop.last -%}, {%- endif -%}",
        "{%- endfor -%}",
        "]",
        "{%- endif -%}"
    );

    /// LFM2-shaped gating: `Some(tools)` renders the tool list with each
    /// bare function object inlined as JSON; `None` renders the section as
    /// empty because the guard sees an undefined `tools`.
    #[test]
    fn lfm2_tools_section_gated_on_tools_presence() {
        let tmpl = JinjaChatTemplate::new(LFM2_TOOLS_FRAGMENT, None, None);

        let with_tools = RenderOptions {
            tools: Some(serde_json::json!([{
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }])),
            ..RenderOptions::default()
        };
        let out = tmpl
            .render(&[ChatMessage::user("hi")], &with_tools)
            .unwrap();
        assert!(
            out.contains("List of tools: ["),
            "tools present must render the list header: {out}"
        );
        assert!(
            out.contains("get_weather"),
            "function name must be inlined via tojson: {out}"
        );

        let without_tools = tmpl
            .render(&[ChatMessage::user("hi")], &RenderOptions::default())
            .unwrap();
        assert!(
            !without_tools.contains("List of tools"),
            "tools: None must skip the section entirely: {without_tools}"
        );
    }

    /// `tools: None` must OMIT the context key, not pass it as null: a
    /// null/none value satisfies `is defined`, so templates in the wild
    /// that check `tools is defined` would misfire on a present-as-null
    /// key. This pins the HuggingFace `apply_chat_template` omission
    /// contract.
    #[test]
    fn tools_none_leaves_key_undefined_not_null() {
        let tmpl = JinjaChatTemplate::new(
            "{% if tools is defined %}defined{% else %}undefined{% endif %}",
            None,
            None,
        );

        let none = tmpl
            .render(&[ChatMessage::user("x")], &RenderOptions::default())
            .unwrap();
        assert_eq!(none, "undefined");

        // Contrast: even an EMPTY array flips `is defined` — proving the
        // None case above exercised true key absence, not falsiness.
        let empty = tmpl
            .render(
                &[ChatMessage::user("x")],
                &RenderOptions {
                    tools: Some(serde_json::json!([])),
                    ..RenderOptions::default()
                },
            )
            .unwrap();
        assert_eq!(empty, "defined");
    }

    /// The registered `tojson` must match HF's plain-`json.dumps` contract:
    /// no HTML escaping (comparison operators reach the model verbatim) and
    /// keyword `indent` support for Llama-3.1-style `tojson(indent=N)`.
    #[test]
    fn tojson_matches_hf_plain_json_contract() {
        let plain = JinjaChatTemplate::new("{{ tools[0] | tojson }}", None, None);
        let opts = RenderOptions {
            tools: Some(serde_json::json!([
                {"name": "cmp", "description": "true when a < b & b > c"}
            ])),
            ..RenderOptions::default()
        };
        let out = plain.render(&[], &opts).unwrap();
        assert!(
            out.contains("a < b & b > c"),
            "tojson must not HTML-escape prompt text: {out}"
        );

        let indented = JinjaChatTemplate::new("{{ tools[0] | tojson(indent=2) }}", None, None);
        let out = indented.render(&[], &opts).unwrap();
        assert!(
            out.contains("\n  \"name\""),
            "indent kwarg must pretty-print: {out}"
        );
    }

    /// LFM2.5-230M's template wraps assistant spans in HF
    /// `{%- generation -%}` / `{%- endgeneration -%}` tags. They must be
    /// neutralized so the template compiles, with whitespace-control
    /// preserved (the span content renders, the tags vanish).
    #[test]
    fn generation_tags_are_neutralized_and_preserve_trim() {
        let tmpl = JinjaChatTemplate::new(
            "a  {%- generation -%}  {{ messages[0].content }}  {%- endgeneration -%}  b",
            None,
            None,
        );
        let out = tmpl
            .render(&[ChatMessage::user("X")], &RenderOptions::default())
            .unwrap();
        // Both dashed sides trim: every whitespace run touching a dashed tag
        // edge vanishes, exactly as under HF's Jinja2 generation extension.
        assert_eq!(out, "aXb");
    }

    /// Undashed variant keeps surrounding whitespace, mirroring Jinja
    /// semantics for a no-op statement without trim markers.
    #[test]
    fn generation_tags_without_dashes_keep_whitespace() {
        let tmpl = JinjaChatTemplate::new("a {% generation %}x{% endgeneration %} b", None, None);
        let out = tmpl
            .render(&[ChatMessage::user("unused")], &RenderOptions::default())
            .unwrap();
        assert_eq!(out, "a x b");
    }

    /// A statement that merely STARTS with the keyword has unknown semantics
    /// and must still fail compilation loudly rather than being rewritten.
    #[test]
    fn generation_like_statements_with_extra_tokens_still_error() {
        let tmpl = JinjaChatTemplate::new("{% generation foo %}", None, None);
        let result = tmpl.render(&[ChatMessage::user("hi")], &RenderOptions::default());
        assert!(matches!(result, Err(JinjaTemplateError::Render(_))));
    }
}
