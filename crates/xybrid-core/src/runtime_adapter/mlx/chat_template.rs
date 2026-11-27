//! Chat templates for the MLX adapter.
//!
//! MLX-LM bundles ship the Jinja2 template the model was trained with, in
//! `tokenizer_config.json`'s `chat_template` field or in a
//! `chat_template.jinja` beside it. It turns a `[{role, content}]` message
//! list, and any tool definitions, into the raw text the model expects. Qwen 3
//! uses ChatML, Gemma 4 its own `<|turn>` markers, LFM2 ChatML with a tool
//! list. Rather than ship per-model templates, this module loads the bundle's
//! own and renders it.
//!
//! Rendering itself is [`crate::runtime_adapter::jinja_template`], the one
//! renderer both local text backends use: the same HuggingFace context
//! (`messages`, `add_generation_prompt`, `bos_token`, `eos_token`,
//! `enable_thinking`, `tools`), the same plain-JSON `tojson`, the same
//! handling of `{% generation %}` tags. A template therefore renders the same
//! prompt from an MLX bundle as from the GGUF of the same model.
//!
//! [`ChatTemplate`] is stateless and `Send + Sync`: one is loaded per bundle
//! and kept in [`super::model::LoadedState`].

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::runtime_adapter::jinja_template::{self, JinjaChatTemplate};
use crate::runtime_adapter::llm::ChatMessage;

/// Errors from chat-template loading or rendering.
#[derive(Debug, thiserror::Error)]
pub enum ChatTemplateError {
    /// `tokenizer_config.json` couldn't be read.
    #[error("failed to read tokenizer_config.json: {0}")]
    Io(#[from] std::io::Error),

    /// `tokenizer_config.json` failed to parse as JSON.
    #[error("failed to parse tokenizer_config.json: {0}")]
    Parse(#[from] serde_json::Error),

    /// `tokenizer_config.json` has no `chat_template` field. Required — we
    /// don't try to infer ChatML/etc. from heuristics because the MLX-LM
    /// upstream canonicalises on having the template in the bundle.
    #[error("tokenizer_config.json has no `chat_template` field; bundle is not chat-capable")]
    MissingTemplate,

    /// Template failed to parse (Jinja2 syntax error) or render (undefined
    /// variable, type mismatch, etc.). Wraps the minijinja error.
    #[error("chat template render failed: {0}")]
    Render(String),
}

pub type ChatTemplateResult<T> = Result<T, ChatTemplateError>;

/// `tokenizer_config.json` subset we care about for chat rendering.
///
/// HuggingFace stores either a single template string or a list of
/// `{ name, template }` objects (for multi-mode tokenizers — Mistral's
/// "default" vs "tool_use"). We pick the `default` entry or the single
/// string; anything else (including `tool_use`) is ignored until a story
/// needs it.
#[derive(Debug, Clone, Deserialize)]
struct TokenizerConfigRaw {
    #[serde(default)]
    chat_template: Option<ChatTemplateField>,

    /// BOS / EOS tokens used by some templates — we pass them through to the
    /// Jinja context. Missing fields deserialise to `None`.
    #[serde(default)]
    bos_token: Option<TokenField>,
    #[serde(default)]
    eos_token: Option<TokenField>,
}

/// The `chat_template` field is either a raw Jinja string OR a list of
/// `{ name, template }` entries. We collapse to the string form by picking
/// the `default`-named entry or the first entry.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ChatTemplateField {
    Single(String),
    Multi(Vec<ChatTemplateEntry>),
}

#[derive(Debug, Clone, Deserialize)]
struct ChatTemplateEntry {
    name: String,
    template: String,
}

/// The BOS / EOS fields are sometimes a plain string, sometimes a
/// `{ content, ... }` map (Mistral's `AddedToken` shape). Extract the
/// content in either case.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TokenField {
    Plain(String),
    Rich {
        #[serde(default)]
        content: String,
    },
}

impl TokenField {
    fn into_string(self) -> String {
        match self {
            TokenField::Plain(s) => s,
            TokenField::Rich { content } => content,
        }
    }
}

impl ChatTemplateField {
    fn into_template_string(self) -> ChatTemplateResult<String> {
        match self {
            ChatTemplateField::Single(s) => Ok(s),
            ChatTemplateField::Multi(v) => {
                // Prefer the "default" entry if present, otherwise pick the
                // first. Matches the HuggingFace transformers tokenizer
                // behaviour.
                v.iter()
                    .find(|e| e.name == "default")
                    .map(|e| e.template.clone())
                    .or_else(|| v.first().map(|e| e.template.clone()))
                    .ok_or(ChatTemplateError::MissingTemplate)
            }
        }
    }
}

/// Options the caller passes at render time. Kept tiny on purpose —
/// expanding it is cheaper than inventing a second rendering path.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Whether to append the assistant-turn scaffolding (e.g. Qwen 3's
    /// `<|im_start|>assistant\n`) so the LLM continues mid-turn. Almost
    /// always `true` when the caller plans to call `generate` on the
    /// rendered prompt; set `false` for analysis or dry-run paths.
    pub add_generation_prompt: bool,
    /// Whether reasoning-model templates should enter their thinking mode.
    /// Qwen 3 and Gemma 4 instruction templates both honor this switch to
    /// skip reasoning-channel scaffolding for direct user-facing answers.
    pub enable_thinking: bool,
    /// Tool definitions for the template's `tools` variable, as a JSON array
    /// already in the shape the template expects (see
    /// `ToolCallProtocol::template_tool_entries`). `None` leaves `tools`
    /// undefined, as HuggingFace's `apply_chat_template` does for a request
    /// without tools.
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

/// Loaded chat template, ready to render message lists into raw prompt text.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// The raw Jinja template string. Kept around for debugging and for the
    /// per-call environment we build in [`Self::render`].
    template: String,
    /// Optional BOS token (e.g. Llama's `<s>`) — passed into the Jinja
    /// context as `bos_token` for templates that interpolate it.
    bos_token: Option<String>,
    /// Optional EOS token (e.g. Qwen's `<|im_end|>`) — same passthrough.
    eos_token: Option<String>,
}

impl ChatTemplate {
    /// Construct a template from a raw Jinja string. Used by both the
    /// file-loader and by tests that want to exercise the renderer without
    /// writing a fixture to disk.
    pub fn from_str(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            bos_token: None,
            eos_token: None,
        }
    }

    /// Built-in Gemma 4 chat fallback used for MLX/VLM bundles that expose
    /// Gemma 4 turn tokens but do not ship a HuggingFace `chat_template`.
    pub fn gemma4_fallback() -> Self {
        Self::from_str(concat!(
            "{{- '<bos>' -}}",
            "{%- for message in messages -%}",
            "{%- set role = 'model' if message.role == 'assistant' else message.role -%}",
            "{{- '<|turn>' + role + '\\n' + (message.content | trim) + '<turn|>\\n' -}}",
            "{%- endfor -%}",
            "{%- if add_generation_prompt -%}",
            "{%- if enable_thinking is defined and enable_thinking -%}",
            "{{- '<|turn>model\\n<|channel>thought\\n<channel|>' -}}",
            "{%- else -%}",
            "{{- '<|turn>model\\n' -}}",
            "{%- endif -%}",
            "{%- endif -%}"
        ))
    }

    /// Builder: set the BOS token.
    pub fn with_bos(mut self, bos: impl Into<String>) -> Self {
        self.bos_token = Some(bos.into());
        self
    }

    /// Builder: set the EOS token.
    pub fn with_eos(mut self, eos: impl Into<String>) -> Self {
        self.eos_token = Some(eos.into());
        self
    }

    /// Load from a `tokenizer_config.json` in the given directory. Returns
    /// [`ChatTemplateError::MissingTemplate`] if the file exists but has no
    /// `chat_template` field (which is the only "expected non-fatal" miss —
    /// base models without chat templates are legitimately not chat-capable).
    pub fn load_from_dir(model_dir: &Path) -> ChatTemplateResult<Self> {
        let path = model_dir.join("tokenizer_config.json");
        let file_template_path = model_dir.join("chat_template.jinja");
        let cfg = match fs::read_to_string(&path) {
            Ok(raw) => Some(serde_json::from_str::<TokenizerConfigRaw>(&raw)?),
            Err(err)
                if err.kind() == std::io::ErrorKind::NotFound && file_template_path.is_file() =>
            {
                None
            }
            Err(err) => return Err(ChatTemplateError::Io(err)),
        };

        let template_str = match cfg.as_ref().and_then(|cfg| cfg.chat_template.clone()) {
            Some(template) => template.into_template_string()?,
            None if file_template_path.is_file() => fs::read_to_string(&file_template_path)?,
            None => return Err(ChatTemplateError::MissingTemplate),
        };

        let bos_token = cfg
            .as_ref()
            .and_then(|cfg| cfg.bos_token.clone())
            .map(TokenField::into_string);
        let eos_token = cfg
            .as_ref()
            .and_then(|cfg| cfg.eos_token.clone())
            .map(TokenField::into_string);

        Ok(Self {
            template: template_str,
            bos_token,
            eos_token,
        })
    }

    /// The template's Jinja source, as the bundle ships it. The tool-call
    /// protocol a model speaks is read off its markers.
    pub fn source(&self) -> &str {
        &self.template
    }

    /// Render a message list to a prompt string, with the context
    /// HuggingFace's `apply_chat_template` builds, so a template copied
    /// verbatim from any MLX-LM bundle renders without modification.
    ///
    /// `enable_thinking` is always passed, never left undefined: templates
    /// such as Qwen 3's close the think block only for an explicit `false`
    /// and treat a missing flag as thinking on.
    ///
    /// # Errors
    ///
    /// [`ChatTemplateError::Render`] when the template does not compile or
    /// fails to render.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        options: &RenderOptions,
    ) -> ChatTemplateResult<String> {
        let template = JinjaChatTemplate::new(
            self.template.as_str(),
            self.bos_token.clone(),
            self.eos_token.clone(),
        );
        let options = jinja_template::RenderOptions {
            add_generation_prompt: options.add_generation_prompt,
            enable_thinking: Some(options.enable_thinking),
            tools: options.tools.clone(),
        };
        template
            .render(messages, &options)
            .map_err(|err| match err {
                jinja_template::JinjaTemplateError::Render(message) => {
                    ChatTemplateError::Render(message)
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Canonical Qwen 3 / ChatML template (simplified — drops the
    /// tool-calling branches that Qwen 3 also ships, since they're not
    /// exercised by our integration path). Pulled from
    /// `mlx-community/Qwen3.5-0.5B-Instruct-4bit/tokenizer_config.json`
    /// with the `{% if tools %}` branch elided.
    const QWEN_TEMPLATE: &str = concat!(
        "{%- for message in messages %}",
        "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}",
        "{%- endfor %}",
        "{%- if add_generation_prompt %}",
        "{{- '<|im_start|>assistant\n' }}",
        "{%- if enable_thinking is defined and enable_thinking is false %}",
        "{{- '<think>\n\n</think>\n\n' }}",
        "{%- endif %}",
        "{%- endif %}"
    );

    #[test]
    fn from_str_then_render_single_turn() {
        let tmpl = ChatTemplate::from_str(QWEN_TEMPLATE);
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
        ];
        let out = tmpl.render(&msgs, &RenderOptions::default()).unwrap();
        // Message boundaries + trailing assistant-turn scaffold.
        assert!(out.starts_with("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        assert!(out.contains("<|im_start|>user\nHello!<|im_end|>\n"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn render_four_turn_chat() {
        // 4-turn chat shape: system + (user/assistant)*2 + final user -> assistant.
        let tmpl = ChatTemplate::from_str(QWEN_TEMPLATE);
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("u1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("u2"),
            ChatMessage::assistant("a2"),
            ChatMessage::user("u3"),
        ];
        let out = tmpl.render(&msgs, &RenderOptions::default()).unwrap();

        // No template leakage: every message appears inside its own
        // <|im_start|>...<|im_end|> bracket.
        for (role, body) in [
            ("system", "sys"),
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
            ("user", "u3"),
        ] {
            let token = format!("<|im_start|>{role}\n{body}<|im_end|>\n");
            assert!(out.contains(&token), "missing message token: {token}");
        }

        // The scaffolded trailing assistant turn must be present so the
        // generate loop continues mid-turn (part of the PRD's "no template
        // leakage" criterion — we neither drop nor duplicate it).
        assert!(out.trim_end().ends_with("<|im_start|>assistant"));
        // And the final user turn comes BEFORE the scaffold.
        let scaffold_pos = out.rfind("<|im_start|>assistant").unwrap();
        let u3_pos = out.rfind("u3").unwrap();
        assert!(u3_pos < scaffold_pos, "user u3 must precede the scaffold");
    }

    #[test]
    fn add_generation_prompt_false_skips_scaffold() {
        let tmpl = ChatTemplate::from_str(QWEN_TEMPLATE);
        let msgs = vec![ChatMessage::user("hi")];
        let opts = RenderOptions {
            add_generation_prompt: false,
            ..RenderOptions::default()
        };
        let out = tmpl.render(&msgs, &opts).unwrap();
        assert!(!out.contains("<|im_start|>assistant"));
    }

    #[test]
    fn render_can_disable_qwen_thinking() {
        let tmpl = ChatTemplate::from_str(QWEN_TEMPLATE);
        let opts = RenderOptions {
            enable_thinking: false,
            ..RenderOptions::default()
        };
        let out = tmpl.render(&[ChatMessage::user("hi")], &opts).unwrap();
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn render_supports_macro_templates() {
        let tmpl = ChatTemplate::from_str(concat!(
            "{%- macro turn(message) -%}",
            "<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n",
            "{%- endmacro -%}",
            "{%- for message in messages -%}",
            "{{ turn(message) }}",
            "{%- endfor -%}",
            "{%- if add_generation_prompt -%}<start_of_turn>assistant\n{%- endif -%}"
        ));
        let out = tmpl
            .render(&[ChatMessage::user("hello")], &RenderOptions::default())
            .unwrap();
        assert!(out.contains("<start_of_turn>user\nhello<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>assistant"));
    }

    #[test]
    fn gemma4_fallback_uses_turn_tokens() {
        let tmpl = ChatTemplate::gemma4_fallback();
        let out = tmpl
            .render(&[ChatMessage::user(" hello ")], &RenderOptions::default())
            .unwrap();

        assert_eq!(
            out,
            "<bos><|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[test]
    fn gemma4_fallback_can_disable_thinking() {
        let tmpl = ChatTemplate::gemma4_fallback();
        let opts = RenderOptions {
            enable_thinking: false,
            ..RenderOptions::default()
        };
        let out = tmpl.render(&[ChatMessage::user("hello")], &opts).unwrap();

        assert_eq!(out, "<bos><|turn>user\nhello<turn|>\n<|turn>model\n");
    }

    #[test]
    fn bos_eos_flow_through_to_context() {
        // Template that references bos_token/eos_token so we can observe
        // the flow-through.
        let tmpl =
            ChatTemplate::from_str("{{ bos_token }}[{{ messages[0].content }}]{{ eos_token }}")
                .with_bos("<s>")
                .with_eos("</s>");
        let out = tmpl
            .render(&[ChatMessage::user("x")], &RenderOptions::default())
            .unwrap();
        assert_eq!(out, "<s>[x]</s>");
    }

    #[test]
    fn load_from_dir_reads_single_string_template() {
        let tmp = TempDir::new().unwrap();
        let cfg_json = serde_json::json!({
            "chat_template": QWEN_TEMPLATE,
            "bos_token": "<|endoftext|>",
            "eos_token": { "content": "<|im_end|>" }
        });
        let mut f = fs::File::create(tmp.path().join("tokenizer_config.json")).unwrap();
        f.write_all(cfg_json.to_string().as_bytes()).unwrap();

        let tmpl = ChatTemplate::load_from_dir(tmp.path()).expect("load");
        assert_eq!(tmpl.bos_token.as_deref(), Some("<|endoftext|>"));
        assert_eq!(tmpl.eos_token.as_deref(), Some("<|im_end|>"));

        let out = tmpl
            .render(&[ChatMessage::user("hello")], &RenderOptions::default())
            .unwrap();
        assert!(out.contains("<|im_start|>user\nhello<|im_end|>"));
    }

    #[test]
    fn load_from_dir_reads_multi_template_picks_default() {
        let tmp = TempDir::new().unwrap();
        let cfg_json = serde_json::json!({
            "chat_template": [
                { "name": "default", "template": "DEFAULT:{{ messages[0].content }}" },
                { "name": "tool_use", "template": "TOOL:{{ messages[0].content }}" }
            ]
        });
        let mut f = fs::File::create(tmp.path().join("tokenizer_config.json")).unwrap();
        f.write_all(cfg_json.to_string().as_bytes()).unwrap();

        let tmpl = ChatTemplate::load_from_dir(tmp.path()).expect("load");
        let out = tmpl
            .render(&[ChatMessage::user("x")], &RenderOptions::default())
            .unwrap();
        assert_eq!(out, "DEFAULT:x");
    }

    #[test]
    fn load_from_dir_errors_without_template_field() {
        let tmp = TempDir::new().unwrap();
        let cfg_json = serde_json::json!({ "bos_token": "<s>" });
        let mut f = fs::File::create(tmp.path().join("tokenizer_config.json")).unwrap();
        f.write_all(cfg_json.to_string().as_bytes()).unwrap();

        let err = ChatTemplate::load_from_dir(tmp.path()).unwrap_err();
        assert!(matches!(err, ChatTemplateError::MissingTemplate));
    }

    #[test]
    fn load_from_dir_uses_chat_template_file_fallback() {
        let tmp = TempDir::new().unwrap();
        let cfg_json = serde_json::json!({
            "bos_token": "<|startoftext|>",
            "eos_token": "<|im_end|>"
        });
        let mut cfg = fs::File::create(tmp.path().join("tokenizer_config.json")).unwrap();
        cfg.write_all(cfg_json.to_string().as_bytes()).unwrap();
        let mut template = fs::File::create(tmp.path().join("chat_template.jinja")).unwrap();
        template
            .write_all(b"{{ bos_token }}{{ messages[0].content }}{{ eos_token }}")
            .unwrap();

        let tmpl = ChatTemplate::load_from_dir(tmp.path()).expect("load");
        let out = tmpl
            .render(&[ChatMessage::user("hello")], &RenderOptions::default())
            .unwrap();
        assert_eq!(out, "<|startoftext|>hello<|im_end|>");
    }

    #[test]
    fn load_from_dir_uses_chat_template_file_without_tokenizer_config() {
        let tmp = TempDir::new().unwrap();
        let mut template = fs::File::create(tmp.path().join("chat_template.jinja")).unwrap();
        template.write_all(b"{{ messages[0].content }}").unwrap();

        let tmpl = ChatTemplate::load_from_dir(tmp.path()).expect("load");
        let out = tmpl
            .render(&[ChatMessage::user("hello")], &RenderOptions::default())
            .unwrap();
        assert_eq!(out, "hello");
    }

    #[test]
    fn load_from_dir_errors_on_missing_file() {
        let tmp = TempDir::new().unwrap();
        let err = ChatTemplate::load_from_dir(tmp.path()).unwrap_err();
        assert!(matches!(err, ChatTemplateError::Io(_)));
    }

    /// `ChatTemplate` should be Send + Sync so it can live inside
    /// `MlxLlmAdapter` (which the runtime selector boxes as
    /// `Box<dyn LlmBackend + Send + Sync>`).
    /// The tool-listing half of LFM2.5's shipped template.
    const LFM2_TOOLS_TEMPLATE: &str = concat!(
        "{{- bos_token -}}",
        "{%- if tools -%}",
        "{{- '<|im_start|>system\nList of tools: [' -}}",
        "{%- for tool in tools -%}",
        "{%- if tool is not string -%}{%- set tool = tool | tojson -%}{%- endif -%}",
        "{{- tool -}}{%- if not loop.last -%}{{- ', ' -}}{%- endif -%}",
        "{%- endfor -%}",
        "{{- ']<|im_end|>\n' -}}",
        "{%- endif -%}",
        "{%- for message in messages -%}",
        "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' -}}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
    );

    #[test]
    fn tools_reach_the_template_as_plain_json() {
        let tmpl = ChatTemplate::from_str(LFM2_TOOLS_TEMPLATE).with_bos("<|startoftext|>");
        let tools = serde_json::json!([{
            "name": "compare",
            "description": "true when a < b",
            "parameters": { "type": "object", "properties": {} }
        }]);
        let out = tmpl
            .render(
                &[ChatMessage::user("Hello")],
                &RenderOptions {
                    tools: Some(tools),
                    ..RenderOptions::default()
                },
            )
            .unwrap();

        assert!(
            out.starts_with("<|startoftext|><|im_start|>system\nList of tools: [{"),
            "{out}"
        );
        assert!(out.contains(r#""name":"compare""#), "{out}");
        // Plain JSON, as `apply_chat_template` renders it: `<` is not
        // HTML-escaped to `\u003c`.
        assert!(out.contains("true when a < b"), "{out}");
        assert!(out.ends_with("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"));
    }

    #[test]
    fn a_request_without_tools_leaves_the_tool_list_out() {
        let tmpl = ChatTemplate::from_str(LFM2_TOOLS_TEMPLATE);
        let out = tmpl
            .render(&[ChatMessage::user("Hello")], &RenderOptions::default())
            .unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// LFM2.5-230M's template marks assistant spans with HuggingFace's
    /// `{% generation %}` tags. They are a rendering no-op, and minijinja
    /// rejects them as unknown statements unless they are neutralized.
    #[test]
    fn generation_tags_render_as_nothing() {
        let tmpl = ChatTemplate::from_str(concat!(
            "{%- for message in messages -%}",
            "{{- '<|im_start|>' + message.role + '\n' -}}",
            "{%- generation -%}{{- message.content -}}{%- endgeneration -%}",
            "{{- '<|im_end|>\n' -}}",
            "{%- endfor -%}"
        ));
        let out = tmpl
            .render(&[ChatMessage::user("Hello")], &RenderOptions::default())
            .unwrap();
        assert_eq!(out, "<|im_start|>user\nHello<|im_end|>\n");
    }

    #[test]
    fn template_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ChatTemplate>();
    }
}
