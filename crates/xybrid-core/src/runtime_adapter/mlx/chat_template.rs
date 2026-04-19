//! HuggingFace chat-template renderer for the MLX adapter (US-014).
//!
//! MLX-LM bundles ship a `tokenizer_config.json` containing a `chat_template`
//! field — a Jinja2 template that turns a `[{role, content}]` message list
//! into the raw text the model was trained on. Qwen 3 uses ChatML, Gemma 4
//! uses its own `<start_of_turn>` markers, and LFM uses a Llama-style
//! `[INST] ... [/INST]` variant. Rather than ship per-model templates, we
//! read the bundle's template and render it via [`minijinja`].
//!
//! Design notes:
//!
//! - The renderer is stateless: construct a fresh [`ChatTemplate`] per bundle
//!   and keep it in [`super::model::LoadedState`]. `Send + Sync` so the
//!   adapter stays thread-shareable.
//! - We pass the standard HuggingFace variables: `messages` (list of
//!   `{role, content}` maps), `add_generation_prompt` (bool), and the
//!   tokenizer's `bos_token` / `eos_token` when present. That covers every
//!   template in the MLX-LM zoo today.
//! - We do NOT support `tools` / function-calling templates yet — MLX-LM
//!   models that use them (e.g. Hermes) are not on the US-018 priority list.
//!   Adding them later is a single new variable in [`RenderOptions`].

use std::fs;
use std::path::Path;

use minijinja::{context, Environment, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::Deserialize;

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
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
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
        let raw = fs::read_to_string(&path)?;
        let cfg: TokenizerConfigRaw = serde_json::from_str(&raw)?;

        let template_str = cfg
            .chat_template
            .ok_or(ChatTemplateError::MissingTemplate)?
            .into_template_string()?;

        let bos_token = cfg.bos_token.map(TokenField::into_string);
        let eos_token = cfg.eos_token.map(TokenField::into_string);

        Ok(Self {
            template: template_str,
            bos_token,
            eos_token,
        })
    }

    /// Render a message list to a prompt string. The Jinja context mirrors
    /// the HuggingFace transformers tokenizer's `apply_chat_template` —
    /// templates copied verbatim from any MLX-LM bundle render without
    /// modification.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        options: &RenderOptions,
    ) -> ChatTemplateResult<String> {
        // Build an Environment per-call. minijinja's envs are cheap to
        // construct (~1us) and the alternative — caching a compiled template
        // on the struct — would require a self-referential lifetime or
        // `Arc<Environment<'static>>` with a leaked template string. The
        // per-call cost is invisible relative to the forward pass.
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_template("chat", &self.template)
            .map_err(|e| ChatTemplateError::Render(format!("compile: {e}")))?;

        // Messages flow into the template as a list of maps. Using
        // `Value::from_serialize` preserves the `{role, content}` shape
        // every HF template expects.
        let messages_value: Vec<Value> = messages
            .iter()
            .map(|m| {
                Value::from_serialize(serde_json::json!({
                    "role": m.role.as_str(),
                    "content": &m.content,
                }))
            })
            .collect();

        let ctx = context! {
            messages => messages_value,
            add_generation_prompt => options.add_generation_prompt,
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone().unwrap_or_default(),
        };

        let tmpl = env
            .get_template("chat")
            .map_err(|e| ChatTemplateError::Render(format!("lookup: {e}")))?;
        tmpl.render(ctx)
            .map_err(|e| ChatTemplateError::Render(format!("render: {e}")))
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
        };
        let out = tmpl.render(&msgs, &opts).unwrap();
        assert!(!out.contains("<|im_start|>assistant"));
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
    fn load_from_dir_errors_on_missing_file() {
        let tmp = TempDir::new().unwrap();
        let err = ChatTemplate::load_from_dir(tmp.path()).unwrap_err();
        assert!(matches!(err, ChatTemplateError::Io(_)));
    }

    /// `ChatTemplate` should be Send + Sync so it can live inside
    /// `MlxLlmAdapter` (which the runtime selector boxes as
    /// `Box<dyn LlmBackend + Send + Sync>`).
    #[test]
    fn template_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ChatTemplate>();
    }
}
