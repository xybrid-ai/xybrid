//! Always-available streaming and chat types for LLM inference.
//!
//! This module contains types that are used by FFI/binding code and need to be
//! available regardless of which LLM backend (if any) is enabled.
//!
//! Types here:
//! - `PartialToken` - Token emitted during streaming generation
//! - `StreamingCallback` - Callback type for streaming token generation
//! - `StreamingError` - Error type for streaming callback failures
//! - `ChatMessage` - Chat message for multi-turn conversations
//! - `GenerationConfig` - Generation parameters for LLM inference
//! - `LlmConfig` - Configuration for loading a local LLM

use crate::gateway::{Tool, ToolCall};
use crate::ir::MessageRole;
use crate::{
    conversation::ConversationContext,
    ir::{Envelope, EnvelopeKind, ImageDimensions, ImageFormat, ImageSource},
    runtime_adapter::AdapterError,
};

/// Envelope-metadata key carrying stop sequences.
///
/// The metadata map is `String`-valued, so a list needs an encoding.
/// [`encode_stop_sequences`] writes a JSON array and [`parse_stop_sequences`]
/// reads one; both the cloud adapter and the local LLM strategy go through
/// that pair, so one envelope yields the same stop list either side of the
/// local/cloud seam.
pub const STOP_SEQUENCES_METADATA_KEY: &str = "stop_sequences";

/// Decode the [`STOP_SEQUENCES_METADATA_KEY`] metadata value.
///
/// A JSON array of strings is the canonical form and round-trips any sequence
/// exactly — including the whitespace-only and comma-bearing values a
/// delimiter-based encoding mangles (`"\n\n"` would trim away to nothing).
///
/// Anything else is read as the legacy comma-separated list, trimming each
/// entry and dropping empties, so hand-written metadata like
/// `stop_sequences = "<|im_end|>,<|im_start|>"` keeps working. That legacy form
/// stays lossy by construction; write JSON for anything whitespace- or
/// comma-bearing.
pub fn parse_stop_sequences(value: &str) -> Vec<String> {
    if let Ok(parsed) = serde_json::from_str::<Vec<String>>(value) {
        return parsed;
    }
    value
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Encode stop sequences for [`STOP_SEQUENCES_METADATA_KEY`] as a JSON array.
///
/// Lossless: [`parse_stop_sequences`] returns exactly what went in.
pub fn encode_stop_sequences(stop_sequences: &[String]) -> String {
    serde_json::to_string(stop_sequences).expect("a Vec<String> always serialises to JSON")
}

use serde::{Deserialize, Serialize};

// =============================================================================
// Streaming Types
// =============================================================================

/// Partial token emitted during streaming generation.
///
/// This is passed to the callback function during `generate_streaming()` calls.
#[derive(Debug, Clone)]
pub struct PartialToken {
    /// The decoded token text (may be partial UTF-8 for some tokenizers)
    pub token: String,
    /// Raw token ID if available (backend-specific)
    pub token_id: Option<i64>,
    /// Zero-based index of this token in the generation sequence
    pub index: usize,
    /// Cumulative text generated so far (all tokens concatenated)
    pub cumulative_text: String,
    /// Finish reason if this is the final token, None otherwise.
    /// Values: "stop" (hit stop sequence/EOS), "length" (hit max_tokens),
    /// "tool_calls" (the turn ended on a parseable tool-call block).
    pub finish_reason: Option<String>,
    /// Tool calls parsed from the completed turn. Only ever populated on the
    /// **terminal** token (the one carrying `finish_reason`), and only for a
    /// request that offered tools.
    ///
    /// A streaming caller dispatches on this instead of re-parsing the token
    /// text: the protocol blocks are suppressed from the emitted stream, so
    /// the raw call text never reaches the callback in the first place.
    /// Empty on every mid-stream token and on turns that emitted no call.
    pub tool_calls: Vec<ToolCall>,
    /// The completed turn's raw output text, tool-call block included —
    /// exactly what `Envelope::tool_results` wants as `prior_assistant_text`.
    ///
    /// `Some` only alongside a non-empty [`Self::tool_calls`], which is the
    /// only situation it is useful in. It exists because `cumulative_text`
    /// deliberately reports the *emitted* text (protocol blocks suppressed),
    /// so without this a streaming caller would have no way to compose the
    /// continuation turn.
    pub raw_text: Option<String>,
}

impl PartialToken {
    /// Create a new partial token.
    pub fn new(token: String, index: usize, cumulative_text: String) -> Self {
        Self {
            token,
            token_id: None,
            index,
            cumulative_text,
            finish_reason: None,
            tool_calls: Vec::new(),
            raw_text: None,
        }
    }

    /// Set the token ID.
    pub fn with_token_id(mut self, id: i64) -> Self {
        self.token_id = Some(id);
        self
    }

    /// Mark this as the final token with the given finish reason.
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Attach the tool calls parsed from the completed turn, plus the raw
    /// turn text needed to compose the continuation. Terminal tokens only —
    /// see [`PartialToken::tool_calls`] and [`PartialToken::raw_text`].
    ///
    /// A call-free turn is left untouched: `raw_text` is only meaningful next
    /// to calls, and stamping it unconditionally would put the unsuppressed
    /// protocol text on every terminal token for no consumer.
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>, raw_text: &str) -> Self {
        if calls.is_empty() {
            return self;
        }
        self.tool_calls = calls;
        self.raw_text = Some(raw_text.to_string());
        self
    }

    /// Check if this is the final token.
    pub fn is_final(&self) -> bool {
        self.finish_reason.is_some()
    }
}

/// Error type for streaming callback failures.
pub type StreamingError = Box<dyn std::error::Error + Send + Sync>;

/// Callback type for streaming token generation.
///
/// This is a boxed function that receives partial tokens during streaming generation.
/// Return `Ok(())` to continue generation, or `Err(...)` to stop.
pub type StreamingCallback<'a> =
    Box<dyn FnMut(PartialToken) -> Result<(), StreamingError> + Send + 'a>;

// =============================================================================
// Chat Message
// =============================================================================

/// Chat message for multi-turn conversations.
///
/// This is the unified ChatMessage type used by the LLM runtime adapter.
/// It uses `MessageRole` from `xybrid_core::ir` to ensure type-safe role handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender (system, user, or assistant)
    pub role: MessageRole,
    /// Message content
    pub content: String,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

// =============================================================================
// Multimodal Chat Message
// =============================================================================

/// Image part carried through the backend-neutral multimodal chat contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalImagePart {
    /// Image source. Debug and human-readable serialization redact bytes.
    pub source: ImageSource,
    /// Envelope-local ID for diagnostics and future cache keys.
    pub local_id: String,
}

impl MultimodalImagePart {
    /// Create an image part from an envelope image source.
    pub fn new(source: ImageSource, local_id: impl Into<String>) -> Self {
        Self {
            source,
            local_id: local_id.into(),
        }
    }

    /// Encoded byte length for diagnostics and marker planning.
    pub fn byte_len(&self) -> usize {
        self.source.byte_len()
    }

    /// Encoded image format, when available.
    pub fn format(&self) -> Option<ImageFormat> {
        self.source.encoded_format()
    }

    /// Validated decoded dimensions, when available.
    pub fn dimensions(&self) -> Option<ImageDimensions> {
        self.source.dimensions()
    }
}

/// Ordered part in a backend-neutral multimodal chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MultimodalMessagePart {
    /// Text fragment.
    Text(String),
    /// Image fragment. The source may carry bytes, but diagnostics stay redacted.
    Image(MultimodalImagePart),
}

impl MultimodalMessagePart {
    /// Return the contained text if this is a text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Image(_) => None,
        }
    }

    /// Return true when this part is an image.
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image(_))
    }
}

/// Backend-neutral multimodal message preserving ordered text and image parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalChatMessage {
    /// Role of the message sender.
    pub role: MessageRole,
    /// Ordered message parts.
    pub parts: Vec<MultimodalMessagePart>,
}

impl MultimodalChatMessage {
    /// Build a multimodal message from a single envelope.
    pub fn from_envelope(envelope: &Envelope) -> Result<Self, AdapterError> {
        let role = envelope.role().unwrap_or(MessageRole::User);
        let parts = parts_from_envelope(envelope)?;
        if parts.is_empty() {
            return Err(AdapterError::InvalidInput(
                "multimodal message must contain at least one part".to_string(),
            ));
        }

        Ok(Self { role, parts })
    }

    /// Build multimodal messages from a conversation context in replay order.
    pub fn from_context(context: &ConversationContext) -> Result<Vec<Self>, AdapterError> {
        context
            .context_for_llm()
            .into_iter()
            .map(Self::from_envelope)
            .collect()
    }

    /// Count image parts.
    pub fn image_count(&self) -> usize {
        self.parts.iter().filter(|part| part.is_image()).count()
    }

    /// Convert ordered parts to a marker prompt for backends such as llama.cpp mtmd.
    ///
    /// Text fragments must not already contain the reserved marker because that
    /// would make marker/image-count parity ambiguous at the backend boundary.
    pub fn marker_prompt(&self, marker: &str) -> Result<String, AdapterError> {
        if marker.is_empty() {
            return Err(AdapterError::InvalidInput(
                "media marker must not be empty".to_string(),
            ));
        }

        let mut prompt = String::new();
        for part in &self.parts {
            match part {
                MultimodalMessagePart::Text(text) => {
                    if text.contains(marker) {
                        return Err(AdapterError::InvalidInput(
                            "text part contains reserved media marker; marker/image-count parity would be ambiguous".to_string(),
                        ));
                    }
                    prompt.push_str(text);
                }
                MultimodalMessagePart::Image(_) => prompt.push_str(marker),
            }
        }

        let marker_count = prompt.matches(marker).count();
        let image_count = self.image_count();
        if marker_count != image_count {
            return Err(AdapterError::InvalidInput(format!(
                "media marker count {} does not match image count {}",
                marker_count, image_count
            )));
        }

        Ok(prompt)
    }
}

fn parts_from_envelope(envelope: &Envelope) -> Result<Vec<MultimodalMessagePart>, AdapterError> {
    match &envelope.kind {
        EnvelopeKind::Text(text) => Ok(vec![MultimodalMessagePart::Text(text.clone())]),
        EnvelopeKind::Image { source } => Ok(vec![MultimodalMessagePart::Image(
            MultimodalImagePart::new(source.clone(), envelope.local_id().to_string()),
        )]),
        EnvelopeKind::MultiPart(parts) => parts
            .iter()
            .map(part_from_multipart_fragment)
            .collect::<Result<Vec<_>, _>>(),
        EnvelopeKind::Audio(_) | EnvelopeKind::Embedding(_) => {
            Err(AdapterError::InvalidInput(format!(
                "unsupported multimodal envelope kind {}",
                envelope.kind.as_str()
            )))
        }
    }
}

fn part_from_multipart_fragment(
    envelope: &Envelope,
) -> Result<MultimodalMessagePart, AdapterError> {
    match &envelope.kind {
        EnvelopeKind::Text(text) => Ok(MultimodalMessagePart::Text(text.clone())),
        EnvelopeKind::Image { source } => Ok(MultimodalMessagePart::Image(
            MultimodalImagePart::new(source.clone(), envelope.local_id().to_string()),
        )),
        other => Err(AdapterError::InvalidInput(format!(
            "unsupported multimodal part kind {}",
            other.as_str()
        ))),
    }
}

// =============================================================================
// Generation Configuration
// =============================================================================

/// Generation parameters for LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate. Default: 2048
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for sampling. Default: 0.0 (deterministic / greedy).
    /// Callers who want sampling must set this explicitly — see `greedy()`
    /// and `creative()` constructors.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold. Default: 0.9
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Min-p sampling threshold. Default: 0.05
    ///
    /// Prunes tokens with probability below `min_p * max_probability`.
    /// This is more adaptive than top_p: for confident predictions it
    /// aggressively prunes, for uncertain ones it keeps more candidates.
    /// Set to 0.0 to disable.
    #[serde(default = "default_min_p")]
    pub min_p: f32,

    /// Top-k sampling (0 = disabled). Default: 40
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Repetition penalty (1.0 = disabled). Default: 1.1
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Stop sequences.
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Optional GBNF grammar constraining generation. When set, the local
    /// llama.cpp backend masks any token the grammar would reject, guaranteeing
    /// structured output (e.g. schema-valid JSON for data extraction).
    ///
    /// Set the raw grammar via [`GenerationConfig::with_grammar`], or convert a
    /// JSON Schema with [`GenerationConfig::with_json_schema`]. `None` = free
    /// (unconstrained) generation. Ignored by non-llama backends.
    #[serde(default)]
    pub grammar: Option<String>,

    /// Tools (functions) the model may call, in the OpenAI `Tool` shape.
    ///
    /// When non-empty, the local llama.cpp backend renders the definitions
    /// into the model's embedded chat template and the executor parses any
    /// emitted tool-call blocks (LFM2-family pythonic and gemma-4-family
    /// `call:` notation) into the response envelope's `tool_calls` metadata.
    /// Tool calling is llama.cpp-only today and unsupported paths fail
    /// closed: a model without an embedded chat template, the mistralrs
    /// backend, and the SDK's cloud-fallback leg all reject tool-bearing
    /// requests instead of silently generating without the tools. Empty
    /// means no tool calling (existing behavior, unchanged).
    #[serde(default)]
    pub tools: Vec<Tool>,
}

fn default_max_tokens() -> usize {
    2048
}

fn default_temperature() -> f32 {
    // Deterministic-by-default: 0.0 triggers mistralrs's
    // `set_deterministic_sampler()` path, making local inference
    // reproducible without explicit config. Callers who want
    // sampling must opt in (e.g. `GenerationConfig::creative()`).
    0.0
}

fn default_top_p() -> f32 {
    0.9
}

fn default_min_p() -> f32 {
    0.05
}

fn default_top_k() -> usize {
    40
}

fn default_repetition_penalty() -> f32 {
    1.1
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            min_p: default_min_p(),
            top_k: default_top_k(),
            repetition_penalty: default_repetition_penalty(),
            stop_sequences: Vec::new(),
            grammar: None,
            tools: Vec::new(),
        }
    }
}

impl GenerationConfig {
    /// Create config for greedy decoding (deterministic).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        }
    }

    /// Create config for creative generation.
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Add stop sequence.
    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    /// Constrain generation to a raw [GBNF] grammar (entry rule `root`).
    ///
    /// This is the escape hatch for callers who already have a grammar; most
    /// callers should use [`GenerationConfig::with_json_schema`] instead. Only
    /// the local llama.cpp backend honors this; other backends ignore it.
    ///
    /// [GBNF]: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
    pub fn with_grammar(mut self, grammar: impl Into<String>) -> Self {
        self.grammar = Some(grammar.into());
        self
    }

    /// Constrain generation to a JSON Schema, converting it to a GBNF grammar.
    ///
    /// Guarantees the model emits schema-valid JSON — the basis for reliable
    /// on-device data extraction. Supports a subset of JSON Schema (objects,
    /// arrays, scalars, enums); see [`crate::runtime_adapter::grammar`].
    ///
    /// # Errors
    ///
    /// Returns [`XybridError::Grammar`] if the schema is malformed or uses an
    /// unsupported construct.
    ///
    /// [`XybridError::Grammar`]: crate::error::XybridError::Grammar
    pub fn with_json_schema(
        mut self,
        schema: &serde_json::Value,
    ) -> crate::error::XybridResult<Self> {
        self.grammar = Some(crate::runtime_adapter::grammar::json_schema_to_gbnf(
            schema,
        )?);
        Ok(self)
    }

    /// Offer tools (functions) the model may call. See [`GenerationConfig::tools`].
    ///
    /// Tools are plain data — define your own with [`Tool::function`], run
    /// the request, execute whatever calls come back in the response
    /// envelope's `tool_calls` metadata, and feed the results into the next
    /// turn with `Envelope::tool_results`. The multi-turn loop lives in app
    /// code.
    ///
    /// # Examples
    ///
    /// ```
    /// use xybrid_core::gateway::Tool;
    /// use xybrid_core::runtime_adapter::types::GenerationConfig;
    ///
    /// let config = GenerationConfig::default().with_tools([Tool::function(
    ///     "get_weather",
    ///     "Current weather for a city.",
    ///     serde_json::json!({
    ///         "type": "object",
    ///         "properties": { "city": { "type": "string" } },
    ///         "required": ["city"]
    ///     }),
    /// )]);
    /// assert_eq!(config.tools.len(), 1);
    /// ```
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }
}

// =============================================================================
// LLM Configuration
// =============================================================================

/// Configuration for loading a local LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Path to the model file (GGUF format).
    pub model_path: String,

    /// Path to chat template file (optional).
    pub chat_template: Option<String>,

    /// Path to sibling vision encoder / mmproj artifact (optional).
    ///
    /// Embedding-style backends may load a separate vision encoder from this
    /// path. llama.cpp VLMs use it as the mmproj artifact for their backend-owned
    /// mtmd chunk/eval path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_encoder_path: Option<String>,

    /// Maximum context length (tokens). Default: 4096
    #[serde(default = "default_context_length")]
    pub context_length: usize,

    /// Number of GPU layers to offload. 0 = CPU only, 99 = all layers on GPU (default).
    /// Use 99 as default to enable GPU acceleration when available.
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,

    /// Enable paged attention for memory efficiency. Default: true
    #[serde(default = "default_paged_attention")]
    pub paged_attention: bool,

    /// Enable logging during inference. Default: false
    #[serde(default)]
    pub logging: bool,

    /// Number of threads for inference. 0 = auto-detect (uses all available cores).
    ///
    /// On Android, defaults to 4 (performance cores only on big.LITTLE SoCs).
    /// On other platforms, defaults to 0 (auto-detect).
    #[serde(default = "default_n_threads")]
    pub n_threads: usize,

    /// Batch size for prompt processing. 0 = default (512).
    ///
    /// On Android, defaults to 256 (reduced memory bandwidth pressure).
    /// On other platforms, defaults to 0 (512).
    #[serde(default = "default_n_batch")]
    pub n_batch: usize,

    /// Enable flash attention for faster inference on longer contexts.
    /// Can provide 2-4x speedup. Default: true.
    #[serde(default = "default_flash_attn")]
    pub flash_attn: bool,

    /// Whether this is a reasoning ("thinking") model that emits a
    /// chain-of-thought before its answer. Sourced from the model metadata's
    /// `reasoning` flag. When set, the chat-prompt builder primes the
    /// `<think>` channel so the model produces its reasoning, which the backend
    /// captures into `reasoning_content`. Default: false.
    #[serde(default)]
    pub reasoning: bool,
}

fn default_context_length() -> usize {
    4096
}

fn default_paged_attention() -> bool {
    true
}

fn default_flash_attn() -> bool {
    true
}

fn default_gpu_layers() -> i32 {
    // Default to 99 layers on GPU for maximum performance
    // llama.cpp will automatically use fewer if the model has fewer layers
    // or fall back to CPU if no GPU is available
    99
}

/// Platform-adaptive thread count.
///
/// On Android big.LITTLE SoCs (e.g., Pixel 8: 4x Cortex-X3 + 4x Cortex-A715),
/// using all 8 cores causes thread contention — efficiency cores are much slower
/// and drag down throughput. Using 4 threads targets only the performance cores.
///
/// On desktop/macOS/iOS, auto-detect (0) uses all cores effectively since they
/// have uniform performance or GPU offloading handles the heavy lifting.
fn default_n_threads() -> usize {
    if cfg!(target_os = "android") {
        4
    } else {
        0 // auto-detect
    }
}

/// Platform-adaptive batch size for prompt processing.
///
/// On Android, memory bandwidth is limited — a batch size of 512 causes
/// excessive memory traffic on 1B+ models. 256 is a good balance between
/// prompt processing speed and memory pressure.
///
/// On desktop, 0 means the default (512) which is fine for higher-bandwidth systems.
fn default_n_batch() -> usize {
    if cfg!(target_os = "android") {
        256
    } else {
        0 // default (512)
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            chat_template: None,
            vision_encoder_path: None,
            context_length: default_context_length(),
            gpu_layers: default_gpu_layers(),
            paged_attention: default_paged_attention(),
            logging: false,
            n_threads: default_n_threads(),
            n_batch: default_n_batch(),
            flash_attn: default_flash_attn(),
            reasoning: false,
        }
    }
}

impl LlmConfig {
    /// Create a new config with the model path.
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    /// Set the chat template path.
    pub fn with_chat_template(mut self, path: impl Into<String>) -> Self {
        self.chat_template = Some(path.into());
        self
    }

    /// Set the sibling vision encoder / mmproj artifact path.
    pub fn with_vision_encoder(mut self, path: impl Into<String>) -> Self {
        self.vision_encoder_path = Some(path.into());
        self
    }

    /// Set the context length.
    pub fn with_context_length(mut self, length: usize) -> Self {
        self.context_length = length;
        self
    }

    /// Mark this as a reasoning ("thinking") model. Primes the `<think>`
    /// channel during chat-prompt construction. See [`Self::reasoning`].
    pub fn with_reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Set GPU layers to offload.
    pub fn with_gpu_layers(mut self, layers: i32) -> Self {
        self.gpu_layers = layers;
        self
    }

    /// Enable or disable paged attention.
    pub fn with_paged_attention(mut self, enabled: bool) -> Self {
        self.paged_attention = enabled;
        self
    }

    /// Enable or disable logging.
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.logging = enabled;
        self
    }

    /// Set the number of threads for inference. 0 = auto-detect.
    pub fn with_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// Set the batch size for prompt processing. 0 = default (512).
    pub fn with_batch_size(mut self, n_batch: usize) -> Self {
        self.n_batch = n_batch;
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::gateway::{FunctionDefinition, Tool};

    use super::*;

    fn test_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "f".into(),
                description: None,
                parameters: None,
            },
        }
    }

    /// Regression: the metadata bridge used a comma-delimited encoding, which
    /// silently corrupted the most common stop sequences of all. `"\n\n"`
    /// trimmed away to empty and was dropped entirely, so a cloud leg that
    /// should have stopped at a blank line ran on to `max_tokens`; `"\nUser:"`
    /// arrived as `"User:"`; and anything containing a comma was split in two.
    /// The local leg kept the caller's exact values, so the two legs stopped
    /// on different text.
    #[test]
    fn stop_sequences_round_trip_losslessly() {
        let hostile = vec![
            "\n\n".to_string(),
            "\nUser:".to_string(),
            "one, two".to_string(),
            "  ".to_string(),
            "<|im_end|>".to_string(),
        ];

        assert_eq!(
            parse_stop_sequences(&encode_stop_sequences(&hostile)),
            hostile
        );
    }

    #[test]
    fn stop_sequences_round_trip_an_empty_list() {
        let empty: Vec<String> = Vec::new();
        assert_eq!(parse_stop_sequences(&encode_stop_sequences(&empty)), empty);
    }

    /// Hand-written metadata predates the JSON encoding and must keep working.
    #[test]
    fn stop_sequences_still_read_the_legacy_comma_form() {
        assert_eq!(
            parse_stop_sequences("<|im_end|>,<|im_start|>"),
            vec!["<|im_end|>".to_string(), "<|im_start|>".to_string()]
        );
        assert_eq!(
            parse_stop_sequences(" STOP , END "),
            vec!["STOP".to_string(), "END".to_string()]
        );
        assert!(parse_stop_sequences("").is_empty());
    }

    #[test]
    fn test_partial_token_new() {
        let token = PartialToken::new("hello".to_string(), 0, "hello".to_string());
        assert_eq!(token.token, "hello");
        assert_eq!(token.index, 0);
        assert_eq!(token.cumulative_text, "hello");
        assert!(token.token_id.is_none());
        assert!(token.finish_reason.is_none());
        assert!(!token.is_final());
    }

    #[test]
    fn test_partial_token_with_token_id() {
        let token =
            PartialToken::new("world".to_string(), 1, "hello world".to_string()).with_token_id(42);
        assert_eq!(token.token_id, Some(42));
    }

    #[test]
    fn test_partial_token_with_finish_reason() {
        let token = PartialToken::new("".to_string(), 5, "final text".to_string())
            .with_finish_reason("stop");
        assert_eq!(token.finish_reason, Some("stop".to_string()));
        assert!(token.is_final());
    }

    #[test]
    fn test_partial_token_chained_builders() {
        let token = PartialToken::new("token".to_string(), 3, "all tokens".to_string())
            .with_token_id(100)
            .with_finish_reason("length");
        assert_eq!(token.token, "token");
        assert_eq!(token.index, 3);
        assert_eq!(token.token_id, Some(100));
        assert_eq!(token.finish_reason, Some("length".to_string()));
        assert!(token.is_final());
    }

    #[test]
    fn test_chat_message_constructors() {
        let user = ChatMessage::user("hello");
        assert_eq!(user.role, MessageRole::User);
        assert_eq!(user.content, "hello");

        let system = ChatMessage::system("you are helpful");
        assert_eq!(system.role, MessageRole::System);
        assert_eq!(system.content, "you are helpful");

        let assistant = ChatMessage::assistant("hi there");
        assert_eq!(assistant.role, MessageRole::Assistant);
        assert_eq!(assistant.content, "hi there");
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage::user("test");
        let json = serde_json::to_string(&msg).unwrap();
        // Role should serialize to lowercase
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"test\""));

        // Deserialize back
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, MessageRole::User);
        assert_eq!(parsed.content, "test");
    }

    #[test]
    fn test_chat_message_role_as_str() {
        let system = ChatMessage::system("sys");
        let user = ChatMessage::user("usr");
        let assistant = ChatMessage::assistant("ast");

        assert_eq!(system.role.as_str(), "system");
        assert_eq!(user.role.as_str(), "user");
        assert_eq!(assistant.role.as_str(), "assistant");
    }

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 2048);
        // Deterministic-by-default: temperature=0.0 is the SDK's documented
        // default (see `default_temperature` doc-comment). Callers that want
        // sampling opt in via `GenerationConfig::creative()` or explicit
        // `with_temperature(x)`.
        assert!(config.temperature.abs() < f32::EPSILON);
        assert!((config.top_p - 0.9).abs() < f32::EPSILON);
        assert!((config.min_p - 0.05).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 40);
        assert!((config.repetition_penalty - 1.1).abs() < f32::EPSILON);
        assert!(config.stop_sequences.is_empty());
    }

    #[test]
    fn test_generation_config_default_has_empty_tools() {
        let config = GenerationConfig::default();

        assert!(config.tools.is_empty());
    }

    #[test]
    fn test_generation_config_with_tools_sets_tools() {
        let config = GenerationConfig::default().with_tools([test_tool()]);

        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.tools[0].tool_type, "function");
        assert_eq!(config.tools[0].function.name, "f");
    }

    #[test]
    fn test_generation_config_deserializes_legacy_json_with_empty_tools() {
        let config: GenerationConfig = serde_json::from_str(r#"{"max_tokens":128}"#).unwrap();

        assert_eq!(config.max_tokens, 128);
        assert!(config.tools.is_empty());
    }

    #[test]
    fn test_generation_config_tools_serde_round_trip() {
        let config = GenerationConfig::default().with_tools([test_tool()]);

        let json = serde_json::to_string(&config).unwrap();
        let parsed: GenerationConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.tools.len(), 1);
        assert_eq!(parsed.tools[0].tool_type, "function");
        assert_eq!(parsed.tools[0].function.name, "f");
    }

    #[test]
    fn test_generation_config_with_max_tokens() {
        let config = GenerationConfig::default().with_max_tokens(1024);
        assert_eq!(config.max_tokens, 1024);
    }

    #[test]
    fn test_generation_config_with_stop_sequences() {
        let config = GenerationConfig::default()
            .with_stop("<|end|>")
            .with_stop("STOP");
        assert_eq!(config.stop_sequences.len(), 2);
        assert_eq!(config.stop_sequences[0], "<|end|>");
        assert_eq!(config.stop_sequences[1], "STOP");
    }

    #[test]
    fn test_llm_config_defaults() {
        let config = LlmConfig::new("/path/to/model.gguf");
        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.context_length, 4096);
        assert_eq!(config.gpu_layers, 99);
        assert!(config.chat_template.is_none());
        assert!(!config.logging);
        assert!(config.paged_attention); // Default is true for better memory efficiency
    }

    #[test]
    fn test_llm_config_with_context_length() {
        let config = LlmConfig::new("/path/to/model.gguf").with_context_length(8192);
        assert_eq!(config.context_length, 8192);
    }

    #[test]
    fn test_llm_config_with_chat_template() {
        let config = LlmConfig::new("/path/to/model.gguf").with_chat_template("chatml".to_string());
        assert_eq!(config.chat_template, Some("chatml".to_string()));
    }

    #[test]
    fn test_llm_config_with_vision_encoder_path() {
        let config =
            LlmConfig::new("/path/to/model.gguf").with_vision_encoder("/path/to/mmproj.gguf");
        assert_eq!(
            config.vision_encoder_path.as_deref(),
            Some("/path/to/mmproj.gguf")
        );
    }

    #[test]
    fn test_llm_config_platform_adaptive_defaults() {
        let config = LlmConfig::default();
        // On Android: n_threads=4, n_batch=256 (mobile-optimized)
        // On other platforms: n_threads=0 (auto), n_batch=0 (default 512)
        if cfg!(target_os = "android") {
            assert_eq!(config.n_threads, 4);
            assert_eq!(config.n_batch, 256);
        } else {
            assert_eq!(config.n_threads, 0);
            assert_eq!(config.n_batch, 0);
        }
    }
}
