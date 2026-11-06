//! Content capture: message model, capture policy, redaction, and bounded
//! truncation that keeps structured messages valid.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::semconv::{
    CAPTURE_CONTENT_ENV, DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES, REDACTION_REPLACEMENT,
    TRUNCATION_MARKER, TRUNCATION_MIN_STRING_BYTES,
};

/// Where captured content is recorded.
///
/// Values match the GenAI `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`
/// configuration. The default is [`ContentCapture::SpanOnly`]: unlike the
/// upstream default, text is captured once telemetry is enabled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentCapture {
    /// Metadata only: no prompts, outputs, tool payloads, or prompt variables.
    NoContent,
    /// Content as JSON-string span attributes.
    #[default]
    SpanOnly,
    /// Content as structured log events only.
    EventOnly,
    /// Content on both spans and events.
    SpanAndEvent,
}

impl ContentCapture {
    /// Parses a configuration value case-insensitively. Unknown values mean
    /// [`ContentCapture::NoContent`], as the conventions require.
    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "span_only" => Self::SpanOnly,
            "event_only" => Self::EventOnly,
            "span_and_event" => Self::SpanAndEvent,
            _ => Self::NoContent,
        }
    }

    /// Reads `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`; `None` when unset.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        std::env::var(CAPTURE_CONTENT_ENV)
            .ok()
            .map(|v| Self::parse(&v))
    }

    /// The configuration value (`span_only`, …).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoContent => "no_content",
            Self::SpanOnly => "span_only",
            Self::EventOnly => "event_only",
            Self::SpanAndEvent => "span_and_event",
        }
    }

    pub(crate) fn on_spans(self) -> bool {
        matches!(self, Self::SpanOnly | Self::SpanAndEvent)
    }

    pub(crate) fn on_events(self) -> bool {
        matches!(self, Self::EventOnly | Self::SpanAndEvent)
    }

    pub(crate) fn any(self) -> bool {
        self != Self::NoContent
    }
}

/// Rewrites content before it is recorded.
///
/// `kind` is the attribute key (`gen_ai.input.messages`, …, or
/// `error.message` for error descriptions). Returning `None` drops the value.
/// Runs before truncation and serialization, on the caller's thread.
pub trait Redactor: Send + Sync {
    /// Returns the value to record, or `None` to drop it.
    fn redact(&self, kind: &str, value: Value) -> Option<Value>;
}

impl<F> Redactor for F
where
    F: Fn(&str, Value) -> Option<Value> + Send + Sync,
{
    fn redact(&self, kind: &str, value: Value) -> Option<Value> {
        self(kind, value)
    }
}

/// Replaces every occurrence of configured substrings in every JSON string
/// value with `[REDACTED]`. The reference redactor used by the contract
/// fixtures.
#[derive(Clone, Debug, Default)]
pub struct SubstringRedactor {
    needles: Vec<String>,
}

impl SubstringRedactor {
    /// Creates a redactor for the given substrings (empty ones are ignored).
    pub fn new<I, S>(needles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            needles: needles
                .into_iter()
                .map(Into::into)
                .filter(|s: &String| !s.is_empty())
                .collect(),
        }
    }
}

impl Redactor for SubstringRedactor {
    fn redact(&self, _kind: &str, mut value: Value) -> Option<Value> {
        map_strings(&mut value, &mut |s| {
            for needle in &self.needles {
                if s.contains(needle.as_str()) {
                    *s = s.replace(needle.as_str(), REDACTION_REPLACEMENT);
                }
            }
        });
        Some(value)
    }
}

/// Content capture configuration.
#[derive(Clone)]
pub struct ContentPolicy {
    capture: ContentCapture,
    max_attribute_bytes: usize,
    redactor: Option<Arc<dyn Redactor>>,
}

impl Default for ContentPolicy {
    /// Product default: text captured on spans, 16 KiB per attribute, no
    /// redaction hook. Does not read the environment; see
    /// [`ContentPolicy::from_env`].
    fn default() -> Self {
        Self {
            capture: ContentCapture::default(),
            max_attribute_bytes: DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES,
            redactor: None,
        }
    }
}

impl ContentPolicy {
    /// Default policy with the capture mode overridden by
    /// `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` when it is set.
    #[must_use]
    pub fn from_env() -> Self {
        let mut policy = Self::default();
        if let Some(capture) = ContentCapture::from_env() {
            policy.capture = capture;
        }
        policy
    }

    /// Metadata-only policy.
    #[must_use]
    pub fn metadata_only() -> Self {
        Self::default().with_capture(ContentCapture::NoContent)
    }

    /// Sets the capture mode.
    #[must_use]
    pub fn with_capture(mut self, capture: ContentCapture) -> Self {
        self.capture = capture;
        self
    }

    /// Sets the per-attribute byte budget (minimum 64 bytes).
    #[must_use]
    pub fn with_max_attribute_bytes(mut self, bytes: usize) -> Self {
        self.max_attribute_bytes = bytes.max(64);
        self
    }

    /// Installs a redaction hook.
    #[must_use]
    pub fn with_redactor(mut self, redactor: impl Redactor + 'static) -> Self {
        self.redactor = Some(Arc::new(redactor));
        self
    }

    /// The capture mode.
    #[must_use]
    pub fn capture(&self) -> ContentCapture {
        self.capture
    }

    /// The per-attribute byte budget.
    #[must_use]
    pub fn max_attribute_bytes(&self) -> usize {
        self.max_attribute_bytes
    }

    pub(crate) fn redact(&self, kind: &str, value: Value) -> Option<Value> {
        match &self.redactor {
            Some(r) => r.redact(kind, value),
            None => Some(value),
        }
    }

    /// Redacts and bounds a structured content value. Returns the JSON string
    /// to record (if any) and whether truncation applied.
    pub(crate) fn prepare_json(&self, kind: &str, value: Value) -> Prepared<Value> {
        let Some(value) = self.redact(kind, value) else {
            return Prepared {
                value: None,
                truncated: false,
            };
        };
        truncate_json(value, self.max_attribute_bytes)
    }

    /// Redacts and bounds a plain-string content value.
    pub(crate) fn prepare_text(&self, kind: &str, text: String) -> Prepared<String> {
        match self.redact(kind, Value::String(text)) {
            Some(Value::String(s)) => {
                let (s, truncated) = truncate_text(s, self.max_attribute_bytes);
                Prepared {
                    value: Some(s),
                    truncated,
                }
            }
            Some(other) => {
                let (s, truncated) = truncate_text(other.to_string(), self.max_attribute_bytes);
                Prepared {
                    value: Some(s),
                    truncated,
                }
            }
            None => Prepared {
                value: None,
                truncated: false,
            },
        }
    }
}

impl fmt::Debug for ContentPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContentPolicy")
            .field("capture", &self.capture)
            .field("max_attribute_bytes", &self.max_attribute_bytes)
            .field("redactor", &self.redactor.as_ref().map(|_| "<hook>"))
            .finish()
    }
}

/// A bounded content value.
#[derive(Debug)]
pub(crate) struct Prepared<T> {
    /// The value to record, `None` when dropped by redaction or truncation.
    pub value: Option<T>,
    /// Whether the truncation algorithm changed or omitted the value.
    pub truncated: bool,
}

/// Media modality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Text.
    Text,
    /// Images.
    Image,
    /// Audio.
    Audio,
    /// Video.
    Video,
}

impl Modality {
    /// Lowercase name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// Metadata describing media without its bytes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MediaMetadata {
    /// Modality of the media.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<Modality>,
    /// MIME type, e.g. `audio/wav`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Size of the media in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Duration for audio/video, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<f64>,
    /// Width in pixels for images/video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Height in pixels for images/video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

/// One part of a message, following the pinned GenAI message schemas.
///
/// There is deliberately no variant for raw bytes: media is described by
/// [`Part::Media`] metadata or by caller-provided references.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Part {
    /// Text content.
    Text {
        /// The text.
        content: String,
    },
    /// Model reasoning content.
    Reasoning {
        /// The reasoning text.
        content: String,
    },
    /// A tool call requested by the model.
    ToolCall {
        /// Tool call id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Tool name.
        name: String,
        /// Call arguments.
        #[serde(default, skip_serializing_if = "Value::is_null")]
        arguments: Value,
    },
    /// A tool's response.
    ToolCallResponse {
        /// Tool call id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The response payload.
        response: Value,
    },
    /// A caller-provided URI reference to media.
    Uri {
        /// Modality of the referenced media.
        modality: Modality,
        /// MIME type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// The URI.
        uri: String,
    },
    /// A caller-provided file id reference to media.
    File {
        /// Modality of the referenced media.
        modality: Modality,
        /// MIME type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Provider or application file id.
        file_id: String,
    },
    /// Media described by metadata only (generic part `media`).
    Media(MediaMetadata),
}

impl Part {
    /// A text part.
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text {
            content: content.into(),
        }
    }
}

/// An input (chat history) message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputMessage {
    /// `system`, `user`, `assistant`, `tool`, or a custom role.
    pub role: String,
    /// Message parts.
    pub parts: Vec<Part>,
    /// Optional participant name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl InputMessage {
    /// A single-text-part message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            parts: vec![Part::text(content)],
            name: None,
        }
    }
}

/// A model output message (one per choice).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputMessage {
    /// Usually `assistant`.
    pub role: String,
    /// Message parts.
    pub parts: Vec<Part>,
    /// Optional participant name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `stop`, `length`, `content_filter`, `tool_call`, `compaction`, `error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

impl OutputMessage {
    /// A single-text-part assistant message.
    pub fn assistant_text(content: impl Into<String>, finish_reason: Option<&str>) -> Self {
        Self {
            role: "assistant".into(),
            parts: vec![Part::text(content)],
            name: None,
            finish_reason: finish_reason.map(str::to_owned),
        }
    }
}

fn map_strings(value: &mut Value, f: &mut impl FnMut(&mut String)) {
    match value {
        Value::String(s) => f(s),
        Value::Array(items) => items.iter_mut().for_each(|v| map_strings(v, f)),
        Value::Object(map) => map.values_mut().for_each(|v| map_strings(v, f)),
        _ => {}
    }
}

fn max_string_bytes(value: &Value) -> usize {
    match value {
        Value::String(s) => s.len(),
        Value::Array(items) => items.iter().map(max_string_bytes).max().unwrap_or(0),
        Value::Object(map) => map.values().map(max_string_bytes).max().unwrap_or(0),
        _ => 0,
    }
}

fn serialized_len(value: &Value) -> usize {
    serde_json::to_string(value).map_or(usize::MAX, |s| s.len())
}

/// Longest prefix of at most `max` UTF-8 bytes that does not split a code point.
fn utf8_prefix(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn cap_strings(original: &Value, k: usize) -> Value {
    let mut value = original.clone();
    map_strings(&mut value, &mut |s| {
        if s.len() > k {
            let mut capped = String::with_capacity(k + TRUNCATION_MARKER.len());
            capped.push_str(utf8_prefix(s, k));
            capped.push_str(TRUNCATION_MARKER);
            *s = capped;
        }
    });
    value
}

/// The contract's deterministic truncation algorithm (CONTRACT.md,
/// "Truncation").
pub(crate) fn truncate_json(original: Value, budget: usize) -> Prepared<Value> {
    if serialized_len(&original) <= budget {
        return Prepared {
            value: Some(original),
            truncated: false,
        };
    }
    let mut k = max_string_bytes(&original);
    let mut current = original.clone();
    while serialized_len(&current) > budget && k > TRUNCATION_MIN_STRING_BYTES {
        k /= 2;
        current = cap_strings(&original, k);
    }
    if let Value::Array(items) = &mut current {
        while items.len() > 1 && serialized_len(&Value::Array(items.clone())) > budget {
            items.remove(0);
        }
    }
    if serialized_len(&current) > budget {
        return Prepared {
            value: None,
            truncated: true,
        };
    }
    Prepared {
        value: Some(current),
        truncated: true,
    }
}

/// Plain-string truncation with `K = budget`.
pub(crate) fn truncate_text(s: String, budget: usize) -> (String, bool) {
    if s.len() <= budget {
        return (s, false);
    }
    let mut out = utf8_prefix(&s, budget).to_owned();
    out.push_str(TRUNCATION_MARKER);
    (out, true)
}

/// Bounds a human-readable message to `max_chars` characters.
pub(crate) fn bound_chars(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parts_serialize_with_schema_shapes() {
        let msg = InputMessage {
            role: "user".into(),
            parts: vec![
                Part::text("hi"),
                Part::Uri {
                    modality: Modality::Audio,
                    mime_type: Some("audio/wav".into()),
                    uri: "file:///a.wav".into(),
                },
                Part::Media(MediaMetadata {
                    modality: Some(Modality::Image),
                    mime_type: None,
                    size_bytes: Some(10),
                    ..Default::default()
                }),
            ],
            name: None,
        };
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(
            v,
            json!({"role":"user","parts":[
                {"type":"text","content":"hi"},
                {"type":"uri","modality":"audio","mime_type":"audio/wav","uri":"file:///a.wav"},
                {"type":"media","modality":"image","size_bytes":10}
            ]})
        );
    }

    #[test]
    fn small_values_are_untouched() {
        let p = truncate_json(json!(["a"]), 100);
        assert!(!p.truncated);
        assert_eq!(p.value.unwrap(), json!(["a"]));
    }

    #[test]
    fn halves_strings_without_splitting_code_points() {
        let long = "é".repeat(100); // 200 bytes
        let p = truncate_json(json!([{ "content": long }]), 120);
        assert!(p.truncated);
        let v = p.value.unwrap();
        let s = v[0]["content"].as_str().unwrap();
        assert!(s.ends_with(TRUNCATION_MARKER));
        let prefix = s.strip_suffix(TRUNCATION_MARKER).unwrap();
        assert!(prefix.chars().all(|c| c == 'é'));
        assert!(serde_json::to_string(&v).unwrap().len() <= 120);
    }

    #[test]
    fn surrogate_pair_characters_are_kept_whole() {
        let s = "😀".repeat(10); // 4 bytes each
        assert_eq!(utf8_prefix(&s, 6), "😀");
        assert_eq!(utf8_prefix(&s, 8), "😀😀");
    }

    #[test]
    fn drops_oldest_messages_then_omits() {
        let msgs = json!([
            {"role":"user","parts":[{"type":"text","content":"x".repeat(40)}]},
            {"role":"user","parts":[{"type":"text","content":"y".repeat(40)}]}
        ]);
        let p = truncate_json(msgs, 90);
        assert!(p.truncated);
        let v = p.value.unwrap();
        assert_eq!(v.as_array().unwrap().len(), 1);
        assert!(v[0]["parts"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with('y'));

        let p = truncate_json(json!({"k": "v".repeat(1000), "other": 1}), 10);
        assert!(p.truncated);
        assert!(p.value.is_none());
    }

    #[test]
    fn capture_parsing_defaults_to_no_content_on_garbage() {
        assert_eq!(ContentCapture::parse("SPAN_ONLY"), ContentCapture::SpanOnly);
        assert_eq!(
            ContentCapture::parse("Span_And_Event"),
            ContentCapture::SpanAndEvent
        );
        assert_eq!(ContentCapture::parse("yes"), ContentCapture::NoContent);
    }

    #[test]
    fn substring_redactor_rewrites_nested_strings() {
        let r = SubstringRedactor::new(["s3cret"]);
        let v = r
            .redact("k", json!({"a":["x s3cret y"],"b":{"c":"s3cret"}}))
            .unwrap();
        assert_eq!(v, json!({"a":["x [REDACTED] y"],"b":{"c":"[REDACTED]"}}));
    }
}
