//! Content capture: message model, capture policy, redaction (callback hook
//! and declarative rules), preflight budgets, and bounded truncation that
//! keeps structured messages valid.
//!
//! Every governed value goes through the pipeline of CONTRACT.md "Content
//! pipeline": capture gate, omit rules, preflight before cloning, clone,
//! callback redactor, output validation, replacement rules, truncation,
//! retained-operation budget, record. A failing stage fails closed: the value
//! is omitted and listed in `xybrid.genai.content.truncated_attributes`.

use std::env::VarError;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::preflight::{self, PreflightError};
use crate::semconv::{
    CAPTURE_CONTENT_ENV, CAPTURE_CONTENT_LEGACY_ENV, DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES,
    DEFAULT_MAX_RETAINED_OPERATION_BYTES, DEFAULT_MAX_TRAVERSAL_BYTES, REDACTION_REPLACEMENT,
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
    /// Parses a configuration value (trimmed, case-insensitive). Unknown
    /// values mean [`ContentCapture::NoContent`], as the conventions require.
    #[must_use]
    pub fn parse(value: &str) -> Self {
        Self::parse_known(value).unwrap_or(Self::NoContent)
    }

    fn parse_known(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "no_content" => Some(Self::NoContent),
            "span_only" => Some(Self::SpanOnly),
            "event_only" => Some(Self::EventOnly),
            "span_and_event" => Some(Self::SpanAndEvent),
            _ => None,
        }
    }

    /// The capture mode configured in the environment, `None` when neither
    /// variable is set.
    ///
    /// `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` wins; otherwise the legacy
    /// `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` applies (`true`
    /// means `span_only`). Unknown values, and values that are not valid
    /// UTF-8, mean `no_content` (fail closed). Empty values are unset.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        Self::from_vars(|name| std::env::var(name))
    }

    pub(crate) fn from_vars(
        var: impl Fn(&str) -> std::result::Result<String, VarError>,
    ) -> Option<Self> {
        match read_env(var(CAPTURE_CONTENT_ENV)) {
            EnvValue::Set(v) => return Some(Self::parse(&v)),
            EnvValue::Unreadable => return Some(Self::NoContent),
            EnvValue::Unset => {}
        }
        match read_env(var(CAPTURE_CONTENT_LEGACY_ENV)) {
            EnvValue::Set(v) => Some(match v.to_ascii_lowercase().as_str() {
                "true" => Self::SpanOnly,
                "false" => Self::NoContent,
                other => Self::parse(other),
            }),
            EnvValue::Unreadable => Some(Self::NoContent),
            EnvValue::Unset => None,
        }
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
        self.on_spans() || self.on_events()
    }
}

enum EnvValue {
    Unset,
    Unreadable,
    Set(String),
}

fn read_env(value: std::result::Result<String, VarError>) -> EnvValue {
    match value {
        Ok(v) if v.trim().is_empty() => EnvValue::Unset,
        Ok(v) => EnvValue::Set(v.trim().to_owned()),
        Err(VarError::NotPresent) => EnvValue::Unset,
        Err(VarError::NotUnicode(_)) => EnvValue::Unreadable,
    }
}

/// A redaction hook failed. Opaque: it never carries content or the hook's
/// own error message.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RedactionError;

impl fmt::Display for RedactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("content redaction failed")
    }
}

impl std::error::Error for RedactionError {}

/// Rewrites content before it is recorded.
///
/// `kind` is the attribute key (`gen_ai.input.messages`,
/// `gen_ai.prompt.variable.<name>`, `error.message`,
/// `xybrid.genai.feedback.comment`, …). `value` is the observer's canonical
/// JSON clone, never the caller's object; string kinds (prompt variables,
/// error messages, feedback comments) receive a string and must return one.
/// Returning `None` drops the value; `Some(Value::Null)` records JSON null.
/// Runs on the caller's thread, before declarative rules and truncation.
pub trait Redactor: Send + Sync {
    /// Returns the value to record, or `None` to drop it.
    fn redact(&self, kind: &str, value: Value) -> Option<Value>;

    /// Fallible form; the observer calls this one. An error fails closed:
    /// the value is omitted and listed as truncated, and later values are
    /// redacted normally.
    ///
    /// # Errors
    ///
    /// [`RedactionError`] when the hook cannot redact this value.
    fn try_redact(
        &self,
        kind: &str,
        value: Value,
    ) -> std::result::Result<Option<Value>, RedactionError> {
        Ok(self.redact(kind, value))
    }
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

/// A declarative redaction rule (same JSON shape in every package and binding).
///
/// ```json
/// { "type": "replace_literal", "find": "<non-empty>", "replace": "<string>" }
/// { "type": "omit_attribute",  "key":  "<non-empty attribute key>" }
/// ```
///
/// `omit_attribute` drops exactly the named attribute key (case-sensitive, no
/// patterns) before preflight and the callback hook. `replace_literal` rules
/// run after the callback hook, in order, each on the previous rule's output:
/// every non-overlapping occurrence of `find` in every JSON string value
/// (never object keys) is replaced left to right. Numbers, booleans, null,
/// structure and keys are untouched. Rules can only be built through the
/// validating constructors or `Deserialize`, so an invalid rule cannot exist.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", try_from = "RawRule")]
pub enum RedactionRule {
    /// Replaces a literal substring in every JSON string value.
    #[non_exhaustive]
    ReplaceLiteral {
        /// Non-empty text to find.
        find: String,
        /// Replacement text.
        replace: String,
    },
    /// Omits the content attribute with exactly this key.
    #[non_exhaustive]
    OmitAttribute {
        /// Non-empty attribute key (`gen_ai.input.messages`,
        /// `gen_ai.prompt.variable.<name>`, `error.message`, …).
        key: String,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawRule {
    ReplaceLiteral { find: String, replace: String },
    OmitAttribute { key: String },
}

impl TryFrom<RawRule> for RedactionRule {
    type Error = Error;

    fn try_from(raw: RawRule) -> Result<Self> {
        match raw {
            RawRule::ReplaceLiteral { find, replace } => Self::replace_literal(find, replace),
            RawRule::OmitAttribute { key } => Self::omit_attribute(key),
        }
    }
}

impl RedactionRule {
    /// A literal replacement rule.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when `find` is empty.
    pub fn replace_literal(find: impl Into<String>, replace: impl Into<String>) -> Result<Self> {
        let find = find.into();
        if find.is_empty() {
            return Err(Error::Config(
                "redaction rule replace_literal needs a non-empty find".into(),
            ));
        }
        Ok(Self::ReplaceLiteral {
            find,
            replace: replace.into(),
        })
    }

    /// An attribute omission rule.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when `key` is empty.
    pub fn omit_attribute(key: impl Into<String>) -> Result<Self> {
        let key = key.into();
        if key.is_empty() {
            return Err(Error::Config(
                "redaction rule omit_attribute needs a non-empty key".into(),
            ));
        }
        Ok(Self::OmitAttribute { key })
    }
}

/// Content capture configuration.
///
/// [`ContentPolicy::default`] leaves the capture mode unset: an observer
/// built with it resolves the mode from the environment once, at build time
/// (see [`ContentCapture::from_env`]), falling back to `span_only`.
/// [`ContentPolicy::with_capture`] sets it explicitly.
#[derive(Clone)]
pub struct ContentPolicy {
    capture: Option<ContentCapture>,
    snapshot: Option<ContentCapture>,
    max_attribute_bytes: usize,
    max_retained_operation_bytes: usize,
    max_traversal_bytes: usize,
    redactor: Option<Arc<dyn Redactor>>,
    rules: Arc<[RedactionRule]>,
}

impl Default for ContentPolicy {
    /// Product default: capture mode from the environment (else span
    /// attributes), 16 KiB per attribute, 480 KiB retained per operation and
    /// traversed per value, no redaction hook, no rules.
    fn default() -> Self {
        Self {
            capture: None,
            snapshot: None,
            max_attribute_bytes: DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES,
            max_retained_operation_bytes: DEFAULT_MAX_RETAINED_OPERATION_BYTES,
            max_traversal_bytes: DEFAULT_MAX_TRAVERSAL_BYTES,
            redactor: None,
            rules: Arc::from(Vec::new()),
        }
    }
}

impl ContentPolicy {
    /// The default policy; the capture mode is resolved from the environment.
    /// Equivalent to [`ContentPolicy::default`].
    #[must_use]
    pub fn from_env() -> Self {
        Self::default()
    }

    /// Metadata-only policy (explicit `no_content`).
    #[must_use]
    pub fn metadata_only() -> Self {
        Self::default().with_capture(ContentCapture::NoContent)
    }

    /// Sets the capture mode explicitly; the environment is then ignored.
    #[must_use]
    pub fn with_capture(mut self, capture: ContentCapture) -> Self {
        self.capture = Some(capture);
        self.snapshot = None;
        self
    }

    /// Sets the per-attribute byte budget `B` (minimum 1).
    #[must_use]
    pub fn with_max_attribute_bytes(mut self, bytes: usize) -> Self {
        self.max_attribute_bytes = bytes.max(1);
        self
    }

    /// Sets the budget of content retained by one operation (minimum 1).
    #[must_use]
    pub fn with_max_retained_operation_bytes(mut self, bytes: usize) -> Self {
        self.max_retained_operation_bytes = bytes.max(1);
        self
    }

    /// Sets the preflight traversal budget per value (minimum 1).
    #[must_use]
    pub fn with_max_traversal_bytes(mut self, bytes: usize) -> Self {
        self.max_traversal_bytes = bytes.max(1);
        self
    }

    /// Installs a redaction hook. Declarative rules run after it.
    #[must_use]
    pub fn with_redactor(mut self, redactor: impl Redactor + 'static) -> Self {
        self.redactor = Some(Arc::new(redactor));
        self
    }

    /// Appends declarative redaction rules (applied in order).
    #[must_use]
    pub fn with_rules(mut self, rules: impl IntoIterator<Item = RedactionRule>) -> Self {
        let mut all = self.rules.to_vec();
        all.extend(rules);
        self.rules = Arc::from(all);
        self
    }

    /// Appends one declarative redaction rule.
    #[must_use]
    pub fn with_rule(self, rule: RedactionRule) -> Self {
        self.with_rules([rule])
    }

    /// The explicitly configured capture mode, if any.
    #[must_use]
    pub fn explicit_capture(&self) -> Option<ContentCapture> {
        self.capture
    }

    /// The resolved capture mode: the explicit mode, else the snapshot an
    /// observer took when it was built, else the environment now, else
    /// `span_only`.
    #[must_use]
    pub fn capture(&self) -> ContentCapture {
        self.capture
            .or(self.snapshot)
            .or_else(ContentCapture::from_env)
            .unwrap_or_default()
    }

    /// Freezes the resolved capture mode (observers call this once at build).
    pub(crate) fn resolved(mut self) -> Self {
        self.snapshot = Some(self.capture());
        self
    }

    #[cfg(test)]
    pub(crate) fn resolved_with(
        mut self,
        var: impl Fn(&str) -> std::result::Result<String, VarError>,
    ) -> Self {
        self.snapshot = Some(
            self.capture
                .or_else(|| ContentCapture::from_vars(var))
                .unwrap_or_default(),
        );
        self
    }

    /// The per-attribute byte budget.
    #[must_use]
    pub fn max_attribute_bytes(&self) -> usize {
        self.max_attribute_bytes
    }

    /// The retained-operation byte budget.
    #[must_use]
    pub fn max_retained_operation_bytes(&self) -> usize {
        self.max_retained_operation_bytes
    }

    /// The traversal byte budget.
    #[must_use]
    pub fn max_traversal_bytes(&self) -> usize {
        self.max_traversal_bytes
    }

    /// The declarative rules, in application order.
    #[must_use]
    pub fn rules(&self) -> &[RedactionRule] {
        &self.rules
    }

    fn omits(&self, kind: &str) -> bool {
        self.rules
            .iter()
            .any(|r| matches!(r, RedactionRule::OmitAttribute { key } if key == kind))
    }

    fn traversal(&self) -> u64 {
        self.max_traversal_bytes as u64
    }

    /// Runs the content pipeline for a structured value (governed or schema
    /// content). The caller's value is preflighted before it is cloned.
    pub(crate) fn prepare_json<T: Serialize + ?Sized>(
        &self,
        kind: &str,
        value: &T,
    ) -> Outcome<BoundedJson> {
        if self.omits(kind) {
            return Outcome::Dropped;
        }
        let budget = self.traversal();
        let cost = match preflight::measure(value, budget) {
            Ok(cost) => cost,
            Err(e) => return failed(kind, e.into()),
        };
        let Ok(clone) = serde_json::to_value(value) else {
            return failed(kind, Failure::Unrepresentable);
        };
        let (mut value, cost) = match self.call_redactor(kind, clone, cost) {
            Ok(Some(v)) => v,
            Ok(None) => return Outcome::Dropped,
            Err(f) => return failed(kind, f),
        };
        if let Err(f) = self.apply_replacements(&mut value, cost) {
            return failed(kind, f);
        }
        match truncate_json(value, self.max_attribute_bytes) {
            Some(bounded) => Outcome::Keep(bounded),
            None => failed(kind, Failure::Truncation),
        }
    }

    /// Runs the content pipeline for a plain-string value (prompt variable,
    /// error message, feedback comment).
    pub(crate) fn prepare_text(
        &self,
        kind: &str,
        text: &str,
        bound: TextBound,
    ) -> Outcome<BoundedText> {
        if self.omits(kind) {
            return Outcome::Dropped;
        }
        let budget = self.traversal();
        let cost = match preflight::measure(text, budget) {
            Ok(cost) => cost,
            Err(e) => return failed(kind, e.into()),
        };
        let (mut value, cost) = match self.call_redactor(kind, Value::String(text.to_owned()), cost)
        {
            Ok(Some(v)) => v,
            Ok(None) => return Outcome::Dropped,
            Err(f) => return failed(kind, f),
        };
        if !value.is_string() {
            return failed(kind, Failure::InvalidOutput);
        }
        if let Err(f) = self.apply_replacements(&mut value, cost) {
            return failed(kind, f);
        }
        let Value::String(s) = value else {
            return failed(kind, Failure::InvalidOutput);
        };
        let (text, truncated) = match bound {
            TextBound::Bytes(b) => truncate_text(s, b),
            TextBound::Chars(n) => {
                let bounded = bound_chars(&s, n);
                let truncated = bounded.len() < s.len();
                (bounded, truncated)
            }
        };
        Outcome::Keep(BoundedText { text, truncated })
    }

    /// The plain-string byte bound `K = B`.
    pub(crate) fn text_bound(&self) -> TextBound {
        TextBound::Bytes(self.max_attribute_bytes)
    }

    /// Calls the hook (if any) on the clone and validates its output. Returns
    /// the value to continue with and its preflight cost, or `None` to drop.
    fn call_redactor(
        &self,
        kind: &str,
        clone: Value,
        cost: u64,
    ) -> std::result::Result<Option<(Value, u64)>, Failure> {
        let Some(redactor) = &self.redactor else {
            return Ok(Some((clone, cost)));
        };
        match redactor.try_redact(kind, clone) {
            Err(RedactionError) => Err(Failure::Redactor),
            Ok(None) => Ok(None),
            Ok(Some(out)) => match preflight::measure(&out, self.traversal()) {
                Ok(cost) => Ok(Some((out, cost))),
                Err(_) => Err(Failure::InvalidOutput),
            },
        }
    }

    /// Applies the replacement rules to every string value. Fails when the
    /// result's preflight cost exceeds the traversal budget, or when one
    /// string alone outgrows it while its rules are applied (bounded work).
    fn apply_replacements(&self, value: &mut Value, cost: u64) -> std::result::Result<(), Failure> {
        if !self
            .rules
            .iter()
            .any(|r| matches!(r, RedactionRule::ReplaceLiteral { .. }))
        {
            return Ok(());
        }
        let budget = self.traversal();
        let mut total = i128::from(cost);
        let mut overflow = false;
        map_strings(value, &mut |s| {
            if overflow {
                return;
            }
            let before = s.len();
            for rule in self.rules.iter() {
                if let RedactionRule::ReplaceLiteral { find, replace } = rule {
                    // Amendment A1 (b): the string's cost after this rule is
                    // computed from the match count before the new string is
                    // built, so an expanding rule never allocates beyond the
                    // budget (work stays bounded by the input size).
                    let matches = s.matches(find.as_str()).count();
                    if matches == 0 {
                        continue;
                    }
                    let grown = matches
                        .checked_mul(replace.len())
                        .and_then(|added| (s.len() - matches * find.len()).checked_add(added));
                    match grown {
                        Some(len) if preflight_len_cost(len) <= budget => {
                            *s = s.replace(find.as_str(), replace);
                        }
                        _ => {
                            overflow = true;
                            return;
                        }
                    }
                }
            }
            total += s.len() as i128 - before as i128;
        });
        if overflow || total > i128::from(budget) {
            return Err(Failure::RulesBudget);
        }
        Ok(())
    }
}

/// Preflight cost of a string of `len` UTF-8 bytes.
fn preflight_len_cost(len: usize) -> u64 {
    (crate::semconv::PREFLIGHT_NODE_BYTES as u64).saturating_add(len as u64)
}

impl fmt::Debug for ContentPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContentPolicy")
            .field("capture", &self.capture())
            .field("explicit", &self.capture.is_some())
            .field("max_attribute_bytes", &self.max_attribute_bytes)
            .field(
                "max_retained_operation_bytes",
                &self.max_retained_operation_bytes,
            )
            .field("max_traversal_bytes", &self.max_traversal_bytes)
            .field("redactor", &self.redactor.as_ref().map(|_| "<hook>"))
            .field("rules", &self.rules.len())
            .finish()
    }
}

/// How a plain string is bounded.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TextBound {
    /// Prefix of at most `K = B` bytes plus the truncation marker.
    Bytes(usize),
    /// At most this many code points, no marker (error messages).
    Chars(usize),
}

/// Result of the content pipeline for one value.
#[derive(Debug)]
pub(crate) enum Outcome<T> {
    /// Record this value.
    Keep(T),
    /// Intentionally dropped (omit rule or the hook returned drop): omitted,
    /// not listed as truncated.
    Dropped,
    /// Failed closed: omitted and listed as truncated.
    Failed,
}

/// Why a value failed closed (diagnostics only; never content).
#[derive(Clone, Copy, Debug)]
enum Failure {
    Depth,
    Traversal,
    Unrepresentable,
    Redactor,
    InvalidOutput,
    RulesBudget,
    Truncation,
}

impl From<PreflightError> for Failure {
    fn from(e: PreflightError) -> Self {
        match e {
            PreflightError::Depth => Self::Depth,
            PreflightError::Budget => Self::Traversal,
            PreflightError::Unrepresentable => Self::Unrepresentable,
        }
    }
}

fn failed<T>(kind: &str, reason: Failure) -> Outcome<T> {
    log::debug!("xybrid-observability: content attribute {kind} omitted ({reason:?})");
    Outcome::Failed
}

/// A bounded structured value: the value (for events) and its compact JSON
/// serialization (for spans).
#[derive(Debug)]
pub(crate) struct BoundedJson {
    pub value: Value,
    pub json: String,
    /// Whether truncation changed the value.
    pub truncated: bool,
}

/// A bounded plain string.
#[derive(Debug)]
pub(crate) struct BoundedText {
    pub text: String,
    /// Whether the bound changed the value.
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
        /// Call arguments (omitted when absent or `null`).
        #[serde(default, skip_serializing_if = "Value::is_null")]
        arguments: Value,
    },
    /// A tool's response.
    ToolCallResponse {
        /// Tool call id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The response payload (always serialized; `null` when absent).
        #[serde(default)]
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

fn to_json(value: &Value) -> String {
    serde_json::to_string(value).expect("a serde_json::Value always serializes")
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
    match original {
        Value::String(s) if s.len() > k => {
            let mut capped = String::with_capacity(k + TRUNCATION_MARKER.len());
            capped.push_str(utf8_prefix(s, k));
            capped.push_str(TRUNCATION_MARKER);
            Value::String(capped)
        }
        Value::Array(items) => Value::Array(items.iter().map(|v| cap_strings(v, k)).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, v)| (key.clone(), cap_strings(v, k)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The contract's deterministic truncation algorithm (CONTRACT.md,
/// "Truncation"), without quadratic work: halving rebuilds from the original a
/// logarithmic number of times, and dropping the oldest array elements uses
/// one pass over per-element sizes and serializes the kept suffix once.
/// Returns `None` when the value cannot fit `budget` (omitted, listed).
pub(crate) fn truncate_json(original: Value, budget: usize) -> Option<BoundedJson> {
    let json = to_json(&original);
    if json.len() <= budget {
        return Some(BoundedJson {
            value: original,
            json,
            truncated: false,
        });
    }
    let mut k = max_string_bytes(&original);
    let mut current: Option<(Value, String)> = None;
    let mut len = json.len();
    while len > budget && k > TRUNCATION_MIN_STRING_BYTES {
        k /= 2;
        let capped = cap_strings(&original, k);
        let capped_json = to_json(&capped);
        len = capped_json.len();
        current = Some((capped, capped_json));
    }
    let (value, json) = current.unwrap_or((original, json));
    if len <= budget {
        return Some(BoundedJson {
            value,
            json,
            truncated: true,
        });
    }
    let Value::Array(mut items) = value else {
        return None;
    };
    let parts: Vec<String> = items.iter().map(to_json).collect();
    // `[` + elements joined by `,` + `]`.
    let mut total =
        2 + parts.iter().map(String::len).sum::<usize>() + parts.len().saturating_sub(1);
    let mut start = 0;
    while total > budget && parts.len() - start > 1 {
        total -= parts[start].len() + 1;
        start += 1;
    }
    if total > budget {
        return None;
    }
    let mut kept = String::with_capacity(total);
    kept.push('[');
    for (i, part) in parts[start..].iter().enumerate() {
        if i > 0 {
            kept.push(',');
        }
        kept.push_str(part);
    }
    kept.push(']');
    items.drain(..start);
    Some(BoundedJson {
        value: Value::Array(items),
        json: kept,
        truncated: true,
    })
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

    fn keep<T>(o: Outcome<T>) -> T {
        match o {
            Outcome::Keep(v) => v,
            other => panic!("expected a kept value, got {:?}", other.kind()),
        }
    }

    impl<T> Outcome<T> {
        fn kind(&self) -> &'static str {
            match self {
                Self::Keep(_) => "keep",
                Self::Dropped => "dropped",
                Self::Failed => "failed",
            }
        }
    }

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
    fn tool_parts_keep_nulls_as_the_schema_requires() {
        let response: Part =
            serde_json::from_value(json!({"type": "tool_call_response", "id": "c"})).unwrap();
        assert_eq!(
            serde_json::to_value(&response).unwrap(),
            json!({"type": "tool_call_response", "id": "c", "response": null})
        );
        let call: Part =
            serde_json::from_value(json!({"type": "tool_call", "name": "f", "arguments": null}))
                .unwrap();
        assert_eq!(
            serde_json::to_value(&call).unwrap(),
            json!({"type": "tool_call", "name": "f"})
        );
    }

    #[test]
    fn small_values_are_untouched() {
        let p = truncate_json(json!(["a"]), 100).unwrap();
        assert!(!p.truncated);
        assert_eq!(p.value, json!(["a"]));
        assert_eq!(p.json, r#"["a"]"#);
    }

    #[test]
    fn halves_strings_without_splitting_code_points() {
        let long = "é".repeat(100); // 200 bytes
        let p = truncate_json(json!([{ "content": long }]), 120).unwrap();
        assert!(p.truncated);
        let s = p.value[0]["content"].as_str().unwrap();
        assert!(s.ends_with(TRUNCATION_MARKER));
        let prefix = s.strip_suffix(TRUNCATION_MARKER).unwrap();
        assert!(prefix.chars().all(|c| c == 'é'));
        assert!(p.json.len() <= 120);
        assert_eq!(p.json, to_json(&p.value));
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
        let p = truncate_json(msgs, 90).unwrap();
        assert!(p.truncated);
        assert_eq!(p.value.as_array().unwrap().len(), 1);
        assert!(p.value[0]["parts"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with('y'));
        assert_eq!(p.json, to_json(&p.value));

        assert!(truncate_json(json!({"k": "v".repeat(1000), "other": 1}), 10).is_none());
    }

    /// The v1 implementation (clone and re-serialize per removed element),
    /// kept as the reference for R-CONTENT-1.
    fn reference_truncate(original: Value, budget: usize) -> Option<Value> {
        let len = |v: &Value| to_json(v).len();
        if len(&original) <= budget {
            return Some(original);
        }
        let mut k = max_string_bytes(&original);
        let mut current = original.clone();
        while len(&current) > budget && k > TRUNCATION_MIN_STRING_BYTES {
            k /= 2;
            current = cap_strings(&original, k);
        }
        if let Value::Array(items) = &mut current {
            while items.len() > 1 && len(&Value::Array(items.clone())) > budget {
                items.remove(0);
            }
        }
        (len(&current) <= budget).then_some(current)
    }

    fn messages(n: usize) -> Value {
        Value::Array(
            (0..n)
                .map(|i| {
                    json!({"role": "user", "parts": [{"type": "text", "content": format!("message number {i:06} with some text")}]})
                })
                .collect(),
        )
    }

    #[test]
    fn linear_truncation_matches_the_reference_algorithm() {
        // R-CONTENT-1 (equality on a smaller input).
        for (n, budget) in [(300, 4096), (50, 700), (10, 64), (1, 30), (5, 1)] {
            let v = messages(n);
            let fast = truncate_json(v.clone(), budget).map(|b| {
                assert_eq!(b.json, to_json(&b.value));
                b.value
            });
            assert_eq!(fast, reference_truncate(v, budget), "n={n} budget={budget}");
        }
    }

    #[test]
    fn twenty_thousand_messages_truncate_quickly() {
        // R-CONTENT-1: 20 000 messages of ~70 bytes with a raised traversal
        // budget finish within the time bound (1 s optimized; debug builds
        // get a looser bound).
        let policy = ContentPolicy::default()
            .with_capture(ContentCapture::SpanOnly)
            .with_max_traversal_bytes(64 * 1024 * 1024);
        let v = messages(20_000);
        let started = std::time::Instant::now();
        let bounded = keep(policy.prepare_json("gen_ai.input.messages", &v));
        let elapsed = started.elapsed();
        assert!(bounded.truncated);
        assert!(bounded.json.len() <= DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES);
        let limit = if cfg!(debug_assertions) { 15.0 } else { 1.0 };
        assert!(elapsed.as_secs_f64() < limit, "took {elapsed:?}");
    }

    #[test]
    fn capture_parsing_defaults_to_no_content_on_garbage() {
        assert_eq!(ContentCapture::parse("SPAN_ONLY"), ContentCapture::SpanOnly);
        assert_eq!(
            ContentCapture::parse(" Span_And_Event "),
            ContentCapture::SpanAndEvent
        );
        assert_eq!(ContentCapture::parse("yes"), ContentCapture::NoContent);
    }

    /// Variable name and value (`None` = not valid UTF-8).
    type Env = &'static [(&'static str, Option<&'static str>)];

    fn vars(pairs: Env) -> impl Fn(&str) -> std::result::Result<String, VarError> {
        move |name| match pairs.iter().find(|(k, _)| *k == name) {
            Some((_, Some(v))) => Ok((*v).to_owned()),
            Some((_, None)) => Err(VarError::NotUnicode(std::ffi::OsString::from("x"))),
            None => Err(VarError::NotPresent),
        }
    }

    #[test]
    fn capture_resolution_follows_the_contract() {
        // R-CFG-7
        use ContentCapture::*;
        let cases: [(Env, Option<ContentCapture>); 10] = [
            (&[], None),
            (
                &[(CAPTURE_CONTENT_ENV, Some(" EVENT_ONLY "))],
                Some(EventOnly),
            ),
            (&[(CAPTURE_CONTENT_ENV, Some("bogus"))], Some(NoContent)),
            (&[(CAPTURE_CONTENT_ENV, Some(""))], None),
            (&[(CAPTURE_CONTENT_ENV, None)], Some(NoContent)),
            (
                &[(CAPTURE_CONTENT_LEGACY_ENV, Some("true"))],
                Some(SpanOnly),
            ),
            (
                &[(CAPTURE_CONTENT_LEGACY_ENV, Some("FALSE"))],
                Some(NoContent),
            ),
            (
                &[(CAPTURE_CONTENT_LEGACY_ENV, Some("span_and_event"))],
                Some(SpanAndEvent),
            ),
            (
                &[(CAPTURE_CONTENT_LEGACY_ENV, Some("maybe"))],
                Some(NoContent),
            ),
            (
                &[
                    (CAPTURE_CONTENT_ENV, Some("span_only")),
                    (CAPTURE_CONTENT_LEGACY_ENV, Some("false")),
                ],
                Some(SpanOnly),
            ),
        ];
        for (env, expected) in cases {
            assert_eq!(ContentCapture::from_vars(vars(env)), expected, "{env:?}");
        }
        // A default policy with only a redactor honours the environment; an
        // explicit capture ignores it.
        let env = vars(&[(CAPTURE_CONTENT_ENV, Some("no_content"))]);
        let policy = ContentPolicy::default()
            .with_redactor(SubstringRedactor::new(["x"]))
            .resolved_with(&env);
        assert_eq!(policy.capture(), NoContent);
        assert_eq!(policy.explicit_capture(), None);
        let explicit = ContentPolicy::default()
            .with_capture(SpanOnly)
            .resolved_with(&env);
        assert_eq!(explicit.capture(), SpanOnly);
        let unreadable =
            ContentPolicy::default().resolved_with(vars(&[(CAPTURE_CONTENT_ENV, None)]));
        assert_eq!(unreadable.capture(), NoContent);
        assert_eq!(
            ContentPolicy::default().resolved_with(vars(&[])).capture(),
            SpanOnly
        );
    }

    #[test]
    fn budgets_are_configurable_and_default_to_the_catalog() {
        // R-BUF-3
        let p = ContentPolicy::default();
        assert_eq!(p.max_attribute_bytes(), DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES);
        assert_eq!(
            p.max_retained_operation_bytes(),
            DEFAULT_MAX_RETAINED_OPERATION_BYTES
        );
        assert_eq!(p.max_traversal_bytes(), DEFAULT_MAX_TRAVERSAL_BYTES);
        let p = p
            .with_max_attribute_bytes(0)
            .with_max_retained_operation_bytes(0)
            .with_max_traversal_bytes(7);
        assert_eq!(p.max_attribute_bytes(), 1);
        assert_eq!(p.max_retained_operation_bytes(), 1);
        assert_eq!(p.max_traversal_bytes(), 7);
        // No 64-byte floor: B = 10 bounds a plain string to 10 bytes.
        let p = ContentPolicy::default().with_max_attribute_bytes(10);
        let t = keep(p.prepare_text("k", &"a".repeat(20), p.text_bound()));
        assert_eq!(t.text, format!("{}{TRUNCATION_MARKER}", "a".repeat(10)));
        assert!(t.truncated);
        // The traversal budget rejects values whose preflight cost exceeds it.
        let p = ContentPolicy::default().with_max_traversal_bytes(16 + 3);
        assert!(matches!(
            p.prepare_text("k", "abc", p.text_bound()),
            Outcome::Keep(_)
        ));
        assert!(matches!(
            p.prepare_text("k", "abcd", p.text_bound()),
            Outcome::Failed
        ));
    }

    #[test]
    fn substring_redactor_rewrites_nested_strings() {
        let r = SubstringRedactor::new(["s3cret"]);
        let v = r
            .redact("k", json!({"a":["x s3cret y"],"b":{"c":"s3cret"}}))
            .unwrap();
        assert_eq!(v, json!({"a":["x [REDACTED] y"],"b":{"c":"[REDACTED]"}}));
    }

    struct Flaky;

    impl Redactor for Flaky {
        fn redact(&self, _kind: &str, value: Value) -> Option<Value> {
            Some(value)
        }

        fn try_redact(
            &self,
            kind: &str,
            value: Value,
        ) -> std::result::Result<Option<Value>, RedactionError> {
            match kind {
                "fail" => Err(RedactionError),
                "drop" => Ok(None),
                "null" => Ok(Some(Value::Null)),
                _ => Ok(Some(value)),
            }
        }
    }

    #[test]
    fn redactor_errors_fail_closed_without_affecting_later_values() {
        // R-CONTENT-2
        let p = ContentPolicy::default().with_redactor(Flaky);
        assert!(matches!(
            p.prepare_json("fail", &json!({"a": 1})),
            Outcome::Failed
        ));
        let next = keep(p.prepare_json("ok", &json!({"a": 1})));
        assert_eq!(next.value, json!({"a": 1}));
        assert!(matches!(
            p.prepare_json("drop", &json!(1)),
            Outcome::Dropped
        ));
        // JSON null is valid for JSON kinds and invalid for string kinds.
        assert_eq!(keep(p.prepare_json("null", &json!(1))).json, "null");
        assert!(matches!(
            p.prepare_text("null", "x", p.text_bound()),
            Outcome::Failed
        ));
        assert!(matches!(
            p.prepare_text("fail", "x", p.text_bound()),
            Outcome::Failed
        ));
        assert!(matches!(
            p.prepare_text("drop", "x", p.text_bound()),
            Outcome::Dropped
        ));
        assert_eq!(keep(p.prepare_text("ok", "x", p.text_bound())).text, "x");
    }

    #[test]
    fn redactor_output_is_validated() {
        let deep = |_: &str, _: Value| {
            let mut v = json!("x");
            for _ in 0..129 {
                v = Value::Array(vec![v]);
            }
            Some(v)
        };
        let p = ContentPolicy::default().with_redactor(deep);
        assert!(matches!(p.prepare_json("k", &json!(1)), Outcome::Failed));
        let big = |_: &str, _: Value| Some(json!("x".repeat(100)));
        let p = ContentPolicy::default()
            .with_redactor(big)
            .with_max_traversal_bytes(64);
        assert!(matches!(p.prepare_json("k", &json!(1)), Outcome::Failed));
    }

    #[test]
    fn rules_validate_and_round_trip() {
        // R-CONTENT-4
        assert!(RedactionRule::replace_literal("", "x").is_err());
        assert!(RedactionRule::omit_attribute("").is_err());
        let rules = vec![
            RedactionRule::replace_literal("sk-", "[KEY]").unwrap(),
            RedactionRule::omit_attribute("gen_ai.input.messages").unwrap(),
        ];
        let json = serde_json::to_value(&rules).unwrap();
        assert_eq!(
            json,
            json!([
                {"type": "replace_literal", "find": "sk-", "replace": "[KEY]"},
                {"type": "omit_attribute", "key": "gen_ai.input.messages"}
            ])
        );
        let back: Vec<RedactionRule> = serde_json::from_value(json).unwrap();
        assert_eq!(back, rules);
        for bad in [
            json!({"type": "replace_literal", "find": "", "replace": "x"}),
            json!({"type": "omit_attribute", "key": ""}),
            json!({"type": "regex", "pattern": "x"}),
            json!({"type": "replace_literal", "find": 1, "replace": "x"}),
            json!({"type": "omit_attribute"}),
        ] {
            assert!(serde_json::from_value::<RedactionRule>(bad).is_err());
        }
    }

    #[test]
    fn rules_run_after_the_hook_in_order_and_preserve_structure() {
        let p = ContentPolicy::default()
            .with_redactor(SubstringRedactor::new(["secret"]))
            .with_rules([
                RedactionRule::replace_literal("[REDACTED]", "<r>").unwrap(),
                RedactionRule::replace_literal("a", "b").unwrap(),
                RedactionRule::replace_literal("b", "c").unwrap(),
                RedactionRule::omit_attribute("gone").unwrap(),
            ]);
        let v = keep(p.prepare_json(
            "k",
            &json!({"a": "secret a", "n": 1, "t": true, "z": null, "l": ["ab"]}),
        ));
        assert_eq!(
            v.value,
            json!({"a": "<r> c", "n": 1, "t": true, "z": null, "l": ["cc"]})
        );
        assert!(matches!(
            p.prepare_json("gone", &json!(1)),
            Outcome::Dropped
        ));
        assert!(matches!(
            p.prepare_text("gone", "x", p.text_bound()),
            Outcome::Dropped
        ));
        let expand = ContentPolicy::default()
            .with_max_traversal_bytes(100)
            .with_rule(RedactionRule::replace_literal("x", "#".repeat(50)).unwrap());
        assert!(matches!(
            expand.prepare_json("k", &json!(["xx"])),
            Outcome::Failed
        ));
        assert!(matches!(
            expand.prepare_json("k", &json!(["x"])),
            Outcome::Keep(_)
        ));
    }
}
