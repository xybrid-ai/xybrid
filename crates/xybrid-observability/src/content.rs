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
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::preflight::{self, PreflightError};
use crate::semconv::{
    span_content_ceiling, CAPTURE_CONTENT_ENV, CAPTURE_CONTENT_LEGACY_ENV,
    DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES, DEFAULT_MAX_REQUEST_BODY_BYTES,
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
///
/// A hook that panics fails closed exactly like one that returns
/// [`RedactionError`]: the value is omitted and listed as truncated, the
/// operation continues, and later values are redacted normally. The process's
/// panic hook still runs (and reports the panic as it is configured to); a
/// binary built with `panic = "abort"` aborts.
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
///
/// The content budgets are clamped so that one span's content cannot push
/// its export request past `max_request_body_bytes` (CONTRACT.md "Content
/// pipeline", budgets): the retained-operation budget is at most
/// [`crate::semconv::span_content_ceiling`] of the request bound, and the
/// attribute budget at most the retained-operation budget. Clamping never
/// fails; the getters return the effective values.
#[derive(Clone)]
pub struct ContentPolicy {
    capture: Option<ContentCapture>,
    snapshot: Option<ContentCapture>,
    /// Configured values; the getters clamp them.
    max_attribute_bytes: usize,
    max_retained_operation_bytes: usize,
    max_request_body_bytes: usize,
    max_traversal_bytes: usize,
    /// Redaction hooks in order, each on the previous one's output: none,
    /// the one [`ContentPolicy::with_redactor`] installs, or those
    /// [`ContentPolicy::capped_by`] combines (each hook once).
    redactors: Arc<[Arc<dyn Redactor>]>,
    rules: Arc<[RedactionRule]>,
}

impl Default for ContentPolicy {
    /// Product default: capture mode from the environment (else span
    /// attributes), 16 KiB per attribute, 208 KiB retained per operation
    /// (the ceiling of the default 480 KiB request bound), 480 KiB traversed
    /// per value, no redaction hook, no rules.
    fn default() -> Self {
        Self {
            capture: None,
            snapshot: None,
            max_attribute_bytes: DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES,
            max_retained_operation_bytes: DEFAULT_MAX_RETAINED_OPERATION_BYTES,
            max_request_body_bytes: DEFAULT_MAX_REQUEST_BODY_BYTES,
            max_traversal_bytes: DEFAULT_MAX_TRAVERSAL_BYTES,
            redactors: Arc::from(Vec::new()),
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

    /// Sets the per-attribute byte budget `B` (minimum 1). The effective
    /// budget is at most the effective retained-operation budget.
    #[must_use]
    pub fn with_max_attribute_bytes(mut self, bytes: usize) -> Self {
        self.max_attribute_bytes = bytes.max(1);
        self
    }

    /// Sets the budget of content retained by one operation (minimum 1). The
    /// effective budget is at most the span content ceiling of the request
    /// bound.
    #[must_use]
    pub fn with_max_retained_operation_bytes(mut self, bytes: usize) -> Self {
        self.max_retained_operation_bytes = bytes.max(1);
        self
    }

    /// Sets the `max_request_body_bytes` of the exporter that sends this
    /// policy's spans (minimum 1; default 480 KiB), which the content
    /// budgets are clamped to fit. An [`crate::otlp::ObservabilityPipeline`]
    /// with a traces signal applies its own bound when it builds an observer
    /// (the smaller one wins); set it here for providers the application
    /// owns.
    #[must_use]
    pub fn with_max_request_body_bytes(mut self, bytes: usize) -> Self {
        self.max_request_body_bytes = bytes.max(1);
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
        self.redactors = Arc::from(vec![Arc::new(redactor) as Arc<dyn Redactor>]);
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

    /// This policy, never capturing more content than `ceiling`:
    ///
    /// - content goes only to the destinations both capture modes include
    ///   (`span_only` under `event_only` captures nothing);
    /// - each byte budget and the request bound is the smaller of the two;
    /// - `ceiling`'s redaction hooks run first and this policy's hooks that
    ///   `ceiling` does not already hold run on their output, so no hook
    ///   redacts a value twice; `ceiling`'s rules apply first, followed by
    ///   this policy's rules that `ceiling` does not already have. A value
    ///   that any hook drops or fails on is dropped or fails.
    ///
    /// Capping by a policy at least as permissive changes nothing, and
    /// capping twice by the same ceiling, or again by a policy already
    /// capped by this one, is the same as capping once.
    #[must_use]
    pub fn capped_by(&self, ceiling: &ContentPolicy) -> Self {
        let (own, cap) = (self.capture(), ceiling.capture());
        let spans = own.on_spans() && cap.on_spans();
        let events = own.on_events() && cap.on_events();
        let capture = match (spans, events) {
            (true, true) => ContentCapture::SpanAndEvent,
            (true, false) => ContentCapture::SpanOnly,
            (false, true) => ContentCapture::EventOnly,
            (false, false) => ContentCapture::NoContent,
        };
        let mut redactors = ceiling.redactors.to_vec();
        redactors.extend(
            self.redactors
                .iter()
                .filter(|own| !ceiling.redactors.iter().any(|hook| Arc::ptr_eq(hook, own)))
                .cloned(),
        );
        let mut rules = ceiling.rules.to_vec();
        rules.extend(
            self.rules
                .iter()
                .filter(|rule| !ceiling.rules.contains(rule))
                .cloned(),
        );
        Self {
            capture: Some(capture),
            snapshot: None,
            max_attribute_bytes: self
                .max_attribute_bytes()
                .min(ceiling.max_attribute_bytes()),
            max_retained_operation_bytes: self
                .max_retained_operation_bytes()
                .min(ceiling.max_retained_operation_bytes()),
            max_request_body_bytes: self
                .max_request_body_bytes
                .min(ceiling.max_request_body_bytes),
            max_traversal_bytes: self.max_traversal_bytes.min(ceiling.max_traversal_bytes),
            redactors: Arc::from(redactors),
            rules: Arc::from(rules),
        }
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

    /// The effective per-attribute byte budget: the configured value, at
    /// most [`ContentPolicy::max_retained_operation_bytes`].
    #[must_use]
    pub fn max_attribute_bytes(&self) -> usize {
        self.max_attribute_bytes
            .min(self.max_retained_operation_bytes())
    }

    /// The effective retained-operation byte budget: the configured value, at
    /// most the span content ceiling of
    /// [`ContentPolicy::max_request_body_bytes`].
    #[must_use]
    pub fn max_retained_operation_bytes(&self) -> usize {
        self.max_retained_operation_bytes
            .min(span_content_ceiling(self.max_request_body_bytes))
    }

    /// The request body bound the content budgets fit.
    #[must_use]
    pub fn max_request_body_bytes(&self) -> usize {
        self.max_request_body_bytes
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
        match truncate_json(value, self.max_attribute_bytes()) {
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
        TextBound::Bytes(self.max_attribute_bytes())
    }

    /// Calls the hooks (if any) in order on the clone and validates the last
    /// one's output. Returns the value to continue with and its preflight
    /// cost, or `None` to drop.
    fn call_redactor(
        &self,
        kind: &str,
        clone: Value,
        cost: u64,
    ) -> std::result::Result<Option<(Value, u64)>, Failure> {
        if self.redactors.is_empty() {
            return Ok(Some((clone, cost)));
        }
        // A panicking hook fails closed like one that returns an error (as a
        // throwing JS hook does). Unwind safety: the hooks own the value they
        // are handed and borrow nothing of ours mutably, and the panic stops
        // here, before it can unwind through a lock a caller holds (an
        // operation's end redacts its error message under the operation's
        // state lock).
        let redacted = std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.redactors
                .iter()
                .try_fold(Some(clone), |value, redactor| match value {
                    Some(value) => redactor.try_redact(kind, value),
                    None => Ok(None),
                })
        }));
        match redacted.unwrap_or(Err(RedactionError)) {
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
    /// string alone outgrows it while its rules are applied (amendment A1).
    ///
    /// Both conditions depend only on each string's own rule results and on
    /// the final cost, so the outcome never depends on the order strings are
    /// visited. The walk also keeps a lower bound of the final cost
    /// (amendment A9): the structural cost (every node, key and scalar, plus
    /// 16 per string), plus the final length of each string already
    /// rewritten, plus the least length each other string can end with (its
    /// current length when no rule shrinks text, else zero). A rule rebuilds a
    /// string only while that bound, with the string at the least length it
    /// can end with after the rule, fits the budget, so nothing more is built
    /// once the strings already rewritten prove the value over budget. The
    /// bound never decreases and equals the final cost after the last string,
    /// so failing on it never changes an outcome.
    fn apply_replacements(&self, value: &mut Value, cost: u64) -> std::result::Result<(), Failure> {
        let rules: Vec<(&str, &str)> = self
            .rules
            .iter()
            .filter_map(|r| match r {
                RedactionRule::ReplaceLiteral { find, replace } => {
                    Some((find.as_str(), replace.as_str()))
                }
                RedactionRule::OmitAttribute { .. } => None,
            })
            .collect();
        if rules.is_empty() {
            return Ok(());
        }
        let shrinks = rules
            .iter()
            .any(|(find, replace)| replace.len() < find.len());
        // The least final length of a string that is `len` bytes long now.
        let least = |len: usize| if shrinks { 0 } else { len as i128 };
        let per_string = self.traversal();
        let budget = i128::from(per_string);
        // Lower bound of the final cost before any string is rewritten.
        let mut bound = i128::from(cost);
        if shrinks {
            bound -= string_bytes(value) as i128;
        }
        let mut overflow = false;
        map_strings(value, &mut |s| {
            if overflow {
                return;
            }
            // This string's share of the bound is recomputed from its result.
            bound -= least(s.len());
            for &(find, replace) in &rules {
                // Amendment A1 (b): the string's cost after this rule is
                // computed from the match count before the new string is
                // built, so an expanding rule never allocates beyond the
                // budget for one string.
                let matches = s.matches(find).count();
                if matches == 0 {
                    continue;
                }
                let grown = matches
                    .checked_mul(replace.len())
                    .and_then(|added| (s.len() - matches * find.len()).checked_add(added));
                match grown {
                    Some(len)
                        if preflight_len_cost(len) <= per_string
                            && bound + least(len) <= budget =>
                    {
                        *s = s.replace(find, replace);
                    }
                    _ => {
                        overflow = true;
                        return;
                    }
                }
            }
            bound += s.len() as i128;
        });
        // After the last string the bound is the final cost.
        if overflow || bound > budget {
            return Err(Failure::RulesBudget);
        }
        Ok(())
    }
}

/// UTF-8 length of `s` written inside a JSON string (quotes excluded): `"`,
/// `\` and the short control escapes take two bytes, other control
/// characters six (`\u00XX`); everything else is kept as is. Both packages'
/// JSON writers escape exactly these.
pub(crate) fn json_escaped_len(s: &str) -> usize {
    s.len()
        + s.bytes()
            .map(|b| match b {
                b'"' | b'\\' | b'\n' | b'\r' | b'\t' | 0x08 | 0x0c => 1,
                0x00..=0x1f => 5,
                _ => 0,
            })
            .sum::<usize>()
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
            .field("max_attribute_bytes", &self.max_attribute_bytes())
            .field(
                "max_retained_operation_bytes",
                &self.max_retained_operation_bytes(),
            )
            .field("max_request_body_bytes", &self.max_request_body_bytes)
            .field("max_traversal_bytes", &self.max_traversal_bytes)
            .field("redactors", &self.redactors.len())
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

/// Total UTF-8 bytes of every JSON string value (object keys excluded).
fn string_bytes(value: &Value) -> usize {
    match value {
        Value::String(s) => s.len(),
        Value::Array(items) => items
            .iter()
            .fold(0, |sum, v| sum.saturating_add(string_bytes(v))),
        Value::Object(map) => map
            .values()
            .fold(0, |sum, v| sum.saturating_add(string_bytes(v))),
        _ => 0,
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
    fn a_panicking_redactor_fails_closed_without_affecting_later_values() {
        // A panic is the same failure as `Err(RedactionError)`.
        let hook = |kind: &str, value: Value| {
            assert!(kind != "panics", "redactor panicked");
            Some(value)
        };
        let p = ContentPolicy::default().with_redactor(hook);
        assert!(matches!(
            p.prepare_json("panics", &json!({"a": 1})),
            Outcome::Failed
        ));
        assert!(matches!(
            p.prepare_text("panics", "x", p.text_bound()),
            Outcome::Failed
        ));
        assert_eq!(
            keep(p.prepare_json("ok", &json!({"a": 1}))).value,
            json!({"a": 1})
        );
        assert_eq!(keep(p.prepare_text("ok", "x", p.text_bound())).text, "x");
    }

    #[test]
    fn budgets_are_clamped_to_fit_one_span_in_the_request_bound() {
        // Retained <= (request - allowance) / growth, attribute
        // <= retained; clamping never fails and the getters are effective.
        use crate::semconv::{
            span_content_ceiling, SPAN_CONTENT_ENCODING_GROWTH, SPAN_ENVELOPE_ALLOWANCE_BYTES,
        };
        let ceiling = (DEFAULT_MAX_REQUEST_BODY_BYTES - SPAN_ENVELOPE_ALLOWANCE_BYTES)
            / SPAN_CONTENT_ENCODING_GROWTH;
        assert_eq!(ceiling, 212_992);
        assert_eq!(
            span_content_ceiling(DEFAULT_MAX_REQUEST_BODY_BYTES),
            ceiling
        );
        // A bound at or below the allowance leaves one byte.
        assert_eq!(span_content_ceiling(SPAN_ENVELOPE_ALLOWANCE_BYTES), 1);
        assert_eq!(span_content_ceiling(1), 1);

        let raised = ContentPolicy::default()
            .with_max_attribute_bytes(400_000)
            .with_max_retained_operation_bytes(1_000_000);
        assert_eq!(raised.max_retained_operation_bytes(), ceiling);
        assert_eq!(raised.max_attribute_bytes(), ceiling);
        // The attribute budget follows a lower retained budget, in either
        // order of the builder calls.
        let lower = ContentPolicy::default()
            .with_max_retained_operation_bytes(1000)
            .with_max_attribute_bytes(5000);
        assert_eq!(lower.max_attribute_bytes(), 1000);
        let lower = ContentPolicy::default()
            .with_max_attribute_bytes(5000)
            .with_max_retained_operation_bytes(1000);
        assert_eq!(lower.max_attribute_bytes(), 1000);
        // A smaller request bound lowers both; values under it are kept.
        let bound = SPAN_ENVELOPE_ALLOWANCE_BYTES + 2 * 10_000;
        let small = raised.clone().with_max_request_body_bytes(bound);
        assert_eq!(small.max_request_body_bytes(), bound);
        assert_eq!(small.max_retained_operation_bytes(), 10_000);
        assert_eq!(small.max_attribute_bytes(), 10_000);
        let kept = ContentPolicy::default()
            .with_max_attribute_bytes(64)
            .with_max_request_body_bytes(bound);
        assert_eq!(kept.max_attribute_bytes(), 64);
        // Truncation uses the effective budget.
        let t = keep(small.prepare_text("k", &"a".repeat(20_000), small.text_bound()));
        assert!(t.truncated);
        assert_eq!(t.text.len(), 10_000 + TRUNCATION_MARKER.len());
        let j = keep(small.prepare_json("k", &json!(["b".repeat(20_000)])));
        assert!(j.truncated && j.json.len() <= 10_000, "{}", j.json.len());
    }

    #[test]
    fn json_escaped_len_counts_what_json_writers_escape() {
        for s in [
            "plain",
            "",
            "é漢🎉",
            "\u{7f}",
            "a\"b\\c",
            "\n\r\t\u{8}\u{c}",
            "\u{0}\u{1f}",
        ] {
            let written = serde_json::to_string(s).unwrap();
            assert_eq!(json_escaped_len(s), written.len() - 2, "{s:?}");
        }
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

    /// The amendment A1 checks before the A9 early exit: every string is
    /// rewritten, then the final cost is compared once.
    fn reference_replacements(
        rules: &[RedactionRule],
        budget: u64,
        mut value: Value,
        cost: u64,
    ) -> Option<Value> {
        let mut total = i128::from(cost);
        let mut overflow = false;
        map_strings(&mut value, &mut |s| {
            if overflow {
                return;
            }
            let before = s.len();
            for rule in rules {
                let RedactionRule::ReplaceLiteral { find, replace } = rule else {
                    continue;
                };
                let matches = s.matches(find.as_str()).count();
                if matches == 0 {
                    continue;
                }
                let len = s.len() - matches * find.len() + matches * replace.len();
                if preflight_len_cost(len) > budget {
                    overflow = true;
                    return;
                }
                *s = s.replace(find.as_str(), replace);
            }
            total += s.len() as i128 - before as i128;
        });
        (!overflow && total <= i128::from(budget)).then_some(value)
    }

    fn replace(find: &str, with: &str) -> RedactionRule {
        RedactionRule::replace_literal(find, with).unwrap()
    }

    /// Runs the rules alone (no hook, no truncation) under `budget`.
    fn run_rules(rules: &[RedactionRule], budget: u64, value: &Value) -> Option<Value> {
        let policy = ContentPolicy::default()
            .with_max_traversal_bytes(budget as usize)
            .with_rules(rules.iter().cloned());
        let mut out = value.clone();
        policy
            .apply_replacements(&mut out, preflight::cost_of(value))
            .ok()
            .map(|()| out)
    }

    #[test]
    fn the_early_exit_keeps_every_outcome_of_the_final_cost_check() {
        // Amendment A9: stopping on the lower bound never changes an outcome.
        // Rule sets that only grow, only shrink, grow then shrink (the bound
        // then counts other strings as empty), and an omit rule mixed in.
        let rule_sets: Vec<Vec<RedactionRule>> = vec![
            vec![replace("a", "bbbb")],
            vec![replace("ab", "c")],
            vec![replace("a", "bbbbbbbb"), replace("bbbb", "")],
            vec![replace("b", ""), replace("a", "xyz"), replace("z", "zz")],
            vec![
                RedactionRule::omit_attribute("k").unwrap(),
                replace("c", "cc"),
                replace("q", "q"),
            ],
        ];
        // Deterministic documents: arrays and objects of strings over a small
        // alphabet, plus scalars, so every rule matches somewhere.
        let mut seed = 0x2545_f491_u64;
        let mut next = move |n: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % n
        };
        let mut outcomes = [0_usize; 2];
        for doc in 0..60 {
            let strings: Vec<Value> = (0..1 + next(12))
                .map(|_| {
                    let len = next(40) as usize;
                    let text: String = (0..len)
                        .map(|_| ['a', 'b', 'c', 'z', ' '][next(5) as usize])
                        .collect();
                    Value::String(text)
                })
                .collect();
            let value = if doc % 2 == 0 {
                json!({"n": 1, "list": strings, "nested": {"s": "abab", "t": true}})
            } else {
                Value::Array(strings)
            };
            let cost = preflight::cost_of(&value);
            for rules in &rule_sets {
                // Budgets around the final cost and the largest intermediate.
                let reach = reference_replacements(rules, u64::MAX, value.clone(), cost)
                    .map(|v| preflight::cost_of(&v))
                    .unwrap();
                let mut budgets = vec![cost, cost + 1, reach.saturating_sub(1), reach, reach + 1];
                budgets.extend((0..6).map(|_| cost + next(reach.max(cost) - cost + 64)));
                for budget in budgets.into_iter().filter(|b| *b >= cost) {
                    let expected = reference_replacements(rules, budget, value.clone(), cost);
                    assert_eq!(
                        run_rules(rules, budget, &value),
                        expected,
                        "doc {doc}, budget {budget}, rules {rules:?}"
                    );
                    outcomes[usize::from(expected.is_some())] += 1;
                }
            }
        }
        // Both outcomes are exercised, many times each.
        assert!(outcomes[0] > 200 && outcomes[1] > 200, "{outcomes:?}");
    }

    #[test]
    fn the_rules_outcome_does_not_depend_on_string_order() {
        // One string expands, another shrinks; only the final cost decides.
        let rules = [replace("x", "yyyy"), replace("zz", "")];
        let grow = json!("x".repeat(10));
        let shrink = json!("z".repeat(20));
        let forward = Value::Array(vec![grow.clone(), shrink.clone()]);
        let backward = Value::Array(vec![shrink, grow]);
        let cost = preflight::cost_of(&forward);
        // Final: 16 + (16 + 40) + (16 + 0) = 88; preflight was 16 + 26 + 36 = 78.
        assert_eq!(cost, 78);
        for budget in [78, 87, 88, 89] {
            let a = run_rules(&rules, budget, &forward);
            let b = run_rules(&rules, budget, &backward);
            assert_eq!(a.is_some(), budget >= 88, "budget {budget}");
            assert_eq!(a.is_some(), b.is_some(), "budget {budget}");
        }
    }

    #[test]
    fn a_document_over_budget_builds_nothing_once_its_rewritten_strings_prove_it() {
        // Amendment A9. 600 strings of 480 `a` cost 600 * 496 + 16 = 297,616,
        // under the 491,520 default; each would grow to 480,000 bytes, which
        // passes the per-string guard on its own.
        let value = Value::Array(vec![json!("a".repeat(480)); 600]);
        let policy = ContentPolicy::default().with_rule(replace("a", &"b".repeat(1000)));
        let cost = preflight::cost_of(&value);
        assert!(cost <= policy.traversal());
        let mut out = value.clone();
        assert!(policy.apply_replacements(&mut out, cost).is_err());
        // No rule shrinks text, so the bound fails before the first rewritten
        // string is built: nothing changed.
        assert_eq!(out, value, "a string was rewritten before failing closed");

        // With a rule that can shrink text the other strings may end empty,
        // so rewritten strings are built and only their final lengths count.
        // The structural cost is 297,616 - 600 * 480 = 9,616; each rewritten
        // string adds 480,000, so after the second one the bound is over
        // budget and the third is never built.
        let policy = policy.with_rule(replace("\u{0}", ""));
        let mut out = value.clone();
        assert!(policy.apply_replacements(&mut out, cost).is_err());
        let Value::Array(items) = &out else {
            unreachable!()
        };
        let rewritten = items.iter().filter(|s| **s != value[0]).count();
        assert_eq!(rewritten, 2, "a string was built after the bound failed");

        // A growth spread over many strings stops part way: each string grows
        // from 480 to 1,920 bytes, and 600 * 1,440 more bytes exceed the
        // budget; the first string whose growth would cross it is not built.
        let policy = ContentPolicy::default().with_rule(replace("a", "bbbb"));
        let mut out = value.clone();
        assert!(policy.apply_replacements(&mut out, cost).is_err());
        let Value::Array(items) = &out else {
            unreachable!()
        };
        let rewritten = items.iter().filter(|s| **s != value[0]).count();
        let room = (policy.traversal() - cost) / 1440;
        assert_eq!(rewritten as u64, room, "only strings that fit were built");
    }

    #[test]
    fn a_capped_policy_captures_only_what_both_policies_allow() {
        // A per-operation policy can only lower capture
        // below the policy the operation would otherwise run under.
        use ContentCapture::{EventOnly, NoContent, SpanAndEvent, SpanOnly};
        let modes = [NoContent, SpanOnly, EventOnly, SpanAndEvent];
        for own in modes {
            for ceiling in modes {
                let capped = ContentPolicy::default()
                    .with_capture(own)
                    .capped_by(&ContentPolicy::default().with_capture(ceiling))
                    .capture();
                assert_eq!(
                    (capped.on_spans(), capped.on_events()),
                    (
                        own.on_spans() && ceiling.on_spans(),
                        own.on_events() && ceiling.on_events()
                    ),
                    "{own:?} capped by {ceiling:?}"
                );
            }
        }
        // The environment cannot raise a capped mode: it is explicit.
        let capped = ContentPolicy::default().capped_by(&ContentPolicy::metadata_only());
        assert_eq!(capped.explicit_capture(), Some(NoContent));

        // Each budget is the smaller one, whichever side sets it.
        let small = ContentPolicy::default()
            .with_max_attribute_bytes(64)
            .with_max_retained_operation_bytes(1024)
            .with_max_traversal_bytes(2048)
            .with_max_request_body_bytes(100_000);
        let large = ContentPolicy::default()
            .with_max_attribute_bytes(100_000)
            .with_max_traversal_bytes(1_000_000);
        for capped in [large.capped_by(&small), small.capped_by(&large)] {
            assert_eq!(capped.max_attribute_bytes(), 64);
            assert_eq!(capped.max_retained_operation_bytes(), 1024);
            assert_eq!(capped.max_traversal_bytes(), 2048);
            assert_eq!(capped.max_request_body_bytes(), 100_000);
        }
    }

    #[test]
    fn a_capped_policy_redacts_with_the_ceiling_first_then_its_own() {
        let ceiling = ContentPolicy::default()
            .with_redactor(SubstringRedactor::new(["floor-secret"]))
            .with_rules([
                RedactionRule::replace_literal("floor-word", "<floor>").unwrap(),
                RedactionRule::omit_attribute("floor-key").unwrap(),
            ]);
        let own = ContentPolicy::default()
            .with_redactor(|_kind: &str, value: Value| {
                // Sees the ceiling hook's output.
                let text = value.as_str()?.replace("[REDACTED]", "<own-saw-floor>");
                Some(Value::String(text.replace("own-secret", "<own>")))
            })
            .with_rules([
                RedactionRule::replace_literal("<floor>", "<floor-then-own>").unwrap(),
                RedactionRule::omit_attribute("own-key").unwrap(),
            ]);
        let capped = own.capped_by(&ceiling);
        let text = keep(capped.prepare_text(
            "k",
            "floor-secret own-secret floor-word",
            capped.text_bound(),
        ))
        .text;
        assert_eq!(text, "<own-saw-floor> <own> <floor-then-own>");
        for key in ["floor-key", "own-key"] {
            assert!(matches!(
                capped.prepare_text(key, "x", capped.text_bound()),
                Outcome::Dropped
            ));
        }

        // Either hook failing fails the value closed.
        let failing = ContentPolicy::default().with_redactor(Flaky);
        for capped in [own.capped_by(&failing), failing.capped_by(&own)] {
            assert!(matches!(
                capped.prepare_text("fail", "x", capped.text_bound()),
                Outcome::Failed
            ));
            assert!(matches!(
                capped.prepare_text("drop", "x", capped.text_bound()),
                Outcome::Dropped
            ));
        }

        // Capping again by the same ceiling changes nothing: its hook and
        // rules are not applied twice.
        let counting = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = counting.clone();
        let ceiling = ContentPolicy::default()
            .with_redactor(move |_kind: &str, value: Value| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Some(value)
            })
            .with_rule(RedactionRule::replace_literal("a", "aa").unwrap());
        let twice = ContentPolicy::default()
            .capped_by(&ceiling)
            .capped_by(&ceiling);
        assert_eq!(twice.rules(), ceiling.rules());
        assert_eq!(
            keep(twice.prepare_text("k", "a", twice.text_bound())).text,
            "aa"
        );
        assert_eq!(counting.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn capping_again_never_runs_a_hook_twice() {
        // A nested operation re-caps a per-run policy by its parent's policy,
        // which that same per-run policy already shaped. Each hook still runs
        // once per value, in the ceiling-first order, however often the
        // policies are combined.
        let marker = |mark: &'static str| {
            move |_kind: &str, value: Value| {
                Some(Value::String(format!("{}{mark}", value.as_str()?)))
            }
        };
        let ceiling = ContentPolicy::default().with_redactor(marker("+c"));
        let own = ContentPolicy::default().with_redactor(marker("+o"));
        let once = own.capped_by(&ceiling);
        let combinations = [
            ("once", once.clone()),
            ("twice by the ceiling", once.capped_by(&ceiling)),
            ("by its own capped policy", own.capped_by(&once)),
            ("the capped policy by itself", once.capped_by(&once)),
        ];
        for (name, policy) in combinations {
            assert_eq!(
                keep(policy.prepare_text("k", "x", policy.text_bound())).text,
                "x+c+o",
                "{name}"
            );
        }
        // The other way round, the other order.
        let reversed = ceiling.capped_by(&own);
        assert_eq!(
            keep(reversed.prepare_text("k", "x", reversed.text_bound())).text,
            "x+o+c"
        );
    }
}
