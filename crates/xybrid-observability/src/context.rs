//! Portable trace context: W3C `traceparent`/`tracestate` and observation
//! references that cross threads, processes, FFI, and language boundaries.
//!
//! Parsing follows W3C Trace Context level 1 as pinned by CONTRACT.md
//! ("ObservationContext and W3C Trace Context"): flags keep only the sampled
//! bit and `tracestate` is normalized or discarded as a whole.

use std::fmt;
use std::str::FromStr;

use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use opentelemetry::Context;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, Result};
use crate::semconv::{MAX_TRACESTATE_CHARS, MAX_TRACESTATE_MEMBERS};

/// Length of a version-00 `traceparent`.
const TRACEPARENT_LEN: usize = 55;

/// A portable W3C trace context.
///
/// Capture it from an operation with `context()` before a thread, task,
/// worker, FFI, or network hop, and pass it as the explicit parent on the
/// other side. A value of this type is valid by construction; its serialized
/// form is `{"traceparent": "...", "tracestate": "..."}`.
#[derive(Clone, PartialEq, Eq)]
pub struct ObservationContext {
    trace_id: [u8; 16],
    span_id: [u8; 8],
    sampled: bool,
    trace_state: Option<String>,
    is_remote: bool,
}

impl ObservationContext {
    /// Parses a W3C `traceparent` header and optional `tracestate`.
    ///
    /// Leading and trailing spaces and tabs are trimmed. Flags keep only the
    /// sampled bit. The `tracestate` is normalized; an invalid one is
    /// discarded (the `traceparent` stays valid), as the specification
    /// requires.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTraceparent`] when the header does not follow
    /// W3C Trace Context level 1 (lowercase hex, version `00` of exactly 55
    /// characters, later versions extended only after a `-`, non-zero ids).
    pub fn from_traceparent(traceparent: &str, tracestate: Option<&str>) -> Result<Self> {
        let (trace_id, span_id, flags) = parse_traceparent(trim_ows(traceparent))
            .ok_or_else(|| Error::InvalidTraceparent(bounded(traceparent)))?;
        Ok(Self {
            trace_id,
            span_id,
            sampled: flags & 0x01 == 0x01,
            trace_state: tracestate.and_then(normalize_tracestate),
            is_remote: true,
        })
    }

    /// Formats this context as a `traceparent` header value (version `00`,
    /// flags `00` or `01`).
    #[must_use]
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            hex(&self.trace_id),
            hex(&self.span_id),
            u8::from(self.sampled)
        )
    }

    /// The normalized `tracestate` header value, if any.
    #[must_use]
    pub fn tracestate(&self) -> Option<&str> {
        self.trace_state.as_deref()
    }

    /// Lowercase hex trace id.
    #[must_use]
    pub fn trace_id(&self) -> String {
        hex(&self.trace_id)
    }

    /// Lowercase hex span id.
    #[must_use]
    pub fn span_id(&self) -> String {
        hex(&self.span_id)
    }

    /// Whether the sampled flag is set.
    #[must_use]
    pub fn is_sampled(&self) -> bool {
        self.sampled
    }

    /// The reference (`trace_id`, `span_id`) of the span this context points at.
    #[must_use]
    pub fn observation_ref(&self) -> ObservationRef {
        ObservationRef {
            trace_id: self.trace_id(),
            span_id: self.span_id(),
        }
    }

    /// Extracts the active span's context from an OpenTelemetry [`Context`].
    /// Flags are masked to the sampled bit and the trace state is normalized.
    #[must_use]
    pub fn from_otel(cx: &Context) -> Option<Self> {
        let span = cx.span();
        Self::from_span_context(span.span_context())
    }

    pub(crate) fn from_span_context(sc: &SpanContext) -> Option<Self> {
        if !sc.is_valid() {
            return None;
        }
        Some(Self {
            trace_id: sc.trace_id().to_bytes(),
            span_id: sc.span_id().to_bytes(),
            sampled: sc.is_sampled(),
            trace_state: normalize_tracestate(&sc.trace_state().header()),
            is_remote: sc.is_remote(),
        })
    }

    /// The ambient context of the current thread, if a span is active.
    #[must_use]
    pub fn current() -> Option<Self> {
        Self::from_otel(&Context::current())
    }

    /// Converts to an OpenTelemetry [`Context`] carrying this span context as
    /// the parent.
    #[must_use]
    pub fn to_otel(&self) -> Context {
        Context::new().with_remote_span_context(self.span_context())
    }

    pub(crate) fn span_context(&self) -> SpanContext {
        SpanContext::new(
            TraceId::from_bytes(self.trace_id),
            SpanId::from_bytes(self.span_id),
            if self.sampled {
                TraceFlags::SAMPLED
            } else {
                TraceFlags::default()
            },
            self.is_remote,
            self.trace_state
                .as_deref()
                .map(otel_trace_state)
                .unwrap_or_default(),
        )
    }
}

/// Builds an OpenTelemetry trace state from a normalized `tracestate`, member
/// by member, skipping only the members the OpenTelemetry type refuses
/// (opentelemetry 0.33 rejects multi-tenant keys whose system id has 14
/// characters). Members keep their order.
pub(crate) fn otel_trace_state(normalized: &str) -> TraceState {
    let mut state = TraceState::default();
    // `TraceState::insert` prepends, so insert in reverse to keep the order.
    for member in normalized.rsplit(',') {
        if let Some((key, value)) = member.split_once('=') {
            if let Ok(next) = state.insert(key, value) {
                state = next;
            }
        }
    }
    state
}

impl fmt::Debug for ObservationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObservationContext")
            .field("traceparent", &self.to_traceparent())
            .field("tracestate", &self.trace_state)
            .field("is_remote", &self.is_remote)
            .finish()
    }
}

impl fmt::Display for ObservationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_traceparent())
    }
}

impl FromStr for ObservationContext {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_traceparent(s, None)
    }
}

#[derive(Serialize, Deserialize)]
struct SerializedContext<T> {
    traceparent: T,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tracestate: Option<T>,
}

impl Serialize for ObservationContext {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        SerializedContext {
            traceparent: self.to_traceparent(),
            tracestate: self.trace_state.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ObservationContext {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let raw = SerializedContext::<String>::deserialize(deserializer)?;
        Self::from_traceparent(&raw.traceparent, raw.tracestate.as_deref())
            .map_err(|_| serde::de::Error::custom("invalid traceparent"))
    }
}

/// Identifies a recorded span: what results, feedback, and evaluations link to.
///
/// The JSON form is exactly `{"trace_id": "<32 lowercase hex>", "span_id":
/// "<16 lowercase hex>"}`; neither id may be all zeros. Deserialization
/// validates (camelCase aliases, uppercase hex and zero ids are rejected;
/// extra members are ignored).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawRef")]
pub struct ObservationRef {
    /// Lowercase 32-hex trace id.
    pub trace_id: String,
    /// Lowercase 16-hex span id.
    pub span_id: String,
}

#[derive(Deserialize)]
struct RawRef {
    trace_id: String,
    span_id: String,
}

impl TryFrom<RawRef> for ObservationRef {
    type Error = Error;

    fn try_from(raw: RawRef) -> Result<Self> {
        Self::new(raw.trace_id, raw.span_id)
    }
}

impl ObservationRef {
    /// Builds a reference from hex ids.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidObservationRef`] unless both ids are lowercase
    /// hex of the right length and non-zero.
    pub fn new(trace_id: impl Into<String>, span_id: impl Into<String>) -> Result<Self> {
        let trace_id = trace_id.into();
        let span_id = span_id.into();
        if parse_hex::<16>(trace_id.as_bytes()).is_none()
            || parse_hex::<8>(span_id.as_bytes()).is_none()
        {
            return Err(Error::InvalidObservationRef);
        }
        Ok(Self { trace_id, span_id })
    }

    /// The parsed ids, `None` when the (public) fields were set to invalid
    /// values after construction.
    pub(crate) fn ids(&self) -> Option<(TraceId, SpanId)> {
        Some((
            TraceId::from_bytes(parse_hex::<16>(self.trace_id.as_bytes())?),
            SpanId::from_bytes(parse_hex::<8>(self.span_id.as_bytes())?),
        ))
    }
}

impl fmt::Display for ObservationRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.trace_id, self.span_id)
    }
}

/// Normalizes a W3C `tracestate` value.
///
/// Members are split on `,`, trimmed of spaces and tabs, and empty members
/// are skipped. Each remaining member must be a valid `key=value`
/// (simple or multi-tenant key; 1–256 printable ASCII value characters
/// without `,`, `=` or a trailing space). Any invalid member, a duplicate
/// key, more than 32 members, or a normalized length above 512 characters
/// discards the whole value. Returns the members joined with `,` in input
/// order, or `None`.
#[must_use]
pub fn normalize_tracestate(value: &str) -> Option<String> {
    let mut members: Vec<(&str, &str)> = Vec::new();
    for member in value.split(',') {
        let member = trim_ows(member);
        if member.is_empty() {
            continue;
        }
        if members.len() == MAX_TRACESTATE_MEMBERS {
            return None;
        }
        let (key, val) = member.split_once('=')?;
        if !valid_tracestate_key(key) || !valid_tracestate_value(val) {
            return None;
        }
        if members.iter().any(|(k, _)| *k == key) {
            return None;
        }
        members.push((key, val));
    }
    if members.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(value.len());
    for (i, (k, v)) in members.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(k);
        out.push('=');
        out.push_str(v);
    }
    (out.len() <= MAX_TRACESTATE_CHARS).then_some(out)
}

fn key_char(b: u8) -> bool {
    b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'-' | b'*' | b'/')
}

fn valid_tracestate_key(key: &str) -> bool {
    let bytes = key.as_bytes();
    match key.split_once('@') {
        None => {
            (1..=256).contains(&bytes.len())
                && bytes[0].is_ascii_lowercase()
                && bytes[1..].iter().all(|b| key_char(*b))
        }
        Some((tenant, system)) => {
            let (t, s) = (tenant.as_bytes(), system.as_bytes());
            (1..=241).contains(&t.len())
                && (t[0].is_ascii_lowercase() || t[0].is_ascii_digit())
                && t[1..].iter().all(|b| key_char(*b))
                && (1..=14).contains(&s.len())
                && s[0].is_ascii_lowercase()
                && s[1..].iter().all(|b| key_char(*b))
        }
    }
}

fn valid_tracestate_value(value: &str) -> bool {
    let bytes = value.as_bytes();
    (1..=256).contains(&bytes.len())
        && bytes
            .iter()
            .all(|b| (0x20..=0x7e).contains(b) && *b != b',' && *b != b'=')
        && bytes.last() != Some(&b' ')
}

/// Trims W3C optional whitespace (spaces and horizontal tabs) only.
fn trim_ows(s: &str) -> &str {
    s.trim_matches([' ', '\t'])
}

fn parse_traceparent(value: &str) -> Option<([u8; 16], [u8; 8], u8)> {
    let b = value.as_bytes();
    if b.len() < TRACEPARENT_LEN {
        return None;
    }
    let [version] = parse_hex_any::<1>(&b[0..2])?;
    if version == 0xff {
        return None;
    }
    if b[2] != b'-' || b[35] != b'-' || b[52] != b'-' {
        return None;
    }
    // Version 00 is exactly 55 characters; later versions may append fields
    // after a `-`.
    match (version, b.len()) {
        (_, TRACEPARENT_LEN) => {}
        (0, _) => return None,
        _ if b[TRACEPARENT_LEN] == b'-' => {}
        _ => return None,
    }
    let trace_id = parse_hex::<16>(&b[3..35])?;
    let span_id = parse_hex::<8>(&b[36..52])?;
    let [flags] = parse_hex_any::<1>(&b[53..55])?;
    Some((trace_id, span_id, flags))
}

/// Parses exactly `N` bytes of lowercase hex; all-zero ids are invalid.
fn parse_hex<const N: usize>(s: &[u8]) -> Option<[u8; N]> {
    let out = parse_hex_any::<N>(s)?;
    (!out.iter().all(|b| *b == 0)).then_some(out)
}

/// Parses exactly `N` bytes of lowercase hex.
fn parse_hex_any<const N: usize>(s: &[u8]) -> Option<[u8; N]> {
    if s.len() != N * 2 {
        return None;
    }
    let mut out = [0u8; N];
    for (i, [hi, lo]) in s.as_chunks::<2>().0.iter().enumerate() {
        out[i] = (nibble(*hi)? << 4) | nibble(*lo)?;
    }
    Some(out)
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn bounded(s: &str) -> String {
    s.chars().take(64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TP: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn round_trips_traceparent_and_tracestate() {
        let cx = ObservationContext::from_traceparent(TP, Some("vendor=abc")).unwrap();
        assert_eq!(cx.to_traceparent(), TP);
        assert_eq!(cx.tracestate(), Some("vendor=abc"));
        assert!(cx.is_sampled());
        let back = ObservationContext::from_otel(&cx.to_otel()).unwrap();
        assert_eq!(back, cx);
    }

    #[test]
    fn rejects_invalid_traceparents() {
        for bad in [
            "",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
            "0-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01future",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-0A",
            "00_4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",
            // Non-ASCII in place of hex never panics on slicing.
            "00-4bf92f3577b34da6a3ce929d0e0e47é-00f067aa0ba902b7-01",
            "\n00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        ] {
            assert!(
                ObservationContext::from_traceparent(bad, None).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn future_versions_may_carry_extra_fields_and_flags_are_masked() {
        let cx = ObservationContext::from_traceparent(
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-03-future",
            None,
        )
        .unwrap();
        assert_eq!(cx.trace_id(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(cx.to_traceparent(), TP);
        let cx = ObservationContext::from_traceparent(
            " \t00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-fe\t",
            None,
        )
        .unwrap();
        assert!(!cx.is_sampled());
        assert!(cx.to_traceparent().ends_with("-00"));
    }

    #[test]
    fn tracestate_normalization() {
        // R-CTX-1
        let cases: [(&str, Option<&str>); 16] = [
            (" a=1 ,, b=2 ,\t", Some("a=1,b=2")),
            ("a=b c", Some("a=b c")),
            ("t@abcdefghijklmn=1", Some("t@abcdefghijklmn=1")),
            ("t@abcdefghijklmno=1", None),
            ("1a@vendor=b", Some("1a@vendor=b")),
            ("1a=b", None),
            ("A=1", None),
            ("a=1,a=2", None),
            ("a=", None),
            ("=b", None),
            ("a==b", None),
            ("a=\u{e9}", None),
            ("a=b ", Some("a=b")),
            ("novalue", None),
            ("", None),
            (" , ", None),
        ];
        for (input, expected) in cases {
            assert_eq!(
                normalize_tracestate(input).as_deref(),
                expected,
                "{input:?}"
            );
        }
        let members: Vec<String> = (0..32).map(|i| format!("k{i}=v")).collect();
        assert!(normalize_tracestate(&members.join(",")).is_some());
        let mut over = members.clone();
        over.push("k32=v".into());
        assert!(normalize_tracestate(&over.join(",")).is_none());
        let long = ["a", "b", "c"].map(|k| format!("{k}={}", "x".repeat(200)));
        assert!(normalize_tracestate(&long.join(",")).is_none());
        assert_eq!(
            normalize_tracestate(&format!("a=1{}b=2", ",".repeat(35))).as_deref(),
            Some("a=1,b=2")
        );
    }

    #[test]
    fn span_trace_state_skips_only_members_opentelemetry_refuses() {
        // R-CTX-1: opentelemetry 0.33 refuses 14-character system ids.
        let cx =
            ObservationContext::from_traceparent(TP, Some("a=1,t@abcdefghijklmn=2,b=3")).unwrap();
        assert_eq!(cx.tracestate(), Some("a=1,t@abcdefghijklmn=2,b=3"));
        let header = cx.span_context().trace_state().header();
        assert!(header == "a=1,b=3" || header == "a=1,t@abcdefghijklmn=2,b=3");
        assert!(header.starts_with("a=1,"));
        assert!(header.ends_with("b=3"));
    }

    #[test]
    fn observation_ref_validates_ids_and_serde() {
        // R-CTX-4
        assert!(
            ObservationRef::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").is_ok()
        );
        assert!(ObservationRef::new("xyz", "00f067aa0ba902b7").is_err());
        assert!(
            ObservationRef::new("00000000000000000000000000000000", "00f067aa0ba902b7").is_err()
        );
        let ok: ObservationRef = serde_json::from_value(serde_json::json!({
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "span_id": "00f067aa0ba902b7",
            "extra": 1
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(&ok).unwrap(),
            serde_json::json!({"trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "span_id": "00f067aa0ba902b7"})
        );
        for bad in [
            serde_json::json!({"traceId": "4bf92f3577b34da6a3ce929d0e0e4736", "spanId": "00f067aa0ba902b7"}),
            serde_json::json!({"trace_id": "4BF92F3577B34DA6A3CE929D0E0E4736", "span_id": "00f067aa0ba902b7"}),
            serde_json::json!({"trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "span_id": "0000000000000000"}),
            serde_json::json!({"trace_id": " 4bf92f3577b34da6a3ce929d0e0e4736", "span_id": "00f067aa0ba902b7"}),
        ] {
            assert!(serde_json::from_value::<ObservationRef>(bad).is_err());
        }
    }

    #[test]
    fn context_serde_round_trip_and_validation() {
        // R-CTX-4
        let cx = ObservationContext::from_traceparent(TP, Some("a=1, b=2")).unwrap();
        let json = serde_json::to_value(&cx).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"traceparent": TP, "tracestate": "a=1,b=2"})
        );
        let back: ObservationContext = serde_json::from_value(json).unwrap();
        assert_eq!(back, cx);
        let bare = ObservationContext::from_traceparent(TP, None).unwrap();
        assert_eq!(
            serde_json::to_value(&bare).unwrap(),
            serde_json::json!({"traceparent": TP})
        );
        let with_null: ObservationContext =
            serde_json::from_value(serde_json::json!({"traceparent": TP, "tracestate": null}))
                .unwrap();
        assert_eq!(with_null.tracestate(), None);
        assert!(serde_json::from_value::<ObservationContext>(
            serde_json::json!({"traceparent": "00-abc"})
        )
        .is_err());
        let dropped: ObservationContext =
            serde_json::from_value(serde_json::json!({"traceparent": TP, "tracestate": "a=1,a=2"}))
                .unwrap();
        assert_eq!(dropped.tracestate(), None);
    }
}
