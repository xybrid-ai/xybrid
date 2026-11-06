//! Portable trace context: W3C `traceparent`/`tracestate` and observation
//! references that cross threads, processes, FFI, and language boundaries.

use std::fmt;
use std::str::FromStr;

use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use opentelemetry::Context;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Maximum `tracestate` length accepted (W3C Trace Context limit).
const MAX_TRACESTATE_LEN: usize = 512;

/// A portable W3C trace context.
///
/// Capture it from an operation with `context()` before a thread, task,
/// worker, FFI, or network hop, and pass it as the explicit parent on the
/// other side.
#[derive(Clone, PartialEq, Eq)]
pub struct ObservationContext {
    trace_id: [u8; 16],
    span_id: [u8; 8],
    flags: u8,
    trace_state: Option<String>,
    is_remote: bool,
}

impl ObservationContext {
    /// Parses a W3C `traceparent` header and optional `tracestate`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTraceparent`] when the header does not follow
    /// W3C Trace Context level 1 (lowercase hex, version `00` with exactly four
    /// fields, non-zero ids). An invalid or oversized `tracestate` is
    /// discarded rather than rejected, as the specification requires.
    pub fn from_traceparent(traceparent: &str, tracestate: Option<&str>) -> Result<Self> {
        let (trace_id, span_id, flags) = parse_traceparent(traceparent.trim())
            .ok_or_else(|| Error::InvalidTraceparent(bounded(traceparent)))?;
        let trace_state = tracestate
            .map(str::trim)
            .filter(|s| !s.is_empty() && s.len() <= MAX_TRACESTATE_LEN)
            .filter(|s| TraceState::from_str(s).is_ok())
            .map(str::to_owned);
        Ok(Self {
            trace_id,
            span_id,
            flags,
            trace_state,
            is_remote: true,
        })
    }

    /// Formats this context as a `traceparent` header value.
    #[must_use]
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            hex(&self.trace_id),
            hex(&self.span_id),
            self.flags
        )
    }

    /// The `tracestate` header value, if any.
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
        self.flags & 0x01 == 0x01
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
    #[must_use]
    pub fn from_otel(cx: &Context) -> Option<Self> {
        let span = cx.span();
        let sc = span.span_context();
        if !sc.is_valid() {
            return None;
        }
        let state = sc.trace_state().header();
        Some(Self {
            trace_id: sc.trace_id().to_bytes(),
            span_id: sc.span_id().to_bytes(),
            flags: sc.trace_flags().to_u8(),
            trace_state: (!state.is_empty()).then_some(state),
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
        let state = self
            .trace_state
            .as_deref()
            .and_then(|s| TraceState::from_str(s).ok())
            .unwrap_or_default();
        SpanContext::new(
            TraceId::from_bytes(self.trace_id),
            SpanId::from_bytes(self.span_id),
            TraceFlags::new(self.flags),
            self.is_remote,
            state,
        )
    }
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

/// Identifies a recorded span: what results, feedback, and evaluations link to.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObservationRef {
    /// Lowercase 32-hex trace id.
    pub trace_id: String,
    /// Lowercase 16-hex span id.
    pub span_id: String,
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
        if parse_hex::<16>(&trace_id).is_none() || parse_hex::<8>(&span_id).is_none() {
            return Err(Error::InvalidObservationRef);
        }
        Ok(Self { trace_id, span_id })
    }

    pub(crate) fn ids(&self) -> Option<(TraceId, SpanId)> {
        Some((
            TraceId::from_bytes(parse_hex::<16>(&self.trace_id)?),
            SpanId::from_bytes(parse_hex::<8>(&self.span_id)?),
        ))
    }
}

impl fmt::Display for ObservationRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.trace_id, self.span_id)
    }
}

fn parse_traceparent(value: &str) -> Option<([u8; 16], [u8; 8], u8)> {
    let mut fields = value.split('-');
    let version = fields.next()?;
    let trace_id = fields.next()?;
    let span_id = fields.next()?;
    let flags = fields.next()?;
    let rest: Vec<&str> = fields.collect();

    let [version] = parse_hex::<1>(version)?;
    if version == 0xff {
        return None;
    }
    // Version 00 has exactly four fields; later versions may append fields.
    if version == 0 && !rest.is_empty() {
        return None;
    }
    let trace_id = parse_hex::<16>(trace_id)?;
    let span_id = parse_hex::<8>(span_id)?;
    let [flags] = parse_hex::<1>(flags)?;
    Some((trace_id, span_id, flags))
}

/// Parses exactly `N` bytes of lowercase hex; all-zero ids are invalid for
/// `N > 1`.
fn parse_hex<const N: usize>(s: &str) -> Option<[u8; N]> {
    let bytes = s.as_bytes();
    if bytes.len() != N * 2 {
        return None;
    }
    let mut out = [0u8; N];
    for (i, [hi, lo]) in bytes.as_chunks::<2>().0.iter().enumerate() {
        out[i] = (nibble(*hi)? << 4) | nibble(*lo)?;
    }
    if N > 1 && out.iter().all(|b| *b == 0) {
        return None;
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
        ] {
            assert!(
                ObservationContext::from_traceparent(bad, None).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn future_versions_may_carry_extra_fields() {
        let cx = ObservationContext::from_traceparent(
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-future",
            None,
        )
        .unwrap();
        assert_eq!(cx.trace_id(), "4bf92f3577b34da6a3ce929d0e0e4736");
    }

    #[test]
    fn invalid_tracestate_is_dropped_not_fatal() {
        let cx = ObservationContext::from_traceparent(TP, Some("not a valid=state=")).unwrap();
        assert_eq!(cx.tracestate(), None);
    }

    #[test]
    fn observation_ref_validates_ids() {
        assert!(
            ObservationRef::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").is_ok()
        );
        assert!(ObservationRef::new("xyz", "00f067aa0ba902b7").is_err());
    }
}
