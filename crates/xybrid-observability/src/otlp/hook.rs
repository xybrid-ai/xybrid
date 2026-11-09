//! Per-destination response hooks.
//!
//! A [`SignalEndpoint`](super::SignalEndpoint) may carry one
//! [`ResponseHook`]. The transport calls it exactly once for every HTTP
//! response that destination answers: scheduled batches, flushes, retries and
//! the final shutdown drain alike. Transport errors and timeouts produce no
//! response and no call. The hook reads the status, headers and the bounded
//! body, and either lets the destination continue or disables it for the rest
//! of the pipeline's (or exporter's) life.
//!
//! Disabling is per destination and per build: the other signals of the same
//! pipeline, other pipelines, and local instrumentation are unaffected, and a
//! pipeline built again from the same configuration starts enabled. Once
//! disabled, the destination makes no further request: items still pending
//! and items offered later are dropped and counted (`SignalStats::dropped`);
//! items whose delivery was already attempted fail. Nothing is reported as
//! exported that the endpoint did not acknowledge.
//!
//! The package interprets no vendor header itself. Adapters (the Xybrid SDK's
//! workspace feedback, for instance) attach a hook to the destinations they
//! trust.

use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use reqwest::header::HeaderMap;

use super::config::Protocol;
use super::encode::Signal;

/// What a destination does after its [`ResponseHook`] saw a response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponseAction {
    /// Keep exporting: the response is handled as usual (a retryable status
    /// is retried within the export's deadline).
    #[default]
    Continue,
    /// Stop this destination for good. The response itself is still handled
    /// as usual (a 2xx acknowledges its items; any other status fails them),
    /// but no further request is made: pending and later items are dropped.
    Disable,
}

/// The `google.rpc.Status` of an OTLP/HTTP failure response, decoded in the
/// request's encoding (protobuf or OTLP/JSON).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RpcStatus {
    /// `google.rpc.Code` (for example 16, `UNAUTHENTICATED`).
    pub code: i32,
    /// The status message.
    pub message: String,
}

impl RpcStatus {
    /// A status with `code` and `message`.
    #[must_use]
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// One HTTP response to an export request, as a [`ResponseHook`] sees it.
pub struct ExportResponse<'a> {
    signal: Signal,
    protocol: Protocol,
    status: u16,
    headers: &'a HeaderMap,
    body: Option<&'a [u8]>,
}

impl<'a> ExportResponse<'a> {
    pub(crate) fn new(
        signal: Signal,
        protocol: Protocol,
        status: u16,
        headers: &'a HeaderMap,
        body: Option<&'a [u8]>,
    ) -> Self {
        Self {
            signal,
            protocol,
            status,
            headers,
            body,
        }
    }

    /// The signal of the destination: `traces`, `metrics` or `logs`.
    #[must_use]
    pub fn signal(&self) -> &'static str {
        self.signal.as_str()
    }

    /// The encoding of the request (and, per OTLP/HTTP, of its response).
    #[must_use]
    pub fn protocol(&self) -> Protocol {
        self.protocol
    }

    /// The HTTP status code.
    #[must_use]
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Whether the status is 2xx.
    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// The first value of header `name` (case-insensitive), when it is
    /// visible ASCII.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&'a str> {
        let headers: &'a HeaderMap = self.headers;
        headers.get(name)?.to_str().ok()
    }

    /// Every value of header `name` (case-insensitive), in order, as raw
    /// bytes: an adapter can tell a malformed value from an absent one.
    pub fn header_values(&self, name: &str) -> impl Iterator<Item = &'a [u8]> + 'a {
        let headers: &'a HeaderMap = self.headers;
        headers.get_all(name).iter().map(|value| value.as_bytes())
    }

    /// The response body, at most `max_response_body_bytes` of it; `None`
    /// when it was larger, could not be read, or the export's deadline ended
    /// the read.
    #[must_use]
    pub fn body(&self) -> Option<&'a [u8]> {
        self.body
    }

    /// The `google.rpc.Status` of a non-2xx response, decoded in the
    /// request's encoding; `None` for a 2xx, or an absent, empty or
    /// undecodable body.
    #[must_use]
    pub fn rpc_status(&self) -> Option<RpcStatus> {
        if self.is_success() {
            return None;
        }
        let body = self.body.filter(|b| !b.is_empty())?;
        match self.protocol {
            Protocol::HttpProtobuf => decode_status_protobuf(body),
            Protocol::HttpJson => decode_status_json(body),
        }
    }
}

impl fmt::Debug for ExportResponse<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header values and bodies stay out of diagnostics.
        f.debug_struct("ExportResponse")
            .field("signal", &self.signal.as_str())
            .field("protocol", &self.protocol)
            .field("status", &self.status)
            .field("body_bytes", &self.body.map(<[u8]>::len))
            .finish_non_exhaustive()
    }
}

type HookFn = dyn Fn(&ExportResponse<'_>) -> ResponseAction + Send + Sync;

/// A per-destination callback over every HTTP response of its exports.
///
/// Set it with [`SignalEndpoint::with_response_hook`](super::SignalEndpoint::with_response_hook).
/// It runs on the transport runtime, once per response, so it must return
/// quickly and never block. A panic inside it is caught and treated as
/// [`ResponseAction::Continue`]. Clones share the same callback; two hooks
/// are equal when they are clones of each other.
#[derive(Clone)]
pub struct ResponseHook {
    inner: Arc<HookFn>,
}

impl ResponseHook {
    /// A hook calling `f` for every response.
    pub fn new(f: impl Fn(&ExportResponse<'_>) -> ResponseAction + Send + Sync + 'static) -> Self {
        Self { inner: Arc::new(f) }
    }

    /// Calls the hook (adapters use this to chain a hook they replace).
    pub fn call(&self, response: &ExportResponse<'_>) -> ResponseAction {
        catch_unwind(AssertUnwindSafe(|| (self.inner)(response))).unwrap_or_else(|_| {
            log::warn!(
                "xybrid-observability: a {} response hook panicked; the destination continues",
                response.signal()
            );
            ResponseAction::Continue
        })
    }
}

impl fmt::Debug for ResponseHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ResponseHook(..)")
    }
}

impl PartialEq for ResponseHook {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for ResponseHook {}

/// Reads one base-128 varint (at most ten bytes).
fn varint(body: &mut &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let (&byte, rest) = body.split_first()?;
        *body = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

fn skip(body: &mut &[u8], len: u64) -> Option<()> {
    let len = usize::try_from(len).ok()?;
    *body = body.get(len..)?;
    Some(())
}

/// `google.rpc.Status` in protobuf: `int32 code = 1; string message = 2;`
/// (`details = 3` and unknown fields skipped). A wrong wire type for a known
/// field, a truncated field or invalid UTF-8 is undecodable.
fn decode_status_protobuf(mut body: &[u8]) -> Option<RpcStatus> {
    let mut status = RpcStatus::new(0, "");
    while !body.is_empty() {
        let key = varint(&mut body)?;
        let (field, wire) = (key >> 3, key & 7);
        match (field, wire) {
            // int32 is sign-extended to 64 bits on the wire; the low 32 bits
            // are the value.
            (1, 0) => status.code = varint(&mut body)? as i32,
            (2, 2) => {
                let len = usize::try_from(varint(&mut body)?).ok()?;
                let bytes = body.get(..len)?;
                status.message = std::str::from_utf8(bytes).ok()?.to_owned();
                body = &body[len..];
            }
            (0..=2, _) => return None,
            (_, 0) => {
                varint(&mut body)?;
            }
            (_, 1) => skip(&mut body, 8)?,
            (_, 2) => {
                let len = varint(&mut body)?;
                skip(&mut body, len)?;
            }
            (_, 5) => skip(&mut body, 4)?,
            _ => return None,
        }
    }
    Some(status)
}

/// `google.rpc.Status` in OTLP/JSON: `{"code": 16, "message": "..."}`; the
/// code may also be a numeric string (proto3 JSON). Unknown fields are
/// ignored; a field of the wrong type is undecodable.
fn decode_status_json(body: &[u8]) -> Option<RpcStatus> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let object = value.as_object()?;
    let code = match object.get("code") {
        None | Some(serde_json::Value::Null) => 0,
        Some(serde_json::Value::Number(n)) => i32::try_from(n.as_i64()?).ok()?,
        Some(serde_json::Value::String(s)) => s.trim().parse().ok()?,
        Some(_) => return None,
    };
    let message = match object.get("message") {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(_) => return None,
    };
    Some(RpcStatus { code, message })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Status { code: 16, message: "workspace_expired" }` as prost encodes it.
    const EXPIRED: &[u8] = b"\x08\x10\x12\x11workspace_expired";

    fn response(status: u16, protocol: Protocol, body: &[u8]) -> Option<RpcStatus> {
        let headers = HeaderMap::new();
        ExportResponse::new(Signal::Traces, protocol, status, &headers, Some(body)).rpc_status()
    }

    #[test]
    fn protobuf_status_decodes_known_fields_and_skips_unknown_ones() {
        assert_eq!(
            decode_status_protobuf(EXPIRED),
            Some(RpcStatus::new(16, "workspace_expired"))
        );
        // details (field 3) and unknown fixed-width fields are skipped.
        let mut with_details = EXPIRED.to_vec();
        with_details.extend_from_slice(
            b"\x1a\x03abc\x25\x01\x02\x03\x04\x29\x01\x02\x03\x04\x05\x06\x07\x08\x30\x05",
        );
        assert_eq!(
            decode_status_protobuf(&with_details),
            Some(RpcStatus::new(16, "workspace_expired"))
        );
        // A negative int32 is a ten-byte varint.
        let negative = b"\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01";
        assert_eq!(decode_status_protobuf(negative).unwrap().code, -1);
    }

    #[test]
    fn malformed_protobuf_status_is_undecodable() {
        for body in [
            &b"\x08"[..],                                        // truncated varint
            b"\x12\x05abc",                                      // truncated string
            b"\x0a\x01x",                                        // field 1 with the wrong wire type
            b"\x10\x01",                                         // field 2 as a varint
            b"\x12\x02\xff\xfe",                                 // invalid UTF-8
            b"\x0b",                                             // a group
            b"\x08\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01", // varint too long
        ] {
            assert_eq!(decode_status_protobuf(body), None, "{body:?}");
        }
    }

    #[test]
    fn json_status_accepts_numbers_and_numeric_strings() {
        assert_eq!(
            decode_status_json(br#"{"code":16,"message":"workspace_expired"}"#),
            Some(RpcStatus::new(16, "workspace_expired"))
        );
        assert_eq!(
            decode_status_json(br#"{"code":"16","message":"m","details":[]}"#),
            Some(RpcStatus::new(16, "m"))
        );
        assert_eq!(decode_status_json(br"{}"), Some(RpcStatus::new(0, "")));
        for body in [
            &br#"{"code":16.5}"#[..],
            br#"{"code":true}"#,
            br#"{"message":7}"#,
            br"[]",
            br"not json",
            br#"{"code":99999999999}"#,
        ] {
            assert_eq!(decode_status_json(body), None, "{body:?}");
        }
    }

    #[test]
    fn rpc_status_follows_the_request_encoding_and_ignores_success() {
        assert_eq!(
            response(401, Protocol::HttpProtobuf, EXPIRED),
            Some(RpcStatus::new(16, "workspace_expired"))
        );
        assert_eq!(response(401, Protocol::HttpJson, EXPIRED), None);
        assert_eq!(
            response(
                401,
                Protocol::HttpJson,
                br#"{"code":16,"message":"workspace_expired"}"#
            ),
            Some(RpcStatus::new(16, "workspace_expired"))
        );
        assert_eq!(response(200, Protocol::HttpProtobuf, EXPIRED), None);
        assert_eq!(response(401, Protocol::HttpProtobuf, b""), None);
    }

    #[test]
    fn headers_are_case_insensitive_and_debug_hides_values() {
        let mut headers = HeaderMap::new();
        headers.append("x-test", "one".parse().unwrap());
        headers.append("x-test", "two".parse().unwrap());
        headers.append(
            "x-bad",
            reqwest::header::HeaderValue::from_bytes(b"\xff").unwrap(),
        );
        let view = ExportResponse::new(
            Signal::Logs,
            Protocol::HttpJson,
            204,
            &headers,
            Some(b"secret body"),
        );
        assert_eq!(view.header("X-Test"), Some("one"));
        assert_eq!(
            view.header_values("x-test").collect::<Vec<_>>(),
            [&b"one"[..], b"two"]
        );
        assert_eq!(
            view.header_values("x-bad").collect::<Vec<_>>(),
            [&b"\xff"[..]]
        );
        assert_eq!(view.header("x-bad"), None);
        assert_eq!(view.header("bad name"), None);
        assert_eq!(view.header_values("bad name").count(), 0);
        assert!(view.is_success());
        assert_eq!(view.signal(), "logs");
        let printed = format!("{view:?}");
        assert!(
            !printed.contains("one") && !printed.contains("secret"),
            "{printed}"
        );
    }

    #[test]
    fn a_panicking_hook_continues_and_clones_are_equal() {
        let headers = HeaderMap::new();
        let view = ExportResponse::new(Signal::Traces, Protocol::HttpProtobuf, 500, &headers, None);
        let hook = ResponseHook::new(|_| panic!("hook bug"));
        assert_eq!(hook.call(&view), ResponseAction::Continue);
        assert_eq!(hook.clone(), hook);
        assert_ne!(ResponseHook::new(|_| ResponseAction::Disable), hook);
        assert_eq!(format!("{hook:?}"), "ResponseHook(..)");
    }
}
