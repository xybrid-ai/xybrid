//! Xybrid workspace feedback from trace ingest.
//!
//! A key from `xybrid connect` belongs to a provisional workspace until a
//! human claims it. Xybrid's trace ingest reports that lifecycle on the
//! responses of the Xybrid preset's trace destination:
//!
//! - every accepted export carries `X-Xybrid-Workspace: provisional;
//!   expires_at=<RFC 3339>`, the key's deadline;
//! - once the deadline passed, ingest answers HTTP 401 with
//!   `X-Xybrid-Workspace: expired` and a `google.rpc.Status` of code 16
//!   (`UNAUTHENTICATED`) and message `workspace_expired`, encoded like the
//!   request (protobuf or OTLP/JSON).
//!
//! [`WorkspaceFeedback`] is the adapter: attached to a destination built by
//! the Xybrid preset ([`ObservabilityConfig::xybrid`],
//! [`ObservabilityConfig::xybrid_at`], [`OtlpConfig::xybrid_traces`],
//! [`OtlpConfig::xybrid_traces_to`]), never to a generic Collector, it keeps
//! the latest [`WorkspaceNotice`] and stops that destination on an expiry.
//! Expiry is terminal only when all three agree: HTTP 401, the `expired`
//! marker, and that status body. A generic 401, a malformed or ambiguous
//! marker, a 429 or 503 is never an expiry. Stopping the destination keeps
//! local inference, local spans and every other destination running; the
//! spans it no longer sends are counted as dropped, never as exported. A new
//! session (another [`init_observability`] after the claim) starts fresh.
//!
//! Notices are also logged once per session (target `xybrid_sdk`) for
//! applications that install a logger; the log carries no key or token.

use std::sync::atomic::{AtomicBool, Ordering};

use chrono::{DateTime, SecondsFormat, Utc};
use xybrid_observability::otlp::{ExportResponse, ResponseAction, ResponseHook, RpcStatus};

use super::*;

/// The response header carrying the workspace lifecycle.
const WORKSPACE_HEADER: &str = "x-xybrid-workspace";
/// `google.rpc.Code` `UNAUTHENTICATED`.
const UNAUTHENTICATED: i32 = 16;
/// The status message of an expired workspace's key.
const WORKSPACE_EXPIRED: &str = "workspace_expired";

/// The workspace lifecycle Xybrid's trace ingest last reported for a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorkspaceNotice {
    /// The key belongs to an unclaimed workspace. It stops working at
    /// `key_expires_at` unless the workspace is claimed.
    Provisional {
        /// The key's deadline.
        key_expires_at: SystemTime,
    },
    /// The unclaimed workspace passed its key deadline. Ingest refused the
    /// key and export to this destination stopped for the session (spans it
    /// accepted before that were stored). Claiming the workspace restores
    /// the key; a new session exports again.
    Expired,
}

#[derive(Debug, Default)]
struct FeedbackState {
    notice: Mutex<Option<WorkspaceNotice>>,
    provisional_logged: AtomicBool,
}

/// A handle to the workspace feedback of one Xybrid trace destination.
///
/// Clones share the state, which lives as long as any handle or the pipeline
/// does: read it after the pipeline shut down to see a notice that first
/// arrived with the final drain. Capture it (from
/// [`ObservabilitySession::workspace_feedback`] or [`workspace_feedback`])
/// before shutting the session down, since shutdown uninstalls the default.
#[derive(Clone, Default)]
pub struct WorkspaceFeedback {
    state: Arc<FeedbackState>,
}

impl WorkspaceFeedback {
    /// Attaches workspace feedback to `config`'s trace destination when the
    /// Xybrid preset built it ([`SignalEndpoint::is_xybrid_preset`]); `None`
    /// for any other configuration. A response hook already set on that
    /// destination keeps running first, and either hook may stop it.
    ///
    /// [`ObservabilitySession::new`] does this itself; this is for hosts that
    /// build an [`xybrid_observability::otlp::ObservabilityPipeline`] from the
    /// configuration directly. Clones of the attached configuration share
    /// this feedback: attach once per build for a notice per pipeline. Every
    /// build starts with its destination enabled either way.
    pub fn attach(config: &mut OtlpConfig) -> Option<Self> {
        let traces = config
            .traces
            .as_mut()
            .filter(|traces| traces.is_xybrid_preset())?;
        let feedback = Self::default();
        let state = feedback.state.clone();
        let previous = traces.response_hook.take();
        traces.response_hook = Some(ResponseHook::new(move |response| {
            let theirs = previous
                .as_ref()
                .map_or(ResponseAction::Continue, |hook| hook.call(response));
            let ours = state.observe(response);
            if theirs == ResponseAction::Disable || ours == ResponseAction::Disable {
                ResponseAction::Disable
            } else {
                ResponseAction::Continue
            }
        }));
        Some(feedback)
    }

    /// The latest notice: [`WorkspaceNotice::Expired`] once ingest refused
    /// the key, else the deadline of the latest accepted export that carried
    /// one. `None` before any response, and after an accepted export without
    /// the header (a claimed workspace or a full key). Responses that say
    /// nothing about the workspace (a 429, a generic 401, a malformed
    /// marker) leave it as it was.
    ///
    /// A destination stops at its expiry, so within one pipeline nothing
    /// replaces [`WorkspaceNotice::Expired`]. Only an accepted export from a
    /// pipeline built again with this feedback attached (after a claim) does.
    #[must_use]
    pub fn notice(&self) -> Option<WorkspaceNotice> {
        *lock(&self.state.notice)
    }

    /// Whether ingest reported the workspace expired (export stopped).
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.notice() == Some(WorkspaceNotice::Expired)
    }
}

impl fmt::Debug for WorkspaceFeedback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkspaceFeedback")
            .field("notice", &self.notice())
            .finish()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl FeedbackState {
    /// Folds one response of the trusted destination into the notice.
    ///
    /// Only the expiry response itself disables the destination: the
    /// pipeline keeps that destination stopped from then on, and a pipeline
    /// built again (after a claim) starts enabled, even when it was built
    /// from a configuration this feedback is already attached to.
    fn observe(&self, response: &ExportResponse<'_>) -> ResponseAction {
        let markers: Vec<&[u8]> = response.header_values(WORKSPACE_HEADER).collect();
        let rpc = response.rpc_status();
        self.fold(classify(response.status(), &markers, rpc.as_ref()))
    }

    fn fold(&self, observation: Observation) -> ResponseAction {
        let mut notice = lock(&self.notice);
        match observation {
            Observation::Expired => {
                let already =
                    notice.replace(WorkspaceNotice::Expired) == Some(WorkspaceNotice::Expired);
                drop(notice);
                if already {
                    return ResponseAction::Disable;
                }
                log::warn!(
                    target: "xybrid_sdk",
                    "Xybrid trace export stopped for this session: the key's unclaimed \
                     workspace expired. Claim the workspace (`xybrid auth claim`) to restore \
                     the key, then start observability again. Local inference is unaffected."
                );
                ResponseAction::Disable
            }
            Observation::Provisional(key_expires_at) => {
                *notice = Some(WorkspaceNotice::Provisional { key_expires_at });
                drop(notice);
                if !self.provisional_logged.swap(true, Ordering::Relaxed) {
                    log::info!(
                        target: "xybrid_sdk",
                        "Xybrid trace export: this key belongs to an unclaimed workspace and \
                         stops working at {} unless the workspace is claimed (`xybrid auth claim`).",
                        DateTime::<Utc>::from(key_expires_at).to_rfc3339_opts(SecondsFormat::Secs, true)
                    );
                }
                ResponseAction::Continue
            }
            Observation::Unrestricted => {
                *notice = None;
                ResponseAction::Continue
            }
            Observation::Ignored => ResponseAction::Continue,
        }
    }
}

/// What one response says about the workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Observation {
    /// An accepted export of a provisional key, with its deadline.
    Provisional(SystemTime),
    /// An accepted export without the header: no lifecycle (claimed, or a
    /// key that never had one).
    Unrestricted,
    /// HTTP 401, the `expired` marker and the `workspace_expired` status.
    Expired,
    /// Nothing to act on: any other status, or a malformed, ambiguous or
    /// unconfirmed marker.
    Ignored,
}

/// A parsed `X-Xybrid-Workspace` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Marker {
    Provisional(SystemTime),
    Expired,
}

/// Classifies one response of the trusted destination. `markers` are the raw
/// `X-Xybrid-Workspace` values; more than one is ambiguous.
fn classify(status: u16, markers: &[&[u8]], rpc: Option<&RpcStatus>) -> Observation {
    let marker = match markers {
        [] => None,
        [one] => match std::str::from_utf8(one).ok().and_then(parse_marker) {
            Some(marker) => Some(marker),
            None => return Observation::Ignored,
        },
        _ => return Observation::Ignored,
    };
    if (200..300).contains(&status) {
        return match marker {
            None => Observation::Unrestricted,
            Some(Marker::Provisional(at)) => Observation::Provisional(at),
            Some(Marker::Expired) => Observation::Ignored,
        };
    }
    let confirmed =
        rpc.is_some_and(|s| s.code == UNAUTHENTICATED && s.message == WORKSPACE_EXPIRED);
    if status == 401 && marker == Some(Marker::Expired) && confirmed {
        Observation::Expired
    } else {
        Observation::Ignored
    }
}

/// `provisional; expires_at=<RFC 3339>` or `expired`. The state is
/// case-insensitive; parameters are `name=value` (a value may be quoted),
/// unknown names are ignored, and anything else makes the value malformed.
fn parse_marker(value: &str) -> Option<Marker> {
    let mut parts = value.split(';');
    let state = parts.next()?.trim();
    let mut expires_at = None;
    for parameter in parts {
        let (name, raw) = parameter.split_once('=')?;
        let name = name.trim();
        if name.is_empty() {
            return None;
        }
        let raw = raw.trim();
        let raw = raw
            .strip_prefix('"')
            .and_then(|r| r.strip_suffix('"'))
            .unwrap_or(raw);
        if name.eq_ignore_ascii_case("expires_at") {
            if expires_at.is_some() {
                return None;
            }
            expires_at = Some(deadline(raw)?);
        }
    }
    if state.eq_ignore_ascii_case("provisional") {
        expires_at.map(Marker::Provisional)
    } else if state.eq_ignore_ascii_case("expired") {
        Some(Marker::Expired)
    } else {
        None
    }
}

/// An RFC 3339 key deadline. One before the Unix epoch is malformed: no key
/// expires then, and not every platform's `SystemTime` reaches that far back.
fn deadline(raw: &str) -> Option<SystemTime> {
    let at = DateTime::parse_from_rfc3339(raw).ok()?;
    let seconds = u64::try_from(at.timestamp()).ok()?;
    let since_epoch = Duration::from_secs(seconds)
        .checked_add(Duration::from_nanos(u64::from(at.timestamp_subsec_nanos())))?;
    SystemTime::UNIX_EPOCH.checked_add(since_epoch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;

    const KEY: &str = concat!("sk_test", "_workspaceFeedbackKey");
    /// `Status { code: 16, message: "workspace_expired" }` in protobuf.
    const EXPIRED_PROTOBUF: &[u8] = b"\x08\x10\x12\x11workspace_expired";
    const EXPIRED_JSON: &str = r#"{"code":16,"message":"workspace_expired"}"#;
    const DEADLINE: &str = "2026-11-15T12:00:00Z";

    fn at(rfc3339: &str) -> SystemTime {
        SystemTime::from(DateTime::parse_from_rfc3339(rfc3339).unwrap())
    }

    fn expired_status() -> RpcStatus {
        RpcStatus::new(16, "workspace_expired")
    }

    fn one(marker: &str) -> Vec<&[u8]> {
        vec![marker.as_bytes()]
    }

    #[test]
    fn provisional_markers_parse_the_deadline() {
        let deadline = at(DEADLINE);
        for value in [
            "provisional; expires_at=2026-11-15T12:00:00Z",
            "provisional;expires_at=2026-11-15T12:00:00Z",
            " Provisional ; EXPIRES_AT = \"2026-11-15T12:00:00Z\" ",
            "provisional; expires_at=2026-11-15T13:00:00+01:00",
            "provisional; claim_expires_at=2026-11-22T12:00:00Z; expires_at=2026-11-15T12:00:00Z",
        ] {
            assert_eq!(
                classify(200, &one(value), None),
                Observation::Provisional(deadline),
                "{value}"
            );
        }
        // Fractional seconds (A2's precision) are accepted too.
        assert_eq!(
            classify(
                202,
                &one("provisional; expires_at=2026-11-15T12:00:00.250Z"),
                None
            ),
            Observation::Provisional(deadline + Duration::from_millis(250))
        );
        assert_eq!(classify(200, &[], None), Observation::Unrestricted);
        assert_eq!(
            classify(
                200,
                &one("provisional; expires_at=1970-01-01T00:00:00Z"),
                None
            ),
            Observation::Provisional(SystemTime::UNIX_EPOCH)
        );
    }

    #[test]
    fn malformed_or_ambiguous_markers_are_ignored() {
        for value in [
            "",
            "provisional",
            "provisional; expires_at=",
            "provisional; expires_at=tomorrow",
            "provisional; expires_at=2026-11-15",
            "provisional expires_at=2026-11-15T12:00:00Z",
            "provisonal; expires_at=2026-11-15T12:00:00Z",
            "provisional; =2026-11-15T12:00:00Z",
            "provisional; expires_at",
            "provisional; expires_at=2026-11-15T12:00:00Z; expires_at=2026-11-16T12:00:00Z",
            // Before the Unix epoch: no key expires then (and a Windows
            // `SystemTime` cannot hold year 1).
            "provisional; expires_at=0001-01-01T00:00:00Z",
            "provisional; expires_at=1969-12-31T23:59:59.999Z",
            "claimed",
        ] {
            assert_eq!(
                classify(200, &one(value), None),
                Observation::Ignored,
                "{value:?}"
            );
        }
        let both: Vec<&[u8]> = vec![
            b"provisional; expires_at=2026-11-15T12:00:00Z",
            b"provisional; expires_at=2026-11-15T12:00:00Z",
        ];
        assert_eq!(classify(200, &both, None), Observation::Ignored);
        assert_eq!(
            classify(200, &[b"provisional\xff"], None),
            Observation::Ignored
        );
        // `expired` on an accepted export is not an expiry.
        assert_eq!(classify(200, &one("expired"), None), Observation::Ignored);
    }

    #[test]
    fn only_401_with_the_marker_and_the_status_body_is_expiry() {
        let expired = expired_status();
        assert_eq!(
            classify(401, &one("expired"), Some(&expired)),
            Observation::Expired
        );
        assert_eq!(
            classify(401, &one(" EXPIRED "), Some(&expired)),
            Observation::Expired
        );
        let generic = RpcStatus::new(16, "Invalid API key");
        let wrong_code = RpcStatus::new(14, "workspace_expired");
        /// Status, `X-Xybrid-Workspace` values, status body, case name.
        type Case<'a> = (u16, Vec<&'a [u8]>, Option<&'a RpcStatus>, &'a str);
        let cases: Vec<Case<'_>> = vec![
            (401, vec![], Some(&generic), "generic 401"),
            (401, vec![], Some(&expired), "body without the marker"),
            (
                401,
                one("expired"),
                Some(&generic),
                "marker with a generic body",
            ),
            (401, one("expired"), None, "marker without a body"),
            (401, one("expired"), Some(&wrong_code), "wrong code"),
            (401, one("expird"), Some(&expired), "misspelled marker"),
            (
                401,
                one("expired; junk"),
                Some(&expired),
                "malformed parameter",
            ),
            (
                401,
                vec![b"expired", b"expired"],
                Some(&expired),
                "two markers",
            ),
            (429, one("expired"), Some(&expired), "rate limited"),
            (503, one("expired"), Some(&expired), "unavailable"),
            (403, one("expired"), Some(&expired), "forbidden"),
            (500, one("expired"), Some(&expired), "server error"),
        ];
        for (status, markers, rpc, case) in cases {
            assert_eq!(
                classify(status, &markers, rpc),
                Observation::Ignored,
                "{case}"
            );
        }
    }

    /// Fast, bounded export settings for loopback tests.
    fn quick(config: OtlpConfig) -> OtlpConfig {
        config
            .with_environment(xybrid_observability::otlp::EnvironmentSource::Disabled)
            .with_batch(BatchLimits {
                scheduled_delay: Duration::from_secs(60),
                ..BatchLimits::default()
            })
            .with_retry(RetryLimits {
                max_retries: 0,
                ..RetryLimits::default()
            })
    }

    fn xybrid_session(server: &MockServer, protocol: Protocol) -> ObservabilitySession {
        let mut otlp = OtlpConfig::xybrid_traces_to(&server.url("/v1/traces"), KEY);
        if let Some(traces) = otlp.traces.as_mut() {
            traces.protocol = protocol;
        }
        ObservabilitySession::new(ObservabilityConfig::otlp(quick(otlp)).device_attributes(false))
            .unwrap()
    }

    fn span(session: &ObservabilitySession) {
        session
            .observer()
            .start_inference(
                InferenceRequest::new("chat", "llama.cpp"),
                StartOptions::default(),
            )
            .end();
    }

    #[test]
    fn a_provisional_header_yields_a_notice_and_a_plain_success_clears_it() {
        let server = MockServer::start();
        let mut provisional = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/traces")
                .header("authorization", format!("Bearer {KEY}"));
            then.status(200).header(
                "X-Xybrid-Workspace",
                format!("provisional; expires_at={DEADLINE}"),
            );
        });
        let session = xybrid_session(&server, Protocol::HttpProtobuf);
        let feedback = session.workspace_feedback().expect("Xybrid preset");
        assert_eq!(feedback.notice(), None);
        span(&session);
        session.flush(Duration::from_secs(5)).unwrap();
        assert_eq!(
            feedback.notice(),
            Some(WorkspaceNotice::Provisional {
                key_expires_at: at(DEADLINE)
            })
        );

        // Claimed meanwhile: ingest stops sending the header.
        provisional.delete();
        let plain = server.mock(|when, then| {
            when.method(POST).path("/v1/traces");
            then.status(200);
        });
        span(&session);
        session.flush(Duration::from_secs(5)).unwrap();
        plain.assert_hits(1);
        assert_eq!(feedback.notice(), None);
        session.shutdown(Duration::from_secs(5)).unwrap();
    }

    #[test]
    fn an_expired_answer_stops_the_destination_in_both_encodings() {
        for protocol in [Protocol::HttpProtobuf, Protocol::HttpJson] {
            let server = MockServer::start();
            let refused = server.mock(|when, then| {
                when.method(POST)
                    .path("/v1/traces")
                    .header("content-type", protocol.content_type());
                let then = then.status(401).header("X-Xybrid-Workspace", "expired");
                match protocol {
                    Protocol::HttpProtobuf => then
                        .header("content-type", "application/x-protobuf")
                        .body(EXPIRED_PROTOBUF),
                    _ => then
                        .header("content-type", "application/json")
                        .body(EXPIRED_JSON),
                };
            });
            let session = xybrid_session(&server, protocol);
            let feedback = session.workspace_feedback().unwrap();
            span(&session);
            assert!(session.flush(Duration::from_secs(5)).is_err());
            assert_eq!(
                feedback.notice(),
                Some(WorkspaceNotice::Expired),
                "{protocol:?}"
            );
            assert!(feedback.is_expired());

            // No further request, and the refused spans are never "exported".
            span(&session);
            span(&session);
            let _ = session.flush(Duration::from_secs(5));
            session.shutdown(Duration::from_secs(5)).unwrap();
            refused.assert_hits(1);
            let traces = session.stats().unwrap().traces;
            assert_eq!(
                (
                    traces.exported,
                    traces.failed,
                    traces.dropped,
                    traces.pending
                ),
                (0, 1, 2, 0),
                "{protocol:?}"
            );
        }
    }

    #[test]
    fn lookalike_answers_never_stop_the_destination() {
        let generic = r#"{"code":16,"message":"Invalid API key"}"#;
        let cases: Vec<(u16, Option<&str>, &str)> = vec![
            (401, None, generic),
            (401, Some("expired"), generic),
            (401, None, EXPIRED_JSON),
            (401, Some("expired; nope"), EXPIRED_JSON),
            (429, Some("expired"), EXPIRED_JSON),
            (503, Some("expired"), EXPIRED_JSON),
        ];
        for (status, marker, body) in cases {
            let server = MockServer::start();
            let answer = server.mock(|when, then| {
                when.method(POST).path("/v1/traces");
                let then = then.status(status).body(body);
                if let Some(marker) = marker {
                    then.header("X-Xybrid-Workspace", marker);
                }
            });
            let session = xybrid_session(&server, Protocol::HttpJson);
            let feedback = session.workspace_feedback().unwrap();
            for _ in 0..2 {
                span(&session);
                assert!(session.flush(Duration::from_secs(5)).is_err());
            }
            session.shutdown(Duration::from_secs(5)).unwrap();
            answer.assert_hits(2);
            assert_eq!(feedback.notice(), None, "{status} {marker:?}");
            assert_eq!(session.stats().unwrap().traces.dropped, 0);
        }
    }

    #[test]
    fn a_generic_collector_gets_no_workspace_feedback() {
        let server = MockServer::start();
        let collector = server.mock(|when, then| {
            when.method(POST).path("/v1/traces");
            then.status(401)
                .header("X-Xybrid-Workspace", "expired")
                .body(EXPIRED_PROTOBUF);
        });
        // The same URL and bearer header, configured as a generic endpoint.
        let endpoint = SignalEndpoint::new(server.url("/v1/traces"))
            .with_header("Authorization", format!("Bearer {KEY}"));
        let session = ObservabilitySession::new(
            ObservabilityConfig::otlp(quick(OtlpConfig::default().with_traces(Some(endpoint))))
                .device_attributes(false),
        )
        .unwrap();
        assert!(session.workspace_feedback().is_none());
        for _ in 0..2 {
            span(&session);
            assert!(session.flush(Duration::from_secs(5)).is_err());
        }
        session.shutdown(Duration::from_secs(5)).unwrap();
        collector.assert_hits(2);
        let mut collector_config = OtlpConfig::collector("http://localhost:4318");
        assert!(WorkspaceFeedback::attach(&mut collector_config).is_none());
        assert!(collector_config.traces.unwrap().response_hook.is_none());
    }

    #[test]
    fn a_notice_first_received_during_shutdown_survives_it() {
        let _guard = test_support::default_session_lock();
        let server = MockServer::start();
        let accepted = server.mock(|when, then| {
            when.method(POST).path("/v1/traces");
            then.status(200).header(
                "X-Xybrid-Workspace",
                format!("provisional; expires_at={DEADLINE}"),
            );
        });
        init_observability(
            ObservabilityConfig::xybrid_at(&server.url("/v1/traces"), KEY).device_attributes(false),
        )
        .unwrap();
        // Captured before shutdown: shutdown uninstalls the default.
        let feedback = workspace_feedback().expect("the default session's feedback");
        span(&default_session().unwrap());
        assert_eq!(feedback.notice(), None, "nothing exported yet");
        shutdown_observability(Duration::from_secs(5)).unwrap();
        accepted.assert_hits(1);
        assert!(workspace_feedback().is_none(), "the default is gone");
        assert_eq!(
            feedback.notice(),
            Some(WorkspaceNotice::Provisional {
                key_expires_at: at(DEADLINE)
            })
        );
    }

    #[test]
    fn a_new_session_after_expiry_starts_fresh_and_exports_again() {
        let server = MockServer::start();
        let mut refused = server.mock(|when, then| {
            when.method(POST).path("/v1/traces");
            then.status(401)
                .header("X-Xybrid-Workspace", "expired")
                .body(EXPIRED_PROTOBUF);
        });
        let config = ObservabilityConfig::otlp(quick(OtlpConfig::xybrid_traces_to(
            &server.url("/v1/traces"),
            KEY,
        )))
        .device_attributes(false);
        let first = ObservabilitySession::new(config.clone()).unwrap();
        span(&first);
        let _ = first.flush(Duration::from_secs(5));
        assert!(first.workspace_feedback().unwrap().is_expired());
        first.shutdown(Duration::from_secs(5)).unwrap();

        // Claimed: the same key works again for a new exporter.
        refused.delete();
        let accepted = server.mock(|when, then| {
            when.method(POST).path("/v1/traces");
            then.status(200);
        });
        let second = ObservabilitySession::new(config).unwrap();
        let feedback = second.workspace_feedback().unwrap();
        assert_eq!(feedback.notice(), None);
        span(&second);
        second.flush(Duration::from_secs(5)).unwrap();
        second.shutdown(Duration::from_secs(5)).unwrap();
        accepted.assert_hits(1);
        assert_eq!(second.stats().unwrap().traces.exported, 1);
        assert!(first.workspace_feedback().unwrap().is_expired());
    }

    #[test]
    fn a_pipeline_rebuilt_from_an_attached_configuration_recovers_after_a_claim() {
        use xybrid_observability::otlp::ObservabilityPipeline;

        let server = MockServer::start();
        let mut refused = server.mock(|when, then| {
            when.method(POST).path("/v1/traces");
            then.status(401)
                .header("X-Xybrid-Workspace", "expired")
                .body(EXPIRED_PROTOBUF);
        });
        // A host attaches once and builds from clones of the configuration.
        let mut otlp = quick(OtlpConfig::xybrid_traces_to(&server.url("/v1/traces"), KEY));
        let feedback = WorkspaceFeedback::attach(&mut otlp).expect("Xybrid preset");
        let first = ObservabilityPipeline::build(otlp.clone()).unwrap();
        first
            .observer()
            .start_inference(
                InferenceRequest::new("chat", "llama.cpp"),
                StartOptions::default(),
            )
            .end();
        assert!(first.force_flush(Duration::from_secs(5)).is_err());
        first.shutdown(Duration::from_secs(5)).unwrap();
        refused.assert_hits(1);
        assert!(feedback.is_expired());

        // Claimed: ingest accepts the key again, without the header.
        refused.delete();
        let accepted = server.mock(|when, then| {
            when.method(POST).path("/v1/traces");
            then.status(200);
        });
        let second = ObservabilityPipeline::build(otlp.clone()).unwrap();
        for _ in 0..2 {
            second
                .observer()
                .start_inference(
                    InferenceRequest::new("chat", "llama.cpp"),
                    StartOptions::default(),
                )
                .end();
            second.force_flush(Duration::from_secs(5)).unwrap();
        }
        second.shutdown(Duration::from_secs(5)).unwrap();
        let traces = second.stats().traces;
        assert_eq!((traces.exported, traces.dropped), (2, 0));
        assert_eq!(feedback.notice(), None, "the accepted export replaced it");

        // A session given the already attached configuration exports too.
        let session =
            ObservabilitySession::new(ObservabilityConfig::otlp(otlp).device_attributes(false))
                .unwrap();
        span(&session);
        session.flush(Duration::from_secs(5)).unwrap();
        session.shutdown(Duration::from_secs(5)).unwrap();
        accepted.assert_hits(3);
        assert_eq!(session.stats().unwrap().traces.exported, 1);
    }

    #[test]
    fn responses_without_workspace_news_keep_the_notice() {
        let state = FeedbackState::default();
        let deadline = at(DEADLINE);
        let provisional = Observation::Provisional(deadline);
        assert_eq!(state.fold(provisional), ResponseAction::Continue);
        assert_eq!(state.fold(Observation::Ignored), ResponseAction::Continue);
        assert_eq!(
            *lock(&state.notice),
            Some(WorkspaceNotice::Provisional {
                key_expires_at: deadline
            })
        );
        // Each expiry answer disables the destination it came from; nothing
        // else does, whatever the notice already says.
        for _ in 0..2 {
            assert_eq!(state.fold(Observation::Expired), ResponseAction::Disable);
        }
        assert_eq!(state.fold(Observation::Ignored), ResponseAction::Continue);
        assert_eq!(*lock(&state.notice), Some(WorkspaceNotice::Expired));
        // An accepted export (a pipeline built again after the claim).
        assert_eq!(
            state.fold(Observation::Unrestricted),
            ResponseAction::Continue
        );
        assert_eq!(*lock(&state.notice), None);
    }

    #[test]
    fn an_application_hook_keeps_running_alongside() {
        let server = MockServer::start();
        let accepted = server.mock(|when, then| {
            when.method(POST).path("/v1/traces");
            then.status(200).header(
                "X-Xybrid-Workspace",
                format!("provisional; expires_at={DEADLINE}"),
            );
        });
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = calls.clone();
        let mut otlp = quick(OtlpConfig::xybrid_traces_to(&server.url("/v1/traces"), KEY));
        if let Some(traces) = otlp.traces.as_mut() {
            traces.response_hook = Some(ResponseHook::new(move |_| {
                counted.fetch_add(1, Ordering::Relaxed);
                ResponseAction::Continue
            }));
        }
        let session =
            ObservabilitySession::new(ObservabilityConfig::otlp(otlp).device_attributes(false))
                .unwrap();
        span(&session);
        session.flush(Duration::from_secs(5)).unwrap();
        session.shutdown(Duration::from_secs(5)).unwrap();
        accepted.assert_hits(1);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(matches!(
            session.workspace_feedback().unwrap().notice(),
            Some(WorkspaceNotice::Provisional { .. })
        ));
    }
}
