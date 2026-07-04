//! Remote failure-inbox client (eliXir) — the read seam to the platform's
//! `GET /v1/telemetry/feedback` endpoint.
//!
//! `xybrid eval inbox` surfaces the same triage queue the console shows:
//! explicit `Feedback` flags (`result.report()`) plus implicit monitor `Signal`
//! auto-flags, collected from the field. This is the *read* side of the harness
//! collect loop — the "go look at the failure logs" view, the terminal twin of
//! the console inbox.
//!
//! Scope: this client is **read-only**. Minting full eval [`Case`]s from these
//! items additionally needs the original inference *input*, which is not carried
//! on a `Feedback` event (only the captured `expected`/`note` are) — it rides on
//! payload capture, joined by `trace_id`. That backfill is deliberately out of
//! scope here; `eval pull` keeps its local-inbox path for case minting.
//!
//! [`Case`]: crate::eval::format::Case

use serde::Deserialize;
use std::time::Duration;

use crate::model::SdkError;

/// Connection timeout for inbox requests.
const CONNECT_TIMEOUT_MS: u64 = 5_000;
/// Request timeout for inbox requests.
const REQUEST_TIMEOUT_MS: u64 = 15_000;

/// Default platform API base (override with `XYBRID_API_URL`).
pub const DEFAULT_API_URL: &str = "https://api.xybrid.dev";

/// One triage item — mirrors `FeedbackInboxItem` on the platform.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InboxItem {
    /// Telemetry event id.
    pub id: String,
    /// Originating inference trace id, when threaded through.
    #[serde(default)]
    pub trace_id: Option<String>,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub task: Option<String>,
    /// `report` (explicit flag) or `signal` (monitor auto-flag).
    pub source: String,
    /// `up` / `down` for explicit feedback.
    #[serde(default)]
    pub rating: Option<String>,
    /// `structural` / `behavioral` for signals.
    #[serde(default)]
    pub signal_kind: Option<String>,
    /// Signal name (`truncated`, `regenerated`, …).
    #[serde(default)]
    pub signal_name: Option<String>,
    /// Captured correction (opt-in payload capture only).
    #[serde(default)]
    pub expected: Option<String>,
    /// Captured note (opt-in).
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default)]
    pub payload_captured: bool,
    #[serde(default)]
    pub device_id: Option<String>,
    #[serde(default)]
    pub platform: Option<String>,
}

/// A `(key, count)` breakdown row.
#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct InboxCountRow {
    pub key: String,
    pub count: i64,
}

/// Aggregate inbox stats — mirrors `FeedbackInboxSummary`.
#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct InboxSummary {
    pub total: i64,
    pub down_count: i64,
    pub up_count: i64,
    pub signal_count: i64,
    #[serde(default)]
    pub negative_rate: Option<f64>,
    #[serde(default)]
    pub by_model: Vec<InboxCountRow>,
    #[serde(default)]
    pub top_signals: Vec<InboxCountRow>,
}

/// Response from the platform failure inbox — mirrors `FeedbackInboxResponse`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InboxResponse {
    pub period: String,
    pub items: Vec<InboxItem>,
    pub total: i64,
    #[serde(default)]
    pub limit: i64,
    #[serde(default)]
    pub offset: i64,
    pub summary: InboxSummary,
    #[serde(default)]
    pub data_source: String,
}

/// Query filters for an inbox fetch (all optional; the server applies defaults).
#[derive(Debug, Clone, Default)]
pub struct InboxQuery {
    /// `1d` | `7d` | `30d` | `all`.
    pub period: Option<String>,
    pub model_id: Option<String>,
    /// `report` | `signal` | `all`.
    pub source: Option<String>,
    /// `up` | `down`.
    pub rating: Option<String>,
    pub limit: Option<u32>,
}

/// Read-only client for the platform failure inbox. Mirrors the
/// [`RegistryClient`](crate::registry_client::RegistryClient) HTTP conventions
/// (blocking `ureq` agent with connect/request timeouts).
pub struct InboxClient {
    base_url: String,
    api_key: String,
    agent: ureq::Agent,
}

impl InboxClient {
    /// A client targeting `base_url` authenticated with a platform API key
    /// (`xy_live_…`). A trailing slash on `base_url` is tolerated.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(CONNECT_TIMEOUT_MS))
            .timeout(Duration::from_millis(REQUEST_TIMEOUT_MS))
            .build();
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            agent,
        }
    }

    /// Build from the environment: `XYBRID_API_KEY` (required) + the API base
    /// (optional; defaults to the production API). Returns `None` when no API
    /// key is set so the caller can surface the auth hint.
    ///
    /// The base URL is resolved from `XYBRID_API_URL` first (the console-style
    /// var), then `XYBRID_PLATFORM_URL` (the env form of the `xybrid` CLI's
    /// global `--platform-url`), then the production default. (The CLI itself
    /// builds the client with the already-resolved flag value via
    /// [`InboxClient::new`]; this constructor is the env-only convenience path
    /// for SDK callers.)
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("XYBRID_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())?;
        let base_url = std::env::var("XYBRID_API_URL")
            .ok()
            .filter(|u| !u.is_empty())
            .or_else(|| {
                std::env::var("XYBRID_PLATFORM_URL")
                    .ok()
                    .filter(|u| !u.is_empty())
            })
            .unwrap_or_else(|| DEFAULT_API_URL.to_string());
        Some(Self::new(base_url, api_key))
    }

    /// The resolved endpoint URL (exposed for diagnostics / tests).
    pub fn endpoint(&self) -> String {
        format!(
            "{}/v1/telemetry/feedback",
            self.base_url.trim_end_matches('/')
        )
    }

    /// Fetch the failure inbox. Network/transport failures map to
    /// [`SdkError::Offline`]; non-2xx and parse failures to
    /// [`SdkError::NetworkError`].
    pub fn fetch(&self, query: &InboxQuery) -> Result<InboxResponse, SdkError> {
        let mut req = self
            .agent
            .get(&self.endpoint())
            .set("Authorization", &format!("Bearer {}", self.api_key));
        if let Some(p) = &query.period {
            req = req.query("period", p);
        }
        if let Some(m) = &query.model_id {
            req = req.query("model_id", m);
        }
        if let Some(s) = &query.source {
            req = req.query("source", s);
        }
        if let Some(r) = &query.rating {
            req = req.query("rating", r);
        }
        if let Some(l) = query.limit {
            req = req.query("limit", &l.to_string());
        }

        match req.call() {
            Ok(resp) => resp
                .into_json::<InboxResponse>()
                .map_err(|e| SdkError::network(format!("failed to parse inbox response: {e}"))),
            Err(ureq::Error::Status(status, _)) => Err(SdkError::network(format!(
                "failure inbox request failed: HTTP {status}"
            ))),
            Err(ureq::Error::Transport(t)) => Err(SdkError::offline(format!(
                "failure inbox request failed: {t}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_trims_trailing_slash() {
        let c = InboxClient::new("https://api.example.test/", "xy_test_k");
        assert_eq!(
            c.endpoint(),
            "https://api.example.test/v1/telemetry/feedback"
        );
        let c2 = InboxClient::new("https://api.example.test", "xy_test_k");
        assert_eq!(
            c2.endpoint(),
            "https://api.example.test/v1/telemetry/feedback"
        );
    }

    #[test]
    fn deserializes_platform_response() {
        // Mirrors the exact JSON the platform `FeedbackInboxResponse` serializes.
        let json = r#"{
            "period": "7d",
            "items": [
                {
                    "id": "evt_1",
                    "trace_id": "tr_9a31",
                    "created_at": "2026-06-27T10:00:00.000Z",
                    "model_id": "qwen3.5-0.8b",
                    "task": "chat",
                    "source": "report",
                    "rating": "down",
                    "signal_kind": null,
                    "signal_name": null,
                    "expected": "Decline politely.",
                    "note": "Hallucinated a policy",
                    "payload_captured": true,
                    "device_id": "mirage-vault",
                    "platform": "macos-arm64"
                },
                {
                    "id": "evt_2",
                    "trace_id": null,
                    "created_at": "2026-06-27T09:00:00.000Z",
                    "model_id": "qwen3.5-0.8b",
                    "task": "chat",
                    "source": "signal",
                    "rating": null,
                    "signal_kind": "structural",
                    "signal_name": "truncated",
                    "payload_captured": false,
                    "device_id": null,
                    "platform": null
                }
            ],
            "total": 2,
            "limit": 50,
            "offset": 0,
            "summary": {
                "total": 2,
                "down_count": 1,
                "up_count": 0,
                "signal_count": 1,
                "negative_rate": 1.0,
                "by_model": [{ "key": "qwen3.5-0.8b", "count": 2 }],
                "top_signals": [{ "key": "truncated", "count": 1 }]
            },
            "data_source": "telemetry"
        }"#;

        let resp: InboxResponse = serde_json::from_str(json).expect("parses");
        assert_eq!(resp.period, "7d");
        assert_eq!(resp.total, 2);
        assert_eq!(resp.items.len(), 2);

        let reported = &resp.items[0];
        assert_eq!(reported.source, "report");
        assert_eq!(reported.rating.as_deref(), Some("down"));
        assert_eq!(reported.expected.as_deref(), Some("Decline politely."));
        assert!(reported.payload_captured);

        let signal = &resp.items[1];
        assert_eq!(signal.source, "signal");
        assert_eq!(signal.signal_name.as_deref(), Some("truncated"));
        assert!(signal.trace_id.is_none());
        assert!(!signal.payload_captured);

        assert_eq!(resp.summary.down_count, 1);
        assert_eq!(resp.summary.signal_count, 1);
        assert_eq!(resp.summary.negative_rate, Some(1.0));
        assert_eq!(resp.summary.by_model[0].key, "qwen3.5-0.8b");
        assert_eq!(resp.summary.top_signals[0].key, "truncated");
    }

    #[test]
    fn tolerates_minimal_response() {
        // Defaults keep an older/leaner server response parseable.
        let json = r#"{
            "period": "all",
            "items": [],
            "total": 0,
            "summary": { "total": 0, "down_count": 0, "up_count": 0, "signal_count": 0 }
        }"#;
        let resp: InboxResponse = serde_json::from_str(json).expect("parses");
        assert!(resp.items.is_empty());
        assert_eq!(resp.summary.total, 0);
        assert!(resp.summary.by_model.is_empty());
        assert_eq!(resp.data_source, "");
    }
}
