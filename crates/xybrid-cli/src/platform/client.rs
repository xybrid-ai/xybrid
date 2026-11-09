//! The platform's agent-signup HTTP contract (`/v1/agent/*`, platform #332).
//!
//! | Call | Auth | Result |
//! |---|---|---|
//! | `GET /v1/agent/challenge` | none | [`Challenge`] |
//! | `POST /v1/agent/connect` | proof of work + `Idempotency-Key` | [`Provision`] with a claim token |
//! | `POST /v1/agent/projects` | claim token (body) + `Idempotency-Key` | [`Provision`] for another project |
//! | `GET /v1/agent/workspace` | `Authorization: Bearer <key>` | [`WorkspaceStatus`] |
//!
//! Errors are `{"error": code, "message": …, …details}`. The client never
//! follows redirects (a credential-bearing request must not be bounced to
//! another host), bounds response bodies, and honors the standard proxy
//! variables except for a loopback host (ureq 2 ignores `NO_PROXY`, and a
//! plain-HTTP local platform must not go through a proxy).

use std::io::Read;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::credentials::{NamedRef, Secret};
use super::host::PlatformHost;

/// Response bodies are a few hundred bytes; never buffer more than this.
const MAX_RESPONSE_BYTES: u64 = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// `GET /v1/agent/challenge`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Challenge {
    pub challenge: String,
    pub difficulty: u8,
    #[serde(default)]
    pub algorithm: Option<String>,
}

/// The allowlisted, non-identifying client facts sent with `connect`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ClientInfo {
    pub cli_version: &'static str,
    pub os: &'static str,
    pub arch: &'static str,
    pub binding: &'static str,
}

impl ClientInfo {
    pub(crate) fn current() -> Self {
        Self {
            cli_version: env!("CARGO_PKG_VERSION"),
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            binding: "cli",
        }
    }
}

/// `POST /v1/agent/connect` body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ConnectBody<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub install_id: Option<&'a str>,
    pub client_info: ClientInfo,
    pub challenge: &'a str,
    pub nonce: &'a str,
}

/// Gateway and project limits for a provisional workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Limits {
    #[serde(default)]
    pub gateway_daily_requests: Option<u32>,
    #[serde(default)]
    pub max_projects: Option<u32>,
}

/// New credentials from `connect` or `projects`. The key is shown once.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Provision {
    pub api_key: Secret,
    #[serde(default)]
    pub scope: Option<String>,
    pub organization: NamedRef,
    pub project: NamedRef,
    pub key_expires_at: DateTime<Utc>,
    pub claim_expires_at: DateTime<Utc>,
    /// Only from `connect`; `projects` callers already hold it.
    #[serde(default)]
    pub claim_token: Option<Secret>,
    #[serde(default)]
    pub limits: Option<Limits>,
    #[serde(default)]
    pub terms_url: Option<String>,
    #[serde(default)]
    pub privacy_url: Option<String>,
    /// An `Idempotency-Key` retry re-issued these (the old key is revoked).
    #[serde(default)]
    pub replayed: bool,
}

/// Whether OTLP spans from the key's project have been stored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Verification {
    /// `received`, `none` or `unavailable` (unknown right now, not `none`).
    pub state: String,
    #[serde(default)]
    pub traces_received: Option<bool>,
    #[serde(default)]
    pub spans: Option<u64>,
    #[serde(default)]
    pub traces: Option<u64>,
    #[serde(default)]
    pub first_ingested_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_ingested_at: Option<DateTime<Utc>>,
}

/// `GET /v1/agent/workspace`: lifecycle and verification for the calling key.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WorkspaceStatus {
    pub organization: NamedRef,
    pub project: NamedRef,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub provisional: bool,
    #[serde(default)]
    pub claimed: bool,
    #[serde(default)]
    pub expired: bool,
    #[serde(default)]
    pub key_expires_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub claim_expires_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub verification: Option<Verification>,
}

/// Why a platform call failed.
#[derive(Debug, Clone)]
pub(crate) enum ApiFailure {
    /// The host could not be reached (DNS, connect, TLS, timeout).
    Offline(String),
    /// A 3xx answer; never followed.
    Redirect { status: u16 },
    /// A 4xx/5xx answer.
    Status {
        status: u16,
        code: Option<String>,
        message: Option<String>,
        retry_after: Option<u64>,
        /// The whole error body (`reason`, `min_cli_version`, deadlines, …).
        details: Value,
        /// `X-Xybrid-Workspace`, e.g. `expired`.
        workspace: Option<String>,
    },
    /// A 2xx answer that does not match the contract.
    Invalid(String),
}

impl ApiFailure {
    /// The platform's error code, if it sent one.
    pub(crate) fn code(&self) -> Option<&str> {
        match self {
            ApiFailure::Status { code, .. } => code.as_deref(),
            _ => None,
        }
    }

    /// A human-readable description without secrets.
    pub(crate) fn describe(&self) -> String {
        match self {
            ApiFailure::Offline(detail) => format!("could not reach the platform ({detail})"),
            ApiFailure::Redirect { status } => format!(
                "the platform answered with a redirect (HTTP {status}); redirects are never \
                 followed for credential requests"
            ),
            ApiFailure::Status {
                status,
                code,
                message,
                ..
            } => match (code, message) {
                (Some(code), Some(message)) => format!("{message} (HTTP {status}, {code})"),
                (Some(code), None) => format!("HTTP {status} ({code})"),
                _ => format!("HTTP {status}"),
            },
            ApiFailure::Invalid(detail) => format!("unexpected platform response ({detail})"),
        }
    }

    /// Whether the outcome is unknown, so the request may have been applied
    /// (keep idempotency state and retry with the same key).
    pub(crate) fn outcome_unknown(&self) -> bool {
        match self {
            ApiFailure::Offline(_) | ApiFailure::Invalid(_) => true,
            ApiFailure::Redirect { .. } => false,
            ApiFailure::Status { status, .. } => *status >= 500,
        }
    }
}

/// A client for one pinned or explicitly allowed platform origin.
pub(crate) struct PlatformClient {
    agent: ureq::Agent,
    host: PlatformHost,
}

impl PlatformClient {
    pub(crate) fn new(host: PlatformHost) -> Self {
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .timeout_connect(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .user_agent(&format!("xybrid-cli/{}", env!("CARGO_PKG_VERSION")))
            .try_proxy_from_env(!host.is_loopback())
            .build();
        Self { agent, host }
    }

    pub(crate) fn challenge(&self) -> Result<Challenge, ApiFailure> {
        let request = self.agent.get(&self.host.endpoint("/v1/agent/challenge"));
        decode(finish(request.set("Accept", "application/json").call())?)
    }

    pub(crate) fn connect(
        &self,
        idempotency_key: &Secret,
        body: &ConnectBody<'_>,
    ) -> Result<Provision, ApiFailure> {
        let body = serde_json::to_value(body).map_err(|e| ApiFailure::Invalid(e.to_string()))?;
        let request = self
            .agent
            .post(&self.host.endpoint("/v1/agent/connect"))
            .set("Accept", "application/json")
            .set("Idempotency-Key", idempotency_key.expose());
        decode(finish(request.send_json(body))?)
    }

    pub(crate) fn create_project(
        &self,
        idempotency_key: &Secret,
        claim_token: &Secret,
        project_name: Option<&str>,
    ) -> Result<Provision, ApiFailure> {
        let mut body = serde_json::json!({ "claim_token": claim_token.expose() });
        if let Some(name) = project_name {
            body["project_name"] = Value::String(name.to_string());
        }
        let request = self
            .agent
            .post(&self.host.endpoint("/v1/agent/projects"))
            .set("Accept", "application/json")
            .set("Idempotency-Key", idempotency_key.expose());
        decode(finish(request.send_json(body))?)
    }

    pub(crate) fn workspace(&self, api_key: &Secret) -> Result<WorkspaceStatus, ApiFailure> {
        let request = self
            .agent
            .get(&self.host.endpoint("/v1/agent/workspace"))
            .set("Accept", "application/json")
            .set("Authorization", &format!("Bearer {}", api_key.expose()));
        decode(finish(request.call())?)
    }
}

/// Turn a ureq result into a 2xx body or an [`ApiFailure`].
fn finish(result: Result<ureq::Response, ureq::Error>) -> Result<Vec<u8>, ApiFailure> {
    match result {
        Ok(response) if (200..300).contains(&response.status()) => read_body(response),
        Ok(response) => Err(ApiFailure::Redirect {
            status: response.status(),
        }),
        Err(ureq::Error::Status(status, response)) => {
            let retry_after = response
                .header("retry-after")
                .and_then(|v| v.trim().parse::<u64>().ok());
            let workspace = response
                .header("x-xybrid-workspace")
                .map(|v| v.chars().filter(|c| !c.is_control()).take(128).collect());
            let details: Value = read_body(response)
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or(Value::Null);
            let text = |field: &str| {
                details
                    .get(field)
                    .and_then(Value::as_str)
                    .map(|s| s.chars().filter(|c| !c.is_control()).take(300).collect())
            };
            Err(ApiFailure::Status {
                status,
                code: text("error"),
                message: text("message"),
                retry_after,
                details: details.clone(),
                workspace,
            })
        }
        Err(ureq::Error::Transport(transport)) => Err(ApiFailure::Offline(transport.to_string())),
    }
}

fn read_body(response: ureq::Response) -> Result<Vec<u8>, ApiFailure> {
    let mut body = Vec::new();
    response
        .into_reader()
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut body)
        .map_err(|e| ApiFailure::Offline(e.to_string()))?;
    if body.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(ApiFailure::Invalid("response too large".into()));
    }
    Ok(body)
}

fn decode<T: serde::de::DeserializeOwned>(body: Vec<u8>) -> Result<T, ApiFailure> {
    // serde_json's messages quote no input values, so no secret can leak here.
    serde_json::from_slice(&body).map_err(|e| ApiFailure::Invalid(e.to_string()))
}

/// A server-supplied link worth showing: an absolute http(s) URL.
pub(crate) fn safe_link(raw: Option<&str>) -> Option<String> {
    let raw = raw?.trim();
    let url = url::Url::parse(raw).ok()?;
    matches!(url.scheme(), "https" | "http").then(|| url.to_string())
}

/// A server-supplied display name, cleaned for terminals and files.
pub(crate) fn safe_name(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provision_debug_redacts_secrets() {
        let provision: Provision = serde_json::from_value(serde_json::json!({
            "api_key": "sk_live_SECRETKEYSECRETKEY",
            "key_prefix": "SECRETKE",
            "scope": "client",
            "organization": {"slug": "my-app", "name": "my-app"},
            "project": {"slug": "default", "name": "Default"},
            "key_expires_at": "2026-11-15T12:00:00Z",
            "claim_expires_at": "2026-11-22T12:00:00Z",
            "claim_token": "CLAIMCLAIMCLAIMCLAIMCLAIMCLAIMCLAIMCLAIM123",
            "limits": {"gateway_daily_requests": 25, "max_projects": 5},
            "replayed": false
        }))
        .unwrap();
        let debug = format!("{provision:?}");
        assert!(!debug.contains("SECRETKEYSECRET"), "{debug}");
        assert!(!debug.contains("CLAIMCLAIM"), "{debug}");
    }

    #[test]
    fn unknown_outcomes_keep_idempotency_state() {
        let status = |status| ApiFailure::Status {
            status,
            code: None,
            message: None,
            retry_after: None,
            details: Value::Null,
            workspace: None,
        };
        assert!(ApiFailure::Offline("timeout".into()).outcome_unknown());
        assert!(status(502).outcome_unknown());
        assert!(!status(403).outcome_unknown());
        assert!(!ApiFailure::Redirect { status: 302 }.outcome_unknown());
    }

    #[test]
    fn server_text_is_sanitized() {
        assert_eq!(safe_name("  my\napp\u{1b}[31m  "), "my app [31m");
        assert_eq!(
            safe_link(Some("https://www.xybrid.ai/terms")).as_deref(),
            Some("https://www.xybrid.ai/terms")
        );
        assert_eq!(safe_link(Some("javascript:alert(1)")), None);
        assert_eq!(safe_link(Some("not a url")), None);
        assert_eq!(safe_link(None), None);
    }
}
