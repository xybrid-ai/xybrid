//! Structured results of `connect` and `auth`, as JSON or human text.
//!
//! Every JSON object carries `schema_version`, `status` and
//! `tell_the_user`. It never contains a raw key: only a preview. The claim
//! link appears only where recovery was explicitly requested (`connect`,
//! `auth claim`, `auth status --show-claim-url`).
//!
//! | Exit | Status |
//! |---|---|
//! | 0 | `created`, `existing`, `connected`, `claimable`, `claimed`, `logged_out` |
//! | 3 | `signup_closed` |
//! | 4 | `offline` |
//! | 5 | `ci_skipped`, `opted_out` |
//! | 6 | `env_file_tracked` |
//! | 7 | `workspace_expired` |
//! | 1 | `error` / `not_connected` (`error.code` says why) |

use chrono::{DateTime, Utc};
use serde::Serialize;

use super::client::{Limits, Verification};
use super::credentials::NamedRef;
use super::env_file::EnvFormat;
use super::host::{Environment, PlatformHost};
use super::notices::human_date;
use super::resolve::KeySource;
use super::CliError;

/// Bumped only for incompatible changes to the JSON shape.
pub(crate) const SCHEMA_VERSION: u32 = 1;

/// The outcome, which also fixes the exit code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Status {
    Created,
    Existing,
    Connected,
    Claimable,
    Claimed,
    LoggedOut,
    SignupClosed,
    Offline,
    CiSkipped,
    OptedOut,
    EnvFileTracked,
    WorkspaceExpired,
    NotConnected,
    Error,
}

impl Status {
    pub(crate) fn exit_code(self) -> i32 {
        match self {
            Status::Created
            | Status::Existing
            | Status::Connected
            | Status::Claimable
            | Status::Claimed
            | Status::LoggedOut => 0,
            Status::SignupClosed => 3,
            Status::Offline => 4,
            Status::CiSkipped | Status::OptedOut => 5,
            Status::EnvFileTracked => 6,
            Status::WorkspaceExpired => 7,
            Status::NotConnected | Status::Error => 1,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ErrorReport {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PlatformReport {
    pub api_origin: String,
    pub environment: Environment,
    pub console_origin: Option<String>,
    pub traces_endpoint: Option<String>,
    /// A `--platform-url` override that is neither production nor staging.
    pub custom_host: bool,
}

impl PlatformReport {
    pub(crate) fn from_host(host: &PlatformHost) -> Self {
        Self {
            api_origin: host.origin().to_string(),
            environment: host.environment(),
            console_origin: host.console_origin().map(str::to_string),
            traces_endpoint: host.traces_endpoint().map(str::to_string),
            custom_host: host.is_custom(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct KeyReport {
    /// `created`, `credentials`, `env_file`, `env` or `flag`.
    pub source: &'static str,
    /// Prefix only, e.g. `sk_live_AbCd1234…`; never the key.
    pub preview: Option<String>,
    pub scope: Option<String>,
}

impl KeyReport {
    pub(crate) fn source_name(source: KeySource) -> &'static str {
        match source {
            KeySource::Flag => "flag",
            KeySource::Env => "env",
            KeySource::Credentials => "credentials",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EnvFileReport {
    pub path: String,
    pub variable: String,
    pub format: EnvFormat,
    /// The key was written by this run.
    pub written: bool,
    /// The file already set a nonempty value (never replaced).
    pub already_configured: bool,
    pub tracked: Option<bool>,
    /// `git check-ignore` confirmed it; false outside a Git repository.
    pub gitignored: bool,
    pub gitignore_updated: bool,
    /// Untracked alternatives when the target is tracked.
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CredentialsReport {
    pub path: String,
    pub saved: bool,
    /// A claim token is stored, so `xybrid auth claim` can recover the link.
    pub recovery: bool,
    pub moved_corrupt_file: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct WorkspaceReport {
    pub organization: Option<NamedRef>,
    pub project: Option<NamedRef>,
    pub provisional: Option<bool>,
    pub claimed: Option<bool>,
    pub expired: Option<bool>,
    pub key_expires_at: Option<DateTime<Utc>>,
    pub claim_expires_at: Option<DateTime<Utc>>,
    /// `server` (just checked), `provision` (from the create response) or
    /// `stored` (saved deadlines, a fallback when the server can't be asked).
    pub lifecycle_source: Option<&'static str>,
    pub new_project_in_existing_workspace: bool,
    pub replayed: bool,
    pub limits: Option<Limits>,
    pub terms_url: Option<String>,
    pub privacy_url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct VerificationReport {
    /// `received`, `none`, `unavailable` (server can't tell right now) or
    /// `not_checked`.
    pub state: String,
    pub traces_received: Option<bool>,
    pub spans: Option<u64>,
    pub traces: Option<u64>,
    pub first_ingested_at: Option<DateTime<Utc>>,
    pub last_ingested_at: Option<DateTime<Utc>>,
}

impl VerificationReport {
    pub(crate) fn not_checked() -> Self {
        Self::state("not_checked")
    }

    pub(crate) fn state(state: &str) -> Self {
        Self {
            state: state.to_string(),
            traces_received: None,
            spans: None,
            traces: None,
            first_ingested_at: None,
            last_ingested_at: None,
        }
    }

    pub(crate) fn from_server(verification: Option<&Verification>) -> Self {
        match verification {
            None => Self::state("unavailable"),
            Some(v) => Self {
                state: match v.state.as_str() {
                    "received" | "none" | "unavailable" => v.state.clone(),
                    _ => "unavailable".into(),
                },
                traces_received: v.traces_received,
                spans: v.spans,
                traces: v.traces,
                first_ingested_at: v.first_ingested_at,
                last_ingested_at: v.last_ingested_at,
            },
        }
    }
}

/// Saved deadlines, reported apart from the server's answer.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct StoredReport {
    pub key_expires_at: Option<DateTime<Utc>>,
    pub claim_expires_at: Option<DateTime<Utc>>,
    pub claimed: bool,
    pub recovery: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct NextStep {
    /// `wire_init`, `verify` or `relay`.
    pub kind: &'static str,
    pub summary: String,
    pub command: Option<String>,
}

/// One command's result.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Report {
    pub schema_version: u32,
    pub command: &'static str,
    pub status: Status,
    pub error: Option<ErrorReport>,
    pub platform: Option<PlatformReport>,
    pub key: Option<KeyReport>,
    pub env_file: Option<EnvFileReport>,
    pub credentials: Option<CredentialsReport>,
    pub workspace: Option<WorkspaceReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored: Option<StoredReport>,
    pub claim_url: Option<String>,
    pub verification: Option<VerificationReport>,
    /// `auth claim`: whether a browser was opened.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opened_browser: Option<bool>,
    /// `auth logout`: whether saved credentials were removed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removed: Option<bool>,
    pub next_steps: Vec<NextStep>,
    pub tell_the_user: String,
}

impl Report {
    pub(crate) fn new(command: &'static str) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            command,
            status: Status::Error,
            error: None,
            platform: None,
            key: None,
            env_file: None,
            credentials: None,
            workspace: None,
            stored: None,
            claim_url: None,
            verification: None,
            opened_browser: None,
            removed: None,
            next_steps: Vec::new(),
            tell_the_user: String::new(),
        }
    }

    /// Record an error outcome.
    pub(crate) fn fail(&mut self, err: &CliError) {
        self.status = Status::Error;
        self.error = Some(ErrorReport {
            code: err.code.to_string(),
            message: err.message.clone(),
        });
        if self.tell_the_user.is_empty() {
            self.tell_the_user = format!(
                "Couldn't finish setting up the Xybrid console connection: {} \
                 On-device inference works without it.",
                ensure_period(&err.message)
            );
        }
    }

    /// Print the report and return the process exit code.
    pub(crate) fn emit(&self, json: bool, quiet: bool) -> i32 {
        if json {
            match serde_json::to_string_pretty(self) {
                Ok(text) => println!("{text}"),
                Err(err) => {
                    eprintln!("could not serialize the result: {err}");
                    return 1;
                }
            }
        } else {
            self.print_human(quiet);
        }
        self.status.exit_code()
    }

    fn print_human(&self, quiet: bool) {
        use crate::ui;
        if quiet {
            println!("{}", self.tell_the_user);
            return;
        }
        println!();
        match self.status {
            Status::Created
            | Status::Existing
            | Status::Connected
            | Status::Claimable
            | Status::Claimed
            | Status::LoggedOut => ui::ok(&headline(self)),
            Status::Error | Status::NotConnected => ui::err(&headline(self)),
            _ => ui::warning(&headline(self)),
        }
        if let Some(platform) = self.platform.as_ref().filter(|p| p.custom_host) {
            ui::kv(
                "Platform",
                &format!("{} (custom host)", platform.api_origin),
            );
        }
        if let Some(workspace) = &self.workspace {
            if let Some(org) = &workspace.organization {
                let project = workspace
                    .project
                    .as_ref()
                    .map(|p| format!(" / {}", p.name))
                    .unwrap_or_default();
                ui::kv("Workspace", &format!("{}{project}", org.name));
            }
            if workspace.provisional == Some(true) {
                if let Some(at) = workspace.key_expires_at {
                    ui::kv("Key stops", &format!("{} unless claimed", human_date(at)));
                }
                if let Some(at) = workspace.claim_expires_at {
                    ui::kv("Deleted", &format!("{} unless claimed", human_date(at)));
                }
            }
        }
        if let Some(key) = &self.key {
            let preview = key.preview.as_deref().unwrap_or("(hidden)");
            ui::kv("Key", &format!("{preview} from {}", key.source));
        }
        if let Some(env) = &self.env_file {
            let ignored = if env.gitignored {
                "gitignored"
            } else {
                "not gitignored"
            };
            ui::kv(
                "Project file",
                &format!("{} ({}, {ignored})", env.path, env.variable),
            );
        }
        if let Some(credentials) = &self.credentials {
            ui::kv("Credentials", &credentials.path);
        }
        if let Some(verification) = &self.verification {
            if verification.state != "not_checked" {
                ui::kv("Traces", &verification.state);
            }
        }
        if let Some(url) = &self.claim_url {
            ui::kv("Claim link", &format!("{url} (private)"));
        }
        if let Some(error) = &self.error {
            ui::kv("Error", &format!("{} ({})", error.message, error.code));
        }
        println!();
        println!("  {}", self.tell_the_user);
        println!();
    }
}

fn headline(report: &Report) -> String {
    match report.status {
        Status::Created => "Connected this project to a provisional Xybrid workspace".into(),
        Status::Existing => "This project already has a Xybrid key".into(),
        Status::Connected => "Xybrid key is valid".into(),
        Status::Claimable => "Claim link".into(),
        Status::Claimed => "Workspace already claimed".into(),
        Status::LoggedOut => "Saved credentials".into(),
        Status::SignupClosed => "Agent signup is closed".into(),
        Status::Offline => "Couldn't reach Xybrid".into(),
        Status::CiSkipped => "Skipped: CI environment".into(),
        Status::OptedOut => "Skipped: telemetry opt-out".into(),
        Status::EnvFileTracked => "Skipped: the project file is tracked by Git".into(),
        Status::WorkspaceExpired => "The workspace's key has expired".into(),
        Status::NotConnected => "No Xybrid key for this project".into(),
        Status::Error => "Error".into(),
    }
}

/// `text` ending in sentence punctuation.
pub(crate) fn ensure_period(text: &str) -> String {
    let text = text.trim();
    if text.ends_with(['.', '!', '?']) {
        text.to_string()
    } else {
        format!("{text}.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_the_contract() {
        for (status, code) in [
            (Status::Created, 0),
            (Status::Existing, 0),
            (Status::SignupClosed, 3),
            (Status::Offline, 4),
            (Status::CiSkipped, 5),
            (Status::OptedOut, 5),
            (Status::EnvFileTracked, 6),
            (Status::WorkspaceExpired, 7),
            (Status::Error, 1),
        ] {
            assert_eq!(status.exit_code(), code, "{status:?}");
        }
    }

    #[test]
    fn json_has_a_stable_shape() {
        let mut report = Report::new("connect");
        report.fail(&CliError::new("custom_host_not_allowed", "nope"));
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["status"], "error");
        assert_eq!(value["error"]["code"], "custom_host_not_allowed");
        for field in [
            "platform",
            "key",
            "env_file",
            "credentials",
            "workspace",
            "claim_url",
        ] {
            assert!(value.get(field).is_some(), "{field} is always present");
        }
        assert!(value["tell_the_user"].as_str().unwrap().contains("nope."));
    }
}
