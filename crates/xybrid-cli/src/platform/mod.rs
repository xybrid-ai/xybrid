//! Xybrid platform credentials for `xybrid connect` and `xybrid auth`.
//!
//! The CLI's on-device commands never need an account. This module is the
//! additive platform half: it provisions a claimable workspace for a project,
//! keeps its restricted key in a private credentials store, and resolves that
//! key for later commands.
//!
//! | Module | Contents |
//! |---|---|
//! | [`host`] | Pinned production/staging origins and the custom-host policy |
//! | [`client`] | The `/v1/agent/*` HTTP contract, with redirects disabled |
//! | [`pow`] | The challenge proof of work |
//! | [`credentials`] | `~/.xybrid/credentials.json`: locked, atomic, private |
//! | [`install_id`] | A random installation ID, sent only to `connect` |
//! | [`git`] | Repository roots, tracked-file checks and `.gitignore` entries |
//! | [`env_file`] | Writers for dotenv, Java properties and xcconfig files |
//! | [`resolve`] | The one key resolver: flag, then env, then stored credentials |
//! | [`notices`] | The daily claim reminder and the no-export tip (stderr only) |
//! | [`report`] | The JSON/human result shared by `connect` and `auth`, and exit codes |
//!
//! Nothing here starts an exporter: provisioning a key and exporting traces
//! are separate decisions, and `--observability` stays explicit.

pub(crate) mod client;
pub(crate) mod credentials;
pub(crate) mod env_file;
pub(crate) mod git;
pub(crate) mod host;
pub(crate) mod install_id;
pub(crate) mod notices;
pub(crate) mod pow;
pub(crate) mod report;
pub(crate) mod resolve;

use std::fmt;

/// A failure with a stable machine code, reported as `error.code` in JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CliError {
    pub code: &'static str,
    pub message: String,
}

impl CliError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CliError {}

/// The directory for the CLI's platform state (install ID, notice dates):
/// the directory of `XYBRID_CREDENTIALS_FILE` when it is set, so one variable
/// isolates everything, else `~/.xybrid`.
pub(crate) fn state_dir() -> Option<std::path::PathBuf> {
    if let Some(raw) = std::env::var_os(credentials::CREDENTIALS_FILE_ENV).filter(|v| !v.is_empty())
    {
        let path = std::path::absolute(std::path::PathBuf::from(raw)).ok()?;
        return path.parent().map(std::path::Path::to_path_buf);
    }
    dirs::home_dir().map(|home| home.join(".xybrid"))
}

/// `1`, `true` or `yes` (any case), the convention of `XYBRID_TELEMETRY_OPTOUT`.
pub(crate) fn is_truthy(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes")
    )
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Environment variables that mark a CI or build-service run.
const CI_MARKERS: &[&str] = &[
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "JENKINS_URL",
    "TF_BUILD",
    "TEAMCITY_VERSION",
    "BITBUCKET_BUILD_NUMBER",
    "CODEBUILD_BUILD_ID",
    "TRAVIS",
    "DRONE",
];

/// The first CI marker set in the environment. `CI=false` / `CI=0` don't count.
pub(crate) fn ci_marker() -> Option<&'static str> {
    CI_MARKERS.iter().copied().find(|name| {
        env(name).is_some_and(|v| {
            let v = v.trim().to_ascii_lowercase();
            v != "false" && v != "0" && v != "no"
        })
    })
}

/// The first privacy opt-out in the environment, as `VAR=value`.
///
/// `OTEL_SDK_DISABLED` follows the OpenTelemetry rule (only `true` counts);
/// the others accept `1`, `true` or `yes`.
pub(crate) fn opt_out() -> Option<String> {
    for name in ["XYBRID_TELEMETRY_OPTOUT", "DO_NOT_TRACK"] {
        if let Some(value) = env(name).filter(|v| is_truthy(Some(v))) {
            return Some(format!("{name}={}", value.trim()));
        }
    }
    if otel_sdk_disabled() {
        return Some("OTEL_SDK_DISABLED=true".to_string());
    }
    None
}

/// Whether `OTEL_SDK_DISABLED=true` (the only value the OpenTelemetry spec honors).
pub(crate) fn otel_sdk_disabled() -> bool {
    env("OTEL_SDK_DISABLED").is_some_and(|v| v.trim().eq_ignore_ascii_case("true"))
}

/// Whether `XYBRID_QUIET` silences reminders and tips.
pub(crate) fn xybrid_quiet() -> bool {
    is_truthy(env("XYBRID_QUIET").as_deref())
}

/// A short, non-secret preview of a key: its environment prefix plus the
/// eight characters the console also shows, e.g. `sk_live_AbCd1234…`.
pub(crate) fn key_preview(key: &str) -> String {
    let (env_prefix, rest) = ["sk_live_", "sk_test_"]
        .iter()
        .find_map(|p| key.strip_prefix(p).map(|rest| (*p, rest)))
        .unwrap_or(("", key));
    // Never reveal more than a third of an unusually short key.
    let shown = rest.chars().count().min(8).min(rest.chars().count() / 3);
    let visible: String = rest.chars().take(shown).collect();
    format!("{env_prefix}{visible}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_values_follow_the_opt_out_convention() {
        for value in ["1", "true", "TRUE", " yes "] {
            assert!(is_truthy(Some(value)), "{value}");
        }
        for value in ["0", "false", "", "no", "enabled"] {
            assert!(!is_truthy(Some(value)), "{value}");
        }
        assert!(!is_truthy(None));
    }

    #[test]
    fn key_previews_never_reveal_the_secret_part() {
        let key = concat!("sk_live", "_AbCd1234EfGh5678IjKl9012MnOp3456");
        assert_eq!(key_preview(key), "sk_live_AbCd1234…");
        assert_eq!(key_preview("sk_live_abc"), "sk_live_a…");
        assert_eq!(key_preview("short"), "s…");
        assert!(!key_preview(key).contains("EfGh"));
    }
}
