//! Pinned platform hosts.
//!
//! The production and staging API origins map to fixed console and trace
//! ingest origins, so a claim link or a stored key can never be steered to
//! another host by an injected `--platform-url` / `XYBRID_PLATFORM_URL`.
//! Anything else is a custom host: `connect` and `auth` refuse it unless
//! `--allow-custom-host` is passed, and require HTTPS unless it is loopback.

use std::net::IpAddr;

use serde::Serialize;
use url::{Host, Url};

use super::CliError;

/// Production API origin, the CLI's default `--platform-url`.
pub(crate) const PRODUCTION_API: &str = "https://api.xybrid.dev";
const PRODUCTION_CONSOLE: &str = "https://console.xybrid.dev";
const PRODUCTION_TRACES: &str = "https://ingest.xybrid.dev/v1/traces";
const STAGING_API: &str = "https://staging.api.xybrid.dev";
const STAGING_CONSOLE: &str = "https://staging.xybrid.dev";
const STAGING_TRACES: &str = "https://staging.ingest.xybrid.dev/v1/traces";
/// Console origin for a custom (`--allow-custom-host`) platform.
pub(crate) const CONSOLE_URL_ENV: &str = "XYBRID_CONSOLE_URL";

/// Which platform deployment a host is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Environment {
    Production,
    Staging,
    Custom,
}

/// A canonical platform API origin and the origins pinned to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlatformHost {
    origin: String,
    environment: Environment,
    console: Option<String>,
    traces: Option<String>,
    loopback: bool,
    https: bool,
}

impl PlatformHost {
    /// Canonicalize a platform URL. Applies no custom-host policy; see
    /// [`PlatformHost::require_allowed`].
    pub(crate) fn parse(raw: &str) -> Result<Self, CliError> {
        let url = parse_origin_url(raw, "--platform-url")?;
        let origin = url.origin().ascii_serialization();
        let (environment, console, traces) = match origin.as_str() {
            PRODUCTION_API => (
                Environment::Production,
                Some(PRODUCTION_CONSOLE.to_string()),
                Some(PRODUCTION_TRACES.to_string()),
            ),
            STAGING_API => (
                Environment::Staging,
                Some(STAGING_CONSOLE.to_string()),
                Some(STAGING_TRACES.to_string()),
            ),
            _ => (Environment::Custom, None, None),
        };
        Ok(Self {
            loopback: is_loopback(&url),
            https: url.scheme() == "https",
            origin,
            environment,
            console,
            traces,
        })
    }

    /// Refuse a custom host unless explicitly allowed, and plain HTTP unless
    /// it is loopback.
    pub(crate) fn require_allowed(&self, allow_custom_host: bool) -> Result<(), CliError> {
        if self.environment != Environment::Custom {
            return Ok(());
        }
        if !allow_custom_host {
            return Err(CliError::new(
                "custom_host_not_allowed",
                format!(
                    "{} is not a Xybrid platform host. Pass --allow-custom-host to connect to it \
                     on purpose (credentials are kept per host and never sent elsewhere).",
                    self.origin
                ),
            ));
        }
        if !self.https && !self.loopback {
            return Err(CliError::new(
                "insecure_platform_url",
                format!(
                    "{} must use https:// (plain http is only allowed for localhost)",
                    self.origin
                ),
            ));
        }
        Ok(())
    }

    /// The canonical origin, e.g. `https://api.xybrid.dev`. Credentials are keyed by it.
    pub(crate) fn origin(&self) -> &str {
        &self.origin
    }

    pub(crate) fn environment(&self) -> Environment {
        self.environment
    }

    pub(crate) fn is_custom(&self) -> bool {
        self.environment == Environment::Custom
    }

    pub(crate) fn is_loopback(&self) -> bool {
        self.loopback
    }

    /// The pinned console origin, or for a custom host the one chosen with
    /// [`PlatformHost::with_custom_console`].
    pub(crate) fn console_origin(&self) -> Option<&str> {
        self.console.as_deref()
    }

    /// The pinned OTLP trace endpoint (production and staging only).
    pub(crate) fn traces_endpoint(&self) -> Option<&str> {
        self.traces.as_deref()
    }

    /// Set a custom host's console origin. Pinned hosts keep theirs.
    pub(crate) fn with_custom_console(mut self, console: Option<&str>) -> Result<Self, CliError> {
        if self.environment != Environment::Custom {
            return Ok(self);
        }
        self.console = match console.map(str::trim).filter(|c| !c.is_empty()) {
            None => None,
            Some(raw) => {
                let url = parse_origin_url(raw, CONSOLE_URL_ENV)?;
                if url.scheme() != "https" && !is_loopback(&url) {
                    return Err(CliError::new(
                        "insecure_console_url",
                        format!("{CONSOLE_URL_ENV} must use https:// unless it is localhost"),
                    ));
                }
                Some(url.origin().ascii_serialization())
            }
        };
        Ok(self)
    }

    /// An API URL on this origin, e.g. `endpoint("/v1/agent/challenge")`.
    pub(crate) fn endpoint(&self, path: &str) -> String {
        format!("{}{}", self.origin, path)
    }

    /// `<console>/claim/<token>`, built locally from the pinned console origin.
    pub(crate) fn claim_url(&self, token: &str) -> Option<String> {
        let console = self.console.as_deref()?;
        claim_url(console, token)
    }
}

/// `<console>/claim/<token>` when the token has the shape the platform issues
/// (URL-safe base64) and the result stays on `console`.
pub(crate) fn claim_url(console: &str, token: &str) -> Option<String> {
    if !is_claim_token(token) {
        return None;
    }
    let url = format!("{}/claim/{token}", console.trim_end_matches('/'));
    let parsed = Url::parse(&url).ok()?;
    (parsed.origin().ascii_serialization() == console.trim_end_matches('/')).then_some(url)
}

/// Whether `token` has the shape the platform issues: 32-128 URL-safe
/// base64 characters (43 today), so it can sit in a path segment as-is.
pub(crate) fn is_claim_token(token: &str) -> bool {
    (32..=128).contains(&token.len())
        && token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// Whether a full trace endpoint may receive a key stored for `host`.
///
/// A pinned host only sends to its pinned ingest endpoint. A custom host
/// (allowed when its credentials were created) may send to an HTTPS or
/// loopback endpoint on the same host name, or loopback to loopback.
pub(crate) fn trace_endpoint_allowed(host: &PlatformHost, endpoint: &str) -> bool {
    let Ok(url) = Url::parse(endpoint.trim()) else {
        return false;
    };
    if let Some(pinned) = host.traces_endpoint() {
        return Url::parse(pinned).is_ok_and(|p| p == url);
    }
    if url.scheme() != "https" && !(url.scheme() == "http" && is_loopback(&url)) {
        return false;
    }
    let Ok(platform) = Url::parse(host.origin()) else {
        return false;
    };
    (is_loopback(&platform) && is_loopback(&url)) || platform.host_str() == url.host_str()
}

/// Parse an `http(s)` URL that names only an origin (an optional trailing
/// `/` is fine; credentials, paths, queries and fragments are not).
fn parse_origin_url(raw: &str, what: &str) -> Result<Url, CliError> {
    let invalid = |detail: &str| {
        CliError::new(
            "invalid_platform_url",
            format!("{what} must be an http(s) origin such as {PRODUCTION_API} ({detail})"),
        )
    };
    let url = Url::parse(raw.trim()).map_err(|_| invalid("not a URL"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(invalid("unsupported scheme"));
    }
    if url.host().is_none() {
        return Err(invalid("missing host"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(invalid("embedded credentials"));
    }
    if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
        return Err(invalid("paths, queries and fragments are not allowed"));
    }
    Ok(url)
}

fn is_loopback(url: &Url) -> bool {
    match url.host() {
        Some(Host::Domain(domain)) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            domain == "localhost" || domain.ends_with(".localhost")
        }
        Some(Host::Ipv4(ip)) => IpAddr::V4(ip).is_loopback(),
        Some(Host::Ipv6(ip)) => IpAddr::V6(ip).is_loopback(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_and_staging_are_pinned() {
        for raw in [
            "https://api.xybrid.dev",
            "https://API.xybrid.dev/",
            "https://api.xybrid.dev:443",
        ] {
            let host = PlatformHost::parse(raw).unwrap();
            assert_eq!(host.origin(), PRODUCTION_API, "{raw}");
            assert_eq!(host.environment(), Environment::Production);
            assert_eq!(host.console_origin(), Some("https://console.xybrid.dev"));
            assert_eq!(
                host.traces_endpoint(),
                Some("https://ingest.xybrid.dev/v1/traces")
            );
            host.require_allowed(false).unwrap();
        }
        let staging = PlatformHost::parse("https://staging.api.xybrid.dev").unwrap();
        assert_eq!(staging.environment(), Environment::Staging);
        assert_eq!(staging.console_origin(), Some("https://staging.xybrid.dev"));
        assert_eq!(
            staging.traces_endpoint(),
            Some("https://staging.ingest.xybrid.dev/v1/traces")
        );
    }

    #[test]
    fn pinned_hosts_ignore_a_console_override() {
        let host = PlatformHost::parse(PRODUCTION_API)
            .unwrap()
            .with_custom_console(Some("https://evil.example"))
            .unwrap();
        assert_eq!(host.console_origin(), Some("https://console.xybrid.dev"));
    }

    #[test]
    fn lookalike_and_http_variants_are_custom() {
        for raw in [
            "http://api.xybrid.dev",
            "https://api.xybrid.dev:8443",
            "https://api.xybrid.dev.evil.example",
            "https://api-xybrid.dev",
        ] {
            let host = PlatformHost::parse(raw).unwrap();
            assert!(host.is_custom(), "{raw}");
            assert_eq!(host.console_origin(), None);
            assert_eq!(
                host.require_allowed(false).unwrap_err().code,
                "custom_host_not_allowed"
            );
        }
    }

    #[test]
    fn custom_hosts_need_https_unless_loopback() {
        for raw in [
            "http://localhost:3001",
            "http://127.0.0.1:3001",
            "http://[::1]:3001",
            "http://app.localhost",
            "https://platform.example.test",
        ] {
            PlatformHost::parse(raw)
                .unwrap()
                .require_allowed(true)
                .unwrap_or_else(|e| panic!("{raw}: {e}"));
        }
        let plain = PlatformHost::parse("http://platform.example.test").unwrap();
        assert_eq!(
            plain.require_allowed(true).unwrap_err().code,
            "insecure_platform_url"
        );
    }

    #[test]
    fn platform_urls_must_be_origins() {
        for raw in [
            "api.xybrid.dev",
            "ftp://api.xybrid.dev",
            "https://user:pw@api.xybrid.dev",
            "https://api.xybrid.dev/v1",
            "https://api.xybrid.dev/?x=1",
            "https://api.xybrid.dev/#frag",
            "not a url",
        ] {
            assert_eq!(
                PlatformHost::parse(raw).unwrap_err().code,
                "invalid_platform_url",
                "{raw}"
            );
        }
    }

    #[test]
    fn claim_urls_are_built_locally_and_validated() {
        let host = PlatformHost::parse(PRODUCTION_API).unwrap();
        let token = "a".repeat(43);
        assert_eq!(
            host.claim_url(&token).unwrap(),
            format!("https://console.xybrid.dev/claim/{token}")
        );
        for bad in [
            "short",
            "../../evil.example/claim/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            &format!("{token}?x"),
        ] {
            assert_eq!(host.claim_url(bad), None, "{bad}");
        }
        let custom = PlatformHost::parse("http://localhost:3001").unwrap();
        assert_eq!(custom.claim_url(&token), None, "no console configured");
        let custom = custom
            .with_custom_console(Some("http://localhost:5173/"))
            .unwrap();
        assert_eq!(
            custom.claim_url(&token).unwrap(),
            format!("http://localhost:5173/claim/{token}")
        );
        assert_eq!(
            PlatformHost::parse("http://localhost:3001")
                .unwrap()
                .with_custom_console(Some("http://console.example.test"))
                .unwrap_err()
                .code,
            "insecure_console_url"
        );
    }

    #[test]
    fn stored_keys_only_reach_matching_trace_endpoints() {
        let prod = PlatformHost::parse(PRODUCTION_API).unwrap();
        assert!(trace_endpoint_allowed(
            &prod,
            "https://ingest.xybrid.dev/v1/traces"
        ));
        assert!(!trace_endpoint_allowed(
            &prod,
            "https://collector.example.test/v1/traces"
        ));
        assert!(!trace_endpoint_allowed(
            &prod,
            "https://staging.ingest.xybrid.dev/v1/traces"
        ));

        let local = PlatformHost::parse("http://localhost:3001").unwrap();
        assert!(trace_endpoint_allowed(
            &local,
            "http://localhost:8081/v1/traces"
        ));
        assert!(trace_endpoint_allowed(
            &local,
            "http://127.0.0.1:8081/v1/traces"
        ));
        assert!(!trace_endpoint_allowed(
            &local,
            "https://collector.example.test/v1/traces"
        ));

        let custom = PlatformHost::parse("https://platform.example.test").unwrap();
        assert!(trace_endpoint_allowed(
            &custom,
            "https://platform.example.test/ingest/v1/traces"
        ));
        assert!(!trace_endpoint_allowed(
            &custom,
            "http://platform.example.test/v1/traces"
        ));
        assert!(!trace_endpoint_allowed(&custom, "not a url"));
    }
}
