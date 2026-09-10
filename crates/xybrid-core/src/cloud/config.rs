//! Cloud client configuration types.

use serde::{Deserialize, Serialize};
use std::sync::RwLock;

/// Programmatically-set Xybrid gateway API key, held in process memory.
///
/// Set via [`set_xybrid_api_key`] — the `xybrid_sdk::set_api_key` entry point
/// routes here. Kept out of the process environment so the secret is not
/// inherited by child processes the host app spawns.
///
/// Transitional: see issue #213 (invert the cloud composition root). Once the
/// SDK injects the cloud adapter with the key by value, this becomes a
/// last-resort fallback ahead of the `XYBRID_API_KEY` env var rather than the
/// primary programmatic channel.
static XYBRID_API_KEY: RwLock<Option<String>> = RwLock::new(None);

/// Store (or clear, with `None`) the in-memory Xybrid gateway API key.
///
/// Consulted by [`CloudConfig::resolve_api_key`] ahead of the `XYBRID_API_KEY`
/// environment variable.
pub fn set_xybrid_api_key(key: Option<String>) {
    // The lock guards a single `Option<String>`; recover a poisoned guard
    // rather than panic so credential setup can never be wedged.
    let mut guard = XYBRID_API_KEY.write().unwrap_or_else(|e| e.into_inner());
    *guard = key;
}

/// Read the in-memory Xybrid gateway API key, if one has been set.
pub fn xybrid_api_key() -> Option<String> {
    XYBRID_API_KEY
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Programmatically-set Xybrid platform base URL, held in process memory.
///
/// Set via [`set_xybrid_platform_url`] — mirrors [`XYBRID_API_KEY`]. Consulted
/// by [`default_gateway_url`] ahead of the `XYBRID_PLATFORM_URL` environment
/// variable so a host (e.g. the CLI `--platform-url` flag) can point the
/// gateway at a staging endpoint without mutating the process environment after
/// telemetry threads have spawned (a concurrent `setenv`/`getenv` is UB).
static XYBRID_PLATFORM_URL: RwLock<Option<String>> = RwLock::new(None);

/// Store (or clear, with `None`) the in-memory Xybrid platform base URL.
///
/// Consulted by [`default_gateway_url`] ahead of the `XYBRID_PLATFORM_URL`
/// environment variable. The stored value is a bare base URL (no `/v1`); the
/// gateway suffix is applied at read time.
pub fn set_xybrid_platform_url(url: Option<String>) {
    let mut guard = XYBRID_PLATFORM_URL
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *guard = url;
}

/// Read the in-memory Xybrid platform base URL, if one has been set.
pub fn xybrid_platform_url() -> Option<String> {
    XYBRID_PLATFORM_URL
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Report whether an in-memory Xybrid gateway API key has been set.
///
/// Cheaper than [`xybrid_api_key`] for presence checks — it never clones the
/// secret string.
pub fn has_xybrid_api_key() -> bool {
    XYBRID_API_KEY
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
}

/// Cloud execution backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum CloudBackend {
    /// Route through Xybrid Gateway (default, recommended).
    /// Gateway handles authentication, rate limiting, and provider routing.
    #[default]
    Gateway,

    /// Direct API calls (for development/testing only).
    /// Requires API keys in environment or config.
    /// NOT recommended for production mobile apps.
    Direct,
}

/// Cloud client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Which backend to use for cloud requests.
    #[serde(default)]
    pub backend: CloudBackend,

    /// Gateway URL (for Gateway backend).
    /// Defaults to Xybrid's hosted gateway.
    #[serde(default = "default_gateway_url")]
    pub gateway_url: String,

    /// API key for gateway authentication.
    /// Can be:
    /// - Direct value (for testing)
    /// - Environment variable reference: `$XYBRID_API_KEY`
    #[serde(default)]
    pub api_key: Option<String>,

    /// Default model to use when not specified in request.
    #[serde(default)]
    pub default_model: Option<String>,

    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u32,

    /// Enable request/response logging (for debugging).
    #[serde(default)]
    pub debug: bool,

    /// Direct provider (for Direct backend - development only).
    #[serde(default)]
    pub direct_provider: Option<String>,

    /// Base URL for the native direct client (`backend: direct`).
    ///
    /// `None` uses the provider's documented default. The cloud adapter sets
    /// this from a stage's explicit `gateway_url`; unlike `gateway_url` it is
    /// never defaulted to the platform gateway, so a native direct call
    /// cannot accidentally target the Xybrid platform.
    #[serde(default)]
    pub direct_base_url: Option<String>,
}

/// The configured Xybrid platform gateway URL (base + `/v1`).
///
/// Resolution order: `XYBRID_GATEWAY_URL`, the in-memory platform URL set via
/// [`set_xybrid_platform_url`] + `/v1`, `XYBRID_PLATFORM_URL` + `/v1`, then
/// the production default. This is also the only origin that may receive the
/// Xybrid platform API key automatically — see [`CloudConfig::resolve_api_key`].
pub fn platform_gateway_url() -> String {
    default_gateway_url()
}

/// Scheme, lowercase host and effective port of an `http(s)` URL, for
/// comparing destinations. `None` for anything unparsable or non-HTTP.
///
/// Comparing origins (never substrings or suffixes) is what stops
/// `api.xybrid.dev.evil.example` or a different port from looking like the
/// platform gateway.
pub fn url_origin(url: &str) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    let host = parsed.host_str()?.to_ascii_lowercase();
    let port = parsed.port_or_known_default()?;
    Some(format!("{}://{}:{}", parsed.scheme(), host, port))
}

/// True when both URLs parse to the same `http(s)` origin.
pub fn same_origin(a: &str, b: &str) -> bool {
    matches!((url_origin(a), url_origin(b)), (Some(x), Some(y)) if x == y)
}

/// Pure credential resolution for one destination.
///
/// Precedence:
/// 1. An explicit key wins. A literal is used as-is; a `$VAR` reference reads
///    `VAR` through `env` and, if it is unset or empty, yields `None` — it
///    never falls through to another credential.
/// 2. With no explicit key, the Xybrid platform key (`programmatic_platform_key`,
///    then `XYBRID_API_KEY` from `env`) is used **only** when `destination_url`
///    has the same origin as `platform_gateway_url`.
/// 3. Any other destination gets no automatic credential.
///
/// `env` is injected so tests never touch process environment.
pub fn resolve_api_key_for(
    explicit: Option<&str>,
    destination_url: &str,
    platform_gateway_url: &str,
    programmatic_platform_key: Option<String>,
    env: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    fn non_empty(value: Option<String>) -> Option<String> {
        value.filter(|v| !v.trim().is_empty())
    }
    if let Some(key) = explicit {
        if let Some(var) = key.strip_prefix('$') {
            return non_empty(env(var));
        }
        return non_empty(Some(key.to_string()));
    }
    if same_origin(destination_url, platform_gateway_url) {
        return non_empty(programmatic_platform_key).or_else(|| non_empty(env("XYBRID_API_KEY")));
    }
    None
}

fn default_gateway_url() -> String {
    // Priority:
    // 1. XYBRID_GATEWAY_URL env var (explicit override, should include /v1)
    // 2. In-memory platform URL (set via set_xybrid_platform_url) + /v1 suffix
    // 3. XYBRID_PLATFORM_URL env var + /v1 suffix (shared with telemetry)
    // 4. Default production URL (api.xybrid.dev/v1)
    //
    // Note: The /v1 prefix is required for OpenAI-compatible API endpoints.
    // The client appends /chat/completions, so the full path becomes /v1/chat/completions.
    if let Ok(url) = std::env::var("XYBRID_GATEWAY_URL") {
        return url;
    }
    // Programmatic platform URL (set via the SDK/CLI, held in memory) takes
    // precedence over the ambient XYBRID_PLATFORM_URL env var — same ordering as
    // the API key resolution above.
    if let Some(url) = xybrid_platform_url() {
        // Platform URL needs /v1 suffix for gateway endpoints
        return format!("{}/v1", url.trim_end_matches('/'));
    }
    if let Ok(url) = std::env::var("XYBRID_PLATFORM_URL") {
        // Platform URL needs /v1 suffix for gateway endpoints
        return format!("{}/v1", url.trim_end_matches('/'));
    }
    "https://api.xybrid.dev/v1".to_string()
}

fn default_timeout_ms() -> u32 {
    30000
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            backend: CloudBackend::default(),
            gateway_url: default_gateway_url(),
            api_key: None,
            default_model: None,
            timeout_ms: default_timeout_ms(),
            debug: false,
            direct_provider: None,
            direct_base_url: None,
        }
    }
}

impl CloudConfig {
    /// Create a new config with gateway backend.
    pub fn gateway() -> Self {
        Self {
            backend: CloudBackend::Gateway,
            ..Default::default()
        }
    }

    /// Create a new config with direct backend (development only).
    pub fn direct(provider: impl Into<String>) -> Self {
        Self {
            backend: CloudBackend::Direct,
            direct_provider: Some(provider.into()),
            ..Default::default()
        }
    }

    /// Set the gateway URL.
    pub fn with_gateway_url(mut self, url: impl Into<String>) -> Self {
        self.gateway_url = url.into();
        self
    }

    /// Set the API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, timeout_ms: u32) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Enable debug mode.
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Resolve the API key for this config's destination.
    ///
    /// An explicit `api_key` (literal or `$ENV_VAR` reference) always wins and
    /// never falls through. Without one, the Xybrid platform key — the
    /// programmatic key set via the SDK first, then the ambient
    /// `XYBRID_API_KEY` — is supplied only when `gateway_url` has the same
    /// origin as the configured platform gateway. A provider endpoint or a
    /// custom/loopback gateway never inherits the platform credential. See
    /// [`resolve_api_key_for`].
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_for(
            self.api_key.as_deref(),
            &self.gateway_url,
            &default_gateway_url(),
            xybrid_api_key(),
            |var| std::env::var(var).ok(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        // Shares the process-global gateway-URL env/cell with the precedence
        // tests below, so serialize against them.
        let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let config = CloudConfig::default();
        assert_eq!(config.backend, CloudBackend::Gateway);
        // Default URL should be api.xybrid.dev/v1 or from env vars (with /v1)
        assert!(
            config.gateway_url.contains("xybrid") || config.gateway_url.contains("localhost"),
            "gateway_url should contain 'xybrid' or 'localhost', got: {}",
            config.gateway_url
        );
        // Should end with /v1 for OpenAI-compatible endpoints
        assert!(
            config.gateway_url.ends_with("/v1") || std::env::var("XYBRID_GATEWAY_URL").is_ok(),
            "gateway_url should end with '/v1' unless XYBRID_GATEWAY_URL is set, got: {}",
            config.gateway_url
        );
    }

    #[test]
    fn test_gateway_config() {
        let config = CloudConfig::gateway()
            .with_api_key("test-key")
            .with_default_model("gpt-4o-mini");

        assert_eq!(config.backend, CloudBackend::Gateway);
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.default_model, Some("gpt-4o-mini".to_string()));
    }

    #[test]
    fn test_direct_config() {
        let config = CloudConfig::direct("openai");
        assert_eq!(config.backend, CloudBackend::Direct);
        assert_eq!(config.direct_provider, Some("openai".to_string()));
    }

    #[test]
    fn test_resolve_api_key_from_env() {
        std::env::set_var("TEST_CLOUD_KEY", "secret123");

        let config = CloudConfig::default().with_api_key("$TEST_CLOUD_KEY");
        assert_eq!(config.resolve_api_key(), Some("secret123".to_string()));

        std::env::remove_var("TEST_CLOUD_KEY");
    }

    // Serializes tests that mutate process-global state (the `XYBRID_API_KEY`
    // env var and the in-memory cell) so the parallel test runner can't race
    // them.
    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_resolve_api_key_in_memory_precedence() {
        let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // No explicit field: the in-memory cell is consulted before the env.
        std::env::set_var("XYBRID_API_KEY", "env-key");
        set_xybrid_api_key(Some("mem-key".to_string()));

        let config = CloudConfig::default();
        assert_eq!(config.resolve_api_key(), Some("mem-key".to_string()));

        // Clearing the cell falls back to the env var.
        set_xybrid_api_key(None);
        assert_eq!(config.resolve_api_key(), Some("env-key".to_string()));

        // An explicit config field still wins over both.
        let explicit = CloudConfig::default().with_api_key("field-key");
        assert_eq!(explicit.resolve_api_key(), Some("field-key".to_string()));

        std::env::remove_var("XYBRID_API_KEY");
    }

    #[test]
    fn test_gateway_url_in_memory_precedence() {
        let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        // RAII reset: clear the process-global cell + env vars on drop, so a
        // failing assertion below can't leak state into `test_default_config`
        // (which reads the same shared cell) and cascade into a false failure.
        struct ResetOnDrop;
        impl Drop for ResetOnDrop {
            fn drop(&mut self) {
                set_xybrid_platform_url(None);
                std::env::remove_var("XYBRID_GATEWAY_URL");
                std::env::remove_var("XYBRID_PLATFORM_URL");
            }
        }
        let _reset = ResetOnDrop;

        // The in-memory platform URL is consulted before the env var, and the
        // bare base URL gains the /v1 gateway suffix.
        std::env::remove_var("XYBRID_GATEWAY_URL");
        std::env::set_var("XYBRID_PLATFORM_URL", "https://env.example.com");
        set_xybrid_platform_url(Some("https://staging.example.com".to_string()));
        assert_eq!(default_gateway_url(), "https://staging.example.com/v1");

        // Clearing the cell falls back to the env var (also /v1-suffixed).
        set_xybrid_platform_url(None);
        assert_eq!(default_gateway_url(), "https://env.example.com/v1");

        // XYBRID_GATEWAY_URL (already /v1) wins over the in-memory override.
        set_xybrid_platform_url(Some("https://staging.example.com".to_string()));
        std::env::set_var("XYBRID_GATEWAY_URL", "https://explicit.example.com/v1");
        assert_eq!(default_gateway_url(), "https://explicit.example.com/v1");
    }

    // ── destination-scoped credentials ──────────────────────────────────────

    const PLATFORM: &str = "https://api.xybrid.dev/v1";

    fn env_with<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |var| {
            pairs
                .iter()
                .find(|(name, _)| *name == var)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn url_origin_normalizes_host_and_default_port() {
        assert_eq!(
            url_origin("https://API.xybrid.dev/v1/chat").as_deref(),
            Some("https://api.xybrid.dev:443")
        );
        assert_eq!(
            url_origin("http://127.0.0.1:3001/v1").as_deref(),
            Some("http://127.0.0.1:3001")
        );
        assert_eq!(url_origin("ftp://api.xybrid.dev").as_deref(), None);
        assert_eq!(url_origin("not a url").as_deref(), None);
        assert!(same_origin(PLATFORM, "https://api.xybrid.dev/"));
        assert!(same_origin(PLATFORM, "https://api.xybrid.dev:443/v1/"));
        assert!(!same_origin(PLATFORM, "https://api.xybrid.dev:8443/v1"));
        assert!(!same_origin(PLATFORM, "http://api.xybrid.dev/v1"));
        assert!(!same_origin(
            PLATFORM,
            "https://api.xybrid.dev.evil.example/v1"
        ));
        assert!(!same_origin(
            PLATFORM,
            "https://evil.example/api.xybrid.dev/v1"
        ));
    }

    #[test]
    fn explicit_literal_key_wins_for_any_destination() {
        let key = resolve_api_key_for(
            Some("literal-key"),
            "https://api.deepseek.com/v1",
            PLATFORM,
            Some("platform-key".to_string()),
            env_with(&[("XYBRID_API_KEY", "env-platform-key")]),
        );
        assert_eq!(key.as_deref(), Some("literal-key"));
    }

    #[test]
    fn explicit_env_reference_reads_that_variable_only() {
        let env = env_with(&[
            ("DEEPSEEK_API_KEY", "ds-key"),
            ("XYBRID_API_KEY", "env-platform-key"),
        ]);
        let key = resolve_api_key_for(
            Some("$DEEPSEEK_API_KEY"),
            "https://api.deepseek.com/v1",
            PLATFORM,
            Some("platform-key".to_string()),
            &env,
        );
        assert_eq!(key.as_deref(), Some("ds-key"));

        // Unset or empty reference: no fall-through to any other credential,
        // not even at the platform origin.
        for destination in ["https://api.deepseek.com/v1", PLATFORM] {
            assert_eq!(
                resolve_api_key_for(
                    Some("$MISSING_KEY"),
                    destination,
                    PLATFORM,
                    Some("platform-key".to_string()),
                    &env,
                ),
                None
            );
            assert_eq!(
                resolve_api_key_for(
                    Some("$EMPTY_KEY"),
                    destination,
                    PLATFORM,
                    Some("platform-key".to_string()),
                    env_with(&[("EMPTY_KEY", "   ")]),
                ),
                None
            );
        }
    }

    #[test]
    fn platform_key_is_automatic_only_for_the_platform_origin() {
        let env = env_with(&[("XYBRID_API_KEY", "env-platform-key")]);

        // Programmatic key first, then the ambient env var.
        assert_eq!(
            resolve_api_key_for(
                None,
                PLATFORM,
                PLATFORM,
                Some("platform-key".to_string()),
                &env
            )
            .as_deref(),
            Some("platform-key")
        );
        assert_eq!(
            resolve_api_key_for(None, "https://api.xybrid.dev/v1/", PLATFORM, None, &env)
                .as_deref(),
            Some("env-platform-key")
        );
        // A reconfigured platform origin (staging, self-hosted) still gets it.
        assert_eq!(
            resolve_api_key_for(
                None,
                "http://localhost:3000/v1",
                "http://localhost:3000/v1",
                None,
                &env
            )
            .as_deref(),
            Some("env-platform-key")
        );

        // Provider, custom, loopback and lookalike origins never do.
        for destination in [
            "https://api.deepseek.com/v1",
            "https://api.openai.com/v1",
            "http://127.0.0.1:3001/v1",
            "https://api.xybrid.dev.evil.example/v1",
            "https://api.xybrid.dev:8443/v1",
            "http://api.xybrid.dev/v1",
        ] {
            assert_eq!(
                resolve_api_key_for(
                    None,
                    destination,
                    PLATFORM,
                    Some("platform-key".to_string()),
                    &env
                ),
                None,
                "{destination} must not receive the platform key"
            );
        }
    }

    #[test]
    fn config_resolve_api_key_uses_its_gateway_url_as_destination() {
        // Explicit literal on any destination.
        let config = CloudConfig::gateway()
            .with_gateway_url("https://api.deepseek.com/v1")
            .with_api_key("ds-literal");
        assert_eq!(config.resolve_api_key().as_deref(), Some("ds-literal"));

        // Explicit reference to a variable that is certainly unset.
        let config = CloudConfig::gateway()
            .with_gateway_url("https://api.deepseek.com/v1")
            .with_api_key("$XYBRID_TEST_SURELY_UNSET_KEY_7f3a");
        assert_eq!(config.resolve_api_key(), None);
    }
}
