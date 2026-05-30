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
}

fn default_gateway_url() -> String {
    // Priority:
    // 1. XYBRID_GATEWAY_URL env var (explicit override, should include /v1)
    // 2. XYBRID_PLATFORM_URL env var + /v1 suffix (shared with telemetry)
    // 3. Default production URL (api.xybrid.dev/v1)
    //
    // Note: The /v1 prefix is required for OpenAI-compatible API endpoints.
    // The client appends /chat/completions, so the full path becomes /v1/chat/completions.
    if let Ok(url) = std::env::var("XYBRID_GATEWAY_URL") {
        return url;
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

    /// Resolve the API key from environment or config.
    pub fn resolve_api_key(&self) -> Option<String> {
        if let Some(ref key) = self.api_key {
            if let Some(env_var) = key.strip_prefix('$') {
                return std::env::var(env_var).ok();
            }
            return Some(key.clone());
        }

        // Programmatic key (set via the SDK, held in memory — not the
        // environment) takes precedence over the ambient `XYBRID_API_KEY` env
        // var, which remains the fallback for externally-configured keys
        // (CLI `--env`, Flutter `--dart-define`, iOS `ProcessInfo`).
        xybrid_api_key().or_else(|| std::env::var("XYBRID_API_KEY").ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
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
}
