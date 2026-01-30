//! Gateway configuration.

use serde::{Deserialize, Serialize};

/// Gateway configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Gateway base URL.
    pub base_url: String,

    /// Default model to use if not specified in request.
    pub default_model: Option<String>,

    /// Request timeout in milliseconds.
    pub timeout_ms: u32,

    /// Enable debug logging.
    pub debug: bool,

    /// Maximum retries on failure.
    pub max_retries: u32,

    /// Rate limiting configuration.
    pub rate_limit: Option<RateLimitConfig>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.xybrid.dev/v1".to_string(),
            default_model: Some("gpt-4o-mini".to_string()),
            timeout_ms: 60000,
            debug: false,
            max_retries: 3,
            rate_limit: None,
        }
    }
}

impl GatewayConfig {
    /// Create a new gateway config with custom base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            ..Default::default()
        }
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

    /// Set rate limiting.
    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = Some(config);
        self
    }
}

/// Rate limiting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per minute.
    pub requests_per_minute: u32,

    /// Maximum tokens per minute.
    pub tokens_per_minute: u32,

    /// Maximum concurrent requests.
    pub max_concurrent: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 100000,
            max_concurrent: 10,
        }
    }
}

/// Model routing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRouting {
    /// Model alias (user-facing name).
    pub alias: String,

    /// Actual model ID to use.
    pub model_id: String,

    /// Provider to route to.
    pub provider: String,

    /// Priority (lower = higher priority).
    pub priority: u32,

    /// Whether this model is enabled.
    pub enabled: bool,
}

/// Provider credentials configuration.
#[derive(Debug, Clone)]
pub struct ProviderCredentials {
    /// Provider name (e.g., "openai", "anthropic").
    pub provider: String,

    /// API key (loaded from secure storage).
    pub api_key: Option<String>,

    /// Base URL override (for proxies).
    pub base_url: Option<String>,

    /// Organization ID (for OpenAI).
    pub org_id: Option<String>,
}

impl ProviderCredentials {
    /// Create new credentials for a provider.
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            api_key: None,
            base_url: None,
            org_id: None,
        }
    }

    /// Set the API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set organization ID.
    pub fn with_org_id(mut self, org_id: impl Into<String>) -> Self {
        self.org_id = Some(org_id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GatewayConfig::default();
        assert_eq!(config.base_url, "https://api.xybrid.dev/v1");
        assert_eq!(config.default_model, Some("gpt-4o-mini".to_string()));
        assert_eq!(config.timeout_ms, 60000);
    }

    #[test]
    fn test_config_builder() {
        let config = GatewayConfig::new("https://custom.gateway.com/v1")
            .with_default_model("claude-3-sonnet")
            .with_timeout(30000)
            .with_debug(true);

        assert_eq!(config.base_url, "https://custom.gateway.com/v1");
        assert_eq!(config.default_model, Some("claude-3-sonnet".to_string()));
        assert_eq!(config.timeout_ms, 30000);
        assert!(config.debug);
    }

    #[test]
    fn test_provider_credentials() {
        let creds = ProviderCredentials::new("openai")
            .with_api_key("sk-test")
            .with_org_id("org-123");

        assert_eq!(creds.provider, "openai");
        assert_eq!(creds.api_key, Some("sk-test".to_string()));
        assert_eq!(creds.org_id, Some("org-123".to_string()));
    }
}
