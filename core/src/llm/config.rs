//! LLM configuration types.

use serde::{Deserialize, Serialize};

/// LLM execution backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmBackend {
    /// Route through Xybrid Gateway (default, recommended).
    /// Gateway handles authentication, rate limiting, and provider routing.
    Gateway,

    /// Use local on-device model (ONNX).
    /// Best for privacy-sensitive applications or offline use.
    Local,

    /// Direct API calls (for development/testing only).
    /// Requires API keys in environment or config.
    /// NOT recommended for production mobile apps.
    Direct,

    /// Auto-select based on availability and request type.
    Auto,
}

impl Default for LlmBackend {
    fn default() -> Self {
        LlmBackend::Gateway
    }
}

/// LLM client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Which backend to use for LLM requests.
    #[serde(default)]
    pub backend: LlmBackend,

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

    /// Local model path (for Local backend).
    #[serde(default)]
    pub local_model_path: Option<String>,

    /// Direct provider (for Direct backend - development only).
    #[serde(default)]
    pub direct_provider: Option<String>,
}

fn default_gateway_url() -> String {
    // SOOn https://gateway.xybrid.cloud/v1
    // SOON https://gateway.xybrid.ai/v1
    "https://gateway.xybrid.dev/v1".to_string()
}

fn default_timeout_ms() -> u32 {
    30000
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            backend: LlmBackend::default(),
            gateway_url: default_gateway_url(),
            api_key: None,
            default_model: None,
            timeout_ms: default_timeout_ms(),
            debug: false,
            local_model_path: None,
            direct_provider: None,
        }
    }
}

impl LlmConfig {
    /// Create a new config with gateway backend.
    pub fn gateway() -> Self {
        Self {
            backend: LlmBackend::Gateway,
            ..Default::default()
        }
    }

    /// Create a new config with local backend.
    pub fn local(model_path: impl Into<String>) -> Self {
        Self {
            backend: LlmBackend::Local,
            local_model_path: Some(model_path.into()),
            ..Default::default()
        }
    }

    /// Create a new config with direct backend (development only).
    pub fn direct(provider: impl Into<String>) -> Self {
        Self {
            backend: LlmBackend::Direct,
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
            if key.starts_with('$') {
                let env_var = &key[1..];
                return std::env::var(env_var).ok();
            }
            return Some(key.clone());
        }

        // Fall back to XYBRID_API_KEY
        std::env::var("XYBRID_API_KEY").ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlmConfig::default();
        assert_eq!(config.backend, LlmBackend::Gateway);
        assert!(config.gateway_url.contains("xybrid"));
    }

    #[test]
    fn test_gateway_config() {
        let config = LlmConfig::gateway()
            .with_api_key("test-key")
            .with_default_model("gpt-4o-mini");

        assert_eq!(config.backend, LlmBackend::Gateway);
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.default_model, Some("gpt-4o-mini".to_string()));
    }

    #[test]
    fn test_local_config() {
        let config = LlmConfig::local("/models/llama.onnx");
        assert_eq!(config.backend, LlmBackend::Local);
        assert_eq!(
            config.local_model_path,
            Some("/models/llama.onnx".to_string())
        );
    }

    #[test]
    fn test_direct_config() {
        let config = LlmConfig::direct("openai");
        assert_eq!(config.backend, LlmBackend::Direct);
        assert_eq!(config.direct_provider, Some("openai".to_string()));
    }

    #[test]
    fn test_resolve_api_key_from_env() {
        std::env::set_var("TEST_LLM_KEY", "secret123");

        let config = LlmConfig::default().with_api_key("$TEST_LLM_KEY");
        assert_eq!(config.resolve_api_key(), Some("secret123".to_string()));

        std::env::remove_var("TEST_LLM_KEY");
    }
}
