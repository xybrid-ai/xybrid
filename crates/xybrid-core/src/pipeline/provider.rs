//! Integration provider configuration for third-party APIs.
//!
//! Supported providers:
//! - OpenAI (GPT, Whisper, DALL-E)
//! - Anthropic (Claude)
//! - Google (Gemini)
//! - ElevenLabs (TTS)
//! - OpenRouter (multi-provider routing)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported integration providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntegrationProvider {
    /// OpenAI API (GPT, Whisper, DALL-E, embeddings).
    OpenAI,

    /// Anthropic API (Claude models).
    Anthropic,

    /// Google AI (Gemini models).
    Google,

    /// ElevenLabs (TTS).
    ElevenLabs,

    /// OpenRouter (multi-provider).
    OpenRouter,

    /// Custom provider (user-defined).
    Custom,
}

impl IntegrationProvider {
    /// Get the environment variable name for the API key.
    pub fn api_key_env_var(&self) -> &'static str {
        match self {
            IntegrationProvider::OpenAI => "OPENAI_API_KEY",
            IntegrationProvider::Anthropic => "ANTHROPIC_API_KEY",
            IntegrationProvider::Google => "GOOGLE_API_KEY",
            IntegrationProvider::ElevenLabs => "ELEVENLABS_API_KEY",
            IntegrationProvider::OpenRouter => "OPENROUTER_API_KEY",
            IntegrationProvider::Custom => "CUSTOM_API_KEY",
        }
    }

    /// Get the default base URL for the provider.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            IntegrationProvider::OpenAI => "https://api.openai.com/v1",
            IntegrationProvider::Anthropic => "https://api.anthropic.com/v1",
            IntegrationProvider::Google => "https://generativelanguage.googleapis.com/v1beta",
            IntegrationProvider::ElevenLabs => "https://api.elevenlabs.io/v1",
            IntegrationProvider::OpenRouter => "https://openrouter.ai/api/v1",
            IntegrationProvider::Custom => "",
        }
    }

    /// Get the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            IntegrationProvider::OpenAI => "openai",
            IntegrationProvider::Anthropic => "anthropic",
            IntegrationProvider::Google => "google",
            IntegrationProvider::ElevenLabs => "elevenlabs",
            IntegrationProvider::OpenRouter => "openrouter",
            IntegrationProvider::Custom => "custom",
        }
    }
}

impl std::fmt::Display for IntegrationProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for IntegrationProvider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(IntegrationProvider::OpenAI),
            "anthropic" | "claude" => Ok(IntegrationProvider::Anthropic),
            "google" | "gemini" => Ok(IntegrationProvider::Google),
            "elevenlabs" | "eleven" | "eleven_labs" => Ok(IntegrationProvider::ElevenLabs),
            "openrouter" | "open_router" => Ok(IntegrationProvider::OpenRouter),
            "custom" => Ok(IntegrationProvider::Custom),
            _ => Err(format!(
                "Unknown provider: '{}'. Valid values: openai, anthropic, google, elevenlabs, openrouter, custom",
                s
            )),
        }
    }
}

/// Provider-specific configuration options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// The integration provider.
    pub provider: IntegrationProvider,

    /// Override base URL (optional).
    #[serde(default)]
    pub base_url: Option<String>,

    /// API key (usually read from environment).
    /// Can be:
    /// - Direct value (NOT recommended for production)
    /// - Environment variable reference: `$OPENAI_API_KEY`
    /// - Xybrid secrets reference: `xybrid://secrets/openai-key`
    #[serde(default)]
    pub api_key: Option<String>,

    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u32,

    /// Additional provider-specific options.
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

fn default_timeout_ms() -> u32 {
    30000
}

impl ProviderConfig {
    /// Create a new provider config.
    pub fn new(provider: IntegrationProvider) -> Self {
        Self {
            provider,
            base_url: None,
            api_key: None,
            timeout_ms: default_timeout_ms(),
            options: HashMap::new(),
        }
    }

    /// Get the effective base URL (custom or default).
    pub fn effective_base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or_else(|| self.provider.default_base_url())
    }

    /// Resolve the API key from environment or config.
    pub fn resolve_api_key(&self) -> Option<String> {
        if let Some(key) = &self.api_key {
            // Check if it's an environment variable reference
            if let Some(env_var) = key.strip_prefix('$') {
                return std::env::var(env_var).ok();
            }
            // Check if it's a Xybrid secrets reference
            if key.starts_with("xybrid://secrets/") {
                // TODO: Implement Xybrid secrets resolution
                return None;
            }
            // Direct value (not recommended)
            return Some(key.clone());
        }

        // Fall back to default environment variable
        std::env::var(self.provider.api_key_env_var()).ok()
    }

    /// Set an option value.
    pub fn with_option(mut self, key: &str, value: serde_json::Value) -> Self {
        self.options.insert(key.to_string(), value);
        self
    }
}

/// OpenAI-specific options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIOptions {
    /// Temperature for generation (0.0 - 2.0).
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// System prompt for chat models.
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Top-p sampling.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Frequency penalty.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

fn default_temperature() -> f32 {
    0.7
}

impl Default for OpenAIOptions {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            max_tokens: None,
            system_prompt: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

/// Anthropic-specific options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicOptions {
    /// Maximum tokens to generate.
    #[serde(default = "default_anthropic_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for generation (0.0 - 1.0).
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// System prompt.
    #[serde(default)]
    pub system: Option<String>,

    /// Top-p sampling.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<u32>,
}

fn default_anthropic_max_tokens() -> u32 {
    4096
}

impl Default for AnthropicOptions {
    fn default() -> Self {
        Self {
            max_tokens: default_anthropic_max_tokens(),
            temperature: default_temperature(),
            system: None,
            top_p: None,
            top_k: None,
        }
    }
}

/// ElevenLabs-specific options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ElevenLabsOptions {
    /// Voice ID to use.
    #[serde(default)]
    pub voice_id: Option<String>,

    /// Model ID (e.g., "eleven_multilingual_v2").
    #[serde(default)]
    pub model_id: Option<String>,

    /// Stability (0.0 - 1.0).
    #[serde(default)]
    pub stability: Option<f32>,

    /// Similarity boost (0.0 - 1.0).
    #[serde(default)]
    pub similarity_boost: Option<f32>,

    /// Output format (e.g., "mp3_44100_128").
    #[serde(default)]
    pub output_format: Option<String>,
}

/// Google-specific options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoogleOptions {
    /// Temperature for generation (0.0 - 1.0).
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum output tokens.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,

    /// Top-p sampling.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<u32>,

    /// Safety settings level.
    #[serde(default)]
    pub safety_level: Option<String>,
}

impl Default for GoogleOptions {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            max_output_tokens: None,
            top_p: None,
            top_k: None,
            safety_level: None,
        }
    }
}

/// Provider configuration validation result.
#[derive(Debug, Clone)]
pub struct ProviderValidation {
    /// Whether the configuration is valid.
    pub valid: bool,
    /// Error messages if invalid.
    pub errors: Vec<String>,
    /// Warning messages.
    pub warnings: Vec<String>,
}

impl ProviderConfig {
    /// Validate the provider configuration.
    pub fn validate(&self) -> ProviderValidation {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for API key
        if self.resolve_api_key().is_none() {
            warnings.push(format!(
                "No API key found for {}. Set {} environment variable or provide api_key in config.",
                self.provider,
                self.provider.api_key_env_var()
            ));
        }

        // Check for custom provider without base URL
        if self.provider == IntegrationProvider::Custom && self.base_url.is_none() {
            errors.push("Custom provider requires a base_url".to_string());
        }

        // Validate timeout
        if self.timeout_ms < 1000 {
            warnings.push(format!(
                "Timeout {}ms is very short, consider increasing to at least 5000ms",
                self.timeout_ms
            ));
        }

        ProviderValidation {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Check if the provider is ready for use (has API key).
    pub fn is_ready(&self) -> bool {
        self.resolve_api_key().is_some()
    }

    /// Get headers for API requests.
    pub fn auth_headers(&self) -> Option<Vec<(String, String)>> {
        let api_key = self.resolve_api_key()?;

        let headers = match self.provider {
            IntegrationProvider::OpenAI | IntegrationProvider::OpenRouter => {
                vec![("Authorization".to_string(), format!("Bearer {}", api_key))]
            }
            IntegrationProvider::Anthropic => {
                vec![
                    ("x-api-key".to_string(), api_key),
                    ("anthropic-version".to_string(), "2023-06-01".to_string()),
                ]
            }
            IntegrationProvider::Google => {
                // Google uses query parameter or API key header
                vec![("x-goog-api-key".to_string(), api_key)]
            }
            IntegrationProvider::ElevenLabs => {
                vec![("xi-api-key".to_string(), api_key)]
            }
            IntegrationProvider::Custom => {
                vec![("Authorization".to_string(), format!("Bearer {}", api_key))]
            }
        };

        Some(headers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_from_str() {
        assert_eq!(
            "openai".parse::<IntegrationProvider>().unwrap(),
            IntegrationProvider::OpenAI
        );
        assert_eq!(
            "anthropic".parse::<IntegrationProvider>().unwrap(),
            IntegrationProvider::Anthropic
        );
        assert_eq!(
            "claude".parse::<IntegrationProvider>().unwrap(),
            IntegrationProvider::Anthropic
        );
        assert_eq!(
            "elevenlabs".parse::<IntegrationProvider>().unwrap(),
            IntegrationProvider::ElevenLabs
        );
    }

    #[test]
    fn test_provider_env_var() {
        assert_eq!(
            IntegrationProvider::OpenAI.api_key_env_var(),
            "OPENAI_API_KEY"
        );
        assert_eq!(
            IntegrationProvider::Anthropic.api_key_env_var(),
            "ANTHROPIC_API_KEY"
        );
    }

    #[test]
    fn test_provider_config_base_url() {
        let config = ProviderConfig::new(IntegrationProvider::OpenAI);
        assert_eq!(config.effective_base_url(), "https://api.openai.com/v1");

        let config = ProviderConfig {
            provider: IntegrationProvider::OpenAI,
            base_url: Some("https://custom.openai.azure.com".to_string()),
            api_key: None,
            timeout_ms: 30000,
            options: HashMap::new(),
        };
        assert_eq!(
            config.effective_base_url(),
            "https://custom.openai.azure.com"
        );
    }

    #[test]
    fn test_provider_config_serde() {
        let config = ProviderConfig::new(IntegrationProvider::OpenAI)
            .with_option("temperature", serde_json::json!(0.7));

        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: ProviderConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.provider, parsed.provider);
    }

    #[test]
    fn test_provider_validation() {
        let config = ProviderConfig::new(IntegrationProvider::Custom);
        let validation = config.validate();
        assert!(!validation.valid);
        assert!(validation.errors.iter().any(|e| e.contains("base_url")));

        let config = ProviderConfig::new(IntegrationProvider::OpenAI);
        let validation = config.validate();
        assert!(validation.valid); // No errors, just warnings about API key
    }

    #[test]
    fn test_provider_auth_headers() {
        // Set a test API key
        std::env::set_var("OPENAI_API_KEY", "test-key-123");

        let config = ProviderConfig::new(IntegrationProvider::OpenAI);
        let headers = config.auth_headers().unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "Authorization");
        assert!(headers[0].1.contains("Bearer test-key-123"));

        // Clean up
        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn test_google_options_default() {
        let options = GoogleOptions::default();
        assert!((options.temperature - 0.7).abs() < f32::EPSILON);
        assert!(options.max_output_tokens.is_none());
    }

    #[test]
    fn test_provider_is_ready() {
        std::env::set_var("TEST_API_KEY", "key123");

        let mut config = ProviderConfig::new(IntegrationProvider::Custom);
        config.api_key = Some("$TEST_API_KEY".to_string());
        assert!(config.is_ready());

        config.api_key = Some("$NONEXISTENT_KEY".to_string());
        assert!(!config.is_ready());

        std::env::remove_var("TEST_API_KEY");
    }
}
