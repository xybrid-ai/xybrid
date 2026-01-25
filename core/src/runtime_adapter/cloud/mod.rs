//! Cloud Runtime Adapter - Third-party API integrations (OpenAI, Anthropic, etc.)
//!
//! This adapter implements `RuntimeAdapter` for cloud-based LLM providers,
//! routing requests through the Xybrid gateway or directly to provider APIs.
//!
//! ## Architecture
//!
//! The cloud adapter extracts stage configuration from the `Envelope`'s metadata,
//! allowing the `Executor` to remain agnostic to cloud-specific details.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::runtime_adapter::CloudRuntimeAdapter;
//!
//! let adapter = CloudRuntimeAdapter::new();
//! // Or with custom gateway URL:
//! let adapter = CloudRuntimeAdapter::with_gateway("https://my-gateway.example.com");
//! ```

use crate::cloud::{Cloud, CloudBackend, CloudConfig, CompletionRequest};
use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::IntegrationProvider;
use crate::runtime_adapter::{AdapterError, AdapterResult, RuntimeAdapter};
use crate::tracing as trace;

/// Cloud runtime adapter for third-party LLM API integrations.
///
/// This adapter handles cloud-based inference through providers like OpenAI,
/// Anthropic, Google, etc. It can route through the Xybrid gateway (recommended)
/// or directly to provider APIs.
///
/// ## Metadata Keys
///
/// The adapter reads the following keys from `Envelope.metadata`:
///
/// | Key | Type | Description |
/// |-----|------|-------------|
/// | `provider` | String | Provider name: "openai", "anthropic", "google" |
/// | `model` | String | Model identifier, e.g., "gpt-4o-mini" |
/// | `system_prompt` | String | System message for the conversation |
/// | `temperature` | f32 | Sampling temperature (0.0-2.0) |
/// | `max_tokens` | u32 | Maximum tokens in response |
/// | `backend` | String | "gateway" (default) or "direct" |
/// | `gateway_url` | String | Custom gateway URL |
/// | `api_key` | String | API key (for direct mode) |
/// | `timeout_ms` | u32 | Request timeout in milliseconds |
///
pub struct CloudRuntimeAdapter {
    /// Default gateway URL
    gateway_url: String,
    /// Default timeout in milliseconds
    timeout_ms: u32,
    /// Debug mode
    debug: bool,
}

impl CloudRuntimeAdapter {
    /// Creates a new CloudRuntimeAdapter with default settings.
    ///
    /// Uses the default Xybrid gateway URL.
    pub fn new() -> Self {
        Self {
            gateway_url: "http://localhost:3000".to_string(),
            timeout_ms: 60000,
            debug: false,
        }
    }

    /// Creates a CloudRuntimeAdapter with a custom gateway URL.
    pub fn with_gateway(gateway_url: &str) -> Self {
        Self {
            gateway_url: gateway_url.to_string(),
            timeout_ms: 60000,
            debug: false,
        }
    }

    /// Sets the default timeout.
    pub fn with_timeout(mut self, timeout_ms: u32) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Enables debug mode.
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Extracts provider from envelope metadata.
    fn get_provider(&self, envelope: &Envelope) -> AdapterResult<IntegrationProvider> {
        let provider_str = envelope
            .metadata
            .get("provider")
            .ok_or_else(|| AdapterError::InvalidInput("Missing 'provider' in metadata".into()))?;

        // Parse provider string
        match provider_str.to_lowercase().as_str() {
            "openai" => Ok(IntegrationProvider::OpenAI),
            "anthropic" => Ok(IntegrationProvider::Anthropic),
            "google" => Ok(IntegrationProvider::Google),
            "elevenlabs" => Ok(IntegrationProvider::ElevenLabs),
            other => Err(AdapterError::InvalidInput(format!(
                "Unknown provider: {}",
                other
            ))),
        }
    }

    /// Builds CloudConfig from envelope metadata.
    fn build_config(&self, envelope: &Envelope) -> CloudConfig {
        let mut config = CloudConfig {
            gateway_url: self.gateway_url.clone(),
            timeout_ms: self.timeout_ms,
            debug: self.debug,
            ..Default::default()
        };

        // Override with metadata if present
        if let Some(gateway_url) = envelope.metadata.get("gateway_url") {
            config.gateway_url = gateway_url.clone();
        }

        if let Some(api_key) = envelope.metadata.get("api_key") {
            config.api_key = Some(api_key.clone());
        }

        if let Some(timeout_str) = envelope.metadata.get("timeout_ms") {
            if let Ok(timeout) = timeout_str.parse::<u32>() {
                config.timeout_ms = timeout;
            }
        }

        if let Some(debug_str) = envelope.metadata.get("debug") {
            config.debug = debug_str == "true";
        }

        // Backend selection
        if let Some(backend) = envelope.metadata.get("backend") {
            match backend.to_lowercase().as_str() {
                "direct" => {
                    config.backend = CloudBackend::Direct;
                    if let Some(provider) = envelope.metadata.get("provider") {
                        config.direct_provider = Some(provider.clone());
                    }
                }
                _ => {
                    config.backend = CloudBackend::Gateway;
                }
            }
        }

        config
    }

    /// Builds CompletionRequest from envelope metadata.
    fn build_request(&self, input_text: &str, envelope: &Envelope) -> CompletionRequest {
        let mut request = CompletionRequest::new(input_text);

        // Model
        if let Some(model) = envelope.metadata.get("model") {
            request = request.with_model(model);
        }

        // System prompt
        if let Some(system) = envelope.metadata.get("system_prompt") {
            request = request.with_system(system);
        }

        // Temperature
        if let Some(temp_str) = envelope.metadata.get("temperature") {
            if let Ok(temp) = temp_str.parse::<f32>() {
                request = request.with_temperature(temp);
            }
        }

        // Max tokens
        if let Some(max_str) = envelope.metadata.get("max_tokens") {
            if let Ok(max) = max_str.parse::<u32>() {
                request = request.with_max_tokens(max);
            }
        }

        request
    }
}

impl Default for CloudRuntimeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for CloudRuntimeAdapter {
    fn name(&self) -> &str {
        "cloud"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        // Cloud adapter doesn't use file formats
        vec![]
    }

    fn load_model(&mut self, _path: &str) -> AdapterResult<()> {
        // Cloud adapter doesn't load local models
        // Model is specified via metadata
        Ok(())
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        // Validate provider is specified
        let provider = self.get_provider(input)?;

        // Start tracing span
        let model_name = input
            .metadata
            .get("model")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let _exec_span = trace::SpanGuard::new(format!("cloud_execute:{}", model_name));
        trace::add_metadata("provider", provider.as_str());
        trace::add_metadata("adapter", "cloud");

        // Build configuration
        let config = self.build_config(input);
        let backend_str = match config.backend {
            CloudBackend::Gateway => "gateway",
            CloudBackend::Direct => "direct",
        };
        trace::add_metadata("backend", backend_str);

        // Create cloud client
        let client = Cloud::with_config(config).map_err(|e| {
            AdapterError::RuntimeError(format!("Failed to create cloud client: {}", e))
        })?;

        // Extract text input
        let input_text = match &input.kind {
            EnvelopeKind::Text(text) => text.clone(),
            other => {
                return Err(AdapterError::InvalidInput(format!(
                    "Cloud adapter expects Text input, got: {:?}",
                    other
                )));
            }
        };

        // Build and execute request
        let request = self.build_request(&input_text, input);

        let response = {
            let _llm_span = trace::SpanGuard::new("llm_inference");
            client
                .complete(request)
                .map_err(|e| AdapterError::InferenceFailed(format!("LLM request failed: {}", e)))?
        };

        // Build output envelope with response metadata
        let mut output = Envelope::new(EnvelopeKind::Text(response.text));

        // Add response metadata
        if let Some(backend) = response.backend {
            output.metadata.insert("backend".to_string(), backend);
        }
        output
            .metadata
            .insert("provider".to_string(), provider.as_str().to_string());

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_adapter_creation() {
        let adapter = CloudRuntimeAdapter::new();
        assert_eq!(adapter.name(), "cloud");
        assert!(adapter.supported_formats().is_empty());
    }

    #[test]
    fn test_cloud_adapter_with_gateway() {
        let adapter = CloudRuntimeAdapter::with_gateway("https://custom.gateway.com");
        assert_eq!(adapter.gateway_url, "https://custom.gateway.com");
    }

    #[test]
    fn test_load_model_is_noop() {
        let mut adapter = CloudRuntimeAdapter::new();
        // Should succeed (no-op)
        assert!(adapter.load_model("/any/path").is_ok());
    }

    #[test]
    fn test_execute_without_provider_fails() {
        let adapter = CloudRuntimeAdapter::new();
        let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));

        let result = adapter.execute(&input);
        assert!(matches!(result, Err(AdapterError::InvalidInput(_))));
    }

    #[test]
    fn test_execute_with_non_text_input_fails() {
        let adapter = CloudRuntimeAdapter::new();
        let mut input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 100]));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());

        let result = adapter.execute(&input);
        assert!(matches!(result, Err(AdapterError::InvalidInput(_))));
    }
}
