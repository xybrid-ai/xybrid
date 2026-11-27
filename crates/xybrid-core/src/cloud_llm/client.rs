//! LLM client implementation for OpenAI and Anthropic APIs.

use super::error::LlmError;
use super::request::{LlmRequest, Role};
use super::response::{
    AnthropicError, AnthropicResponse, LlmResponse, OpenAIError, OpenAIResponse,
};
use crate::pipeline::{IntegrationProvider, ProviderConfig};
use serde_json::json;
use std::time::Duration;

/// Default models for each provider.
const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-3-5-sonnet-20241022";

/// LLM client for cloud API calls.
pub struct LlmClient {
    provider: IntegrationProvider,
    config: ProviderConfig,
    agent: ureq::Agent,
}

impl LlmClient {
    /// Create a new LLM client for the specified provider.
    ///
    /// Uses default configuration and reads API key from environment.
    pub fn new(provider: IntegrationProvider) -> Result<Self, LlmError> {
        let config = ProviderConfig::new(provider);
        Self::with_config(config)
    }

    /// Create a new LLM client with custom configuration.
    pub fn with_config(config: ProviderConfig) -> Result<Self, LlmError> {
        // Validate provider is supported
        match config.provider {
            IntegrationProvider::OpenAI | IntegrationProvider::Anthropic => {}
            _ => {
                return Err(LlmError::UnsupportedProvider(format!(
                    "LLM client only supports OpenAI and Anthropic, got: {}",
                    config.provider
                )));
            }
        }

        // Check for API key
        if config.resolve_api_key().is_none() {
            return Err(LlmError::ApiKeyMissing {
                provider: config.provider.to_string(),
                env_var: config.provider.api_key_env_var().to_string(),
            });
        }

        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_millis(config.timeout_ms as u64))
            .build();

        Ok(Self {
            provider: config.provider,
            config,
            agent,
        })
    }

    /// Create an OpenAI client.
    pub fn openai() -> Result<Self, LlmError> {
        Self::new(IntegrationProvider::OpenAI)
    }

    /// Create an Anthropic client.
    pub fn anthropic() -> Result<Self, LlmError> {
        Self::new(IntegrationProvider::Anthropic)
    }

    /// Send a completion request.
    pub fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        match self.provider {
            IntegrationProvider::OpenAI => self.complete_openai(request),
            IntegrationProvider::Anthropic => self.complete_anthropic(request),
            _ => Err(LlmError::UnsupportedProvider(self.provider.to_string())),
        }
    }

    /// Simple prompt completion (convenience method).
    pub fn prompt(&self, prompt: &str) -> Result<String, LlmError> {
        let request = LlmRequest::prompt(prompt);
        let response = self.complete(request)?;
        Ok(response.text)
    }

    /// Chat completion with system prompt (convenience method).
    pub fn chat(&self, system: &str, user_message: &str) -> Result<String, LlmError> {
        let request = LlmRequest::prompt(user_message).with_system(system);
        let response = self.complete(request)?;
        Ok(response.text)
    }

    /// OpenAI completion implementation.
    fn complete_openai(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let api_key = self
            .config
            .resolve_api_key()
            .ok_or_else(|| LlmError::ApiKeyMissing {
                provider: "OpenAI".into(),
                env_var: "OPENAI_API_KEY".into(),
            })?;

        let model = request.model.as_deref().unwrap_or(DEFAULT_OPENAI_MODEL);
        let messages = self.build_openai_messages(&request);

        let mut body = json!({
            "model": model,
            "messages": messages,
        });

        // Add optional parameters
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(ref stop) = request.stop {
            body["stop"] = json!(stop);
        }
        if let Some(freq_penalty) = request.frequency_penalty {
            body["frequency_penalty"] = json!(freq_penalty);
        }
        if let Some(presence_penalty) = request.presence_penalty {
            body["presence_penalty"] = json!(presence_penalty);
        }

        let url = format!("{}/chat/completions", self.config.effective_base_url());

        let response = crate::tracing::inject_provider_trace_headers(self.agent.post(&url))
            .set("Authorization", &format!("Bearer {}", api_key))
            .set("Content-Type", "application/json")
            .send_json(&body);

        match response {
            Ok(resp) => {
                let openai_resp: OpenAIResponse = resp
                    .into_json()
                    .map_err(|e| LlmError::ParseError(e.to_string()))?;
                Ok(openai_resp.into())
            }
            Err(ureq::Error::Status(status, resp)) => {
                let error_body: Result<OpenAIError, _> = resp.into_json();
                let (message, spent_quota) = match error_body {
                    Ok(e) => {
                        let spent_quota = crate::cloud::is_insufficient_quota(
                            e.error.code.as_deref(),
                            e.error.error_type.as_deref(),
                        );
                        (e.error.message, spent_quota)
                    }
                    Err(_) => ("Unknown error".into(), false),
                };

                // A spent quota (`insufficient_quota`) is not a rate limit:
                // it stays an `ApiError` so its message reaches the caller
                // and nothing retries it.
                if status == 429 && !spent_quota {
                    return Err(LlmError::RateLimited {
                        retry_after_secs: 60,
                    });
                }

                Err(LlmError::ApiError { status, message })
            }
            Err(ureq::Error::Transport(transport)) => {
                // Check if it's a timeout by examining the error message
                let msg = transport.to_string();
                if msg.contains("timed out") || msg.contains("timeout") {
                    return Err(LlmError::Timeout {
                        timeout_ms: self.config.timeout_ms,
                    });
                }
                Err(LlmError::HttpError(msg))
            }
        }
    }

    /// Anthropic completion implementation.
    fn complete_anthropic(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let api_key = self
            .config
            .resolve_api_key()
            .ok_or_else(|| LlmError::ApiKeyMissing {
                provider: "Anthropic".into(),
                env_var: "ANTHROPIC_API_KEY".into(),
            })?;

        let model = request.model.as_deref().unwrap_or(DEFAULT_ANTHROPIC_MODEL);
        let messages = self.build_anthropic_messages(&request);
        let max_tokens = request.max_tokens.unwrap_or(4096);

        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        });

        // Add system prompt separately (Anthropic API style)
        if let Some(ref system) = request.system {
            body["system"] = json!(system);
        }

        // Add optional parameters
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(top_k) = request.top_k {
            body["top_k"] = json!(top_k);
        }
        if let Some(ref stop) = request.stop {
            body["stop_sequences"] = json!(stop);
        }

        let url = format!("{}/messages", self.config.effective_base_url());

        let response = crate::tracing::inject_provider_trace_headers(self.agent.post(&url))
            .set("x-api-key", &api_key)
            .set("anthropic-version", "2023-06-01")
            .set("Content-Type", "application/json")
            .send_json(&body);

        match response {
            Ok(resp) => {
                let anthropic_resp: AnthropicResponse = resp
                    .into_json()
                    .map_err(|e| LlmError::ParseError(e.to_string()))?;
                Ok(anthropic_resp.into())
            }
            Err(ureq::Error::Status(status, resp)) => {
                let error_body: Result<AnthropicError, _> = resp.into_json();
                let message = error_body
                    .map(|e| e.error.message)
                    .unwrap_or_else(|_| "Unknown error".into());

                if status == 429 {
                    return Err(LlmError::RateLimited {
                        retry_after_secs: 60,
                    });
                }

                Err(LlmError::ApiError { status, message })
            }
            Err(ureq::Error::Transport(transport)) => {
                // Check if it's a timeout by examining the error message
                let msg = transport.to_string();
                if msg.contains("timed out") || msg.contains("timeout") {
                    return Err(LlmError::Timeout {
                        timeout_ms: self.config.timeout_ms,
                    });
                }
                Err(LlmError::HttpError(msg))
            }
        }
    }

    /// Build OpenAI-format messages array.
    fn build_openai_messages(&self, request: &LlmRequest) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(json!({
                "role": "system",
                "content": system
            }));
        }

        // Add conversation messages or single prompt
        if let Some(ref msgs) = request.messages {
            for msg in msgs {
                messages.push(json!({
                    "role": self.role_to_string(&msg.role),
                    "content": &msg.content
                }));
            }
        } else if let Some(ref prompt) = request.prompt {
            messages.push(json!({
                "role": "user",
                "content": prompt
            }));
        }

        messages
    }

    /// Build Anthropic-format messages array.
    /// Note: Anthropic doesn't include system message in messages array.
    fn build_anthropic_messages(&self, request: &LlmRequest) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // Add conversation messages or single prompt
        // Skip system messages (handled separately by Anthropic API)
        if let Some(ref msgs) = request.messages {
            for msg in msgs {
                if msg.role != Role::System {
                    messages.push(json!({
                        "role": self.role_to_string(&msg.role),
                        "content": &msg.content
                    }));
                }
            }
        } else if let Some(ref prompt) = request.prompt {
            messages.push(json!({
                "role": "user",
                "content": prompt
            }));
        }

        messages
    }

    /// Convert role enum to string.
    fn role_to_string(&self, role: &Role) -> &'static str {
        match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    /// Get the current provider.
    pub fn provider(&self) -> IntegrationProvider {
        self.provider
    }

    /// Get the configuration.
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::super::request::Message;
    use super::*;

    /// Helper to create an OpenAI client with a direct API key (no env var manipulation).
    fn openai_client_with_key(key: &str) -> LlmClient {
        let mut config = ProviderConfig::new(IntegrationProvider::OpenAI);
        config.api_key = Some(key.to_string());
        LlmClient::with_config(config).unwrap()
    }

    /// Helper to create an Anthropic client with a direct API key (no env var manipulation).
    fn anthropic_client_with_key(key: &str) -> LlmClient {
        let mut config = ProviderConfig::new(IntegrationProvider::Anthropic);
        config.api_key = Some(key.to_string());
        LlmClient::with_config(config).unwrap()
    }

    #[test]
    fn test_client_creation_requires_key() {
        // Use an env var reference to a nonexistent variable, so resolve_api_key()
        // returns None without touching real env vars (avoids test race conditions).
        let mut config = ProviderConfig::new(IntegrationProvider::OpenAI);
        config.api_key = Some("$XYBRID_TEST_NONEXISTENT_KEY_12345".to_string());

        let result = LlmClient::with_config(config);
        assert!(matches!(result, Err(LlmError::ApiKeyMissing { .. })));
    }

    #[test]
    fn test_client_with_key() {
        let mut config = ProviderConfig::new(IntegrationProvider::OpenAI);
        config.api_key = Some("test-key-123".to_string());

        let client = LlmClient::with_config(config);
        assert!(client.is_ok());

        let client = client.unwrap();
        assert_eq!(client.provider(), IntegrationProvider::OpenAI);
    }

    #[test]
    fn test_unsupported_provider() {
        let mut config = ProviderConfig::new(IntegrationProvider::ElevenLabs);
        config.api_key = Some("test".to_string());

        let result = LlmClient::with_config(config);
        assert!(matches!(result, Err(LlmError::UnsupportedProvider(_))));
    }

    #[test]
    fn test_build_openai_messages() {
        let client = openai_client_with_key("test");

        let request = LlmRequest::prompt("Hello").with_system("Be helpful");

        let messages = client.build_openai_messages(&request);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn test_build_anthropic_messages() {
        let client = anthropic_client_with_key("test");

        // System message should not be in the messages array for Anthropic
        let request = LlmRequest::chat(vec![Message::system("Be helpful"), Message::user("Hello")]);

        let messages = client.build_anthropic_messages(&request);

        // Should only have user message (system is handled separately)
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    /// OpenAI answers a spent quota with 429 `insufficient_quota`: that is
    /// reported with its message (not as a rate limit), while an ordinary 429
    /// stays `RateLimited`.
    #[test]
    fn openai_429_insufficient_quota_is_an_api_error_not_a_rate_limit() {
        use httpmock::prelude::*;

        let server = MockServer::start();
        let mut config = ProviderConfig::new(IntegrationProvider::OpenAI);
        config.api_key = Some("test-key".to_string());
        config.base_url = Some(server.base_url());
        let client = LlmClient::with_config(config).unwrap();

        let mut quota = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(429)
                .header("content-type", "application/json")
                .body(
                    json!({ "error": {
                        "message": "You exceeded your current quota.",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                    } })
                    .to_string(),
                );
        });
        match client.complete(LlmRequest::prompt("hello")) {
            Err(LlmError::ApiError { status, message }) => {
                assert_eq!(status, 429);
                assert_eq!(message, "You exceeded your current quota.");
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
        quota.assert_hits(1);
        quota.delete();

        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(429)
                .header("content-type", "application/json")
                .body(
                    json!({ "error": {
                        "message": "Rate limit reached.",
                        "type": "requests",
                        "code": "rate_limit_exceeded",
                    } })
                    .to_string(),
                );
        });
        assert!(matches!(
            client.complete(LlmRequest::prompt("hello")),
            Err(LlmError::RateLimited { .. })
        ));
    }
}
