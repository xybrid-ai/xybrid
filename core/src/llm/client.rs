//! LLM client implementation.

use super::completion::{CompletionRequest, CompletionResponse};
use super::config::{LlmBackend, LlmConfig};
use super::error::LlmError;
use serde_json::json;
use std::time::{Duration, Instant};

/// LLM client for completions.
///
/// Routes requests through the configured backend:
/// - Gateway (default): Xybrid's managed gateway
/// - Local: On-device ONNX model
/// - Direct: Direct API calls (development only)
pub struct Llm {
    config: LlmConfig,
    agent: ureq::Agent,
}

impl Llm {
    /// Create a new LLM client with default configuration.
    /// Uses gateway backend by default.
    pub fn new() -> Result<Self, LlmError> {
        Self::with_config(LlmConfig::default())
    }

    /// Create a new LLM client with custom configuration.
    pub fn with_config(config: LlmConfig) -> Result<Self, LlmError> {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_millis(config.timeout_ms as u64))
            .build();

        Ok(Self { config, agent })
    }

    /// Create a client that uses the gateway.
    pub fn gateway() -> Result<Self, LlmError> {
        Self::with_config(LlmConfig::gateway())
    }

    /// Create a client that uses direct API calls (development only).
    pub fn direct(provider: &str) -> Result<Self, LlmError> {
        Self::with_config(LlmConfig::direct(provider))
    }

    /// Send a completion request.
    pub fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let start = Instant::now();

        let mut response = match self.config.backend {
            LlmBackend::Gateway => self.complete_gateway(request)?,
            LlmBackend::Local => self.complete_local(request)?,
            LlmBackend::Direct => self.complete_direct(request)?,
            LlmBackend::Auto => self.complete_auto(request)?,
        };

        response.latency_ms = Some(start.elapsed().as_millis() as u32);
        Ok(response)
    }

    /// Simple prompt completion (convenience method).
    pub fn prompt(&self, prompt: &str) -> Result<String, LlmError> {
        let request = CompletionRequest::new(prompt);
        let response = self.complete(request)?;
        Ok(response.text)
    }

    /// Chat completion with system prompt (convenience method).
    pub fn chat(&self, system: &str, user_message: &str) -> Result<String, LlmError> {
        let request = CompletionRequest::new(user_message).with_system(system);
        let response = self.complete(request)?;
        Ok(response.text)
    }

    /// Complete through Xybrid Gateway.
    fn complete_gateway(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let api_key = self.config.resolve_api_key();

        // Build OpenAI-compatible request body
        let messages = request.to_messages();
        let openai_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "role": match m.role {
                        super::completion::Role::System => "system",
                        super::completion::Role::User => "user",
                        super::completion::Role::Assistant => "assistant",
                    },
                    "content": &m.content
                })
            })
            .collect();

        let model = request
            .model
            .clone()
            .or_else(|| self.config.default_model.clone())
            .unwrap_or_else(|| "gpt-4o-mini".to_string());

        let mut body = json!({
            "model": model,
            "messages": openai_messages,
        });

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
        if request.stream {
            body["stream"] = json!(true);
        }

        let url = format!("{}/chat/completions", self.config.gateway_url);

        if self.config.debug {
            eprintln!("[LLM] Gateway request to: {}", url);
            eprintln!("[LLM] Body: {}", serde_json::to_string_pretty(&body).unwrap_or_default());
        }

        let mut req = self.agent.post(&url).set("Content-Type", "application/json");

        if let Some(ref key) = api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let response = req.send_json(&body);

        match response {
            Ok(resp) => {
                let json_resp: serde_json::Value = resp
                    .into_json()
                    .map_err(|e| LlmError::ParseError(e.to_string()))?;

                if self.config.debug {
                    eprintln!("[LLM] Response: {}", serde_json::to_string_pretty(&json_resp).unwrap_or_default());
                }

                // Parse OpenAI-format response
                let text = json_resp["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();

                let model = json_resp["model"].as_str().unwrap_or("unknown").to_string();

                let finish_reason = json_resp["choices"][0]["finish_reason"]
                    .as_str()
                    .map(|s| s.to_string());

                let usage = json_resp.get("usage").map(|u| super::completion::Usage {
                    prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                    completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
                    total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
                });

                let id = json_resp["id"].as_str().map(|s| s.to_string());

                Ok(CompletionResponse {
                    text,
                    model,
                    finish_reason,
                    usage,
                    id,
                    latency_ms: None,
                    backend: Some("gateway".to_string()),
                })
            }
            Err(ureq::Error::Status(status, resp)) => {
                let error_body: Result<serde_json::Value, _> = resp.into_json();
                let message = error_body
                    .ok()
                    .and_then(|v| v["error"]["message"].as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "Unknown error".into());

                if status == 429 {
                    return Err(LlmError::RateLimited {
                        retry_after_secs: 60,
                    });
                }

                Err(LlmError::ApiError { status, message })
            }
            Err(ureq::Error::Transport(transport)) => {
                let msg = transport.to_string();
                if msg.contains("timed out") || msg.contains("timeout") {
                    return Err(LlmError::Timeout {
                        timeout_ms: self.config.timeout_ms,
                    });
                }
                Err(LlmError::NetworkError(msg))
            }
        }
    }

    /// Complete using local ONNX model.
    fn complete_local(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let model_path = self.config.local_model_path.as_ref().ok_or_else(|| {
            LlmError::ConfigError("Local model path not configured".to_string())
        })?;

        // TODO: Implement local ONNX model inference
        // For now, return an error indicating local models aren't implemented yet
        Err(LlmError::BackendUnavailable(format!(
            "Local model inference not yet implemented. Model path: {}",
            model_path
        )))
    }

    /// Complete using direct API calls (development only).
    fn complete_direct(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let provider = self.config.direct_provider.as_ref().ok_or_else(|| {
            LlmError::ConfigError("Direct provider not configured".to_string())
        })?;

        // Use cloud_llm for direct API calls
        let llm_provider: crate::pipeline::IntegrationProvider = provider
            .parse()
            .map_err(|e: String| LlmError::ConfigError(e))?;

        let client = crate::cloud_llm::LlmClient::new(llm_provider)?;
        let llm_request: crate::cloud_llm::LlmRequest = request.into();
        let response = client.complete(llm_request)?;

        let mut completion_response: CompletionResponse = response.into();
        completion_response.backend = Some(format!("direct:{}", provider));

        Ok(completion_response)
    }

    /// Auto-select backend based on availability.
    /// TODO better availabilities criterias v0.2.0 ?
    fn complete_auto(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // Try gateway first
        match self.complete_gateway(request.clone()) {
            Ok(resp) => return Ok(resp),
            Err(e) if self.config.debug => {
                eprintln!("[LLM] Gateway failed, trying fallback: {}", e);
            }
            Err(_) => {}
        }

        // Try local if configured
        if self.config.local_model_path.is_some() {
            if let Ok(resp) = self.complete_local(request.clone()) {
                return Ok(resp);
            }
        }

        // Try direct if configured
        if self.config.direct_provider.is_some() {
            return self.complete_direct(request);
        }

        Err(LlmError::BackendUnavailable(
            "No available backend. Configure gateway API key or enable local/direct mode."
                .to_string(),
        ))
    }

    /// Get the current configuration.
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }
}

impl Default for Llm {
    fn default() -> Self {
        Self::new().expect("Failed to create default LLM client")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_new() {
        let llm = Llm::new();
        assert!(llm.is_ok());
        let llm = llm.unwrap();
        assert_eq!(llm.config().backend, LlmBackend::Gateway);
    }

    #[test]
    fn test_llm_with_config() {
        let config = LlmConfig::gateway()
            .with_default_model("gpt-4o-mini")
            .with_timeout(60000);

        let llm = Llm::with_config(config).unwrap();
        assert_eq!(llm.config().default_model, Some("gpt-4o-mini".to_string()));
        assert_eq!(llm.config().timeout_ms, 60000);
    }

    #[test]
    fn test_llm_direct() {
        // This will fail without API key, but should create the client
        std::env::set_var("OPENAI_API_KEY", "test");
        let llm = Llm::direct("openai");
        assert!(llm.is_ok());
        std::env::remove_var("OPENAI_API_KEY");
    }
}
