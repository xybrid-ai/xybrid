//! Cloud client implementation.

use super::completion::{CompletionRequest, CompletionResponse};
use super::config::{CloudBackend, CloudConfig};
use super::error::CloudError;
use crate::http::{with_retry, CircuitBreaker, CircuitConfig, RetryPolicy, RetryResult};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default timeout for HTTP connections (10 seconds).
const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 10_000;

/// Cloud client for completions.
///
/// Routes requests through the configured backend:
/// - Gateway (default): Xybrid's managed gateway
/// - Direct: Direct API calls (development only)
///
/// For local/on-device inference, use `target: device` in your pipeline YAML,
/// which routes to [`crate::execution::TemplateExecutor`] instead.
///
/// # Resilience Features
///
/// The client includes production-hardening features:
/// - **Circuit breaker**: Prevents hammering failing gateway endpoints
/// - **Automatic retry**: Exponential backoff with jitter for transient failures
/// - **Connection timeouts**: Fail fast on unresponsive endpoints
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::cloud::Cloud;
///
/// let cloud = Cloud::new()?;
/// let response = cloud.prompt("Hello, world!")?;
/// println!("Response: {}", response);
/// ```
pub struct Cloud {
    config: CloudConfig,
    agent: ureq::Agent,
    /// Circuit breaker for gateway endpoint.
    gateway_circuit: Arc<CircuitBreaker>,
    /// Retry policy for gateway requests.
    retry_policy: RetryPolicy,
}

impl Cloud {
    /// Create a new cloud client with default configuration.
    /// Uses gateway backend by default.
    pub fn new() -> Result<Self, CloudError> {
        Self::with_config(CloudConfig::default())
    }

    /// Create a new cloud client with custom configuration.
    pub fn with_config(config: CloudConfig) -> Result<Self, CloudError> {
        // Configure HTTP agent with both connection and request timeouts
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(DEFAULT_CONNECT_TIMEOUT_MS))
            .timeout(Duration::from_millis(config.timeout_ms as u64))
            .build();

        // Circuit breaker: open after 3 failures, stay open for 30s
        let gateway_circuit = Arc::new(CircuitBreaker::new(CircuitConfig::default()));

        // Retry policy: 3 attempts with exponential backoff (conservative for LLM calls)
        let retry_policy = RetryPolicy::conservative();

        Ok(Self {
            config,
            agent,
            gateway_circuit,
            retry_policy,
        })
    }

    /// Create a client that uses the gateway.
    pub fn gateway() -> Result<Self, CloudError> {
        Self::with_config(CloudConfig::gateway())
    }

    /// Create a client that uses direct API calls (development only).
    pub fn direct(provider: &str) -> Result<Self, CloudError> {
        Self::with_config(CloudConfig::direct(provider))
    }

    /// Check if the gateway circuit breaker is open.
    pub fn is_circuit_open(&self) -> bool {
        self.gateway_circuit.is_open()
    }

    /// Reset the gateway circuit breaker (use with caution).
    pub fn reset_circuit(&self) {
        self.gateway_circuit.reset();
    }

    /// Send a completion request.
    ///
    /// For gateway backend, this wraps the call with retry logic and circuit breaker.
    /// For direct backend, retries are not applied (direct calls are for development only).
    pub fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, CloudError> {
        let start = Instant::now();

        let mut response = match self.config.backend {
            CloudBackend::Gateway => self.complete_via_gateway(request)?,
            CloudBackend::Direct => self.call_direct(request)?,
        };

        response.latency_ms = Some(start.elapsed().as_millis() as u32);
        Ok(response)
    }

    /// Complete via gateway with retry logic and circuit breaker.
    fn complete_via_gateway(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CloudError> {
        // Check circuit breaker before attempting
        if !self.gateway_circuit.can_execute() {
            return Err(CloudError::CircuitOpen(
                "Gateway circuit breaker is open due to recent failures. Try again later.".into(),
            ));
        }

        // Clone request data needed for retry closure
        let request_clone = request.clone();

        let result: RetryResult<CompletionResponse, CloudError> =
            with_retry(&self.retry_policy, Some(&self.gateway_circuit), || {
                self.call_gateway(request_clone.clone())
            });

        result.into_result()
    }

    /// Simple prompt completion (convenience method).
    pub fn prompt(&self, prompt: &str) -> Result<String, CloudError> {
        let request = CompletionRequest::new(prompt);
        let response = self.complete(request)?;
        Ok(response.text)
    }

    /// Chat completion with system prompt (convenience method).
    pub fn chat(&self, system: &str, user_message: &str) -> Result<String, CloudError> {
        let request = CompletionRequest::new(user_message).with_system(system);
        let response = self.complete(request)?;
        Ok(response.text)
    }

    /// Complete through Xybrid Gateway.
    fn call_gateway(&self, request: CompletionRequest) -> Result<CompletionResponse, CloudError> {
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
            eprintln!("[Cloud] Gateway request to: {}", url);
            eprintln!(
                "[Cloud] Body: {}",
                serde_json::to_string_pretty(&body).unwrap_or_default()
            );
        }

        let mut req = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");

        if let Some(ref key) = api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let response = req.send_json(&body);

        match response {
            Ok(resp) => {
                let json_resp: serde_json::Value = resp
                    .into_json()
                    .map_err(|e| CloudError::ParseError(e.to_string()))?;

                if self.config.debug {
                    eprintln!(
                        "[Cloud] Response: {}",
                        serde_json::to_string_pretty(&json_resp).unwrap_or_default()
                    );
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
                // Parse Retry-After header for rate limiting (before consuming response)
                let retry_after_secs = resp
                    .header("Retry-After")
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(60); // Default to 60 seconds if not specified

                let error_body: Result<serde_json::Value, _> = resp.into_json();
                let message = error_body
                    .ok()
                    .and_then(|v| v["error"]["message"].as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "Unknown error".into());

                match status {
                    429 => Err(CloudError::RateLimited { retry_after_secs }),
                    502..=504 => Err(CloudError::GatewayError(format!(
                        "Gateway returned {}: {}",
                        status, message
                    ))),
                    _ => Err(CloudError::ApiError { status, message }),
                }
            }
            Err(ureq::Error::Transport(transport)) => {
                let msg = transport.to_string();
                if msg.contains("timed out") || msg.contains("timeout") {
                    return Err(CloudError::Timeout {
                        timeout_ms: self.config.timeout_ms,
                    });
                }
                Err(CloudError::NetworkError(msg))
            }
        }
    }

    /// Complete using direct API calls (development only).
    fn call_direct(&self, request: CompletionRequest) -> Result<CompletionResponse, CloudError> {
        let provider =
            self.config.direct_provider.as_ref().ok_or_else(|| {
                CloudError::ConfigError("Direct provider not configured".to_string())
            })?;

        // Use cloud_llm for direct API calls
        let llm_provider: crate::pipeline::IntegrationProvider = provider
            .parse()
            .map_err(|e: String| CloudError::ConfigError(e))?;

        let client = crate::cloud_llm::LlmClient::new(llm_provider)?;
        let llm_request: crate::cloud_llm::LlmRequest = request.into();
        let response = client.complete(llm_request)?;

        let mut completion_response: CompletionResponse = response.into();
        completion_response.backend = Some(format!("direct:{}", provider));

        Ok(completion_response)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &CloudConfig {
        &self.config
    }
}

impl Default for Cloud {
    fn default() -> Self {
        Self::new().expect("Failed to create default Cloud client")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_new() {
        let cloud = Cloud::new();
        assert!(cloud.is_ok());
        let cloud = cloud.unwrap();
        assert_eq!(cloud.config().backend, CloudBackend::Gateway);
    }

    #[test]
    fn test_cloud_with_config() {
        let config = CloudConfig::gateway()
            .with_default_model("gpt-4o-mini")
            .with_timeout(60000);

        let cloud = Cloud::with_config(config).unwrap();
        assert_eq!(
            cloud.config().default_model,
            Some("gpt-4o-mini".to_string())
        );
        assert_eq!(cloud.config().timeout_ms, 60000);
    }

    #[test]
    fn test_cloud_direct() {
        // This will fail without API key, but should create the client
        std::env::set_var("OPENAI_API_KEY", "test");
        let cloud = Cloud::direct("openai");
        assert!(cloud.is_ok());
        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn test_cloud_circuit_breaker_initial_state() {
        let cloud = Cloud::new().unwrap();
        assert!(!cloud.is_circuit_open());
    }

    #[test]
    fn test_cloud_circuit_breaker_reset() {
        let cloud = Cloud::new().unwrap();

        // Manually trigger failures to open the circuit
        for _ in 0..3 {
            cloud.gateway_circuit.record_failure();
        }
        assert!(cloud.is_circuit_open());

        // Reset should close it
        cloud.reset_circuit();
        assert!(!cloud.is_circuit_open());
    }

    #[test]
    fn test_cloud_error_circuit_open() {
        let err = CloudError::CircuitOpen("test".to_string());
        assert!(matches!(err, CloudError::CircuitOpen(_)));
        assert_eq!(err.to_string(), "Circuit breaker open: test");
    }

    #[test]
    fn test_circuit_open_not_retryable() {
        use crate::http::RetryableError;

        let err = CloudError::CircuitOpen("test".to_string());
        assert!(!err.is_retryable());
    }
}
