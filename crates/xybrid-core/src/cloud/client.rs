//! Cloud client implementation.

use super::completion::{CompletionRequest, CompletionResponse, Role};
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
/// ```no_run
/// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
/// use xybrid_core::cloud::Cloud;
///
/// let cloud = Cloud::new()?;
/// let response = cloud.prompt("Hello, world!")?;
/// println!("Response: {}", response);
/// # Ok(())
/// # }
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

    /// Complete through an OpenAI-compatible `/chat/completions` endpoint
    /// (the Xybrid gateway or a provider reached over the same transport).
    fn call_gateway(&self, request: CompletionRequest) -> Result<CompletionResponse, CloudError> {
        let api_key = self.config.resolve_api_key();

        // No silent default: falling back to a hosted model here would route a
        // caller who merely forgot to set `model` to OpenAI, quietly billing a
        // third-party provider instead of running the model they asked for.
        let body =
            openai_chat_body(&request, &self.config, request.stream).map_err(|MissingModel| {
                CloudError::GatewayError(
                    "no model specified for the gateway request: set CompletionRequest::model or \
                     CloudConfig::default_model"
                        .to_string(),
                )
            })?;

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
                let model = json_resp["model"].as_str().unwrap_or("unknown").to_string();

                let finish_reason = json_resp["choices"][0]["finish_reason"]
                    .as_str()
                    .map(|s| s.to_string());

                let text = assistant_content(&json_resp, finish_reason.as_deref())?;

                let usage = json_resp.get("usage").map(parse_gateway_usage);

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

/// Extract `choices[0].message.content` from an OpenAI-shaped 200 response.
///
/// An HTTP 200 is not an answer. A body with no choices, a non-string
/// content, or an empty string (typically `finish_reason: length` after hidden
/// reasoning consumed the whole output budget) is reported as
/// [`CloudError::ParseError`] — non-retryable, so the caller sees the failure
/// instead of an empty "success".
fn assistant_content(
    response: &serde_json::Value,
    finish_reason: Option<&str>,
) -> Result<String, CloudError> {
    match response["choices"][0]["message"]["content"].as_str() {
        Some(content) if !content.is_empty() => Ok(content.to_string()),
        Some(_) => Err(CloudError::ParseError(format!(
            "completion returned empty content (finish_reason: {}); the output budget may have \
             been exhausted before an answer was produced",
            finish_reason.unwrap_or("none")
        ))),
        None => Err(CloudError::ParseError(
            "response has no choices[0].message.content string".to_string(),
        )),
    }
}

/// The request names no model and the config has no `default_model`.
///
/// Returned by [`openai_chat_body`] so each transport can map it to its own
/// error type and message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MissingModel;

/// Build the OpenAI-compatible `/chat/completions` request body.
///
/// The single body builder shared by the batch transport
/// (`Cloud::call_gateway`) and the SSE transport
/// (`CloudRuntimeAdapter::execute_streaming`), so the two wire bodies are
/// identical except for `stream`. Carries `model`, `messages`, `max_tokens`,
/// `temperature`, `top_p`, `stop`, `stream` (only when `true`), and DeepSeek's
/// `"thinking": {"type": ...}` when [`CompletionRequest::thinking`] is set.
///
/// The model is `request.model`, then `config.default_model`; with neither,
/// [`MissingModel`] is returned rather than guessing a hosted default.
pub(crate) fn openai_chat_body(
    request: &CompletionRequest,
    config: &CloudConfig,
    stream: bool,
) -> Result<serde_json::Value, MissingModel> {
    let model = request
        .model
        .clone()
        .or_else(|| config.default_model.clone())
        .ok_or(MissingModel)?;

    let messages: Vec<serde_json::Value> = request
        .to_messages()
        .into_iter()
        .map(|m| {
            json!({
                "role": match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                "content": m.content,
            })
        })
        .collect();

    let mut body = json!({
        "model": model,
        "messages": messages,
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
    if let Some(stop) = request.stop.as_ref() {
        body["stop"] = json!(stop);
    }
    if stream {
        body["stream"] = json!(true);
    }
    if let Some(thinking) = request.thinking {
        body["thinking"] = json!({ "type": thinking.as_str() });
    }

    Ok(body)
}

/// Parse a raw gateway `usage` JSON value into canonical `Usage`.
///
/// Maps DeepSeek's `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
/// onto `cache_read_input_tokens` + derived uncached. The DeepSeek miss
/// count isn't stored — downstream code derives it as
/// `prompt_tokens - cache_read - cache_creation`. DeepSeek has no
/// cache-creation concept; Anthropic's creation field is plumbed via
/// the direct-path adapter in `cloud_llm::response`, not this function.
pub(crate) fn parse_gateway_usage(u: &serde_json::Value) -> super::completion::Usage {
    // Presence on either field surfaces as `Some(0)` for cold-cache
    // responses (where only the miss key is present). Preserves the
    // "provider didn't report" (None) vs "cold cache" (Some(0))
    // distinction downstream.
    let has_cache_fields =
        u.get("prompt_cache_hit_tokens").is_some() || u.get("prompt_cache_miss_tokens").is_some();
    let cache_read_input_tokens = if has_cache_fields {
        Some(
            u.get("prompt_cache_hit_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        )
    } else {
        None
    };
    super::completion::Usage {
        prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
        cache_read_input_tokens,
        cache_creation_input_tokens: None,
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
    fn gateway_usage_maps_deepseek_cache_fields() {
        // Shape of a real DeepSeek `usage` block on a warm-cache call.
        let blob = serde_json::json!({
            "prompt_tokens": 1000,
            "completion_tokens": 120,
            "total_tokens": 1120,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        });
        let usage = parse_gateway_usage(&blob);
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(usage.completion_tokens, 120);
        assert_eq!(usage.cache_read_input_tokens, Some(800));
        assert_eq!(usage.cache_creation_input_tokens, None);
        // Derived uncached matches the provider's reported miss count.
        let derived_uncached = usage
            .prompt_tokens
            .saturating_sub(usage.cache_read_input_tokens.unwrap_or(0))
            .saturating_sub(usage.cache_creation_input_tokens.unwrap_or(0));
        assert_eq!(derived_uncached, 200);
    }

    #[test]
    fn gateway_usage_cold_cache_only_miss_field() {
        // First-call cold-cache response may include only the miss key.
        // Either-field presence detection should surface read as Some(0)
        // so the UI can render "0% cached" (vs hiding the receipt).
        let blob = serde_json::json!({
            "prompt_tokens": 500,
            "completion_tokens": 50,
            "total_tokens": 550,
            "prompt_cache_miss_tokens": 500,
        });
        let usage = parse_gateway_usage(&blob);
        assert_eq!(usage.cache_read_input_tokens, Some(0));
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn gateway_usage_no_cache_fields_stays_none() {
        // Providers that don't report caching at all (Groq, short-prompt
        // OpenAI) → both cache fields None.
        let blob = serde_json::json!({
            "prompt_tokens": 300,
            "completion_tokens": 30,
            "total_tokens": 330,
        });
        let usage = parse_gateway_usage(&blob);
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn openai_chat_body_serializes_thinking_as_typed_object() {
        use super::super::completion::ThinkingMode;

        let config = CloudConfig::gateway();
        let request = CompletionRequest::new("hello")
            .with_model("deepseek-flash")
            .with_thinking(ThinkingMode::Disabled);

        let body = openai_chat_body(&request, &config, false).expect("model is set");

        assert_eq!(body["thinking"], json!({ "type": "disabled" }));
        assert!(body.get("stream").is_none(), "stream is omitted when false");

        let plain = CompletionRequest::new("hello").with_model("deepseek-flash");
        let body = openai_chat_body(&plain, &config, false).expect("model is set");
        assert!(body.get("thinking").is_none(), "unset thinking is omitted");
    }

    #[test]
    fn openai_chat_body_reports_missing_model() {
        let config = CloudConfig::gateway();
        assert!(config.default_model.is_none(), "guard the premise");

        assert_eq!(
            openai_chat_body(&CompletionRequest::new("hello"), &config, false),
            Err(MissingModel)
        );

        let with_default = CloudConfig::gateway().with_default_model("lfm2.5-350m");
        let body = openai_chat_body(&CompletionRequest::new("hello"), &with_default, true)
            .expect("config supplies the model");
        assert_eq!(body["model"], "lfm2.5-350m");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn assistant_content_rejects_missing_and_empty_answers() {
        let ok = json!({"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]});
        assert_eq!(assistant_content(&ok, Some("stop")).unwrap(), "hi");

        let no_choices = json!({"choices":[]});
        assert!(matches!(
            assistant_content(&no_choices, None),
            Err(CloudError::ParseError(_))
        ));

        let empty = json!({"choices":[{"message":{"content":""},"finish_reason":"length"}]});
        match assistant_content(&empty, Some("length")) {
            Err(CloudError::ParseError(msg)) => {
                assert!(msg.contains("empty content"), "got {msg}");
                assert!(msg.contains("length"), "got {msg}");
            }
            other => panic!("expected ParseError, got {other:?}"),
        }

        let non_string = json!({"choices":[{"message":{"content":null}}]});
        assert!(matches!(
            assistant_content(&non_string, None),
            Err(CloudError::ParseError(_))
        ));
    }

    /// 401 is permanent; 429 is retryable with the server's `Retry-After`.
    /// The retry loop itself sleeps for the policy delay, so classification is
    /// what keeps a rate-limited request bounded, not wall-clock assertions.
    #[test]
    fn status_errors_classify_for_retry() {
        use crate::http::RetryableError;

        let unauthorized = CloudError::ApiError {
            status: 401,
            message: "bad key".to_string(),
        };
        assert!(!unauthorized.is_retryable());
        assert_eq!(unauthorized.retry_after(), None);

        let limited = CloudError::RateLimited {
            retry_after_secs: 7,
        };
        assert!(limited.is_retryable());
        assert_eq!(limited.retry_after(), Some(Duration::from_secs(7)));

        let empty_answer = CloudError::ParseError("empty".to_string());
        assert!(!empty_answer.is_retryable());
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
