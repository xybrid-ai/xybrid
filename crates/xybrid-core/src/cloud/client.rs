//! Cloud client implementation.

use super::completion::{CompletionRequest, CompletionResponse, Role};
use super::config::{same_origin, CloudBackend, CloudConfig};
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

        // Use cloud_llm for direct API calls. The explicit `api_key` (literal
        // or `$VAR`) and `direct_base_url` reach the native client; absent
        // values fall back to the provider's own environment variable and
        // documented base URL.
        let client = crate::cloud_llm::LlmClient::with_config(direct_provider_config(
            provider,
            &self.config,
        )?)?;
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

/// Build the native-client [`crate::pipeline::ProviderConfig`] for a
/// `backend: direct` call.
///
/// Threads the stage's explicit `api_key` (literal or `$VAR` reference) and
/// `direct_base_url` into the native client. Credential selection is
/// destination-scoped: an explicit key always wins; without one, the
/// provider's own environment variable is used only when the effective
/// destination is the provider's documented origin. A custom (loopback,
/// staging, lookalike) destination requires an explicit key and fails here —
/// before any HTTP — so the native client can never attach an ambient
/// provider credential to an unrelated endpoint.
fn direct_provider_config(
    provider: &str,
    config: &CloudConfig,
) -> Result<crate::pipeline::ProviderConfig, CloudError> {
    direct_provider_config_with_env(provider, config, |var| std::env::var(var).ok())
}

/// [`direct_provider_config`] with an injected environment lookup, so the
/// credential matrix is testable without touching process environment.
fn direct_provider_config_with_env(
    provider: &str,
    config: &CloudConfig,
    env: impl Fn(&str) -> Option<String>,
) -> Result<crate::pipeline::ProviderConfig, CloudError> {
    let llm_provider: crate::pipeline::IntegrationProvider = provider
        .parse()
        .map_err(|e: String| CloudError::ConfigError(e))?;
    let destination = config
        .direct_base_url
        .as_deref()
        .unwrap_or_else(|| llm_provider.default_base_url());
    let api_key =
        resolve_direct_api_key(llm_provider, config.api_key.as_deref(), destination, env)?;

    let mut provider_config = crate::pipeline::ProviderConfig::new(llm_provider);
    provider_config.base_url = config.direct_base_url.clone();
    provider_config.timeout_ms = config.timeout_ms;
    // Always concrete: `ProviderConfig::resolve_api_key` falls back to the
    // provider's environment variable when this is `None`, which would
    // re-open the ambient credential path the destination check just closed.
    provider_config.api_key = Some(api_key);
    Ok(provider_config)
}

/// Destination-scoped credential for a `backend: direct` native call.
///
/// Rules, in order:
/// 1. An explicit key wins. A literal is used as-is; a `$VAR` reference reads
///    only that variable.
/// 2. An empty explicit key, or a reference that resolves to nothing, is an
///    error — never a fall-through to another credential.
/// 3. With no explicit key, the provider's own environment variable is used
///    only when `destination` has the provider's documented origin.
/// 4. Any other destination demands an explicit key; the Xybrid platform key
///    (`XYBRID_API_KEY`) is never substituted for a provider key.
///
/// Errors name the configuration problem, never a secret value.
fn resolve_direct_api_key(
    provider: crate::pipeline::IntegrationProvider,
    explicit: Option<&str>,
    destination: &str,
    env: impl Fn(&str) -> Option<String>,
) -> Result<String, CloudError> {
    fn non_empty(value: Option<String>) -> Option<String> {
        value.filter(|v| !v.trim().is_empty())
    }

    if let Some(key) = explicit {
        if key.trim().is_empty() {
            return Err(CloudError::ConfigError(format!(
                "backend 'direct' for {}: 'api_key' is empty; set a literal key or a '${}' \
                 reference",
                provider,
                provider.api_key_env_var()
            )));
        }
        if let Some(var) = key.strip_prefix('$') {
            return non_empty(env(var)).ok_or_else(|| {
                CloudError::ConfigError(format!(
                    "backend 'direct' for {}: 'api_key' references ${}, which is unset or \
                     empty; refusing to fall back to another credential",
                    provider, var
                ))
            });
        }
        return Ok(key.to_string());
    }

    if same_origin(destination, provider.default_base_url()) {
        return non_empty(env(provider.api_key_env_var())).ok_or_else(|| {
            CloudError::ConfigError(format!(
                "no API key for {}: set {} or the stage's 'api_key'",
                provider,
                provider.api_key_env_var()
            ))
        });
    }

    Err(CloudError::ConfigError(format!(
        "backend 'direct' for {} at a custom destination ('{}') requires an explicit \
         'api_key'; refusing to send {} to a different origin",
        provider,
        destination,
        provider.api_key_env_var()
    )))
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

    /// Environment lookup backed by a literal list, so credential selection
    /// is tested without touching process environment.
    fn env_with<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |var| {
            pairs
                .iter()
                .find(|(name, _)| *name == var)
                .map(|(_, value)| value.to_string())
        }
    }

    /// Minimal loopback Anthropic `/messages` responder that records every
    /// request verbatim, for exercising the public `Cloud` entry point.
    struct FakeAnthropic {
        base_url: String,
        requests: Arc<std::sync::Mutex<Vec<String>>>,
        stop: Arc<std::sync::atomic::AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl FakeAnthropic {
        fn start() -> Self {
            use std::io::ErrorKind;
            use std::net::TcpListener;

            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let thread = {
                let requests = requests.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                        match listener.accept() {
                            Ok((stream, _)) => Self::serve(stream, &requests),
                            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                                std::thread::sleep(Duration::from_millis(10));
                            }
                            Err(_) => break,
                        }
                    }
                })
            };
            Self {
                base_url: format!("http://{addr}"),
                requests,
                stop,
                thread: Some(thread),
            }
        }

        fn serve(mut stream: std::net::TcpStream, requests: &Arc<std::sync::Mutex<Vec<String>>>) {
            use std::io::{Read, Write};

            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let read = match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                request.extend_from_slice(&buf[..read]);
                if let Some(header_end) = request
                    .windows(4)
                    .position(|w| w == b"\r\n\r\n")
                    .map(|pos| pos + 4)
                {
                    let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    while request.len() < header_end + content_length {
                        match stream.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => request.extend_from_slice(&buf[..n]),
                        }
                    }
                    break;
                }
            }
            requests
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(&request).into_owned());

            let body = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","content":[{"type":"text","text":"FAKE_ANTHROPIC_REPLY"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
        }

        fn requests(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for FakeAnthropic {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    #[test]
    fn direct_credential_explicit_literal_wins_for_any_destination() {
        let env = env_with(&[
            ("ANTHROPIC_API_KEY", "env-anthropic"),
            ("XYBRID_API_KEY", "env-platform"),
        ]);

        for destination in ["https://api.anthropic.com/v1", "http://127.0.0.1:8080/v1"] {
            let key = resolve_direct_api_key(
                crate::pipeline::IntegrationProvider::Anthropic,
                Some("literal-key"),
                destination,
                &env,
            )
            .expect("literal key wins");
            assert_eq!(key, "literal-key");
        }
    }

    #[test]
    fn direct_credential_env_reference_reads_only_that_variable() {
        let env = env_with(&[
            ("ANTHROPIC_API_KEY", "env-anthropic"),
            ("MY_KEY", "my-key"),
            ("EMPTY_KEY", "   "),
            ("XYBRID_API_KEY", "env-platform"),
        ]);

        let key = resolve_direct_api_key(
            crate::pipeline::IntegrationProvider::Anthropic,
            Some("$MY_KEY"),
            "https://api.anthropic.com/v1",
            &env,
        )
        .expect("reference resolves");
        assert_eq!(key, "my-key");

        for destination in ["https://api.anthropic.com/v1", "http://127.0.0.1:8080/v1"] {
            for reference in ["$MISSING_KEY", "$EMPTY_KEY"] {
                let err = resolve_direct_api_key(
                    crate::pipeline::IntegrationProvider::Anthropic,
                    Some(reference),
                    destination,
                    &env,
                )
                .expect_err("unresolved reference must fail");
                assert!(matches!(err, CloudError::ConfigError(_)), "{err:?}");
                assert!(err.to_string().contains("unset or empty"), "{err}");
            }

            let err = resolve_direct_api_key(
                crate::pipeline::IntegrationProvider::Anthropic,
                Some("   "),
                destination,
                &env,
            )
            .expect_err("whitespace literal is empty");
            assert!(err.to_string().contains("is empty"), "{err}");
        }
    }

    #[test]
    fn direct_credential_provider_env_only_at_provider_origin() {
        let env = env_with(&[
            ("ANTHROPIC_API_KEY", "env-anthropic"),
            ("XYBRID_API_KEY", "env-platform"),
        ]);

        for destination in [
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com:443/v1/",
        ] {
            let key = resolve_direct_api_key(
                crate::pipeline::IntegrationProvider::Anthropic,
                None,
                destination,
                &env,
            )
            .expect("provider origin may use the provider env var");
            assert_eq!(key, "env-anthropic");
        }

        // Custom, loopback, different port, downgraded scheme, and lookalike
        // hosts must all demand an explicit key and must never receive the
        // platform key or the provider env var.
        for destination in [
            "http://127.0.0.1:8080/v1",
            "http://api.anthropic.com/v1",
            "https://api.anthropic.com:8443/v1",
            "https://api.anthropic.com.evil.example/v1",
            "https://evil.example/api.anthropic.com/v1",
        ] {
            let err = resolve_direct_api_key(
                crate::pipeline::IntegrationProvider::Anthropic,
                None,
                destination,
                &env,
            )
            .expect_err("custom destination must require an explicit key");
            assert!(matches!(err, CloudError::ConfigError(_)), "{err:?}");
            let message = err.to_string();
            assert!(
                message.contains("explicit 'api_key'"),
                "{destination}: {message}"
            );
            assert!(!message.contains("env-anthropic"), "{message}");
            assert!(!message.contains("env-platform"), "{message}");
        }

        // The provider origin with no provider env var fails instead of
        // reaching for the platform key.
        let platform_only = env_with(&[("XYBRID_API_KEY", "env-platform")]);
        let err = resolve_direct_api_key(
            crate::pipeline::IntegrationProvider::Anthropic,
            None,
            "https://api.anthropic.com/v1",
            &platform_only,
        )
        .expect_err("missing provider key");
        let message = err.to_string();
        assert!(message.contains("ANTHROPIC_API_KEY"), "{message}");
        assert!(!message.contains("XYBRID"), "{message}");
    }

    #[test]
    fn direct_provider_config_threads_explicit_key_url_and_timeout() {
        let config = CloudConfig {
            backend: CloudBackend::Direct,
            direct_provider: Some("anthropic".to_string()),
            direct_base_url: Some("http://127.0.0.1:8080/v1".to_string()),
            api_key: Some("sk-ant-explicit".to_string()),
            timeout_ms: 1234,
            ..Default::default()
        };

        let env = env_with(&[("ANTHROPIC_API_KEY", "env-anthropic")]);
        let provider_config =
            direct_provider_config_with_env("anthropic", &config, &env).expect("valid provider");

        assert_eq!(
            provider_config.base_url.as_deref(),
            Some("http://127.0.0.1:8080/v1")
        );
        assert_eq!(provider_config.api_key.as_deref(), Some("sk-ant-explicit"));
        assert_eq!(provider_config.timeout_ms, 1234);
        // The explicit key is what the client resolves, whatever the origin.
        assert_eq!(
            provider_config.resolve_api_key().as_deref(),
            Some("sk-ant-explicit")
        );
    }

    #[test]
    fn direct_provider_config_resolves_provider_env_at_its_own_origin() {
        let config = CloudConfig::direct("anthropic");
        let env = env_with(&[("ANTHROPIC_API_KEY", "sk-ant-env")]);

        let provider_config =
            direct_provider_config_with_env("anthropic", &config, &env).expect("valid provider");

        assert!(
            provider_config.base_url.is_none(),
            "None must fall back to the provider's documented base URL"
        );
        assert_eq!(
            provider_config.effective_base_url(),
            "https://api.anthropic.com/v1"
        );
        // Resolved eagerly to a literal; the native client must not re-read
        // the environment at request time.
        assert_eq!(provider_config.api_key.as_deref(), Some("sk-ant-env"));
    }

    #[test]
    fn direct_provider_config_rejects_custom_destination_without_explicit_key() {
        let config = CloudConfig {
            backend: CloudBackend::Direct,
            direct_provider: Some("anthropic".to_string()),
            direct_base_url: Some("http://127.0.0.1:9/v1".to_string()),
            ..Default::default()
        };
        let env = env_with(&[("ANTHROPIC_API_KEY", "env-anthropic")]);

        let err = direct_provider_config_with_env("anthropic", &config, &env)
            .expect_err("custom destination without an explicit key");

        assert!(matches!(err, CloudError::ConfigError(_)), "{err:?}");
        assert!(err.to_string().contains("custom destination"), "{err}");
    }

    #[test]
    fn direct_cloud_refuses_custom_origin_without_explicit_key_and_makes_no_request() {
        let server = FakeAnthropic::start();
        let config = CloudConfig {
            backend: CloudBackend::Direct,
            direct_provider: Some("anthropic".to_string()),
            direct_base_url: Some(format!("{}/v1", server.base_url)),
            ..Default::default()
        };

        let cloud = Cloud::with_config(config).expect("client construction");
        let err = cloud
            .complete(CompletionRequest::new("hello"))
            .expect_err("custom destination without a key must fail");

        assert!(matches!(err, CloudError::ConfigError(_)), "{err:?}");
        assert!(
            server.requests().is_empty(),
            "no HTTP before the configuration error: {:?}",
            server.requests()
        );
    }

    #[test]
    fn direct_cloud_sends_only_the_explicit_key_to_a_custom_origin() {
        let server = FakeAnthropic::start();
        let config = CloudConfig {
            backend: CloudBackend::Direct,
            direct_provider: Some("anthropic".to_string()),
            direct_base_url: Some(server.base_url.clone()),
            api_key: Some("sk-ant-explicit".to_string()),
            ..Default::default()
        };

        let cloud = Cloud::with_config(config).expect("client construction");
        let response = cloud
            .complete(CompletionRequest::new("hello").with_max_tokens(16))
            .expect("explicit key request");

        assert_eq!(response.text, "FAKE_ANTHROPIC_REPLY");
        let requests = server.requests();
        assert_eq!(requests.len(), 1, "{requests:?}");
        let request = &requests[0];
        assert!(request.starts_with("POST /messages "), "{request}");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("x-api-key: sk-ant-explicit"),
            "{request}"
        );
    }

    #[test]
    fn direct_provider_config_rejects_unknown_provider() {
        let config = CloudConfig::direct("nope");
        let err = direct_provider_config("nope", &config).expect_err("unknown provider");
        assert!(matches!(err, CloudError::ConfigError(_)), "{err:?}");
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
