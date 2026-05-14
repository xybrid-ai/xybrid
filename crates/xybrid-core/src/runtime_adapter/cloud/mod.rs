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

use crate::cloud::{
    parse_gateway_usage, Cloud, CloudBackend, CloudConfig, CompletionRequest, CompletionResponse,
    Role, Usage,
};
use crate::gateway::ChatCompletionChunk;
use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::IntegrationProvider;
use crate::runtime_adapter::types::{PartialToken, StreamingCallback};
use crate::runtime_adapter::{AdapterError, AdapterResult, RuntimeAdapter};
use crate::tracing as trace;
use serde_json::json;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

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
            gateway_url: CloudConfig::default().gateway_url,
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
            "deepseek" => Ok(IntegrationProvider::DeepSeek),
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
            complete_with_cloud_telemetry(&client, request)?
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

/// Cloud adapter trait for emitting response tokens incrementally.
///
/// `execute_streaming` is the seam the SDK uses to thread cloud retries
/// through `run_streaming_with_fallback`.
///
/// The default implementation on [`CloudRuntimeAdapter`] consumes
/// OpenAI-compatible Server-Sent Events from the configured Xybrid gateway.
/// The non-streaming [`RuntimeAdapter::execute`] path remains backed by
/// `Cloud::complete()` for compatibility.
pub trait CloudStreaming: Send + Sync {
    /// Stream the cloud completion as [`PartialToken`]s through `on_token`,
    /// returning the assembled [`Envelope`] (same shape as
    /// [`RuntimeAdapter::execute`]) once the stream finishes.
    fn execute_streaming(
        &self,
        input: &Envelope,
        on_token: StreamingCallback<'_>,
    ) -> AdapterResult<Envelope>;
}

impl CloudStreaming for CloudRuntimeAdapter {
    fn execute_streaming(
        &self,
        input: &Envelope,
        mut on_token: StreamingCallback<'_>,
    ) -> AdapterResult<Envelope> {
        let provider = self.get_provider(input)?;

        let model_name = input
            .metadata
            .get("model")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let _exec_span = trace::SpanGuard::new(format!("cloud_execute_streaming:{}", model_name));
        trace::add_metadata("provider", provider.as_str());
        trace::add_metadata("adapter", "cloud");
        trace::add_metadata("streaming", "sse");

        let config = self.build_config(input);
        let backend_str = match config.backend {
            CloudBackend::Gateway => "gateway",
            CloudBackend::Direct => "direct",
        };
        trace::add_metadata("backend", backend_str);

        let input_text = match &input.kind {
            EnvelopeKind::Text(text) => text.clone(),
            other => {
                return Err(AdapterError::InvalidInput(format!(
                    "Cloud adapter expects Text input, got: {:?}",
                    other
                )));
            }
        };

        let request = self.build_request(&input_text, input);

        let response = {
            let _llm_span = trace::SpanGuard::new("llm_inference");
            stream_with_gateway_sse(&config, request, &mut on_token)?
        };

        let mut output = Envelope::new(EnvelopeKind::Text(response.text));
        if let Some(backend) = response.backend {
            output.metadata.insert("backend".to_string(), backend);
        }
        output
            .metadata
            .insert("provider".to_string(), provider.as_str().to_string());
        output
            .metadata
            .insert("streaming_mode".to_string(), "sse".to_string());

        Ok(output)
    }
}

/// Issue `client.complete(request)`, time the gateway round-trip, and
/// emit `ttft_ms` + (when present) `tokens_in` / `tokens_out` on the
/// currently-active tracing span — typically the `llm_inference` span
/// the caller wraps around the call.
///
/// Token counts come from the upstream `usage` block when populated;
/// absent usage leaves the fields unset rather than writing 0 (which
/// would pollute aggregations).
fn complete_with_cloud_telemetry(
    client: &Cloud,
    request: CompletionRequest,
) -> AdapterResult<CompletionResponse> {
    let gateway_start = Instant::now();
    let response = client
        .complete(request)
        .map_err(|e| AdapterError::InferenceFailed(format!("LLM request failed: {}", e)))?;
    let gateway_rtt_ms = gateway_start.elapsed().as_millis() as u64;
    trace::add_metadata("ttft_ms", gateway_rtt_ms.to_string());
    if let Some(usage) = response.usage.as_ref() {
        trace::add_metadata("tokens_in", usage.prompt_tokens.to_string());
        trace::add_metadata("tokens_out", usage.completion_tokens.to_string());
    }
    Ok(response)
}

fn stream_with_gateway_sse(
    config: &CloudConfig,
    request: CompletionRequest,
    on_token: &mut StreamingCallback<'_>,
) -> AdapterResult<CompletionResponse> {
    if !matches!(config.backend, CloudBackend::Gateway) {
        return Err(AdapterError::RuntimeError(
            "Cloud streaming is only supported through the gateway backend".to_string(),
        ));
    }

    let body = gateway_chat_body(&request, config, true);
    let url = format!("{}/chat/completions", config.gateway_url);
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_millis(10_000))
        .timeout(Duration::from_millis(config.timeout_ms as u64))
        .build();

    if config.debug {
        eprintln!("[Cloud] Gateway stream request to: {}", url);
        eprintln!(
            "[Cloud] Body: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }

    let mut http_req = agent
        .post(&url)
        .set("Accept", "text/event-stream")
        .set("Content-Type", "application/json");

    if let Some(key) = config.resolve_api_key() {
        http_req = http_req.set("Authorization", &format!("Bearer {}", key));
    }

    let stream_start = Instant::now();
    let response = http_req
        .send_json(&body)
        .map_err(|e| gateway_stream_error(e, config.timeout_ms))?;

    let mut reader = BufReader::new(response.into_reader());
    let mut line = String::new();
    let mut cumulative = String::new();
    let mut model = request
        .model
        .clone()
        .or_else(|| config.default_model.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let mut id = None;
    let mut finish_reason = None;
    let mut usage = None;
    let mut token_index = 0usize;
    let mut terminal_emitted = false;
    let mut first_token_recorded = false;

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(AdapterError::IOError)?;
        if bytes == 0 {
            break;
        }

        let line = line.trim_end_matches(['\r', '\n']);
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim_start();
        if data == "[DONE]" {
            break;
        }
        if data.is_empty() {
            continue;
        }

        let chunk: ChatCompletionChunk = serde_json::from_str(data)
            .map_err(|e| AdapterError::SerializationError(e.to_string()))?;
        if id.is_none() {
            id = Some(chunk.id.clone());
        }
        model = chunk.model.clone();
        usage = usage.or_else(|| stream_usage_from_json(data));

        for choice in chunk.choices {
            let choice_finish = choice.finish_reason;
            let content = choice.delta.content.unwrap_or_default();

            if let Some(reason) = choice_finish.as_ref() {
                finish_reason = Some(reason.clone());
            }

            if content.is_empty() {
                continue;
            }

            cumulative.push_str(&content);
            if !first_token_recorded {
                trace::add_metadata("ttft_ms", stream_start.elapsed().as_millis().to_string());
                first_token_recorded = true;
            }

            let token = PartialToken {
                token: content,
                token_id: None,
                index: token_index,
                cumulative_text: cumulative.clone(),
                finish_reason: choice_finish.clone(),
            };
            terminal_emitted = choice_finish.is_some();
            token_index += 1;
            on_token(token).map_err(|e| {
                AdapterError::InferenceFailed(format!("streaming callback error: {}", e))
            })?;
        }
    }

    if !terminal_emitted {
        let reason = finish_reason.clone().unwrap_or_else(|| "stop".to_string());
        let token = PartialToken {
            token: String::new(),
            token_id: None,
            index: token_index,
            cumulative_text: cumulative.clone(),
            finish_reason: Some(reason.clone()),
        };
        finish_reason = Some(reason);
        on_token(token).map_err(|e| {
            AdapterError::InferenceFailed(format!("streaming callback error: {}", e))
        })?;
    }

    if !first_token_recorded {
        trace::add_metadata("ttft_ms", stream_start.elapsed().as_millis().to_string());
    }
    if let Some(usage) = usage.as_ref() {
        trace::add_metadata("tokens_in", usage.prompt_tokens.to_string());
        trace::add_metadata("tokens_out", usage.completion_tokens.to_string());
    }

    Ok(CompletionResponse {
        text: cumulative,
        model,
        finish_reason,
        usage,
        id,
        latency_ms: Some(stream_start.elapsed().as_millis() as u32),
        backend: Some("gateway".to_string()),
    })
}

fn gateway_chat_body(
    request: &CompletionRequest,
    config: &CloudConfig,
    force_stream: bool,
) -> serde_json::Value {
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

    let model = request
        .model
        .clone()
        .or_else(|| config.default_model.clone())
        .unwrap_or_else(|| "gpt-4o-mini".to_string());

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
    if force_stream || request.stream {
        body["stream"] = json!(true);
    }

    body
}

fn gateway_stream_error(error: ureq::Error, timeout_ms: u32) -> AdapterError {
    match error {
        ureq::Error::Status(status, resp) => {
            let error_body: Result<serde_json::Value, _> = resp.into_json();
            let message = error_body
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| "Unknown error".to_string());
            AdapterError::InferenceFailed(format!("Gateway returned {status}: {message}"))
        }
        ureq::Error::Transport(transport) => {
            let msg = transport.to_string();
            if msg.contains("timed out") || msg.contains("timeout") {
                AdapterError::InferenceFailed(format!(
                    "Gateway request timed out after {timeout_ms} ms"
                ))
            } else {
                AdapterError::InferenceFailed(format!("Gateway stream failed: {msg}"))
            }
        }
    }
}

fn stream_usage_from_json(data: &str) -> Option<Usage> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    value.get("usage").map(parse_gateway_usage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};

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

    #[test]
    fn gateway_chat_body_forces_stream_true() {
        let config = CloudConfig::gateway().with_default_model("default-model");
        let request = CompletionRequest::new("hello")
            .with_model("explicit-model")
            .with_temperature(0.2)
            .with_max_tokens(42);

        let body = gateway_chat_body(&request, &config, true);

        assert_eq!(body["stream"], true);
        assert_eq!(body["model"], "explicit-model");
        assert!((body["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert_eq!(body["max_tokens"], 42);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn execute_streaming_consumes_gateway_sse_in_order() {
        let sse = concat!(
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (gateway_url, request_rx) = start_sse_server(sse, 200);
        let adapter = CloudRuntimeAdapter::with_gateway(&gateway_url);
        let mut input = Envelope::new(EnvelopeKind::Text("original prompt".to_string()));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "gpt-test".to_string());

        let collected: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_for_cb = collected.clone();
        let cb: StreamingCallback<'_> = Box::new(move |t: PartialToken| {
            collected_for_cb.lock().unwrap().push(t);
            Ok(())
        });

        let output = adapter.execute_streaming(&input, cb).unwrap();
        let request = request_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        assert!(request.starts_with("POST /chat/completions "));
        assert!(request.contains("\"stream\":true"));
        assert!(request.contains("\"content\":\"original prompt\""));
        assert_eq!(output.metadata["streaming_mode"], "sse");
        assert_eq!(output.metadata["backend"], "gateway");
        assert_eq!(output.kind, EnvelopeKind::Text("hello world".to_string()));

        let tokens = collected.lock().unwrap().clone();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].token, "hello ");
        assert_eq!(tokens[0].index, 0);
        assert_eq!(tokens[0].cumulative_text, "hello ");
        assert_eq!(tokens[0].finish_reason, None);
        assert_eq!(tokens[1].token, "world");
        assert_eq!(tokens[1].index, 1);
        assert_eq!(tokens[1].cumulative_text, "hello world");
        assert_eq!(tokens[1].finish_reason, None);
        assert_eq!(tokens[2].token, "");
        assert_eq!(tokens[2].index, 2);
        assert_eq!(tokens[2].cumulative_text, "hello world");
        assert_eq!(tokens[2].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn execute_streaming_marks_content_chunk_final_when_finish_reason_coincides() {
        let sse = concat!(
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (gateway_url, _request_rx) = start_sse_server(sse, 200);
        let adapter = CloudRuntimeAdapter::with_gateway(&gateway_url);
        let mut input = Envelope::new(EnvelopeKind::Text("prompt".to_string()));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "gpt-test".to_string());

        let collected: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_for_cb = collected.clone();
        let cb: StreamingCallback<'_> = Box::new(move |t: PartialToken| {
            collected_for_cb.lock().unwrap().push(t);
            Ok(())
        });

        let output = adapter.execute_streaming(&input, cb).unwrap();

        assert_eq!(output.kind, EnvelopeKind::Text("done".to_string()));
        let tokens = collected.lock().unwrap().clone();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token, "done");
        assert_eq!(tokens[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn stream_usage_from_json_reuses_gateway_usage_parser() {
        let mut usage = serde_json::Map::new();
        usage.insert("prompt_tokens".to_string(), serde_json::json!(1000));
        usage.insert("completion_tokens".to_string(), serde_json::json!(50));
        usage.insert("total_tokens".to_string(), serde_json::json!(1050));
        usage.insert(
            format!("prompt{}cache{}hit{}tokens", "_", "_", "_"),
            serde_json::json!(800),
        );
        usage.insert(
            format!("prompt{}cache{}miss{}tokens", "_", "_", "_"),
            serde_json::json!(200),
        );

        let mut chunk = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "choices": [],
        });
        chunk["usage"] = serde_json::Value::Object(usage);

        let parsed = stream_usage_from_json(&chunk.to_string()).unwrap();

        assert_eq!(parsed.prompt_tokens, 1000);
        assert_eq!(parsed.completion_tokens, 50);
        assert_eq!(parsed.total_tokens, 1050);
        assert_eq!(parsed.cache_read_input_tokens, Some(800));
        assert_eq!(parsed.cache_creation_input_tokens, None);
    }

    #[test]
    fn execute_streaming_propagates_callback_errors() {
        let sse = concat!(
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (gateway_url, _request_rx) = start_sse_server(sse, 200);
        let adapter = CloudRuntimeAdapter::with_gateway(&gateway_url);
        let mut input = Envelope::new(EnvelopeKind::Text("prompt".to_string()));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "gpt-test".to_string());

        let cb: StreamingCallback<'_> = Box::new(|_| Err("user cancelled".into()));
        let result = adapter.execute_streaming(&input, cb);

        match result {
            Err(AdapterError::InferenceFailed(msg)) => {
                assert!(msg.contains("user cancelled"));
            }
            other => panic!("expected InferenceFailed, got {:?}", other),
        }
    }

    fn start_sse_server(body: &'static str, status: u16) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buf = [0; 1024];
            loop {
                let read = stream.read(&mut buf).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..read]);
                if request.windows(4).any(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            line.strip_prefix("Content-Length:")
                                .or_else(|| line.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    let header_end = request
                        .windows(4)
                        .position(|w| w == b"\r\n\r\n")
                        .map(|pos| pos + 4)
                        .unwrap();
                    while request.len() < header_end + content_length {
                        let read = stream.read(&mut buf).unwrap();
                        if read == 0 {
                            break;
                        }
                        request.extend_from_slice(&buf[..read]);
                    }
                    break;
                }
            }
            tx.send(String::from_utf8_lossy(&request).into_owned())
                .unwrap();

            let response = format!(
                "HTTP/1.1 {status} OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        (format!("http://{}", addr), rx)
    }
}
