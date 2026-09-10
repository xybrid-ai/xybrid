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
//! ```no_run
//! use xybrid_core::runtime_adapter::CloudRuntimeAdapter;
//!
//! let adapter = CloudRuntimeAdapter::new();
//! // Or with custom gateway URL:
//! let adapter = CloudRuntimeAdapter::with_gateway("https://my-gateway.example.com");
//! ```

use crate::cloud::{
    openai_chat_body, parse_gateway_usage, same_origin, Cloud, CloudBackend, CloudConfig,
    CompletionRequest, CompletionResponse, MissingModel, ThinkingMode, Usage,
};
use crate::gateway::ChatCompletionChunk;
use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::IntegrationProvider;
use crate::runtime_adapter::types::{
    parse_stop_sequences, PartialToken, StreamingCallback, STOP_SEQUENCES_METADATA_KEY,
};
use crate::runtime_adapter::{AdapterError, AdapterResult, RuntimeAdapter};
use crate::tracing as trace;
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
        provider_str
            .parse::<IntegrationProvider>()
            .map_err(AdapterError::InvalidInput)
    }

    /// Builds CloudConfig from envelope metadata.
    ///
    /// Transport selection:
    /// - `backend` omitted or `gateway`: the adapter's gateway URL (or the
    ///   `gateway_url` override) over the OpenAI-compatible transport.
    /// - `backend: direct` with an OpenAI-compatible provider (OpenAI,
    ///   DeepSeek, OpenRouter, Custom): the same transport pointed at the
    ///   provider's documented base URL (`gateway_url` may override; Custom
    ///   requires it).
    /// - `backend: direct` with Anthropic: the native direct client, using
    ///   the provider's documented base URL (a stage `gateway_url` overrides)
    ///   and its explicit `api_key` when set, then `$ANTHROPIC_API_KEY`.
    ///   Google and ElevenLabs have no native client and are rejected here.
    ///
    /// Credentials are scoped to the destination: an explicit `api_key` always
    /// wins; otherwise the provider's own `$<PROVIDER>_API_KEY` is selected
    /// when the destination is that provider's origin; the Xybrid platform key
    /// is supplied only for the configured platform gateway origin (see
    /// [`CloudConfig::resolve_api_key`]); anything else is anonymous. A known
    /// provider origin with no resolvable key fails here, before any HTTP.
    ///
    /// # Errors
    ///
    /// [`AdapterError::InvalidInput`] for an unknown `backend`, a malformed or
    /// non-HTTP URL, `custom` + `direct` without a `gateway_url`, a provider
    /// with no native direct client, or a known provider origin without a
    /// usable key.
    fn build_config(
        &self,
        envelope: &Envelope,
        provider: IntegrationProvider,
    ) -> AdapterResult<CloudConfig> {
        let mut config = CloudConfig {
            gateway_url: self.gateway_url.clone(),
            timeout_ms: self.timeout_ms,
            debug: self.debug,
            ..Default::default()
        };

        if let Some(timeout_str) = envelope.metadata.get("timeout_ms") {
            if let Ok(timeout) = timeout_str.parse::<u32>() {
                config.timeout_ms = timeout;
            }
        }
        if let Some(debug_str) = envelope.metadata.get("debug") {
            config.debug = debug_str == "true";
        }

        let explicit_url = envelope.metadata.get("gateway_url").cloned();
        let explicit_key = envelope.metadata.get("api_key").cloned();
        let backend = envelope
            .metadata
            .get("backend")
            .map(|b| b.trim().to_ascii_lowercase());

        match backend.as_deref() {
            None | Some("gateway") => {
                config.backend = CloudBackend::Gateway;
                if let Some(url) = explicit_url {
                    config.gateway_url = url;
                }
            }
            Some("direct") if openai_compatible(provider) => {
                config.backend = CloudBackend::Gateway;
                config.gateway_url = match explicit_url {
                    Some(url) => url,
                    None => {
                        let base = provider.default_base_url();
                        if base.is_empty() {
                            return Err(AdapterError::InvalidInput(format!(
                                "provider '{}' with backend 'direct' requires a 'gateway_url'",
                                provider
                            )));
                        }
                        base.to_string()
                    }
                };
            }
            Some("direct") => {
                // Native direct client. Only the providers LlmClient can
                // actually serve; OpenAI-compatible providers were handled
                // above and ride the gateway transport.
                if !native_direct_supported(provider) {
                    return Err(AdapterError::InvalidInput(format!(
                        "provider '{}' with backend 'direct' is not supported: the native \
                         direct client serves anthropic only. Use backend 'gateway' with an \
                         explicit gateway_url, or an OpenAI-compatible provider (openai, \
                         deepseek, openrouter, custom)",
                        provider
                    )));
                }
                config.backend = CloudBackend::Direct;
                config.direct_provider = Some(provider.as_str().to_string());
                config.direct_base_url = match explicit_url {
                    Some(url) => Some(normalize_gateway_url(&url)?),
                    None => None,
                };
                config.api_key = explicit_key;
                return Ok(config);
            }
            Some(other) => {
                return Err(AdapterError::InvalidInput(format!(
                    "unknown backend '{}' (expected 'gateway' or 'direct')",
                    other
                )));
            }
        }

        config.gateway_url = normalize_gateway_url(&config.gateway_url)?;
        config.api_key = match explicit_key {
            Some(key) => Some(key),
            None => provider_key_for_destination(provider, &config.gateway_url),
        };

        // Fail fast: a request to a known provider origin without a usable key
        // would only produce a 401 after the round trip.
        if let Some(origin_provider) = provider_for_origin(&config.gateway_url) {
            if config.resolve_api_key().is_none() {
                return Err(AdapterError::InvalidInput(format!(
                    "no API key for {}: set {} or the stage's 'api_key'",
                    origin_provider,
                    origin_provider.api_key_env_var()
                )));
            }
        }

        Ok(config)
    }

    /// Builds CompletionRequest from envelope metadata.
    ///
    /// # Errors
    ///
    /// [`AdapterError::InvalidInput`] when `thinking` metadata is present for a
    /// provider other than DeepSeek, or holds anything but `enabled` /
    /// `disabled` (case-insensitive).
    fn build_request(
        &self,
        input_text: &str,
        envelope: &Envelope,
        provider: IntegrationProvider,
    ) -> AdapterResult<CompletionRequest> {
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

        // Top-p (nucleus) sampling
        if let Some(top_p_str) = envelope.metadata.get("top_p") {
            if let Ok(top_p) = top_p_str.parse::<f32>() {
                request = request.with_top_p(top_p);
            }
        }

        // Stop sequences
        if let Some(stop_str) = envelope.metadata.get(STOP_SEQUENCES_METADATA_KEY) {
            let stop = parse_stop_sequences(stop_str);
            if !stop.is_empty() {
                request = request.with_stop(stop);
            }
        }

        // Thinking mode: a DeepSeek request capability (`"thinking": {"type":
        // ...}`), not a generic OpenAI field. Rejecting it for other providers
        // keeps a stage option from being silently dropped on the floor.
        if let Some(raw) = envelope.metadata.get("thinking") {
            if provider != IntegrationProvider::DeepSeek {
                return Err(AdapterError::InvalidInput(format!(
                    "'thinking' is only supported for provider 'deepseek', not '{}'",
                    provider
                )));
            }
            let mode: ThinkingMode = raw.parse().map_err(AdapterError::InvalidInput)?;
            request = request.with_thinking(mode);
        }

        Ok(request)
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
        let config = self.build_config(input, provider)?;
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
        let request = self.build_request(&input_text, input, provider)?;

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

/// Providers whose API speaks the OpenAI chat-completions dialect and can be
/// reached over the gateway transport with a bearer key.
fn openai_compatible(provider: IntegrationProvider) -> bool {
    matches!(
        provider,
        IntegrationProvider::OpenAI
            | IntegrationProvider::DeepSeek
            | IntegrationProvider::OpenRouter
            | IntegrationProvider::Custom
    )
}

/// Providers the native direct client (`LlmClient`) can serve.
///
/// OpenAI and the OpenAI-compatible providers never reach this check: they
/// ride the gateway transport.
fn native_direct_supported(provider: IntegrationProvider) -> bool {
    matches!(provider, IntegrationProvider::Anthropic)
}

const KNOWN_PROVIDERS: [IntegrationProvider; 6] = [
    IntegrationProvider::OpenAI,
    IntegrationProvider::Anthropic,
    IntegrationProvider::Google,
    IntegrationProvider::DeepSeek,
    IntegrationProvider::ElevenLabs,
    IntegrationProvider::OpenRouter,
];

/// The provider whose documented origin `url` points at, if any. Compared by
/// origin (scheme + host + port), never by substring.
fn provider_for_origin(url: &str) -> Option<IntegrationProvider> {
    KNOWN_PROVIDERS
        .into_iter()
        .find(|candidate| same_origin(url, candidate.default_base_url()))
}

/// Automatic credential for `destination`: the provider's own environment
/// variable, and only when the destination is *that* provider's origin. A
/// platform gateway, a different provider's origin, a lookalike host or a
/// loopback server gets nothing here (the platform key is handled by
/// [`CloudConfig::resolve_api_key`]).
fn provider_key_for_destination(
    provider: IntegrationProvider,
    destination: &str,
) -> Option<String> {
    (provider_for_origin(destination) == Some(provider))
        .then(|| format!("${}", provider.api_key_env_var()))
}

/// Validate an `http(s)` base URL and strip trailing slashes so
/// `/chat/completions` can be appended without a double slash.
fn normalize_gateway_url(url: &str) -> AdapterResult<String> {
    let trimmed = url.trim();
    let parsed = url::Url::parse(trimmed)
        .map_err(|e| AdapterError::InvalidInput(format!("invalid gateway_url '{trimmed}': {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(AdapterError::InvalidInput(format!(
            "invalid gateway_url '{trimmed}': expected an http(s) URL"
        )));
    }
    Ok(trimmed.trim_end_matches('/').to_string())
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

        let config = self.build_config(input, provider)?;
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

        let request = self.build_request(&input_text, input, provider)?;

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

    let body = gateway_chat_body(&request, config, true)?;
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
                // The cloud gateway rejects tool-bearing requests today, so a
                // cloud stream never carries parsed calls.
                tool_calls: Vec::new(),
                raw_text: None,
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
            tool_calls: Vec::new(),
            raw_text: None,
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

/// Build the SSE variant of the OpenAI-compatible chat body.
///
/// Thin wrapper over the shared [`openai_chat_body`] builder (also used by the
/// batch transport) that forces `stream: true` and maps a missing model to
/// [`AdapterError::InvalidInput`]. This used to fall back to `"gpt-4o-mini"`,
/// which the gateway routes to OpenAI — so a caller that simply forgot the
/// model silently billed a third-party provider instead of running the model it
/// asked for.
fn gateway_chat_body(
    request: &CompletionRequest,
    config: &CloudConfig,
    force_stream: bool,
) -> AdapterResult<serde_json::Value> {
    openai_chat_body(request, config, force_stream || request.stream).map_err(|MissingModel| {
        AdapterError::InvalidInput(
            "no model specified for the cloud request: set it on the envelope's `model` \
             metadata or via CloudConfig::default_model"
                .to_string(),
        )
    })
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
    use serde_json::json;
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

    /// The adapter reads every request setting off envelope metadata, so a key
    /// it ignores is a caller value silently dropped. `top_p` and
    /// `stop_sequences` were ignored even though `gateway_chat_body` already
    /// serialises both.
    #[test]
    fn build_request_reads_top_p_and_stop_sequences_from_metadata() {
        let adapter = CloudRuntimeAdapter::new();
        let mut input = Envelope::new(EnvelopeKind::Text("hello".to_string()));
        input
            .metadata
            .insert("top_p".to_string(), "0.72".to_string());
        input.metadata.insert(
            STOP_SEQUENCES_METADATA_KEY.to_string(),
            r#"["STOP","END"]"#.to_string(),
        );

        let request = adapter
            .build_request("hello", &input, IntegrationProvider::OpenAI)
            .unwrap();

        assert!((request.top_p.unwrap() - 0.72).abs() < 1e-6);
        assert_eq!(
            request.stop.as_deref(),
            Some(["STOP".to_string(), "END".to_string()].as_slice())
        );
    }

    /// An empty value must not become `"stop": []` on the wire.
    #[test]
    fn build_request_omits_an_empty_stop_list() {
        let adapter = CloudRuntimeAdapter::new();
        let mut input = Envelope::new(EnvelopeKind::Text("hello".to_string()));
        input
            .metadata
            .insert(STOP_SEQUENCES_METADATA_KEY.to_string(), String::new());

        let request = adapter
            .build_request("hello", &input, IntegrationProvider::OpenAI)
            .unwrap();

        assert!(request.stop.is_none());
    }

    #[test]
    fn build_request_parses_thinking_for_deepseek_only() {
        let adapter = CloudRuntimeAdapter::new();
        let mut input = Envelope::new(EnvelopeKind::Text("hello".to_string()));
        input
            .metadata
            .insert("thinking".to_string(), "Disabled".to_string());

        let request = adapter
            .build_request("hello", &input, IntegrationProvider::DeepSeek)
            .unwrap();
        assert_eq!(request.thinking, Some(ThinkingMode::Disabled));

        input
            .metadata
            .insert("thinking".to_string(), "enabled".to_string());
        let request = adapter
            .build_request("hello", &input, IntegrationProvider::DeepSeek)
            .unwrap();
        assert_eq!(request.thinking, Some(ThinkingMode::Enabled));

        // Other providers reject the option instead of dropping it.
        let err = adapter
            .build_request("hello", &input, IntegrationProvider::OpenAI)
            .expect_err("thinking is DeepSeek-only");
        assert!(
            matches!(err, AdapterError::InvalidInput(ref m) if m.contains("deepseek") && m.contains("openai")),
            "unexpected error: {err:?}"
        );

        // Unknown values are rejected.
        input
            .metadata
            .insert("thinking".to_string(), "maybe".to_string());
        let err = adapter
            .build_request("hello", &input, IntegrationProvider::DeepSeek)
            .expect_err("invalid thinking value");
        assert!(
            matches!(err, AdapterError::InvalidInput(ref m) if m.contains("maybe")),
            "unexpected error: {err:?}"
        );

        // Absent metadata leaves the field unset (provider default applies).
        input.metadata.remove("thinking");
        let request = adapter
            .build_request("hello", &input, IntegrationProvider::DeepSeek)
            .unwrap();
        assert_eq!(request.thinking, None);
    }

    /// End to end through the body serialiser: metadata in, wire fields out.
    #[test]
    fn gateway_chat_body_carries_top_p_and_stop_from_metadata() {
        let adapter = CloudRuntimeAdapter::new();
        let config = CloudConfig::gateway();
        let mut input = Envelope::new(EnvelopeKind::Text("hello".to_string()));
        input
            .metadata
            .insert("model".to_string(), "lfm2.5-350m".to_string());
        input
            .metadata
            .insert("top_p".to_string(), "0.72".to_string());
        input.metadata.insert(
            STOP_SEQUENCES_METADATA_KEY.to_string(),
            r#"["STOP","END"]"#.to_string(),
        );

        let request = adapter
            .build_request("hello", &input, IntegrationProvider::OpenAI)
            .unwrap();
        let body = gateway_chat_body(&request, &config, false).expect("model is set");

        assert!((body["top_p"].as_f64().unwrap() - 0.72).abs() < 1e-6);
        assert_eq!(body["stop"], json!(["STOP", "END"]));
    }

    /// Batch and SSE share one body builder: the only difference on the wire
    /// is `stream`, and `thinking` plus every generation option ride both.
    #[test]
    fn batch_and_sse_bodies_match_except_stream() {
        let config = CloudConfig::gateway();
        let request = CompletionRequest::new("hello")
            .with_model("deepseek-flash")
            .with_system("You are terse.")
            .with_temperature(0.0)
            .with_max_tokens(16)
            .with_top_p(0.9)
            .with_stop(vec!["END".to_string()])
            .with_thinking(ThinkingMode::Disabled);

        let batch = openai_chat_body(&request, &config, request.stream).expect("model is set");
        let mut sse = gateway_chat_body(&request, &config, true).expect("model is set");

        assert!(batch.get("stream").is_none());
        assert_eq!(sse["stream"], true);
        sse.as_object_mut().unwrap().remove("stream");
        assert_eq!(batch, sse);

        assert_eq!(batch["model"], "deepseek-flash");
        assert_eq!(batch["thinking"], json!({ "type": "disabled" }));
        assert_eq!(batch["messages"][0]["role"], "system");
        assert_eq!(batch["messages"][0]["content"], "You are terse.");
        assert_eq!(batch["messages"][1]["role"], "user");
        assert_eq!(batch["messages"][1]["content"], "hello");
        assert_eq!(batch["max_tokens"], 16);
        assert_eq!(batch["temperature"], 0.0);
        assert!((batch["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(batch["stop"], json!(["END"]));
    }

    #[test]
    fn gateway_chat_body_forces_stream_true() {
        let config = CloudConfig::gateway().with_default_model("default-model");
        let request = CompletionRequest::new("hello")
            .with_model("explicit-model")
            .with_temperature(0.2)
            .with_max_tokens(42);

        let body = gateway_chat_body(&request, &config, true).expect("model is set");

        assert_eq!(body["stream"], true);
        assert_eq!(body["model"], "explicit-model");
        assert!((body["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert_eq!(body["max_tokens"], 42);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    /// A request naming no model must fail loudly. It used to default to
    /// `gpt-4o-mini`, so a caller who forgot `model` silently ran (and paid
    /// for) OpenAI instead of the model they meant — and the resulting
    /// upstream failure surfaced as an opaque gateway error.
    #[test]
    fn gateway_chat_body_rejects_a_missing_model() {
        let config = CloudConfig::gateway();
        assert!(config.default_model.is_none(), "guard the premise");
        let request = CompletionRequest::new("hello");

        let err = gateway_chat_body(&request, &config, false)
            .expect_err("a model-less request must not fall back to a hosted default");

        assert!(
            matches!(err, AdapterError::InvalidInput(ref m) if m.contains("no model specified")),
            "unexpected error: {err:?}"
        );
    }

    /// The config default still satisfies the requirement.
    #[test]
    fn gateway_chat_body_accepts_the_config_default_model() {
        let config = CloudConfig::gateway().with_default_model("lfm2.5-350m");
        let request = CompletionRequest::new("hello");

        let body = gateway_chat_body(&request, &config, false).expect("config supplies the model");

        assert_eq!(body["model"], "lfm2.5-350m");
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

    /// One-shot JSON server: captures the raw request (headers + body) and
    /// answers with `body` under `status`.
    fn start_json_server(body: &'static str, status: u16) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut buf = [0; 1024];
            loop {
                let read = stream.read(&mut buf).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..read]);
                if let Some(header_end) = request
                    .windows(4)
                    .position(|w| w == b"\r\n\r\n")
                    .map(|pos| pos + 4)
                {
                    let headers = String::from_utf8_lossy(&request[..header_end]);
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
                "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        (format!("http://{}", addr), rx)
    }

    const FAKE_DEEPSEEK_BODY: &str = r#"{"id":"chatcmpl-1","model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant","content":"FAKE_DEEPSEEK_REPLY"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#;

    /// A `$VAR` reference that no environment will have set.
    const SURELY_UNSET_KEY_REF: &str = "$XYBRID_TEST_SURELY_UNSET_KEY_7f3a";

    fn text_with(pairs: &[(&str, &str)]) -> Envelope {
        let mut envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));
        for (key, value) in pairs {
            envelope.metadata.insert(key.to_string(), value.to_string());
        }
        envelope
    }

    fn header_value<'a>(request: &'a str, name: &str) -> Option<&'a str> {
        request.lines().find_map(|line| {
            let (candidate, value) = line.split_once(':')?;
            candidate.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }

    // ── Phase 7: provider parsing, transport mapping, credential scoping ───

    #[test]
    fn get_provider_uses_from_str_aliases_and_rejects_unknown() {
        let adapter = CloudRuntimeAdapter::new();
        for (raw, expected) in [
            ("deepseek", IntegrationProvider::DeepSeek),
            ("Deep_Seek", IntegrationProvider::DeepSeek),
            ("openrouter", IntegrationProvider::OpenRouter),
            ("custom", IntegrationProvider::Custom),
            ("claude", IntegrationProvider::Anthropic),
        ] {
            let provider = adapter
                .get_provider(&text_with(&[("provider", raw)]))
                .unwrap_or_else(|e| panic!("{raw}: {e}"));
            assert_eq!(provider, expected);
        }
        let err = adapter
            .get_provider(&text_with(&[("provider", "nope")]))
            .expect_err("unknown provider");
        assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn direct_openai_compatible_providers_use_the_gateway_transport_at_the_provider_origin() {
        let adapter = CloudRuntimeAdapter::new();
        for (name, provider) in [
            ("deepseek", IntegrationProvider::DeepSeek),
            ("openai", IntegrationProvider::OpenAI),
            ("openrouter", IntegrationProvider::OpenRouter),
        ] {
            // Explicit literal key keeps the check independent of the env.
            let envelope =
                text_with(&[("provider", name), ("backend", "direct"), ("api_key", "k")]);
            let config = adapter
                .build_config(&envelope, provider)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(config.backend, CloudBackend::Gateway, "{name}");
            assert_eq!(config.gateway_url, provider.default_base_url(), "{name}");
            assert_eq!(config.api_key.as_deref(), Some("k"), "{name}");
            assert_eq!(config.direct_provider, None, "{name}");
        }
    }

    #[test]
    fn direct_custom_requires_a_gateway_url() {
        let adapter = CloudRuntimeAdapter::new();
        let err = adapter
            .build_config(
                &text_with(&[("provider", "custom"), ("backend", "direct")]),
                IntegrationProvider::Custom,
            )
            .expect_err("custom without a URL");
        assert!(err.to_string().contains("gateway_url"), "{err}");

        let config = adapter
            .build_config(
                &text_with(&[
                    ("provider", "custom"),
                    ("backend", "direct"),
                    ("gateway_url", "http://127.0.0.1:9/v1/"),
                ]),
                IntegrationProvider::Custom,
            )
            .expect("custom with a URL");
        assert_eq!(config.backend, CloudBackend::Gateway);
        assert_eq!(config.gateway_url, "http://127.0.0.1:9/v1");
        // Loopback: anonymous unless a key is given explicitly.
        assert_eq!(config.api_key, None);
    }

    #[test]
    fn direct_anthropic_keeps_the_native_direct_backend() {
        let adapter = CloudRuntimeAdapter::new();
        let config = adapter
            .build_config(
                &text_with(&[("provider", "anthropic"), ("backend", "direct")]),
                IntegrationProvider::Anthropic,
            )
            .expect("anthropic direct");
        assert_eq!(config.backend, CloudBackend::Direct);
        assert_eq!(config.direct_provider.as_deref(), Some("anthropic"));
        // No explicit URL/key: provider default URL and env-var fallback.
        assert_eq!(config.direct_base_url, None);
        assert_eq!(config.api_key, None);

        // Explicit URL and key are stored (normalized) for the native client.
        let config = adapter
            .build_config(
                &text_with(&[
                    ("provider", "anthropic"),
                    ("backend", "direct"),
                    ("gateway_url", "http://127.0.0.1:8080/v1/"),
                    ("api_key", "$ANTHROPIC_API_KEY"),
                ]),
                IntegrationProvider::Anthropic,
            )
            .expect("anthropic direct with overrides");
        assert_eq!(
            config.direct_base_url.as_deref(),
            Some("http://127.0.0.1:8080/v1")
        );
        assert_eq!(config.api_key.as_deref(), Some("$ANTHROPIC_API_KEY"));
    }

    #[test]
    fn direct_native_providers_without_a_client_are_rejected() {
        let adapter = CloudRuntimeAdapter::new();

        for provider in [IntegrationProvider::Google, IntegrationProvider::ElevenLabs] {
            let err = adapter
                .build_config(
                    &text_with(&[("provider", provider.as_str()), ("backend", "direct")]),
                    provider,
                )
                .expect_err("no native client for this provider");
            assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
            let message = err.to_string();
            assert!(message.contains(provider.as_str()), "{message}");
            assert!(message.contains("anthropic"), "{message}");
        }
    }

    #[test]
    fn explicit_gateway_url_and_api_key_override_provider_defaults() {
        let adapter = CloudRuntimeAdapter::new();
        let config = adapter
            .build_config(
                &text_with(&[
                    ("provider", "deepseek"),
                    ("backend", "direct"),
                    ("gateway_url", "http://127.0.0.1:9/v1/"),
                    ("api_key", "k"),
                ]),
                IntegrationProvider::DeepSeek,
            )
            .expect("overrides accepted");
        assert_eq!(config.backend, CloudBackend::Gateway);
        assert_eq!(config.gateway_url, "http://127.0.0.1:9/v1");
        assert_eq!(config.api_key.as_deref(), Some("k"));

        // Backend omitted + explicit provider URL behaves the same.
        let config = adapter
            .build_config(
                &text_with(&[
                    ("provider", "deepseek"),
                    ("gateway_url", "https://api.deepseek.com/v1/"),
                    ("api_key", "k"),
                ]),
                IntegrationProvider::DeepSeek,
            )
            .expect("provider url accepted");
        assert_eq!(config.gateway_url, "https://api.deepseek.com/v1");
    }

    #[test]
    fn unknown_backend_and_malformed_urls_are_rejected() {
        let adapter = CloudRuntimeAdapter::new();
        let cases: [(&[(&str, &str)], &str); 3] = [
            (
                &[("provider", "openai"), ("backend", "proxy")],
                "unknown backend",
            ),
            (
                &[("provider", "openai"), ("gateway_url", "ftp://x/v1")],
                "http(s)",
            ),
            (
                &[("provider", "openai"), ("gateway_url", "not a url")],
                "invalid gateway_url",
            ),
        ];
        for (pairs, expected) in cases {
            let err = adapter
                .build_config(&text_with(pairs), IntegrationProvider::OpenAI)
                .expect_err("rejected");
            assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
            assert!(err.to_string().contains(expected), "{err}");
        }
    }

    #[test]
    fn provider_key_is_selected_only_for_that_providers_origin() {
        assert_eq!(
            provider_key_for_destination(
                IntegrationProvider::DeepSeek,
                "https://api.deepseek.com/v1/"
            )
            .as_deref(),
            Some("$DEEPSEEK_API_KEY")
        );
        assert_eq!(
            provider_key_for_destination(IntegrationProvider::OpenAI, "https://api.openai.com/v1")
                .as_deref(),
            Some("$OPENAI_API_KEY")
        );
        // A different provider's origin, lookalikes, other ports and loopback
        // get nothing automatically.
        for destination in [
            "https://api.openai.com/v1",
            "https://api.deepseek.com.evil.example/v1",
            "https://api.deepseek.com:8443/v1",
            "http://api.deepseek.com/v1",
            "http://127.0.0.1:3001/v1",
            "https://api.xybrid.dev/v1",
        ] {
            assert_eq!(
                provider_key_for_destination(IntegrationProvider::DeepSeek, destination),
                None,
                "{destination}"
            );
        }
        assert_eq!(
            provider_for_origin("https://API.DeepSeek.com:443/v1"),
            Some(IntegrationProvider::DeepSeek)
        );
        assert_eq!(provider_for_origin("https://api.xybrid.dev/v1"), None);
    }

    #[test]
    fn provider_origin_without_a_resolvable_key_fails_before_http() {
        let adapter = CloudRuntimeAdapter::new();
        let err = adapter
            .build_config(
                &text_with(&[
                    ("provider", "deepseek"),
                    ("backend", "direct"),
                    ("api_key", SURELY_UNSET_KEY_REF),
                ]),
                IntegrationProvider::DeepSeek,
            )
            .expect_err("no key for the provider origin");
        assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");
        let message = err.to_string();
        assert!(message.contains("DEEPSEEK_API_KEY"), "{message}");
        assert!(message.contains("deepseek"), "{message}");

        // Same check when the provider origin comes from an explicit URL with
        // the backend omitted.
        let err = adapter
            .build_config(
                &text_with(&[
                    ("provider", "openai"),
                    ("gateway_url", "https://api.openai.com/v1"),
                    ("api_key", SURELY_UNSET_KEY_REF),
                ]),
                IntegrationProvider::OpenAI,
            )
            .expect_err("no key for the provider origin");
        assert!(err.to_string().contains("OPENAI_API_KEY"), "{err}");
    }

    #[test]
    fn execute_posts_an_openai_shaped_chat_completion_with_bearer_and_thinking() {
        let (url, rx) = start_json_server(FAKE_DEEPSEEK_BODY, 200);
        let adapter = CloudRuntimeAdapter::new();
        let envelope = text_with(&[
            ("provider", "deepseek"),
            ("model", "deepseek-flash"),
            ("backend", "direct"),
            ("gateway_url", &format!("{url}/v1/")),
            ("api_key", "test-key"),
            ("thinking", "disabled"),
            ("system_prompt", "Be terse."),
            ("temperature", "0"),
            ("max_tokens", "16"),
        ]);

        let output = adapter.execute(&envelope).expect("fake provider answers");

        assert_eq!(output.as_text(), Some("FAKE_DEEPSEEK_REPLY"));
        assert_eq!(
            output.metadata.get("backend").map(String::as_str),
            Some("gateway")
        );
        assert_eq!(
            output.metadata.get("provider").map(String::as_str),
            Some("deepseek")
        );

        let request = rx.recv().expect("request captured");
        assert!(
            request.starts_with("POST /v1/chat/completions "),
            "{request}"
        );
        assert_eq!(
            header_value(&request, "authorization"),
            Some("Bearer test-key")
        );
        let body_start = request.find("\r\n\r\n").unwrap() + 4;
        let body: serde_json::Value = serde_json::from_str(&request[body_start..]).unwrap();
        assert_eq!(body["model"], "deepseek-flash");
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
        assert_eq!(body["max_tokens"], 16);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "Be terse.");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "hello");
        // The batch body never asks for a stream (the key is omitted when false).
        assert_ne!(body.get("stream"), Some(&json!(true)));
    }

    #[test]
    fn anonymous_loopback_gateway_sends_no_authorization_header() {
        let (url, rx) = start_json_server(FAKE_DEEPSEEK_BODY, 200);
        let adapter = CloudRuntimeAdapter::with_gateway(&url);
        let envelope = text_with(&[("provider", "openai"), ("model", "deepseek-flash")]);

        let output = adapter.execute(&envelope).expect("anonymous fake answers");
        assert_eq!(output.as_text(), Some("FAKE_DEEPSEEK_REPLY"));

        let request = rx.recv().expect("request captured");
        assert!(request.starts_with("POST /chat/completions "), "{request}");
        assert_eq!(
            header_value(&request, "authorization"),
            None,
            "a loopback gateway must not inherit any credential: {request}"
        );
    }

    #[test]
    fn execute_surfaces_an_unauthorized_response_as_an_error() {
        let (url, _rx) = start_json_server(r#"{"error":{"message":"bad key"}}"#, 401);
        let adapter = CloudRuntimeAdapter::with_gateway(&url);
        let envelope = text_with(&[
            ("provider", "openai"),
            ("model", "deepseek-flash"),
            ("api_key", "wrong"),
        ]);

        let err = adapter.execute(&envelope).expect_err("401 is an error");
        let message = err.to_string();
        assert!(
            message.contains("401") || message.contains("bad key"),
            "{message}"
        );
        assert!(!message.contains("FAKE_DEEPSEEK_REPLY"));
    }
}
