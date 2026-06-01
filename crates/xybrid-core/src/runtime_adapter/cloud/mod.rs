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
    parse_gateway_usage, Cloud, CloudBackend, CloudConfig, CompletionRequest, CompletionResponse,
    Role, Usage,
};
use crate::gateway::ChatCompletionChunk;
use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::IntegrationProvider;
use crate::runtime_adapter::types::{PartialToken, StreamingCallback};
use crate::runtime_adapter::{AdapterError, AdapterResult, RuntimeAdapter};
use crate::tracing as trace;
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Read};
use std::time::{Duration, Instant};

const GATEWAY_STREAM_ERROR_FINISH_REASON: &str = "error";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GatewayRoute {
    Chat,
    Embeddings,
    AudioTranscriptions,
    AudioTranslations,
    AudioSpeech,
}

#[derive(Debug, Deserialize)]
struct GatewayEmbeddingsResponse {
    data: Vec<GatewayEmbeddingData>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GatewayEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct GatewayTranscriptionResponse {
    text: String,
}

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
/// | `top_p` | f32 | Nucleus sampling value |
/// | `max_tokens` | u32 | Maximum tokens in response |
/// | `stop` / `stop_sequences` | String or JSON array | Stop sequences |
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

        // Top-p
        if let Some(top_p_str) = envelope.metadata.get("top_p") {
            if let Ok(top_p) = top_p_str.parse::<f32>() {
                request.top_p = Some(top_p);
            }
        }

        // Max tokens
        if let Some(max_str) = envelope.metadata.get("max_tokens") {
            if let Ok(max) = max_str.parse::<u32>() {
                request = request.with_max_tokens(max);
            }
        }

        let mut stop_sequences = Vec::new();
        for key in ["stop", "stop_sequences"] {
            if let Some(raw_stop) = envelope.metadata.get(key) {
                stop_sequences.extend(parse_stop_metadata(raw_stop));
            }
        }
        if !stop_sequences.is_empty() {
            request = request.with_stop(stop_sequences);
        }

        request
    }

    fn gateway_route(&self, input: &Envelope) -> GatewayRoute {
        let output_type = input
            .metadata
            .get("output_type")
            .map(|value| normalize_metadata_value(value));
        let task = input
            .metadata
            .get("task")
            .or_else(|| input.metadata.get("metadata.task"))
            .map(|value| normalize_metadata_value(value));

        if matches!(
            task.as_deref(),
            Some("translate" | "translation" | "audiotranslation" | "speechtranslation")
        ) && matches!(&input.kind, EnvelopeKind::Audio(_))
        {
            return GatewayRoute::AudioTranslations;
        }
        if matches!(&input.kind, EnvelopeKind::Audio(_)) {
            return GatewayRoute::AudioTranscriptions;
        }
        if output_type.as_deref() == Some("embedding")
            || matches!(
                task.as_deref(),
                Some(
                    "embedding"
                        | "embeddings"
                        | "textembedding"
                        | "textembeddings"
                        | "featureextraction"
                )
            )
        {
            return GatewayRoute::Embeddings;
        }
        if output_type.as_deref() == Some("audio")
            || matches!(
                task.as_deref(),
                Some("tts" | "texttospeech" | "speechsynthesis")
            )
        {
            return GatewayRoute::AudioSpeech;
        }
        GatewayRoute::Chat
    }

    fn gateway_model(&self, input: &Envelope, config: &CloudConfig) -> AdapterResult<String> {
        input
            .metadata
            .get("model")
            .cloned()
            .or_else(|| config.default_model.clone())
            .filter(|model| !model.trim().is_empty())
            .ok_or_else(|| {
                AdapterError::InvalidInput(
                    "Cloud gateway request requires 'model' metadata".to_string(),
                )
            })
    }

    fn execute_gateway_route(
        &self,
        input: &Envelope,
        config: &CloudConfig,
        route: GatewayRoute,
    ) -> AdapterResult<Envelope> {
        match route {
            GatewayRoute::Chat => {
                let input_text = text_input(input)?;
                let request = self.build_request(&input_text, input);
                let client = Cloud::with_config(config.clone()).map_err(|e| {
                    AdapterError::RuntimeError(format!("Failed to create cloud client: {}", e))
                })?;
                let response = complete_with_cloud_telemetry(&client, request)?;
                let mut output = Envelope::new(EnvelopeKind::Text(response.text));
                if let Some(backend) = response.backend {
                    output.metadata.insert("backend".to_string(), backend);
                }
                Ok(output)
            }
            GatewayRoute::Embeddings => self.execute_gateway_embeddings(input, config),
            GatewayRoute::AudioTranscriptions => self.execute_gateway_transcription(input, config),
            GatewayRoute::AudioTranslations => self.execute_gateway_translation(input, config),
            GatewayRoute::AudioSpeech => self.execute_gateway_speech(input, config),
        }
    }

    fn execute_gateway_embeddings(
        &self,
        input: &Envelope,
        config: &CloudConfig,
    ) -> AdapterResult<Envelope> {
        ensure_gateway_backend(config, "embeddings")?;
        let input_text = text_input(input)?;
        let model = self.gateway_model(input, config)?;
        let mut body = json!({
            "model": model,
            "input": [input_text],
            "encoding_format": "float",
        });
        if let Some(dimensions) = input.metadata.get("dimensions") {
            if let Ok(dimensions) = dimensions.parse::<u32>() {
                body["dimensions"] = json!(dimensions);
            }
        }
        let response: GatewayEmbeddingsResponse =
            gateway_json_request(config, "/embeddings", body)?;
        let embedding = response
            .data
            .into_iter()
            .next()
            .map(|data| data.embedding)
            .ok_or_else(|| {
                AdapterError::InferenceFailed("Gateway embeddings response was empty".to_string())
            })?;
        let mut output = Envelope::new(EnvelopeKind::Embedding(embedding));
        output
            .metadata
            .insert("backend".to_string(), "gateway".to_string());
        if let Some(model) = response.model {
            output.metadata.insert("model".to_string(), model);
        }
        Ok(output)
    }

    fn execute_gateway_transcription(
        &self,
        input: &Envelope,
        config: &CloudConfig,
    ) -> AdapterResult<Envelope> {
        self.execute_gateway_audio_text(
            input,
            config,
            "/audio/transcriptions",
            true,
            "audio transcription",
        )
    }

    fn execute_gateway_translation(
        &self,
        input: &Envelope,
        config: &CloudConfig,
    ) -> AdapterResult<Envelope> {
        self.execute_gateway_audio_text(
            input,
            config,
            "/audio/translations",
            false,
            "audio translation",
        )
    }

    fn execute_gateway_audio_text(
        &self,
        input: &Envelope,
        config: &CloudConfig,
        path: &str,
        include_language: bool,
        operation: &str,
    ) -> AdapterResult<Envelope> {
        ensure_gateway_backend(config, operation)?;
        let audio = match &input.kind {
            EnvelopeKind::Audio(bytes) => bytes.as_slice(),
            other => {
                return Err(AdapterError::InvalidInput(format!(
                    "Cloud transcription expects Audio input, got: {:?}",
                    other
                )));
            }
        };
        let model = self.gateway_model(input, config)?;
        let filename = input
            .metadata
            .get("filename")
            .map(String::as_str)
            .unwrap_or("input.wav");
        let content_type = input
            .metadata
            .get("content_type")
            .or_else(|| input.metadata.get("mime_type"))
            .map(String::as_str)
            .unwrap_or_else(|| audio_content_type(input));
        let mut fields = vec![
            ("model".to_string(), model),
            ("response_format".to_string(), "json".to_string()),
        ];
        let forwarded = if include_language {
            ["language", "prompt", "temperature"].as_slice()
        } else {
            ["prompt", "temperature"].as_slice()
        };
        for &key in forwarded {
            if let Some(value) = input.metadata.get(key) {
                fields.push((key.to_string(), value.clone()));
            }
        }
        let response: GatewayTranscriptionResponse =
            gateway_multipart_request(config, path, fields, "file", filename, content_type, audio)?;
        let mut output = Envelope::new(EnvelopeKind::Text(response.text));
        output
            .metadata
            .insert("backend".to_string(), "gateway".to_string());
        Ok(output)
    }

    fn execute_gateway_speech(
        &self,
        input: &Envelope,
        config: &CloudConfig,
    ) -> AdapterResult<Envelope> {
        ensure_gateway_backend(config, "audio speech")?;
        let input_text = text_input(input)?;
        let model = self.gateway_model(input, config)?;
        let voice = input
            .metadata
            .get("voice_id")
            .or_else(|| input.metadata.get("voice"))
            .cloned()
            .unwrap_or_else(|| "default".to_string());
        let response_format = input
            .metadata
            .get("response_format")
            .cloned()
            .unwrap_or_else(|| "wav".to_string());
        let mut body = json!({
            "model": model,
            "input": input_text,
            "voice": voice,
            "response_format": response_format,
        });
        if let Some(speed) = input.metadata.get("speed") {
            if let Ok(speed) = speed.parse::<f32>() {
                body["speed"] = json!(speed);
            }
        }
        let bytes = gateway_binary_request(config, "/audio/speech", body)?;
        let mut output = Envelope::new(EnvelopeKind::Audio(bytes));
        output
            .metadata
            .insert("backend".to_string(), "gateway".to_string());
        output
            .metadata
            .insert("format".to_string(), response_format);
        Ok(output)
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
        let route = self.gateway_route(input);
        trace::add_metadata("gateway_route", format!("{route:?}"));

        let mut output = {
            let _llm_span = trace::SpanGuard::new("llm_inference");
            self.execute_gateway_route(input, &config, route)?
        };

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
        let route = self.gateway_route(input);
        trace::add_metadata("gateway_route", format!("{route:?}"));

        if route != GatewayRoute::Chat {
            let mut output = {
                let _llm_span = trace::SpanGuard::new("gateway_http_inference");
                self.execute_gateway_route(input, &config, route)?
            };
            output
                .metadata
                .insert("provider".to_string(), provider.as_str().to_string());
            output
                .metadata
                .insert("streaming_mode".to_string(), "gateway_http".to_string());
            return Ok(output);
        }

        let input_text = text_input(input)?;
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
            if choice_finish.as_deref() == Some(GATEWAY_STREAM_ERROR_FINISH_REASON) {
                continue;
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

    if finish_reason.as_deref() == Some(GATEWAY_STREAM_ERROR_FINISH_REASON) {
        return Err(AdapterError::InferenceFailed(
            "Gateway stream finished with error".to_string(),
        ));
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

fn gateway_json_request<T: for<'de> Deserialize<'de>>(
    config: &CloudConfig,
    path: &str,
    body: serde_json::Value,
) -> AdapterResult<T> {
    let response = gateway_request(config, path)
        .set("Accept", "application/json")
        .set("Content-Type", "application/json")
        .send_json(&body)
        .map_err(|error| gateway_http_error(error, config.timeout_ms, path))?;

    response
        .into_json()
        .map_err(|error| AdapterError::SerializationError(error.to_string()))
}

fn gateway_binary_request(
    config: &CloudConfig,
    path: &str,
    body: serde_json::Value,
) -> AdapterResult<Vec<u8>> {
    let response = gateway_request(config, path)
        .set("Accept", "application/octet-stream")
        .set("Content-Type", "application/json")
        .send_json(&body)
        .map_err(|error| gateway_http_error(error, config.timeout_ms, path))?;

    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(AdapterError::IOError)?;
    Ok(bytes)
}

fn gateway_multipart_request<T: for<'de> Deserialize<'de>>(
    config: &CloudConfig,
    path: &str,
    fields: Vec<(String, String)>,
    file_field: &str,
    filename: &str,
    content_type: &str,
    file_bytes: &[u8],
) -> AdapterResult<T> {
    let boundary = "xybrid-cloud-boundary";
    let body = multipart_body(
        boundary,
        fields,
        file_field,
        filename,
        content_type,
        file_bytes,
    );
    let response = gateway_request(config, path)
        .set("Accept", "application/json")
        .set(
            "Content-Type",
            &format!("multipart/form-data; boundary={boundary}"),
        )
        .send_bytes(&body)
        .map_err(|error| gateway_http_error(error, config.timeout_ms, path))?;

    response
        .into_json()
        .map_err(|error| AdapterError::SerializationError(error.to_string()))
}

fn gateway_request(config: &CloudConfig, path: &str) -> ureq::Request {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_millis(10_000))
        .timeout(Duration::from_millis(config.timeout_ms as u64))
        .build();
    let url = format!("{}{}", config.gateway_url, path);
    if config.debug {
        eprintln!("[Cloud] Gateway request to: {}", url);
    }
    let mut request = agent.post(&url);
    if let Some(key) = config.resolve_api_key() {
        request = request.set("Authorization", &format!("Bearer {}", key));
    }
    request
}

fn gateway_http_error(error: ureq::Error, timeout_ms: u32, path: &str) -> AdapterError {
    match error {
        ureq::Error::Status(status, resp) => {
            let error_body: Result<serde_json::Value, _> = resp.into_json();
            let message = error_body
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| "Unknown error".to_string());
            AdapterError::InferenceFailed(format!("Gateway {path} returned {status}: {message}"))
        }
        ureq::Error::Transport(transport) => {
            let msg = transport.to_string();
            if msg.contains("timed out") || msg.contains("timeout") {
                AdapterError::InferenceFailed(format!(
                    "Gateway request timed out after {timeout_ms} ms"
                ))
            } else {
                AdapterError::InferenceFailed(format!("Gateway {path} failed: {msg}"))
            }
        }
    }
}

fn multipart_body(
    boundary: &str,
    fields: Vec<(String, String)>,
    file_field: &str,
    filename: &str,
    content_type: &str,
    file_bytes: &[u8],
) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{}\"\r\n\r\n", name).as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n",
            file_field, filename
        )
        .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(file_bytes);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

fn text_input(input: &Envelope) -> AdapterResult<String> {
    match &input.kind {
        EnvelopeKind::Text(text) => Ok(text.clone()),
        other => Err(AdapterError::InvalidInput(format!(
            "Cloud adapter expects Text input, got: {:?}",
            other
        ))),
    }
}

fn ensure_gateway_backend(config: &CloudConfig, operation: &str) -> AdapterResult<()> {
    if matches!(config.backend, CloudBackend::Gateway) {
        Ok(())
    } else {
        Err(AdapterError::RuntimeError(format!(
            "Cloud {operation} is only supported through the gateway backend"
        )))
    }
}

fn normalize_metadata_value(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn audio_content_type(input: &Envelope) -> &'static str {
    match input
        .metadata
        .get("format")
        .map(|format| normalize_metadata_value(format))
        .as_deref()
    {
        Some("mp3") => "audio/mpeg",
        Some("m4a") => "audio/mp4",
        Some("ogg") => "audio/ogg",
        Some("webm") => "audio/webm",
        _ => "audio/wav",
    }
}

fn stream_usage_from_json(data: &str) -> Option<Usage> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    value.get("usage").map(parse_gateway_usage)
}

fn parse_stop_metadata(raw: &str) -> Vec<String> {
    let value = serde_json::from_str::<serde_json::Value>(raw).ok();
    match value {
        Some(serde_json::Value::Array(values)) => values
            .into_iter()
            .filter_map(|value| match value {
                serde_json::Value::String(stop) => Some(stop),
                _ => None,
            })
            .map(|stop| stop.trim().to_string())
            .filter(|stop| !stop.is_empty())
            .collect(),
        Some(serde_json::Value::String(stop)) => {
            let stop = stop.trim();
            if stop.is_empty() {
                Vec::new()
            } else {
                vec![stop.to_string()]
            }
        }
        _ => raw
            .split(',')
            .map(str::trim)
            .filter(|stop| !stop.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
    }
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
    fn execute_routes_embedding_to_gateway_embeddings() {
        let body = r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],"model":"embed-test"}"#;
        let (gateway_url, request_rx) =
            start_http_server(body.as_bytes().to_vec(), 200, "application/json");
        let adapter = CloudRuntimeAdapter::with_gateway(&gateway_url);
        let mut input = Envelope::new(EnvelopeKind::Text("embed me".to_string()));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "embed-test".to_string());
        input
            .metadata
            .insert("output_type".to_string(), "embedding".to_string());

        let output = adapter.execute(&input).unwrap();
        let request = request_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        assert!(request.starts_with("POST /embeddings "));
        assert!(request.contains("\"model\":\"embed-test\""));
        assert!(request.contains("\"input\":[\"embed me\"]"));
        assert_eq!(output.kind, EnvelopeKind::Embedding(vec![0.1, 0.2, 0.3]));
        assert_eq!(output.metadata["backend"], "gateway");
        assert_eq!(output.metadata["provider"], "openai");
    }

    #[test]
    fn execute_routes_audio_to_gateway_transcriptions() {
        let (gateway_url, request_rx) = start_http_server(
            br#"{"text":"transcribed"}"#.to_vec(),
            200,
            "application/json",
        );
        let adapter = CloudRuntimeAdapter::with_gateway(&gateway_url);
        let mut input = Envelope::new(EnvelopeKind::Audio(vec![1, 2, 3, 4]));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "whisper-test".to_string());
        input
            .metadata
            .insert("filename".to_string(), "sample.wav".to_string());
        input
            .metadata
            .insert("language".to_string(), "en".to_string());

        let output = adapter.execute(&input).unwrap();
        let request = request_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        assert!(request.starts_with("POST /audio/transcriptions "));
        assert!(request.contains("multipart/form-data"));
        assert!(request.contains("name=\"model\""));
        assert!(request.contains("whisper-test"));
        assert!(request.contains("name=\"file\"; filename=\"sample.wav\""));
        assert_eq!(output.kind, EnvelopeKind::Text("transcribed".to_string()));
        assert_eq!(output.metadata["backend"], "gateway");
    }

    #[test]
    fn execute_routes_audio_translation_to_gateway_translations() {
        let (gateway_url, request_rx) = start_http_server(
            br#"{"text":"translated"}"#.to_vec(),
            200,
            "application/json",
        );
        let adapter = CloudRuntimeAdapter::with_gateway(&gateway_url);
        let mut input = Envelope::new(EnvelopeKind::Audio(vec![1, 2, 3, 4]));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "whisper-test".to_string());
        input
            .metadata
            .insert("filename".to_string(), "sample.wav".to_string());
        input
            .metadata
            .insert("task".to_string(), "translate".to_string());
        input
            .metadata
            .insert("language".to_string(), "en".to_string());
        input
            .metadata
            .insert("prompt".to_string(), "preserve names".to_string());

        let output = adapter.execute(&input).unwrap();
        let request = request_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        assert!(request.starts_with("POST /audio/translations "));
        assert!(request.contains("multipart/form-data"));
        assert!(request.contains("name=\"model\""));
        assert!(request.contains("whisper-test"));
        assert!(request.contains("name=\"prompt\""));
        assert!(request.contains("preserve names"));
        assert!(!request.contains("name=\"language\""));
        assert_eq!(output.kind, EnvelopeKind::Text("translated".to_string()));
        assert_eq!(output.metadata["backend"], "gateway");
    }

    #[test]
    fn execute_routes_tts_to_gateway_speech() {
        let speech = b"RIFFtestwave".to_vec();
        let (gateway_url, request_rx) =
            start_http_server(speech.clone(), 200, "application/octet-stream");
        let adapter = CloudRuntimeAdapter::with_gateway(&gateway_url);
        let mut input = Envelope::new(EnvelopeKind::Text("say this".to_string()));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "tts-test".to_string());
        input
            .metadata
            .insert("output_type".to_string(), "audio".to_string());
        input
            .metadata
            .insert("voice".to_string(), "alloy".to_string());

        let output = adapter.execute(&input).unwrap();
        let request = request_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        assert!(request.starts_with("POST /audio/speech "));
        assert!(request.contains("\"model\":\"tts-test\""));
        assert!(request.contains("\"input\":\"say this\""));
        assert!(request.contains("\"voice\":\"alloy\""));
        assert_eq!(output.kind, EnvelopeKind::Audio(speech));
        assert_eq!(output.metadata["backend"], "gateway");
        assert_eq!(output.metadata["format"], "wav");
    }

    #[test]
    fn execute_streaming_non_chat_uses_single_gateway_http_result() {
        let body = r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.4]}],"model":"embed-test"}"#;
        let (gateway_url, request_rx) =
            start_http_server(body.as_bytes().to_vec(), 200, "application/json");
        let adapter = CloudRuntimeAdapter::with_gateway(&gateway_url);
        let mut input = Envelope::new(EnvelopeKind::Text("embed me".to_string()));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "embed-test".to_string());
        input
            .metadata
            .insert("output_type".to_string(), "embedding".to_string());

        let mut tokens = Vec::new();
        let output = adapter
            .execute_streaming(
                &input,
                Box::new(|token| {
                    tokens.push(token);
                    Ok(())
                }),
            )
            .unwrap();
        let request = request_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        assert!(request.starts_with("POST /embeddings "));
        assert_eq!(tokens.len(), 0);
        assert_eq!(output.kind, EnvelopeKind::Embedding(vec![0.4]));
        assert_eq!(output.metadata["streaming_mode"], "gateway_http");
    }

    #[test]
    fn gateway_chat_body_forces_stream_true() {
        let config = CloudConfig::gateway().with_default_model("default-model");
        let mut request = CompletionRequest::new("hello")
            .with_model("explicit-model")
            .with_temperature(0.2)
            .with_max_tokens(42);
        request.top_p = Some(0.9);
        request.stop = Some(vec!["STOP".to_string(), "END".to_string()]);

        let body = gateway_chat_body(&request, &config, true);

        assert_eq!(body["stream"], true);
        assert_eq!(body["model"], "explicit-model");
        assert!((body["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert!((body["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(body["stop"], serde_json::json!(["STOP", "END"]));
        assert_eq!(body["max_tokens"], 42);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn build_request_forwards_sampling_metadata() {
        let adapter = CloudRuntimeAdapter::new();
        let mut input = Envelope::new(EnvelopeKind::Text("prompt".to_string()));
        input
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        input
            .metadata
            .insert("model".to_string(), "gpt-test".to_string());
        input
            .metadata
            .insert("top_p".to_string(), "0.73".to_string());
        input.metadata.insert(
            "stop_sequences".to_string(),
            "[\"STOP\", \"END\"]".to_string(),
        );

        let request = adapter.build_request("prompt", &input);

        assert_eq!(request.model.as_deref(), Some("gpt-test"));
        assert_eq!(request.top_p, Some(0.73));
        assert_eq!(
            request.stop,
            Some(vec!["STOP".to_string(), "END".to_string()])
        );
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
    fn execute_streaming_returns_error_on_gateway_error_finish() {
        let sse = concat!(
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\"}]}\n\n",
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

        let result = adapter.execute_streaming(&input, cb);

        match result {
            Err(AdapterError::InferenceFailed(msg)) => {
                assert!(msg.contains("Gateway stream finished with error"));
            }
            other => panic!("expected gateway error finish to fail, got {:?}", other),
        }
        let tokens = collected.lock().unwrap().clone();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token, "partial");
        assert_eq!(tokens[0].finish_reason, None);
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
        start_http_server(body.as_bytes().to_vec(), status, "text/event-stream")
    }

    fn start_http_server(
        body: Vec<u8>,
        status: u16,
        content_type: &'static str,
    ) -> (String, mpsc::Receiver<String>) {
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
                "HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
        });

        (format!("http://{}", addr), rx)
    }
}
