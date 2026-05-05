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

use crate::cloud::{Cloud, CloudBackend, CloudConfig, CompletionRequest, CompletionResponse};
use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::IntegrationProvider;
use crate::runtime_adapter::types::{PartialToken, StreamingCallback};
use crate::runtime_adapter::{AdapterError, AdapterResult, RuntimeAdapter};
use crate::tracing as trace;
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
/// > **DEMO-ONLY synthetic streaming.** The default implementation on
/// > [`CloudRuntimeAdapter`] buffers the full `Cloud::complete()` round-trip
/// > and emits chunks afterward. Cloud-leg TTFT = full upstream completion
/// > time + N×25 ms sleep — there is **no real upstream SSE streaming**
/// > here. Behaviour against a slow or degraded gateway: the user sees
/// > nothing for the full round-trip, then a burst of synthetic chunks.
/// > That is acceptable only inside the recorded `cloud_fallback_demo`
/// > example, where the local-leg pre-abort window absorbs the wait.
/// > Real SSE is tracked on the open Linear issue for the streaming cloud
/// > adapter; **do not ship this adapter behind any user-visible
/// > streaming UX** until that work lands. Production code that needs
/// > a streaming-shaped cloud path must call `Cloud::complete()`
/// > explicitly and manage the wait itself.
///
/// Timeouts honor `CloudConfig.timeout_ms` (wired through `Cloud::complete`);
/// errors propagate via [`AdapterError::InferenceFailed`]. The synthetic
/// chunk loop has no separate cancellation hook — once `Cloud::complete()`
/// returns, all chunks fire.
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
    /// **Demo-only synthetic streaming.** See the [`CloudStreaming`] trait
    /// docs for the production-readiness caveat. This implementation is
    /// intentionally simple: one blocking `Cloud::complete()` call followed
    /// by a synchronous chunk loop. Cloud-leg TTFT measured at the synthetic
    /// chunker's first emission is dominated by the upstream round-trip; the
    /// real-gateway TTFT goal (sub-500 ms first byte) is not addressed here.
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
        trace::add_metadata("streaming", "synthetic");

        let config = self.build_config(input);
        let backend_str = match config.backend {
            CloudBackend::Gateway => "gateway",
            CloudBackend::Direct => "direct",
        };
        trace::add_metadata("backend", backend_str);

        let client = Cloud::with_config(config).map_err(|e| {
            AdapterError::RuntimeError(format!("Failed to create cloud client: {}", e))
        })?;

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
            complete_with_cloud_telemetry(&client, request)?
        };

        let finish_reason = response
            .finish_reason
            .clone()
            .unwrap_or_else(|| "stop".to_string());

        synthetic_chunk_emit(&response.text, &finish_reason, &mut on_token, true)?;

        let mut output = Envelope::new(EnvelopeKind::Text(response.text));
        if let Some(backend) = response.backend {
            output.metadata.insert("backend".to_string(), backend);
        }
        output
            .metadata
            .insert("provider".to_string(), provider.as_str().to_string());
        output
            .metadata
            .insert("streaming_mode".to_string(), "synthetic".to_string());

        Ok(output)
    }
}

/// Issue `client.complete(request)`, time the gateway round-trip, and
/// emit `ttft_ms` + (when present) `tokens_in` / `tokens_out` on the
/// currently-active tracing span — typically the `llm_inference` span
/// the caller wraps around the call.
///
/// Centralizes the telemetry contract for both `execute` and
/// `execute_streaming` so the two paths can't drift. Gateway RTT is the
/// honest TTFT for the synthetic-streaming adapter (the synthetic chunker
/// emits the first chunk within ~25 ms of return); real upstream SSE
/// will measure differently when implemented. Token counts come from the
/// upstream `usage` block when populated; absent usage leaves the fields
/// unset rather than writing 0 (which would pollute aggregations).
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

/// Split `text` on whitespace and emit one [`PartialToken`] per chunk
/// through `on_token`. Sleeps 25 ms between non-final chunks when
/// `with_delay` is `true`; tests pass `false` to keep them fast.
/// Empty `text` still emits exactly one terminal token so callers can
/// observe completion. Returns the number of chunks emitted.
fn synthetic_chunk_emit(
    text: &str,
    finish_reason: &str,
    on_token: &mut StreamingCallback<'_>,
    with_delay: bool,
) -> AdapterResult<usize> {
    if text.is_empty() {
        let token = PartialToken {
            token: String::new(),
            token_id: None,
            index: 0,
            cumulative_text: String::new(),
            finish_reason: Some(finish_reason.to_string()),
        };
        on_token(token).map_err(|e| {
            AdapterError::InferenceFailed(format!("streaming callback error: {}", e))
        })?;
        return Ok(1);
    }

    let chunks: Vec<&str> = text.split_inclusive(char::is_whitespace).collect();
    let total = chunks.len();
    let mut cumulative = String::with_capacity(text.len());

    for (idx, chunk) in chunks.into_iter().enumerate() {
        cumulative.push_str(chunk);
        let is_last = idx + 1 == total;
        let token = PartialToken {
            token: chunk.to_string(),
            token_id: None,
            index: idx,
            cumulative_text: cumulative.clone(),
            finish_reason: is_last.then(|| finish_reason.to_string()),
        };
        on_token(token).map_err(|e| {
            AdapterError::InferenceFailed(format!("streaming callback error: {}", e))
        })?;
        if !is_last && with_delay {
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn synthetic_chunk_emit_splits_on_whitespace_and_marks_final() {
        let collected: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_for_cb = collected.clone();
        let mut cb: StreamingCallback<'_> = Box::new(move |t: PartialToken| {
            collected_for_cb.lock().unwrap().push(t);
            Ok(())
        });

        let count = synthetic_chunk_emit("hello world", "stop", &mut cb, false).unwrap();

        assert_eq!(count, 2);
        let tokens = collected.lock().unwrap().clone();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].token, "hello ");
        assert_eq!(tokens[0].index, 0);
        assert_eq!(tokens[0].cumulative_text, "hello ");
        assert_eq!(tokens[0].finish_reason, None);
        assert_eq!(tokens[1].token, "world");
        assert_eq!(tokens[1].index, 1);
        assert_eq!(tokens[1].cumulative_text, "hello world");
        assert_eq!(tokens[1].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn synthetic_chunk_emit_emits_one_terminal_token_for_empty_text() {
        let collected: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_for_cb = collected.clone();
        let mut cb: StreamingCallback<'_> = Box::new(move |t: PartialToken| {
            collected_for_cb.lock().unwrap().push(t);
            Ok(())
        });

        let count = synthetic_chunk_emit("", "stop", &mut cb, false).unwrap();

        assert_eq!(count, 1);
        let tokens = collected.lock().unwrap().clone();
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].token.is_empty());
        assert_eq!(tokens[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn synthetic_chunk_emit_propagates_callback_errors() {
        let mut cb: StreamingCallback<'_> = Box::new(|_| Err("user cancelled".into()));
        let result = synthetic_chunk_emit("hello world", "stop", &mut cb, false);
        match result {
            Err(AdapterError::InferenceFailed(msg)) => {
                assert!(msg.contains("user cancelled"));
            }
            other => panic!("expected InferenceFailed, got {:?}", other),
        }
    }

    #[test]
    fn synthetic_chunk_emit_handles_multi_whitespace_runs() {
        let collected: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_for_cb = collected.clone();
        let mut cb: StreamingCallback<'_> = Box::new(move |t: PartialToken| {
            collected_for_cb.lock().unwrap().push(t);
            Ok(())
        });

        let count = synthetic_chunk_emit("a\nb c", "length", &mut cb, false).unwrap();

        assert_eq!(count, 3);
        let tokens = collected.lock().unwrap().clone();
        // Last token's cumulative_text reconstructs the original input verbatim
        assert_eq!(tokens.last().unwrap().cumulative_text, "a\nb c");
        assert_eq!(
            tokens.last().unwrap().finish_reason.as_deref(),
            Some("length")
        );
    }
}
