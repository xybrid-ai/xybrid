//! Model loading FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::sync::Arc;
use std::time::Duration;
use url::Url;
use xybrid_core::device::{ResourceSnapshot, ResourceSnapshotProvider};
use xybrid_core::runtime_adapter::CloudRuntimeAdapter;
use xybrid_sdk::{
    AbortPolicy, AbortSignal, GenerationConfig, ModelLoader, RunOptions, XybridModel,
};

use crate::frb_generated::StreamSink;

use super::context::FfiConversationContext;
use super::device;
use super::result::FfiResult;

/// Generation parameters for LLM inference.
///
/// All fields are optional. When `None`, the model's default value is used.
/// This is non-opaque so FRB auto-generates a plain Dart data class.
pub struct FfiGenerationConfig {
    /// Maximum tokens to generate. Default: 2048
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0.0 = deterministic, higher = more random). Default: 0.7
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling threshold. Default: 0.9
    pub top_p: Option<f32>,
    /// Min-p sampling threshold. Default: 0.05
    pub min_p: Option<f32>,
    /// Top-k sampling (0 = disabled). Default: 40
    pub top_k: Option<u32>,
    /// Repetition penalty (1.0 = disabled). Default: 1.1
    pub repetition_penalty: Option<f32>,
    /// Stop sequences. When `None` or empty, only EOS token stops generation.
    pub stop_sequences: Option<Vec<String>>,
}

impl FfiGenerationConfig {
    /// Create a greedy decoding config (deterministic, temperature=0).
    #[frb(sync)]
    pub fn greedy() -> FfiGenerationConfig {
        FfiGenerationConfig {
            max_tokens: None,
            temperature: Some(0.0),
            top_p: Some(1.0),
            min_p: None,
            top_k: Some(0),
            repetition_penalty: None,
            stop_sequences: None,
        }
    }

    /// Create a creative generation config (higher temperature).
    #[frb(sync)]
    pub fn creative() -> FfiGenerationConfig {
        FfiGenerationConfig {
            max_tokens: None,
            temperature: Some(0.9),
            top_p: Some(0.95),
            min_p: None,
            top_k: Some(50),
            repetition_penalty: None,
            stop_sequences: None,
        }
    }

    pub(crate) fn to_sdk(&self) -> GenerationConfig {
        let mut config = GenerationConfig::default();
        if let Some(v) = self.max_tokens {
            config.max_tokens = v as usize;
        }
        if let Some(v) = self.temperature {
            config.temperature = v;
        }
        if let Some(v) = self.top_p {
            config.top_p = v;
        }
        if let Some(v) = self.min_p {
            config.min_p = v;
        }
        if let Some(v) = self.top_k {
            config.top_k = v as usize;
        }
        if let Some(v) = self.repetition_penalty {
            config.repetition_penalty = v;
        }
        if let Some(ref v) = self.stop_sequences {
            config.stop_sequences = v.clone();
        }
        config
    }
}

/// Cloud fallback controls exposed to Flutter callers.
///
/// This mirrors the customer-facing Dart `RunOptions` wrapper and maps into
/// `xybrid_sdk::RunOptions` at the FFI boundary.
pub struct FfiRunOptions {
    pub cloud_provider: Option<String>,
    pub cloud_model: Option<String>,
    pub cloud_gateway_url: Option<String>,
    pub correlation_id: Option<String>,
    pub abort_on_memory_pressure_critical: bool,
    pub abort_on_thermal_critical: bool,
    pub fallback_to_cloud: bool,
    pub max_grace_tokens: Option<u32>,
}

impl FfiRunOptions {
    fn to_sdk(&self, generation_config: Option<GenerationConfig>) -> RunOptions {
        let mut policy = AbortPolicy::default()
            .with_cloud_fallback(self.fallback_to_cloud)
            .with_max_grace_tokens(self.max_grace_tokens.unwrap_or(0));
        if self.abort_on_memory_pressure_critical {
            policy = policy.stop_on(AbortSignal::MemoryPressureCritical);
        }
        if self.abort_on_thermal_critical {
            policy = policy.stop_on(AbortSignal::ThermalCritical);
        }

        let mut options = RunOptions::new()
            .with_abort_policy(policy)
            .with_resource_provider(Arc::new(FlutterFallbackResourceProvider));

        if let Some(config) = generation_config {
            options = options.with_generation_config(config);
        }

        if let Some(correlation_id) = non_empty(self.correlation_id.as_deref()) {
            options = options.with_correlation_id(correlation_id.to_string());
        }

        options
    }
}

#[derive(Debug)]
struct FlutterFallbackResourceProvider;

impl ResourceSnapshotProvider for FlutterFallbackResourceProvider {
    fn current_snapshot(&self, max_age: Duration) -> ResourceSnapshot {
        device::current_snapshot_with_debug_memory_pressure(max_age)
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.and_then(|v| {
        let trimmed = v.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    })
}

fn apply_cloud_fallback_metadata(
    envelope: &mut xybrid_sdk::ir::Envelope,
    options: &FfiRunOptions,
    config: Option<&FfiGenerationConfig>,
) -> Result<Option<String>, String> {
    let provider = non_empty(options.cloud_provider.as_deref()).unwrap_or("openai");
    let model = non_empty(options.cloud_model.as_deref()).unwrap_or("gpt-4o-mini");
    envelope
        .metadata
        .insert("provider".to_string(), provider.to_string());
    envelope
        .metadata
        .insert("model".to_string(), model.to_string());
    envelope
        .metadata
        .insert("backend".to_string(), "gateway".to_string());

    let gateway_url = options.validated_cloud_gateway_url()?;
    if let Some(gateway_url) = gateway_url.as_deref() {
        envelope
            .metadata
            .insert("gateway_url".to_string(), gateway_url.to_string());
    }

    if let Some(config) = config {
        if let Some(max_tokens) = config.max_tokens {
            envelope
                .metadata
                .insert("max_tokens".to_string(), max_tokens.to_string());
        }
        if let Some(temperature) = config.temperature {
            envelope
                .metadata
                .insert("temperature".to_string(), temperature.to_string());
        }
    }

    Ok(gateway_url)
}

impl FfiRunOptions {
    fn validated_cloud_gateway_url(&self) -> Result<Option<String>, String> {
        non_empty(self.cloud_gateway_url.as_deref())
            .map(validate_cloud_gateway_url)
            .transpose()
    }
}

fn validate_cloud_gateway_url(gateway_url: &str) -> Result<String, String> {
    let parsed = Url::parse(gateway_url)
        .map_err(|e| format!("Invalid cloud gateway URL '{}': {}", gateway_url, e))?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(format!(
                "Invalid cloud gateway URL '{}': unsupported scheme '{}'",
                gateway_url, scheme
            ));
        }
    }

    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err("Invalid cloud gateway URL: credentials are not allowed".to_string());
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(
            "Invalid cloud gateway URL: query strings and fragments are not allowed".to_string(),
        );
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| "Invalid cloud gateway URL: host is required".to_string())?;
    if !is_v1_gateway_base(&parsed) {
        return Err("Invalid cloud gateway URL: base URL must include /v1".to_string());
    }

    if parsed.scheme() == "https" && is_xybrid_gateway_host(host) {
        return Ok(normalize_gateway_url(parsed));
    }

    #[cfg(debug_assertions)]
    {
        if is_debug_gateway_host(host) {
            return Ok(normalize_gateway_url(parsed));
        }
    }

    Err(
        "Invalid cloud gateway URL: release builds only allow HTTPS Xybrid gateway hosts"
            .to_string(),
    )
}

fn normalize_gateway_url(parsed: Url) -> String {
    parsed.as_str().trim_end_matches('/').to_string()
}

fn is_v1_gateway_base(parsed: &Url) -> bool {
    let path = parsed.path().trim_end_matches('/');
    path == "/v1" || path.starts_with("/v1/")
}

fn is_xybrid_gateway_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "xybrid.dev" || host.ends_with(".xybrid.dev")
}

#[cfg(debug_assertions)]
fn is_debug_gateway_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }

    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        Ok(std::net::IpAddr::V6(ip)) => {
            ip.is_loopback() || is_ipv6_link_local(ip) || is_ipv6_unique_local(ip)
        }
        Err(_) => false,
    }
}

#[cfg(debug_assertions)]
fn is_ipv6_link_local(ip: std::net::Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xffc0) == 0xfe80
}

#[cfg(debug_assertions)]
fn is_ipv6_unique_local(ip: std::net::Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xfe00) == 0xfc00
}

/// Event emitted during model loading with progress.
#[derive(Clone)]
pub enum FfiLoadEvent {
    /// Download progress update (0.0 to 1.0)
    Progress(f64),
    /// Model loaded successfully - contains the model handle ID
    Complete,
    /// An error occurred during loading
    Error(String),
}

/// Event emitted during streaming inference.
/// Follows the "everything is a stream" pattern from the SDK.
#[derive(Clone)]
pub enum FfiStreamEvent {
    /// A token was generated
    Token(FfiStreamToken),
    /// Inference completed with final result
    Complete(FfiResult),
    /// An error occurred
    Error(String),
}

/// Token received during streaming inference.
/// Mirrors the SDK's StreamToken structure for FFI.
#[derive(Clone)]
pub struct FfiStreamToken {
    /// The generated token text
    pub token: String,
    /// The token ID (if available)
    pub token_id: Option<i64>,
    /// Index of this token in the sequence
    pub index: u32,
    /// Cumulative text generated so far
    pub cumulative_text: String,
    /// Reason for stopping (if this is the final token)
    pub finish_reason: Option<String>,
}

/// FFI wrapper for ModelLoader (preparatory step before loading).
#[frb(opaque)]
pub struct FfiModelLoader(ModelLoader);

/// FFI wrapper for a loaded XybridModel ready for inference.
#[frb(opaque)]
pub struct FfiModel(Arc<XybridModel>);

impl From<xybrid_sdk::StreamEvent> for FfiStreamEvent {
    fn from(event: xybrid_sdk::StreamEvent) -> Self {
        match event {
            xybrid_sdk::StreamEvent::Token(token) => FfiStreamEvent::Token(FfiStreamToken {
                token: token.token,
                token_id: token.token_id,
                index: token.index as u32,
                cumulative_text: token.cumulative_text,
                finish_reason: token.finish_reason,
            }),
            xybrid_sdk::StreamEvent::Complete(result) => {
                FfiStreamEvent::Complete(FfiResult::from_inference_result(&result))
            }
            xybrid_sdk::StreamEvent::Error(e) => FfiStreamEvent::Error(e),
        }
    }
}

impl FfiModelLoader {
    #[frb(sync)]
    pub fn from_registry(model_id: String) -> FfiModelLoader {
        FfiModelLoader(ModelLoader::from_registry(&model_id))
    }

    #[frb(sync)]
    pub fn from_bundle(path: String) -> Result<FfiModelLoader, String> {
        ModelLoader::from_bundle(&path)
            .map(FfiModelLoader)
            .map_err(|e| e.to_string())
    }

    #[frb(sync)]
    pub fn from_directory(path: String) -> Result<FfiModelLoader, String> {
        ModelLoader::from_directory(&path)
            .map(FfiModelLoader)
            .map_err(|e| e.to_string())
    }

    /// Create a loader for a model from a HuggingFace Hub repository.
    ///
    /// Downloads model files from HuggingFace and caches them locally.
    /// Requires the `huggingface` feature flag to be enabled.
    #[frb(sync)]
    pub fn from_huggingface(repo: String) -> FfiModelLoader {
        FfiModelLoader(ModelLoader::from_huggingface(&repo))
    }

    /// Load the model without progress updates.
    pub async fn load(&self) -> Result<FfiModel, String> {
        self.0
            .load_async()
            .await
            .map(|m| FfiModel(Arc::new(m)))
            .map_err(|e| e.to_string())
    }

    /// Load the model with download progress updates.
    ///
    /// Streams FfiLoadEvent during download:
    /// - `Progress(f64)` for download progress (0.0 to 1.0)
    /// - `Complete` when the model is ready
    /// - `Error(String)` if loading fails
    ///
    /// After receiving `Complete`, call `load()` to get the cached model instantly.
    pub fn load_with_progress(&self, sink: StreamSink<FfiLoadEvent>) {
        let loader = self.0.clone();

        // Run loading in a background thread to not block
        std::thread::spawn(move || {
            let result = loader.load_with_progress(|progress| {
                // Send progress as f64 (0.0 to 1.0)
                let _ = sink.add(FfiLoadEvent::Progress(progress as f64));
            });

            match result {
                Ok(_) => {
                    // Model is now cached, send complete event
                    let _ = sink.add(FfiLoadEvent::Complete);
                }
                Err(e) => {
                    let _ = sink.add(FfiLoadEvent::Error(e.to_string()));
                }
            }
        });
    }
}

impl FfiModel {
    /// Run batch inference (non-streaming).
    ///
    /// Pass an optional `config` to control generation parameters.
    /// When `None`, the model's default parameters are used.
    pub fn run(
        &self,
        envelope: super::envelope::FfiEnvelope,
        config: Option<FfiGenerationConfig>,
    ) -> Result<FfiResult, String> {
        let sdk_config = config.as_ref().map(|c| c.to_sdk());
        let result = self
            .0
            .run(&envelope.into_envelope(), sdk_config.as_ref())
            .map_err(|e| e.to_string())?;
        Ok(FfiResult::from_inference_result(&result))
    }

    /// Run inference with streaming output.
    ///
    /// Returns a stream of events:
    /// - `FfiStreamEvent::Token` for each generated token (LLM models)
    /// - `FfiStreamEvent::Complete` when inference finishes
    /// - `FfiStreamEvent::Error` if an error occurs
    ///
    /// For non-LLM models, a single Token event is emitted with the full result.
    ///
    /// Pass an optional `config` to control generation parameters.
    /// When `None`, the model's default parameters are used.
    pub fn run_stream(
        &self,
        envelope: super::envelope::FfiEnvelope,
        config: Option<FfiGenerationConfig>,
        sink: StreamSink<FfiStreamEvent>,
    ) {
        use tokio_stream::StreamExt;

        let model = self.0.clone();
        let env = envelope.into_envelope();
        let sdk_config = config.map(|c| c.to_sdk());

        // Spawn a background thread with its own Tokio runtime
        // (same pattern as load_with_progress which works)
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = sink.add(FfiStreamEvent::Error(format!(
                        "Failed to create runtime: {}",
                        e
                    )));
                    return;
                }
            };

            rt.block_on(async move {
                let mut stream = model.run_stream(env, sdk_config);

                while let Some(event) = stream.next().await {
                    let ffi_event = FfiStreamEvent::from(event);
                    // Send to Dart stream (ignore errors if sink is closed)
                    if sink.add(ffi_event).is_err() {
                        break;
                    }
                }
            });
        });
    }

    /// Check if this model supports true token-by-token streaming.
    ///
    /// Returns `true` for LLM models (GGUF), `false` for other model types.
    #[frb(sync)]
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    pub fn supports_token_streaming(&self) -> bool {
        self.0.supports_token_streaming()
    }

    /// Run inference with conversation context.
    ///
    /// The context provides conversation history which is formatted into
    /// the prompt using the model's chat template.
    ///
    /// Note: The context is NOT automatically updated with the result.
    /// Call `context.push_text(result.text(), FfiMessageRole::Assistant)` to add the response.
    ///
    /// Pass an optional `config` to control generation parameters.
    /// When `None`, the model's default parameters are used.
    pub fn run_with_context(
        &self,
        envelope: super::envelope::FfiEnvelope,
        context: &FfiConversationContext,
        config: Option<FfiGenerationConfig>,
    ) -> Result<FfiResult, String> {
        let sdk_config = config.as_ref().map(|c| c.to_sdk());
        let ctx_guard = context
            .0
            .read()
            .map_err(|e| format!("Failed to read context: {}", e))?;

        let result = self
            .0
            .run_with_context(&envelope.into_envelope(), &ctx_guard, sdk_config.as_ref())
            .map_err(|e| e.to_string())?;

        Ok(FfiResult::from_inference_result(&result))
    }

    /// Run inference with streaming output and conversation context.
    ///
    /// Combines streaming output with multi-turn conversation memory.
    /// The model sees the full conversation history when generating responses.
    ///
    /// Returns a stream of events:
    /// - `FfiStreamEvent::Token` for each generated token (LLM models)
    /// - `FfiStreamEvent::Complete` when inference finishes
    /// - `FfiStreamEvent::Error` if an error occurs
    ///
    /// Pass an optional `config` to control generation parameters.
    /// When `None`, the model's default parameters are used.
    pub fn run_stream_with_context(
        &self,
        envelope: super::envelope::FfiEnvelope,
        context: &FfiConversationContext,
        config: Option<FfiGenerationConfig>,
        sink: StreamSink<FfiStreamEvent>,
    ) {
        let model = self.0.clone();
        let env = envelope.into_envelope();
        let ctx = context.0.clone();
        let sdk_config = config.map(|c| c.to_sdk());

        // Spawn a background thread
        std::thread::spawn(move || {
            // Get read lock on context
            let ctx_guard = match ctx.read() {
                Ok(guard) => guard,
                Err(e) => {
                    let _ = sink.add(FfiStreamEvent::Error(format!(
                        "Failed to read context: {}",
                        e
                    )));
                    return;
                }
            };

            // Track token index for the stream
            let mut token_index = 0u32;

            // Use callback-based streaming
            let result =
                model.run_streaming_with_context(&env, &ctx_guard, sdk_config.as_ref(), |token| {
                    let ffi_token = FfiStreamToken {
                        token: token.token.clone(),
                        token_id: token.token_id,
                        index: token_index,
                        cumulative_text: token.cumulative_text.clone(),
                        finish_reason: token.finish_reason.clone(),
                    };
                    token_index += 1;
                    let _ = sink.add(FfiStreamEvent::Token(ffi_token));
                    Ok(())
                });

            match result {
                Ok(inference_result) => {
                    let ffi_result = FfiResult::from_inference_result(&inference_result);
                    let _ = sink.add(FfiStreamEvent::Complete(ffi_result));
                }
                Err(e) => {
                    let _ = sink.add(FfiStreamEvent::Error(e.to_string()));
                }
            }
        });
    }

    /// Run streaming inference with local abort and Xybrid cloud fallback.
    ///
    /// The Rust SDK owns the fallback semantics: abort on configured resource
    /// pressure, re-check cloud policy, retry the original prompt through the
    /// authenticated gateway, and emit local/cloud telemetry under one
    /// correlation id.
    pub fn run_stream_with_fallback(
        &self,
        envelope: super::envelope::FfiEnvelope,
        options: FfiRunOptions,
        config: Option<FfiGenerationConfig>,
        sink: StreamSink<FfiStreamEvent>,
    ) {
        let model = self.0.clone();
        let mut env = envelope.into_envelope();
        let gateway_url = match apply_cloud_fallback_metadata(&mut env, &options, config.as_ref()) {
            Ok(gateway_url) => gateway_url,
            Err(e) => {
                let _ = sink.add(FfiStreamEvent::Error(e));
                return;
            }
        };
        let sdk_config = config.map(|c| c.to_sdk());
        let run_options = options.to_sdk(sdk_config);
        let cloud_adapter = match gateway_url.as_deref() {
            Some(gateway_url) => CloudRuntimeAdapter::with_gateway(gateway_url),
            None => CloudRuntimeAdapter::new(),
        };

        std::thread::spawn(move || {
            let mut token_index = 0u32;
            let result = {
                let mut on_token = |token: xybrid_core::runtime_adapter::types::PartialToken| {
                    let ffi_token = FfiStreamToken {
                        token: token.token.clone(),
                        token_id: token.token_id,
                        index: token_index,
                        cumulative_text: token.cumulative_text.clone(),
                        finish_reason: token.finish_reason.clone(),
                    };
                    token_index = token_index.saturating_add(1);
                    let _ = sink.add(FfiStreamEvent::Token(ffi_token));
                    Ok(())
                };
                let mut on_seam = |_seam: xybrid_sdk::model::SeamInfo| {};

                model.run_streaming_with_fallback(
                    &env,
                    &run_options,
                    &cloud_adapter,
                    &mut on_token,
                    &mut on_seam,
                )
            };

            match result {
                Ok(inference_result) => {
                    let ffi_result = FfiResult::from_inference_result(&inference_result);
                    let _ = sink.add(FfiStreamEvent::Complete(ffi_result));
                }
                Err(e) => {
                    let _ = sink.add(FfiStreamEvent::Error(e.to_string()));
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_run_options_maps_abort_policy_to_sdk() {
        let ffi = FfiRunOptions {
            cloud_provider: Some("openai".to_string()),
            cloud_model: Some("gpt-4o-mini".to_string()),
            cloud_gateway_url: Some("http://127.0.0.1:3001/v1".to_string()),
            correlation_id: Some("corr-flutter".to_string()),
            abort_on_memory_pressure_critical: true,
            abort_on_thermal_critical: false,
            fallback_to_cloud: false,
            max_grace_tokens: Some(2),
        };

        let sdk = ffi.to_sdk(None);

        assert!(sdk
            .abort_policy
            .observes(AbortSignal::MemoryPressureCritical));
        assert!(!sdk.abort_policy.observes(AbortSignal::ThermalCritical));
        assert!(!sdk.abort_policy.fallback_to_cloud);
        assert_eq!(sdk.abort_policy.max_grace_tokens, 2);
        assert_eq!(sdk.correlation_id.as_deref(), Some("corr-flutter"));
    }

    #[test]
    fn cloud_gateway_url_accepts_xybrid_https_v1_base() {
        let url = validate_cloud_gateway_url("https://api.xybrid.dev/v1/").unwrap();
        assert_eq!(url, "https://api.xybrid.dev/v1");
    }

    #[test]
    fn cloud_gateway_url_rejects_public_http_hosts() {
        let err = validate_cloud_gateway_url("http://example.com/v1").unwrap_err();
        assert!(err.contains("HTTPS Xybrid gateway hosts"));
    }

    #[test]
    fn cloud_gateway_url_rejects_embedded_credentials() {
        let err = validate_cloud_gateway_url("https://token@api.xybrid.dev/v1").unwrap_err();
        assert!(err.contains("credentials"));
    }

    #[test]
    fn cloud_gateway_url_rejects_missing_v1_base() {
        let err = validate_cloud_gateway_url("https://api.xybrid.dev/").unwrap_err();
        assert!(err.contains("/v1"));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn cloud_gateway_url_accepts_debug_localhost_gateway() {
        let url = validate_cloud_gateway_url("http://127.0.0.1:3001/v1").unwrap();
        assert_eq!(url, "http://127.0.0.1:3001/v1");
    }
}
