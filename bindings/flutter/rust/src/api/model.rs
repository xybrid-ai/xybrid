//! Model loading FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::sync::Arc;
use std::time::Duration;
use url::Url;
use xybrid_core::device::{ResourceSnapshot, ResourceSnapshotProvider};
use xybrid_core::runtime_adapter::CloudRuntimeAdapter;
use xybrid_ffi_facade as facade;
use xybrid_sdk::{
    AbortPolicy, AbortSignal, CancellationToken, GenerationConfig, ModelLoader, RunOptions,
    XybridModel,
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

    /// Re-shape into the facade POD. The facade owns the single canonical
    /// mapping into [`xybrid_sdk::GenerationConfig`] (`Option` overrides →
    /// SDK defaults); calling [`to_sdk`](Self::to_sdk) delegates through
    /// it instead of duplicating the 20-line `if let Some(...)` chain we
    /// used to maintain here.
    pub(crate) fn to_facade(&self) -> facade::GenerationConfig {
        facade::GenerationConfig {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            min_p: self.min_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            stop_sequences: self.stop_sequences.clone().unwrap_or_default(),
        }
    }

    pub(crate) fn to_sdk(&self) -> GenerationConfig {
        self.to_facade().to_sdk()
    }
}

/// Opaque cooperative cancellation handle shared across the FFI boundary.
///
/// Wraps the SDK [`CancellationToken`] (an `Arc<AtomicBool>` inside, hence
/// `Clone + Send + Sync`). A Dart caller constructs one, passes it into a
/// streaming run via the run options, and calls [`FfiCancellationToken::cancel`]
/// to halt Rust generation at the next token boundary — releasing the model
/// write lock promptly.
///
/// The handle is cheap to clone; cloning shares the same underlying flag, so
/// the copy held by the spawned streaming thread observes a `cancel()` issued
/// from Dart.
#[frb(opaque)]
pub struct FfiCancellationToken(pub(crate) CancellationToken);

impl FfiCancellationToken {
    /// Create a fresh, un-cancelled token.
    #[frb(sync)]
    pub fn new() -> FfiCancellationToken {
        FfiCancellationToken(CancellationToken::new())
    }

    /// Request cooperative cancellation of the associated run.
    ///
    /// Takes effect at the next token boundary (cancellation is cooperative,
    /// not preemptive — it never interrupts mid-token).
    #[frb(sync)]
    pub fn cancel(&self) {
        self.0.cancel();
    }

    /// Whether cancellation has been requested on this token.
    #[frb(sync)]
    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

impl Default for FfiCancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

// Allow cloning the handle (shares the same underlying cancellation flag).
impl Clone for FfiCancellationToken {
    fn clone(&self) -> Self {
        FfiCancellationToken(self.0.clone())
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
    /// Caller-supplied UUID identifying one continuous live-capture session
    /// (e.g. the Flutter vision-live loop). When present, the run is tagged via
    /// `RunOptions::with_frame_session`, which flips `live_mode = true` and
    /// makes the SDK rate-limit live telemetry to ~1 wire row/sec per session.
    /// `None` for one-shot / chat runs (telemetry path unchanged).
    pub frame_session_id: Option<String>,
}

impl FfiRunOptions {
    /// Re-shape into the facade POD. Drops cloud_provider/cloud_model/
    /// cloud_gateway_url (those ride on the envelope metadata via
    /// [`apply_cloud_fallback_metadata`], not on `RunOptions`).
    fn to_facade(&self, generation_config: Option<facade::GenerationConfig>) -> facade::RunOptions {
        let mut abort_on = Vec::new();
        if self.abort_on_memory_pressure_critical {
            abort_on.push(facade::AbortSignal::MemoryPressureCritical);
        }
        if self.abort_on_thermal_critical {
            abort_on.push(facade::AbortSignal::ThermalCritical);
        }
        facade::RunOptions {
            generation_config,
            abort_on,
            fallback_to_cloud: self.fallback_to_cloud,
            max_grace_tokens: self.max_grace_tokens.unwrap_or(0),
            correlation_id: non_empty(self.correlation_id.as_deref()).map(str::to_string),
        }
    }

    /// Build SDK [`RunOptions`], optionally wiring a cancellation token.
    ///
    /// The facade owns the base `AbortPolicy` -> SDK assembly (memory/thermal
    /// signals, cloud fallback, grace tokens, correlation id). On top of the
    /// canonical SDK options it returns, we layer the bits the facade keeps out
    /// of its FFI-safe surface: the Flutter resource provider, the cancellation
    /// token (plus `UserCancelled`, which the facade's `AbortSignal` omits by
    /// design because cancellation is expressed through the token), and the
    /// vision live-mode frame session.
    ///
    /// When `cancellation_token` is `None` the abort policy is left untouched,
    /// preserving the default chat / cloud-fallback semantics exactly.
    fn to_sdk_with_cancellation(
        &self,
        generation_config: Option<GenerationConfig>,
        cancellation_token: Option<&FfiCancellationToken>,
    ) -> RunOptions {
        let facade_gc = generation_config
            .as_ref()
            .map(|cfg| facade::GenerationConfig {
                max_tokens: Some(cfg.max_tokens as u32),
                temperature: Some(cfg.temperature),
                top_p: Some(cfg.top_p),
                min_p: Some(cfg.min_p),
                top_k: Some(cfg.top_k as u32),
                repetition_penalty: Some(cfg.repetition_penalty),
                stop_sequences: cfg.stop_sequences.clone(),
            });
        let mut options = self.to_facade(facade_gc).to_sdk(None);

        // Flutter-specific resource provider; the facade omits this field so it
        // stays FFI-safe (the trait object isn't portable across generators).
        options = options.with_resource_provider(Arc::new(FlutterFallbackResourceProvider));

        // Observe UserCancelled and attach the token only when one is supplied,
        // so callers that pass no token keep the default abort semantics.
        if let Some(token) = cancellation_token {
            let policy = options
                .abort_policy
                .clone()
                .stop_on(AbortSignal::UserCancelled);
            options = options
                .with_abort_policy(policy)
                .with_cancellation_token(token.0.clone());
        }

        // Vision live-mode: flips live_mode and rate-limits live telemetry.
        if let Some(frame_session_id) = non_empty(self.frame_session_id.as_deref()) {
            options = options.with_frame_session(frame_session_id.to_string());
        }

        options
    }
}

/// Tracks whether a streaming run has reached its terminal state (final token
/// emitted or natural completion). A sink-close (Dart unsubscribe) is only
/// treated as a user cancellation while the run is still mid-stream — a close
/// that races in after the last token must NOT trigger a false cancel.
fn should_cancel_on_sink_close(reached_terminal: bool) -> bool {
    !reached_terminal
}

/// Build minimal SDK [`RunOptions`] for the plain (non-cloud-fallback)
/// streaming FFI paths (`run_stream` / `run_stream_with_context`).
///
/// When `cancellation_token` is `Some`, the abort policy observes
/// [`AbortSignal::UserCancelled`] and the token is attached so the per-token
/// abort check halts generation on cancel. When `None`, the returned options
/// carry only the optional generation config and an empty abort policy — which
/// makes `run_streaming_with_options` behave exactly like the pre-cancellation
/// `run_streaming` path (no abort observation, no resource checks).
fn streaming_run_options(
    generation_config: Option<GenerationConfig>,
    cancellation_token: Option<&FfiCancellationToken>,
    frame_session_id: Option<&str>,
) -> RunOptions {
    let mut options = RunOptions::new();
    if let Some(config) = generation_config {
        options = options.with_generation_config(config);
    }
    if let Some(token) = cancellation_token {
        options = options
            .with_abort_policy(AbortPolicy::default().stop_on(AbortSignal::UserCancelled))
            .with_cancellation_token(token.0.clone());
    }
    // Tag the run as part of a live-capture session when the caller supplies a
    // frame session id (the Flutter vision-live loop). Absent / empty → plain
    // per-run telemetry, byte-for-byte the pre-live-mode path.
    if let Some(frame_session_id) = non_empty(frame_session_id) {
        options = options.with_frame_session(frame_session_id.to_string());
    }
    options
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

/// Event emitted during streaming TTS. Audio rides the stream as raw 16-bit LE
/// PCM chunks (one per sentence-chunk) plus the sample rate, so the Dart side
/// wraps each chunk into a correctly-headed WAV as it arrives.
#[derive(Clone)]
pub enum FfiTtsStreamEvent {
    /// One synthesized chunk: raw 16-bit LE PCM and its sample rate (Hz).
    AudioChunk { pcm: Vec<u8>, sample_rate: u32 },
    /// Synthesis completed.
    Complete,
    /// An error occurred during synthesis.
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
    ///
    /// Pass an optional `cancellation_token` to make the run cancellable: when
    /// the token is cancelled (or the Dart sink is closed mid-stream), Rust
    /// generation halts at the next token boundary and releases the model write
    /// lock. When `None`, behavior matches the pre-cancellation streaming path
    /// (no `UserCancelled` observation).
    ///
    /// Pass `preempt = true` (latest-frame-wins) **together with** a
    /// `cancellation_token` to make this run cancel the model's previously
    /// in-flight streaming run *before* it acquires the model write lock — so a
    /// new frame's stream does not head-of-line block behind a still-running
    /// one. The displaced run halts at its next token and releases the lock.
    /// `preempt` defaults to `false`: chat and any caller that wants
    /// drop-if-busy / serialized semantics passes `false` (or omits it) and the
    /// behavior is byte-for-byte the pre-preempt path. Preempt with no token is
    /// a no-op (there is nothing to register/cancel).
    ///
    /// Pass an optional `frame_session_id` (a caller-supplied UUID) to tag every
    /// run in a continuous live-capture session. The SDK then rate-limits the
    /// session's telemetry to ~1 wire row/sec instead of one row per frame.
    /// `None` (chat and one-shot runs) leaves telemetry as plain per-run rows.
    pub fn run_stream(
        &self,
        envelope: super::envelope::FfiEnvelope,
        config: Option<FfiGenerationConfig>,
        cancellation_token: Option<FfiCancellationToken>,
        preempt: bool,
        frame_session_id: Option<String>,
        sink: StreamSink<FfiStreamEvent>,
    ) {
        let model = self.0.clone();
        let env = envelope.into_envelope();
        let sdk_config = config.map(|c| c.to_sdk());

        // Build per-run options carrying the cancellation token (when present).
        // The token is also kept as `cancel_handle` so a closed/unsubscribed
        // sink can drive the same cancellation flag the abort check observes.
        // A non-empty `frame_session_id` tags the run as live-capture so the SDK
        // rate-limits its telemetry per session.
        let run_options = streaming_run_options(
            sdk_config,
            cancellation_token.as_ref(),
            frame_session_id.as_deref(),
        );
        let cancel_handle = cancellation_token;

        std::thread::spawn(move || {
            let reached_terminal = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mut token_index = 0u32;
            let result = {
                let reached_terminal = reached_terminal.clone();
                let cancel_handle = cancel_handle.clone();
                // Clone the sink for the callback so the original remains
                // available for the terminal Complete/Error emit below.
                let token_sink = sink.clone();
                let on_token = move |token: xybrid_core::runtime_adapter::types::PartialToken| {
                    let is_final = token.finish_reason.is_some();
                    let ffi_token = FfiStreamToken {
                        token: token.token.clone(),
                        token_id: token.token_id,
                        index: token_index,
                        cumulative_text: token.cumulative_text.clone(),
                        finish_reason: token.finish_reason.clone(),
                    };
                    token_index = token_index.saturating_add(1);
                    // Mark terminal *before* the final emit so a sink-close that
                    // races the last token does not look like a mid-stream cancel.
                    if is_final {
                        reached_terminal.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    if token_sink.add(FfiStreamEvent::Token(ffi_token)).is_err()
                        && should_cancel_on_sink_close(
                            reached_terminal.load(std::sync::atomic::Ordering::SeqCst),
                        )
                    {
                        if let Some(handle) = cancel_handle.as_ref() {
                            handle.0.cancel();
                        }
                    }
                    Ok(())
                };
                model.run_streaming_with_options_preempt(&env, &run_options, preempt, on_token)
            };

            match result {
                Ok(inference_result) => {
                    reached_terminal.store(true, std::sync::atomic::Ordering::SeqCst);
                    let ffi_result = FfiResult::from_inference_result(&inference_result);
                    let _ = sink.add(FfiStreamEvent::Complete(ffi_result));
                }
                Err(e) => {
                    let _ = sink.add(FfiStreamEvent::Error(e.to_string()));
                }
            }
        });
    }

    /// Streaming TTS: synthesize the envelope's text sentence-chunk by
    /// sentence-chunk and emit each chunk's PCM through `sink` as it is produced
    /// (instead of one batched WAV), so playback can start after the first
    /// sentence. Runs on a worker thread.
    ///
    /// Cancellation mirrors [`run_stream`]: an optional `cancellation_token`
    /// stops synthesis at the next chunk boundary, and a closed/unsubscribed
    /// `sink` (Dart cancelled the stream — i.e. barge-in) drives the same
    /// cancel via the `should_cancel_on_sink_close` handshake.
    pub fn run_tts_stream(
        &self,
        envelope: super::envelope::FfiEnvelope,
        config: Option<FfiGenerationConfig>,
        cancellation_token: Option<FfiCancellationToken>,
        sink: StreamSink<FfiTtsStreamEvent>,
    ) {
        let model = self.0.clone();
        let env = envelope.into_envelope();
        let sdk_config = config.map(|c| c.to_sdk());
        let run_options = streaming_run_options(sdk_config, cancellation_token.as_ref(), None);
        let cancel_handle = cancellation_token;

        std::thread::spawn(move || {
            let reached_terminal = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let result = {
                let reached_terminal = reached_terminal.clone();
                let cancel_handle = cancel_handle.clone();
                // Clone the sink for the per-chunk callback; the original stays
                // for the terminal Complete/Error emit below.
                let chunk_sink = sink.clone();
                let on_chunk = move |pcm: Vec<u8>, sample_rate: u32| -> bool {
                    if chunk_sink
                        .add(FfiTtsStreamEvent::AudioChunk { pcm, sample_rate })
                        .is_err()
                        && should_cancel_on_sink_close(
                            reached_terminal.load(std::sync::atomic::Ordering::SeqCst),
                        )
                    {
                        // Dart unsubscribed mid-stream (barge-in): cancel the
                        // synthesis and stop at this chunk boundary.
                        if let Some(handle) = cancel_handle.as_ref() {
                            handle.0.cancel();
                        }
                        return false;
                    }
                    true
                };
                model.run_tts_streaming(&env, &run_options, on_chunk)
            };

            match result {
                Ok(()) => {
                    reached_terminal.store(true, std::sync::atomic::Ordering::SeqCst);
                    // A cancelled run (barge-in via sink-close, or the caller
                    // cancelling the token) stopped early — don't surface it as a
                    // successful Complete. On sink-close the add would no-op
                    // anyway, but a token-only cancel keeps the sink open.
                    let cancelled = cancel_handle
                        .as_ref()
                        .map(|h| h.0.is_cancelled())
                        .unwrap_or(false);
                    if !cancelled {
                        let _ = sink.add(FfiTtsStreamEvent::Complete);
                    }
                }
                Err(e) => {
                    let _ = sink.add(FfiTtsStreamEvent::Error(e.to_string()));
                }
            }
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
    ///
    /// Pass an optional `cancellation_token` to make the run cancellable: when
    /// the token is cancelled (or the Dart sink is closed mid-stream), Rust
    /// generation halts at the next token boundary and releases the model write
    /// lock. When `None`, behavior matches the pre-cancellation streaming path.
    ///
    /// Pass `preempt = true` (latest-frame-wins) together with a
    /// `cancellation_token` to cancel the model's previously in-flight
    /// streaming run before acquiring the write lock — see
    /// [`Self::run_stream`] for the full semantics. Defaults to `false`
    /// (drop-if-busy / serialized); chat passes `false` and is unaffected.
    ///
    /// Pass an optional `frame_session_id` (a caller-supplied UUID) to tag the
    /// run as part of a continuous live-capture session — see [`Self::run_stream`]
    /// for the telemetry rate-limit semantics. `None` for chat / one-shot runs.
    // FRB-exported boundary fn: each param maps to a named Dart argument, so
    // they cannot be bundled into a struct without reshaping the generated Dart
    // API. The arg count (envelope, context, config, token, preempt,
    // frame_session_id, sink) is inherent to the binding surface.
    #[allow(clippy::too_many_arguments)]
    pub fn run_stream_with_context(
        &self,
        envelope: super::envelope::FfiEnvelope,
        context: &FfiConversationContext,
        config: Option<FfiGenerationConfig>,
        cancellation_token: Option<FfiCancellationToken>,
        preempt: bool,
        frame_session_id: Option<String>,
        sink: StreamSink<FfiStreamEvent>,
    ) {
        let model = self.0.clone();
        let env = envelope.into_envelope();
        let ctx = context.0.clone();
        let sdk_config = config.map(|c| c.to_sdk());
        let run_options = streaming_run_options(
            sdk_config,
            cancellation_token.as_ref(),
            frame_session_id.as_deref(),
        );
        let cancel_handle = cancellation_token;

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

            let reached_terminal = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mut token_index = 0u32;

            // Use callback-based streaming through the options-aware path so the
            // per-token abort check runs (honouring the cancellation token).
            let result = {
                let reached_terminal = reached_terminal.clone();
                let cancel_handle = cancel_handle.clone();
                let token_sink = sink.clone();
                model.run_streaming_with_context_options_preempt(
                    &env,
                    &ctx_guard,
                    &run_options,
                    preempt,
                    move |token| {
                        let is_final = token.finish_reason.is_some();
                        let ffi_token = FfiStreamToken {
                            token: token.token.clone(),
                            token_id: token.token_id,
                            index: token_index,
                            cumulative_text: token.cumulative_text.clone(),
                            finish_reason: token.finish_reason.clone(),
                        };
                        token_index = token_index.saturating_add(1);
                        if is_final {
                            reached_terminal.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        if token_sink.add(FfiStreamEvent::Token(ffi_token)).is_err()
                            && should_cancel_on_sink_close(
                                reached_terminal.load(std::sync::atomic::Ordering::SeqCst),
                            )
                        {
                            if let Some(handle) = cancel_handle.as_ref() {
                                handle.0.cancel();
                            }
                        }
                        Ok(())
                    },
                )
            };

            match result {
                Ok(inference_result) => {
                    reached_terminal.store(true, std::sync::atomic::Ordering::SeqCst);
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
        cancellation_token: Option<FfiCancellationToken>,
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
        let run_options = options.to_sdk_with_cancellation(sdk_config, cancellation_token.as_ref());
        let cancel_handle = cancellation_token;
        let cloud_adapter = match gateway_url.as_deref() {
            Some(gateway_url) => CloudRuntimeAdapter::with_gateway(gateway_url),
            None => CloudRuntimeAdapter::new(),
        };

        std::thread::spawn(move || {
            let reached_terminal = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mut token_index = 0u32;
            let result = {
                let reached_terminal = reached_terminal.clone();
                let cancel_handle = cancel_handle.clone();
                let token_sink = sink.clone();
                let mut on_token = |token: xybrid_core::runtime_adapter::types::PartialToken| {
                    let is_final = token.finish_reason.is_some();
                    let ffi_token = FfiStreamToken {
                        token: token.token.clone(),
                        token_id: token.token_id,
                        index: token_index,
                        cumulative_text: token.cumulative_text.clone(),
                        finish_reason: token.finish_reason.clone(),
                    };
                    token_index = token_index.saturating_add(1);
                    if is_final {
                        reached_terminal.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    if token_sink.add(FfiStreamEvent::Token(ffi_token)).is_err()
                        && should_cancel_on_sink_close(
                            reached_terminal.load(std::sync::atomic::Ordering::SeqCst),
                        )
                    {
                        if let Some(handle) = cancel_handle.as_ref() {
                            handle.0.cancel();
                        }
                    }
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
                    reached_terminal.store(true, std::sync::atomic::Ordering::SeqCst);
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

    fn sample_options() -> FfiRunOptions {
        FfiRunOptions {
            cloud_provider: None,
            cloud_model: None,
            cloud_gateway_url: None,
            correlation_id: None,
            abort_on_memory_pressure_critical: false,
            abort_on_thermal_critical: false,
            fallback_to_cloud: false,
            max_grace_tokens: None,
            frame_session_id: None,
        }
    }

    #[test]
    fn to_sdk_without_cancellation_token_does_not_observe_user_cancelled() {
        let sdk = sample_options().to_sdk_with_cancellation(None, None);

        assert!(!sdk.abort_policy.observes(AbortSignal::UserCancelled));
        assert!(sdk.cancellation_token.is_none());
    }

    #[test]
    fn to_sdk_with_cancellation_token_observes_user_cancelled_and_sets_token() {
        let token = FfiCancellationToken::new();
        let sdk = sample_options().to_sdk_with_cancellation(None, Some(&token));

        assert!(sdk.abort_policy.observes(AbortSignal::UserCancelled));
        assert!(sdk.cancellation_token.is_some());

        // Cancelling the FFI handle must flip the SDK token it was cloned into,
        // proving they share the same underlying flag across the boundary.
        let sdk_token = sdk.cancellation_token.expect("token attached");
        assert!(!sdk_token.is_cancelled());
        token.cancel();
        assert!(sdk_token.is_cancelled());
    }

    #[test]
    fn to_sdk_with_cancellation_token_preserves_existing_abort_signals() {
        let mut ffi = sample_options();
        ffi.abort_on_memory_pressure_critical = true;
        ffi.abort_on_thermal_critical = true;
        let token = FfiCancellationToken::new();

        let sdk = ffi.to_sdk_with_cancellation(None, Some(&token));

        assert!(sdk
            .abort_policy
            .observes(AbortSignal::MemoryPressureCritical));
        assert!(sdk.abort_policy.observes(AbortSignal::ThermalCritical));
        assert!(sdk.abort_policy.observes(AbortSignal::UserCancelled));
    }

    #[test]
    fn streaming_run_options_wires_token_only_when_present() {
        let with_none = streaming_run_options(None, None, None);
        assert!(!with_none.abort_policy.observes(AbortSignal::UserCancelled));
        assert!(with_none.cancellation_token.is_none());

        let token = FfiCancellationToken::new();
        let with_token = streaming_run_options(None, Some(&token), None);
        assert!(with_token.abort_policy.observes(AbortSignal::UserCancelled));
        assert!(with_token.cancellation_token.is_some());
    }

    #[test]
    fn streaming_run_options_tags_live_session_when_frame_id_present() {
        // A non-empty frame session id flips the run into live-capture mode.
        let live = streaming_run_options(None, None, Some("frame-sess-7"));
        assert!(live.live_mode);
        assert_eq!(live.frame_session_id.as_deref(), Some("frame-sess-7"));

        // Absent id → plain per-run telemetry (not live).
        let non_live = streaming_run_options(None, None, None);
        assert!(!non_live.live_mode);
        assert_eq!(non_live.frame_session_id, None);

        // Empty / whitespace id is treated as absent (no live tag).
        let blank = streaming_run_options(None, None, Some("   "));
        assert!(!blank.live_mode);
        assert_eq!(blank.frame_session_id, None);
    }

    #[test]
    fn to_sdk_with_frame_session_id_enables_live_mode() {
        let mut ffi = sample_options();
        ffi.frame_session_id = Some("frame-sess-9".to_string());
        let sdk = ffi.to_sdk_with_cancellation(None, None);
        assert!(sdk.live_mode);
        assert_eq!(sdk.frame_session_id.as_deref(), Some("frame-sess-9"));
    }

    #[test]
    fn sink_close_cancels_only_before_terminal() {
        // Non-terminal sink close (Dart unsubscribed mid-stream) → cancel.
        assert!(should_cancel_on_sink_close(/* reached_terminal */ false));
        // Terminal sink close (race after the last token) → no false cancel.
        assert!(!should_cancel_on_sink_close(/* reached_terminal */ true));
    }

    #[test]
    fn ffi_cancellation_token_is_cooperative() {
        let token = FfiCancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        // Cloned handle shares the same flag.
        let clone = token.clone();
        assert!(clone.is_cancelled());
    }

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
            frame_session_id: None,
        };

        let sdk = ffi.to_sdk_with_cancellation(None, None);

        assert!(sdk
            .abort_policy
            .observes(AbortSignal::MemoryPressureCritical));
        assert!(!sdk.abort_policy.observes(AbortSignal::ThermalCritical));
        // No cancellation token supplied → UserCancelled must NOT be observed,
        // preserving the existing chat / cloud-fallback abort semantics.
        assert!(!sdk.abort_policy.observes(AbortSignal::UserCancelled));
        assert!(!sdk.abort_policy.fallback_to_cloud);
        assert_eq!(sdk.abort_policy.max_grace_tokens, 2);
        assert_eq!(sdk.correlation_id.as_deref(), Some("corr-flutter"));
        // No frame session id → run is not live-tagged.
        assert!(!sdk.live_mode);
        assert_eq!(sdk.frame_session_id, None);
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
