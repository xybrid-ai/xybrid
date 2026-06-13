//! Model loading and execution for xybrid-sdk.
//!
//! This module provides:
//! - `ModelLoader`: Preparatory step for loading models (from registry, bundle, or directory)
//! - `XybridModel`: Loaded model ready for inference
//! - `ModelHandle`: Internal state management for the loaded model
//! - `StreamEvent`: Events emitted during streaming inference

use crate::registry_client::RegistryClient;
use crate::result::{InferenceResult, OutputType};
use crate::run_options::{
    check_abort_for_streaming, AbortState, CancellationToken, LiveModeTag, RunOptions,
};
use crate::source::{detect_platform, ModelSource};
use crate::stream::XybridStream;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio_stream::wrappers::ReceiverStream;
use xybrid_core::conversation::ConversationContext;
use xybrid_core::execution::{
    ExecutionTemplate, ModelMetadata, TemplateExecutor, VoiceConfig, VoiceInfo,
};
use xybrid_core::ir::Envelope;
use xybrid_core::orchestrator::authority::{
    ExecutionOutcome, LocalAuthority, OrchestrationAuthority, OutcomeCategory, PolicyOutcome,
    PolicyRequest, ResolvedTarget, SignalContext, StageContext,
};
use xybrid_core::orchestrator::routing_engine::LocalReliabilityHint;
use xybrid_core::runtime_adapter::types::GenerationConfig;
use xybrid_core::streaming::{StreamConfig as CoreStreamConfig, VadStreamConfig as CoreVadConfig};

/// A token generated during streaming inference.
///
/// This is the SDK's version of the core `PartialToken`, re-exported for convenience.
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// The generated token text
    pub token: String,
    /// The token ID (if available from the model)
    pub token_id: Option<i64>,
    /// Index of this token in the generation sequence
    pub index: usize,
    /// All text generated so far (cumulative)
    pub cumulative_text: String,
    /// Reason for stopping (only set on the final token)
    pub finish_reason: Option<String>,
}

/// Events emitted during streaming inference.
///
/// Use this with `run_stream()` to handle tokens as they're generated.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A token was generated (emitted for each token during LLM inference)
    Token(StreamToken),
    /// Inference completed successfully with final result
    Complete(InferenceResult),
    /// An error occurred during inference
    Error(String),
}

/// A boxed, thread-safe error cause carried by the wrapping [`SdkError`]
/// variants so the underlying error stays `source()`-walkable and
/// downcastable instead of being flattened into a string.
pub type SdkErrorSource = Box<dyn std::error::Error + Send + Sync>;

/// SDK-level error type.
///
/// The variants that wrap an underlying failure (`LoadError`,
/// `InferenceError`, `NetworkError`, `CacheError`, `PipelineError`,
/// `Offline`) carry the original error as a `#[source]` cause rather than
/// pre-formatting it into the message, so callers can walk
/// [`std::error::Error::source`] and downcast to the real type. Construct
/// them with the [`SdkError::inference`] / [`SdkError::inference_src`]
/// family of helpers.
#[derive(Debug, thiserror::Error)]
pub enum SdkError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),
    #[error("model_metadata.json not found in directory: {0}")]
    MetadataNotFound(String),
    #[error("model_metadata.json is invalid: {0}")]
    MetadataInvalid(String),
    #[error("Failed to load model: {message}")]
    LoadError {
        message: String,
        #[source]
        source: Option<SdkErrorSource>,
    },
    #[error("Inference failed: {message}")]
    InferenceError {
        message: String,
        #[source]
        source: Option<SdkErrorSource>,
    },
    #[error("Missing artifact: {artifact} at {path}")]
    MissingArtifact { artifact: String, path: String },
    #[error(
        "Unsupported model capability: model '{model_id}' does not support {capability}; {hint}"
    )]
    UnsupportedModelCapability {
        model_id: String,
        capability: String,
        hint: String,
    },
    #[error("Unsupported backend capability: model '{model_id}' requires {capability}, but backend/build '{backend}' does not support {capability}; {hint}")]
    UnsupportedBackendCapability {
        model_id: String,
        backend: String,
        capability: String,
        hint: String,
    },
    /// Local streaming aborted under resource pressure with the caller's
    /// permission to retry on cloud (`AbortPolicy::fallback_to_cloud`).
    /// `run_streaming_with_fallback` catches this variant; lower-level
    /// streaming entry points (e.g. `run_streaming_with_options`) propagate
    /// it so callers can choose their own retry strategy.
    #[error("Aborted for cloud fallback: {reason}")]
    AbortedForCloudFallback {
        reason: xybrid_core::abort::AbortReason,
    },
    #[error("Streaming not supported by this model")]
    StreamingNotSupported,
    #[error("Model not loaded")]
    NotLoaded,
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Network error: {message}")]
    NetworkError {
        message: String,
        #[source]
        source: Option<SdkErrorSource>,
    },
    /// The registry could not be reached at all (DNS failure, connection refused,
    /// network unreachable, interface down). This is distinct from `NetworkError`
    /// because it represents *local* unreachability rather than a server-side problem,
    /// and the circuit breaker treats it differently — offline errors don't count
    /// toward the failure threshold so callers aren't punished for being offline.
    #[error("Registry unreachable: {message}")]
    Offline {
        message: String,
        #[source]
        source: Option<SdkErrorSource>,
    },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Cache error: {message}")]
    CacheError {
        message: String,
        #[source]
        source: Option<SdkErrorSource>,
    },
    #[error("Pipeline error: {message}")]
    PipelineError {
        message: String,
        #[source]
        source: Option<SdkErrorSource>,
    },
    #[error("Circuit breaker open: {0}")]
    CircuitOpen(String),
    #[error("Rate limited, retry after {retry_after_secs} seconds")]
    RateLimited { retry_after_secs: u64 },
    #[error("Request timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
}

/// Generates the `message`-only and `_src` (cause-chaining) constructors for
/// the wrapping `SdkError` variants, so call sites stay terse
/// (`SdkError::cache_src("…", e)`) while preserving the underlying cause.
macro_rules! sdk_error_ctors {
    ($($msg_fn:ident, $src_fn:ident => $variant:ident);+ $(;)?) => {
        impl SdkError {
            $(
                #[doc = concat!("Build [`SdkError::", stringify!($variant), "`] from a message with no underlying cause.")]
                pub(crate) fn $msg_fn(message: impl Into<String>) -> Self {
                    SdkError::$variant { message: message.into(), source: None }
                }

                #[doc = concat!("Build [`SdkError::", stringify!($variant), "`] from a message, chaining `source` as the `#[source]` cause.")]
                pub(crate) fn $src_fn(
                    message: impl Into<String>,
                    source: impl Into<SdkErrorSource>,
                ) -> Self {
                    SdkError::$variant { message: message.into(), source: Some(source.into()) }
                }
            )+
        }
    };
}

sdk_error_ctors! {
    load, load_src => LoadError;
    inference, inference_src => InferenceError;
    network, network_src => NetworkError;
    cache, cache_src => CacheError;
    pipeline, pipeline_src => PipelineError;
    offline, offline_src => Offline;
}

/// Result type for SDK operations.
pub type SdkResult<T> = Result<T, SdkError>;

impl SdkError {
    /// Whether retrying the operation that produced this error could
    /// succeed without the caller changing anything.
    ///
    /// Transient failures (`NetworkError`, `RateLimited`, `Timeout`,
    /// `Offline`) are retryable; everything else — including
    /// `CircuitOpen`, `ConfigError`, `ModelNotFound`, `LoadError`,
    /// `InferenceError`, and `AbortedForCloudFallback` — is not. `Offline`
    /// is retryable only across *different* registry URLs (a fallback
    /// registry may be reachable when the primary isn't); within a single
    /// URL the retry loop short-circuits (see `registry_client`).
    ///
    /// This is the inherent form of the [`xybrid_core::http::RetryableError`]
    /// trait method, exposed directly on `SdkError` so callers (and the
    /// FFI / UniFFI layers) can query retryability without importing the
    /// trait. The trait impl forwards here.
    pub fn is_retryable(&self) -> bool {
        match self {
            // Retryable errors (transient failures)
            SdkError::NetworkError { .. } => true,
            SdkError::RateLimited { .. } => true,
            SdkError::Timeout { .. } => true,
            // Offline is "retryable" only across URLs — the fallback registry
            // may be reachable even when the primary isn't. Within a single URL
            // the retry loop short-circuits immediately (see registry_client).
            SdkError::Offline { .. } => true,

            // Non-retryable errors (permanent failures)
            SdkError::ModelNotFound(_) => false,
            SdkError::DirectoryNotFound(_) => false,
            SdkError::MetadataNotFound(_) => false,
            SdkError::MetadataInvalid(_) => false,
            SdkError::LoadError { .. } => false,
            SdkError::InferenceError { .. } => false,
            SdkError::MissingArtifact { .. } => false,
            SdkError::UnsupportedModelCapability { .. } => false,
            SdkError::UnsupportedBackendCapability { .. } => false,
            // Resource-driven abort is not retryable on the same path; the
            // wrapper redirects to cloud instead.
            SdkError::AbortedForCloudFallback { .. } => false,
            SdkError::StreamingNotSupported => false,
            SdkError::NotLoaded => false,
            SdkError::ConfigError(_) => false,
            SdkError::IoError(_) => false,
            SdkError::CacheError { .. } => false,
            SdkError::PipelineError { .. } => false,
            SdkError::CircuitOpen(_) => false, // Don't retry when circuit is open
        }
    }

    /// The minimum delay a caller should wait before retrying, when the
    /// error itself dictates one. Only `RateLimited` carries a
    /// server-specified backoff; every other variant returns `None`
    /// (the caller picks its own backoff if [`Self::is_retryable`]).
    ///
    /// Inherent form of [`xybrid_core::http::RetryableError::retry_after`];
    /// the trait impl forwards here.
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            SdkError::RateLimited { retry_after_secs } => {
                Some(std::time::Duration::from_secs(*retry_after_secs))
            }
            _ => None,
        }
    }
}

fn sdk_execution_error<E>(context: &str, error: E) -> SdkError
where
    E: Into<xybrid_core::error::XybridError>,
{
    let error = error.into();
    match error {
        xybrid_core::error::XybridError::MissingArtifact { artifact, path } => {
            SdkError::MissingArtifact { artifact, path }
        }
        xybrid_core::error::XybridError::UnsupportedModelCapability {
            model_id,
            capability,
            hint,
        } => SdkError::UnsupportedModelCapability {
            model_id,
            capability,
            hint,
        },
        xybrid_core::error::XybridError::UnsupportedBackendCapability {
            model_id,
            backend,
            capability,
            hint,
        } => SdkError::UnsupportedBackendCapability {
            model_id,
            backend,
            capability,
            hint,
        },
        other => SdkError::inference_src(context, other),
    }
}

impl xybrid_core::http::RetryableError for SdkError {
    fn is_retryable(&self) -> bool {
        SdkError::is_retryable(self)
    }

    fn retry_after(&self) -> Option<std::time::Duration> {
        SdkError::retry_after(self)
    }

    fn circuit_open() -> Self {
        SdkError::CircuitOpen(
            "circuit breaker stayed open for the entire retry window; no request was sent"
                .to_string(),
        )
    }
}

fn streaming_execution_error(error: xybrid_core::runtime_adapter::AdapterError) -> SdkError {
    match error {
        xybrid_core::runtime_adapter::AdapterError::AbortedForCloudFallback { reason } => {
            SdkError::AbortedForCloudFallback { reason }
        }
        other => SdkError::inference_src("Streaming execution failed", other),
    }
}

fn streaming_callback_error(error: Box<dyn std::error::Error + Send + Sync>) -> SdkError {
    if let Some(reason) = xybrid_core::abort::cloud_fallback_reason_from_error(error.as_ref()) {
        return SdkError::AbortedForCloudFallback { reason };
    }
    SdkError::inference_src("Streaming callback failed", error)
}

fn streaming_pre_run_abort_error(
    reason: crate::run_options::AbortReason,
    fallback_to_cloud: bool,
) -> SdkError {
    if fallback_to_cloud && !matches!(reason, crate::run_options::AbortReason::UserCancelled) {
        return SdkError::AbortedForCloudFallback {
            reason: reason.to_core_abort_reason(),
        };
    }
    SdkError::inference(format!("Execution aborted: {reason}"))
}

/// Stamp the live-capture tag onto a telemetry-event `data` object.
///
/// When `live_tag` is `Some`, inserts the flat `live_mode = true` +
/// `frame_session_id = <uuid>` fields that `convert_to_platform_event` hoists
/// to the wire payload top level and the dispatch funnel uses to rate-limit per
/// session. When `None` (every non-live run), `data` is left **byte-for-byte
/// unchanged** so the existing telemetry path is unaffected.
fn stamp_live_mode_tag(data: &mut serde_json::Value, live_tag: Option<&LiveModeTag>) {
    let Some(tag) = live_tag else {
        return;
    };
    if let Some(obj) = data.as_object_mut() {
        obj.insert("live_mode".to_string(), serde_json::json!(true));
        obj.insert(
            "frame_session_id".to_string(),
            serde_json::json!(tag.frame_session_id),
        );
    }
}

/// Information about a local→cloud handoff "seam" surfaced by
/// [`XybridModel::run_streaming_with_fallback`].
///
/// The wrapper invokes the caller's `on_seam` once when a local stream
/// aborts under resource pressure and the run is about to continue on
/// cloud. Callers use this to render UX cues ("switching to cloud…") or
/// reconcile telemetry against the same `correlation_id`.
#[derive(Debug, Clone)]
pub struct SeamInfo {
    /// Why the local run aborted.
    pub reason: xybrid_core::abort::AbortReason,
    /// Correlation id linking the local-aborted and cloud-retry telemetry events.
    pub correlation_id: String,
    /// Number of `PartialToken`s the local leg emitted before aborting.
    pub local_tokens: u32,
    /// Wall-clock latency of the local leg, in milliseconds.
    pub local_latency_ms: u32,
}

const FALLBACK_POLICY_RESOURCE_MAX_AGE: Duration = Duration::from_millis(500);

static FALLBACK_AUTHORITY: OnceLock<LocalAuthority> = OnceLock::new();

fn fallback_authority() -> &'static dyn OrchestrationAuthority {
    FALLBACK_AUTHORITY.get_or_init(LocalAuthority::new)
}

fn fallback_policy_metrics(options: &RunOptions) -> xybrid_core::context::DeviceMetrics {
    let snapshot = options
        .resource_provider
        .as_ref()
        .map(|provider| provider.current_snapshot(FALLBACK_POLICY_RESOURCE_MAX_AGE))
        .unwrap_or_else(|| {
            xybrid_core::device::ResourceMonitor::global()
                .current_snapshot(FALLBACK_POLICY_RESOURCE_MAX_AGE)
        });
    // Prefer caller-supplied DeviceMetrics so RTT- and battery-based deny
    // rules in the policy engine see real values. `with_live_snapshot` then
    // overlays the freshly sampled resource snapshot on top — best of both
    // worlds. Falls back to `DeviceMetrics::default()` when the caller has
    // no device adapter wired (the historical behaviour).
    let base = options.device_metrics.clone().unwrap_or_default();
    base.with_live_snapshot(snapshot)
}

fn cloud_target(provider: Option<&str>) -> ResolvedTarget {
    ResolvedTarget::Cloud {
        provider: provider.unwrap_or("xybrid").to_string(),
    }
}

fn record_local_abort_outcome(
    authority: &dyn OrchestrationAuthority,
    model_id: &str,
    reason: xybrid_core::abort::AbortReason,
    latency_ms: u32,
    signal_context: Option<SignalContext>,
) {
    authority.record_outcome(&ExecutionOutcome {
        stage_id: model_id.to_string(),
        target: ResolvedTarget::Device,
        latency_ms: latency_ms as u64,
        success: false,
        error: Some(reason.as_str().to_string()),
        category: Some(OutcomeCategory::AbortedForCloudFallback { reason }),
        model_id: Some(model_id.to_string()),
        signal_context,
    });
}

fn record_cloud_outcome(
    authority: &dyn OrchestrationAuthority,
    model_id: &str,
    provider: Option<&str>,
    latency_ms: u32,
    success: bool,
    error: Option<String>,
    category: OutcomeCategory,
    signal_context: Option<SignalContext>,
) {
    authority.record_outcome(&ExecutionOutcome {
        stage_id: model_id.to_string(),
        target: cloud_target(provider),
        latency_ms: latency_ms as u64,
        success,
        error,
        category: Some(category),
        model_id: Some(model_id.to_string()),
        signal_context,
    });
}

fn local_reliability_hint_after_abort(
    authority: &dyn OrchestrationAuthority,
    model_id: &str,
    envelope: &Envelope,
    metrics: &xybrid_core::context::DeviceMetrics,
) -> Option<LocalReliabilityHint> {
    let context = StageContext {
        stage_id: model_id.to_string(),
        model_id: model_id.to_string(),
        input_kind: envelope.kind.clone(),
        metrics: metrics.clone(),
        resource_monitor: xybrid_core::device::ResourceMonitor::global(),
        explicit_target: None,
        local_availability: None,
        device_class: Some(metrics.canonical_device_class()),
        device_class_schema_version: Some(xybrid_core::context::DEVICE_CLASS_SCHEMA_VERSION),
    };
    authority
        .resolve_target_with_feedback(&context)
        .local_reliability_hint
}

/// Inspect the local-leg result and, on a typed cloud-fallback abort, fire
/// `on_seam`, retry on the cloud adapter, and return the cloud
/// [`InferenceResult`]. On any other shape the original result is returned
/// unchanged.
///
/// `cancellation_token`, when set, makes the cloud retry leg honour
/// caller-driven cancellation. The cloud leg cannot meaningfully react to
/// resource pressure on the device, so only `UserCancelled` is consulted —
/// matching the local leg's contract.
///
/// Lives as a free function so unit tests can drive it directly without
/// constructing a real [`XybridModel`].
#[allow(clippy::too_many_arguments)]
fn dispatch_after_local<F, S>(
    local_result: SdkResult<InferenceResult>,
    envelope: &Envelope,
    cloud_adapter: &dyn xybrid_core::runtime_adapter::CloudStreaming,
    correlation_id: String,
    model_id: &str,
    local_tokens: u32,
    local_latency_ms: u32,
    local_resource_summary: Option<xybrid_core::device::ResourceUsageSummary>,
    authority: &dyn OrchestrationAuthority,
    policy_metrics: xybrid_core::context::DeviceMetrics,
    signal_context: Option<SignalContext>,
    cancellation_token: Option<CancellationToken>,
    on_token: &mut F,
    on_seam: &mut S,
) -> SdkResult<InferenceResult>
where
    F: FnMut(
            xybrid_core::runtime_adapter::types::PartialToken,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
        + Send,
    S: FnMut(SeamInfo) + Send,
{
    match local_result {
        Ok(result) => Ok(result),
        Err(SdkError::AbortedForCloudFallback { reason }) => {
            record_local_abort_outcome(
                authority,
                model_id,
                reason,
                local_latency_ms,
                signal_context,
            );
            let local_reliability_hint =
                local_reliability_hint_after_abort(authority, model_id, envelope, &policy_metrics);
            // Emit `LocalAborted` before the user's `on_seam` callback fires so
            // the audit trail exists even if the caller's seam handler panics.
            crate::telemetry::publish_local_aborted_with_details(
                &correlation_id,
                model_id,
                reason,
                local_latency_ms,
                local_tokens,
                local_resource_summary,
                local_reliability_hint,
            );

            let seam = SeamInfo {
                reason,
                correlation_id: correlation_id.clone(),
                local_tokens,
                local_latency_ms,
            };
            on_seam(seam);

            if cancellation_token
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Err(SdkError::inference(format!(
                    "Execution aborted: {}",
                    crate::run_options::AbortReason::UserCancelled
                )));
            }

            // FR-6: reuse the original prompt; no partial-token reuse.
            let cloud_envelope = envelope.clone();
            let cloud_provider = cloud_envelope.metadata.get("provider").cloned();
            // The cloud leg almost always runs a different model than the
            // local leg (e.g. local `qwen2.5-0.5b-instruct` falling back to
            // cloud `deepseek-chat`). The dashboard reads `model_id` off
            // the published event, so passing the local `model_id` through
            // to the cloud event makes the trace lie about which model
            // actually produced the cloud tokens. Pull the cloud model from
            // the envelope's `model` metadata (the same field the gateway
            // dispatches on) and fall back to the local id only when the
            // caller didn't set one — in that case the dispatch would have
            // failed at the gateway anyway, so the local id is a fine
            // last-resort label.
            let cloud_model_id = cloud_envelope
                .metadata
                .get("model")
                .map(|s| s.as_str())
                .unwrap_or(model_id);
            let policy_decision = authority.apply_policy(&PolicyRequest {
                stage_id: cloud_model_id.to_string(),
                envelope: cloud_envelope.clone(),
                metrics: policy_metrics,
            });
            match policy_decision.result {
                PolicyOutcome::Allow => {}
                PolicyOutcome::Deny {
                    reason: policy_reason,
                } => {
                    crate::telemetry::publish_cloud_denied_by_policy(
                        &correlation_id,
                        cloud_model_id,
                        reason,
                        &policy_reason,
                        local_latency_ms,
                    );
                    record_cloud_outcome(
                        authority,
                        cloud_model_id,
                        cloud_provider.as_deref(),
                        0,
                        false,
                        Some(format!("cloud_denied_by_policy: {}", policy_reason)),
                        OutcomeCategory::HardFail {
                            reason: "cloud_denied_by_policy".to_string(),
                        },
                        signal_context,
                    );
                    return Err(SdkError::inference(format!(
                        "cloud_denied_by_policy: {}",
                        policy_reason
                    )));
                }
                PolicyOutcome::Transform { transforms } => {
                    // Defensive hard-fail: the orchestrator path applies
                    // PolicyEngine::redact when this variant fires, but the
                    // SDK fallback path does not yet plumb a redact seam.
                    // Treating Transform as Allow would silently dispatch
                    // the un-redacted envelope to cloud — a privacy
                    // regression the orchestrator path explicitly avoids.
                    // Fail closed until the redact seam is wired in.
                    let policy_reason =
                        format!("transforms_unsupported_in_fallback: {:?}", transforms);
                    crate::telemetry::publish_cloud_denied_by_policy(
                        &correlation_id,
                        cloud_model_id,
                        reason,
                        &policy_reason,
                        local_latency_ms,
                    );
                    record_cloud_outcome(
                        authority,
                        cloud_model_id,
                        cloud_provider.as_deref(),
                        0,
                        false,
                        Some(format!("cloud_denied_by_policy: {}", policy_reason)),
                        OutcomeCategory::HardFail {
                            reason: "cloud_denied_by_policy".to_string(),
                        },
                        signal_context,
                    );
                    return Err(SdkError::inference(format!(
                        "cloud_denied_by_policy: {}",
                        policy_reason
                    )));
                }
            }

            let cloud_start = Instant::now();
            let cloud_tokens = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
            let cloud_tokens_for_cb = cloud_tokens.clone();
            let cancellation_for_cb = cancellation_token.clone();
            let cloud_callback: xybrid_core::runtime_adapter::types::StreamingCallback<'_> =
                Box::new(move |token| {
                    // Honour user cancellation across the local→cloud seam.
                    // Resource-pressure signals are not consulted: the device's
                    // CPU/memory state says nothing about a cloud round-trip's
                    // viability, and treating it as a cloud-leg abort would
                    // surface CloudFallbackAbort errors at the wrong layer.
                    if let Some(token_handle) = cancellation_for_cb.as_ref() {
                        if token_handle.is_cancelled() {
                            return Err(Box::new(crate::run_options::AbortReason::UserCancelled)
                                as Box<dyn std::error::Error + Send + Sync>);
                        }
                    }
                    cloud_tokens_for_cb.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    on_token(token)
                });
            // Match instead of `?` so we publish a `CloudRetry` event on
            // BOTH branches. Without this the audit trail loses any record
            // that cloud was even attempted when the cloud leg fails (auth
            // misconfiguration, gateway 5xx, network partition, etc.) —
            // the trace would just stop after `LocalAborted` and the
            // dashboard would show no cloud activity at all.
            let cloud_result = cloud_adapter.execute_streaming(&cloud_envelope, cloud_callback);
            let cloud_latency_ms = cloud_start.elapsed().as_millis() as u32;
            let cloud_token_count = cloud_tokens.load(std::sync::atomic::Ordering::SeqCst);

            match cloud_result {
                Ok(cloud_output) => {
                    crate::telemetry::publish_cloud_retry(
                        &correlation_id,
                        cloud_model_id,
                        cloud_provider.as_deref(),
                        cloud_latency_ms,
                        cloud_token_count,
                        None,
                    );
                    record_cloud_outcome(
                        authority,
                        cloud_model_id,
                        cloud_provider.as_deref(),
                        cloud_latency_ms,
                        true,
                        None,
                        OutcomeCategory::Success,
                        signal_context,
                    );
                    let total_latency_ms = local_latency_ms.saturating_add(cloud_latency_ms);
                    Ok(InferenceResult::new(
                        cloud_output,
                        cloud_model_id,
                        total_latency_ms,
                    ))
                }
                Err(adapter_err) => {
                    let error_message = adapter_err.to_string();
                    let telemetry_error =
                        crate::telemetry::redact_error_for_telemetry(&error_message);
                    crate::telemetry::publish_cloud_retry(
                        &correlation_id,
                        cloud_model_id,
                        cloud_provider.as_deref(),
                        cloud_latency_ms,
                        cloud_token_count,
                        Some(&error_message),
                    );
                    record_cloud_outcome(
                        authority,
                        cloud_model_id,
                        cloud_provider.as_deref(),
                        cloud_latency_ms,
                        false,
                        Some(telemetry_error.clone()),
                        OutcomeCategory::HardFail {
                            reason: telemetry_error,
                        },
                        signal_context,
                    );
                    Err(streaming_execution_error(adapter_err))
                }
            }
        }
        Err(other) => Err(other),
    }
}

/// Configuration for streaming ASR sessions.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Enable VAD (Voice Activity Detection) for smart chunking
    pub enable_vad: bool,
    /// VAD threshold (0.0-1.0)
    pub vad_threshold: f32,
    /// Language hint for ASR
    pub language: Option<String>,
    /// Path to VAD model (uses default if None)
    pub vad_model_dir: Option<String>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            enable_vad: false,
            vad_threshold: 0.5,
            language: Some("en".to_string()),
            vad_model_dir: None,
        }
    }
}

impl StreamConfig {
    /// Create config with VAD enabled.
    pub fn with_vad() -> Self {
        Self {
            enable_vad: true,
            ..Default::default()
        }
    }

    /// Set language hint.
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set VAD threshold.
    pub fn vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold;
        self
    }
}

/// Internal handle holding the loaded model state.
struct ModelHandle {
    /// Template executor for running inference
    executor: TemplateExecutor,
    /// Model metadata
    metadata: ModelMetadata,
    /// Model directory path (permanent extraction in cache)
    model_dir: PathBuf,
    /// Whether model is currently loaded
    loaded: bool,
}

/// Represents a model that can be loaded.
///
/// Created by `Xybrid::model()`, must call `.load()` to use.
/// This is a preparatory step that doesn't download or load anything.
///
/// # Example (Recommended - Registry-based)
///
/// ```no_run
/// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
/// # use xybrid_sdk::ModelLoader;
/// # use xybrid_sdk::ir::Envelope;
/// # let envelope: Envelope = unimplemented!();
/// // Load using registry (recommended - auto-resolves to best variant)
/// let loader = ModelLoader::from_registry("kokoro-82m");
/// let model = loader.load()?;
/// let result = model.run(&envelope, None)?;
/// # Ok(())
/// # }
/// ```
///
/// # Example (With progress callback)
///
/// ```no_run
/// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
/// # use xybrid_sdk::ModelLoader;
/// let loader = ModelLoader::from_registry("kokoro-82m");
/// let model = loader.load_with_progress(|progress| {
///     println!("Download: {:.1}%", progress * 100.0);
/// })?;
/// # Ok(())
/// # }
/// ```
/// GGUF quantization preference order for automatic selection.
/// Q4_K_M is the default — best quality/size tradeoff for edge devices.
const GGUF_PREFERENCE_ORDER: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "BF16", "F32",
];

/// Select the best GGUF file from a list based on user preference or default ranking.
///
/// If `variant` is specified, finds a file containing that quantization string (case-insensitive).
/// Otherwise, selects the file matching the highest-priority quantization from `GGUF_PREFERENCE_ORDER`.
fn select_gguf_variant(gguf_files: &[&str], variant: Option<&str>) -> SdkResult<String> {
    if let Some(v) = variant {
        let v_upper = v.to_uppercase();
        // Find a file containing the variant string (case-insensitive)
        if let Some(found) = gguf_files
            .iter()
            .find(|f| f.to_uppercase().contains(&v_upper))
        {
            return Ok(found.to_string());
        }
        return Err(SdkError::load(format!(
            "No GGUF file matching variant '{}'. Available: {}",
            v,
            gguf_files.join(", ")
        )));
    }

    // Auto-select: try each preferred quantization in order
    for pref in GGUF_PREFERENCE_ORDER {
        if let Some(found) = gguf_files.iter().find(|f| f.to_uppercase().contains(pref)) {
            return Ok(found.to_string());
        }
    }

    // Fallback: pick the smallest file (likely the most quantized)
    Ok(gguf_files
        .first()
        .ok_or_else(|| SdkError::load("No GGUF files found"))?
        .to_string())
}

#[derive(Debug, Clone)]
pub struct ModelLoader {
    source: ModelSource,
    model_id: Option<String>,
    version: Option<String>,
}

impl ModelLoader {
    /// Create loader from registry (recommended).
    ///
    /// Uses the registry API to resolve the model ID to the best variant
    /// for the current platform, then downloads from HuggingFace with
    /// caching and SHA256 verification.
    ///
    /// # Example
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::ModelLoader;
    /// let loader = ModelLoader::from_registry("kokoro-82m");
    /// let model = loader.load()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_registry(id: &str) -> Self {
        Self {
            source: ModelSource::registry(id),
            model_id: Some(id.to_string()),
            version: None, // Version is resolved by registry API
        }
    }

    /// Create loader from registry with explicit platform.
    ///
    /// # Example
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::ModelLoader;
    /// let loader = ModelLoader::from_registry_with_platform("kokoro-82m", "macos-arm64");
    /// let model = loader.load()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_registry_with_platform(id: &str, platform: &str) -> Self {
        Self {
            source: ModelSource::registry_with_platform(id, platform),
            model_id: Some(id.to_string()),
            version: None,
        }
    }

    /// Create loader from legacy registry with direct URL.
    ///
    /// # Deprecated
    /// Use `from_registry()` instead for automatic platform resolution and caching.
    #[deprecated(since = "0.0.17", note = "Use ModelLoader::from_registry() instead")]
    #[allow(deprecated)]
    pub fn from_legacy_registry(url: &str, model_id: &str, version: &str) -> Self {
        Self {
            source: ModelSource::legacy_registry(url, model_id, version),
            model_id: Some(model_id.to_string()),
            version: Some(version.to_string()),
        }
    }

    /// Create loader from legacy registry with explicit platform.
    ///
    /// # Deprecated
    /// Use `from_registry_with_platform()` instead.
    #[deprecated(
        since = "0.0.17",
        note = "Use ModelLoader::from_registry_with_platform() instead"
    )]
    #[allow(deprecated)]
    pub fn from_legacy_registry_with_platform(
        url: &str,
        model_id: &str,
        version: &str,
        platform: &str,
    ) -> Self {
        Self {
            source: ModelSource::legacy_registry_with_platform(url, model_id, version, platform),
            model_id: Some(model_id.to_string()),
            version: Some(version.to_string()),
        }
    }

    /// Create loader from local bundle file.
    pub fn from_bundle(path: impl Into<PathBuf>) -> SdkResult<Self> {
        let path: PathBuf = path.into();
        if !path.exists() {
            return Err(SdkError::ModelNotFound(format!(
                "Bundle not found: {:?}",
                path
            )));
        }
        Ok(Self {
            source: ModelSource::bundle(path),
            model_id: None,
            version: None,
        })
    }

    /// Create loader from local model directory.
    ///
    /// The directory must exist and contain a valid `model_metadata.json` file.
    ///
    /// # Errors
    ///
    /// - `SdkError::DirectoryNotFound` if the path does not exist
    /// - `SdkError::MetadataNotFound` if `model_metadata.json` is missing
    /// - `SdkError::MetadataInvalid` if `model_metadata.json` contains invalid JSON
    pub fn from_directory(path: impl Into<PathBuf>) -> SdkResult<Self> {
        let path: PathBuf = path.into();
        if !path.exists() {
            return Err(SdkError::DirectoryNotFound(path.display().to_string()));
        }
        let metadata_path = path.join("model_metadata.json");
        if !metadata_path.exists() {
            return Err(SdkError::MetadataNotFound(path.display().to_string()));
        }
        // Validate that the metadata is valid JSON
        let metadata_str = std::fs::read_to_string(&metadata_path).map_err(|e| {
            SdkError::MetadataInvalid(format!("failed to read model_metadata.json: {}", e))
        })?;
        let _metadata: xybrid_core::execution::ModelMetadata = serde_json::from_str(&metadata_str)
            .map_err(|e| {
                SdkError::MetadataInvalid(format!("invalid model_metadata.json: {}", e))
            })?;
        Ok(Self {
            source: ModelSource::directory(path),
            model_id: None,
            version: None,
        })
    }

    /// Create loader from a HuggingFace Hub repository.
    ///
    /// Downloads model files from the HuggingFace Hub and caches them locally.
    /// Subsequent calls use the cached files. The repository must contain a
    /// `model_metadata.json` for the model to be loadable (auto-generation
    /// is planned for a future version).
    ///
    /// Requires the `huggingface` feature flag at load time.
    /// The constructor itself is always available, but `load()` will return
    /// `SdkError::ConfigError` if the feature is not enabled.
    ///
    /// # Example
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::ModelLoader;
    /// let loader = ModelLoader::from_huggingface("xybrid-ai/kokoro-82m");
    /// let model = loader.load()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_huggingface(repo: &str) -> Self {
        Self {
            source: ModelSource::huggingface(repo),
            model_id: Some(repo.to_string()),
            version: None,
        }
    }

    /// Create loader from a HuggingFace Hub repository with explicit revision.
    ///
    /// # Example
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::ModelLoader;
    /// let loader = ModelLoader::from_huggingface_with_revision("xybrid-ai/kokoro-82m", "v1.0");
    /// let model = loader.load()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_huggingface_with_revision(repo: &str, revision: &str) -> Self {
        Self {
            source: ModelSource::huggingface_with_revision(repo, revision),
            model_id: Some(repo.to_string()),
            version: Some(revision.to_string()),
        }
    }

    /// Create loader from a HuggingFace repo string, parsing optional variant suffix.
    ///
    /// Supports `"org/repo:Q8_0"` syntax to select a specific GGUF quantization.
    /// Without a variant, defaults to Q4_K_M for GGUF repos.
    ///
    /// # Example
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::ModelLoader;
    /// let loader = ModelLoader::from_huggingface_parsed("LiquidAI/LFM2.5-350M-GGUF:Q8_0");
    /// let model = loader.load()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_huggingface_parsed(input: &str) -> Self {
        let source = ModelSource::parse_huggingface(input);
        let repo = source.model_id().unwrap_or(input).to_string();
        Self {
            source,
            model_id: Some(repo),
            version: None,
        }
    }

    /// Get the model ID (if known).
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Get the version (if known).
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    /// Get the source type.
    pub fn source_type(&self) -> &'static str {
        self.source.source_type()
    }

    /// Load the model into memory (synchronous).
    ///
    /// This will:
    /// - For registry: Resolve via registry API, download from HuggingFace (with caching)
    /// - For legacy_registry (deprecated): Download the bundle if not cached, extract it
    /// - For bundle: Extract the bundle to a temp directory
    /// - For directory: Load directly from the directory
    ///
    /// Returns a loaded `XybridModel` ready for inference.
    #[allow(deprecated)]
    pub fn load(&self) -> SdkResult<XybridModel> {
        self.load_with_progress(|_| {})
    }

    /// Load the model with a progress callback.
    ///
    /// The callback receives progress as a float from 0.0 to 1.0.
    /// Only applies to registry-based loading (downloads from HuggingFace).
    ///
    /// # Example
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::ModelLoader;
    /// # let loader: ModelLoader = unimplemented!();
    /// let model = loader.load_with_progress(|progress| {
    ///     println!("Download: {:.1}%", progress * 100.0);
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(deprecated)]
    pub fn load_with_progress<F>(&self, progress_callback: F) -> SdkResult<XybridModel>
    where
        F: Fn(f32),
    {
        match &self.source {
            ModelSource::Registry { id, platform } => {
                self.load_from_registry_api(id, platform.as_deref(), progress_callback)
            }
            ModelSource::LegacyRegistry {
                url,
                model_id,
                version,
                platform,
            } => self.load_from_legacy_registry(url, model_id, version, platform.as_deref()),
            ModelSource::Bundle { path } => self.load_from_bundle(path),
            ModelSource::Directory { path } => self.load_from_directory(path),
            ModelSource::HuggingFace {
                repo,
                revision,
                variant,
            } => self.load_from_huggingface(
                repo,
                revision.as_deref(),
                variant.as_deref(),
                progress_callback,
            ),
        }
    }

    /// Load the model asynchronously.
    pub async fn load_async(&self) -> SdkResult<XybridModel> {
        // For now, wrap the sync version. Real async would use tokio::fs and async HTTP.
        let loader = self.clone();
        tokio::task::spawn_blocking(move || loader.load())
            .await
            .map_err(|e| SdkError::load_src("Task join error", e))?
    }

    /// Load model from registry using RegistryClient.
    ///
    /// This is the recommended loading method - it uses the registry API to resolve
    /// the model ID to the best variant for the platform, downloads from HuggingFace,
    /// and caches locally with SHA256 verification.
    fn load_from_registry_api<F>(
        &self,
        id: &str,
        platform: Option<&str>,
        progress_callback: F,
    ) -> SdkResult<XybridModel>
    where
        F: Fn(f32),
    {
        // Create registry client (uses default API or environment variable)
        let client = RegistryClient::from_env()?;

        // Fetch and extract model (handles both .xyb bundles and passthrough GGUF files)
        let model_dir = client.fetch_extracted(id, platform, progress_callback)?;

        // Load from extracted directory
        self.load_from_directory(&model_dir)
    }

    /// Load from legacy registry (deprecated - use load_from_registry_api instead).
    fn load_from_legacy_registry(
        &self,
        url: &str,
        model_id: &str,
        version: &str,
        platform: Option<&str>,
    ) -> SdkResult<XybridModel> {
        let platform = platform.map(String::from).unwrap_or_else(detect_platform);

        // Build bundle URL
        let bundle_url = format!(
            "{}/bundles/{}/{}/{}/{}.xyb",
            url.trim_end_matches('/'),
            model_id,
            version,
            platform,
            model_id
        );

        // Download bundle to temp file
        let temp_dir = TempDir::new().map_err(SdkError::IoError)?;
        let bundle_path = temp_dir.path().join(format!("{}.xyb", model_id));

        // Use blocking HTTP client
        let response = ureq::get(&bundle_url)
            .call()
            .map_err(|e| SdkError::network_src("Failed to download bundle", e))?;

        if response.status() != 200 {
            return Err(SdkError::ModelNotFound(format!(
                "Bundle not found at registry: {} (status {})",
                bundle_url,
                response.status()
            )));
        }

        // Write bundle to temp file
        let mut file = std::fs::File::create(&bundle_path)?;
        std::io::copy(&mut response.into_reader(), &mut file)?;

        // Extract using CacheManager (extracts to permanent cache location)
        // The temp_dir will be dropped after this, but extracted files persist
        self.load_from_bundle(&bundle_path)
    }

    fn load_from_bundle(&self, path: &PathBuf) -> SdkResult<XybridModel> {
        // Use CacheManager for unified extraction (single source of truth)
        let cache = crate::cache::CacheManager::new()?;
        let extract_dir = cache.ensure_extracted(path)?;

        // Load from extracted directory (extraction is permanent in cache)
        let handle = Self::create_model_handle(&extract_dir)?;

        let model_id = handle.metadata.model_id.clone();
        let version = handle.metadata.version.clone();
        let supports_streaming = Self::check_streaming_support(&handle.metadata);
        let output_type = Self::infer_output_type(&handle.metadata);

        Ok(XybridModel {
            handle: Arc::new(RwLock::new(handle)),
            model_id,
            version,
            output_type,
            supports_streaming,
            current_run: Arc::new(Mutex::new(None)),
        })
    }

    /// Load a model from HuggingFace Hub.
    ///
    /// Only downloads the selected GGUF variant (defaults to Q4_K_M) plus essential
    /// supporting files (config, tokenizer, README). Avoids downloading all variants.
    #[cfg(feature = "huggingface")]
    fn load_from_huggingface<F>(
        &self,
        repo: &str,
        revision: Option<&str>,
        variant: Option<&str>,
        _progress_callback: F,
    ) -> SdkResult<XybridModel>
    where
        F: Fn(f32),
    {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        // Determine our cache directory
        let cache_dir = Self::hf_cache_dir(repo)?;

        // Check if we already have a cached copy with model_metadata.json
        let metadata_path = cache_dir.join("model_metadata.json");
        if metadata_path.exists() {
            log::info!(target: "xybrid_sdk", "Loading HuggingFace model from cache: {}", cache_dir.display());
            return self.load_from_directory(&cache_dir);
        }

        log::info!(target: "xybrid_sdk", "Downloading model from HuggingFace: {}", repo);

        // Create HF API client
        let api = Api::new()
            .map_err(|e| SdkError::network_src("Failed to create HuggingFace API client", e))?;

        // Create repo reference with optional revision
        let hf_repo = if let Some(rev) = revision {
            Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string())
        } else {
            Repo::new(repo.to_string(), RepoType::Model)
        };
        let repo_api = api.repo(hf_repo);

        // Get repo info to list all files
        let repo_info = repo_api.info().map_err(|e| {
            SdkError::network_src(
                format!("Failed to get HuggingFace repo info for '{}'", repo),
                e,
            )
        })?;

        let siblings = repo_info.siblings;
        if siblings.is_empty() {
            return Err(SdkError::load(format!(
                "HuggingFace repo '{}' has no files",
                repo
            )));
        }

        // Classify files by type to enable smart filtering
        let all_filenames: Vec<&str> = siblings.iter().map(|s| s.rfilename.as_str()).collect();
        let gguf_files: Vec<&str> = all_filenames
            .iter()
            .filter(|f| f.ends_with(".gguf"))
            .copied()
            .collect();

        // If multiple GGUF files exist, select the best one instead of downloading all
        let selected_gguf = if gguf_files.len() > 1 {
            Some(select_gguf_variant(&gguf_files, variant)?)
        } else {
            None
        };

        if let Some(ref selected) = selected_gguf {
            log::info!(
                target: "xybrid_sdk",
                "Selected GGUF variant: {} (from {} available)",
                selected, gguf_files.len()
            );
        }

        // Create cache directory
        std::fs::create_dir_all(&cache_dir)?;

        // Filter to only files we need
        let files_to_download: Vec<&str> = all_filenames
            .iter()
            .filter(|filename| {
                // Skip hidden files and directories
                if filename.starts_with('.') || filename.ends_with('/') {
                    return false;
                }

                // If we have a selected GGUF, skip other GGUF files
                if let Some(ref selected) = selected_gguf {
                    if filename.ends_with(".gguf") && **filename != *selected {
                        return false;
                    }
                }

                // Skip non-essential files (LICENSE, subdirectories like leap/)
                let dominated_by_model = selected_gguf.is_some() || gguf_files.len() == 1;
                if dominated_by_model {
                    Self::is_essential_file(filename)
                } else {
                    true
                }
            })
            .copied()
            .collect();

        let total_files = files_to_download.len();
        for (i, filename) in files_to_download.iter().enumerate() {
            log::debug!(target: "xybrid_sdk", "Downloading [{}/{}]: {}", i + 1, total_files, filename);

            // Report approximate progress
            _progress_callback((i as f32) / (total_files as f32));

            // Download file (hf-hub caches internally)
            let cached_path = repo_api.get(filename).map_err(|e| {
                SdkError::network_src(
                    format!("Failed to download '{}' from '{}'", filename, repo),
                    e,
                )
            })?;

            // Create target path in our cache directory
            let target_path = cache_dir.join(filename);

            // Create parent directories if the file is in a subdirectory
            if let Some(parent) = target_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // Skip if already exists in our cache
            if target_path.exists() {
                continue;
            }

            // Create symlink to hf-hub's cached file (avoids duplication)
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(&cached_path, &target_path).map_err(|e| {
                    SdkError::IoError(std::io::Error::other(format!(
                        "Failed to symlink {} -> {}: {}",
                        cached_path.display(),
                        target_path.display(),
                        e
                    )))
                })?;
            }

            // On Windows, copy the file instead
            #[cfg(not(unix))]
            {
                std::fs::copy(&cached_path, &target_path)?;
            }
        }

        // Report completion
        _progress_callback(1.0);

        // Auto-generate model_metadata.json if not provided by the repo
        if !metadata_path.exists() {
            log::info!(
                target: "xybrid_sdk",
                "No model_metadata.json in repo '{}', attempting auto-generation...",
                repo
            );
            match crate::metadata_gen::generate_metadata(&cache_dir, repo) {
                Ok((_metadata, _task_inference)) => {
                    log::info!(
                        target: "xybrid_sdk",
                        "Auto-generated model_metadata.json for '{}'. \
                         Review and adjust if inference results are unexpected.",
                        repo
                    );
                }
                Err(e) => {
                    return Err(SdkError::MetadataNotFound(format!(
                        "HuggingFace repo '{}' does not contain model_metadata.json and \
                         auto-generation failed: {}. \
                         Create one manually — see docs/sdk/MODEL_METADATA.md",
                        repo, e
                    )));
                }
            }
        }

        log::info!(target: "xybrid_sdk", "Model cached at: {}", cache_dir.display());
        self.load_from_directory(&cache_dir)
    }

    /// Not available without the `huggingface` feature.
    #[cfg(not(feature = "huggingface"))]
    fn load_from_huggingface<F>(
        &self,
        _repo: &str,
        _revision: Option<&str>,
        _variant: Option<&str>,
        _progress_callback: F,
    ) -> SdkResult<XybridModel>
    where
        F: Fn(f32),
    {
        Err(SdkError::ConfigError(
            "HuggingFace loading requires the 'huggingface' feature flag. \
             Enable it with: cargo build --features huggingface"
                .to_string(),
        ))
    }

    /// Get the cache directory for a HuggingFace repo.
    ///
    /// Returns `~/.xybrid/cache/hf/{sanitized_repo}/` or the SDK-configured cache path.
    fn hf_cache_dir(repo: &str) -> SdkResult<PathBuf> {
        let base_cache = if let Some(sdk_cache) = crate::get_sdk_cache_dir() {
            sdk_cache.join("hf")
        } else {
            let home = dirs::home_dir()
                .ok_or_else(|| SdkError::cache("Cannot determine home directory"))?;
            home.join(".xybrid").join("cache").join("hf")
        };

        // Sanitize repo name for filesystem (e.g., "xybrid-ai/kokoro-82m" -> "xybrid-ai--kokoro-82m")
        let sanitized = repo.replace('/', "--");
        Ok(base_cache.join(sanitized))
    }

    /// Check if a file is essential and should always be downloaded.
    ///
    /// Essential files are model files (.gguf, .onnx, .safetensors), metadata files
    /// (config.json, tokenizer, vocab), and the README (for model card parsing).
    /// Non-essential files (LICENSE, leap/, etc.) are skipped.
    fn is_essential_file(filename: &str) -> bool {
        // Model files
        if filename.ends_with(".gguf")
            || filename.ends_with(".onnx")
            || filename.ends_with(".safetensors")
        {
            return true;
        }

        // Metadata and supporting files
        let basename = filename.rsplit('/').next().unwrap_or(filename);
        matches!(
            basename,
            "model_metadata.json"
                | "config.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
                | "vocab.json"
                | "vocab.txt"
                | "tokens.txt"
                | "merges.txt"
                | "preprocessor_config.json"
                | "generation_config.json"
                | "README.md"
        )
    }

    fn load_from_directory(&self, path: &PathBuf) -> SdkResult<XybridModel> {
        let handle = Self::create_model_handle(path)?;

        let model_id = handle.metadata.model_id.clone();
        let version = handle.metadata.version.clone();
        let supports_streaming = Self::check_streaming_support(&handle.metadata);
        let output_type = Self::infer_output_type(&handle.metadata);

        Ok(XybridModel {
            handle: Arc::new(RwLock::new(handle)),
            model_id,
            version,
            output_type,
            supports_streaming,
            current_run: Arc::new(Mutex::new(None)),
        })
    }

    fn create_model_handle(model_dir: &PathBuf) -> SdkResult<ModelHandle> {
        // Load metadata
        let metadata_path = model_dir.join("model_metadata.json");
        let metadata_str = std::fs::read_to_string(&metadata_path)
            .map_err(|e| SdkError::load_src("Failed to read model_metadata.json", e))?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_str)
            .map_err(|e| SdkError::load_src("Failed to parse metadata", e))?;

        // Create executor with base path
        let executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap_or("."));

        Ok(ModelHandle {
            executor,
            metadata,
            model_dir: model_dir.clone(),
            loaded: true,
        })
    }

    fn is_llm_template(metadata: &ModelMetadata) -> bool {
        matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. })
    }

    fn check_streaming_support(metadata: &ModelMetadata) -> bool {
        if Self::is_llm_template(metadata) {
            return true;
        }

        // Check if this is an ASR model (supports streaming)
        // Look at metadata task or model type (metadata is HashMap<String, serde_json::Value>)
        if let Some(task) = metadata.metadata.get("task").and_then(|v| v.as_str()) {
            if task == "speech-recognition" || task == "asr" {
                return true;
            }
        }

        // Check execution template type
        match &metadata.execution_template {
            ExecutionTemplate::SafeTensors { architecture, .. } => {
                architecture.as_deref() == Some("whisper")
            }
            ExecutionTemplate::Onnx { .. } => {
                // Check if preprocessing includes AudioDecode (likely ASR)
                metadata.preprocessing.iter().any(|step| {
                    matches!(
                        step,
                        xybrid_core::execution_template::PreprocessingStep::AudioDecode { .. }
                    )
                })
            }
            _ => false,
        }
    }

    fn infer_output_type(metadata: &ModelMetadata) -> OutputType {
        if Self::is_llm_template(metadata) {
            return OutputType::Text;
        }

        // Check metadata hints (metadata is HashMap<String, serde_json::Value>)
        if let Some(task) = metadata.metadata.get("task").and_then(|v| v.as_str()) {
            match task {
                "speech-recognition" | "asr" | "transcription" => return OutputType::Text,
                "text-to-speech" | "tts" | "speech-synthesis" => return OutputType::Audio,
                "embedding" | "feature-extraction" => return OutputType::Embedding,
                _ => {}
            }
        }

        // Check postprocessing steps for hints
        for step in &metadata.postprocessing {
            match step {
                xybrid_core::execution_template::PostprocessingStep::CTCDecode { .. }
                | xybrid_core::execution_template::PostprocessingStep::WhisperDecode { .. } => {
                    return OutputType::Text
                }
                xybrid_core::execution_template::PostprocessingStep::TTSAudioEncode { .. } => {
                    return OutputType::Audio
                }
                _ => {}
            }
        }

        OutputType::Unknown
    }
}

/// Represents a loaded model ready for inference.
///
/// Created by `ModelLoader::load()`. Provides both batch and streaming inference.
///
/// # Example
///
/// ```no_run
/// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
/// # use xybrid_sdk::{ModelLoader, StreamConfig};
/// # use xybrid_sdk::ir::Envelope;
/// # let loader: ModelLoader = unimplemented!();
/// # let audio_envelope: Envelope = unimplemented!();
/// # let samples: Vec<f32> = vec![];
/// let model = loader.load()?;
///
/// // Batch inference
/// let result = model.run(&audio_envelope, None)?;
/// println!("Transcription: {}", result.unwrap_text());
///
/// // Streaming inference (if supported)
/// if model.supports_streaming() {
///     let stream = model.stream(StreamConfig::with_vad())?;
///     stream.feed(&samples)?;
///     let transcript = stream.flush()?;
/// }
///
/// // Cleanup
/// model.unload()?;
/// # Ok(())
/// # }
/// ```
pub struct XybridModel {
    handle: Arc<RwLock<ModelHandle>>,
    model_id: String,
    version: String,
    output_type: OutputType,
    supports_streaming: bool,
    /// In-flight cancellation token for the preemptive cancel-and-replace
    /// ("latest-frame-wins") streaming slot. Guarded by its **own** mutex,
    /// deliberately separate from `handle`'s write lock: a preempting start
    /// swaps the new token into this slot and cancels the old one *before*
    /// acquiring `handle.write()`, so the previous run halts at its next token
    /// and drops the write guard promptly instead of head-of-line blocking the
    /// new run. `current_run.lock()` is only ever held for the brief swap — it
    /// is never held across `handle.write()`, so the two locks cannot deadlock.
    /// Non-preempting callers (chat) never touch this slot.
    ///
    /// `Arc`-shared so all clones of a model (the FFI clones the model into
    /// each streaming worker thread) coordinate through one slot — preemption
    /// must see the previous concurrent run's token even though it ran on a
    /// different clone.
    current_run: Arc<Mutex<Option<CancellationToken>>>,
}

struct WarmupEventFields {
    model_id: String,
    version: String,
    output_type: OutputType,
}

fn cap_warmup_generation(mut warmup_input: Envelope) -> Envelope {
    warmup_input
        .metadata
        .insert("max_tokens".to_string(), "1".to_string());
    warmup_input
}

fn publish_model_warmup_event(
    fields: WarmupEventFields,
    latency_ms: u32,
    resource_guard: xybrid_core::device::RunGuard,
) {
    let event = crate::telemetry::TelemetryEvent {
        event_type: "ModelWarmup".to_string(),
        stage_name: Some(fields.model_id.clone()),
        target: Some("local".to_string()),
        latency_ms: Some(latency_ms),
        error: None,
        data: Some(
            serde_json::json!({
                "model_id": fields.model_id,
                "version": fields.version,
                "output_type": format!("{:?}", fields.output_type),
            })
            .to_string(),
        ),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    };
    crate::telemetry::publish_with_resource_summary(event, resource_guard);
}

impl XybridModel {
    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the model version.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Check if the model is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.handle.read().map(|h| h.loaded).unwrap_or(false)
    }

    /// Check if this model supports streaming.
    pub fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }

    /// Get the expected output type for this model.
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }

    /// Check if this is an LLM model (uses GGUF execution template).
    ///
    /// LLM models support multi-turn conversation contexts. Use this to
    /// determine if conversation history should be maintained.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::{ModelLoader, ConversationContext};
    /// # let loader: ModelLoader = unimplemented!();
    /// let model = loader.load()?;
    /// if model.is_llm() {
    ///     // Create conversation context for multi-turn chat
    ///     let mut ctx = ConversationContext::new();
    ///     // ... manage conversation history
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_llm(&self) -> bool {
        self.handle
            .read()
            .ok()
            .map(|h| {
                matches!(
                    h.metadata.execution_template,
                    ExecutionTemplate::Gguf { .. }
                )
            })
            .unwrap_or(false)
    }

    // =========================================================================
    // Voice Discovery (TTS models only)
    // =========================================================================

    /// Get the voice configuration for this model, if available.
    ///
    /// Returns `None` for non-TTS models or TTS models without voice configuration.
    pub fn voice_config(&self) -> Option<VoiceConfig> {
        self.handle
            .read()
            .ok()
            .and_then(|h| h.metadata.voices.clone())
    }

    /// Get all available voices for this TTS model.
    ///
    /// Returns `None` for non-TTS models or TTS models without voice configuration.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use xybrid_sdk::XybridModel;
    /// # let model: XybridModel = unimplemented!();
    /// if let Some(voices) = model.voices() {
    ///     for voice in voices {
    ///         println!("{}: {} ({})", voice.id, voice.name, voice.language.unwrap_or_default());
    ///     }
    /// }
    /// ```
    pub fn voices(&self) -> Option<Vec<VoiceInfo>> {
        self.voice_config().map(|vc| vc.catalog)
    }

    /// Get the default voice for this TTS model.
    ///
    /// Returns `None` for non-TTS models or if no default is configured.
    pub fn default_voice(&self) -> Option<VoiceInfo> {
        self.voice_config().and_then(|vc| {
            let default_id = &vc.default;
            vc.catalog.into_iter().find(|v| &v.id == default_id)
        })
    }

    /// Check if this model has voice configuration.
    ///
    /// Returns `true` for TTS models with voice support.
    pub fn has_voices(&self) -> bool {
        self.voice_config().is_some()
    }

    /// Get a specific voice by ID.
    ///
    /// Returns `None` if the voice is not found or the model has no voice support.
    ///
    /// # Arguments
    ///
    /// * `voice_id` - The voice identifier (e.g., "af_bella")
    pub fn voice(&self, voice_id: &str) -> Option<VoiceInfo> {
        self.voice_config()
            .and_then(|vc| vc.catalog.into_iter().find(|v| v.id == voice_id))
    }

    // =========================================================================
    // Warmup Methods (for pre-loading models)
    // =========================================================================

    /// Warm up the model by running a minimal inference.
    ///
    /// This pre-loads the model into memory, ensuring that the first real inference
    /// is fast. For LLM models, this loads the model weights and creates the context.
    ///
    /// Call this at app startup or after `load()` to eliminate cold-start latency.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::ModelLoader;
    /// # use xybrid_sdk::ir::Envelope;
    /// # let loader: ModelLoader = unimplemented!();
    /// # let envelope: Envelope = unimplemented!();
    /// let model = loader.load()?;
    /// model.warmup()?;  // Pre-load model
    ///
    /// // First inference is now fast
    /// let result = model.run(&envelope, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn warmup(&self) -> SdkResult<()> {
        use xybrid_core::ir::EnvelopeKind;

        log::info!(target: "xybrid_sdk", "Warming up model: {}", self.model_id);
        let is_llm = self.is_llm();

        // Create a minimal input based on expected input type
        let warmup_input = match self.output_type {
            // For TTS models, use a short text
            OutputType::Audio => Envelope {
                kind: EnvelopeKind::Text("Hi".to_string()),
                metadata: std::collections::HashMap::new(),
            },
            // For ASR models, use minimal audio (1 second of silence at 16kHz)
            OutputType::Text if self.supports_streaming && !is_llm => {
                // Create a minimal WAV file with silence
                let silence_samples = vec![0i16; 16000]; // 1 second at 16kHz
                let audio_bytes = Self::create_wav_bytes(&silence_samples, 16000);
                Envelope {
                    kind: EnvelopeKind::Audio(audio_bytes),
                    metadata: std::collections::HashMap::new(),
                }
            }
            // For LLM/text models, use a short prompt
            OutputType::Text | OutputType::Embedding | OutputType::Unknown => Envelope {
                kind: EnvelopeKind::Text("Hi".to_string()),
                metadata: std::collections::HashMap::new(),
            },
        };

        // Warmup measures model-load + first-token latency, not full
        // generation. Cap LLM decoding at 1 token so a 2048-token
        // `GenerationConfig::default()` doesn't turn warmup into a real
        // inference. `executor::execute_llm` reads this from envelope
        // metadata when no explicit `GenerationConfig` is passed;
        // non-LLM paths ignore it.
        let warmup_input = cap_warmup_generation(warmup_input);

        // Run the inference inline (rather than delegating to `self.run`)
        // so the publish at the end is a `ModelWarmup` event rather than
        // a `ModelComplete`. Warmups should be visible to billing /
        // perf-debugging but distinguishable from real inferences on
        // the Traces dashboard — `ModelWarmup` carries the same
        // attribution fields (`stage_name`, `target`, `latency_ms`) but
        // its own event_type so the platform can render with a `warmup`
        // badge and default-filter it out of cost-attribution rollups.
        let start = Instant::now();
        let resource_guard = crate::telemetry::begin_resource_run();
        let trace_id = uuid::Uuid::new_v4();
        let _telemetry_ctx =
            crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

        let event_fields = {
            let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());
            if !handle.loaded {
                return Err(SdkError::NotLoaded);
            }
            let metadata = handle.metadata.clone();
            handle
                .executor
                .execute(&metadata, &warmup_input, None)
                .map_err(|e| SdkError::inference_src("Warmup execution failed", e))?;

            WarmupEventFields {
                model_id: self.model_id.clone(),
                version: metadata.version,
                output_type: self.output_type,
            }
        };

        let latency_ms = start.elapsed().as_millis() as u32;
        publish_model_warmup_event(event_fields, latency_ms, resource_guard);

        log::info!(
            target: "xybrid_sdk",
            "Model {} warmed up in {}ms",
            self.model_id,
            latency_ms
        );

        Ok(())
    }

    /// Warm up the model asynchronously.
    ///
    /// This is useful for background pre-loading at app startup without blocking the UI.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::ModelLoader;
    /// # let loader: ModelLoader = unimplemented!();
    /// let model = loader.load()?;
    ///
    /// // Start warmup in background
    /// let warmup_handle = tokio::spawn(async move {
    ///     model.warmup_async().await
    /// });
    ///
    /// // Do other initialization...
    ///
    /// // Wait for warmup if needed
    /// warmup_handle.await??;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn warmup_async(&self) -> SdkResult<()> {
        let handle = self.handle.clone();
        let model_id = self.model_id.clone();
        let output_type = self.output_type;
        let supports_streaming = self.supports_streaming;
        let is_llm = self.is_llm();

        tokio::task::spawn_blocking(move || {
            use xybrid_core::ir::EnvelopeKind;

            log::info!(target: "xybrid_sdk", "Warming up model (async): {}", model_id);

            // Create a minimal input based on expected input type
            let warmup_input = match output_type {
                OutputType::Audio => Envelope {
                    kind: EnvelopeKind::Text("Hi".to_string()),
                    metadata: std::collections::HashMap::new(),
                },
                OutputType::Text if supports_streaming && !is_llm => {
                    let silence_samples = vec![0i16; 16000];
                    let audio_bytes = Self::create_wav_bytes(&silence_samples, 16000);
                    Envelope {
                        kind: EnvelopeKind::Audio(audio_bytes),
                        metadata: std::collections::HashMap::new(),
                    }
                }
                OutputType::Text | OutputType::Embedding | OutputType::Unknown => Envelope {
                    kind: EnvelopeKind::Text("Hi".to_string()),
                    metadata: std::collections::HashMap::new(),
                },
            };

            // See sync `warmup()` for rationale — cap LLM decoding at
            // 1 token so warmup doesn't run a full generation.
            let warmup_input = cap_warmup_generation(warmup_input);

            let start = Instant::now();
            let resource_guard = crate::telemetry::begin_resource_run();
            let trace_id = uuid::Uuid::new_v4();
            let _telemetry_ctx =
                crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

            // Run inference inline and publish a `ModelWarmup` event —
            // same shape as the sync `warmup()` above. Previously this
            // path published nothing at all, so async warmups were
            // silent on the wire (visible only via logs).
            let event_fields = {
                let mut guard = handle.write().unwrap_or_else(|e| e.into_inner());
                if !guard.loaded {
                    return Err(SdkError::NotLoaded);
                }

                let metadata = guard.metadata.clone();
                guard
                    .executor
                    .execute(&metadata, &warmup_input, None)
                    .map_err(|e| SdkError::inference_src("Warmup failed", e))?;

                WarmupEventFields {
                    model_id: model_id.clone(),
                    version: metadata.version,
                    output_type,
                }
            };

            let latency_ms = start.elapsed().as_millis() as u32;
            publish_model_warmup_event(event_fields, latency_ms, resource_guard);

            log::info!(
                target: "xybrid_sdk",
                "Model {} warmed up (async) in {}ms",
                model_id,
                latency_ms
            );

            Ok(())
        })
        .await
        .map_err(|e| SdkError::inference_src("Task join error", e))?
    }

    /// Create a minimal WAV file bytes from samples for warmup.
    fn create_wav_bytes(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        let num_samples = samples.len();
        let data_size = (num_samples * 2) as u32;
        let file_size = 36 + data_size;

        // RIFF header
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&file_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        // fmt chunk
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes()); // Chunk size
        bytes.extend_from_slice(&1u16.to_le_bytes()); // Audio format (PCM)
        bytes.extend_from_slice(&1u16.to_le_bytes()); // Num channels
        bytes.extend_from_slice(&sample_rate.to_le_bytes()); // Sample rate
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // Byte rate
        bytes.extend_from_slice(&2u16.to_le_bytes()); // Block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample

        // data chunk
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        bytes
    }

    /// Run batch inference with an Envelope.
    ///
    /// # Arguments
    ///
    /// * `envelope` - Input data wrapped in an Envelope
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the output with convenient accessors.
    pub fn run(
        &self,
        envelope: &Envelope,
        config: Option<&GenerationConfig>,
    ) -> SdkResult<InferenceResult> {
        crate::telemetry::maybe_emit_dev_nudge();
        let start = Instant::now();
        // Begin a resource-telemetry scope for this run. When
        // `resource_telemetry_mode()` is `Off` the guard is a no-op; otherwise
        // it captures start snapshots (and launches a sampler for Summary /
        // DebugLocal). Summary is produced by
        // `publish_with_resource_summary` at the end of this function.
        let resource_guard = crate::telemetry::begin_resource_run();

        // Install a per-call trace_id for the lifetime of this run. Any
        // telemetry events emitted between install and `Drop` (including
        // any deferred adapter events) share the same `trace_id`, so the
        // dashboard collapses them to a single Traces row. Same
        // discipline `run_with_context` uses; see `docs/sdk/trace-model.md`.
        let trace_id = uuid::Uuid::new_v4();
        let _telemetry_ctx =
            crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

        // Recover from poisoned RwLock to prevent permanent lock errors
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to avoid borrow conflict with executor
        let metadata = handle.metadata.clone();
        let output = handle
            .executor
            .execute(&metadata, envelope, config)
            .map_err(|e| sdk_execution_error("Execution failed", e))?;

        let latency_ms = start.elapsed().as_millis() as u32;

        // Emit ModelComplete telemetry event
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(
                serde_json::json!({
                    "model_id": self.model_id,
                    "version": self.version,
                    "output_type": format!("{:?}", self.output_type),
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_with_resource_summary(event, resource_guard);

        // `_telemetry_ctx` drops here, clearing the pipeline context after
        // the publish — same ordering as `run_with_context`.

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Streaming TTS: synthesize `envelope`'s text sentence-chunk by
    /// sentence-chunk and hand each chunk's PCM (with its sample rate) to
    /// `on_chunk` as it is produced, instead of returning one batched WAV. For
    /// long text this lets playback start after the first sentence.
    ///
    /// Audio rides the callback; there is no batched return value. `on_chunk`
    /// returning `false` stops early, as does a cancelled
    /// `options.cancellation_token` — both honored at the next chunk boundary
    /// (one chunk's ONNX forward is uninterruptible). The model write-lock is
    /// held for the whole synthesis, exactly like [`run`].
    pub fn run_tts_streaming<F>(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
        mut on_chunk: F,
    ) -> SdkResult<()>
    where
        F: FnMut(Vec<u8>, u32) -> bool,
    {
        crate::telemetry::maybe_emit_dev_nudge();
        let start = Instant::now();
        let resource_guard = crate::telemetry::begin_resource_run();
        let trace_id = uuid::Uuid::new_v4();
        let _telemetry_ctx =
            crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());
        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }
        let metadata = handle.metadata.clone();

        // Between-chunk cancellation: a chunk's ONNX forward can't be aborted
        // mid-way, so the token (and the caller's `on_chunk`) is consulted at
        // chunk boundaries.
        let cancel = options.cancellation_token.clone();
        let mut adapter = |pcm: Vec<u8>, sample_rate: u32| -> bool {
            if let Some(token) = &cancel {
                if token.is_cancelled() {
                    return false;
                }
            }
            on_chunk(pcm, sample_rate)
        };

        handle
            .executor
            .execute_tts_streaming(&metadata, envelope, &mut adapter)
            .map_err(|e| sdk_execution_error("TTS streaming failed", e))?;

        let latency_ms = start.elapsed().as_millis() as u32;
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(
                serde_json::json!({
                    "model_id": self.model_id,
                    "version": self.version,
                    "output_type": format!("{:?}", self.output_type),
                    "streaming": true,
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_with_resource_summary(event, resource_guard);

        Ok(())
    }

    /// Run batch inference with per-run controls.
    pub fn run_with_options(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
    ) -> SdkResult<InferenceResult> {
        let mut abort_state = AbortState::new(options);
        abort_state
            .check_before_run()
            .map_err(|reason| SdkError::inference(format!("Execution aborted: {reason}")))?;
        self.run(envelope, options.generation_config.as_ref())
    }

    /// Run inference with conversation context.
    ///
    /// This method passes the conversation history to the model, allowing it to
    /// generate context-aware responses. The model uses its chat template to
    /// format the conversation history into a prompt.
    ///
    /// **Important:** This method does not mutate the context. The caller is
    /// responsible for pushing the result to the context if desired.
    ///
    /// # Arguments
    ///
    /// * `envelope` - The current user input (should have `MessageRole::User`)
    /// * `context` - Conversation history (system prompt + previous turns)
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the assistant's response (tagged with `MessageRole::Assistant`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// use xybrid_sdk::{ModelLoader, ConversationContext};
    /// use xybrid_sdk::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let model = ModelLoader::from_registry("gemma-3-1b").load()?;
    /// let mut ctx = ConversationContext::new();
    ///
    /// // Add user message to context
    /// let user_input = Envelope::new(EnvelopeKind::Text("Hello!".into()))
    ///     .with_role(MessageRole::User);
    /// ctx.push(user_input.clone());
    ///
    /// // Run with context (model sees the full history)
    /// let result = model.run_with_context(&user_input, &ctx, None)?;
    ///
    /// // Add assistant response to context
    /// ctx.push(result.envelope().clone());
    ///
    /// println!("{}", result.text().unwrap_or_default());
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_with_context(
        &self,
        envelope: &Envelope,
        context: &ConversationContext,
        config: Option<&GenerationConfig>,
    ) -> SdkResult<InferenceResult> {
        let start = Instant::now();
        let resource_guard = crate::telemetry::begin_resource_run();

        // Install a turn-scoped trace_id for the lifetime of this call.
        // The guard's `Drop` clears the pipeline context on every exit —
        // including the `?` error paths and panics below — so every
        // telemetry event emitted between install and drop shares the
        // same `trace_id` and the dashboard collapses the turn to one
        // Traces row. Same discipline `Pipeline::run` uses for stages.
        let trace_id = uuid::Uuid::new_v4();
        let _telemetry_ctx =
            crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

        // Recover from poisoned RwLock to prevent permanent lock errors
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to avoid borrow conflict with executor
        let metadata = handle.metadata.clone();
        let output = handle
            .executor
            .execute_with_context(&metadata, envelope, context, config)
            .map_err(|e| sdk_execution_error("Execution failed", e))?;

        let latency_ms = start.elapsed().as_millis() as u32;

        // Emit ModelComplete telemetry event
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(
                serde_json::json!({
                    "model_id": self.model_id,
                    "version": self.version,
                    "output_type": format!("{:?}", self.output_type),
                    "context_messages": context.history().len(),
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_with_resource_summary(event, resource_guard);

        // `_telemetry_ctx` drops here (and on every early return / unwind
        // above), clearing the pipeline context after the publish.

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Run inference with conversation context and per-run controls.
    pub fn run_with_context_options(
        &self,
        envelope: &Envelope,
        context: &ConversationContext,
        options: &RunOptions,
    ) -> SdkResult<InferenceResult> {
        let mut abort_state = AbortState::new(options);
        abort_state
            .check_before_run()
            .map_err(|reason| SdkError::inference(format!("Execution aborted: {reason}")))?;
        self.run_with_context(envelope, context, options.generation_config.as_ref())
    }

    /// Run streaming inference with conversation context.
    ///
    /// Combines streaming output with multi-turn conversation memory.
    /// The model sees the full conversation history when generating responses.
    ///
    /// # Arguments
    ///
    /// * `envelope` - Current user input wrapped in an Envelope
    /// * `context` - Conversation history for multi-turn chat
    /// * `on_token` - Callback invoked for each token (LLM) or once (other models)
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the final output.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::{XybridModel, ConversationContext};
    /// # use xybrid_sdk::ir::{Envelope, EnvelopeKind, MessageRole};
    /// # let model: XybridModel = unimplemented!();
    /// let mut ctx = ConversationContext::new();
    ///
    /// // Add user message and run with streaming
    /// let input = Envelope::new(EnvelopeKind::Text("Tell me a joke".into()))
    ///     .with_role(MessageRole::User);
    /// ctx.push(input.clone());
    ///
    /// let result = model.run_streaming_with_context(&input, &ctx, None, |token| {
    ///     print!("{}", token.token);
    ///     std::io::Write::flush(&mut std::io::stdout())?;
    ///     Ok(())
    /// })?;
    ///
    /// // Add assistant response to context
    /// ctx.push(result.envelope().clone());
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_streaming_with_context<F>(
        &self,
        envelope: &Envelope,
        context: &ConversationContext,
        config: Option<&GenerationConfig>,
        on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        // No live-capture tag on the bare context streaming path.
        self.run_streaming_with_context_tagged(envelope, context, config, None, on_token)
    }

    /// Internal context-streaming entry point that optionally stamps a
    /// live-capture telemetry tag onto the emitted `ModelComplete` event. See
    /// [`Self::run_streaming_tagged`] for the rationale; this is the
    /// conversation-context variant.
    fn run_streaming_with_context_tagged<F>(
        &self,
        envelope: &Envelope,
        context: &ConversationContext,
        config: Option<&GenerationConfig>,
        live_tag: Option<&LiveModeTag>,
        mut on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        use xybrid_core::execution::ExecutionTemplate;
        use xybrid_core::runtime_adapter::types::PartialToken;

        let start = Instant::now();

        // RAII pipeline context — see `run_with_context` for rationale.
        // Cleared on drop at end of scope (after publish) or on any
        // early return / unwind below.
        let trace_id = uuid::Uuid::new_v4();
        let _telemetry_ctx =
            crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

        // Get write lock on handle
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to check execution template
        let metadata = handle.metadata.clone();

        // Check if this is an LLM model (GGUF template)
        let is_llm = matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. });

        let output = if is_llm {
            // True streaming with context for LLM models
            handle
                .executor
                .execute_streaming_with_context(
                    &metadata,
                    envelope,
                    context,
                    Box::new(&mut on_token),
                    config,
                )
                .map_err(streaming_execution_error)?
        } else {
            // For non-LLM models: run with context and emit single "token" with full result
            let result = handle
                .executor
                .execute_with_context(&metadata, envelope, context, config)
                .map_err(|e| sdk_execution_error("Execution failed", e))?;

            // Extract text from result (if any) and emit as single token
            if let xybrid_core::ir::EnvelopeKind::Text(text) = &result.kind {
                let token = PartialToken {
                    token: text.clone(),
                    token_id: None,
                    index: 0,
                    cumulative_text: text.clone(),
                    finish_reason: Some("stop".to_string()),
                };
                on_token(token).map_err(streaming_callback_error)?;
            }

            result
        };

        let latency_ms = start.elapsed().as_millis() as u32;

        // Emit telemetry event
        let mut data = serde_json::json!({
            "model_id": self.model_id,
            "version": self.version,
            "output_type": format!("{:?}", self.output_type),
            "streaming": true,
            "context_messages": context.history().len(),
        });
        stamp_live_mode_tag(&mut data, live_tag);
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(data.to_string()),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_telemetry_event(event);

        // `_telemetry_ctx` drops here, clearing pipeline context after
        // the publish — same ordering as before.

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Run streaming inference with conversation context and per-token abort checks.
    pub fn run_streaming_with_context_options<F>(
        &self,
        envelope: &Envelope,
        context: &ConversationContext,
        options: &RunOptions,
        mut on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        let mut abort_state = AbortState::new(options);
        // Honour pre-run cancellation so a `token.cancel()` issued before
        // invocation aborts immediately rather than running the full batch
        // (or model load + prompt fill on streaming models) before the first
        // token boundary checks the abort state.
        let fallback_to_cloud = options.abort_policy.fallback_to_cloud;
        abort_state
            .check_before_run()
            .map_err(|reason| streaming_pre_run_abort_error(reason, fallback_to_cloud))?;
        let live_tag = options.live_mode_tag();
        self.run_streaming_with_context_tagged(
            envelope,
            context,
            options.generation_config.as_ref(),
            live_tag.as_ref(),
            move |token| {
                if let Err(reason) = abort_state.check_before_token() {
                    return Err(reason.into_streaming_error(fallback_to_cloud));
                }
                on_token(token)
            },
        )
    }

    /// Run inference with streaming output.
    ///
    /// This method provides a unified streaming interface for all model types:
    /// - **LLM models (GGUF)**: True token-by-token streaming via the callback
    /// - **Other models (TTS, ASR, etc.)**: Single callback with the full result
    ///
    /// This "everything is a stream" pattern allows consumers to use the same
    /// API regardless of model type, while LLMs get the latency benefits of
    /// true streaming.
    ///
    /// # Arguments
    ///
    /// * `envelope` - Input data wrapped in an Envelope
    /// * `on_token` - Callback invoked for each token (LLM) or once (other models)
    ///
    /// # Returns
    ///
    /// `InferenceResult` containing the final output.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::XybridModel;
    /// # use xybrid_sdk::ir::Envelope;
    /// # let model: XybridModel = unimplemented!();
    /// # let envelope: Envelope = unimplemented!();
    /// // Works for both LLM and non-LLM models
    /// let result = model.run_streaming(&envelope, None, |token| {
    ///     print!("{}", token.token);
    ///     std::io::Write::flush(&mut std::io::stdout())?;
    ///     Ok(())
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_streaming<F>(
        &self,
        envelope: &Envelope,
        config: Option<&GenerationConfig>,
        on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        // No live-capture tag on the bare `run_streaming` path — telemetry is
        // byte-for-byte the pre-live-mode shape.
        self.run_streaming_tagged(envelope, config, None, on_token)
    }

    /// Internal streaming entry point that optionally stamps a live-capture
    /// telemetry tag onto the emitted `ModelComplete` event.
    ///
    /// `run_streaming` delegates here with `live_tag = None` (unchanged wire
    /// payload). The options-aware streaming entry points pass
    /// `RunOptions::live_mode_tag()` so live-capture sessions carry
    /// `live_mode` + `frame_session_id` on the wire, which the telemetry
    /// dispatch funnel then rate-limits per session.
    fn run_streaming_tagged<F>(
        &self,
        envelope: &Envelope,
        config: Option<&GenerationConfig>,
        live_tag: Option<&LiveModeTag>,
        mut on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        use xybrid_core::execution::ExecutionTemplate;
        use xybrid_core::runtime_adapter::types::PartialToken;

        let start = Instant::now();

        // Per-call trace_id scope — see `run` for rationale. Cleared on
        // drop at end of scope (after publish) or on any early return.
        let trace_id = uuid::Uuid::new_v4();
        let _telemetry_ctx =
            crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

        // Get write lock on handle
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Clone metadata to check execution template
        let metadata = handle.metadata.clone();

        // Check if this is an LLM model (GGUF template)
        let is_llm = matches!(metadata.execution_template, ExecutionTemplate::Gguf { .. });

        let output = if is_llm {
            // True streaming for LLM models
            handle
                .executor
                .execute_streaming(&metadata, envelope, Box::new(&mut on_token), config)
                .map_err(streaming_execution_error)?
        } else {
            // For non-LLM models: run batch and emit single "token" with full result
            let result = handle
                .executor
                .execute(&metadata, envelope, config)
                .map_err(|e| sdk_execution_error("Execution failed", e))?;

            // Extract text from result (if any) and emit as single token
            if let xybrid_core::ir::EnvelopeKind::Text(text) = &result.kind {
                let token = PartialToken {
                    token: text.clone(),
                    token_id: None,
                    index: 0,
                    cumulative_text: text.clone(),
                    finish_reason: Some("stop".to_string()),
                };
                on_token(token).map_err(streaming_callback_error)?;
            }

            result
        };

        let latency_ms = start.elapsed().as_millis() as u32;

        // Emit telemetry event
        let mut data = serde_json::json!({
            "model_id": self.model_id,
            "version": self.version,
            "output_type": format!("{:?}", self.output_type),
            "streaming": true,
        });
        stamp_live_mode_tag(&mut data, live_tag);
        let event = crate::telemetry::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some(self.model_id.clone()),
            target: Some("local".to_string()),
            latency_ms: Some(latency_ms),
            error: None,
            data: Some(data.to_string()),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        crate::telemetry::publish_telemetry_event(event);

        Ok(InferenceResult::new(output, &self.model_id, latency_ms))
    }

    /// Run streaming inference with per-token abort checks.
    pub fn run_streaming_with_options<F>(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
        mut on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        // Capture supports_streaming before the closure takes ownership.
        // The check_abort_for_streaming helper short-circuits when this is
        // false so non-streaming models (batch executor + single synthetic
        // token at the end) don't trigger cloud fallback after local
        // inference already succeeded — see helper docs for the full
        // privacy/cost rationale.
        let supports_streaming = self.supports_streaming;
        let mut abort_state = AbortState::new(options);
        // Honour pre-run cancellation: without this, a non-streaming model
        // whose `supports_streaming = false` would silently execute the full
        // batch even when the cancellation token was set before invocation,
        // because `check_abort_for_streaming` short-circuits in that case.
        let fallback_to_cloud = options.abort_policy.fallback_to_cloud;
        abort_state
            .check_before_run()
            .map_err(|reason| streaming_pre_run_abort_error(reason, fallback_to_cloud))?;
        let live_tag = options.live_mode_tag();
        self.run_streaming_tagged(
            envelope,
            options.generation_config.as_ref(),
            live_tag.as_ref(),
            move |token| {
                check_abort_for_streaming(supports_streaming, &mut abort_state, fallback_to_cloud)?;
                on_token(token)
            },
        )
    }

    /// Swap `token` into the preemptive cancel-and-replace slot and cancel the
    /// run it replaced. Returns the token that was previously in flight (if
    /// any), already cancelled.
    ///
    /// **Locking contract (race-free, codex-reviewed):** holds `current_run`'s
    /// mutex *only* for the brief swap, then releases it before issuing the
    /// `cancel()` on the displaced token. This ordering is load-bearing:
    /// 1. `current_run.lock()` is never held while acquiring `handle.write()`,
    ///    so the two locks can never deadlock and a third concurrent start
    ///    cannot wedge waiting on the swap.
    /// 2. The displaced run is cancelled *before* the caller goes on to
    ///    acquire `handle.write()`, so it halts at its next token boundary and
    ///    drops the write guard promptly — the new run acquires the lock
    ///    without waiting for the old run's natural completion (latest-frame-
    ///    wins instead of head-of-line blocking).
    fn preempt_register(&self, token: CancellationToken) -> Option<CancellationToken> {
        let old = {
            let mut slot = self.current_run.lock().unwrap_or_else(|e| e.into_inner());
            slot.replace(token)
        };
        if let Some(ref old) = old {
            old.cancel();
        }
        old
    }

    /// Clear the in-flight slot **iff it still holds `token`**.
    ///
    /// A newer preempting start may have already replaced the slot with its own
    /// token while this run was finishing; clearing unconditionally would
    /// clobber that newer run's registration and let a stale frame escape
    /// cancellation. The Arc-identity check ([`CancellationToken::same_token`])
    /// makes the clear a no-op unless the slot is still ours.
    fn clear_current_run(&self, token: &CancellationToken) {
        let mut slot = self.current_run.lock().unwrap_or_else(|e| e.into_inner());
        if slot.as_ref().is_some_and(|t| t.same_token(token)) {
            *slot = None;
        }
    }

    /// Preemptive ("latest-frame-wins") variant of
    /// [`Self::run_streaming_with_options`].
    ///
    /// When `preempt` is `true` **and** `options` carries a cancellation token,
    /// this registers the token in the model's in-flight slot and cancels the
    /// previously-registered run *before* acquiring the model write lock. The
    /// displaced run halts at its next token and releases the lock, so this
    /// call starts promptly instead of head-of-line blocking behind it. On
    /// completion the slot is cleared if it still holds this run's token.
    ///
    /// When `preempt` is `false` (the default for chat and every existing
    /// caller), this delegates straight to `run_streaming_with_options` and
    /// never touches the slot — behavior is byte-for-byte identical to the
    /// non-preempt path.
    pub fn run_streaming_with_options_preempt<F>(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
        preempt: bool,
        on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        // Only engage the slot when preemption is requested AND a token is
        // present to register (the token is what a later preempt cancels). A
        // preempt with no token has nothing to register, so we run plain.
        //
        // NOTE (known limitation): when `preempt` is true but `options` carries
        // no cancellation token, the previous in-flight run is NOT cancelled —
        // there is nothing to register in the slot, hence nothing for a later
        // run to clear or cancel, and this run cannot itself be preempted.
        // Cancel-and-replace requires a token. Callers that want latest-frame-
        // wins must pass a fresh per-call token (the FFI/Dart streaming paths
        // and the live-capture loop already create one per call/frame).
        let registered = preempt
            .then(|| options.cancellation_token.clone())
            .flatten();
        if let Some(ref token) = registered {
            // Steps 1-2: swap in + cancel the displaced run before the lock.
            self.preempt_register(token.clone());
        }
        let result = self.run_streaming_with_options(envelope, options, on_token);
        if let Some(ref token) = registered {
            // Step 4: clear the slot iff it is still ours (a newer preempt may
            // have replaced it — don't clobber that).
            self.clear_current_run(token);
        }
        result
    }

    /// Preemptive ("latest-frame-wins") variant of
    /// [`Self::run_streaming_with_context_options`]. See
    /// [`Self::run_streaming_with_options_preempt`] for the slot semantics and
    /// locking contract; the only difference is the conversation context is
    /// threaded through to the underlying run.
    pub fn run_streaming_with_context_options_preempt<F>(
        &self,
        envelope: &Envelope,
        context: &ConversationContext,
        options: &RunOptions,
        preempt: bool,
        on_token: F,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
    {
        let registered = preempt
            .then(|| options.cancellation_token.clone())
            .flatten();
        if let Some(ref token) = registered {
            self.preempt_register(token.clone());
        }
        let result = self.run_streaming_with_context_options(envelope, context, options, on_token);
        if let Some(ref token) = registered {
            self.clear_current_run(token);
        }
        result
    }

    /// Run streaming inference with automatic cloud fallback on resource-driven
    /// abort.
    ///
    /// On a healthy run this behaves identically to
    /// [`Self::run_streaming_with_options`] — `on_token` fires once per local
    /// token, `on_seam` is never invoked, and the returned [`InferenceResult`]
    /// reflects the local execution.
    ///
    /// When the configured [`AbortPolicy`](crate::run_options::AbortPolicy)
    /// trips mid-stream **and** the policy permits cloud fallback, the wrapper:
    /// 1. Captures the local token count and elapsed latency.
    /// 2. Fires `on_seam(SeamInfo { … })` with the abort reason and a
    ///    shared `correlation_id`. Callers use this to render a UX cue.
    /// 3. Records the local abort outcome and re-checks policy before cloud.
    ///    If policy denies cloud, the wrapper emits `cloud_denied_by_policy`
    ///    telemetry and returns an explicit error without retrying.
    /// 4. Builds a fresh cloud envelope from the **original prompt** (no
    ///    partial-token reuse) and calls
    ///    [`CloudStreaming::execute_streaming`](xybrid_core::runtime_adapter::CloudStreaming::execute_streaming)
    ///    on `cloud_adapter`.
    /// 5. Routes cloud chunks through the same `on_token` so the user sees
    ///    one continuous token stream.
    /// 6. Records the cloud outcome and returns the cloud-leg
    ///    [`InferenceResult`].
    ///
    /// On any other error (no abort policy fired, or `fallback_to_cloud` is
    /// false), the original [`SdkError`] is returned unchanged.
    ///
    /// # Notes
    ///
    /// - The same `on_token` is invoked for both legs; the seam is observable
    ///   via `on_seam`, not via the token stream itself.
    /// - `correlation_id` is taken from `options.correlation_id` if set,
    ///   otherwise generated via `uuid::Uuid::new_v4()`.
    /// - The `envelope` must carry cloud-side routing metadata (`provider`,
    ///   `model`, `system_prompt`, `temperature`, …) for the retry leg. See
    ///   [`CloudRuntimeAdapter`](xybrid_core::runtime_adapter::CloudRuntimeAdapter)
    ///   for supported keys.
    /// - The wrapper is fully synchronous; the default `CloudRuntimeAdapter`
    ///   consumes OpenAI-compatible gateway SSE via `CloudStreaming`.
    /// - **Cancellation timing across the seam.** A cancel set on the
    ///   `cancellation_token` *before* `execute_streaming` is invoked
    ///   (including from inside `on_seam`) short-circuits before the cloud
    ///   adapter is called — the prompt is never sent. A cancel that arrives
    ///   while the cloud leg is waiting on SSE is observed at the next gateway
    ///   chunk: the user-visible token stream is suppressed, but the prompt may
    ///   already have been transmitted and billed. Treat the token as
    ///   "responsive on the seam, best-effort during cloud."
    pub fn run_streaming_with_fallback<F, S>(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
        cloud_adapter: &dyn xybrid_core::runtime_adapter::CloudStreaming,
        on_token: &mut F,
        on_seam: &mut S,
    ) -> SdkResult<InferenceResult>
    where
        F: FnMut(
                xybrid_core::runtime_adapter::types::PartialToken,
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send,
        S: FnMut(SeamInfo) + Send,
    {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let correlation_id = options
            .correlation_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let local_tokens = Arc::new(AtomicU32::new(0));
        let local_start = Instant::now();
        let local_resource_guard = crate::telemetry::begin_resource_run();

        let local_result = {
            let local_tokens = local_tokens.clone();
            self.run_streaming_with_options(envelope, options, |token| {
                local_tokens.fetch_add(1, Ordering::SeqCst);
                on_token(token)
            })
        };

        let local_latency_ms = local_start.elapsed().as_millis() as u32;
        let local_resource_summary = local_resource_guard.finish();
        let policy_metrics = fallback_policy_metrics(options);
        let signal_context = Some(SignalContext::from_metrics(&policy_metrics));

        dispatch_after_local(
            local_result,
            envelope,
            cloud_adapter,
            correlation_id,
            &self.model_id,
            local_tokens.load(Ordering::SeqCst),
            local_latency_ms,
            local_resource_summary,
            fallback_authority(),
            policy_metrics,
            signal_context,
            options.cancellation_token.clone(),
            on_token,
            on_seam,
        )
    }

    /// Run inference returning a stream of events.
    ///
    /// This is the idiomatic Rust streaming API that returns a `Stream` instead of
    /// using callbacks. Events are emitted as they occur:
    /// - `StreamEvent::Token` - for each generated token (LLM models)
    /// - `StreamEvent::Complete` - when inference finishes successfully
    /// - `StreamEvent::Error` - if an error occurs
    ///
    /// For non-LLM models, a single `Token` event is emitted with the full result,
    /// followed by `Complete`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn _example() {
    /// # use xybrid_sdk::{XybridModel, StreamEvent};
    /// # use xybrid_sdk::ir::Envelope;
    /// # let model: XybridModel = unimplemented!();
    /// # let envelope: Envelope = unimplemented!();
    /// use tokio_stream::StreamExt;
    ///
    /// let mut stream = model.run_stream(envelope, None);
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         StreamEvent::Token(token) => print!("{}", token.token),
    ///         StreamEvent::Complete(result) => println!("\nDone: {}ms", result.latency_ms()),
    ///         StreamEvent::Error(e) => eprintln!("Error: {}", e),
    ///     }
    /// }
    /// # }
    /// ```
    pub fn run_stream(
        &self,
        envelope: Envelope,
        config: Option<GenerationConfig>,
    ) -> Pin<Box<dyn tokio_stream::Stream<Item = StreamEvent> + Send + '_>> {
        use tokio::sync::mpsc;
        use xybrid_core::runtime_adapter::types::PartialToken;

        let (tx, rx) = mpsc::channel::<StreamEvent>(100);
        let handle = self.handle.clone();
        let model_id = self.model_id.clone();
        let version = self.version.clone();
        let output_type = self.output_type;

        // Clone tx for the completion event (before moving into spawn_blocking)
        let tx_completion = tx.clone();

        // Spawn blocking task to run inference
        tokio::task::spawn(async move {
            let result = tokio::task::spawn_blocking(move || {
                let start = Instant::now();

                // Per-call trace_id scope — see `run` for rationale. Lives
                // inside the spawn_blocking closure so the install + drop
                // happen on the same thread the publish runs on.
                let trace_id = uuid::Uuid::new_v4();
                let _telemetry_ctx =
                    crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

                // Get write lock on handle
                let mut guard = handle.write().unwrap_or_else(|e| e.into_inner());

                if !guard.loaded {
                    return Err(SdkError::NotLoaded);
                }

                let metadata = guard.metadata.clone();
                let is_llm = matches!(
                    metadata.execution_template,
                    xybrid_core::execution::ExecutionTemplate::Gguf { .. }
                );

                // Clone tx for the streaming callback (so we can use tx in the else branch)
                let tx_for_callback = tx.clone();

                let output = if is_llm {
                    // True streaming for LLM models
                    guard
                        .executor
                        .execute_streaming(
                            &metadata,
                            &envelope,
                            Box::new(move |token: PartialToken| {
                                let stream_token = StreamToken {
                                    token: token.token.clone(),
                                    token_id: token.token_id.map(|id| id),
                                    index: token.index,
                                    cumulative_text: token.cumulative_text.clone(),
                                    finish_reason: token.finish_reason.clone(),
                                };
                                // Ignore send errors (receiver dropped)
                                let _ =
                                    tx_for_callback.blocking_send(StreamEvent::Token(stream_token));
                                Ok(())
                            }),
                            config.as_ref(),
                        )
                        .map_err(streaming_execution_error)?
                } else {
                    // Non-LLM: batch execution, emit single token
                    let result = guard
                        .executor
                        .execute(&metadata, &envelope, config.as_ref())
                        .map_err(|e| sdk_execution_error("Execution failed", e))?;

                    // Emit single token with full result
                    if let xybrid_core::ir::EnvelopeKind::Text(text) = &result.kind {
                        let stream_token = StreamToken {
                            token: text.clone(),
                            token_id: None,
                            index: 0,
                            cumulative_text: text.clone(),
                            finish_reason: Some("stop".to_string()),
                        };
                        let _ = tx.blocking_send(StreamEvent::Token(stream_token));
                    }
                    result
                };

                let latency_ms = start.elapsed().as_millis() as u32;

                // Emit telemetry
                let event = crate::telemetry::TelemetryEvent {
                    event_type: "ModelComplete".to_string(),
                    stage_name: Some(model_id.clone()),
                    target: Some("local".to_string()),
                    latency_ms: Some(latency_ms),
                    error: None,
                    data: Some(
                        serde_json::json!({
                            "model_id": model_id,
                            "version": version,
                            "output_type": format!("{:?}", output_type),
                            "streaming": true,
                        })
                        .to_string(),
                    ),
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                };
                crate::telemetry::publish_telemetry_event(event);

                Ok(InferenceResult::new(output, &model_id, latency_ms))
            })
            .await;

            // Send completion or error event
            match result {
                Ok(Ok(inference_result)) => {
                    let _ = tx_completion
                        .send(StreamEvent::Complete(inference_result))
                        .await;
                }
                Ok(Err(e)) => {
                    let _ = tx_completion.send(StreamEvent::Error(e.to_string())).await;
                }
                Err(e) => {
                    let _ = tx_completion
                        .send(StreamEvent::Error(format!("Task failed: {}", e)))
                        .await;
                }
            }
        });

        Box::pin(ReceiverStream::new(rx))
    }

    /// Check if this model supports true token streaming.
    ///
    /// Returns `true` for LLM models (GGUF) when LLM features are enabled,
    /// `false` for other model types or when LLM features are disabled.
    /// Note: `run_streaming()` works for all models, but only LLM models
    /// get true token-by-token streaming; others emit a single result.
    pub fn supports_token_streaming(&self) -> bool {
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        {
            use xybrid_core::execution::ExecutionTemplate;

            self.handle
                .read()
                .ok()
                .map(|h| {
                    matches!(
                        h.metadata.execution_template,
                        ExecutionTemplate::Gguf { .. }
                    )
                })
                .unwrap_or(false)
        }
        #[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp")))]
        {
            false
        }
    }

    /// Run batch inference asynchronously.
    pub async fn run_async(
        &self,
        envelope: &Envelope,
        config: Option<&GenerationConfig>,
    ) -> SdkResult<InferenceResult> {
        crate::telemetry::maybe_emit_dev_nudge();
        let handle = self.handle.clone();
        let model_id = self.model_id.clone();
        let version = self.version.clone();
        let output_type = self.output_type;
        let envelope = envelope.clone();
        let config = config.cloned();

        tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let resource_guard = crate::telemetry::begin_resource_run();

            // Per-call trace_id scope — see `run` for rationale. Cleared
            // on drop at end of the spawn_blocking closure (after publish)
            // or on any early return.
            let trace_id = uuid::Uuid::new_v4();
            let _telemetry_ctx =
                crate::telemetry::TelemetryPipelineContextGuard::install(None, Some(trace_id));

            // Recover from poisoned RwLock to prevent permanent lock errors
            let mut guard = handle.write().unwrap_or_else(|e| e.into_inner());

            if !guard.loaded {
                return Err(SdkError::NotLoaded);
            }

            // Clone metadata to avoid borrow conflict with executor
            let metadata = guard.metadata.clone();
            let output = guard
                .executor
                .execute(&metadata, &envelope, config.as_ref())
                .map_err(|e| sdk_execution_error("Execution failed", e))?;

            let latency_ms = start.elapsed().as_millis() as u32;

            // Emit ModelComplete telemetry event
            let event = crate::telemetry::TelemetryEvent {
                event_type: "ModelComplete".to_string(),
                stage_name: Some(model_id.clone()),
                target: Some("local".to_string()),
                latency_ms: Some(latency_ms),
                error: None,
                data: Some(
                    serde_json::json!({
                        "model_id": model_id,
                        "version": version,
                        "output_type": format!("{:?}", output_type),
                    })
                    .to_string(),
                ),
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            };
            crate::telemetry::publish_with_resource_summary(event, resource_guard);

            Ok(InferenceResult::new(output, &model_id, latency_ms))
        })
        .await
        .map_err(|e| SdkError::inference_src("Task join error", e))?
    }

    /// Create a streaming session for real-time ASR.
    ///
    /// Returns an error if `!supports_streaming()`.
    ///
    /// # Arguments
    ///
    /// * `config` - Streaming configuration (VAD, language, etc.)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::{XybridModel, StreamConfig};
    /// # let model: XybridModel = unimplemented!();
    /// # let audio_samples: Vec<f32> = vec![];
    /// let stream = model.stream(StreamConfig::with_vad())?;
    ///
    /// // Feed audio chunks
    /// stream.feed(&audio_samples)?;
    ///
    /// // Get partial results
    /// if let Some(partial) = stream.partial_result() {
    ///     println!("Partial: {}", partial.text);
    /// }
    ///
    /// // Get final transcript
    /// let transcript = stream.flush()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn stream(&self, config: StreamConfig) -> SdkResult<XybridStream> {
        if !self.supports_streaming {
            return Err(SdkError::StreamingNotSupported);
        }

        // Recover from poisoned RwLock to prevent permanent lock errors
        let handle = self.handle.read().unwrap_or_else(|e| e.into_inner());

        if !handle.loaded {
            return Err(SdkError::NotLoaded);
        }

        // Convert to core StreamConfig
        let core_config = CoreStreamConfig {
            vad: CoreVadConfig {
                enabled: config.enable_vad,
                model_dir: config.vad_model_dir,
                threshold: config.vad_threshold,
                ..Default::default()
            },
            language: config.language,
            ..Default::default()
        };

        XybridStream::new(&handle.model_dir, core_config, &self.model_id)
    }

    /// Unload the model from memory.
    ///
    /// The model can be reloaded by creating a new ModelLoader.
    pub fn unload(&self) -> SdkResult<()> {
        // Recover from poisoned RwLock to prevent permanent lock errors
        let mut handle = self.handle.write().unwrap_or_else(|e| e.into_inner());

        handle.loaded = false;
        // Clear the session cache (drop executor and recreate empty)
        handle.executor = TemplateExecutor::default();

        Ok(())
    }
}

// Make XybridModel cloneable (shares the handle)
impl Clone for XybridModel {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            model_id: self.model_id.clone(),
            version: self.version.clone(),
            output_type: self.output_type,
            supports_streaming: self.supports_streaming,
            // Share the in-flight slot so all clones coordinate preemption
            // through one mutex (see field docs).
            current_run: self.current_run.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inherent `SdkError::is_retryable` / `retry_after` accessors and
    /// the `RetryableError` trait impl must agree for every variant — the
    /// trait forwards to the inherent methods, so a divergence would be a
    /// refactor slip. Covers the four retryable variants, a representative
    /// non-retryable one, and the `RateLimited` retry-after passthrough.
    #[test]
    fn inherent_and_trait_retryability_agree() {
        use xybrid_core::http::RetryableError;

        let cases = [
            (SdkError::network("x"), true),
            (
                SdkError::RateLimited {
                    retry_after_secs: 5,
                },
                true,
            ),
            (SdkError::Timeout { timeout_ms: 100 }, true),
            (SdkError::offline("x"), true),
            (SdkError::CircuitOpen("x".into()), false),
            (SdkError::NotLoaded, false),
            (SdkError::ConfigError("x".into()), false),
        ];
        for (err, expected) in &cases {
            assert_eq!(err.is_retryable(), *expected, "inherent for {err:?}");
            assert_eq!(
                RetryableError::is_retryable(err),
                *expected,
                "trait for {err:?}"
            );
        }

        // Only RateLimited carries a server-specified backoff.
        let rl = SdkError::RateLimited {
            retry_after_secs: 7,
        };
        assert_eq!(rl.retry_after(), Some(std::time::Duration::from_secs(7)));
        assert_eq!(rl.retry_after(), RetryableError::retry_after(&rl));
        assert_eq!(SdkError::NotLoaded.retry_after(), None);
    }

    /// The wrapping variants must keep the underlying cause walkable via
    /// `std::error::Error::source` and downcastable to its concrete type,
    /// rather than flattening it into the message string. This is the whole
    /// point of the `#[source]` cause: a consumer can inspect the real error
    /// instead of string-grepping. Also asserts the message no longer
    /// embeds the cause (no double-rendering).
    #[test]
    fn wrapping_variants_chain_source_cause() {
        use std::error::Error as _;

        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "boom");
        let err = SdkError::load_src("metadata unreadable", io);

        // Display shows the variant prefix + context message, not the cause —
        // the cause is reachable through `source()`, not flattened into the text.
        assert_eq!(err.to_string(), "Failed to load model: metadata unreadable");

        // The cause is preserved, walkable, and downcasts to the real type.
        let source = err.source().expect("source cause should be present");
        let io_cause = source
            .downcast_ref::<std::io::Error>()
            .expect("source should downcast to io::Error");
        assert_eq!(io_cause.kind(), std::io::ErrorKind::PermissionDenied);

        // A message-only constructor carries no source.
        assert!(SdkError::load("no cause here").source().is_none());
    }

    #[test]
    fn streaming_execution_error_preserves_typed_cloud_fallback_abort() {
        let error = streaming_execution_error(
            xybrid_core::runtime_adapter::AdapterError::AbortedForCloudFallback {
                reason: xybrid_core::abort::AbortReason::StressMemory,
            },
        );

        match error {
            SdkError::AbortedForCloudFallback { reason } => {
                assert_eq!(reason, xybrid_core::abort::AbortReason::StressMemory);
            }
            other => panic!("expected AbortedForCloudFallback, got {other:?}"),
        }
    }

    #[test]
    fn streaming_callback_error_preserves_typed_cloud_fallback_abort() {
        let error =
            streaming_callback_error(Box::new(xybrid_core::abort::CloudFallbackAbort::new(
                xybrid_core::abort::AbortReason::StressThermal,
            )));

        match error {
            SdkError::AbortedForCloudFallback { reason } => {
                assert_eq!(reason, xybrid_core::abort::AbortReason::StressThermal);
            }
            other => panic!("expected AbortedForCloudFallback, got {other:?}"),
        }
    }

    #[test]
    fn streaming_callback_error_keeps_non_fallback_abort_generic() {
        let error =
            streaming_callback_error(Box::new(crate::run_options::AbortReason::UserCancelled));

        match error {
            SdkError::InferenceError { message, source } => {
                assert!(message.contains("Streaming callback failed"));
                // The callback's cause is now chained as `#[source]`, not
                // flattened into the message.
                let cause = source.expect("callback error should chain its cause");
                assert!(cause.to_string().contains("user_cancelled"));
            }
            other => panic!("expected inference error, got {other:?}"),
        }
    }

    #[test]
    fn execution_error_preserves_unsupported_model_capability() {
        let error = sdk_execution_error(
            "Execution failed",
            xybrid_core::error::XybridError::UnsupportedModelCapability {
                model_id: "smollm2-360m".to_string(),
                capability: "image input".to_string(),
                hint: "select a VisionLanguage model".to_string(),
            },
        );

        match error {
            SdkError::UnsupportedModelCapability {
                model_id,
                capability,
                hint,
            } => {
                assert_eq!(model_id, "smollm2-360m");
                assert_eq!(capability, "image input");
                assert!(hint.contains("VisionLanguage"));
            }
            other => panic!("expected unsupported model capability, got {other:?}"),
        }
    }

    #[test]
    fn execution_error_preserves_unsupported_backend_capability() {
        let error = sdk_execution_error(
            "Execution failed",
            xybrid_core::error::XybridError::UnsupportedBackendCapability {
                model_id: "lfm2-vl-450m".to_string(),
                backend: "llama.cpp".to_string(),
                capability: "vision input".to_string(),
                hint: "rebuild with llm-llamacpp-vision".to_string(),
            },
        );

        match error {
            SdkError::UnsupportedBackendCapability {
                model_id,
                backend,
                capability,
                hint,
            } => {
                assert_eq!(model_id, "lfm2-vl-450m");
                assert_eq!(backend, "llama.cpp");
                assert_eq!(capability, "vision input");
                assert!(hint.contains("llm-llamacpp-vision"));
            }
            other => panic!("expected unsupported backend capability, got {other:?}"),
        }
    }

    #[test]
    fn execution_error_preserves_missing_artifact() {
        let error = sdk_execution_error(
            "Execution failed",
            xybrid_core::error::XybridError::MissingArtifact {
                artifact: "vision_encoder".to_string(),
                path: "/models/mmproj.gguf".to_string(),
            },
        );

        match error {
            SdkError::MissingArtifact { artifact, path } => {
                assert_eq!(artifact, "vision_encoder");
                assert_eq!(path, "/models/mmproj.gguf");
            }
            other => panic!("expected missing artifact, got {other:?}"),
        }
    }

    #[test]
    fn test_model_loader_from_registry() {
        let loader = ModelLoader::from_registry("kokoro-82m");
        assert_eq!(loader.model_id(), Some("kokoro-82m"));
        assert_eq!(loader.version(), None); // Version resolved by registry
        assert_eq!(loader.source_type(), "registry");
    }

    #[test]
    fn test_model_loader_from_registry_with_platform() {
        let loader = ModelLoader::from_registry_with_platform("whisper-tiny", "macos-arm64");
        assert_eq!(loader.model_id(), Some("whisper-tiny"));
        assert_eq!(loader.source_type(), "registry");
    }

    #[test]
    fn gguf_models_are_streaming_text_models() {
        let mut metadata = ModelMetadata::onnx("qwen2.5-0.5b-instruct", "1.0", "model.gguf");
        metadata.execution_template = ExecutionTemplate::Gguf {
            model_file: "model.gguf".to_string(),
            chat_template: None,
            context_length: 2048,
            generation_params: None,
        };

        assert!(
            ModelLoader::check_streaming_support(&metadata),
            "GGUF LLMs stream tokens and must run abort checks between chunks"
        );
        assert_eq!(ModelLoader::infer_output_type(&metadata), OutputType::Text);
    }

    #[test]
    #[allow(deprecated)]
    fn test_model_loader_from_legacy_registry() {
        let loader = ModelLoader::from_legacy_registry("http://localhost:8080", "whisper", "1.0");
        assert_eq!(loader.model_id(), Some("whisper"));
        assert_eq!(loader.version(), Some("1.0"));
        assert_eq!(loader.source_type(), "legacy_registry");
    }

    #[test]
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();
        assert!(!config.enable_vad);
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_stream_config_with_vad() {
        let config = StreamConfig::with_vad().language("fr").vad_threshold(0.7);
        assert!(config.enable_vad);
        assert_eq!(config.language, Some("fr".to_string()));
        assert_eq!(config.vad_threshold, 0.7);
    }

    // ========================================================================
    // run_streaming_with_fallback / dispatch_after_local
    //
    // We test the testable inner helper `dispatch_after_local` directly so the
    // fallback decision logic is covered without standing up a real model. The
    // public wrapper is a thin shim over `run_streaming_with_options` plus this
    // helper, so its behavior is derivable from these tests + the existing
    // streaming-error tests above.
    // ========================================================================

    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Mutex;

    /// Records calls and emits a fixed response as one or more synthetic tokens.
    /// Used as a stand-in for `CloudRuntimeAdapter` in unit tests.
    struct FakeCloudAdapter {
        response_text: String,
        calls: Mutex<Vec<xybrid_core::ir::Envelope>>,
    }

    impl FakeCloudAdapter {
        fn new(response: &str) -> Self {
            Self {
                response_text: response.to_string(),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }

        fn calls(&self) -> Vec<xybrid_core::ir::Envelope> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl xybrid_core::runtime_adapter::CloudStreaming for FakeCloudAdapter {
        fn execute_streaming(
            &self,
            input: &xybrid_core::ir::Envelope,
            mut on_token: xybrid_core::runtime_adapter::types::StreamingCallback<'_>,
        ) -> xybrid_core::runtime_adapter::AdapterResult<xybrid_core::ir::Envelope> {
            self.calls.lock().unwrap().push(input.clone());

            let token = xybrid_core::runtime_adapter::types::PartialToken {
                token: self.response_text.clone(),
                token_id: None,
                index: 0,
                cumulative_text: self.response_text.clone(),
                finish_reason: Some("stop".to_string()),
            };
            on_token(token).map_err(|e| {
                xybrid_core::runtime_adapter::AdapterError::InferenceFailed(format!("{}", e))
            })?;

            Ok(xybrid_core::ir::Envelope::new(
                xybrid_core::ir::EnvelopeKind::Text(self.response_text.clone()),
            ))
        }
    }

    struct FailingCloudAdapter {
        calls: AtomicUsize,
    }

    impl FailingCloudAdapter {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl xybrid_core::runtime_adapter::CloudStreaming for FailingCloudAdapter {
        fn execute_streaming(
            &self,
            _input: &xybrid_core::ir::Envelope,
            _on_token: xybrid_core::runtime_adapter::types::StreamingCallback<'_>,
        ) -> xybrid_core::runtime_adapter::AdapterResult<xybrid_core::ir::Envelope> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(xybrid_core::runtime_adapter::AdapterError::RuntimeError(
                "gateway unavailable".to_string(),
            ))
        }
    }

    #[derive(Debug)]
    struct FixedUnitResourceProvider {
        snapshot: xybrid_core::device::ResourceSnapshot,
    }

    impl FixedUnitResourceProvider {
        fn new(snapshot: xybrid_core::device::ResourceSnapshot) -> Self {
            Self { snapshot }
        }
    }

    impl xybrid_core::device::ResourceSnapshotProvider for FixedUnitResourceProvider {
        fn current_snapshot(
            &self,
            _max_age: std::time::Duration,
        ) -> xybrid_core::device::ResourceSnapshot {
            self.snapshot
        }
    }

    fn text_envelope(text: &str) -> xybrid_core::ir::Envelope {
        xybrid_core::ir::Envelope::new(xybrid_core::ir::EnvelopeKind::Text(text.to_string()))
    }

    fn test_loaded_model(supports_streaming: bool) -> XybridModel {
        let metadata =
            xybrid_core::execution::ModelMetadata::onnx("local-test-model", "1.0", "model.onnx");
        XybridModel {
            handle: Arc::new(RwLock::new(ModelHandle {
                executor: TemplateExecutor::default(),
                metadata,
                model_dir: PathBuf::from("."),
                loaded: true,
            })),
            model_id: "local-test-model".to_string(),
            version: "1.0".to_string(),
            output_type: OutputType::Text,
            supports_streaming,
            current_run: Arc::new(Mutex::new(None)),
        }
    }

    fn default_metrics() -> xybrid_core::context::DeviceMetrics {
        xybrid_core::context::DeviceMetrics::default()
    }

    fn default_signal() -> xybrid_core::orchestrator::authority::SignalContext {
        xybrid_core::orchestrator::authority::SignalContext::from_metrics(&default_metrics())
    }

    struct FakeAuthority {
        allow_policy: bool,
        deny_reason: String,
        policy_requests: Mutex<Vec<xybrid_core::orchestrator::authority::PolicyRequest>>,
        outcomes: Mutex<Vec<xybrid_core::orchestrator::authority::ExecutionOutcome>>,
    }

    impl FakeAuthority {
        fn allow() -> Self {
            Self {
                allow_policy: true,
                deny_reason: String::new(),
                policy_requests: Mutex::new(Vec::new()),
                outcomes: Mutex::new(Vec::new()),
            }
        }

        fn deny(reason: &str) -> Self {
            Self {
                allow_policy: false,
                deny_reason: reason.to_string(),
                policy_requests: Mutex::new(Vec::new()),
                outcomes: Mutex::new(Vec::new()),
            }
        }

        fn policy_requests(&self) -> Vec<xybrid_core::orchestrator::authority::PolicyRequest> {
            self.policy_requests.lock().unwrap().clone()
        }

        fn outcomes(&self) -> Vec<xybrid_core::orchestrator::authority::ExecutionOutcome> {
            self.outcomes.lock().unwrap().clone()
        }
    }

    impl xybrid_core::orchestrator::authority::OrchestrationAuthority for FakeAuthority {
        fn apply_policy(
            &self,
            request: &xybrid_core::orchestrator::authority::PolicyRequest,
        ) -> xybrid_core::orchestrator::authority::AuthorityDecision<
            xybrid_core::orchestrator::authority::PolicyOutcome,
        > {
            self.policy_requests.lock().unwrap().push(request.clone());
            if self.allow_policy {
                xybrid_core::orchestrator::authority::AuthorityDecision::local(
                    xybrid_core::orchestrator::authority::PolicyOutcome::Allow,
                    "policy allowed",
                )
            } else {
                xybrid_core::orchestrator::authority::AuthorityDecision::local(
                    xybrid_core::orchestrator::authority::PolicyOutcome::Deny {
                        reason: self.deny_reason.clone(),
                    },
                    self.deny_reason.clone(),
                )
            }
        }

        fn resolve_target(
            &self,
            _context: &xybrid_core::orchestrator::authority::StageContext,
        ) -> xybrid_core::orchestrator::authority::AuthorityDecision<
            xybrid_core::orchestrator::authority::ResolvedTarget,
        > {
            xybrid_core::orchestrator::authority::AuthorityDecision::local(
                xybrid_core::orchestrator::authority::ResolvedTarget::Device,
                "test",
            )
        }

        fn select_model(
            &self,
            request: &xybrid_core::orchestrator::authority::ModelRequest,
        ) -> xybrid_core::orchestrator::authority::AuthorityDecision<
            xybrid_core::orchestrator::authority::ModelSelection,
        > {
            xybrid_core::orchestrator::authority::AuthorityDecision::local(
                xybrid_core::orchestrator::authority::ModelSelection {
                    model_id: request.model_id.clone(),
                    variant: None,
                    source: xybrid_core::orchestrator::authority::ModelSource::Local {
                        path: "test".to_string(),
                    },
                },
                "test",
            )
        }

        fn record_outcome(&self, outcome: &xybrid_core::orchestrator::authority::ExecutionOutcome) {
            self.outcomes.lock().unwrap().push(outcome.clone());
        }

        fn name(&self) -> &str {
            "fake"
        }
    }

    #[test]
    fn dispatch_after_local_retries_on_typed_abort_and_emits_seam() {
        let cloud = FakeCloudAdapter::new("hello from cloud");
        let authority = FakeAuthority::allow();
        let envelope = text_envelope("write me a haiku");
        let collected: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_for_cb = collected.clone();
        let mut on_token = move |t: xybrid_core::runtime_adapter::types::PartialToken| {
            collected_for_cb.lock().unwrap().push(t.token);
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        };
        let seam_count = Arc::new(AtomicUsize::new(0));
        let seam_reason: Arc<Mutex<Option<xybrid_core::abort::AbortReason>>> =
            Arc::new(Mutex::new(None));
        let seam_count_for_cb = seam_count.clone();
        let seam_reason_for_cb = seam_reason.clone();
        let mut on_seam = move |s: SeamInfo| {
            seam_count_for_cb.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            *seam_reason_for_cb.lock().unwrap() = Some(s.reason);
            assert_eq!(s.correlation_id, "corr-1");
            assert_eq!(s.local_tokens, 3);
            assert_eq!(s.local_latency_ms, 100);
        };

        let local_result: SdkResult<InferenceResult> = Err(SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory,
        });

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-1".to_string(),
            "test-model",
            3,
            100,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        )
        .expect("retry should succeed");

        assert_eq!(seam_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            *seam_reason.lock().unwrap(),
            Some(xybrid_core::abort::AbortReason::StressMemory)
        );
        assert_eq!(cloud.call_count(), 1);
        assert_eq!(result.text(), Some("hello from cloud"));
        assert_eq!(result.model_id(), "test-model");
        let tokens = collected.lock().unwrap().clone();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], "hello from cloud");
    }

    #[test]
    fn run_streaming_with_fallback_retries_cloud_on_pre_run_resource_pressure() {
        let model = test_loaded_model(true);
        let cloud = FakeCloudAdapter::new("hello from cloud");
        let mut critical_snapshot = xybrid_core::device::ResourceSnapshot::unknown();
        critical_snapshot.memory_pressure = xybrid_core::device::MemoryPressure::Critical;
        let resource_provider = Arc::new(FixedUnitResourceProvider::new(critical_snapshot));
        let options = RunOptions::new()
            .with_abort_policy(
                crate::run_options::AbortPolicy::default()
                    .stop_on(crate::run_options::AbortSignal::MemoryPressureCritical)
                    .with_cloud_fallback(true)
                    .with_max_grace_tokens(0),
            )
            .with_resource_provider(resource_provider)
            .with_correlation_id("corr-pre-run");
        let envelope = text_envelope("write me a haiku");
        let collected: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_for_cb = collected.clone();
        let mut on_token = move |t: xybrid_core::runtime_adapter::types::PartialToken| {
            collected_for_cb.lock().unwrap().push(t.token);
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        };
        let seam_count = Arc::new(AtomicUsize::new(0));
        let seam_count_for_cb = seam_count.clone();
        let mut on_seam = move |s: SeamInfo| {
            seam_count_for_cb.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(s.reason, xybrid_core::abort::AbortReason::StressMemory);
            assert_eq!(s.correlation_id, "corr-pre-run");
            assert_eq!(s.local_tokens, 0);
        };

        let result = model
            .run_streaming_with_fallback(&envelope, &options, &cloud, &mut on_token, &mut on_seam)
            .expect("pre-run resource pressure should retry on cloud");

        assert_eq!(seam_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(cloud.call_count(), 1);
        assert_eq!(result.text(), Some("hello from cloud"));
        assert_eq!(collected.lock().unwrap().as_slice(), ["hello from cloud"]);
    }

    #[test]
    fn run_streaming_with_fallback_retries_cloud_on_pre_run_thermal_pressure() {
        let model = test_loaded_model(true);
        let cloud = FakeCloudAdapter::new("hello from cloud");
        let mut critical_snapshot = xybrid_core::device::ResourceSnapshot::unknown();
        critical_snapshot.thermal_state = xybrid_core::device::ThermalState::Critical;
        let resource_provider = Arc::new(FixedUnitResourceProvider::new(critical_snapshot));
        let options = RunOptions::new()
            .with_abort_policy(
                crate::run_options::AbortPolicy::default()
                    .stop_on(crate::run_options::AbortSignal::ThermalCritical)
                    .with_cloud_fallback(true)
                    .with_max_grace_tokens(0),
            )
            .with_resource_provider(resource_provider)
            .with_correlation_id("corr-thermal-pre-run");
        let envelope = text_envelope("write me a haiku");
        let collected: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_for_cb = collected.clone();
        let mut on_token = move |t: xybrid_core::runtime_adapter::types::PartialToken| {
            collected_for_cb.lock().unwrap().push(t.token);
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        };
        let seam_count = Arc::new(AtomicUsize::new(0));
        let seam_count_for_cb = seam_count.clone();
        let mut on_seam = move |s: SeamInfo| {
            seam_count_for_cb.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(s.reason, xybrid_core::abort::AbortReason::StressThermal);
            assert_eq!(s.correlation_id, "corr-thermal-pre-run");
            assert_eq!(s.local_tokens, 0);
        };

        let result = model
            .run_streaming_with_fallback(&envelope, &options, &cloud, &mut on_token, &mut on_seam)
            .expect("pre-run thermal pressure should retry on cloud");

        assert_eq!(seam_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(cloud.call_count(), 1);
        assert_eq!(result.text(), Some("hello from cloud"));
        assert_eq!(collected.lock().unwrap().as_slice(), ["hello from cloud"]);
    }

    #[test]
    fn dispatch_after_local_records_local_abort_and_cloud_success_outcomes() {
        let cloud = FakeCloudAdapter::new("hello from cloud");
        let authority = FakeAuthority::allow();
        let mut envelope = text_envelope("write a haiku");
        envelope
            .metadata
            .insert("model".to_string(), "deepseek-chat".to_string());
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let mut on_seam = |_s: SeamInfo| {};
        let local_result: SdkResult<InferenceResult> = Err(SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory,
        });

        dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-record".to_string(),
            "local-model",
            7,
            321,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        )
        .expect("cloud retry should succeed");

        let outcomes = authority.outcomes();
        assert_eq!(outcomes.len(), 2);
        assert!(matches!(
            outcomes[0].category,
            Some(
                xybrid_core::orchestrator::authority::OutcomeCategory::AbortedForCloudFallback {
                    reason: xybrid_core::abort::AbortReason::StressMemory
                }
            )
        ));
        assert!(matches!(
            outcomes[0].target,
            xybrid_core::orchestrator::authority::ResolvedTarget::Device
        ));
        assert!(matches!(
            outcomes[1].category,
            Some(xybrid_core::orchestrator::authority::OutcomeCategory::Success)
        ));
        assert!(matches!(
            outcomes[1].target,
            xybrid_core::orchestrator::authority::ResolvedTarget::Cloud { .. }
        ));
        assert_eq!(outcomes[1].model_id.as_deref(), Some("deepseek-chat"));
    }

    #[test]
    fn dispatch_after_local_records_cloud_retry_failure() {
        let cloud = FailingCloudAdapter::new();
        let authority = FakeAuthority::allow();
        let mut envelope = text_envelope("write a haiku");
        envelope
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        envelope
            .metadata
            .insert("model".to_string(), "gpt-4o-mini".to_string());
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let mut on_seam = |_s: SeamInfo| {};
        let local_result: SdkResult<InferenceResult> = Err(SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory,
        });

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-cloud-fail".to_string(),
            "local-model",
            2,
            120,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        );

        match result {
            Err(SdkError::InferenceError { message, source }) => {
                // The underlying cause is chained as `#[source]`; reconstruct
                // the full chain to assert on the original failure text.
                let full = source.map_or(message.clone(), |s| format!("{message}: {s}"));
                assert!(full.contains("gateway unavailable"), "{full}");
            }
            other => panic!("expected cloud retry failure, got {other:?}"),
        }
        assert_eq!(cloud.call_count(), 1);

        let outcomes = authority.outcomes();
        assert_eq!(outcomes.len(), 2);
        assert!(matches!(
            outcomes[0].category,
            Some(
                xybrid_core::orchestrator::authority::OutcomeCategory::AbortedForCloudFallback { .. }
            )
        ));
        assert!(matches!(
            outcomes[1].category,
            Some(xybrid_core::orchestrator::authority::OutcomeCategory::HardFail { .. })
        ));
        assert_eq!(outcomes[1].model_id.as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn dispatch_after_local_rechecks_policy_and_retries_with_original_envelope() {
        let cloud = FakeCloudAdapter::new("cloud continuation");
        let authority = FakeAuthority::allow();
        let mut envelope = text_envelope("original prompt, not partial local output");
        envelope
            .metadata
            .insert("provider".to_string(), "openai".to_string());
        envelope
            .metadata
            .insert("model".to_string(), "gpt-4o-mini".to_string());
        envelope
            .metadata
            .insert("temperature".to_string(), "0.2".to_string());

        let received: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let received_for_cb = received.clone();
        let mut on_token = move |t: xybrid_core::runtime_adapter::types::PartialToken| {
            received_for_cb.lock().unwrap().push(t.token);
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        };
        let mut on_seam = |_s: SeamInfo| {};
        let local_result: SdkResult<InferenceResult> = Err(SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory,
        });

        dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-original-envelope".to_string(),
            "local-model",
            4,
            123,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        )
        .expect("cloud retry should succeed");

        let policy_requests = authority.policy_requests();
        assert_eq!(policy_requests.len(), 1);
        assert_eq!(policy_requests[0].stage_id, "gpt-4o-mini");
        assert_eq!(policy_requests[0].envelope, envelope);

        let cloud_calls = cloud.calls();
        assert_eq!(cloud_calls.len(), 1);
        assert_eq!(cloud_calls[0], envelope);
        assert_eq!(received.lock().unwrap().as_slice(), ["cloud continuation"]);
    }

    #[test]
    fn dispatch_after_local_stops_before_cloud_when_cancelled_after_seam() {
        let cloud = FakeCloudAdapter::new("must not run");
        let authority = FakeAuthority::allow();
        let envelope = text_envelope("prompt");
        let cancellation = CancellationToken::new();
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let seam_count = Arc::new(AtomicUsize::new(0));
        let seam_count_for_cb = seam_count.clone();
        let cancellation_for_seam = cancellation.clone();
        let mut on_seam = move |_s: SeamInfo| {
            seam_count_for_cb.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            cancellation_for_seam.cancel();
        };
        let local_result: SdkResult<InferenceResult> = Err(SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory,
        });

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-cancel-after-seam".to_string(),
            "local-model",
            0,
            0,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            Some(cancellation),
            &mut on_token,
            &mut on_seam,
        );

        match result {
            Err(SdkError::InferenceError { message, .. }) => {
                assert!(message.contains("user_cancelled"))
            }
            other => panic!("expected user_cancelled error, got {other:?}"),
        }
        assert_eq!(seam_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(cloud.call_count(), 0);
        assert_eq!(authority.outcomes().len(), 1);
    }

    /// Short-circuit case (FFI issue 10): a token cancelled *before* the run
    /// starts must abort in the pre-run gate (`check_before_run`) with
    /// `UserCancelled`, before the model write lock is ever taken — and the
    /// lock must be free afterwards.
    ///
    /// This intentionally does NOT prove the held-then-released path (the lock
    /// is never acquired here because the pre-run gate fires first). The
    /// mid-stream held-lock release is covered by
    /// `run_streaming_with_options_mid_stream_cancel_releases_held_write_lock`.
    #[test]
    fn run_streaming_with_options_pre_run_cancellation_short_circuits_before_lock() {
        let model = test_loaded_model(true);
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(
                crate::run_options::AbortPolicy::default()
                    .stop_on(crate::run_options::AbortSignal::UserCancelled),
            );
        let envelope = text_envelope("tell me a long story");

        // Cancel before the run so the pre-run gate aborts deterministically
        // without needing a real token stream.
        token.cancel();

        let started = Instant::now();
        let err = model
            .run_streaming_with_options(&envelope, &options, |_token| Ok(()))
            .expect_err("cancelled run must abort, not produce a result");
        let elapsed = started.elapsed();

        // (a) The run halts with a user-cancellation error (terminal; never a
        // cloud-fallback marker).
        match err {
            SdkError::InferenceError {
                ref message,
                ref source,
                ..
            } => {
                let full = match source {
                    Some(s) => format!("{message}: {s}"),
                    None => message.clone(),
                };
                assert!(
                    full.contains("user_cancelled"),
                    "expected user_cancelled abort, got: {full}"
                );
            }
            other => panic!("expected user_cancelled InferenceError, got {other:?}"),
        }

        // (b) The model write lock must be free immediately after the aborted
        // run returns — `try_write` succeeding proves the run did not leave the
        // lock held. (acquirable well within one token).
        assert!(
            model.handle.try_write().is_ok(),
            "model write lock must be released after a cancelled streaming run"
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "cancelled run should return promptly, took {elapsed:?}"
        );
    }

    /// Fake `"onnx"` runtime that blocks inside `execute()` until released, so a
    /// test can observe the model write lock being *held* mid-run. On release it
    /// returns a text envelope, after which the streaming path emits a single
    /// synthetic token through the per-token abort gate — exactly the boundary
    /// that halts a real LLM stream when the token is cancelled.
    struct BlockingFakeRuntime {
        entered_tx: Mutex<Option<std::sync::mpsc::Sender<()>>>,
        release_rx: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl xybrid_core::runtime_adapter::ModelRuntime for BlockingFakeRuntime {
        fn name(&self) -> &str {
            "onnx"
        }

        fn supported_formats(&self) -> Vec<&str> {
            vec!["onnx"]
        }

        fn load(
            &mut self,
            _model_path: &std::path::Path,
        ) -> xybrid_core::runtime_adapter::AdapterResult<()> {
            Ok(())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn execute(
            &mut self,
            _input: &xybrid_core::ir::Envelope,
        ) -> xybrid_core::runtime_adapter::AdapterResult<xybrid_core::ir::Envelope> {
            // Signal that we are inside execute() — the write lock is held now.
            if let Some(tx) = self.entered_tx.lock().unwrap().take() {
                let _ = tx.send(());
            }
            // Block until the test releases us (after it has cancelled the
            // token and confirmed the lock is held).
            let _ = self.release_rx.lock().unwrap().recv();
            Ok(xybrid_core::ir::Envelope::new(
                xybrid_core::ir::EnvelopeKind::Text("partial output".to_string()),
            ))
        }
    }

    fn test_model_with_runtime(
        runtime: Box<dyn xybrid_core::runtime_adapter::ModelRuntime>,
    ) -> XybridModel {
        let metadata =
            xybrid_core::execution::ModelMetadata::onnx("local-test-model", "1.0", "model.onnx");
        let mut executor = TemplateExecutor::default();
        // Register under "onnx" so the bare-Onnx execute path uses our fake.
        executor.register_runtime("onnx", runtime);
        XybridModel {
            handle: Arc::new(RwLock::new(ModelHandle {
                executor,
                metadata,
                model_dir: PathBuf::from("."),
                loaded: true,
            })),
            model_id: "local-test-model".to_string(),
            version: "1.0".to_string(),
            output_type: OutputType::Text,
            supports_streaming: true,
            current_run: Arc::new(Mutex::new(None)),
        }
    }

    /// Acceptance for the reachable-cancellation path (FFI issue 10),
    /// held-then-released case: a cancel issued *during* generation (while the
    /// model write lock is HELD) must (a) halt the run with `UserCancelled`
    /// rather than completing naturally, and (b) release the held write lock so
    /// a follow-up run can acquire it.
    ///
    /// Unlike the pre-run short-circuit test, this drives a fake runtime that
    /// blocks inside `execute()` — so the test can assert `try_write()` FAILS
    /// while the run is mid-flight (lock genuinely held), then cancel, then
    /// assert `try_write()` succeeds after the run unwinds.
    #[test]
    fn run_streaming_with_options_mid_stream_cancel_releases_held_write_lock() {
        use std::sync::mpsc;

        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let runtime = BlockingFakeRuntime {
            entered_tx: Mutex::new(Some(entered_tx)),
            release_rx: Mutex::new(release_rx),
        };
        let model = test_model_with_runtime(Box::new(runtime));

        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(
                crate::run_options::AbortPolicy::default()
                    .stop_on(crate::run_options::AbortSignal::UserCancelled),
            );

        // Run the streaming call on a worker thread so we can inspect lock state
        // from the test thread while the run is mid-flight.
        let model_for_worker = model.clone();
        let worker = std::thread::spawn(move || {
            let envelope = text_envelope("tell me a long story");
            model_for_worker.run_streaming_with_options(&envelope, &options, |_token| Ok(()))
        });

        // Wait until the runtime is inside execute() — the write lock is now held.
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("runtime should enter execute() and hold the write lock");

        // Prove the lock is genuinely HELD mid-run: a write acquisition must fail.
        assert!(
            model.handle.try_write().is_err(),
            "write lock must be held while the streaming run is mid-flight"
        );

        // Cancel mid-generation, then let execute() return so the per-token
        // abort gate observes the cancellation at the (single) token boundary.
        token.cancel();
        release_tx.send(()).expect("release the blocked runtime");

        let result = worker.join().expect("streaming worker should not panic");

        // (a) The run halted on user cancellation rather than completing
        // naturally with the runtime's "partial output".
        match result {
            Err(SdkError::InferenceError {
                ref message,
                ref source,
                ..
            }) => {
                let full = match source {
                    Some(s) => format!("{message}: {s}"),
                    None => message.clone(),
                };
                assert!(
                    full.contains("user_cancelled"),
                    "expected user_cancelled abort, got: {full}"
                );
            }
            Ok(r) => panic!(
                "mid-stream cancel must abort, but run completed naturally: {:?}",
                r.text()
            ),
            Err(other) => panic!("expected user_cancelled InferenceError, got {other:?}"),
        }

        // (b) The previously-HELD write lock must now be released — bounded wait
        // to allow the worker's lock guard to drop on its own thread.
        let mut acquired = false;
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if model.handle.try_write().is_ok() {
                acquired = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            acquired,
            "the HELD model write lock must be released after a mid-stream cancellation"
        );
    }

    /// Fake `"onnx"` runtime for the preemptive cancel-and-replace tests. Each
    /// `execute()` call increments `runtime_entered` (proving a run actually
    /// entered the runtime *while holding the model write lock*), signals entry
    /// (so the test sees the write lock held), and blocks until the test
    /// releases *that* call. Unlike `BlockingFakeRuntime` (one-shot entry
    /// signal), this supports the N sequential `execute()` calls that the
    /// preempt + stress tests drive against the single shared model write lock.
    struct MultiBlockingFakeRuntime {
        /// Count of `execute()` calls that reached the runtime while holding the
        /// write lock. Lets the stress test assert preemption happened under
        /// *real* lock contention (a replacement acquired the freed lock and
        /// ran), not vacuously via everyone aborting at the pre-run gate.
        runtime_entered: Arc<AtomicUsize>,
        entered_tx: Mutex<std::sync::mpsc::Sender<()>>,
        release_rx: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl xybrid_core::runtime_adapter::ModelRuntime for MultiBlockingFakeRuntime {
        fn name(&self) -> &str {
            "onnx"
        }

        fn supported_formats(&self) -> Vec<&str> {
            vec!["onnx"]
        }

        fn load(
            &mut self,
            _model_path: &std::path::Path,
        ) -> xybrid_core::runtime_adapter::AdapterResult<()> {
            Ok(())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn execute(
            &mut self,
            _input: &xybrid_core::ir::Envelope,
        ) -> xybrid_core::runtime_adapter::AdapterResult<xybrid_core::ir::Envelope> {
            // We are inside the runtime holding the write lock — record it.
            self.runtime_entered
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Signal entry: the write lock is held for this call now.
            let _ = self.entered_tx.lock().unwrap().send(());
            // Block until the test releases this specific call.
            let _ = self.release_rx.lock().unwrap().recv();
            Ok(xybrid_core::ir::Envelope::new(
                xybrid_core::ir::EnvelopeKind::Text("partial output".to_string()),
            ))
        }
    }

    fn preempt_options(token: &CancellationToken) -> RunOptions {
        RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(
                crate::run_options::AbortPolicy::default()
                    .stop_on(crate::run_options::AbortSignal::UserCancelled),
            )
    }

    /// Issue 11 acceptance — **preempt frees the lock without waiting for A's
    /// natural completion**. Run A blocks inside the fake runtime holding the
    /// write lock. Run B starts WITH preempt + a registered token; B's preempt
    /// cancels A *before* B tries to acquire the lock. We then let A's
    /// `execute()` return so A observes the cancel at its (single) token gate,
    /// returns `user_cancelled`, and drops the write guard — at which point B
    /// acquires the lock and completes. The key assertion is that A halts on
    /// cancellation (latest-frame-wins) and the lock becomes acquirable for B
    /// promptly, rather than B head-of-line blocking behind A.
    #[test]
    fn preempt_cancels_in_flight_run_and_frees_lock_for_replacement() {
        use std::sync::mpsc;

        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let runtime = MultiBlockingFakeRuntime {
            runtime_entered: Arc::new(AtomicUsize::new(0)),
            entered_tx: Mutex::new(entered_tx),
            release_rx: Mutex::new(release_rx),
        };
        let model = test_model_with_runtime(Box::new(runtime));

        // Run A: a normal streaming run that blocks inside execute() holding
        // the write lock. It carries its own token registered in the slot so
        // B's preempt has something to cancel.
        let token_a = CancellationToken::new();
        let options_a = preempt_options(&token_a);
        let model_a = model.clone();
        let worker_a = std::thread::spawn(move || {
            let envelope = text_envelope("frame A");
            model_a.run_streaming_with_options_preempt(
                &envelope,
                &options_a,
                /* preempt */ true,
                |_token| Ok(()),
            )
        });

        // Wait until A is inside execute() — the write lock is held.
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("run A should enter execute() and hold the write lock");
        assert!(
            model.handle.try_write().is_err(),
            "run A must hold the write lock while mid-flight"
        );

        // Run B: preempting start. preempt_register cancels A's token BEFORE B
        // touches the write lock. B will then block on handle.write() until A
        // releases it.
        let token_b = CancellationToken::new();
        let options_b = preempt_options(&token_b);
        let model_b = model.clone();
        let (b_done_tx, b_done_rx) = mpsc::channel::<()>();
        let worker_b = std::thread::spawn(move || {
            let envelope = text_envelope("frame B");
            let r = model_b.run_streaming_with_options_preempt(
                &envelope,
                &options_b,
                /* preempt */ true,
                |_token| Ok(()),
            );
            let _ = b_done_tx.send(());
            r
        });

        // B's preempt must flip A's token even though A still holds the lock.
        let cancel_deadline = Instant::now() + Duration::from_secs(2);
        while !token_a.is_cancelled() && Instant::now() < cancel_deadline {
            std::thread::yield_now();
        }
        assert!(
            token_a.is_cancelled(),
            "B's preempt must cancel A's in-flight token before acquiring the write lock"
        );

        // B must NOT have finished yet — it is blocked on handle.write() behind
        // A, which still holds the lock. This proves preemption did not somehow
        // bypass the lock.
        assert!(
            b_done_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "B must still be waiting on the write lock while A holds it"
        );

        // Release A's execute() so A reaches its token gate, sees the cancel,
        // and unwinds — freeing the lock. Then release B's execute() so B can
        // complete once it acquires the lock.
        release_tx.send(()).expect("release A");
        release_tx.send(()).expect("release B");

        let result_a = worker_a.join().expect("run A worker should not panic");
        let result_b = worker_b.join().expect("run B worker should not panic");

        // A halted on user cancellation (its partial output was discarded),
        // proving latest-frame-wins rather than A completing naturally.
        match result_a {
            Err(SdkError::InferenceError {
                ref message,
                ref source,
                ..
            }) => {
                let full = match source {
                    Some(s) => format!("{message}: {s}"),
                    None => message.clone(),
                };
                assert!(
                    full.contains("user_cancelled"),
                    "run A must abort on preempt, got: {full}"
                );
            }
            Ok(r) => panic!(
                "preempted run A must abort, but completed naturally: {:?}",
                r.text()
            ),
            Err(other) => panic!("expected user_cancelled for A, got {other:?}"),
        }

        // B was not preempted by anyone, so it completes normally.
        assert!(
            result_b.is_ok(),
            "replacement run B should complete, got {result_b:?}"
        );

        // Slot ends holding B's token (B cleared only if still its own; nothing
        // displaced B, so B's clear-if-ours emptied it).
        let slot = model.current_run.lock().unwrap();
        assert!(
            slot.is_none(),
            "after the last run clears its own token, the slot must be empty"
        );
    }

    /// Issue 11 mutation-check companion: the `preempt_cancels_…` assertion
    /// that A aborts is load-bearing. This test documents what happens when
    /// preemption is NOT requested — A runs to natural completion even though B
    /// starts — so the contrast confirms the preempt path is what cancels A.
    /// (Temporarily breaking `preempt_register` to a no-op makes
    /// `preempt_cancels_in_flight_run_and_frees_lock_for_replacement` fail at
    /// the `token_a.is_cancelled()` assertion — verified during development.)
    #[test]
    fn without_preempt_in_flight_run_is_not_cancelled_by_a_later_start() {
        use std::sync::mpsc;

        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let runtime = MultiBlockingFakeRuntime {
            runtime_entered: Arc::new(AtomicUsize::new(0)),
            entered_tx: Mutex::new(entered_tx),
            release_rx: Mutex::new(release_rx),
        };
        let model = test_model_with_runtime(Box::new(runtime));

        let token_a = CancellationToken::new();
        let options_a = preempt_options(&token_a);
        let model_a = model.clone();
        let worker_a = std::thread::spawn(move || {
            let envelope = text_envelope("frame A");
            // preempt = false: A registers nothing in the slot.
            model_a.run_streaming_with_options_preempt(
                &envelope,
                &options_a,
                /* preempt */ false,
                |_token| Ok(()),
            )
        });

        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("run A should enter execute()");

        // A non-preempting "later start" cannot cancel A: with preempt=false the
        // slot is never touched. Release A so it completes naturally.
        release_tx.send(()).expect("release A");
        let result_a = worker_a.join().expect("run A worker should not panic");

        assert!(
            !token_a.is_cancelled(),
            "without preempt, A's token must never be cancelled by the slot"
        );
        assert!(
            result_a.is_ok(),
            "non-preempt run A should complete naturally, got {result_a:?}"
        );
        // The slot was never engaged.
        assert!(
            model.current_run.lock().unwrap().is_none(),
            "non-preempt runs must not register in the in-flight slot"
        );
    }

    /// Issue 11 stress / no-deadlock: rapidly cycle start-replace N times
    /// against the fake runtime. Each cycle starts a preempting run that
    /// cancels its predecessor, then is itself released. Asserts the whole
    /// cycle completes within a bounded time (no deadlock), no panic, and the
    /// slot ends consistent (the last run's token, then None after it clears).
    #[test]
    fn rapid_preempt_cycling_does_not_deadlock_and_leaves_consistent_slot() {
        use std::sync::mpsc;

        const N: usize = 20;

        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let runtime_entered = Arc::new(AtomicUsize::new(0));
        let runtime = MultiBlockingFakeRuntime {
            runtime_entered: runtime_entered.clone(),
            entered_tx: Mutex::new(entered_tx),
            release_rx: Mutex::new(release_rx),
        };
        let model = test_model_with_runtime(Box::new(runtime));

        let started = Instant::now();

        // Deterministic cycle to force preemption under *real* lock contention
        // (not a vacuous pass where every run aborts at the pre-run gate before
        // ever acquiring the write lock):
        //
        //   1. Start run i (preempt=true) on a worker thread.
        //   2. Wait until run i has actually ENTERED the runtime — at which
        //      point it holds the model write lock and blocks inside execute().
        //   3. Start run i+1 (preempt=true). Its preempt_register cancels run
        //      i's token BEFORE run i+1 tries handle.write(), so run i+1 then
        //      blocks on the write lock that run i still holds.
        //   4. Release run i's execute() block. Run i hits its token gate, sees
        //      the cancel, returns user_cancelled, and DROPS the write guard —
        //      freeing the lock so run i+1 acquires it and enters the runtime.
        //
        // Each replacement therefore observably acquires a lock freed by the
        // preemption it triggered. If the preempt cancel were neutered, run i
        // would never be cancelled, run i+1 would block forever on
        // handle.write(), and step 2 for run i+1 (`entered_rx.recv_timeout`)
        // would time out — the test fails/hangs-then-fails rather than passing.
        let mut workers = Vec::with_capacity(N);
        let mut prev: Option<CancellationToken> = None;
        for i in 0..N {
            let token = CancellationToken::new();
            let options = preempt_options(&token);
            let model_i = model.clone();
            workers.push(std::thread::spawn(move || {
                let envelope = text_envelope(&format!("frame {i}"));
                model_i.run_streaming_with_options_preempt(
                    &envelope,
                    &options,
                    /* preempt */ true,
                    |_token| Ok(()),
                )
            }));

            // Wait for THIS run to enter the runtime (holding the write lock).
            // Its own preempt already cancelled the previous run's token; now we
            // release the previous run so it unwinds and frees the lock for this
            // one to acquire and enter.
            if let Some(prev_token) = prev.take() {
                // The new run's preempt must have flipped the previous token.
                let cancel_deadline = Instant::now() + Duration::from_secs(5);
                while !prev_token.is_cancelled() && Instant::now() < cancel_deadline {
                    std::thread::yield_now();
                }
                assert!(
                    prev_token.is_cancelled(),
                    "run {i}'s preempt must cancel the previous in-flight run's token"
                );
                // Release the previous run's execute() so it unwinds and frees
                // the lock for run i.
                release_tx.send(()).expect("release previous run");
            }
            entered_rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap_or_else(|_| {
                    panic!(
                        "run {i} must enter the runtime under contention — a timeout here means \
                         the previous run's lock was never freed (preempt cancel broken / deadlock)"
                    )
                });
            prev = Some(token);
        }

        // Release the final still-blocked run so it completes naturally.
        release_tx.send(()).expect("release final run");

        let mut panics = 0;
        for worker in workers {
            if worker.join().is_err() {
                panics += 1;
            }
        }
        let elapsed = started.elapsed();

        assert_eq!(panics, 0, "no preempt worker should panic");
        // A true deadlock would block a worker join (or the per-cycle
        // recv_timeout) indefinitely and blow well past this bound.
        assert!(
            elapsed < Duration::from_secs(30),
            "rapid preempt cycling must not deadlock; took {elapsed:?}"
        );

        // LOAD-BEARING: every run actually entered the runtime while holding the
        // write lock, so all N preemptions happened under genuine lock
        // contention (a replacement acquired a lock freed by the preemption it
        // triggered). This is the assertion that fails if preemption regresses
        // to head-of-line blocking or the test passes vacuously.
        let entered = runtime_entered.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            entered, N,
            "all {N} runs must enter the runtime under real lock contention; only {entered} did \
             (a vacuous pass / broken preemption would leave this short of {N})"
        );

        // Slot is in a consistent terminal state: the mutex is not poisoned
        // (the `.expect` proves it). The final run cleared its own token, so the
        // slot is empty.
        let slot = model
            .current_run
            .lock()
            .expect("slot mutex must not be poisoned after rapid cycling");
        assert!(
            slot.is_none(),
            "after the final run clears its own token, the slot must be empty"
        );
        drop(slot);

        // And the model write lock must be free — no run left a guard held.
        assert!(
            model.handle.try_write().is_ok(),
            "model write lock must be free after rapid preempt cycling (no held guard)"
        );
    }

    #[test]
    fn dispatch_after_local_stops_when_cloud_policy_is_denied() {
        let cloud = FakeCloudAdapter::new("should not run");
        let authority = FakeAuthority::deny("Policy rule 'rtt_rule' matched");
        let mut envelope = text_envelope("write a haiku");
        envelope
            .metadata
            .insert("model".to_string(), "deepseek-chat".to_string());
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let mut on_seam = |_s: SeamInfo| {};
        let local_result: SdkResult<InferenceResult> = Err(SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory,
        });

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-denied".to_string(),
            "local-model",
            7,
            321,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        );

        match result {
            Err(SdkError::InferenceError { message, .. }) => {
                assert!(message.contains("cloud_denied_by_policy"));
            }
            other => panic!("expected cloud_denied_by_policy error, got {other:?}"),
        }
        assert_eq!(cloud.call_count(), 0);

        let outcomes = authority.outcomes();
        assert_eq!(outcomes.len(), 2);
        assert!(matches!(
            outcomes[1].category,
            Some(xybrid_core::orchestrator::authority::OutcomeCategory::HardFail { ref reason })
                if reason == "cloud_denied_by_policy"
        ));
    }

    #[test]
    fn dispatch_after_local_uses_cloud_model_name_from_envelope() {
        // Regression: previously the cloud-leg `CloudRetry` event and the
        // returned `InferenceResult` both reported the LOCAL model_id
        // (e.g. `qwen2.5-0.5b-instruct`) for tokens that actually came from
        // the cloud-side model (e.g. `deepseek-chat`), making the dashboard
        // misattribute cloud output to the local model.
        let cloud = FakeCloudAdapter::new("hello from deepseek");
        let authority = FakeAuthority::allow();
        let mut envelope = text_envelope("write a haiku");
        envelope
            .metadata
            .insert("provider".to_string(), "deepseek".to_string());
        envelope
            .metadata
            .insert("model".to_string(), "deepseek-chat".to_string());
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let mut on_seam = |_s: SeamInfo| {};

        let local_result: SdkResult<InferenceResult> = Err(SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory,
        });

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-cloud-model".to_string(),
            "qwen2.5-0.5b-instruct",
            5,
            220,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        )
        .expect("cloud retry should succeed");

        assert_eq!(result.model_id(), "deepseek-chat");
    }

    #[test]
    fn dispatch_after_local_falls_back_to_local_model_when_envelope_missing_model() {
        // When no `model` metadata is on the envelope the dispatch would
        // typically fail at the gateway, but if it somehow succeeds we
        // shouldn't leave the result struct's model_id empty — fall back
        // to the local id so the trace at least labels something.
        let cloud = FakeCloudAdapter::new("ok");
        let authority = FakeAuthority::allow();
        let envelope = text_envelope("p");
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let mut on_seam = |_s: SeamInfo| {};
        let local_result: SdkResult<InferenceResult> = Err(SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory,
        });

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-fallback".to_string(),
            "local-only-model",
            0,
            0,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        )
        .expect("cloud retry should succeed");

        assert_eq!(result.model_id(), "local-only-model");
    }

    #[test]
    fn dispatch_after_local_passes_through_ok_result() {
        let cloud = FakeCloudAdapter::new("unused");
        let authority = FakeAuthority::allow();
        let envelope = text_envelope("prompt");
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let seam_fired = Arc::new(AtomicBool::new(false));
        let seam_fired_for_cb = seam_fired.clone();
        let mut on_seam = move |_s: SeamInfo| {
            seam_fired_for_cb.store(true, std::sync::atomic::Ordering::SeqCst);
        };

        let local_inner = InferenceResult::new(text_envelope("local result"), "test-model", 50);
        let local_result: SdkResult<InferenceResult> = Ok(local_inner);

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-2".to_string(),
            "test-model",
            10,
            200,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        )
        .expect("ok should pass through");

        assert!(!seam_fired.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(cloud.call_count(), 0);
        assert_eq!(result.text(), Some("local result"));
        assert_eq!(result.latency_ms(), 50);
    }

    #[test]
    fn dispatch_after_local_passes_through_other_errors() {
        let cloud = FakeCloudAdapter::new("unused");
        let authority = FakeAuthority::allow();
        let envelope = text_envelope("prompt");
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let seam_fired = Arc::new(AtomicBool::new(false));
        let seam_fired_for_cb = seam_fired.clone();
        let mut on_seam = move |_s: SeamInfo| {
            seam_fired_for_cb.store(true, std::sync::atomic::Ordering::SeqCst);
        };

        let local_result: SdkResult<InferenceResult> = Err(SdkError::inference("local failed"));

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-3".to_string(),
            "test-model",
            0,
            0,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        );

        match result {
            Err(SdkError::InferenceError { message: msg, .. }) => {
                assert!(msg.contains("local failed"))
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }
        assert!(!seam_fired.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(cloud.call_count(), 0);
    }

    #[test]
    fn dispatch_after_local_never_retries_user_cancelled_with_fallback_enabled() {
        let cloud = FakeCloudAdapter::new("must not run");
        let authority = FakeAuthority::allow();
        let envelope = text_envelope("prompt");
        let mut on_token =
            |_: xybrid_core::runtime_adapter::types::PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) };
        let seam_fired = Arc::new(AtomicBool::new(false));
        let seam_fired_for_cb = seam_fired.clone();
        let mut on_seam = move |_s: SeamInfo| {
            seam_fired_for_cb.store(true, std::sync::atomic::Ordering::SeqCst);
        };

        let cancellation_error =
            crate::run_options::AbortReason::UserCancelled.into_streaming_error(true);
        let local_result: SdkResult<InferenceResult> =
            Err(streaming_callback_error(cancellation_error));

        let result = dispatch_after_local(
            local_result,
            &envelope,
            &cloud,
            "corr-cancel".to_string(),
            "test-model",
            0,
            0,
            None,
            &authority,
            default_metrics(),
            Some(default_signal()),
            None,
            &mut on_token,
            &mut on_seam,
        );

        match result {
            Err(SdkError::InferenceError { message, source }) => {
                let full = source.map_or(message.clone(), |s| format!("{message}: {s}"));
                assert!(full.contains("user_cancelled"), "{full}");
            }
            other => panic!("expected terminal user_cancelled error, got {other:?}"),
        }
        assert!(!seam_fired.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(cloud.call_count(), 0);
    }
}
