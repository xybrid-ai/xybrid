//! OpenTelemetry observability for the Xybrid SDK.
//!
//! The SDK reports every model load, inference, streaming run, cloud attempt,
//! and pipeline through the portable [`xybrid_observability`] instrumentation:
//! GenAI-convention spans, metrics, and log events that any OpenTelemetry
//! backend accepts (Xybrid, Langfuse, Phoenix, a Collector).
//!
//! Export is **off until configured explicitly** — an API key alone no longer
//! starts it:
//!
//! ```no_run
//! use std::time::Duration;
//! use xybrid_sdk::observability::{self, ObservabilityConfig, OtlpConfig};
//!
//! // Traces to the Xybrid dashboard:
//! observability::init_observability(ObservabilityConfig::xybrid("xy_live_..."))?;
//! // ...or everything to an OpenTelemetry Collector:
//! observability::init_observability(ObservabilityConfig::otlp(OtlpConfig::collector(
//!     "http://localhost:4318",
//! )))?;
//! // At exit:
//! observability::shutdown_observability(Duration::from_secs(5))?;
//! # Ok::<(), xybrid_sdk::SdkError>(())
//! ```
//!
//! Applications that already own OpenTelemetry providers pass an
//! [`Observer`] built from them with [`ObservabilityConfig::with_observer`].
//! Prompts and outputs are captured once observability is enabled; use
//! [`ContentPolicy::metadata_only`] to record metadata only.
//!
//! Per run, [`ObservationOptions`] on `RunOptions` sets an explicit parent
//! (for example a `traceparent` from another process or language), a
//! conversation id, prompt metadata, and trace suppression. Results return
//! their [`ObservationRef`] for feedback and evaluations.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use xybrid_core::device::{ResourceTelemetryMode, ResourceUsageSummary, ThermalState};
use xybrid_core::execution::{ExecutionTemplate, ModelMetadata};
use xybrid_core::ir::{Envelope, EnvelopeKind, ImageSource};
use xybrid_core::tracing::{RecorderScope, TraceHeaders, TraceRecorder};

pub use xybrid_observability::otlp::{
    BatchLimits, OtlpConfig, PipelineStats, Protocol, ResourceConfig, RetryLimits, SignalEndpoint,
    SignalStats, XYBRID_TRACES_ENDPOINT,
};
pub use xybrid_observability::{
    semconv, AttemptInfo, AttemptOperation, AttemptReason, ContentCapture, ContentPolicy,
    ErrorInfo, Evaluation, Feedback, InferenceOperation, InferenceRequest, InputMessage, KeyValue,
    Location, MediaMetadata, Modality, ModelLoadInfo, ModelLoadOperation, ObservationContext,
    ObservationRef, Observer, ObserverBuilder, OutputMessage, Part, PromptRef, RedactionRule,
    Redactor, ResourceSample, SpanOperation, StartOptions, SubstringRedactor, ToolCallInfo,
    ToolOperation, TraceMode, Usage, WorkflowOperation,
};

use crate::model::{SdkError, SdkResult};

mod asr;
mod manual;
pub use manual::{
    start_inference, start_model_load, start_tool, start_workflow, ManualObservation,
};
mod session;
pub(crate) use asr::AsrObservation;
use session::resolve_session;
pub use session::{
    default_session, install_default_session, CapturedContext, CapturedScope, ObservabilitySession,
};

/// How observability is exported.
#[derive(Clone)]
enum Exporter {
    Otlp(Box<OtlpConfig>),
    Observer(Observer),
}

/// Observability configuration for [`init_observability`].
#[derive(Clone)]
#[must_use]
pub struct ObservabilityConfig {
    exporter: Exporter,
    content: Option<ContentPolicy>,
    resource_telemetry: ResourceTelemetryMode,
    live_trace_interval: Duration,
    device_attributes: bool,
    device_id: Option<String>,
    provider_origins: Vec<String>,
}

impl ObservabilityConfig {
    /// Export through the packaged OTLP/HTTP pipeline.
    pub fn otlp(config: OtlpConfig) -> Self {
        Self {
            exporter: Exporter::Otlp(Box::new(config)),
            content: None,
            resource_telemetry: ResourceTelemetryMode::Off,
            live_trace_interval: Duration::from_secs(1),
            device_attributes: true,
            device_id: None,
            provider_origins: Vec::new(),
        }
    }

    /// Xybrid dashboard preset: traces only, bearer-authenticated. Metrics and
    /// logs are not sent; use [`ObservabilityConfig::otlp`] with a Collector
    /// to route them to your own stores.
    pub fn xybrid(api_key: &str) -> Self {
        Self::otlp(OtlpConfig::xybrid_traces(api_key))
    }

    /// Xybrid preset against a custom trace endpoint (staging, self-hosted).
    pub fn xybrid_at(traces_endpoint: &str, api_key: &str) -> Self {
        Self::otlp(OtlpConfig::xybrid_traces_to(traces_endpoint, api_key))
    }

    /// Use an observer built from application-owned providers. The SDK never
    /// flushes or shuts those providers down; the application does.
    pub fn with_observer(observer: Observer) -> Self {
        Self {
            exporter: Exporter::Observer(observer),
            content: None,
            resource_telemetry: ResourceTelemetryMode::Off,
            live_trace_interval: Duration::from_secs(1),
            device_attributes: true,
            device_id: None,
            provider_origins: Vec::new(),
        }
    }

    /// Configuration from standard `OTEL_EXPORTER_OTLP_*` variables, or `None`
    /// when no endpoint is configured.
    pub fn from_env() -> Option<Self> {
        OtlpConfig::from_env().map(Self::otlp)
    }

    /// Sets the content policy (packaged pipeline only; an application-owned
    /// observer keeps its own policy).
    pub fn content(mut self, policy: ContentPolicy) -> Self {
        self.content = Some(policy);
        self
    }

    /// Samples device resources (CPU, memory, thermal, battery) around runs
    /// and records them on spans. Defaults to `Off`.
    pub fn resource_telemetry(mut self, mode: ResourceTelemetryMode) -> Self {
        self.resource_telemetry = mode;
        self
    }

    /// Minimum interval between traced frames of one live session. Every frame
    /// is still counted in metrics. Defaults to one second.
    pub fn live_trace_interval(mut self, interval: Duration) -> Self {
        self.live_trace_interval = interval;
        self
    }

    /// Whether to attach detected device attributes (OS, architecture, chip,
    /// RAM) to the resource. Defaults to `true`; hostnames are never sent.
    pub fn device_attributes(mut self, enabled: bool) -> Self {
        self.device_attributes = enabled;
        self
    }

    /// Sets a caller-owned device identifier. Never synthesized from hardware.
    pub fn device_id(mut self, id: impl Into<String>) -> Self {
        self.device_id = Some(id.into());
        self
    }

    /// Allows context propagation to one exact third-party HTTP(S) origin.
    /// Xybrid-owned hops propagate automatically. Provider propagation defaults off.
    ///
    /// # Errors
    /// Returns a configuration error unless `origin` is an absolute HTTP(S) origin.
    pub fn propagate_context_to(mut self, origin: &str) -> SdkResult<Self> {
        let origin = xybrid_core::tracing::provider_origin(origin).ok_or_else(|| {
            SdkError::ConfigError("provider propagation requires an HTTP(S) origin".into())
        })?;
        if !self.provider_origins.contains(&origin) {
            self.provider_origins.push(origin);
        }
        Ok(self)
    }

    /// Sets the service name and version reported as the OTel resource.
    pub fn service(mut self, name: impl Into<String>, version: Option<String>) -> Self {
        if let Exporter::Otlp(config) = &mut self.exporter {
            config.resource.service_name = Some(name.into());
            config.resource.service_version = version;
        }
        self
    }
}

impl fmt::Debug for ObservabilityConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exporter = match &self.exporter {
            Exporter::Otlp(c) => format!("{c:?}"),
            Exporter::Observer(o) => format!("{o:?}"),
        };
        f.debug_struct("ObservabilityConfig")
            .field("exporter", &exporter)
            .field("content", &self.content)
            .field("resource_telemetry", &self.resource_telemetry)
            .field("live_trace_interval", &self.live_trace_interval)
            .finish()
    }
}

/// Creates and installs the helper-owned default session.
///
/// A previous helper-owned pipeline with no operation pinned to it shuts down
/// before this returns (bounded to one second); one still pinned by in-flight
/// operations drains on a background worker after the last of them releases
/// it. Application-installed sessions retain lifecycle control.
///
/// # Errors
/// [`SdkError::ConfigError`] for invalid configuration.
pub fn init_observability(config: ObservabilityConfig) -> SdkResult<()> {
    let session = ObservabilitySession::new(config)?;
    if let Some(previous) = session::replace_default(session, true) {
        session::retire_replaced(previous);
    }
    Ok(())
}

/// The default observer. Disabled until explicitly configured.
pub fn observer() -> Observer {
    default_session().map_or_else(Observer::disabled, |s| s.observer().clone())
}

/// Whether the default observer is enabled.
pub fn is_enabled() -> bool {
    observer().is_enabled()
}

/// Per-signal delivery counters, unavailable for borrowed providers.
pub fn stats() -> Option<PipelineStats> {
    default_session().and_then(|s| s.stats())
}

/// Flushes the default session's owned pipeline.
///
/// # Errors
/// Returns timeout or delivery errors from the pipeline.
pub fn flush_observability(timeout: Duration) -> SdkResult<()> {
    default_session().map_or(Ok(()), |s| s.flush(timeout))
}

/// Async [`flush_observability`].
///
/// # Errors
/// As for [`flush_observability`].
pub async fn flush_observability_async(timeout: Duration) -> SdkResult<()> {
    match default_session() {
        Some(s) => s.flush_async(timeout).await,
        None => Ok(()),
    }
}

/// Uninstalls the default session, shutting down a helper-owned pipeline.
/// Application-installed sessions retain application-controlled lifecycle.
///
/// # Errors
/// Returns timeout or delivery errors from a helper-owned pipeline.
pub fn shutdown_observability(timeout: Duration) -> SdkResult<()> {
    match session::take_default() {
        Some(slot) if slot.helper_owned => slot.session.shutdown(timeout),
        _ => Ok(()),
    }
}

/// Async [`shutdown_observability`].
///
/// # Errors
/// As for [`shutdown_observability`].
pub async fn shutdown_observability_async(timeout: Duration) -> SdkResult<()> {
    match session::take_default() {
        Some(slot) if slot.helper_owned => slot.session.shutdown_async(timeout).await,
        _ => Ok(()),
    }
}

/// Records end-user feedback linked to a run (`InferenceResult::observation_ref`)
/// as a `gen_ai.evaluation.result` log event. Returns `false` when no logs
/// signal is configured (the Xybrid trace preset sends traces only).
pub fn record_feedback(target: &ObservationRef, feedback: &Feedback) -> bool {
    observer().record_feedback(target, feedback)
}

/// Records an evaluation result linked to a run. Returns `false` when no logs
/// signal is configured.
pub fn record_evaluation(target: &ObservationRef, evaluation: &Evaluation) -> bool {
    observer().record_evaluation(target, evaluation)
}

fn lifecycle_error(e: xybrid_observability::Error, timeout: Duration) -> SdkError {
    if e.is_timeout() {
        SdkError::Timeout {
            timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
        }
    } else {
        SdkError::TelemetryDelivery(e.to_string())
    }
}

/// The OpenTelemetry resource attributes the SDK adds to sessions it builds:
/// `telemetry.sdk.name`, `xybrid.sdk.*`, detected device attributes when
/// `device_attributes` is set (never hostnames), and a caller-owned
/// `device.id` when given.
///
/// Hosts that build their own provider around the SDK (the CLI's local trace
/// capture, for example) use this to describe themselves the same way.
///
/// # Errors
/// [`SdkError::ConfigError`] unless `device_id` has 1–128 characters and no
/// control characters.
pub fn resource_attributes(
    device_attributes: bool,
    device_id: Option<&str>,
) -> SdkResult<Vec<KeyValue>> {
    let mut attributes = sdk_resource_attributes(device_attributes);
    if let Some(id) = device_id {
        attributes.push(KeyValue::new("device.id", validated_device_id(id)?));
    }
    Ok(attributes)
}

pub(crate) fn validated_device_id(id: &str) -> SdkResult<String> {
    if id.is_empty() || id.chars().count() > 128 || id.chars().any(char::is_control) {
        return Err(SdkError::ConfigError(
            "device.id must contain 1–128 characters without controls".into(),
        ));
    }
    Ok(id.to_owned())
}

fn sdk_resource_attributes(device: bool) -> Vec<KeyValue> {
    let mut attrs = vec![
        KeyValue::new("telemetry.sdk.name", "xybrid"),
        KeyValue::new("xybrid.sdk.version", crate::SDK_VERSION),
        KeyValue::new("xybrid.sdk.binding", crate::get_binding()),
    ];
    if device && !crate::telemetry_optout::is_telemetry_opted_out() {
        let profile = xybrid_core::device::DeviceProfile::detect();
        if let Some(os) = profile.os {
            attrs.push(KeyValue::new("os.type", os));
        }
        if let Some(v) = profile.os_version {
            attrs.push(KeyValue::new("os.version", v));
        }
        if let Some(v) = profile.arch {
            attrs.push(KeyValue::new("host.arch", v));
        }
        if let Some(v) = profile.chip_family {
            attrs.push(KeyValue::new("xybrid.device.chip", v));
        }
        if let Some(v) = profile.ram_gb {
            attrs.push(KeyValue::new("xybrid.device.ram_gb", i64::from(v)));
        }
    }
    attrs
}

/// Per-run observation options (`RunOptions::observation`).
#[derive(Clone, Debug, Default)]
pub struct ObservationOptions {
    /// Explicit parent; wins over the ambient OpenTelemetry context.
    pub parent: Option<ObservationContext>,
    /// Caller-owned conversation id (never synthesized).
    pub conversation_id: Option<String>,
    /// Prompt template metadata.
    pub prompt: Option<PromptRef>,
    /// Trace hint. Metrics are recorded either way.
    pub trace: TraceMode,
    /// Session override, retained for the operation and all its children.
    pub session: Option<ObservabilitySession>,
    /// Observer override for this run (borrowed provider lifecycle).
    pub observer: Option<Observer>,
    /// Extra span attributes.
    pub attributes: Vec<KeyValue>,
}

impl ObservationOptions {
    /// Options parented to a W3C `traceparent` (and optional `tracestate`).
    ///
    /// # Errors
    ///
    /// [`SdkError::ConfigError`] when `traceparent` is invalid.
    pub fn from_traceparent(traceparent: &str, tracestate: Option<&str>) -> SdkResult<Self> {
        let parent = ObservationContext::from_traceparent(traceparent, tracestate)
            .map_err(|e| SdkError::ConfigError(e.to_string()))?;
        Ok(Self {
            parent: Some(parent),
            ..Self::default()
        })
    }

    /// Pins this operation to an application-owned session.
    #[must_use]
    pub fn with_session(mut self, session: &ObservabilitySession) -> Self {
        self.session = Some(session.clone());
        self
    }

    /// Sets the parent.
    #[must_use]
    pub fn with_parent(mut self, parent: ObservationContext) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Sets the conversation id.
    #[must_use]
    pub fn with_conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Run instrumentation
// ---------------------------------------------------------------------------

/// What a model run is, in GenAI terms.
#[derive(Clone, Debug)]
pub(crate) struct RunSubject {
    pub operation: &'static str,
    pub task: Option<String>,
    pub runtime: Option<&'static str>,
    pub model: String,
    pub format: Option<&'static str>,
    pub size_bytes: Option<u64>,
    pub input_modalities: Vec<Modality>,
    pub output_modalities: Vec<Modality>,
    pub output_type: Option<&'static str>,
    pub is_llm: bool,
}

/// Maps the runtime wire label to the contract's runtime name.
pub(crate) fn runtime_name(label: &str) -> &'static str {
    match label {
        "ort" | "onnx" => "onnxruntime",
        "llamacpp" | "llama.cpp" => "llama.cpp",
        "mistralrs" | "mistral" => "mistral.rs",
        "whispercpp" => "whisper.cpp",
        "candle" => "candle",
        "zzz" => "zzz",
        "coreml" => "coreml",
        "cloud" | "gateway" => "http",
        _ => "unknown",
    }
}

impl RunSubject {
    pub(crate) fn from_metadata(model_id: &str, metadata: &ModelMetadata) -> Self {
        let hint = metadata.metadata.get("backend").and_then(|v| v.as_str());
        let runtime = xybrid_core::execution::template::backend_label_from_template(
            &metadata.execution_template,
            hint,
        )
        .map(runtime_name);
        let task = metadata
            .metadata
            .get("task")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let task_l = task.as_deref().unwrap_or("").to_ascii_lowercase();
        let (operation, task, inputs, outputs, output_type, format, is_llm) =
            match &metadata.execution_template {
                ExecutionTemplate::Gguf { .. } => (
                    "chat",
                    Some("text_generation".into()),
                    vec![Modality::Text],
                    vec![Modality::Text],
                    None,
                    Some("gguf"),
                    true,
                ),
                ExecutionTemplate::VisionLanguage { .. } => (
                    "chat",
                    Some("vision_language".into()),
                    vec![Modality::Text, Modality::Image],
                    vec![Modality::Text],
                    None,
                    Some("gguf"),
                    true,
                ),
                ExecutionTemplate::GgmlWhisper { .. } => (
                    "generate_content",
                    Some("speech_to_text".into()),
                    vec![Modality::Audio],
                    vec![Modality::Text],
                    None,
                    Some("ggml"),
                    false,
                ),
                ExecutionTemplate::ZzzEmbed { .. } | ExecutionTemplate::ZzzPhonon { .. } => (
                    "generate_content",
                    Some("text_to_speech".into()),
                    vec![Modality::Text],
                    vec![Modality::Audio],
                    Some("speech"),
                    None,
                    false,
                ),
                ExecutionTemplate::ChoiceScorer { .. } => (
                    "predict",
                    Some("decision".into()),
                    vec![Modality::Text],
                    Vec::new(),
                    None,
                    Some("onnx"),
                    false,
                ),
                template => {
                    let format = match template {
                        ExecutionTemplate::Onnx { .. } => Some("onnx"),
                        ExecutionTemplate::SafeTensors { .. } => Some("safetensors"),
                        ExecutionTemplate::CoreMl { .. } => Some("coreml"),
                        ExecutionTemplate::TfLite { .. } => Some("tflite"),
                        ExecutionTemplate::LiteRtLm { .. } => Some("litertlm"),
                        _ => None,
                    };
                    if (task_l.contains("speech") && task_l.starts_with("text"))
                        || task_l.contains("tts")
                    {
                        (
                            "generate_content",
                            Some("text_to_speech".into()),
                            vec![Modality::Text],
                            vec![Modality::Audio],
                            Some("speech"),
                            format,
                            false,
                        )
                    } else if task_l.contains("speech")
                        || task_l.contains("asr")
                        || task_l.contains("transcri")
                    {
                        (
                            "generate_content",
                            Some("speech_to_text".into()),
                            vec![Modality::Audio],
                            vec![Modality::Text],
                            None,
                            format,
                            false,
                        )
                    } else if task_l.contains("embed") {
                        (
                            "embeddings",
                            Some("embedding".into()),
                            vec![Modality::Text],
                            Vec::new(),
                            None,
                            format,
                            false,
                        )
                    } else if task_l.contains("generation") || task_l.contains("chat") {
                        (
                            "chat",
                            task.clone(),
                            vec![Modality::Text],
                            vec![Modality::Text],
                            None,
                            format,
                            true,
                        )
                    } else {
                        (
                            "predict",
                            task.clone(),
                            Vec::new(),
                            Vec::new(),
                            None,
                            format,
                            false,
                        )
                    }
                }
            };
        Self {
            operation,
            task,
            runtime,
            model: model_id.to_owned(),
            format,
            size_bytes: declared_model_size(metadata),
            input_modalities: inputs,
            output_modalities: outputs,
            output_type,
            is_llm,
        }
    }

    pub(crate) fn pipeline_stage(stage: &xybrid_core::context::StageDescriptor) -> Self {
        let model = stage.model.as_deref().unwrap_or(&stage.name);
        if let Some(metadata) = stage
            .bundle_path
            .as_deref()
            .and_then(|path| {
                std::fs::read_to_string(std::path::Path::new(path).join("model_metadata.json")).ok()
            })
            .and_then(|json| serde_json::from_str::<ModelMetadata>(&json).ok())
        {
            return Self::from_metadata(model, &metadata);
        }
        // Integration stages are served by core's chat-completions client
        // (gateway or direct); ElevenLabs is speech synthesis. Other stages
        // without local metadata report only facts known at dispatch: never
        // infer modalities or a task from an id.
        match stage.provider {
            Some(xybrid_core::pipeline::IntegrationProvider::ElevenLabs) => {
                return Self {
                    operation: "generate_content",
                    task: Some("text_to_speech".into()),
                    runtime: Some("http"),
                    model: model.to_owned(),
                    format: None,
                    size_bytes: None,
                    input_modalities: vec![Modality::Text],
                    output_modalities: vec![Modality::Audio],
                    output_type: Some("speech"),
                    is_llm: false,
                };
            }
            Some(_) => return Self::remote_chat(model),
            None => {}
        }
        Self {
            operation: "predict",
            task: None,
            runtime: stage.is_cloud().then_some("http"),
            model: model.to_owned(),
            format: None,
            size_bytes: None,
            input_modalities: Vec::new(),
            output_modalities: Vec::new(),
            output_type: None,
            is_llm: false,
        }
    }

    /// A subject for a remote (gateway) leg serving `model`.
    pub(crate) fn remote_chat(model: &str) -> Self {
        Self {
            operation: "chat",
            task: Some("text_generation".into()),
            runtime: Some("http"),
            model: model.to_owned(),
            format: None,
            size_bytes: None,
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Text],
            output_type: None,
            is_llm: true,
        }
    }
}

/// Inputs to [`RunObservation::start`].
pub(crate) struct RunStart<'a> {
    pub subject: RunSubject,
    pub input: Option<&'a Envelope>,
    pub history: &'a [Envelope],
    pub options: Option<&'a crate::run_options::RunOptions>,
    pub generation: Option<&'a xybrid_core::runtime_adapter::types::GenerationConfig>,
    pub streaming: bool,
    pub location: Location,
    /// Parent captured on the caller's thread before a hop to a blocking
    /// worker. Explicit `ObservationOptions::parent` still wins.
    pub parent: Option<ObservationContext>,
    /// Provider override (defaults to the runtime locally, `xybrid` remotely).
    pub provider: Option<&'a str>,
    /// Remote server address and port, when known.
    pub server: Option<(Option<String>, Option<i64>)>,
}

/// Observation of one SDK run: the logical inference span plus a stage
/// recorder installed on the executing thread.
pub(crate) struct RunObservation {
    op: Arc<InferenceOperation>,
    stages: Arc<StageRecorder>,
    _scope: Option<RecorderScope>,
    started: SystemTime,
    subject: RunSubject,
    server: Option<(Option<String>, Option<i64>)>,
    /// Shared with attempt recorders, whose transport retries start attempts too.
    attempts: Arc<std::sync::atomic::AtomicU32>,
}

/// Records streamed chunks from token callbacks (which must be `Send`).
#[derive(Clone)]
pub(crate) struct ChunkSink {
    op: Arc<InferenceOperation>,
}

impl ChunkSink {
    /// One streamed output chunk (token callback, audio packet). Performs no
    /// I/O. A chunk never sets first-token time: that comes only from the
    /// runtime's measured `ttft_ms` (see `RunObservation::apply_harvest`).
    pub(crate) fn chunk(&self) {
        self.op.record_chunk();
    }
}

/// Where an attempt ran.
pub(crate) struct AttemptSpec<'a> {
    pub reason: AttemptReason,
    pub location: Location,
    /// Defaults to the local runtime, or `xybrid` (the gateway) for remote.
    pub provider: Option<&'a str>,
    pub model: Option<&'a str>,
}

/// One concrete execution attempt of a run (speculative cloud leg, reactive
/// fallback). Outbound requests made on this thread while it lives carry the
/// attempt's trace context; stages become its children. A transport retry
/// inside it becomes the run's next (`retry`) attempt (see [`AttemptChain`]).
pub(crate) struct AttemptScope {
    chain: Arc<AttemptChain>,
    stages: Arc<StageRecorder>,
    parent_stages: Arc<StageRecorder>,
    started: SystemTime,
    logical_started: SystemTime,
    _scope: Option<RecorderScope>,
}

impl AttemptScope {
    /// Ends the attempt with the outcome of its leg.
    pub(crate) fn end(self, result: &SdkResult<crate::result::InferenceResult>) {
        self.end_with_output(result, result.as_ref().ok().map(|r| r.envelope()));
    }

    fn end_with_output<T>(mut self, result: &SdkResult<T>, output: Option<&Envelope>) {
        self._scope.take();
        // The live attempt: the initial one, or the last retry core started.
        let op = self.chain.finish();
        let mut h = self.stages.harvest();
        if let Some(output) = output {
            // Runtime envelopes may carry the only usage report. Absorb it
            // within its attempt before folding distinct attempts into totals.
            h.absorb(
                output
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str())),
            );
        }
        let usage = Usage {
            input_tokens: h.tokens_in,
            output_tokens: h.tokens_out,
            cache_read_input_tokens: h.cached_tokens,
            ..Usage::default()
        };
        if let Some(op) = op.as_ref().filter(|_| !usage.is_empty()) {
            op.record_usage(&usage);
        }
        // Runtime TTFT is relative to this attempt, never the earlier logical
        // start. Failed attempts cannot establish delivered-token timing.
        h.ttft_ms = h.ttft_ms.filter(|_| result.is_ok()).and_then(|ms| {
            self.started
                .duration_since(self.logical_started)
                .ok()
                .map(|offset| offset.as_secs_f64() * 1000.0 + ms)
        });
        self.parent_stages.merge(&h);
        let Some(op) = op else {
            return;
        };
        match result {
            Ok(_) => op.end(),
            Err(SdkError::Cancelled { .. }) => op.cancel(),
            Err(e) => op.end_with_error(&error_info(e)),
        }
    }
}

/// The live attempt behind an attempt recorder.
///
/// Core reports each transport retry it makes inside the attempt
/// ([`xybrid_core::tracing::record_request_retry`]); the live attempt then
/// ends as failed with the try's error class and the run's next attempt
/// (reason `retry`, same target) replaces it. Requests and stages started
/// afterwards use the new attempt's context, so every try has its own parent
/// span and `traceparent`. Stages already open stay under the attempt they
/// started in.
pub(crate) struct AttemptChain {
    run: Arc<InferenceOperation>,
    counter: Arc<std::sync::atomic::AtomicU32>,
    /// The first attempt's target; retries keep it with a new index and reason.
    target: AttemptInfo,
    observer: Observer,
    /// The live attempt and when it started.
    live: Mutex<Option<(AttemptOperation, SystemTime)>>,
}

impl AttemptChain {
    /// Starts the chain's first attempt (`target`) at `started`.
    fn new(
        run: &Arc<InferenceOperation>,
        counter: &Arc<std::sync::atomic::AtomicU32>,
        target: AttemptInfo,
        observer: &Observer,
        started: SystemTime,
    ) -> Self {
        Self {
            live: Mutex::new(Some((
                run.start_attempt_at(target.clone(), started),
                started,
            ))),
            run: run.clone(),
            counter: counter.clone(),
            target,
            observer: observer.clone(),
        }
    }

    fn live(&self) -> std::sync::MutexGuard<'_, Option<(AttemptOperation, SystemTime)>> {
        self.live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Context of the live attempt, for its stages and outbound requests.
    fn context(&self) -> Option<ObservationContext> {
        self.live().as_ref().and_then(|(op, _)| op.context())
    }

    /// Ends the live attempt as failed when its try failed (`failed_at`) and
    /// starts the next `retry` attempt now, so the backoff between the two
    /// belongs to neither. A no-op once the attempt scope has finished.
    fn retry(&self, error_type: &str, failed_at: std::time::Instant) {
        let mut live = self.live();
        let Some((failed, started)) = live.take() else {
            return;
        };
        let now = self.observer.now();
        let failed_end = now
            .checked_sub(failed_at.elapsed())
            .map_or(now, |end| end.max(started).min(now));
        failed.end_with_error_at(&ErrorInfo::new(error_type), failed_end);
        let mut next = self.target.clone();
        next.index = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        next.reason = AttemptReason::Retry;
        *live = Some((self.run.start_attempt_at(next, now), now));
    }

    /// Takes the live attempt for its final outcome.
    fn finish(&self) -> Option<AttemptOperation> {
        self.live().take().map(|(op, _)| op)
    }
}

/// Makes `parent` the ambient parent on this thread (for work moved to a
/// blocking worker): runs started while it lives become its children.
pub(crate) struct ParentScope {
    _scope: RecorderScope,
}

impl ParentScope {
    pub(crate) fn install(parent: ObservationContext) -> Self {
        Self {
            _scope: RecorderScope::install(Arc::new(StageRecorder::new(observer(), Some(parent)))),
        }
    }
}

impl AttemptScope {
    /// Ends the attempt with the outcome of a runtime adapter call (errors
    /// are recorded as `inference_error`).
    pub(crate) fn end_adapter<E: fmt::Display>(self, result: &Result<Envelope, E>) {
        let mapped: SdkResult<()> = match result {
            Ok(_) => Ok(()),
            Err(e) => Err(SdkError::inference(e.to_string())),
        };
        self.end_with_output(&mapped, result.as_ref().ok());
    }
}

/// A pipeline run: an `invoke_workflow` span whose execution stages (and any
/// model runs started beneath it) become its children.
pub(crate) struct WorkflowScope {
    op: WorkflowOperation,
    _scope: Option<RecorderScope>,
}

impl WorkflowScope {
    pub(crate) fn start(name: &str, options: &crate::run_options::RunOptions) -> Self {
        let obs_opts = options.observation.as_ref();
        let session = resolve_session(obs_opts);
        let observer = session.observer().clone();
        let mut start = StartOptions::default().with_trace(option_trace(obs_opts));
        start.parent = obs_opts
            .and_then(|o| o.parent.clone())
            .or_else(ambient_parent);
        if let Some(o) = obs_opts {
            start.attributes = o.attributes.clone();
        }
        if let Some(id) = &options.correlation_id {
            start
                .attributes
                .push(KeyValue::new("xybrid.correlation_id", id.clone()));
        }
        let conversation_id = conversation_id(obs_opts);
        if let Some(id) = &conversation_id {
            start
                .attributes
                .push(KeyValue::new(semconv::attr::CONVERSATION_ID, id.clone()));
        }
        let start_suppressed = start.trace == TraceMode::Suppress;
        let op = observer.start_workflow(name, start);
        let stages = Arc::new(
            StageRecorder::for_session(session, op.context(), start_suppressed)
                .with_conversation_id(conversation_id),
        );
        let scope = RecorderScope::install(stages);
        Self {
            op,
            _scope: Some(scope),
        }
    }

    /// Carries the run's controls into an inference nested under this workflow.
    /// An explicit caller parent belongs to the workflow; its inference is a child of it.
    pub(crate) fn child_options(
        &self,
        options: &crate::run_options::RunOptions,
    ) -> crate::run_options::RunOptions {
        let mut child = options.clone();
        if let Some(observation) = child.observation.as_mut() {
            observation.parent = self.op.context();
        }
        child
    }

    /// The workflow's context, the parent of its stages.
    pub(crate) fn context(&self) -> Option<ObservationContext> {
        self.op.context()
    }

    /// Ends the workflow with `result`, returning its reference on success.
    pub(crate) fn end<T>(mut self, result: &SdkResult<T>) -> Option<ObservationRef> {
        self._scope.take();
        let reference = self.op.observation_ref();
        match result {
            Ok(_) => {
                self.op.end();
                reference
            }
            Err(e) if is_cancellation(e) => {
                self.op.cancel();
                None
            }
            Err(e) => {
                self.op.end_with_error(&error_info(e));
                None
            }
        }
    }
}

/// A generic span with a stage recorder (warmup, pipelines' internal steps).
pub(crate) struct SpanScope {
    span: SpanOperation,
    _scope: Option<RecorderScope>,
}

impl SpanScope {
    /// Starts `name` under the ambient run (if any) and records stages
    /// beneath it.
    pub(crate) fn start(name: &str, attributes: Vec<KeyValue>) -> Self {
        Self::start_loading(name, attributes, None)
    }

    /// [`Self::start`], reporting weights core loads beneath it as
    /// `load_model` operations of `load`.
    pub(crate) fn start_loading(
        name: &str,
        attributes: Vec<KeyValue>,
        load: Option<Arc<LoadSubject>>,
    ) -> Self {
        let session = resolve_session(None);
        let observer = session.observer().clone();
        let options = StartOptions::default()
            .with_parent(ambient_parent())
            .with_trace(inherited_trace(TraceMode::Default));
        let span = observer.start_span(name.to_owned(), attributes, options);
        let mut stages =
            StageRecorder::for_session(session, span.context(), span.context().is_none());
        stages.load = load;
        let scope = RecorderScope::install(Arc::new(stages));
        Self {
            span,
            _scope: Some(scope),
        }
    }

    /// The span's context.
    pub(crate) fn context(&self) -> Option<ObservationContext> {
        self.span.context()
    }

    /// Ends the span with `result`.
    pub(crate) fn end<T>(mut self, result: &SdkResult<T>) {
        self._scope.take();
        match result {
            Ok(_) => self.span.end(),
            Err(e) => self.span.end_with_error(&error_info(e)),
        }
    }
}

impl RunObservation {
    /// Starts observing a run on the current thread.
    pub(crate) fn start(run: RunStart<'_>) -> Self {
        let obs_opts = run.options.and_then(|o| o.observation.as_ref());
        let observation_session = resolve_session(obs_opts);
        let observer = observation_session.observer().clone();
        let mut trace = option_trace(obs_opts);
        if let Some(opts) = run.options {
            if opts.live_mode && trace == TraceMode::Default {
                if let Some(session) = opts.frame_session_id.as_deref() {
                    trace = observation_session.live_trace_mode(session);
                }
            }
        }
        let subject = run.subject;
        let mut request = InferenceRequest::new(
            subject.operation,
            run.provider.unwrap_or(match run.location {
                Location::Remote => "xybrid",
                Location::Local => subject.runtime.unwrap_or("xybrid"),
            }),
        );
        let server = run.server;
        if let (Location::Remote, Some((address, port))) = (run.location, server.clone()) {
            request.server_address = address;
            request.server_port = port;
        }
        request.model = Some(subject.model.clone());
        request.location = Some(run.location);
        request.runtime = subject.runtime.map(str::to_owned);
        request.task = subject.task.clone();
        request.stream = run.streaming;
        request.input_modalities = subject.input_modalities.clone();
        request.output_modalities = subject.output_modalities.clone();
        request.output_type = subject.output_type.map(str::to_owned);
        if let Some(cfg) = run.generation {
            request.max_tokens = Some(i64::from(cfg.max_tokens as u32));
            request.temperature = Some(f64::from(cfg.temperature));
            request.top_p = Some(f64::from(cfg.top_p));
            if cfg.top_k > 0 {
                request.top_k = Some(i64::from(cfg.top_k as u32));
            }
            if !cfg.stop_sequences.is_empty() {
                request.stop_sequences = cfg.stop_sequences.clone();
            }
        }
        request.conversation_id = conversation_id(obs_opts);
        if let Some(o) = obs_opts {
            request.prompt = o.prompt.clone();
        }
        let mut start = StartOptions::default().with_trace(trace);
        start.parent = obs_opts
            .and_then(|o| o.parent.clone())
            .or(run.parent)
            .or_else(ambient_parent);
        if let Some(o) = obs_opts {
            start.attributes = o.attributes.clone();
        }
        if let Some(opts) = run.options {
            if let Some(id) = &opts.correlation_id {
                start
                    .attributes
                    .push(KeyValue::new("xybrid.correlation_id", id.clone()));
            }
            if let Some(id) = &opts.frame_session_id {
                start
                    .attributes
                    .push(KeyValue::new("xybrid.live.session_id", id.clone()));
            }
        }
        if let Some(format) = subject.format {
            start
                .attributes
                .push(KeyValue::new("xybrid.genai.model.format", format));
        }
        let started = observer.now();
        start.start_time = Some(started);
        let op = observer.start_inference(request, start);

        let (system, messages) = input_messages(run.history, run.input);
        if !system.is_empty() {
            op.record_system_instructions(&system);
        }
        if !messages.is_empty() {
            op.record_input_messages(&messages);
        }
        if let Some(definitions) = run.generation.and_then(tool_definitions) {
            op.record_tool_definitions(&definitions);
        }

        let stages = Arc::new(
            StageRecorder::for_session(
                observation_session,
                op.context(),
                trace == TraceMode::Suppress,
            )
            .with_conversation_id(conversation_id(obs_opts))
            .with_load(LoadSubject::new(
                &subject.model,
                subject.format,
                "lazy",
                subject.size_bytes,
            )),
        );
        let scope = RecorderScope::install(stages.clone());
        Self {
            op: Arc::new(op),
            stages,
            _scope: Some(scope),
            started,
            subject,
            server,
            attempts: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// A `Send` recorder for streamed chunks.
    pub(crate) fn chunk_sink(&self) -> ChunkSink {
        ChunkSink {
            op: self.op.clone(),
        }
    }

    /// Starts the next execution attempt of this run.
    pub(crate) fn attempt(&self, spec: AttemptSpec<'_>) -> AttemptScope {
        let index = self
            .attempts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (server_address, server_port) = match spec.location {
            Location::Remote => self.server.clone().unwrap_or_else(gateway_address),
            Location::Local => (None, None),
        };
        let started = self.stages.observer.now();
        let target = AttemptInfo {
            index,
            reason: spec.reason,
            provider: spec
                .provider
                .or(match spec.location {
                    Location::Remote => Some("xybrid"),
                    Location::Local => self.subject.runtime,
                })
                .unwrap_or("xybrid")
                .to_owned(),
            model: spec.model.map(str::to_owned),
            location: spec.location,
            runtime: match spec.location {
                Location::Remote => Some("http".to_owned()),
                Location::Local => self.subject.runtime.map(str::to_owned),
            },
            server_address,
            server_port,
        };
        let chain = Arc::new(AttemptChain::new(
            &self.op,
            &self.attempts,
            target,
            &self.stages.observer,
            started,
        ));
        let mut stages = StageRecorder::for_session(
            self.stages.session.clone(),
            chain.context(),
            self.stages.suppressed,
        );
        stages.load = self.stages.load.clone();
        stages.conversation_id = self.stages.conversation_id.clone();
        stages.attempt = Some(chain.clone());
        let stages = Arc::new(stages);
        let scope = RecorderScope::install(stages.clone());
        AttemptScope {
            chain,
            stages,
            parent_stages: self.stages.clone(),
            started,
            logical_started: self.started,
            _scope: Some(scope),
        }
    }

    /// Number of attempts started so far.
    pub(crate) fn attempt_count(&self) -> u32 {
        self.attempts.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Ends the run from its result, attaching the observation reference to
    /// a successful result.
    pub(crate) fn finish(
        self,
        result: SdkResult<crate::result::InferenceResult>,
    ) -> SdkResult<crate::result::InferenceResult> {
        match result {
            Ok(r) => {
                let reference = self.succeed(r.envelope());
                Ok(r.with_observation(reference))
            }
            Err(e) => {
                self.fail(&e);
                Err(e)
            }
        }
    }

    /// The logical span's context, for children and outbound requests.
    pub(crate) fn context(&self) -> Option<ObservationContext> {
        self.op.context()
    }

    /// The operation handle (attempts, extra attributes).
    pub(crate) fn operation(&self) -> &InferenceOperation {
        &self.op
    }

    /// Records a resource summary measured around the run.
    pub(crate) fn resource(&self, summary: Option<&ResourceUsageSummary>) {
        if let Some(sample) = summary.map(resource_sample) {
            self.op.record_resource(&sample);
        }
    }

    /// Ends the run successfully and returns its reference.
    pub(crate) fn succeed(self, output: &Envelope) -> Option<ObservationRef> {
        self.apply_harvest(Some(output));
        if let Some(message) = output_message(output) {
            self.op.record_output_messages(&[message]);
        }
        let reference = self.op.observation_ref();
        self.op.end();
        reference
    }

    /// Ends the run successfully without an output envelope (TTS sinks).
    pub(crate) fn succeed_without_output(self) -> Option<ObservationRef> {
        self.apply_harvest(None);
        let reference = self.op.observation_ref();
        self.op.end();
        reference
    }

    /// Ends the run with an error (user cancellation ends it as cancelled).
    pub(crate) fn fail(self, error: &SdkError) {
        self.apply_harvest(None);
        if is_cancellation(error) {
            self.op.cancel();
        } else {
            self.op.end_with_error(&error_info(error));
        }
    }

    /// Ends the run as cancelled.
    pub(crate) fn cancel(self) -> Option<ObservationRef> {
        self.apply_harvest(None);
        let reference = self.op.observation_ref();
        self.op.cancel();
        reference
    }

    fn apply_harvest(&self, output: Option<&Envelope>) {
        let mut h = self.stages.harvest();
        if let Some(meta) = output.map(|o| &o.metadata) {
            let attempts = self.attempt_count() > 0;
            let totals = (h.tokens_in, h.tokens_out, h.cached_tokens, h.ttft_ms);
            h.absorb(meta.iter().map(|(k, v)| (k.as_str(), v.as_str())));
            if attempts {
                // The terminal envelope repeats the delivering attempt's
                // cumulative counts; it is not another attempt or a new total.
                (h.tokens_in, h.tokens_out, h.cached_tokens, h.ttft_ms) = totals;
            }
        }
        if h.finish_reason.is_none()
            && output.is_some_and(|o| o.metadata.contains_key(Envelope::TOOL_CALLS_METADATA_KEY))
            && output.is_some_and(|o| !tool_call_parts(o).is_empty())
        {
            // The turn ended on tool calls and no runtime reported a reason.
            h.finish_reason = Some("tool_call".to_owned());
        }
        let usage = Usage {
            input_tokens: h.tokens_in,
            output_tokens: h.tokens_out,
            cache_read_input_tokens: h.cached_tokens,
            text_input_tokens: if self.subject.input_modalities == [Modality::Text] {
                h.tokens_in
            } else {
                None
            },
            text_output_tokens: if self.subject.output_modalities == [Modality::Text] {
                h.tokens_out
            } else {
                None
            },
            ..Usage::default()
        };
        if !usage.is_empty() {
            self.op.record_usage(&usage);
        }
        if let Some(reason) = &h.finish_reason {
            self.op.record_finish_reasons([finish_reason(reason)]);
        }
        if let Some(at) = h
            .ttft_ms
            .and_then(|ms| Duration::try_from_secs_f64(ms / 1000.0).ok())
            .and_then(|duration| self.started.checked_add(duration))
        {
            // Runtime-measured first token (first call wins).
            self.op.record_first_token_at(at);
        }
        if let Some(itl) = h.mean_itl_ms {
            self.op.record_token_timing(itl / 1000.0);
        }
        if let Some(provider) = &h.provider {
            self.op
                .set_attribute(KeyValue::new("xybrid.upstream.provider", provider.clone()));
        }
        if let Some(accel) = &h.execution_provider {
            self.op
                .set_attribute(KeyValue::new("xybrid.genai.accelerator", accel.clone()));
        }
        if let Some(backend) = &h.backend {
            self.op.set_attribute(KeyValue::new(
                "xybrid.genai.runtime.name",
                runtime_name(backend),
            ));
        }
        if let Some(q) = &h.quantization {
            self.op
                .set_attribute(KeyValue::new("xybrid.genai.model.quantization", q.clone()));
        }
    }
}

impl Drop for RunObservation {
    fn drop(&mut self) {
        // Uninstall the stage recorder before the operation ends (Drop of
        // `op` cancels a run abandoned by a panic or early return).
        self._scope.take();
    }
}

fn is_cancellation(error: &SdkError) -> bool {
    match error {
        SdkError::Cancelled { .. } => true,
        SdkError::InferenceError { message, .. } => {
            message.contains(crate::run_options::AbortReason::UserCancelled.as_str())
        }
        _ => false,
    }
}

/// Host and port of the configured LLM gateway.
fn gateway_address() -> (Option<String>, Option<i64>) {
    url_address(&xybrid_core::cloud::platform_gateway_url())
}

fn finish_reason(raw: &str) -> String {
    match raw {
        "stop" | "eos" | "end_turn" | "stop_sequence" => "stop",
        "length" | "max_tokens" => "length",
        "tool_calls" | "tool_call" | "tool_use" => "tool_call",
        "content_filter" => "content_filter",
        "error" => "error",
        _ => "_OTHER",
    }
    .to_owned()
}

/// Maps an SDK error to a low-cardinality `error.type`.
pub(crate) fn error_info(error: &SdkError) -> ErrorInfo {
    let error_type = match error {
        SdkError::Timeout { .. } => "timeout",
        SdkError::RateLimited { .. } => "rate_limited",
        SdkError::NetworkError { .. } => "network_error",
        SdkError::Offline { .. } => "offline",
        SdkError::AbortedForCloudFallback { .. } => "aborted_for_cloud_fallback",
        SdkError::Cancelled { .. } => "cancelled",
        SdkError::TelemetryDelivery(_) => "telemetry_delivery",
        SdkError::ConfigError(_) => "config_error",
        SdkError::ModelNotFound(_) => "model_not_found",
        SdkError::LoadError { .. } => "load_error",
        SdkError::InferenceError { .. } => "inference_error",
        SdkError::PipelineError { .. } => "pipeline_error",
        SdkError::CircuitOpen(_) => "circuit_open",
        SdkError::StreamingNotSupported
        | SdkError::UnsupportedModelCapability { .. }
        | SdkError::UnsupportedBackendCapability { .. } => "unsupported",
        _ => "_OTHER",
    };
    // Automatic diagnostics can contain prompts, credentials and model paths.
    // Preserve the original error for the caller; export only its class. The
    // public ErrorInfo API still permits explicitly supplied, redacted text.
    ErrorInfo::new(error_type)
}

fn role(envelope: &Envelope, default: &str) -> String {
    envelope
        .role()
        .map_or_else(|| default.to_owned(), |r| r.as_str().to_owned())
}

fn parts(envelope: &Envelope) -> Vec<Part> {
    match &envelope.kind {
        EnvelopeKind::Text(text) => vec![Part::text(text.clone())],
        EnvelopeKind::Audio(bytes) => vec![Part::Media(MediaMetadata {
            modality: Some(Modality::Audio),
            mime_type: None,
            size_bytes: Some(bytes.len() as u64),
            ..MediaMetadata::default()
        })],
        EnvelopeKind::Image { source } => vec![Part::Media(image_metadata(source))],
        EnvelopeKind::MultiPart(items) => items.iter().flat_map(parts).collect(),
        EnvelopeKind::Embedding(_)
        | EnvelopeKind::ChoiceRequest(_)
        | EnvelopeKind::ChoiceScores(_) => Vec::new(),
    }
}

fn image_metadata(source: &ImageSource) -> MediaMetadata {
    match source {
        ImageSource::Encoded {
            bytes,
            format,
            dimensions,
        } => MediaMetadata {
            modality: Some(Modality::Image),
            mime_type: Some(format!("image/{}", format.to_string().to_ascii_lowercase())),
            size_bytes: Some(bytes.len() as u64),
            width: Some(dimensions.width),
            height: Some(dimensions.height),
            ..MediaMetadata::default()
        },
        _ => MediaMetadata {
            modality: Some(Modality::Image),
            ..MediaMetadata::default()
        },
    }
}

/// Tool definitions offered to the model, as recorded schema content.
fn tool_definitions(
    config: &xybrid_core::runtime_adapter::types::GenerationConfig,
) -> Option<serde_json::Value> {
    if config.tools.is_empty() {
        return None;
    }
    serde_json::to_value(&config.tools).ok()
}

/// Tool-call parts for calls the model requested, from the envelope's
/// `tool_calls` metadata (the same parse as `InferenceResult::tool_calls`).
fn tool_call_parts(envelope: &Envelope) -> Vec<Part> {
    envelope
        .metadata
        .get(Envelope::TOOL_CALLS_METADATA_KEY)
        .and_then(|raw| serde_json::from_str::<Vec<xybrid_core::gateway::ToolCall>>(raw).ok())
        .unwrap_or_default()
        .into_iter()
        .map(tool_call_part)
        .collect()
}

fn tool_call_part(call: xybrid_core::gateway::ToolCall) -> Part {
    // Arguments are a JSON string on the wire; keep the raw string when it
    // does not parse rather than dropping what the model sent.
    let arguments = serde_json::from_str(&call.function.arguments)
        .unwrap_or(serde_json::Value::String(call.function.arguments));
    Part::ToolCall {
        id: Some(call.id).filter(|id| !id.is_empty()),
        name: call.function.name,
        arguments,
    }
}

fn is_empty_text(part: &Part) -> bool {
    matches!(part, Part::Text { content } if content.is_empty())
}

/// Expands a tool-continuation envelope into the conversation it replays:
/// the user turn, the assistant turn that requested the tools and one `tool`
/// message per result. `None` when the envelope carries no tool results.
fn tool_continuation_messages(envelope: &Envelope, role: &str) -> Option<Vec<InputMessage>> {
    let results: Vec<xybrid_core::ir::ToolCallResult> = envelope
        .metadata
        .get(Envelope::TOOL_RESPONSES_METADATA_KEY)
        .and_then(|raw| serde_json::from_str(raw).ok())?;
    let mut messages = Vec::with_capacity(results.len() + 2);
    let user = parts(envelope);
    if !user.is_empty() {
        messages.push(InputMessage {
            role: role.to_owned(),
            parts: user,
            name: None,
        });
    }
    let mut calls: Vec<Part> = envelope
        .metadata
        .get(Envelope::TOOL_PRIOR_TEXT_METADATA_KEY)
        .map(|text| xybrid_core::runtime_adapter::tool_call::parse_tool_calls(text))
        .unwrap_or_default()
        .into_iter()
        .map(tool_call_part)
        .collect();
    if calls.is_empty() {
        // The prior turn's protocol text did not parse: the results still
        // name the calls they answer.
        calls = results
            .iter()
            .map(|result| Part::ToolCall {
                id: Some(result.call_id.clone()),
                name: result.name.clone(),
                arguments: serde_json::Value::Null,
            })
            .collect();
    }
    if !calls.is_empty() {
        messages.push(InputMessage {
            role: "assistant".into(),
            parts: calls,
            name: None,
        });
    }
    for result in results {
        messages.push(InputMessage {
            role: "tool".into(),
            parts: vec![Part::ToolCallResponse {
                id: Some(result.call_id),
                response: result.content,
            }],
            name: None,
        });
    }
    Some(messages)
}

/// Builds the GenAI system instructions and input messages for a run.
fn input_messages(
    history: &[Envelope],
    input: Option<&Envelope>,
) -> (Vec<Part>, Vec<InputMessage>) {
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for envelope in history.iter().chain(input) {
        let role = role(envelope, "user");
        if let Some(expanded) = tool_continuation_messages(envelope, &role) {
            messages.extend(expanded);
            continue;
        }
        let mut p = parts(envelope);
        // An assistant history turn that requested tools keeps its calls.
        let calls = tool_call_parts(envelope);
        if !calls.is_empty() {
            p.retain(|part| !is_empty_text(part));
            p.extend(calls);
        }
        if p.is_empty() {
            continue;
        }
        if role == "system" {
            system.extend(p);
        } else {
            messages.push(InputMessage {
                role,
                parts: p,
                name: None,
            });
        }
    }
    if let Some(EnvelopeKind::ChoiceRequest(request)) = input.map(|e| &e.kind) {
        if let Ok(value) = serde_json::to_string(request) {
            messages.push(InputMessage::text("user", value));
        }
    }
    (system, messages)
}

fn output_message(output: &Envelope) -> Option<OutputMessage> {
    let mut parts = parts(output);
    if let Some(EnvelopeKind::ChoiceScores(scores)) = Some(&output.kind) {
        if let Ok(value) = serde_json::to_string(scores) {
            parts = vec![Part::text(value)];
        }
    }
    if let Some(reasoning) = output.metadata.get("reasoning_content") {
        parts.insert(
            0,
            Part::Reasoning {
                content: reasoning.clone(),
            },
        );
    }
    let calls = tool_call_parts(output);
    let called = !calls.is_empty();
    if called {
        parts.retain(|part| !is_empty_text(part));
        parts.extend(calls);
    }
    if parts.is_empty() {
        return None;
    }
    Some(OutputMessage {
        role: "assistant".into(),
        parts,
        name: None,
        finish_reason: output
            .metadata
            .get("finish_reason")
            .map(|r| finish_reason(r))
            // A turn that ended on tool calls without a reported reason.
            .or_else(|| called.then(|| "tool_call".to_owned())),
    })
}

/// Converts a measured resource summary to a portable sample.
///
/// `cpu_peak_pct`, `available_mem_min_mb` and `memory_pressure_peak` are not
/// exported: the contract's resource catalog has no peak-CPU, available-memory
/// or memory-pressure attribute, and `memory.used_bytes` (system memory in
/// use) cannot be derived from available memory without the device total,
/// which the summary does not carry.
pub(crate) fn resource_sample(summary: &ResourceUsageSummary) -> ResourceSample {
    ResourceSample {
        cpu_utilization: summary.cpu_avg_pct.map(|p| f64::from(p) / 100.0),
        memory_used_bytes: None,
        process_rss_bytes: None,
        process_peak_rss_bytes: summary
            .process_rss_peak_mb
            .map(|mb| u64::from(mb) * 1024 * 1024),
        thermal_state: summary.thermal_measured.then(|| {
            match summary.thermal_state_peak {
                ThermalState::Normal => "nominal",
                ThermalState::Warm => "fair",
                ThermalState::Hot => "serious",
                ThermalState::Critical => "critical",
            }
            .to_owned()
        }),
        battery_level: summary.battery_pct_end.map(|p| f64::from(p) / 100.0),
        battery_charging: None,
        accelerator_utilization: None,
        accelerator_memory_used_bytes: None,
    }
}

// ---------------------------------------------------------------------------
// Stage recorder: core `SpanGuard`s become child spans of the run.
// ---------------------------------------------------------------------------

pub(crate) type Harvest = StageMeasurements;

/// Measured stage values; missing measurements remain unavailable.
#[derive(Clone, Debug, Default)]
pub struct StageMeasurements {
    pub tokens_in: Option<u64>,
    pub tokens_out: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub ttft_ms: Option<f64>,
    pub mean_itl_ms: Option<f64>,
    pub finish_reason: Option<String>,
    pub backend: Option<String>,
    pub execution_provider: Option<String>,
    pub provider: Option<String>,
    pub quantization: Option<String>,
}

impl StageMeasurements {
    fn absorb<'a>(&mut self, pairs: impl Iterator<Item = (&'a str, &'a str)>) {
        for (key, value) in pairs {
            // Output envelopes are caller/runtime-owned too. Use the same
            // boundary as stages before promoting metadata to the logical run.
            let Some(value) = stage_metadata_value(key, value) else {
                continue;
            };
            let n = || value.parse::<u64>().ok();
            let f = || value.parse::<f64>().ok();
            match key {
                "tokens_in" | "prompt_tokens" => self.tokens_in = n().or(self.tokens_in),
                "tokens_out" | "completion_tokens" | "tokens_generated" => {
                    self.tokens_out = n().or(self.tokens_out);
                }
                "prompt_cached_tokens" | "cache_read_input_tokens" => {
                    self.cached_tokens = n().or(self.cached_tokens);
                }
                "ttft_ms" => self.ttft_ms = f().or(self.ttft_ms),
                "mean_itl_ms" => self.mean_itl_ms = f().or(self.mean_itl_ms),
                "finish_reason" => self.finish_reason = Some(value.to_owned()),
                "backend" => self.backend = Some(value.to_owned()),
                "execution_provider" => self.execution_provider = Some(value.to_owned()),
                "provider" => self.provider = Some(value.to_owned()),
                "quantization" => self.quantization = Some(value.to_owned()),
                _ => {}
            }
        }
    }
}

/// Reviewed, content-free metadata from core stages and output envelopes.
///
/// These are not application observation attributes: core may place paths,
/// snapshots, errors or arbitrary bundle strings here. Content is recorded by
/// the logical operation's message/error APIs, which apply its content policy.
/// Keep descriptor values closed as well as keys; token-shaped secrets are
/// still secrets. Explicit public observation APIs continue to accept custom
/// model/provider identities and application-owned attributes.
/// Validates the SDK's reviewed, content-free core metadata vocabulary.
#[must_use]
pub fn stage_metadata_value(key: &str, value: &str) -> Option<String> {
    let known = match key {
        "backend" => matches!(
            value,
            "ort"
                | "onnx"
                | "llamacpp"
                | "llama.cpp"
                | "mistralrs"
                | "mistral"
                | "whispercpp"
                | "candle"
                | "zzz"
                | "coreml"
                | "cloud"
                | "gateway"
        ),
        "execution_provider" => matches!(
            value,
            "cpu"
                | "metal"
                | "cuda"
                | "vulkan"
                | "coreml"
                | "directml"
                | "nnapi"
                | "webgpu"
                | "wasm"
                | "tensorrt"
                | "rocm"
        ),
        "provider" => matches!(
            value,
            "openai"
                | "anthropic"
                | "google"
                | "groq"
                | "together"
                | "fireworks"
                | "mistral"
                | "deepseek"
                | "openrouter"
                | "elevenlabs"
                | "custom"
                | "xybrid"
        ),
        "stage_kind" => matches!(
            value,
            "asr" | "tts" | "llm" | "translate" | "vision" | "embed" | "audio" | "decision"
        ),
        "span_kind" => matches!(value, "cpu" | "gpu" | "io" | "tool"),
        "task" => matches!(
            value,
            "speech-recognition"
                | "speech-to-text"
                | "asr"
                | "text-to-speech"
                | "tts"
                | "text-generation"
                | "text_generation"
                | "chat"
                | "llm"
                | "translation"
                | "translate"
                | "image-classification"
                | "image-to-text"
                | "vision"
                | "vision-language"
                | "vision_language"
                | "vlm"
                | "embedding"
                | "sentence-embedding"
                | "audio-classification"
                | "vad"
                | "decision"
        ),
        "quantization" => {
            let normalized = value.to_ascii_lowercase();
            return matches!(
                normalized.as_str(),
                "f16"
                    | "f32"
                    | "fp16"
                    | "fp32"
                    | "bf16"
                    | "int8"
                    | "q2_k"
                    | "q2_k_s"
                    | "q3_k"
                    | "q3_k_s"
                    | "q3_k_m"
                    | "q3_k_l"
                    | "q4_0"
                    | "q4_1"
                    | "q4_k"
                    | "q4_k_s"
                    | "q4_k_m"
                    | "q5_0"
                    | "q5_1"
                    | "q5_k"
                    | "q5_k_s"
                    | "q5_k_m"
                    | "q6_k"
                    | "q8_0"
                    | "q8_1"
                    | "q8_k"
                    | "iq1_s"
                    | "iq1_m"
                    | "iq2_xxs"
                    | "iq2_xs"
                    | "iq2_s"
                    | "iq3_xxs"
                    | "iq3_s"
                    | "iq4_xs"
                    | "iq4_nl"
            )
            .then_some(normalized);
        }
        "tokens_in"
        | "tokens_out"
        | "prompt_tokens"
        | "completion_tokens"
        | "tokens_generated"
        | "prompt_cached_tokens"
        | "cache_read_input_tokens"
        | "emitted_chunks"
        | "choice_count"
        | "image_count"
        | "placeholder_tokens"
        | "raw_image_source_vec_allocations"
        | "raw_image_source_extra_pixel_copies"
        | "raw_image_conversion_buffer_allocations"
        | "raw_image_pixel_bytes" => {
            return value.parse::<u64>().ok().map(|n| n.to_string());
        }
        "ttft_ms"
        | "mean_itl_ms"
        | "p95_itl_ms"
        | "generation_time_ms"
        | "image_preprocess_ms"
        | "tokens_per_second"
        | "decode_tps"
        | "prefill_tps" => {
            return value
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite() && *n >= 0.0)
                .map(|n| n.to_string());
        }
        "finish_reason" => {
            return Some(
                match value {
                    "stop" | "eos" | "end_turn" | "stop_sequence" => "stop",
                    "length" | "max_tokens" => "length",
                    "tool_calls" | "tool_call" | "tool_use" => "tool_call",
                    "content_filter" => "content_filter",
                    "error" => "error",
                    _ => "_OTHER",
                }
                .to_owned(),
            );
        }
        _ => false,
    };
    known.then(|| value.to_owned())
}

pub(crate) type StageRecorder = StageCapture;

/// SDK attribute naming why a `load_model` operation ran: `load` (explicit
/// load), `lazy` (first execution after a load), `reload` (after automatic
/// release), `warmup`, or `speculative_background`.
pub(crate) const LOAD_TRIGGER: &str = "xybrid.load.trigger";

/// The model that lazy materializations beneath a recorder load, and why.
/// Shared by a run's logical recorder and its attempts.
pub(crate) struct LoadSubject {
    model: String,
    format: Option<&'static str>,
    size_bytes: Option<u64>,
    trigger: Mutex<&'static str>,
}

impl LoadSubject {
    pub(crate) fn new(
        model: &str,
        format: Option<&'static str>,
        trigger: &'static str,
        size_bytes: Option<u64>,
    ) -> Arc<Self> {
        Arc::new(Self {
            model: model.to_owned(),
            format,
            size_bytes,
            trigger: Mutex::new(trigger),
        })
    }

    fn trigger(&self) -> &'static str {
        *self
            .trigger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Declared artifact size only; missing or invalid metadata stays absent.
pub(crate) fn declared_model_size(metadata: &ModelMetadata) -> Option<u64> {
    metadata
        .metadata
        .get("size_bytes")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            metadata
                .metadata
                .get("size_mb")
                .and_then(serde_json::Value::as_u64)?
                .checked_mul(1024 * 1024)
        })
        .filter(|bytes| i64::try_from(*bytes).is_ok())
}

fn conversation_id(options: Option<&ObservationOptions>) -> Option<String> {
    options.and_then(|o| o.conversation_id.clone()).or_else(|| {
        xybrid_core::tracing::current_recorder().and_then(|recorder| {
            recorder
                .as_any()
                .and_then(|r| r.downcast_ref::<StageRecorder>())
                .and_then(|r| r.conversation_id.clone())
        })
    })
}

struct ActiveLoad {
    operation: Arc<ModelLoadOperation>,
    facts: Mutex<LoadFacts>,
}

#[derive(Clone, Copy, Default)]
struct LoadFacts {
    size: Option<(i64, LoadSizeSource)>,
    cache_hit: Option<bool>,
    runtime: Option<&'static str>,
    format: Option<&'static str>,
}

/// A model's own declaration takes precedence over measured model files;
/// transfer/archive bytes are only a fallback while model size is unknown.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LoadSizeSource {
    Transfer,
    MeasuredFiles,
    ModelDeclaration,
}

impl ActiveLoad {
    fn record_size(&self, bytes: u64, source: LoadSizeSource) {
        let Ok(bytes) = i64::try_from(bytes) else {
            return;
        };
        let mut facts = self
            .facts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if facts.size.is_none_or(|(_, previous)| source >= previous) {
            facts.size = Some((bytes, source));
        }
    }

    /// The OTel SDK appends attributes; emit each final fact once to avoid
    /// duplicate keys and spending the attribute budget on progress updates.
    fn finish_metadata(&self) {
        let facts = *self
            .facts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((bytes, _)) = facts.size {
            self.operation
                .set_attribute(KeyValue::new(semconv::attr::MODEL_SIZE_BYTES, bytes));
        }
        if let Some(hit) = facts.cache_hit {
            self.operation.record_cache_hit(hit);
        }
        if let Some(runtime) = facts.runtime {
            self.operation
                .set_attribute(KeyValue::new(semconv::attr::RUNTIME_NAME, runtime));
        }
        if let Some(format) = facts.format {
            self.operation
                .set_attribute(KeyValue::new(semconv::attr::MODEL_FORMAT, format));
        }
    }
}

fn current_load() -> Option<Arc<ActiveLoad>> {
    xybrid_core::tracing::current_recorder().and_then(|recorder| {
        recorder
            .as_any()
            .and_then(|r| r.downcast_ref::<StageRecorder>())
            .and_then(|r| r.active_load.clone())
    })
}

/// The first cache decision describes the payload source. A cached bundle that
/// still needs extraction is a hit; extraction must not overwrite that decision.
/// For a directly supplied local bundle, the extraction cache decides instead.
pub(crate) fn record_load_cache_hit(hit: bool) {
    if let Some(load) = current_load() {
        let mut facts = load
            .facts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if facts.cache_hit.is_none() {
            facts.cache_hit = Some(hit);
        }
    }
}

/// The resolved artifact size is known before its transfer, including on errors.
pub(crate) fn record_load_size(bytes: u64) {
    if let Some(load) = current_load() {
        load.record_size(bytes, LoadSizeSource::Transfer);
    }
}

struct MeasuredLoad {
    operation: ModelLoadOperation,
    resources: xybrid_core::device::RunGuard,
}

impl MeasuredLoad {
    fn finish_resources(self) -> ModelLoadOperation {
        if let Some(summary) = self.resources.finish() {
            self.operation.record_resource(&resource_sample(&summary));
        }
        self.operation
    }
}

/// Records that the run on this thread reloads an automatically released
/// model: the materialization it triggers is a `reload`, not a lazy load.
pub(crate) fn mark_reload() {
    if let Some(recorder) = xybrid_core::tracing::current_recorder() {
        if let Some(load) = recorder
            .as_any()
            .and_then(|r| r.downcast_ref::<StageRecorder>())
            .and_then(|r| r.load.as_ref())
        {
            *load
                .trigger
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = "reload";
        }
    }
}

/// Records execution stages as `INTERNAL` child spans of the run and keeps
/// the last value of the usage/timing keys stages stamp.
///
/// A core `model_load:<runtime>` stage (weights actually loaded into a
/// runtime) becomes a `load_model` operation when the model is known
/// ([`StageCapture::for_model`]).
pub struct StageCapture {
    observer: Observer,
    session: ObservabilitySession,
    suppressed: bool,
    root: Option<ObservationContext>,
    spans: Mutex<HashMap<u64, SpanOperation>>,
    loads: Mutex<HashMap<u64, MeasuredLoad>>,
    failed: Mutex<std::collections::HashSet<u64>>,
    next: std::sync::atomic::AtomicU64,
    harvest: Mutex<Harvest>,
    load: Option<Arc<LoadSubject>>,
    conversation_id: Option<String>,
    active_load: Option<Arc<ActiveLoad>>,
    /// Set on an attempt's recorder: its root follows the live attempt.
    attempt: Option<Arc<AttemptChain>>,
}

impl fmt::Debug for StageCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageCapture")
            .field("suppressed", &self.suppressed)
            .field("root", &self.root())
            .field("model", &self.load.as_ref().map(|l| l.model.as_str()))
            .finish_non_exhaustive()
    }
}

impl StageCapture {
    /// Records core stages beneath an explicitly captured parent.
    pub fn new(observer: Observer, root: Option<ObservationContext>) -> Self {
        let suppressed = root.is_none();
        Self::for_session(ObservabilitySession::borrowed(observer), root, suppressed)
    }

    /// Reports weights that core loads lazily beneath this recorder as
    /// `load_model {model}` operations (trigger `lazy`).
    #[must_use]
    pub fn for_model(self, model: &str, metadata: Option<&ModelMetadata>) -> Self {
        let format = metadata.and_then(|m| RunSubject::from_metadata(model, m).format);
        self.with_load(LoadSubject::new(
            model,
            format,
            "lazy",
            metadata.and_then(declared_model_size),
        ))
    }

    pub(crate) fn with_load(mut self, load: Arc<LoadSubject>) -> Self {
        self.load = Some(load);
        self
    }

    fn for_session(
        session: ObservabilitySession,
        root: Option<ObservationContext>,
        suppressed: bool,
    ) -> Self {
        Self {
            observer: session.observer().clone(),
            session,
            suppressed,
            root,
            spans: Mutex::new(HashMap::new()),
            loads: Mutex::new(HashMap::new()),
            failed: Mutex::new(std::collections::HashSet::new()),
            next: std::sync::atomic::AtomicU64::new(1),
            harvest: Mutex::new(Harvest::default()),
            load: None,
            conversation_id: conversation_id(None),
            active_load: None,
            attempt: None,
        }
    }

    /// Parent of stages without an open parent stage and of outbound
    /// requests: the live attempt for an attempt's recorder, else the root.
    fn root(&self) -> Option<ObservationContext> {
        match &self.attempt {
            Some(attempt) => attempt.context(),
            None => self.root.clone(),
        }
    }

    fn with_conversation_id(mut self, id: Option<String>) -> Self {
        self.conversation_id = id;
        self
    }

    fn with_active_load(mut self, load: Arc<ActiveLoad>) -> Self {
        self.active_load = Some(load);
        self
    }

    /// The latest known measurement snapshot; repeated cumulative reports replace values.
    pub fn harvest(&self) -> StageMeasurements {
        self.harvest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Folds an attempt's harvest into this (logical) one.
    fn merge(&self, other: &Harvest) {
        let mut h = self
            .harvest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        fn sum(a: Option<u64>, b: Option<u64>) -> Option<u64> {
            match (a, b) {
                (Some(a), Some(b)) => Some(a.saturating_add(b)),
                (a, b) => a.or(b),
            }
        }
        h.tokens_in = sum(h.tokens_in, other.tokens_in);
        h.tokens_out = sum(h.tokens_out, other.tokens_out);
        h.cached_tokens = sum(h.cached_tokens, other.cached_tokens);
        h.ttft_ms = match (h.ttft_ms, other.ttft_ms) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        h.mean_itl_ms = other.mean_itl_ms.or(h.mean_itl_ms);
        h.finish_reason = other.finish_reason.clone().or(h.finish_reason.take());
        h.backend = other.backend.clone().or(h.backend.take());
        h.execution_provider = other
            .execution_provider
            .clone()
            .or(h.execution_provider.take());
        h.provider = other.provider.clone().or(h.provider.take());
        h.quantization = other.quantization.clone().or(h.quantization.take());
    }

    fn spans(&self) -> std::sync::MutexGuard<'_, HashMap<u64, SpanOperation>> {
        self.spans
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn loads(&self) -> std::sync::MutexGuard<'_, HashMap<u64, MeasuredLoad>> {
        self.loads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Context of the open stage `id` (generic span or load), else the root.
    fn context_of(&self, id: Option<u64>) -> Option<ObservationContext> {
        id.and_then(|id| {
            self.spans()
                .get(&id)
                .map(SpanOperation::context)
                .or_else(|| self.loads().get(&id).map(|load| load.operation.context()))
        })
        .unwrap_or_else(|| self.root())
    }
}

impl TraceRecorder for StageCapture {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn start_span(&self, name: &str, parent: Option<u64>) -> u64 {
        let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let parent_cx = match parent {
            Some(p) => self.context_of(Some(p)),
            None => self.root(),
        };
        let mut options = match parent_cx {
            Some(cx) => StartOptions::child_of(cx),
            // A run whose logical span was suppressed keeps its stages
            // untraced too.
            None => StartOptions::default().with_trace(TraceMode::Suppress),
        };
        if let Some(id) = &self.conversation_id {
            options
                .attributes
                .push(KeyValue::new(semconv::attr::CONVERSATION_ID, id.clone()));
        }
        if let (Some(runtime), Some(load)) = (
            name.strip_prefix(xybrid_core::tracing::MODEL_LOAD_STAGE_PREFIX),
            &self.load,
        ) {
            let runtime = Some(runtime_name(runtime))
                .filter(|r| *r != "unknown")
                .map(str::to_owned);
            options
                .attributes
                .push(KeyValue::new(LOAD_TRIGGER, load.trigger()));
            let operation = self.observer.start_model_load(
                ModelLoadInfo {
                    model: load.model.clone(),
                    provider: runtime.clone(),
                    runtime,
                    format: load.format.map(str::to_owned),
                    size_bytes: load.size_bytes,
                },
                options,
            );
            self.loads().insert(
                id,
                MeasuredLoad {
                    operation,
                    resources: crate::telemetry::begin_resource_run(),
                },
            );
            return id;
        }
        let span = self.observer.start_span(
            name.to_owned(),
            vec![KeyValue::new("xybrid.stage.name", name.to_owned())],
            options,
        );
        self.spans().insert(id, span);
        id
    }

    fn end_span(&self, id: u64) {
        let failed = self
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id);
        if let Some(span) = self.spans().remove(&id) {
            if failed {
                span.end_with_error(&ErrorInfo::new("stage_error"));
            } else {
                span.end();
            }
        } else if let Some(load) = self.loads().remove(&id) {
            let load = load.finish_resources();
            if failed {
                load.end_with_error(&ErrorInfo::new("load_error"));
            } else {
                load.end();
            }
        }
    }

    fn add_metadata(&self, id: u64, key: &str, value: &str) {
        if key == xybrid_core::tracing::STAGE_OUTCOME_KEY {
            // Only the fact of failure crosses; core never sends a reason here.
            if value == "error" {
                self.failed
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(id);
            }
            return;
        }
        let Some(value) = stage_metadata_value(key, value) else {
            return;
        };
        self.harvest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .absorb(std::iter::once((key, value.as_str())));
        if let Some(span) = self.spans().get(&id) {
            span.set_attribute(KeyValue::new(format!("xybrid.stage.{key}"), value));
        }
    }

    fn allows_provider_context(&self, origin: &str) -> bool {
        self.session.allows_provider_context(origin)
    }

    fn retry_request(&self, _id: Option<u64>, error_type: &str, failed_at: std::time::Instant) {
        if let Some(attempt) = &self.attempt {
            attempt.retry(error_type, failed_at);
        }
    }

    fn trace_headers(&self, _id: Option<u64>) -> Option<TraceHeaders> {
        // Network children belong to the enclosing run/attempt, not an
        // INTERNAL preprocessing stage that happens to be open at the call.
        let cx = self.root()?;
        Some(TraceHeaders {
            traceparent: cx.to_traceparent(),
            tracestate: cx.tracestate().map(str::to_owned),
        })
    }
}

fn option_trace(options: Option<&ObservationOptions>) -> TraceMode {
    let trace = options.map(|o| o.trace).unwrap_or_default();
    if options.is_some_and(|o| o.parent.is_some()) {
        trace
    } else {
        inherited_trace(trace)
    }
}

fn inherited_trace(trace: TraceMode) -> TraceMode {
    if trace == TraceMode::Suppress {
        return trace;
    }
    let suppressed = xybrid_core::tracing::current_recorder().is_some_and(|r| {
        r.as_any()
            .and_then(|r| r.downcast_ref::<StageRecorder>())
            .is_some_and(|r| r.suppressed)
    });
    if suppressed {
        TraceMode::Suppress
    } else {
        trace
    }
}

/// The parent for an operation started on this thread: the innermost open
/// stage of an installed SDK recorder (else that recorder's run), a foreign
/// recorder's trace headers, or the application's active OpenTelemetry span.
pub(crate) fn ambient_parent() -> Option<ObservationContext> {
    let Some(recorder) = xybrid_core::tracing::current_recorder() else {
        return ObservationContext::current();
    };
    if let Some(stages) = recorder
        .as_any()
        .and_then(|r| r.downcast_ref::<StageRecorder>())
    {
        // A suppressed recorder has no root or stage contexts: do not bypass
        // it by resurrecting an outer application's span.
        return stages.context_of(xybrid_core::tracing::current_span_id());
    }
    // An installed recorder with no headers can represent a suppressed workflow.
    xybrid_core::tracing::current_trace_headers().and_then(|headers| {
        ObservationContext::from_traceparent(&headers.traceparent, headers.tracestate.as_deref())
            .ok()
    })
}

/// Observation of an explicit model load (`load_model {model}`).
pub(crate) struct LoadScope {
    load: Arc<ActiveLoad>,
    resources: Option<xybrid_core::device::RunGuard>,
    _scope: Option<RecorderScope>,
}

impl LoadScope {
    /// Starts an explicit load (`xybrid.load.trigger = load`).
    pub(crate) fn start(
        model: &str,
        metadata: Option<&ModelMetadata>,
        options: Option<&ObservationOptions>,
    ) -> Self {
        Self::start_with_trigger(model, metadata, options, "load")
    }

    /// Starts a load observation for `trigger` under the explicit or ambient parent.
    pub(crate) fn start_with_trigger(
        model: &str,
        metadata: Option<&ModelMetadata>,
        options: Option<&ObservationOptions>,
        trigger: &'static str,
    ) -> Self {
        let session = resolve_session(options);
        let subject = metadata.map(|m| RunSubject::from_metadata(model, m));
        let trace = option_trace(options);
        let mut start = StartOptions::default().with_trace(trace).with_parent(
            options
                .and_then(|o| o.parent.clone())
                .or_else(ambient_parent),
        );
        if let Some(options) = options {
            start.attributes = options.attributes.clone();
        }
        let conversation_id = conversation_id(options);
        if let Some(id) = &conversation_id {
            start
                .attributes
                .push(KeyValue::new(semconv::attr::CONVERSATION_ID, id.clone()));
        }
        start.attributes.push(KeyValue::new(LOAD_TRIGGER, trigger));
        let runtime = subject
            .as_ref()
            .and_then(|s| s.runtime)
            .filter(|r| *r != "unknown");
        let op = Arc::new(session.observer().start_model_load(
            ModelLoadInfo {
                model: model.to_owned(),
                provider: runtime.map(str::to_owned),
                // Mutable metadata is emitted once with its final known facts.
                runtime: None,
                format: None,
                size_bytes: None,
            },
            start,
        ));
        let load = Arc::new(ActiveLoad {
            operation: op.clone(),
            facts: Mutex::new(LoadFacts {
                runtime,
                format: subject.as_ref().and_then(|s| s.format),
                ..Default::default()
            }),
        });
        if let Some(bytes) = metadata.and_then(declared_model_size) {
            load.record_size(bytes, LoadSizeSource::ModelDeclaration);
        }
        let _scope = Some(RecorderScope::install(Arc::new(
            StageRecorder::for_session(session, op.context(), trace == TraceMode::Suppress)
                .with_conversation_id(conversation_id)
                .with_active_load(load.clone()),
        )));
        Self {
            load,
            _scope,
            resources: Some(crate::telemetry::begin_resource_run()),
        }
    }

    /// Records what was loaded once it is known (runtime and format of the
    /// resolved model). Values that are not known stay absent.
    pub(crate) fn describe(&self, model_id: &str, metadata: &ModelMetadata) {
        let subject = RunSubject::from_metadata(model_id, metadata);
        {
            let mut facts = self
                .load
                .facts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            facts.runtime = subject
                .runtime
                .filter(|r| *r != "unknown")
                .or(facts.runtime);
            facts.format = subject.format.or(facts.format);
        }
        if let Some(bytes) = subject.size_bytes {
            self.load
                .record_size(bytes, LoadSizeSource::ModelDeclaration);
        }
    }

    /// Metadata read from a successfully fetched extraction, without requiring weights.
    pub(crate) fn describe_directory(&self, model_id: &str, directory: &std::path::Path) {
        if let Some(metadata) = std::fs::read(directory.join("model_metadata.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ModelMetadata>(&bytes).ok())
        {
            self.describe(model_id, &metadata);
            if declared_model_size(&metadata).is_none() && !metadata.files.is_empty() {
                let size = metadata.files.iter().try_fold(0_u64, |total, file| {
                    let file = std::fs::metadata(directory.join(file)).ok()?;
                    if !file.is_file() {
                        return None;
                    }
                    total.checked_add(file.len())
                });
                if let Some(bytes) = size {
                    self.load.record_size(bytes, LoadSizeSource::MeasuredFiles);
                }
            }
        }
    }

    /// Retains transfer/archive size as a fallback; unrepresentable values stay absent.
    pub(crate) fn size_bytes(&self, bytes: u64) {
        self.load.record_size(bytes, LoadSizeSource::Transfer);
    }

    /// Records the resources measured around the load.
    pub(crate) fn resource(&self, summary: Option<&ResourceUsageSummary>) {
        if let Some(sample) = summary.map(resource_sample) {
            self.load.operation.record_resource(&sample);
        }
    }

    pub(crate) fn end<T>(mut self, result: &SdkResult<T>) -> Option<ObservationRef> {
        if let Some(resources) = self.resources.take() {
            self.resource(resources.finish().as_ref());
        }
        self._scope.take();
        self.load.finish_metadata();
        let reference = self.load.operation.observation_ref();
        match result {
            Ok(_) => {
                self.load.operation.end();
                reference
            }
            Err(e) if is_cancellation(e) => {
                self.load.operation.cancel();
                None
            }
            Err(e) => {
                self.load.operation.end_with_error(&error_info(e));
                None
            }
        }
    }
}

/// Why a pipeline executes: a real run, or a warmup that loads models
/// without counting as inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PipelinePurpose {
    Run,
    Warmup,
}

/// Closed-vocabulary routing attributes of a routed stage. The authority's
/// free-text reason (model ids, policy and remote-advice text) is never exported.
fn routing_attributes(route: &xybrid_core::orchestrator::StageRoute<'_>) -> Vec<KeyValue> {
    use xybrid_core::orchestrator::routing_engine::RouteTarget;
    use xybrid_core::orchestrator::DecisionSource;
    vec![
        KeyValue::new(
            "xybrid.routing.target",
            match route.decision.target {
                RouteTarget::Local => "local",
                RouteTarget::Cloud => "cloud",
                RouteTarget::Fallback(_) => "fallback",
            },
        ),
        KeyValue::new(
            "xybrid.routing.source",
            match route.source {
                DecisionSource::Local => "local",
                DecisionSource::Remote => "remote",
                DecisionSource::Cached => "cached",
                DecisionSource::Default => "default",
            },
        ),
        KeyValue::new("xybrid.routing.policy_allowed", route.policy_allowed),
    ]
}

/// Where a routed stage executes, decided from the route (not from whether a
/// bundle happens to be present): provider, location and server.
struct StageDestination {
    location: Location,
    provider: Option<String>,
    server: Option<(Option<String>, Option<i64>)>,
}

fn stage_destination(
    stage: &xybrid_core::context::StageDescriptor,
    route: &xybrid_core::orchestrator::StageRoute<'_>,
) -> StageDestination {
    use xybrid_core::orchestrator::routing_engine::RouteTarget;
    if matches!(route.decision.target, RouteTarget::Local) {
        return StageDestination {
            location: Location::Local,
            provider: None,
            server: None,
        };
    }
    let option = |key: &str| stage.options.as_ref().and_then(|o| o.get::<String>(key));
    let direct = option("backend").as_deref() == Some("direct");
    let url = option("gateway_url").or_else(|| match (direct, stage.provider.as_ref()) {
        (true, Some(provider)) => Some(provider.default_base_url().to_owned()),
        (true, None) => None,
        (false, _) => Some(xybrid_core::cloud::platform_gateway_url()),
    });
    let provider = match (direct, stage.provider.as_ref()) {
        (true, Some(provider)) => provider.as_str().to_owned(),
        _ => "xybrid".to_owned(),
    };
    StageDestination {
        location: Location::Remote,
        provider: Some(provider),
        server: Some(url.map_or((None, None), |url| url_address(&url))),
    }
}

/// HTTP host and optional non-default port, excluding credentials, path,
/// query and fragment. Invalid or non-HTTP URLs provide no server attributes.
fn url_address(url: &str) -> (Option<String>, Option<i64>) {
    let Ok(parsed) = url::Url::parse(url) else {
        return (None, None);
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        return (None, None);
    }
    let host = parsed.host().map(|host| match host {
        url::Host::Domain(domain) => domain.to_owned(),
        url::Host::Ipv4(address) => address.to_string(),
        url::Host::Ipv6(address) => address.to_string(),
    });
    (host, parsed.port().map(i64::from))
}

/// Observation of one routed pipeline stage.
pub(crate) enum StageObservation {
    /// A real run: the stage's logical inference, measured on its own.
    Inference(Box<StageInference>),
    /// A warmup: execution stages stay generic spans and lazy loads are
    /// `load_model` operations with trigger `warmup`; nothing is an inference.
    Warmup { _scope: RecorderScope },
}

pub(crate) struct StageInference {
    run: RunObservation,
    attempt: AttemptScope,
    resources: xybrid_core::device::RunGuard,
}

impl StageObservation {
    /// Starts observing `stage` once its route is known, under `parent`.
    pub(crate) fn start(
        stage: &xybrid_core::context::StageDescriptor,
        input: &Envelope,
        route: &xybrid_core::orchestrator::StageRoute<'_>,
        options: &crate::run_options::RunOptions,
        parent: Option<ObservationContext>,
        purpose: PipelinePurpose,
    ) -> Self {
        let mut subject = RunSubject::pipeline_stage(stage);
        if purpose == PipelinePurpose::Warmup {
            let session = resolve_session(options.observation.as_ref());
            let suppressed = parent.is_none();
            let recorder = StageRecorder::for_session(session, parent, suppressed).with_load(
                LoadSubject::new(&subject.model, subject.format, "warmup", subject.size_bytes),
            );
            return Self::Warmup {
                _scope: RecorderScope::install(Arc::new(recorder)),
            };
        }
        let StageDestination {
            location,
            provider,
            server,
        } = stage_destination(stage, route);
        if location == Location::Remote {
            subject.runtime = Some("http");
        }
        let mut options = options.clone();
        // A prompt describes a model run, not the stages of a pipeline.
        if let Some(observation) = options.observation.as_mut() {
            observation.prompt = None;
            observation.parent = parent.clone();
        }
        let run = RunObservation::start(RunStart {
            subject,
            input: Some(input),
            history: &[],
            options: Some(&options),
            generation: None,
            streaming: false,
            location,
            parent,
            provider: provider.as_deref(),
            server,
        });
        run.operation()
            .set_attribute(KeyValue::new("xybrid.stage.name", stage.name.clone()));
        for attribute in routing_attributes(route) {
            run.operation().set_attribute(attribute);
        }
        let attempt = run.attempt(AttemptSpec {
            reason: AttemptReason::Initial,
            location,
            provider: provider.as_deref(),
            model: stage.model.as_deref(),
        });
        Self::Inference(Box::new(StageInference {
            run,
            attempt,
            resources: crate::telemetry::begin_resource_run(),
        }))
    }

    /// Ends the stage from its outcome, returning the stage inference reference.
    pub(crate) fn finish(
        self,
        result: &Result<
            xybrid_core::orchestrator::StageExecutionResult,
            xybrid_core::orchestrator::OrchestratorError,
        >,
    ) -> Option<ObservationRef> {
        match self {
            Self::Warmup { _scope } => None,
            Self::Inference(inference) => {
                let StageInference {
                    run,
                    attempt,
                    resources,
                } = *inference;
                run.resource(resources.finish().as_ref());
                match result {
                    Ok(stage) => {
                        attempt.end_with_output(&Ok::<(), SdkError>(()), Some(&stage.output));
                        run.succeed(&stage.output)
                    }
                    Err(error) => {
                        let error = SdkError::pipeline(error.to_string());
                        attempt.end_with_output(
                            &Err::<(), _>(SdkError::pipeline(error.to_string())),
                            None,
                        );
                        run.fail(&error);
                        None
                    }
                }
            }
        }
    }
}

/// A future whose every poll runs with `recorder` installed, so stages of
/// work that moves between executor threads stay under one operation without
/// holding a thread-local guard across an await.
pub(crate) struct Instrumented<F> {
    recorder: Arc<StageRecorder>,
    inner: std::pin::Pin<Box<F>>,
}

impl<F: std::future::Future> std::future::Future for Instrumented<F> {
    type Output = F::Output;
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<F::Output> {
        let _scope = RecorderScope::install(self.recorder.clone());
        self.inner.as_mut().poll(cx)
    }
}

/// An `invoke_workflow` operation for async work: no thread-local guard is
/// held; [`AsyncWorkflow::instrument`] installs it on every poll instead.
pub(crate) struct AsyncWorkflow {
    op: WorkflowOperation,
    recorder: Arc<StageRecorder>,
}

impl AsyncWorkflow {
    pub(crate) fn start(name: &str, options: &crate::run_options::RunOptions) -> Self {
        let obs_opts = options.observation.as_ref();
        let session = resolve_session(obs_opts);
        let mut start = StartOptions::default().with_trace(option_trace(obs_opts));
        start.parent = obs_opts
            .and_then(|o| o.parent.clone())
            .or_else(ambient_parent);
        if let Some(o) = obs_opts {
            start.attributes = o.attributes.clone();
        }
        let conversation_id = conversation_id(obs_opts);
        if let Some(id) = &conversation_id {
            start
                .attributes
                .push(KeyValue::new(semconv::attr::CONVERSATION_ID, id.clone()));
        }
        let suppressed = start.trace == TraceMode::Suppress;
        let op = session.observer().start_workflow(name, start);
        let recorder = Arc::new(
            StageRecorder::for_session(session, op.context(), suppressed)
                .with_conversation_id(conversation_id),
        );
        Self { op, recorder }
    }

    /// The workflow's context, the parent of its stages.
    pub(crate) fn context(&self) -> Option<ObservationContext> {
        self.op.context()
    }

    pub(crate) fn instrument<F: std::future::Future>(&self, future: F) -> Instrumented<F> {
        Instrumented {
            recorder: self.recorder.clone(),
            inner: Box::pin(future),
        }
    }

    pub(crate) fn end<T, E: fmt::Display>(self, result: &Result<T, E>) -> Option<ObservationRef> {
        let reference = self.op.observation_ref();
        match result {
            Ok(_) => {
                self.op.end();
                reference
            }
            Err(_) => {
                self.op.end_with_error(&ErrorInfo::new("pipeline_error"));
                None
            }
        }
    }
}

/// In-memory span capture through an application-owned tracer provider, for
/// tests that assert exported spans without a network collector.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use opentelemetry_sdk::error::OTelSdkResult;
    use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanProcessor};

    /// Every ended span, in end order.
    #[derive(Clone, Debug, Default)]
    pub(crate) struct Memory(Arc<Mutex<Vec<SpanData>>>);

    impl SpanProcessor for Memory {
        fn on_start(&self, _: &mut opentelemetry_sdk::trace::Span, _: &opentelemetry::Context) {}
        fn on_end(&self, span: SpanData) {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(span);
        }
        fn force_flush(&self) -> OTelSdkResult {
            Ok(())
        }
        fn shutdown_with_timeout(&self, _: Duration) -> OTelSdkResult {
            Ok(())
        }
    }

    impl Memory {
        pub(crate) fn spans(&self) -> Vec<SpanData> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }

        /// The only span named `name`.
        pub(crate) fn named(&self, name: &str) -> SpanData {
            let matching: Vec<_> = self
                .spans()
                .into_iter()
                .filter(|s| s.name == name)
                .collect();
            assert_eq!(
                matching.len(),
                1,
                "expected one span {name:?}, have {:?}",
                self.spans()
                    .iter()
                    .map(|s| s.name.to_string())
                    .collect::<Vec<_>>()
            );
            matching.into_iter().next().unwrap()
        }

        /// The span with `reference`'s id.
        pub(crate) fn by_ref(&self, reference: &ObservationRef) -> SpanData {
            self.spans()
                .into_iter()
                .find(|s| span_id(s) == reference.span_id)
                .unwrap_or_else(|| panic!("no exported span for {reference:?}"))
        }
    }

    /// A borrowed session over an application-owned provider that keeps every
    /// span in memory. Keep the provider alive for the test.
    pub(crate) fn memory_session() -> (ObservabilitySession, Memory, SdkTracerProvider) {
        let memory = Memory::default();
        let provider = SdkTracerProvider::builder()
            .with_span_processor(memory.clone())
            .build();
        let observer = Observer::builder().tracer_provider(&provider).build();
        (ObservabilitySession::borrowed(observer), memory, provider)
    }

    /// [`memory_session`] whose session allowlists trace context to each of
    /// `origins` (third-party HTTP(S) origins such as a loopback gateway).
    pub(crate) fn memory_session_allowing(
        origins: &[&str],
    ) -> (ObservabilitySession, Memory, SdkTracerProvider) {
        let memory = Memory::default();
        let provider = SdkTracerProvider::builder()
            .with_span_processor(memory.clone())
            .build();
        let mut config = ObservabilityConfig::with_observer(
            Observer::builder().tracer_provider(&provider).build(),
        );
        for origin in origins {
            config = config
                .propagate_context_to(origin)
                .expect("test origins are HTTP(S) origins");
        }
        (ObservabilitySession::new(config).unwrap(), memory, provider)
    }

    /// Serializes tests that replace the process-wide default session, so
    /// parallel test threads never observe each other's defaults.
    pub(crate) fn default_session_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn attr(span: &SpanData, key: &str) -> Option<String> {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.to_string())
    }

    pub(crate) fn span_id(span: &SpanData) -> String {
        span.span_context.span_id().to_string()
    }

    pub(crate) fn parent_id(span: &SpanData) -> String {
        span.parent_span_id.to_string()
    }

    pub(crate) fn trace_id(span: &SpanData) -> String {
        span.span_context.trace_id().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    #[test]
    fn url_address_extracts_only_http_host_and_port() {
        for (url, host, port) in [
            ("https://[::1]", Some("::1"), None),
            ("http://[::1]:8080/v1", Some("::1"), Some(8080)),
            (
                "https://user:secret@gateway.example:8443/v1?token=private#fragment",
                Some("gateway.example"),
                Some(8443),
            ),
            (
                "https://user:secret@[2001:db8::1]?token=private#fragment",
                Some("2001:db8::1"),
                None,
            ),
            (
                "https://gateway.example?token=private#fragment",
                Some("gateway.example"),
                None,
            ),
            (
                "https://gateway.example:443/v1",
                Some("gateway.example"),
                None,
            ),
            ("not a URL", None, None),
            ("https://[::1", None, None),
            ("https://gateway.example:invalid", None, None),
            ("file:///private/model", None, None),
        ] {
            assert_eq!(url_address(url), (host.map(str::to_owned), port), "{url}");
        }
    }

    /// One bounded local request; the long batch delay lets shutdown export all
    /// test spans together, without sleeps or a dependency on a collector service.
    fn collector() -> (
        OtlpConfig,
        mpsc::Receiver<String>,
        std::thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "no OTLP export received");
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("collector accept: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut chunk = [0_u8; 4096];
            let (start, length) = loop {
                let n = stream.read(&mut chunk).unwrap();
                assert_ne!(n, 0, "collector received an incomplete request");
                request.extend_from_slice(&chunk[..n]);
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]);
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .expect("OTLP HTTP export has a content length");
                    if request.len() >= end + 4 + length {
                        break (end + 4, length);
                    }
                }
            };
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
            tx.send(String::from_utf8(request[start..start + length].to_vec()).unwrap())
                .unwrap();
        });
        let config = OtlpConfig::default()
            .with_traces(Some(
                SignalEndpoint::new(format!("http://{address}/v1/traces"))
                    .with_protocol(Protocol::HttpJson)
                    .with_timeout(Duration::from_secs(5)),
            ))
            .with_batch(BatchLimits {
                scheduled_delay: Duration::from_secs(60),
                ..BatchLimits::default()
            });
        (config, rx, handle)
    }

    fn spans(body: &serde_json::Value) -> Vec<&serde_json::Value> {
        body["resourceSpans"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r["scopeSpans"].as_array().unwrap())
            .flat_map(|s| s["spans"].as_array().unwrap())
            .collect()
    }

    fn attribute<'a>(span: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
        span["attributes"]
            .as_array()?
            .iter()
            .find(|a| a["key"] == key)
            .map(|a| &a["value"])
    }

    #[test]
    fn stage_metadata_exports_only_validated_measurements_and_descriptors() {
        use xybrid_observability::otlp::ObservabilityPipeline;

        for capture in [false, true] {
            let (config, exports, collector) = collector();
            let pipeline = ObservabilityPipeline::build(config).unwrap();
            let policy = if capture {
                ContentPolicy::default().with_redactor(SubstringRedactor::new(["secret-content"]))
            } else {
                ContentPolicy::metadata_only()
            };
            let observer = pipeline.observer_with(policy);
            let options = crate::RunOptions {
                observation: Some(ObservationOptions {
                    observer: Some(observer.clone()),
                    ..ObservationOptions::default()
                }),
                ..crate::RunOptions::default()
            };
            let input = Envelope::new(EnvelopeKind::Text("prompt-text secret-content".into()));
            let system = Envelope::new(EnvelopeKind::Text("system-text secret-content".into()))
                .with_role(xybrid_core::ir::MessageRole::System);
            let run = RunObservation::start(RunStart {
                subject: RunSubject {
                    operation: "chat",
                    task: Some("text_generation".into()),
                    runtime: Some("llama.cpp"),
                    model: "test-model".into(),
                    format: Some("gguf"),
                    size_bytes: None,
                    input_modalities: vec![Modality::Text],
                    output_modalities: vec![Modality::Text],
                    output_type: None,
                    is_llm: true,
                },
                input: Some(&input),
                history: &[system],
                options: Some(&options),
                generation: None,
                streaming: false,
                location: Location::Local,
                parent: None,
                provider: None,
                server: None,
            });
            {
                let _stage = xybrid_core::tracing::SpanGuard::new("core-stage");
                for (key, value) in [
                    ("prompt_tokens", "8"),
                    ("tokens_generated", "3"),
                    ("prompt_cached_tokens", "2"),
                    ("ttft_ms", "2.5"),
                    ("mean_itl_ms", "1.5"),
                    ("p95_itl_ms", "4"),
                    ("emitted_chunks", "3"),
                    ("choice_count", "0005"),
                    ("backend", "llamacpp"),
                    ("execution_provider", "cpu"),
                    ("provider", "openai"),
                    ("quantization", "Q4_K_M"),
                    ("task", "chat"),
                    ("span_kind", "cpu"),
                    ("stage_kind", "llm"),
                    ("finish_reason", "max_tokens"),
                    ("unknown", "unknown-private"),
                    ("model", "/private/model-path.gguf"),
                    ("encoder_file", "/private/encoder-path.onnx"),
                    ("model_id", "/private/model-id"),
                    ("version", "version-private"),
                    ("stages", "snapshot-private secret-content"),
                    ("input", "input-snapshot-private secret-content"),
                    ("output", "output-snapshot-private secret-content"),
                    ("error", "stage-error-private secret-content"),
                    // Even a known descriptor key must not become a side channel.
                    ("backend", "backend-private"),
                    ("execution_provider", "accelerator-private"),
                    ("provider", "provider-private"),
                    ("quantization", "quantization-private"),
                    ("stage_kind", "stage-kind-private"),
                    ("task", "task-private"),
                    ("prompt_tokens", "token-count-private"),
                    ("choice_count", "choice-count-private"),
                    ("choice_count", "-1"),
                    ("choice_count", "18446744073709551616"),
                    ("ttft_ms", "NaN"),
                    ("mean_itl_ms", "-1"),
                ] {
                    xybrid_core::tracing::add_metadata(key, value);
                }
            }
            let harvest = run.stages.harvest();
            assert_eq!(harvest.tokens_in, Some(8));
            assert_eq!(harvest.tokens_out, Some(3));
            assert_eq!(harvest.cached_tokens, Some(2));
            assert_eq!(harvest.ttft_ms, Some(2.5));
            assert_eq!(harvest.mean_itl_ms, Some(1.5));
            assert_eq!(harvest.backend.as_deref(), Some("llamacpp"));
            assert_eq!(harvest.quantization.as_deref(), Some("q4_k_m"));
            let mut output = Envelope::new(EnvelopeKind::Text("output-text secret-content".into()));
            for (key, value) in [
                ("completion_tokens", "4"),
                ("backend", "envelope-backend-private"),
                ("provider", "envelope-provider-private"),
                ("execution_provider", "envelope-accelerator-private"),
                ("quantization", "envelope-quantization-private"),
                ("finish_reason", "envelope-finish-private"),
                ("untrusted", "envelope-unknown-private"),
                ("reasoning_content", "reasoning-text secret-content"),
            ] {
                output.metadata.insert(key.into(), value.into());
            }
            run.succeed(&output).expect("logical inference is traced");

            // Automatic errors are classification-only; the returned SDK error
            // remains unchanged. Explicit custom identities are still supported.
            let error = SdkError::inference("sdk-error-private secret-content");
            let failure = observer.start_inference(
                InferenceRequest::new("chat", "my-custom-provider")
                    .with_model("organization/custom-model"),
                StartOptions::default(),
            );
            failure.end_with_error(&error_info(&error));
            assert!(error.to_string().contains("sdk-error-private"));

            pipeline.shutdown(Duration::from_secs(5)).unwrap();
            let raw = exports.recv_timeout(Duration::from_secs(5)).unwrap();
            collector.join().unwrap();
            assert!(
                !raw.contains("private"),
                "untrusted metadata escaped: {raw}"
            );
            assert!(
                !raw.contains("secret-content"),
                "content redaction was bypassed"
            );
            for content in [
                "prompt-text",
                "system-text",
                "output-text",
                "reasoning-text",
            ] {
                assert_eq!(
                    raw.contains(content),
                    capture,
                    "content policy for {content}"
                );
            }
            assert_eq!(raw.contains("[REDACTED]"), capture);
            let body: serde_json::Value = serde_json::from_str(&raw).unwrap();
            let spans = spans(&body);
            assert_eq!(
                spans.len(),
                3,
                "stage, logical inference and explicit failure"
            );
            let stage = spans.iter().find(|s| s["name"] == "core-stage").unwrap();
            let logical = spans
                .iter()
                .find(|s| s["spanId"] == stage["parentSpanId"])
                .unwrap();
            for (key, expected) in [
                ("prompt_tokens", "8"),
                ("tokens_generated", "3"),
                ("backend", "llamacpp"),
                ("execution_provider", "cpu"),
                ("provider", "openai"),
                ("quantization", "q4_k_m"),
                ("task", "chat"),
                ("stage_kind", "llm"),
                ("span_kind", "cpu"),
                ("ttft_ms", "2.5"),
                ("mean_itl_ms", "1.5"),
                ("p95_itl_ms", "4"),
                ("emitted_chunks", "3"),
                ("choice_count", "5"),
                ("finish_reason", "length"),
            ] {
                assert_eq!(
                    attribute(stage, &format!("xybrid.stage.{key}")).unwrap()["stringValue"],
                    expected
                );
            }
            for (key, expected) in [
                ("gen_ai.usage.input_tokens", "8"),
                ("gen_ai.usage.output_tokens", "4"),
                ("gen_ai.usage.cache_read.input_tokens", "2"),
            ] {
                assert_eq!(attribute(logical, key).unwrap()["intValue"], expected);
            }
            assert_eq!(
                attribute(logical, "xybrid.genai.runtime.name").unwrap()["stringValue"],
                "llama.cpp"
            );
            assert_eq!(
                attribute(logical, "xybrid.upstream.provider").unwrap()["stringValue"],
                "openai"
            );
            assert_eq!(
                attribute(logical, "xybrid.genai.accelerator").unwrap()["stringValue"],
                "cpu"
            );
            assert_eq!(
                attribute(logical, "xybrid.genai.model.quantization").unwrap()["stringValue"],
                "q4_k_m"
            );
            assert_eq!(
                attribute(logical, "gen_ai.response.finish_reasons").unwrap()["arrayValue"]
                    ["values"][0]["stringValue"],
                "_OTHER"
            );
            let failed = spans.iter().find(|s| s["status"]["code"] == 2).unwrap();
            assert_eq!(
                attribute(failed, "error.type").unwrap()["stringValue"],
                "inference_error"
            );
            assert_eq!(failed["status"]["message"], "inference_error");
            assert_eq!(
                attribute(failed, "gen_ai.provider.name").unwrap()["stringValue"],
                "my-custom-provider"
            );
            assert_eq!(
                attribute(failed, "gen_ai.request.model").unwrap()["stringValue"],
                "organization/custom-model"
            );
        }
    }

    #[test]
    fn harvest_rejects_invalid_measurements_and_normalizes_unknown_finish_reason() {
        let mut h = Harvest::default();
        h.absorb([("tokens_in", "8"), ("ttft_ms", "2.5"), ("mean_itl_ms", "1")].into_iter());
        for value in ["NaN", "inf", "-1", "arbitrary-secret"] {
            h.absorb(
                [
                    ("ttft_ms", value),
                    ("mean_itl_ms", value),
                    ("tokens_in", value),
                ]
                .into_iter(),
            );
        }
        h.absorb(
            [
                ("finish_reason", "arbitrary-secret"),
                ("provider", "arbitrary-secret"),
            ]
            .into_iter(),
        );
        assert_eq!(h.tokens_in, Some(8));
        assert_eq!(h.ttft_ms, Some(2.5));
        assert_eq!(h.mean_itl_ms, Some(1.0));
        assert_eq!(h.finish_reason.as_deref(), Some("_OTHER"));
        assert_eq!(h.provider, None);
    }

    #[test]
    fn ambient_application_context_survives_worker_hops_and_recorder_precedence() {
        let application = ObservationContext::from_traceparent(
            "00-11111111111111111111111111111111-2222222222222222-01",
            Some("vendor=app"),
        )
        .unwrap();
        let _application = application.to_otel().attach();
        assert_eq!(ambient_parent(), Some(application.clone()));
        let captured = ambient_parent().unwrap();
        std::thread::spawn(move || {
            assert!(ambient_parent().is_none());
            let _scope = ParentScope::install(captured.clone());
            assert_eq!(ambient_parent(), Some(captured));
        })
        .join()
        .unwrap();
        let nested = ObservationContext::from_traceparent(
            "00-33333333333333333333333333333333-4444444444444444-01",
            None,
        )
        .unwrap();
        {
            let _scope = ParentScope::install(nested.clone());
            assert_eq!(ambient_parent(), Some(nested));
        }
        {
            let _scope =
                RecorderScope::install(Arc::new(StageRecorder::new(Observer::disabled(), None)));
            assert!(
                ambient_parent().is_none(),
                "a suppressed recorder must not inherit the outer application span"
            );
        }
        assert_eq!(ambient_parent(), Some(application));
    }

    #[test]
    fn streaming_workflow_children_keep_observer_and_suppression_controls() {
        use crate::run_options::RunOptions;
        use httpmock::prelude::*;
        use xybrid_observability::otlp::ObservabilityPipeline;

        let server = MockServer::start();
        let pipeline = ObservabilityPipeline::build(OtlpConfig::default().with_traces(Some(
            SignalEndpoint::new(server.url("/v1/traces")).with_protocol(Protocol::HttpJson),
        )))
        .unwrap();
        let observer = pipeline.observer();
        let caller = ObservationContext::from_traceparent(
            "00-11111111111111111111111111111111-2222222222222222-01",
            None,
        )
        .unwrap();
        for trace in [TraceMode::Default, TraceMode::Suppress] {
            let options = RunOptions::default().with_observation(ObservationOptions {
                parent: Some(caller.clone()),
                observer: Some(observer.clone()),
                trace,
                conversation_id: Some("conversation".into()),
                attributes: vec![KeyValue::new("custom", "kept")],
                ..ObservationOptions::default()
            });
            let workflow = WorkflowScope::start("streaming", &options);
            let workflow_context = workflow.op.context();
            let child = workflow.child_options(&options);
            let controls = child.observation.as_ref().unwrap();
            assert_eq!(controls.trace, trace);
            assert_eq!(controls.parent, workflow_context);
            assert_eq!(controls.conversation_id.as_deref(), Some("conversation"));
            let obs = RunObservation::start(RunStart {
                subject: RunSubject {
                    operation: "chat",
                    task: None,
                    runtime: Some("llama.cpp"),
                    model: "tiny".into(),
                    format: Some("gguf"),
                    size_bytes: None,
                    input_modalities: vec![Modality::Text],
                    output_modalities: vec![Modality::Text],
                    output_type: None,
                    is_llm: true,
                },
                input: None,
                history: &[],
                options: Some(&child),
                generation: None,
                streaming: true,
                location: Location::Local,
                parent: None,
                provider: None,
                server: None,
            });
            if trace == TraceMode::Suppress {
                assert!(obs.context().is_none());
            } else {
                assert_eq!(obs.context().unwrap().trace_id(), caller.trace_id());
            }
            obs.succeed_without_output();
            workflow.end(&Ok::<(), SdkError>(()));
            if let Some(parent) = workflow_context {
                let sent = server.mock(|when, then| {
                    when.method(POST)
                        .path("/v1/traces")
                        .body_contains(format!("\"parentSpanId\":\"{}\"", parent.span_id()))
                        .body_contains("chat tiny");
                    then.status(200);
                });
                pipeline.force_flush(Duration::from_secs(2)).unwrap();
                sent.assert_hits(1);
            }
        }
        assert_eq!(
            pipeline.stats().traces.accepted,
            2,
            "the suppressed workflow and inference must enqueue no spans"
        );
        pipeline.shutdown(Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn harvest_keeps_last_known_values() {
        let mut h = Harvest::default();
        h.absorb([("tokens_out", "3"), ("ttft_ms", "12.5")].into_iter());
        h.absorb([("tokens_out", "9"), ("tokens_in", "4"), ("tokens_out", "x")].into_iter());
        assert_eq!(h.tokens_out, Some(9));
        assert_eq!(h.tokens_in, Some(4));
        assert_eq!(h.ttft_ms, Some(12.5));
    }

    fn test_run(options: &crate::RunOptions) -> RunObservation {
        RunObservation::start(RunStart {
            subject: RunSubject {
                operation: "chat",
                task: None,
                runtime: Some("llama.cpp"),
                model: "test".into(),
                format: Some("gguf"),
                size_bytes: None,
                input_modalities: vec![Modality::Text],
                output_modalities: vec![Modality::Text],
                output_type: None,
                is_llm: true,
            },
            input: None,
            history: &[],
            options: Some(options),
            generation: None,
            streaming: true,
            location: Location::Local,
            parent: None,
            provider: None,
            server: None,
        })
    }

    #[test]
    fn attempts_sum_usage_once_and_offset_only_successful_first_token() {
        let (config, exports, collector) = collector();
        let session = ObservabilitySession::new(ObservabilityConfig::otlp(config)).unwrap();
        let options = crate::RunOptions::default()
            .with_observation(ObservationOptions::default().with_session(&session));
        let run = test_run(&options);
        let failed = run.attempt(AttemptSpec {
            reason: AttemptReason::Initial,
            location: Location::Local,
            provider: None,
            model: None,
        });
        {
            let _stage = xybrid_core::tracing::SpanGuard::new("failed-runtime");
            xybrid_core::tracing::add_metadata("tokens_out", "1");
            xybrid_core::tracing::add_metadata("tokens_out", "3");
            xybrid_core::tracing::add_metadata("tokens_in", "4");
            xybrid_core::tracing::add_metadata("ttft_ms", "1");
        }
        failed.end(&Err(SdkError::inference("retry")));
        assert_eq!(run.stages.harvest().ttft_ms, None);
        let mut winner = run.attempt(AttemptSpec {
            reason: AttemptReason::Fallback,
            location: Location::Remote,
            provider: None,
            model: None,
        });
        // Deterministic timing input: the runtime's 5 ms starts 100 ms
        // after the logical request, even though this test does no waiting.
        winner.started = run.started + Duration::from_millis(100);
        let mut output = Envelope::new(EnvelopeKind::Text("done".into()));
        for (key, value) in [
            ("tokens_in", "10"),
            ("tokens_out", "7"),
            ("cache_read_input_tokens", "0"),
            ("ttft_ms", "5"),
        ] {
            output.metadata.insert(key.into(), value.into());
        }
        winner.end_adapter(&Ok::<_, SdkError>(output.clone()));
        assert_eq!(run.stages.harvest().tokens_in, Some(14));
        assert_eq!(run.stages.harvest().tokens_out, Some(10));
        assert_eq!(run.stages.harvest().cached_tokens, Some(0));
        assert_eq!(run.stages.harvest().ttft_ms, Some(105.0));
        run.succeed(&output);
        session.shutdown(Duration::from_secs(2)).unwrap();
        let body: serde_json::Value = serde_json::from_str(&exports.recv().unwrap()).unwrap();
        let all = spans(&body);
        let logical = all.iter().find(|s| s["name"] == "chat test").unwrap();
        assert_eq!(
            attribute(logical, "gen_ai.usage.input_tokens").unwrap()["intValue"],
            "14"
        );
        assert_eq!(
            attribute(logical, "gen_ai.usage.output_tokens").unwrap()["intValue"],
            "10"
        );
        assert_eq!(
            attribute(logical, semconv::attr::RESPONSE_TIME_TO_FIRST_TOKEN).unwrap()["doubleValue"],
            0.105
        );
        collector.join().unwrap();
    }

    #[test]
    fn asr_workflows_start_on_feed_and_keep_decoder_and_terminal_references() {
        use test_support::*;
        use xybrid_core::streaming::{DecoderObserver, StreamError};

        let (session, memory, _provider) = memory_session();
        let parent = ObservationContext::from_traceparent(
            "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
            None,
        )
        .unwrap();
        let options = ObservationOptions::default()
            .with_session(&session)
            .with_parent(parent.clone())
            .with_conversation_id("utterance");
        let mut metadata = ModelMetadata::onnx("decoder", "1", "model.onnx");
        metadata
            .metadata
            .insert("task".into(), serde_json::json!("speech-recognition"));
        let asr = AsrObservation::for_model(Some(&options), Some(("decoder", &metadata)));

        // Opening, warming and flushing a stream that was never fed records no workflow.
        assert_eq!(asr.reference(), None);
        drop(asr.warmup_span().1);
        {
            let _scope = asr.enter();
        }
        assert_eq!(asr.finish(&Ok::<_, SdkError>(())), None);

        // An abandoned utterance is cancelled by reset.
        drop(asr.enter_feed());
        let abandoned = asr.reference().unwrap();
        asr.reset();
        assert_eq!(asr.reference(), None);

        let input = Envelope::new(EnvelopeKind::Audio(vec![0; 32]));
        let output = Envelope::new(EnvelopeKind::Text("transcription".into()));
        let mut decoders = Vec::new();
        for _ in 0..2 {
            let _scope = asr.enter_feed();
            asr.start(&metadata, &input).finish(&Ok(output.clone()));
            let reference = asr.decoder_reference().unwrap();
            // Cached partial reads keep the reference and record nothing.
            for _ in 0..5 {
                assert_eq!(asr.decoder_reference(), Some(reference.clone()));
            }
            decoders.push(reference);
        }
        let workflow = asr.reference().unwrap();
        {
            let _scope = asr.enter_feed();
            asr.start(&metadata, &input)
                .finish(&Err(StreamError::InferenceError("decode failed".into())));
        }
        assert_eq!(
            asr.decoder_reference().as_ref(),
            decoders.last(),
            "a failed decode keeps the last transcript's reference"
        );
        assert_eq!(asr.reference(), Some(workflow.clone()));
        {
            let _scope = asr.enter();
            assert_eq!(asr.finish(&Ok::<_, SdkError>(())), Some(workflow.clone()));
        }
        // A repeated flush returns the cached reference and starts nothing.
        assert_eq!(
            asr.finish(&Err::<(), _>(SdkError::inference("late"))),
            Some(workflow.clone())
        );
        asr.reset();
        assert_eq!(asr.decoder_reference(), None);
        drop(asr.enter_feed());
        let dropped = asr.reference().unwrap();
        drop(asr);

        let spans = memory.spans();
        let workflows: Vec<_> = spans
            .iter()
            .filter(|s| s.name == "invoke_workflow asr_session")
            .collect();
        assert_eq!(
            workflows.len(),
            3,
            "abandoned, completed and dropped utterances"
        );
        assert_eq!(
            spans
                .iter()
                .filter(|s| s.name == "generate_content decoder")
                .count(),
            3,
            "only real decoder executions are inferences"
        );
        assert!(spans.iter().any(|s| s.name == "warmup"));
        for reference in &decoders {
            let decode = memory.by_ref(reference);
            assert_eq!(parent_id(&decode), workflow.span_id);
            assert_eq!(trace_id(&decode), parent.trace_id());
        }
        let failed = spans
            .iter()
            .find(|s| {
                s.name == "generate_content decoder" && attr(s, semconv::attr::ERROR_TYPE).is_some()
            })
            .unwrap();
        assert_eq!(parent_id(failed), workflow.span_id);
        for reference in [&abandoned, &dropped] {
            assert_eq!(
                attr(&memory.by_ref(reference), semconv::attr::ERROR_TYPE).as_deref(),
                Some("cancelled")
            );
        }
        let completed = memory.by_ref(&workflow);
        assert_eq!(parent_id(&completed), parent.span_id());
        assert_eq!(attr(&completed, semconv::attr::ERROR_TYPE), None);
        assert_eq!(
            attr(&completed, semconv::attr::CONVERSATION_ID).as_deref(),
            Some("utterance")
        );
        assert_eq!(
            attr(&completed, semconv::attr::WORKFLOW_NAME).as_deref(),
            Some("asr_session")
        );
    }

    #[test]
    fn tool_definitions_calls_and_results_are_captured_automatically() {
        use test_support::*;
        use xybrid_core::gateway::Tool;
        use xybrid_core::runtime_adapter::types::GenerationConfig;

        let tool: Tool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }))
        .unwrap();
        let config = GenerationConfig {
            tools: vec![tool],
            ..GenerationConfig::default()
        };
        let call = |id: &str| {
            serde_json::json!([{
                "id": id, "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }])
            .to_string()
        };
        // A history assistant turn that requested a tool, then the continuation.
        let mut assistant = Envelope::new(EnvelopeKind::Text(String::new()))
            .with_role(xybrid_core::ir::MessageRole::Assistant);
        assistant
            .metadata
            .insert(Envelope::TOOL_CALLS_METADATA_KEY.into(), call("call_h"));
        let continuation = Envelope::tool_results(
            "what is the weather?",
            "<unparseable prior turn>",
            &[xybrid_core::ir::ToolCallResult {
                call_id: "call_0".into(),
                name: "get_weather".into(),
                content: serde_json::json!({"temp_c": 21}),
            }],
        );
        let mut output = Envelope::new(EnvelopeKind::Text(String::new()));
        output
            .metadata
            .insert(Envelope::TOOL_CALLS_METADATA_KEY.into(), call("call_1"));

        for capture in [true, false] {
            let (session, memory, _provider) = memory_session();
            let session = if capture {
                session
            } else {
                ObservabilitySession::borrowed(
                    Observer::builder()
                        .tracer_provider(&_provider)
                        .content_policy(ContentPolicy::metadata_only())
                        .build(),
                )
            };
            let options = crate::RunOptions::default()
                .with_observation(ObservationOptions::default().with_session(&session));
            let run = RunObservation::start(RunStart {
                subject: RunSubject::remote_chat("tool-model"),
                input: Some(&continuation),
                history: std::slice::from_ref(&assistant),
                options: Some(&options),
                generation: Some(&config),
                streaming: false,
                location: Location::Local,
                parent: None,
                provider: None,
                server: None,
            });
            let reference = run.succeed(&output).unwrap();
            let span = memory.by_ref(&reference);
            let definitions = attr(&span, semconv::attr::TOOL_DEFINITIONS)
                .expect("tool definitions are schema content in every capture mode");
            assert!(definitions.contains("get_weather"), "{definitions}");
            assert!(
                attr(&span, semconv::attr::RESPONSE_FINISH_REASONS)
                    .is_some_and(|r| r.contains("tool_call")),
                "a turn that ended on tool calls reports tool_call"
            );
            let input = attr(&span, semconv::attr::INPUT_MESSAGES);
            let output_messages = attr(&span, semconv::attr::OUTPUT_MESSAGES);
            if !capture {
                assert_eq!(input, None);
                assert_eq!(output_messages, None);
                continue;
            }
            let input: serde_json::Value = serde_json::from_str(&input.unwrap()).unwrap();
            let roles: Vec<&str> = input
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m["role"].as_str().unwrap())
                .collect();
            assert_eq!(roles, ["assistant", "user", "assistant", "tool"]);
            assert_eq!(input[0]["parts"][0]["type"], "tool_call");
            assert_eq!(input[0]["parts"][0]["id"], "call_h");
            assert_eq!(input[1]["parts"][0]["content"], "what is the weather?");
            // The prior protocol text did not parse: the results name their calls.
            assert_eq!(input[2]["parts"][0]["type"], "tool_call");
            assert_eq!(input[2]["parts"][0]["id"], "call_0");
            assert_eq!(input[3]["parts"][0]["type"], "tool_call_response");
            assert_eq!(input[3]["parts"][0]["id"], "call_0");
            assert_eq!(input[3]["parts"][0]["response"]["temp_c"], 21);
            let output: serde_json::Value =
                serde_json::from_str(&output_messages.unwrap()).unwrap();
            let call = output[0]["parts"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["type"] == "tool_call")
                .expect("the model's tool call is an output part");
            assert_eq!(call["name"], "get_weather");
            assert_eq!(call["id"], "call_1");
            assert_eq!(call["arguments"]["city"], "Paris");
        }
    }

    #[test]
    fn lazy_reload_and_warmup_materializations_are_load_operations() {
        use test_support::*;
        let (session, memory, _provider) = memory_session();
        let options = crate::RunOptions::default().with_observation(
            ObservationOptions::default()
                .with_session(&session)
                .with_conversation_id("lazy-conversation"),
        );
        let mut subject = test_run_subject();
        subject.model = "lazy-model".into();
        subject.size_bytes = Some(8192);
        let run = RunObservation::start(RunStart {
            subject,
            input: None,
            history: &[],
            options: Some(&options),
            generation: None,
            streaming: false,
            location: Location::Local,
            parent: None,
            provider: None,
            server: None,
        });
        {
            let _execute = xybrid_core::tracing::SpanGuard::new("execute:lazy-model");
            drop(xybrid_core::tracing::model_load_stage("llamacpp"));
        }
        // An automatically released model reloads on its next run.
        mark_reload();
        {
            let attempt = run.attempt(AttemptSpec {
                reason: AttemptReason::Initial,
                location: Location::Local,
                provider: None,
                model: None,
            });
            {
                let _load = xybrid_core::tracing::model_load_stage("ort");
                xybrid_core::tracing::mark_stage_failed();
            }
            attempt.end_adapter(&Err::<Envelope, _>("load failed"));
        }
        let run_ref = run.succeed_without_output().unwrap();
        let warmup_root = memory_session_root(&session);
        let warmup_scope = warmup_root.enter();
        let warmup = SpanScope::start_loading(
            "warmup lazy-model",
            Vec::new(),
            Some(LoadSubject::new("lazy-model", Some("gguf"), "warmup", None)),
        );
        drop(xybrid_core::tracing::model_load_stage("llamacpp"));
        warmup.end(&Ok::<(), SdkError>(()));
        drop(warmup_scope);
        warmup_root.end();
        // Without a known model a load stays a generic stage (no fabricated name).
        {
            let root = memory_session_root(&session);
            let _scope = RecorderScope::install(Arc::new(StageCapture::new(
                session.observer().clone(),
                root.context(),
            )));
            drop(xybrid_core::tracing::model_load_stage("ort"));
            root.end();
        }

        let loads: Vec<_> = memory
            .spans()
            .into_iter()
            .filter(|s| s.name == "load_model lazy-model")
            .collect();
        assert_eq!(loads.len(), 3);
        let trigger = |s: &opentelemetry_sdk::trace::SpanData| attr(s, LOAD_TRIGGER).unwrap();
        let lazy = &loads[0];
        assert_eq!(trigger(lazy), "lazy");
        assert_eq!(
            parent_id(lazy),
            span_id(&memory.named("execute:lazy-model")),
            "a load is a child of the stage that triggered it"
        );
        assert_eq!(
            attr(lazy, semconv::attr::RUNTIME_NAME).as_deref(),
            Some("llama.cpp")
        );
        assert_eq!(
            attr(lazy, semconv::attr::MODEL_FORMAT).as_deref(),
            Some("gguf")
        );
        let reload = &loads[1];
        for load in [lazy, reload] {
            assert_eq!(
                attr(load, semconv::attr::CONVERSATION_ID).as_deref(),
                Some("lazy-conversation")
            );
            assert_eq!(
                attr(load, semconv::attr::MODEL_SIZE_BYTES).as_deref(),
                Some("8192")
            );
            assert_eq!(
                attr(load, semconv::attr::MODEL_CACHE_HIT),
                None,
                "materializing local weights does not establish their cache origin"
            );
        }
        assert_eq!(
            attr(&loads[2], semconv::attr::MODEL_SIZE_BYTES),
            None,
            "unknown size stays absent"
        );
        assert_eq!(trigger(reload), "reload");
        assert_eq!(
            attr(reload, semconv::attr::ERROR_TYPE).as_deref(),
            Some("load_error")
        );
        let attempt = memory
            .spans()
            .into_iter()
            .find(|s| span_id(s) == parent_id(reload))
            .unwrap();
        assert_eq!(parent_id(&attempt), run_ref.span_id);
        assert_eq!(trigger(&loads[2]), "warmup");
        assert!(memory.spans().iter().any(|s| s.name == "model_load:ort"));
        assert_eq!(
            memory
                .spans()
                .iter()
                .filter(|s| attr(s, semconv::attr::OPERATION_NAME).as_deref() == Some("load_model"))
                .count(),
            3
        );
    }

    #[test]
    fn automatic_workflows_and_worker_loads_retain_conversation_controls() {
        use test_support::*;
        let (session, memory, _provider) = memory_session();
        let controls = ObservationOptions::default()
            .with_session(&session)
            .with_conversation_id("automatic-conversation");
        let options = crate::RunOptions::default().with_observation(controls);
        let workflow = WorkflowScope::start("automatic", &options);
        let inference = RunObservation::start(RunStart {
            subject: test_run_subject(),
            input: None,
            history: &[],
            options: None,
            generation: None,
            streaming: false,
            location: Location::Local,
            parent: None,
            provider: None,
            server: None,
        });
        inference.succeed_without_output();
        let captured = CapturedContext::capture();
        std::thread::spawn(move || {
            let _context = captured.enter();
            let load =
                LoadScope::start_with_trigger("worker", None, None, "speculative_background");
            load.end(&Err::<(), _>(SdkError::load("missing artifact")));
        })
        .join()
        .unwrap();
        workflow.end(&Ok::<(), SdkError>(()));
        let asynchronous = AsyncWorkflow::start("asynchronous", &options);
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(asynchronous.instrument(async {
                let load = LoadScope::start("async-worker", None, None);
                load.end(&Ok::<(), SdkError>(()));
            }));
        asynchronous.end(&Ok::<(), SdkError>(()));
        for span in memory.spans() {
            assert_eq!(
                attr(&span, semconv::attr::CONVERSATION_ID).as_deref(),
                Some("automatic-conversation")
            );
            if attr(&span, semconv::attr::OPERATION_NAME).as_deref() == Some("load_model") {
                assert_eq!(attr(&span, semconv::attr::MODEL_SIZE_BYTES), None);
                assert_eq!(attr(&span, semconv::attr::MODEL_CACHE_HIT), None);
            }
        }
        assert_eq!(memory.spans().len(), 5);
    }

    #[test]
    fn final_load_metadata_is_unique_and_model_size_beats_transfer_progress() {
        use test_support::*;
        let (session, memory, _provider) = memory_session();
        let options = ObservationOptions::default().with_session(&session);
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("model.onnx"), b"1234").unwrap();
        std::fs::write(directory.path().join("tokens.json"), b"abc").unwrap();
        for (name, initial, describe, declared, expected) in [
            ("initial", true, true, Some(4_u64), Some("4")),
            ("late", false, true, Some(4), Some("4")),
            ("measured", false, true, None, Some("7")),
            ("transfer", false, false, None, Some("900")),
            ("unknown", false, false, None, None),
        ] {
            let mut metadata = ModelMetadata::onnx(name, "1", "model.onnx");
            metadata.files.push("tokens.json".into());
            if let Some(bytes) = declared {
                metadata.metadata.insert("size_bytes".into(), bytes.into());
            }
            std::fs::write(
                directory.path().join("model_metadata.json"),
                serde_json::to_vec(&metadata).unwrap(),
            )
            .unwrap();
            let load = LoadScope::start(name, initial.then_some(&metadata), Some(&options));
            let reference = load.load.operation.observation_ref().unwrap();
            if expected.is_some() {
                // More progress reports than the default span-attribute budget.
                for bytes in 396..652 {
                    record_load_size(bytes);
                    record_load_cache_hit(true);
                }
                if describe {
                    load.describe_directory(name, directory.path());
                    load.describe_directory(name, directory.path());
                }
                // A late archive/header update cannot replace known model bytes.
                load.size_bytes(900);
                record_load_size(u64::MAX);
                record_load_cache_hit(false);
            }
            if name == "measured" {
                load.end(&Err::<(), _>(SdkError::load("after metadata resolution")));
            } else {
                load.end(&Ok::<(), SdkError>(()));
            }
            let span = memory.by_ref(&reference);
            assert_eq!(
                attr(&span, semconv::attr::MODEL_SIZE_BYTES).as_deref(),
                expected,
                "{name}"
            );
            assert_eq!(
                attr(&span, semconv::attr::MODEL_CACHE_HIT).as_deref(),
                expected.map(|_| "true"),
                "{name}"
            );
            for key in [
                semconv::attr::MODEL_SIZE_BYTES,
                semconv::attr::MODEL_CACHE_HIT,
                semconv::attr::MODEL_FORMAT,
                semconv::attr::RUNTIME_NAME,
            ] {
                assert!(
                    span.attributes
                        .iter()
                        .filter(|a| a.key.as_str() == key)
                        .count()
                        <= 1,
                    "duplicate {key} on {name}"
                );
            }
            assert_eq!(span.dropped_attributes_count, 0, "{name}");
        }
    }

    #[test]
    fn declared_load_size_is_exported_without_inventing_missing_values() {
        use test_support::*;
        let (session, memory, _provider) = memory_session();
        let options = ObservationOptions::default().with_session(&session);
        for (declared, size_mb, expected) in [
            (
                serde_json::json!(1234),
                serde_json::Value::Null,
                Some("1234"),
            ),
            (
                serde_json::Value::Null,
                serde_json::json!(2),
                Some("2097152"),
            ),
            (serde_json::json!(-1), serde_json::Value::Null, None),
            (serde_json::json!("unknown"), serde_json::Value::Null, None),
            (serde_json::Value::Null, serde_json::json!(u64::MAX), None),
            (serde_json::json!(u64::MAX), serde_json::Value::Null, None),
        ] {
            let mut metadata = ModelMetadata::onnx("size-model", "1", "missing.onnx");
            metadata.metadata.insert("size_bytes".into(), declared);
            metadata.metadata.insert("size_mb".into(), size_mb);
            let load = LoadScope::start("size-model", Some(&metadata), Some(&options));
            let reference = load.end(&Ok::<(), SdkError>(())).unwrap();
            let span = memory.by_ref(&reference);
            assert_eq!(
                attr(&span, semconv::attr::MODEL_SIZE_BYTES).as_deref(),
                expected
            );
            assert_eq!(attr(&span, semconv::attr::MODEL_CACHE_HIT), None);
        }
        let directory = tempfile::tempdir().unwrap();
        let mut metadata = ModelMetadata::onnx("measured-size", "1", "model.onnx");
        metadata.files = vec!["model.onnx".into(), "tokens.json".into()];
        std::fs::write(
            directory.path().join("model_metadata.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        std::fs::write(directory.path().join("model.onnx"), b"1234").unwrap();
        std::fs::write(directory.path().join("tokens.json"), b"abc").unwrap();
        for (model, expected) in [("complete-files", Some("7")), ("missing-files", None)] {
            if expected.is_none() {
                std::fs::remove_file(directory.path().join("tokens.json")).unwrap();
            }
            let load = LoadScope::start(model, None, Some(&options));
            load.describe_directory(model, directory.path());
            let reference = load.end(&Ok::<(), SdkError>(())).unwrap();
            assert_eq!(
                attr(&memory.by_ref(&reference), semconv::attr::MODEL_SIZE_BYTES).as_deref(),
                expected
            );
        }
    }

    fn test_run_subject() -> RunSubject {
        RunSubject {
            operation: "chat",
            task: None,
            runtime: Some("llama.cpp"),
            model: "test".into(),
            format: Some("gguf"),
            size_bytes: None,
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Text],
            output_type: None,
            is_llm: true,
        }
    }

    fn memory_session_root(session: &ObservabilitySession) -> ManualObservation<WorkflowOperation> {
        start_workflow("root", &ObservationOptions::default().with_session(session))
    }

    #[test]
    fn operations_started_inside_an_open_stage_are_its_children() {
        use test_support::*;
        let (session, memory, _provider) = memory_session();
        let options = crate::RunOptions::default()
            .with_observation(ObservationOptions::default().with_session(&session));
        let workflow = WorkflowScope::start("nested", &options);
        let workflow_ref = workflow.op.observation_ref().unwrap();
        let stage_id;
        let captured;
        {
            let stage = xybrid_core::tracing::SpanGuard::new("inference:model");
            stage_id = stage.id();
            let tool = start_tool(
                ToolCallInfo::new("in-stage"),
                &ObservationOptions::default(),
            );
            tool.end();
            captured = CapturedContext::capture();
        }
        std::thread::spawn(move || {
            let _scope = captured.enter();
            start_tool(ToolCallInfo::new("worker"), &ObservationOptions::default()).end();
        })
        .join()
        .unwrap();
        // After the stage closes, new operations attach to the workflow again.
        start_tool(ToolCallInfo::new("after"), &ObservationOptions::default()).end();
        workflow.end(&Ok::<(), SdkError>(()));
        assert_ne!(stage_id, 0);
        let stage = memory.named("inference:model");
        assert_eq!(parent_id(&stage), workflow_ref.span_id);
        for name in ["execute_tool in-stage", "execute_tool worker"] {
            assert_eq!(parent_id(&memory.named(name)), span_id(&stage), "{name}");
        }
        assert_eq!(
            parent_id(&memory.named("execute_tool after")),
            workflow_ref.span_id
        );
    }

    #[test]
    fn routing_attributes_never_export_free_text_or_endpoints() {
        use xybrid_core::orchestrator::routing_engine::{
            LocalReliabilityHint, RouteTarget, RoutingDecision,
        };
        use xybrid_core::orchestrator::{DecisionSource, StageRoute};
        let decision = RoutingDecision {
            stage: "stage".into(),
            target: RouteTarget::Fallback("https://private.example/endpoint".into()),
            reason: "[remote] policy_deny: secret-model-id is private (confidence: 90%)".into(),
            timestamp_ms: 0,
            local_reliability_hint: LocalReliabilityHint::EMPTY,
        };
        let attributes =
            routing_attributes(&StageRoute::new(&decision, DecisionSource::Remote, false));
        let rendered: Vec<(String, String)> = attributes
            .iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();
        assert_eq!(
            rendered,
            [
                ("xybrid.routing.target".to_owned(), "fallback".to_owned()),
                ("xybrid.routing.source".to_owned(), "remote".to_owned()),
                (
                    "xybrid.routing.policy_allowed".to_owned(),
                    "false".to_owned()
                ),
            ]
        );
    }

    #[test]
    fn manual_workflow_tool_and_load_record_the_conversation_id() {
        use test_support::*;
        let (session, memory, _provider) = memory_session();
        let options = ObservationOptions::default()
            .with_session(&session)
            .with_conversation_id("conversation-7");
        start_workflow("conversation", &options).end();
        start_tool(ToolCallInfo::new("lookup"), &options).end();
        start_model_load(
            ModelLoadInfo {
                model: "m".into(),
                ..Default::default()
            },
            &options,
        )
        .end();
        start_inference(InferenceRequest::new("chat", "app"), &options).end();
        let spans = memory.spans();
        assert_eq!(spans.len(), 4);
        for span in &spans {
            assert_eq!(
                attr(span, semconv::attr::CONVERSATION_ID).as_deref(),
                Some("conversation-7"),
                "{}",
                span.name
            );
        }
    }

    #[test]
    fn async_workflows_parent_stages_across_polls_without_holding_a_guard() {
        use test_support::*;
        fn assert_send<T: Send>(_: &T) {}
        let (session, memory, _provider) = memory_session();
        let options = crate::RunOptions::default()
            .with_observation(ObservationOptions::default().with_session(&session));
        let workflow = AsyncWorkflow::start("legacy", &options);
        let future = workflow.instrument(async {
            {
                let _first = xybrid_core::tracing::SpanGuard::new("before-yield");
            }
            // Core's `pipeline:` guard in `execute_pipeline_async` is held
            // across its awaits: the poll that drops it runs under a fresh
            // scope, yet the stage still ends (not cancelled after the run).
            let held = xybrid_core::tracing::SpanGuard::new("held-across-yield");
            tokio::task::yield_now().await;
            let _second = xybrid_core::tracing::SpanGuard::new("after-yield");
            drop(held);
            Ok::<(), String>(())
        });
        assert_send(&future);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap();
        let result = runtime.block_on(future);
        let reference = workflow.end(&result).unwrap();
        assert!(xybrid_core::tracing::current_recorder().is_none());
        for name in ["before-yield", "after-yield"] {
            assert_eq!(parent_id(&memory.named(name)), reference.span_id);
        }
        let workflow_span = memory.by_ref(&reference);
        assert_eq!(workflow_span.name, "invoke_workflow legacy");
        let held = memory.named("held-across-yield");
        assert_eq!(parent_id(&held), reference.span_id);
        assert_eq!(attr(&held, semconv::attr::ERROR_TYPE), None);
        assert!(held.end_time <= workflow_span.end_time);
    }

    #[test]
    fn replaced_helper_sessions_shut_down_now_unless_pinned() {
        let _default = test_support::default_session_lock();
        // Unpinned: the previous pipeline exports before replacement returns.
        let (config, exports, first) = collector();
        let unpinned = ObservabilitySession::new(ObservabilityConfig::otlp(config)).unwrap();
        start_workflow(
            "before",
            &ObservationOptions::default().with_session(&unpinned),
        )
        .end();
        session::retire_replaced(session::DefaultSlot {
            session: unpinned,
            helper_owned: true,
        });
        assert!(
            exports.try_recv().is_ok(),
            "an unpinned replaced pipeline shuts down synchronously"
        );
        first.join().unwrap();

        // Pinned: it drains after the last pinned operation releases it.
        let (config, exports, second) = collector();
        let pinned = ObservabilitySession::new(ObservabilityConfig::otlp(config)).unwrap();
        let operation = start_workflow(
            "in-flight",
            &ObservationOptions::default().with_session(&pinned),
        );
        session::retire_replaced(session::DefaultSlot {
            session: pinned,
            helper_owned: true,
        });
        assert!(
            exports.try_recv().is_err(),
            "a pinned pipeline keeps running"
        );
        operation.end();
        drop(operation);
        let body: serde_json::Value =
            serde_json::from_str(&exports.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
        assert_eq!(spans(&body).len(), 1);
        second.join().unwrap();

        // A helper-owned default handed back by install_default_session is retired.
        let helper = ObservabilitySession::borrowed(Observer::disabled());
        let previous = session::replace_default(helper, true);
        let returned =
            install_default_session(ObservabilitySession::borrowed(Observer::disabled())).unwrap();
        assert!(returned.is_retired());
        let installed = session::take_default();
        assert!(installed.is_some_and(|slot| !slot.helper_owned));
        if let Some(previous) = previous {
            session::replace_default(previous.session, previous.helper_owned);
        }
    }

    #[test]
    fn device_id_reaches_the_exported_resource_and_is_validated() {
        let (config, exports, collector) = collector();
        let session = ObservabilitySession::new(
            ObservabilityConfig::otlp(config)
                .device_attributes(false)
                .device_id("device-42"),
        )
        .unwrap();
        start_workflow(
            "device",
            &ObservationOptions::default().with_session(&session),
        )
        .end();
        session.shutdown(Duration::from_secs(2)).unwrap();
        let body: serde_json::Value = serde_json::from_str(&exports.recv().unwrap()).unwrap();
        collector.join().unwrap();
        let resource = &body["resourceSpans"][0]["resource"]["attributes"];
        let value = |key: &str| {
            resource
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["key"] == key)
                .map(|a| a["value"]["stringValue"].clone())
        };
        assert_eq!(value("device.id"), Some(serde_json::json!("device-42")));
        assert_eq!(
            value("telemetry.sdk.name"),
            Some(serde_json::json!("xybrid"))
        );

        for invalid in ["", "line\nbreak", &"x".repeat(129)] {
            assert!(matches!(
                ObservabilitySession::new(
                    ObservabilityConfig::otlp(OtlpConfig::default()).device_id(invalid)
                ),
                Err(SdkError::ConfigError(_))
            ));
            assert!(resource_attributes(false, Some(invalid)).is_err());
        }
        let conflicting =
            OtlpConfig::default().with_resource_attribute(KeyValue::new("device.id", "other"));
        assert!(ObservabilitySession::new(
            ObservabilityConfig::otlp(conflicting).device_id("device-42")
        )
        .is_err());
        assert!(ObservabilitySession::new(
            ObservabilityConfig::with_observer(Observer::disabled()).device_id("device-42")
        )
        .is_err());
        let attributes = resource_attributes(false, Some("device-42")).unwrap();
        assert!(attributes
            .iter()
            .any(|kv| kv.key.as_str() == "device.id" && kv.value.to_string() == "device-42"));
    }

    #[test]
    fn retired_helper_keeps_pipeline_alive_until_pinned_operation_finishes() {
        let (config, exports, collector) = collector();
        let session = ObservabilitySession::new(ObservabilityConfig::otlp(config)).unwrap();
        let options = crate::RunOptions::default()
            .with_observation(ObservationOptions::default().with_session(&session));
        let run = test_run(&options);
        session.retire();
        drop(options);
        drop(session);
        assert!(run.succeed_without_output().is_some());
        let body: serde_json::Value =
            serde_json::from_str(&exports.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
        assert_eq!(spans(&body).len(), 1);
        collector.join().unwrap();
    }

    #[test]
    fn explicit_parent_overrides_inherited_suppression_unless_explicitly_suppressed() {
        let (config, exports, collector) = collector();
        let session = ObservabilitySession::new(ObservabilityConfig::otlp(config)).unwrap();
        let options = crate::RunOptions::default().with_observation(ObservationOptions {
            session: Some(session.clone()),
            trace: TraceMode::Suppress,
            ..Default::default()
        });
        let workflow = WorkflowScope::start("suppressed", &options);
        let parent = ObservationContext::from_traceparent(
            "00-11111111111111111111111111111111-2222222222222222-01",
            None,
        )
        .unwrap();
        for trace in [TraceMode::Default, TraceMode::Suppress] {
            let child = crate::RunOptions::default().with_observation(ObservationOptions {
                parent: Some(parent.clone()),
                session: Some(session.clone()),
                trace,
                ..Default::default()
            });
            let run = test_run(&child);
            if trace == TraceMode::Default {
                assert_eq!(run.context().unwrap().trace_id(), parent.trace_id());
            } else {
                assert!(run.context().is_none());
            }
            run.succeed_without_output();
        }
        workflow.end(&Ok::<(), SdkError>(()));
        session.shutdown(Duration::from_secs(2)).unwrap();
        let body: serde_json::Value = serde_json::from_str(&exports.recv().unwrap()).unwrap();
        assert_eq!(spans(&body).len(), 1);
        collector.join().unwrap();
    }

    #[test]
    fn provider_policy_survives_a_captured_worker_hop() {
        let original = ObservabilitySession::new(
            ObservabilityConfig::with_observer(Observer::disabled())
                .propagate_context_to("https://API.example:443/")
                .unwrap(),
        )
        .unwrap();
        let _scope =
            RecorderScope::install(Arc::new(StageRecorder::for_session(original, None, true)));
        assert!(xybrid_core::tracing::current_recorder()
            .unwrap()
            .allows_provider_context("https://api.example"));
        let captured = CapturedContext::capture();
        std::thread::spawn(move || {
            let _scope = captured.enter();
            let recorder = xybrid_core::tracing::current_recorder().unwrap();
            assert!(recorder.allows_provider_context("https://api.example"));
            assert!(!recorder.allows_provider_context("https://other.example"));
            assert!(xybrid_core::tracing::current_trace_headers().is_none());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn manual_handles_retain_retired_sessions_and_parent_worker_children() {
        let (config, exports, collector) = collector();
        let session = ObservabilitySession::new(ObservabilityConfig::otlp(config)).unwrap();
        let root = start_workflow(
            "manual-root",
            &ObservationOptions::default().with_session(&session),
        );
        let reference = root.observation_ref().unwrap();
        session.retire();
        drop(session);
        let captured = {
            let _scope = root.enter();
            CapturedContext::capture()
        };
        std::thread::spawn(move || {
            let _scope = captured.enter();
            let tool = start_tool(
                ToolCallInfo {
                    name: "worker-tool".into(),
                    ..Default::default()
                },
                &ObservationOptions::default(),
            );
            assert_eq!(tool.context().unwrap().trace_id(), reference.trace_id);
            tool.record_result(&serde_json::json!({"count": 0}));
            tool.end();
            tool.end_with_error(&ErrorInfo::new("late_error"));
            tool.record_result(&serde_json::json!({"count": 99}));
            assert!(tool.is_finished());
        })
        .join()
        .unwrap();
        root.end();
        root.cancel();
        drop(root);
        let body: serde_json::Value =
            serde_json::from_str(&exports.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
        let all = spans(&body);
        assert_eq!(all.len(), 2);
        let root = all
            .iter()
            .find(|s| s["name"] == "invoke_workflow manual-root")
            .unwrap();
        let tool = all
            .iter()
            .find(|s| s["name"] == "execute_tool worker-tool")
            .unwrap();
        assert_eq!(tool["parentSpanId"], root["spanId"]);
        assert!(attribute(tool, semconv::attr::ERROR_TYPE).is_none());
        assert_eq!(
            attribute(tool, "gen_ai.tool.call.result").unwrap()["stringValue"],
            "{\"count\":0}"
        );
        collector.join().unwrap();
    }

    #[test]
    fn manual_drop_cancels_once_and_explicit_parent_overrides_suppression() {
        let (config, exports, collector) = collector();
        let session = ObservabilitySession::new(ObservabilityConfig::otlp(config)).unwrap();
        let options = ObservationOptions::default().with_session(&session);
        let suppressed = start_workflow(
            "suppressed",
            &ObservationOptions {
                trace: TraceMode::Suppress,
                ..options.clone()
            },
        );
        let parent = ObservationContext::from_traceparent(
            "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
            None,
        )
        .unwrap();
        let reference;
        {
            let _scope = suppressed.enter();
            let child = start_inference(
                InferenceRequest::default(),
                &options.clone().with_parent(parent.clone()),
            );
            reference = child.observation_ref().unwrap();
            assert_eq!(reference.trace_id, parent.trace_id());
            let inherited = start_model_load(
                ModelLoadInfo {
                    model: "hidden".into(),
                    ..Default::default()
                },
                &options,
            );
            assert!(inherited.observation_ref().is_none());
            assert!(inherited.context().is_none());
            inherited.end();
            // Child drops unfinished here; it must emit exactly one cancellation.
        }
        suppressed.end();
        session.shutdown(Duration::from_secs(2)).unwrap();
        let body: serde_json::Value = serde_json::from_str(&exports.recv().unwrap()).unwrap();
        let all = spans(&body);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0]["spanId"], reference.span_id);
        assert_eq!(
            attribute(all[0], semconv::attr::ERROR_TYPE).unwrap()["stringValue"],
            "cancelled"
        );
        collector.join().unwrap();
    }

    #[test]
    fn manual_unsampled_operations_keep_references_and_children() {
        let (config, exports, collector) = collector();
        let session = ObservabilitySession::new(ObservabilityConfig::otlp(config)).unwrap();
        let options = ObservationOptions::default().with_session(&session);
        let parent = ObservationContext::from_traceparent(
            "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-00",
            None,
        )
        .unwrap();
        let workflow = start_workflow("unsampled", &options.clone().with_parent(parent.clone()));
        assert!(workflow.observation_ref().is_some());
        assert!(!workflow.context().unwrap().is_sampled());
        {
            let _scope = workflow.enter();
            let inference = start_inference(InferenceRequest::default(), &options);
            assert!(inference.observation_ref().is_some());
            assert_eq!(inference.context().unwrap().trace_id(), parent.trace_id());
            assert!(!inference.context().unwrap().is_sampled());
            inference.end();
        }
        workflow.end();
        assert_eq!(session.stats().unwrap().traces.accepted, 0);
        let anchor = start_workflow("sampled", &options);
        anchor.end();
        session.shutdown(Duration::from_secs(2)).unwrap();
        let body: serde_json::Value = serde_json::from_str(&exports.recv().unwrap()).unwrap();
        assert_eq!(spans(&body).len(), 1);
        collector.join().unwrap();
    }

    #[test]
    fn unavailable_thermal_state_is_not_reported_as_nominal() {
        let summary = ResourceUsageSummary {
            cpu_avg_pct: None,
            cpu_peak_pct: None,
            process_rss_peak_mb: None,
            available_mem_min_mb: None,
            memory_pressure_peak: xybrid_core::device::MemoryPressure::Unknown,
            thermal_state_peak: ThermalState::Normal,
            thermal_measured: false,
            battery_pct_end: None,
            sample_count: 0,
            sampling_mode: "off".into(),
            sampling_interval_ms: None,
        };
        assert_eq!(resource_sample(&summary).thermal_state, None);
        let measured = ResourceUsageSummary {
            thermal_measured: true,
            ..summary
        };
        assert_eq!(
            resource_sample(&measured).thermal_state.as_deref(),
            Some("nominal")
        );
    }

    #[test]
    fn finish_reasons_map_to_the_schema_enum() {
        assert_eq!(finish_reason("eos"), "stop");
        assert_eq!(finish_reason("max_tokens"), "length");
        assert_eq!(finish_reason("tool_use"), "tool_call");
    }

    #[test]
    fn runtime_labels_map_to_contract_names() {
        assert_eq!(runtime_name("ort"), "onnxruntime");
        assert_eq!(runtime_name("llamacpp"), "llama.cpp");
        assert_eq!(runtime_name("whispercpp"), "whisper.cpp");
    }

    #[test]
    fn messages_split_system_and_describe_media_without_bytes() {
        let history = vec![Envelope::new(EnvelopeKind::Text("be brief".into()))
            .with_role(xybrid_core::ir::MessageRole::System)];
        let input = Envelope::new(EnvelopeKind::Audio(vec![0; 64]));
        let (system, messages) = input_messages(&history, Some(&input));
        assert_eq!(system, vec![Part::text("be brief")]);
        assert_eq!(messages.len(), 1);
        match &messages[0].parts[0] {
            Part::Media(m) => assert_eq!(m.size_bytes, Some(64)),
            other => panic!("unexpected part {other:?}"),
        }
    }

    #[test]
    fn sessions_remain_pinned_across_default_replacement_and_worker_hops() {
        let _default = test_support::default_session_lock();
        let (config, exports, collector) = collector();
        let original =
            ObservabilitySession::new(ObservabilityConfig::otlp(config).device_attributes(false))
                .unwrap();
        let options = crate::RunOptions::default()
            .with_observation(ObservationOptions::default().with_session(&original));
        let workflow = WorkflowScope::start("pinned", &options);
        let workflow_ref = workflow.op.observation_ref().unwrap();
        let captured = CapturedContext::capture();
        let replacement = ObservabilitySession::borrowed(Observer::disabled());
        let previous = install_default_session(replacement.clone());
        std::thread::spawn(move || {
            let _scope = captured.enter();
            let load = LoadScope::start("worker-model", None, None);
            let reference = load.end(&Ok::<(), SdkError>(())).unwrap();
            assert_eq!(reference.trace_id, workflow_ref.trace_id);
        })
        .join()
        .unwrap();
        workflow.end(&Ok::<(), SdkError>(()));
        original.shutdown(Duration::from_secs(2)).unwrap();
        assert_eq!(original.stats().unwrap().traces.exported, 2);
        let body: serde_json::Value = serde_json::from_str(&exports.recv().unwrap()).unwrap();
        assert_eq!(spans(&body).len(), 2);
        collector.join().unwrap();
        let installed = session::take_default().unwrap();
        assert!(!installed.session.observer().is_enabled());
        if let Some(previous) = previous {
            install_default_session(previous);
        }
    }

    #[test]
    fn suppression_and_borrowed_lifecycle_survive_hops() {
        let borrowed = ObservabilitySession::borrowed(Observer::disabled());
        assert!(!borrowed.owns_pipeline());
        assert_eq!(borrowed.stats(), None);
        borrowed.flush(Duration::ZERO).unwrap();
        borrowed.shutdown(Duration::ZERO).unwrap();
        let options = crate::RunOptions::default().with_observation(ObservationOptions {
            session: Some(borrowed),
            trace: TraceMode::Suppress,
            ..Default::default()
        });
        let workflow = WorkflowScope::start("suppressed", &options);
        let captured = CapturedContext::capture();
        std::thread::spawn(move || {
            let _scope = captured.enter();
            assert_eq!(inherited_trace(TraceMode::Default), TraceMode::Suppress);
            assert!(ambient_parent().is_none());
        })
        .join()
        .unwrap();
        workflow.end(&Ok::<(), SdkError>(()));
    }

    #[test]
    fn live_sampler_traces_one_frame_per_interval() {
        let session = ObservabilitySession::borrowed(Observer::disabled());
        assert_eq!(session.live_trace_mode("stream"), TraceMode::Default);
        assert_eq!(session.live_trace_mode("stream"), TraceMode::Suppress);
        let other = ObservabilitySession::borrowed(Observer::disabled());
        assert_eq!(other.live_trace_mode("stream"), TraceMode::Default);
    }
}
