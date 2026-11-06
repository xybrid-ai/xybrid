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
    BatchLimits, ExportStats, OtlpConfig, Protocol, ResourceConfig, RetryLimits, SignalEndpoint,
    XYBRID_TRACES_ENDPOINT,
};
pub use xybrid_observability::{
    semconv, AttemptInfo, AttemptOperation, AttemptReason, ContentCapture, ContentPolicy,
    ErrorInfo, Evaluation, Feedback, InferenceOperation, InferenceRequest, InputMessage, KeyValue,
    Location, MediaMetadata, Modality, ModelLoadInfo, ModelLoadOperation, ObservationContext,
    ObservationRef, Observer, ObserverBuilder, OutputMessage, Part, PromptRef, Redactor,
    ResourceSample, SpanOperation, StartOptions, SubstringRedactor, ToolCallInfo, ToolOperation,
    TraceMode, Usage, WorkflowOperation,
};

use crate::model::{SdkError, SdkResult};

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

struct Installed {
    observer: Observer,
    pipeline: Option<Arc<xybrid_observability::otlp::ObservabilityPipeline>>,
    live_trace_interval: Duration,
}

static INSTALLED: RwLock<Option<Installed>> = RwLock::new(None);

fn installed() -> std::sync::RwLockReadGuard<'static, Option<Installed>> {
    INSTALLED
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Enables observability for this process. Replaces a previous configuration
/// (its packaged pipeline is shut down within one second).
///
/// # Errors
///
/// [`SdkError::ConfigError`] when the exporter configuration is rejected.
pub fn init_observability(config: ObservabilityConfig) -> SdkResult<()> {
    crate::telemetry::activate_resource_telemetry(config.resource_telemetry);
    let (observer, pipeline) = match config.exporter {
        Exporter::Observer(observer) => (observer, None),
        Exporter::Otlp(mut otlp) => {
            otlp.resource
                .attributes
                .extend(sdk_resource_attributes(config.device_attributes));
            if otlp.resource.service_name.is_none() {
                otlp.resource.service_name = Some("xybrid-app".to_owned());
            }
            let pipeline = xybrid_observability::otlp::ObservabilityPipeline::build(*otlp)
                .map_err(|e| SdkError::ConfigError(format!("observability: {e}")))?;
            let content = config.content.unwrap_or_else(ContentPolicy::from_env);
            (pipeline.observer_with(content), Some(Arc::new(pipeline)))
        }
    };
    let previous = {
        let mut slot = INSTALLED
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slot.replace(Installed {
            observer,
            pipeline,
            live_trace_interval: config.live_trace_interval,
        })
    };
    if let Some(previous) = previous.and_then(|p| p.pipeline) {
        let _ = previous.shutdown(Duration::from_secs(1));
    }
    Ok(())
}

/// The observer runs report to. Disabled until [`init_observability`].
pub fn observer() -> Observer {
    installed()
        .as_ref()
        .map_or_else(Observer::disabled, |i| i.observer.clone())
}

/// Whether observability has been configured.
pub fn is_enabled() -> bool {
    installed()
        .as_ref()
        .is_some_and(|i| i.observer.is_enabled())
}

/// Delivery counters of the packaged pipeline, if one is running.
pub fn stats() -> Option<ExportStats> {
    installed()
        .as_ref()
        .and_then(|i| i.pipeline.as_ref().map(|p| p.stats()))
}

/// Exports queued telemetry, waiting at most `timeout`. No-op without a
/// packaged pipeline.
///
/// # Errors
///
/// [`SdkError::Timeout`] when the bound is exceeded.
pub fn flush_observability(timeout: Duration) -> SdkResult<()> {
    let pipeline = installed().as_ref().and_then(|i| i.pipeline.clone());
    match pipeline {
        Some(p) => p
            .force_flush(timeout)
            .map_err(|e| lifecycle_error(e, timeout)),
        None => Ok(()),
    }
}

/// Flushes and stops the packaged pipeline, waiting at most `timeout`, and
/// disables observability. Idempotent.
///
/// # Errors
///
/// [`SdkError::Timeout`] when the bound is exceeded.
pub fn shutdown_observability(timeout: Duration) -> SdkResult<()> {
    let previous = INSTALLED
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    match previous.and_then(|p| p.pipeline) {
        Some(p) => p.shutdown(timeout).map_err(|e| lifecycle_error(e, timeout)),
        None => Ok(()),
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
        SdkError::network(e.to_string())
    }
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
    /// Observer override for this run.
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
            input_modalities: inputs,
            output_modalities: outputs,
            output_type,
            is_llm,
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
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Text],
            output_type: None,
            is_llm: true,
        }
    }
}

/// Live-session trace sampler: one traced frame per interval per session.
static LIVE_TRACED: std::sync::LazyLock<Mutex<HashMap<String, Instant>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

const LIVE_SESSION_TTL: Duration = Duration::from_secs(300);

fn live_trace_mode(session: &str) -> TraceMode {
    let interval = installed()
        .as_ref()
        .map_or(Duration::from_secs(1), |i| i.live_trace_interval);
    let now = Instant::now();
    let mut map = LIVE_TRACED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    map.retain(|id, last| id == session || now.duration_since(*last) <= LIVE_SESSION_TTL);
    match map.get(session) {
        Some(last) if now.duration_since(*last) < interval => TraceMode::Suppress,
        _ => {
            map.insert(session.to_owned(), now);
            TraceMode::Default
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
}

/// Observation of one SDK run: the logical inference span plus a stage
/// recorder installed on the executing thread.
pub(crate) struct RunObservation {
    op: Arc<InferenceOperation>,
    stages: Arc<StageRecorder>,
    _scope: Option<RecorderScope>,
    chunks: Arc<Mutex<ChunkState>>,
    started: SystemTime,
    subject: RunSubject,
    attempts: std::sync::atomic::AtomicU32,
}

#[derive(Default)]
struct ChunkState {
    first_token_marked: bool,
}

/// Records streamed chunks from token callbacks (which must be `Send`).
#[derive(Clone)]
pub(crate) struct ChunkSink {
    op: Arc<InferenceOperation>,
    chunks: Arc<Mutex<ChunkState>>,
    stages: Arc<StageRecorder>,
    is_llm: bool,
}

impl ChunkSink {
    /// One streamed output chunk (token callback, audio packet). Performs no
    /// I/O.
    pub(crate) fn chunk(&self) {
        self.op.record_chunk();
        if !self.is_llm {
            return;
        }
        let mut state = self
            .chunks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.first_token_marked {
            state.first_token_marked = true;
            drop(state);
            if self.stages.harvest().ttft_ms.is_none() {
                self.op.record_first_token();
            }
        }
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
/// attempt's trace context; stages become its children.
pub(crate) struct AttemptScope {
    op: AttemptOperation,
    stages: Arc<StageRecorder>,
    parent_stages: Arc<StageRecorder>,
    _scope: Option<RecorderScope>,
}

impl AttemptScope {
    /// Ends the attempt with the outcome of its leg.
    pub(crate) fn end<T>(mut self, result: &SdkResult<T>) {
        self._scope.take();
        let h = self.stages.harvest();
        let usage = Usage {
            input_tokens: h.tokens_in,
            output_tokens: h.tokens_out,
            cache_read_input_tokens: h.cached_tokens.filter(|n| *n > 0),
            ..Usage::default()
        };
        if !usage.is_empty() {
            self.op.record_usage(&usage);
        }
        self.parent_stages.merge(&h);
        match result {
            Ok(_) => self.op.end(),
            Err(SdkError::Cancelled { .. }) => self.op.cancel(),
            Err(e) => self.op.end_with_error(&error_info(e)),
        }
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
    pub(crate) fn end_adapter<T, E: fmt::Display>(self, result: &Result<T, E>) {
        let mapped: SdkResult<()> = match result {
            Ok(_) => Ok(()),
            Err(e) => Err(SdkError::inference(e.to_string())),
        };
        self.end(&mapped);
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
        let observer = obs_opts
            .and_then(|o| o.observer.clone())
            .unwrap_or_else(observer);
        let mut start =
            StartOptions::default().with_trace(obs_opts.map(|o| o.trace).unwrap_or_default());
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
        let op = observer.start_workflow(name, start);
        let stages = Arc::new(StageRecorder::new(observer, op.context()));
        let scope = RecorderScope::install(stages);
        Self {
            op,
            _scope: Some(scope),
        }
    }

    /// Carries the run's controls into an inference nested under this workflow.
    /// An explicit caller parent belongs to the workflow; its inference is a child of it.
    #[cfg(any(test, feature = "llm-mistral", feature = "llm-llamacpp"))]
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
        let observer = observer();
        let options = StartOptions::default().with_parent(ambient_parent());
        let span = observer.start_span(name.to_owned(), attributes, options);
        let stages = Arc::new(StageRecorder::new(observer, span.context()));
        let scope = RecorderScope::install(stages);
        Self {
            span,
            _scope: Some(scope),
        }
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
        let observer = obs_opts
            .and_then(|o| o.observer.clone())
            .unwrap_or_else(observer);
        let mut trace = obs_opts.map(|o| o.trace).unwrap_or_default();
        if let Some(opts) = run.options {
            if opts.live_mode && trace == TraceMode::Default {
                if let Some(session) = opts.frame_session_id.as_deref() {
                    trace = live_trace_mode(session);
                }
            }
        }
        let subject = run.subject;
        let mut request = InferenceRequest::new(
            subject.operation,
            match run.location {
                Location::Remote => "xybrid",
                Location::Local => subject.runtime.unwrap_or("xybrid"),
            },
        );
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
        if let Some(o) = obs_opts {
            request.conversation_id = o.conversation_id.clone();
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
        let op = observer.start_inference(request, start);

        let (system, messages) = input_messages(run.history, run.input);
        if !system.is_empty() {
            op.record_system_instructions(&system);
        }
        if !messages.is_empty() {
            op.record_input_messages(&messages);
        }

        let stages = Arc::new(StageRecorder::new(observer, op.context()));
        let scope = RecorderScope::install(stages.clone());
        Self {
            op: Arc::new(op),
            stages,
            _scope: Some(scope),
            chunks: Arc::new(Mutex::new(ChunkState::default())),
            started,
            subject,
            attempts: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// A `Send` recorder for streamed chunks.
    pub(crate) fn chunk_sink(&self) -> ChunkSink {
        ChunkSink {
            op: self.op.clone(),
            chunks: self.chunks.clone(),
            stages: self.stages.clone(),
            is_llm: self.subject.is_llm,
        }
    }

    /// Starts the next execution attempt of this run.
    pub(crate) fn attempt(&self, spec: AttemptSpec<'_>) -> AttemptScope {
        let index = self
            .attempts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (server_address, server_port) = match spec.location {
            Location::Remote => gateway_address(),
            Location::Local => (None, None),
        };
        let op = self.op.start_attempt(AttemptInfo {
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
        });
        let stages = Arc::new(StageRecorder::new(
            self.stages.observer.clone(),
            op.context(),
        ));
        let scope = RecorderScope::install(stages.clone());
        AttemptScope {
            op,
            stages,
            parent_stages: self.stages.clone(),
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
    pub(crate) fn cancel(self) {
        self.apply_harvest(None);
        self.op.cancel();
    }

    fn apply_harvest(&self, output: Option<&Envelope>) {
        let mut h = self.stages.harvest();
        if let Some(meta) = output.map(|o| &o.metadata) {
            h.absorb(meta.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        }
        let usage = Usage {
            input_tokens: h.tokens_in,
            output_tokens: h.tokens_out,
            cache_read_input_tokens: h.cached_tokens.filter(|n| *n > 0),
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
        if let Some(ttft) = h.ttft_ms {
            let state = self
                .chunks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !state.first_token_marked || self.subject.is_llm {
                self.op
                    .record_first_token_at(self.started + Duration::from_secs_f64(ttft / 1000.0));
            }
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
    let url = xybrid_core::cloud::platform_gateway_url();
    let rest = url.split_once("://").map_or(url.as_str(), |(_, r)| r);
    let authority = rest.split('/').next().unwrap_or_default();
    match authority.rsplit_once(':') {
        Some((host, port)) if !host.contains(']') || host.ends_with(']') => {
            (Some(host.to_owned()), port.parse().ok())
        }
        _ if authority.is_empty() => (None, None),
        _ => (Some(authority.to_owned()), None),
    }
}

fn finish_reason(raw: &str) -> String {
    match raw {
        "stop" | "eos" | "end_turn" | "stop_sequence" => "stop",
        "length" | "max_tokens" => "length",
        "tool_calls" | "tool_call" | "tool_use" => "tool_call",
        "content_filter" => "content_filter",
        "error" => "error",
        other => other,
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
    ErrorInfo::new(error_type).with_message(crate::telemetry::redact_error_for_telemetry(
        &error.to_string(),
    ))
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

/// Builds the GenAI system instructions and input messages for a run.
fn input_messages(
    history: &[Envelope],
    input: Option<&Envelope>,
) -> (Vec<Part>, Vec<InputMessage>) {
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for envelope in history.iter().chain(input) {
        let role = role(envelope, "user");
        let p = parts(envelope);
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
            .map(|r| finish_reason(r)),
    })
}

/// Converts a measured resource summary to a portable sample.
pub(crate) fn resource_sample(summary: &ResourceUsageSummary) -> ResourceSample {
    ResourceSample {
        cpu_utilization: summary.cpu_avg_pct.map(|p| f64::from(p) / 100.0),
        memory_used_bytes: None,
        process_rss_bytes: None,
        process_peak_rss_bytes: summary
            .process_rss_peak_mb
            .map(|mb| u64::from(mb) * 1024 * 1024),
        thermal_state: Some(
            match summary.thermal_state_peak {
                ThermalState::Normal => "nominal",
                ThermalState::Warm => "fair",
                ThermalState::Hot => "serious",
                ThermalState::Critical => "critical",
            }
            .to_owned(),
        ),
        battery_level: summary.battery_pct_end.map(|p| f64::from(p) / 100.0),
        battery_charging: None,
        accelerator_utilization: None,
        accelerator_memory_used_bytes: None,
    }
}

// ---------------------------------------------------------------------------
// Stage recorder: core `SpanGuard`s become child spans of the run.
// ---------------------------------------------------------------------------

/// Values harvested from stage metadata (the keys execution code stamps).
#[derive(Clone, Debug, Default)]
pub(crate) struct Harvest {
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

impl Harvest {
    fn absorb<'a>(&mut self, pairs: impl Iterator<Item = (&'a str, &'a str)>) {
        for (key, value) in pairs {
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

/// Records execution stages as `INTERNAL` child spans of the run and keeps
/// the last value of the usage/timing keys stages stamp.
pub(crate) struct StageRecorder {
    observer: Observer,
    root: Option<ObservationContext>,
    spans: Mutex<HashMap<u64, SpanOperation>>,
    next: std::sync::atomic::AtomicU64,
    harvest: Mutex<Harvest>,
}

impl StageRecorder {
    pub(crate) fn new(observer: Observer, root: Option<ObservationContext>) -> Self {
        Self {
            observer,
            root,
            spans: Mutex::new(HashMap::new()),
            next: std::sync::atomic::AtomicU64::new(1),
            harvest: Mutex::new(Harvest::default()),
        }
    }

    pub(crate) fn harvest(&self) -> Harvest {
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
        h.tokens_in = other.tokens_in.or(h.tokens_in);
        h.tokens_out = other.tokens_out.or(h.tokens_out);
        h.cached_tokens = other.cached_tokens.or(h.cached_tokens);
        h.ttft_ms = other.ttft_ms.or(h.ttft_ms);
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
}

impl TraceRecorder for StageRecorder {
    fn start_span(&self, name: &str, parent: Option<u64>) -> u64 {
        let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let parent_cx = match parent {
            Some(p) => self.spans().get(&p).and_then(SpanOperation::context),
            None => None,
        }
        .or_else(|| self.root.clone());
        let options = match parent_cx {
            Some(cx) => StartOptions::child_of(cx),
            // A run whose logical span was suppressed keeps its stages
            // untraced too.
            None => StartOptions::default().with_trace(TraceMode::Suppress),
        };
        let span = self.observer.start_span(
            name.to_owned(),
            vec![KeyValue::new("xybrid.stage.name", name.to_owned())],
            options,
        );
        self.spans().insert(id, span);
        id
    }

    fn end_span(&self, id: u64) {
        if let Some(span) = self.spans().remove(&id) {
            span.end();
        }
    }

    fn add_metadata(&self, id: u64, key: &str, value: &str) {
        self.harvest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .absorb(std::iter::once((key, value)));
        if let Some(span) = self.spans().get(&id) {
            span.set_attribute(KeyValue::new(
                format!("xybrid.stage.{key}"),
                value.to_owned(),
            ));
        }
    }

    fn trace_headers(&self, id: Option<u64>) -> Option<TraceHeaders> {
        let cx = id
            .and_then(|id| self.spans().get(&id).and_then(SpanOperation::context))
            .or_else(|| self.root.clone())?;
        Some(TraceHeaders {
            traceparent: cx.to_traceparent(),
            tracestate: cx.tracestate().map(str::to_owned),
        })
    }
}

/// The parent for a run started inside another observed run on this thread
/// (for example a pipeline stage), taken from the installed stage recorder,
/// or the application's active OpenTelemetry span before a worker hop.
pub(crate) fn ambient_parent() -> Option<ObservationContext> {
    if xybrid_core::tracing::current_recorder().is_some() {
        // An installed recorder with no headers can represent a suppressed workflow.
        // Do not bypass it by resurrecting an outer application's span.
        return xybrid_core::tracing::current_trace_headers().and_then(|headers| {
            ObservationContext::from_traceparent(
                &headers.traceparent,
                headers.tracestate.as_deref(),
            )
            .ok()
        });
    }
    ObservationContext::current()
}

/// Starts a model-load observation.
pub(crate) fn start_model_load(
    model_id: &str,
    metadata: Option<&ModelMetadata>,
) -> ModelLoadOperation {
    let subject = metadata.map(|m| RunSubject::from_metadata(model_id, m));
    observer().start_model_load(
        ModelLoadInfo {
            model: model_id.to_owned(),
            provider: subject.as_ref().and_then(|s| s.runtime).map(str::to_owned),
            runtime: subject.as_ref().and_then(|s| s.runtime).map(str::to_owned),
            format: subject.as_ref().and_then(|s| s.format).map(str::to_owned),
            size_bytes: None,
        },
        StartOptions::default().with_parent(ambient_parent()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
                        .body_contains(format!("\"parentSpanId\": \"{}\"", parent.span_id()))
                        .body_contains("chat tiny");
                    then.status(200);
                });
                pipeline.force_flush(Duration::from_secs(2)).unwrap();
                sent.assert_hits(1);
            }
        }
        assert_eq!(
            pipeline.stats().spans_queued,
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
    fn live_sampler_traces_one_frame_per_interval() {
        let session = format!("s-{}", uuid::Uuid::new_v4());
        assert_eq!(live_trace_mode(&session), TraceMode::Default);
        assert_eq!(live_trace_mode(&session), TraceMode::Suppress);
    }
}
