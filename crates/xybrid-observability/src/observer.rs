//! The [`Observer`]: entry point that turns AI operations into OpenTelemetry
//! spans, metrics, and log events.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;

use opentelemetry::global::{BoxedTracer, ObjectSafeTracerProvider};
use opentelemetry::logs::{AnyValue, LogRecord as _, Logger, LoggerProvider, Severity};
use opentelemetry::metrics::{Counter, Histogram, Meter, MeterProvider};
use opentelemetry::trace::{
    SpanId, SpanKind, TraceContextExt, TraceFlags, TraceId, Tracer, TracerProvider,
};
use opentelemetry::{Context, InstrumentationScope, Key, KeyValue};
use serde_json::Value;

use crate::clock::Clock;
use crate::content::ContentPolicy;
use crate::context::ObservationRef;
use crate::operation::{
    InferenceOperation, ModelLoadOperation, SpanOperation, ToolOperation, WorkflowOperation,
};
use crate::semconv::{attr, event, metric, DURATION_BUCKETS, SCOPE_NAME, TOKEN_BUCKETS};
use crate::types::{
    Evaluation, Feedback, InferenceRequest, ModelLoadInfo, StartOptions, ToolCallInfo, TraceMode,
};

/// Records AI operations as OpenTelemetry signals.
///
/// Cheap to clone and safe to share across threads. Built from
/// application-owned providers; it never installs global providers. A
/// [`Observer::disabled`] observer records nothing and costs almost nothing.
#[derive(Clone)]
pub struct Observer {
    pub(crate) inner: Arc<ObserverInner>,
}

pub(crate) struct ObserverInner {
    pub tracer: Option<BoxedTracer>,
    pub instruments: Option<Instruments>,
    pub events: Option<Box<dyn EventSink>>,
    pub content: ContentPolicy,
    pub clock: Clock,
}

impl Observer {
    /// Starts building an observer.
    pub fn builder() -> ObserverBuilder {
        ObserverBuilder::default()
    }

    /// An observer that records nothing.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            inner: Arc::new(ObserverInner {
                tracer: None,
                instruments: None,
                events: None,
                content: ContentPolicy::metadata_only(),
                clock: Clock::system(),
            }),
        }
    }

    /// Whether any signal is wired.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.inner.tracer.is_some()
            || self.inner.instruments.is_some()
            || self.inner.events.is_some()
    }

    /// Whether a logger is wired (required for evaluation and feedback events).
    #[must_use]
    pub fn has_events(&self) -> bool {
        self.inner.events.is_some()
    }

    /// The content policy.
    #[must_use]
    pub fn content_policy(&self) -> &ContentPolicy {
        &self.inner.content
    }

    /// Current time from the observer clock.
    #[must_use]
    pub fn now(&self) -> SystemTime {
        self.inner.clock.now()
    }

    /// Starts a logical inference operation (`chat`, `generate_content`,
    /// `embeddings`, `predict`, …).
    #[must_use]
    pub fn start_inference(
        &self,
        request: InferenceRequest,
        options: StartOptions,
    ) -> InferenceOperation {
        InferenceOperation::start(self.inner.clone(), request, options)
    }

    /// Starts a workflow (`invoke_workflow {name}`).
    #[must_use]
    pub fn start_workflow(
        &self,
        name: impl Into<String>,
        options: StartOptions,
    ) -> WorkflowOperation {
        WorkflowOperation::start(self.inner.clone(), name.into(), options)
    }

    /// Starts a tool execution (`execute_tool {name}`).
    #[must_use]
    pub fn start_tool(&self, call: ToolCallInfo, options: StartOptions) -> ToolOperation {
        ToolOperation::start(self.inner.clone(), call, options)
    }

    /// Starts loading a model (`load_model {model}`).
    #[must_use]
    pub fn start_model_load(
        &self,
        info: ModelLoadInfo,
        options: StartOptions,
    ) -> ModelLoadOperation {
        ModelLoadOperation::start(self.inner.clone(), info, options)
    }

    /// Starts a generic internal span (runtime stages, routing decisions).
    /// It carries no GenAI attributes and records no metrics.
    #[must_use]
    pub fn start_span(
        &self,
        name: impl Into<String>,
        attributes: Vec<KeyValue>,
        options: StartOptions,
    ) -> SpanOperation {
        SpanOperation::start(self.inner.clone(), name.into(), attributes, options)
    }

    /// Emits a `gen_ai.evaluation.result` event linked to `target`.
    ///
    /// Returns `false` when no logger is configured or `target` is invalid.
    pub fn record_evaluation(&self, target: &ObservationRef, evaluation: &Evaluation) -> bool {
        let Some(events) = &self.inner.events else {
            return false;
        };
        let Some((trace_id, span_id)) = target.ids() else {
            return false;
        };
        let mut attributes = vec![(
            Key::from_static_str(attr::EVALUATION_NAME),
            AnyValue::from(evaluation.name.clone()),
        )];
        push_any(
            &mut attributes,
            attr::EVALUATION_SCORE_VALUE,
            evaluation.score_value.map(AnyValue::from),
        );
        push_any(
            &mut attributes,
            attr::EVALUATION_SCORE_LABEL,
            evaluation.score_label.clone().map(AnyValue::from),
        );
        push_any(
            &mut attributes,
            attr::EVALUATION_EXPLANATION,
            evaluation.explanation.clone().map(AnyValue::from),
        );
        push_any(
            &mut attributes,
            attr::RESPONSE_ID,
            evaluation.response_id.clone().map(AnyValue::from),
        );
        events.emit(LogEvent {
            name: event::EVALUATION_RESULT,
            severity: Severity::Info,
            severity_text: "INFO",
            timestamp: self.now(),
            trace: Some((trace_id, span_id, None)),
            attributes,
        });
        true
    }

    /// Emits end-user feedback as a `gen_ai.evaluation.result` event named
    /// `user_feedback`. The comment is content and follows the capture policy.
    ///
    /// Returns `false` when no logger is configured or `target` is invalid.
    pub fn record_feedback(&self, target: &ObservationRef, feedback: &Feedback) -> bool {
        let Some(events) = &self.inner.events else {
            return false;
        };
        let Some((trace_id, span_id)) = target.ids() else {
            return false;
        };
        let mut attributes = vec![(
            Key::from_static_str(attr::EVALUATION_NAME),
            AnyValue::from(event::USER_FEEDBACK_EVALUATION),
        )];
        push_any(
            &mut attributes,
            attr::EVALUATION_SCORE_VALUE,
            feedback.score.map(AnyValue::from),
        );
        push_any(
            &mut attributes,
            attr::EVALUATION_SCORE_LABEL,
            feedback.label.clone().map(AnyValue::from),
        );
        if let Some(comment) = &feedback.comment {
            if self.inner.content.capture().any() {
                let prepared = self
                    .inner
                    .content
                    .prepare_text(attr::FEEDBACK_COMMENT, comment.clone());
                push_any(
                    &mut attributes,
                    attr::FEEDBACK_COMMENT,
                    prepared.value.map(AnyValue::from),
                );
            }
        }
        events.emit(LogEvent {
            name: event::EVALUATION_RESULT,
            severity: Severity::Info,
            severity_text: "INFO",
            timestamp: self.now(),
            trace: Some((trace_id, span_id, None)),
            attributes,
        });
        true
    }
}

impl fmt::Debug for Observer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Observer")
            .field("traces", &self.inner.tracer.is_some())
            .field("metrics", &self.inner.instruments.is_some())
            .field("events", &self.inner.events.is_some())
            .field("content", &self.inner.content)
            .finish()
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Builds an [`Observer`] from application-owned providers.
#[derive(Default)]
#[must_use]
pub struct ObserverBuilder {
    tracer: Option<BoxedTracer>,
    meter: Option<Meter>,
    events: Option<Box<dyn EventSink>>,
    content: Option<ContentPolicy>,
    clock: Option<Clock>,
}

impl ObserverBuilder {
    /// Records spans with `provider`.
    pub fn tracer_provider<P>(mut self, provider: &P) -> Self
    where
        P: TracerProvider,
        P::Tracer: Send + Sync + 'static,
        <P::Tracer as Tracer>::Span: Send + Sync + 'static,
    {
        self.tracer = Some(BoxedTracer::new(provider.boxed_tracer(scope())));
        self
    }

    /// Records spans with the global tracer provider (read, never installed).
    pub fn global_tracer_provider(mut self) -> Self {
        self.tracer = Some(opentelemetry::global::tracer_provider().tracer_with_scope(scope()));
        self
    }

    /// Records metrics with `provider`.
    pub fn meter_provider(mut self, provider: &impl MeterProvider) -> Self {
        self.meter = Some(provider.meter_with_scope(scope()));
        self
    }

    /// Records metrics with the global meter provider (read, never installed).
    pub fn global_meter_provider(mut self) -> Self {
        self.meter = Some(opentelemetry::global::meter_provider().meter_with_scope(scope()));
        self
    }

    /// Emits log events (content details, exceptions, evaluations) with
    /// `provider`.
    pub fn logger_provider<P>(mut self, provider: &P) -> Self
    where
        P: LoggerProvider,
        P::Logger: Send + Sync + 'static,
    {
        self.events = Some(Box::new(LoggerSink(provider.logger_with_scope(scope()))));
        self
    }

    /// Sets the content policy. Without this call the policy is the product
    /// default with `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` honored.
    pub fn content_policy(mut self, policy: ContentPolicy) -> Self {
        self.content = Some(policy);
        self
    }

    /// Sets the clock (tests use a manual clock).
    pub fn clock(mut self, clock: Clock) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Builds the observer.
    #[must_use]
    pub fn build(self) -> Observer {
        Observer {
            inner: Arc::new(ObserverInner {
                tracer: self.tracer,
                instruments: self.meter.as_ref().map(Instruments::new),
                events: self.events,
                content: self.content.unwrap_or_else(ContentPolicy::from_env),
                clock: self.clock.unwrap_or_default(),
            }),
        }
    }
}

impl fmt::Debug for ObserverBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObserverBuilder").finish_non_exhaustive()
    }
}

fn scope() -> InstrumentationScope {
    InstrumentationScope::builder(SCOPE_NAME)
        .with_version(env!("CARGO_PKG_VERSION"))
        .build()
}

/// Metric instruments, created once per observer.
pub(crate) struct Instruments {
    pub inference_duration: Histogram<f64>,
    pub embeddings_duration: Histogram<f64>,
    pub operation_duration: Histogram<f64>,
    pub time_to_first_chunk: Histogram<f64>,
    pub time_per_output_chunk: Histogram<f64>,
    pub time_to_first_token: Histogram<f64>,
    pub usage_input: Counter<u64>,
    pub usage_output: Counter<u64>,
    pub usage_cache_read: Counter<u64>,
    pub usage_cache_write: Counter<u64>,
    pub usage_reasoning: Counter<u64>,
    pub operation_input_tokens: Histogram<u64>,
    pub operation_output_tokens: Histogram<u64>,
    pub embeddings_input_tokens: Histogram<u64>,
    pub workflow_duration: Histogram<f64>,
    pub tool_duration: Histogram<f64>,
    pub attempts: Counter<u64>,
}

impl Instruments {
    fn new(meter: &Meter) -> Self {
        let seconds = |name: &'static str, description: &'static str| {
            meter
                .f64_histogram(name)
                .with_unit("s")
                .with_description(description)
                .with_boundaries(DURATION_BUCKETS.to_vec())
                .build()
        };
        let tokens_hist = |name: &'static str, description: &'static str| {
            meter
                .u64_histogram(name)
                .with_unit("{token}")
                .with_description(description)
                .with_boundaries(TOKEN_BUCKETS.to_vec())
                .build()
        };
        let tokens = |name: &'static str, description: &'static str| {
            meter
                .u64_counter(name)
                .with_unit("{token}")
                .with_description(description)
                .build()
        };
        Self {
            inference_duration: seconds(metric::INFERENCE_DURATION, "GenAI client inference operation duration."),
            embeddings_duration: seconds(metric::EMBEDDINGS_DURATION, "GenAI client embeddings operation duration."),
            operation_duration: seconds(metric::OPERATION_DURATION, "GenAI operation duration."),
            time_to_first_chunk: seconds(
                metric::TIME_TO_FIRST_CHUNK,
                "Time from the start of a logical inference operation to the first chunk in its streaming response.",
            ),
            time_per_output_chunk: seconds(
                metric::TIME_PER_OUTPUT_CHUNK,
                "Time between consecutive output chunks after the first one.",
            ),
            time_to_first_token: seconds(
                metric::TIME_TO_FIRST_TOKEN,
                "Time from the start of a logical inference operation to the first output token.",
            ),
            usage_input: tokens(metric::USAGE_INPUT_TOKENS, "Input tokens used, including cached tokens."),
            usage_output: tokens(metric::USAGE_OUTPUT_TOKENS, "Output tokens used, including reasoning tokens."),
            usage_cache_read: tokens(metric::USAGE_CACHE_READ_INPUT_TOKENS, "Input tokens served from cache."),
            usage_cache_write: tokens(metric::USAGE_CACHE_WRITE_INPUT_TOKENS, "Input tokens written to cache."),
            usage_reasoning: tokens(metric::USAGE_REASONING_OUTPUT_TOKENS, "Output tokens used for reasoning."),
            operation_input_tokens: tokens_hist(
                metric::OPERATION_INPUT_TOKENS,
                "Input tokens used per inference operation.",
            ),
            operation_output_tokens: tokens_hist(
                metric::OPERATION_OUTPUT_TOKENS,
                "Output tokens used per inference operation.",
            ),
            embeddings_input_tokens: tokens_hist(
                metric::EMBEDDINGS_INPUT_TOKENS,
                "Input tokens used per embeddings operation.",
            ),
            workflow_duration: seconds(metric::WORKFLOW_DURATION, "Duration of GenAI workflows."),
            tool_duration: seconds(metric::TOOL_DURATION, "Duration of tool executions."),
            attempts: meter
                .u64_counter(metric::ATTEMPTS)
                .with_unit("{attempt}")
                .with_description("Execution attempts of logical inference requests.")
                .build(),
        }
    }
}

/// A log event ready to emit.
pub(crate) struct LogEvent {
    pub name: &'static str,
    pub severity: Severity,
    pub severity_text: &'static str,
    pub timestamp: SystemTime,
    pub trace: Option<(TraceId, SpanId, Option<TraceFlags>)>,
    pub attributes: Vec<(Key, AnyValue)>,
}

/// Object-safe logger wrapper.
pub(crate) trait EventSink: Send + Sync {
    fn emit(&self, event: LogEvent);
}

struct LoggerSink<L>(L);

impl<L: Logger + Send + Sync> EventSink for LoggerSink<L> {
    fn emit(&self, event: LogEvent) {
        let mut record = self.0.create_log_record();
        record.set_event_name(event.name);
        record.set_severity_number(event.severity);
        record.set_severity_text(event.severity_text);
        record.set_timestamp(event.timestamp);
        record.set_observed_timestamp(event.timestamp);
        if let Some((trace_id, span_id, flags)) = event.trace {
            record.set_trace_context(trace_id, span_id, flags);
        }
        record.add_attributes(event.attributes);
        self.0.emit(record);
    }
}

pub(crate) fn push_any(out: &mut Vec<(Key, AnyValue)>, key: &'static str, value: Option<AnyValue>) {
    if let Some(value) = value {
        out.push((Key::from_static_str(key), value));
    }
}

/// Structured JSON → `AnyValue` for log events.
pub(crate) fn json_to_any(value: Value) -> AnyValue {
    match value {
        Value::Null => AnyValue::String("".into()),
        Value::Bool(b) => AnyValue::Boolean(b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => AnyValue::Int(i),
            None => AnyValue::Double(n.as_f64().unwrap_or_default()),
        },
        Value::String(s) => AnyValue::String(s.into()),
        Value::Array(items) => {
            AnyValue::ListAny(Box::new(items.into_iter().map(json_to_any).collect()))
        }
        Value::Object(map) => AnyValue::Map(Box::new(
            map.into_iter()
                .map(|(k, v)| (Key::new(k), json_to_any(v)))
                .collect(),
        )),
    }
}

impl ObserverInner {
    /// Starts a span unless suppressed or no tracer is configured.
    pub(crate) fn start_span(
        &self,
        name: String,
        kind: SpanKind,
        mut attributes: Vec<KeyValue>,
        options: &mut StartOptions,
        start: SystemTime,
    ) -> Option<Context> {
        if options.trace == TraceMode::Suppress {
            return None;
        }
        let tracer = self.tracer.as_ref()?;
        let parent = match &options.parent {
            Some(parent) => parent.to_otel(),
            None => Context::current(),
        };
        attributes.append(&mut options.attributes);
        let span = tracer
            .span_builder(Cow::Owned(name))
            .with_kind(kind)
            .with_start_time(start)
            .with_attributes(attributes)
            .start_with_context(tracer, &parent);
        Some(parent.with_span(span))
    }
}
