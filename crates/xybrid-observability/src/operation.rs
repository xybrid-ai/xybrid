//! Operation handles. Every handle is `Send + Sync`, records through `&self`,
//! never performs I/O, and ends at most once. Dropping an un-ended handle
//! ends it as cancelled.
//!
//! Content (governed, schema and prompt variables) is processed when it is
//! recorded, retained on the operation within the retained-operation budget,
//! and written once, at end (last write wins; one attribute per key).
//! Metadata and measurements keep their timing and are written before
//! content, so span attribute limits drop content first.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::SystemTime;

use opentelemetry::logs::{AnyValue, Severity};
use opentelemetry::trace::{SpanId, SpanKind, Status, TraceContextExt, TraceFlags, TraceId};
use opentelemetry::{Array, Key, KeyValue, StringValue, Value as OtelValue};
use serde::Serialize;
use serde_json::Value;

use crate::clock::{seconds_between, OperationClock};
use crate::content::{InputMessage, Outcome, OutputMessage, Part, TextBound};
use crate::context::{ObservationContext, ObservationRef};
use crate::observer::{push_any, Instruments, LogEvent, ObserverInner, SpanState};
use crate::preflight;
use crate::semconv::{
    attr, event, op, ERROR_MESSAGE_KIND, MAX_ERROR_MESSAGE_CHARS, PREFLIGHT_NODE_BYTES,
};
use crate::types::{
    AttemptInfo, ErrorInfo, InferenceRequest, Location, ModelLoadInfo, ResourceSample,
    StartOptions, ToolCallInfo, TraceMode, Usage,
};
use crate::wire::{json_to_any, otel_to_any};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn kind_for(location: Option<Location>) -> SpanKind {
    match location {
        Some(Location::Remote) => SpanKind::Client,
        _ => SpanKind::Internal,
    }
}

/// `operation + " " + subject`, built literally (no template expansion); the
/// operation alone when the subject is absent or empty.
fn span_name(operation: &str, subject: Option<&str>) -> String {
    match subject {
        Some(s) if !s.is_empty() => {
            let mut name = String::with_capacity(operation.len() + 1 + s.len());
            name.push_str(operation);
            name.push(' ');
            name.push_str(s);
            name
        }
        _ => operation.to_owned(),
    }
}

fn string_array(values: impl IntoIterator<Item = String>) -> OtelValue {
    OtelValue::Array(Array::String(
        values.into_iter().map(StringValue::from).collect(),
    ))
}

fn u64_attr(key: &'static str, v: u64) -> KeyValue {
    KeyValue::new(key, i64::try_from(v).unwrap_or(i64::MAX))
}

fn finite(v: Option<f64>) -> Option<f64> {
    v.filter(|v| v.is_finite())
}

fn usage_attributes(usage: &Usage) -> Vec<KeyValue> {
    let pairs = [
        (attr::USAGE_INPUT_TOKENS, usage.input_tokens),
        (attr::USAGE_OUTPUT_TOKENS, usage.output_tokens),
        (
            attr::USAGE_CACHE_READ_INPUT_TOKENS,
            usage.cache_read_input_tokens,
        ),
        (
            attr::USAGE_CACHE_WRITE_INPUT_TOKENS,
            usage.cache_write_input_tokens,
        ),
        (
            attr::USAGE_REASONING_OUTPUT_TOKENS,
            usage.reasoning_output_tokens,
        ),
        (attr::USAGE_TEXT_INPUT_TOKENS, usage.text_input_tokens),
        (attr::USAGE_IMAGE_INPUT_TOKENS, usage.image_input_tokens),
        (attr::USAGE_AUDIO_INPUT_TOKENS, usage.audio_input_tokens),
        (attr::USAGE_TEXT_OUTPUT_TOKENS, usage.text_output_tokens),
        (attr::USAGE_IMAGE_OUTPUT_TOKENS, usage.image_output_tokens),
        (attr::USAGE_AUDIO_OUTPUT_TOKENS, usage.audio_output_tokens),
    ];
    pairs
        .into_iter()
        .filter_map(|(k, v)| v.map(|v| u64_attr(k, v)))
        .collect()
}

fn resource_attributes(sample: &ResourceSample) -> Vec<KeyValue> {
    let mut out = Vec::new();
    if let Some(v) = finite(sample.cpu_utilization) {
        out.push(KeyValue::new(attr::RESOURCE_CPU_UTILIZATION, v));
    }
    if let Some(v) = sample.memory_used_bytes {
        out.push(u64_attr(attr::RESOURCE_MEMORY_USED_BYTES, v));
    }
    if let Some(v) = sample.process_rss_bytes {
        out.push(u64_attr(attr::RESOURCE_PROCESS_RSS_BYTES, v));
    }
    if let Some(v) = sample.process_peak_rss_bytes {
        out.push(u64_attr(attr::RESOURCE_PROCESS_PEAK_RSS_BYTES, v));
    }
    if let Some(v) = &sample.thermal_state {
        out.push(KeyValue::new(attr::RESOURCE_THERMAL_STATE, v.clone()));
    }
    if let Some(v) = finite(sample.battery_level) {
        out.push(KeyValue::new(attr::RESOURCE_BATTERY_LEVEL, v));
    }
    if let Some(v) = sample.battery_charging {
        out.push(KeyValue::new(attr::RESOURCE_BATTERY_CHARGING, v));
    }
    if let Some(v) = finite(sample.accelerator_utilization) {
        out.push(KeyValue::new(attr::RESOURCE_ACCELERATOR_UTILIZATION, v));
    }
    if let Some(v) = sample.accelerator_memory_used_bytes {
        out.push(u64_attr(attr::RESOURCE_ACCELERATOR_MEMORY_USED_BYTES, v));
    }
    out
}

/// One retained content attribute.
struct Entry {
    /// Span value: a compact JSON string, or the text of a prompt variable.
    span: Option<String>,
    /// Structured event value.
    event: Option<Value>,
    /// Retained-operation bytes charged for this entry.
    bytes: usize,
}

/// Content retained by one operation within its budget, plus the
/// `truncated_attributes` set (first-marked order).
#[derive(Default)]
struct Retained {
    entries: BTreeMap<String, Entry>,
    bytes: usize,
    truncated: Vec<String>,
}

impl Retained {
    fn mark(&mut self, key: &str) {
        if !self.truncated.iter().any(|k| k == key) {
            self.truncated.push(key.to_owned());
        }
    }

    fn remove(&mut self, key: &str) {
        if let Some(entry) = self.entries.remove(key) {
            self.bytes -= entry.bytes;
        }
    }

    /// Applies one pipeline outcome for `key` (last write wins).
    fn apply(&mut self, key: &str, outcome: Recorded, budget: usize) {
        self.remove(key);
        match outcome {
            Recorded::Dropped => {}
            Recorded::Failed => self.mark(key),
            Recorded::Value {
                span,
                event,
                truncated,
            } => {
                let node = PREFLIGHT_NODE_BYTES + key.len();
                let bytes = span.as_ref().map_or(0, |s| node + s.len())
                    + event.as_ref().map_or(0, |v| {
                        node.saturating_add(
                            usize::try_from(preflight::cost_of(v)).unwrap_or(usize::MAX),
                        )
                    });
                if self.bytes.saturating_add(bytes) > budget {
                    self.mark(key);
                    return;
                }
                if truncated {
                    self.mark(key);
                }
                self.bytes += bytes;
                self.entries
                    .insert(key.to_owned(), Entry { span, event, bytes });
            }
        }
    }

    /// Span attributes: metadata (`truncated_attributes`) is written by the
    /// caller first; these are the content attributes.
    fn span_attributes(&self) -> Vec<KeyValue> {
        self.entries
            .iter()
            .filter_map(|(k, e)| e.span.clone().map(|s| KeyValue::new(k.clone(), s)))
            .collect()
    }

    fn take_event_values(&mut self) -> Vec<(Key, AnyValue)> {
        std::mem::take(&mut self.entries)
            .into_iter()
            .filter_map(|(k, e)| e.event.map(|v| (Key::new(k), json_to_any(v))))
            .collect()
    }

    fn has_event_values(&self) -> bool {
        self.entries.values().any(|e| e.event.is_some())
    }

    fn truncated_value(&self) -> Option<OtelValue> {
        (!self.truncated.is_empty()).then(|| string_array(self.truncated.iter().cloned()))
    }

    fn truncated_any(&self) -> Option<AnyValue> {
        (!self.truncated.is_empty()).then(|| {
            AnyValue::ListAny(Box::new(
                self.truncated.iter().cloned().map(AnyValue::from).collect(),
            ))
        })
    }
}

/// A content pipeline outcome routed to the enabled destinations.
enum Recorded {
    Value {
        span: Option<String>,
        event: Option<Value>,
        truncated: bool,
    },
    Dropped,
    Failed,
}

/// Where a content value goes, decided before any work (capture gate).
#[derive(Clone, Copy)]
struct Destinations {
    span: bool,
    event: bool,
}

impl Destinations {
    fn any(self) -> bool {
        self.span || self.event
    }
}

/// State shared by every handle type: the span (if any), the operation
/// clock and the start time.
struct Core {
    observer: Arc<ObserverInner>,
    span: Option<SpanState>,
    clock: OperationClock,
    start: SystemTime,
}

impl Core {
    fn now(&self) -> SystemTime {
        self.clock.now()
    }

    fn set_attributes(&self, attributes: Vec<KeyValue>) {
        if let Some(s) = &self.span {
            if !attributes.is_empty() {
                s.cx.span().set_attributes(attributes);
            }
        }
    }

    fn set_attribute(&self, kv: KeyValue) {
        if let Some(s) = &self.span {
            s.cx.span().set_attribute(kv);
        }
    }

    /// The context for children: the operation's span, or the parent's
    /// context passed through by an API-only tracer.
    fn context(&self) -> Option<ObservationContext> {
        self.span
            .as_ref()
            .and_then(|s| ObservationContext::from_otel(&s.cx))
    }

    /// The context of the operation's own span, if it has one.
    fn own_context(&self) -> Option<ObservationContext> {
        self.span
            .as_ref()
            .filter(|s| s.own)
            .and_then(|s| ObservationContext::from_otel(&s.cx))
    }

    fn observation_ref(&self) -> Option<ObservationRef> {
        self.own_context().map(|c| c.observation_ref())
    }

    /// Trace context for events: the operation's own span, else none (events
    /// of span-less operations never attach to the ambient span).
    fn trace_ids(&self) -> Option<(TraceId, SpanId, Option<TraceFlags>)> {
        let s = self.span.as_ref().filter(|s| s.own)?;
        let span = s.cx.span();
        let sc = span.span_context();
        sc.is_valid().then(|| {
            let flags = if sc.is_sampled() {
                TraceFlags::SAMPLED
            } else {
                TraceFlags::default()
            };
            (sc.trace_id(), sc.span_id(), Some(flags))
        })
    }

    /// Routes governed content through the capture gate.
    fn governed(&self, has_details_event: bool) -> Destinations {
        let capture = self.observer.content.capture();
        Destinations {
            span: capture.on_spans(),
            event: has_details_event && capture.on_events() && self.observer.events.is_some(),
        }
    }

    /// Schema content: recorded on spans in every mode, on events when event
    /// capture is enabled.
    fn schema(&self) -> Destinations {
        let capture = self.observer.content.capture();
        Destinations {
            span: true,
            event: capture.on_events() && self.observer.events.is_some(),
        }
    }

    /// Runs the content pipeline for a structured value.
    fn prepare_json<T: Serialize + ?Sized>(
        &self,
        key: &str,
        value: &T,
        to: Destinations,
    ) -> Recorded {
        match self.observer.content.prepare_json(key, value) {
            Outcome::Keep(b) => Recorded::Value {
                span: to.span.then_some(b.json),
                event: to.event.then_some(b.value),
                truncated: b.truncated,
            },
            Outcome::Dropped => Recorded::Dropped,
            Outcome::Failed => Recorded::Failed,
        }
    }

    /// Runs the content pipeline for a plain-string content value.
    fn prepare_text(&self, key: &str, text: &str, to: Destinations) -> Recorded {
        let content = &self.observer.content;
        match content.prepare_text(key, text, content.text_bound()) {
            Outcome::Keep(b) => Recorded::Value {
                event: to.event.then(|| Value::String(b.text.clone())),
                span: to.span.then_some(b.text),
                truncated: b.truncated,
            },
            Outcome::Dropped => Recorded::Dropped,
            Outcome::Failed => Recorded::Failed,
        }
    }

    fn retained_budget(&self) -> usize {
        self.observer.content.max_retained_operation_bytes()
    }

    /// Marks the span failed. Returns the error message that survived the
    /// content pipeline (redaction, rules, 256 code points), if any; the span
    /// status description falls back to the error type.
    fn fail(&self, error: &ErrorInfo) -> Option<String> {
        let message = error.message.as_deref().and_then(|m| {
            match self.observer.content.prepare_text(
                ERROR_MESSAGE_KIND,
                m,
                TextBound::Chars(MAX_ERROR_MESSAGE_CHARS),
            ) {
                Outcome::Keep(b) => Some(b.text),
                Outcome::Dropped | Outcome::Failed => None,
            }
        });
        if let Some(s) = &self.span {
            let span = s.cx.span();
            span.set_attribute(KeyValue::new(attr::ERROR_TYPE, error.error_type.clone()));
            span.set_status(Status::error(
                message.clone().unwrap_or_else(|| error.error_type.clone()),
            ));
        }
        message
    }

    /// Ends the span; it never ends before it starts.
    fn end_span(&self, at: SystemTime) {
        if let Some(s) = &self.span {
            s.cx.span().end_with_timestamp(at.max(self.start));
        }
    }
}

/// Inference state guarded by a mutex. Streaming state is constant-size.
#[derive(Default)]
struct InferenceState {
    ended: bool,
    last_chunk: Option<SystemTime>,
    chunk_count: u64,
    first_token: bool,
    usage: Usage,
    finish_reasons: Option<Vec<String>>,
    output_recorded: bool,
    response_model: Option<String>,
    response_id: Option<String>,
    attempt_count: u32,
    content: Retained,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Category {
    Inference,
    Embeddings,
    Custom,
}

impl Category {
    /// Catalog category for `start_inference` names: only
    /// `catalog.inference_operations` resolve through the catalog; any other
    /// name is a custom inference operation (generative inference metrics).
    fn of(operation: &str) -> Self {
        match operation {
            op::EMBEDDINGS => Self::Embeddings,
            op::PREDICT => Self::Custom,
            _ => Self::Inference,
        }
    }
}

/// A logical inference request. See the crate docs for the span shape.
pub struct InferenceOperation {
    core: Core,
    request: InferenceRequest,
    category: Category,
    dims: Vec<KeyValue>,
    state: Mutex<InferenceState>,
}

impl InferenceOperation {
    pub(crate) fn start(
        observer: Arc<ObserverInner>,
        request: InferenceRequest,
        mut options: StartOptions,
    ) -> Self {
        let clock = observer.clock.anchor();
        let start = options.start_time.unwrap_or_else(|| clock.now());
        let category = Category::of(&request.operation);
        let mut dims = vec![
            KeyValue::new(attr::OPERATION_NAME, request.operation.clone()),
            KeyValue::new(attr::PROVIDER_NAME, request.provider.clone()),
        ];
        if let Some(model) = &request.model {
            dims.push(KeyValue::new(attr::REQUEST_MODEL, model.clone()));
        }
        if let Some(addr) = &request.server_address {
            dims.push(KeyValue::new(attr::SERVER_ADDRESS, addr.clone()));
        }
        if let Some(port) = request.server_port {
            dims.push(KeyValue::new(attr::SERVER_PORT, port));
        }
        if let Some(location) = request.location {
            dims.push(KeyValue::new(attr::EXECUTION_LOCATION, location.as_str()));
        }

        let mut attributes = dims.clone();
        push_request_attributes(&mut attributes, &request);
        let name = span_name(&request.operation, request.model.as_deref());
        let span = observer.start_span(
            name,
            kind_for(request.location),
            attributes,
            &mut options,
            start,
        );
        let operation = Self {
            core: Core {
                observer,
                span,
                clock,
                start,
            },
            request,
            category,
            dims,
            state: Mutex::new(InferenceState::default()),
        };
        operation.record_prompt_variables();
        operation
    }

    /// Prompt variables are content: processed in ascending name order and
    /// retained like other content.
    fn record_prompt_variables(&self) {
        let Some(prompt) = &self.request.prompt else {
            return;
        };
        let to = self.core.governed(true);
        if !to.any() {
            return;
        }
        let budget = self.core.retained_budget();
        for (name, value) in &prompt.variables {
            let key = format!("{}{name}", attr::PROMPT_VARIABLE_PREFIX);
            let recorded = self.core.prepare_text(&key, value, to);
            lock(&self.state).content.apply(&key, recorded, budget);
        }
    }

    /// The context for child operations: this operation's span (also when it
    /// is not sampled), or the parent context passed through by an API-only
    /// tracer. `None` when tracing is suppressed or no tracer is configured.
    #[must_use]
    pub fn context(&self) -> Option<ObservationContext> {
        self.core.context()
    }

    /// Reference to this operation's own span (sampled or not); `None` when
    /// the operation has no span of its own.
    #[must_use]
    pub fn observation_ref(&self) -> Option<ObservationRef> {
        self.core.observation_ref()
    }

    /// The request this operation was started with.
    #[must_use]
    pub fn request(&self) -> &InferenceRequest {
        &self.request
    }

    /// Whether the operation has reached its first terminal transition.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.is_ended()
    }

    fn is_ended(&self) -> bool {
        lock(&self.state).ended
    }

    fn record_content<T: Serialize + ?Sized>(
        &self,
        key: &'static str,
        value: &T,
        to: Destinations,
    ) {
        if !to.any() || self.is_ended() {
            return;
        }
        let recorded = self.core.prepare_json(key, value, to);
        let budget = self.core.retained_budget();
        let mut state = lock(&self.state);
        if !state.ended {
            state.content.apply(key, recorded, budget);
        }
    }

    /// Records the system instructions (content).
    pub fn record_system_instructions(&self, parts: &[Part]) {
        self.record_content(attr::SYSTEM_INSTRUCTIONS, parts, self.core.governed(true));
    }

    /// Records the input messages (content).
    pub fn record_input_messages(&self, messages: &[InputMessage]) {
        self.record_content(attr::INPUT_MESSAGES, messages, self.core.governed(true));
    }

    /// Records the output messages (content).
    pub fn record_output_messages(&self, messages: &[OutputMessage]) {
        {
            let mut state = lock(&self.state);
            if state.ended {
                return;
            }
            state.output_recorded = true;
        }
        self.record_content(attr::OUTPUT_MESSAGES, messages, self.core.governed(true));
    }

    /// Records the tool definitions offered to the model. Schema content:
    /// recorded in every capture mode, still redacted and bounded.
    pub fn record_tool_definitions(&self, definitions: &Value) {
        self.record_content(attr::TOOL_DEFINITIONS, definitions, self.core.schema());
    }

    fn stream_dims(&self, state: &InferenceState) -> Vec<KeyValue> {
        let mut dims = self.dims.clone();
        if let Some(model) = &state.response_model {
            dims.push(KeyValue::new(attr::RESPONSE_MODEL, model.clone()));
        }
        dims
    }

    /// Marks one delivered output chunk now. A chunk never sets or influences
    /// first-token time.
    pub fn record_chunk(&self) {
        self.record_chunk_at(self.core.now());
    }

    /// Marks one delivered output chunk at `at`. The first call records the
    /// time to first chunk; each later call records the time since the
    /// previous call's timestamp (clamped at zero).
    pub fn record_chunk_at(&self, at: SystemTime) {
        let mut state = lock(&self.state);
        if state.ended {
            return;
        }
        state.chunk_count += 1;
        let instruments = self.core.observer.instruments.as_ref();
        match state.last_chunk {
            None => {
                let ttfc = seconds_between(self.core.start, at);
                self.core
                    .set_attribute(KeyValue::new(attr::RESPONSE_TIME_TO_FIRST_CHUNK, ttfc));
                if let Some(i) = instruments {
                    i.time_to_first_chunk
                        .record(ttfc, &self.stream_dims(&state));
                }
            }
            Some(previous) => {
                if let Some(i) = instruments {
                    i.time_per_output_chunk
                        .record(seconds_between(previous, at), &self.stream_dims(&state));
                }
            }
        }
        state.last_chunk = Some(at);
    }

    /// Records the runtime-measured first output token now. The first call
    /// wins. Pass only measured times, never a chunk-callback estimate.
    pub fn record_first_token(&self) {
        self.record_first_token_at(self.core.now());
    }

    /// Records the runtime-measured first output token at `at`. The first
    /// call wins; chunks never set it, so a measured time reported after the
    /// chunks is still recorded.
    pub fn record_first_token_at(&self, at: SystemTime) {
        let mut state = lock(&self.state);
        if state.ended || state.first_token {
            return;
        }
        state.first_token = true;
        let ttft = seconds_between(self.core.start, at);
        self.core
            .set_attribute(KeyValue::new(attr::RESPONSE_TIME_TO_FIRST_TOKEN, ttft));
        if let Some(i) = self.core.observer.instruments.as_ref() {
            i.time_to_first_token
                .record(ttft, &self.stream_dims(&state));
        }
    }

    /// Records the runtime's measured mean time per output token, seconds.
    /// Non-finite values are ignored.
    pub fn record_token_timing(&self, time_per_output_token_s: f64) {
        if !time_per_output_token_s.is_finite() || self.is_ended() {
            return;
        }
        self.core.set_attribute(KeyValue::new(
            attr::RESPONSE_TIME_PER_OUTPUT_TOKEN,
            time_per_output_token_s,
        ));
    }

    /// Records request usage totals (merged field by field).
    pub fn record_usage(&self, usage: &Usage) {
        let mut state = lock(&self.state);
        if !state.ended {
            state.usage.merge(usage);
        }
    }

    /// Records finish reasons, one per choice (last call wins). An empty
    /// list clears them: empty lists are never recorded.
    pub fn record_finish_reasons<I, S>(&self, reasons: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let reasons: Vec<String> = reasons.into_iter().map(Into::into).collect();
        let mut state = lock(&self.state);
        if !state.ended {
            state.finish_reasons = (!reasons.is_empty()).then_some(reasons);
        }
    }

    /// Records the response id and/or model. The response model joins the
    /// metric dimensions from now on (last recorded value wins).
    pub fn record_response(&self, response_id: Option<&str>, model: Option<&str>) {
        let mut state = lock(&self.state);
        if state.ended {
            return;
        }
        if let Some(id) = response_id {
            state.response_id = Some(id.to_owned());
        }
        if let Some(m) = model {
            state.response_model = Some(m.to_owned());
        }
    }

    /// Records a host-measured resource sample (non-finite values ignored).
    pub fn record_resource(&self, sample: &ResourceSample) {
        self.core.set_attributes(resource_attributes(sample));
    }

    /// Sets an extra span attribute (adapter extensions). Never changes
    /// metric dimensions.
    pub fn set_attribute(&self, kv: KeyValue) {
        self.core.set_attribute(kv);
    }

    /// Starts a concrete execution attempt as a child span.
    #[must_use]
    pub fn start_attempt(&self, attempt: AttemptInfo) -> AttemptOperation {
        self.start_attempt_at(attempt, self.core.now())
    }

    /// Starts a concrete execution attempt at `at`. Attempts of an operation
    /// without a span of its own are suppressed.
    #[must_use]
    pub fn start_attempt_at(&self, attempt: AttemptInfo, at: SystemTime) -> AttemptOperation {
        lock(&self.state).attempt_count += 1;
        AttemptOperation::start(
            self.core.observer.clone(),
            self.request.operation.clone(),
            self.core.own_context(),
            attempt,
            at,
        )
    }

    /// Ends the operation successfully now.
    pub fn end(&self) {
        self.end_at(self.core.now());
    }

    /// Ends the operation successfully at `at`.
    pub fn end_at(&self, at: SystemTime) {
        self.finish(at, None);
    }

    /// Ends the operation with an error now.
    pub fn end_with_error(&self, error: &ErrorInfo) {
        self.end_with_error_at(error, self.core.now());
    }

    /// Ends the operation with an error at `at`.
    pub fn end_with_error_at(&self, error: &ErrorInfo, at: SystemTime) {
        self.finish(at, Some(error));
    }

    /// Ends the operation as cancelled now.
    pub fn cancel(&self) {
        self.end_with_error(&ErrorInfo::new("cancelled"));
    }

    /// Ends the operation as cancelled at `at`.
    pub fn cancel_at(&self, at: SystemTime) {
        self.end_with_error_at(&ErrorInfo::new("cancelled"), at);
    }

    fn finish(&self, at: SystemTime, error: Option<&ErrorInfo>) {
        let mut state = lock(&self.state);
        if state.ended {
            return;
        }
        state.ended = true;
        let core = &self.core;

        if error.is_some() && state.output_recorded && state.finish_reasons.is_none() {
            state.finish_reasons = Some(vec!["error".to_owned()]);
        }

        let mut end_attrs = usage_attributes(&state.usage);
        if let Some(id) = &state.response_id {
            end_attrs.push(KeyValue::new(attr::RESPONSE_ID, id.clone()));
        }
        if let Some(model) = &state.response_model {
            end_attrs.push(KeyValue::new(attr::RESPONSE_MODEL, model.clone()));
        }
        if let Some(reasons) = &state.finish_reasons {
            end_attrs.push(KeyValue::new(
                attr::RESPONSE_FINISH_REASONS,
                string_array(reasons.iter().cloned()),
            ));
        }
        if state.chunk_count > 0 {
            end_attrs.push(u64_attr(attr::RESPONSE_CHUNK_COUNT, state.chunk_count));
        }
        if state.attempt_count > 0 {
            end_attrs.push(KeyValue::new(
                attr::ATTEMPT_COUNT,
                i64::from(state.attempt_count),
            ));
        }
        if let Some(v) = state.content.truncated_value() {
            end_attrs.push(KeyValue::new(attr::CONTENT_TRUNCATED_ATTRIBUTES, v));
        }
        core.set_attributes(end_attrs);
        let message = error.map(|e| core.fail(e));
        core.set_attributes(state.content.span_attributes());

        self.record_metrics(&state, at, error);
        self.emit_events(&mut state, at, error, message.flatten());
        core.end_span(at);
    }

    fn record_metrics(&self, state: &InferenceState, at: SystemTime, error: Option<&ErrorInfo>) {
        let Some(instruments) = self.core.observer.instruments.as_ref() else {
            return;
        };
        let base = self.stream_dims(state);
        let mut with_error = base.clone();
        if let Some(e) = error {
            with_error.push(KeyValue::new(attr::ERROR_TYPE, e.error_type.clone()));
        }
        let duration = seconds_between(self.core.start, at);
        match self.category {
            Category::Inference => instruments.inference_duration.record(duration, &with_error),
            Category::Embeddings => instruments
                .embeddings_duration
                .record(duration, &with_error),
            Category::Custom => instruments.operation_duration.record(duration, &with_error),
        }

        let usage = &state.usage;
        match self.category {
            Category::Custom => {}
            Category::Embeddings => {
                if let Some(n) = usage.input_tokens {
                    instruments.embeddings_input_tokens.record(n, &with_error);
                }
            }
            Category::Inference => {
                if let Some(n) = usage.input_tokens {
                    instruments.operation_input_tokens.record(n, &with_error);
                }
                if let Some(n) = usage.output_tokens {
                    instruments.operation_output_tokens.record(n, &with_error);
                }
                let with_modality = |modality: &'static str| {
                    let mut dims = base.clone();
                    dims.push(KeyValue::new(attr::TOKEN_MODALITY, modality));
                    dims
                };
                let split = |total: Option<u64>, parts: [(&'static str, Option<u64>); 3]| {
                    let mut points: Vec<(&'static str, u64)> = parts
                        .iter()
                        .filter_map(|(m, v)| v.filter(|v| *v > 0).map(|v| (*m, v)))
                        .collect();
                    let covered: u64 = points.iter().map(|(_, v)| *v).sum();
                    if let Some(total) = total {
                        if total > covered {
                            points.push(("unknown", total - covered));
                        }
                    }
                    points
                };
                for (m, v) in split(
                    usage.input_tokens,
                    [
                        ("text", usage.text_input_tokens),
                        ("image", usage.image_input_tokens),
                        ("audio", usage.audio_input_tokens),
                    ],
                ) {
                    instruments.usage_input.add(v, &with_modality(m));
                }
                for (m, v) in split(
                    usage.output_tokens,
                    [
                        ("text", usage.text_output_tokens),
                        ("image", usage.image_output_tokens),
                        ("audio", usage.audio_output_tokens),
                    ],
                ) {
                    instruments.usage_output.add(v, &with_modality(m));
                }
                let input_modality = single_token_modality(&self.request.input_modalities);
                let output_modality = single_token_modality(&self.request.output_modalities);
                if let Some(v) = usage.cache_read_input_tokens.filter(|v| *v > 0) {
                    instruments
                        .usage_cache_read
                        .add(v, &with_modality(input_modality));
                }
                if let Some(v) = usage.cache_write_input_tokens.filter(|v| *v > 0) {
                    instruments
                        .usage_cache_write
                        .add(v, &with_modality(input_modality));
                }
                if let Some(v) = usage.reasoning_output_tokens.filter(|v| *v > 0) {
                    instruments
                        .usage_reasoning
                        .add(v, &with_modality(output_modality));
                }
            }
        }
    }

    fn emit_events(
        &self,
        state: &mut InferenceState,
        at: SystemTime,
        error: Option<&ErrorInfo>,
        message: Option<String>,
    ) {
        let Some(events) = self.core.observer.events.as_ref() else {
            return;
        };
        let trace = self.core.trace_ids();
        if let Some(e) = error.filter(|e| e.error_type != "cancelled") {
            let mut attributes = vec![(
                Key::from_static_str(attr::EXCEPTION_TYPE),
                AnyValue::from(e.error_type.clone()),
            )];
            // Only a message that survived redaction is emitted; it is never
            // replaced by the error type.
            push_any(
                &mut attributes,
                attr::EXCEPTION_MESSAGE,
                message.map(AnyValue::from),
            );
            events.emit(LogEvent {
                name: event::OPERATION_EXCEPTION,
                severity: Severity::Warn,
                severity_text: "WARN",
                timestamp: at,
                trace,
                attributes,
            });
        }

        if !self.core.observer.content.capture().on_events() {
            return;
        }
        let mut attributes: Vec<(Key, AnyValue)> = vec![
            (
                Key::from_static_str(attr::OPERATION_NAME),
                AnyValue::from(self.request.operation.clone()),
            ),
            (
                Key::from_static_str(attr::PROVIDER_NAME),
                AnyValue::from(self.request.provider.clone()),
            ),
        ];
        push_any(
            &mut attributes,
            attr::REQUEST_MODEL,
            self.request.model.clone().map(AnyValue::from),
        );
        push_any(
            &mut attributes,
            attr::RESPONSE_MODEL,
            state.response_model.clone().map(AnyValue::from),
        );
        push_any(
            &mut attributes,
            attr::RESPONSE_ID,
            state.response_id.clone().map(AnyValue::from),
        );
        push_any(
            &mut attributes,
            attr::USAGE_INPUT_TOKENS,
            state
                .usage
                .input_tokens
                .map(|v| AnyValue::Int(i64::try_from(v).unwrap_or(i64::MAX))),
        );
        push_any(
            &mut attributes,
            attr::USAGE_OUTPUT_TOKENS,
            state
                .usage
                .output_tokens
                .map(|v| AnyValue::Int(i64::try_from(v).unwrap_or(i64::MAX))),
        );
        if let Some(reasons) = &state.finish_reasons {
            attributes.push((
                Key::from_static_str(attr::RESPONSE_FINISH_REASONS),
                AnyValue::ListAny(Box::new(
                    reasons.iter().cloned().map(AnyValue::from).collect(),
                )),
            ));
        }
        if let Some(e) = error {
            push_any(
                &mut attributes,
                attr::ERROR_TYPE,
                Some(AnyValue::from(e.error_type.clone())),
            );
        }
        let truncated = state.content.truncated_any();
        attributes.extend(state.content.take_event_values());
        push_any(
            &mut attributes,
            attr::CONTENT_TRUNCATED_ATTRIBUTES,
            truncated,
        );
        events.emit(LogEvent {
            name: event::INFERENCE_DETAILS,
            severity: Severity::Debug,
            severity_text: "DEBUG",
            timestamp: at,
            trace,
            attributes,
        });
    }
}

fn single_token_modality(modalities: &[crate::content::Modality]) -> &'static str {
    use crate::content::Modality;
    match modalities {
        [Modality::Text] => "text",
        [Modality::Image] => "image",
        [Modality::Audio] => "audio",
        _ => "unknown",
    }
}

fn push_request_attributes(out: &mut Vec<KeyValue>, request: &InferenceRequest) {
    if let Some(v) = request.max_tokens {
        out.push(KeyValue::new(attr::REQUEST_MAX_TOKENS, v));
    }
    if let Some(v) = finite(request.temperature) {
        out.push(KeyValue::new(attr::REQUEST_TEMPERATURE, v));
    }
    if let Some(v) = finite(request.top_p) {
        out.push(KeyValue::new(attr::REQUEST_TOP_P, v));
    }
    if let Some(v) = request.top_k {
        out.push(KeyValue::new(attr::REQUEST_TOP_K, v));
    }
    if let Some(v) = request.seed {
        out.push(KeyValue::new(attr::REQUEST_SEED, v));
    }
    if !request.stop_sequences.is_empty() {
        out.push(KeyValue::new(
            attr::REQUEST_STOP_SEQUENCES,
            string_array(request.stop_sequences.iter().cloned()),
        ));
    }
    if let Some(v) = finite(request.frequency_penalty) {
        out.push(KeyValue::new(attr::REQUEST_FREQUENCY_PENALTY, v));
    }
    if let Some(v) = finite(request.presence_penalty) {
        out.push(KeyValue::new(attr::REQUEST_PRESENCE_PENALTY, v));
    }
    if request.stream {
        out.push(KeyValue::new(attr::REQUEST_STREAM, true));
    }
    if let Some(v) = &request.output_type {
        out.push(KeyValue::new(attr::OUTPUT_TYPE, v.clone()));
    }
    if let Some(v) = &request.conversation_id {
        out.push(KeyValue::new(attr::CONVERSATION_ID, v.clone()));
    }
    if let Some(prompt) = &request.prompt {
        out.push(KeyValue::new(attr::PROMPT_NAME, prompt.name.clone()));
        if let Some(v) = &prompt.version {
            out.push(KeyValue::new(attr::PROMPT_VERSION, v.clone()));
        }
    }
    if let Some(v) = &request.runtime {
        out.push(KeyValue::new(attr::RUNTIME_NAME, v.clone()));
    }
    if let Some(v) = &request.runtime_version {
        out.push(KeyValue::new(attr::RUNTIME_VERSION, v.clone()));
    }
    if let Some(v) = &request.accelerator {
        out.push(KeyValue::new(attr::ACCELERATOR, v.clone()));
    }
    if let Some(v) = &request.task {
        out.push(KeyValue::new(attr::TASK, v.clone()));
    }
    if !request.input_modalities.is_empty() {
        out.push(KeyValue::new(
            attr::INPUT_MODALITIES,
            string_array(
                request
                    .input_modalities
                    .iter()
                    .map(|m| m.as_str().to_owned()),
            ),
        ));
    }
    if !request.output_modalities.is_empty() {
        out.push(KeyValue::new(
            attr::OUTPUT_MODALITIES,
            string_array(
                request
                    .output_modalities
                    .iter()
                    .map(|m| m.as_str().to_owned()),
            ),
        ));
    }
    if let Some(v) = request.embedding_dimensions {
        out.push(KeyValue::new(attr::EMBEDDINGS_DIMENSION_COUNT, v));
    }
}

impl Drop for InferenceOperation {
    fn drop(&mut self) {
        if !self.is_ended() {
            self.cancel();
        }
    }
}

impl fmt::Debug for InferenceOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceOperation")
            .field("operation", &self.request.operation)
            .field("model", &self.request.model)
            .field("observation_ref", &self.observation_ref())
            .finish_non_exhaustive()
    }
}

/// One concrete execution attempt of a logical request.
pub struct AttemptOperation {
    core: Core,
    dims: Vec<KeyValue>,
    state: Mutex<(bool, Usage)>,
}

impl AttemptOperation {
    fn start(
        observer: Arc<ObserverInner>,
        operation: String,
        parent: Option<ObservationContext>,
        attempt: AttemptInfo,
        start: SystemTime,
    ) -> Self {
        let clock = observer.clock.anchor();
        let mut dims = vec![
            KeyValue::new(attr::OPERATION_NAME, operation),
            KeyValue::new(attr::PROVIDER_NAME, attempt.provider.clone()),
        ];
        if let Some(model) = &attempt.model {
            dims.push(KeyValue::new(attr::REQUEST_MODEL, model.clone()));
        }
        dims.push(KeyValue::new(
            attr::EXECUTION_LOCATION,
            attempt.location.as_str(),
        ));
        dims.push(KeyValue::new(attr::ATTEMPT_REASON, attempt.reason.as_str()));

        let mut attributes = vec![
            KeyValue::new(attr::ATTEMPT_INDEX, i64::from(attempt.index)),
            KeyValue::new(attr::ATTEMPT_REASON, attempt.reason.as_str()),
            KeyValue::new(attr::PROVIDER_NAME, attempt.provider.clone()),
            KeyValue::new(attr::EXECUTION_LOCATION, attempt.location.as_str()),
        ];
        if let Some(model) = &attempt.model {
            attributes.push(KeyValue::new(attr::REQUEST_MODEL, model.clone()));
        }
        if let Some(runtime) = &attempt.runtime {
            attributes.push(KeyValue::new(attr::RUNTIME_NAME, runtime.clone()));
        }
        if let Some(addr) = &attempt.server_address {
            attributes.push(KeyValue::new(attr::SERVER_ADDRESS, addr.clone()));
        }
        if let Some(port) = attempt.server_port {
            attributes.push(KeyValue::new(attr::SERVER_PORT, port));
        }
        // An attempt without a parent span of its own stays untraced so it
        // never becomes a stray root.
        let mut options = StartOptions {
            trace: if parent.is_some() {
                TraceMode::Default
            } else {
                TraceMode::Suppress
            },
            parent,
            start_time: Some(start),
            attributes: Vec::new(),
        };
        let span = observer.start_span(
            format!("attempt {}", attempt.index),
            kind_for(Some(attempt.location)),
            attributes,
            &mut options,
            start,
        );
        Self {
            core: Core {
                observer,
                span,
                clock,
                start,
            },
            dims,
            state: Mutex::new((false, Usage::default())),
        }
    }

    /// Context for children (e.g. HTTP client spans).
    #[must_use]
    pub fn context(&self) -> Option<ObservationContext> {
        self.core.context()
    }

    /// Reference to this attempt's own span.
    #[must_use]
    pub fn observation_ref(&self) -> Option<ObservationRef> {
        self.core.observation_ref()
    }

    /// Records this attempt's own usage.
    pub fn record_usage(&self, usage: &Usage) {
        let mut state = lock(&self.state);
        if !state.0 {
            state.1.merge(usage);
        }
    }

    /// Records a resource sample (non-finite values ignored).
    pub fn record_resource(&self, sample: &ResourceSample) {
        self.core.set_attributes(resource_attributes(sample));
    }

    /// Sets an extra span attribute.
    pub fn set_attribute(&self, kv: KeyValue) {
        self.core.set_attribute(kv);
    }

    /// Ends the attempt successfully now.
    pub fn end(&self) {
        self.finish(self.core.now(), None);
    }

    /// Ends the attempt successfully at `at`.
    pub fn end_at(&self, at: SystemTime) {
        self.finish(at, None);
    }

    /// Ends the attempt with an error now.
    pub fn end_with_error(&self, error: &ErrorInfo) {
        self.finish(self.core.now(), Some(error));
    }

    /// Ends the attempt with an error at `at`.
    pub fn end_with_error_at(&self, error: &ErrorInfo, at: SystemTime) {
        self.finish(at, Some(error));
    }

    /// Ends the attempt as cancelled now.
    pub fn cancel(&self) {
        self.end_with_error(&ErrorInfo::new("cancelled"));
    }

    /// Ends the attempt as cancelled at `at`.
    pub fn cancel_at(&self, at: SystemTime) {
        self.end_with_error_at(&ErrorInfo::new("cancelled"), at);
    }

    fn finish(&self, at: SystemTime, error: Option<&ErrorInfo>) {
        let mut state = lock(&self.state);
        if state.0 {
            return;
        }
        state.0 = true;
        self.core.set_attributes(usage_attributes(&state.1));
        if let Some(e) = error {
            self.core.fail(e);
        }
        if let Some(i) = self.core.observer.instruments.as_ref() {
            let mut dims = self.dims.clone();
            if let Some(e) = error {
                dims.push(KeyValue::new(attr::ERROR_TYPE, e.error_type.clone()));
            }
            i.attempts.add(1, &dims);
        }
        self.core.end_span(at);
    }
}

impl Drop for AttemptOperation {
    fn drop(&mut self) {
        if !lock(&self.state).0 {
            self.cancel();
        }
    }
}

impl fmt::Debug for AttemptOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AttemptOperation")
            .field("observation_ref", &self.observation_ref())
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct SimpleState {
    ended: bool,
    content: Retained,
}

/// A simple timed operation (workflow, tool, model load).
struct Simple {
    core: Core,
    dims: Vec<KeyValue>,
    state: Mutex<SimpleState>,
}

impl Simple {
    fn new(
        observer: Arc<ObserverInner>,
        clock: OperationClock,
        start: SystemTime,
        span: Option<SpanState>,
        dims: Vec<KeyValue>,
    ) -> Self {
        Self {
            core: Core {
                observer,
                span,
                clock,
                start,
            },
            dims,
            state: Mutex::new(SimpleState::default()),
        }
    }

    fn finish(
        &self,
        at: SystemTime,
        error: Option<&ErrorInfo>,
        record: impl FnOnce(&Instruments, f64, &[KeyValue]),
    ) {
        let mut state = lock(&self.state);
        if state.ended {
            return;
        }
        state.ended = true;
        let core = &self.core;
        if let Some(v) = state.content.truncated_value() {
            core.set_attribute(KeyValue::new(attr::CONTENT_TRUNCATED_ATTRIBUTES, v));
        }
        if let Some(e) = error {
            core.fail(e);
        }
        core.set_attributes(state.content.span_attributes());
        if let Some(i) = core.observer.instruments.as_ref() {
            let mut dims = self.dims.clone();
            if let Some(e) = error {
                dims.push(KeyValue::new(attr::ERROR_TYPE, e.error_type.clone()));
            }
            record(i, seconds_between(core.start, at), &dims);
        }
        if let Some(events) = core
            .observer
            .events
            .as_ref()
            .filter(|_| state.content.has_event_values())
        {
            let mut attributes: Vec<(Key, AnyValue)> = self
                .dims
                .iter()
                .map(|kv| (kv.key.clone(), otel_to_any(&kv.value)))
                .collect();
            let truncated = state.content.truncated_any();
            attributes.extend(state.content.take_event_values());
            if let Some(e) = error {
                attributes.push((
                    Key::from_static_str(attr::ERROR_TYPE),
                    AnyValue::from(e.error_type.clone()),
                ));
            }
            push_any(
                &mut attributes,
                attr::CONTENT_TRUNCATED_ATTRIBUTES,
                truncated,
            );
            events.emit(LogEvent {
                name: event::TOOL_DETAILS,
                severity: Severity::Debug,
                severity_text: "DEBUG",
                timestamp: at,
                trace: core.trace_ids(),
                attributes,
            });
        }
        core.end_span(at);
    }

    fn is_ended(&self) -> bool {
        lock(&self.state).ended
    }
}

macro_rules! lifecycle {
    ($ty:ident, $record:expr) => {
        impl $ty {
            /// Context for child operations (see
            /// [`InferenceOperation::context`]).
            #[must_use]
            pub fn context(&self) -> Option<ObservationContext> {
                self.simple.core.context()
            }

            /// Reference to this operation's own span (sampled or not).
            #[must_use]
            pub fn observation_ref(&self) -> Option<ObservationRef> {
                self.simple.core.observation_ref()
            }

            /// Whether the operation has reached its first terminal transition.
            #[must_use]
            pub fn is_finished(&self) -> bool {
                self.simple.is_ended()
            }

            /// Sets an extra span attribute. Never changes metric dimensions.
            pub fn set_attribute(&self, kv: KeyValue) {
                self.simple.core.set_attribute(kv);
            }

            /// Ends successfully now.
            pub fn end(&self) {
                self.end_at(self.simple.core.now());
            }

            /// Ends successfully at `at`.
            pub fn end_at(&self, at: SystemTime) {
                self.simple.finish(at, None, $record);
            }

            /// Ends with an error now.
            pub fn end_with_error(&self, error: &ErrorInfo) {
                self.end_with_error_at(error, self.simple.core.now());
            }

            /// Ends with an error at `at`.
            pub fn end_with_error_at(&self, error: &ErrorInfo, at: SystemTime) {
                self.simple.finish(at, Some(error), $record);
            }

            /// Ends as cancelled now.
            pub fn cancel(&self) {
                self.end_with_error(&ErrorInfo::new("cancelled"));
            }

            /// Ends as cancelled at `at`.
            pub fn cancel_at(&self, at: SystemTime) {
                self.end_with_error_at(&ErrorInfo::new("cancelled"), at);
            }
        }

        impl Drop for $ty {
            fn drop(&mut self) {
                if !self.simple.is_ended() {
                    self.cancel();
                }
            }
        }

        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($ty))
                    .field("observation_ref", &self.observation_ref())
                    .finish_non_exhaustive()
            }
        }
    };
}

/// A workflow invocation (`invoke_workflow {name}`).
pub struct WorkflowOperation {
    simple: Simple,
}

impl WorkflowOperation {
    pub(crate) fn start(
        observer: Arc<ObserverInner>,
        name: String,
        mut options: StartOptions,
    ) -> Self {
        let clock = observer.clock.anchor();
        let start = options.start_time.unwrap_or_else(|| clock.now());
        let dims = vec![
            KeyValue::new(attr::OPERATION_NAME, op::INVOKE_WORKFLOW),
            KeyValue::new(attr::WORKFLOW_NAME, name.clone()),
        ];
        let span = observer.start_span(
            span_name(op::INVOKE_WORKFLOW, Some(&name)),
            SpanKind::Internal,
            dims.clone(),
            &mut options,
            start,
        );
        Self {
            simple: Simple::new(observer, clock, start, span, dims),
        }
    }
}

lifecycle!(
    WorkflowOperation,
    |i: &Instruments, d, dims: &[KeyValue]| {
        i.workflow_duration.record(d, dims);
    }
);

/// A tool execution (`execute_tool {name}`).
pub struct ToolOperation {
    simple: Simple,
}

impl ToolOperation {
    pub(crate) fn start(
        observer: Arc<ObserverInner>,
        call: ToolCallInfo,
        mut options: StartOptions,
    ) -> Self {
        let clock = observer.clock.anchor();
        let start = options.start_time.unwrap_or_else(|| clock.now());
        let mut dims = vec![
            KeyValue::new(attr::OPERATION_NAME, op::EXECUTE_TOOL),
            KeyValue::new(attr::TOOL_NAME, call.name.clone()),
        ];
        if let Some(t) = &call.tool_type {
            dims.push(KeyValue::new(attr::TOOL_TYPE, t.clone()));
        }
        let mut attributes = dims.clone();
        if let Some(id) = &call.call_id {
            attributes.push(KeyValue::new(attr::TOOL_CALL_ID, id.clone()));
        }
        if let Some(d) = &call.description {
            attributes.push(KeyValue::new(attr::TOOL_DESCRIPTION, d.clone()));
        }
        let span = observer.start_span(
            span_name(op::EXECUTE_TOOL, Some(&call.name)),
            SpanKind::Internal,
            attributes,
            &mut options,
            start,
        );
        let operation = Self {
            simple: Simple::new(observer, clock, start, span, dims),
        };
        if let Some(args) = &call.arguments {
            operation.record_content(attr::TOOL_CALL_ARGUMENTS, args);
        }
        operation
    }

    /// Records the tool result (content). Recording again replaces it.
    pub fn record_result(&self, result: &Value) {
        self.record_content(attr::TOOL_CALL_RESULT, result);
    }

    fn record_content(&self, key: &'static str, value: &Value) {
        let core = &self.simple.core;
        let to = core.governed(true);
        if !to.any() || self.simple.is_ended() {
            return;
        }
        let recorded = core.prepare_json(key, value, to);
        let budget = core.retained_budget();
        // Recording is serialized with finish so no content arrives after
        // the details event.
        let mut state = lock(&self.simple.state);
        if !state.ended {
            state.content.apply(key, recorded, budget);
        }
    }
}

lifecycle!(ToolOperation, |i: &Instruments, d, dims: &[KeyValue]| {
    i.tool_duration.record(d, dims);
});

/// Loading a model into a runtime (`load_model {model}`).
pub struct ModelLoadOperation {
    simple: Simple,
}

impl ModelLoadOperation {
    pub(crate) fn start(
        observer: Arc<ObserverInner>,
        info: ModelLoadInfo,
        mut options: StartOptions,
    ) -> Self {
        let clock = observer.clock.anchor();
        let start = options.start_time.unwrap_or_else(|| clock.now());
        // Provider, else the runtime, else absent (never fabricated).
        let provider = info.provider.clone().or_else(|| info.runtime.clone());
        let mut dims = vec![KeyValue::new(attr::OPERATION_NAME, op::LOAD_MODEL)];
        if let Some(provider) = provider {
            dims.push(KeyValue::new(attr::PROVIDER_NAME, provider));
        }
        dims.push(KeyValue::new(attr::REQUEST_MODEL, info.model.clone()));
        let mut attributes = dims.clone();
        if let Some(v) = &info.runtime {
            attributes.push(KeyValue::new(attr::RUNTIME_NAME, v.clone()));
        }
        if let Some(v) = &info.format {
            attributes.push(KeyValue::new(attr::MODEL_FORMAT, v.clone()));
        }
        if let Some(v) = info.size_bytes {
            attributes.push(u64_attr(attr::MODEL_SIZE_BYTES, v));
        }
        let span = observer.start_span(
            span_name(op::LOAD_MODEL, Some(&info.model)),
            SpanKind::Internal,
            attributes,
            &mut options,
            start,
        );
        Self {
            simple: Simple::new(observer, clock, start, span, dims),
        }
    }

    /// Records whether the model came from a local cache.
    pub fn record_cache_hit(&self, hit: bool) {
        self.simple
            .core
            .set_attribute(KeyValue::new(attr::MODEL_CACHE_HIT, hit));
    }

    /// Records a host-measured resource sample (non-finite values ignored).
    pub fn record_resource(&self, sample: &ResourceSample) {
        self.simple.core.set_attributes(resource_attributes(sample));
    }
}

lifecycle!(
    ModelLoadOperation,
    |i: &Instruments, d, dims: &[KeyValue]| {
        i.operation_duration.record(d, dims);
    }
);

/// A generic internal span for runtime-internal work (pre-processing,
/// decoding, routing decisions). Carries no GenAI semantics and records no
/// metrics, so GenAI-aware backends never count it as a model call.
pub struct SpanOperation {
    core: Core,
    ended: Mutex<bool>,
}

impl SpanOperation {
    pub(crate) fn start(
        observer: Arc<ObserverInner>,
        name: String,
        attributes: Vec<KeyValue>,
        mut options: StartOptions,
    ) -> Self {
        let clock = observer.clock.anchor();
        let start = options.start_time.unwrap_or_else(|| clock.now());
        let span = observer.start_span(name, SpanKind::Internal, attributes, &mut options, start);
        Self {
            core: Core {
                observer,
                span,
                clock,
                start,
            },
            ended: Mutex::new(false),
        }
    }

    /// Context for child operations.
    #[must_use]
    pub fn context(&self) -> Option<ObservationContext> {
        self.core.context()
    }

    /// Reference to this span (sampled or not).
    #[must_use]
    pub fn observation_ref(&self) -> Option<ObservationRef> {
        self.core.observation_ref()
    }

    /// Sets a span attribute.
    pub fn set_attribute(&self, kv: KeyValue) {
        self.core.set_attribute(kv);
    }

    /// Ends successfully now.
    pub fn end(&self) {
        self.end_at(self.core.now());
    }

    /// Ends successfully at `at`.
    pub fn end_at(&self, at: SystemTime) {
        self.finish(at, None);
    }

    /// Ends with an error now.
    pub fn end_with_error(&self, error: &ErrorInfo) {
        self.finish(self.core.now(), Some(error));
    }

    /// Ends with an error at `at`.
    pub fn end_with_error_at(&self, error: &ErrorInfo, at: SystemTime) {
        self.finish(at, Some(error));
    }

    /// Ends as cancelled now.
    pub fn cancel(&self) {
        self.end_with_error(&ErrorInfo::new("cancelled"));
    }

    fn finish(&self, at: SystemTime, error: Option<&ErrorInfo>) {
        {
            let mut ended = lock(&self.ended);
            if *ended {
                return;
            }
            *ended = true;
        }
        if let Some(e) = error {
            self.core.fail(e);
        }
        self.core.end_span(at);
    }
}

impl Drop for SpanOperation {
    fn drop(&mut self) {
        if !*lock(&self.ended) {
            self.cancel();
        }
    }
}

impl fmt::Debug for SpanOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpanOperation")
            .field("observation_ref", &self.observation_ref())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
