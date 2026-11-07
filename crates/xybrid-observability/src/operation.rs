//! Operation handles. Every handle is `Send + Sync`, records through `&self`,
//! never performs I/O, and ends at most once. Dropping an un-ended handle
//! ends it as cancelled.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::SystemTime;

use opentelemetry::logs::{AnyValue, Severity};
use opentelemetry::trace::{SpanKind, Status, TraceContextExt};
use opentelemetry::{Array, Context, Key, KeyValue, StringValue, Value as OtelValue};
use serde_json::Value;

use crate::clock::seconds_between;
use crate::content::{bound_chars, InputMessage, OutputMessage, Part};
use crate::context::{ObservationContext, ObservationRef};
use crate::observer::{json_to_any, push_any, LogEvent, ObserverInner};
use crate::semconv::{attr, event, op, MAX_ERROR_MESSAGE_CHARS};
use crate::types::{
    AttemptInfo, ErrorInfo, InferenceRequest, Location, ModelLoadInfo, ResourceSample,
    StartOptions, ToolCallInfo, Usage,
};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn kind_for(location: Option<Location>) -> SpanKind {
    match location {
        Some(Location::Remote) => SpanKind::Client,
        _ => SpanKind::Internal,
    }
}

fn span_name(operation: &str, subject: Option<&str>) -> String {
    match subject {
        Some(s) if !s.is_empty() => format!("{operation} {s}"),
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
    if let Some(v) = sample.cpu_utilization {
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
    if let Some(v) = sample.battery_level {
        out.push(KeyValue::new(attr::RESOURCE_BATTERY_LEVEL, v));
    }
    if let Some(v) = sample.battery_charging {
        out.push(KeyValue::new(attr::RESOURCE_BATTERY_CHARGING, v));
    }
    if let Some(v) = sample.accelerator_utilization {
        out.push(KeyValue::new(attr::RESOURCE_ACCELERATOR_UTILIZATION, v));
    }
    if let Some(v) = sample.accelerator_memory_used_bytes {
        out.push(u64_attr(attr::RESOURCE_ACCELERATOR_MEMORY_USED_BYTES, v));
    }
    out
}

/// State shared by every handle type: the span (if any), timing, and the
/// end-once guard.
struct Core {
    observer: Arc<ObserverInner>,
    cx: Option<Context>,
    start: SystemTime,
}

impl Core {
    fn now(&self) -> SystemTime {
        self.observer.clock.now()
    }

    fn set_attributes(&self, attributes: Vec<KeyValue>) {
        if let Some(cx) = &self.cx {
            cx.span().set_attributes(attributes);
        }
    }

    fn set_attribute(&self, kv: KeyValue) {
        if let Some(cx) = &self.cx {
            cx.span().set_attribute(kv);
        }
    }

    fn context(&self) -> Option<ObservationContext> {
        self.cx.as_ref().and_then(ObservationContext::from_otel)
    }

    fn observation_ref(&self) -> Option<ObservationRef> {
        self.context().map(|c| c.observation_ref())
    }

    fn trace_ids(
        &self,
    ) -> Option<(
        opentelemetry::trace::TraceId,
        opentelemetry::trace::SpanId,
        Option<opentelemetry::trace::TraceFlags>,
    )> {
        let cx = self.cx.as_ref()?;
        let span = cx.span();
        let sc = span.span_context();
        sc.is_valid()
            .then(|| (sc.trace_id(), sc.span_id(), Some(sc.trace_flags())))
    }

    /// Marks the span failed and returns the bounded description.
    fn fail(&self, error: &ErrorInfo) -> String {
        let description = error
            .message
            .as_ref()
            .and_then(|m| {
                match self
                    .observer
                    .content
                    .redact("error.message", Value::String(m.clone()))
                {
                    Some(Value::String(s)) => Some(s),
                    Some(other) => Some(other.to_string()),
                    None => None,
                }
            })
            .map_or_else(
                || error.error_type.clone(),
                |m| bound_chars(&m, MAX_ERROR_MESSAGE_CHARS),
            );
        if let Some(cx) = &self.cx {
            let span = cx.span();
            span.set_attribute(KeyValue::new(attr::ERROR_TYPE, error.error_type.clone()));
            span.set_status(Status::error(description.clone()));
        }
        description
    }

    fn end_span(&self, at: SystemTime) {
        if let Some(cx) = &self.cx {
            cx.span().end_with_timestamp(at);
        }
    }

    /// Records a JSON content attribute on the span. Returns the bounded value
    /// (for events) and whether it was truncated.
    fn content_json(
        &self,
        key: &'static str,
        value: Value,
        on_span: bool,
    ) -> (Option<Value>, bool) {
        let prepared = self.observer.content.prepare_json(key, value);
        if on_span {
            if let Some(v) = &prepared.value {
                if let Ok(s) = serde_json::to_string(v) {
                    self.set_attribute(KeyValue::new(key, s));
                }
            }
        }
        (prepared.value, prepared.truncated)
    }
}

/// Inference state guarded by a mutex.
#[derive(Default)]
struct InferenceState {
    ended: bool,
    first_chunk: Option<SystemTime>,
    last_chunk: Option<SystemTime>,
    chunk_count: u64,
    first_token: Option<SystemTime>,
    usage: Usage,
    finish_reasons: Option<Vec<String>>,
    output_recorded: bool,
    response_model: Option<String>,
    response_id: Option<String>,
    truncated: Vec<String>,
    attempt_count: u32,
    event_content: Vec<(String, Value)>,
}

impl InferenceState {
    fn mark_truncated(&mut self, key: &str) {
        if !self.truncated.iter().any(|k| k == key) {
            self.truncated.push(key.to_owned());
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Category {
    Inference,
    Embeddings,
    Custom,
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
        let start = options.start_time.unwrap_or_else(|| observer.clock.now());
        let category = match request.operation.as_str() {
            op::EMBEDDINGS => Category::Embeddings,
            op::PREDICT => Category::Custom,
            _ => Category::Inference,
        };
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
        let mut state = InferenceState::default();
        push_request_attributes(&mut attributes, &request);

        let capture = observer.content.capture();
        if let Some(prompt) = &request.prompt {
            if capture.any() {
                for (name, value) in &prompt.variables {
                    let key = format!("{}{name}", attr::PROMPT_VARIABLE_PREFIX);
                    let prepared = observer.content.prepare_text(&key, value.clone());
                    if prepared.truncated {
                        state.mark_truncated(&key);
                    }
                    if let Some(v) = prepared.value {
                        if capture.on_spans() {
                            attributes.push(KeyValue::new(key.clone(), v.clone()));
                        }
                        if capture.on_events() {
                            state.event_content.push((key, Value::String(v)));
                        }
                    }
                }
            }
        }

        let name = span_name(&request.operation, request.model.as_deref());
        let cx = observer.start_span(
            name,
            kind_for(request.location),
            attributes,
            &mut options,
            start,
        );
        Self {
            core: Core {
                observer,
                cx,
                start,
            },
            request,
            category,
            dims,
            state: Mutex::new(state),
        }
    }

    /// The context for child operations, `None` when no span was recorded.
    #[must_use]
    pub fn context(&self) -> Option<ObservationContext> {
        self.core.context()
    }

    /// Reference to this operation's span, `None` when no span was recorded.
    #[must_use]
    pub fn observation_ref(&self) -> Option<ObservationRef> {
        self.core.observation_ref()
    }

    /// The request this operation was started with.
    #[must_use]
    pub fn request(&self) -> &InferenceRequest {
        &self.request
    }

    fn record_content(&self, key: &'static str, value: Value) {
        let capture = self.core.observer.content.capture();
        if !capture.any() {
            return;
        }
        let (bounded, truncated) = self.core.content_json(key, value, capture.on_spans());
        let mut state = lock(&self.state);
        if truncated {
            state.mark_truncated(key);
        }
        if capture.on_events() {
            if let Some(v) = bounded {
                state.event_content.retain(|(k, _)| k != key);
                state.event_content.push((key.to_owned(), v));
            }
        }
    }

    /// Records the system instructions (content).
    pub fn record_system_instructions(&self, parts: &[Part]) {
        if let Ok(v) = serde_json::to_value(parts) {
            self.record_content(attr::SYSTEM_INSTRUCTIONS, v);
        }
    }

    /// Records the input messages (content).
    pub fn record_input_messages(&self, messages: &[InputMessage]) {
        if let Ok(v) = serde_json::to_value(messages) {
            self.record_content(attr::INPUT_MESSAGES, v);
        }
    }

    /// Records the output messages (content).
    pub fn record_output_messages(&self, messages: &[OutputMessage]) {
        lock(&self.state).output_recorded = true;
        if let Ok(v) = serde_json::to_value(messages) {
            self.record_content(attr::OUTPUT_MESSAGES, v);
        }
    }

    /// Records the tool definitions offered to the model. Schema content:
    /// recorded in every capture mode, still redacted and bounded.
    pub fn record_tool_definitions(&self, definitions: &Value) {
        let capture = self.core.observer.content.capture();
        let (bounded, truncated) =
            self.core
                .content_json(attr::TOOL_DEFINITIONS, definitions.clone(), true);
        let mut state = lock(&self.state);
        if truncated {
            state.mark_truncated(attr::TOOL_DEFINITIONS);
        }
        if capture.on_events() {
            if let Some(v) = bounded {
                state
                    .event_content
                    .retain(|(k, _)| k != attr::TOOL_DEFINITIONS);
                state
                    .event_content
                    .push((attr::TOOL_DEFINITIONS.to_owned(), v));
            }
        }
    }

    /// Marks one decoded output chunk now.
    pub fn record_chunk(&self) {
        self.record_chunk_at(self.core.now());
    }

    /// Marks one decoded output chunk at `at`.
    pub fn record_chunk_at(&self, at: SystemTime) {
        let mut state = lock(&self.state);
        if state.ended {
            return;
        }
        state.chunk_count += 1;
        let instruments = self.core.observer.instruments.as_ref();
        match state.last_chunk {
            None => {
                state.first_chunk = Some(at);
                let ttfc = seconds_between(self.core.start, at);
                self.core
                    .set_attribute(KeyValue::new(attr::RESPONSE_TIME_TO_FIRST_CHUNK, ttfc));
                if let Some(i) = instruments {
                    i.time_to_first_chunk.record(ttfc, &self.dims);
                }
            }
            Some(previous) => {
                if let Some(i) = instruments {
                    i.time_per_output_chunk
                        .record(seconds_between(previous, at), &self.dims);
                }
            }
        }
        state.last_chunk = Some(at);
    }

    /// Marks the first output token now. Later calls are ignored.
    pub fn record_first_token(&self) {
        self.record_first_token_at(self.core.now());
    }

    /// Marks the first output token at `at`. Later calls are ignored.
    pub fn record_first_token_at(&self, at: SystemTime) {
        let mut state = lock(&self.state);
        if state.ended || state.first_token.is_some() {
            return;
        }
        state.first_token = Some(at);
        let ttft = seconds_between(self.core.start, at);
        self.core
            .set_attribute(KeyValue::new(attr::RESPONSE_TIME_TO_FIRST_TOKEN, ttft));
        if let Some(i) = self.core.observer.instruments.as_ref() {
            i.time_to_first_token.record(ttft, &self.dims);
        }
    }

    /// Records the runtime's measured mean time per output token, seconds.
    pub fn record_token_timing(&self, time_per_output_token_s: f64) {
        self.core.set_attribute(KeyValue::new(
            attr::RESPONSE_TIME_PER_OUTPUT_TOKEN,
            time_per_output_token_s,
        ));
    }

    /// Records request usage totals (merged field by field).
    pub fn record_usage(&self, usage: &Usage) {
        lock(&self.state).usage.merge(usage);
    }

    /// Records finish reasons, one per choice.
    pub fn record_finish_reasons<I, S>(&self, reasons: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        lock(&self.state).finish_reasons = Some(reasons.into_iter().map(Into::into).collect());
    }

    /// Records the response id and/or model.
    pub fn record_response(&self, response_id: Option<&str>, model: Option<&str>) {
        let mut state = lock(&self.state);
        if let Some(id) = response_id {
            state.response_id = Some(id.to_owned());
        }
        if let Some(m) = model {
            state.response_model = Some(m.to_owned());
        }
    }

    /// Records a host-measured resource sample.
    pub fn record_resource(&self, sample: &ResourceSample) {
        self.core.set_attributes(resource_attributes(sample));
    }

    /// Sets an extra span attribute (adapter extensions).
    pub fn set_attribute(&self, kv: KeyValue) {
        self.core.set_attribute(kv);
    }

    /// Starts a concrete execution attempt as a child span.
    #[must_use]
    pub fn start_attempt(&self, attempt: AttemptInfo) -> AttemptOperation {
        self.start_attempt_at(attempt, self.core.now())
    }

    /// Starts a concrete execution attempt at `at`.
    #[must_use]
    pub fn start_attempt_at(&self, attempt: AttemptInfo, at: SystemTime) -> AttemptOperation {
        lock(&self.state).attempt_count += 1;
        AttemptOperation::start(
            self.core.observer.clone(),
            self.request.operation.clone(),
            self.context(),
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
        if !state.truncated.is_empty() {
            end_attrs.push(KeyValue::new(
                attr::CONTENT_TRUNCATED_ATTRIBUTES,
                string_array(state.truncated.iter().cloned()),
            ));
        }
        core.set_attributes(end_attrs);

        let description = error.map(|e| core.fail(e));
        self.record_metrics(&state, at, error);
        self.emit_events(&mut state, at, error, description.as_deref());
        core.end_span(at);
    }

    fn record_metrics(&self, state: &InferenceState, at: SystemTime, error: Option<&ErrorInfo>) {
        let Some(instruments) = self.core.observer.instruments.as_ref() else {
            return;
        };
        let mut base = self.dims.clone();
        if let Some(model) = &state.response_model {
            base.push(KeyValue::new(attr::RESPONSE_MODEL, model.clone()));
        }
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
        description: Option<&str>,
    ) {
        let Some(events) = self.core.observer.events.as_ref() else {
            return;
        };
        let trace = self.core.trace_ids();
        if let (Some(e), Some(description)) =
            (error.filter(|e| e.error_type != "cancelled"), description)
        {
            let mut attributes = vec![(
                Key::from_static_str(attr::EXCEPTION_TYPE),
                AnyValue::from(e.error_type.clone()),
            )];
            if e.message.is_some() {
                push_any(
                    &mut attributes,
                    attr::EXCEPTION_MESSAGE,
                    Some(AnyValue::from(description.to_owned())),
                );
            }
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
        for (key, value) in std::mem::take(&mut state.event_content) {
            attributes.push((Key::new(key), json_to_any(value)));
        }
        if !state.truncated.is_empty() {
            attributes.push((
                Key::from_static_str(attr::CONTENT_TRUNCATED_ATTRIBUTES),
                AnyValue::ListAny(Box::new(
                    state
                        .truncated
                        .iter()
                        .cloned()
                        .map(AnyValue::from)
                        .collect(),
                )),
            ));
        }
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
    if let Some(v) = request.temperature {
        out.push(KeyValue::new(attr::REQUEST_TEMPERATURE, v));
    }
    if let Some(v) = request.top_p {
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
    if let Some(v) = request.frequency_penalty {
        out.push(KeyValue::new(attr::REQUEST_FREQUENCY_PENALTY, v));
    }
    if let Some(v) = request.presence_penalty {
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
        if !lock(&self.state).ended {
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
        // An attempt without a recorded parent span stays untraced so it
        // never becomes a stray root.
        let mut options = StartOptions {
            parent: parent.clone(),
            trace: if parent.is_some() {
                crate::types::TraceMode::Default
            } else {
                crate::types::TraceMode::Suppress
            },
            start_time: Some(start),
            attributes: Vec::new(),
        };
        let cx = observer.start_span(
            format!("attempt {}", attempt.index),
            kind_for(Some(attempt.location)),
            attributes,
            &mut options,
            start,
        );
        Self {
            core: Core {
                observer,
                cx,
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

    /// Reference to this attempt's span.
    #[must_use]
    pub fn observation_ref(&self) -> Option<ObservationRef> {
        self.core.observation_ref()
    }

    /// Records this attempt's own usage.
    pub fn record_usage(&self, usage: &Usage) {
        lock(&self.state).1.merge(usage);
    }

    /// Records a resource sample.
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

    /// Ends the attempt as cancelled.
    pub fn cancel(&self) {
        self.end_with_error(&ErrorInfo::new("cancelled"));
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

/// A simple timed operation (workflow, tool, model load).
struct Simple {
    core: Core,
    dims: Vec<KeyValue>,
    ended: Mutex<bool>,
    truncated: Mutex<Vec<String>>,
    event_content: Mutex<BTreeMap<&'static str, Value>>,
}

impl Simple {
    fn finish(
        &self,
        at: SystemTime,
        error: Option<&ErrorInfo>,
        record: impl FnOnce(&crate::observer::Instruments, f64, &[KeyValue]),
    ) {
        {
            let mut ended = lock(&self.ended);
            if *ended {
                return;
            }
            *ended = true;
        }
        let truncated = lock(&self.truncated).clone();
        if !truncated.is_empty() {
            self.core.set_attribute(KeyValue::new(
                attr::CONTENT_TRUNCATED_ATTRIBUTES,
                string_array(truncated.clone()),
            ));
        }
        if let Some(e) = error {
            self.core.fail(e);
        }
        if let Some(i) = self.core.observer.instruments.as_ref() {
            let mut dims = self.dims.clone();
            if let Some(e) = error {
                dims.push(KeyValue::new(attr::ERROR_TYPE, e.error_type.clone()));
            }
            record(i, seconds_between(self.core.start, at), &dims);
        }
        let content = std::mem::take(&mut *lock(&self.event_content));
        if let Some(events) = self
            .core
            .observer
            .events
            .as_ref()
            .filter(|_| !content.is_empty())
        {
            let mut attributes: Vec<(Key, AnyValue)> = self
                .dims
                .iter()
                .map(|kv| {
                    (
                        kv.key.clone(),
                        json_to_any(serde_json::Value::String(kv.value.to_string())),
                    )
                })
                .collect();
            attributes.extend(
                content
                    .into_iter()
                    .map(|(key, value)| (Key::from_static_str(key), json_to_any(value))),
            );
            if let Some(e) = error {
                attributes.push((
                    Key::from_static_str(attr::ERROR_TYPE),
                    AnyValue::from(e.error_type.clone()),
                ));
            }
            if !truncated.is_empty() {
                attributes.push((
                    Key::from_static_str(attr::CONTENT_TRUNCATED_ATTRIBUTES),
                    AnyValue::ListAny(Box::new(
                        truncated.into_iter().map(AnyValue::from).collect(),
                    )),
                ));
            }
            events.emit(LogEvent {
                name: event::TOOL_DETAILS,
                severity: Severity::Debug,
                severity_text: "DEBUG",
                timestamp: at,
                trace: self.core.trace_ids(),
                attributes,
            });
        }
        self.core.end_span(at);
    }

    fn is_ended(&self) -> bool {
        *lock(&self.ended)
    }
}

macro_rules! lifecycle {
    ($ty:ident, $record:expr) => {
        impl $ty {
            /// Context for child operations.
            #[must_use]
            pub fn context(&self) -> Option<ObservationContext> {
                self.simple.core.context()
            }

            /// Reference to this operation's span.
            #[must_use]
            pub fn observation_ref(&self) -> Option<ObservationRef> {
                self.simple.core.observation_ref()
            }

            /// Sets an extra span attribute.
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
        let start = options.start_time.unwrap_or_else(|| observer.clock.now());
        let dims = vec![
            KeyValue::new(attr::OPERATION_NAME, op::INVOKE_WORKFLOW),
            KeyValue::new(attr::WORKFLOW_NAME, name.clone()),
        ];
        let cx = observer.start_span(
            span_name(op::INVOKE_WORKFLOW, Some(&name)),
            SpanKind::Internal,
            dims.clone(),
            &mut options,
            start,
        );
        Self {
            simple: Simple {
                core: Core {
                    observer,
                    cx,
                    start,
                },
                dims,
                ended: Mutex::new(false),
                truncated: Mutex::new(Vec::new()),
                event_content: Mutex::new(BTreeMap::new()),
            },
        }
    }
}

lifecycle!(
    WorkflowOperation,
    |i: &crate::observer::Instruments, d, dims: &[KeyValue]| {
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
        let start = options.start_time.unwrap_or_else(|| observer.clock.now());
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
        let cx = observer.start_span(
            span_name(op::EXECUTE_TOOL, Some(&call.name)),
            SpanKind::Internal,
            attributes,
            &mut options,
            start,
        );
        let operation = Self {
            simple: Simple {
                core: Core {
                    observer,
                    cx,
                    start,
                },
                dims,
                ended: Mutex::new(false),
                truncated: Mutex::new(Vec::new()),
                event_content: Mutex::new(BTreeMap::new()),
            },
        };
        if let Some(args) = call.arguments {
            operation.record_content(attr::TOOL_CALL_ARGUMENTS, args);
        }
        operation
    }

    /// Records the tool result (content).
    pub fn record_result(&self, result: &Value) {
        self.record_content(attr::TOOL_CALL_RESULT, result.clone());
    }

    fn record_content(&self, key: &'static str, value: Value) {
        // Serialize recording with finish so no content can arrive after the event.
        let ended = lock(&self.simple.ended);
        if *ended {
            return;
        }
        let core = &self.simple.core;
        let capture = core.observer.content.capture();
        if !capture.any() {
            return;
        }
        let (bounded, truncated) = core.content_json(key, value, capture.on_spans());
        if capture.on_events() {
            if let Some(value) = bounded {
                lock(&self.simple.event_content).insert(key, value);
            }
        }
        if truncated {
            lock(&self.simple.truncated).push(key.to_owned());
        }
    }
}

lifecycle!(
    ToolOperation,
    |i: &crate::observer::Instruments, d, dims: &[KeyValue]| {
        i.tool_duration.record(d, dims);
    }
);

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
        let start = options.start_time.unwrap_or_else(|| observer.clock.now());
        let provider = info
            .provider
            .clone()
            .or_else(|| info.runtime.clone())
            .unwrap_or_else(|| "unknown".to_owned());
        let dims = vec![
            KeyValue::new(attr::OPERATION_NAME, op::LOAD_MODEL),
            KeyValue::new(attr::PROVIDER_NAME, provider),
            KeyValue::new(attr::REQUEST_MODEL, info.model.clone()),
        ];
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
        let cx = observer.start_span(
            span_name(op::LOAD_MODEL, Some(&info.model)),
            SpanKind::Internal,
            attributes,
            &mut options,
            start,
        );
        Self {
            simple: Simple {
                core: Core {
                    observer,
                    cx,
                    start,
                },
                dims,
                ended: Mutex::new(false),
                truncated: Mutex::new(Vec::new()),
                event_content: Mutex::new(BTreeMap::new()),
            },
        }
    }

    /// Records whether the model came from a local cache.
    pub fn record_cache_hit(&self, hit: bool) {
        self.simple
            .core
            .set_attribute(KeyValue::new(attr::MODEL_CACHE_HIT, hit));
    }

    /// Records a host-measured resource sample.
    pub fn record_resource(&self, sample: &ResourceSample) {
        self.simple.core.set_attributes(resource_attributes(sample));
    }
}

lifecycle!(
    ModelLoadOperation,
    |i: &crate::observer::Instruments, d, dims: &[KeyValue]| {
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
        let start = options.start_time.unwrap_or_else(|| observer.clock.now());
        let cx = observer.start_span(name, SpanKind::Internal, attributes, &mut options, start);
        Self {
            core: Core {
                observer,
                cx,
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

    /// Reference to this span.
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
            self.end_with_error(&ErrorInfo::new("cancelled"));
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
mod tests {
    use super::*;
    use crate::Observer;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};

    #[test]
    fn dropped_generic_spans_are_cancelled_but_explicit_completion_is_preserved() {
        let spans = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(spans.clone())
            .build();
        let observer = Observer::builder().tracer_provider(&provider).build();
        drop(observer.start_span("interrupted", vec![], StartOptions::default()));
        let completed = observer.start_span("completed", vec![], StartOptions::default());
        completed.end();
        drop(completed);
        let failed = observer.start_span("failed", vec![], StartOptions::default());
        failed.end_with_error(&ErrorInfo::new("network"));
        drop(failed);
        let finished = spans.get_finished_spans().unwrap();
        assert_eq!(finished.len(), 3);
        assert!(matches!(finished[0].status, Status::Error { .. }));
        assert!(finished[0]
            .attributes
            .contains(&KeyValue::new(attr::ERROR_TYPE, "cancelled")));
        assert_eq!(finished[1].status, Status::Unset);
        assert!(finished[2]
            .attributes
            .contains(&KeyValue::new(attr::ERROR_TYPE, "network")));
    }
}
