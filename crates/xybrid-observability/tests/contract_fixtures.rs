//! Runs every shared contract fixture (`contract/fixtures/*.json`, format v2,
//! CONTRACT.md "Fixture format") and checks the normalized spans, metrics, and
//! log events against its `expect` block, plus every inline assertion.
//!
//! `XYBRID_FIXTURE_DUMP=<dir>` writes the normalized output of each fixture to
//! `<dir>/<name>.json`; `XYBRID_FIXTURE_ONLY=<name>` runs one fixture. Fixture
//! expectations are owned by the contract: never re-bless them from this
//! runner (file an amendment instead).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use opentelemetry::trace::noop::NoopTracerProvider;
use opentelemetry::trace::{SpanKind, Status};
use opentelemetry::{KeyValue, Value as OtelValue};
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use serde_json::{json, Map, Value};
use xybrid_observability::semconv::{self, attr, metric};
use xybrid_observability::wire::{self, CatalogType};
use xybrid_observability::{
    AttemptInfo, AttemptOperation, AttemptReason, Clock, ContentCapture, ContentPolicy, ErrorInfo,
    Evaluation, Feedback, InferenceOperation, InferenceRequest, InputMessage, Location,
    ManualClock, ModelLoadInfo, ModelLoadOperation, ObservationContext, ObservationRef, Observer,
    OutputMessage, Part, PromptRef, RedactionError, RedactionRule, Redactor, ResourceSample,
    StartOptions, SubstringRedactor, ToolCallInfo, ToolOperation, TraceMode, Usage,
    WorkflowOperation,
};

const BASE_MS: u64 = 1_700_000_000_000;

fn contract_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("contract")
}

fn catalog() -> Value {
    serde_json::from_str(&std::fs::read_to_string(contract_dir().join("catalog.json")).unwrap())
        .unwrap()
}

/// Parses JSON without serde_json's 128-level recursion limit (fixtures nest
/// content up to the contract's depth guard inside their `expect` blocks).
/// Structure is parsed here; strings and numbers are delegated to serde_json.
fn parse_unbounded(text: &str) -> Value {
    fn ws(b: &[u8], mut i: usize) -> usize {
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        i
    }
    fn string_end(b: &[u8], mut i: usize) -> usize {
        // `i` is at the opening quote; returns the index after the closing one.
        i += 1;
        while b[i] != b'"' {
            i += if b[i] == b'\\' { 2 } else { 1 };
        }
        i + 1
    }
    fn value(t: &str, i: usize) -> (Value, usize) {
        let b = t.as_bytes();
        let i = ws(b, i);
        match b[i] {
            b'[' => {
                let mut items = Vec::new();
                let mut j = ws(b, i + 1);
                if b[j] == b']' {
                    return (Value::Array(items), j + 1);
                }
                loop {
                    let (v, next) = value(t, j);
                    items.push(v);
                    j = ws(b, next);
                    match b[j] {
                        b',' => j += 1,
                        b']' => return (Value::Array(items), j + 1),
                        c => panic!("unexpected {:?} at {j}", c as char),
                    }
                }
            }
            b'{' => {
                let mut map = Map::new();
                let mut j = ws(b, i + 1);
                if b[j] == b'}' {
                    return (Value::Object(map), j + 1);
                }
                loop {
                    let end = string_end(b, j);
                    let key: String = serde_json::from_str(&t[j..end]).unwrap();
                    j = ws(b, end);
                    assert_eq!(b[j], b':', "expected ':' at {j}");
                    let (v, next) = value(t, j + 1);
                    map.insert(key, v);
                    j = ws(b, next);
                    match b[j] {
                        b',' => j = ws(b, j + 1),
                        b'}' => return (Value::Object(map), j + 1),
                        c => panic!("unexpected {:?} at {j}", c as char),
                    }
                }
            }
            b'"' => {
                let end = string_end(b, i);
                (serde_json::from_str(&t[i..end]).unwrap(), end)
            }
            _ => {
                let mut end = i;
                while end < b.len()
                    && !matches!(b[end], b',' | b']' | b'}' | b' ' | b'\t' | b'\n' | b'\r')
                {
                    end += 1;
                }
                (serde_json::from_str(&t[i..end]).unwrap(), end)
            }
        }
    }
    let (v, end) = value(text, 0);
    assert_eq!(ws(text.as_bytes(), end), text.len(), "trailing data");
    v
}

fn base_time() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_millis(BASE_MS)
}

fn at(ms: u64) -> SystemTime {
    base_time() + Duration::from_millis(ms)
}

enum Handle {
    Inference(Box<InferenceOperation>),
    Attempt(AttemptOperation),
    Workflow(WorkflowOperation),
    Tool(ToolOperation),
    ModelLoad(ModelLoadOperation),
}

macro_rules! each {
    ($self:ident, $h:ident => $e:expr) => {
        match $self {
            Handle::Inference($h) => $e,
            Handle::Attempt($h) => $e,
            Handle::Workflow($h) => $e,
            Handle::Tool($h) => $e,
            Handle::ModelLoad($h) => $e,
        }
    };
}

impl Handle {
    fn context(&self) -> Option<ObservationContext> {
        each!(self, h => h.context())
    }

    fn observation_ref(&self) -> Option<ObservationRef> {
        each!(self, h => h.observation_ref())
    }

    fn end(&self, time: Option<SystemTime>) {
        match time {
            Some(t) => each!(self, h => h.end_at(t)),
            None => each!(self, h => h.end()),
        }
    }

    fn end_with_error(&self, e: &ErrorInfo, time: Option<SystemTime>) {
        match time {
            Some(t) => each!(self, h => h.end_with_error_at(e, t)),
            None => each!(self, h => h.end_with_error(e)),
        }
    }

    fn cancel(&self, time: Option<SystemTime>) {
        match time {
            Some(t) => each!(self, h => h.cancel_at(t)),
            None => each!(self, h => h.cancel()),
        }
    }

    fn inference(&self) -> &InferenceOperation {
        match self {
            Self::Inference(h) => h,
            _ => panic!("step requires an inference handle"),
        }
    }
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_owned)
}

fn i(v: &Value, k: &str) -> Option<i64> {
    v.get(k).and_then(Value::as_i64)
}

fn u(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(Value::as_u64)
}

fn f(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

fn b(v: &Value, k: &str) -> Option<bool> {
    v.get(k).and_then(Value::as_bool)
}

fn strings(v: &Value, k: &str) -> Vec<String> {
    v.get(k)
        .map(|m| serde_json::from_value(m.clone()).expect(k))
        .unwrap_or_default()
}

fn modalities(v: &Value, k: &str) -> Vec<xybrid_observability::Modality> {
    v.get(k)
        .map(|m| serde_json::from_value(m.clone()).expect("modalities"))
        .unwrap_or_default()
}

fn location(v: &Value, k: &str) -> Option<Location> {
    v.get(k)
        .map(|m| serde_json::from_value(m.clone()).expect("location"))
}

const REQUEST_FIELDS: &[&str] = &[
    "operation",
    "provider",
    "model",
    "location",
    "runtime",
    "runtime_version",
    "accelerator",
    "task",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "stop_sequences",
    "frequency_penalty",
    "presence_penalty",
    "output_type",
    "conversation_id",
    "prompt_name",
    "prompt_version",
    "prompt_variables",
    "input_modalities",
    "output_modalities",
    "server_address",
    "server_port",
    "embedding_dimensions",
];

fn inference_request(r: &Value) -> InferenceRequest {
    check_fields("request", r, REQUEST_FIELDS);
    InferenceRequest {
        operation: s(r, "operation").expect("operation"),
        provider: s(r, "provider").expect("provider"),
        model: s(r, "model"),
        location: location(r, "location"),
        runtime: s(r, "runtime"),
        runtime_version: s(r, "runtime_version"),
        accelerator: s(r, "accelerator"),
        task: s(r, "task"),
        stream: b(r, "stream").unwrap_or(false),
        max_tokens: i(r, "max_tokens"),
        temperature: f(r, "temperature"),
        top_p: f(r, "top_p"),
        top_k: i(r, "top_k"),
        seed: i(r, "seed"),
        stop_sequences: strings(r, "stop_sequences"),
        frequency_penalty: f(r, "frequency_penalty"),
        presence_penalty: f(r, "presence_penalty"),
        output_type: s(r, "output_type"),
        conversation_id: s(r, "conversation_id"),
        prompt: s(r, "prompt_name").map(|name| PromptRef {
            name,
            version: s(r, "prompt_version"),
            variables: r
                .get("prompt_variables")
                .map(|v| serde_json::from_value(v.clone()).expect("prompt_variables"))
                .unwrap_or_default(),
        }),
        input_modalities: modalities(r, "input_modalities"),
        output_modalities: modalities(r, "output_modalities"),
        server_address: s(r, "server_address"),
        server_port: i(r, "server_port"),
        embedding_dimensions: i(r, "embedding_dimensions"),
    }
}

const USAGE_FIELDS: &[&str] = &[
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
    "text_input_tokens",
    "image_input_tokens",
    "audio_input_tokens",
    "text_output_tokens",
    "image_output_tokens",
    "audio_output_tokens",
];

fn usage(v: &Value) -> Usage {
    Usage {
        input_tokens: u(v, "input_tokens"),
        output_tokens: u(v, "output_tokens"),
        cache_read_input_tokens: u(v, "cache_read_input_tokens"),
        cache_write_input_tokens: u(v, "cache_write_input_tokens"),
        reasoning_output_tokens: u(v, "reasoning_output_tokens"),
        text_input_tokens: u(v, "text_input_tokens"),
        image_input_tokens: u(v, "image_input_tokens"),
        audio_input_tokens: u(v, "audio_input_tokens"),
        text_output_tokens: u(v, "text_output_tokens"),
        image_output_tokens: u(v, "image_output_tokens"),
        audio_output_tokens: u(v, "audio_output_tokens"),
    }
}

const RESOURCE_FIELDS: &[&str] = &[
    "cpu_utilization",
    "cpu_peak_utilization",
    "memory_used_bytes",
    "memory_min_available_bytes",
    "memory_pressure",
    "process_rss_bytes",
    "process_peak_rss_bytes",
    "thermal_state",
    "battery_level",
    "battery_charging",
    "accelerator_utilization",
    "accelerator_memory_used_bytes",
];

const PARENT_FIELDS: &[&str] = &[
    "parent",
    "parent_form",
    "parent_traceparent",
    "parent_tracestate",
    "trace",
    "time_ms",
    "expect_error",
];

/// Allowed fields per op (beyond `at_ms` and `op`).
fn op_fields(op: &str) -> Vec<&'static str> {
    let with_parent = |extra: &[&'static str]| {
        let mut v = vec!["id"];
        v.extend_from_slice(PARENT_FIELDS);
        v.extend_from_slice(extra);
        v
    };
    match op {
        "start_inference" => with_parent(&["request"]),
        "start_workflow" => with_parent(&["name"]),
        "start_tool" => with_parent(&["name", "call_id", "tool_type", "description", "arguments"]),
        "start_model_load" => {
            with_parent(&["model", "provider", "runtime", "format", "size_bytes"])
        }
        "start_attempt" => vec![
            "id",
            "parent",
            "index",
            "reason",
            "provider",
            "model",
            "location",
            "runtime",
            "server_address",
            "server_port",
            "time_ms",
        ],
        "input_messages"
        | "output_messages"
        | "system_instructions"
        | "tool_definitions"
        | "tool_result"
        | "finish_reasons"
        | "model_cache_hit" => vec!["id", "value"],
        "chunk" | "first_token" | "end" | "cancel" => vec!["id", "time_ms"],
        "token_timing" => vec!["id", "time_per_output_token_s"],
        "usage" => [&["id"][..], USAGE_FIELDS].concat(),
        "resource" => [&["id"][..], RESOURCE_FIELDS].concat(),
        "response" => vec!["id", "response_id", "model"],
        "end_with_error" => vec!["id", "error_type", "message", "time_ms"],
        "evaluation" => vec![
            "target",
            "target_ref",
            "name",
            "score_value",
            "score_label",
            "explanation",
            "response_id",
            "expect_result",
        ],
        "feedback" => vec![
            "target",
            "target_ref",
            "score",
            "label",
            "comment",
            "expect_result",
        ],
        "assert_ref" => vec!["target", "present", "trace_id"],
        "assert_context" => vec![
            "target",
            "present",
            "sampled",
            "trace_id",
            "traceparent",
            "tracestate",
        ],
        "parse_context" => vec!["traceparent", "tracestate", "expect"],
        other => panic!("unknown step op {other}"),
    }
}

fn check_fields(what: &str, v: &Value, allowed: &[&str]) {
    for key in v
        .as_object()
        .unwrap_or_else(|| panic!("{what} is an object"))
        .keys()
    {
        assert!(
            allowed.contains(&key.as_str()),
            "unknown {what} field {key}"
        );
    }
}

/// Expands the `$nested_array` and `$repeat_string` value macros.
fn expand(v: &Value) -> Value {
    match v {
        Value::Object(map) if map.len() == 1 && map.contains_key("$nested_array") => {
            let spec = &map["$nested_array"];
            let mut out = expand(&spec["leaf"]);
            for _ in 0..spec["depth"].as_u64().expect("depth") {
                out = Value::Array(vec![out]);
            }
            out
        }
        Value::Object(map) if map.len() == 1 && map.contains_key("$repeat_string") => {
            let spec = &map["$repeat_string"];
            let count = usize::try_from(spec["count"].as_u64().expect("count")).unwrap();
            Value::String(spec["value"].as_str().expect("value").repeat(count))
        }
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), expand(v))).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(expand).collect()),
        other => other.clone(),
    }
}

fn nested(depth: usize, leaf: Value) -> Value {
    let mut v = leaf;
    for _ in 0..depth {
        v = Value::Array(vec![v]);
    }
    v
}

/// The fixtures' reference callback (`redact_substrings` + `test_redactor`).
struct FixtureRedactor {
    substrings: SubstringRedactor,
    drop_kinds: Vec<String>,
    fail_kinds: Vec<String>,
    invalid_output_kinds: Vec<String>,
}

impl Redactor for FixtureRedactor {
    fn redact(&self, kind: &str, value: Value) -> Option<Value> {
        self.try_redact(kind, value).ok().flatten()
    }

    fn try_redact(&self, kind: &str, value: Value) -> Result<Option<Value>, RedactionError> {
        let has = |list: &[String]| list.iter().any(|k| k == kind);
        if has(&self.fail_kinds) {
            return Err(RedactionError);
        }
        if has(&self.drop_kinds) {
            return Ok(None);
        }
        if has(&self.invalid_output_kinds) {
            return Ok(Some(nested(129, json!("x"))));
        }
        Ok(self.substrings.redact(kind, value))
    }
}

const CONFIG_FIELDS: &[&str] = &[
    "content_capture",
    "max_content_attribute_bytes",
    "max_retained_operation_bytes",
    "max_traversal_bytes",
    "redact_substrings",
    "test_redactor",
    "redaction_rules",
    "logs",
    "tracer",
];

fn content_policy(config: &Value) -> ContentPolicy {
    let size = |k: &str, default: usize| {
        config
            .get(k)
            .map_or(default, |v| usize::try_from(v.as_u64().expect(k)).unwrap())
    };
    let mut policy = ContentPolicy::default()
        .with_capture(ContentCapture::parse(
            config
                .get("content_capture")
                .map_or("span_only", |v| v.as_str().expect("content_capture")),
        ))
        .with_max_attribute_bytes(size(
            "max_content_attribute_bytes",
            semconv::DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES,
        ))
        .with_max_retained_operation_bytes(size(
            "max_retained_operation_bytes",
            semconv::DEFAULT_MAX_RETAINED_OPERATION_BYTES,
        ))
        .with_max_traversal_bytes(size(
            "max_traversal_bytes",
            semconv::DEFAULT_MAX_TRAVERSAL_BYTES,
        ));
    let substrings = strings(config, "redact_substrings");
    let test = config.get("test_redactor");
    if let Some(t) = test {
        check_fields(
            "test_redactor",
            t,
            &["drop_kinds", "fail_kinds", "invalid_output_kinds"],
        );
    }
    if !substrings.is_empty() || test.is_some() {
        let list = |k: &str| test.map(|t| strings(t, k)).unwrap_or_default();
        policy = policy.with_redactor(FixtureRedactor {
            substrings: SubstringRedactor::new(substrings),
            drop_kinds: list("drop_kinds"),
            fail_kinds: list("fail_kinds"),
            invalid_output_kinds: list("invalid_output_kinds"),
        });
    }
    if let Some(rules) = config.get("redaction_rules") {
        let rules: Vec<RedactionRule> =
            serde_json::from_value(rules.clone()).expect("valid redaction_rules");
        policy = policy.with_rules(rules);
    }
    policy
}

struct Run {
    spans: InMemorySpanExporter,
    metrics: InMemoryMetricExporter,
    logs: InMemoryLogExporter,
    tracer_provider: SdkTracerProvider,
    meter_provider: SdkMeterProvider,
    logger_provider: SdkLoggerProvider,
    /// Span id -> handle id, from `observation_ref()` only.
    span_ids: HashMap<String, String>,
    /// Trace ids named by some step's `parent_traceparent`.
    remote_traces: HashSet<String>,
}

fn build_parent(
    step: &Value,
    handles: &HashMap<String, Handle>,
) -> Result<Option<ObservationContext>, xybrid_observability::Error> {
    if let Some(tp) = s(step, "parent_traceparent") {
        return ObservationContext::from_traceparent(&tp, s(step, "parent_tracestate").as_deref())
            .map(Some);
    }
    let Some(parent) = s(step, "parent") else {
        return Ok(None);
    };
    let Some(cx) = handles[&parent].context() else {
        return Ok(None);
    };
    match s(step, "parent_form").as_deref().unwrap_or("object") {
        "object" => Ok(Some(cx)),
        "cloned" => Ok(Some(cx.clone())),
        "serialized" => {
            let json = serde_json::to_value(&cx).unwrap();
            serde_json::from_value(json)
                .map(Some)
                .map_err(|_| xybrid_observability::Error::InvalidTraceparent(String::new()))
        }
        "traceparent" => {
            ObservationContext::from_traceparent(&cx.to_traceparent(), cx.tracestate()).map(Some)
        }
        other => panic!("unknown parent_form {other}"),
    }
}

fn run_fixture(fixture: &Value) -> Value {
    check_fields(
        "fixture",
        fixture,
        &["name", "description", "config", "steps", "expect"],
    );
    let config = &fixture["config"];
    check_fields("config", config, CONFIG_FIELDS);
    let clock = Arc::new(ManualClock::new(base_time()));
    let spans = InMemorySpanExporter::default();
    let metrics = InMemoryMetricExporter::default();
    let logs = InMemoryLogExporter::default();
    let tracer_provider = SdkTracerProvider::builder()
        .with_simple_exporter(spans.clone())
        .build();
    let meter_provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(metrics.clone()).build())
        .build();
    let logger_provider = SdkLoggerProvider::builder()
        .with_simple_exporter(logs.clone())
        .build();

    let mut builder = Observer::builder()
        .meter_provider(&meter_provider)
        .content_policy(content_policy(config))
        .clock(Clock::manual(clock.clone()));
    match config
        .get("tracer")
        .map_or("sdk", |v| v.as_str().expect("tracer"))
    {
        "sdk" => builder = builder.tracer_provider(&tracer_provider),
        "none" => {}
        "noop" => builder = builder.tracer_provider(&NoopTracerProvider::new()),
        other => panic!("unknown tracer {other}"),
    }
    if b(config, "logs").unwrap_or(true) {
        builder = builder.logger_provider(&logger_provider);
    }
    let observer = builder.build();

    let mut run = Run {
        spans,
        metrics,
        logs,
        tracer_provider,
        meter_provider,
        logger_provider,
        span_ids: HashMap::new(),
        remote_traces: HashSet::new(),
    };
    let mut handles: HashMap<String, Handle> = HashMap::new();

    for (index, step) in fixture["steps"]
        .as_array()
        .expect("steps")
        .iter()
        .enumerate()
    {
        let op = step["op"].as_str().expect("op");
        let mut allowed = op_fields(op);
        allowed.extend(["at_ms", "op"]);
        check_fields(&format!("step {index} ({op})"), step, &allowed);
        clock.set(at(step["at_ms"].as_u64().expect("at_ms")));
        let id = s(step, "id");
        let time = u(step, "time_ms").map(at);
        if let Some(tp) = s(step, "parent_traceparent") {
            if let Some(trace_id) = tp.get(3..35) {
                run.remote_traces.insert(trace_id.to_owned());
            }
        }
        let target = || &handles[id.as_deref().expect("id")];
        let named = |k: &str| &handles[&s(step, k).unwrap_or_else(|| panic!("{k}"))];

        if op.starts_with("start_") && op != "start_attempt" {
            let parent = build_parent(step, &handles);
            let expect_error = s(step, "expect_error");
            let parent = match (parent, expect_error.as_deref()) {
                (Err(_), Some("invalid_parent")) => continue,
                (Ok(_), Some("invalid_parent")) => {
                    panic!("step {index}: parent was accepted but must be rejected")
                }
                (_, Some(other)) => panic!("unknown expect_error {other}"),
                (Err(e), None) => panic!("step {index}: invalid parent: {e}"),
                (Ok(parent), None) => parent,
            };
            let mut options = StartOptions::default().with_parent(parent);
            match s(step, "trace").as_deref() {
                None | Some("default") => {}
                Some("suppress") => options = options.with_trace(TraceMode::Suppress),
                Some(other) => panic!("unknown trace {other}"),
            }
            if let Some(t) = time {
                options = options.with_start_time(t);
            }
            let handle = match op {
                "start_inference" => Handle::Inference(Box::new(
                    observer.start_inference(inference_request(&step["request"]), options),
                )),
                "start_workflow" => {
                    Handle::Workflow(observer.start_workflow(s(step, "name").unwrap(), options))
                }
                "start_tool" => Handle::Tool(observer.start_tool(
                    ToolCallInfo {
                        name: s(step, "name").unwrap(),
                        call_id: s(step, "call_id"),
                        tool_type: s(step, "tool_type"),
                        description: s(step, "description"),
                        arguments: step.get("arguments").map(expand),
                    },
                    options,
                )),
                "start_model_load" => Handle::ModelLoad(observer.start_model_load(
                    ModelLoadInfo {
                        model: s(step, "model").unwrap(),
                        provider: s(step, "provider"),
                        runtime: s(step, "runtime"),
                        format: s(step, "format"),
                        size_bytes: u(step, "size_bytes"),
                    },
                    options,
                )),
                other => panic!("unknown step op {other}"),
            };
            handles.insert(id.clone().expect("id"), handle);
        } else {
            match op {
                "start_attempt" => {
                    let parent = named("parent").inference();
                    let attempt = AttemptInfo {
                        index: u32::try_from(u(step, "index").unwrap()).unwrap(),
                        reason: serde_json::from_value::<AttemptReason>(step["reason"].clone())
                            .unwrap(),
                        provider: s(step, "provider").unwrap(),
                        model: s(step, "model"),
                        location: location(step, "location").unwrap(),
                        runtime: s(step, "runtime"),
                        server_address: s(step, "server_address"),
                        server_port: i(step, "server_port"),
                    };
                    let h = match time {
                        Some(t) => parent.start_attempt_at(attempt, t),
                        None => parent.start_attempt(attempt),
                    };
                    handles.insert(id.clone().unwrap(), Handle::Attempt(h));
                }
                "input_messages" => {
                    let m: Vec<InputMessage> =
                        serde_json::from_value(expand(&step["value"])).unwrap();
                    target().inference().record_input_messages(&m);
                }
                "output_messages" => {
                    let m: Vec<OutputMessage> =
                        serde_json::from_value(expand(&step["value"])).unwrap();
                    target().inference().record_output_messages(&m);
                }
                "system_instructions" => {
                    let p: Vec<Part> = serde_json::from_value(expand(&step["value"])).unwrap();
                    target().inference().record_system_instructions(&p);
                }
                "tool_definitions" => target()
                    .inference()
                    .record_tool_definitions(&expand(&step["value"])),
                "tool_result" => match target() {
                    Handle::Tool(h) => h.record_result(&expand(&step["value"])),
                    _ => panic!("tool_result needs a tool"),
                },
                "chunk" => match time {
                    Some(t) => target().inference().record_chunk_at(t),
                    None => target().inference().record_chunk(),
                },
                "first_token" => match time {
                    Some(t) => target().inference().record_first_token_at(t),
                    None => target().inference().record_first_token(),
                },
                "token_timing" => target()
                    .inference()
                    .record_token_timing(f(step, "time_per_output_token_s").unwrap()),
                "usage" => match target() {
                    Handle::Inference(h) => h.record_usage(&usage(step)),
                    Handle::Attempt(h) => h.record_usage(&usage(step)),
                    _ => panic!("usage needs inference or attempt"),
                },
                "finish_reasons" => {
                    let r: Vec<String> = serde_json::from_value(step["value"].clone()).unwrap();
                    target().inference().record_finish_reasons(r);
                }
                "response" => target().inference().record_response(
                    s(step, "response_id").as_deref(),
                    s(step, "model").as_deref(),
                ),
                "resource" => {
                    let mut sample = step.clone();
                    let obj = sample.as_object_mut().unwrap();
                    obj.retain(|k, _| RESOURCE_FIELDS.contains(&k.as_str()));
                    let sample: ResourceSample = serde_json::from_value(sample).unwrap();
                    match target() {
                        Handle::Inference(h) => h.record_resource(&sample),
                        Handle::Attempt(h) => h.record_resource(&sample),
                        Handle::ModelLoad(h) => h.record_resource(&sample),
                        _ => panic!("resource target"),
                    }
                }
                "model_cache_hit" => match target() {
                    Handle::ModelLoad(h) => h.record_cache_hit(step["value"].as_bool().unwrap()),
                    _ => panic!("model_cache_hit needs a model load"),
                },
                "end" => target().end(time),
                "end_with_error" => target().end_with_error(
                    &ErrorInfo {
                        error_type: s(step, "error_type").unwrap(),
                        message: s(step, "message"),
                    },
                    time,
                ),
                "cancel" => target().cancel(time),
                "evaluation" | "feedback" => {
                    let expect = b(step, "expect_result").unwrap_or(true);
                    let reference = if let Some(t) = s(step, "target") {
                        let r = handles[&t].observation_ref();
                        assert!(
                            r.is_some() || !expect,
                            "step {index}: target {t} has no reference"
                        );
                        r
                    } else {
                        serde_json::from_value::<ObservationRef>(
                            step.get("target_ref")
                                .expect("target or target_ref")
                                .clone(),
                        )
                        .ok()
                    };
                    let result = reference.is_some_and(|r| {
                        if op == "evaluation" {
                            observer.record_evaluation(
                                &r,
                                &Evaluation {
                                    name: s(step, "name").unwrap(),
                                    score_value: f(step, "score_value"),
                                    score_label: s(step, "score_label"),
                                    explanation: s(step, "explanation"),
                                    response_id: s(step, "response_id"),
                                },
                            )
                        } else {
                            observer.record_feedback(
                                &r,
                                &Feedback {
                                    score: f(step, "score"),
                                    label: s(step, "label"),
                                    comment: s(step, "comment"),
                                },
                            )
                        }
                    });
                    assert_eq!(result, expect, "step {index}: {op} result");
                }
                "assert_ref" => {
                    let h = named("target");
                    let r = h.observation_ref();
                    let present = b(step, "present").expect("present");
                    assert_eq!(r.is_some(), present, "step {index}: reference presence");
                    if let Some(r) = r {
                        assert!(ObservationRef::new(&r.trace_id, &r.span_id).is_ok());
                        if let Some(t) = s(step, "trace_id") {
                            assert_eq!(r.trace_id, t, "step {index}: trace id");
                        }
                        if let Some(cx) = h.context() {
                            assert_eq!(r.span_id, cx.span_id(), "step {index}: span id");
                        }
                    }
                }
                "assert_context" => {
                    let cx = named("target").context();
                    let present = b(step, "present").expect("present");
                    assert_eq!(cx.is_some(), present, "step {index}: context presence");
                    if let Some(cx) = cx {
                        if let Some(sampled) = b(step, "sampled") {
                            assert_eq!(cx.is_sampled(), sampled, "step {index}: sampled");
                        }
                        if let Some(t) = s(step, "trace_id") {
                            assert_eq!(cx.trace_id(), t, "step {index}: trace id");
                        }
                        if let Some(tp) = s(step, "traceparent") {
                            assert_eq!(cx.to_traceparent(), tp, "step {index}: traceparent");
                        }
                        if let Some(ts) = step.get("tracestate") {
                            assert_eq!(cx.tracestate(), ts.as_str(), "step {index}: tracestate");
                        }
                    }
                }
                "parse_context" => {
                    let expect = &step["expect"];
                    check_fields(
                        "parse_context expect",
                        expect,
                        &["valid", "traceparent", "tracestate", "sampled"],
                    );
                    let parsed = ObservationContext::from_traceparent(
                        step["traceparent"].as_str().expect("traceparent"),
                        step.get("tracestate").and_then(Value::as_str),
                    );
                    let valid = b(expect, "valid").expect("valid");
                    assert_eq!(
                        parsed.is_ok(),
                        valid,
                        "step {index}: {:?}",
                        step["traceparent"]
                    );
                    if let Ok(cx) = parsed {
                        if let Some(tp) = s(expect, "traceparent") {
                            assert_eq!(cx.to_traceparent(), tp, "step {index}");
                        }
                        if let Some(ts) = expect.get("tracestate") {
                            assert_eq!(cx.tracestate(), ts.as_str(), "step {index}: tracestate");
                        }
                        if let Some(sampled) = b(expect, "sampled") {
                            assert_eq!(cx.is_sampled(), sampled, "step {index}: sampled");
                        }
                    }
                }
                other => panic!("unknown step op {other}"),
            }
        }

        if let Some(id) = &id {
            if let Some(r) = handles.get(id).and_then(Handle::observation_ref) {
                run.span_ids.entry(r.span_id).or_insert_with(|| id.clone());
            }
        }
    }

    normalize(&mut run)
}

fn otel_to_json(v: &OtelValue) -> Value {
    match v {
        OtelValue::Bool(b) => json!(b),
        OtelValue::I64(n) => json!(n),
        OtelValue::F64(n) => json!(n),
        OtelValue::String(s) => json!(s.as_str()),
        OtelValue::Array(a) => match a {
            opentelemetry::Array::Bool(v) => json!(v),
            opentelemetry::Array::I64(v) => json!(v),
            opentelemetry::Array::F64(v) => json!(v),
            opentelemetry::Array::String(v) => {
                json!(v.iter().map(|s| s.as_str().to_owned()).collect::<Vec<_>>())
            }
            _ => json!(null),
        },
        _ => json!(null),
    }
}

/// Checks a span/metric attribute's runtime type against the catalog.
fn assert_catalog_type(key: &str, v: &OtelValue) {
    let ok = match wire::attribute_type(key) {
        None => true,
        Some(CatalogType::Int) => matches!(v, OtelValue::I64(_)),
        Some(CatalogType::Double) => matches!(v, OtelValue::F64(_)),
        Some(CatalogType::Boolean) => matches!(v, OtelValue::Bool(_)),
        Some(CatalogType::String | CatalogType::Json) => matches!(v, OtelValue::String(_)),
        Some(CatalogType::StringArray) => {
            matches!(v, OtelValue::Array(opentelemetry::Array::String(_)))
        }
    };
    assert!(ok, "{key} has the wrong value type: {v:?}");
}

const JSON_ATTRIBUTES: [&str; 6] = [
    attr::INPUT_MESSAGES,
    attr::OUTPUT_MESSAGES,
    attr::SYSTEM_INSTRUCTIONS,
    attr::TOOL_CALL_ARGUMENTS,
    attr::TOOL_CALL_RESULT,
    attr::TOOL_DEFINITIONS,
];

fn attributes_json<'a>(attrs: impl Iterator<Item = &'a KeyValue>) -> Value {
    let mut out = Map::new();
    for kv in attrs {
        let key = kv.key.as_str();
        assert_catalog_type(key, &kv.value);
        let mut value = otel_to_json(&kv.value);
        if JSON_ATTRIBUTES.contains(&key) {
            value = parse_unbounded(value.as_str().expect("json attribute is a string"));
        }
        assert!(
            out.insert(key.to_owned(), value).is_none(),
            "attribute {key} appears twice"
        );
    }
    Value::Object(out)
}

fn ms(t: SystemTime) -> u64 {
    u64::try_from(t.duration_since(base_time()).unwrap().as_millis()).unwrap()
}

fn normalize(run: &mut Run) -> Value {
    run.tracer_provider.force_flush().unwrap();
    run.meter_provider.force_flush().unwrap();
    run.logger_provider.force_flush().unwrap();

    let finished = run.spans.get_finished_spans().unwrap();
    let mut traces: Vec<String> = Vec::new();
    let mut spans = Vec::new();
    for span in &finished {
        let span_id = span.span_context.span_id().to_string();
        let trace_id = span.span_context.trace_id().to_string();
        if !traces.contains(&trace_id) {
            traces.push(trace_id.clone());
        }
        let parent_hex = span.parent_span_id.to_string();
        let parent = run
            .span_ids
            .get(&parent_hex)
            .map_or(Value::Null, |p| json!(p));
        let status = match &span.status {
            Status::Unset => json!({"code": "UNSET"}),
            Status::Ok => json!({"code": "OK"}),
            Status::Error { description } if description.is_empty() => {
                json!({"code": "ERROR"})
            }
            Status::Error { description } => {
                json!({"code": "ERROR", "description": description.as_ref()})
            }
        };
        let kind = match span.span_kind {
            SpanKind::Client => "CLIENT",
            SpanKind::Server => "SERVER",
            SpanKind::Producer => "PRODUCER",
            SpanKind::Consumer => "CONSUMER",
            SpanKind::Internal => "INTERNAL",
        };
        let mut entry = json!({
            "id": run.span_ids.get(&span_id).cloned().unwrap_or_else(|| span_id.clone()),
            "name": span.name.as_ref(),
            "kind": kind,
            "parent": parent,
            "trace": traces.iter().position(|t| *t == trace_id).unwrap(),
            "status": status,
            "start_ms": ms(span.start_time),
            "end_ms": ms(span.end_time),
            "attributes": attributes_json(span.attributes.iter()),
        });
        if run.remote_traces.contains(&trace_id) {
            entry["trace_id"] = json!(trace_id);
            entry["tracestate"] = json!(span.span_context.trace_state().header());
            if entry["parent"].is_null()
                && span.parent_span_id != opentelemetry::trace::SpanId::INVALID
            {
                entry["parent_span_id"] = json!(parent_hex);
            }
        }
        assert!(
            span.events.is_empty(),
            "spans carry no events in the contract"
        );
        spans.push(entry);
    }

    let catalog = catalog();
    let mut metrics = Vec::new();
    let finished = run.metrics.get_finished_metrics().unwrap();
    if let Some(rm) = finished.last() {
        for sm in rm.scope_metrics() {
            assert_eq!(sm.scope().name(), semconv::SCOPE_NAME);
            for m in sm.metrics() {
                let descriptor = &catalog["metrics"][m.name()];
                assert!(descriptor.is_object(), "{} is not in the catalog", m.name());
                assert_eq!(descriptor["unit"], m.unit(), "{} unit", m.name());
                let buckets = descriptor
                    .get("buckets")
                    .map(|b| &catalog["bucket_boundaries"][b.as_str().unwrap()]);
                let check_bounds = |bounds: Vec<f64>| {
                    assert_matches(
                        &format!("{} buckets", m.name()),
                        &json!(bounds),
                        buckets.unwrap(),
                    );
                };
                let (instrument, value_type) = match m.data() {
                    AggregatedMetrics::F64(MetricData::Histogram(_)) => ("histogram", "double"),
                    AggregatedMetrics::U64(MetricData::Histogram(_)) => ("histogram", "int"),
                    AggregatedMetrics::U64(MetricData::Sum(_)) => ("counter", "int"),
                    _ => panic!("unexpected aggregation for {}", m.name()),
                };
                assert_eq!(descriptor["instrument"], instrument, "{}", m.name());
                assert_eq!(descriptor["value_type"], value_type, "{}", m.name());
                let mut points: Vec<Value> = match m.data() {
                    AggregatedMetrics::F64(MetricData::Histogram(h)) => h
                        .data_points()
                        .map(|p| {
                            check_bounds(p.bounds().collect());
                            json!({"attributes": attributes_json(p.attributes()), "count": p.count(), "sum": p.sum()})
                        })
                        .collect(),
                    AggregatedMetrics::U64(MetricData::Histogram(h)) => h
                        .data_points()
                        .map(|p| {
                            check_bounds(p.bounds().collect());
                            json!({"attributes": attributes_json(p.attributes()), "count": p.count(), "sum": p.sum()})
                        })
                        .collect(),
                    AggregatedMetrics::U64(MetricData::Sum(s)) => {
                        assert!(s.is_monotonic(), "{} is a counter", m.name());
                        s.data_points()
                            .map(|p| json!({"attributes": attributes_json(p.attributes()), "value": p.value()}))
                            .collect()
                    }
                    _ => unreachable!(),
                };
                if points.is_empty() {
                    continue;
                }
                points.sort_by_key(|p| p["attributes"].to_string());
                metrics.push(json!({"name": m.name(), "unit": m.unit(), "points": points}));
            }
        }
    }
    metrics.sort_by_key(|m| m["name"].as_str().unwrap().to_owned());

    let mut logs = Vec::new();
    for log in run.logs.get_emitted_logs().unwrap() {
        let record = &log.record;
        let span = record.trace_context().map_or(Value::Null, |t| {
            let id = t.span_id.to_string();
            match run.span_ids.get(&id) {
                Some(handle) => json!(handle),
                None if t.span_id != opentelemetry::trace::SpanId::INVALID => json!(id),
                None => Value::Null,
            }
        });
        let mut attributes = Map::new();
        for (k, v) in record.attributes_iter() {
            assert!(
                attributes
                    .insert(k.as_str().to_owned(), wire::any_value_to_json(v))
                    .is_none(),
                "log attribute {} appears twice",
                k.as_str()
            );
        }
        logs.push(json!({
            "event_name": record.event_name(),
            "severity_number": record.severity_number().map(|s| s as i32),
            "span": span,
            "attributes": attributes,
        }));
    }

    json!({"spans": spans, "metrics": metrics, "logs": logs})
}

/// Structural comparison with float tolerance.
fn assert_matches(path: &str, actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!((a - e).abs() <= 1e-9, "{path}: {a} != {e}");
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(
                a.len(),
                e.len(),
                "{path}: length {} != {}\nactual: {actual:#}",
                a.len(),
                e.len()
            );
            for (idx, (a, e)) in a.iter().zip(e).enumerate() {
                assert_matches(&format!("{path}[{idx}]"), a, e);
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            let ak: Vec<_> = a.keys().collect();
            let ek: Vec<_> = e.keys().collect();
            assert_eq!(ak, ek, "{path}: keys differ\nactual: {actual:#}");
            for (k, ev) in e {
                assert_matches(&format!("{path}.{k}"), &a[k], ev);
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn contract_fixtures() {
    let dump = std::env::var("XYBRID_FIXTURE_DUMP").ok().map(PathBuf::from);
    let only = std::env::var("XYBRID_FIXTURE_ONLY").ok();
    let mut paths: Vec<_> = std::fs::read_dir(contract_dir().join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty());
    let mut missing = Vec::new();
    let mut ran = 0;
    for path in paths {
        let fixture = parse_unbounded(&std::fs::read_to_string(&path).unwrap());
        let name = fixture["name"].as_str().unwrap().to_owned();
        if only.as_ref().is_some_and(|o| *o != name) {
            continue;
        }
        let actual = run_fixture(&fixture);
        ran += 1;
        if let Some(dir) = &dump {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(
                dir.join(format!("{name}.json")),
                serde_json::to_string_pretty(&actual).unwrap(),
            )
            .unwrap();
        }
        match &fixture["expect"] {
            Value::Null => missing.push(name),
            expected => assert_matches(&name, &actual, expected),
        }
    }
    assert!(ran > 0, "no fixture ran");
    assert!(
        missing.is_empty(),
        "fixtures without expectations: {missing:?}"
    );
}

#[test]
fn catalog_matches_constants() {
    let catalog = catalog();
    assert_eq!(
        catalog["genai_semconv"]["revision"],
        semconv::GENAI_SEMCONV_REVISION
    );
    assert_eq!(catalog["contract_version"], semconv::CONTRACT_VERSION);
    assert_eq!(
        catalog["instrumentation_scope"]["name"],
        semconv::SCOPE_NAME
    );
    assert_eq!(
        catalog["inference_operations"],
        json!(semconv::op::INFERENCE_OPERATIONS)
    );

    let metrics: BTreeMap<&str, &str> = [
        (metric::INFERENCE_DURATION, "s"),
        (metric::EMBEDDINGS_DURATION, "s"),
        (metric::OPERATION_DURATION, "s"),
        (metric::TIME_TO_FIRST_CHUNK, "s"),
        (metric::TIME_PER_OUTPUT_CHUNK, "s"),
        (metric::USAGE_INPUT_TOKENS, "{token}"),
        (metric::USAGE_OUTPUT_TOKENS, "{token}"),
        (metric::USAGE_CACHE_READ_INPUT_TOKENS, "{token}"),
        (metric::USAGE_CACHE_WRITE_INPUT_TOKENS, "{token}"),
        (metric::USAGE_REASONING_OUTPUT_TOKENS, "{token}"),
        (metric::OPERATION_INPUT_TOKENS, "{token}"),
        (metric::OPERATION_OUTPUT_TOKENS, "{token}"),
        (metric::EMBEDDINGS_INPUT_TOKENS, "{token}"),
        (metric::WORKFLOW_DURATION, "s"),
        (metric::TOOL_DURATION, "s"),
        (metric::TIME_TO_FIRST_TOKEN, "s"),
        (metric::ATTEMPTS, "{attempt}"),
    ]
    .into_iter()
    .collect();
    let catalog_metrics = catalog["metrics"].as_object().unwrap();
    assert_eq!(catalog_metrics.len(), metrics.len());
    for (name, unit) in &metrics {
        assert_eq!(catalog_metrics[*name]["unit"], *unit, "{name}");
    }
    let buckets = &catalog["bucket_boundaries"];
    assert_matches(
        "duration",
        &buckets["duration"],
        &json!(semconv::DURATION_BUCKETS.to_vec()),
    );
    assert_matches(
        "tokens",
        &buckets["tokens"],
        &json!(semconv::TOKEN_BUCKETS.to_vec()),
    );

    let defaults = &catalog["defaults"];
    assert_eq!(
        defaults["max_content_attribute_bytes"],
        semconv::DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES
    );
    assert_eq!(
        defaults["max_error_message_chars"],
        semconv::MAX_ERROR_MESSAGE_CHARS
    );
    assert_eq!(defaults["truncation_marker"], semconv::TRUNCATION_MARKER);
    assert_eq!(
        defaults["truncation_min_string_bytes"],
        semconv::TRUNCATION_MIN_STRING_BYTES
    );
    assert_eq!(
        defaults["redaction_replacement"],
        semconv::REDACTION_REPLACEMENT
    );
    assert_eq!(
        defaults["content_capture"],
        ContentCapture::default().as_str()
    );
    assert_eq!(
        defaults["max_retained_operation_bytes"],
        semconv::DEFAULT_MAX_RETAINED_OPERATION_BYTES
    );
    assert_eq!(
        defaults["max_traversal_bytes"],
        semconv::DEFAULT_MAX_TRAVERSAL_BYTES
    );
    assert_eq!(defaults["max_json_depth"], semconv::MAX_JSON_DEPTH);
    assert_eq!(
        defaults["preflight_node_bytes"],
        semconv::PREFLIGHT_NODE_BYTES
    );
    assert_eq!(
        defaults["max_tracestate_members"],
        semconv::MAX_TRACESTATE_MEMBERS
    );
    assert_eq!(
        defaults["max_tracestate_chars"],
        semconv::MAX_TRACESTATE_CHARS
    );
    let policy = ContentPolicy::default();
    assert_eq!(
        defaults["max_content_attribute_bytes"],
        policy.max_attribute_bytes()
    );
    assert_eq!(
        defaults["max_retained_operation_bytes"],
        policy.max_retained_operation_bytes()
    );
    assert_eq!(
        defaults["max_traversal_bytes"],
        policy.max_traversal_bytes()
    );
    assert_eq!(
        catalog["env"]["content_capture"],
        semconv::CAPTURE_CONTENT_ENV
    );
    assert_eq!(
        catalog["env"]["content_capture_legacy"],
        semconv::CAPTURE_CONTENT_LEGACY_ENV
    );
    assert_eq!(
        catalog["enums"]["redaction_rule"],
        json!(["replace_literal", "omit_attribute"])
    );
    assert_eq!(
        catalog["enums"][attr::RESOURCE_MEMORY_PRESSURE],
        json!(semconv::MEMORY_PRESSURE_LEVELS)
    );
    for (rule, kind) in [
        (
            RedactionRule::replace_literal("a", "b").unwrap(),
            "replace_literal",
        ),
        (
            RedactionRule::omit_attribute("k").unwrap(),
            "omit_attribute",
        ),
    ] {
        assert_eq!(serde_json::to_value(rule).unwrap()["type"], kind);
    }

    // Every catalog attribute's value type matches the wire table.
    let mut checked = 0;
    for group in ["standard", "content", "schema_content", "extension"] {
        for (key, ty) in catalog["attributes"][group].as_object().unwrap() {
            if key.starts_with('$') {
                continue;
            }
            let key = key.replace("{key}", "example");
            assert_eq!(
                wire::attribute_type(&key).map(CatalogType::as_str),
                ty.as_str(),
                "{key}"
            );
            checked += 1;
        }
    }
    assert!(checked > 80);
    for key in [
        attr::OPERATION_NAME,
        attr::RESPONSE_TIME_TO_FIRST_CHUNK,
        attr::RESPONSE_TIME_TO_FIRST_TOKEN,
        attr::ATTEMPT_INDEX,
        attr::CONTENT_TRUNCATED_ATTRIBUTES,
        attr::TOOL_DEFINITIONS,
        attr::FEEDBACK_COMMENT,
        attr::RESOURCE_ACCELERATOR_MEMORY_USED_BYTES,
        attr::RESOURCE_CPU_PEAK_UTILIZATION,
        attr::RESOURCE_MEMORY_MIN_AVAILABLE_BYTES,
        attr::RESOURCE_MEMORY_PRESSURE,
    ] {
        assert!(
            wire::attribute_type(key).is_some(),
            "{key} missing from catalog"
        );
    }
    #[cfg(feature = "otlp")]
    assert_eq!(
        catalog["presets"]["xybrid_traces"]["traces_endpoint"],
        xybrid_observability::otlp::XYBRID_TRACES_ENDPOINT
    );
    #[cfg(feature = "otlp")]
    transport_defaults_match_the_catalog(defaults);
}

/// The OTLP transport defaults (`catalog.defaults`), as constants and as the
/// public configuration defaults.
#[cfg(feature = "otlp")]
fn transport_defaults_match_the_catalog(d: &Value) {
    use xybrid_observability::otlp::{defaults, BatchLimits, OtlpConfig, RetryLimits};
    let ms = |v: &Value| Duration::from_millis(v.as_u64().expect("milliseconds"));
    assert_eq!(d["batch"]["max_queue_size"], defaults::MAX_QUEUE_SIZE);
    assert_eq!(
        d["batch"]["max_export_batch_size"],
        defaults::MAX_EXPORT_BATCH_SIZE
    );
    assert_eq!(
        ms(&d["batch"]["scheduled_delay_ms"]),
        defaults::SCHEDULED_DELAY
    );
    assert_eq!(
        ms(&d["batch"]["export_timeout_ms"]),
        defaults::EXPORT_TIMEOUT
    );
    assert_eq!(d["retry"]["max_retries"], defaults::MAX_RETRIES);
    assert_eq!(ms(&d["retry"]["initial_delay_ms"]), defaults::INITIAL_DELAY);
    assert_eq!(ms(&d["retry"]["max_delay_ms"]), defaults::MAX_DELAY);
    assert_eq!(
        d["retry"]["retryable_status"],
        json!(defaults::RETRYABLE_STATUS)
    );
    assert_eq!(
        d["max_request_body_bytes"],
        defaults::MAX_REQUEST_BODY_BYTES
    );
    assert_eq!(
        d["max_buffered_bytes_per_signal"],
        defaults::MAX_BUFFERED_BYTES_PER_SIGNAL
    );
    assert_eq!(
        d["max_response_body_bytes"],
        defaults::MAX_RESPONSE_BODY_BYTES
    );
    assert_eq!(ms(&d["shutdown_timeout_ms"]), defaults::SHUTDOWN_TIMEOUT);
    assert_eq!(ms(&d["flush_timeout_ms"]), defaults::FLUSH_TIMEOUT);
    assert_eq!(
        ms(&d["metric_export_interval_ms"]),
        defaults::METRIC_EXPORT_INTERVAL
    );

    let batch = BatchLimits::default();
    assert_eq!(batch.max_queue_size, defaults::MAX_QUEUE_SIZE);
    assert_eq!(batch.max_export_batch_size, defaults::MAX_EXPORT_BATCH_SIZE);
    assert_eq!(batch.scheduled_delay, defaults::SCHEDULED_DELAY);
    assert_eq!(batch.export_timeout, defaults::EXPORT_TIMEOUT);
    assert_eq!(
        batch.max_buffered_bytes,
        defaults::MAX_BUFFERED_BYTES_PER_SIGNAL
    );
    let retry = RetryLimits::default();
    assert_eq!(retry.max_retries, defaults::MAX_RETRIES);
    assert_eq!(retry.initial_delay, defaults::INITIAL_DELAY);
    assert_eq!(retry.max_delay, defaults::MAX_DELAY);
    let config = OtlpConfig::default();
    assert_eq!(config.batch, batch);
    assert_eq!(config.retry, retry);
    assert_eq!(
        config.max_request_body_bytes,
        defaults::MAX_REQUEST_BODY_BYTES
    );
    assert_eq!(
        config.metric_export_interval,
        defaults::METRIC_EXPORT_INTERVAL
    );
}
