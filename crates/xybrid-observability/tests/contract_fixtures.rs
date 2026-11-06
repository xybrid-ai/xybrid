//! Runs every shared contract fixture (`contract/fixtures/*.json`) and checks
//! the normalized spans, metrics, and log events against its `expect` block.
//!
//! `XYBRID_FIXTURE_DUMP=<dir>` writes the normalized output of each fixture to
//! `<dir>/<name>.json`; `XYBRID_FIXTURE_BLESS=1` writes it into the fixture's
//! `expect` block (review the diff before committing).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use opentelemetry::logs::AnyValue;
use opentelemetry::trace::{SpanKind, Status};
use opentelemetry::{KeyValue, Value as OtelValue};
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use serde_json::{json, Map, Value};
use xybrid_observability::semconv::{self, attr, metric};
use xybrid_observability::{
    AttemptInfo, AttemptOperation, AttemptReason, Clock, ContentCapture, ContentPolicy, ErrorInfo,
    Evaluation, Feedback, InferenceOperation, InferenceRequest, InputMessage, Location,
    ManualClock, ModelLoadInfo, ModelLoadOperation, ObservationContext, Observer, OutputMessage,
    Part, PromptRef, ResourceSample, StartOptions, SubstringRedactor, ToolCallInfo, ToolOperation,
    TraceMode, Usage, WorkflowOperation,
};

const BASE_MS: u64 = 1_700_000_000_000;

fn contract_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("contract")
}

fn base_time() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_millis(BASE_MS)
}

enum Handle {
    Inference(Box<InferenceOperation>),
    Attempt(AttemptOperation),
    Workflow(WorkflowOperation),
    Tool(ToolOperation),
    ModelLoad(ModelLoadOperation),
}

impl Handle {
    fn context(&self) -> Option<ObservationContext> {
        match self {
            Self::Inference(h) => h.context(),
            Self::Attempt(h) => h.context(),
            Self::Workflow(h) => h.context(),
            Self::Tool(h) => h.context(),
            Self::ModelLoad(h) => h.context(),
        }
    }

    fn end(&self) {
        match self {
            Self::Inference(h) => h.end(),
            Self::Attempt(h) => h.end(),
            Self::Workflow(h) => h.end(),
            Self::Tool(h) => h.end(),
            Self::ModelLoad(h) => h.end(),
        }
    }

    fn end_with_error(&self, e: &ErrorInfo) {
        match self {
            Self::Inference(h) => h.end_with_error(e),
            Self::Attempt(h) => h.end_with_error(e),
            Self::Workflow(h) => h.end_with_error(e),
            Self::Tool(h) => h.end_with_error(e),
            Self::ModelLoad(h) => h.end_with_error(e),
        }
    }

    fn cancel(&self) {
        match self {
            Self::Inference(h) => h.cancel(),
            Self::Attempt(h) => h.cancel(),
            Self::Workflow(h) => h.cancel(),
            Self::Tool(h) => h.cancel(),
            Self::ModelLoad(h) => h.cancel(),
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

fn modalities(v: &Value, k: &str) -> Vec<xybrid_observability::Modality> {
    v.get(k)
        .map(|m| serde_json::from_value(m.clone()).expect("modalities"))
        .unwrap_or_default()
}

fn location(v: &Value, k: &str) -> Option<Location> {
    v.get(k)
        .map(|m| serde_json::from_value(m.clone()).expect("location"))
}

fn inference_request(r: &Value) -> InferenceRequest {
    InferenceRequest {
        operation: s(r, "operation").expect("operation"),
        provider: s(r, "provider").expect("provider"),
        model: s(r, "model"),
        location: location(r, "location"),
        runtime: s(r, "runtime"),
        runtime_version: s(r, "runtime_version"),
        accelerator: s(r, "accelerator"),
        task: s(r, "task"),
        stream: r.get("stream").and_then(Value::as_bool).unwrap_or(false),
        max_tokens: i(r, "max_tokens"),
        temperature: f(r, "temperature"),
        top_p: f(r, "top_p"),
        top_k: i(r, "top_k"),
        seed: i(r, "seed"),
        stop_sequences: r
            .get("stop_sequences")
            .map(|v| serde_json::from_value(v.clone()).expect("stop_sequences"))
            .unwrap_or_default(),
        frequency_penalty: None,
        presence_penalty: None,
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

struct Run {
    spans: InMemorySpanExporter,
    metrics: InMemoryMetricExporter,
    logs: InMemoryLogExporter,
    tracer_provider: SdkTracerProvider,
    meter_provider: SdkMeterProvider,
    logger_provider: SdkLoggerProvider,
    span_ids: HashMap<String, String>,
}

fn run_fixture(fixture: &Value) -> Value {
    let config = &fixture["config"];
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

    let mut policy = ContentPolicy::default()
        .with_capture(ContentCapture::parse(
            config["content_capture"].as_str().unwrap_or("span_only"),
        ))
        .with_max_attribute_bytes(
            config["max_content_attribute_bytes"]
                .as_u64()
                .unwrap_or(16_384) as usize,
        );
    let needles: Vec<String> = config
        .get("redact_substrings")
        .map(|v| serde_json::from_value(v.clone()).expect("redact_substrings"))
        .unwrap_or_default();
    if !needles.is_empty() {
        policy = policy.with_redactor(SubstringRedactor::new(needles));
    }
    let mut builder = Observer::builder()
        .tracer_provider(&tracer_provider)
        .meter_provider(&meter_provider)
        .content_policy(policy)
        .clock(Clock::manual(clock.clone()));
    if config["logs"].as_bool().unwrap_or(true) {
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
    };
    let mut handles: HashMap<String, Handle> = HashMap::new();
    let mut remote_parent = false;

    for step in fixture["steps"].as_array().expect("steps") {
        let at = step["at_ms"].as_u64().expect("at_ms");
        clock.set(base_time() + Duration::from_millis(at));
        let op = step["op"].as_str().expect("op");
        let id = s(step, "id");
        let target = || &handles[id.as_deref().expect("id")];

        let options = |handles: &HashMap<String, Handle>| {
            let mut options = StartOptions::default();
            if let Some(parent) = s(step, "parent") {
                options.parent = handles[&parent].context();
            }
            if let Some(tp) = s(step, "parent_traceparent") {
                options.parent = Some(
                    ObservationContext::from_traceparent(
                        &tp,
                        step.get("parent_tracestate").and_then(Value::as_str),
                    )
                    .expect("traceparent"),
                );
            }
            if s(step, "trace").as_deref() == Some("suppress") {
                options.trace = TraceMode::Suppress;
            }
            options
        };

        match op {
            "start_inference" => {
                remote_parent |= step.get("parent_traceparent").is_some();
                let h = observer
                    .start_inference(inference_request(&step["request"]), options(&handles));
                handles.insert(id.clone().unwrap(), Handle::Inference(Box::new(h)));
            }
            "start_workflow" => {
                let h = observer.start_workflow(s(step, "name").unwrap(), options(&handles));
                handles.insert(id.clone().unwrap(), Handle::Workflow(h));
            }
            "start_tool" => {
                let call = ToolCallInfo {
                    name: s(step, "name").unwrap(),
                    call_id: s(step, "call_id"),
                    tool_type: s(step, "tool_type"),
                    description: s(step, "description"),
                    arguments: step.get("arguments").cloned(),
                };
                let h = observer.start_tool(call, options(&handles));
                handles.insert(id.clone().unwrap(), Handle::Tool(h));
            }
            "start_model_load" => {
                let info = ModelLoadInfo {
                    model: s(step, "model").unwrap(),
                    provider: s(step, "provider"),
                    runtime: s(step, "runtime"),
                    format: s(step, "format"),
                    size_bytes: u(step, "size_bytes"),
                };
                let h = observer.start_model_load(info, options(&handles));
                handles.insert(id.clone().unwrap(), Handle::ModelLoad(h));
            }
            "start_attempt" => {
                let parent = handles[&s(step, "parent").unwrap()].inference();
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
                let h = parent.start_attempt(attempt);
                handles.insert(id.clone().unwrap(), Handle::Attempt(h));
            }
            "input_messages" => {
                let m: Vec<InputMessage> = serde_json::from_value(step["value"].clone()).unwrap();
                target().inference().record_input_messages(&m);
            }
            "output_messages" => {
                let m: Vec<OutputMessage> = serde_json::from_value(step["value"].clone()).unwrap();
                target().inference().record_output_messages(&m);
            }
            "system_instructions" => {
                let p: Vec<Part> = serde_json::from_value(step["value"].clone()).unwrap();
                target().inference().record_system_instructions(&p);
            }
            "tool_definitions" => target().inference().record_tool_definitions(&step["value"]),
            "tool_result" => match target() {
                Handle::Tool(h) => h.record_result(&step["value"]),
                _ => panic!("tool_result needs a tool"),
            },
            "chunk" => target().inference().record_chunk(),
            "first_token" => target().inference().record_first_token(),
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
                let sample: ResourceSample = serde_json::from_value(step.clone()).unwrap();
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
            "end" => target().end(),
            "end_with_error" => target().end_with_error(&ErrorInfo {
                error_type: s(step, "error_type").unwrap(),
                message: s(step, "message"),
            }),
            "cancel" => target().cancel(),
            "evaluation" => {
                let t = handles[&s(step, "target").unwrap()]
                    .context()
                    .expect("evaluated span")
                    .observation_ref();
                assert!(observer.record_evaluation(
                    &t,
                    &Evaluation {
                        name: s(step, "name").unwrap(),
                        score_value: f(step, "score_value"),
                        score_label: s(step, "score_label"),
                        explanation: s(step, "explanation"),
                        response_id: s(step, "response_id"),
                    }
                ));
            }
            "feedback" => {
                let t = handles[&s(step, "target").unwrap()]
                    .context()
                    .expect("evaluated span")
                    .observation_ref();
                assert!(observer.record_feedback(
                    &t,
                    &Feedback {
                        score: f(step, "score"),
                        label: s(step, "label"),
                        comment: s(step, "comment"),
                    }
                ));
            }
            other => panic!("unknown step op {other}"),
        }

        if let Some(id) = &id {
            if let Some(cx) = handles.get(id).and_then(Handle::context) {
                run.span_ids
                    .entry(cx.span_id())
                    .or_insert_with(|| id.clone());
            }
        }
    }

    normalize(&mut run, remote_parent)
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

fn any_to_json(v: &AnyValue) -> Value {
    match v {
        AnyValue::Int(n) => json!(n),
        AnyValue::Double(n) => json!(n),
        AnyValue::String(s) => json!(s.as_str()),
        AnyValue::Boolean(b) => json!(b),
        AnyValue::ListAny(items) => Value::Array(items.iter().map(any_to_json).collect()),
        AnyValue::Map(map) => {
            let mut out = Map::new();
            for (k, v) in map.iter() {
                out.insert(k.as_str().to_owned(), any_to_json(v));
            }
            Value::Object(out)
        }
        _ => json!(null),
    }
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
        let mut value = otel_to_json(&kv.value);
        if JSON_ATTRIBUTES.contains(&key) {
            value = serde_json::from_str(value.as_str().expect("json attribute is a string"))
                .expect("json attribute parses");
        }
        out.insert(key.to_owned(), value);
    }
    Value::Object(out)
}

fn ms(t: SystemTime) -> u64 {
    u64::try_from(t.duration_since(base_time()).unwrap().as_millis()).unwrap()
}

fn normalize(run: &mut Run, remote_parent: bool) -> Value {
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
        if remote_parent {
            entry["trace_id"] = json!(trace_id);
            entry["tracestate"] = json!(span.span_context.trace_state().header());
            if entry["parent"].is_null() {
                entry["parent_span_id"] = json!(parent_hex);
            }
        }
        assert!(
            span.events.is_empty(),
            "spans carry no events in contract v1"
        );
        spans.push(entry);
    }

    let mut metrics = Vec::new();
    let finished = run.metrics.get_finished_metrics().unwrap();
    if let Some(rm) = finished.last() {
        for sm in rm.scope_metrics() {
            assert_eq!(sm.scope().name(), semconv::SCOPE_NAME);
            for m in sm.metrics() {
                let mut points: Vec<Value> = match m.data() {
                    AggregatedMetrics::F64(MetricData::Histogram(h)) => h
                        .data_points()
                        .map(|p| json!({"attributes": attributes_json(p.attributes()), "count": p.count(), "sum": p.sum()}))
                        .collect(),
                    AggregatedMetrics::U64(MetricData::Histogram(h)) => h
                        .data_points()
                        .map(|p| json!({"attributes": attributes_json(p.attributes()), "count": p.count(), "sum": p.sum()}))
                        .collect(),
                    AggregatedMetrics::U64(MetricData::Sum(s)) => s
                        .data_points()
                        .map(|p| json!({"attributes": attributes_json(p.attributes()), "value": p.value()}))
                        .collect(),
                    _ => panic!("unexpected aggregation for {}", m.name()),
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
        let span = record
            .trace_context()
            .and_then(|t| run.span_ids.get(&t.span_id.to_string()).cloned())
            .map_or(Value::Null, |s| json!(s));
        let mut attributes = Map::new();
        for (k, v) in record.attributes_iter() {
            attributes.insert(k.as_str().to_owned(), any_to_json(v));
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
    let bless = std::env::var("XYBRID_FIXTURE_BLESS").is_ok_and(|v| v == "1");
    let mut paths: Vec<_> = std::fs::read_dir(contract_dir().join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty());
    let mut missing = Vec::new();
    for path in paths {
        let mut fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let name = fixture["name"].as_str().unwrap().to_owned();
        let actual = run_fixture(&fixture);
        if let Some(dir) = &dump {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(
                dir.join(format!("{name}.json")),
                serde_json::to_string_pretty(&actual).unwrap(),
            )
            .unwrap();
        }
        if bless {
            fixture["expect"] = actual;
            let mut text = serde_json::to_string_pretty(&fixture).unwrap();
            text.push('\n');
            std::fs::write(&path, text).unwrap();
            continue;
        }
        match &fixture["expect"] {
            Value::Null => missing.push(name),
            expected => assert_matches(&name, &actual, expected),
        }
    }
    assert!(
        missing.is_empty(),
        "fixtures without expectations: {missing:?}"
    );
}

#[test]
fn catalog_matches_constants() {
    let catalog: Value = serde_json::from_str(
        &std::fs::read_to_string(contract_dir().join("catalog.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        catalog["genai_semconv"]["revision"],
        semconv::GENAI_SEMCONV_REVISION
    );
    assert_eq!(catalog["contract_version"], semconv::CONTRACT_VERSION);
    assert_eq!(
        catalog["instrumentation_scope"]["name"],
        semconv::SCOPE_NAME
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
        catalog["env"]["content_capture"],
        semconv::CAPTURE_CONTENT_ENV
    );

    let all_attributes: Vec<&str> = ["standard", "content", "schema_content", "extension"]
        .iter()
        .flat_map(|group| catalog["attributes"][*group].as_object().unwrap().keys())
        .map(String::as_str)
        .filter(|k| !k.starts_with('$'))
        .collect();
    for key in [
        attr::OPERATION_NAME,
        attr::RESPONSE_TIME_TO_FIRST_CHUNK,
        attr::RESPONSE_TIME_TO_FIRST_TOKEN,
        attr::ATTEMPT_INDEX,
        attr::CONTENT_TRUNCATED_ATTRIBUTES,
        attr::TOOL_DEFINITIONS,
        attr::FEEDBACK_COMMENT,
        attr::RESOURCE_ACCELERATOR_MEMORY_USED_BYTES,
    ] {
        assert!(all_attributes.contains(&key), "{key} missing from catalog");
    }
    #[cfg(feature = "otlp")]
    assert_eq!(
        catalog["presets"]["xybrid_traces"]["traces_endpoint"],
        xybrid_observability::otlp::XYBRID_TRACES_ENDPOINT
    );
}
