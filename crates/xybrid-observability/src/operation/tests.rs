use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use opentelemetry::logs::AnyValue;
use opentelemetry::trace::{
    noop::NoopTracerProvider, Status, TraceContextExt, Tracer, TracerProvider,
};
use opentelemetry::{Context, InstrumentationScope, KeyValue};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::logs::{InMemoryLogExporter, LogProcessor, SdkLogRecord, SdkLoggerProvider};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, Sampler, SdkTracerProvider};
use serde_json::json;

use super::*;
use crate::semconv::metric;
use crate::{ContentCapture, ContentPolicy, Evaluation, Feedback, Observer, RedactionRule};

struct Signals {
    spans: InMemorySpanExporter,
    logs: InMemoryLogExporter,
    tracer: SdkTracerProvider,
    logger: SdkLoggerProvider,
}

fn signals(sampler: Sampler) -> Signals {
    let spans = InMemorySpanExporter::default();
    let logs = InMemoryLogExporter::default();
    let tracer = SdkTracerProvider::builder()
        .with_simple_exporter(spans.clone())
        .with_sampler(sampler)
        .build();
    let logger = SdkLoggerProvider::builder()
        .with_simple_exporter(logs.clone())
        .build();
    Signals {
        spans,
        logs,
        tracer,
        logger,
    }
}

fn observer(s: &Signals, content: ContentPolicy) -> Observer {
    Observer::builder()
        .tracer_provider(&s.tracer)
        .logger_provider(&s.logger)
        .content_policy(content)
        .build()
}

fn chat() -> InferenceRequest {
    InferenceRequest::new("chat", "llama.cpp").with_model("m")
}

fn attr<'a>(attrs: &'a [KeyValue], key: &str) -> Vec<&'a OtelValue> {
    attrs
        .iter()
        .filter(|kv| kv.key.as_str() == key)
        .map(|kv| &kv.value)
        .collect()
}

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

/// Every data point of the metric `name` in the latest export, as its
/// `error.type` and its count (histograms) or value (counters).
fn points(metrics: &InMemoryMetricExporter, name: &str) -> Vec<(Option<String>, u64)> {
    let finished = metrics.get_finished_metrics().unwrap();
    fn error_type<'a>(mut attributes: impl Iterator<Item = &'a KeyValue>) -> Option<String> {
        attributes
            .find(|kv| kv.key.as_str() == attr::ERROR_TYPE)
            .map(|kv| kv.value.to_string())
    }
    let mut out = Vec::new();
    for sm in finished.last().expect("one metric export").scope_metrics() {
        for m in sm.metrics().filter(|m| m.name() == name) {
            match m.data() {
                AggregatedMetrics::F64(MetricData::Histogram(h)) => {
                    for p in h.data_points() {
                        out.push((error_type(p.attributes()), p.count()));
                    }
                }
                AggregatedMetrics::U64(MetricData::Sum(s)) => {
                    for p in s.data_points() {
                        out.push((error_type(p.attributes()), p.value()));
                    }
                }
                _ => panic!("unexpected data for {name}"),
            }
        }
    }
    out
}

#[test]
fn dropped_typed_operations_are_cancelled_in_spans_and_metrics() {
    // A3: Drop of an un-ended handle ends it as cancelled, metric included.
    let s = signals(Sampler::AlwaysOn);
    let metrics = InMemoryMetricExporter::default();
    let meter = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(metrics.clone()).build())
        .build();
    let obs = Observer::builder()
        .tracer_provider(&s.tracer)
        .logger_provider(&s.logger)
        .meter_provider(&meter)
        .build();

    let inference = obs.start_inference(chat(), StartOptions::default());
    drop(inference.start_attempt(AttemptInfo {
        index: 0,
        reason: crate::AttemptReason::Initial,
        provider: "llama.cpp".into(),
        model: Some("m".into()),
        location: Location::Local,
        runtime: None,
        server_address: None,
        server_port: None,
    }));
    drop(inference);
    drop(obs.start_tool(ToolCallInfo::new("lookup"), StartOptions::default()));
    drop(obs.start_workflow("flow", StartOptions::default()));
    drop(obs.start_model_load(
        ModelLoadInfo {
            model: "m".into(),
            ..ModelLoadInfo::default()
        },
        StartOptions::default(),
    ));

    let spans = s.spans.get_finished_spans().unwrap();
    let mut names: Vec<_> = spans.iter().map(|s| s.name.to_string()).collect();
    names.sort();
    assert_eq!(
        names,
        [
            "attempt 0",
            "chat m",
            "execute_tool lookup",
            "invoke_workflow flow",
            "load_model m"
        ],
        "one span per dropped handle"
    );
    for span in &spans {
        assert_eq!(span.status, Status::error("cancelled"), "{}", span.name);
        assert_eq!(
            attr(&span.attributes, attr::ERROR_TYPE),
            [&OtelValue::from("cancelled")],
            "{}",
            span.name
        );
    }

    meter.force_flush().unwrap();
    let cancelled = vec![(Some("cancelled".to_owned()), 1)];
    for name in [
        metric::INFERENCE_DURATION,
        metric::ATTEMPTS,
        metric::TOOL_DURATION,
        metric::WORKFLOW_DURATION,
        metric::OPERATION_DURATION,
    ] {
        assert_eq!(points(&metrics, name), cancelled, "{name}");
    }

    let exceptions = s
        .logs
        .get_emitted_logs()
        .unwrap()
        .into_iter()
        .filter(|l| l.record.event_name() == Some(event::OPERATION_EXCEPTION))
        .count();
    assert_eq!(exceptions, 0, "a cancellation is not an exception");
}

#[test]
fn unsampled_spans_keep_references_and_suppression_has_none() {
    // D3 / context-catalog-2: references exist for unsampled spans of their own.
    let s = signals(Sampler::AlwaysOff);
    let obs = observer(&s, ContentPolicy::default());
    let op = obs.start_inference(chat(), StartOptions::default());
    let cx = op.context().expect("unsampled span context");
    assert!(!cx.is_sampled());
    let r = op.observation_ref().expect("unsampled span reference");
    assert_eq!(r.span_id, cx.span_id());
    let child = obs.start_tool(ToolCallInfo::new("t"), StartOptions::child_of(cx.clone()));
    assert_ne!(child.observation_ref().unwrap().span_id, r.span_id);
    assert_eq!(child.observation_ref().unwrap().trace_id, r.trace_id);
    child.end();
    let suppressed = obs.start_inference(
        chat(),
        StartOptions::default().with_trace(TraceMode::Suppress),
    );
    assert!(suppressed.context().is_none());
    assert!(suppressed.observation_ref().is_none());
    // A reference to an unsampled span is a valid evaluation target.
    assert!(obs.record_evaluation(
        &r,
        &Evaluation {
            name: "q".into(),
            ..Default::default()
        }
    ));
    op.end();
    suppressed.end();
    assert!(s.spans.get_finished_spans().unwrap().is_empty());
}

#[test]
fn api_only_tracers_pass_the_parent_through_without_a_reference() {
    let obs = Observer::builder()
        .tracer_provider(&NoopTracerProvider::new())
        .build();
    let parent = ObservationContext::from_traceparent(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        Some("vendor=abc"),
    )
    .unwrap();
    let op = obs.start_inference(chat(), StartOptions::child_of(parent.clone()));
    assert!(op.observation_ref().is_none());
    assert_eq!(
        op.context().unwrap().to_traceparent(),
        parent.to_traceparent()
    );
    // Attempts of an operation without a span of its own are suppressed.
    let attempt = op.start_attempt(AttemptInfo {
        index: 0,
        reason: crate::AttemptReason::Initial,
        provider: "llama.cpp".into(),
        model: None,
        location: Location::Local,
        runtime: None,
        server_address: None,
        server_port: None,
    });
    assert!(attempt.context().is_none());
    attempt.end();
    let root = obs.start_inference(chat(), StartOptions::default());
    assert!(root.context().is_none() && root.observation_ref().is_none());
}

#[test]
fn span_less_operation_events_never_attach_to_the_ambient_span() {
    // R-CTX-3
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(
        &s,
        ContentPolicy::default().with_capture(ContentCapture::EventOnly),
    );
    let tracer = s.tracer.tracer("app");
    let ambient = tracer.start("ambient");
    let _guard = Context::current_with_span(ambient).attach();
    let op = obs.start_inference(
        chat(),
        StartOptions::default().with_trace(TraceMode::Suppress),
    );
    op.record_input_messages(&[InputMessage::text("user", "hi")]);
    op.end_with_error(&ErrorInfo::new("boom").with_message("failed"));
    let tool = obs.start_tool(
        ToolCallInfo {
            arguments: Some(json!({"q": 1})),
            ..ToolCallInfo::new("lookup")
        },
        StartOptions::default().with_trace(TraceMode::Suppress),
    );
    tool.end();
    let logs = s.logs.get_emitted_logs().unwrap();
    let names: Vec<_> = logs.iter().map(|l| l.record.event_name()).collect();
    assert_eq!(
        names,
        [
            Some(event::OPERATION_EXCEPTION),
            Some(event::INFERENCE_DETAILS),
            Some(event::TOOL_DETAILS)
        ]
    );
    for log in &logs {
        assert!(
            log.record.trace_context().is_none(),
            "{:?}",
            log.record.event_name()
        );
    }
}

#[test]
fn evaluation_records_carry_the_reference_ids_with_zero_flags() {
    // R-LOG-2
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(&s, ContentPolicy::default());
    let target =
        ObservationRef::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").unwrap();
    assert!(obs.record_feedback(
        &target,
        &Feedback {
            score: Some(1.0),
            ..Default::default()
        }
    ));
    let logs = s.logs.get_emitted_logs().unwrap();
    let tc = logs[0].record.trace_context().unwrap();
    assert_eq!(tc.trace_id.to_string(), target.trace_id);
    assert_eq!(tc.span_id.to_string(), target.span_id);
    assert!(tc.trace_flags.is_none_or(|f| f.to_u8() == 0));
    // Invalid public fields and missing loggers return false.
    let mut broken = target.clone();
    broken.span_id = "XYZ".into();
    assert!(!obs.record_feedback(&broken, &Feedback::default()));
    let no_logs = Observer::builder().tracer_provider(&s.tracer).build();
    assert!(!no_logs.record_evaluation(&target, &Evaluation::default()));
    // A logger that reports the event disabled records nothing.
    let disabled = SdkLoggerProvider::builder()
        .with_log_processor(Disabled)
        .build();
    let obs = Observer::builder().logger_provider(&disabled).build();
    assert!(!obs.record_evaluation(&target, &Evaluation::default()));
    let empty = SdkLoggerProvider::builder().build();
    let obs = Observer::builder().logger_provider(&empty).build();
    assert!(!obs.record_feedback(&target, &Feedback::default()));
}

#[derive(Debug)]
struct Disabled;

impl LogProcessor for Disabled {
    fn emit(&self, _data: &mut SdkLogRecord, _scope: &InstrumentationScope) {
        panic!("disabled processors receive nothing");
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn event_enabled(
        &self,
        _level: opentelemetry::logs::Severity,
        _target: &str,
        _name: Option<&str>,
    ) -> bool {
        false
    }
}

#[derive(Debug, Default)]
struct Seen(Mutex<Vec<(String, AnyValue)>>);

#[derive(Debug)]
struct Recording(Arc<Seen>);

impl LogProcessor for Recording {
    fn emit(&self, data: &mut SdkLogRecord, _scope: &InstrumentationScope) {
        let mut seen = self.0 .0.lock().unwrap();
        for (k, v) in data.attributes_iter() {
            seen.push((k.as_str().to_owned(), v.clone()));
        }
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }
}

#[test]
fn preceding_log_processors_see_the_null_carrier() {
    // R-WIRE-5 (processor side; the paired exporter decodes it).
    let seen = Arc::new(Seen::default());
    let logger = SdkLoggerProvider::builder()
        .with_log_processor(Recording(seen.clone()))
        .build();
    let obs = Observer::builder()
        .logger_provider(&logger)
        .content_policy(ContentPolicy::default().with_capture(ContentCapture::EventOnly))
        .build();
    let tool = obs.start_tool(
        ToolCallInfo {
            arguments: Some(json!({"x": null, "e": ""})),
            ..ToolCallInfo::new("lookup")
        },
        StartOptions::default(),
    );
    tool.record_result(&Value::Null);
    tool.end();
    let seen = seen.0.lock().unwrap();
    let result = &seen
        .iter()
        .find(|(k, _)| k == attr::TOOL_CALL_RESULT)
        .unwrap()
        .1;
    assert!(crate::wire::is_null(result));
    let args = &seen
        .iter()
        .find(|(k, _)| k == attr::TOOL_CALL_ARGUMENTS)
        .unwrap()
        .1;
    assert_eq!(
        crate::wire::any_value_to_json(args),
        json!({"x": null, "e": ""})
    );
    let AnyValue::Map(map) = args else {
        panic!("map")
    };
    assert!(crate::wire::is_null(&map[&opentelemetry::Key::from("x")]));
    assert_eq!(map[&opentelemetry::Key::from("e")], AnyValue::from(""));
}

#[test]
fn re_recorded_content_keeps_one_attribute_and_one_truncation_marker() {
    // content-11 / core-17
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(
        &s,
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanOnly)
            .with_max_attribute_bytes(64),
    );
    let tool = obs.start_tool(ToolCallInfo::new("t"), StartOptions::default());
    tool.record_result(&json!({"r": "y".repeat(500)}));
    tool.record_result(&json!({"r": "z".repeat(500)}));
    tool.end();
    let chat_op = obs.start_inference(chat(), StartOptions::default());
    chat_op.record_input_messages(&[InputMessage::text("user", "first")]);
    chat_op.record_input_messages(&[InputMessage::text("user", "second")]);
    chat_op.end();
    let spans = s.spans.get_finished_spans().unwrap();
    let results = attr(&spans[0].attributes, attr::TOOL_CALL_RESULT);
    assert_eq!(results.len(), 1);
    assert!(results[0].as_str().contains("zzz"));
    assert_eq!(
        attr(&spans[0].attributes, attr::CONTENT_TRUNCATED_ATTRIBUTES),
        [&string_array([attr::TOOL_CALL_RESULT.to_owned()])]
    );
    let inputs = attr(&spans[1].attributes, attr::INPUT_MESSAGES);
    assert_eq!(inputs.len(), 1);
    assert!(inputs[0].as_str().contains("second"));
}

#[test]
fn dropped_error_messages_fall_back_to_the_error_type_without_exception_message() {
    // critic-7
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(
        &s,
        ContentPolicy::default()
            .with_rule(RedactionRule::omit_attribute(ERROR_MESSAGE_KIND).unwrap()),
    );
    let op = obs.start_inference(chat(), StartOptions::default());
    op.end_with_error(&ErrorInfo::new("auth").with_message("bad key sk-1"));
    let spans = s.spans.get_finished_spans().unwrap();
    assert_eq!(spans[0].status, Status::error("auth"));
    let logs = s.logs.get_emitted_logs().unwrap();
    let keys: Vec<_> = logs[0]
        .record
        .attributes_iter()
        .map(|(k, _)| k.as_str().to_owned())
        .collect();
    assert_eq!(keys, [attr::EXCEPTION_TYPE]);
}

#[test]
fn streaming_points_are_recorded_when_observed_with_known_dimensions() {
    let metrics = InMemoryMetricExporter::default();
    let meter = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(metrics.clone()).build())
        .build();
    let obs = Observer::builder().meter_provider(&meter).build();
    let op = obs.start_inference(chat().streaming(true), StartOptions::default());
    op.record_chunk();
    op.record_response(None, Some("m-2024"));
    op.record_chunk();
    // Points are visible before `end`.
    meter.force_flush().unwrap();
    let finished = metrics.get_finished_metrics().unwrap();
    let mut seen = Vec::new();
    for sm in finished.last().unwrap().scope_metrics() {
        for m in sm.metrics() {
            if let AggregatedMetrics::F64(MetricData::Histogram(h)) = m.data() {
                for p in h.data_points() {
                    let has_model = p
                        .attributes()
                        .any(|kv| kv.key.as_str() == attr::RESPONSE_MODEL);
                    seen.push((m.name().to_owned(), has_model));
                }
            }
        }
    }
    seen.sort();
    assert_eq!(
        seen,
        [
            (
                "gen_ai.client.inference.time_per_output_chunk".to_owned(),
                true
            ),
            (
                "gen_ai.client.inference.time_to_first_chunk".to_owned(),
                false
            ),
        ]
    );
    op.end();
}

#[test]
fn measured_first_token_after_chunks_is_recorded() {
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(&s, ContentPolicy::default());
    let start = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
    let op = obs.start_inference(chat(), StartOptions::default().with_start_time(start));
    op.record_chunk_at(start + Duration::from_millis(100));
    op.record_chunk_at(start + Duration::from_millis(150));
    op.record_first_token_at(start + Duration::from_millis(80));
    op.record_first_token_at(start + Duration::from_millis(60));
    op.end_at(start + Duration::from_millis(200));
    let spans = s.spans.get_finished_spans().unwrap();
    assert_eq!(
        attr(&spans[0].attributes, attr::RESPONSE_TIME_TO_FIRST_TOKEN),
        [&OtelValue::F64(0.08)]
    );
}

#[test]
fn model_loads_fall_back_to_the_runtime_and_never_fabricate_a_provider() {
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(&s, ContentPolicy::default());
    obs.start_model_load(
        ModelLoadInfo {
            model: "a".into(),
            runtime: Some("whisper.cpp".into()),
            ..ModelLoadInfo::default()
        },
        StartOptions::default(),
    )
    .end();
    obs.start_model_load(
        ModelLoadInfo {
            model: "b".into(),
            ..ModelLoadInfo::default()
        },
        StartOptions::default(),
    )
    .end();
    let spans = s.spans.get_finished_spans().unwrap();
    assert_eq!(
        attr(&spans[0].attributes, attr::PROVIDER_NAME),
        [&OtelValue::from("whisper.cpp")]
    );
    assert!(attr(&spans[1].attributes, attr::PROVIDER_NAME).is_empty());
}

#[test]
fn spans_never_end_before_they_start() {
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(&s, ContentPolicy::default());
    let start = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
    let op = obs.start_inference(chat(), StartOptions::default().with_start_time(start));
    op.end_at(start - Duration::from_secs(5));
    let spans = s.spans.get_finished_spans().unwrap();
    assert_eq!(spans[0].end_time, start);
}

#[test]
fn resource_samples_record_measured_values_only() {
    let empty = resource_attributes(&ResourceSample::default());
    assert!(empty.is_empty(), "an empty sample records nothing");

    let sample = ResourceSample {
        cpu_peak_utilization: Some(f64::NAN),
        memory_min_available_bytes: Some(0),
        memory_pressure: Some("high".into()),
        ..ResourceSample::default()
    };
    let recorded = resource_attributes(&sample);
    assert!(attr(&recorded, attr::RESOURCE_CPU_PEAK_UTILIZATION).is_empty());
    assert_eq!(
        attr(&recorded, attr::RESOURCE_MEMORY_MIN_AVAILABLE_BYTES),
        [&OtelValue::I64(0)],
        "zero available memory is a measurement"
    );
    assert!(attr(&recorded, attr::RESOURCE_MEMORY_PRESSURE).is_empty());

    for level in MEMORY_PRESSURE_LEVELS {
        let sample = ResourceSample {
            memory_pressure: Some(level.into()),
            ..ResourceSample::default()
        };
        assert_eq!(
            attr(
                &resource_attributes(&sample),
                attr::RESOURCE_MEMORY_PRESSURE
            ),
            [&OtelValue::from(level)]
        );
    }
}

/// The kinds a redaction hook was called for, in call order.
#[derive(Clone, Default)]
struct Calls(Arc<Mutex<Vec<String>>>);

impl Calls {
    fn policy(&self, capture: ContentCapture) -> ContentPolicy {
        let calls = self.0.clone();
        ContentPolicy::default()
            .with_capture(capture)
            .with_redactor(move |kind: &str, value: Value| {
                calls.lock().unwrap().push(kind.to_owned());
                Some(value)
            })
    }

    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// Every content key the operations below record, in recording order.
const CONTENT_KEYS: [&str; 7] = [
    "gen_ai.prompt.variable.city",
    attr::SYSTEM_INSTRUCTIONS,
    attr::INPUT_MESSAGES,
    attr::TOOL_DEFINITIONS,
    attr::OUTPUT_MESSAGES,
    attr::TOOL_CALL_ARGUMENTS,
    attr::TOOL_CALL_RESULT,
];

/// An inference and a tool, each recording every kind of content.
fn record_every_content_kind(obs: &Observer, options: &StartOptions) {
    let mut request = chat();
    request.prompt = Some(crate::PromptRef {
        name: "p".into(),
        variables: [("city".to_owned(), "Lisbon".to_owned())].into(),
        ..Default::default()
    });
    let op = obs.start_inference(request, options.clone());
    op.record_system_instructions(&[Part::text("be brief")]);
    op.record_input_messages(&[InputMessage::text("user", "hi")]);
    op.record_tool_definitions(&json!([{"name": "lookup"}]));
    op.record_output_messages(&[OutputMessage::assistant_text("hello", Some("stop"))]);
    op.end();
    let call = ToolCallInfo {
        arguments: Some(json!({"q": "x"})),
        ..ToolCallInfo::new("lookup")
    };
    let tool = obs.start_tool(call, options.clone());
    tool.record_result(&json!({"r": "y"}));
    tool.end();
}

/// Content attribute keys on a span or a log record.
fn content_keys<'a>(keys: impl Iterator<Item = &'a str>) -> Vec<String> {
    keys.filter(|k| CONTENT_KEYS.contains(k))
        .map(str::to_owned)
        .collect()
}

#[test]
fn content_for_a_span_that_does_not_record_is_never_processed() {
    // Span-destined content costs nothing (no preflight, hook,
    // rules or truncation) when the operation's span does not record: a
    // sampler drop, an unsampled parent, suppression, no tracer, an API-only
    // tracer. The event destination is unchanged: event capture still
    // processes and emits every value.
    let unsampled_parent = || {
        let s = signals(Sampler::AlwaysOff);
        let parent =
            observer(&s, ContentPolicy::default()).start_workflow("w", StartOptions::default());
        let cx = parent.context().expect("an unsampled context");
        assert!(!cx.is_sampled());
        cx
    };
    let setup = |case: &str| match case {
        "sampler drop" => (Sampler::AlwaysOff, StartOptions::default()),
        "unsampled parent" => (
            Sampler::ParentBased(Box::new(Sampler::AlwaysOn)),
            StartOptions::child_of(unsampled_parent()),
        ),
        "suppressed" => (
            Sampler::AlwaysOn,
            StartOptions::default().with_trace(TraceMode::Suppress),
        ),
        // The SDK provider (always on) is not wired to the observer.
        _ => (Sampler::AlwaysOn, StartOptions::default()),
    };
    let build = |case: &str, s: &Signals, content: ContentPolicy| match case {
        "no tracer" => Observer::builder()
            .logger_provider(&s.logger)
            .content_policy(content)
            .build(),
        "API-only tracer" => Observer::builder()
            .tracer_provider(&NoopTracerProvider::new())
            .logger_provider(&s.logger)
            .content_policy(content)
            .build(),
        _ => observer(s, content),
    };
    let mut every_key: Vec<String> = CONTENT_KEYS.iter().map(|k| (*k).to_owned()).collect();
    every_key.sort();
    for case in [
        "sampler drop",
        "unsampled parent",
        "suppressed",
        "no tracer",
        "API-only tracer",
    ] {
        let calls = Calls::default();
        let (sampler, options) = setup(case);
        let s = signals(sampler);
        let obs = build(case, &s, calls.policy(ContentCapture::SpanOnly));
        record_every_content_kind(&obs, &options);
        assert_eq!(calls.take(), Vec::<String>::new(), "{case}: span_only");
        assert!(s.spans.get_finished_spans().unwrap().is_empty(), "{case}");
        assert!(s.logs.get_emitted_logs().unwrap().is_empty(), "{case}");

        for capture in [ContentCapture::SpanAndEvent, ContentCapture::EventOnly] {
            let calls = Calls::default();
            let (sampler, options) = setup(case);
            let s = signals(sampler);
            let obs = build(case, &s, calls.policy(capture));
            record_every_content_kind(&obs, &options);
            assert_eq!(calls.take(), CONTENT_KEYS, "{case}: {capture:?}");
            // The details events carry every value, as before.
            let mut on_events: Vec<String> = s
                .logs
                .get_emitted_logs()
                .unwrap()
                .iter()
                .flat_map(|l| content_keys(l.record.attributes_iter().map(|(k, _)| k.as_str())))
                .collect();
            on_events.sort();
            assert_eq!(on_events, every_key, "{case}: {capture:?}");
        }
    }

    // A recording span: every value is processed and recorded, as before.
    let calls = Calls::default();
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(&s, calls.policy(ContentCapture::SpanOnly));
    record_every_content_kind(&obs, &StartOptions::default());
    assert_eq!(calls.take(), CONTENT_KEYS);
    let mut on_spans: Vec<String> = s
        .spans
        .get_finished_spans()
        .unwrap()
        .iter()
        .flat_map(|span| content_keys(span.attributes.iter().map(|kv| kv.key.as_str())))
        .collect();
    on_spans.sort();
    assert_eq!(on_spans, every_key);
}

/// A redaction hook that panics for the output messages and error messages
/// and passes everything else through.
fn panicking_hook(kind: &str, value: Value) -> Option<Value> {
    assert!(
        kind != attr::OUTPUT_MESSAGES && kind != ERROR_MESSAGE_KIND,
        "redactor panicked on {kind}"
    );
    Some(value)
}

#[test]
fn a_panicking_redactor_fails_closed_like_an_error() {
    // A hook that panics behaves exactly like one that returns
    // `RedactionError` (and like a JS hook that throws): the value is omitted
    // and listed, the error message falls back to the error type, the
    // operation ends normally, and later values are redacted as usual.
    let s = signals(Sampler::AlwaysOn);
    let obs = observer(
        &s,
        ContentPolicy::default()
            .with_capture(ContentCapture::SpanAndEvent)
            .with_redactor(panicking_hook),
    );
    for round in 0..2 {
        let op = obs.start_inference(chat(), StartOptions::default());
        op.record_input_messages(&[InputMessage::text("user", "hi")]);
        op.record_output_messages(&[OutputMessage::assistant_text("secret", None)]);
        op.record_chunk();
        op.end_with_error(&ErrorInfo::new("boom").with_message("secret message"));
        assert!(op.is_finished(), "round {round}");
    }
    let spans = s.spans.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);
    for span in &spans {
        assert_eq!(span.status, Status::error("boom"));
        assert_eq!(attr(&span.attributes, attr::INPUT_MESSAGES).len(), 1);
        assert!(attr(&span.attributes, attr::OUTPUT_MESSAGES).is_empty());
        assert_eq!(
            attr(&span.attributes, attr::CONTENT_TRUNCATED_ATTRIBUTES),
            [&string_array([attr::OUTPUT_MESSAGES.to_owned()])]
        );
    }
    let logs = s.logs.get_emitted_logs().unwrap();
    for log in &logs {
        let keys: Vec<&str> = log
            .record
            .attributes_iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert!(!keys.contains(&attr::OUTPUT_MESSAGES), "{keys:?}");
        assert!(!keys.contains(&attr::EXCEPTION_MESSAGE), "{keys:?}");
    }
    // Two exception events and two details events, each with the input.
    assert_eq!(logs.len(), 4);
}

/// An application span processor that panics in `on_end` for spans named
/// `chat boom`, and counts every span it sees.
#[derive(Debug, Default)]
struct PanicsOnEnd(Arc<Mutex<Vec<String>>>);

impl opentelemetry_sdk::trace::SpanProcessor for PanicsOnEnd {
    fn on_start(&self, _span: &mut opentelemetry_sdk::trace::Span, _cx: &Context) {}

    fn on_end(&self, span: opentelemetry_sdk::trace::SpanData) {
        self.0.lock().unwrap().push(span.name.to_string());
        assert!(span.name != "chat boom", "application processor panicked");
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }
}

#[test]
fn an_application_span_processor_panic_in_on_end_is_contained() {
    // Ending an operation whose span an application processor
    // panics on returns normally; the operation's metrics are recorded; the
    // processor and every later span keep working. A processor registered
    // after the panicking one does not see that span.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let memory = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(PanicsOnEnd(seen.clone()))
        .with_simple_exporter(memory.clone())
        .build();
    let metrics = InMemoryMetricExporter::default();
    let meter = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(metrics.clone()).build())
        .build();
    let obs = Observer::builder()
        .tracer_provider(&provider)
        .meter_provider(&meter)
        .build();
    let request = |model: &str| InferenceRequest::new("chat", "llama.cpp").with_model(model);
    obs.start_inference(request("before"), StartOptions::default())
        .end();
    let boom = obs.start_inference(request("boom"), StartOptions::default());
    boom.end();
    assert!(boom.is_finished());
    // Dropping a handle ends it too (cancelled), through the same path.
    drop(obs.start_inference(request("boom"), StartOptions::default()));
    obs.start_inference(request("after"), StartOptions::default())
        .end();

    assert_eq!(
        *seen.lock().unwrap(),
        ["chat before", "chat boom", "chat boom", "chat after"]
    );
    let exported: Vec<String> = memory
        .get_finished_spans()
        .unwrap()
        .into_iter()
        .map(|s| s.name.into_owned())
        .collect();
    assert_eq!(exported, ["chat before", "chat after"]);
    meter.force_flush().unwrap();
    let durations: u64 = points(&metrics, metric::INFERENCE_DURATION)
        .iter()
        .map(|(_, count)| count)
        .sum();
    assert_eq!(durations, 4, "every operation recorded its duration");
}

#[test]
fn plain_text_is_charged_at_its_escaped_length_against_the_retained_budget() {
    // A prompt variable's span value is plain text, which
    // OTLP/JSON escapes; it is charged as written there. 16 + 24
    // ("gen_ai.prompt.variable.a") + 100 fits twice in 300; written as JSON
    // 100 quotes take 200 bytes, so only the first variable fits.
    for (value, kept) in [("x".repeat(100), 2), ("\"".repeat(100), 1)] {
        let s = signals(Sampler::AlwaysOn);
        let obs = observer(
            &s,
            ContentPolicy::default()
                .with_capture(ContentCapture::SpanOnly)
                .with_max_retained_operation_bytes(300),
        );
        let mut request = chat();
        request.prompt = Some(crate::PromptRef {
            name: "p".into(),
            variables: [("a".to_owned(), value.clone()), ("b".to_owned(), value)].into(),
            ..Default::default()
        });
        obs.start_inference(request, StartOptions::default()).end();
        let spans = s.spans.get_finished_spans().unwrap();
        let variables = spans[0]
            .attributes
            .iter()
            .filter(|kv| kv.key.as_str().starts_with(attr::PROMPT_VARIABLE_PREFIX))
            .count();
        assert_eq!(variables, kept);
    }
}

#[test]
fn a_capped_observer_records_through_the_same_providers_with_less_content() {
    // The capped observer exports to the providers of
    // the observer it came from, and captures only what both policies allow.
    let s = signals(Sampler::AlwaysOn);
    let metrics = InMemoryMetricExporter::default();
    let meter = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(metrics.clone()).build())
        .build();
    let full = Observer::builder()
        .tracer_provider(&s.tracer)
        .logger_provider(&s.logger)
        .meter_provider(&meter)
        .content_policy(ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent))
        .build();
    let run = |obs: &Observer, text: &str| {
        let op = obs.start_inference(chat(), StartOptions::default());
        op.record_input_messages(&[InputMessage::text("user", text)]);
        op.end();
    };
    run(&full, "full content");
    // Spans only, under a ceiling that captures spans only.
    let spans_only =
        full.content_capped_by(&ContentPolicy::default().with_capture(ContentCapture::SpanOnly));
    assert_eq!(
        spans_only.content_policy().capture(),
        ContentCapture::SpanOnly
    );
    run(&spans_only, "span content");
    // Nothing, under a metadata-only ceiling.
    let none = full.content_capped_by(&ContentPolicy::metadata_only());
    run(&none, "no content");
    // A ceiling that captures more raises nothing.
    let raised = none
        .content_capped_by(&ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent));
    run(&raised, "still no content");

    let spans = s.spans.get_finished_spans().unwrap();
    let inputs: Vec<_> = spans
        .iter()
        .map(|span| {
            attr(&span.attributes, attr::INPUT_MESSAGES)
                .first()
                .map(|v| v.to_string())
        })
        .collect();
    assert_eq!(spans.len(), 4, "every observer exports through one tracer");
    assert!(inputs[0]
        .as_deref()
        .is_some_and(|v| v.contains("full content")));
    assert!(inputs[1]
        .as_deref()
        .is_some_and(|v| v.contains("span content")));
    assert_eq!(inputs[2..], [None, None]);
    let details: Vec<String> = s
        .logs
        .get_emitted_logs()
        .unwrap()
        .iter()
        .filter(|l| l.record.event_name() == Some(event::INFERENCE_DETAILS))
        .map(|l| format!("{:?}", l.record))
        .collect();
    assert_eq!(details.len(), 1, "only the uncapped run emits details");
    assert!(details[0].contains("full content"));
    meter.force_flush().unwrap();
    assert_eq!(
        points(&metrics, metric::INFERENCE_DURATION),
        [(None, 4)],
        "every observer records through one meter"
    );
}
