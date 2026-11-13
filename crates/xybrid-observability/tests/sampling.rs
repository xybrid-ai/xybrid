//! Spans the sampler drops, through the real OTLP/HTTP transport
//! (reliability C2, C3): a dropped operation exports no span, yet its
//! duration, outcome and usage reach `/v1/metrics` with exact counts, and
//! evaluations and feedback recorded against its reference reach `/v1/logs`
//! linked to it (trace flags `0`, CONTRACT.md "Observation references").
//!
//! Each workload also ends one sampled anchor operation, so the traces
//! endpoint is proven to export: what is missing from it was dropped, not
//! lost.
#![cfg(feature = "otlp")]

mod support;

use std::time::Duration;

use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::common::v1::any_value::Value as Wire;
use opentelemetry_proto::tonic::logs::v1::LogRecord;
use opentelemetry_proto::tonic::metrics::v1::metric::Data;
use opentelemetry_proto::tonic::metrics::v1::number_data_point::Value as Number;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
use support::Server;
use xybrid_observability::otlp::{exporters, ObservabilityPipeline, Protocol};
use xybrid_observability::semconv::{attr, event, metric};
use xybrid_observability::{
    AttemptInfo, AttemptReason, ErrorInfo, Evaluation, Feedback, InferenceRequest, Location,
    ObservationContext, ObservationRef, Observer, StartOptions, ToolCallInfo, Usage,
};

/// Bound of every flush: a hang detector, far above a loopback export.
const FLUSH: Duration = Duration::from_secs(20);

/// The trace W3C `traceparent`s below belong to.
const TRACE: &str = "4bf92f3577b34da6a3ce929d0e0e4736";

fn chat_request() -> InferenceRequest {
    InferenceRequest::new("chat", "llama.cpp")
        .with_model("tiny")
        .with_location(Location::Local)
}

/// `(name, trace id, span id)` of every span the endpoint received, in
/// arrival and body order.
fn exported_spans(server: &Server) -> Vec<(String, String, String)> {
    server
        .requests_to("/v1/traces")
        .iter()
        .flat_map(|r| r.traces().resource_spans)
        .flat_map(|r| r.scope_spans)
        .flat_map(|s| s.spans)
        .map(|s| (s.name, hex(&s.trace_id), hex(&s.span_id)))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The latest metrics body (temporality is cumulative, so it holds every
/// point recorded so far).
fn last_metrics(server: &Server) -> ExportMetricsServiceRequest {
    server
        .requests_to("/v1/metrics")
        .last()
        .expect("a metrics export")
        .metrics()
}

/// Every point of the metric `name` as its `error.type` and its count
/// (histograms) or integer value (counters), sorted.
fn points(request: &ExportMetricsServiceRequest, name: &str) -> Vec<(Option<String>, u64)> {
    let error_type = |attributes: &[opentelemetry_proto::tonic::common::v1::KeyValue]| {
        attributes
            .iter()
            .find(|kv| kv.key == attr::ERROR_TYPE)
            .and_then(|kv| match kv.value.as_ref()?.value.as_ref()? {
                Wire::StringValue(s) => Some(s.clone()),
                other => panic!("error.type is a string, not {other:?}"),
            })
    };
    let mut out = Vec::new();
    for m in request
        .resource_metrics
        .iter()
        .flat_map(|r| &r.scope_metrics)
        .flat_map(|s| &s.metrics)
        .filter(|m| m.name == name)
    {
        match m.data.as_ref().expect("metric data") {
            Data::Histogram(h) => out.extend(
                h.data_points
                    .iter()
                    .map(|p| (error_type(&p.attributes), p.count)),
            ),
            Data::Sum(s) => out.extend(s.data_points.iter().map(|p| match p.value {
                Some(Number::AsInt(v)) => (
                    error_type(&p.attributes),
                    u64::try_from(v).expect("a count"),
                ),
                other => panic!("{name} counts integers, not {other:?}"),
            })),
            other => panic!("{name}: unexpected data {other:?}"),
        }
    }
    out.sort();
    out
}

/// The sum of every point of the counter `name`.
fn total(request: &ExportMetricsServiceRequest, name: &str) -> u64 {
    points(request, name).iter().map(|(_, v)| v).sum()
}

/// Every `gen_ai.evaluation.result` record the endpoint received.
fn evaluations(server: &Server) -> Vec<LogRecord> {
    server
        .requests_to("/v1/logs")
        .iter()
        .flat_map(|r| r.logs().resource_logs)
        .flat_map(|r| r.scope_logs)
        .flat_map(|s| s.log_records)
        .filter(|r| r.event_name == event::EVALUATION_RESULT)
        .collect()
}

/// The `gen_ai.evaluation.name` of an evaluation record.
fn evaluation_name(record: &LogRecord) -> String {
    record
        .attributes
        .iter()
        .find(|kv| kv.key == attr::EVALUATION_NAME)
        .and_then(|kv| match kv.value.as_ref()?.value.as_ref()? {
            Wire::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        .expect("an evaluation name")
}

/// Asserts the evaluation records are exactly `expected` (name, target),
/// each linked to its target with trace flags `0`.
fn assert_linked_evaluations(server: &Server, expected: &[(&str, &ObservationRef)]) {
    let mut received: Vec<_> = evaluations(server)
        .iter()
        .map(|r| {
            assert_eq!(r.flags & 0xff, 0, "a reference carries no flags: {r:?}");
            (evaluation_name(r), hex(&r.trace_id), hex(&r.span_id))
        })
        .collect();
    received.sort();
    let mut expected: Vec<_> = expected
        .iter()
        .map(|(name, target)| {
            (
                (*name).to_owned(),
                target.trace_id.clone(),
                target.span_id.clone(),
            )
        })
        .collect();
    expected.sort();
    assert_eq!(received, expected);
}

fn attempt_info() -> AttemptInfo {
    AttemptInfo {
        index: 0,
        reason: AttemptReason::Initial,
        provider: "llama.cpp".into(),
        model: Some("tiny".into()),
        location: Location::Local,
        runtime: None,
        server_address: None,
        server_port: None,
    }
}

#[test]
fn sampler_dropped_operations_export_exact_metrics_and_linked_evaluations() {
    // C2: an application tracer whose root sampler keeps nothing
    // (ParentBased(TraceIdRatioBased(0))) exports spans through the paired
    // OTLP span exporter; metrics and logs go through the pipeline's own
    // providers. The packaged pipeline itself samples ParentBased(AlwaysOn)
    // and takes no sampler, so this is how a ratio drop reaches the transport.
    let server = Server::ok();
    let protocol = Protocol::HttpProtobuf;
    let pipeline = ObservabilityPipeline::build(server.all_config(protocol).with_traces(None))
        .expect("metrics and logs pipeline");
    let span_exporter = exporters(&server.traces_config(protocol))
        .expect("paired exporters")
        .traces
        .expect("a traces exporter");
    let tracer = SdkTracerProvider::builder()
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
            0.0,
        ))))
        .with_simple_exporter(span_exporter)
        .build();
    let observer = Observer::builder()
        .tracer_provider(&tracer)
        .meter_provider(pipeline.meter_provider().expect("metrics"))
        .logger_provider(pipeline.logger_provider().expect("logs"))
        .build();

    // Five roots the ratio drops: three succeed with usage, one fails, one
    // is cancelled.
    let mut dropped = Vec::new();
    for outcome in ["ok", "ok", "ok", "network", "cancelled"] {
        let op = observer.start_inference(chat_request(), StartOptions::default());
        let cx = op.context().expect("a dropped span keeps its context");
        assert!(!cx.is_sampled(), "{outcome}: the ratio sampler dropped it");
        let reference = op
            .observation_ref()
            .expect("a dropped span keeps its reference");
        assert_eq!(reference.span_id, cx.span_id(), "{outcome}");
        assert_eq!(reference.trace_id, cx.trace_id(), "{outcome}");
        match outcome {
            "ok" => {
                op.record_usage(&Usage::tokens(Some(2), Some(4)));
                op.end();
            }
            "network" => op.end_with_error(&ErrorInfo::new("network")),
            _ => op.cancel(),
        }
        dropped.push(reference);
    }
    assert!(observer.record_evaluation(
        &dropped[0],
        &Evaluation {
            name: "relevance".into(),
            score_value: Some(0.5),
            ..Evaluation::default()
        }
    ));
    assert!(observer.record_feedback(
        &dropped[3],
        &Feedback {
            score: Some(1.0),
            ..Feedback::default()
        }
    ));

    // A sampled anchor: the traces endpoint exports what the sampler keeps.
    let anchor_parent =
        ObservationContext::from_traceparent(&format!("00-{TRACE}-00f067aa0ba902b7-01"), None)
            .unwrap();
    let anchor = observer.start_tool(
        ToolCallInfo::new("anchor"),
        StartOptions::child_of(anchor_parent),
    );
    let anchor_ref = anchor.observation_ref().expect("sampled anchor");
    anchor.end();

    tracer.force_flush().expect("span flush");
    pipeline.force_flush(FLUSH).expect("metrics and logs flush");

    let spans = exported_spans(&server);
    assert_eq!(
        spans,
        [(
            "execute_tool anchor".to_owned(),
            anchor_ref.trace_id.clone(),
            anchor_ref.span_id.clone()
        )],
        "only the sampled anchor reaches /v1/traces"
    );

    let metrics = last_metrics(&server);
    assert_eq!(
        points(&metrics, metric::INFERENCE_DURATION),
        [
            (None, 3),
            (Some("cancelled".to_owned()), 1),
            (Some("network".to_owned()), 1),
        ],
        "every dropped inference is counted once, by outcome"
    );
    assert_eq!(total(&metrics, metric::USAGE_INPUT_TOKENS), 6);
    assert_eq!(total(&metrics, metric::USAGE_OUTPUT_TOKENS), 12);
    assert_eq!(
        points(&metrics, metric::OPERATION_INPUT_TOKENS),
        [(None, 3)],
        "one usage record per successful inference"
    );
    assert_eq!(points(&metrics, metric::TOOL_DURATION), [(None, 1)]);

    assert_linked_evaluations(
        &server,
        &[("relevance", &dropped[0]), ("user_feedback", &dropped[3])],
    );

    // Spans go through the application's tracer, not the pipeline: the
    // anchor-only /v1/traces body above is their check.
    let stats = pipeline.stats();
    for (signal, s) in [("logs", stats.logs), ("metrics", stats.metrics)] {
        assert!(s.exported > 0, "{signal}: {s:?}");
        assert_eq!((s.failed, s.dropped, s.pending), (0, 0, 0), "{signal}");
    }
    tracer.shutdown().expect("tracer shutdown");
    pipeline.shutdown(FLUSH).expect("pipeline shutdown");
}

#[test]
fn unsampled_parent_exports_exact_metrics_and_linked_evaluations_but_no_spans() {
    // C3: the packaged pipeline (ParentBased(AlwaysOn)) under an incoming
    // traceparent with sampled flag 0. The inference, its attempt and its
    // tool child are dropped; metrics count them exactly and evaluations and
    // feedback on their references reach /v1/logs.
    let server = Server::ok();
    let pipeline = ObservabilityPipeline::build(server.all_config(Protocol::HttpProtobuf)).unwrap();
    let observer = pipeline.observer();
    let parent =
        ObservationContext::from_traceparent(&format!("00-{TRACE}-00f067aa0ba902b7-00"), None)
            .unwrap();
    assert!(!parent.is_sampled());

    let chat = observer.start_inference(chat_request(), StartOptions::child_of(parent.clone()));
    let cx = chat.context().expect("unsampled context");
    assert!(!cx.is_sampled());
    let chat_ref = chat.observation_ref().expect("unsampled reference");
    assert_eq!(chat_ref.trace_id, TRACE, "in the incoming trace");
    assert_ne!(chat_ref.span_id, parent.span_id(), "a span of its own");

    let attempt = chat.start_attempt(attempt_info());
    let attempt_ref = attempt.observation_ref().expect("attempt reference");
    assert_eq!(attempt_ref.trace_id, TRACE);
    attempt.end();
    let tool = observer.start_tool(ToolCallInfo::new("lookup"), StartOptions::child_of(cx));
    let tool_ref = tool.observation_ref().expect("tool reference");
    assert_eq!(tool_ref.trace_id, TRACE);
    tool.end();
    chat.record_usage(&Usage::tokens(Some(2), Some(4)));
    chat.end();

    assert!(observer.record_evaluation(
        &chat_ref,
        &Evaluation {
            name: "relevance".into(),
            score_value: Some(0.5),
            ..Evaluation::default()
        }
    ));
    assert!(observer.record_feedback(
        &tool_ref,
        &Feedback {
            score: Some(1.0),
            ..Feedback::default()
        }
    ));

    // A sampled root anchor: the traces endpoint exports what is sampled.
    let anchor = observer.start_workflow("anchor", StartOptions::default());
    let anchor_ref = anchor.observation_ref().expect("sampled anchor");
    anchor.end();

    pipeline.force_flush(FLUSH).expect("flush");

    assert_eq!(
        exported_spans(&server),
        [(
            "invoke_workflow anchor".to_owned(),
            anchor_ref.trace_id.clone(),
            anchor_ref.span_id.clone()
        )],
        "nothing under the unsampled parent reaches /v1/traces"
    );
    let stats = pipeline.stats();
    assert_eq!(
        (stats.traces.accepted, stats.traces.exported),
        (1, 1),
        "only the anchor is enqueued"
    );

    let metrics = last_metrics(&server);
    let once = [(None, 1)];
    assert_eq!(points(&metrics, metric::INFERENCE_DURATION), once);
    assert_eq!(points(&metrics, metric::ATTEMPTS), once);
    assert_eq!(points(&metrics, metric::TOOL_DURATION), once);
    assert_eq!(points(&metrics, metric::WORKFLOW_DURATION), once);
    assert_eq!(total(&metrics, metric::USAGE_INPUT_TOKENS), 2);
    assert_eq!(total(&metrics, metric::USAGE_OUTPUT_TOKENS), 4);

    assert_linked_evaluations(
        &server,
        &[("relevance", &chat_ref), ("user_feedback", &tool_ref)],
    );
    for (signal, s) in [("logs", stats.logs), ("metrics", stats.metrics)] {
        assert!(s.exported > 0, "{signal}: {s:?}");
        assert_eq!((s.failed, s.dropped, s.pending), (0, 0, 0), "{signal}");
    }
    pipeline.shutdown(FLUSH).expect("shutdown");
}
