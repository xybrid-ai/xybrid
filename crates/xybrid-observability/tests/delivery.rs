//! Delivery regressions over real HTTP: redirects (R-CFG-6), splitting by
//! final encoded size (R-BATCH-3..5), one absolute deadline per export
//! (R-DEADLINE-1..5), counters (R-CNT-1..6) and the byte budget (R-BUF-2).
#![cfg(feature = "otlp")]

mod support;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use opentelemetry::logs::{AnyValue, LogRecord as _, Logger as _, LoggerProvider as _};
use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
use opentelemetry::InstrumentationScope;
use opentelemetry_proto::tonic::collector::logs::v1::{
    ExportLogsPartialSuccess, ExportLogsServiceResponse,
};
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTracePartialSuccess, ExportTraceServiceResponse,
};
use opentelemetry_proto::tonic::common::v1::any_value::Value as Wire;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::logs::{LogBatch, LogExporter, SdkLoggerProvider};
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider, Temporality};
use prost::Message;
use support::{Action, Server};
use xybrid_observability::otlp::{
    exporters, BatchLimits, ObservabilityPipeline, OtlpConfig, OtlpLogExporter, OtlpMetricExporter,
    PipelineStats, Protocol, RetryLimits, SignalStats,
};
use xybrid_observability::{
    ContentCapture, ContentPolicy, Error, InferenceRequest, KeyValue, Location, Observer,
    StartOptions, Usage,
};

const PROTOCOLS: [Protocol; 2] = [Protocol::HttpProtobuf, Protocol::HttpJson];

fn fast_retry(max_retries: usize) -> RetryLimits {
    RetryLimits {
        max_retries,
        initial_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(20),
    }
}

fn limits(export_timeout: Duration) -> BatchLimits {
    BatchLimits {
        scheduled_delay: Duration::from_secs(60),
        export_timeout,
        ..BatchLimits::default()
    }
}

/// `n` spans of the application scope `app` with a `payload` attribute.
fn spans(pipeline: &ObservabilityPipeline, n: usize, payload: usize) {
    let tracer = pipeline.tracer_provider().unwrap().tracer("app");
    for i in 0..n {
        let mut span = tracer.start(format!("span-{i}"));
        span.set_attribute(KeyValue::new("payload", "x".repeat(payload)));
        span.end();
    }
}

fn logs(pipeline: &ObservabilityPipeline, names: &[String], payload: usize) {
    let logger = pipeline.logger_provider().unwrap().logger("app");
    for name in names {
        let mut record = logger.create_log_record();
        record.set_body(AnyValue::from(name.clone()));
        record.add_attribute("payload", "x".repeat(payload));
        logger.emit(record);
    }
}

fn inference(observer: &Observer) {
    let op = observer.start_inference(
        InferenceRequest::new("chat", "fake")
            .with_model("m")
            .with_location(Location::Local),
        StartOptions::default(),
    );
    op.record_usage(&Usage::tokens(Some(2), Some(3)));
    op.end();
}

fn delivery(result: &xybrid_observability::Result<()>) -> Option<(u64, u64)> {
    match result {
        Err(Error::Delivery {
            failed, dropped, ..
        }) => Some((*failed, *dropped)),
        _ => None,
    }
}

#[test]
fn redirects_are_never_followed() {
    // R-CFG-6: 3xx is a non-retryable failure; nothing goes to Location.
    for status in [301, 302, 307, 308] {
        let server = Server::start(move |r| {
            if r.path == "/v1/traces" {
                Action::status(status).header("Location", "/elsewhere")
            } else {
                Action::ok()
            }
        });
        let pipeline = ObservabilityPipeline::build(
            server
                .traces_config(Protocol::HttpProtobuf)
                .with_retry(fast_retry(2)),
        )
        .unwrap();
        spans(&pipeline, 2, 10);
        let flushed = pipeline.force_flush(Duration::from_secs(3));
        assert_eq!(delivery(&flushed), Some((2, 0)), "{status}: {flushed:?}");
        thread::sleep(Duration::from_millis(50));
        assert_eq!(server.requests_to("/elsewhere").len(), 0, "{status}");
        assert_eq!(server.count(), 1, "{status}: never retried");
        let stats = pipeline.stats().traces;
        assert_eq!((stats.failed, stats.exported, stats.pending), (2, 0, 0));
    }
}

#[test]
fn log_batches_split_by_encoded_size_and_drop_only_oversized_singletons() {
    // R-BATCH-3 (R-BATCH-1/2 for logs).
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let mut config = server
            .traces_config(protocol)
            .with_traces(None)
            .with_logs(Some(server.endpoint("logs", protocol)));
        config.max_request_body_bytes = 1800;
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        let names: Vec<String> = (0..9).map(|i| format!("log-{i}")).collect();
        logs(&pipeline, &names, 300);
        logs(&pipeline, &["oversized".to_owned()], 5000);
        let flushed = pipeline.force_flush(Duration::from_secs(3));
        assert_eq!(delivery(&flushed), Some((0, 1)), "{protocol:?}");

        let requests = server.requests_to("/v1/logs");
        assert!(requests.len() > 1, "{protocol:?}: split");
        let mut bodies = Vec::new();
        for request in &requests {
            assert!(request.body.len() <= 1800, "{}", request.body.len());
            let decoded = request.logs();
            assert_eq!(decoded.resource_logs.len(), 1);
            let resource = &decoded.resource_logs[0];
            assert!(resource
                .resource
                .as_ref()
                .unwrap()
                .attributes
                .iter()
                .any(|kv| kv.key == "service.name"));
            assert_eq!(resource.scope_logs.len(), 1);
            assert_eq!(resource.scope_logs[0].scope.as_ref().unwrap().name, "app");
            for record in &resource.scope_logs[0].log_records {
                let Some(Wire::StringValue(body)) =
                    record.body.as_ref().and_then(|b| b.value.clone())
                else {
                    panic!("body")
                };
                bodies.push(body);
            }
        }
        assert_eq!(bodies, names, "{protocol:?}: every log, in order");
        let stats = pipeline.stats().logs;
        assert_eq!((stats.exported, stats.dropped, stats.failed), (9, 1, 0));
        assert!(stats.is_consistent());
    }
}

#[test]
fn two_scope_batches_keep_their_grouping_in_every_part() {
    // R-BATCH-5.
    for protocol in PROTOCOLS {
        let server = Server::ok();
        // Small enough to split ten spans, large enough for parts of several
        // spans (OTLP/JSON is roughly twice as large).
        let bound = match protocol {
            Protocol::HttpProtobuf => 2000,
            Protocol::HttpJson => 3500,
        };
        let mut config = server.traces_config(protocol);
        config.max_request_body_bytes = bound;
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        let provider = pipeline.tracer_provider().unwrap();
        let tracers = [provider.tracer("scope-a"), provider.tracer("scope-b")];
        for i in 0..10 {
            let prefix = if i % 2 == 0 { "a" } else { "b" };
            let mut span = tracers[i % 2].start(format!("{prefix}-{i}"));
            span.set_attribute(KeyValue::new("payload", "x".repeat(300)));
            span.end();
        }
        pipeline.force_flush(Duration::from_secs(3)).unwrap();

        let requests = server.requests_to("/v1/traces");
        assert!(requests.len() > 1, "{protocol:?}: split");
        let (mut a, mut b, mut mixed) = (Vec::new(), Vec::new(), 0);
        for request in &requests {
            assert!(request.body.len() <= bound);
            let decoded = request.traces();
            assert_eq!(decoded.resource_spans.len(), 1);
            assert!(!decoded.resource_spans[0]
                .resource
                .as_ref()
                .unwrap()
                .attributes
                .is_empty());
            let groups = &decoded.resource_spans[0].scope_spans;
            let mut scopes: Vec<_> = groups
                .iter()
                .map(|g| g.scope.as_ref().unwrap().name.clone())
                .collect();
            let count = scopes.len();
            scopes.dedup();
            assert_eq!(scopes.len(), count, "one group per scope in a part");
            if count == 2 {
                mixed += 1;
            }
            for group in groups {
                let scope = &group.scope.as_ref().unwrap().name;
                for span in &group.spans {
                    match scope.as_str() {
                        "scope-a" => {
                            assert!(span.name.starts_with("a-"));
                            a.push(span.name.clone());
                        }
                        "scope-b" => {
                            assert!(span.name.starts_with("b-"));
                            b.push(span.name.clone());
                        }
                        other => panic!("unexpected scope {other}"),
                    }
                }
            }
        }
        assert!(mixed > 0, "{protocol:?}: some part carries both scopes");
        assert_eq!(a, ["a-0", "a-2", "a-4", "a-6", "a-8"]);
        assert_eq!(b, ["b-1", "b-3", "b-5", "b-7", "b-9"]);
        assert_eq!(pipeline.stats().traces.exported, 10);
    }
}

#[test]
fn oversized_metrics_collections_fail_without_a_request() {
    // R-BATCH-4.
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let mut config = server
            .traces_config(protocol)
            .with_traces(None)
            .with_metrics(Some(server.endpoint("metrics", protocol)));
        config.max_request_body_bytes = 200;
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        inference(&pipeline.observer());
        let flushed = pipeline.force_flush(Duration::from_secs(3));
        let (failed, dropped) = delivery(&flushed).expect("the flush reports the failure");
        assert!(failed > 0 && dropped == 0, "{protocol:?}: {flushed:?}");
        let stats = pipeline.stats().metrics;
        assert_eq!(stats.failed, failed);
        assert_eq!((stats.exported, stats.pending), (0, 0));
        assert_eq!(stats.accepted, stats.failed);
        assert_eq!(server.count(), 0, "{protocol:?}: nothing sent");
    }
}

#[test]
fn a_response_after_the_deadline_is_aborted_and_fails() {
    // R-DEADLINE-1: 200 arrives only after the export deadline.
    let server = Server::always(Action::ok().after(Duration::from_secs(3)));
    let pipeline = ObservabilityPipeline::build(
        server
            .traces_config(Protocol::HttpProtobuf)
            .with_batch(limits(Duration::from_millis(300))),
    )
    .unwrap();
    spans(&pipeline, 2, 10);
    let started = Instant::now();
    let flushed = pipeline.force_flush(Duration::from_secs(3));
    let elapsed = started.elapsed();
    assert_eq!(delivery(&flushed), Some((2, 0)), "{flushed:?}");
    assert!(elapsed < Duration::from_millis(1500), "{elapsed:?}");
    let request = server.requests()[0].clone();
    let disconnect = server.wait_disconnect(0, Duration::from_secs(2));
    assert!(
        disconnect.duration_since(request.at) < Duration::from_secs(1),
        "the request was aborted at the deadline"
    );
    assert!(server.response_of(0).is_none());
    assert_eq!(server.count(), 1, "no time left to retry");
    let stats = pipeline.stats().traces;
    assert_eq!((stats.failed, stats.pending), (2, 0));
}

#[test]
fn split_siblings_are_not_sent_after_the_deadline() {
    // R-DEADLINE-2.
    let server = Server::sequence(vec![Action::Hang, Action::ok()]);
    let mut config = server
        .traces_config(Protocol::HttpProtobuf)
        .with_batch(limits(Duration::from_millis(300)));
    config.max_request_body_bytes = 1800;
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    spans(&pipeline, 6, 300);
    let flushed = pipeline.force_flush(Duration::from_secs(3));
    assert_eq!(delivery(&flushed), Some((6, 0)), "{flushed:?}");
    thread::sleep(Duration::from_millis(300));
    assert_eq!(server.count(), 1, "no part was sent after the deadline");
    let stats = pipeline.stats().traces;
    assert_eq!((stats.failed, stats.exported, stats.dropped), (6, 0, 0));
}

#[test]
fn retry_after_beyond_the_deadline_fails_at_once() {
    // R-DEADLINE-3, first half.
    let server = Server::always(Action::status(503).header("Retry-After", "5"));
    let pipeline = ObservabilityPipeline::build(
        server
            .traces_config(Protocol::HttpProtobuf)
            .with_batch(limits(Duration::from_secs(2))),
    )
    .unwrap();
    spans(&pipeline, 1, 10);
    let started = Instant::now();
    let flushed = pipeline.force_flush(Duration::from_secs(5));
    assert_eq!(delivery(&flushed), Some((1, 0)));
    assert!(started.elapsed() < Duration::from_millis(800));
    assert_eq!(server.count(), 1);
}

#[test]
fn retry_after_above_max_delay_is_honoured_within_the_deadline() {
    // R-DEADLINE-3, second half: Retry-After 3 s, max_delay 2 s, deadline 10 s.
    let server = Server::sequence(vec![
        Action::status(503).header("Retry-After", "3"),
        Action::ok(),
    ]);
    let pipeline = ObservabilityPipeline::build(
        server
            .traces_config(Protocol::HttpProtobuf)
            .with_batch(limits(Duration::from_secs(10)))
            .with_retry(RetryLimits {
                max_retries: 2,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(2),
            }),
    )
    .unwrap();
    spans(&pipeline, 1, 10);
    pipeline.force_flush(Duration::from_secs(15)).unwrap();
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    let gap = requests[1].at.duration_since(requests[0].at);
    assert!(
        gap >= Duration::from_millis(2900) && gap < Duration::from_millis(4500),
        "{gap:?}"
    );
    assert_eq!(pipeline.stats().traces.exported, 1);
}

#[test]
fn a_2xx_with_a_slow_or_oversized_body_counts_exported() {
    // R-DEADLINE-4: the status decides; the body read stops at the deadline
    // and at 64 KiB.
    let server = Server::always(Action::Trickle {
        status: 200,
        total: 1000,
        chunk: 10,
        every: Duration::from_millis(50),
    });
    let pipeline = ObservabilityPipeline::build(
        server
            .traces_config(Protocol::HttpJson)
            .with_batch(limits(Duration::from_millis(500))),
    )
    .unwrap();
    spans(&pipeline, 2, 10);
    let started = Instant::now();
    pipeline.force_flush(Duration::from_secs(5)).unwrap();
    assert!(started.elapsed() < Duration::from_millis(2000));
    let stats = pipeline.stats().traces;
    assert_eq!((stats.exported, stats.failed), (2, 0));

    // A partial success the client would honour if it read it whole...
    let partial = |message: usize| {
        format!(
            r#"{{"partialSuccess":{{"rejectedSpans":"1","errorMessage":"{}"}}}}"#,
            "m".repeat(message)
        )
    };
    for (message, exported, failed) in [(1024, 1, 1), (70 * 1024, 2, 0)] {
        let server = Server::always(
            Action::ok()
                .header("Content-Type", "application/json")
                .body(partial(message)),
        );
        let pipeline =
            ObservabilityPipeline::build(server.traces_config(Protocol::HttpJson)).unwrap();
        spans(&pipeline, 2, 10);
        let _ = pipeline.force_flush(Duration::from_secs(5));
        let stats = pipeline.stats().traces;
        // ...is ignored once the body exceeds max_response_body_bytes.
        assert_eq!(
            (stats.exported, stats.failed),
            (exported, failed),
            "message of {message} bytes"
        );
        assert_eq!(server.count(), 1);
    }
}

#[test]
fn each_attempt_is_bounded_by_the_signal_timeout_and_the_deadline() {
    // R-DEADLINE-5: attempt timeout = min(signal timeout, remaining).
    let server = Server::sequence(vec![Action::Hang, Action::ok()]);
    let config = server
        .traces_config(Protocol::HttpProtobuf)
        .with_traces(Some(
            server
                .endpoint("traces", Protocol::HttpProtobuf)
                .with_timeout(Duration::from_millis(200)),
        ))
        .with_batch(limits(Duration::from_secs(5)))
        .with_retry(fast_retry(1));
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    spans(&pipeline, 1, 10);
    pipeline.force_flush(Duration::from_secs(5)).unwrap();
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    let aborted = server.wait_disconnect(0, Duration::from_secs(1));
    let attempt = aborted.duration_since(requests[0].at);
    assert!(
        attempt >= Duration::from_millis(150) && attempt < Duration::from_secs(1),
        "signal timeout bounds the attempt: {attempt:?}"
    );
    let gap = requests[1].at.duration_since(requests[0].at);
    assert!(gap < Duration::from_millis(1500), "{gap:?}");
    assert_eq!(pipeline.stats().traces.exported, 1);

    // The other bound: a 10 s signal timeout is clamped to the 300 ms left.
    let server = Server::always(Action::Hang);
    let config = server
        .traces_config(Protocol::HttpProtobuf)
        .with_traces(Some(
            server
                .endpoint("traces", Protocol::HttpProtobuf)
                .with_timeout(Duration::from_secs(10)),
        ))
        .with_batch(limits(Duration::from_millis(300)));
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    spans(&pipeline, 1, 10);
    assert!(pipeline.force_flush(Duration::from_secs(5)).is_err());
    let request = server.requests()[0].clone();
    let aborted = server.wait_disconnect(0, Duration::from_secs(2));
    assert!(aborted.duration_since(request.at) < Duration::from_secs(1));
}

#[test]
fn retryable_statuses_then_success_export_once() {
    // R-CNT-2: 503, 503, 200 -> exported once, three requests.
    let server = Server::sequence(vec![Action::status(503), Action::status(503), Action::ok()]);
    let pipeline = ObservabilityPipeline::build(
        server
            .traces_config(Protocol::HttpProtobuf)
            .with_retry(fast_retry(2)),
    )
    .unwrap();
    spans(&pipeline, 2, 10);
    pipeline.force_flush(Duration::from_secs(5)).unwrap();
    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    assert!(requests.iter().all(|r| r.body == requests[0].body));
    let stats = pipeline.stats().traces;
    assert_eq!((stats.accepted, stats.exported, stats.failed), (2, 2, 0));
}

#[test]
fn lost_acknowledgements_with_and_without_retries() {
    // R-CNT-3: the server reads the body and drops the connection.
    let server = Server::always(Action::Lost);
    let pipeline = ObservabilityPipeline::build(
        server
            .traces_config(Protocol::HttpProtobuf)
            .with_retry(fast_retry(0)),
    )
    .unwrap();
    spans(&pipeline, 2, 10);
    assert_eq!(
        delivery(&pipeline.force_flush(Duration::from_secs(5))),
        Some((2, 0))
    );
    assert_eq!(server.count(), 1);
    assert_eq!(pipeline.stats().traces.failed, 2);

    let server = Server::sequence(vec![Action::Lost, Action::ok()]);
    let pipeline = ObservabilityPipeline::build(
        server
            .traces_config(Protocol::HttpProtobuf)
            .with_retry(fast_retry(2)),
    )
    .unwrap();
    spans(&pipeline, 2, 10);
    pipeline.force_flush(Duration::from_secs(5)).unwrap();
    let requests = server.requests();
    assert_eq!(requests.len(), 2, "the endpoint received the batch twice");
    assert_eq!(requests[0].body, requests[1].body);
    let stats = pipeline.stats().traces;
    assert_eq!((stats.exported, stats.failed), (2, 0), "counted once");
}

#[test]
fn partial_success_in_both_encodings_and_signals() {
    // R-CNT-4: rejected items fail, the rest are exported, nothing retried.
    for protocol in PROTOCOLS {
        for signal in ["traces", "logs"] {
            let (content_type, body) = match (protocol, signal) {
                (Protocol::HttpProtobuf, "traces") => (
                    "application/x-protobuf",
                    ExportTraceServiceResponse {
                        partial_success: Some(ExportTracePartialSuccess {
                            rejected_spans: 1,
                            error_message: "no".into(),
                        }),
                    }
                    .encode_to_vec(),
                ),
                (Protocol::HttpProtobuf, _) => (
                    "application/x-protobuf",
                    ExportLogsServiceResponse {
                        partial_success: Some(ExportLogsPartialSuccess {
                            rejected_log_records: 1,
                            error_message: "no".into(),
                        }),
                    }
                    .encode_to_vec(),
                ),
                (_, "traces") => (
                    "application/json",
                    br#"{"partialSuccess":{"rejectedSpans":"1"}}"#.to_vec(),
                ),
                _ => (
                    "application/json",
                    br#"{"partialSuccess":{"rejectedLogRecords":1}}"#.to_vec(),
                ),
            };
            let server =
                Server::always(Action::ok().header("Content-Type", content_type).body(body));
            let mut config = server.traces_config(protocol).with_retry(fast_retry(2));
            if signal == "logs" {
                config = config
                    .with_traces(None)
                    .with_logs(Some(server.endpoint("logs", protocol)));
            }
            let pipeline = ObservabilityPipeline::build(config).unwrap();
            if signal == "logs" {
                let names: Vec<String> = (0..3).map(|i| format!("log-{i}")).collect();
                logs(&pipeline, &names, 10);
            } else {
                spans(&pipeline, 3, 10);
            }
            let flushed = pipeline.force_flush(Duration::from_secs(5));
            assert_eq!(delivery(&flushed), Some((1, 0)), "{protocol:?} {signal}");
            assert_eq!(server.count(), 1, "{protocol:?} {signal}: not retried");
            let stats = pipeline.stats();
            let stats = if signal == "logs" {
                stats.logs
            } else {
                stats.traces
            };
            assert_eq!(
                (stats.exported, stats.failed),
                (2, 1),
                "{protocol:?} {signal}"
            );
        }
    }
}

#[test]
fn the_byte_budget_drops_items_beyond_it() {
    // R-BUF-2 and the byte-budget half of R-CNT-5: with 3 KB items and a
    // 10 000-byte budget, two items fit (queued plus in-flight), the rest
    // are dropped and still counted as accepted.
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let config = server
            .traces_config(protocol)
            .with_logs(Some(server.endpoint("logs", protocol)))
            .with_batch(BatchLimits {
                max_buffered_bytes: 10_000,
                max_export_batch_size: 512,
                scheduled_delay: Duration::from_secs(60),
                ..BatchLimits::default()
            });
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        spans(&pipeline, 5, 3000);
        let names: Vec<String> = (0..5).map(|i| format!("log-{i}")).collect();
        logs(&pipeline, &names, 3000);
        let stats = pipeline.stats();
        for (signal, s) in [("traces", stats.traces), ("logs", stats.logs)] {
            assert_eq!(
                (s.accepted, s.pending, s.dropped),
                (5, 2, 3),
                "{protocol:?} {signal}"
            );
        }
        pipeline.shutdown(Duration::from_secs(5)).unwrap();
        let stats = pipeline.stats();
        for s in [stats.traces, stats.logs] {
            assert_eq!((s.exported, s.dropped, s.pending), (2, 3, 0));
            assert!(s.is_consistent());
        }
        assert_eq!(
            server
                .requests_to("/v1/traces")
                .iter()
                .map(|r| r.span_names().len())
                .sum::<usize>(),
            2
        );
    }
}

#[cfg(feature = "test-util")]
#[test]
fn buffered_bytes_report_the_byte_budget_running_total() {
    // The benchmark diagnostic behind the saturation bench's peak
    // library-owned bytes: against an endpoint that never answers, the
    // admitted items stay counted (queued plus in flight) within the budget,
    // and shutdown releases them.
    let server = Server::always(Action::Hang);
    let config = server
        .traces_config(Protocol::HttpProtobuf)
        .with_logs(Some(server.endpoint("logs", Protocol::HttpProtobuf)))
        .with_batch(BatchLimits {
            max_buffered_bytes: 10_000,
            scheduled_delay: Duration::from_secs(60),
            ..BatchLimits::default()
        });
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    assert_eq!(pipeline.buffered_bytes().traces, 0);
    spans(&pipeline, 5, 3000);
    let names: Vec<String> = (0..5).map(|i| format!("log-{i}")).collect();
    logs(&pipeline, &names, 3000);
    let held = pipeline.buffered_bytes();
    for (signal, bytes) in [("traces", held.traces), ("logs", held.logs)] {
        assert!(bytes > 6000 && bytes <= 10_000, "{signal}: {bytes}");
    }
    assert_eq!(held.metrics, 0, "metrics not configured");
    let _ = pipeline.shutdown(Duration::from_millis(200));
    assert_eq!(
        pipeline.buffered_bytes(),
        xybrid_observability::otlp::BufferedBytes::default()
    );
}

fn large_distinct_scope(index: usize) -> InstrumentationScope {
    InstrumentationScope::builder(format!("scope-{index}-{}", "n".repeat(6000)))
        .with_version("v".repeat(128))
        .with_schema_url(format!("https://scope/{index}/{}", "s".repeat(128)))
        .with_attributes([KeyValue::new("scope.attribute", "a".repeat(512))])
        .build()
}

#[test]
fn large_distinct_span_scopes_saturate_the_retained_byte_budget() {
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let pipeline =
            ObservabilityPipeline::build(server.traces_config(protocol).with_batch(BatchLimits {
                max_buffered_bytes: 10_000,
                scheduled_delay: Duration::from_secs(60),
                ..BatchLimits::default()
            }))
            .unwrap();
        for i in 0..5 {
            pipeline
                .tracer_provider()
                .unwrap()
                .tracer_with_scope(large_distinct_scope(i))
                .start("small")
                .end();
        }
        let stats = pipeline.stats().traces;
        assert_eq!(
            (stats.accepted, stats.pending, stats.dropped),
            (5, 1, 4),
            "{protocol:?} {stats:?}"
        );
        pipeline.shutdown(Duration::from_secs(5)).unwrap();
        let stats = pipeline.stats().traces;
        assert_eq!((stats.exported, stats.pending, stats.dropped), (1, 0, 4));
        assert!(stats.is_consistent());
    }
}

#[test]
fn large_distinct_log_scopes_saturate_even_with_a_short_target_override() {
    for protocol in PROTOCOLS {
        for target in [None, Some("wire-target")] {
            let server = Server::ok();
            let pipeline = ObservabilityPipeline::build(
                server
                    .traces_config(protocol)
                    .with_traces(None)
                    .with_logs(Some(server.endpoint("logs", protocol)))
                    .with_batch(BatchLimits {
                        max_buffered_bytes: 10_000,
                        scheduled_delay: Duration::from_secs(60),
                        ..BatchLimits::default()
                    }),
            )
            .unwrap();
            for i in 0..5 {
                let logger = pipeline
                    .logger_provider()
                    .unwrap()
                    .logger_with_scope(large_distinct_scope(i));
                let mut record = logger.create_log_record();
                record.set_body(AnyValue::from("small"));
                if let Some(target) = target {
                    record.set_target(target);
                }
                logger.emit(record);
            }
            let stats = pipeline.stats().logs;
            assert_eq!(
                (stats.accepted, stats.pending, stats.dropped),
                (5, 1, 4),
                "{protocol:?} {target:?} {stats:?}"
            );
            pipeline.shutdown(Duration::from_secs(5)).unwrap();
            let stats = pipeline.stats().logs;
            assert_eq!((stats.exported, stats.pending, stats.dropped), (1, 0, 4));
            assert!(stats.is_consistent());
        }
    }
}

#[test]
fn standalone_exporters_charge_distinct_scope_metadata_at_admission() {
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanExporter as _};

    for protocol in PROTOCOLS {
        let server = Server::ok();
        let set = exporters(
            &server
                .traces_config(protocol)
                .with_logs(Some(server.endpoint("logs", protocol)))
                .with_batch(BatchLimits {
                    max_buffered_bytes: 10_000,
                    ..BatchLimits::default()
                }),
        )
        .unwrap();
        let memory = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(memory.clone())
            .build();
        for i in 0..5 {
            provider
                .tracer_with_scope(large_distinct_scope(i))
                .start("small")
                .end();
        }
        let logger_provider = SdkLoggerProvider::builder().build();
        let logger = logger_provider.logger("app");
        let records: Vec<_> = (0..5)
            .map(|i| {
                let mut record = logger.create_log_record();
                record.set_target("short-target");
                (record, large_distinct_scope(i))
            })
            .collect();
        let batch: Vec<_> = records.iter().map(|(r, s)| (r, s)).collect();
        let spans = set.traces.unwrap();
        let logs = set.logs.unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        assert!(runtime
            .block_on(spans.export(memory.get_finished_spans().unwrap()))
            .is_err());
        assert!(runtime
            .block_on(logs.export(LogBatch::new(&batch)))
            .is_err());
        for stats in [spans.stats(), logs.stats()] {
            assert_eq!(
                (stats.accepted, stats.dropped, stats.pending),
                (5, 5, 0),
                "{protocol:?} {stats:?}"
            );
            assert!(stats.is_consistent());
        }
        assert_eq!(
            server.count(),
            0,
            "over-budget batches never reach transport"
        );
        spans.shutdown_with_timeout(Duration::from_secs(1)).unwrap();
        logs.shutdown_with_timeout(Duration::from_secs(1)).unwrap();
    }
}

/// Lets a test keep reading a standalone exporter's counters after the SDK
/// took ownership of it.
#[derive(Debug)]
struct Shared<E>(Arc<E>);

impl LogExporter for Shared<OtlpLogExporter> {
    async fn export(&self, batch: LogBatch<'_>) -> OTelSdkResult {
        self.0.export(batch).await
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.0.shutdown_with_timeout(timeout)
    }
}

impl PushMetricExporter for Shared<OtlpMetricExporter> {
    async fn export(&self, metrics: &ResourceMetrics) -> OTelSdkResult {
        self.0.export(metrics).await
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.0.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.0.shutdown_with_timeout(timeout)
    }

    fn temporality(&self) -> Temporality {
        self.0.temporality()
    }
}

#[test]
fn logs_and_metrics_counters_and_standalone_exporter_stats() {
    // R-CNT-6.
    let server = Server::ok();
    let config = server.all_config(Protocol::HttpProtobuf).with_traces(None);
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    let observer =
        pipeline.observer_with(ContentPolicy::default().with_capture(ContentCapture::EventOnly));
    inference(&observer);
    pipeline.force_flush(Duration::from_secs(5)).unwrap();
    let stats = pipeline.stats();
    assert_eq!(stats.traces, SignalStats::default(), "unconfigured");
    for (signal, s) in [("logs", stats.logs), ("metrics", stats.metrics)] {
        assert!(s.accepted > 0, "{signal}: {s:?}");
        assert_eq!((s.exported, s.pending), (s.accepted, 0), "{signal}");
    }
    pipeline.shutdown(Duration::from_secs(5)).unwrap();

    let server = Server::ok();
    let pipeline =
        ObservabilityPipeline::build(server.traces_config(Protocol::HttpProtobuf)).unwrap();
    inference(&pipeline.observer());
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    let stats = pipeline.stats();
    assert_eq!(stats.traces.exported, 1);
    assert_eq!(
        (stats.logs, stats.metrics),
        (SignalStats::default(), SignalStats::default())
    );
    assert_eq!(PipelineStats::default().logs, SignalStats::default());

    // Standalone exporters in application-owned providers.
    let server = Server::ok();
    let set = exporters(&server.all_config(Protocol::HttpProtobuf).with_traces(None)).unwrap();
    let log_exporter = Arc::new(set.logs.unwrap());
    let metric_exporter = Arc::new(set.metrics.unwrap());
    let logger_provider = SdkLoggerProvider::builder()
        .with_simple_exporter(Shared(log_exporter.clone()))
        .build();
    let meter_provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(Shared(metric_exporter.clone())).build())
        .build();
    let observer = Observer::builder()
        .logger_provider(&logger_provider)
        .meter_provider(&meter_provider)
        .content_policy(ContentPolicy::default().with_capture(ContentCapture::EventOnly))
        .build();
    inference(&observer);
    meter_provider.force_flush().unwrap();
    let logs = log_exporter.stats();
    let metrics = metric_exporter.stats();
    assert!(
        logs.accepted > 0 && logs.exported == logs.accepted,
        "{logs:?}"
    );
    assert!(
        metrics.accepted > 0 && metrics.exported == metrics.accepted,
        "{metrics:?}"
    );
    logger_provider.shutdown().unwrap();
    meter_provider.shutdown().unwrap();
    assert!(!server.requests_to("/v1/logs").is_empty());
    assert!(!server.requests_to("/v1/metrics").is_empty());
}

#[test]
fn counters_stay_consistent_under_concurrent_load() {
    // R-CNT-1: every snapshot of every signal satisfies
    // accepted == exported + dropped + failed + pending.
    let server = Server::always(Action::ok().after(Duration::from_millis(3)));
    let config = server
        .all_config(Protocol::HttpProtobuf)
        .with_batch(BatchLimits {
            max_queue_size: 64,
            max_export_batch_size: 16,
            scheduled_delay: Duration::from_millis(5),
            ..BatchLimits::default()
        })
        // Also the metric export deadline (min(export_timeout, interval)).
        .with_metric_export_interval(Duration::from_millis(500));
    let pipeline = Arc::new(ObservabilityPipeline::build(config).unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let snapshots = Arc::new(AtomicUsize::new(0));
    let sampler = {
        let (pipeline, stop, snapshots) = (pipeline.clone(), stop.clone(), snapshots.clone());
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let s = pipeline.stats();
                for (signal, stats) in [
                    ("traces", s.traces),
                    ("logs", s.logs),
                    ("metrics", s.metrics),
                ] {
                    assert!(stats.is_consistent(), "{signal}: {stats:?}");
                }
                snapshots.fetch_add(1, Ordering::Relaxed);
            }
        })
    };
    let workers: Vec<_> = (0..4)
        .map(|w| {
            let pipeline = pipeline.clone();
            thread::spawn(move || {
                let observer = pipeline.observer_with(
                    ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent),
                );
                for i in 0..300 {
                    inference(&observer);
                    if (w + i) % 97 == 0 {
                        let _ = pipeline.force_flush(Duration::from_millis(500));
                    }
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    sampler.join().unwrap();
    assert!(snapshots.load(Ordering::Relaxed) > 100);
    pipeline.shutdown(Duration::from_secs(10)).unwrap();
    let s = pipeline.stats();
    for (signal, stats) in [
        ("traces", s.traces),
        ("logs", s.logs),
        ("metrics", s.metrics),
    ] {
        assert!(stats.is_consistent(), "{signal}: {stats:?}");
        assert_eq!(stats.pending, 0, "{signal}");
        assert_eq!(stats.failed, 0, "{signal}: {stats:?}");
        assert!(stats.exported > 0, "{signal}");
    }
    assert_eq!(s.traces.accepted, 1200);
}

#[test]
fn an_unconfigured_signal_config_is_rejected_before_anything_starts() {
    // Sanity for the helpers above: a configuration without signals fails.
    assert!(matches!(
        ObservabilityPipeline::build(OtlpConfig::default()),
        Err(Error::Config(_))
    ));
}
