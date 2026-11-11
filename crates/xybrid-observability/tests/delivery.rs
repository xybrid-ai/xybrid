//! Delivery regressions over real HTTP: redirects (R-CFG-6), splitting by
//! final encoded size (R-BATCH-3..5), one absolute deadline per export
//! (R-DEADLINE-1..5), counters (R-CNT-1..6) and the byte budget (R-BUF-2);
//! per signal, failure replies, lost acknowledgements, an outage and its
//! recovery, saturation and receiver reconciliation.
#![cfg(feature = "otlp")]

mod support;

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use opentelemetry::logs::{AnyValue, LogRecord as _, Logger as _, LoggerProvider as _};
use opentelemetry::metrics::MeterProvider as _;
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
use support::{Action, Received, Server};
use xybrid_observability::otlp::{
    exporters, BatchLimits, EnvironmentSource, ObservabilityPipeline, OtlpConfig, OtlpLogExporter,
    OtlpMetricExporter, PipelineStats, Protocol, RetryLimits, SignalEndpoint, SignalStats,
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

#[test]
fn stats_report_the_bytes_the_byte_budget_admits_against() {
    // `SignalStats::buffered_bytes` is the running total the
    // `max_buffered_bytes` budget admits against, per signal. Against an
    // endpoint that never answers, the admitted spans and log records stay
    // counted (queued plus in flight) within the budget; the metrics
    // collection in flight holds exactly its encoded body; refused items add
    // nothing; shutdown releases everything.
    use opentelemetry::metrics::MeterProvider as _;

    let server = Server::always(Action::Hang);
    let endpoint = |signal: &str| {
        Some(
            server
                .endpoint(signal, Protocol::HttpProtobuf)
                .with_timeout(Duration::from_secs(120)),
        )
    };
    let config = server
        .all_config(Protocol::HttpProtobuf)
        .with_traces(endpoint("traces"))
        .with_logs(endpoint("logs"))
        .with_metrics(endpoint("metrics"))
        .with_batch(BatchLimits {
            max_buffered_bytes: 10_000,
            scheduled_delay: Duration::from_secs(60),
            export_timeout: Duration::from_secs(120),
            ..BatchLimits::default()
        });
    let pipeline = Arc::new(ObservabilityPipeline::build(config).unwrap());
    let stats = pipeline.stats();
    for s in [stats.traces, stats.logs, stats.metrics] {
        assert_eq!(s.buffered_bytes, 0, "{s:?}");
    }
    spans(&pipeline, 5, 3000);
    let names: Vec<String> = (0..5).map(|i| format!("log-{i}")).collect();
    logs(&pipeline, &names, 3000);
    let stats = pipeline.stats();
    for (signal, s) in [("traces", stats.traces), ("logs", stats.logs)] {
        assert_eq!((s.accepted, s.pending, s.dropped), (5, 2, 3), "{signal}");
        // Two 3 KB items admitted; the three refused ones hold nothing.
        assert!(
            s.buffered_bytes > 6000 && s.buffered_bytes <= 10_000,
            "{signal}: {s:?}"
        );
    }
    assert_eq!(stats.metrics.buffered_bytes, 0, "no collection yet");

    pipeline
        .meter_provider()
        .unwrap()
        .meter("app")
        .u64_counter("items")
        .build()
        .add(1, &[]);
    let flush = {
        let pipeline = pipeline.clone();
        thread::spawn(move || pipeline.force_flush(Duration::from_secs(60)))
    };
    // The bound only detects a request that never arrives.
    server.wait_for_path("/v1/metrics", 1, Duration::from_secs(10));
    let parked = server.requests_to("/v1/metrics");
    let metrics = pipeline.stats().metrics;
    assert_eq!(metrics.pending, 1, "{metrics:?}");
    assert_eq!(
        metrics.buffered_bytes,
        parked[0].body.len() as u64,
        "the collection in flight holds its encoded body"
    );

    let _ = pipeline.shutdown(Duration::from_millis(200));
    assert!(flush.join().unwrap().is_err());
    let stats = pipeline.stats();
    for s in [stats.traces, stats.logs, stats.metrics] {
        assert_eq!((s.pending, s.buffered_bytes), (0, 0), "{s:?}");
        assert!(s.is_consistent(), "{s:?}");
    }
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
fn a_shutdown_cut_off_fails_the_export_in_flight_and_drops_what_is_queued() {
    // Every item of every signal ends in exactly one bucket,
    // and the receiver agrees with the counters. Per path the first request
    // is rejected (503, no retries), the second acknowledged and every later
    // one parked. When shutdown starts, each signal has one export parked:
    // its items fail at the deadline, and what waits behind it (the fourth
    // span and log batch, the metrics collection shutdown takes) is dropped.
    use opentelemetry::metrics::MeterProvider as _;
    use std::collections::HashMap;
    use std::sync::Mutex;

    let per_path = Mutex::new(HashMap::<String, usize>::new());
    let server = Server::start(move |r| {
        let mut per_path = per_path.lock().unwrap();
        let n = per_path.entry(r.path.clone()).or_default();
        *n += 1;
        match *n {
            1 => Action::status(503),
            2 => Action::ok(),
            _ => Action::Hang,
        }
    });
    // Only the shutdown ends a parked request.
    let endpoint = |signal: &str| {
        Some(
            server
                .endpoint(signal, Protocol::HttpProtobuf)
                .with_timeout(Duration::from_secs(120)),
        )
    };
    let config = server
        .all_config(Protocol::HttpProtobuf)
        .with_traces(endpoint("traces"))
        .with_logs(endpoint("logs"))
        .with_metrics(endpoint("metrics"))
        .with_batch(BatchLimits {
            max_export_batch_size: 4,
            scheduled_delay: Duration::from_secs(60),
            export_timeout: Duration::from_secs(120),
            ..BatchLimits::default()
        })
        .with_retry(fast_retry(0));
    let pipeline = Arc::new(ObservabilityPipeline::build(config).unwrap());
    // Batches of 4: rejected, acknowledged, parked, and 2 items left queued.
    spans(&pipeline, 14, 10);
    let names: Vec<String> = (0..14).map(|i| format!("log-{i}")).collect();
    logs(&pipeline, &names, 10);
    // Three series: every (cumulative) collection carries three points.
    let counter = pipeline
        .meter_provider()
        .unwrap()
        .meter("app")
        .u64_counter("items")
        .build();
    for series in ["a", "b", "c"] {
        counter.add(1, &[KeyValue::new("series", series)]);
    }

    // The bounds only detect a request that never arrives.
    let wait = |path: &str, n: usize| server.wait_for_path(path, n, Duration::from_secs(10));
    let flush = |p: &Arc<ObservabilityPipeline>| {
        let p = p.clone();
        thread::spawn(move || p.force_flush(Duration::from_secs(60)))
    };
    // The first flush drives every span and log batch until one parks; each
    // flush takes one metrics collection, exported one after the other.
    let mut flushes = vec![flush(&pipeline)];
    wait("/v1/traces", 3);
    wait("/v1/logs", 3);
    wait("/v1/metrics", 1);
    flushes.push(flush(&pipeline));
    wait("/v1/metrics", 2);
    flushes.push(flush(&pipeline));
    wait("/v1/metrics", 3);

    let result = pipeline.shutdown(Duration::from_millis(100));
    assert!(result.as_ref().is_err_and(Error::is_timeout), "{result:?}");
    for flush in flushes {
        assert!(flush.join().unwrap().is_err());
    }
    // The metrics collection shutdown takes waits behind the parked export
    // on the SDK's reader thread; it is refused once that export is cut off.
    // The bound only detects a collection that never comes.
    let until = Instant::now() + Duration::from_secs(10);
    while pipeline.stats().metrics.accepted < 12 {
        assert!(Instant::now() < until, "{:?}", pipeline.stats().metrics);
        thread::sleep(Duration::from_millis(1));
    }

    let stats = pipeline.stats();
    for (path, s, queued) in [
        ("/v1/traces", stats.traces, 2),
        ("/v1/logs", stats.logs, 2),
        ("/v1/metrics", stats.metrics, 3),
    ] {
        let sent = server.requests_to(path);
        assert_eq!(sent.len(), 3, "{path}: a request after the cut off");
        let [rejected, acknowledged, parked] = [0, 1, 2].map(|i| sent[i].item_count());
        assert!(
            server.response_of(sent[1].index).is_some(),
            "{path}: the receiver acknowledged the second request"
        );
        // The client closed the request in flight. A blocking shutdown also
        // releases the transport before it returns, so this cannot tell the
        // deadline's cancellation from the teardown: `lifecycle.rs`
        // (shutdown_aborts_the_export_in_flight_and_sends_nothing_after_it)
        // checks the cancellation alone.
        server.wait_disconnect(sent[2].index, Duration::from_secs(10));
        // What the receiver acknowledged is exactly what was exported.
        assert_eq!(s.exported, acknowledged, "{path}: {s:?}");
        assert_eq!(
            s.failed,
            rejected + parked,
            "{path}: the rejected items and those in flight fail: {s:?}"
        );
        assert_eq!(s.dropped, queued, "{path}: what was still queued: {s:?}");
        assert_eq!(s.pending, 0, "{path}: {s:?}");
        assert_eq!(
            s.accepted,
            rejected + acknowledged + parked + queued,
            "{path}: {s:?}"
        );
        assert!(s.is_consistent(), "{path}: {s:?}");
        if path != "/v1/metrics" {
            assert_eq!([rejected, acknowledged, parked], [4, 4, 4], "{path}");
        } else {
            assert_eq!([rejected, acknowledged, parked], [3, 3, 3], "{path}");
        }
    }
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

// Per signal: failure replies, lost acknowledgements, an outage and
// its recovery, saturation and receiver reconciliation.

/// One signal of the per-signal tables below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    Traces,
    Logs,
    Metrics,
}

impl Signal {
    const ALL: [Self; 3] = [Self::Traces, Self::Logs, Self::Metrics];

    fn path(self) -> &'static str {
        match self {
            Self::Traces => "/v1/traces",
            Self::Logs => "/v1/logs",
            Self::Metrics => "/v1/metrics",
        }
    }

    fn of(self, stats: PipelineStats) -> SignalStats {
        match self {
            Self::Traces => stats.traces,
            Self::Logs => stats.logs,
            Self::Metrics => stats.metrics,
        }
    }
}

/// The per-attempt timeout of [`pipeline_to`]: above every scripted reply
/// delay and every flush bound below, so only the export deadline can end an
/// attempt early.
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(120);

/// A pipeline exporting `signals` over OTLP/protobuf to `base`, with no
/// environment; exports happen on flush and shutdown only.
fn pipeline_to(
    base: &str,
    signals: &[Signal],
    export_timeout: Duration,
    retry: RetryLimits,
) -> ObservabilityPipeline {
    let endpoint = |signal: Signal| {
        Some(SignalEndpoint::new(format!("{base}{}", signal.path())).with_timeout(ATTEMPT_TIMEOUT))
    };
    let mut config = OtlpConfig::default()
        .with_environment(EnvironmentSource::Disabled)
        .with_batch(limits(export_timeout))
        .with_retry(retry)
        .with_metric_export_interval(Duration::from_secs(600));
    for &signal in signals {
        config = match signal {
            Signal::Traces => config.with_traces(endpoint(signal)),
            Signal::Logs => config.with_logs(endpoint(signal)),
            Signal::Metrics => config.with_metrics(endpoint(signal)),
        };
    }
    ObservabilityPipeline::build(config).unwrap()
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}-{i}")).collect()
}

fn named_spans(pipeline: &ObservabilityPipeline, names: &[String]) {
    let tracer = pipeline.tracer_provider().unwrap().tracer("app");
    for name in names {
        tracer.start(name.clone()).end();
    }
}

/// Offers one batch of `signal` and returns its size: two spans, two log
/// records, or a counter with three series (three data points in every
/// collection).
fn offer(pipeline: &ObservabilityPipeline, signal: Signal) -> u64 {
    match signal {
        Signal::Traces => {
            spans(pipeline, 2, 10);
            2
        }
        Signal::Logs => {
            logs(pipeline, &names("log", 2), 10);
            2
        }
        Signal::Metrics => {
            let counter = pipeline
                .meter_provider()
                .unwrap()
                .meter("app")
                .u64_counter("items")
                .build();
            for series in ["a", "b", "c"] {
                counter.add(1, &[KeyValue::new("series", series)]);
            }
            3
        }
    }
}

/// Log record bodies in body order.
fn log_bodies(request: &Received) -> Vec<String> {
    request
        .logs()
        .resource_logs
        .into_iter()
        .flat_map(|r| r.scope_logs)
        .flat_map(|s| s.log_records)
        .map(|record| match record.body.and_then(|b| b.value) {
            Some(Wire::StringValue(body)) => body,
            other => panic!("a string log body, not {other:?}"),
        })
        .collect()
}

/// The cumulative value of the `work` counter per `phase`, as decoded.
fn work_totals(request: &Received) -> BTreeMap<String, i64> {
    use opentelemetry_proto::tonic::metrics::v1::metric::Data;
    use opentelemetry_proto::tonic::metrics::v1::number_data_point::Value;
    use opentelemetry_proto::tonic::metrics::v1::AggregationTemporality;

    let mut totals = BTreeMap::new();
    let metrics = request
        .metrics()
        .resource_metrics
        .into_iter()
        .flat_map(|r| r.scope_metrics)
        .flat_map(|s| s.metrics);
    for metric in metrics.filter(|m| m.name == "work") {
        let Some(Data::Sum(sum)) = metric.data else {
            panic!("work is a sum")
        };
        assert_eq!(
            sum.aggregation_temporality,
            AggregationTemporality::Cumulative as i32
        );
        for point in sum.data_points {
            let phase = point
                .attributes
                .iter()
                .find(|kv| kv.key == "phase")
                .and_then(|kv| kv.value.clone())
                .and_then(|v| v.value);
            let (Some(Wire::StringValue(phase)), Some(Value::AsInt(value))) = (phase, point.value)
            else {
                panic!("a work point with a phase")
            };
            totals.insert(phase, value);
        }
    }
    totals
}

fn totals(pairs: &[(&str, i64)]) -> BTreeMap<String, i64> {
    pairs.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
}

/// A loopback port that refuses connections. [`Refusing::listen`] brings an
/// endpoint up on it.
///
/// On Linux the port stays the test's own ([`Refusing::held`]): a socket bound
/// to it that never listens. Linux resets a connect to it as if nothing were
/// there, and no other test (each binds port 0 in parallel) can be given the
/// port. That does not carry over: macOS drops a SYN sent to a socket that is
/// bound but not listening, so a connect would wait for the OS to time it out
/// instead of being refused. Everywhere else the port is bound and released
/// ([`Refusing::released`]) and `listen` binds it again; in between, the
/// kernel can (rarely) hand it to another test.
enum Refusing {
    Held(tokio::net::TcpSocket),
    Released(SocketAddr),
}

impl Refusing {
    fn bind() -> Self {
        if cfg!(target_os = "linux") {
            Self::held()
        } else {
            Self::released()
        }
    }

    /// Bound and never listening (refuses only on Linux).
    fn held() -> Self {
        // Without `SO_REUSEADDR`, which std's listeners set on Unix: no
        // other socket shares the port while this one holds it.
        let socket = tokio::net::TcpSocket::new_v4().expect("a TCP socket");
        socket.bind(LOOPBACK).expect("a loopback port");
        Self::Held(socket)
    }

    /// Bound, then released: nothing holds the port.
    fn released() -> Self {
        let listener = TcpListener::bind(LOOPBACK).expect("a loopback port");
        Self::Released(listener.local_addr().expect("a bound listener"))
    }

    fn addr(&self) -> SocketAddr {
        match self {
            Self::Held(socket) => socket.local_addr().expect("a bound socket"),
            Self::Released(addr) => *addr,
        }
    }

    /// Starts listening on the same port: the endpoint comes up.
    fn listen(self) -> TcpListener {
        match self {
            Self::Held(socket) => {
                // Tokio registers a listener with a reactor; the std listener
                // it hands back no longer needs one.
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("a runtime to listen in");
                let _entered = runtime.enter();
                socket
                    .listen(128)
                    .expect("listening")
                    .into_std()
                    .expect("a std listener")
            }
            Self::Released(addr) => rebind(addr),
        }
    }
}

const LOOPBACK: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);

/// Binds `addr` again. The bound only detects a port that never frees up.
fn rebind(addr: SocketAddr) -> TcpListener {
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        match TcpListener::bind(addr) {
            Ok(listener) => return listener,
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse && Instant::now() < until => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(e) => panic!("rebinding {addr}: {e}"),
        }
    }
}

/// Connects to `refusing`, which must refuse; then brings it up and connects.
fn refuses_then_listens_on_the_same_port(refusing: Refusing) {
    let addr = refusing.addr();
    let refused = std::net::TcpStream::connect(addr).expect_err("nothing listens yet");
    assert_eq!(
        refused.kind(),
        std::io::ErrorKind::ConnectionRefused,
        "{refused}"
    );
    let listener = refusing.listen();
    assert_eq!(listener.local_addr().unwrap(), addr);
    std::net::TcpStream::connect(addr).expect("listening on the same port");
}

#[cfg(target_os = "linux")]
#[test]
fn a_held_port_refuses_connections_and_is_never_handed_to_another_test() {
    let refusing = Refusing::held();
    let taken = TcpListener::bind(refusing.addr()).expect_err("the port is held");
    assert_eq!(taken.kind(), std::io::ErrorKind::AddrInUse, "{taken}");
    refuses_then_listens_on_the_same_port(refusing);
}

#[test]
fn a_released_port_refuses_connections_until_it_is_bound_again() {
    // What `Refusing::bind` falls back to off Linux; exercised everywhere.
    refuses_then_listens_on_the_same_port(Refusing::released());
}

/// One reply sequence for one batch, and what it must come to.
struct Case {
    name: &'static str,
    /// Request `i` gets reply `i`; the last one repeats.
    replies: Vec<Action>,
    export_timeout: Duration,
    /// Requests the batch takes (two retries are allowed).
    requests: usize,
    exported: bool,
    /// The least time between the first two requests (a `Retry-After` wait).
    min_gap: Option<Duration>,
    /// The client gave up on the first request and closed it.
    aborted: bool,
}

impl Case {
    fn new(name: &'static str, replies: Vec<Action>, requests: usize, exported: bool) -> Self {
        Self {
            name,
            replies,
            export_timeout: Duration::from_secs(10),
            requests,
            exported,
            min_gap: None,
            aborted: false,
        }
    }
}

/// Delivers one batch of `signal` against `case.replies`; checks the exact
/// request count, that every attempt resent the same bytes, the flush result
/// and the counters (every item counted once).
fn check_case(signal: Signal, case: Case) {
    let what = format!("{signal:?}, {}", case.name);
    let server = Server::sequence(case.replies);
    let pipeline = pipeline_to(&server.url, &[signal], case.export_timeout, fast_retry(2));
    let n = offer(&pipeline, signal);
    let flushed = pipeline.force_flush(Duration::from_secs(30));
    let requests = server.requests();
    assert_eq!(requests.len(), case.requests, "{what}: requests");
    for (attempt, request) in requests.iter().enumerate() {
        assert_eq!(request.path, signal.path(), "{what}");
        assert!(
            request.body == requests[0].body,
            "{what}: attempt {attempt} did not resend the same bytes"
        );
    }
    assert_eq!(
        requests[0].item_count(),
        n,
        "{what}: one request, every item"
    );
    let stats = signal.of(pipeline.stats());
    let expected = if case.exported {
        assert!(flushed.is_ok(), "{what}: {flushed:?}");
        SignalStats {
            accepted: n,
            exported: n,
            ..SignalStats::default()
        }
    } else {
        assert_eq!(delivery(&flushed), Some((n, 0)), "{what}: {flushed:?}");
        SignalStats {
            accepted: n,
            failed: n,
            ..SignalStats::default()
        }
    };
    assert_eq!(stats, expected, "{what}");
    if let Some(min) = case.min_gap {
        // A lower bound only: the retry waited for the server's hint rather
        // than the 10 ms backoff.
        let gap = requests[1].at.duration_since(requests[0].at);
        assert!(gap >= min, "{what}: retried after {gap:?}");
    }
    if case.aborted {
        server.wait_disconnect(requests[0].index, Duration::from_secs(10));
        assert!(server.response_of(requests[0].index).is_none(), "{what}");
    }
}

/// What each reply class makes of one batch.
fn failure_replies() -> Vec<Case> {
    let throttled = |after: &str| Action::status(429).header("Retry-After", after);
    vec![
        Case::new(
            "429 then 200",
            vec![Action::status(429), Action::ok()],
            2,
            true,
        ),
        Case {
            min_gap: Some(Duration::from_secs(1)),
            ..Case::new(
                "429 with Retry-After 1, then 200",
                vec![throttled("1"), Action::ok()],
                2,
                true,
            )
        },
        Case::new(
            "429 until the retries run out",
            vec![Action::status(429)],
            3,
            false,
        ),
        Case::new(
            "429 with Retry-After 0 until the retries run out",
            vec![throttled("0")],
            3,
            false,
        ),
        // Waiting would pass the export deadline: the batch fails at once. The
        // deadline (120 s) lies beyond the 30 s flush bound, so a client that
        // waited for it, or for the hint, would time the flush out.
        Case {
            export_timeout: Duration::from_secs(120),
            ..Case::new(
                "429 with Retry-After past the deadline",
                vec![throttled("600"), Action::ok()],
                1,
                false,
            )
        },
        Case::new(
            "503 until the retries run out",
            vec![Action::status(503)],
            3,
            false,
        ),
        // Not retryable: the 200 that would follow is never asked for.
        Case::new("500", vec![Action::status(500), Action::ok()], 1, false),
        Case::new("400", vec![Action::status(400), Action::ok()], 1, false),
        // The attempt timeout (ATTEMPT_TIMEOUT) outlasts the reply delay and
        // the flush bound: only the 300 ms export deadline can cut it off.
        Case {
            export_timeout: Duration::from_millis(300),
            aborted: true,
            ..Case::new(
                "200 only after the export deadline",
                vec![Action::ok().after(Duration::from_secs(60))],
                1,
                false,
            )
        },
    ]
}

#[test]
fn failure_replies_settle_a_span_batch_exactly() {
    for case in failure_replies() {
        check_case(Signal::Traces, case);
    }
}

#[test]
fn failure_replies_settle_a_log_batch_exactly() {
    for case in failure_replies() {
        check_case(Signal::Logs, case);
    }
}

#[test]
fn failure_replies_settle_a_metrics_collection_exactly() {
    for case in failure_replies() {
        check_case(Signal::Metrics, case);
    }
}

#[test]
fn a_refused_connection_fails_every_signal() {
    // Nothing listens on the port, so every attempt is a transport error;
    // the batch fails and the flush says so. Whether a refused connect is
    // retried is not pinned: a closed port records no attempts to count. (The
    // lost-acknowledgement tests pin retries of a transport error on a
    // connection that was made.)
    let refusing = Refusing::bind();
    let base = format!("http://{}", refusing.addr());
    for signal in Signal::ALL {
        let pipeline = pipeline_to(&base, &[signal], Duration::from_secs(10), fast_retry(2));
        let n = offer(&pipeline, signal);
        let flushed = pipeline.force_flush(Duration::from_secs(60));
        assert_eq!(delivery(&flushed), Some((n, 0)), "{signal:?}: {flushed:?}");
        assert_eq!(
            signal.of(pipeline.stats()),
            SignalStats {
                accepted: n,
                failed: n,
                ..SignalStats::default()
            },
            "{signal:?}"
        );
    }
}

#[test]
fn lost_acknowledgements_resend_the_same_bytes_and_count_each_item_once() {
    // R-CNT-3 for every signal: the receiver reads each body, then
    // closes the connection unanswered. Every attempt resends the batch byte
    // for byte, and its items count once: exported when a later attempt is
    // acknowledged; failed (delivery unknown) when the retries run out,
    // although the receiver read them three times.
    for signal in Signal::ALL {
        check_case(
            signal,
            Case::new(
                "lost, then acknowledged",
                vec![Action::Lost, Action::ok()],
                2,
                true,
            ),
        );
        check_case(
            signal,
            Case::new("lost every time", vec![Action::Lost], 3, false),
        );
    }
}

/// What the endpoint does with a request while it is down.
#[derive(Clone, Copy, Debug)]
enum Down {
    /// Answers 503.
    Unavailable,
    /// Reads the request, then closes the connection unanswered.
    LostAcknowledgement,
}

/// An endpoint availability the test switches. While down, requests that
/// arrive before the test opens the gate are parked there, so the test knows
/// which batch is in flight when it offers more.
struct Switch {
    down: AtomicBool,
    open: Mutex<bool>,
    opened: Condvar,
    arrived: Mutex<mpsc::Sender<String>>,
    /// Whether each request (by arrival index) was acknowledged.
    acknowledged: Mutex<BTreeMap<usize, bool>>,
}

impl Switch {
    fn new(arrived: mpsc::Sender<String>) -> Self {
        Self {
            down: AtomicBool::new(false),
            open: Mutex::new(false),
            opened: Condvar::new(),
            arrived: Mutex::new(arrived),
            acknowledged: Mutex::default(),
        }
    }

    fn answer(&self, request: &Received, down: Down) -> Action {
        if !self.down.load(Ordering::SeqCst) {
            self.acknowledged
                .lock()
                .unwrap()
                .insert(request.index, true);
            return Action::ok();
        }
        let _ = self.arrived.lock().unwrap().send(request.path.clone());
        // Bounded so that a failing test cannot park the server forever.
        let open = self.open.lock().unwrap();
        drop(
            self.opened
                .wait_timeout_while(open, Duration::from_secs(60), |open| !*open)
                .unwrap(),
        );
        self.acknowledged
            .lock()
            .unwrap()
            .insert(request.index, false);
        match down {
            Down::Unavailable => Action::status(503),
            Down::LostAcknowledgement => Action::Lost,
        }
    }

    fn open_gate(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }
}

/// Opens the gate when dropped: a failing assertion then releases the
/// parked requests, so the server can stop at once.
struct OpenOnDrop(Arc<Switch>);

impl Drop for OpenOnDrop {
    fn drop(&mut self) {
        self.0.open_gate();
    }
}

/// Healthy, down, healthy again; the phases are driven by flushes, not
/// by time. Queue 8, batch 4, one retry, all three signals. Measured loss:
///
/// | phase  | spans, and log records alike    | metrics (`work` counter)        |
/// |--------|---------------------------------|---------------------------------|
/// | before | 2 offered: 2 exported           | collection of 1 point: exported |
/// | down   | 10 offered: 8 failed, 2 dropped | collection of 2 points: failed  |
/// | after  | 2 offered: 2 exported           | 2 collections of 3: exported    |
///
/// While down, the batch in flight and the batch queued behind it fail once
/// their retry is spent (two requests each); the 2 offers beyond the queue
/// bound (queued plus in flight) are dropped at admission and never sent.
/// Nothing offered before or after is lost: no outage item is resent after
/// recovery, and the first export after it is acknowledged on its first
/// attempt. With lost acknowledgements the receiver did read the 8 failed
/// items, twice each: "failed" means "delivery unknown". Metrics lose no
/// value: the failed collection's increment arrives with the next
/// acknowledged (cumulative) collection.
///
/// Timing: no hold-off carries over from the outage, so the first
/// acknowledgement after recovery is one round trip of the next export (in
/// production, at most `scheduled_delay` plus that round trip). The test
/// prints the time from recovery to that request (`--nocapture`); it asserts
/// only the 30 s flush bound, as a hang detector.
///
/// The exact request counts pin today's per-export retry policy: nothing
/// carries a `Retry-After` hold or a circuit breaker across exports. Whether
/// sustained failures should suppress later exports is an open product
/// decision; adopting one changes these counts.
fn an_outage_then_recovery(down: Down) {
    let (arrived, arrivals) = mpsc::channel();
    let switch = Arc::new(Switch::new(arrived));
    let server = {
        let switch = switch.clone();
        Server::start(move |r| switch.answer(r, down))
    };
    // Dropped before the server.
    let _release = OpenOnDrop(switch.clone());
    // The bound only detects a request that never arrives.
    let parked = |path: &str| {
        let next = arrivals
            .recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|_| panic!("no request to {path}"));
        assert_eq!(next, path, "{down:?}: the next request parked");
    };
    let pipeline = Arc::new(
        ObservabilityPipeline::build(
            server
                .all_config(Protocol::HttpProtobuf)
                .with_batch(BatchLimits {
                    max_queue_size: 8,
                    max_export_batch_size: 4,
                    scheduled_delay: Duration::from_secs(60),
                    ..BatchLimits::default()
                })
                .with_retry(fast_retry(1)),
        )
        .unwrap(),
    );
    let work = pipeline
        .meter_provider()
        .unwrap()
        .meter("app")
        .u64_counter("work")
        .build();
    let phase = |name: &'static str| [KeyValue::new("phase", name)];

    let before = names("a", 2);
    named_spans(&pipeline, &before);
    logs(&pipeline, &before, 10);
    work.add(1, &phase("a"));
    pipeline.force_flush(Duration::from_secs(30)).unwrap();

    // Down. The first span and log batches park in flight, the next ones
    // fill the queue, and the last two offers of each are dropped.
    switch.down.store(true, Ordering::SeqCst);
    let during = names("b", 10);
    named_spans(&pipeline, &during[..4]);
    parked("/v1/traces");
    named_spans(&pipeline, &during[4..]);
    logs(&pipeline, &during[..4], 10);
    parked("/v1/logs");
    logs(&pipeline, &during[4..], 10);
    let stats = pipeline.stats();
    for (signal, s) in [("traces", stats.traces), ("logs", stats.logs)] {
        assert!(s.buffered_bytes > 0, "{down:?}: {signal} holds its items");
        let held = SignalStats {
            accepted: 12,
            exported: 2,
            dropped: 2,
            failed: 0,
            pending: 8,
            buffered_bytes: s.buffered_bytes,
        };
        assert_eq!(s, held, "{down:?}: {signal} while down");
    }
    work.add(10, &phase("b"));
    let flush = {
        let pipeline = pipeline.clone();
        thread::spawn(move || pipeline.force_flush(Duration::from_secs(30)))
    };
    parked("/v1/metrics");
    switch.open_gate();
    // Both span batches, both log batches and the collection fail; the
    // admission drops were never covered by the flush.
    let flushed = flush.join().unwrap();
    assert_eq!(
        delivery(&flushed),
        Some((8 + 8 + 2, 0)),
        "{down:?}: {flushed:?}"
    );

    switch.down.store(false, Ordering::SeqCst);
    // Every outage request arrived before the flush above returned.
    let recovered = Instant::now();
    let after = names("c", 2);
    named_spans(&pipeline, &after);
    logs(&pipeline, &after, 10);
    work.add(100, &phase("c"));
    pipeline.force_flush(Duration::from_secs(30)).unwrap();
    // Shutdown takes one more collection.
    pipeline.shutdown(Duration::from_secs(10)).unwrap();

    let stats = pipeline.stats();
    let lost = SignalStats {
        accepted: 14,
        exported: 4,
        dropped: 2,
        failed: 8,
        pending: 0,
        buffered_bytes: 0,
    };
    assert_eq!(stats.traces, lost, "{down:?}");
    assert_eq!(stats.logs, lost, "{down:?}");
    let metrics = SignalStats {
        accepted: 1 + 2 + 3 + 3,
        exported: 1 + 3 + 3,
        failed: 2,
        ..SignalStats::default()
    };
    assert_eq!(stats.metrics, metrics, "{down:?}");

    // The receiver agrees: what it acknowledged is exactly what was exported.
    let acknowledged = switch.acknowledged.lock().unwrap().clone();
    let mut requests = server.requests();
    requests.sort_by_key(|r| r.index);
    assert_eq!(acknowledged.len(), requests.len());
    let first = requests
        .iter()
        .filter(|r| r.at >= recovered)
        .min_by_key(|r| r.at)
        .expect("a request after recovery");
    assert!(acknowledged[&first.index], "{down:?}: first after recovery");
    eprintln!(
        "{down:?}: first acknowledged request {:?} after recovery",
        first.at.duration_since(recovered)
    );
    for signal in Signal::ALL {
        let sent: Vec<&Received> = requests
            .iter()
            .filter(|r| r.path == signal.path())
            .collect();
        let expected = if signal == Signal::Metrics { 5 } else { 6 };
        assert_eq!(sent.len(), expected, "{down:?}: requests to {signal:?}");
        let delivered: u64 = sent
            .iter()
            .filter(|r| acknowledged[&r.index])
            .map(|r| r.item_count())
            .sum();
        assert_eq!(signal.of(stats).exported, delivered, "{down:?}: {signal:?}");
    }
    let healthy: BTreeSet<String> = before.iter().chain(&after).cloned().collect();
    let mut tried_twice: Vec<String> = during[..8]
        .iter()
        .flat_map(|name| [name.clone(), name.clone()])
        .collect();
    tried_twice.sort();
    let items = |request: &Received| match request.path.as_str() {
        "/v1/traces" => request.span_names(),
        _ => log_bodies(request),
    };
    for path in ["/v1/traces", "/v1/logs"] {
        let (mut ok, mut failed) = (Vec::new(), Vec::new());
        for request in requests.iter().filter(|r| r.path == path) {
            if acknowledged[&request.index] {
                ok.extend(items(request));
            } else {
                failed.extend(items(request));
            }
        }
        failed.sort();
        assert_eq!(ok.len(), healthy.len(), "{down:?}: {path}: {ok:?}");
        assert_eq!(
            ok.into_iter().collect::<BTreeSet<_>>(),
            healthy,
            "{down:?}: {path}"
        );
        assert_eq!(failed, tried_twice, "{down:?}: {path}");
    }
    let collections: Vec<(bool, BTreeMap<String, i64>)> = requests
        .iter()
        .filter(|r| r.path == "/v1/metrics")
        .map(|r| (acknowledged[&r.index], work_totals(r)))
        .collect();
    let down_total = totals(&[("a", 1), ("b", 10)]);
    let recovered = totals(&[("a", 1), ("b", 10), ("c", 100)]);
    assert_eq!(
        collections,
        [
            (true, totals(&[("a", 1)])),
            (false, down_total.clone()),
            (false, down_total),
            (true, recovered.clone()),
            (true, recovered),
        ],
        "{down:?}"
    );
}

#[test]
fn an_outage_of_503s_then_recovery_loses_only_what_was_offered_while_down() {
    an_outage_then_recovery(Down::Unavailable);
}

#[test]
fn an_outage_of_lost_acknowledgements_then_recovery_loses_only_what_was_offered_while_down() {
    an_outage_then_recovery(Down::LostAcknowledgement);
}

#[test]
fn an_endpoint_that_refuses_then_comes_up_delivers_from_then_on() {
    // Connection refused: down from the start, then listening on the
    // same port. Measured loss: what was offered while it refused (2 spans,
    // 2 log records and the collection's point: failed); nothing after. The
    // first collection after it comes up carries the increment recorded
    // while it was down. The time from coming up to the first request is
    // printed (`--nocapture`), not asserted.
    let refusing = Refusing::bind();
    let pipeline = pipeline_to(
        &format!("http://{}", refusing.addr()),
        &Signal::ALL,
        Duration::from_secs(10),
        fast_retry(1),
    );
    let work = pipeline
        .meter_provider()
        .unwrap()
        .meter("app")
        .u64_counter("work")
        .build();

    let during = names("b", 2);
    named_spans(&pipeline, &during);
    logs(&pipeline, &during, 10);
    work.add(10, &[KeyValue::new("phase", "b")]);
    let flushed = pipeline.force_flush(Duration::from_secs(60));
    assert_eq!(delivery(&flushed), Some((2 + 2 + 1, 0)), "{flushed:?}");

    let server = Server::start_on(refusing.listen(), |_| Action::ok());
    let up = Instant::now();
    let after = names("c", 2);
    named_spans(&pipeline, &after);
    logs(&pipeline, &after, 10);
    work.add(100, &[KeyValue::new("phase", "c")]);
    pipeline.force_flush(Duration::from_secs(30)).unwrap();
    pipeline.shutdown(Duration::from_secs(10)).unwrap();

    let stats = pipeline.stats();
    let once = SignalStats {
        accepted: 4,
        exported: 2,
        failed: 2,
        ..SignalStats::default()
    };
    assert_eq!(stats.traces, once);
    assert_eq!(stats.logs, once);
    let metrics = SignalStats {
        accepted: 1 + 2 + 2,
        exported: 2 + 2,
        failed: 1,
        ..SignalStats::default()
    };
    assert_eq!(stats.metrics, metrics);
    // Only what was offered after it came up reached it, in one request per
    // span and log batch and one per collection.
    let traces = server.requests_to("/v1/traces");
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].span_names(), after);
    let records = server.requests_to("/v1/logs");
    assert_eq!(records.len(), 1);
    assert_eq!(log_bodies(&records[0]), after);
    let collections = server.requests_to("/v1/metrics");
    assert_eq!(collections.len(), 2);
    for collection in &collections {
        assert_eq!(work_totals(collection), totals(&[("b", 10), ("c", 100)]));
    }
    let points: u64 = collections.iter().map(Received::item_count).sum();
    assert_eq!(points, stats.metrics.exported);
    let first = server.requests().iter().map(|r| r.at).min();
    let first = first.expect("a request after it came up");
    eprintln!(
        "refused, then up: first acknowledged request {:?} after it came up",
        first.duration_since(up)
    );
}

#[test]
fn a_metrics_byte_budget_below_one_collection_drops_it_without_a_request() {
    // Metrics are admitted by bytes only: a collection whose encoded
    // body exceeds max_buffered_bytes is dropped whole at admission (local
    // policy, not a failed delivery). No request; every point dropped.
    for protocol in PROTOCOLS {
        let server = Server::ok();
        let config = server
            .traces_config(protocol)
            .with_traces(None)
            .with_metrics(Some(server.endpoint("metrics", protocol)))
            .with_batch(BatchLimits {
                max_buffered_bytes: 64,
                scheduled_delay: Duration::from_secs(60),
                ..BatchLimits::default()
            });
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        let n = offer(&pipeline, Signal::Metrics);
        let flushed = pipeline.force_flush(Duration::from_secs(5));
        assert_eq!(
            delivery(&flushed),
            Some((0, n)),
            "{protocol:?}: {flushed:?}"
        );
        let dropped = |n| SignalStats {
            accepted: n,
            dropped: n,
            ..SignalStats::default()
        };
        assert_eq!(pipeline.stats().metrics, dropped(n), "{protocol:?}");
        // The final collection goes the same way.
        let result = pipeline.shutdown(Duration::from_secs(5));
        assert_eq!(delivery(&result), Some((0, n)), "{protocol:?}: {result:?}");
        assert_eq!(pipeline.stats().metrics, dropped(2 * n), "{protocol:?}");
        assert_eq!(server.count(), 0, "{protocol:?}: nothing sent");
    }
}

#[test]
fn a_hung_endpoint_saturates_the_log_queue_exactly() {
    // The log twin of the JS "a hung endpoint: queue 8, batch 4, 20
    // spans": the queue bound counts the batch in flight as well as the
    // queue, so 8 records are pending and the other 12 are dropped.
    let server = Server::always(Action::Hang);
    let config = server
        .traces_config(Protocol::HttpProtobuf)
        .with_traces(None)
        .with_logs(Some(server.endpoint("logs", Protocol::HttpProtobuf)))
        .with_batch(BatchLimits {
            max_queue_size: 8,
            max_export_batch_size: 4,
            scheduled_delay: Duration::from_secs(60),
            ..BatchLimits::default()
        });
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    let records = names("log", 20);
    logs(&pipeline, &records[..4], 10);
    // The bound only detects a request that never arrives.
    server.wait_for_path("/v1/logs", 1, Duration::from_secs(10));
    logs(&pipeline, &records[4..], 10);
    let held = pipeline.stats().logs;
    assert!(held.buffered_bytes > 0, "{held:?}");
    let saturated = SignalStats {
        accepted: 20,
        dropped: 12,
        pending: 8,
        buffered_bytes: held.buffered_bytes,
        ..SignalStats::default()
    };
    assert_eq!(held, saturated);
    let sent = server.requests();
    assert_eq!(sent.len(), 1);
    assert_eq!(log_bodies(&sent[0]), records[..4]);

    // Shutdown cuts the parked export off: its 4 records fail; the 4 queued
    // behind it are dropped.
    let result = pipeline.shutdown(Duration::from_millis(100));
    assert!(result.as_ref().is_err_and(Error::is_timeout), "{result:?}");
    let settled = SignalStats {
        accepted: 20,
        failed: 4,
        dropped: 16,
        ..SignalStats::default()
    };
    assert_eq!(pipeline.stats().logs, settled);
    assert_eq!(server.count(), 1, "nothing sent after the parked request");
}

#[test]
fn exported_metric_points_are_the_points_the_receiver_acknowledged() {
    // A mixed run over every aggregation the default views produce
    // (sum, non-monotonic sum, histogram, gauge). The first collection is
    // rejected (503, no retry), the later ones acknowledged. The points the
    // receiver decodes from acknowledged requests are exactly `exported`;
    // those of the rejected one exactly `failed`.
    use opentelemetry_proto::tonic::metrics::v1::metric::Data;

    let server = Server::start(|r| {
        if r.index == 0 {
            Action::status(503)
        } else {
            Action::ok()
        }
    });
    let pipeline = pipeline_to(
        &server.url,
        &[Signal::Metrics],
        Duration::from_secs(10),
        fast_retry(0),
    );
    let meter = pipeline.meter_provider().unwrap().meter("app");
    let requests = meter.u64_counter("requests").build();
    let latency = meter.f64_histogram("latency").build();
    let active = meter.i64_up_down_counter("active").build();
    let depth = meter.u64_gauge("depth").build();
    for route in ["a", "b"] {
        requests.add(1, &[KeyValue::new("route", route)]);
        latency.record(0.5, &[KeyValue::new("route", route)]);
    }
    active.add(2, &[]);
    depth.record(7, &[]);
    let flushed = pipeline.force_flush(Duration::from_secs(10));
    assert_eq!(delivery(&flushed), Some((6, 0)), "{flushed:?}");
    // A new series; two acknowledged collections, and shutdown takes a third.
    requests.add(1, &[KeyValue::new("route", "c")]);
    pipeline.force_flush(Duration::from_secs(10)).unwrap();
    pipeline.force_flush(Duration::from_secs(10)).unwrap();
    pipeline.shutdown(Duration::from_secs(10)).unwrap();

    let mut sent = server.requests();
    sent.sort_by_key(|r| r.index);
    assert_eq!(sent.len(), 4);
    let kinds: BTreeSet<&str> = sent[0]
        .metrics()
        .resource_metrics
        .into_iter()
        .flat_map(|r| r.scope_metrics)
        .flat_map(|s| s.metrics)
        .filter_map(|m| match m.data? {
            Data::Sum(s) if s.is_monotonic => Some("sum"),
            Data::Sum(_) => Some("non-monotonic sum"),
            Data::Histogram(_) => Some("histogram"),
            Data::Gauge(_) => Some("gauge"),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        BTreeSet::from(["gauge", "histogram", "non-monotonic sum", "sum"])
    );
    let points: Vec<u64> = sent.iter().map(Received::metric_point_count).collect();
    assert_eq!(points, [6, 7, 7, 7]);
    let stats = pipeline.stats().metrics;
    let reconciled = SignalStats {
        accepted: points.iter().sum(),
        exported: points[1..].iter().sum(),
        failed: points[0],
        ..SignalStats::default()
    };
    assert_eq!(stats, reconciled);
}

/// The string value of span attribute `key` in an OTLP request.
fn span_string(request: &Received, key: &str) -> Option<String> {
    request
        .traces()
        .resource_spans
        .into_iter()
        .flat_map(|r| r.scope_spans)
        .flat_map(|s| s.spans)
        .flat_map(|span| span.attributes)
        .find(|kv| kv.key == key)
        .and_then(|kv| kv.value?.value)
        .map(|v| match v {
            Wire::StringValue(s) => s,
            Wire::ArrayValue(a) => format!("{:?}", a.values),
            other => panic!("{key}: {other:?}"),
        })
}

#[test]
fn raised_content_budgets_are_clamped_so_the_span_is_exported() {
    // An attribute budget raised past what one request can carry
    // is clamped at configuration, so a quote-heavy 600 KB output is
    // truncated and its span exported, not dropped whole as an oversized
    // item. The ceiling follows the traces protocol: OTLP/JSON can double the
    // content (every quote is escaped), OTLP/protobuf sends it as retained
    // and so carries twice as much. Each pipeline also fits the budgets to
    // its own, smaller, request bound, and a bound under the envelope
    // allowance still carries content.
    use xybrid_observability::semconv::{
        attr, span_content_ceiling, DEFAULT_MAX_REQUEST_BODY_BYTES,
        DEFAULT_MAX_RETAINED_OPERATION_BYTES, SPAN_ENVELOPE_ALLOWANCE_BYTES,
    };
    use xybrid_observability::{OutputMessage, SpanContentEncoding};

    let quoted = "\"".repeat(600_000);
    for (bound, protocol, encoding, expected) in [
        (
            DEFAULT_MAX_REQUEST_BODY_BYTES,
            Protocol::HttpJson,
            SpanContentEncoding::Json,
            212_992,
        ),
        (
            DEFAULT_MAX_REQUEST_BODY_BYTES,
            Protocol::HttpProtobuf,
            SpanContentEncoding::Protobuf,
            DEFAULT_MAX_RETAINED_OPERATION_BYTES,
        ),
        (
            200_000,
            Protocol::HttpJson,
            SpanContentEncoding::Json,
            (200_000 - SPAN_ENVELOPE_ALLOWANCE_BYTES) / 2,
        ),
        (
            200_000,
            Protocol::HttpProtobuf,
            SpanContentEncoding::Protobuf,
            200_000 - SPAN_ENVELOPE_ALLOWANCE_BYTES,
        ),
        // Under the 64 KiB allowance: half the bound is reserved instead.
        (
            40_000,
            Protocol::HttpJson,
            SpanContentEncoding::Json,
            10_000,
        ),
        (
            40_000,
            Protocol::HttpProtobuf,
            SpanContentEncoding::Protobuf,
            20_000,
        ),
    ] {
        let what = format!("{protocol:?}, bound {bound}");
        let server = Server::ok();
        let mut config = server.traces_config(protocol);
        config.max_request_body_bytes = bound;
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        // The policy names the other encoding: the pipeline's protocol wins.
        let other = match encoding {
            SpanContentEncoding::Json => SpanContentEncoding::Protobuf,
            SpanContentEncoding::Protobuf => SpanContentEncoding::Json,
        };
        let observer = pipeline.observer_with(
            ContentPolicy::default()
                .with_capture(ContentCapture::SpanOnly)
                .with_span_content_encoding(other)
                .with_max_attribute_bytes(1_000_000)
                .with_max_retained_operation_bytes(1_000_000)
                // The 600 KB value is past the default 480 KiB preflight
                // bound, which would omit it before any truncation.
                .with_max_traversal_bytes(1_000_000),
        );
        let ceiling = span_content_ceiling(bound, encoding);
        assert_eq!(ceiling, expected, "{what}");
        let policy = observer.content_policy();
        assert_eq!(policy.span_content_encoding(), encoding, "{what}");
        assert_eq!(policy.max_request_body_bytes(), bound, "{what}");
        assert_eq!(policy.max_retained_operation_bytes(), ceiling, "{what}");
        assert_eq!(policy.max_attribute_bytes(), ceiling, "{what}");

        let op = observer.start_inference(
            InferenceRequest::new("chat", "llama.cpp").with_model("m"),
            StartOptions::default(),
        );
        op.record_output_messages(&[OutputMessage::assistant_text(quoted.clone(), None)]);
        op.end();
        pipeline.force_flush(Duration::from_secs(10)).unwrap();
        let stats = pipeline.stats().traces;
        assert_eq!(
            (stats.accepted, stats.exported, stats.dropped),
            (1, 1, 0),
            "{what}"
        );
        let sent = server.requests_to("/v1/traces");
        assert_eq!(sent.len(), 1, "{what}");
        assert!(
            sent[0].body.len() <= bound,
            "{what}: {}",
            sent[0].body.len()
        );
        let output = span_string(&sent[0], attr::OUTPUT_MESSAGES)
            .unwrap_or_else(|| panic!("{what}: the output was omitted"));
        assert!(output.len() <= ceiling, "{what}: {}", output.len());
        // The budget is used, not just respected. Truncation halves the
        // string until it fits, so more than half the ceiling arrives: over
        // protobuf that is more than the same bound carries as JSON.
        assert!(output.len() > ceiling / 2, "{what}: {}", output.len());
        assert!(output.contains("…[truncated]"), "{what}");
        let listed = span_string(&sent[0], attr::CONTENT_TRUNCATED_ATTRIBUTES).unwrap();
        assert!(listed.contains(attr::OUTPUT_MESSAGES), "{what}: {listed}");
        pipeline.shutdown(Duration::from_secs(5)).unwrap();
    }
}

#[test]
fn a_default_observer_keeps_the_protobuf_ceiling_of_content() {
    // With no budget configured, the packaged pipeline over OTLP/protobuf
    // (the default protocol) retains 416 KiB of content per operation; over
    // OTLP/JSON the same request bound carries 208 KiB.
    use xybrid_observability::semconv::attr;

    for (protocol, retained) in [
        (Protocol::HttpProtobuf, 425_984),
        (Protocol::HttpJson, 212_992),
    ] {
        let what = format!("{protocol:?}");
        let server = Server::ok();
        let pipeline = ObservabilityPipeline::build(server.traces_config(protocol)).unwrap();
        let observer =
            pipeline.observer_with(ContentPolicy::default().with_capture(ContentCapture::SpanOnly));
        let policy = observer.content_policy();
        assert_eq!(policy.max_retained_operation_bytes(), retained, "{what}");

        // Thirty 16,000-byte prompt variables ask for 480 KB: the operation
        // keeps what fits the retained budget and lists the rest.
        let mut request = InferenceRequest::new("chat", "llama.cpp").with_model("m");
        request.prompt = Some(xybrid_observability::PromptRef {
            name: "template".into(),
            version: None,
            variables: (0..30)
                .map(|index| (format!("variable_{index:02}"), "v".repeat(16_000)))
                .collect(),
        });
        let op = observer.start_inference(request, StartOptions::default());
        op.end();
        pipeline.force_flush(Duration::from_secs(10)).unwrap();
        let stats = pipeline.stats().traces;
        assert_eq!(
            (stats.accepted, stats.exported, stats.dropped),
            (1, 1, 0),
            "{what}"
        );
        let sent = server.requests_to("/v1/traces");
        assert_eq!(sent.len(), 1, "{what}");
        assert!(
            sent[0].body.len() <= 491_520,
            "{what}: {}",
            sent[0].body.len()
        );
        let key = |index: usize| format!("{}variable_{index:02}", attr::PROMPT_VARIABLE_PREFIX);
        let kept = (0..30)
            .filter(|index| span_string(&sent[0], &key(*index)).is_some())
            .count();
        // Each variable is charged 16 + its key + 16,000 bytes.
        let each = 16 + key(0).len() + 16_000;
        assert_eq!(kept, retained / each, "{what}");
        pipeline.shutdown(Duration::from_secs(5)).unwrap();
    }
}

#[test]
fn an_observer_takes_the_pipeline_traces_bound_only_when_it_is_smaller() {
    // The policy's own
    // request bound (default 480 KiB) is replaced by the traces bound only
    // when that is smaller. A larger pipeline bound does not raise it, and a
    // pipeline without traces has no span request to fit.
    use xybrid_observability::semconv::DEFAULT_MAX_REQUEST_BODY_BYTES as DEFAULT;
    let server = Server::ok();
    let fitted = |config: OtlpConfig, policy: ContentPolicy| {
        let pipeline = ObservabilityPipeline::build(config).unwrap();
        let bound = pipeline
            .observer_with(policy)
            .content_policy()
            .max_request_body_bytes();
        pipeline.shutdown(Duration::from_secs(5)).unwrap();
        bound
    };
    let traces = |bound: usize| {
        let mut config = server.traces_config(Protocol::HttpProtobuf);
        config.max_request_body_bytes = bound;
        config
    };
    let own = |bound: usize| ContentPolicy::default().with_max_request_body_bytes(bound);

    assert_eq!(fitted(traces(DEFAULT), own(100_000)), 100_000);
    assert_eq!(fitted(traces(DEFAULT), own(900_000)), DEFAULT);
    assert_eq!(fitted(traces(200_000), ContentPolicy::default()), 200_000);
    assert_eq!(fitted(traces(1_000_000), ContentPolicy::default()), DEFAULT);
    assert_eq!(fitted(traces(1_000_000), own(900_000)), 900_000);
    let logs_only = traces(70_000)
        .with_traces(None)
        .with_logs(Some(server.endpoint("logs", Protocol::HttpProtobuf)));
    assert_eq!(fitted(logs_only, ContentPolicy::default()), DEFAULT);
}
