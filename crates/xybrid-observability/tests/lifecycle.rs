//! Lifecycle regressions: flush barriers (R-CANCEL-4..6), idempotent shutdown
//! and drops after it (R-CANCEL-7, R-CNT-5), hot paths against a hung
//! endpoint (R-CANCEL-9), the metrics drain honouring the shutdown timeout
//! (D30.3/D30.4), durations of any size, application span processors and
//! local trace serialization.
#![cfg(feature = "otlp")]

mod support;

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use opentelemetry::logs::{AnyValue, LogRecord as _, Logger as _, LoggerProvider as _};
use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
use opentelemetry::Context;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::any_value::Value as Wire;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::trace::{
    InMemorySpanExporter, SdkTracerProvider, SimpleSpanProcessor, Span, SpanData, SpanExporter,
    SpanProcessor,
};
use opentelemetry_sdk::Resource;
use support::{Action, Server};
use xybrid_observability::otlp::{
    exporters, trace_json, BatchLimits, EnvironmentSource, ObservabilityPipeline, OtlpConfig,
    Protocol, RetryLimits, SignalEndpoint,
};
use xybrid_observability::semconv::{attr, SCOPE_NAME};
use xybrid_observability::{
    ContentCapture, ContentPolicy, Error, InferenceRequest, KeyValue, Location, Observer,
    StartOptions, Usage,
};

fn spans(pipeline: &ObservabilityPipeline, names: &[&str]) {
    let tracer = pipeline.tracer_provider().unwrap().tracer("app");
    for name in names {
        tracer.start(name.to_string()).end();
    }
}

fn logs(pipeline: &ObservabilityPipeline, n: usize) {
    let logger = pipeline.logger_provider().unwrap().logger("app");
    for i in 0..n {
        let mut record = logger.create_log_record();
        record.set_body(AnyValue::from(format!("log-{i}")));
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

/// Long scheduled delay and big batches: spans are exported only by a
/// flush, a shutdown or a full batch.
fn held(batch: usize) -> BatchLimits {
    BatchLimits {
        max_export_batch_size: batch,
        scheduled_delay: Duration::from_secs(60),
        ..BatchLimits::default()
    }
}

fn text(result: &xybrid_observability::Result<()>) -> Result<(), String> {
    result.as_ref().map(|_| ()).map_err(ToString::to_string)
}

#[test]
fn shutdown_during_a_flush_delivers_every_batch_and_succeeds() {
    // R-CANCEL-4.
    let server = Server::always(Action::ok().after(Duration::from_millis(100)));
    let pipeline = Arc::new(
        ObservabilityPipeline::build(
            server
                .traces_config(Protocol::HttpProtobuf)
                .with_batch(held(2)),
        )
        .unwrap(),
    );
    spans(
        &pipeline,
        &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    );
    let copy = pipeline.clone();
    let flush = thread::spawn(move || copy.force_flush(Duration::from_secs(10)));
    server.wait_for(1, Duration::from_secs(3));
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    assert_eq!(server.count(), 5, "every batch was delivered");
    assert!(flush.join().unwrap().is_ok());
    let stats = pipeline.stats().traces;
    assert_eq!((stats.exported, stats.pending), (10, 0));
}

#[test]
fn concurrent_flushes_each_wait_for_their_own_barrier() {
    // R-CANCEL-5: flush 1 covers A only; flush 2, started while flush 1 is
    // running and after B was accepted, waits for B.
    let server = Server::always(Action::ok().after(Duration::from_millis(300)));
    let pipeline = Arc::new(
        ObservabilityPipeline::build(
            server
                .traces_config(Protocol::HttpProtobuf)
                .with_batch(held(512)),
        )
        .unwrap(),
    );
    spans(&pipeline, &["A"]);
    let flush = |p: &Arc<ObservabilityPipeline>| {
        let p = p.clone();
        thread::spawn(move || {
            let result = p.force_flush(Duration::from_secs(5));
            (result, Instant::now())
        })
    };
    let first = flush(&pipeline);
    // A is exported only because flush 1 registered its barrier.
    server.wait_for(1, Duration::from_secs(3));
    spans(&pipeline, &["B"]);
    let second = flush(&pipeline);
    let (r1, t1) = first.join().unwrap();
    let (r2, t2) = second.join().unwrap();
    assert!(r1.is_ok() && r2.is_ok(), "{r1:?} {r2:?}");
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].span_names(), ["A"]);
    assert_eq!(requests[1].span_names(), ["B"]);
    let b_sent = requests[1].at;
    assert!(
        t1 < b_sent + Duration::from_millis(200),
        "flush 1 did not wait for B"
    );
    assert!(
        t2 >= b_sent + Duration::from_millis(250),
        "flush 2 waited for B and did not resolve with flush 1"
    );
    assert_eq!(pipeline.stats().traces.exported, 2);
}

#[test]
fn items_accepted_after_a_flush_call_do_not_hold_it() {
    // R-CANCEL-6: B, accepted after the call, is stuck behind a hung
    // endpoint; the flush still completes once A is delivered.
    let server = Server::sequence(vec![
        Action::ok().after(Duration::from_millis(100)),
        Action::Hang,
    ]);
    let pipeline = Arc::new(
        ObservabilityPipeline::build(
            server
                .traces_config(Protocol::HttpProtobuf)
                .with_batch(held(512)),
        )
        .unwrap(),
    );
    spans(&pipeline, &["A"]);
    let copy = pipeline.clone();
    let started = Instant::now();
    let flush = thread::spawn(move || copy.force_flush(Duration::from_secs(3)));
    server.wait_for(1, Duration::from_secs(3));
    spans(&pipeline, &["B"]);
    assert!(flush.join().unwrap().is_ok());
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(pipeline.stats().traces.pending, 1, "B is still pending");
    let result = pipeline.shutdown(Duration::from_millis(200));
    assert!(result.as_ref().is_err_and(Error::is_timeout), "{result:?}");
    let stats = pipeline.stats().traces;
    assert_eq!(
        (stats.exported, stats.failed, stats.pending),
        (1, 1, 0),
        "{stats:?}"
    );
}

#[test]
fn shutdown_is_idempotent_and_drops_items_offered_after_it() {
    // R-CANCEL-7: identical result, no new requests, flush returns it.
    for action in [Action::ok(), Action::status(400)] {
        let server = Server::always(action);
        let pipeline = ObservabilityPipeline::build(
            server
                .traces_config(Protocol::HttpProtobuf)
                .with_logs(Some(server.endpoint("logs", Protocol::HttpProtobuf)))
                .with_retry(RetryLimits {
                    max_retries: 0,
                    ..RetryLimits::default()
                }),
        )
        .unwrap();
        spans(&pipeline, &["a", "b"]);
        logs(&pipeline, 2);
        let first = pipeline.shutdown(Duration::from_secs(5));
        let requests = server.count();
        assert_eq!(
            text(&pipeline.shutdown(Duration::from_secs(5))),
            text(&first)
        );
        assert_eq!(
            text(&pipeline.force_flush(Duration::from_secs(5))),
            text(&first)
        );
        assert_eq!(server.count(), requests, "no new request");
        if first.is_err() {
            assert!(
                matches!(first, Err(Error::Delivery { failed: 4, .. })),
                "{first:?}"
            );
        }
    }

    // Items offered after shutdown began are dropped and counted, while the
    // providers still observe them (the drain is still running).
    let server = Server::always(Action::Hang);
    let pipeline = Arc::new(
        ObservabilityPipeline::build(
            server
                .traces_config(Protocol::HttpProtobuf)
                .with_logs(Some(server.endpoint("logs", Protocol::HttpProtobuf)))
                .with_batch(held(512)),
        )
        .unwrap(),
    );
    spans(&pipeline, &["in-flight"]);
    let copy = pipeline.clone();
    let shutdown = thread::spawn(move || copy.shutdown(Duration::from_millis(800)));
    while !pipeline.is_shut_down() {
        thread::yield_now();
    }
    spans(&pipeline, &["late-1", "late-2", "late-3"]);
    logs(&pipeline, 3);
    let result = shutdown.join().unwrap();
    assert!(result.as_ref().is_err_and(Error::is_timeout), "{result:?}");
    let stats = pipeline.stats();
    assert_eq!(
        (
            stats.traces.accepted,
            stats.traces.dropped,
            stats.traces.failed,
            stats.traces.pending
        ),
        (4, 3, 1, 0),
        "{stats:?}"
    );
    assert_eq!((stats.logs.accepted, stats.logs.dropped), (3, 3));
}

#[test]
fn hot_paths_return_promptly_with_a_hung_endpoint() {
    // R-CANCEL-9 for spans, log records and metric recordings.
    let server = Server::always(Action::Hang);
    let config = server
        .all_config(Protocol::HttpProtobuf)
        .with_batch(BatchLimits {
            max_queue_size: 16,
            max_export_batch_size: 4,
            scheduled_delay: Duration::from_millis(1),
            export_timeout: Duration::from_secs(30),
            ..BatchLimits::default()
        })
        .with_metric_export_interval(Duration::from_millis(20));
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    let observer =
        pipeline.observer_with(ContentPolicy::default().with_capture(ContentCapture::SpanAndEvent));
    let logger = pipeline.logger_provider().unwrap().logger("app");
    // Let every signal get an export stuck on the endpoint first.
    inference(&observer);
    server.wait_for(3, Duration::from_secs(3));
    let started = Instant::now();
    for i in 0..500 {
        inference(&observer);
        let mut record = logger.create_log_record();
        record.set_body(AnyValue::from(format!("hot-{i}")));
        logger.emit(record);
    }
    let hot = started.elapsed();
    assert!(hot < Duration::from_secs(2), "hot paths took {hot:?}");
    let stats = pipeline.stats();
    assert!(stats.traces.pending <= 16 && stats.traces.dropped > 0);
    assert!(stats.logs.pending <= 16 && stats.logs.dropped > 0);
    let started = Instant::now();
    assert!(pipeline.shutdown(Duration::from_millis(300)).is_err());
    assert!(started.elapsed() < Duration::from_secs(3));
    let stats = pipeline.stats();
    for s in [stats.traces, stats.logs, stats.metrics] {
        assert!(s.is_consistent() && s.pending == 0, "{s:?}");
    }
}

fn metrics_only(server: &Server) -> xybrid_observability::otlp::OtlpConfig {
    server
        .traces_config(Protocol::HttpProtobuf)
        .with_traces(None)
        .with_metrics(Some(server.endpoint("metrics", Protocol::HttpProtobuf)))
        .with_batch(BatchLimits {
            export_timeout: Duration::from_secs(20),
            ..BatchLimits::default()
        })
}

#[test]
fn the_metrics_drain_honours_a_shutdown_timeout_above_five_seconds() {
    // D30.3(b, d): the SDK's periodic reader stops waiting after a fixed
    // 5 s; the pipeline must keep draining until its own deadline and settle
    // the counters before it returns.
    let server = Server::always(Action::ok().after(Duration::from_millis(6500)));
    let pipeline = ObservabilityPipeline::build(metrics_only(&server)).unwrap();
    inference(&pipeline.observer());
    let started = Instant::now();
    let result = pipeline.shutdown(Duration::from_secs(15));
    let elapsed = started.elapsed();
    let stats = pipeline.stats().metrics;
    assert!(result.is_ok(), "{result:?} after {elapsed:?}, {stats:?}");
    assert!(elapsed >= Duration::from_millis(6400), "{elapsed:?}");
    assert!(stats.accepted > 0);
    assert_eq!(
        (stats.exported, stats.failed, stats.pending),
        (stats.accepted, 0, 0),
        "{stats:?}"
    );
    assert_eq!(server.requests_to("/v1/metrics").len(), 1);
}

#[test]
fn the_metrics_drain_reports_a_timeout_and_settles_when_cut_off() {
    // D30.3(c, d), D30.4: the final export outlives the shutdown timeout.
    let server = Server::always(Action::ok().after(Duration::from_secs(3)));
    let pipeline = ObservabilityPipeline::build(metrics_only(&server)).unwrap();
    inference(&pipeline.observer());
    let started = Instant::now();
    let result = pipeline.shutdown(Duration::from_secs(1));
    let elapsed = started.elapsed();
    assert!(result.as_ref().is_err_and(Error::is_timeout), "{result:?}");
    assert!(elapsed < Duration::from_millis(2500), "{elapsed:?}");
    let stats = pipeline.stats().metrics;
    assert!(stats.accepted > 0);
    assert_eq!(
        (stats.failed, stats.pending),
        (stats.accepted, 0),
        "{stats:?}"
    );
    server.wait_disconnect(0, Duration::from_secs(2));
    // The remembered result does not change.
    assert_eq!(
        text(&pipeline.shutdown(Duration::from_secs(1))),
        text(&result)
    );
}

#[test]
fn the_metrics_drain_reports_delivery_failures() {
    // D30.4: completed in time with losses.
    let server = Server::always(Action::status(400));
    let pipeline = ObservabilityPipeline::build(metrics_only(&server)).unwrap();
    inference(&pipeline.observer());
    let result = pipeline.shutdown(Duration::from_secs(5));
    let stats = pipeline.stats().metrics;
    assert!(
        matches!(result, Err(Error::Delivery { failed, dropped: 0, .. }) if failed == stats.accepted),
        "{result:?} {stats:?}"
    );
    assert_eq!(stats.pending, 0);
}

#[test]
fn concurrent_metric_flushes_never_starve_host_name_resolution() {
    // Each metrics flush parks a blocking SDK call until the periodic reader
    // exported its collection, and reqwest resolves host names on the same
    // blocking pool. Concurrent flushes must leave room for the resolution
    // their own exports need (the endpoint is `localhost`, not an address).
    let server = Server::ok();
    let port = server.url.rsplit(':').next().unwrap().to_owned();
    let config = OtlpConfig::default()
        .with_environment(EnvironmentSource::Disabled)
        .with_metrics(Some(SignalEndpoint::new(format!(
            "http://localhost:{port}/v1/metrics"
        ))))
        // Metric export deadline: min(export_timeout, interval) = 2 s.
        .with_metric_export_interval(Duration::from_secs(2));
    let pipeline = Arc::new(ObservabilityPipeline::build(config).unwrap());
    inference(&pipeline.observer());
    let started = Instant::now();
    let flushes: Vec<_> = (0..4)
        .map(|_| {
            let p = pipeline.clone();
            thread::spawn(move || p.force_flush(Duration::from_secs(5)))
        })
        .collect();
    for flush in flushes {
        let result = flush.join().unwrap();
        assert!(result.is_ok(), "{result:?}");
    }
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_millis(1500), "{elapsed:?}");
    let stats = pipeline.stats().metrics;
    assert_eq!(stats.failed, 0, "{stats:?}");
    assert!(server.requests_to("/v1/metrics").len() >= 4);
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
}

#[test]
fn durations_of_any_size_are_valid_and_never_panic() {
    // `Duration::MAX` is a valid limit and lifecycle timeout ("practically
    // never"); deadline arithmetic saturates instead of overflowing.
    let server = Server::ok();
    let endpoint = |signal: &str| {
        server
            .endpoint(signal, Protocol::HttpProtobuf)
            .with_timeout(Duration::MAX)
    };
    let config = server
        .all_config(Protocol::HttpProtobuf)
        .with_traces(Some(endpoint("traces")))
        .with_logs(Some(endpoint("logs")))
        .with_metrics(Some(endpoint("metrics")))
        .with_batch(BatchLimits {
            scheduled_delay: Duration::MAX,
            export_timeout: Duration::MAX,
            ..BatchLimits::default()
        })
        .with_retry(RetryLimits {
            max_retries: 3,
            initial_delay: Duration::MAX,
            max_delay: Duration::MAX,
        })
        .with_metric_export_interval(Duration::MAX);
    let pipeline = ObservabilityPipeline::build(config.clone()).unwrap();
    let observer =
        pipeline.observer_with(ContentPolicy::default().with_capture(ContentCapture::EventOnly));
    inference(&observer);
    pipeline.force_flush(Duration::MAX).unwrap();
    inference(&observer);
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(pipeline.force_flush_async(Duration::MAX))
        .unwrap();
    inference(&observer);
    pipeline.shutdown(Duration::MAX).unwrap();
    let stats = pipeline.stats();
    for (signal, s) in [
        ("traces", stats.traces),
        ("logs", stats.logs),
        ("metrics", stats.metrics),
    ] {
        assert!(s.exported > 0, "{signal}: {s:?}");
        assert_eq!((s.exported, s.pending), (s.accepted, 0), "{signal}");
    }
    assert_eq!(stats.traces.exported, 3);

    // Standalone exporters with the same limits.
    let exporter = exporters(&config).unwrap().traces.unwrap();
    let memory = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(memory.clone())
        .build();
    inference(&Observer::builder().tracer_provider(&provider).build());
    let finished = memory.get_finished_spans().unwrap();
    futures_lite_block_on(exporter.export(finished)).unwrap();
    assert_eq!(exporter.stats().exported, 1);
    exporter.shutdown_with_timeout(Duration::MAX).unwrap();
}

/// Drives a future on a throwaway Tokio runtime (the exporter never needs
/// the caller's runtime; any executor works).
fn futures_lite_block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(future)
}

#[test]
fn an_application_span_processor_sees_the_spans_and_traces_can_stay_local() {
    // `build_with_span_processor`: the application's processor observes the
    // same sampled spans the transport exports.
    let server = Server::ok();
    let memory = InMemorySpanExporter::default();
    let pipeline = ObservabilityPipeline::build_with_span_processor(
        server.traces_config(Protocol::HttpProtobuf),
        SimpleSpanProcessor::new(memory.clone()),
    )
    .unwrap();
    inference(&pipeline.observer());
    // The in-memory exporter forgets its spans on shutdown: read first.
    let local: Vec<String> = memory
        .get_finished_spans()
        .unwrap()
        .into_iter()
        .map(|s| s.name.into_owned())
        .collect();
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    assert_eq!(local.len(), 1);
    let remote: Vec<String> = server
        .requests_to("/v1/traces")
        .iter()
        .flat_map(|r| r.span_names())
        .collect();
    assert_eq!(remote, local);

    // Without a traces endpoint, spans stay local while logs export.
    let server = Server::ok();
    let memory = InMemorySpanExporter::default();
    let config = server
        .traces_config(Protocol::HttpProtobuf)
        .with_traces(None)
        .with_logs(Some(server.endpoint("logs", Protocol::HttpProtobuf)));
    let pipeline = ObservabilityPipeline::build_with_span_processor(
        config,
        SimpleSpanProcessor::new(memory.clone()),
    )
    .unwrap();
    assert!(pipeline.tracer_provider().is_some());
    inference(
        &pipeline.observer_with(ContentPolicy::default().with_capture(ContentCapture::EventOnly)),
    );
    assert_eq!(memory.get_finished_spans().unwrap().len(), 1);
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    assert!(server.requests_to("/v1/traces").is_empty());
    assert!(!server.requests_to("/v1/logs").is_empty());
    let stats = pipeline.stats();
    assert_eq!(stats.traces, Default::default());
    assert!(stats.logs.exported > 0);
}

/// An application processor whose shutdown blocks far longer than the
/// pipeline's shutdown timeout.
#[derive(Debug)]
struct Stubborn;

impl SpanProcessor for Stubborn {
    fn on_start(&self, _span: &mut Span, _cx: &Context) {}

    fn on_end(&self, _span: SpanData) {}

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        thread::sleep(Duration::from_secs(4));
        Ok(())
    }
}

#[test]
fn a_blocking_application_processor_cannot_hold_the_shutdown() {
    let server = Server::ok();
    let pipeline = ObservabilityPipeline::build_with_span_processor(
        server.traces_config(Protocol::HttpProtobuf),
        Stubborn,
    )
    .unwrap();
    inference(&pipeline.observer());
    let started = Instant::now();
    let result = pipeline.shutdown(Duration::from_millis(200));
    let elapsed = started.elapsed();
    assert!(result.as_ref().is_err_and(Error::is_timeout), "{result:?}");
    // Bounded by the deadline plus one settle grace (1 s), not by the
    // processor and not by the caller's extra wait.
    assert!(elapsed < Duration::from_millis(1900), "{elapsed:?}");
    assert_eq!(pipeline.stats().traces.exported, 1);
}

/// An application processor whose shutdown fails: returns an error, or
/// panics.
#[derive(Debug)]
struct Failing {
    panics: bool,
}

impl SpanProcessor for Failing {
    fn on_start(&self, _span: &mut Span, _cx: &Context) {}

    fn on_end(&self, _span: SpanData) {}

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        assert!(!self.panics, "application processor panicked on shutdown");
        Err(OTelSdkError::InternalFailure(
            "private processor detail".into(),
        ))
    }
}

#[test]
fn a_failing_application_processor_makes_shutdown_fail_and_is_remembered() {
    // D30.5: the result is truthful even when every item was delivered.
    for panics in [false, true] {
        let server = Server::ok();
        let pipeline = ObservabilityPipeline::build_with_span_processor(
            server.traces_config(Protocol::HttpProtobuf),
            Failing { panics },
        )
        .unwrap();
        inference(&pipeline.observer());
        let result = pipeline.shutdown(Duration::from_secs(5));
        let Err(Error::Lifecycle { operation, message }) = &result else {
            panic!("panics={panics}: expected a lifecycle failure, got {result:?}");
        };
        assert_eq!(*operation, "shutdown");
        assert!(
            !message.contains("private"),
            "application error text stays out of the result: {message}"
        );
        assert_eq!(pipeline.stats().traces.exported, 1, "panics={panics}");
        // Every later call (and a flush) returns the first call's result.
        assert_eq!(text(&pipeline.shutdown(Duration::ZERO)), text(&result));
        assert_eq!(text(&pipeline.force_flush(Duration::ZERO)), text(&result));
    }

    // A provider the application already shut down is not a failure.
    let server = Server::ok();
    let pipeline = ObservabilityPipeline::build_with_span_processor(
        server.traces_config(Protocol::HttpProtobuf),
        SimpleSpanProcessor::new(InMemorySpanExporter::default()),
    )
    .unwrap();
    inference(&pipeline.observer());
    pipeline.tracer_provider().unwrap().shutdown().unwrap();
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    assert_eq!(pipeline.stats().traces.exported, 1);
}

#[test]
fn trace_json_uses_the_export_grouping_and_catalog_types() {
    let memory = InMemorySpanExporter::default();
    let resource = Resource::builder_empty().with_service_name("svc").build();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(memory.clone())
        .with_resource(resource.clone())
        .build();
    let observer = Observer::builder().tracer_provider(&provider).build();
    let op = observer.start_inference(
        InferenceRequest::new("chat", "fake"),
        StartOptions::default(),
    );
    op.set_attribute(KeyValue::new(attr::REQUEST_TOP_P, 1_i64));
    op.end();
    let mut app = provider.tracer("app").start("app-span");
    app.set_attribute(KeyValue::new(attr::REQUEST_TEMPERATURE, 1_i64));
    app.end();

    let json = trace_json(memory.get_finished_spans().unwrap(), &resource).unwrap();
    let request: ExportTraceServiceRequest = serde_json::from_slice(&json).unwrap();
    assert_eq!(request.resource_spans.len(), 1);
    let resource_spans = &request.resource_spans[0];
    assert!(resource_spans
        .resource
        .as_ref()
        .unwrap()
        .attributes
        .iter()
        .any(|kv| kv.key == "service.name"
            && kv.value.as_ref().unwrap().value == Some(Wire::StringValue("svc".into()))));
    let groups = &resource_spans.scope_spans;
    assert_eq!(groups.len(), 2);
    let value = |scope: &str, key: &str| {
        groups
            .iter()
            .find(|g| g.scope.as_ref().unwrap().name == scope)
            .unwrap()
            .spans[0]
            .attributes
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| kv.value.as_ref())
            .and_then(|v| v.value.clone())
    };
    assert_eq!(
        value(SCOPE_NAME, attr::REQUEST_TOP_P),
        Some(Wire::DoubleValue(1.0))
    );
    assert_eq!(
        value("app", attr::REQUEST_TEMPERATURE),
        Some(Wire::IntValue(1))
    );
}
