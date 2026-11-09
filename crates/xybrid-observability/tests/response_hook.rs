//! Per-destination response hooks (`SignalEndpoint::with_response_hook`):
//! every response seen once, a disabling hook stopping only its own
//! destination with truthful counters, and a rebuilt pipeline starting
//! enabled again. Loopback `httpmock` endpoints, no network.

#![cfg(feature = "otlp")]

mod support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use httpmock::prelude::*;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SimpleSpanProcessor};
use xybrid_observability::otlp::{
    BatchLimits, EnvironmentSource, ObservabilityPipeline, OtlpConfig, Protocol, ResponseAction,
    ResponseHook, RetryLimits, RpcStatus, SignalEndpoint,
};
use xybrid_observability::{
    ContentCapture, ContentPolicy, Feedback, InferenceRequest, InputMessage, Location, Observer,
    StartOptions,
};

use support::{Action, Server};

fn span(observer: &Observer) {
    observer
        .start_inference(
            InferenceRequest::new("chat", "llama.cpp")
                .with_model("tiny")
                .with_location(Location::Local),
            StartOptions::default(),
        )
        .end();
}

/// What a hook saw of one response.
#[derive(Clone, Debug, PartialEq)]
struct Seen {
    signal: &'static str,
    status: u16,
    marker: Option<String>,
    body: Option<usize>,
    rpc: Option<RpcStatus>,
}

/// Records every response a hook sees and answers with `decide`.
#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<Vec<Seen>>>);

impl Recorder {
    fn hook(&self, decide: fn(&Seen) -> ResponseAction) -> ResponseHook {
        let seen = self.0.clone();
        ResponseHook::new(move |response| {
            let record = Seen {
                signal: response.signal(),
                status: response.status(),
                marker: response.header("x-marker").map(str::to_owned),
                body: response.body().map(<[u8]>::len),
                rpc: response.rpc_status(),
            };
            let action = decide(&record);
            seen.lock().unwrap().push(record);
            action
        })
    }

    fn seen(&self) -> Vec<Seen> {
        self.0.lock().unwrap().clone()
    }

    fn statuses(&self, signal: &str) -> Vec<u16> {
        self.seen()
            .iter()
            .filter(|s| s.signal == signal)
            .map(|s| s.status)
            .collect()
    }
}

fn keep(_: &Seen) -> ResponseAction {
    ResponseAction::Continue
}

/// Disables on a 401 carrying `x-marker: stop` (an adapter's explicit marker).
fn stop_on_marker(seen: &Seen) -> ResponseAction {
    if seen.status == 401 && seen.marker.as_deref() == Some("stop") {
        ResponseAction::Disable
    } else {
        ResponseAction::Continue
    }
}

/// No environment, retries fast, exports only on a full batch, flush or
/// shutdown.
fn base() -> OtlpConfig {
    OtlpConfig::default()
        .with_environment(EnvironmentSource::Disabled)
        .with_batch(BatchLimits {
            scheduled_delay: Duration::from_secs(60),
            ..BatchLimits::default()
        })
        .with_retry(RetryLimits {
            max_retries: 2,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(20),
        })
        .with_metric_export_interval(Duration::from_secs(600))
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let until = Instant::now() + Duration::from_secs(5);
    while !done() {
        assert!(Instant::now() < until, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn the_hook_sees_every_response_once_for_batch_flush_retry_and_shutdown() {
    let server = MockServer::start();
    let traces = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(200).header("x-marker", "seen");
    });
    let logs = server.mock(|when, then| {
        when.method(POST).path("/v1/logs");
        then.status(503);
    });
    let recorder = Recorder::default();
    let config = base()
        .with_traces(Some(
            SignalEndpoint::new(server.url("/v1/traces")).with_response_hook(recorder.hook(keep)),
        ))
        .with_logs(Some(
            SignalEndpoint::new(server.url("/v1/logs")).with_response_hook(recorder.hook(keep)),
        ))
        .with_batch(BatchLimits {
            max_export_batch_size: 2,
            scheduled_delay: Duration::from_secs(60),
            ..BatchLimits::default()
        });
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    let observer = pipeline.observer();

    // A full batch, exported by the worker on its own.
    span(&observer);
    span(&observer);
    wait_until("the batch export", || traces.hits() == 1);
    // A flush export.
    span(&observer);
    pipeline.force_flush(Duration::from_secs(5)).unwrap();
    assert_eq!(traces.hits(), 2);
    // A retried export: three 503 responses, each seen once. (The rated
    // span is the fourth span; it goes out with this flush.)
    let op = observer.start_inference(
        InferenceRequest::new("chat", "llama.cpp"),
        StartOptions::default(),
    );
    let target = op.observation_ref().expect("sampled");
    op.end();
    assert!(observer.record_feedback(
        &target,
        &Feedback {
            score: Some(1.0),
            ..Feedback::default()
        }
    ));
    assert!(pipeline
        .force_flush(Duration::from_secs(5))
        .unwrap_err()
        .is_delivery());
    logs.assert_hits(3);
    // The shutdown drain.
    span(&observer);
    pipeline.shutdown(Duration::from_secs(5)).unwrap();

    traces.assert_hits(4);
    assert_eq!(recorder.statuses("traces"), [200, 200, 200, 200]);
    assert_eq!(recorder.statuses("logs"), [503, 503, 503]);
    let seen = recorder.seen();
    assert!(seen
        .iter()
        .filter(|s| s.signal == "traces")
        .all(|s| s.marker.as_deref() == Some("seen") && s.body == Some(0) && s.rpc.is_none()));
    let stats = pipeline.stats();
    assert_eq!((stats.traces.exported, stats.traces.failed), (5, 0));
    assert_eq!((stats.logs.exported, stats.logs.failed), (0, 1));
    assert!(stats.traces.is_consistent() && stats.logs.is_consistent());
}

#[test]
fn a_disabled_destination_stops_and_drops_pending_and_later_items() {
    let server = MockServer::start();
    // The delay keeps the first export in flight while more spans queue up.
    let rejected = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(401)
            .header("x-marker", "stop")
            .body([0x08, 0x10, 0x12, 0x04, b's', b't', b'o', b'p'])
            .delay(Duration::from_millis(300));
    });
    let recorder = Recorder::default();
    let config = base()
        .with_traces(Some(
            SignalEndpoint::new(server.url("/v1/traces"))
                .with_response_hook(recorder.hook(stop_on_marker)),
        ))
        .with_batch(BatchLimits {
            max_export_batch_size: 2,
            scheduled_delay: Duration::from_secs(60),
            ..BatchLimits::default()
        });
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    let observer = pipeline.observer();

    for _ in 0..5 {
        span(&observer);
    }
    // The first two are in flight; the flush covers all five.
    let flushed = pipeline.force_flush(Duration::from_secs(5)).unwrap_err();
    assert!(
        matches!(
            flushed,
            xybrid_observability::Error::Delivery {
                failed: 2,
                dropped: 3,
                ..
            }
        ),
        "{flushed:?}"
    );
    assert_eq!(
        recorder.seen()[0].rpc,
        Some(RpcStatus::new(16, "stop")),
        "the hook reads the bounded failure body"
    );

    // Later work is refused at admission: no queue, no request.
    for _ in 0..3 {
        span(&observer);
    }
    pipeline.force_flush(Duration::from_secs(5)).unwrap();
    pipeline.shutdown(Duration::from_secs(5)).unwrap();

    rejected.assert_hits(1);
    assert_eq!(recorder.seen().len(), 1, "one response, seen once");
    let stats = pipeline.stats().traces;
    assert_eq!(
        (
            stats.accepted,
            stats.exported,
            stats.failed,
            stats.dropped,
            stats.pending
        ),
        (8, 0, 2, 6, 0),
        "nothing discarded is counted as exported"
    );
    assert!(stats.is_consistent());
}

#[test]
fn a_disabling_retryable_response_ends_the_retries() {
    let server = MockServer::start();
    let busy = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(503).header("x-marker", "stop");
    });
    let config = base()
        .with_traces(Some(
            SignalEndpoint::new(server.url("/v1/traces")).with_response_hook(ResponseHook::new(
                |response| {
                    if response.status() == 503 {
                        ResponseAction::Disable
                    } else {
                        ResponseAction::Continue
                    }
                },
            )),
        ))
        .with_retry(RetryLimits {
            max_retries: 5,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(10),
        });
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    span(&pipeline.observer());
    assert!(pipeline
        .force_flush(Duration::from_secs(5))
        .unwrap_err()
        .is_delivery());
    span(&pipeline.observer());
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    busy.assert_hits(1);
    let stats = pipeline.stats().traces;
    assert_eq!((stats.failed, stats.dropped, stats.exported), (1, 1, 0));
}

#[test]
fn a_disabling_retryable_response_fails_at_once_without_its_retry_wait() {
    let server = MockServer::start();
    // Retry-After fits the export budget: a continuing hook would wait it out.
    let busy = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(503).header("Retry-After", "4");
    });
    let config = base()
        .with_traces(Some(
            SignalEndpoint::new(server.url("/v1/traces")).with_response_hook(ResponseHook::new(
                |response| {
                    if response.status() == 503 {
                        ResponseAction::Disable
                    } else {
                        ResponseAction::Continue
                    }
                },
            )),
        ))
        .with_batch(BatchLimits {
            scheduled_delay: Duration::from_secs(60),
            export_timeout: Duration::from_secs(10),
            ..BatchLimits::default()
        });
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    span(&pipeline.observer());
    let started = Instant::now();
    assert!(pipeline
        .force_flush(Duration::from_secs(8))
        .unwrap_err()
        .is_delivery());
    let took = started.elapsed();
    assert!(
        took < Duration::from_secs(2),
        "the failed batch settled without the 4 s Retry-After wait: {took:?}"
    );
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    busy.assert_hits(1);
    let stats = pipeline.stats().traces;
    assert_eq!((stats.failed, stats.dropped, stats.exported), (1, 0, 0));
}

#[test]
fn a_stalled_failure_body_keeps_the_retry_budget() {
    // Two 503s whose bodies stall after the headers, then a 200. Without a
    // hook the 503 bodies are never read; with one they are, briefly.
    let stalled = Action::Trickle {
        status: 503,
        total: 64,
        chunk: 1,
        every: Duration::from_secs(30),
    };
    let server = Server::sequence(vec![stalled.clone(), stalled, Action::ok()]);
    let recorder = Recorder::default();
    let config = base()
        .with_traces(Some(
            server
                .endpoint("traces", Protocol::HttpProtobuf)
                .with_response_hook(recorder.hook(keep)),
        ))
        .with_batch(BatchLimits {
            scheduled_delay: Duration::from_secs(60),
            export_timeout: Duration::from_secs(4),
            ..BatchLimits::default()
        });
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    span(&pipeline.observer());
    let started = Instant::now();
    pipeline
        .force_flush(Duration::from_secs(8))
        .expect("the third attempt is acknowledged within the 4 s budget");
    let took = started.elapsed();
    assert!(took < Duration::from_secs(4), "{took:?}");
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    assert_eq!(server.count(), 3);
    let seen = recorder.seen();
    assert_eq!(recorder.statuses("traces"), [503, 503, 200]);
    assert_eq!(seen[0].body, None, "a stalled body is unreadable, not waited for");
    let stats = pipeline.stats().traces;
    assert_eq!((stats.exported, stats.failed), (1, 0));
}

#[test]
fn parts_of_a_split_batch_after_the_disabling_response_are_dropped() {
    let server = MockServer::start();
    let rejected = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(401).header("x-marker", "stop");
    });
    let recorder = Recorder::default();
    // Each span carries ~9 KB of captured input: one fits the 14 KB body
    // bound, two do not, so a batch of four is sent one span per request.
    let config = base()
        .with_traces(Some(
            SignalEndpoint::new(server.url("/v1/traces"))
                .with_response_hook(recorder.hook(stop_on_marker)),
        ))
        .with_max_request_body_bytes(14_000);
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    let observer =
        pipeline.observer_with(ContentPolicy::default().with_capture(ContentCapture::SpanOnly));
    let text = "x".repeat(9_000);
    for _ in 0..4 {
        let op = observer.start_inference(
            InferenceRequest::new("chat", "llama.cpp"),
            StartOptions::default(),
        );
        op.record_input_messages(&[InputMessage::text("user", text.as_str())]);
        op.end();
    }
    let flushed = pipeline.force_flush(Duration::from_secs(5)).unwrap_err();
    assert!(
        matches!(
            flushed,
            xybrid_observability::Error::Delivery {
                failed: 1,
                dropped: 3,
                ..
            }
        ),
        "{flushed:?}"
    );
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    rejected.assert_hits(1);
    assert!(pipeline.stats().traces.is_consistent());
}

#[test]
fn other_destinations_pipelines_and_local_spans_are_unaffected() {
    let server = MockServer::start();
    let rejected = server.mock(|when, then| {
        when.method(POST).path("/xybrid/v1/traces");
        then.status(401).header("x-marker", "stop");
    });
    let logs = server.mock(|when, then| {
        when.method(POST).path("/xybrid/v1/logs");
        then.status(200);
    });
    let metrics = server.mock(|when, then| {
        when.method(POST).path("/xybrid/v1/metrics");
        then.status(200);
    });
    let collector = server.mock(|when, then| {
        when.method(POST).path("/collector/v1/traces");
        then.status(401).header("x-marker", "stop");
    });
    let recorder = Recorder::default();
    let local = InMemorySpanExporter::default();
    let pipeline = ObservabilityPipeline::build_with_span_processor(
        base()
            .with_traces(Some(
                SignalEndpoint::new(server.url("/xybrid/v1/traces"))
                    .with_response_hook(recorder.hook(stop_on_marker)),
            ))
            .with_logs(Some(SignalEndpoint::new(server.url("/xybrid/v1/logs"))))
            .with_metrics(Some(SignalEndpoint::new(server.url("/xybrid/v1/metrics")))),
        SimpleSpanProcessor::new(local.clone()),
    )
    .unwrap();
    // Another pipeline to a Collector answering the same 401, without a hook.
    let other = ObservabilityPipeline::build(base().with_traces(Some(SignalEndpoint::new(
        server.url("/collector/v1/traces"),
    ))))
    .unwrap();

    let observer = pipeline.observer();
    span(&observer);
    assert!(pipeline.force_flush(Duration::from_secs(5)).is_err());
    rejected.assert_hits(1);

    // The disabled traces destination sends nothing more...
    let op = observer.start_inference(
        InferenceRequest::new("chat", "llama.cpp"),
        StartOptions::default(),
    );
    let target = op.observation_ref().expect("local spans are still sampled");
    op.end();
    // ...while this pipeline's logs and metrics, the local span processor and
    // the other pipeline keep working.
    assert!(observer.record_feedback(&target, &Feedback::default()));
    let _ = pipeline.force_flush(Duration::from_secs(5));
    for _ in 0..2 {
        span(&other.observer());
        let _ = other.force_flush(Duration::from_secs(5));
    }
    assert_eq!(local.get_finished_spans().unwrap().len(), 2);
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    let _ = other.shutdown(Duration::from_secs(5));

    rejected.assert_hits(1);
    logs.assert_hits(1);
    assert!(metrics.hits() >= 1);
    collector.assert_hits(2);
    let stats = pipeline.stats();
    assert_eq!(
        (
            stats.traces.failed,
            stats.traces.dropped,
            stats.logs.exported
        ),
        (1, 1, 1)
    );
    assert_eq!(other.stats().traces.failed, 2);
}

#[test]
fn a_disabled_metrics_destination_refuses_later_collections() {
    let server = MockServer::start();
    let rejected = server.mock(|when, then| {
        when.method(POST).path("/v1/metrics");
        then.status(401).header("x-marker", "stop");
    });
    let recorder = Recorder::default();
    let pipeline = ObservabilityPipeline::build(
        base().with_metrics(Some(
            SignalEndpoint::new(server.url("/v1/metrics"))
                .with_response_hook(recorder.hook(stop_on_marker)),
        )),
    )
    .unwrap();
    let observer = pipeline.observer();
    span(&observer);
    assert!(pipeline.force_flush(Duration::from_secs(5)).is_err());
    span(&observer);
    assert!(pipeline.force_flush(Duration::from_secs(5)).is_err());
    let _ = pipeline.shutdown(Duration::from_secs(5));
    rejected.assert_hits(1);
    let stats = pipeline.stats().metrics;
    assert!(stats.failed > 0 && stats.dropped > 0, "{stats:?}");
    assert_eq!(stats.exported, 0);
    assert!(stats.is_consistent());
}

#[test]
fn a_rebuilt_pipeline_starts_with_the_destination_enabled() {
    let server = MockServer::start();
    let mut rejected = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(401).header("x-marker", "stop");
    });
    let recorder = Recorder::default();
    // One configuration, one hook, two builds.
    let config = base().with_traces(Some(
        SignalEndpoint::new(server.url("/v1/traces"))
            .with_protocol(Protocol::HttpJson)
            .with_response_hook(recorder.hook(stop_on_marker)),
    ));

    let first = ObservabilityPipeline::build(config.clone()).unwrap();
    span(&first.observer());
    assert!(first.force_flush(Duration::from_secs(5)).is_err());
    span(&first.observer());
    first.shutdown(Duration::from_secs(5)).unwrap();
    rejected.assert_hits(1);
    assert_eq!(first.stats().traces.dropped, 1);

    // The workspace was claimed: the endpoint accepts the key again.
    rejected.delete();
    let accepted = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(200);
    });
    let second = ObservabilityPipeline::build(config).unwrap();
    span(&second.observer());
    second.force_flush(Duration::from_secs(5)).unwrap();
    second.shutdown(Duration::from_secs(5)).unwrap();
    accepted.assert_hits(1);
    assert_eq!(second.stats().traces.exported, 1);
    assert_eq!(recorder.statuses("traces"), [401, 200]);
}

#[test]
fn a_continuing_hook_leaves_failures_and_retries_as_they_were() {
    let server = MockServer::start();
    let unauthorized = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(401);
    });
    let recorder = Recorder::default();
    let pipeline = ObservabilityPipeline::build(
        base().with_traces(Some(
            SignalEndpoint::new(server.url("/v1/traces"))
                .with_response_hook(recorder.hook(stop_on_marker)),
        )),
    )
    .unwrap();
    for _ in 0..3 {
        span(&pipeline.observer());
        assert!(pipeline.force_flush(Duration::from_secs(5)).is_err());
    }
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    // A 401 without the marker is an ordinary, non-retryable failure each time.
    unauthorized.assert_hits(3);
    let stats = pipeline.stats().traces;
    assert_eq!((stats.failed, stats.dropped), (3, 0));
}
