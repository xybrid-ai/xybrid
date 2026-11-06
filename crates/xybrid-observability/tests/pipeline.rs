//! OTLP pipeline delivery: bounds, failure handling, lifecycle, presets, and
//! propagation across threads.

#![cfg(feature = "otlp")]

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use httpmock::prelude::*;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use xybrid_observability::otlp::{
    BatchLimits, ObservabilityPipeline, OtlpConfig, Protocol, RetryLimits, SignalEndpoint,
};
use xybrid_observability::{
    InferenceRequest, InputMessage, Location, ObservationContext, Observer, StartOptions,
    ToolCallInfo, Usage,
};

fn chat(observer: &Observer, options: StartOptions) -> xybrid_observability::InferenceOperation {
    observer.start_inference(
        InferenceRequest::new("chat", "llama.cpp")
            .with_model("tiny")
            .with_location(Location::Local),
        options,
    )
}

#[test]
fn xybrid_preset_sends_only_traces_with_bearer_auth() {
    let server = MockServer::start();
    let traces = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/traces")
            .header("authorization", "Bearer sk-test")
            .header("content-type", "application/x-protobuf");
        then.status(200);
    });
    let anything_else = server.mock(|when, then| {
        when.path_matches(httpmock::Regex::new("^/v1/(metrics|logs)").unwrap());
        then.status(200);
    });

    let pipeline = ObservabilityPipeline::build(OtlpConfig::xybrid_traces_to(
        &server.url("/v1/traces"),
        "sk-test",
    ))
    .unwrap();
    let observer = pipeline.observer();
    let op = chat(&observer, StartOptions::default());
    op.record_input_messages(&[InputMessage::text("user", "hello")]);
    op.record_usage(&Usage::tokens(Some(1), Some(2)));
    op.end();
    pipeline.force_flush(Duration::from_secs(5)).unwrap();
    pipeline.shutdown(Duration::from_secs(5)).unwrap();

    traces.assert_hits(1);
    anything_else.assert_hits(0);
    let stats = pipeline.stats();
    assert_eq!(stats.spans_exported, 1);
    assert_eq!(stats.spans_dropped, 0);
}

#[test]
fn json_protocol_is_supported() {
    let server = MockServer::start();
    let traces = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/traces")
            .header("content-type", "application/json")
            .body_contains("\"gen_ai.operation.name\"");
        then.status(200);
    });
    let pipeline = ObservabilityPipeline::build(OtlpConfig::default().with_traces(Some(
        SignalEndpoint::new(server.url("/v1/traces")).with_protocol(Protocol::HttpJson),
    )))
    .unwrap();
    chat(&pipeline.observer(), StartOptions::default()).end();
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    traces.assert_hits(1);
}

#[test]
fn queue_saturation_drops_and_counts_without_blocking() {
    // Unroutable endpoint (TEST-NET-1) with a long scheduled delay so spans
    // pile up in the bounded queue.
    let config = OtlpConfig::default()
        .with_traces(Some(
            SignalEndpoint::new("http://192.0.2.1:9/v1/traces")
                .with_timeout(Duration::from_millis(200)),
        ))
        .with_batch(BatchLimits {
            max_queue_size: 8,
            max_export_batch_size: 512,
            scheduled_delay: Duration::from_secs(60),
            export_timeout: Duration::from_millis(200),
        })
        .with_retry(RetryLimits {
            max_retries: 0,
            ..RetryLimits::default()
        });
    let pipeline = ObservabilityPipeline::build(config).unwrap();
    let observer = pipeline.observer();

    let started = Instant::now();
    for _ in 0..100 {
        chat(&observer, StartOptions::default()).end();
    }
    let hot_path = started.elapsed();
    assert!(
        hot_path < Duration::from_millis(500),
        "hot path took {hot_path:?}"
    );

    let stats = pipeline.stats();
    assert_eq!(stats.spans_queued, 8);
    assert_eq!(stats.spans_dropped, 92);

    // Shutdown is bounded even though the destination never answers.
    let started = Instant::now();
    let _ = pipeline.shutdown(Duration::from_secs(2));
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(
        pipeline.shutdown(Duration::from_secs(2)).is_ok(),
        "second shutdown is a no-op"
    );
    assert!(
        pipeline.force_flush(Duration::from_secs(2)).is_ok(),
        "flush after shutdown is a no-op"
    );
}

#[test]
fn unavailable_exporter_counts_failures() {
    let server = MockServer::start();
    let failing = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(400);
    });
    let pipeline = ObservabilityPipeline::build(
        OtlpConfig::default()
            .with_traces(Some(SignalEndpoint::new(server.url("/v1/traces"))))
            .with_retry(RetryLimits {
                max_retries: 0,
                ..RetryLimits::default()
            }),
    )
    .unwrap();
    for _ in 0..3 {
        chat(&pipeline.observer(), StartOptions::default()).end();
    }
    let _ = pipeline.force_flush(Duration::from_secs(5));
    pipeline.shutdown(Duration::from_secs(5)).ok();
    failing.assert_hits(1);
    let stats = pipeline.stats();
    assert_eq!(stats.export_failures, 1);
    assert_eq!(stats.spans_failed, 3);
    assert_eq!(stats.spans_exported, 0);
}

#[test]
fn retries_are_bounded() {
    let server = MockServer::start();
    let unavailable = server.mock(|when, then| {
        when.method(POST).path("/v1/traces");
        then.status(503);
    });
    let pipeline = ObservabilityPipeline::build(
        OtlpConfig::default()
            .with_traces(Some(SignalEndpoint::new(server.url("/v1/traces"))))
            .with_retry(RetryLimits {
                max_retries: 2,
                initial_delay: Duration::from_millis(10),
                max_delay: Duration::from_millis(20),
            }),
    )
    .unwrap();
    chat(&pipeline.observer(), StartOptions::default()).end();
    let _ = pipeline.force_flush(Duration::from_secs(5));
    pipeline.shutdown(Duration::from_secs(5)).ok();
    unavailable.assert_hits(3);
    assert_eq!(pipeline.stats().export_failures, 1);
}

#[test]
fn concurrent_operations_keep_separate_hierarchies() {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let observer = Observer::builder().tracer_provider(&provider).build();

    let handles: Vec<_> = (0..8)
        .map(|n| {
            let observer = observer.clone();
            thread::spawn(move || {
                let workflow = observer.start_workflow(format!("wf{n}"), StartOptions::default());
                let parent = workflow.context().unwrap();
                // The child runs on another thread with an explicit parent,
                // like `spawn_blocking` or a native callback thread.
                let child_observer = observer.clone();
                let tool_parent = parent.clone();
                thread::spawn(move || {
                    let tool = child_observer.start_tool(
                        ToolCallInfo::new(format!("tool{n}")),
                        StartOptions::child_of(tool_parent),
                    );
                    tool.end();
                })
                .join()
                .unwrap();
                let inference = chat(&observer, StartOptions::child_of(parent.clone()));
                inference.end();
                workflow.end();
                parent
            })
        })
        .collect();
    let parents: Vec<ObservationContext> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 24);
    for parent in &parents {
        let children: Vec<_> = spans
            .iter()
            .filter(|s| s.parent_span_id.to_string() == parent.span_id())
            .collect();
        assert_eq!(
            children.len(),
            2,
            "each workflow owns exactly its tool and inference"
        );
        assert!(children
            .iter()
            .all(|s| s.span_context.trace_id().to_string() == parent.trace_id()));
    }
}

#[test]
fn traceparent_crosses_process_boundaries() {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let observer = Observer::builder().tracer_provider(&provider).build();
    let op = chat(&observer, StartOptions::default());
    let header = op.context().unwrap().to_traceparent();

    // "Another process" receives only the header string.
    let remote = ObservationContext::from_traceparent(&header, None).unwrap();
    let child = observer.start_tool(ToolCallInfo::new("remote"), StartOptions::child_of(remote));
    let child_ref = child.observation_ref().unwrap();
    child.end();
    op.end();

    let spans = exporter.get_finished_spans().unwrap();
    let child_span = spans
        .iter()
        .find(|s| s.span_context.span_id().to_string() == child_ref.span_id)
        .unwrap();
    assert_eq!(
        child_span.parent_span_id.to_string(),
        op.observation_ref().unwrap().span_id
    );
}

#[test]
fn disabled_observer_is_inert() {
    let observer = Observer::disabled();
    let op = chat(&observer, StartOptions::default());
    op.record_chunk();
    assert!(op.observation_ref().is_none());
    op.end();
    let shared = Arc::new(observer);
    assert!(!shared.is_enabled());
}
