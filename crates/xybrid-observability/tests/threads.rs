//! R-CFG-5 and R-CANCEL-8: no transport thread exists before a valid build,
//! none remains after shutdown (beyond the runtime's short teardown), even one
//! that cut exports off, nor after dropping a pipeline whose exports are
//! parked, and the lifecycle works from Tokio worker threads and through
//! the async variants.
//!
//! This binary holds a single test so that no other test's threads are
//! counted. Thread names are read from `/proc` (Linux); elsewhere the
//! lifecycle still runs but thread counts are not asserted.
#![cfg(feature = "otlp")]

mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::SdkTracerProvider;
use support::{Action, Server};
use xybrid_observability::otlp::{
    exporters, BatchLimits, EnvironmentSource, ObservabilityPipeline, OtlpConfig, Protocol,
    RetryLimits, SignalEndpoint,
};
use xybrid_observability::{
    ContentCapture, ContentPolicy, Error, InferenceRequest, Observer, StartOptions, Usage,
};

/// Threads whose (15-byte, truncated) name starts with `prefix`.
fn threads(prefix: &str) -> Option<usize> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let tasks = std::fs::read_dir("/proc/self/task").ok()?;
    Some(
        tasks
            .filter_map(|t| std::fs::read_to_string(t.ok()?.path().join("comm")).ok())
            .filter(|name| name.trim_end().starts_with(prefix))
            .count(),
    )
}

/// `xybrid-observability-transport`, as the kernel truncates it.
const TRANSPORT: &str = "xybrid-observab";
/// The SDK's `OpenTelemetry.Metrics.PeriodicReader` thread.
const READER: &str = "OpenTelemetry.M";

/// Waits up to `within` for no thread named `prefix` to remain; returns how
/// many remain (`0` when not measurable).
fn remaining(prefix: &str, within: Duration) -> usize {
    let until = Instant::now() + within;
    loop {
        let n = threads(prefix).unwrap_or(0);
        if n == 0 || Instant::now() >= until {
            return n;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Waits up to `within` for at least one thread named `prefix`; returns how
/// many exist (`None` when not measurable). A new thread sets its own name
/// once it is first scheduled, so an immediate count can miss it under load.
fn appeared(prefix: &str, within: Duration) -> Option<usize> {
    let until = Instant::now() + within;
    loop {
        let n = threads(prefix)?;
        if n > 0 || Instant::now() >= until {
            return Some(n);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn assert_none_left(context: &str) {
    // Teardown waits at most 100 ms for its threads; allow scheduling slack.
    assert_eq!(
        remaining(TRANSPORT, Duration::from_millis(500)),
        0,
        "{context}: transport threads remain"
    );
    assert_eq!(
        remaining(READER, Duration::from_millis(500)),
        0,
        "{context}: periodic reader threads remain"
    );
}

fn inference(observer: &Observer) {
    let op = observer.start_inference(
        InferenceRequest::new("chat", "fake"),
        StartOptions::default(),
    );
    op.record_usage(&Usage::tokens(Some(1), Some(2)));
    op.end();
}

fn invalid_configs() -> Vec<(OtlpConfig, &'static str)> {
    let base =
        OtlpConfig::collector("http://127.0.0.1:9").with_environment(EnvironmentSource::Disabled);
    let batch = |f: fn(&mut BatchLimits)| {
        let mut limits = BatchLimits::default();
        f(&mut limits);
        base.clone().with_batch(limits)
    };
    vec![
        (batch(|b| b.max_queue_size = 0), "max_queue_size"),
        (
            batch(|b| b.max_export_batch_size = 0),
            "max_export_batch_size",
        ),
        (
            batch(|b| b.scheduled_delay = Duration::ZERO),
            "scheduled_delay",
        ),
        (
            batch(|b| b.export_timeout = Duration::from_micros(999)),
            "export_timeout",
        ),
        (batch(|b| b.max_buffered_bytes = 0), "max_buffered_bytes"),
        (
            base.clone().with_retry(RetryLimits {
                initial_delay: Duration::from_millis(2),
                max_delay: Duration::from_millis(1),
                ..RetryLimits::default()
            }),
            "max_delay",
        ),
        (
            base.clone().with_max_request_body_bytes(0),
            "max_request_body_bytes",
        ),
        (
            base.clone().with_metric_export_interval(Duration::ZERO),
            "metric_export_interval",
        ),
        (
            base.clone().with_logs(Some(
                SignalEndpoint::new("http://127.0.0.1:9/v1/logs").with_timeout(Duration::ZERO),
            )),
            "logs timeout",
        ),
        (
            base.clone()
                .with_traces(Some(SignalEndpoint::new("v1/traces"))),
            "traces endpoint",
        ),
        (
            base.clone()
                .with_metrics(Some(SignalEndpoint::new("ftp://127.0.0.1/v1/metrics"))),
            "metrics endpoint",
        ),
        (OtlpConfig::default(), "no signal"),
    ]
}

#[test]
fn transport_threads_exist_only_between_a_valid_build_and_shutdown() {
    assert_eq!(threads(TRANSPORT).unwrap_or(0), 0, "clean start");

    // R-CFG-5: invalid limits are rejected before any thread exists.
    for (config, option) in invalid_configs() {
        let built = ObservabilityPipeline::build(config.clone());
        assert!(
            matches!(&built, Err(Error::Config(m)) if m.contains(option)),
            "{option}: {built:?}"
        );
        let standalone = exporters(&config);
        assert!(
            matches!(&standalone, Err(Error::Config(m)) if m.contains(option)),
            "{option}"
        );
        assert_eq!(
            threads(TRANSPORT).unwrap_or(0),
            0,
            "{option}: transport thread"
        );
        assert_eq!(threads(READER).unwrap_or(0), 0, "{option}: reader thread");
    }

    let server = Server::ok();
    let config = || server.all_config(Protocol::HttpProtobuf);
    let events = || ContentPolicy::default().with_capture(ContentCapture::EventOnly);

    // R-CANCEL-8: a sync shutdown leaves nothing running.
    let pipeline = ObservabilityPipeline::build(config()).unwrap();
    if let Some(n) = appeared(TRANSPORT, Duration::from_secs(5)) {
        assert!(n >= 1, "the valid build started its transport");
    }
    inference(&pipeline.observer_with(events()));
    pipeline.shutdown(Duration::from_secs(5)).unwrap();
    assert_none_left("sync shutdown");
    let exported = server.count();
    assert!(exported >= 3, "all three signals were exported: {exported}");

    // Sync lifecycle calls from a Tokio worker thread (multi-thread and
    // current-thread runtimes): no `block_on` on the caller's thread.
    for current_thread in [false, true] {
        let runtime = if current_thread {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
        } else {
            tokio::runtime::Runtime::new().unwrap()
        };
        let pipeline = Arc::new(ObservabilityPipeline::build(config()).unwrap());
        let copy = pipeline.clone();
        runtime.block_on(async move {
            tokio::spawn(async move {
                inference(&copy.observer_with(ContentPolicy::default()));
                copy.force_flush(Duration::from_secs(5)).unwrap();
                copy.shutdown(Duration::from_secs(5)).unwrap();
            })
            .await
            .unwrap();
        });
        assert!(pipeline.is_shut_down());
        assert_eq!(pipeline.stats().traces.exported, 1);
        drop(pipeline);
        drop(runtime);
        assert_none_left(if current_thread {
            "sync calls on a current-thread runtime"
        } else {
            "sync calls on a multi-thread worker"
        });
    }

    // The async variants, awaited on an application runtime.
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let pipeline = ObservabilityPipeline::build(config()).unwrap();
        inference(&pipeline.observer_with(events()));
        pipeline
            .force_flush_async(Duration::from_secs(5))
            .await
            .unwrap();
        inference(&pipeline.observer_with(events()));
        pipeline
            .shutdown_async(Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(
            pipeline
                .shutdown_async(Duration::from_secs(1))
                .await
                .map_err(|e| e.to_string()),
            Ok(())
        );
        assert_eq!(pipeline.stats().traces.exported, 2);
    });
    drop(runtime);
    assert_none_left("async variants");

    // Standalone exporters share one transport, released by the last
    // exporter's shutdown.
    let set = exporters(&config()).unwrap();
    if let Some(n) = appeared(TRANSPORT, Duration::from_secs(5)) {
        assert!(n >= 1, "the standalone exporters started their transport");
    }
    let tracer_provider = SdkTracerProvider::builder()
        .with_simple_exporter(set.traces.unwrap())
        .build();
    let logger_provider = SdkLoggerProvider::builder()
        .with_simple_exporter(set.logs.unwrap())
        .build();
    let meter_provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(set.metrics.unwrap()).build())
        .build();
    inference(
        &Observer::builder()
            .tracer_provider(&tracer_provider)
            .logger_provider(&logger_provider)
            .meter_provider(&meter_provider)
            .content_policy(events())
            .build(),
    );
    tracer_provider.shutdown().unwrap();
    logger_provider.shutdown().unwrap();
    meter_provider.shutdown().unwrap();
    assert_none_left("standalone exporters");

    // Dropped without shutdown: transport work is cancelled at once.
    let pipeline = ObservabilityPipeline::build(config()).unwrap();
    inference(&pipeline.observer_with(events()));
    let started = Instant::now();
    drop(pipeline);
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_none_left("dropped pipeline");

    // A shutdown that cut every signal's export off leaves nothing
    // running either, while the application still holds the pipeline. With
    // no thread left to send, the request count after a later emit and
    // flush needs no wait.
    let hung = Server::always(Action::Hang);
    // Only the shutdown ends a parked request.
    let endpoint = |signal: &str| {
        Some(
            hung.endpoint(signal, Protocol::HttpProtobuf)
                .with_timeout(Duration::from_secs(120)),
        )
    };
    let pipeline = Arc::new(
        ObservabilityPipeline::build(
            OtlpConfig::default()
                .with_environment(EnvironmentSource::Disabled)
                .with_traces(endpoint("traces"))
                .with_logs(endpoint("logs"))
                .with_metrics(endpoint("metrics"))
                .with_batch(BatchLimits {
                    scheduled_delay: Duration::from_secs(60),
                    export_timeout: Duration::from_secs(120),
                    ..BatchLimits::default()
                })
                .with_retry(RetryLimits {
                    max_retries: 0,
                    ..RetryLimits::default()
                })
                .with_metric_export_interval(Duration::from_secs(600)),
        )
        .unwrap(),
    );
    let observer = pipeline.observer_with(events());
    inference(&observer);
    let copy = pipeline.clone();
    let flush = std::thread::spawn(move || copy.force_flush(Duration::from_secs(60)));
    // One export per signal parks; the bound only detects one that never
    // arrives.
    hung.wait_for(3, Duration::from_secs(10));
    let result = pipeline.shutdown(Duration::from_millis(100));
    assert!(result.as_ref().is_err_and(Error::is_timeout), "{result:?}");
    assert!(flush.join().unwrap().is_err());
    // These bounds only detect a thread that never stops.
    assert_eq!(
        remaining(TRANSPORT, Duration::from_secs(10)),
        0,
        "cut-off shutdown: transport threads remain"
    );
    assert_eq!(
        remaining(READER, Duration::from_secs(10)),
        0,
        "cut-off shutdown: periodic reader threads remain"
    );
    inference(&observer);
    let _ = pipeline.force_flush(Duration::from_secs(1));
    assert_eq!(hung.count(), 3, "a request after the shutdown");
    let stats = pipeline.stats();
    for signal in [stats.traces, stats.logs, stats.metrics] {
        assert_eq!(signal.pending, 0, "{signal:?}");
        assert!(signal.is_consistent(), "{signal:?}");
    }

    // Dropped while every signal's export is parked, from a plain thread
    // and inside a Tokio runtime: the drop returns although nothing was
    // answered, and leaves no transport or reader thread behind. The bounds
    // only detect a drop that never returns or a thread that never stops.
    for in_runtime in [false, true] {
        let parked = Server::always(Action::Hang);
        // Only the drop ends a parked request.
        let endpoint = |signal: &str| {
            Some(
                parked
                    .endpoint(signal, Protocol::HttpProtobuf)
                    .with_timeout(Duration::from_secs(120)),
            )
        };
        let pipeline = ObservabilityPipeline::build(
            OtlpConfig::default()
                .with_environment(EnvironmentSource::Disabled)
                .with_traces(endpoint("traces"))
                .with_logs(endpoint("logs"))
                .with_metrics(endpoint("metrics"))
                .with_batch(BatchLimits {
                    // One item is a full batch: it is exported at once.
                    max_export_batch_size: 1,
                    scheduled_delay: Duration::from_secs(60),
                    export_timeout: Duration::from_secs(120),
                    ..BatchLimits::default()
                })
                .with_retry(RetryLimits {
                    max_retries: 0,
                    ..RetryLimits::default()
                })
                .with_metric_export_interval(Duration::from_millis(20)),
        )
        .unwrap();
        inference(&pipeline.observer_with(events()));
        for path in ["/v1/traces", "/v1/logs", "/v1/metrics"] {
            parked.wait_for_path(path, 1, Duration::from_secs(10));
        }

        let (dropped, returned) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            if in_runtime {
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap()
                    .block_on(async move { drop(pipeline) });
            } else {
                drop(pipeline);
            }
            let _ = dropped.send(());
        });
        returned
            .recv_timeout(Duration::from_secs(20))
            .unwrap_or_else(|_| panic!("in_runtime={in_runtime}: the drop waited on an export"));
        assert_eq!(
            remaining(TRANSPORT, Duration::from_secs(10)),
            0,
            "in_runtime={in_runtime}: transport threads remain after the drop"
        );
        assert_eq!(
            remaining(READER, Duration::from_secs(10)),
            0,
            "in_runtime={in_runtime}: periodic reader threads remain after the drop"
        );
    }
}
