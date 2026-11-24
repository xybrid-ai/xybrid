//! Bursts and saturation: `T` threads each run `K` short streaming inference
//! operations (16 chunks, representative content) as fast as they can into
//! the packaged OTLP pipeline with default limits (queue 2048 items, 32 MiB
//! per signal), mostly against an endpoint that never answers.
//!
//! What it shows:
//!
//! - the calling threads never wait on export: per-operation latency
//!   percentiles under saturation (`op_ns`) next to the unsaturated
//!   blackhole case, and the burst wall time (Criterion group `burst`);
//! - the bounds hold: accepted / dropped / pending counters, the sampled
//!   peak of library-owned bytes (`SignalStats::buffered_bytes` in
//!   `ObservabilityPipeline::stats`) against the 32 MiB budget, pending
//!   items against the 2048 queue, and no bytes held after shutdown;
//! - process RSS before, at the sampled peak and after shutdown (context
//!   only: RSS includes allocator retention and everything else in the
//!   process).
//!
//! Scenarios: `blocked_small` (8 × 1250 ops, 1 KiB in / 1 KiB out),
//! `blocked_large` (8 × 1250, 15 KiB in / 15 KiB out: the byte budget binds
//! before the queue), `blocked_contended` (64 × 250, 1 KiB / 1 KiB),
//! `blackhole_small` (8 × 1250 to an endpoint that acknowledges at once).
//! `cargo test --benches` runs a tenth of the operations once.
//!
//! ```bash
//! cargo bench -p xybrid-observability --bench saturation
//! ```
//!
//! Side report: `saturation.json` (see `support::report_dir`).

mod support;

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use serde_json::{json, Value};
use support::{Behavior, Endpoint, MaxSampler};
use xybrid_observability::otlp::{defaults, EnvironmentSource, ObservabilityPipeline, OtlpConfig};
use xybrid_observability::{
    ContentCapture, ContentPolicy, InferenceRequest, InputMessage, Location, Observer,
    OutputMessage, StartOptions, Usage,
};

const CHUNKS: usize = 16;
/// Shutdown bound after a burst: long enough to settle, short enough that a
/// blocked endpoint does not dominate the run (in-flight work is cancelled).
const SHUTDOWN: Duration = Duration::from_millis(300);

#[derive(Clone)]
struct Scenario {
    id: &'static str,
    behavior: Behavior,
    threads: usize,
    ops_per_thread: usize,
    input_bytes: usize,
    output_bytes: usize,
}

fn scenarios(scale: usize) -> Vec<Scenario> {
    let k = |n: usize| (n / scale).max(1);
    vec![
        Scenario {
            id: "blocked_small",
            behavior: Behavior::Blocked,
            threads: 8,
            ops_per_thread: k(1250),
            input_bytes: 1024,
            output_bytes: 1024,
        },
        Scenario {
            id: "blocked_large",
            behavior: Behavior::Blocked,
            threads: 8,
            ops_per_thread: k(1250),
            input_bytes: 15 * 1024,
            output_bytes: 15 * 1024,
        },
        Scenario {
            id: "blocked_contended",
            behavior: Behavior::Blocked,
            threads: 64,
            ops_per_thread: k(250),
            input_bytes: 1024,
            output_bytes: 1024,
        },
        Scenario {
            id: "blackhole_small",
            behavior: Behavior::Blackhole,
            threads: 8,
            ops_per_thread: k(1250),
            input_bytes: 1024,
            output_bytes: 1024,
        },
    ]
}

/// Content shared by every thread of a scenario.
struct Content {
    input: Vec<InputMessage>,
    output: String,
}

impl Content {
    fn new(s: &Scenario) -> Self {
        Self {
            input: support::single_message(s.input_bytes),
            output: support::text(s.output_bytes),
        }
    }
}

fn op(observer: &Observer, content: &Content) {
    let op = observer.start_inference(
        InferenceRequest::new("chat", "llama.cpp")
            .with_model("qwen3.5-0.8b")
            .with_location(Location::Local)
            .streaming(true),
        StartOptions::default(),
    );
    op.record_input_messages(&content.input);
    for i in 0..CHUNKS {
        op.record_chunk();
        if i == 0 {
            op.record_first_token();
        }
    }
    op.record_usage(&Usage::tokens(Some(300), Some(CHUNKS as u64)));
    op.record_output_messages(&[OutputMessage::assistant_text(
        content.output.as_str(),
        Some("stop"),
    )]);
    op.end();
    black_box(&op);
}

fn pipeline(endpoint: &Endpoint) -> Arc<ObservabilityPipeline> {
    let config = OtlpConfig::collector(&endpoint.url)
        .with_environment(EnvironmentSource::Disabled)
        .with_service("observability-bench", None);
    Arc::new(ObservabilityPipeline::build(config).expect("a valid pipeline"))
}

fn observer(pipeline: &ObservabilityPipeline) -> Observer {
    pipeline.observer_with(ContentPolicy::default().with_capture(ContentCapture::SpanOnly))
}

/// Runs the burst; returns its wall time and every operation's latency (ns).
fn burst(s: &Scenario, observer: &Observer, content: &Arc<Content>) -> (Duration, Vec<f64>) {
    let barrier = Arc::new(Barrier::new(s.threads + 1));
    let handles: Vec<_> = (0..s.threads)
        .map(|_| {
            let observer = observer.clone();
            let content = content.clone();
            let barrier = barrier.clone();
            let ops = s.ops_per_thread;
            thread::spawn(move || {
                let mut latencies = Vec::with_capacity(ops);
                barrier.wait();
                for _ in 0..ops {
                    let start = Instant::now();
                    op(&observer, &content);
                    latencies.push(support::nanos(start.elapsed()));
                }
                latencies
            })
        })
        .collect();
    barrier.wait();
    let start = Instant::now();
    let mut all = Vec::with_capacity(s.threads * s.ops_per_thread);
    for handle in handles {
        all.extend(handle.join().expect("a burst thread"));
    }
    (start.elapsed(), all)
}

/// One instrumented repetition of a scenario.
fn measured(s: &Scenario, content: &Arc<Content>) -> Value {
    let endpoint = Endpoint::start(s.behavior);
    let pipeline = pipeline(&endpoint);
    let observer = observer(&pipeline);
    let rss_before = support::rss_bytes();
    let bytes = {
        let p = pipeline.clone();
        MaxSampler::start(Duration::from_millis(1), move || {
            let s = p.stats();
            Some(
                s.traces
                    .buffered_bytes
                    .max(s.logs.buffered_bytes)
                    .max(s.metrics.buffered_bytes),
            )
        })
    };
    let rss = MaxSampler::start(support::rss_sample_interval(), support::rss_bytes);
    let (wall, latencies) = burst(s, &observer, content);
    let after_burst = pipeline.stats();
    let (peak_bytes, byte_samples) = bytes.finish();
    let (peak_rss, rss_samples) = rss.finish();
    let (shutdown, shutdown_time) = support::timed(|| pipeline.shutdown(SHUTDOWN));
    let after_shutdown = pipeline.stats();
    let ops = (s.threads * s.ops_per_thread) as u64;
    let peak_bytes = peak_bytes
        .unwrap_or(0)
        .max(after_burst.traces.buffered_bytes);
    let held_after_shutdown = after_shutdown.traces.buffered_bytes
        + after_shutdown.logs.buffered_bytes
        + after_shutdown.metrics.buffered_bytes;
    json!({
        "wall_ms": wall.as_secs_f64() * 1e3,
        "ops": ops,
        "ops_per_s": ops as f64 / wall.as_secs_f64(),
        "op_ns": support::distribution(latencies),
        "stats_after_burst": support::stats_json(&after_burst),
        "buffered_bytes_after_burst": {
            "traces": after_burst.traces.buffered_bytes,
            "logs": after_burst.logs.buffered_bytes,
            "metrics": after_burst.metrics.buffered_bytes,
        },
        "peak_buffered_bytes_sampled": peak_bytes,
        "buffered_byte_samples": byte_samples,
        "bounds": {
            "max_buffered_bytes_per_signal": defaults::MAX_BUFFERED_BYTES_PER_SIGNAL,
            "max_queue_size": defaults::MAX_QUEUE_SIZE,
            "bytes_within_budget": peak_bytes <= defaults::MAX_BUFFERED_BYTES_PER_SIGNAL as u64,
            "pending_within_queue": after_burst.traces.pending <= defaults::MAX_QUEUE_SIZE as u64,
            "accepted_equals_ops": after_burst.traces.accepted == ops,
            "no_bytes_after_shutdown": held_after_shutdown == 0,
        },
        "shutdown": {
            "timeout_ms": SHUTDOWN.as_millis() as u64,
            "took_ms": shutdown_time.as_secs_f64() * 1e3,
            "result": format!("{shutdown:?}"),
            "stats": support::stats_json(&after_shutdown),
        },
        "rss_bytes": {
            "before": rss_before,
            "peak_sampled": peak_rss,
            "samples": rss_samples,
            "after_shutdown": support::rss_bytes(),
            "process_peak_so_far": support::peak_rss_bytes(),
        },
        "endpoint": endpoint.summary(),
    })
}

fn bench_bursts(c: &mut Criterion, scenarios: &[Scenario]) {
    let mut group = c.benchmark_group("burst");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    for s in scenarios {
        let content = Arc::new(Content::new(s));
        group.throughput(Throughput::Elements((s.threads * s.ops_per_thread) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(s.id), s, |b, s| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let endpoint = Endpoint::start(s.behavior);
                    let pipeline = pipeline(&endpoint);
                    let observer = observer(&pipeline);
                    let (wall, _) = burst(s, &observer, &content);
                    total += wall;
                    let _ = pipeline.shutdown(SHUTDOWN);
                }
                total
            });
        });
    }
    group.finish();
}

fn main() {
    let full = support::bench_mode() && !support::list_only();
    let scale = if full { 1 } else { 10 };
    let scenarios = scenarios(scale);
    if !support::list_only() {
        // Side report first, in a fresh process, so RSS reflects the bursts.
        let repetitions = if full && !support::quick() {
            5
        } else if full {
            2
        } else {
            1
        };
        let mut results = Vec::new();
        for s in &scenarios {
            let content = Arc::new(Content::new(s));
            let runs: Vec<Value> = (0..repetitions).map(|_| measured(s, &content)).collect();
            let summary = json!({
                "wall_ms": support::distribution(runs.iter().filter_map(|r| r["wall_ms"].as_f64()).collect()),
                "op_ns_p99": support::distribution(runs.iter().filter_map(|r| r["op_ns"]["p99"].as_f64()).collect()),
                "op_ns_max": support::distribution(runs.iter().filter_map(|r| r["op_ns"]["max"].as_f64()).collect()),
                "dropped": support::distribution(runs.iter().filter_map(|r| r["stats_after_burst"]["traces"]["dropped"].as_f64()).collect()),
                "peak_buffered_bytes_sampled": support::distribution(runs.iter().filter_map(|r| r["peak_buffered_bytes_sampled"].as_f64()).collect()),
                "bounds_held": runs.iter().all(|r| {
                    r["bounds"]["bytes_within_budget"] == json!(true)
                        && r["bounds"]["pending_within_queue"] == json!(true)
                        && r["bounds"]["accepted_equals_ops"] == json!(true)
                        && r["bounds"]["no_bytes_after_shutdown"] == json!(true)
                }),
            });
            results.push(json!({
                "scenario": s.id,
                "endpoint": format!("{:?}", s.behavior),
                "threads": s.threads,
                "ops_per_thread": s.ops_per_thread,
                "chunks_per_op": CHUNKS,
                "input_bytes": s.input_bytes,
                "output_bytes": s.output_bytes,
                "repetitions": repetitions,
                "summary": summary,
                "runs": runs,
            }));
        }
        support::write_report(
            "saturation",
            &json!({
                "bench": "saturation",
                "environment": support::environment(),
                "limits": {
                    "max_queue_size": defaults::MAX_QUEUE_SIZE,
                    "max_export_batch_size": defaults::MAX_EXPORT_BATCH_SIZE,
                    "max_buffered_bytes_per_signal": defaults::MAX_BUFFERED_BYTES_PER_SIGNAL,
                    "export_timeout_ms": defaults::EXPORT_TIMEOUT.as_millis() as u64,
                    "scheduled_delay_ms": defaults::SCHEDULED_DELAY.as_millis() as u64,
                },
                "scenarios": results,
            }),
        );
    }
    let mut criterion = Criterion::default().configure_from_args();
    bench_bursts(&mut criterion, &scenarios);
    criterion.final_summary();
}
