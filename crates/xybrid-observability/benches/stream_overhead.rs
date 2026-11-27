//! Streaming inference overhead: what one instrumented streaming inference
//! costs the calling thread, per operation and per chunk, against a disabled
//! observer.
//!
//! Setups (the same calls in every one, as the SDK makes them):
//!
//! - `disabled`: [`Observer::disabled`].
//! - `memory_metadata` / `memory_content`: SDK tracer (simple processor into
//!   an in-memory exporter), meter (periodic reader, in-memory exporter) and
//!   logger providers; metadata-only vs default content capture.
//! - `otlp_blackhole`: the packaged OTLP pipeline (all three signals, default
//!   limits) to a local endpoint that acknowledges every request at once.
//! - `otlp_blocked`: the same pipeline to an endpoint that never answers
//!   (exports hang until their deadline; admission fills, then drops).
//!
//! One operation = start, system instructions (512 B), input messages
//! (1 KiB), `N` chunks with the first token measured, usage, finish reason,
//! output message (`4 × N` bytes), end. Groups:
//!
//! - `stream_op/<setup>/<N>` for `N` in 1, 32, 256 (throughput in chunks);
//! - `chunk/<setup>`: one `record_chunk` on a running operation.
//!
//! After the Criterion groups, a side report (`stream_overhead.json`, see
//! `support::report_dir`) times every `record_chunk` call and every whole
//! operation individually, for tail percentiles (the latency a token
//! callback sees), plus pipeline counters and endpoint traffic per setup.
//!
//! ```bash
//! cargo bench -p xybrid-observability --bench stream_overhead
//! cargo bench -p xybrid-observability --bench stream_overhead -- --quick   # shorter
//! ```

mod support;

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use serde_json::{json, Value};
use support::{Behavior, Endpoint};
use xybrid_observability::otlp::{EnvironmentSource, ObservabilityPipeline, OtlpConfig};
use xybrid_observability::{
    ContentCapture, ContentPolicy, InferenceRequest, InputMessage, Location, Observer,
    OutputMessage, Part, StartOptions, Usage,
};

const CHUNK_COUNTS: [usize; 3] = [1, 32, 256];
/// Average bytes per streamed token of English text.
const BYTES_PER_CHUNK: usize = 4;
/// Operations between in-memory exporter resets (outside the timed region).
const DRAIN_EVERY: u64 = 512;

/// The content one operation records.
struct Workload {
    system: Vec<Part>,
    input: Vec<InputMessage>,
    /// Output text per chunk count.
    output: Vec<(usize, String)>,
}

impl Workload {
    fn new() -> Self {
        Self {
            system: support::system(512),
            input: support::single_message(1024),
            output: CHUNK_COUNTS
                .iter()
                .map(|&n| (n, support::text(n * BYTES_PER_CHUNK)))
                .collect(),
        }
    }

    fn output(&self, chunks: usize) -> &str {
        self.output
            .iter()
            .find(|(n, _)| *n == chunks)
            .map(|(_, s)| s.as_str())
            .expect("a prepared chunk count")
    }
}

fn request() -> InferenceRequest {
    InferenceRequest::new("chat", "llama.cpp")
        .with_model("qwen3.5-0.8b")
        .with_location(Location::Local)
        .with_runtime("llama.cpp")
        .streaming(true)
}

/// One streaming inference as the SDK records it.
fn stream_op(observer: &Observer, chunks: usize, work: &Workload, output: &str) {
    let op = observer.start_inference(request(), StartOptions::default());
    op.record_system_instructions(&work.system);
    op.record_input_messages(&work.input);
    for i in 0..chunks {
        op.record_chunk();
        if i == 0 {
            op.record_first_token();
        }
    }
    op.record_usage(&Usage::tokens(Some(300), Some(chunks as u64)));
    op.record_finish_reasons(["stop"]);
    op.record_output_messages(&[OutputMessage::assistant_text(output, Some("stop"))]);
    op.end();
    black_box(&op);
}

/// One benchmarked configuration and what it owns.
struct Setup {
    name: &'static str,
    observer: Observer,
    spans: Option<InMemorySpanExporter>,
    providers: Option<(SdkTracerProvider, SdkMeterProvider, SdkLoggerProvider)>,
    pipeline: Option<ObservabilityPipeline>,
    endpoint: Option<Endpoint>,
}

impl Setup {
    fn disabled() -> Self {
        Self {
            name: "disabled",
            observer: Observer::disabled(),
            spans: None,
            providers: None,
            pipeline: None,
            endpoint: None,
        }
    }

    fn memory(name: &'static str, content: ContentPolicy) -> Self {
        let spans = InMemorySpanExporter::default();
        let tracer = SdkTracerProvider::builder()
            .with_simple_exporter(spans.clone())
            .build();
        let meter = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(InMemoryMetricExporter::default()).build())
            .build();
        let logger = SdkLoggerProvider::builder()
            .with_simple_exporter(InMemoryLogExporter::default())
            .build();
        let observer = Observer::builder()
            .tracer_provider(&tracer)
            .meter_provider(&meter)
            .logger_provider(&logger)
            .content_policy(content)
            .build();
        Self {
            name,
            observer,
            spans: Some(spans),
            providers: Some((tracer, meter, logger)),
            pipeline: None,
            endpoint: None,
        }
    }

    fn otlp(name: &'static str, behavior: Behavior) -> Self {
        let endpoint = Endpoint::start(behavior);
        let config = OtlpConfig::collector(&endpoint.url)
            .with_environment(EnvironmentSource::Disabled)
            .with_service("observability-bench", None);
        let pipeline = ObservabilityPipeline::build(config).expect("a valid pipeline");
        let observer = pipeline.observer_with(content_default());
        Self {
            name,
            observer,
            spans: None,
            providers: None,
            pipeline: Some(pipeline),
            endpoint: Some(endpoint),
        }
    }

    /// Releases what the in-memory exporter collected (never timed).
    fn drain(&self) {
        if let Some(spans) = &self.spans {
            spans.reset();
        }
    }

    fn summary(&self) -> Value {
        json!({
            "stats": self.pipeline.as_ref().map(|p| support::stats_json(&p.stats())),
            "endpoint": self.endpoint.as_ref().map(Endpoint::summary),
        })
    }

    fn shutdown(self) -> Value {
        let summary = self.summary();
        let shutdown = self
            .pipeline
            .as_ref()
            .map(|p| format!("{:?}", p.shutdown(Duration::from_millis(500))));
        let after = self
            .pipeline
            .as_ref()
            .map(|p| support::stats_json(&p.stats()));
        if let Some((tracer, meter, logger)) = &self.providers {
            let _ = tracer.shutdown();
            let _ = meter.shutdown();
            let _ = logger.shutdown();
        }
        json!({
            "before_shutdown": summary,
            "shutdown": shutdown,
            "stats_after_shutdown": after,
        })
    }
}

fn content_default() -> ContentPolicy {
    // Explicit, so an inherited OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT
    // cannot change what is measured.
    ContentPolicy::default().with_capture(ContentCapture::SpanOnly)
}

fn setups() -> Vec<Setup> {
    vec![
        Setup::disabled(),
        Setup::memory("memory_metadata", ContentPolicy::metadata_only()),
        Setup::memory("memory_content", content_default()),
        Setup::otlp("otlp_blackhole", Behavior::Blackhole),
        Setup::otlp("otlp_blocked", Behavior::Blocked),
    ]
}

fn bench_ops(c: &mut Criterion, setups: &[Setup], work: &Workload) {
    let mut group = c.benchmark_group("stream_op");
    for setup in setups {
        for &chunks in &CHUNK_COUNTS {
            let output = work.output(chunks);
            group.throughput(Throughput::Elements(chunks as u64));
            group.bench_with_input(BenchmarkId::new(setup.name, chunks), &chunks, |b, &n| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    let mut left = iters;
                    while left > 0 {
                        let batch = left.min(DRAIN_EVERY);
                        let start = Instant::now();
                        for _ in 0..batch {
                            stream_op(&setup.observer, n, work, output);
                        }
                        total += start.elapsed();
                        left -= batch;
                        setup.drain();
                    }
                    total
                });
            });
        }
    }
    group.finish();
}

fn bench_chunk(c: &mut Criterion, setups: &[Setup]) {
    let mut group = c.benchmark_group("chunk");
    group.throughput(Throughput::Elements(1));
    for setup in setups {
        group.bench_function(setup.name, |b| {
            b.iter_custom(|iters| {
                let op = setup
                    .observer
                    .start_inference(request(), StartOptions::default());
                op.record_chunk();
                let start = Instant::now();
                for _ in 0..iters {
                    op.record_chunk();
                }
                let elapsed = start.elapsed();
                op.end();
                setup.drain();
                elapsed
            });
        });
    }
    group.finish();
}

/// Individually timed calls: the distribution a token callback sees.
fn side_report(setups: &[Setup], work: &Workload, ops: usize) -> Value {
    const CHUNKS: usize = 256;
    let output = work.output(CHUNKS);
    let mut results = Vec::new();
    for setup in setups {
        // Warm up allocator and instruments.
        for _ in 0..ops.min(32) {
            stream_op(&setup.observer, CHUNKS, work, output);
        }
        setup.drain();
        let mut chunk_ns = Vec::with_capacity(ops * CHUNKS);
        let mut start_ns = Vec::with_capacity(ops);
        let mut end_ns = Vec::with_capacity(ops);
        let mut op_ns = Vec::with_capacity(ops);
        for i in 0..ops {
            let op_start = Instant::now();
            let (op, d) = support::timed(|| {
                let op = setup
                    .observer
                    .start_inference(request(), StartOptions::default());
                op.record_system_instructions(&work.system);
                op.record_input_messages(&work.input);
                op
            });
            start_ns.push(support::nanos(d));
            for c in 0..CHUNKS {
                let t = Instant::now();
                op.record_chunk();
                if c == 0 {
                    op.record_first_token();
                }
                chunk_ns.push(support::nanos(t.elapsed()));
            }
            let ((), d) = support::timed(|| {
                op.record_usage(&Usage::tokens(Some(300), Some(CHUNKS as u64)));
                op.record_finish_reasons(["stop"]);
                op.record_output_messages(&[OutputMessage::assistant_text(output, Some("stop"))]);
                op.end();
            });
            end_ns.push(support::nanos(d));
            op_ns.push(support::nanos(op_start.elapsed()));
            if (i as u64 + 1).is_multiple_of(DRAIN_EVERY) {
                setup.drain();
            }
        }
        setup.drain();
        results.push(json!({
            "setup": setup.name,
            "ops": ops,
            "chunks_per_op": CHUNKS,
            "record_chunk_ns": support::distribution(chunk_ns),
            "start_and_input_ns": support::distribution(start_ns),
            "output_and_end_ns": support::distribution(end_ns),
            "whole_op_ns": support::distribution(op_ns),
            "pipeline": setup.summary(),
            "rss_bytes": support::rss_bytes(),
        }));
    }
    json!({
        "bench": "stream_overhead",
        "environment": support::environment(),
        "workload": {
            "system_bytes": 512,
            "input_bytes": 1024,
            "output_bytes_per_chunk": BYTES_PER_CHUNK,
            "chunk_counts": CHUNK_COUNTS,
            "timer": "std::time::Instant around each call (includes timer overhead)",
        },
        "timer_overhead_ns": support::distribution(
            (0..10_000)
                .map(|_| {
                    let t = Instant::now();
                    support::nanos(black_box(t).elapsed())
                })
                .collect(),
        ),
        "setups": results,
    })
}

fn main() {
    let work = Workload::new();
    let setups = setups();
    let mut criterion = Criterion::default().configure_from_args();
    bench_ops(&mut criterion, &setups, &work);
    bench_chunk(&mut criterion, &setups);
    criterion.final_summary();
    if support::list_only() {
        return;
    }
    let ops = if support::bench_mode() && !support::quick() {
        2000
    } else {
        50
    };
    let mut report = side_report(&setups, &work, ops);
    report["shutdown"] = setups
        .into_iter()
        .map(|s| (s.name.to_owned(), s.shutdown()))
        .collect::<serde_json::Map<_, _>>()
        .into();
    support::write_report("stream_overhead", &report);
}
