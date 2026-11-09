//! Standalone use of `xybrid-observability`, with no Xybrid dependency.
//!
//! Instruments a fake streaming model: a workflow, a tool call, a model load,
//! an inference with a failed first attempt and a retry, measured first-token
//! time, streamed chunks, usage, an evaluation and end-user feedback, with a
//! declarative redaction rule.
//!
//! - With `OTEL_EXPORTER_OTLP_ENDPOINT` (or a per-signal endpoint) set, the
//!   telemetry goes to that OTLP/HTTP endpoint through an application-owned
//!   [`ObservabilityPipeline`], which is shut down at the end and its delivery
//!   counters printed.
//! - Otherwise the example uses OpenTelemetry SDK providers with in-memory
//!   exporters and prints a summary of the spans, metric streams and log
//!   events it produced.
//!
//! Either way it ends with one `observation refs: {...}` line: the
//! `{trace_id, span_id}` reference of each operation by role, as JSON, which
//! is what an application keeps to find its telemetry in a backend later.
//!
//! ```sh
//! cargo run -p xybrid-observability --example standalone
//! OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
//!     cargo run -p xybrid-observability --example standalone
//! ```

use std::time::{Duration, SystemTime};

use opentelemetry::trace::Status;
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use serde_json::json;
use xybrid_observability::otlp::{ObservabilityPipeline, OtlpConfig};
use xybrid_observability::{
    wire, AttemptInfo, AttemptReason, ContentCapture, ContentPolicy, ErrorInfo, Evaluation,
    Feedback, InferenceRequest, InputMessage, Location, ModelLoadInfo, Observer, OutputMessage,
    RedactionRule, StartOptions, ToolCallInfo, Usage,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let content = ContentPolicy::default()
        .with_capture(ContentCapture::SpanAndEvent)
        .with_rule(RedactionRule::replace_literal("sk-demo-123", "[KEY]")?);

    if let Some(config) = OtlpConfig::from_env() {
        let pipeline = ObservabilityPipeline::build(
            config.with_service("xybrid-observability-standalone", None),
        )?;
        let refs = instrument(&pipeline.observer_with(content));
        let result = pipeline.shutdown(Duration::from_secs(5));
        println!("OTLP pipeline shut down: {result:?}");
        println!("delivery stats: {:?}", pipeline.stats());
        println!("observation refs: {refs}");
        return Ok(());
    }

    let spans = InMemorySpanExporter::default();
    let metrics = InMemoryMetricExporter::default();
    let logs = InMemoryLogExporter::default();
    let tracer_provider = SdkTracerProvider::builder()
        .with_simple_exporter(spans.clone())
        .build();
    let meter_provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(metrics.clone()).build())
        .build();
    let logger_provider = SdkLoggerProvider::builder()
        .with_simple_exporter(logs.clone())
        .build();
    let observer = Observer::builder()
        .tracer_provider(&tracer_provider)
        .meter_provider(&meter_provider)
        .logger_provider(&logger_provider)
        .content_policy(content)
        .build();

    let refs = instrument(&observer);

    tracer_provider.force_flush()?;
    meter_provider.force_flush()?;
    logger_provider.force_flush()?;

    let finished = spans.get_finished_spans()?;
    println!("spans ({}):", finished.len());
    for span in &finished {
        let status = match &span.status {
            Status::Error { description } => format!(" ERROR {description}"),
            _ => String::new(),
        };
        let ms = span
            .end_time
            .duration_since(span.start_time)
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0;
        println!(
            "  {:<32} {:>7.1} ms  {:>2} attributes{status}",
            span.name.as_ref(),
            ms,
            span.attributes.len()
        );
    }

    println!("metric streams:");
    if let Some(collection) = metrics.get_finished_metrics()?.last() {
        for scope in collection.scope_metrics() {
            for metric in scope.metrics() {
                let points = match metric.data() {
                    AggregatedMetrics::F64(MetricData::Histogram(h)) => h.data_points().count(),
                    AggregatedMetrics::U64(MetricData::Histogram(h)) => h.data_points().count(),
                    AggregatedMetrics::U64(MetricData::Sum(s)) => s.data_points().count(),
                    _ => 0,
                };
                println!(
                    "  {:<56} {points} point(s) [{}]",
                    metric.name(),
                    metric.unit()
                );
            }
        }
    }

    let emitted = logs.get_emitted_logs()?;
    println!("log events ({}):", emitted.len());
    for log in &emitted {
        let attributes: serde_json::Map<String, serde_json::Value> = log
            .record
            .attributes_iter()
            .map(|(k, v)| (k.as_str().to_owned(), wire::any_value_to_json(v)))
            .collect();
        println!(
            "  {:<44} {}",
            log.record.event_name().unwrap_or("-"),
            serde_json::Value::Object(attributes)
        );
    }

    tracer_provider.shutdown()?;
    meter_provider.shutdown()?;
    logger_provider.shutdown()?;
    println!("observation refs: {refs}");
    Ok(())
}

/// A fake streaming model instrumented end to end. Returns the reference of
/// each operation by role (`null` for an operation without a span).
fn instrument(observer: &Observer) -> serde_json::Value {
    let workflow = observer.start_workflow("answer_question", StartOptions::default());
    let parent = workflow.context();

    let load = observer.start_model_load(
        ModelLoadInfo {
            model: "tiny-chat".into(),
            runtime: Some("fake-runtime".into()),
            format: Some("gguf".into()),
            size_bytes: Some(1_048_576),
            ..ModelLoadInfo::default()
        },
        StartOptions::default().with_parent(parent.clone()),
    );
    load.record_cache_hit(true);
    load.end();
    let load_ref = load.observation_ref();

    let tool = observer.start_tool(
        ToolCallInfo {
            call_id: Some("call_1".into()),
            tool_type: Some("function".into()),
            arguments: Some(json!({"city": "Paris", "unit": null})),
            ..ToolCallInfo::new("get_weather")
        },
        StartOptions::default().with_parent(parent.clone()),
    );
    tool.record_result(&json!({"temperature_c": 18, "conditions": "cloudy"}));
    tool.end();
    let tool_ref = tool.observation_ref();

    let mut request = InferenceRequest::new("chat", "fake-runtime")
        .with_model("tiny-chat")
        .with_location(Location::Local)
        .with_runtime("fake-runtime")
        .streaming(true);
    request.temperature = Some(0.0);
    request.max_tokens = Some(64);
    let inference = observer.start_inference(request, StartOptions::default().with_parent(parent));
    inference.record_input_messages(&[
        InputMessage::text("system", "You are terse."),
        InputMessage::text("user", "Weather in Paris? My key is sk-demo-123."),
    ]);

    let first = inference.start_attempt(AttemptInfo {
        index: 0,
        reason: AttemptReason::Initial,
        provider: "fake-runtime".into(),
        model: Some("tiny-chat".into()),
        location: Location::Local,
        runtime: Some("fake-runtime".into()),
        server_address: None,
        server_port: None,
    });
    first.end_with_error(&ErrorInfo::new("timeout").with_message("backend busy"));
    let first_ref = first.observation_ref();

    let retry = inference.start_attempt(AttemptInfo {
        index: 1,
        reason: AttemptReason::Retry,
        provider: "fake-runtime".into(),
        model: Some("tiny-chat".into()),
        location: Location::Local,
        runtime: Some("fake-runtime".into()),
        server_address: None,
        server_port: None,
    });
    let tokens = ["It", " is", " 18", "°C", " and", " cloudy", "."];
    let mut first_token: Option<SystemTime> = None;
    for chunk in tokens.chunks(2) {
        for _token in chunk {
            std::thread::sleep(Duration::from_millis(3));
            // The runtime measures when it decodes its first token...
            first_token.get_or_insert_with(SystemTime::now);
        }
        // ...while a chunk delivers buffered tokens later.
        inference.record_chunk();
    }
    if let Some(at) = first_token {
        inference.record_first_token_at(at);
    }
    retry.record_usage(&Usage::tokens(Some(24), Some(tokens.len() as u64)));
    retry.end();
    let retry_ref = retry.observation_ref();

    inference.record_usage(&Usage::tokens(Some(24), Some(tokens.len() as u64)));
    inference.record_response(Some("resp-1"), Some("tiny-chat-q4"));
    inference
        .record_output_messages(&[OutputMessage::assistant_text(tokens.concat(), Some("stop"))]);
    inference.end();
    let reference = inference.observation_ref();
    workflow.end();

    if let Some(target) = &reference {
        let evaluated = observer.record_evaluation(
            target,
            &Evaluation {
                name: "relevance".into(),
                score_value: Some(1.0),
                ..Evaluation::default()
            },
        );
        let rated = observer.record_feedback(
            target,
            &Feedback {
                score: Some(1.0),
                label: Some("thumbs_up".into()),
                comment: Some("Short and correct".into()),
            },
        );
        println!("evaluation recorded: {evaluated}, feedback recorded: {rated}");
    }
    json!({
        "workflow": workflow.observation_ref(),
        "load": load_ref,
        "tool": tool_ref,
        "inference": reference,
        "attempts": [first_ref, retry_ref],
    })
}
