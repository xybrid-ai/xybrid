# xybrid-observability

Portable OpenTelemetry instrumentation for AI workloads: inference and streaming,
retries and fallbacks, workflows, tool calls, model loading, feedback and
evaluations. It follows the OpenTelemetry GenAI semantic conventions and is
independent of the model runtime.

```sh
cargo add xybrid-observability
```

The crate never installs global providers. Supply tracer, meter and logger
providers through `Observer::builder()`, or create and own an OTLP pipeline
through `xybrid_observability::otlp` (enabled by the default `otlp` feature).
An observer without configured providers records nothing. Applications own
flushing and shutdown. Disable default features for instrumentation without
the bundled HTTP exporter.

```rust
use xybrid_observability::{InferenceRequest, Observer, StartOptions, Usage};

let observer = Observer::disabled(); // Supply providers to record telemetry.
let operation = observer.start_inference(
    InferenceRequest::new("chat", "llama.cpp").with_model("my-model"),
    StartOptions::default(),
);
operation.record_usage(&Usage::tokens(Some(20), Some(42)));
operation.end();
```

Content capture defaults to span attributes when recording is enabled; a
`ContentPolicy` without an explicit mode honours
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`. Use `ContentPolicy` to select
metadata-only capture, a redaction hook, declarative `RedactionRule`s (ordered
literal replacements and exact attribute omission), and size budgets. Content
that cannot be processed safely (too deep, over budget, a failing hook) is
omitted and listed in `xybrid.genai.content.truncated_attributes`.

`examples/standalone.rs` instruments a fake streaming model end to end without
any Xybrid dependency. It prints what it recorded, or exports over OTLP/HTTP
when `OTEL_EXPORTER_OTLP_ENDPOINT` is set:

```sh
cargo run -p xybrid-observability --example standalone
```

To own a pipeline directly, build `otlp::ObservabilityPipeline` from an
`otlp::OtlpConfig`, instrument through `pipeline.observer()`, and call
`force_flush` or `shutdown` at an application boundary. The collector preset
configures traces, metrics and logs; `OtlpConfig::xybrid_traces` configures traces only.
Transport uses a private Tokio runtime, so synchronous lifecycle calls also
work inside an application's runtime. Enable export explicitly; credentials
alone never create a pipeline.

Applications that already own providers can use `otlp::exporters(&config)`
and supply the returned span, log and metric exporters to those providers.
Build `Observer::builder()` over the same providers. Each exporter belongs
to one provider and must be its final transport stage; its export timeout
must fit within the calling processor's or reader's timeout. The paired
exporters preserve catalog integer/double wire types and structured nulls.
Before the paired Rust log exporter, processors see null as
`AnyValue::Bytes(empty)` in this package's scope and must preserve it. A
different exporter encodes that carrier as empty bytes rather than null.

Each export shares one absolute deadline across split requests, retries and
response reads. Redirects are disabled. Retained operations and traversal
default to 480 KiB, queues plus in-flight data to 32 MiB per signal, and JSON
depth to 128. These budgets describe library-owned data, not process RSS.
`stats()` reports `accepted`, `exported`, `dropped`, `failed` and `pending`
for each signal. A missing acknowledgement leaves delivery unknown; retries
do not multiply item outcomes. Flush covers its acceptance barrier; shutdown
closes admission, drains, cancels remaining requests and remembers its result.
An application-owned OpenTelemetry metrics provider retains the SDK reader's
own shutdown timeout; see `otlp::OtlpMetricExporter` for that limitation.

See the [API documentation](https://docs.rs/xybrid-observability) for provider
configuration and the [shared contract](CONTRACT.md) for behavioral details.
The JavaScript package `@xybrid/observability` implements the same contract.
[Collector profiles](collector/) adapt canonical telemetry to supported backends.

The crate version follows the Cargo workspace. The branch-driven release
pipeline publishes this crate before `xybrid-core` and `xybrid-sdk`.
