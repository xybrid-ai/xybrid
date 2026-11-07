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

Content capture defaults to span attributes when recording is enabled. Use
`ContentPolicy` to select metadata-only capture, redaction, and size limits.
See the [API documentation](https://docs.rs/xybrid-observability) for provider
configuration and the [shared contract](CONTRACT.md) for behavioral details.
The JavaScript package `@xybrid/observability` implements the same contract.
[Collector profiles](collector/) adapt canonical telemetry to supported backends.

The crate version follows the Cargo workspace. The branch-driven release
pipeline publishes this crate before `xybrid-core` and `xybrid-sdk`.
