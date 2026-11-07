# Rust SDK observability

The supported export API is `xybrid_sdk::observability`. It uses the portable
[`xybrid-observability` contract](../../crates/xybrid-observability/CONTRACT.md)
and OTLP/HTTP. The former `TelemetryConfig`, `HttpTelemetryExporter` and
process-global event-exporter context are retired.

Export is disabled until the application explicitly configures it. Setting
`XYBRID_API_KEY` or calling `set_xybrid_api_key` configures cloud authentication;
it does not enable observability. Local inference needs neither an account nor
an exporter.

## Enable export

```rust,no_run
use std::time::Duration;
use xybrid_sdk::observability::{self, ContentPolicy, ObservabilityConfig, OtlpConfig};

observability::init_observability(
    ObservabilityConfig::otlp(OtlpConfig::collector("http://localhost:4318"))
        .service("my-app", Some("1.0.0".to_owned()))
        .content(ContentPolicy::metadata_only()),
)?;

// Load and run models normally. At a useful batch boundary or before exit:
observability::flush_observability(Duration::from_secs(5))?;
observability::shutdown_observability(Duration::from_secs(5))?;
# Ok::<(), xybrid_sdk::SdkError>(())
```

`OtlpConfig::collector` configures traces, metrics and logs at the usual
`/v1/traces`, `/v1/metrics` and `/v1/logs` paths. To send traces to Xybrid instead,
initialize `ObservabilityConfig::xybrid("your-ingest-key")`. That preset sends
**traces only** to `https://ingest.xybrid.dev/v1/traces`; it does not implicitly
configure metrics or logs. `xybrid_at(full_traces_endpoint, key)` selects a
custom Xybrid trace endpoint.

Use `OtlpConfig::with_traces`, `with_metrics` and `with_logs` to select each
signal's destination, protocol, headers and timeout with `SignalEndpoint`.
`with_batch` and `with_retry` control bounded in-memory queues and retries.
Delivery is best effort; the exporter has no durable offline spool. Inspect
`observability::stats()` for packaged-pipeline export/drop counters.

`ObservabilityConfig::from_env()` explicitly opts into standard
`OTEL_EXPORTER_OTLP_*` endpoint, protocol and header configuration. It returns
`None` when no endpoint is configured or `OTEL_SDK_DISABLED=true`. Supported
protocols are `http/json` and `http/protobuf`. Merely setting endpoint variables
does not initialize the Rust SDK; the application must call
`init_observability` with the returned configuration.

## Content and privacy

Once export is enabled, the default policy captures prompt, system, output and
reasoning content on spans, bounded to 16 KiB per content attribute. Choose the
policy deliberately:

| Configuration | Behavior |
|---|---|
| `ContentPolicy::metadata_only()` | No governed input/output, tool or prompt-variable content |
| `ContentPolicy::default()` | Content on spans |
| `.with_capture(ContentCapture::EventOnly)` | Content on log events; requires logs to be configured |
| `.with_capture(ContentCapture::SpanAndEvent)` | Content on both signals |
| `.with_redactor(...)` | Rewrites or drops governed content before truncation/export |

Without an explicit policy, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` can
select `no_content`, `span_only`, `event_only` or `span_and_event`. An unknown
value selects `no_content`. An explicitly supplied policy takes precedence.
Tool definitions are schema rather than governed content: they remain recorded
in every capture mode, with redaction and truncation still applied.

For example, replace application-specific secrets while keeping useful text:

```rust
use xybrid_sdk::observability::{ContentPolicy, SubstringRedactor};

let policy = ContentPolicy::default()
    .with_redactor(SubstringRedactor::new(["private-token", "customer-email"]));
```

Automatic core-stage and output-envelope metadata pass a reviewed allowlist.
Only validated measurements and known descriptor values are promoted to stage
attributes or logical usage/runtime fields. Unknown keys, local model/encoder
paths, raw snapshots, arbitrary descriptor strings and stage errors are omitted.
Automatic SDK/CLI inference errors export a classification such as
`inference_error`, while the caller retains the original error. Model identity
is reported through the logical request rather than copied from raw stage
metadata.

Explicit public observation fields are caller-owned: custom model/provider
names, conversation IDs and custom attributes remain supported. Do not put
secrets in identifiers or attributes; the content redactor is not a general
attribute scrubber. Under the portable error contract, explicitly supplied
`ErrorInfo` messages are bounded and passed through the `error.message`
redactor even with `metadata_only`; that capture setting does not suppress
explicit exception text. Omit the message or drop it in the redactor when needed.

Detected OS, architecture, chip and RAM fields are attached to the OTLP resource
by default. `.device_attributes(false)` disables that detection; hostnames are
never attached automatically. Set service identity with `.service(...)` and
application-owned resource attributes with `OtlpConfig::with_resource_attribute`.
The HTTP receiver necessarily sees the network source address.

## Trace context and application-owned providers

The SDK observes model loads, logical inferences, concrete execution attempts,
pipelines and internal stages. `RunOptions::observation` accepts
`ObservationOptions` for an explicit W3C parent, conversation/prompt metadata,
per-run observer and trace suppression. Successful inference results expose
`observation_ref()` for linking feedback or evaluations. See the
[trace model](trace-model.md) for parentage and current result limitations.

Applications that already own OpenTelemetry providers build an `Observer` from
their tracer/meter/logger providers and install
`ObservabilityConfig::with_observer(observer)`. The observer retains its own
content policy. The application owns provider flushing and shutdown; SDK
lifecycle calls do not flush or stop application-owned providers. Reinitializing
a packaged pipeline replaces the previous one and shuts it down with a bound.

Feedback and evaluations already have public APIs: `record_feedback` and
`record_evaluation` emit linked `gen_ai.evaluation.result` log events. They return
`false` when no logs signal is configured, including with the Xybrid traces-only
preset. A trace reference alone does not configure a feedback destination.

## Local events are a separate surface

`register_telemetry_sender` subscribes to in-process `TelemetryEvent` values;
`publish_telemetry_event` publishes to those subscribers. Execution listeners
and CLI/local trace capture also remain available without remote export.
Publishing a local event does **not** create an OTLP span, and installing an
observer does not forward arbitrary local `event.data` to a remote endpoint.

Local event payloads and local stage snapshots can contain application data.
OTLP content/redaction settings do not sanitize data the application reads from
these local interfaces or chooses to forward itself. The retired JSON event
format, device-ID generation and flush-time UUID stamping are not the current
OTLP protocol.

Resource sampling is separately opt-in and defaults to `Off` for every exporter
preset. See [resource telemetry](resource-telemetry.md).
