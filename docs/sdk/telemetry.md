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
a packaged pipeline replaces the default session. Operations retain the session
that created them; retirement drains the old owned pipeline once its last
operation releases it.

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
preset. When enabled, observed model loads and inferences carry the run's
measured average and peak CPU utilization, minimum available memory, memory
pressure (`normal`, `warn` or `critical`), peak process RSS, thermal state and
ending battery level as `xybrid.genai.resource.*` span attributes. Values that
were not measured are absent, never zero or a default level. CPU utilization is
system-wide and its first reading covers time before the run, so it shows load
around the run rather than load the run caused. See
[resource telemetry](resource-telemetry.md).

## Sessions, explicit parents and manual operations

`ObservabilitySession::new(config)` creates an owned pipeline without changing
the process default. Pass `ObservationOptions::default().with_session(&session)`
through the existing loader, model, pipeline or ASR options. `install_default_session`
selects the default for subsequent operations. `ObservabilitySession::borrowed(observer)`
retains application-owned provider lifecycle: stats are unavailable and SDK
flush/shutdown are no-ops. Owned sessions expose stats and sync/async flush and shutdown.

[The session example](../../crates/xybrid-sdk/examples/observability_session.rs)
starts a workflow and a manual inference without loading a model. Typed
`start_workflow`, `start_tool`, `start_inference` and `start_model_load` handles
expose context/reference, recording, successful `end`, `end_with_error` and
`cancel`. The first terminal transition wins. Later mutations do nothing.
Dropping the last owner cancels unfinished work without flushing. An `enter()`
guard is thread-local: capture and pass context before a worker hop and never
hold the guard across `.await`.

Model loaders accept `.observation(options)` and expose the completed load's
reference on the returned model. Existing `RunOptions` carries per-run controls;
there are no capability-specific run entry points. Pipelines, streamed TTS and
ASR expose workflow/decoder references on their existing result shapes. See
[the execution matrix and binding examples](observability-coverage.md).

Use `.device_id("application-device-id")` to attach a caller-owned `device.id`
(1–128 characters, no controls). With borrowed providers set it on the provider's
resource instead. There is no generated device identifier. Xybrid-owned network
hops propagate the current context; direct third-party propagation requires
`.propagate_context_to("https://provider.example")?` for each exact origin.
Provider-child spans additionally require that provider's instrumentation.

`RedactionRule::replace_literal(from, to)` and `RedactionRule::omit_attribute(key)`
apply ordered string replacements and exact governed-attribute omission across
native bindings. Structured keys and value types remain intact. Rust/JS callback
redactors run first. Invalid redaction output fails closed before truncation.
Metadata-only mode still permits explicit bounded error messages through the
error redactor; use a classification alone when error text may contain secrets.

The portable defaults include a 480 KiB retained-operation/traversal budget,
32 MiB of queued plus in-flight data per signal, and a 128-level JSON guard.
These bound library-owned data, not process RSS. The existing per-attribute
content limit remains 16 KiB. Oversized singletons are dropped; metrics collections
that cannot fit fail explicitly. `stats()` exposes accepted/exported/dropped/failed/
pending per signal. Retry attempts do not multiply item counts. A lost or invalid
acknowledgement leaves delivery unknown. Flush waits through its acceptance
barrier; shutdown closes admission, drains, cancels outstanding work at its
deadline, and memoizes the result.

CLI `--trace` renders operation-scoped local diagnostics. `--trace-export PATH`
explicitly writes an artifact. Neither flag enables remote OTLP export. Local
trace artifacts can contain application data independently of the OTLP policy.
