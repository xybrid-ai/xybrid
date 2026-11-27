# Observability contract (v2)

This document is the behavioral contract shared by the Rust crate
`xybrid-observability`, the JavaScript package `@xybrid/observability`, and any
backend that ingests their telemetry. The machine-readable half lives in
[`contract/catalog.json`](contract/catalog.json); executable examples live in
[`contract/fixtures/`](contract/fixtures). Both packages run every fixture and
must produce identical normalized output. Behaviour that fixtures cannot express
(wire encoding, transport, batching, cancellation, budgets) is pinned by
regression tests that both packages carry.

The contract follows the OpenTelemetry GenAI semantic conventions pinned at
revision `95aff9cdf605bbc7d7e8f7c4745a693bd0c4eea8` of
[semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai/tree/95aff9cdf605bbc7d7e8f7c4745a693bd0c4eea8).
Those conventions are in Development status. Moving to a newer revision is a
contract change (see "Versioning"): update both packages and the fixtures in
one change, and call it out in the changelog. No schema URL is advertised
because the GenAI repository has not published a stable one.

MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

### Versioning

`contract_version` (catalog and both packages) names the contract a release
ships. Once a release has shipped a version, any change to the catalog or to
behaviour described here bumps it. Until then the version stays open: changes
land in it and are listed under its "Changes from" section. No release has
shipped the contract yet, so v2 is still open.

### Changes from v1

v2 keeps every v1 name, unit, bucket and default. It adds or changes:
references are `{trace_id, span_id}` and exist for unsampled spans; W3C context
is normalized and invalid explicit parents are rejected; clocks are anchored per
operation and derived times are clamped at zero; chunks never set first-token
time and streaming metrics are recorded when observed; custom inference
operation names use inference metrics; model loads fall back to the runtime as
provider; content goes through preflight, the callback redactor, declarative
rules and the unchanged truncation, and fails closed; structured null is
preserved; catalog int/double types are enforced on the wire; exporters split
batches by encoded size under one absolute deadline, never follow redirects and
isolate the Xybrid preset's headers; delivery is bounded by item and byte
budgets with per-signal counters; flush and shutdown have defined barriers and
results; resource samples gained peak CPU utilization, minimum available memory
and a closed memory-pressure vocabulary (see "Resource samples"); the fixture
format gained steps and stricter normalization. Content destined for a span
that does not record is not processed; a redaction hook that panics (Rust)
fails closed like one that throws, and a span processor that panics while a
span ends is contained; the content budgets are clamped so one span's
content fits the request bound as its exporter writes it (default
retained-operation budget 416 KiB over OTLP/protobuf, 208 KiB over OTLP/JSON);
per-signal stats report the bytes held against the byte budget
(`buffered_bytes`).

## Scope and independence

The packages depend only on OpenTelemetry. They know nothing about Xybrid model
types, routing, authentication, or platform services. Xybrid integrates through
adapters that call the public API, the same way any other runtime would
(Transformers.js, Google LiteRT, Apple Foundation Models, a remote API client).

The packages never install global providers or subscribers. Applications either
pass their own tracer/meter/logger providers (optionally with the packages'
exporters, see "Paired observer and exporter adapters"), or ask the package to
build an OTLP/HTTP pipeline that the application owns and shuts down. Export is
enabled only by explicit setup; an API key or an environment variable alone
enables nothing.

## Namespaces

| Namespace | Owner | Notes |
|---|---|---|
| `gen_ai.*`, `error.type`, `server.*`, `exception.*` | OpenTelemetry | Used exactly as the pinned conventions define them. |
| `xybrid.genai.*` | this contract | Portable extensions for measurements the conventions do not cover yet (first token, execution location, runtime, attempts, resources, truncation). |
| `xybrid.routing.*`, `xybrid.device.*`, `xybrid.sdk.*`, `xybrid.pipeline.*` | Xybrid SDK adapter | Never emitted by the portable packages themselves. |

## Identity: provider, runtime, model

- `gen_ai.request.model` is the model the caller asked for; `gen_ai.response.model`
  is what actually answered, when known (the last recorded value wins).
- `gen_ai.provider.name` is who served the request. Remote calls use the
  upstream provider's well-known value (`openai`, `anthropic`, `gcp.gemini`, …).
  Local execution uses the runtime name because there is no upstream provider.
  A model load without a provider uses its runtime; without either, the
  attribute is absent (never `unknown`).
- `xybrid.genai.runtime.name` (and `.version`) always names the execution
  runtime: `llama.cpp`, `whisper.cpp`, `onnxruntime`, `candle`, `mistral.rs`,
  `transformers.js`, `litert`, `apple.foundation_models`, or an HTTP client for
  remote calls (`http`).
- `xybrid.genai.execution.location` is `local` or `remote`.
- `xybrid.genai.accelerator` is the compute device when known (`cpu`, `metal`,
  `cuda`, `webgpu`, `wasm`, `coreml`, `nnapi`, `npu`).

## Operations and span shapes

| API | `gen_ai.operation.name` | Span name | Span kind |
|---|---|---|---|
| inference | `chat`, `text_completion`, `generate_content` | `{op} {request.model}` | `CLIENT` if location is `remote`, otherwise `INTERNAL` |
| inference | `embeddings` | `embeddings {request.model}` | same rule |
| inference | `predict` (non-generative ML: classification, detection, VAD) | `predict {request.model}` | same rule |
| inference | any other name (provider-specific, or a catalog name of another API) | `{op} {request.model}` | same rule |
| model load | `load_model` | `load_model {model}` | `INTERNAL` |
| workflow | `invoke_workflow` | `invoke_workflow {workflow.name}` | `INTERNAL` |
| tool | `execute_tool` | `execute_tool {tool.name}` | `INTERNAL` |
| attempt | — (no `gen_ai.operation.name`) | `attempt {index}` | `CLIENT` if the attempt's location is `remote`, otherwise `INTERNAL` |

When the subject (model, workflow, tool) is unknown or empty, the span name is
the operation name alone. Names are built literally: nothing in an operation,
model, tool or workflow name is interpreted (`{x}`, `$&`, `$1` stay as written).

`start_inference` uses the catalog entry for names in
`catalog.inference_operations` (`chat`, `text_completion`, `generate_content`,
`embeddings`, `predict`). Every other name uses
`catalog.custom_inference_operation`: the generative inference category, with
`gen_ai.client.inference.duration`, token histograms and usage counters.

`start_span(name, attributes)` creates a generic `INTERNAL` span for
runtime-internal work (pre-processing, decoding, routing decisions). It carries
only the caller's attributes, never `gen_ai.operation.name`, and records no
metrics, so GenAI-aware backends do not count it as a model call. Adapters use
it for execution stages under a logical request.

Modalities are data, not separate operation names:

- speech-to-text is `generate_content` with `xybrid.genai.task=speech_to_text`,
  input modalities `["audio"]`, output modalities `["text"]`;
- text-to-speech is `generate_content` with `gen_ai.output.type=speech`,
  `xybrid.genai.task=text_to_speech`, output modalities `["audio"]`;
- vision-language chat is `chat` with input modalities `["text","image"]`;
- classification/detection/VAD is `predict` with the matching task.

Token usage is never fabricated for operations that do not report tokens.

### Attempts (retry and fallback)

A logical request is one inference span. Each concrete execution attempt
(local try, cloud retry, fallback to another target) is a child `attempt {index}`
span carrying:

- `xybrid.genai.attempt.index` (0-based), `xybrid.genai.attempt.reason`
  (`initial`, `retry`, `fallback`);
- `gen_ai.provider.name`, `gen_ai.request.model`,
  `xybrid.genai.execution.location`, `xybrid.genai.runtime.name` of the
  attempt's target, `server.address`/`server.port` for remote attempts;
- `error.type` and `ERROR` status when the attempt failed;
- the attempt's own usage, if any, under `gen_ai.usage.*` **on the attempt span**.

Attempt spans intentionally carry no `gen_ai.operation.name`, so GenAI-aware
backends (Langfuse, Phoenix, Xybrid) count one generation per logical request.
The logical span carries the request totals (`gen_ai.usage.*` as billed for the
request) and `xybrid.genai.attempt.count`. Duration and usage metrics are
recorded once per logical request; `xybrid.genai.client.inference.attempts`
counts attempts with `xybrid.genai.attempt.reason` and `error.type` dimensions.
Attempts of an operation without a span of its own are suppressed.

## Context propagation

### ObservationContext and W3C Trace Context

`ObservationContext` is a portable W3C trace context: 16-byte trace id, 8-byte
span id, trace flags and an optional normalized `tracestate`.

`traceparent` parsing (W3C Trace Context level 1):

1. Leading and trailing spaces and tabs are trimmed.
2. Version is two lowercase hex digits; `ff` is invalid. Version `00` is exactly
   `00-<32 hex>-<16 hex>-<2 hex>` (55 characters). Versions `01`–`fe` use the
   same first 55 characters; anything after them must start with `-` and is
   ignored.
3. All hex is lowercase. Trace id and parent id are not all zeros.
4. Flags keep only the sampled bit (`flags & 0x01`); unknown bits are never
   kept, formatted or propagated. Formatting always emits version `00` and flags
   `00` or `01`.

`tracestate` normalization:

1. Split on `,`, trim spaces and tabs around members, skip empty members.
2. Each member is `key=value` split at the first `=`. Keys are simple keys
   `[a-z][a-z0-9_\-*/]{0,255}` or multi-tenant keys
   `[a-z0-9][a-z0-9_\-*/]{0,240}@[a-z][a-z0-9_\-*/]{0,13}`. Values are 1–256
   printable ASCII characters without `,` or `=` and without a trailing space.
3. An invalid member, a duplicate key, more than 32 members, or a normalized
   length above 512 characters discards the whole tracestate; the traceparent
   stays valid.
4. The normalized form joins members with `,` in input order. Contexts store,
   return and propagate only this form.

Known limitation: opentelemetry-rust 0.33 cannot hold multi-tenant keys whose
system id has 14 characters. The Rust `ObservationContext` keeps them; Rust spans
skip only that member.

### References

An operation handle exposes `context()` for children and `observation_ref()`
for results, feedback and evaluations. A reference is the JSON object
`{"trace_id": "<32 lowercase hex>", "span_id": "<16 lowercase hex>"}`; neither id
is all zeros. Inputs must use exactly these member names (no `traceId`/`spanId`),
lowercase hex and non-zero ids; extra members are ignored.

An operation *has a span of its own* when its tracer returned a valid span whose
span id differs from the parent's.

| Situation | `context()` | `observation_ref()` |
|---|---|---|
| `trace = suppress` | none | none |
| no tracer configured | none | none |
| SDK span, sampled | the span's context | the span |
| SDK span, not sampled (sampler drop, unsampled parent) | the span's context (flags `00`) | the span |
| API-only tracer (no SDK) returning the parent's context | the parent's context (propagation only) | none |

References to unsampled spans are valid targets for evaluations and feedback.
Children of a suppressed operation get no context from it; callers suppress them
explicitly.

### Explicit parents

Every start call takes an optional explicit parent. Without one, the ambient
OpenTelemetry context is used (thread-local `Context::current()` in Rust, the
active context manager in JS). An explicit parent always wins. A present but
invalid explicit parent is rejected at the API boundary, before any span, metric
or content work, and never falls back to the ambient context or becomes a root:

- Rust: `StartOptions.parent` holds a valid `ObservationContext`; invalid input
  fails where the context is built (`ObservationContext::from_traceparent`,
  `StartOptions::with_traceparent`, `Deserialize`) with
  `Error::InvalidTraceparent`. `ObservationContext` serializes as
  `{"traceparent": …, "tracestate": …}` (`tracestate` omitted when none).
- JS: `parent` accepts an `ObservationContext`, a traceparent string (with
  `tracestate`), the serialized `{traceparent, tracestate?}` form, or an
  `ObservationContext`-shaped object `{traceId, spanId, traceFlags, tracestate?,
  isRemote?}` (a structured clone, or an instance from another package copy).
  `undefined` and `null` mean no explicit parent. Anything else, or invalid
  values, throws `TypeError` from the start call.

Crossing threads, `spawn_blocking`, workers, FFI, or HTTP: capture `context()`
before the hop and pass it explicitly. Remote calls inject
`traceparent`/`tracestate` headers.

### Events of span-less operations

Exception and details events of an operation without a span of its own are
emitted with an explicitly empty trace context. They never attach to the ambient
span.

## Timing

All times come from the observer's clock (injectable for tests) unless the
caller passes an explicit timestamp. Every timing call (`start*`, chunk, first
token, `end*`) has an explicit-time form (`*_at(SystemTime)` in Rust, an
optional epoch-milliseconds argument in JS) so work measured elsewhere — a
browser worker, a native runtime, another process — is recorded with its real
times. JS ignores a non-finite explicit time. Durations are seconds as doubles.

- The default clock is wall-clock anchored per operation and monotonic within
  it: the operation captures wall and monotonic time at start and later readings
  add the monotonic elapsed time. In Rust, `InferenceOperation::now()` returns
  that reading, for a time taken before work and recorded through an
  explicit-time call once the work is done (the SDK times a streamed chunk when
  it hands it to the application's callback and records it once the callback
  accepts it), so the recorded time does not follow a wall-clock step either.
- Workers report epoch milliseconds anchored to `Date.now()` (anchor =
  `Date.now() − performance.now()` captured when the worker starts the
  operation, then `anchor + performance.now()`), never
  `performance.timeOrigin + performance.now()` alone and never worker-relative
  offsets.
- Derived values never go negative: duration, time to first chunk, inter-chunk
  time and time to first token are clamped at zero, and a span never ends before
  it starts. Metric points use the clamped values, so they are never lost.
- Operation duration = end − start of the handle (equal to span duration).
- `record_chunk()` marks one delivered output chunk carrying new non-empty
  output. The first call sets `gen_ai.response.time_to_first_chunk` and records
  `gen_ai.client.inference.time_to_first_chunk`. Each later call records
  `gen_ai.client.inference.time_per_output_chunk` (this chunk − the previous
  call's timestamp). `xybrid.genai.response.chunk_count` is the number of calls.
  A chunk never sets or influences first-token time.
- `record_first_token()` records the runtime-measured time of the first output
  token, which can differ from the first chunk (multi-token chunks, buffered
  UTF-8, audio chunks). It sets `xybrid.genai.response.time_to_first_token` and
  records `xybrid.genai.client.inference.time_to_first_token`. The first call
  wins. Callers pass only measured timestamps; because chunks never set it, a
  measured value reported after the chunks is still recorded.
- Streaming timing points are recorded when observed, with the metric
  dimensions known at that moment (see "Metric dimensions"). Streaming state per
  operation is constant-size.
- `xybrid.genai.response.time_per_output_token` is set only when the runtime
  reports a measured mean inter-token time.
- `gen_ai.request.stream=true` is set only for streaming requests; the chunk
  metrics are only recorded when chunks were observed.
- Measurements that were not taken are absent. Nothing defaults to zero.

## Usage

`Usage` fields are all optional: `input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cache_write_input_tokens`,
`reasoning_output_tokens`, and per-modality input/output counts for
`text`, `image`, `audio`. Token counts are non-negative integers; JS ignores
negative, fractional or non-numeric values. Present fields become the matching
`gen_ai.usage.*` span attributes. Calling `record_usage` again replaces the
previous values field by field (later non-null values win).

At the end of a logical inference operation:

- `gen_ai.client.inference.operation.input_tokens` / `.output_tokens`
  histograms record the totals when present (embeddings use
  `gen_ai.client.embeddings.operation.input_tokens`);
- `gen_ai.client.inference.usage.input_tokens` / `.output_tokens` counters
  record one point per known modality (`text`, `image`, `audio`) with a
  strictly positive count, plus an `unknown` point for any positive remainder
  of the total not covered by modality counts;
- `cache_read.input_tokens`, `cache_write.input_tokens`, and
  `reasoning.output_tokens` counters record their totals with
  `gen_ai.token.modality=unknown` unless a single modality was declared for
  the operation's input (cache) or output (reasoning) — in which case that
  modality is used.

Usage metrics are not recorded for `predict`, `load_model` (the model-load API),
workflow, tool, or attempt spans.

Empty request arrays (`stop_sequences`, input/output modalities) and an empty
finish-reason list are absent.

## Resource samples

`record_resource(sample)` exists on inference, attempt and model-load operations
(Rust and JS `ResourceSample`). The host measures; the packages never probe the
device. Every field is optional and becomes the span attribute below only when
present and valid:

| Field (Rust / JS) | Attribute | Type and unit | Meaning |
|---|---|---|---|
| `cpu_utilization` / `cpuUtilization` | `xybrid.genai.resource.cpu.utilization` | double, 0–1 | System CPU utilization: one reading, or the mean of the window's readings |
| `cpu_peak_utilization` / `cpuPeakUtilization` | `xybrid.genai.resource.cpu.peak_utilization` | double, 0–1 | Highest system CPU utilization reading taken in the window |
| `memory_used_bytes` / `memoryUsedBytes` | `xybrid.genai.resource.memory.used_bytes` | int, bytes | System memory in use |
| `memory_min_available_bytes` / `memoryMinAvailableBytes` | `xybrid.genai.resource.memory.min_available_bytes` | int, bytes | Lowest system memory available for allocation in the window |
| `memory_pressure` / `memoryPressure` | `xybrid.genai.resource.memory.pressure` | string, closed vocabulary | Most severe system memory pressure in the window |
| `process_rss_bytes` / `processRssBytes` | `xybrid.genai.resource.process.rss_bytes` | int, bytes | Process resident set size |
| `process_peak_rss_bytes` / `processPeakRssBytes` | `xybrid.genai.resource.process.peak_rss_bytes` | int, bytes | Highest process resident set size in the window |
| `thermal_state` / `thermalState` | `xybrid.genai.resource.thermal.state` | string | Device thermal state: `nominal`, `fair`, `serious`, `critical` |
| `battery_level` / `batteryLevel` | `xybrid.genai.resource.battery.level` | double, 0–1 | Battery charge |
| `battery_charging` / `batteryCharging` | `xybrid.genai.resource.battery.charging` | boolean | Whether the device is charging |
| `accelerator_utilization` / `acceleratorUtilization` | `xybrid.genai.resource.accelerator.utilization` | double, 0–1 | Accelerator utilization |
| `accelerator_memory_used_bytes` / `acceleratorMemoryUsedBytes` | `xybrid.genai.resource.accelerator.memory.used_bytes` | int, bytes | Accelerator memory in use |

- The *window* is the span of time the host sampled for the operation, normally
  its start to its end. Peak, minimum and pressure are extremes of the readings
  the host took in the window, not of continuous time; a host with one reading
  reports it for both the value and the extreme.
- A reading is what the host's probe returned when the host took it. A
  utilization probe usually averages over the time since its previous reading,
  so a reading taken at the start of the window can describe load from before
  the window. A host documents what its readings cover; the packages cannot
  tell.
- `memory.pressure` is one of `normal`, `warn`, `critical`
  (`catalog.enums`, least to most severe): `normal` means memory is not
  constrained, `warn` that the host classifies available memory as low, and
  `critical` a severe shortage in which allocations or the process are at risk.
  The host maps its platform signal (libdispatch's normal/warn/critical
  memory-pressure events, a threshold on available/total memory, …) and
  documents the mapping. There is no `unknown` level: a host that could not
  classify omits the field. Both packages ignore any other value (JS logs a
  debug diagnostic), so the attribute never carries a value outside the
  vocabulary. `thermal.state` is recorded as given; its catalog enum lists the
  expected values.
- Nothing is defaulted: a field that was not measured is absent, never `0`,
  `nominal` or `normal`. Zero is a measurement and is kept (for example
  `memory.min_available_bytes = 0`). Non-finite doubles are ignored; JS also
  ignores byte counts that are negative, fractional or not safe integers. 0–1
  values are neither clamped nor range-checked.
- Recording again sets the attributes of the present, valid fields and keeps
  earlier values of the others.
- Resource attributes are span attributes only. They record no metrics and are
  never metric dimensions.

## Metric dimensions

Metric points only carry the dimensions listed under `metric_dimensions` in the
catalog and only when the value is known. Dimensions come from typed operation
inputs (request, tool call, model load info), `record_response` and the end
error type; extension attributes passed at start or through `set_attribute`
never change them. Content, ids, messages, and error messages never appear as
metric dimensions.

- `gen_ai.response.model` joins the `common` dimensions once recorded.
- Duration metrics and token histograms include `error.type` when the operation
  failed; usage counters never do.
- Streaming timing metrics (`time_to_first_chunk`, `time_per_output_chunk`,
  `time_to_first_token`) carry the `common` dimensions known when each point is
  recorded and never `error.type`.
- Model loads use `metric_dimensions.model_load`.

Metrics are recorded for every operation whether or not its trace is sampled.

## Sampling

Operations accept a `trace` hint. `Default` follows the tracer's sampler (the
packages' own pipeline uses parent-based sampling with every new root
recorded, and takes no ratio or sampler option: metrics are recorded for
every operation whatever is sampled, and an application that wants a ratio
builds its own tracer provider with any sampler and hands it to the
observer). `Suppress` creates no span — the handle still measures time and
records metrics, and its `context()` and `observation_ref()` are empty.
Live/continuous sessions use `Suppress` for frames they do not want traced so
aggregate counts stay exact. Unsampled spans keep their references (see
"References").

## Errors and cancellation

`end_with_error({ error_type, message })`:

- sets `error.type` (low-cardinality: provider code, exception name,
  `timeout`, `cancelled`, `rate_limited`, `_OTHER`);
- sets span status `ERROR` with description = the message after redaction and
  rules, truncated to 256 code points, or `error_type` when no message is given
  or the message was dropped or failed redaction;
- adds `error` to finish reasons only if output was recorded (partial
  streaming) and no non-empty finish-reason list was recorded;
- emits `gen_ai.client.operation.exception` (severity 13) for inference
  operations with `exception.type` = error type and, only when a message
  survived redaction, `exception.message` = the same bounded message — when a
  logger is configured and the error type is not `cancelled` (a cancellation is
  not an exception).

`cancel()` is `end_with_error({ error_type: "cancelled" })`.

Dropping (Rust) or garbage-collecting (JS) an un-ended handle never ends it
implicitly in JS; in Rust `Drop` ends it as cancelled so spans are not leaked.
A JS handle that is never ended, cancelled or released is lost: no span, no
duration metric, no outcome. JS has no reliable finalization, so end handles
in a `finally` block (or cancel them) on every path.
Ending twice is a no-op; the first terminal call wins and later recording calls
are ignored. Recording calls never throw.

An operation's end (Rust) runs the error message through the content pipeline
and ends the span while it holds the operation's state, so a `record_chunk`
(or another recording call) racing with the end on another thread waits for it,
including for a slow redaction hook; once the end has run, that call is
ignored (chunks after the end are not counted). A caller that records chunks
and ends the operation on one thread never waits on this; JS, single-threaded,
never does.

Ending a span runs the tracer's span processors (`on_end` / `onEnd`) on the
calling thread. A processor that panics (Rust) or throws (JS) there is
contained: the operation's end, metrics and events complete, the caller
continues, and later spans still reach every processor (the Rust SDK leaves
only that span's own lock poisoned, never the provider's). What the failing
processor does with later spans is its own: a Rust processor that panics while
it holds a lock of its own may handle no later span. The SDK's
`SimpleSpanProcessor` (what `with_simple_exporter` installs) is one: it exports
under its exporter mutex, so an exporter that panics once poisons that mutex
for good and the processor drops every later span, while the other processors,
the pipeline's export included, keep working. Processors registered after the
failing one never see that span; behind
`ObservabilityPipeline::build_with_span_processor` the application's processor
runs before the pipeline's, so the span is neither exported nor counted. A
failure in `on_start` / `onStart` is not contained: it propagates from the
start call.

## Content capture

Content capture follows the GenAI configuration values, with Xybrid's chosen
default:

| Mode | Spans | Events |
|---|---|---|
| `span_only` (**default once telemetry is enabled**) | content attributes as JSON strings | none |
| `event_only` | none | inference or tool details with structured content |
| `span_and_event` | yes | yes |
| `no_content` (metadata-only) | none | none |

The upstream conventions default to `NO_CONTENT`; this contract deliberately
captures text by default because the product decided so. Resolution, once per
observer:

1. An explicit mode wins. JS accepts it trimmed and case-insensitive and throws
   `RangeError` for any other value; Rust uses the typed enum. A Rust
   `ContentPolicy` without an explicit mode (including
   `ContentPolicy::default().with_redactor(..)`) does not count as explicit.
2. Otherwise `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` (trimmed,
   case-insensitive; an unknown value means `no_content`).
3. Otherwise the legacy `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
   (the same four modes; `true` means `span_only`, `false` and unknown values
   mean `no_content`).
4. Otherwise `span_only`.

An empty variable is unset. A variable that cannot be read (non-UTF-8, a
throwing environment lookup) means `no_content`.

Governed attributes: `gen_ai.input.messages`, `gen_ai.output.messages`,
`gen_ai.system_instructions`, `gen_ai.tool.call.arguments`,
`gen_ai.tool.call.result`, `gen_ai.prompt.variable.<key>`, and the feedback
comment. `gen_ai.tool.definitions` is schema, not content: it is recorded in
every mode but still redacted and bounded.

Tool arguments and results use `gen_ai.execute_tool.operation.details` at completion
when event capture is enabled and a governed payload remains after redaction and
bounding. The DEBUG event is linked to the tool span and includes the tool name,
type, error type and truncation markers. Inference uses
`gen_ai.client.inference.operation.details`. Both modes share the same redaction
and byte budget; metadata-only capture emits neither payload. Content that no
enabled signal receives is not processed and is never marked truncated.

### Message model

Messages follow the pinned JSON schemas. JSON key order is not significant
(byte sizes, and therefore truncation, do not depend on it; fixtures compare
parsed JSON). Parts:

| Part | JSON |
|---|---|
| text | `{"type":"text","content":…}` |
| reasoning | `{"type":"reasoning","content":…}` |
| tool call | `{"type":"tool_call","id":…,"name":…,"arguments":…}` (`id` omitted when unknown; `arguments` omitted when absent or `null`) |
| tool call response | `{"type":"tool_call_response","id":…,"response":…}` (`response` always present, `null` when absent) |
| uri reference | `{"type":"uri","modality":…,"mime_type":…,"uri":…}` (`mime_type` omitted when unknown) |
| file reference | `{"type":"file","modality":…,"mime_type":…,"file_id":…}` |
| media metadata | `{"type":"media","modality":…,"mime_type":…,"size_bytes":…,"duration_s":…,"width":…,"height":…}` (absent fields omitted) |

Input message: `{"role":…,"parts":[…]}` plus `"name"` when set. Output message:
`{"role":…,"parts":[…],"finish_reason":…}`. System instructions: an array of
parts.

Raw media bytes are never accepted or emitted. Media is described by metadata
(`media` generic part) or by caller-provided references (`uri`, `file`). Media
parts live inside messages, so `no_content` omits them with the rest of the
messages; the operation's `xybrid.genai.input.modalities` and
`xybrid.genai.output.modalities` attributes are still recorded.

JSON `null` stays `null` and empty strings stay `""`, on spans and in
structured event content.

### Content pipeline

Every governed value, schema value, prompt variable, error message and feedback
comment is processed in this order:

1. **Capture gate**: skipped entirely when no enabled signal receives it. The
   span destination counts only when the operation's span is recording: for
   an unsampled span (sampler drop, unsampled parent), a suppressed operation
   or one without a span of its own (no tracer, or an OpenTelemetry API
   without an SDK), span-destined content does no work at all (no
   omit check, preflight, clone, callback, rules or truncation) and is never
   marked truncated. The event destination does not depend on the span: with
   an enabled logger, event capture processes and emits every value whether
   or not the span records; without one (no logger provider, a traces-only
   pipeline) it counts as no destination. Output for recording spans is
   unchanged. Error messages (which also feed the exception event) are
   processed as before; a feedback comment is processed only when an enabled
   logger receives the evaluation event, and does no work otherwise.
2. **Omit rules**: an `omit_attribute` rule naming the key drops the value
   (not marked truncated).
3. **Preflight, before cloning**: a bounded walk of the caller's value computes
   the canonical JSON form's nesting depth and preflight cost (below). It fails
   when the depth exceeds 128, the cost exceeds `max_traversal_bytes`, or (JS)
   the value is not representable as JSON (cycle, `BigInt`, binary data,
   throwing getter or `toJSON`). JS otherwise follows `JSON.stringify`
   (`toJSON` is called; `undefined`, functions and symbols are omitted in objects
   and become `null` in arrays; non-finite numbers become `null`; an own
   `__proto__` key stays a key).
4. **Clone** into the canonical JSON value.
5. **Callback redactor** on the clone.
6. **Validation** of the callback's output: JSON data only, depth <= 128, cost <=
   `max_traversal_bytes`, a string for string kinds.
7. **Replacement rules** in order; the result's cost must stay <=
   `max_traversal_bytes`. The value fails closed when (a) its final cost
   exceeds the budget, or (b) one string alone, after any rule, would cost
   more than the budget (`16 + utf8_len`, computed from the match count before
   the string is rebuilt). The outcome depends on nothing else, in particular
   not on the order in which strings or object members are visited. An
   implementation may stop as soon as a lower bound of the final cost exceeds
   the budget, because that cannot change the outcome. Rust does. Its bound is
   the structural cost (16 per string included), plus the final length of each
   string already rewritten, plus the least length each remaining string can
   end with: its current length when no rule's `replace` is shorter than its
   `find`, else 0. Rust rebuilds a string only while that bound, with the
   string at the least length it can end with after the rule, fits the
   budget. When no rule shrinks text, the rewritten text it holds therefore
   stays within the budget; otherwise within about two budgets. Strings
   rewritten before the bound fails can still expand and shrink back, each
   intermediate bounded by (b). JS checks (a) once, after every string.
8. **Truncation** (unchanged algorithm below; error messages are bounded to 256
   code points instead).
9. **Retained-operation budget**: content retained by one operation may not
   exceed `max_retained_operation_bytes`.
10. **Record** once at end: one attribute per key, last write wins.

Budgets (catalog `defaults`; configurable, each >= 1): `max_content_attribute_bytes`
16384, `max_retained_operation_bytes` 425984, `max_traversal_bytes` 491520;
`max_json_depth` 128 (fixed). They describe library-owned data, not process
memory.

The content budgets are clamped, at configuration and never with an error, so
that one span's content cannot push its export request past
`max_request_body_bytes` (`R`, 491520 by default) as the exporter writes it
(`E`, the span content encoding: `json` by default, or `protobuf`):

```
allowance(R)                 = min(span_envelope_allowance_bytes, floor(R / 2))
ceiling(R, E)                = max(1, floor((R - allowance(R)) / span_content_encoding_growth[E]))
max_retained_operation_bytes = min(configured, ceiling(R, E))
max_content_attribute_bytes  = min(configured, max_retained_operation_bytes)
```

`span_envelope_allowance_bytes` (65536) reserves room for the span's
non-content data: ids, name, resource and scope, the package's metadata
attributes, status and the truncation list (attributes an application adds, a
large resource and many long prompt-variable names count against it). A bound
under twice the allowance reserves half of itself instead, so a small bound
still carries content. The allowance is reserved, not enforced: names in
`xybrid.genai.content.truncated_attributes` are not charged to any budget (an
omitted value's key is listed even when the retained budget is spent), so a
span whose non-content data outgrows it (for example thousands of omitted
prompt variables) can still exceed `R` and is then dropped whole as an
oversized item.

`span_content_encoding_growth` is how many encoded bytes one retained byte can
become. OTLP/JSON (`json`, 2) writes each content attribute as a JSON string:
escaping at most doubles a value that is already serialized JSON (`"` and `\`
take two bytes), and plain-text values are charged at their escaped length
(below). OTLP/protobuf (`protobuf`, 1) writes content as it is retained. So
the same request carries twice the content as protobuf: with the default `R`
the ceiling is 425984 (416 KiB), which is the default retained-operation
budget, and 212992 (208 KiB) as JSON, which clamps that default.

`E` is `json` unless it is named: the safe assumption for an exporter the
package does not know (Rust `ContentPolicy::with_span_content_encoding`, JS
`content.spanContentEncoding`). `R` is the policy's own bound (Rust
`ContentPolicy::with_max_request_body_bytes`, JS `content.maxRequestBodyBytes`;
default 491520). An observer built by a packaged pipeline with a traces signal
takes `E` from the pipeline's protocols, whatever the policy names: `protobuf`
when the traces signal, and the logs signal if there is one, both use
`http/protobuf` (the default protocol), otherwise `json`. The logs signal
counts because content sent as events rides it within the same retained
budget. The observer takes the pipeline's `max_request_body_bytes` for `R`
when it is smaller. Both packages
expose the effective values (Rust `ContentPolicy::max_attribute_bytes`,
`max_retained_operation_bytes`, `max_request_body_bytes`,
`span_content_encoding`; JS `Observer.contentBudgets`). When one policy caps
another (Rust `ContentPolicy::capped_by`), each budget of the result is the
smaller of the two effective budgets. Its `R` and `E` come from one policy as
a pair, never one from each: the smaller `R` when both have the same `E`,
otherwise the pair with the lower ceiling. So the result is never fitted
below either policy, and it fits the capped policy's own exporter. A ceiling
that names no encoding is fitted as `json`, like any policy. The clamped
attribute budget can equal the retained-operation budget; a value truncated to
near it still charges its key (and, for `span_and_event` with a recording
span, both copies) against the retained budget, so it can be omitted and
listed rather than kept.

The clamp fits spans. Event-destined content is bounded by the same
retained-operation budget, and a packaged pipeline with a traces signal counts
its logs protocol in `E` (above), so a details event fits a logs request of
the same size. It is not fitted to a logs signal's own
`max_request_body_bytes`: a packaged pipeline fits `R` to its traces bound
only, and without a traces signal it leaves the policy's own bound and
encoding. So a details event can still exceed a lower logs bound and be
dropped whole as an oversized item.

Preflight cost of the canonical JSON form (`preflight_node_bytes` = 16):

```
cost(null | boolean | number) = 16
cost(string)                  = 16 + utf8_len(string)
cost(array)                   = 16 + Σ cost(element)
cost(object)                  = 16 + Σ (16 + utf8_len(key) + cost(member value))
```

Nesting depth: a scalar has depth 0, a container 1 + the deepest child (an empty
container has depth 1).

Retained bytes per entry: span-destined `16 + utf8_len(key) +
utf8_len(recorded JSON string)` for structured values and `16 + utf8_len(key)
+ escaped_len(text)` for plain text (prompt variables), where `escaped_len` is
the UTF-8 length of the text written inside a JSON string (`"`, `\` and the
short escapes `\b \f \n \r \t` take two bytes, other control characters
six, a JS lone surrogate six; quotes excluded); event-destined `16 +
utf8_len(key) + cost(bounded value)`; both for `span_and_event` with a recording
span. Error messages and feedback comments are not retained per operation.

Fail-closed outcomes (the operation always continues; recording never throws):

| Condition | Content attribute or prompt variable | `error.message` | Feedback comment |
|---|---|---|---|
| omit rule, or the callback drops the value | omitted, not listed | description = error type, no `exception.message` | omitted |
| preflight fails (depth, traversal cost, unrepresentable value) | omitted, listed in `xybrid.genai.content.truncated_attributes` | same | omitted |
| callback fails, or its output is invalid | omitted, listed | same | omitted |
| rules push the cost past the traversal budget | omitted, listed | same | omitted |
| truncation cannot fit `B` | omitted, listed | n/a | n/a |
| retained-operation budget exceeded | omitted, listed | n/a | n/a |

Diagnostics name the attribute and reason, never content or callback errors.

### Redaction

**Callback hook.** The hook receives `(kind, value)` where `kind` is the
attribute key (`error.message` for error messages,
`xybrid.genai.feedback.comment` for feedback comments) and `value` is the
canonical clone (a string for string kinds). It returns the value to record or
"drop" (Rust `None`, JS `undefined`); JSON `null` (Rust `Some(Value::Null)`, JS
`null`) records null. For string kinds a non-string result is invalid. Rust
redactors may fail through `Redactor::try_redact` (default: infallible
`redact`); JS catches exceptions and treats a returned promise as a failure.
A Rust hook that panics (in `redact` or `try_redact`) fails closed exactly
like one that returns an error: the value is omitted and listed (an error
message falls back to the error type), the operation continues, and the panic
never reaches the caller; the process's panic hook still runs, and a binary
built with `panic = "abort"` aborts. A failure affects only that value.

**Declarative rules.** Both packages accept an ordered list of rules with this
JSON shape:

```json
{ "type": "replace_literal", "find": "<non-empty>", "replace": "<string>" }
{ "type": "omit_attribute",  "key":  "<non-empty attribute key>" }
```

`omit_attribute` drops exactly the named key (case-sensitive, no patterns)
before preflight and the callback. `replace_literal` rules run after the callback,
in order, each on the previous output: every non-overlapping occurrence of
`find` in every JSON string value (never object keys) is replaced, left to right.
Numbers, booleans, null, structure and keys are untouched. Invalid rules cannot be
configured (Rust constructors return `Err`; JS throws `TypeError`).

The fixtures use a reference callback that replaces every occurrence of
configured substrings in every JSON string value with `[REDACTED]`.

### Truncation

Each governed or schema content attribute has a byte budget `B`
(`max_content_attribute_bytes`, default 16384, minimum 1) measured on the UTF-8
JSON serialization. Deterministic algorithm, identical in both packages:

1. Serialize. If `len ≤ B`, record as is.
2. Let `S` be every JSON string value in the document (object keys excluded).
   Let `K` = the largest UTF-8 byte length in `S`.
3. While `len > B` and `K > 32`: `K = floor(K / 2)`; rebuild the document from
   the original, replacing every string longer than `K` bytes by its longest
   prefix of at most `K` UTF-8 bytes that does not split a code point,
   followed by `…[truncated]`; reserialize.
4. If still `len > B` and the document is an array: while `len > B` and the
   array has more than one element, remove the first element (oldest
   message). Implementations compute this in one pass over per-element sizes
   (`len = 2 + Σ element lengths + commas`) and serialize the kept suffix once.
5. If still `len > B`, omit the attribute.
6. If any step after 1 applied, the attribute key is appended to
   `xybrid.genai.content.truncated_attributes`.

Prompt variables are plain strings and use the same prefix rule with
`K = B`; they are processed in ascending code point order of their names.
`truncated_attributes` lists each key at most once, in first-marked order.
Re-recording a key replaces its value; a re-record that ends in omission removes
the earlier value.

## Wire types

For telemetry of scope `xybrid-observability`, the packages' exporters encode
catalog types exactly, in OTLP/protobuf and OTLP/JSON (`catalog.wire`):
`double` attributes as `doubleValue` even for whole numbers, `int` as
`intValue`, `boolean` as `boolValue`, `string` as `stringValue`, `string[]` as
an array of strings; counters per the metric `value_type` (`asInt`/`asDouble`).
This covers span, span event, log record and metric point attributes. Structured
`json` content on events is an `AnyValue` whose numbers follow the language's JSON
number (consumers treat them as JSON numbers). JSON null is an `AnyValue` with no
value set (OTLP/JSON `{}`).

The Rust OpenTelemetry API has no null `AnyValue`. Between the Rust observer and
the paired exporter, null is carried as `AnyValue::Bytes(empty)` (the package
never emits bytes otherwise); the paired exporter emits a real null. A foreign
exporter emits `bytesValue: ""`, still distinct from `stringValue: ""`.

## Paired observer and exporter adapters

Applications that own their providers pair the package's observer
(`Observer::builder()` / `new Observer({...})` over their providers) with the
package's exporters to keep wire types, null fidelity, transport bounds and
counters:

- Rust (feature `otlp`): `otlp::OtlpSpanExporter`, `OtlpLogExporter`,
  `OtlpMetricExporter` (OpenTelemetry SDK exporter traits), built with
  `otlp::exporters(&OtlpConfig)`.
- JS (`@xybrid/observability/otlp`): `OtlpTraceExporter`, `OtlpLogExporter`,
  `OtlpMetricExporter`, built with `createOtlpExporters(options)`.

Construction requirements: the exporter is the final stage of its pipeline and
belongs to one provider; options are validated at construction; its export
timeout is at most the calling processor's or reader's timeout; Rust exporters
own their transport runtime and never use the caller's; processors in front of a
Rust log exporter see null as `AnyValue::Bytes(empty)` in records of this scope
and must preserve it; JS processors see plain numbers and must not change value
types of catalog keys; only this scope's telemetry is rewritten; standalone
exporter stats count items handed to `export()`. Without the paired exporter, Rust
nulls arrive as `bytesValue: ""` and JS whole-number doubles as `intValue`.

## Lifecycle and delivery

- `Observer` is cheap to clone and safe to share across threads/workers.
- Operation hot paths (start, chunk, usage, end) never do network I/O, never
  block on export, and never flush.
- No disk spool. Delivery is bounded per signal (traces, logs, metrics):
  queued plus in-flight bytes (a conservative per-item estimate computed at
  admission) never exceed `max_buffered_bytes_per_signal` (32 MiB). For traces
  and logs, queued plus in-flight items also never exceed
  `batch.max_queue_size`. Metrics have no item queue: the periodic reader
  exports one collection at a time, admitted whole against the byte budget, so
  metric data points are bounded by that budget and by the instruments'
  cardinality, not by `max_queue_size`. Items over a limit are dropped and
  counted (a refused metrics collection counts all its data points).
- Per-signal counters (`PipelineStats { traces, logs, metrics }`, each
  `accepted`, `exported`, `dropped`, `failed`, `pending`, plus
  `buffered_bytes` / `bufferedBytes`):
  - `accepted`: every item offered (sampled spans that ended, log records
    emitted to the pipeline, metric data points per collection), including items
    refused at admission;
  - `exported`: acknowledged by the endpoint (2xx minus partial-success
    rejections);
  - `dropped`: discarded by local policy without a deciding delivery attempt
    (admission refused, offered after shutdown began, oversized single item,
    still queued when shutdown cancelled, pending or offered after a response
    hook disabled the destination — Rust only, see "Transport");
  - `failed`: export ended without acknowledgement (rejection, retries
    exhausted, deadline or cancellation in flight, transport error, lost
    acknowledgement — delivery unknown —, partial-success rejection, encoding
    failure, oversized metrics collection);
  - `pending`: accepted and not yet terminal;
  - `buffered_bytes`: the bytes the pending items hold against
    `max_buffered_bytes_per_signal`: the conservative per-item estimate
    admitted for each queued or in-flight span and log record, and the encoded
    body of a metrics collection in flight (standalone exporters: the items in
    flight). It is the running total the 32 MiB budget admits against, not
    process memory; refused items add nothing, it returns to zero whenever
    nothing is pending, and so it is zero after shutdown.
  Every snapshot satisfies `accepted = exported + dropped + failed + pending`;
  each item reaches exactly one terminal outcome; retries and splits never count
  an item twice.
- `force_flush(timeout)` covers the items pending when it is called (plus a
  metrics collection it takes). It succeeds when all of them reached a terminal
  outcome in time and none failed or was dropped. Concurrent flushes each wait for
  their own barrier. A flush timeout does not cancel exports.
- `shutdown(timeout)` closes admission, drains queued items and takes a final
  metrics collection, cancels whatever is still running at the deadline
  (in-flight items fail, queued items are dropped), settles the counters and only
  then returns. Its result is a timeout if anything was cancelled, a delivery
  failure if any item pending at the call failed or was dropped, success
  otherwise. Every later call returns the same result without new work; a flush
  after shutdown returns the shutdown result. Exporter shutdown never aborts
  in-flight work before the pipeline's shutdown deadline.
  An export that fails at a deadline capped by shutdown counts as cancelled,
  regardless of which deadline timer fires first. The Rust packaged pipeline
  waits for the metrics reader's final export, including when its shutdown
  timeout exceeds the OpenTelemetry SDK provider's own five-second wait.
- Rust: `Result<()>` with `Error::Timeout` or `Error::Delivery`, plus
  `force_flush_async`/`shutdown_async`; sync calls are safe inside another Tokio
  runtime. JS: `Promise<boolean>`, `true` only for success, resolved after
  counters settle. JS scheduling timers (batch delay, metric interval, flush,
  export and shutdown deadlines) never keep a process alive; an export in
  flight, retry backoff included, does until it completes or its deadline
  aborts it, so a process that exits without shutdown can linger up to
  `export_timeout` against an unresponsive endpoint.
- Rust: dropping an `ObservabilityPipeline` without `shutdown` abandons its
  work at once: queued spans and log records are dropped, exports in flight
  are cancelled (failed) and nothing is flushed. Its providers are shut down
  by the OpenTelemetry SDK when their last owner goes (the pipeline, the
  observers built from it, and any provider clones the application holds), and
  the SDK bounds each such drop-time shutdown by its own wait of about 5
  seconds per provider (tracer, meter, logger). Against an unresponsive
  endpoint, dropping the last owner therefore returns within roughly that bound
  per provider, never by delivering what was abandoned. Call `shutdown(timeout)`
  to bound the wait yourself and to deliver or count what is queued.
- Exporter diagnostics never include payloads, headers, response bodies or
  content.

## Transport

- Encoding: OTLP/HTTP with `http/protobuf` (default) or `http/json`; requests
  group items by resource and then scope. Transport work runs with telemetry
  suppressed so exports never trace themselves.
- Body bound: the final encoded request body may not exceed
  `max_request_body_bytes` (480 KiB). A trace or log batch over the bound is split
  into `ceil(n/2)` and the remaining items (original order), each re-encoded with
  its own resource/scope grouping, recursively; parts are sent in order. Only a
  single item that alone exceeds the bound is dropped. A metrics collection over
  the bound is not sent and fails explicitly.
- One absolute deadline per export: `start + export_timeout` (metrics:
  `min(export_timeout, metric_export_interval)`), shared by every split part,
  attempt, backoff, `Retry-After` wait and response read. Each attempt's timeout
  is `min(signal timeout, remaining)`. At the deadline in-flight requests are
  aborted, backoffs cancelled and unsent parts not sent; their items fail.
  Metrics exports run one at a time.
- Retries: 429, 502, 503, 504 and transport errors are retried up to
  `max_retries` with backoff `min(initial_delay × 2^n, max_delay)` (no jitter).
  `Retry-After` replaces the backoff, even above `max_delay`, unless it would end
  at or after the deadline, in which case the items fail immediately. Other
  statuses fail without retry. Each export retries on its own: a destination
  that keeps answering 429 or 503 does not hold later exports back
  (there is no circuit breaker and no `Retry-After` shared across exports).
  What a struggling destination can receive stays bounded by `max_retries`,
  the export deadline, one export in flight per signal and the
  `max_buffered_bytes_per_signal` budget, beyond which items are dropped
  and counted.
- Redirects are never followed; a 3xx fails without retry.
- Responses: the status decides. At most `max_response_body_bytes` (64 KiB) are
  read within the deadline; a body problem after a 2xx does not change the
  outcome. A decodable OTLP partial success counts its rejected items as failed.
- Rust uses a pipeline-owned Tokio runtime with reqwest (rustls), behind the
  `otlp` feature; caller threads never do network I/O.
- Rust response hooks (optional, per destination): a `SignalEndpoint` may
  carry a `ResponseHook`, called exactly once for every HTTP response of that
  destination (scheduled batches, flushes, retries, the shutdown drain; no call
  for a transport error or timeout). With a hook, failure bodies are read too,
  under the 64 KiB bound and for at most one second (never past the attempt's
  timeout), so a stalled failure body cannot use up the retry budget; the hook
  sees the status, headers, body and the decoded `google.rpc.Status`. It may
  disable its destination for the rest of that pipeline's or exporter set's
  life: the response itself is handled as usual (2xx acknowledges, anything
  else fails at once, with no further retry or retry wait), no further request
  is made, and items pending or offered later are dropped. Other destinations, other pipelines and local processors
  are unaffected; a pipeline built again from the same configuration starts
  enabled. The package interprets no vendor response itself; adapters attach
  hooks to the destinations they trust (`SignalEndpoint::is_xybrid_preset`).

## Configuration sources

Numeric options are validated when a pipeline or exporter is constructed, before
any thread, runtime, timer or connection exists (Rust `Error::Config`, JS
`RangeError`): queue size, batch size, scheduled delay, export timeout, signal
timeout, metric interval, body bytes and buffered bytes are integers >= 1;
retries and initial delay >= 0; `max_delay >= initial_delay`; the effective batch
size is `min(batch, queue)`; endpoints are absolute `http(s)` URLs.

Headers: generic endpoints use `header_policy = merge_environment` —
`OTEL_EXPORTER_OTLP_HEADERS`, then `OTEL_EXPORTER_OTLP_{SIGNAL}_HEADERS`, then
programmatic headers, later sources winning per key (case-insensitive), resolved
once at build. `isolated` endpoints send programmatic headers only. The final
transport adds only `Content-Type` (and a content encoding if ever used).
HTTP framing headers and a client's generic default `Accept: */*` are transport
mechanics. They do not add credentials or merge environment headers into an
isolated endpoint. A caller-supplied `Accept` remains a programmatic header.

| Variable | env constructor (`OtlpConfig::from_env`, `otlpConfigFromEnv`) | generic programmatic config | Xybrid preset |
|---|---|---|---|
| `OTEL_SDK_DISABLED=true` | no signals | ignored | ignored |
| `OTEL_EXPORTER_OTLP_[{SIGNAL}_]ENDPOINT` | creates signals (`{base}/v1/{signal}`) | ignored | ignored |
| `OTEL_EXPORTER_OTLP_[{SIGNAL}_]HEADERS` | applied | merged under programmatic | never |
| `OTEL_EXPORTER_OTLP_[{SIGNAL}_]PROTOCOL` | applied (case-insensitive) | ignored | ignored |
| `OTEL_EXPORTER_OTLP_[{SIGNAL}_]TIMEOUT` | applied | ignored | ignored |
| `OTEL_{SIGNAL}_EXPORTER` containing `none` | signal skipped | ignored | ignored |
| `OTEL_BSP_*`, `OTEL_METRIC_EXPORT_INTERVAL` | applied | ignored | ignored |
| `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES` | applied | applied under programmatic resource | applied under programmatic resource |

Empty values are unset; header and resource lists are percent-decoded
`key=value` pairs and malformed pairs are skipped. Metric temporality is
cumulative. Tests inject or disable the environment (Rust
`OtlpConfig.environment`, JS `environment` option); browsers have none.

## Feedback and evaluation

- `record_evaluation(target_ref, { name, score_value?, score_label?, explanation?, response_id? })`
  emits `gen_ai.evaluation.result` (severity 9) linked to the target's trace and
  span ids (trace flags 0). Unsampled targets are valid.
- `record_feedback(target_ref, { score?, label?, comment? })` emits
  `gen_ai.evaluation.result` with `gen_ai.evaluation.name=user_feedback`; the
  comment is content (`xybrid.genai.feedback.comment`) and follows the capture
  mode and content pipeline (`no_content` drops it).
- Both return `true` only when a record reached an enabled logger. Without a
  logger, with a disabled logger, or with an invalid reference they return
  `false` and record nothing. JS without an explicit logger provider consults the
  `@opentelemetry/api` logs global, then the `@opentelemetry/api-logs` global, at
  call time. Trace-only destinations do not receive evaluation events; route them
  through a Collector logs pipeline.
- The records carry no id of their own. An export whose acknowledgement is
  lost may be sent again, so a destination that must not store the record
  twice deduplicates on what the record is: its target, timestamp, name,
  score, response id and content. Xybrid's ingest derives its stable id from
  exactly those, so a resend replaces the first copy.

## Export presets

- Generic OTLP/HTTP: per-signal endpoints (`/v1/traces`, `/v1/metrics`,
  `/v1/logs`), `http/protobuf` (default) or `http/json`, headers, timeouts,
  standard `OTEL_*` environment variables as in "Configuration sources".
- `xybrid_traces(api_key)`: traces only, to
  `https://ingest.xybrid.dev/v1/traces`, `http/protobuf`, with exactly
  `Authorization: Bearer <key>` (`header_policy = isolated`; no environment
  header ever reaches the endpoint or replaces the key). It never sends metrics
  or logs; route those through a Collector. Rust marks the endpoint it builds
  (`SignalEndpoint::is_xybrid_preset`), so an adapter can trust Xybrid's
  workspace responses from it and from no generic endpoint.
- Collector profiles for Langfuse and Phoenix live in
  [`collector/`](collector/) and translate at export time; instrumentation
  stays canonical.

## JavaScript packaging

The core entry (`@xybrid/observability`) imports only `@opentelemetry/api`
(peer `^1.9.0`) and `@opentelemetry/api-logs` (dependency `0.223.0`). The
`/otlp` entry supports the OpenTelemetry JS SDK 3.0 and 3.1 families with their
aligned transformer: optional peers `~3.0.0 || ~3.1.0` for `core`, `resources`,
`sdk-trace`, `sdk-logs`, `sdk-metrics` and `~0.300.0 || ~0.301.0` for
`otlp-transformer`; an installation uses one family (3.0.x with 0.300.x, or
3.1.x with 0.301.x). SDK 2.x is unsupported.

## Fixture format

Each fixture in `contract/fixtures/*.json` is:

```json
{
  "name": "…",
  "description": "…",
  "config": { "content_capture": "span_only", "max_content_attribute_bytes": 16384,
              "redact_substrings": [], "logs": true },
  "steps": [ { "at_ms": 0, "op": "start_inference", "id": "a", … }, … ],
  "expect": { "spans": […], "metrics": […], "logs": […] }
}
```

Config fields (defaults in parentheses): `content_capture` (`span_only`; runners
never read the environment), `max_content_attribute_bytes` (16384),
`max_retained_operation_bytes` (425984), `max_request_body_bytes` (491520; the
content policy's own bound, which the content budgets are clamped to fit),
`span_content_encoding` (`json`; how that bound's exporter writes content),
`max_traversal_bytes` (491520),
`redact_substrings` (`[]`), `test_redactor` (absent), `redaction_rules` (`[]`),
`logs` (`true`), `tracer` (`sdk`).

- `tracer`: `sdk` (in-memory SDK with the default parent-based always-on
  sampler), `none` (no tracer provider), `noop` (OpenTelemetry API without an SDK:
  Rust `NoopTracerProvider`, JS `ProxyTracerProvider` without a delegate).
- The reference callback exists when `redact_substrings` is non-empty or
  `test_redactor` is present. `test_redactor` is
  `{ "drop_kinds": [], "fail_kinds": [], "invalid_output_kinds": [] }`; per call:
  `fail_kinds` -> the callback fails (Rust `try_redact` error, JS throws);
  `drop_kinds` -> drop; `invalid_output_kinds` -> returns `"x"` nested in 129
  arrays; otherwise substring replacement.

The runner uses a manual clock starting at `1700000000000` ms; `at_ms` is
relative to it and sets the clock before the step runs (it may move backwards).

Step operations (fields beyond `op`, `id`, `at_ms`):

| `op` | Fields |
|---|---|
| `start_inference` | `parent` (handle id, with optional `parent_form`) or `parent_traceparent`/`parent_tracestate`; `trace` (`default`/`suppress`); `time_ms`; `expect_error`; `request` (see below) |
| `start_workflow` | `parent…`, `name`, `time_ms`, `expect_error` |
| `start_tool` | `parent…`, `name`, `call_id`, `tool_type`, `description`, `arguments`, `time_ms`, `expect_error` |
| `start_model_load` | `parent…`, `model`, `provider`, `runtime`, `format`, `size_bytes`, `time_ms`, `expect_error` |
| `start_attempt` | `parent` (inference handle id, required), `index`, `reason`, `provider`, `model`, `location`, `runtime`, `server_address`, `server_port`, `time_ms` |
| `input_messages` / `output_messages` / `system_instructions` / `tool_definitions` | `value` (contract JSON) |
| `tool_result` | `value` |
| `chunk` / `first_token` | `time_ms` |
| `token_timing` | `time_per_output_token_s` |
| `usage` | `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_write_input_tokens`, `reasoning_output_tokens`, `{text,image,audio}_{input,output}_tokens` |
| `finish_reasons` | `value` (string array) |
| `response` | `response_id`, `model` |
| `resource` | `cpu_utilization`, `cpu_peak_utilization`, `memory_used_bytes`, `memory_min_available_bytes`, `memory_pressure`, `process_rss_bytes`, `process_peak_rss_bytes`, `thermal_state`, `battery_level`, `battery_charging`, `accelerator_utilization`, `accelerator_memory_used_bytes` |
| `model_cache_hit` | `value` |
| `end` / `cancel` | `time_ms` |
| `end_with_error` | `error_type`, `message`, `time_ms` |
| `evaluation` | `target` (handle id) or `target_ref` (literal reference JSON), `name`, `score_value`, `score_label`, `explanation`, `response_id`, `expect_result` (default `true`) |
| `feedback` | `target` or `target_ref`, `score`, `label`, `comment`, `expect_result` |
| `assert_ref` | `target`, `present`, optional `trace_id` |
| `assert_context` | `target`, `present`, optional `sampled`, `trace_id`, `traceparent`, `tracestate` |
| `parse_context` | `traceparent`, `tracestate`, `expect` (`valid`, `traceparent`, `tracestate`, `sampled`) |

Step field semantics:

- `time_ms`: an explicit timestamp (base + `time_ms`) passed through the
  explicit-time API.
- `parent_form` (with `parent`): `object` (default; the context value),
  `serialized` (`{traceparent, tracestate}`: Rust serde round trip, JS
  `serializeContext`), `cloned` (Rust `clone()`, JS `structuredClone`),
  `traceparent` (the string plus its tracestate).
- `expect_error: "invalid_parent"`: building the parent or starting must fail;
  no handle is registered and nothing is emitted.
- `target_ref` goes through the language's reference input path (Rust
  `serde_json::from_value::<ObservationRef>`, where an error means result
  `false`; JS the object as is). A `target` without a reference is a runner error
  unless `expect_result` is `false` (then the call is skipped).
- `assert_ref` checks presence, lowercase-hex shape, `trace_id` when given, and
  `span_id == context().span_id` when a context exists. `assert_context` checks
  `context()`. `parse_context` checks `from_traceparent`/`fromTraceparent` only.
- Value macros inside `value` and `arguments`: an object with the single key
  `$nested_array` (`{"depth": N, "leaf": X}` -> X inside N arrays) or
  `$repeat_string` (`{"value": S, "count": N}` -> S repeated N times) is expanded
  before the call.

`request` fields: `operation`, `provider`, `model`, `location`, `runtime`,
`runtime_version`, `accelerator`, `task`, `stream`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, `stop_sequences`,
`frequency_penalty`, `presence_penalty`, `output_type`, `conversation_id`,
`prompt_name`, `prompt_version`, `prompt_variables`, `input_modalities`,
`output_modalities`, `server_address`, `server_port`, `embedding_dimensions`.

Normalized expectations:

- spans, in end order: `{ "id", "name", "kind", "parent" (handle id or null),
  "trace" (0-based index of distinct trace ids in first-seen order), "status":
  {"code", "description"?}, "start_ms", "end_ms", "attributes" }`. The
  `attributes` object is the complete set; JSON-typed content attributes are
  compared after parsing. `description` is omitted when absent or empty. When a
  fixture uses `parent_traceparent`, spans whose trace id appears in a step's
  `parent_traceparent` also list `"trace_id"` and `"tracestate"` (the W3C header
  value, empty string when none), and a span whose valid parent span id is not a
  fixture handle lists `"parent_span_id"`. Spans carry no span events.
- metrics: `{ "name", "unit", "points": [ { "attributes", "count", "sum" } ] }`
  for histograms, `{ "name", "unit", "points": [ { "attributes", "value" } ] }`
  for counters. Points are sorted by their canonical attribute JSON
  (keys sorted). Every metric stream produced must be listed. Descriptors match
  the catalog.
- logs: `{ "event_name", "severity_number", "span", "attributes", "body"? }` in
  emission order; `span` is the handle id, else the raw 16-hex span id when the
  record has a trace context, else `null`. The Rust runner normalizes
  `AnyValue::Bytes(empty)` to `null`.
- Handle ids are assigned to span ids through `observation_ref()`, never through
  `context()`.

Doubles compare with absolute tolerance `1e-9`; runners do not round further. A
fixture with `"expect": null`, an unknown op or field, or a failed inline
assertion fails the run.
