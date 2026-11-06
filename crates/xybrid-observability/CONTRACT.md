# Observability contract (v1)

This document is the behavioral contract shared by the Rust crate
`xybrid-observability`, the JavaScript package `@xybrid/observability`, and any
backend that ingests their telemetry. The machine-readable half lives in
[`contract/catalog.json`](contract/catalog.json); executable examples live in
[`contract/fixtures/`](contract/fixtures). Both packages run every fixture and
must produce identical normalized output.

The contract follows the OpenTelemetry GenAI semantic conventions pinned at
revision `95aff9cdf605bbc7d7e8f7c4745a693bd0c4eea8` of
[semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai/tree/95aff9cdf605bbc7d7e8f7c4745a693bd0c4eea8).
Those conventions are in Development status. Moving to a newer revision is a
contract change: bump `contract_version`, update both packages and the
fixtures in one change, and call it out in the changelog. No schema URL is
advertised because the GenAI repository has not published a stable one.

## Scope and independence

The packages depend only on OpenTelemetry. They know nothing about Xybrid model
types, routing, authentication, or platform services. Xybrid integrates through
adapters that call the public API, the same way any other runtime would
(Transformers.js, Google LiteRT, Apple Foundation Models, a remote API client).

The packages never install global providers or subscribers. Applications either
pass their own tracer/meter/logger providers, or ask the package to build an
OTLP/HTTP pipeline that the application owns and shuts down.

## Namespaces

| Namespace | Owner | Notes |
|---|---|---|
| `gen_ai.*`, `error.type`, `server.*`, `exception.*` | OpenTelemetry | Used exactly as the pinned conventions define them. |
| `xybrid.genai.*` | this contract | Portable extensions for measurements the conventions do not cover yet (first token, execution location, runtime, attempts, resources, truncation). |
| `xybrid.routing.*`, `xybrid.device.*`, `xybrid.sdk.*`, `xybrid.pipeline.*` | Xybrid SDK adapter | Never emitted by the portable packages themselves. |

## Identity: provider, runtime, model

- `gen_ai.request.model` is the model the caller asked for; `gen_ai.response.model`
  is what actually answered, when known.
- `gen_ai.provider.name` is who served the request. Remote calls use the
  upstream provider's well-known value (`openai`, `anthropic`, `gcp.gemini`, …).
  Local execution uses the runtime name because there is no upstream provider.
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
| model load | `load_model` | `load_model {model}` | `INTERNAL` |
| workflow | `invoke_workflow` | `invoke_workflow {workflow.name}` | `INTERNAL` |
| tool | `execute_tool` | `execute_tool {tool.name}` | `INTERNAL` |
| attempt | — (no `gen_ai.operation.name`) | `attempt {index}` | `CLIENT` if the attempt's location is `remote`, otherwise `INTERNAL` |

When the model is unknown the span name is the operation name alone.

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

## Context propagation

- `ObservationContext` is a portable W3C trace context: 16-byte trace id, 8-byte
  span id, trace flags, optional `tracestate`. It parses and formats
  `traceparent`/`tracestate` exactly as W3C Trace Context level 1 specifies
  (version `00`, lowercase hex, all-zero ids invalid).
- Every start call takes an optional explicit parent. Without one, the ambient
  OpenTelemetry context is used (thread-local `Context::current()` in Rust,
  the active context manager in JS). An explicit parent always wins.
- An operation handle exposes `context()` for children and `observation_ref()`
  (`{trace_id, span_id}` lowercase hex) for results, feedback, and evaluations.
- Crossing threads, `spawn_blocking`, workers, FFI, or HTTP: capture
  `context()` before the hop and pass it explicitly (as an
  `ObservationContext` or a `traceparent` string). Remote calls inject
  `traceparent`/`tracestate` headers.

## Timing

All times come from the observer's clock (injectable for tests) unless the
caller passes an explicit timestamp. Every timing call (`start*`, chunk, first
token, `end*`) has an explicit-time form (`*_at(SystemTime)` in Rust, an
optional epoch-milliseconds argument in JS) so work measured elsewhere — a
browser worker, a native runtime, another process — is recorded with its real
times. Workers report wall-clock epoch times (`performance.timeOrigin +
performance.now()` in browsers), never worker-relative offsets. Durations are
seconds as doubles.

- Operation duration = end − start of the handle (equal to span duration).
- `record_chunk()` marks one decoded output chunk carrying new non-empty
  output. The first call sets `gen_ai.response.time_to_first_chunk`
  (first chunk − start) and records `gen_ai.client.inference.time_to_first_chunk`.
  Each later call records `gen_ai.client.inference.time_per_output_chunk`
  (this chunk − previous chunk). `xybrid.genai.response.chunk_count` is the
  number of chunks.
- `record_first_token()` marks when the runtime produced its first output
  token, which can differ from the first chunk (multi-token chunks, buffered
  UTF-8, audio chunks). It sets `xybrid.genai.response.time_to_first_token` and
  records `xybrid.genai.client.inference.time_to_first_token`. Only the first
  call counts.
- `xybrid.genai.response.time_per_output_token` is set only when the runtime
  reports a measured mean inter-token time.
- `gen_ai.request.stream=true` is set only for streaming requests; the chunk
  metrics are only recorded when chunks were observed.
- Measurements that were not taken are absent. Nothing defaults to zero.

## Usage

`Usage` fields are all optional: `input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cache_write_input_tokens`,
`reasoning_output_tokens`, and per-modality input/output counts for
`text`, `image`, `audio`. Present fields become the matching `gen_ai.usage.*`
span attributes. Calling `record_usage` again replaces the previous values
field by field (later non-null values win).

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

Usage metrics are not recorded for `predict`, `load_model`, workflow, tool, or
attempt spans.

## Metrics dimensions

Metric points only carry the dimensions listed under `metric_dimensions` in the
catalog and only when the value is known. Content, ids, messages, and error
messages never appear as metric dimensions. Duration metrics include
`error.type` when the operation failed. Metrics are recorded for every
operation whether or not its trace is sampled.

## Sampling

Operations accept a `trace` hint. `Default` follows the tracer's sampler (the
packages' own pipeline uses parent-based sampling with every new root
recorded). `Suppress` creates no span — the handle still measures time and
records metrics, and its `observation_ref()` is empty. Live/continuous
sessions use `Suppress` for frames they do not want traced so aggregate counts
stay exact.

## Errors and cancellation

`end_with_error({ error_type, message })`:

- sets `error.type` (low-cardinality: provider code, exception name,
  `timeout`, `cancelled`, `rate_limited`, `_OTHER`);
- sets span status `ERROR` with description = the message after redaction,
  truncated to 256 characters, or `error_type` when no message is given;
- adds `error` to finish reasons only if output was recorded (partial
  streaming) and no finish reason was recorded;
- emits `gen_ai.client.operation.exception` (severity 13) for inference
  operations with `exception.type` = error type and, when a message was
  given, `exception.message` = the same bounded message — when a logger is
  configured and the error type is not `cancelled` (a cancellation is not an
  exception).

`cancel()` is `end_with_error({ error_type: "cancelled" })`.

Dropping (Rust) or garbage-collecting (JS) an un-ended handle never ends it
implicitly in JS; in Rust `Drop` ends it as cancelled so spans are not leaked.
Ending twice is a no-op.

## Content capture

Content capture follows the GenAI configuration values, with Xybrid's chosen
default:

| Mode | Spans | Events |
|---|---|---|
| `span_only` (**default once telemetry is enabled**) | content attributes as JSON strings | none |
| `event_only` | none | `gen_ai.client.inference.operation.details` with structured content |
| `span_and_event` | yes | yes |
| `no_content` (metadata-only) | none | none |

The upstream conventions default to `NO_CONTENT`; this contract deliberately
captures text by default because the product decided so. Explicit
configuration wins; otherwise `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`
(case-insensitive, invalid values mean `no_content`) is honored; otherwise
`span_only`.

Governed attributes: `gen_ai.input.messages`, `gen_ai.output.messages`,
`gen_ai.system_instructions`, `gen_ai.tool.call.arguments`,
`gen_ai.tool.call.result`, `gen_ai.prompt.variable.<key>`, and the feedback
comment. `gen_ai.tool.definitions` is schema, not content: it is recorded in
every mode but still redacted and bounded.

### Message model

Messages follow the pinned JSON schemas. JSON key order is not significant
(byte sizes, and therefore truncation, do not depend on it; fixtures compare
parsed JSON). Parts:

| Part | JSON |
|---|---|
| text | `{"type":"text","content":…}` |
| reasoning | `{"type":"reasoning","content":…}` |
| tool call | `{"type":"tool_call","id":…,"name":…,"arguments":…}` (`id` omitted when unknown) |
| tool call response | `{"type":"tool_call_response","id":…,"response":…}` |
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

### Redaction

A redaction hook receives `(kind, value)` where `kind` is the attribute key and
`value` is the structured JSON value, and returns the value to record. It runs
before truncation and before serialization. Error messages pass through the
same hook with kind `error.message`. The fixtures use a reference redactor
that replaces every occurrence of configured substrings in every JSON string
value with `[REDACTED]`.

### Truncation

Each governed or schema content attribute has a byte budget `B`
(`max_content_attribute_bytes`, default 16384) measured on the UTF-8 JSON
serialization. Deterministic algorithm, identical in both packages:

1. Serialize. If `len ≤ B`, record as is.
2. Let `S` be every JSON string value in the document (object keys excluded).
   Let `K` = the largest UTF-8 byte length in `S`.
3. While `len > B` and `K > 32`: `K = floor(K / 2)`; rebuild the document from
   the original, replacing every string longer than `K` bytes by its longest
   prefix of at most `K` UTF-8 bytes that does not split a code point,
   followed by `…[truncated]`; reserialize.
4. If still `len > B` and the document is an array: while `len > B` and the
   array has more than one element, remove the first element (oldest
   message).
5. If still `len > B`, omit the attribute.
6. If any step after 1 applied, the attribute key is appended to
   `xybrid.genai.content.truncated_attributes`.

Prompt variables are plain strings and use the same prefix rule with
`K = B`.

## Lifecycle and delivery

- `Observer` is cheap to clone and safe to share across threads/workers.
- Operation hot paths (start, chunk, usage, end) never do network I/O, never
  block on export, and never flush.
- The packaged OTLP pipeline uses bounded in-memory queues (defaults in the
  catalog), bounded batch sizes, bounded request bodies, bounded retries with
  backoff, and no disk spool. When the queue is full new spans are dropped and
  counted.
- `ExportStats` exposes counters: `spans_queued`, `spans_exported`,
  `spans_dropped`, `export_failures`.
- `force_flush(timeout)` and `shutdown(timeout)` are bounded and idempotent;
  calls after shutdown return immediately.
- Exporter diagnostics never include payloads, headers, or content.

## Feedback and evaluation

- `record_evaluation(target_ref, { name, score_value?, score_label?, explanation?, response_id? })`
  emits `gen_ai.evaluation.result` (severity 9) parented to the target span.
- `record_feedback(target_ref, { score?, label?, comment? })` emits
  `gen_ai.evaluation.result` with `gen_ai.evaluation.name=user_feedback`; the
  comment is content (`xybrid.genai.feedback.comment`) and follows the capture
  mode (`no_content` drops it).
- Both require a logger. Without one the call returns `false` and records
  nothing — trace-only destinations do not receive evaluation events; route
  them through a Collector logs pipeline.

## Export presets

- Generic OTLP/HTTP: per-signal endpoints (`/v1/traces`, `/v1/metrics`,
  `/v1/logs`), `http/protobuf` (default) or `http/json`, headers, timeouts,
  standard `OTEL_EXPORTER_OTLP_*` environment variables.
- `xybrid_traces(api_key)`: traces only, to
  `https://ingest.xybrid.dev/v1/traces` with `Authorization: Bearer <key>`.
  It never sends metrics or logs; route those through a Collector.
- Collector profiles for Langfuse and Phoenix live in
  [`collector/`](collector/) and translate at export time; instrumentation
  stays canonical.

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

The runner uses a manual clock starting at `1700000000000` ms; `at_ms` is
relative to it and advances the clock before the step runs.

Step operations (fields beyond `op`, `id`, `at_ms`):

| `op` | Fields |
|---|---|
| `start_inference` | `parent` (handle id) or `parent_traceparent`/`parent_tracestate`; `trace` (`default`/`suppress`); `request` (see below) |
| `start_workflow` | `parent…`, `name` |
| `start_tool` | `parent…`, `name`, `call_id`, `tool_type`, `description`, `arguments` |
| `start_model_load` | `parent…`, `model`, `provider`, `runtime`, `format`, `size_bytes` |
| `start_attempt` | `parent` (inference handle id, required), `index`, `reason`, `provider`, `model`, `location`, `runtime`, `server_address`, `server_port` |
| `input_messages` / `output_messages` / `system_instructions` / `tool_definitions` | `value` (contract JSON) |
| `tool_result` | `value` |
| `chunk` / `first_token` | — |
| `token_timing` | `time_per_output_token_s` |
| `usage` | `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_write_input_tokens`, `reasoning_output_tokens`, `{text,image,audio}_{input,output}_tokens` |
| `finish_reasons` | `value` (string array) |
| `response` | `response_id`, `model` |
| `resource` | `cpu_utilization`, `memory_used_bytes`, `process_rss_bytes`, `process_peak_rss_bytes`, `thermal_state`, `battery_level`, `battery_charging`, `accelerator_utilization`, `accelerator_memory_used_bytes` |
| `model_cache_hit` | `value` |
| `end` | — |
| `end_with_error` | `error_type`, `message` |
| `cancel` | — |
| `evaluation` | `target` (handle id), `name`, `score_value`, `score_label`, `explanation`, `response_id` |
| `feedback` | `target`, `score`, `label`, `comment` |

`request` fields: `operation`, `provider`, `model`, `location`, `runtime`,
`runtime_version`, `accelerator`, `task`, `stream`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, `stop_sequences`, `output_type`,
`conversation_id`, `prompt_name`, `prompt_version`, `prompt_variables`,
`input_modalities`, `output_modalities`, `server_address`, `server_port`,
`embedding_dimensions`.

Normalized expectations:

- spans, in end order: `{ "id", "name", "kind", "parent" (handle id or null),
  "trace" (0-based index of distinct trace ids in first-seen order), "status":
  {"code", "description"?}, "start_ms", "end_ms", "attributes" }`. The
  `attributes` object is the complete set; JSON-typed content attributes are
  compared after parsing. When a fixture sets `parent_traceparent`, every span
  also lists `"trace_id"` and `"tracestate"` (the W3C header value, empty
  string when none), and spans whose parent is not a fixture handle list
  `"parent_span_id"`.
- metrics: `{ "name", "unit", "points": [ { "attributes", "count", "sum" } ] }`
  for histograms, `{ "name", "unit", "points": [ { "attributes", "value" } ] }`
  for counters. Points are sorted by their canonical attribute JSON
  (keys sorted). Every metric stream produced must be listed.
- logs: `{ "event_name", "severity_number", "span" (handle id or null),
  "attributes", "body"? }` in emission order.

Doubles compare with absolute tolerance `1e-9`.
