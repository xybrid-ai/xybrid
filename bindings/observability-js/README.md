# @xybrid/observability

Portable GenAI observability on OpenTelemetry for JavaScript: inference, tool,
workflow and model-load spans, GenAI metrics, and evaluation/feedback events,
following the OpenTelemetry GenAI semantic conventions. It runs in Node,
browsers and Web Workers, and knows nothing about Xybrid models — any runtime
(Transformers.js, a remote API client, a WASM engine) can use it.

The behavior is defined by the shared contract in
[`crates/xybrid-observability/CONTRACT.md`](../../crates/xybrid-observability/CONTRACT.md);
the Rust crate `xybrid-observability` implements the same contract and both
packages run the same fixtures.

```sh
npm install @xybrid/observability @opentelemetry/api
```

## Two entry points

| Import | Depends on | Use |
|---|---|---|
| `@xybrid/observability` | `@opentelemetry/api` (peer), `@opentelemetry/api-logs` | The `Observer` and operation handles. Records through providers you own. |
| `@xybrid/observability/otlp` | the OpenTelemetry JS SDK (optional peers, see below) | An OTLP/HTTP pipeline the application creates and shuts down. |

The core never installs global providers. Without explicit providers it reads
the `@opentelemetry/api` globals (and records nothing if none are registered).

## Recording operations

```ts
import { Observer } from "@xybrid/observability";

const observer = new Observer({ tracerProvider, meterProvider, loggerProvider });

const op = observer.startInference({
  operation: "chat",
  provider: "llama.cpp",          // who served it; local runtimes use the runtime name
  model: "qwen2.5-0.5b-instruct",
  location: "local",
  runtime: "llama.cpp",
  accelerator: "webgpu",
  stream: true,
  maxTokens: 256,
});
op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: prompt }] }]);
op.recordFirstToken(engine.firstTokenAtMs);               // only a runtime-measured time
for await (const chunk of stream) op.recordChunk();      // first chunk, chunk gaps
op.recordUsage({ inputTokens: 20, outputTokens: 42 });
op.recordOutputMessages([{ role: "assistant", parts: [{ type: "text", content: text }], finish_reason: "stop" }]);
op.recordFinishReasons(["stop"]);
op.end();                                                 // or endWithError({ errorType, message }) / cancel()

const ref = op.observationRef();                          // { trace_id, span_id } for feedback
if (ref) observer.recordFeedback(ref, { score: 1, label: "thumbs_up" });
```

Also: `startWorkflow(name)`, `startTool({ name, callId, toolType, arguments })`
with `recordResult(value)`, `startModelLoad({ model, provider, runtime, format, sizeBytes })`
with `recordCacheHit(hit)`, and `op.startAttempt({ index, reason, provider, location })`
for retries and fallbacks under one logical request. Hot paths never await,
flush, or do I/O. Recording calls never throw, and neither does ending: a span
processor of yours that throws in `onEnd` is contained (processors after it do
not see that span).

End every handle on every path (`try`/`finally`, or `cancel()` when the work
is abandoned). Garbage collection never ends a handle, so one that is never
ended, cancelled or released is lost: no span, no duration metric, no outcome.

Measurements are truthful: what was not measured is absent, never zero.
`recordChunk()` marks delivered output (time to first chunk, then the gap since
the previous call); it never sets first-token time, which comes only from
`recordFirstToken()` with a runtime-measured timestamp (the first call wins).
Streaming timing points are recorded as they happen, with the dimensions known
then (`gen_ai.response.model` once `recordResponse` named it). Token and byte
counts must be non-negative integers and measurements finite; anything else is
ignored. Operation names outside `chat`, `text_completion`, `generate_content`,
`embeddings` and `predict` record generative inference metrics, with the span
name `{operation} {model}` built literally. A model load without a provider
uses its runtime as `gen_ai.provider.name`.

### References

`op.observationRef()` returns `{ trace_id, span_id }` (lowercase hex; a fresh
frozen object), the shape every Xybrid binding and the Rust crate share. A span
that the sampler did not record still has a reference, valid for evaluations
and feedback. There is none for `trace: "suppress"`, without a tracer provider,
or with an API-only tracer (no SDK), whose `context()` passes the parent
through for propagation. `toObservationRef(value)` validates a reference from
elsewhere (`traceId`/`spanId`, uppercase and all-zero ids are rejected);
`recordEvaluation`/`recordFeedback` return `false` for an invalid one.

### Time

Every timing call takes an optional epoch-millisecond time so work measured
elsewhere (a worker, a native runtime) is recorded with its real times; a
non-finite value is ignored. Without one, each operation reads a clock anchored
to `Date.now()` when it starts and advanced by `performance.now()`, so a
drifting `performance.timeOrigin` cannot move spans. Durations, time to first
chunk/token and chunk gaps are clamped at zero, and a span never ends before it
starts, so no metric point is lost.

### Context across workers and processes

An explicit `parent` always wins over the ambient OpenTelemetry context. It may
be an `ObservationContext`, a `traceparent` string (with `tracestate`), the
serialized `{ traceparent, tracestate? }` form, or the plain fields of a context
(a `structuredClone`, or an instance from another copy of the package).
`undefined` and `null` mean "no explicit parent". Anything else, or an invalid
value, throws a `TypeError` from the start call before anything is recorded; it
never silently becomes a root.

```ts
import { serializeContext, injectTraceHeaders } from "@xybrid/observability";

worker.postMessage({ observation: serializeContext(op.context()) });   // structured-clone safe
// in the worker
observer.startTool({ name: "decode" }, { parent: event.data.observation });

await fetch(url, { headers: injectTraceHeaders(op.context(), new Headers()) });
observer.startInference(request, { parent: "00-<trace>-<span>-01", tracestate: "vendor=x" });
```

Contexts follow W3C Trace Context level 1: only the sampled flag is kept, and
`tracestate` is normalized (an invalid one is dropped while the parent stays).
`toObservationContext(value)` converts any accepted object form.

For callbacks that run later on the same thread, `observer.bindContext(callback)`
captures the active application context and restores it when the callback is invoked.
It binds the invocation only; a browser's synchronous context manager still does
not propagate context into native Promise continuations.

Workers report epoch milliseconds anchored to `Date.now()`: capture
`anchor = Date.now() - performance.now()` when the worker begins the operation
and report `anchor + performance.now()`. Never send worker-relative offsets or
`performance.timeOrigin + performance.now()` alone.

## Content capture defaults, and why

| `content.capture` | Spans | Events |
|---|---|---|
| `span_only` (**default**) | messages, system instructions, tool arguments/results as JSON strings | — |
| `event_only` | — | `gen_ai.client.inference.operation.details` with structured content |
| `span_and_event` | yes | yes |
| `no_content` | — | — |

The upstream conventions default to no content; this contract deliberately
captures text by default because inspecting prompts and outputs is the point of
GenAI tracing for most users, and every destination we ship to (Xybrid,
Langfuse, Phoenix) expects it on spans. An explicit `content.capture` wins
(trimmed, case-insensitive; any other value throws `RangeError`), then
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`, then the legacy
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` (`true` = `span_only`), then
`span_only`. Unknown or unreadable environment values mean `no_content`.

Every content value goes through one pipeline, in order:

1. `omit_attribute` rules drop a key outright;
2. a bounded preflight of your value (before anything is copied) fails values
   nested deeper than 128 levels, larger than `maxTraversalBytes` (default
   480 KiB) or not representable as JSON (cycles, `BigInt`, binary data,
   throwing getters); otherwise `JSON.stringify` rules apply (`toJSON`, dropped
   `undefined`, an own `__proto__` key stays a key);
3. a canonical copy goes to your `redact(kind, value)` callback, which returns
   the value, `undefined` (drop) or `null` (JSON null); a throw, a promise or
   non-JSON output fails closed (a Rust hook that panics does the same);
4. `replace_literal` rules replace every occurrence of a literal in every string
   value (never keys), in order;
5. truncation to `maxAttributeBytes` (default 16384): the longest strings are
   halved (at code point boundaries, with `…[truncated]`), then the oldest
   messages are dropped, then the attribute is omitted;
6. content retained by one operation is bounded by `maxRetainedOperationBytes`
   (default 416 KiB) and written once, at the end (a later record of the same
   key replaces it). The budget is clamped to what one export request can
   carry: 416 KiB over OTLP/protobuf, the default protocol of
   `createOtlpPipeline`, and 208 KiB over OTLP/JSON, which can double content
   when it escapes it. An `Observer` over providers the application owns is
   fitted as JSON until `content.spanContentEncoding: "protobuf"` says
   otherwise.

Content headed for a span that is not recording (a sampler drop, an unsampled
parent, `trace: "suppress"`, no tracer) is not processed at all; event capture
(`event_only`, `span_and_event`) still processes and emits it when the observer
has a logger. Without one (`loggerProvider: null`, a traces-only pipeline, the
Xybrid preset) nothing receives the content, so it is not processed either. The budgets are
clamped so one span's content cannot push its export request past
`maxRequestBodyBytes` (default 480 KiB; `pipeline.observer()` uses the
pipeline's traces bound when it is smaller): `maxRetainedOperationBytes` is at most
`(maxRequestBodyBytes - 65536) / 2`, and `maxAttributeBytes` at most
`maxRetainedOperationBytes`. Raising a budget past that clamps it rather than
throwing; `observer.contentBudgets` reports the effective values.

```ts
new Observer({
  content: {
    rules: [
      { type: "replace_literal", find: process.env.API_KEY!, replace: "[KEY]" },
      { type: "omit_attribute", key: "gen_ai.system_instructions" },
    ],
  },
});
```

Anything that fails closed is omitted and listed in
`xybrid.genai.content.truncated_attributes`, as are truncated keys; a drop is
silent. Error messages and feedback comments use the same callback and rules
(an omitted error message falls back to the error type). JSON `null` and empty
strings are preserved. Raw media bytes are never accepted: describe media with
`media`, `uri` or `file` parts. Content never becomes a metric dimension and
never appears in diagnostics.

## Providers you own vs. the packaged pipeline

**Your providers** — pass any OpenTelemetry tracer/meter/logger providers
(Node SDK, a browser `WebTracerProvider`, a vendor SDK). You decide sampling,
processors and exporters.

**The packaged pipeline** — `@xybrid/observability/otlp` builds providers that
are never registered globally:

```ts
import { createOtlpPipeline, otlpConfigFromEnv, xybridTracesPreset } from "@xybrid/observability/otlp";

const pipeline = createOtlpPipeline({
  traces: { endpoint: "http://localhost:4318/v1/traces" },
  metrics: { endpoint: "http://localhost:4318/v1/metrics" },
  logs: { endpoint: "http://localhost:4318/v1/logs", protocol: "http/json" },
  resource: { serviceName: "my-app", serviceVersion: "1.0.0" },
});
const observer = pipeline.observer();          // content options optional
// …
pipeline.stats();                              // { traces, logs, metrics }: SignalStats each
await pipeline.forceFlush(5000);               // bounded; resolves false on timeout
await pipeline.shutdown(5000);                 // bounded, idempotent

createOtlpPipeline(xybridTracesPreset(process.env.XYBRID_API_KEY!)); // traces only, Bearer auth
createOtlpPipeline(otlpConfigFromEnv());       // OTEL_EXPORTER_OTLP_* (Node; {} in browsers)
```

Delivery uses parent-based sampling with every new root recorded, bounded queues
(2048 items, 32 MiB per signal), batches (256), and request bodies (480 KiB).
Trace and log batches split; oversized single items are dropped. Metrics
collections are never split and fail when oversized. Retries use exponential
backoff capped at 2 seconds; `Retry-After` can exceed that cap while fitting the
single absolute export deadline. Nothing is persisted.

Each signal's `SignalStats` contains `accepted`, `exported`, `dropped`, `failed`
and `pending`, plus `bufferedBytes`: the bytes the pending items hold against
the 32 MiB `batch.maxBufferedBytes` budget (the per-item estimate admitted, or a
metrics collection's encoded body; not process memory; zero once nothing is
pending, so always after shutdown). Every snapshot satisfies
`accepted = exported + dropped + failed + pending`. Refused items count as
accepted and dropped; lost acknowledgements and partial-success rejections
count as failed. Retries never multiply item counts. Metrics count data points.

`forceFlush` covers items pending at the call plus one metrics collection. A
wait timeout returns `false` while delivery continues. `shutdown` stops
admission, drains and takes a final metrics collection, cancels pending network
work at its deadline, and settles the counters before returning. A later call
returns the first outcome with its own wait bound. Both return `false` on loss.

Generic destinations merge OTLP environment headers and resource settings;
`headerPolicy: "isolated"` sends programmatic headers only. The Xybrid preset
always isolates its bearer credential. `environment: null` disables process
environment reads; a supplied environment map supports browsers and tests.
`otlpConfigFromEnv()` honors `OTEL_SDK_DISABLED=true` and signal exporter
settings. Tune `batch.maxBufferedBytes` and `batch.exportTimeoutMs` before
construction; invalid limits throw before workers or timers start.

The pipeline's scheduling timers (batch delay, metric interval, flush, export
and shutdown deadlines) never keep a Node process alive, but an export in
flight does, retry backoff included, until it completes or its deadline aborts
it. Against an unresponsive endpoint a process that exits without `shutdown()`
therefore lingers for up to `batch.exportTimeoutMs` (10 s by default). Call
`shutdown(timeoutMs)` before exiting to bound that wait, and to deliver or
count what is still queued.

### Application-owned providers

`createOtlpExporters` returns standalone `traces`, `logs` and `metrics` exporters;
`OtlpTraceExporter`, `OtlpLogExporter` and `OtlpMetricExporter` can also be
constructed separately. Pair these final exporters with this package's
`Observer` over your own providers to preserve catalog integer/double types and
structured null on the wire. The observer's provider options use OTel's public
provider interfaces; nothing installs global providers.

```ts
const { traces } = createOtlpExporters({
  traces: { endpoint: "http://localhost:4318/v1/traces" },
  batch: { exportTimeoutMs: 2000 },
  environment: null,
});
const tracerProvider = new TracerProvider({
  spanProcessors: [new SimpleSpanProcessor({ exporter: traces! })],
});
const observer = new Observer({ tracerProvider, meterProvider: null, loggerProvider: null });
// …
await tracerProvider.shutdown();
```

Import the exporters from `@xybrid/observability/otlp` and the provider/processor
from `@opentelemetry/sdk-trace`. Set the exporter's deadline no longer than the
calling processor or metric reader's timeout. Standalone stats count items
handed to `export()`, and their byte budget bounds in-flight work; the calling
processor owns queued items.

The Xybrid preset sends only traces. Evaluation and feedback events need a logs
pipeline; `recordEvaluation`/`recordFeedback` return `true` only when an enabled
logger received the event, so they return `false` on a traces-only pipeline.

### Langfuse and Phoenix

Instrumentation stays canonical; translation happens in an OpenTelemetry
Collector at export time. Point the pipeline at the Collector
(`http://localhost:4318`) and use the profiles in
[`crates/xybrid-observability/collector/`](../../crates/xybrid-observability/collector/)
to forward traces to Langfuse or Phoenix and route metrics and evaluation
events to the backends that accept them.

## OpenTelemetry versions

The OpenTelemetry JS SDK is mid-transition between its 2.x and 3.x lines, and
the package numbers do not move together. `/otlp` is built and tested against
two aligned families from the 3.x line. Do not mix SDK and transformer families:

| Package | Family 3.0 | Family 3.1 |
|---|---|---|
| `@opentelemetry/api` | 1.10.0 | 1.10.0 |
| `@opentelemetry/core`, `resources`, `sdk-trace`, `sdk-metrics`, `sdk-logs` | 3.0.x | 3.1.x |
| `@opentelemetry/otlp-transformer` | 0.300.x | 0.301.x |

```sh
npm install @opentelemetry/api@1.10.0 @opentelemetry/core@3.0.0 @opentelemetry/resources@3.0.0 \
  @opentelemetry/sdk-trace@3.0.0 @opentelemetry/sdk-metrics@3.0.0 @opentelemetry/sdk-logs@3.0.0 \
  @opentelemetry/otlp-transformer@0.300.0
```

Why this set: `@opentelemetry/api@latest` is 1.10.0, and the 2.x SDK
(`sdk-trace-base@2.12.0`, `sdk-logs@0.223.0`) declares `@opentelemetry/api <1.10.0`,
so a new application installing the latest API would hit peer conflicts with it.
The 3.x line also carries the logs API inside `@opentelemetry/api`. It requires
Node >= 22.15 (Node 20 reached end of life in April 2026); browsers are
unaffected. The SDK packages are **optional** peer dependencies so the core
stays two small packages for applications that bring their own providers. The
OTLP sender is our own (on `otlp-transformer`'s serializers and `fetch`) rather
than `exporter-*-otlp-http`, so request bodies, retries and drop counts are
bounded and exact as the contract requires.

The core's API peer range is `^1.9.0`; its provider inputs are structural.
This package declares optional 3.0/3.1 SDK peer ranges. npm may accept
conflicting optional peers; the OTLP boundary also rejects an unsupported SDK
at runtime: every `/otlp` entry point throws `TypeError` ("requires OpenTelemetry
SDK 3.0 or 3.1") before creating a timer or provider. CI first installs a
complete, valid API 1.9 + SDK 2.x tree (core, resources, sdk-trace, sdk-logs,
sdk-metrics, otlp-transformer), so an unrelated conflict or missing module
cannot make the check pass. It then requires npm to refuse the package with a
peer conflict (or to accept it), installs it anyway, and checks that each entry
point throws that exact error.
SDK 2.x installation compatibility is not supported by this release. The core
reads the global logger provider from `@opentelemetry/api` when present and from
`@opentelemetry/api-logs` otherwise.

## Future runtime adapters

[`examples/runtime-readiness.ts`](examples/runtime-readiness.ts) shows callbacks
for Transformers.js generation, LiteRT embeddings and Foundation Models text
streams. They use the portable observer without claiming those runtimes are
integrated into Xybrid. The caller owns runtime invocation, loading and export
lifecycle. Missing usage stays absent; a text chunk alone never reports first
token latency.

## Development

```sh
pnpm install
pnpm test             # bun test: catalog parity, contract fixtures, trace context, truncation, pipeline, workers
pnpm lint && pnpm typecheck
pnpm build            # dist/index.js and dist/otlp.js with declarations
pnpm pack:release     # pack and verify the archive in an isolated consumer
pnpm test:families   # real packed consumers: aligned 3.0 + 3.1; core-only; unsupported 2.x
pnpm fixtures:dump    # normalized fixture output → .context/fixture-output/js/
pnpm example          # examples/node-basic.ts (console, or OTLP when OTEL_EXPORTER_OTLP_* is set)
pnpm bench            # overhead vs disabled, content pipeline cost (after pnpm build; see tools/scripts/observability_perf)
```

## Releases

The package and instrumentation scope version follow the Cargo workspace version.
`just bump-version` updates the manifest, `src/version.ts`, and the exact
`@xybrid/web` dependency together. See [RELEASING.md](./RELEASING.md) for the
branch-driven release and npm approval process.
