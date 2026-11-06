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
for await (const chunk of stream) op.recordChunk();      // first chunk, chunk gaps
op.recordUsage({ inputTokens: 20, outputTokens: 42 });
op.recordOutputMessages([{ role: "assistant", parts: [{ type: "text", content: text }], finish_reason: "stop" }]);
op.recordFinishReasons(["stop"]);
op.end();                                                 // or endWithError({ errorType, message }) / cancel()

const ref = op.observationRef();                          // { traceId, spanId } for feedback
if (ref) observer.recordFeedback(ref, { score: 1, label: "thumbs_up" });
```

Also: `startWorkflow(name)`, `startTool({ name, callId, toolType, arguments })`
with `recordResult(value)`, `startModelLoad({ model, format, sizeBytes })` with
`recordCacheHit(hit)`, and `op.startAttempt({ index, reason, provider, location })`
for retries and fallbacks under one logical request. Every timing call takes an
optional epoch-millisecond time so work measured elsewhere (a worker, a native
runtime) is recorded with its real times. Hot paths never await, flush, or do
I/O.

### Context across workers and processes

An explicit `parent` always wins over the ambient OpenTelemetry context:

```ts
import { serializeContext, deserializeContext, injectTraceHeaders } from "@xybrid/observability";

worker.postMessage({ observation: serializeContext(op.context()) });   // structured-clone safe
// in the worker
const parent = deserializeContext(event.data.observation);
observer.startTool({ name: "decode" }, { parent });

await fetch(url, { headers: injectTraceHeaders(op.context(), new Headers()) });
observer.startInference(request, { parent: "00-<trace>-<span>-01", tracestate: "vendor=x" });
```

For callbacks that run later on the same thread, `observer.bindContext(callback)`
captures the active application context and restores it when the callback is invoked.
It binds the invocation only; a browser's synchronous context manager still does
not propagate context into native Promise continuations.

Workers report `performance.timeOrigin + performance.now()` (wall-clock epoch
ms), never worker-relative offsets.

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
Langfuse, Phoenix) expects it on spans. Explicit configuration wins, then
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` (Node; invalid values mean
`no_content`), then `span_only`.

Content is bounded: each attribute has a UTF-8 byte budget (`maxAttributeBytes`,
default 16384). Over budget, the longest strings are halved (at code point
boundaries, with `…[truncated]`), then the oldest messages are dropped, then the
attribute is omitted; truncated keys are listed in
`xybrid.genai.content.truncated_attributes`. A `redact(kind, value)` hook runs
before truncation, on error messages too. Raw media bytes are never accepted:
describe media with `media`, `uri` or `file` parts. Content never becomes a
metric dimension and never appears in exporter diagnostics.

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
pipeline.stats();                              // { spansQueued, spansExported, spansDropped, exportFailures }
await pipeline.forceFlush(5000);               // bounded; resolves false on timeout
await pipeline.shutdown(5000);                 // bounded, idempotent

createOtlpPipeline(xybridTracesPreset(process.env.XYBRID_API_KEY!)); // traces only, Bearer auth
createOtlpPipeline(otlpConfigFromEnv());       // OTEL_EXPORTER_OTLP_* (Node; {} in browsers)
```

Delivery: parent-based sampling with every new root recorded; bounded queues
(2048 spans), batches (256), request bodies (480 KiB; larger batches are split,
a single oversized span is dropped) and retries (2, exponential backoff from
200 ms, honoring `Retry-After`, capped at 2 s); no persistence. A full queue
drops new spans and counts them. `spansDropped` counts every span that will
never be delivered (queue full, after shutdown, oversized, or in a batch whose
export failed); `exportFailures` counts failed batches.

The Xybrid preset sends only traces. Evaluation and feedback events need a logs
pipeline; on a traces-only pipeline `recordEvaluation`/`recordFeedback` return
`false`.

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
one exact, mutually compatible set from the 3.x line:

| Package | Version |
|---|---|
| `@opentelemetry/api` | 1.10.0 (the SDK's peer range is `>=1.10.0 <1.11.0`) |
| `@opentelemetry/core`, `resources`, `sdk-trace`, `sdk-metrics`, `sdk-logs` | 3.0.0 |
| `@opentelemetry/otlp-transformer` | 0.300.0 (depends on exactly the 3.0.0 SDK packages) |

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
This package currently declares exact 3.x SDK peers, including optional peers:
npm can reject an application that already installs conflicting 2.x SDK packages.
SDK 2.x installation compatibility is not supported by this release. The core
reads the global logger provider from `@opentelemetry/api` when present and from
`@opentelemetry/api-logs` otherwise.

## Development

```sh
pnpm install
pnpm test             # bun test: catalog parity, contract fixtures, trace context, truncation, pipeline, workers
pnpm lint && pnpm typecheck
pnpm build            # dist/index.js and dist/otlp.js with declarations
pnpm pack:release     # pack and verify the archive in an isolated consumer
pnpm fixtures:dump    # normalized fixture output → .context/fixture-output/js/
pnpm example          # examples/node-basic.ts (console, or OTLP when OTEL_EXPORTER_OTLP_* is set)
```

## Releases

The package and instrumentation scope version follow the Cargo workspace version.
`just bump-version` updates the manifest and `src/version.ts` together. See [RELEASING.md](./RELEASING.md) for the
branch-driven release and npm approval process.
