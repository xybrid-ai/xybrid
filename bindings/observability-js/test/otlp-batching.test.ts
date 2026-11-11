// Splitting by final encoded size, oversized items and collections, and bounded buffers
// (R-BATCH-1..5, R-BUF-1/2, R-CNT-5).
import { afterEach, describe, expect, test } from "bun:test";
import { SeverityNumber } from "@opentelemetry/api";
import {
  InMemoryLogRecordExporter,
  LoggerProvider,
  SimpleLogRecordProcessor,
} from "@opentelemetry/sdk-logs";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  TracerProvider,
} from "@opentelemetry/sdk-trace";
import { DEFAULTS, spanContentCeiling } from "../src/index.ts";
import {
  estimateLogBytes,
  estimateSpanBytes,
  jsonStringBytes,
  logEncoder,
  traceEncoder,
} from "../src/otlp/encoding.ts";
import { createOtlpPipeline, type OtlpPipeline } from "../src/otlp/index.ts";
import { decodeRequest } from "./support/otlp-decode.ts";
import { startServer, type TestServer } from "./support/otlp-server.ts";

// biome-ignore lint/suspicious/noExplicitAny: decoded wire data is inspected dynamically.
type Json = any;

const cleanups: (() => Promise<unknown>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0)) await cleanup();
});
const track = (pipeline: OtlpPipeline, server?: TestServer): OtlpPipeline => {
  cleanups.push(async () => {
    await pipeline.shutdown(500);
    await server?.close();
  });
  return pipeline;
};

const PROTOCOLS = ["http/protobuf", "http/json"] as const;

const decodedBodies = (server: TestServer, signal: "traces" | "logs"): Json[] =>
  server
    .received(`/v1/${signal}`)
    .map((request) => decodeRequest(signal, request.headers["content-type"], request.body));

/** Every request has one resource (with attributes) and each scope at most once. */
const assertGrouping = (requests: Json[], resourceKey: string, scopeKey: string): void => {
  for (const request of requests) {
    expect(request[resourceKey]).toHaveLength(1);
    const resource = request[resourceKey][0];
    expect(resource.resource.attributes.some((entry: Json) => entry.key === "service.name")).toBe(
      true,
    );
    const names = resource[scopeKey].map((scope: Json) => scope.scope.name);
    expect(new Set(names).size).toBe(names.length);
  }
};

describe("trace splitting (R-BATCH-1, R-BATCH-2, R-BATCH-5)", () => {
  for (const protocol of PROTOCOLS) {
    test(`${protocol}: halves until every body fits; grouping and order are kept`, async () => {
      const server = await startServer();
      const bound = protocol === "http/json" ? 4096 : 2048;
      const pipeline = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces`, protocol },
          maxRequestBodyBytes: bound,
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      const provider = pipeline.tracerProvider;
      if (provider === undefined) throw new Error("expected a tracer provider");
      // Two scopes, interleaved (R-BATCH-5).
      const tracers = [provider.getTracer("scope-a", "1.0"), provider.getTracer("scope-b")];
      for (let index = 0; index < 10; index += 1) {
        tracers[index % 2]
          ?.startSpan(`s${index}`, { attributes: { padding: "p".repeat(400) } })
          .end();
      }
      expect(await pipeline.forceFlush(3000)).toBe(true);
      expect(server.requests.length).toBeGreaterThan(2);
      for (const request of server.requests) expect(request.body.length).toBeLessThanOrEqual(bound);
      const requests = decodedBodies(server, "traces");
      assertGrouping(requests, "resourceSpans", "scopeSpans");
      // Parts are sent in order; within a part spans are grouped by scope.
      const delivered = requests.flatMap((request: Json) =>
        request.resourceSpans[0].scopeSpans.flatMap((scope: Json) =>
          scope.spans.map((span: Json) => `${scope.scope.name}:${span.name}`),
        ),
      );
      expect([...delivered].sort()).toEqual(
        Array.from({ length: 10 }, (_, index) => `scope-${index % 2 ? "b" : "a"}:s${index}`).sort(),
      );
      const order = delivered.map((entry: string) => Number(entry.split(":s")[1]));
      const perRequest = requests.map((request: Json) =>
        request.resourceSpans[0].scopeSpans
          .flatMap((scope: Json) => scope.spans.map((span: Json) => Number(span.name.slice(1))))
          .sort((a: number, b: number) => a - b),
      );
      for (let index = 1; index < perRequest.length; index += 1) {
        expect(perRequest[index][0]).toBeGreaterThan(perRequest[index - 1].at(-1));
      }
      expect(order).toHaveLength(10);
      expect(pipeline.stats().traces).toEqual({
        accepted: 10,
        exported: 10,
        dropped: 0,
        failed: 0,
        pending: 0,
        bufferedBytes: 0,
      });
    });

    test(`${protocol}: only an oversized single span is dropped, without a request`, async () => {
      const server = await startServer();
      const pipeline = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces`, protocol },
          maxRequestBodyBytes: 4096,
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      const tracer = pipeline.tracerProvider?.getTracer("app");
      if (tracer === undefined) throw new Error("expected a tracer");
      tracer.startSpan("before", { attributes: { padding: "p".repeat(300) } }).end();
      tracer.startSpan("huge", { attributes: { padding: "h".repeat(8192) } }).end();
      tracer.startSpan("after", { attributes: { padding: "p".repeat(300) } }).end();
      expect(await pipeline.forceFlush(3000)).toBe(false);
      const names = decodedBodies(server, "traces").flatMap((request: Json) =>
        request.resourceSpans[0].scopeSpans[0].spans.map((span: Json) => span.name),
      );
      expect(names).toEqual(["before", "after"]);
      for (const request of server.requests) expect(request.body.includes("hhhhhhhh")).toBe(false);
      expect(pipeline.stats().traces).toEqual({
        accepted: 3,
        exported: 2,
        dropped: 1,
        failed: 0,
        pending: 0,
        bufferedBytes: 0,
      });
    });
  }
});

describe("log splitting (R-BATCH-3)", () => {
  for (const protocol of PROTOCOLS) {
    test(`${protocol}: log batches split with grouping; an oversized record is dropped`, async () => {
      const server = await startServer();
      const bound = protocol === "http/json" ? 4096 : 2048;
      const pipeline = track(
        createOtlpPipeline({
          logs: { endpoint: `${server.url}/v1/logs`, protocol },
          maxRequestBodyBytes: bound,
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      const provider = pipeline.loggerProvider;
      if (provider === undefined) throw new Error("expected a logger provider");
      const loggers = [provider.getLogger("logs-a"), provider.getLogger("logs-b")];
      for (let index = 0; index < 10; index += 1) {
        loggers[index % 2]?.emit({
          eventName: `e${index}`,
          severityNumber: SeverityNumber.INFO,
          body: "b".repeat(400),
        });
      }
      loggers[0]?.emit({ eventName: "huge", body: "h".repeat(8192) });
      expect(await pipeline.forceFlush(3000)).toBe(false);
      expect(server.requests.length).toBeGreaterThan(2);
      for (const request of server.requests) expect(request.body.length).toBeLessThanOrEqual(bound);
      const requests = decodedBodies(server, "logs");
      assertGrouping(requests, "resourceLogs", "scopeLogs");
      const names = requests.flatMap((request: Json) =>
        request.resourceLogs[0].scopeLogs.flatMap((scope: Json) =>
          scope.logRecords.map((record: Json) => record.eventName),
        ),
      );
      expect([...names].sort()).toEqual(
        Array.from({ length: 10 }, (_, index) => `e${index}`).sort(),
      );
      expect(pipeline.stats().logs).toEqual({
        accepted: 11,
        exported: 10,
        dropped: 1,
        failed: 0,
        pending: 0,
        bufferedBytes: 0,
      });
    });
  }
});

describe("metrics collections are never split (R-BATCH-4)", () => {
  test("an oversized collection is not sent and fails explicitly; other signals are unaffected", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        metrics: { endpoint: `${server.url}/v1/metrics` },
        maxRequestBodyBytes: 1500,
        metricExportIntervalMs: 60_000,
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    const observer = pipeline.observer();
    for (let index = 0; index < 4; index += 1) {
      const op = observer.startInference({ operation: "chat", provider: "p", model: `m${index}` });
      op.recordUsage({ inputTokens: 1, outputTokens: 1 });
      op.end();
    }
    expect(await pipeline.forceFlush(3000)).toBe(false);
    expect(server.received("/v1/metrics")).toHaveLength(0);
    const { traces, metrics } = pipeline.stats();
    expect(traces.exported).toBe(4);
    expect(metrics.accepted).toBeGreaterThan(0);
    expect(metrics).toEqual({
      accepted: metrics.accepted,
      exported: 0,
      dropped: 0,
      failed: metrics.accepted,
      pending: 0,
      bufferedBytes: 0,
    });
    expect(await pipeline.shutdown(1000)).toBe(false);
  });
});

describe("bounded buffers (R-BUF-1, R-BUF-2, R-CNT-5)", () => {
  test("a hung endpoint: queue 8, batch 4, 20 spans -> 12 dropped, in-flight items counted", async () => {
    const server = await startServer(() => ({ hang: true }));
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { maxQueueSize: 8, maxExportBatchSize: 4, scheduledDelayMs: 60_000 },
      }),
      server,
    );
    const tracer = pipeline.tracerProvider?.getTracer("app");
    if (tracer === undefined) throw new Error("expected a tracer");
    for (let index = 0; index < 4; index += 1) tracer.startSpan(`s${index}`).end();
    await server.waitFor(() => server.requests.length === 1);
    for (let index = 4; index < 20; index += 1) tracer.startSpan(`s${index}`).end();
    const held = pipeline.stats().traces;
    expect(held).toMatchObject({ accepted: 20, exported: 0, dropped: 12, failed: 0, pending: 8 });
    expect(held.bufferedBytes).toBeGreaterThan(0);
  });

  test("the byte budget drops spans beyond it; estimates cover the encoded size", async () => {
    const server = await startServer(() => ({ hang: true }));
    const memory = new InMemorySpanExporter();
    const local = new TracerProvider({
      spanProcessors: [new SimpleSpanProcessor({ exporter: memory })],
    });
    local
      .getTracer("app")
      .startSpan("large", { attributes: { padding: "x".repeat(10_000) } })
      .end();
    const [sample] = memory.getFinishedSpans();
    if (sample === undefined) throw new Error("expected a span");
    const estimate = estimateSpanBytes(sample);
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { maxBufferedBytes: estimate * 3 + 10, scheduledDelayMs: 60_000 },
      }),
      server,
    );
    const tracer = pipeline.tracerProvider?.getTracer("app");
    if (tracer === undefined) throw new Error("expected a tracer");
    for (let index = 0; index < 5; index += 1) {
      tracer.startSpan("large", { attributes: { padding: "x".repeat(10_000) } }).end();
    }
    // The bytes held are exactly the three admitted estimates.
    expect(pipeline.stats().traces).toEqual({
      accepted: 5,
      exported: 0,
      dropped: 2,
      failed: 0,
      pending: 3,
      bufferedBytes: estimate * 3,
    });
    await local.shutdown();
  });

  test("estimates are at least the singleton request size in both encodings", async () => {
    const spans = new InMemorySpanExporter();
    const logs = new InMemoryLogRecordExporter();
    const tracerProvider = new TracerProvider({
      spanProcessors: [new SimpleSpanProcessor({ exporter: spans })],
    });
    const loggerProvider = new LoggerProvider({
      processors: [new SimpleLogRecordProcessor({ exporter: logs })],
    });
    const tricky = 'quote " backslash \\ newline \n control \u0001 snowman ☃ emoji 😀 lone \ud800';
    const span = tracerProvider.getTracer("scope-with-a-long-name", "1.2.3").startSpan(tricky, {
      attributes: {
        s: tricky,
        n: -9007199254740991,
        d: -1.2345678901234568e-300,
        b: true,
        list: [tricky, "", "x"],
        numbers: [1, 2.5, -3],
        empty: "",
      },
      links: [
        {
          context: {
            traceId: "4bf92f3577b34da6a3ce929d0e0e4736",
            spanId: "00f067aa0ba902b7",
            traceFlags: 1,
          },
          attributes: { link: tricky },
        },
      ],
    });
    span.addEvent(tricky, { e: tricky, k: 1 });
    span.addEvent("empty");
    span.setStatus({ code: 2, message: tricky });
    span.end();
    tracerProvider.getTracer("t").startSpan("").end();
    for (const body of [tricky, { a: null, b: [null, "", { c: tricky }], d: 1.5 }, null, 1]) {
      loggerProvider.getLogger("l").emit({
        body: body as never,
        eventName: tricky,
        severityText: "INFO",
        attributes: { a: tricky, nested: { x: [1, null, tricky] } as never },
      });
    }
    for (const protocol of PROTOCOLS) {
      for (const item of spans.getFinishedSpans()) {
        expect(estimateSpanBytes(item)).toBeGreaterThanOrEqual(
          traceEncoder(protocol)([item]).length,
        );
      }
      for (const item of logs.getFinishedLogRecords()) {
        expect(estimateLogBytes(item)).toBeGreaterThanOrEqual(logEncoder(protocol)([item]).length);
      }
    }
    expect(jsonStringBytes(tricky)).toBeGreaterThanOrEqual(JSON.stringify(tricky).length);
    expect(jsonStringBytes(tricky)).toBeGreaterThanOrEqual(
      new TextEncoder().encode(JSON.stringify(tricky)).length,
    );
    await tracerProvider.shutdown();
    await loggerProvider.shutdown();
  });

  test("estimates charge every scope field: large names, versions, schema URLs, attributes", async () => {
    const spans = new InMemorySpanExporter();
    const logs = new InMemoryLogRecordExporter();
    const tracerProvider = new TracerProvider({
      spanProcessors: [new SimpleSpanProcessor({ exporter: spans })],
    });
    const loggerProvider = new LoggerProvider({
      processors: [new SimpleLogRecordProcessor({ exporter: logs })],
    });
    // Each field alone outgrows the item base, so dropping any one of them is caught.
    for (const size of [5_000, 20_000]) {
      const long = (fill: string): string => fill.repeat(size);
      const scopes: [string, string | undefined, string | undefined, string | undefined][] = [
        [long("n"), undefined, undefined, undefined],
        ["n", long("v"), undefined, undefined],
        ["n", undefined, `https://schema.example/${long("s")}`, undefined],
        ["n", undefined, undefined, long("a")],
        [long("n"), long("v"), `https://schema.example/${long("s")}`, long("a")],
      ];
      for (const [name, version, schemaUrl, attribute] of scopes) {
        const schema = schemaUrl === undefined ? {} : { schemaUrl };
        tracerProvider.getTracer(name, version, schema).startSpan("s").end();
        loggerProvider
          .getLogger(name, version, {
            ...schema,
            ...(attribute === undefined ? {} : { attributes: { scope: attribute } }),
          })
          .emit({ body: "b" });
      }
    }
    const finishedSpans = spans.getFinishedSpans();
    const finishedLogs = logs.getFinishedLogRecords();
    expect(finishedSpans).toHaveLength(10);
    expect(finishedLogs).toHaveLength(10);
    for (const protocol of PROTOCOLS) {
      for (const item of finishedSpans) {
        expect(estimateSpanBytes(item)).toBeGreaterThanOrEqual(
          traceEncoder(protocol)([item]).length,
        );
      }
      for (const item of finishedLogs) {
        expect(estimateLogBytes(item)).toBeGreaterThanOrEqual(logEncoder(protocol)([item]).length);
      }
    }
    await tracerProvider.shutdown();
    await loggerProvider.shutdown();
  });

  test("metrics: a byte budget below one collection drops it whole, without a request", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({
        metrics: { endpoint: `${server.url}/v1/metrics` },
        // Smaller than any encoded collection (the resource alone is larger), well under the
        // request body bound, so the collection is refused by the budget, not as oversized.
        batch: { maxBufferedBytes: 64 },
        metricExportIntervalMs: 60_000,
      }),
      server,
    );
    const meter = pipeline.meterProvider?.getMeter("app");
    if (meter === undefined) throw new Error("expected a meter provider");
    meter.createCounter("one").add(1);
    meter.createCounter("two").add(1);
    expect(await pipeline.forceFlush(10_000)).toBe(false);
    expect(pipeline.stats().metrics).toEqual({
      accepted: 2,
      exported: 0,
      dropped: 2,
      failed: 0,
      pending: 0,
      bufferedBytes: 0,
    });
    // The final collection (cumulative: the same two points) is refused the same way.
    expect(await pipeline.shutdown(10_000)).toBe(false);
    expect(pipeline.stats().metrics).toEqual({
      accepted: 4,
      exported: 0,
      dropped: 4,
      failed: 0,
      pending: 0,
      bufferedBytes: 0,
    });
    expect(server.requests).toHaveLength(0);
  }, 30_000);

  test("large distinct scopes saturate the byte budget for spans and logs", async () => {
    const server = await startServer(() => ({ hang: true }));
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        logs: { endpoint: `${server.url}/v1/logs` },
        batch: { maxBufferedBytes: 10_000, scheduledDelayMs: 60_000 },
      }),
      server,
    );
    const tracerProvider = pipeline.tracerProvider;
    const loggerProvider = pipeline.loggerProvider;
    if (tracerProvider === undefined || loggerProvider === undefined) {
      throw new Error("expected tracer and logger providers");
    }
    // Small items whose only weight is a 6,000-character scope name: one fits in 10,000 bytes.
    for (let index = 0; index < 5; index += 1) {
      const scope = `${index}${"x".repeat(6_000)}`;
      tracerProvider.getTracer(scope).startSpan("s").end();
      loggerProvider.getLogger(scope).emit({ body: "b" });
    }
    const expected = { accepted: 5, exported: 0, dropped: 4, failed: 0, pending: 1 };
    for (const stats of [pipeline.stats().traces, pipeline.stats().logs]) {
      expect(stats).toMatchObject(expected);
      // The one admitted item holds its estimate (its 6,000-character scope name and more).
      expect(stats.bufferedBytes).toBeGreaterThan(6_000);
      expect(stats.bufferedBytes).toBeLessThanOrEqual(10_000);
    }
  });

  test("queue full, byte budget full and after shutdown all count dropped and accepted (logs too)", async () => {
    const server = await startServer(() => ({ hang: true }));
    const pipeline = createOtlpPipeline({
      traces: { endpoint: `${server.url}/v1/traces` },
      logs: { endpoint: `${server.url}/v1/logs` },
      batch: { maxQueueSize: 2, scheduledDelayMs: 60_000, maxBufferedBytes: 20_000 },
    });
    cleanups.push(() => server.close());
    const tracer = pipeline.tracerProvider?.getTracer("app");
    const logger = pipeline.loggerProvider?.getLogger("app");
    if (tracer === undefined || logger === undefined) throw new Error("expected providers");
    tracer.startSpan("big", { attributes: { padding: "x".repeat(30_000) } }).end(); // budget
    for (let index = 0; index < 3; index += 1) tracer.startSpan(`s${index}`).end(); // queue
    logger.emit({ body: "y".repeat(30_000) });
    for (let index = 0; index < 3; index += 1) logger.emit({ body: `l${index}` });
    for (const stats of [pipeline.stats().traces, pipeline.stats().logs]) {
      expect(stats).toMatchObject({ accepted: 4, exported: 0, dropped: 2, failed: 0, pending: 2 });
      expect(stats.bufferedBytes).toBeGreaterThan(0);
      expect(stats.bufferedBytes).toBeLessThanOrEqual(20_000);
    }
    const shutdown = pipeline.shutdown(200);
    tracer.startSpan("late").end();
    logger.emit({ body: "late" });
    expect(await shutdown).toBe(false);
    const { traces, logs } = pipeline.stats();
    expect(traces.accepted).toBe(5);
    expect(traces.pending).toBe(0);
    expect(traces.dropped).toBe(3);
    expect(traces.failed).toBe(2);
    expect(logs.accepted).toBe(5);
    expect(logs.pending).toBe(0);
    expect(logs.dropped).toBe(3);
    expect(logs.failed).toBe(2);
  });
});

describe("content budgets fit one span in the request bound", () => {
  // A quote-heavy 600 KB output doubles in OTLP/JSON. With the attribute budget raised past what
  // one request can carry, the clamp truncates it and the span is exported; unclamped, the span
  // would be dropped whole as an oversized item. The ceiling follows the traces protocol:
  // OTLP/protobuf sends content as retained and so carries twice as much. Each pipeline also fits
  // the budgets to its own, smaller, request bound, and a bound under the envelope allowance
  // still carries content.
  const quoted = '"'.repeat(600_000);
  for (const [bound, protocol, encoding, expected] of [
    [491_520, "http/json", "json", 212_992],
    [491_520, "http/protobuf", "protobuf", 425_984],
    [200_000, "http/json", "json", (200_000 - 65_536) / 2],
    [200_000, "http/protobuf", "protobuf", 200_000 - 65_536],
    [40_000, "http/json", "json", 10_000],
    [40_000, "http/protobuf", "protobuf", 20_000],
  ] as const) {
    test(`${protocol}, bound ${bound}: a raised budget is clamped and the span exported`, async () => {
      const server = await startServer();
      const pipeline = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces`, protocol },
          maxRequestBodyBytes: bound,
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      // The options name the other encoding: the pipeline's protocol wins.
      const observer = pipeline.observer({
        capture: "span_only",
        spanContentEncoding: encoding === "json" ? "protobuf" : "json",
        maxAttributeBytes: 1_000_000,
        maxRetainedOperationBytes: 1_000_000,
        // The 600 KB value is past the default 480 KiB preflight bound, which would omit it
        // before any truncation.
        maxTraversalBytes: 1_000_000,
      });
      const ceiling = spanContentCeiling(bound, encoding);
      expect(ceiling).toBe(expected);
      expect(observer.contentBudgets).toEqual({
        maxAttributeBytes: ceiling,
        maxRetainedOperationBytes: ceiling,
        maxRequestBodyBytes: bound,
        spanContentEncoding: encoding,
      });
      const op = observer.startInference({ operation: "chat", provider: "llama.cpp", model: "m" });
      op.recordOutputMessages([
        { role: "assistant", parts: [{ type: "text", content: quoted }], finish_reason: "stop" },
      ]);
      op.end();
      expect(await pipeline.forceFlush(10_000)).toBe(true);
      expect(pipeline.stats().traces).toMatchObject({ accepted: 1, exported: 1, dropped: 0 });
      const sent = server.received("/v1/traces");
      expect(sent).toHaveLength(1);
      expect(sent[0]?.body.length ?? Infinity).toBeLessThanOrEqual(bound);
      const [request] = decodedBodies(server, "traces");
      const span = request.resourceSpans[0].scopeSpans[0].spans[0];
      const value = (key: string): Json =>
        span.attributes.find((entry: Json) => entry.key === key)?.value;
      const output = String(value("gen_ai.output.messages")?.stringValue);
      expect(Buffer.byteLength(output)).toBeLessThanOrEqual(ceiling);
      // The budget is used, not just respected. Truncation halves the string until it fits, so
      // more than half the ceiling arrives: over protobuf that is more than the same bound
      // carries as JSON.
      expect(Buffer.byteLength(output)).toBeGreaterThan(ceiling / 2);
      expect(output.includes(DEFAULTS.truncation_marker)).toBe(true);
      expect(
        value("xybrid.genai.content.truncated_attributes")?.arrayValue.values.map(
          (entry: Json) => entry.stringValue,
        ),
      ).toEqual(["gen_ai.output.messages"]);
    });
  }

  for (const [logsProtocol, encoding, retained] of [
    ["http/json", "json", 212_992],
    ["http/protobuf", "protobuf", 425_984],
  ] as const) {
    test(`protobuf traces, ${logsProtocol} logs: event content fits the logs request`, async () => {
      // The retained budget also bounds content sent as a details event on the logs signal.
      // Fitted to the traces protocol alone, protobuf traces beside JSON logs would retain twice
      // what the JSON logs request can carry, and the event would be dropped whole. So a pipeline
      // fits content as protobuf only when no signal that carries it is JSON (delivery.rs
      // event_content_fits_a_json_logs_signal_beside_protobuf_traces).
      const server = await startServer();
      const pipeline = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/protobuf" },
          logs: { endpoint: `${server.url}/v1/logs`, protocol: logsProtocol },
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      const observer = pipeline.observer({ capture: "event_only" });
      expect(observer.contentBudgets).toMatchObject({
        maxRetainedOperationBytes: retained,
        spanContentEncoding: encoding,
      });
      // Quote-heavy prompt variables: every byte doubles in OTLP/JSON.
      const variables = Object.fromEntries(
        Array.from({ length: 26 }, (_, index) => [
          `variable_${String(index).padStart(2, "0")}`,
          '"'.repeat(16_000),
        ]),
      );
      observer
        .startInference({
          operation: "chat",
          provider: "llama.cpp",
          model: "m",
          prompt: { name: "template", variables },
        })
        .end();
      expect(await pipeline.forceFlush(10_000)).toBe(true);
      expect(pipeline.stats().logs).toMatchObject({ accepted: 1, exported: 1, dropped: 0 });
      const sent = server.received("/v1/logs");
      expect(sent).toHaveLength(1);
      const size = sent[0]?.body.length ?? Infinity;
      expect(size).toBeLessThanOrEqual(491_520);
      // It still carries content: well over one variable's worth.
      expect(size).toBeGreaterThan(100_000);
    });
  }

  test("JSON traces keep the JSON ceiling whatever the logs protocol is", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
        logs: { endpoint: `${server.url}/v1/logs`, protocol: "http/protobuf" },
      }),
      server,
    );
    expect(pipeline.observer().contentBudgets.spanContentEncoding).toBe("json");
  });

  test("an explicit smaller bound wins; an invalid one still throws", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({ traces: { endpoint: `${server.url}/v1/traces` } }),
      server,
    );
    expect(
      pipeline.observer({ maxRequestBodyBytes: 100_000 }).contentBudgets.maxRequestBodyBytes,
    ).toBe(100_000);
    expect(
      pipeline.observer({ maxRequestBodyBytes: 900_000 }).contentBudgets.maxRequestBodyBytes,
    ).toBe(491_520);
    expect(() => pipeline.observer({ maxRequestBodyBytes: 0 })).toThrow(RangeError);
    // Only a smaller traces bound replaces the observer's own (default 480 KiB), as in Rust
    // (delivery.rs an_observer_takes_the_pipeline_traces_bound_only_when_it_is_smaller).
    const bounded = (bound: number): OtlpPipeline =>
      track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces` },
          maxRequestBodyBytes: bound,
        }),
      );
    expect(bounded(200_000).observer().contentBudgets.maxRequestBodyBytes).toBe(200_000);
    const raised = bounded(1_000_000);
    expect(raised.observer().contentBudgets.maxRequestBodyBytes).toBe(491_520);
    expect(
      raised.observer({ maxRequestBodyBytes: 900_000 }).contentBudgets.maxRequestBodyBytes,
    ).toBe(900_000);
    expect(
      raised.observer({ maxRetainedOperationBytes: 1_000_000 }).contentBudgets
        .maxRetainedOperationBytes,
    ).toBe(spanContentCeiling(491_520, "protobuf"));
    // The default protocol is protobuf, so a default observer keeps the catalog default of
    // 416 KiB; over JSON the same bound carries 208 KiB.
    expect(raised.observer().contentBudgets).toMatchObject({
      maxRetainedOperationBytes: DEFAULTS.max_retained_operation_bytes,
      spanContentEncoding: "protobuf",
    });
    const json = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
      }),
    );
    expect(json.observer().contentBudgets).toMatchObject({
      maxRetainedOperationBytes: 212_992,
      spanContentEncoding: "json",
    });
    // The pipeline's protocol replaces the option; an invalid one still throws.
    expect(
      json.observer({ spanContentEncoding: "protobuf" }).contentBudgets.spanContentEncoding,
    ).toBe("json");
    expect(() => json.observer({ spanContentEncoding: "binary" as never })).toThrow(RangeError);
    // Without a traces signal there is no span request to fit.
    const metricsOnly = track(
      createOtlpPipeline({
        metrics: { endpoint: `${server.url}/v1/metrics` },
        maxRequestBodyBytes: 70_000,
      }),
    );
    expect(metricsOnly.observer().contentBudgets.maxRequestBodyBytes).toBe(491_520);
    // ...and no protocol to take the encoding from: the option (default JSON) stands.
    expect(metricsOnly.observer().contentBudgets.spanContentEncoding).toBe("json");
    expect(
      metricsOnly.observer({ spanContentEncoding: "protobuf" }).contentBudgets.spanContentEncoding,
    ).toBe("protobuf");
  });
});
