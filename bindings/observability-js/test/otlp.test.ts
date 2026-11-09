// The packaged OTLP pipeline: providers, presets, all three signals, saturation on hot paths,
// and export-request tracing suppression.
import { afterAll, afterEach, beforeAll, describe, expect, spyOn, test } from "bun:test";
import {
  type ContextManager,
  context,
  type DiagLogger,
  DiagLogLevel,
  diag,
  ROOT_CONTEXT,
  type Tracer,
  trace,
} from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  createOtlpPipeline,
  type OtlpPipeline,
  type SignalStats,
  xybridTracesPreset,
} from "../src/otlp/index.ts";
import { closedUrl, startServer, type TestServer } from "./support/otlp-server.ts";

const request = { operation: "chat", provider: "llama.cpp", model: "tiny" } as const;

const stats = (partial: Partial<SignalStats>): SignalStats => ({
  accepted: 0,
  exported: 0,
  dropped: 0,
  failed: 0,
  pending: 0,
  ...partial,
});

const cleanups: (() => Promise<unknown>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0)) await cleanup();
});
const track = (pipeline: OtlpPipeline, server?: TestServer): OtlpPipeline => {
  cleanups.push(async () => {
    await pipeline.shutdown(1000);
    await server?.close();
  });
  return pipeline;
};

describe("createOtlpPipeline", () => {
  test("queue saturation drops and counts new spans", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { maxQueueSize: 2, maxExportBatchSize: 2, scheduledDelayMs: 60_000 },
      }),
      server,
    );
    const observer = pipeline.observer();
    for (let index = 0; index < 5; index += 1) observer.startInference(request).end();
    expect(pipeline.stats().traces).toEqual(stats({ accepted: 5, dropped: 3, pending: 2 }));
    expect(await pipeline.forceFlush(2000)).toBe(true);
    expect(pipeline.stats().traces).toEqual(stats({ accepted: 5, exported: 2, dropped: 3 }));
  });

  test("an unavailable endpoint never throws on hot paths and is counted failed", async () => {
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${await closedUrl()}/v1/traces`, timeoutMs: 500 },
        logs: { endpoint: `${await closedUrl()}/v1/logs`, timeoutMs: 500 },
        metrics: { endpoint: `${await closedUrl()}/v1/metrics`, timeoutMs: 500 },
        retry: { maxRetries: 1, initialDelayMs: 10 },
      }),
    );
    const observer = pipeline.observer({ capture: "event_only" });
    const started = performance.now();
    for (let index = 0; index < 3; index += 1) {
      const op = observer.startInference(request);
      op.recordChunk();
      op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "hi" }] }]);
      op.end();
    }
    expect(performance.now() - started).toBeLessThan(100);
    expect(await pipeline.forceFlush(3000)).toBe(false);
    const { traces, logs, metrics } = pipeline.stats();
    expect(traces).toEqual(stats({ accepted: 3, failed: 3 }));
    expect(logs).toEqual(stats({ accepted: 3, failed: 3 }));
    expect(metrics.failed).toBeGreaterThan(0);
    expect(metrics.accepted).toBe(metrics.failed);
  });

  test("never registers global providers", () => {
    const before = trace.getTracerProvider();
    track(createOtlpPipeline({ traces: { endpoint: "http://127.0.0.1:1/v1/traces" } }));
    expect(trace.getTracerProvider()).toBe(before);
  });

  test("the Xybrid preset sends only traces, with the bearer header", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline(xybridTracesPreset("xk_test", { endpoint: `${server.url}/v1/traces` })),
      server,
    );
    expect(pipeline.meterProvider).toBeUndefined();
    expect(pipeline.loggerProvider).toBeUndefined();
    const observer = pipeline.observer();
    const op = observer.startInference(request);
    op.recordUsage({ inputTokens: 3, outputTokens: 4 });
    op.end();
    const target = op.observationRef();
    expect(target).toBeDefined();
    if (target !== undefined) {
      expect(observer.recordEvaluation(target, { name: "relevance", scoreValue: 1 })).toBe(false);
    }
    expect(await pipeline.forceFlush(2000)).toBe(true);
    expect(server.requests.map((received) => received.path)).toEqual(["/v1/traces"]);
    expect(server.requests[0]?.headers["authorization"]).toBe("Bearer xk_test");
    expect(server.requests[0]?.headers["content-type"]).toBe("application/x-protobuf");
    expect(pipeline.stats()).toEqual({
      traces: stats({ accepted: 1, exported: 1 }),
      logs: stats({}),
      metrics: stats({}),
    });
  });

  test("the Xybrid preset defaults to the ingest endpoint and rejects empty keys", () => {
    const preset = xybridTracesPreset("k");
    expect(preset.traces?.endpoint).toBe("https://ingest.xybrid.dev/v1/traces");
    expect(preset.traces?.headerPolicy).toBe("isolated");
    expect(Object.keys(preset)).toEqual(["traces"]);
    expect(() => xybridTracesPreset(" ")).toThrow(TypeError);
  });

  test("http/json exports all three signals to their endpoints", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
        metrics: { endpoint: `${server.url}/v1/metrics`, protocol: "http/json" },
        logs: { endpoint: `${server.url}/v1/logs`, protocol: "http/json" },
        resource: { serviceName: "fixture-app", serviceVersion: "1.2.3" },
        environment: null,
      }),
      server,
    );
    const observer = pipeline.observer({ capture: "event_only" });
    const op = observer.startInference(request);
    op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "hello" }] }]);
    op.end();
    expect(await pipeline.forceFlush(3000)).toBe(true);
    const paths = server.requests.map((received) => received.path).sort();
    expect(paths).toEqual(["/v1/logs", "/v1/metrics", "/v1/traces"]);
    const traces = JSON.parse(server.received("/v1/traces")[0]?.body.toString() ?? "{}");
    const span = traces.resourceSpans[0].scopeSpans[0].spans[0];
    expect(span.name).toBe("chat tiny");
    const service = traces.resourceSpans[0].resource.attributes.find(
      (attribute: { key: string }) => attribute.key === "service.name",
    );
    expect(service.value.stringValue).toBe("fixture-app");
    expect(server.received("/v1/logs")[0]?.body.toString()).toContain(
      "gen_ai.client.inference.operation.details",
    );
    const { traces: spans, logs, metrics } = pipeline.stats();
    expect(spans).toEqual(stats({ accepted: 1, exported: 1 }));
    expect(logs).toEqual(stats({ accepted: 1, exported: 1 }));
    expect(metrics.exported).toBeGreaterThan(0);
    expect(metrics).toEqual(stats({ accepted: metrics.exported, exported: metrics.exported }));
  });

  test("retries are bounded and diagnostics never carry headers or payloads", async () => {
    const server = await startServer(() => ({ status: 503 }));
    const messages: string[] = [];
    const capture = (message: string): void => {
      messages.push(message);
    };
    const logger: DiagLogger = {
      error: capture,
      warn: capture,
      info: capture,
      debug: capture,
      verbose: capture,
    };
    diag.setLogger(logger, DiagLogLevel.ALL);
    try {
      const pipeline = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces`, headers: { "x-secret": "s3cr3t" } },
          retry: { maxRetries: 2, initialDelayMs: 5, maxDelayMs: 10 },
        }),
        server,
      );
      const op = pipeline.observer().startInference(request);
      op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "PAYLOAD" }] }]);
      op.end();
      expect(await pipeline.forceFlush(3000)).toBe(false);
      expect(server.requests).toHaveLength(3);
      expect(pipeline.stats().traces).toEqual(stats({ accepted: 1, failed: 1 }));
      const joined = messages.join("\n");
      expect(joined).toContain("HTTP 503");
      expect(joined).not.toContain("s3cr3t");
      expect(joined).not.toContain("PAYLOAD");
    } finally {
      diag.disable();
    }
  });
});

describe("export request tracing suppression", () => {
  const manager = new AsyncLocalStorageContextManager();
  beforeAll(() => context.setGlobalContextManager(manager.enable()));
  afterAll(() => {
    manager.disable();
    context.disable();
  });

  // Simulate HTTP instrumentation on the pipeline's own tracer. Cap the generated spans so a
  // regression fails an assertion instead of leaving an infinite export loop running in the test.
  const instrumentFetch = (tracer: Tracer) => {
    const original = globalThis.fetch;
    const suppressed: boolean[] = [];
    let requests = 0;
    const fetchWithTracing = async (...[input, init]: Parameters<typeof fetch>) => {
      requests += 1;
      const attempt = requests;
      suppressed.push(isTracingSuppressed(context.active()));
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      suppressed.push(isTracingSuppressed(context.active()));
      const span = attempt <= 8 ? tracer.startSpan("HTTP export") : undefined;
      try {
        const response = await original(input, init);
        suppressed.push(isTracingSuppressed(context.active()));
        return response;
      } finally {
        span?.end();
      }
    };
    const instrumented = spyOn(globalThis, "fetch").mockImplementation(
      Object.assign(fetchWithTracing, original),
    );
    return { suppressed, restore: () => instrumented.mockRestore() };
  };

  for (const protocol of ["http/json", "http/protobuf"] as const) {
    test(`${protocol}: traces, metrics and logs suppress uploads and HTTP retries`, async () => {
      const attempts = new Map<string, number>();
      const server = await startServer((received) => {
        const count = (attempts.get(received.path) ?? 0) + 1;
        attempts.set(received.path, count);
        return { status: count === 1 ? 503 : 200 };
      });
      const pipeline = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces`, protocol },
          metrics: { endpoint: `${server.url}/v1/metrics`, protocol },
          logs: { endpoint: `${server.url}/v1/logs`, protocol },
          batch: { scheduledDelayMs: 60_000 },
          metricExportIntervalMs: 60_000,
          retry: { maxRetries: 1, initialDelayMs: 1, maxDelayMs: 1 },
        }),
        server,
      );
      const tracer = pipeline.tracerProvider?.getTracer("http-instrumentation");
      if (tracer === undefined) throw new Error("expected a tracer provider");
      const instrumented = instrumentFetch(tracer);
      try {
        const op = pipeline.observer({ capture: "event_only" }).startInference(request);
        op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "hello" }] }]);
        op.end();
        expect(isTracingSuppressed(context.active())).toBe(false);
        expect(await pipeline.forceFlush(3000)).toBe(true);
        expect([...attempts.entries()].sort()).toEqual([
          ["/v1/logs", 2],
          ["/v1/metrics", 2],
          ["/v1/traces", 2],
        ]);
        expect(pipeline.stats().traces).toEqual(stats({ accepted: 1, exported: 1 }));
        expect(await pipeline.shutdown(3000)).toBe(true);
        expect(instrumented.suppressed.length).toBeGreaterThanOrEqual(18);
        expect(instrumented.suppressed.every(Boolean)).toBe(true);
        expect(isTracingSuppressed(context.active())).toBe(false);
      } finally {
        instrumented.restore();
      }
    });
  }

  test("scheduled and shutdown trace exports do not create more spans", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 5 },
      }),
      server,
    );
    const tracer = pipeline.tracerProvider?.getTracer("http-instrumentation");
    if (tracer === undefined) throw new Error("expected a tracer provider");
    const instrumented = instrumentFetch(tracer);
    try {
      pipeline.observer().startInference(request).end();
      await server.waitFor(() => pipeline.stats().traces.exported === 1, 2000);
      pipeline.observer().startInference(request).end();
      expect(await pipeline.shutdown(2000)).toBe(true);
      expect(server.requests).toHaveLength(2);
      expect(pipeline.stats().traces).toEqual(stats({ accepted: 2, exported: 2 }));
      expect(instrumented.suppressed).toEqual(Array<boolean>(6).fill(true));
    } finally {
      instrumented.restore();
    }
  });

  for (const failure of ["HTTP 503", "network error"]) {
    test(`a synchronous browser context manager suppresses all signals after ${failure}`, async () => {
      // Like a browser stack context manager, this deliberately restores context as soon as
      // the callback returns a promise; it cannot carry suppression across await or timers.
      let active = ROOT_CONTEXT;
      const synchronous: ContextManager = {
        active: () => active,
        with: (ctx, fn, thisArg, ...args) => {
          const previous = active;
          active = ctx;
          try {
            return fn.call(thisArg, ...args);
          } finally {
            active = previous;
          }
        },
        bind: (_ctx, target) => target,
        enable: () => synchronous,
        disable: () => {
          active = ROOT_CONTEXT;
          return synchronous;
        },
      };
      context.disable();
      context.setGlobalContextManager(synchronous.enable());
      const serverAttempts = new Map<string, number>();
      const server = await startServer((received) => {
        const attempt = (serverAttempts.get(received.path) ?? 0) + 1;
        serverAttempts.set(received.path, attempt);
        return { status: failure === "HTTP 503" && attempt === 1 ? 503 : 200 };
      });
      const pipeline = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces` },
          metrics: { endpoint: `${server.url}/v1/metrics` },
          logs: { endpoint: `${server.url}/v1/logs` },
          batch: { scheduledDelayMs: 60_000 },
          metricExportIntervalMs: 60_000,
          retry: { maxRetries: 1, initialDelayMs: 1, maxDelayMs: 1 },
        }),
        server,
      );
      const tracer = pipeline.tracerProvider?.getTracer("browser-http-instrumentation");
      if (tracer === undefined) throw new Error("expected a tracer provider");
      const original = globalThis.fetch;
      const attempts = new Map<string, number>();
      const atFetch: boolean[] = [];
      const afterAwait: boolean[] = [];
      const fetchWithTracing = async (...[input, init]: Parameters<typeof fetch>) => {
        const path = new URL(String(input)).pathname;
        const attempt = (attempts.get(path) ?? 0) + 1;
        attempts.set(path, attempt);
        atFetch.push(isTracingSuppressed(context.active()));
        // HTTP instrumentations decide whether to trace synchronously at fetch entry.
        const span = atFetch.length <= 8 ? tracer.startSpan("HTTP export") : undefined;
        try {
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
          afterAwait.push(isTracingSuppressed(context.active()));
          if (failure === "network error" && attempt === 1) throw new TypeError("offline");
          return await original(input, init);
        } finally {
          span?.end();
        }
      };
      const instrumented = spyOn(globalThis, "fetch").mockImplementation(
        Object.assign(fetchWithTracing, original),
      );
      try {
        pipeline.observer({ capture: "event_only" }).startInference(request).end();
        expect(await pipeline.forceFlush(3000)).toBe(true);
        expect([...attempts.entries()].sort()).toEqual([
          ["/v1/logs", 2],
          ["/v1/metrics", 2],
          ["/v1/traces", 2],
        ]);
        expect(pipeline.stats().traces.accepted).toBe(1);
        expect(await pipeline.shutdown(3000)).toBe(true);
        expect(atFetch.length).toBeGreaterThanOrEqual(6);
        expect(atFetch.every(Boolean)).toBe(true);
        expect(afterAwait.every((suppressed) => !suppressed)).toBe(true);
        expect(isTracingSuppressed(context.active())).toBe(false);
      } finally {
        instrumented.mockRestore();
        synchronous.disable();
        context.disable();
        context.setGlobalContextManager(manager.enable());
      }
    });
  }
});
