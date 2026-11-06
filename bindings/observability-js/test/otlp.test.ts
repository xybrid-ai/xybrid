import { afterAll, afterEach, beforeAll, describe, expect, spyOn, test } from "bun:test";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import type { AddressInfo } from "node:net";
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
import { OtlpHttpSender } from "../src/otlp/http.ts";
import {
  createOtlpPipeline,
  type OtlpPipeline,
  otlpConfigFromEnv,
  xybridTracesPreset,
} from "../src/otlp/index.ts";
import { createHarness } from "./support/harness.ts";

type Received = { path: string; headers: IncomingHttpHeaders; body: Buffer };

type TestServer = {
  url: string;
  received: Received[];
  close(): Promise<void>;
};

/** A local OTLP endpoint. `respond` decides the status; `null` never answers. */
const startServer = async (
  respond: (path: string) => number | null = () => 200,
): Promise<TestServer> => {
  const received: Received[] = [];
  const open = new Set<import("node:http").ServerResponse>();
  const server: Server = createServer((request, response) => {
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => chunks.push(chunk));
    request.on("end", () => {
      const path = request.url ?? "";
      received.push({ path, headers: request.headers, body: Buffer.concat(chunks) });
      const status = respond(path);
      if (status === null) {
        open.add(response);
        return;
      }
      response.writeHead(status, { "Content-Type": "application/x-protobuf" }).end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    received,
    close: async () => {
      for (const response of open) response.destroy();
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
};

const closedPort = async (): Promise<string> => {
  const server = await startServer();
  const url = server.url;
  await server.close();
  return url;
};

const request = { operation: "chat", provider: "llama.cpp", model: "tiny" } as const;

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
    expect(pipeline.stats()).toEqual({
      spansQueued: 2,
      spansExported: 0,
      spansDropped: 3,
      exportFailures: 0,
    });
    expect(await pipeline.forceFlush(2000)).toBe(true);
    expect(pipeline.stats()).toEqual({
      spansQueued: 2,
      spansExported: 2,
      spansDropped: 3,
      exportFailures: 0,
    });
  });

  test("an unavailable endpoint never throws on hot paths and is counted", async () => {
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${await closedPort()}/v1/traces`, timeoutMs: 500 },
        retry: { maxRetries: 1, initialDelayMs: 10 },
      }),
    );
    const observer = pipeline.observer();
    const started = performance.now();
    for (let index = 0; index < 3; index += 1) {
      const op = observer.startInference(request);
      op.recordChunk();
      op.end();
    }
    expect(performance.now() - started).toBeLessThan(50);
    expect(await pipeline.forceFlush(3000)).toBe(true);
    expect(pipeline.stats()).toEqual({
      spansQueued: 3,
      spansExported: 0,
      spansDropped: 3,
      exportFailures: 1,
    });
  });

  test("flush and shutdown are bounded and idempotent", async () => {
    const server = await startServer(() => null);
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        metrics: { endpoint: `${server.url}/v1/metrics` },
        logs: { endpoint: `${server.url}/v1/logs` },
      }),
      server,
    );
    pipeline.observer().startInference(request).end();
    let started = performance.now();
    expect(await pipeline.forceFlush(150)).toBe(false);
    expect(performance.now() - started).toBeLessThan(1000);
    started = performance.now();
    const first = pipeline.shutdown(150);
    expect(await first).toBe(false);
    expect(performance.now() - started).toBeLessThan(1000);
    started = performance.now();
    expect(pipeline.shutdown(150)).toBe(first);
    expect(await pipeline.forceFlush(150)).toBe(true);
    expect(performance.now() - started).toBeLessThan(50);
    // Spans ended after shutdown are dropped, not queued.
    pipeline.observer().startInference(request).end();
    expect(pipeline.stats().spansDropped).toBe(1);
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
    expect(server.received.map((received) => received.path)).toEqual(["/v1/traces"]);
    expect(server.received[0]?.headers["authorization"]).toBe("Bearer xk_test");
    expect(server.received[0]?.headers["content-type"]).toBe("application/x-protobuf");
    expect(pipeline.stats().spansExported).toBe(1);
  });

  test("the Xybrid preset defaults to the ingest endpoint and rejects empty keys", () => {
    expect(xybridTracesPreset("k").traces?.endpoint).toBe("https://ingest.xybrid.dev/v1/traces");
    expect(Object.keys(xybridTracesPreset("k"))).toEqual(["traces"]);
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
      }),
      server,
    );
    const observer = pipeline.observer({ capture: "event_only" });
    const op = observer.startInference(request);
    op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "hello" }] }]);
    op.end();
    expect(await pipeline.forceFlush(3000)).toBe(true);
    const paths = server.received.map((received) => received.path).sort();
    expect(paths).toEqual(["/v1/logs", "/v1/metrics", "/v1/traces"]);
    const traces = JSON.parse(
      server.received.find((received) => received.path === "/v1/traces")?.body.toString() ?? "{}",
    );
    const span = traces.resourceSpans[0].scopeSpans[0].spans[0];
    expect(span.name).toBe("chat tiny");
    const service = traces.resourceSpans[0].resource.attributes.find(
      (attribute: { key: string }) => attribute.key === "service.name",
    );
    expect(service.value.stringValue).toBe("fixture-app");
    const logs = server.received.find((received) => received.path === "/v1/logs")?.body.toString();
    expect(logs).toContain("gen_ai.client.inference.operation.details");
  });

  test("oversized batches are split and an oversized span is dropped", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        maxRequestBodyBytes: 2048,
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    const observer = pipeline.observer();
    for (let index = 0; index < 8; index += 1) {
      const op = observer.startInference(request);
      op.setAttribute("padding", "p".repeat(400));
      op.end();
    }
    const huge = observer.startInference(request);
    huge.setAttribute("padding", "h".repeat(4096));
    huge.end();
    expect(await pipeline.forceFlush(3000)).toBe(true);
    expect(server.received.length).toBeGreaterThan(1);
    for (const received of server.received)
      expect(received.body.byteLength).toBeLessThanOrEqual(2048);
    expect(pipeline.stats()).toEqual({
      spansQueued: 9,
      spansExported: 8,
      spansDropped: 1,
      exportFailures: 0,
    });
  });

  test("retries are bounded and diagnostics never carry headers or payloads", async () => {
    const server = await startServer(() => 503);
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
      expect(await pipeline.forceFlush(3000)).toBe(true);
      expect(server.received).toHaveLength(3);
      expect(pipeline.stats().exportFailures).toBe(1);
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
  const instrumentFetch = (tracer: Tracer, rejectFirst = false) => {
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
        if (rejectFirst && attempt === 1) throw new TypeError("network unavailable");
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
      const server = await startServer((path) => {
        const count = (attempts.get(path) ?? 0) + 1;
        attempts.set(path, count);
        return count === 1 ? 503 : 200;
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
        expect(pipeline.stats()).toEqual({
          spansQueued: 1,
          spansExported: 1,
          spansDropped: 0,
          exportFailures: 0,
        });
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
      const deadline = Date.now() + 2000;
      while (pipeline.stats().spansExported === 0 && Date.now() < deadline) {
        await new Promise<void>((resolve) => setTimeout(resolve, 5));
      }
      expect(pipeline.stats().spansExported).toBe(1);
      pipeline.observer().startInference(request).end();
      expect(await pipeline.shutdown(2000)).toBe(true);
      expect(server.received).toHaveLength(2);
      expect(pipeline.stats()).toEqual({
        spansQueued: 2,
        spansExported: 2,
        spansDropped: 0,
        exportFailures: 0,
      });
      expect(instrumented.suppressed).toEqual(Array<boolean>(6).fill(true));
    } finally {
      instrumented.restore();
    }
  });

  test("split requests and network-error retries preserve suppression without leaking it", async () => {
    const server = await startServer();
    cleanups.push(() => server.close());
    const { tracerProvider, spans } = createHarness();
    const tracer = tracerProvider.getTracer("http-instrumentation");
    const serializer = {
      serializeRequest: (value: string[]) => new TextEncoder().encode(JSON.stringify(value)),
    };
    const sender = new OtlpHttpSender(
      "traces",
      { endpoint: `${server.url}/v1/traces` },
      { json: serializer, protobuf: serializer },
      { maxRetries: 1, initialDelayMs: 1 },
      12,
    );
    const instrumented = instrumentFetch(tracer, true);
    try {
      expect(await sender.sendItems(["aaaaaa", "bbbbbb"], (items) => items)).toEqual({
        exported: 2,
        failed: 0,
        oversized: 0,
      });
      expect(server.received).toHaveLength(2);
      expect(instrumented.suppressed).toEqual(Array<boolean>(8).fill(true));
      expect(spans.getFinishedSpans()).toHaveLength(0);
      expect(isTracingSuppressed(context.active())).toBe(false);
      tracer.startSpan("application operation").end();
      expect(spans.getFinishedSpans()).toHaveLength(1);
    } finally {
      instrumented.restore();
      sender.abort();
      await tracerProvider.shutdown();
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
      const server = await startServer((path) => {
        const attempt = (serverAttempts.get(path) ?? 0) + 1;
        serverAttempts.set(path, attempt);
        return failure === "HTTP 503" && attempt === 1 ? 503 : 200;
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
        expect(pipeline.stats().spansQueued).toBe(1);
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

describe("otlpConfigFromEnv", () => {
  test("reads the standard variables", () => {
    const config = otlpConfigFromEnv({
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://collector:4318/",
      OTEL_EXPORTER_OTLP_HEADERS: "api-key=abc%20def,bad",
      OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: "http://logs:4318/custom",
      OTEL_EXPORTER_OTLP_METRICS_PROTOCOL: "http/json",
      OTEL_EXPORTER_OTLP_TRACES_TIMEOUT: "2500",
      OTEL_SERVICE_NAME: "svc",
      OTEL_RESOURCE_ATTRIBUTES: "deployment.environment=dev",
      OTEL_BSP_MAX_QUEUE_SIZE: "100",
      OTEL_METRIC_EXPORT_INTERVAL: "5000",
    });
    expect(config.traces).toEqual({
      endpoint: "http://collector:4318/v1/traces",
      headers: { "api-key": "abc def" },
      protocol: "http/protobuf",
      timeoutMs: 2500,
    });
    expect(config.metrics?.protocol).toBe("http/json");
    expect(config.metrics?.endpoint).toBe("http://collector:4318/v1/metrics");
    expect(config.logs?.endpoint).toBe("http://logs:4318/custom");
    expect(config.resource).toEqual({
      serviceName: "svc",
      attributes: { "deployment.environment": "dev" },
    });
    expect(config.batch).toEqual({ maxQueueSize: 100 });
    expect(config.metricExportIntervalMs).toBe(5000);
  });

  test("exporter=none disables a signal; nothing configured means no signals", () => {
    const config = otlpConfigFromEnv({
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://collector:4318",
      OTEL_METRICS_EXPORTER: "none",
      OTEL_LOGS_EXPORTER: "none",
    });
    expect(Object.keys(config)).toEqual(["traces"]);
    expect(otlpConfigFromEnv({})).toEqual({});
  });
});
