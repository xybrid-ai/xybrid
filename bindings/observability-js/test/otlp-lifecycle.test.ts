import { afterEach, describe, expect, test } from "bun:test";
import { createOtlpPipeline, type OtlpPipeline } from "../src/otlp/index.ts";
import { pause, startServer, type TestServer } from "./support/otlp-server.ts";

const cleanups: (() => Promise<unknown>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0)) await cleanup();
});
const track = (pipeline: OtlpPipeline, server: TestServer): OtlpPipeline => {
  cleanups.push(async () => {
    await pipeline.shutdown(100);
    await server.close();
  });
  return pipeline;
};
const emit = (pipeline: OtlpPipeline): void => {
  const op = pipeline.observer().startInference({ operation: "chat", provider: "p", model: "m" });
  op.recordUsage({ inputTokens: 2, outputTokens: 3 });
  op.end();
  pipeline.loggerProvider?.getLogger("app").emit({ body: "hello" });
};

describe("transport and lifecycle (R-LIFE, R-CANCEL, R-CNT)", () => {
  for (const signal of ["traces", "logs", "metrics"] as const) {
    test(`${signal}: shutdown aborts requests, settles counts and memoizes failure`, async () => {
      const server = await startServer(() => ({ hang: true }));
      const pipeline = track(
        createOtlpPipeline({
          [signal]: { endpoint: `${server.url}/v1/${signal}` },
          metricExportIntervalMs: 60_000,
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      emit(pipeline);
      const flush = pipeline.forceFlush(3000);
      await server.waitFor(() => server.requests.length === 1);
      const started = performance.now();
      expect(await pipeline.shutdown(50)).toBe(false);
      expect(performance.now() - started).toBeLessThan(500);
      expect(await flush).toBe(false);
      const stats = pipeline.stats()[signal];
      expect(stats.accepted).toBeGreaterThan(0);
      expect(stats.pending).toBe(0);
      expect(stats.failed).toBeGreaterThan(0);
      expect(stats.accepted).toBe(stats.exported + stats.dropped + stats.failed);
      expect(await pipeline.shutdown(1000)).toBe(false);
      expect(await pipeline.forceFlush(1000)).toBe(false);
      const count = server.requests.length;
      await server.waitFor(() => server.requests[0]?.disconnected === true);
      await pause(30);
      expect(server.requests.length).toBe(count);
    });
  }
  test("shutdown cancels Retry-After backoff without a later retry", async () => {
    const server = await startServer(() => ({ status: 503, headers: { "retry-after": "2" } }));
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    emit(pipeline);
    const flush = pipeline.forceFlush(3000);
    await server.waitFor(() => server.requests.length === 1);
    expect(await pipeline.shutdown(30)).toBe(false);
    expect(await flush).toBe(false);
    await pause(60);
    expect(server.requests).toHaveLength(1);
    expect(pipeline.stats().traces).toEqual({
      accepted: 1,
      exported: 0,
      failed: 1,
      dropped: 0,
      pending: 0,
    });
  });
  test("flush timeout lets delivery continue; its barrier excludes later work", async () => {
    const server = await startServer((request) =>
      request.index === 0 ? { status: 200, delayMs: 100 } : { hang: true },
    );
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000, maxExportBatchSize: 1 },
      }),
      server,
    );
    emit(pipeline);
    expect(await pipeline.forceFlush(10)).toBe(false);
    const barrier = pipeline.forceFlush(1000);
    emit(pipeline);
    expect(await barrier).toBe(true);
    await server.waitFor(() => server.requests.length === 2);
    expect(pipeline.stats().traces.exported).toBe(1);
    expect(pipeline.stats().traces.pending).toBe(1);
    expect(await pipeline.shutdown(30)).toBe(false);
  });
  test("concurrent shutdown and flush callers have their own wait bounds", async () => {
    const server = await startServer(() => ({ hang: true }));
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    emit(pipeline);
    const shutdown = pipeline.shutdown(400);
    await server.waitFor(() => server.requests.length === 1);
    const started = performance.now();
    expect(await pipeline.shutdown(5)).toBe(false);
    expect(await pipeline.forceFlush(5)).toBe(false);
    expect(performance.now() - started).toBeLessThan(200);
    expect(await shutdown).toBe(false);
  });
  test("retries and lost acknowledgements count each item once", async () => {
    const server = await startServer(() => ({ destroy: true }));
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        retry: { maxRetries: 2, initialDelayMs: 1, maxDelayMs: 1 },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    emit(pipeline);
    expect(await pipeline.forceFlush(1000)).toBe(false);
    expect(server.requests).toHaveLength(3);
    expect(server.requests[0]?.body.equals(server.requests[2]?.body ?? Buffer.alloc(0))).toBe(true);
    expect(pipeline.stats().traces).toEqual({
      accepted: 1,
      exported: 0,
      failed: 1,
      dropped: 0,
      pending: 0,
    });
  });
  test("partial acknowledgement fails only rejected items and never retries", async () => {
    const server = await startServer(() => ({
      status: 200,
      body: '{"partialSuccess":{"rejectedSpans":"1"}}',
    }));
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    emit(pipeline);
    emit(pipeline);
    expect(await pipeline.forceFlush(1000)).toBe(false);
    expect(server.requests).toHaveLength(1);
    expect(pipeline.stats().traces).toEqual({
      accepted: 2,
      exported: 1,
      failed: 1,
      dropped: 0,
      pending: 0,
    });
  });
  test("redirects are rejected without contacting the other origin", async () => {
    const destination = await startServer();
    cleanups.push(() => destination.close());
    const server = await startServer(() => ({
      status: 307,
      headers: { location: `${destination.url}/v1/traces` },
    }));
    const pipeline = track(
      createOtlpPipeline({ traces: { endpoint: `${server.url}/v1/traces` } }),
      server,
    );
    emit(pipeline);
    expect(await pipeline.forceFlush(1000)).toBe(false);
    expect(destination.requests).toHaveLength(0);
    expect(server.requests).toHaveLength(1);
  });
});
