import { afterEach, describe, expect, test } from "bun:test";
import {
  createOtlpPipeline,
  type OtlpPipeline,
  type OtlpPipelineOptions,
  type SignalStats,
} from "../src/otlp/index.ts";
import { decodeRequest } from "./support/otlp-decode.ts";
import {
  closedUrl,
  pause,
  type ReceivedRequest,
  startServer,
  type TestServer,
} from "./support/otlp-server.ts";

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
      bufferedBytes: 0,
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
      bufferedBytes: 0,
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
      bufferedBytes: 0,
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

// Mirrors of the Rust reliability tests in crates/xybrid-observability/tests/delivery.rs:
// an outage then recovery, receiver-reconciled metric points, and a shutdown cut-off
// settling every signal exactly. Phases are driven by flushes and by the
// requests the server holds, never by time; every wait is a hang detector of at least 10 s.

const HANG = 10_000;
type SignalName = "traces" | "logs" | "metrics";
const SIGNALS: readonly SignalName[] = ["traces", "logs", "metrics"];
// biome-ignore lint/suspicious/noExplicitAny: decoded wire data is inspected dynamically.
type Json = any;

const decoded = (signal: SignalName, request: ReceivedRequest): Json =>
  decodeRequest(signal, request.headers["content-type"], request.body);
const signalOf = (request: ReceivedRequest): SignalName =>
  request.path.slice("/v1/".length) as SignalName;
const spanNames = (request: ReceivedRequest): string[] =>
  decoded("traces", request)
    .resourceSpans.flatMap((r: Json) => r.scopeSpans)
    .flatMap((s: Json) => s.spans ?? [])
    .map((span: Json) => span.name);
const logBodies = (request: ReceivedRequest): string[] =>
  decoded("logs", request)
    .resourceLogs.flatMap((r: Json) => r.scopeLogs)
    .flatMap((s: Json) => s.logRecords ?? [])
    .map((record: Json) => record.body.stringValue);
const metricsOf = (request: ReceivedRequest): Json[] =>
  decoded("metrics", request)
    .resourceMetrics.flatMap((r: Json) => r.scopeMetrics)
    .flatMap((s: Json) => s.metrics ?? []);
const pointsOf = (metric: Json): Json[] =>
  metric.sum?.dataPoints ?? metric.histogram?.dataPoints ?? metric.gauge?.dataPoints ?? [];
const metricPoints = (request: ReceivedRequest): number =>
  metricsOf(request).reduce((n: number, metric: Json) => n + pointsOf(metric).length, 0);
const itemCount = (request: ReceivedRequest): number => {
  const signal = signalOf(request);
  if (signal === "traces") return spanNames(request).length;
  if (signal === "logs") return logBodies(request).length;
  return metricPoints(request);
};
/** The cumulative `work` counter per `phase`, as decoded. */
const workTotals = (request: ReceivedRequest): Record<string, number> => {
  const totals: Record<string, number> = {};
  for (const metric of metricsOf(request).filter((m: Json) => m.name === "work")) {
    for (const point of metric.sum.dataPoints) {
      const phase = point.attributes.find((a: Json) => a.key === "phase").value.stringValue;
      totals[phase] = Number(point.asInt ?? point.asDouble);
    }
  }
  return totals;
};
const names = (prefix: string, n: number): string[] =>
  Array.from({ length: n }, (_, i) => `${prefix}-${i}`);
const emitSpans = (pipeline: OtlpPipeline, spans: readonly string[]): void => {
  const tracer = pipeline.tracerProvider?.getTracer("app");
  if (tracer === undefined) throw new Error("expected a tracer provider");
  for (const name of spans) tracer.startSpan(name).end();
};
const emitLogs = (pipeline: OtlpPipeline, bodies: readonly string[]): void => {
  const logger = pipeline.loggerProvider?.getLogger("app");
  if (logger === undefined) throw new Error("expected a logger provider");
  for (const body of bodies) logger.emit({ body });
};
const workCounter = (pipeline: OtlpPipeline) => {
  const meter = pipeline.meterProvider?.getMeter("app");
  if (meter === undefined) throw new Error("expected a meter provider");
  return meter.createCounter("work");
};
const allSignals = (
  url: string,
  extra: Partial<OtlpPipelineOptions> = {},
): OtlpPipelineOptions => ({
  traces: { endpoint: `${url}/v1/traces`, timeoutMs: 120_000 },
  logs: { endpoint: `${url}/v1/logs`, timeoutMs: 120_000 },
  metrics: { endpoint: `${url}/v1/metrics`, timeoutMs: 120_000 },
  metricExportIntervalMs: 600_000,
  environment: null,
  ...extra,
});
const settled = (
  stats: SignalStats,
): Pick<SignalStats, "accepted" | "exported" | "dropped" | "failed" | "pending"> => ({
  accepted: stats.accepted,
  exported: stats.exported,
  dropped: stats.dropped,
  failed: stats.failed,
  pending: stats.pending,
});

describe("reliability mirrors (D1, D5a, D6)", () => {
  for (const down of ["503", "lost acknowledgement"] as const) {
    test(`D1: an outage of ${down}s then recovery loses only what was offered while down`, async () => {
      // Healthy, down, healthy again; queue 8, batch 4, one retry, all three signals.
      // | phase  | spans, and log records alike    | metrics (`work` counter)        |
      // | before | 2 offered: 2 exported           | collection of 1 point: exported |
      // | down   | 10 offered: 8 failed, 2 dropped | collection of 2 points: failed  |
      // | after  | 2 offered: 2 exported           | 2 collections of 3: exported    |
      let isDown = false;
      let openGate: () => void = () => undefined;
      const gate = new Promise<void>((resolve) => {
        openGate = resolve;
      });
      const parked: string[] = [];
      const acknowledged = new Map<number, boolean>();
      const server = await startServer(async (request) => {
        if (!isDown) {
          acknowledged.set(request.index, true);
          return { status: 200 };
        }
        // Requests that arrive before the gate opens are held there, so the test knows which
        // batch is in flight when it offers more.
        parked.push(request.path);
        await gate;
        acknowledged.set(request.index, false);
        return down === "503" ? { status: 503 } : { destroy: true };
      });
      cleanups.push(async () => openGate());
      const pipeline = track(
        createOtlpPipeline(
          allSignals(server.url, {
            batch: { maxQueueSize: 8, maxExportBatchSize: 4, scheduledDelayMs: 60_000 },
            retry: { maxRetries: 1, initialDelayMs: 1, maxDelayMs: 1 },
          }),
        ),
        server,
      );
      const work = workCounter(pipeline);

      const before = names("a", 2);
      emitSpans(pipeline, before);
      emitLogs(pipeline, before);
      work.add(1, { phase: "a" });
      expect(await pipeline.forceFlush(HANG)).toBe(true);

      // Down. The first span and log batches are held in flight, the next ones fill the queue,
      // and the last two offers of each are dropped.
      isDown = true;
      const during = names("b", 10);
      emitSpans(pipeline, during.slice(0, 4));
      await server.waitFor(() => parked.includes("/v1/traces"), HANG);
      emitSpans(pipeline, during.slice(4));
      emitLogs(pipeline, during.slice(0, 4));
      await server.waitFor(() => parked.includes("/v1/logs"), HANG);
      emitLogs(pipeline, during.slice(4));
      for (const signal of ["traces", "logs"] as const) {
        const held = pipeline.stats()[signal];
        expect(held.bufferedBytes).toBeGreaterThan(0);
        expect(settled(held)).toEqual({
          accepted: 12,
          exported: 2,
          dropped: 2,
          failed: 0,
          pending: 8,
        });
      }
      work.add(10, { phase: "b" });
      const flush = pipeline.forceFlush(HANG);
      await server.waitFor(() => parked.includes("/v1/metrics"), HANG);
      openGate();
      // Both span batches, both log batches and the collection fail.
      expect(await flush).toBe(false);

      isDown = false;
      const after = names("c", 2);
      emitSpans(pipeline, after);
      emitLogs(pipeline, after);
      work.add(100, { phase: "c" });
      expect(await pipeline.forceFlush(HANG)).toBe(true);
      // Shutdown takes one more collection.
      expect(await pipeline.shutdown(HANG)).toBe(true);

      const stats = pipeline.stats();
      const lost = {
        accepted: 14,
        exported: 4,
        dropped: 2,
        failed: 8,
        pending: 0,
        bufferedBytes: 0,
      };
      expect(stats.traces).toEqual(lost);
      expect(stats.logs).toEqual(lost);
      expect(stats.metrics).toEqual({
        accepted: 1 + 2 + 3 + 3,
        exported: 1 + 3 + 3,
        dropped: 0,
        failed: 2,
        pending: 0,
        bufferedBytes: 0,
      });

      // The receiver agrees: what it acknowledged is exactly what was exported.
      expect(acknowledged.size).toBe(server.requests.length);
      for (const signal of SIGNALS) {
        const sent = server.received(`/v1/${signal}`);
        expect(sent).toHaveLength(signal === "metrics" ? 5 : 6);
        const delivered = sent
          .filter((request) => acknowledged.get(request.index) === true)
          .reduce((n, request) => n + itemCount(request), 0);
        expect(delivered).toBe(stats[signal].exported);
      }
      const healthy = [...before, ...after].sort();
      const triedTwice = during
        .slice(0, 8)
        .flatMap((name) => [name, name])
        .sort();
      for (const [path, items] of [
        ["/v1/traces", spanNames],
        ["/v1/logs", logBodies],
      ] as const) {
        const ok: string[] = [];
        const failed: string[] = [];
        for (const request of server.received(path)) {
          (acknowledged.get(request.index) === true ? ok : failed).push(...items(request));
        }
        expect(ok.sort()).toEqual(healthy);
        expect(failed.sort()).toEqual(triedTwice);
      }
      const recovered = { a: 1, b: 10, c: 100 };
      expect(
        server
          .received("/v1/metrics")
          .map((request) => [acknowledged.get(request.index), workTotals(request)]),
      ).toEqual([
        [true, { a: 1 }],
        [false, { a: 1, b: 10 }],
        [false, { a: 1, b: 10 }],
        [true, recovered],
        [true, recovered],
      ]);
    });
  }

  test("D1: an endpoint that refuses, then comes up, delivers from then on", async () => {
    // Measured loss: what was offered while it refused (2 spans, 2 log records and the
    // collection's point: failed); nothing after. The first collection after it comes up carries
    // the increment recorded while it was down.
    const url = await closedUrl();
    const pipeline = createOtlpPipeline(
      allSignals(url, {
        batch: { scheduledDelayMs: 60_000 },
        retry: { maxRetries: 1, initialDelayMs: 1, maxDelayMs: 1 },
      }),
    );
    const work = workCounter(pipeline);
    const during = names("b", 2);
    emitSpans(pipeline, during);
    emitLogs(pipeline, during);
    work.add(10, { phase: "b" });
    expect(await pipeline.forceFlush(HANG)).toBe(false);
    const stats = pipeline.stats();
    expect([stats.traces.failed, stats.logs.failed, stats.metrics.failed]).toEqual([2, 2, 1]);

    const server = await startServer(undefined, Number(new URL(url).port));
    track(pipeline, server);
    const after = names("c", 2);
    emitSpans(pipeline, after);
    emitLogs(pipeline, after);
    work.add(100, { phase: "c" });
    expect(await pipeline.forceFlush(HANG)).toBe(true);
    expect(await pipeline.shutdown(HANG)).toBe(true);

    const final = pipeline.stats();
    const once = { accepted: 4, exported: 2, dropped: 0, failed: 2, pending: 0, bufferedBytes: 0 };
    expect(final.traces).toEqual(once);
    expect(final.logs).toEqual(once);
    expect(final.metrics).toEqual({
      accepted: 1 + 2 + 2,
      exported: 2 + 2,
      dropped: 0,
      failed: 1,
      pending: 0,
      bufferedBytes: 0,
    });
    const traces = server.received("/v1/traces");
    expect(traces.map(spanNames)).toEqual([after]);
    expect(server.received("/v1/logs").map(logBodies)).toEqual([after]);
    const collections = server.received("/v1/metrics");
    expect(collections.map(workTotals)).toEqual([
      { b: 10, c: 100 },
      { b: 10, c: 100 },
    ]);
    expect(collections.reduce((n, request) => n + metricPoints(request), 0)).toBe(
      final.metrics.exported,
    );
  });

  test("D5a: exported metric points are the points the receiver acknowledged", async () => {
    // A mixed run over every aggregation the default views produce (sum, non-monotonic sum,
    // histogram, gauge). The first collection is rejected (503, no retry), the later ones
    // acknowledged: the points decoded from acknowledged requests are exactly `exported`, those
    // of the rejected one exactly `failed`.
    const server = await startServer((request) => ({ status: request.index === 0 ? 503 : 200 }));
    const pipeline = track(
      createOtlpPipeline({
        metrics: { endpoint: `${server.url}/v1/metrics`, timeoutMs: 120_000 },
        metricExportIntervalMs: 600_000,
        retry: { maxRetries: 0 },
        environment: null,
      }),
      server,
    );
    const meter = pipeline.meterProvider?.getMeter("app");
    if (meter === undefined) throw new Error("expected a meter provider");
    const requests = meter.createCounter("requests");
    const latency = meter.createHistogram("latency");
    const active = meter.createUpDownCounter("active");
    const depth = meter.createGauge("depth");
    for (const route of ["a", "b"]) {
      requests.add(1, { route });
      latency.record(0.5, { route });
    }
    active.add(2);
    depth.record(7);
    expect(await pipeline.forceFlush(HANG)).toBe(false);
    // A new series; two acknowledged collections, and shutdown takes a third.
    requests.add(1, { route: "c" });
    expect(await pipeline.forceFlush(HANG)).toBe(true);
    expect(await pipeline.forceFlush(HANG)).toBe(true);
    expect(await pipeline.shutdown(HANG)).toBe(true);

    const sent = server.received("/v1/metrics");
    expect(sent).toHaveLength(4);
    const kinds = new Set(
      metricsOf(sent[0] as ReceivedRequest).map((metric: Json) =>
        metric.gauge !== undefined
          ? "gauge"
          : metric.histogram !== undefined
            ? "histogram"
            : metric.sum?.isMonotonic === true
              ? "sum"
              : "non-monotonic sum",
      ),
    );
    expect([...kinds].sort()).toEqual(["gauge", "histogram", "non-monotonic sum", "sum"]);
    const points = sent.map(metricPoints);
    expect(points).toEqual([6, 7, 7, 7]);
    const [rejected = 0, ...acknowledged] = points;
    expect(pipeline.stats().metrics).toEqual({
      accepted: rejected + acknowledged.reduce((a, b) => a + b, 0),
      exported: acknowledged.reduce((a, b) => a + b, 0),
      dropped: 0,
      failed: rejected,
      pending: 0,
      bufferedBytes: 0,
    });
  });

  test("D6: a shutdown cut-off fails the export in flight and drops what is queued", async () => {
    // Every item of every signal ends in exactly one bucket, and the receiver agrees with the
    // counters. Per path the first request is rejected (503, no retries), the second acknowledged
    // and every later one held. When shutdown starts, each signal has one export held: its items
    // fail at the deadline, and what waits behind it (the fourth span and log batch, the metrics
    // collection shutdown takes) is dropped.
    const perPath = new Map<string, number>();
    const server = await startServer((request) => {
      const n = (perPath.get(request.path) ?? 0) + 1;
      perPath.set(request.path, n);
      if (n === 1) return { status: 503 };
      if (n === 2) return { status: 200 };
      return { hang: true };
    });
    const pipeline = createOtlpPipeline(
      allSignals(server.url, {
        batch: { maxExportBatchSize: 4, scheduledDelayMs: 60_000, exportTimeoutMs: 120_000 },
        retry: { maxRetries: 0 },
      }),
    );
    cleanups.push(() => server.close());
    // Batches of 4: rejected, acknowledged, held, and 2 items left queued.
    emitSpans(pipeline, names("span", 14));
    emitLogs(pipeline, names("log", 14));
    // Three series: every (cumulative) collection carries three points.
    const counter = workCounter(pipeline);
    for (const series of ["a", "b", "c"]) counter.add(1, { series });

    const count = (path: string, n: number) => () => server.received(path).length >= n;
    // The first flush drives every span and log batch until one is held; each flush takes one
    // metrics collection, delivered one after the other.
    const flushes = [pipeline.forceFlush(HANG * 6)];
    await server.waitFor(count("/v1/traces", 3), HANG);
    await server.waitFor(count("/v1/logs", 3), HANG);
    await server.waitFor(count("/v1/metrics", 1), HANG);
    flushes.push(pipeline.forceFlush(HANG * 6));
    await server.waitFor(count("/v1/metrics", 2), HANG);
    flushes.push(pipeline.forceFlush(HANG * 6));
    await server.waitFor(count("/v1/metrics", 3), HANG);

    expect(await pipeline.shutdown(100)).toBe(false);
    for (const flush of flushes) expect(await flush).toBe(false);

    const stats = pipeline.stats();
    for (const signal of SIGNALS) {
      const s = stats[signal];
      const sent = server.received(`/v1/${signal}`);
      expect(sent).toHaveLength(3);
      const [rejected = 0, acknowledged = 0, held = 0] = sent.map(itemCount);
      const queued = signal === "metrics" ? 3 : 2;
      expect([rejected, acknowledged, held]).toEqual(signal === "metrics" ? [3, 3, 3] : [4, 4, 4]);
      // The client closed the request in flight.
      await server.waitFor(() => sent[2]?.disconnected === true, HANG);
      // What the receiver acknowledged is exactly what was exported.
      expect(s).toEqual({
        accepted: rejected + acknowledged + held + queued,
        exported: acknowledged,
        dropped: queued,
        failed: rejected + held,
        pending: 0,
        bufferedBytes: 0,
      });
    }
    expect(server.requests).toHaveLength(9);
  });
});
