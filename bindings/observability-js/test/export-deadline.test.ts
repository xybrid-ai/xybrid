// Regressions for absolute export deadlines, independent flush barriers and item accounting.
import { afterEach, describe, expect, test } from "bun:test";
import { DEFAULTS } from "../src/catalog.ts";
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
const span = (pipeline: OtlpPipeline, name = "s", padding = 0): void => {
  pipeline.tracerProvider
    ?.getTracer("app")
    .startSpan(name, {
      attributes: { pad: "p".repeat(padding) },
    })
    .end();
};

describe("R-DEADLINE-1 export deadline (no shutdown)", () => {
  for (const signal of ["traces", "logs", "metrics"] as const) {
    test(`${signal}: delayed 200 is aborted at exportTimeoutMs, items failed, disconnect seen`, async () => {
      const server = await startServer(() => ({ status: 200, delayMs: 800 }));
      const p = track(
        createOtlpPipeline({
          [signal]: { endpoint: `${server.url}/v1/${signal}`, timeoutMs: 10_000 },
          batch: { scheduledDelayMs: 60_000, exportTimeoutMs: 150 },
          metricExportIntervalMs: 60_000,
          retry: { maxRetries: 0 },
        }),
        server,
      );
      const op = p.observer().startInference({ operation: "chat", provider: "p", model: "m" });
      op.recordUsage({ inputTokens: 1 });
      op.end();
      p.loggerProvider?.getLogger("app").emit({ body: "x" });
      const t0 = performance.now();
      const ok = await p.forceFlush(5000);
      const took = performance.now() - t0;
      const st = p.stats()[signal];
      expect(ok).toBe(false);
      expect(took).toBeGreaterThanOrEqual(100);
      expect(took).toBeLessThan(600);
      expect(st.pending).toBe(0);
      expect(st.failed).toBeGreaterThan(0);
      expect(st.accepted).toBe(st.failed);
      expect(st.exported).toBe(0);
      expect(server.requests).toHaveLength(1);
      await server.waitFor(() => server.requests[0]?.disconnected === true, 2000);
    });
  }
});

describe("R-DEADLINE-2 split siblings after deadline", () => {
  test("second half not sent once the deadline passed", async () => {
    const server = await startServer((r) =>
      r.index === 0 ? { status: 200, delayMs: 300 } : { status: 200 },
    );
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
        maxRequestBodyBytes: 3000,
        batch: { scheduledDelayMs: 60_000, exportTimeoutMs: 150 },
        retry: { maxRetries: 0 },
      }),
      server,
    );
    for (let i = 0; i < 4; i += 1) span(p, `s${i}`, 900);
    const ok = await p.forceFlush(3000);
    await pause(400);
    expect(ok).toBe(false);
    expect(server.requests.length).toBe(1);
    expect(p.stats().traces).toEqual({
      accepted: 4,
      exported: 0,
      dropped: 0,
      failed: 4,
      pending: 0,
      bufferedBytes: 0,
    });
  });
});

describe("R-DEADLINE-3 Retry-After vs deadline", () => {
  test("Retry-After beyond the deadline fails at once", async () => {
    const server = await startServer(() => ({ status: 503, headers: { "retry-after": "5" } }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000, exportTimeoutMs: 1000 },
      }),
      server,
    );
    span(p);
    const t0 = performance.now();
    const ok = await p.forceFlush(3000);
    const took = performance.now() - t0;
    expect(ok).toBe(false);
    expect(took).toBeLessThan(500);
    expect(server.requests.length).toBe(1);
  });
  test("Retry-After above maxDelay but inside the deadline is honoured", async () => {
    const server = await startServer((r) =>
      r.index === 0 ? { status: 503, headers: { "retry-after": "1" } } : { status: 200 },
    );
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000, exportTimeoutMs: 5000 },
        retry: { initialDelayMs: 10, maxDelayMs: 100 },
      }),
      server,
    );
    span(p);
    const ok = await p.forceFlush(4000);
    const gap = (server.requests[1]?.receivedAt ?? 0) - (server.requests[0]?.receivedAt ?? 0);
    expect(ok).toBe(true);
    expect(gap).toBeGreaterThan(900);
  });
});

describe("R-DEADLINE-4 2xx with slow body", () => {
  test("trickling 2xx body past the deadline still counts exported", async () => {
    const enc = new TextEncoder();
    const server = await startServer(() => ({
      trickle: { status: 200, first: enc.encode("{"), rest: enc.encode("}"), delayMs: 600 },
    }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
        batch: { scheduledDelayMs: 60_000, exportTimeoutMs: 200 },
      }),
      server,
    );
    span(p);
    const ok = await p.forceFlush(3000);
    expect(ok).toBe(true);
    expect(p.stats().traces.exported).toBe(1);
    await server.waitFor(() => server.requests[0]?.disconnected === true);
  });
  test("oversized 2xx body is capped and counts exported", async () => {
    const server = await startServer(() => ({ status: 200, body: "x".repeat(200_000) }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    span(p);
    expect(await p.forceFlush(3000)).toBe(true);
    expect(p.stats().traces.exported).toBe(1);
  });
  // A partial success is only honoured if the client read it: a body one byte over the cap
  // must be ignored (both spans exported); one exactly at the cap must still count.
  for (const [extra, honoured] of [
    [0, true],
    [1, false],
  ] as const) {
    test(`a partial-success body of 64 KiB${extra === 0 ? "" : " + 1 byte"} is ${honoured ? "honoured" : "not read"}`, async () => {
      const json = '{"partialSuccess":{"rejectedSpans":"1"}}';
      // Trailing whitespace keeps the JSON valid, so only the read cap can discard it.
      const body = json + " ".repeat(DEFAULTS.max_response_body_bytes - json.length + extra);
      expect(body.length).toBe(64 * 1024 + extra);
      const server = await startServer(() => ({ status: 200, body }));
      const p = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      span(p, "a");
      span(p, "b");
      expect(await p.forceFlush(3000)).toBe(!honoured);
      expect(server.requests).toHaveLength(1);
      expect(p.stats().traces).toEqual({
        accepted: 2,
        exported: honoured ? 1 : 2,
        dropped: 0,
        failed: honoured ? 1 : 0,
        pending: 0,
        bufferedBytes: 0,
      });
    });
  }
});

describe("R-CFG-6 all redirect codes", () => {
  for (const status of [301, 302, 307, 308]) {
    test(`${status} not followed; item failed`, async () => {
      const dest = await startServer();
      cleanups.push(() => dest.close());
      const server = await startServer(() => ({
        status,
        headers: { location: `${dest.url}/v1/traces` },
      }));
      const p = track(
        createOtlpPipeline({
          traces: { endpoint: `${server.url}/v1/traces` },
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      span(p);
      expect(await p.forceFlush(2000)).toBe(false);
      expect(dest.requests.length).toBe(0);
      expect(server.requests.length).toBe(1);
      expect(p.stats().traces.failed).toBe(1);
    });
  }
});

describe("R-CNT-3 lost ack then success", () => {
  test("destroy then 200 -> exported once, received twice", async () => {
    const server = await startServer((r) => (r.index === 0 ? { destroy: true } : { status: 200 }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        retry: { initialDelayMs: 1, maxDelayMs: 1 },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    span(p);
    expect(await p.forceFlush(2000)).toBe(true);
    expect(server.requests.length).toBe(2);
    expect(p.stats().traces).toEqual({
      accepted: 1,
      exported: 1,
      dropped: 0,
      failed: 0,
      pending: 0,
      bufferedBytes: 0,
    });
  });
  test("max_retries 0 lost ack -> failed", async () => {
    const server = await startServer(() => ({ destroy: true }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        retry: { maxRetries: 0 },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    span(p);
    expect(await p.forceFlush(2000)).toBe(false);
    expect(p.stats().traces.failed).toBe(1);
    expect(server.requests.length).toBe(1);
  });
});

describe("R-CNT-4 partial success protobuf + over-count clamp", () => {
  test("protobuf partial success rejected_spans=1", async () => {
    // ExportTraceServiceResponse { partial_success(1) { rejected_spans(1): 1 } } => 0a 02 08 01
    const body = new Uint8Array([0x0a, 0x02, 0x08, 0x01]);
    const server = await startServer(() => ({ status: 200, body }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    span(p, "a");
    span(p, "b");
    expect(await p.forceFlush(2000)).toBe(false);
    expect(p.stats().traces).toEqual({
      accepted: 2,
      exported: 1,
      dropped: 0,
      failed: 1,
      pending: 0,
      bufferedBytes: 0,
    });
  });
  test("rejected count above item count is clamped", async () => {
    const server = await startServer(() => ({
      status: 200,
      body: '{"partialSuccess":{"rejectedSpans":"99"}}',
    }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
        batch: { scheduledDelayMs: 60_000 },
      }),
      server,
    );
    span(p);
    await p.forceFlush(2000);
    expect(p.stats().traces).toEqual({
      accepted: 1,
      exported: 0,
      dropped: 0,
      failed: 1,
      pending: 0,
      bufferedBytes: 0,
    });
  });
});

describe("R-CANCEL-4/5 barriers", () => {
  test("shutdown during an in-progress flush delivers and succeeds", async () => {
    const server = await startServer(() => ({ status: 200, delayMs: 100 }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000, maxExportBatchSize: 1 },
      }),
      server,
    );
    for (let i = 0; i < 3; i += 1) span(p, `s${i}`);
    const flush = p.forceFlush(3000);
    await server.waitFor(() => server.requests.length === 1);
    expect(await p.shutdown(3000)).toBe(true);
    expect(await flush).toBe(true);
    expect(p.stats().traces).toEqual({
      accepted: 3,
      exported: 3,
      dropped: 0,
      failed: 0,
      pending: 0,
      bufferedBytes: 0,
    });
  });
  test("concurrent flushes: the later one waits for its later items", async () => {
    const server = await startServer(() => ({ status: 200, delayMs: 80 }));
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        batch: { scheduledDelayMs: 60_000, maxExportBatchSize: 1 },
      }),
      server,
    );
    span(p, "a");
    let firstDoneAt = 0;
    const f1 = p.forceFlush(3000).then((v) => {
      firstDoneAt = performance.now();
      return v;
    });
    span(p, "b");
    span(p, "c");
    let secondDoneAt = 0;
    const f2 = p.forceFlush(3000).then((v) => {
      secondDoneAt = performance.now();
      return v;
    });
    expect(await f1).toBe(true);
    expect(secondDoneAt).toBe(0);
    expect(p.stats().traces).toMatchObject({ exported: 1, pending: 2 });
    expect(await f2).toBe(true);
    expect(secondDoneAt).toBeGreaterThan(firstDoneAt);
    expect(p.stats().traces.exported).toBe(3);
  });
});

describe("R-CNT-1 invariant under concurrent load", () => {
  test("every snapshot satisfies the invariant (all signals)", async () => {
    const server = await startServer((r) =>
      r.index % 3 === 0 ? { status: 503 } : { status: 200, delayMs: 2 },
    );
    const p = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces` },
        logs: { endpoint: `${server.url}/v1/logs` },
        metrics: { endpoint: `${server.url}/v1/metrics` },
        metricExportIntervalMs: 5,
        batch: { scheduledDelayMs: 1, maxExportBatchSize: 7, maxQueueSize: 50 },
        retry: { initialDelayMs: 1, maxDelayMs: 2 },
      }),
      server,
    );
    let bad = 0;
    const check = () => {
      for (const s of Object.values(p.stats()))
        if (s.accepted !== s.exported + s.dropped + s.failed + s.pending) bad += 1;
    };
    const logger = p.loggerProvider?.getLogger("app");
    for (let round = 0; round < 40; round += 1) {
      for (let i = 0; i < 20; i += 1) {
        p.observer().startInference({ operation: "chat", provider: "p" }).end();
        logger?.emit({ body: "b" });
        check();
      }
      await pause(1);
      check();
    }
    await p.forceFlush(3000);
    check();
    expect(bad).toBe(0);
  });
});

describe("per-attempt and shutdown deadlines (R-DEADLINE-5, A7)", () => {
  for (const signal of ["traces", "logs", "metrics"] as const) {
    test(`${signal}: a shorter signal timeout permits retries within one export deadline`, async () => {
      const server = await startServer(() => ({ hang: true }));
      const pipeline = track(
        createOtlpPipeline({
          [signal]: { endpoint: `${server.url}/v1/${signal}`, timeoutMs: 40 },
          batch: { scheduledDelayMs: 60_000, exportTimeoutMs: 180 },
          metricExportIntervalMs: 60_000,
          retry: { maxRetries: 10, initialDelayMs: 1, maxDelayMs: 1 },
        }),
        server,
      );
      pipeline.observer().startInference({ operation: "chat", provider: "p" }).end();
      pipeline.loggerProvider?.getLogger("app").emit({ body: "hello" });
      const started = performance.now();
      expect(await pipeline.forceFlush(2000)).toBe(false);
      expect(performance.now() - started).toBeLessThan(1000);
      expect(server.requests.length).toBeGreaterThan(1);
      expect(server.requests[0]?.disconnected).toBe(true);
      const stats = pipeline.stats()[signal];
      expect(stats).toEqual({
        accepted: stats.accepted,
        exported: 0,
        dropped: 0,
        failed: stats.accepted,
        pending: 0,
        bufferedBytes: 0,
      });
    });
    test(`${signal}: shutdown caps a longer export and never reports success on cancellation`, async () => {
      const server = await startServer(() => ({ hang: true }));
      const pipeline = track(
        createOtlpPipeline({
          [signal]: { endpoint: `${server.url}/v1/${signal}`, timeoutMs: 10_000 },
          batch: { scheduledDelayMs: 60_000, exportTimeoutMs: 10_000 },
          metricExportIntervalMs: 60_000,
        }),
        server,
      );
      pipeline.observer().startInference({ operation: "chat", provider: "p" }).end();
      pipeline.loggerProvider?.getLogger("app").emit({ body: "hello" });
      const started = performance.now();
      expect(await pipeline.shutdown(80)).toBe(false);
      expect(performance.now() - started).toBeLessThan(1000);
      expect(await pipeline.shutdown(2000)).toBe(false);
      expect(await pipeline.forceFlush(2000)).toBe(false);
      expect(pipeline.stats()[signal].pending).toBe(0);
      expect(pipeline.stats()[signal].failed).toBeGreaterThan(0);
      await server.waitFor(() => server.requests.every((request) => request.disconnected));
    });
  }
});

test("R-CNT-2: 503, 503, 200 retries identical bytes and counts one item", async () => {
  const server = await startServer((request) => ({ status: request.index < 2 ? 503 : 200 }));
  const pipeline = track(
    createOtlpPipeline({
      traces: { endpoint: `${server.url}/v1/traces` },
      batch: { scheduledDelayMs: 60_000 },
      retry: { initialDelayMs: 1, maxDelayMs: 1 },
    }),
    server,
  );
  span(pipeline);
  expect(await pipeline.forceFlush(2000)).toBe(true);
  expect(server.requests).toHaveLength(3);
  for (const request of server.requests)
    expect(request.body).toEqual(server.requests[0]?.body ?? Buffer.alloc(0));
  expect(pipeline.stats().traces).toEqual({
    accepted: 1,
    exported: 1,
    failed: 0,
    dropped: 0,
    pending: 0,
    bufferedBytes: 0,
  });
});

for (const protocol of ["http/json", "http/protobuf"] as const) {
  for (const signal of ["traces", "logs", "metrics"] as const) {
    test(`R-CNT-4: ${signal} ${protocol} partial success settles individual items without retry`, async () => {
      const key =
        signal === "traces"
          ? "rejectedSpans"
          : signal === "logs"
            ? "rejectedLogRecords"
            : "rejectedDataPoints";
      const body =
        protocol === "http/json"
          ? JSON.stringify({ partialSuccess: { [key]: "1" } })
          : new Uint8Array([0x0a, 0x02, 0x08, 0x01]);
      const server = await startServer(() => ({ status: 200, body }));
      const pipeline = track(
        createOtlpPipeline({
          [signal]: { endpoint: `${server.url}/v1/${signal}`, protocol },
          metricExportIntervalMs: 60_000,
          batch: { scheduledDelayMs: 60_000 },
        }),
        server,
      );
      span(pipeline, "one");
      span(pipeline, "two");
      pipeline.loggerProvider?.getLogger("app").emit({ body: "one" });
      pipeline.loggerProvider?.getLogger("app").emit({ body: "two" });
      pipeline.meterProvider?.getMeter("app").createCounter("one").add(1);
      pipeline.meterProvider?.getMeter("app").createCounter("two").add(1);
      expect(await pipeline.forceFlush(2000)).toBe(false);
      expect(server.requests).toHaveLength(1);
      expect(pipeline.stats()[signal]).toEqual({
        accepted: 2,
        exported: 1,
        failed: 1,
        dropped: 0,
        pending: 0,
        bufferedBytes: 0,
      });
    });
  }
}

test("A7: shutdown bounds final async metrics collection and never exports it later", async () => {
  const server = await startServer();
  const pipeline = track(
    createOtlpPipeline({
      metrics: { endpoint: `${server.url}/v1/metrics` },
      metricExportIntervalMs: 60_000,
      batch: { exportTimeoutMs: 10_000 },
    }),
    server,
  );
  let release = (): void => undefined;
  const wait = new Promise<void>((resolve) => {
    release = resolve;
  });
  const meter = pipeline.meterProvider?.getMeter("async");
  const counter = meter?.createObservableCounter("slow");
  if (counter === undefined) throw new Error("expected a meter");
  counter.addCallback(async (result) => {
    await wait;
    result.observe(1);
  });
  const started = performance.now();
  const shutdown = pipeline.shutdown(80);
  try {
    expect(await shutdown).toBe(false);
    expect(performance.now() - started).toBeLessThan(1000);
    expect(pipeline.stats().metrics.pending).toBe(0);
  } finally {
    release();
  }
  await pause(30);
  expect(server.requests).toHaveLength(0);
  expect(await pipeline.shutdown(2000)).toBe(false);
});

test("R-CANCEL-9: span end, log emit and metric record stay prompt with every endpoint hung", async () => {
  const server = await startServer(() => ({ hang: true }));
  const pipeline = track(
    createOtlpPipeline({
      traces: { endpoint: `${server.url}/v1/traces` },
      logs: { endpoint: `${server.url}/v1/logs` },
      metrics: { endpoint: `${server.url}/v1/metrics` },
      metricExportIntervalMs: 60_000,
      batch: { scheduledDelayMs: 60_000, maxQueueSize: 8 },
    }),
    server,
  );
  const observer = pipeline.observer();
  const logger = pipeline.loggerProvider?.getLogger("app");
  const metric = pipeline.meterProvider?.getMeter("app").createCounter("hot_path");
  const emit = (): void => {
    observer.startInference({ operation: "chat", provider: "p" }).end();
    logger?.emit({ body: "hello" });
    metric?.add(1);
  };
  emit();
  const flushing = pipeline.forceFlush(2000);
  await server.waitFor(() => server.requests.length === 3);
  const started = performance.now();
  for (let index = 0; index < 100; index += 1) emit();
  expect(performance.now() - started).toBeLessThan(1000);
  expect(pipeline.stats().traces.dropped).toBeGreaterThan(0);
  expect(pipeline.stats().logs.dropped).toBeGreaterThan(0);
  expect(await pipeline.shutdown(50)).toBe(false);
  expect(await flushing).toBe(false);
  for (const stats of Object.values(pipeline.stats())) expect(stats.pending).toBe(0);
});

const SIGNALS = ["traces", "logs", "metrics"] as const;
type SignalName = (typeof SIGNALS)[number];

/** A pipeline exporting only `signal`; exports start from `forceFlush` alone. */
const onlySignal = (
  server: TestServer,
  signal: SignalName,
  retry: { maxRetries?: number; initialDelayMs: number; maxDelayMs: number },
): OtlpPipeline =>
  track(
    createOtlpPipeline({
      [signal]: { endpoint: `${server.url}/v1/${signal}` },
      batch: { scheduledDelayMs: 60_000, exportTimeoutMs: 10_000 },
      metricExportIntervalMs: 60_000,
      retry,
    }),
    server,
  );

/** Two items of `signal`: spans, log records or metric data points. */
const emitTwo = (pipeline: OtlpPipeline, signal: SignalName): void => {
  if (signal === "traces") {
    span(pipeline, "one");
    span(pipeline, "two");
  } else if (signal === "logs") {
    pipeline.loggerProvider?.getLogger("app").emit({ body: "one" });
    pipeline.loggerProvider?.getLogger("app").emit({ body: "two" });
  } else {
    pipeline.meterProvider?.getMeter("app").createCounter("one").add(1);
    pipeline.meterProvider?.getMeter("app").createCounter("two").add(1);
  }
};

/** Every request re-sent the first one's bytes. */
const expectIdenticalResends = (server: TestServer): void => {
  for (const request of server.requests)
    expect(request.body).toEqual(server.requests[0]?.body ?? Buffer.alloc(0));
};

const FAST = { maxRetries: 2, initialDelayMs: 1, maxDelayMs: 1 };
/** Hang guard only: every export below ends by its 10 s deadline. */
const HANG_GUARD_MS = 30_000;

describe("D2: HTTP failure statuses, per signal", () => {
  const cases = [
    {
      // A backoff of 60 s cannot fit the 10 s deadline, so only an honoured header retries.
      name: "429 with Retry-After: 0, then 200, is retried at once",
      reply: (index: number) =>
        index === 0 ? { status: 429, headers: { "retry-after": "0" } } : { status: 200 },
      retry: { maxRetries: 2, initialDelayMs: 60_000, maxDelayMs: 60_000 },
      requests: 2,
      exported: 2,
    },
    {
      name: "429 without Retry-After is retried until the retry budget runs out",
      reply: () => ({ status: 429 }),
      retry: FAST,
      requests: FAST.maxRetries + 1,
      exported: 0,
    },
    {
      name: "429 with a Retry-After past the deadline fails without a retry",
      reply: () => ({ status: 429, headers: { "retry-after": "30" } }),
      retry: FAST,
      requests: 1,
      exported: 0,
    },
    {
      name: "400 is not retried",
      reply: () => ({ status: 400 }),
      retry: FAST,
      requests: 1,
      exported: 0,
    },
    {
      name: "500 is not retried",
      reply: () => ({ status: 500 }),
      retry: FAST,
      requests: 1,
      exported: 0,
    },
  ];
  for (const signal of SIGNALS) {
    for (const { name, reply, retry, requests, exported } of cases) {
      test(
        `${signal}: ${name}`,
        async () => {
          const server = await startServer((request) => reply(request.index));
          const pipeline = onlySignal(server, signal, retry);
          emitTwo(pipeline, signal);
          expect(await pipeline.forceFlush(20_000)).toBe(exported === 2);
          expect(server.requests).toHaveLength(requests);
          expect(server.received(`/v1/${signal}`)).toHaveLength(requests);
          expectIdenticalResends(server);
          expect(pipeline.stats()[signal]).toEqual({
            accepted: 2,
            exported,
            dropped: 0,
            failed: 2 - exported,
            pending: 0,
            bufferedBytes: 0,
          });
        },
        HANG_GUARD_MS,
      );
    }
  }
});

describe("D3: lost acknowledgements for logs and metrics", () => {
  for (const signal of ["logs", "metrics"] as const) {
    test(
      `${signal}: a lost ack then 200 resends the same bytes and counts the items once`,
      async () => {
        const server = await startServer((request) =>
          request.index === 0 ? { destroy: true } : { status: 200 },
        );
        const pipeline = onlySignal(server, signal, FAST);
        emitTwo(pipeline, signal);
        expect(await pipeline.forceFlush(20_000)).toBe(true);
        expect(server.requests).toHaveLength(2);
        expectIdenticalResends(server);
        expect(pipeline.stats()[signal]).toEqual({
          accepted: 2,
          exported: 2,
          dropped: 0,
          failed: 0,
          pending: 0,
          bufferedBytes: 0,
        });
      },
      HANG_GUARD_MS,
    );

    test(
      `${signal}: acks lost on every attempt fail the items after maxRetries + 1 requests`,
      async () => {
        const server = await startServer(() => ({ destroy: true }));
        const pipeline = onlySignal(server, signal, FAST);
        emitTwo(pipeline, signal);
        expect(await pipeline.forceFlush(20_000)).toBe(false);
        expect(server.requests).toHaveLength(FAST.maxRetries + 1);
        expectIdenticalResends(server);
        expect(pipeline.stats()[signal]).toEqual({
          accepted: 2,
          exported: 0,
          dropped: 0,
          failed: 2,
          pending: 0,
          bufferedBytes: 0,
        });
      },
      HANG_GUARD_MS,
    );
  }
});
