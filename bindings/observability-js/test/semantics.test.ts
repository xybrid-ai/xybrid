// Contract v2 semantic regressions that the shared fixtures cannot express (contract-decisions
// R-CTX-2/3, R-CLOCK-1, R-TIME-1, R-NUM-1, R-STREAM-1, R-LOG-1/2, R-CFG-7; core-2, core-3,
// critic-1, critic-2, context-catalog-7).
import { heapSize } from "bun:jsc";
import { afterAll, afterEach, beforeAll, describe, expect, spyOn, test } from "bun:test";
import * as api from "@opentelemetry/api";
import { context, ProxyTracerProvider, trace } from "@opentelemetry/api";
import { logs as legacyLogs } from "@opentelemetry/api-logs";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  JsonLogsSerializer,
  JsonMetricsSerializer,
  JsonTraceSerializer,
  ProtobufLogsSerializer,
  ProtobufMetricsSerializer,
  ProtobufTraceSerializer,
} from "@opentelemetry/otlp-transformer";
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
import {
  ENUMS,
  type LoggerProviderLike,
  ObservationContext,
  Observer,
  resolveContentCapture,
  serializeContext,
} from "../src/index.ts";
import { CLOCK_START_MS, createHarness, normalizeMetrics } from "./support/harness.ts";

const request = {
  operation: "chat",
  provider: "llama.cpp",
  model: "tiny",
  location: "local",
} as const;
const TRACE = "4bf92f3577b34da6a3ce929d0e0e4736";
const SPAN = "00f067aa0ba902b7";

const metricNames = async (collect: () => Promise<Parameters<typeof normalizeMetrics>[0]>) =>
  normalizeMetrics(await collect()).map((metric) => metric.name);

describe("references (D2, D3)", () => {
  test("an unsampled span keeps its reference; suppression has none", async () => {
    const { observer, spans } = createHarness();
    const op = observer.startInference(request, { parent: `00-${TRACE}-${SPAN}-00` });
    const ref = op.observationRef();
    expect(ref?.trace_id).toBe(TRACE);
    expect(ref?.span_id).not.toBe(SPAN);
    expect(ref?.span_id).toBe(op.context()?.spanId ?? "");
    expect(op.context()?.sampled).toBe(false);
    const attempt = op.startAttempt({
      index: 0,
      reason: "initial",
      provider: "p",
      location: "local",
    });
    expect(attempt.observationRef()?.trace_id).toBe(TRACE);
    attempt.end();
    op.end();
    expect(spans.getFinishedSpans()).toHaveLength(0);
    const suppressed = observer.startInference(request, { trace: "suppress" });
    expect(suppressed.observationRef()).toBeUndefined();
    expect(suppressed.context()).toBeUndefined();
    suppressed.end();
  });

  test("an API-only tracer passes the parent through without a reference", () => {
    const observer = new Observer({
      tracerProvider: new ProxyTracerProvider(),
      meterProvider: null,
      loggerProvider: null,
    });
    const child = observer.startInference(request, { parent: `00-${TRACE}-${SPAN}-01` });
    expect(child.observationRef()).toBeUndefined();
    expect(child.context()?.toTraceparent()).toBe(`00-${TRACE}-${SPAN}-01`);
    // Attempts of an operation without a span of its own are suppressed.
    const attempt = child.startAttempt({
      index: 0,
      reason: "initial",
      provider: "p",
      location: "local",
    });
    expect(attempt.context()).toBeUndefined();
    child.end();
    const root = observer.startInference(request);
    expect(root.context()).toBeUndefined();
    expect(root.observationRef()).toBeUndefined();
  });

  test("no tracer provider: no context or reference, even with an explicit parent", () => {
    const { observer } = createHarness({ tracer: "none" });
    const op = observer.startInference(request, { parent: `00-${TRACE}-${SPAN}-01` });
    expect(op.context()).toBeUndefined();
    expect(op.observationRef()).toBeUndefined();
  });
});

describe("explicit parents (D6, R-CTX-2)", () => {
  test("every accepted form gives the same parent; null and undefined use the ambient context", () => {
    const { observer, spans } = createHarness();
    const parent = observer.startWorkflow("wf", { parent: `00-${TRACE}-${SPAN}-01` });
    const ctx = parent.context();
    if (ctx === undefined) throw new Error("expected a context");
    class OtherCopy {
      readonly traceId = ctx?.traceId;
      readonly spanId = ctx?.spanId;
      readonly traceFlags = ctx?.traceFlags;
      readonly tracestate = undefined;
      readonly isRemote = false;
    }
    for (const form of [ctx, structuredClone(ctx), serializeContext(ctx), new OtherCopy()]) {
      observer.startTool({ name: "t" }, { parent: form as never }).end();
    }
    for (const parentSpan of spans.getFinishedSpans().map((span) => span.parentSpanContext)) {
      expect(parentSpan?.spanId).toBe(ctx.spanId);
    }
    for (const ambient of [null, undefined]) {
      observer.startTool({ name: "root" }, { parent: ambient }).end();
    }
    const roots = spans.getFinishedSpans().slice(4);
    expect(roots.map((span) => span.parentSpanContext)).toEqual([undefined, undefined]);
    parent.end();
  });

  test("invalid parents throw TypeError before any span, metric or content work", async () => {
    const redacted: string[] = [];
    const { observer, spans, collectMetrics } = createHarness({
      content: {
        redact: (kind, value) => {
          redacted.push(kind);
          return value;
        },
      },
    });
    const secret = "00-0AF7651916CD43DD8448EB211C80319C-b7ad6b7169203331-01";
    for (const parent of [
      secret,
      "garbage",
      `00-${TRACE}-${"0".repeat(16)}-01`,
      { traceparent: "nope" },
      { traceId: TRACE, spanId: SPAN },
      { traceId: TRACE, spanId: SPAN, traceFlags: Number.NaN },
      42,
      true,
      [`00-${TRACE}-${SPAN}-01`],
    ]) {
      const start = () =>
        observer.startInference(
          { ...request, prompt: { name: "p", variables: { v: "private" } } },
          { parent: parent as never },
        );
      expect(start).toThrow(TypeError);
      try {
        start();
      } catch (error) {
        expect(String((error as Error).message)).not.toContain("0AF7651916CD43DD");
      }
      expect(() => observer.startTool({ name: "t" }, { parent: parent as never })).toThrow(
        TypeError,
      );
      expect(() => observer.startWorkflow("w", { parent: parent as never })).toThrow(TypeError);
      expect(() => observer.startModelLoad({ model: "m" }, { parent: parent as never })).toThrow(
        TypeError,
      );
      expect(() => observer.startSpan("s", {}, { parent: parent as never })).toThrow(TypeError);
      // Suppression does not skip validation.
      expect(() =>
        observer.startInference(request, { parent: parent as never, trace: "suppress" }),
      ).toThrow(TypeError);
    }
    expect(redacted).toEqual([]);
    expect(spans.getFinishedSpans()).toHaveLength(0);
    expect(await metricNames(collectMetrics)).toEqual([]);
  });
});

describe("events of span-less operations (D7, R-CTX-3)", () => {
  const manager = new AsyncLocalStorageContextManager();
  beforeAll(() => context.setGlobalContextManager(manager.enable()));
  afterAll(() => {
    manager.disable();
    context.disable();
  });

  test("exception and details events never attach to the ambient span", () => {
    const { observer, logs, tracerProvider } = createHarness({
      content: { capture: "event_only" },
    });
    const ambient = tracerProvider.getTracer("app").startSpan("request");
    context.with(trace.setSpan(context.active(), ambient), () => {
      const op = observer.startInference(request, { trace: "suppress" });
      op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "hi" }] }]);
      op.endWithError({ errorType: "boom", message: "failed" });
      const tool = observer.startTool({ name: "t", arguments: { q: 1 } }, { trace: "suppress" });
      tool.end();
    });
    ambient.end();
    const records = logs.getFinishedLogRecords();
    expect(records.map((record) => record.eventName)).toEqual([
      "gen_ai.client.operation.exception",
      "gen_ai.client.inference.operation.details",
      "gen_ai.execute_tool.operation.details",
    ]);
    for (const record of records) expect(record.spanContext).toBeUndefined();
  });

  test("events of an operation with a span carry that span, even unsampled", () => {
    const { observer, logs } = createHarness();
    const op = observer.startInference(request, { parent: `00-${TRACE}-${SPAN}-00` });
    op.endWithError({ errorType: "boom" });
    const [record] = logs.getFinishedLogRecords();
    expect(record?.spanContext?.spanId).toBe(op.observationRef()?.span_id ?? "");
  });
});

describe("clocks and timing (D8)", () => {
  afterEach(() => {
    // Restore any stubbed performance clock.
    for (const mock of mocks.splice(0)) mock.mockRestore();
  });
  const mocks: { mockRestore(): void }[] = [];

  test("R-CLOCK-1: a lagging performance clock does not move spans off the wall clock", async () => {
    const realNow = performance.now.bind(performance);
    mocks.push(spyOn(performance, "now").mockImplementation(() => realNow() - 60_000));
    const spans = new InMemorySpanExporter();
    const tracerProvider = new TracerProvider({
      spanProcessors: [new SimpleSpanProcessor({ exporter: spans })],
    });
    const observer = new Observer({ tracerProvider, meterProvider: null, loggerProvider: null });
    const before = Date.now();
    const op = observer.startInference(request);
    await new Promise((resolve) => setTimeout(resolve, 20));
    op.recordChunk();
    op.end();
    const [span] = spans.getFinishedSpans();
    const startMs = (span?.startTime[0] ?? 0) * 1000 + (span?.startTime[1] ?? 0) / 1e6;
    const endMs = (span?.endTime[0] ?? 0) * 1000 + (span?.endTime[1] ?? 0) / 1e6;
    expect(Math.abs(startMs - before)).toBeLessThan(1000);
    expect(endMs - startMs).toBeGreaterThanOrEqual(15);
    expect(span?.attributes["gen_ai.response.time_to_first_chunk"] as number).toBeGreaterThan(
      0.015,
    );
  });

  test("R-TIME-1: non-finite explicit times are ignored and every encoder serializes", async () => {
    const { observer, spans, logs, clock, collectMetrics } = createHarness();
    const op = observer.startInference(request, { startTimeMs: Number.NaN });
    clock.advance(40);
    op.recordFirstToken(Number.POSITIVE_INFINITY);
    op.recordChunk(Number.NaN);
    clock.advance(10);
    op.endWithError({ errorType: "x" }, Number.NEGATIVE_INFINITY);
    const [span] = spans.getFinishedSpans();
    expect(span?.startTime).toEqual([CLOCK_START_MS / 1000, 0]);
    expect(span?.attributes["xybrid.genai.response.time_to_first_token"]).toBeCloseTo(0.04, 9);
    expect(span?.attributes["gen_ai.response.time_to_first_chunk"]).toBeCloseTo(0.04, 9);
    const finished = spans.getFinishedSpans();
    const records = logs.getFinishedLogRecords();
    const resourceMetrics = await collectMetrics();
    for (const serializer of [JsonTraceSerializer, ProtobufTraceSerializer]) {
      expect(() => serializer.serializeRequest(finished)).not.toThrow();
    }
    for (const serializer of [JsonLogsSerializer, ProtobufLogsSerializer]) {
      expect(() => serializer.serializeRequest(records)).not.toThrow();
    }
    for (const serializer of [JsonMetricsSerializer, ProtobufMetricsSerializer]) {
      expect(() => serializer.serializeRequest(resourceMetrics)).not.toThrow();
    }
  });

  test("core-3: an end before the start clamps to zero and keeps the metric point", async () => {
    const { observer, spans, collectMetrics } = createHarness();
    const op = observer.startTool({ name: "t" }, { startTimeMs: CLOCK_START_MS + 500 });
    op.end(CLOCK_START_MS + 100);
    const [span] = spans.getFinishedSpans();
    expect(span?.endTime).toEqual(span?.startTime);
    const [metric] = normalizeMetrics(await collectMetrics());
    expect(metric?.points).toEqual([
      {
        attributes: { "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "t" },
        count: 1,
        sum: 0,
      },
    ]);
  });
});

describe("streaming (D9, R-STREAM-1)", () => {
  test("chunk points are visible before end and many chunks keep constant state", async () => {
    const { observer, spans, clock, collectMetrics } = createHarness();
    const op = observer.startInference({ ...request, stream: true });
    clock.advance(10);
    op.recordFirstToken();
    const chunks = 100_000;
    for (let index = 0; index < chunks; index += 1) {
      clock.advance(1);
      op.recordChunk();
    }
    const before = normalizeMetrics(await collectMetrics());
    const point = (name: string) =>
      before.find((metric) => metric.name === name)?.points[0] as { count: number; sum: number };
    expect(point("gen_ai.client.inference.time_to_first_chunk").count).toBe(1);
    expect(point("gen_ai.client.inference.time_per_output_chunk").count).toBe(chunks - 1);
    expect(point("gen_ai.client.inference.time_per_output_chunk").sum).toBeCloseTo(
      (chunks - 1) / 1000,
      6,
    );
    expect(point("xybrid.genai.client.inference.time_to_first_token").count).toBe(1);
    expect(before.map((metric) => metric.name)).not.toContain("gen_ai.client.inference.duration");
    op.end();
    expect(spans.getFinishedSpans()[0]?.attributes["xybrid.genai.response.chunk_count"]).toBe(
      chunks,
    );
  });

  test("streaming state does not grow with the number of chunks", () => {
    const { observer } = createHarness({ metrics: false, traces: false });
    const op = observer.startInference({ ...request, stream: true });
    op.recordChunk();
    Bun.gc(true);
    const baseline = heapSize();
    for (let index = 0; index < 400_000; index += 1) op.recordChunk();
    Bun.gc(true);
    // A per-chunk timestamp array would hold 400 000 doubles (about 3.5 MB of JS heap).
    expect(heapSize() - baseline).toBeLessThan(1_000_000);
    op.end();
  });

  test("chunks never set first-token time", () => {
    const { observer, spans } = createHarness();
    const op = observer.startInference({ ...request, stream: true });
    op.recordChunk();
    op.recordChunk();
    op.end();
    const attributes = spans.getFinishedSpans()[0]?.attributes ?? {};
    expect(attributes["xybrid.genai.response.time_to_first_token"]).toBeUndefined();
    expect(attributes["xybrid.genai.response.chunk_count"]).toBe(2);
  });
});

describe("numeric inputs (D10, R-NUM-1)", () => {
  test("invalid counts and measurements are ignored on spans and metrics alike", async () => {
    const { observer, spans, collectMetrics } = createHarness();
    const op = observer.startInference({
      ...request,
      maxTokens: 1.5,
      topK: Number.NaN,
      seed: -3,
      temperature: Number.POSITIVE_INFINITY,
      topP: 1,
      serverPort: 443.5,
      stopSequences: [],
      inputModalities: [],
    });
    op.recordUsage({
      inputTokens: -1,
      outputTokens: 2.5,
      cacheReadInputTokens: Number.NaN,
      reasoningOutputTokens: "7" as never,
      textInputTokens: 3,
    });
    op.recordUsage({ inputTokens: 5 });
    op.recordResource({
      cpuUtilization: Number.NaN,
      cpuPeakUtilization: Number.POSITIVE_INFINITY,
      memoryUsedBytes: -10,
      memoryMinAvailableBytes: 1.5,
      memoryPressure: "high" as never,
      processRssBytes: 1024,
      batteryLevel: Number.POSITIVE_INFINITY,
    });
    op.recordTokenTiming({ timePerOutputTokenS: Number.NaN });
    op.recordFinishReasons([]);
    op.end();
    const attributes = spans.getFinishedSpans()[0]?.attributes ?? {};
    expect(attributes["gen_ai.request.max_tokens"]).toBeUndefined();
    expect(attributes["gen_ai.request.top_k"]).toBeUndefined();
    expect(attributes["gen_ai.request.seed"]).toBe(-3);
    expect(attributes["gen_ai.request.temperature"]).toBeUndefined();
    expect(attributes["gen_ai.request.top_p"]).toBe(1);
    expect(attributes["server.port"]).toBeUndefined();
    expect(attributes["gen_ai.request.stop_sequences"]).toBeUndefined();
    expect(attributes["xybrid.genai.input.modalities"]).toBeUndefined();
    expect(attributes["gen_ai.usage.input_tokens"]).toBe(5);
    expect(attributes["gen_ai.usage.output_tokens"]).toBeUndefined();
    expect(attributes["gen_ai.usage.cache_read.input_tokens"]).toBeUndefined();
    expect(attributes["gen_ai.usage.reasoning.output_tokens"]).toBeUndefined();
    expect(attributes["xybrid.genai.resource.cpu.utilization"]).toBeUndefined();
    expect(attributes["xybrid.genai.resource.cpu.peak_utilization"]).toBeUndefined();
    expect(attributes["xybrid.genai.resource.memory.used_bytes"]).toBeUndefined();
    expect(attributes["xybrid.genai.resource.memory.min_available_bytes"]).toBeUndefined();
    expect(attributes["xybrid.genai.resource.memory.pressure"]).toBeUndefined();
    expect(attributes["xybrid.genai.resource.process.rss_bytes"]).toBe(1024);
    expect(attributes["xybrid.genai.resource.battery.level"]).toBeUndefined();
    expect(attributes["xybrid.genai.response.time_per_output_token"]).toBeUndefined();
    expect(attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
    const metrics = normalizeMetrics(await collectMetrics());
    const names = metrics.map((metric) => metric.name);
    expect(names).not.toContain("gen_ai.client.inference.operation.output_tokens");
    expect(names).not.toContain("gen_ai.client.inference.usage.reasoning.output_tokens");
    expect(
      metrics.find((metric) => metric.name === "gen_ai.client.inference.operation.input_tokens")
        ?.points,
    ).toEqual([
      {
        attributes: {
          "gen_ai.operation.name": "chat",
          "gen_ai.provider.name": "llama.cpp",
          "gen_ai.request.model": "tiny",
          "xybrid.genai.execution.location": "local",
        },
        count: 1,
        sum: 5,
      },
    ]);
  });

  test("memory pressure accepts exactly the catalog levels; zero available memory is kept", () => {
    const { observer, spans } = createHarness();
    for (const level of ENUMS["xybrid.genai.resource.memory.pressure"]) {
      const load = observer.startModelLoad({ model: level });
      load.recordResource({ memoryPressure: level, memoryMinAvailableBytes: 0 });
      load.end();
    }
    const finished = spans.getFinishedSpans();
    expect(
      finished.map((span) => span.attributes["xybrid.genai.resource.memory.pressure"]),
    ).toEqual(["normal", "warn", "critical"]);
    for (const span of finished) {
      expect(span.attributes["xybrid.genai.resource.memory.min_available_bytes"]).toBe(0);
    }
  });
});

describe("operation names and metric dimensions (D11, D12)", () => {
  test("names are literal and custom operations use inference metrics", async () => {
    const { observer, spans, collectMetrics } = createHarness();
    observer.startInference({ ...request, model: "$&-$1" }).end();
    observer.startInference({ ...request, operation: "load_model", model: "m" }).end();
    expect(spans.getFinishedSpans().map((span) => span.name)).toEqual([
      "chat $&-$1",
      "load_model m",
    ]);
    expect(await metricNames(collectMetrics)).toEqual(["gen_ai.client.inference.duration"]);
  });

  test("extension attributes and setAttribute never change dimensions", async () => {
    const { observer, spans, collectMetrics } = createHarness();
    const op = observer.startInference(request, {
      attributes: { "server.address": "injected", "gen_ai.response.model": "fake" },
    });
    op.setAttribute("gen_ai.request.model", "renamed");
    op.setAttribute("server.port", 1);
    op.recordResponse({ model: "served" });
    op.end();
    expect(spans.getFinishedSpans()[0]?.attributes["server.address"]).toBe("injected");
    const [duration] = normalizeMetrics(await collectMetrics());
    expect(duration?.points[0]?.attributes).toEqual({
      "gen_ai.operation.name": "chat",
      "gen_ai.provider.name": "llama.cpp",
      "gen_ai.request.model": "tiny",
      "gen_ai.response.model": "served",
      "xybrid.genai.execution.location": "local",
    });
  });
});

describe("evaluation and feedback (D23, R-LOG-1, R-LOG-2)", () => {
  afterEach(() => {
    api.logs.disable();
    legacyLogs.disable();
  });

  const sdkLogger = () => {
    const exporter = new InMemoryLogRecordExporter();
    const provider = new LoggerProvider({
      processors: [new SimpleLogRecordProcessor({ exporter })],
    });
    return { exporter, provider };
  };
  const target = { trace_id: TRACE, span_id: SPAN };

  test("false without an SDK logger; true after late registration through api 1.10", () => {
    const observer = new Observer({ tracerProvider: null, meterProvider: null });
    expect(observer.recordEvaluation(target, { name: "relevance", scoreValue: 1 })).toBe(false);
    expect(observer.recordFeedback(target, { score: 1 })).toBe(false);
    const { exporter, provider } = sdkLogger();
    expect(api.logs.setGlobalLoggerProvider(provider)).toBe(true);
    expect(observer.recordEvaluation(target, { name: "relevance", scoreValue: 1 })).toBe(true);
    expect(exporter.getFinishedLogRecords()).toHaveLength(1);
  });

  test("a provider registered only through @opentelemetry/api-logs is used", () => {
    const observer = new Observer({ tracerProvider: null, meterProvider: null });
    const { exporter, provider } = sdkLogger();
    legacyLogs.setGlobalLoggerProvider(provider as never);
    expect(observer.recordFeedback(target, { label: "thumbs_up" })).toBe(true);
    expect(exporter.getFinishedLogRecords()[0]?.attributes["gen_ai.evaluation.name"]).toBe(
      "user_feedback",
    );
  });

  test("a disabled logger, a shut-down provider or an invalid reference record nothing", async () => {
    const emitted: unknown[] = [];
    const disabled: LoggerProviderLike = {
      getLogger: () => ({ emit: (record) => emitted.push(record), enabled: () => false }),
    };
    const observer = new Observer({ loggerProvider: disabled, tracerProvider: null });
    expect(observer.recordEvaluation(target, { name: "x" })).toBe(false);
    expect(emitted).toEqual([]);
    const { provider } = sdkLogger();
    const sdk = new Observer({
      loggerProvider: provider as LoggerProviderLike,
      tracerProvider: null,
    });
    expect(sdk.recordEvaluation({ traceId: TRACE, spanId: SPAN } as never, { name: "x" })).toBe(
      false,
    );
    await provider.shutdown();
    expect(sdk.recordEvaluation(target, { name: "x" })).toBe(false);
  });

  test("records carry the reference's ids with trace flags 0", () => {
    const { observer, logs } = createHarness();
    const op = observer.startInference(request);
    op.end();
    const ref = op.observationRef();
    if (ref === undefined) throw new Error("expected a reference");
    expect(observer.recordEvaluation(ref, { name: "relevance", scoreValue: 1 })).toBe(true);
    expect(observer.recordFeedback(target, { score: 0 })).toBe(true);
    const [evaluation, feedback] = logs.getFinishedLogRecords();
    expect(evaluation?.spanContext).toMatchObject({
      traceId: ref.trace_id,
      spanId: ref.span_id,
      traceFlags: 0,
    });
    expect(feedback?.spanContext).toMatchObject({ traceId: TRACE, spanId: SPAN, traceFlags: 0 });
  });
});

describe("content capture resolution (D16, R-CFG-7)", () => {
  const CURRENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT";
  const LEGACY = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";
  const withEnv = (values: Record<string, string | undefined>, check: () => void): void => {
    const saved = { [CURRENT]: process.env[CURRENT], [LEGACY]: process.env[LEGACY] };
    try {
      for (const [name, value] of Object.entries(values)) {
        if (value === undefined) delete process.env[name];
        else process.env[name] = value;
      }
      check();
    } finally {
      for (const [name, value] of Object.entries(saved)) {
        if (value === undefined) delete process.env[name];
        else process.env[name] = value;
      }
    }
  };

  test("environment and legacy variable", () => {
    withEnv({ [CURRENT]: undefined, [LEGACY]: undefined }, () =>
      expect(resolveContentCapture()).toBe("span_only"),
    );
    withEnv({ [CURRENT]: " Event_Only ", [LEGACY]: "true" }, () =>
      expect(resolveContentCapture()).toBe("event_only"),
    );
    withEnv({ [CURRENT]: "maybe", [LEGACY]: "true" }, () =>
      expect(resolveContentCapture()).toBe("no_content"),
    );
    withEnv({ [CURRENT]: "", [LEGACY]: "TRUE" }, () =>
      expect(resolveContentCapture()).toBe("span_only"),
    );
    withEnv({ [CURRENT]: undefined, [LEGACY]: "false" }, () =>
      expect(resolveContentCapture()).toBe("no_content"),
    );
    withEnv({ [CURRENT]: undefined, [LEGACY]: "span_and_event" }, () =>
      expect(resolveContentCapture()).toBe("span_and_event"),
    );
    withEnv({ [CURRENT]: undefined, [LEGACY]: "1" }, () =>
      expect(resolveContentCapture()).toBe("no_content"),
    );
  });

  test("explicit values are trimmed and case-insensitive; anything else throws RangeError", () => {
    withEnv({ [CURRENT]: "no_content" }, () => {
      expect(resolveContentCapture("SPAN_AND_EVENT")).toBe("span_and_event");
      expect(new Observer({ content: { capture: " NO_CONTENT " as never } }).contentCapture).toBe(
        "no_content",
      );
      for (const invalid of ["verbatim", "", "true", 1, null]) {
        expect(() => resolveContentCapture(invalid as never)).toThrow(RangeError);
        expect(() => new Observer({ content: { capture: invalid as never } })).toThrow(RangeError);
      }
    });
  });

  test("an unreadable environment fails closed", () => {
    const saved = process.env;
    try {
      (process as { env: unknown }).env = new Proxy(
        {},
        {
          get() {
            throw new Error("denied");
          },
        },
      );
      expect(resolveContentCapture()).toBe("no_content");
      expect(resolveContentCapture("span_only")).toBe("span_only");
    } finally {
      (process as { env: unknown }).env = saved;
    }
  });
});

test("ObservationContext reads back masked flags from OpenTelemetry contexts", () => {
  const ctx = ObservationContext.fromOtelContext(
    trace.setSpanContext(api.ROOT_CONTEXT, { traceId: TRACE, spanId: SPAN, traceFlags: 0x03 }),
  );
  expect(ctx?.toTraceparent()).toBe(`00-${TRACE}-${SPAN}-01`);
});
