import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import type { SpanProcessor } from "@opentelemetry/sdk-trace";
import {
  type JsonValue,
  type LoggerProviderLike,
  ObservationContext,
  Observer,
  resolveContentCapture,
} from "../src/index.ts";
import { createHarness, normalizeMetrics } from "./support/harness.ts";

const request = {
  operation: "chat",
  provider: "llama.cpp",
  model: "tiny",
  location: "local",
} as const;

describe("parenting", () => {
  // The application, not the package, owns the context manager.
  const manager = new AsyncLocalStorageContextManager();
  beforeAll(() => context.setGlobalContextManager(manager.enable()));
  afterAll(() => {
    manager.disable();
    context.disable();
  });

  test("the ambient context parents a new operation", () => {
    const { observer, spans, tracerProvider } = createHarness();
    const outer = tracerProvider.getTracer("app").startSpan("request");
    context.with(trace.setSpan(context.active(), outer), () => {
      observer.startInference(request).end();
    });
    outer.end();
    const [inference] = spans.getFinishedSpans();
    expect(inference?.parentSpanContext?.spanId).toBe(outer.spanContext().spanId);
  });

  test("bound callbacks retain their captured parent and restore the invoking context", async () => {
    const { observer, spans, tracerProvider } = createHarness();
    const tracer = tracerProvider.getTracer("app");
    const parent = tracer.startSpan("captured");
    const other = tracer.startSpan("other");
    const bound = context.with(trace.setSpan(context.active(), parent), () =>
      observer.bindContext((model: string) => {
        observer.startModelLoad({ model }).end();
        return model.length;
      }),
    );
    await Promise.resolve();
    context.with(trace.setSpan(context.active(), other), () => {
      expect(bound("tiny")).toBe(4);
      expect(trace.getSpan(context.active())).toBe(other);
    });
    expect(spans.getFinishedSpans()[0]?.parentSpanContext?.spanId).toBe(
      parent.spanContext().spanId,
    );
    const throwing = context.with(trace.setSpan(context.active(), parent), () =>
      observer.bindContext(() => {
        throw new Error("callback failure");
      }),
    );
    expect(() => throwing()).toThrow("callback failure");
    expect(trace.getSpan(context.active())).toBeUndefined();
    parent.end();
    other.end();
  });

  test("an explicit parent wins over the ambient context", () => {
    const { observer, spans, tracerProvider } = createHarness();
    const outer = tracerProvider.getTracer("app").startSpan("ambient");
    const parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    context.with(trace.setSpan(context.active(), outer), () => {
      observer.startInference(request, { parent, tracestate: "vendor=abc" }).end();
    });
    const [inference] = spans.getFinishedSpans();
    expect(inference?.spanContext().traceId).toBe("4bf92f3577b34da6a3ce929d0e0e4736");
    expect(inference?.parentSpanContext?.spanId).toBe("00f067aa0ba902b7");
    expect(inference?.spanContext().traceState?.get("vendor")).toBe("abc");
  });

  test("interleaved async flows each keep their own ambient parent", async () => {
    const { observer, spans, tracerProvider, collectMetrics } = createHarness();
    const flows = 10;
    const parents = Array.from({ length: flows }, (_, flow) =>
      tracerProvider.getTracer("app").startSpan(`request ${flow}`),
    );
    // The test resumes each flow at each step, in an order it chooses; nothing is timed.
    const deferred = () => {
      let resolve = (): void => undefined;
      const promise = new Promise<void>((settle) => {
        resolve = settle;
      });
      return { promise, resolve };
    };
    const steps = parents.map(() => ({
      start: deferred(),
      started: deferred(),
      end: deferred(),
      ended: deferred(),
    }));
    const running = parents.map((parent, flow) =>
      context.with(trace.setSpan(context.active(), parent), async () => {
        const step = steps[flow];
        if (step === undefined) throw new Error("no step for the flow");
        await step.start.promise;
        const op = observer.startInference(request);
        op.setAttribute("flow", flow);
        step.started.resolve();
        await step.end.promise;
        op.end();
        step.ended.resolve();
      }),
    );
    const resume = async (order: readonly number[], phase: "start" | "end"): Promise<void> => {
      for (const flow of order) {
        const step = steps[flow];
        if (step === undefined) throw new Error("no step for the flow");
        step[phase].resolve();
        // A flow that throws instead of reaching its step fails the test here.
        await Promise.race([step[phase === "start" ? "started" : "ended"].promise, running[flow]]);
        // The flow's context never leaks into the code that resumed it.
        expect(trace.getSpan(context.active())).toBeUndefined();
      }
    };
    const reverse = [...parents.keys()].reverse();
    const evensThenOdds = [...parents.keys()].sort((a, b) => (a % 2) - (b % 2) || a - b);
    // Every inference starts in reverse order and is open before any of them ends.
    await resume(reverse, "start");
    await resume(evensThenOdds, "end");
    await Promise.all(running);
    for (const parent of parents) parent.end();

    const inferences = spans.getFinishedSpans().filter((span) => span.name === "chat tiny");
    expect(inferences.map((span) => span.attributes["flow"])).toEqual(evensThenOdds);
    for (const span of inferences) {
      const parent = parents[Number(span.attributes["flow"])]?.spanContext();
      expect(span.parentSpanContext?.spanId).toBe(parent?.spanId ?? "");
      expect(span.spanContext().traceId).toBe(parent?.traceId ?? "");
    }
    const duration = normalizeMetrics(await collectMetrics()).find(
      (metric) => metric.name === "gen_ai.client.inference.duration",
    );
    expect(duration?.points.map((point) => ("count" in point ? point.count : 0))).toEqual([flows]);
  });

  test("operation contexts parent children explicitly", () => {
    const { observer, spans } = createHarness();
    const workflow = observer.startWorkflow("agent");
    const tool = observer.startTool({ name: "search" }, { parent: workflow.context() });
    tool.end();
    workflow.end();
    const [toolSpan, workflowSpan] = spans.getFinishedSpans();
    expect(toolSpan?.parentSpanContext?.spanId).toBe(workflowSpan?.spanContext().spanId);
    expect(workflow.observationRef()).toEqual({
      trace_id: workflowSpan?.spanContext().traceId ?? "",
      span_id: workflowSpan?.spanContext().spanId ?? "",
    });
  });
});

describe("lifecycle", () => {
  test("suppressed operations create no span but keep metrics", async () => {
    const { observer, spans, collectMetrics } = createHarness();
    const op = observer.startInference(request, { trace: "suppress" });
    op.recordChunk();
    expect(op.observationRef()).toBeUndefined();
    expect(op.context()).toBeUndefined();
    op.end();
    expect(spans.getFinishedSpans()).toHaveLength(0);
    const names = normalizeMetrics(await collectMetrics()).map((metric) => metric.name);
    expect(names).toContain("gen_ai.client.inference.duration");
    expect(names).toContain("gen_ai.client.inference.time_to_first_chunk");
  });

  test("ending twice and recording after the end are no-ops", async () => {
    const { observer, spans, clock, collectMetrics } = createHarness();
    const op = observer.startInference(request);
    clock.advance(100);
    op.end();
    clock.advance(100);
    op.endWithError({ errorType: "late" });
    op.cancel();
    op.recordUsage({ inputTokens: 5 });
    op.setAttribute("late", true);
    const [span] = spans.getFinishedSpans();
    expect(spans.getFinishedSpans()).toHaveLength(1);
    expect(span?.attributes["error.type"]).toBeUndefined();
    expect(span?.attributes["late"]).toBeUndefined();
    const [duration] = normalizeMetrics(await collectMetrics());
    expect(duration?.points).toEqual([
      {
        attributes: {
          "gen_ai.operation.name": "chat",
          "gen_ai.provider.name": "llama.cpp",
          "gen_ai.request.model": "tiny",
          "xybrid.genai.execution.location": "local",
        },
        count: 1,
        sum: 0.1,
      },
    ]);
  });

  test("explicit timestamps come from elsewhere (a worker) verbatim", () => {
    const { observer, spans } = createHarness();
    const start = 1_800_000_000_000.25;
    const op = observer.startInference({ ...request, stream: true }, { startTimeMs: start });
    op.recordFirstToken(start + 40);
    op.recordChunk(start + 50);
    op.recordChunk(start + 75);
    op.end(start + 100);
    const [span] = spans.getFinishedSpans();
    expect(span?.attributes["gen_ai.response.time_to_first_chunk"]).toBeCloseTo(0.05, 9);
    expect(span?.attributes["xybrid.genai.response.time_to_first_token"]).toBeCloseTo(0.04, 9);
    expect(span?.attributes["xybrid.genai.response.chunk_count"]).toBe(2);
    expect(span?.startTime).toEqual([1_800_000_000, 250_000]);
  });

  test("usage replaces field by field", () => {
    const { observer, spans } = createHarness();
    const op = observer.startInference(request);
    op.recordUsage({ inputTokens: 3, outputTokens: 1 });
    op.recordUsage({ outputTokens: 7 });
    op.end();
    const attributes = spans.getFinishedSpans()[0]?.attributes;
    expect(attributes?.["gen_ai.usage.input_tokens"]).toBe(3);
    expect(attributes?.["gen_ai.usage.output_tokens"]).toBe(7);
  });

  test("an unended handle is never ended implicitly", () => {
    const { observer, spans } = createHarness();
    observer.startInference(request);
    expect(spans.getFinishedSpans()).toHaveLength(0);
  });
});

describe("events", () => {
  test("evaluation and feedback need a logger", () => {
    const { observer } = createHarness({ logs: false });
    const op = observer.startInference(request);
    op.end();
    const target = op.observationRef();
    expect(target).toBeDefined();
    if (target === undefined) return;
    expect(observer.recordEvaluation(target, { name: "relevance", scoreValue: 1 })).toBe(false);
    expect(observer.recordFeedback(target, { score: 1 })).toBe(false);
  });

  test("without an enabled logger the feedback comment is not processed", () => {
    const calls: string[] = [];
    const content = {
      capture: "span_and_event",
      redact: (kind: string, value: JsonValue): JsonValue => {
        calls.push(kind);
        return value;
      },
    } as const;
    const { observer } = createHarness({ logs: false, content });
    const op = observer.startInference(request);
    op.end();
    const target = op.observationRef();
    if (target === undefined) throw new Error("expected a recorded span");
    expect(observer.recordFeedback(target, { score: 1, comment: "secret" })).toBe(false);
    const emitted: unknown[] = [];
    const disabled: LoggerProviderLike = {
      getLogger: () => ({ emit: (record) => emitted.push(record), enabled: () => false }),
    };
    const quiet = new Observer({
      tracerProvider: null,
      meterProvider: null,
      loggerProvider: disabled,
      content,
    });
    expect(quiet.recordFeedback(target, { score: 1, comment: "secret" })).toBe(false);
    expect(calls).toEqual([]);
    expect(emitted).toEqual([]);
    // With an enabled logger the comment goes through the pipeline once.
    const { observer: enabled, logs } = createHarness({ content });
    expect(enabled.recordFeedback(target, { score: 1, comment: "secret" })).toBe(true);
    expect(calls).toEqual(["xybrid.genai.feedback.comment"]);
    const [record] = logs.getFinishedLogRecords();
    expect(record?.attributes["xybrid.genai.feedback.comment"]).toBe("secret");
  });

  test("no_content drops the feedback comment", () => {
    const { observer, logs } = createHarness({ content: { capture: "no_content" } });
    const op = observer.startInference(request);
    op.end();
    const target = op.observationRef();
    if (target === undefined) throw new Error("expected a recorded span");
    expect(observer.recordFeedback(target, { score: 0, comment: "private" })).toBe(true);
    const [record] = logs.getFinishedLogRecords();
    expect(record?.attributes["xybrid.genai.feedback.comment"]).toBeUndefined();
    expect(record?.attributes["gen_ai.evaluation.name"]).toBe("user_feedback");
    expect(record?.spanContext?.spanId).toBe(target.span_id);
  });

  test("content never reaches metric dimensions", async () => {
    const { observer, collectMetrics } = createHarness({ content: { capture: "span_and_event" } });
    const op = observer.startInference({ ...request, conversationId: "secret-conversation" });
    op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "secret-text" }] }]);
    op.recordUsage({ inputTokens: 2, outputTokens: 2 });
    op.endWithError({ errorType: "timeout", message: "secret-message" });
    const serialized = JSON.stringify(normalizeMetrics(await collectMetrics()));
    expect(serialized).not.toContain("secret");
  });
});

describe("configuration", () => {
  test("content capture: explicit, then environment, then span_only", () => {
    const name = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT";
    const previous = process.env[name];
    try {
      delete process.env[name];
      expect(resolveContentCapture()).toBe("span_only");
      process.env[name] = "EVENT_ONLY";
      expect(resolveContentCapture()).toBe("event_only");
      expect(resolveContentCapture("span_and_event")).toBe("span_and_event");
      process.env[name] = "true";
      expect(resolveContentCapture()).toBe("no_content");
    } finally {
      if (previous === undefined) delete process.env[name];
      else process.env[name] = previous;
    }
  });

  test("default providers are the API globals, read but never installed", () => {
    const before = trace.getTracerProvider();
    const observer = new Observer();
    const op = observer.startInference(request);
    op.end();
    expect(op.observationRef()).toBeUndefined();
    expect(trace.getTracerProvider()).toBe(before);
  });

  test("rejects a non-positive content budget", () => {
    expect(() => new Observer({ content: { maxAttributeBytes: 0 } })).toThrow(RangeError);
  });

  test("ObservationContext converts to and from OpenTelemetry contexts", () => {
    const ctx = ObservationContext.fromTraceparent(
      "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    );
    const roundTrip = ObservationContext.fromOtelContext(ctx?.toOtelContext() ?? context.active());
    expect(roundTrip?.toTraceparent()).toBe(ctx?.toTraceparent() ?? "");
  });
});

describe("generic spans", () => {
  test("startSpan carries only the caller's attributes and records no metrics", async () => {
    const { observer, spans, clock, collectMetrics } = createHarness();
    const parent = observer.startInference(request);
    const stage = observer.startSpan(
      "decode",
      { "xybrid.pipeline.stage": "decode", skipped: undefined },
      { parent: parent.context() },
    );
    clock.advance(25);
    stage.setAttribute("tokens", 4);
    stage.endWithError({ errorType: "timeout", message: "slow" });
    stage.end();
    parent.end();
    const [span] = spans.getFinishedSpans();
    expect(span?.name).toBe("decode");
    expect(span?.kind).toBe(0);
    expect(span?.parentSpanContext?.spanId).toBe(parent.observationRef()?.span_id ?? "");
    expect(span?.attributes).toEqual({
      "xybrid.pipeline.stage": "decode",
      tokens: 4,
      "error.type": "timeout",
    });
    expect(span?.status).toEqual({ code: 2, message: "slow" });
    expect(stage.observationRef()?.span_id).toBe(span?.spanContext().spanId ?? "");
    const names = normalizeMetrics(await collectMetrics()).map((metric) => metric.name);
    expect(names).toEqual(["gen_ai.client.inference.duration"]);
  });
});

describe("a span processor that throws in onEnd", () => {
  test("is contained: the operation ends, metrics are recorded, later spans export", async () => {
    // Mirror of the Rust an_application_span_processor_panic_in_on_end_is_contained. The throwing
    // processor comes first, so the in-memory exporter after it never sees that span.
    const seen: string[] = [];
    const throwing: SpanProcessor = {
      onStart: () => undefined,
      onEnd: (span) => {
        seen.push(span.name);
        if (span.name === "chat boom") throw new Error("application processor threw");
      },
      forceFlush: async () => undefined,
      shutdown: async () => undefined,
    };
    const { observer, spans, collectMetrics } = createHarness({ spanProcessors: [throwing] });
    const named = (model: string) => ({ ...request, model });
    observer.startInference(named("before")).end();
    const boom = observer.startInference(named("boom"));
    expect(() => boom.end()).not.toThrow();
    expect(boom.ended).toBe(true);
    observer.startInference(named("after")).end();
    expect(seen).toEqual(["chat before", "chat boom", "chat after"]);
    expect(spans.getFinishedSpans().map((span) => span.name)).toEqual([
      "chat before",
      "chat after",
    ]);
    const durations = normalizeMetrics(await collectMetrics()).filter(
      (metric) => metric.name === "gen_ai.client.inference.duration",
    );
    const counts = durations.flatMap((metric) =>
      metric.points.map((point) => ("count" in point ? point.count : 0)),
    );
    expect(counts.reduce((a, b) => a + b, 0)).toBe(3);
  });
});
