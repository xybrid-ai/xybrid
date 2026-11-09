// Decoded OTLP wire fidelity (R-WIRE-1..4, R-WIRE-6): captured request bodies are decoded, for
// OTLP/protobuf and OTLP/JSON, through the packaged pipeline and through the standalone exporters
// in application-owned providers.
import { afterEach, describe, expect, test } from "bun:test";
import { LoggerProvider, SimpleLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { SimpleSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import { BUCKET_BOUNDARIES, INSTRUMENTATION_SCOPE_NAME } from "../src/catalog.ts";
import { type LoggerProviderLike, Observer } from "../src/index.ts";
import { createOtlpExporters, createOtlpPipeline, type OtlpProtocol } from "../src/otlp/index.ts";
import { ManualClock } from "./support/harness.ts";
import { attribute, decodeRequest } from "./support/otlp-decode.ts";
import { startServer, type TestServer } from "./support/otlp-server.ts";

type Providers = {
  readonly tracerProvider: TracerProvider;
  readonly meterProvider: MeterProvider;
  readonly loggerProvider: LoggerProvider;
};

const cleanups: (() => Promise<unknown>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0)) await cleanup();
});

/** One inference with whole-number doubles, two tools with structured nulls, an evaluation. */
const scenario = (providers: Providers): void => {
  const clock = new ManualClock();
  const observer = new Observer({
    tracerProvider: providers.tracerProvider,
    meterProvider: providers.meterProvider,
    loggerProvider: providers.loggerProvider as LoggerProviderLike,
    clock: clock.now,
    content: { capture: "span_and_event" },
  });
  const op = observer.startInference({
    operation: "chat",
    provider: "openai",
    model: "m",
    location: "remote",
    temperature: 0,
    topP: 1,
    maxTokens: 64,
    serverAddress: "api.example.com",
    serverPort: 443,
    stream: true,
  });
  op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "" }] }]);
  clock.advance(1000);
  op.recordChunk();
  op.recordTokenTiming({ timePerOutputTokenS: 2 });
  op.recordResource({ cpuUtilization: 1, batteryLevel: 1 });
  op.recordUsage({ inputTokens: 3, outputTokens: 4 });
  op.end();
  const tool = observer.startTool({ name: "lookup", arguments: { x: null, e: "" } });
  tool.recordResult([null, "", { k: null, e: "" }]);
  tool.end();
  const empty = observer.startTool({ name: "noop" });
  empty.recordResult(null);
  empty.end();
  const ref = op.observationRef();
  if (ref === undefined) throw new Error("expected a reference");
  expect(observer.recordEvaluation(ref, { name: "relevance", scoreValue: 1 })).toBe(true);
  // Telemetry of other scopes keeps the transformer's encoding (R-WIRE-6).
  providers.tracerProvider
    .getTracer("app")
    .startSpan("other", { attributes: { "gen_ai.request.temperature": 0 } })
    .end();
  providers.loggerProvider
    .getLogger("app")
    .emit({ eventName: "app.event", attributes: { "gen_ai.evaluation.score.value": 1 } });
};

// biome-ignore lint/suspicious/noExplicitAny: decoded wire data is inspected dynamically.
type Json = any;

const decoded = (server: TestServer, signal: "traces" | "logs" | "metrics"): Json[] =>
  server
    .received(`/v1/${signal}`)
    .map((request) => decodeRequest(signal, request.headers["content-type"], request.body));

/** Items of `scope` across every decoded request of a signal. */
const itemsOf = (
  requests: Json[],
  resourceKey: string,
  scopeKey: string,
  itemKey: string,
  scope: string,
): Json[] =>
  requests.flatMap((request: Json) =>
    (request[resourceKey] ?? []).flatMap((resource: Json) =>
      (resource[scopeKey] ?? [])
        .filter((entry: Json) => entry.scope?.name === scope)
        .flatMap((entry: Json) => entry[itemKey] ?? []),
    ),
  );

const assertWire = (server: TestServer, protocol: OtlpProtocol): void => {
  const contentType = protocol === "http/json" ? "application/json" : "application/x-protobuf";
  for (const request of server.requests) expect(request.headers["content-type"]).toBe(contentType);

  // R-WIRE-1 / R-WIRE-2: span attributes.
  const traces = decoded(server, "traces");
  const spans = itemsOf(traces, "resourceSpans", "scopeSpans", "spans", INSTRUMENTATION_SCOPE_NAME);
  const chat = spans.find((span: Json) => span.name === "chat m");
  expect(chat).toBeDefined();
  const spanAttribute = (key: string) => attribute(chat.attributes, key);
  expect(spanAttribute("gen_ai.request.temperature")).toEqual({ doubleValue: 0 });
  expect(spanAttribute("gen_ai.request.top_p")).toEqual({ doubleValue: 1 });
  expect(spanAttribute("gen_ai.response.time_to_first_chunk")).toEqual({ doubleValue: 1 });
  expect(spanAttribute("xybrid.genai.resource.cpu.utilization")).toEqual({ doubleValue: 1 });
  expect(spanAttribute("xybrid.genai.resource.battery.level")).toEqual({ doubleValue: 1 });
  expect(spanAttribute("xybrid.genai.response.time_per_output_token")).toEqual({ doubleValue: 2 });
  expect(spanAttribute("gen_ai.request.max_tokens")).toEqual({ intValue: 64 });
  expect(spanAttribute("server.port")).toEqual({ intValue: 443 });
  expect(spanAttribute("gen_ai.usage.input_tokens")).toEqual({ intValue: 3 });
  expect(spanAttribute("gen_ai.usage.output_tokens")).toEqual({ intValue: 4 });
  expect(spanAttribute("xybrid.genai.response.chunk_count")).toEqual({ intValue: 1 });
  expect(spanAttribute("gen_ai.request.stream")).toEqual({ boolValue: true });
  // Span content stays a compact JSON string: null stays null, "" stays "".
  const lookup = spans.find((span: Json) => span.name === "execute_tool lookup");
  expect(attribute(lookup.attributes, "gen_ai.tool.call.result")).toEqual({
    stringValue: '[null,"",{"k":null,"e":""}]',
  });
  // R-WIRE-6: another scope's attributes are not rewritten.
  const other = itemsOf(traces, "resourceSpans", "scopeSpans", "spans", "app");
  expect(attribute(other[0]?.attributes, "gen_ai.request.temperature")).toEqual({ intValue: 0 });

  // R-WIRE-3: log records.
  const logRequests = decoded(server, "logs");
  const records = itemsOf(
    logRequests,
    "resourceLogs",
    "scopeLogs",
    "logRecords",
    INSTRUMENTATION_SCOPE_NAME,
  );
  const evaluation = records.find(
    (record: Json) => record.eventName === "gen_ai.evaluation.result",
  );
  expect(attribute(evaluation.attributes, "gen_ai.evaluation.score.value")).toEqual({
    doubleValue: 1,
  });
  const details = records.find(
    (record: Json) => record.eventName === "gen_ai.client.inference.operation.details",
  );
  expect(attribute(details.attributes, "gen_ai.usage.input_tokens")).toEqual({ intValue: 3 });
  expect(attribute(details.attributes, "gen_ai.usage.output_tokens")).toEqual({ intValue: 4 });
  const messages = attribute(details.attributes, "gen_ai.input.messages") as Json;
  const part = messages.arrayValue.values[0].kvlistValue.values.find(
    (entry: Json) => entry.key === "parts",
  ).value.arrayValue.values[0].kvlistValue.values;
  expect(attribute(part, "content")).toEqual({ stringValue: "" });
  const toolEvents = records.filter(
    (record: Json) => record.eventName === "gen_ai.execute_tool.operation.details",
  );
  const lookupEvent = toolEvents.find(
    (record: Json) =>
      attribute(record.attributes, "gen_ai.tool.name")?.["stringValue"] === "lookup",
  );
  // Null in a map, "" in a map.
  expect(attribute(lookupEvent.attributes, "gen_ai.tool.call.arguments")).toEqual({
    kvlistValue: {
      values: [
        { key: "x", value: {} },
        { key: "e", value: { stringValue: "" } },
      ],
    },
  });
  // Null and "" in a list, and inside a nested map.
  expect(attribute(lookupEvent.attributes, "gen_ai.tool.call.result")).toEqual({
    arrayValue: {
      values: [
        {},
        { stringValue: "" },
        {
          kvlistValue: {
            values: [
              { key: "k", value: {} },
              { key: "e", value: { stringValue: "" } },
            ],
          },
        },
      ],
    },
  });
  // A top-level null is an AnyValue without a value.
  const noopEvent = toolEvents.find(
    (record: Json) => attribute(record.attributes, "gen_ai.tool.name")?.["stringValue"] === "noop",
  );
  expect(attribute(noopEvent.attributes, "gen_ai.tool.call.result")).toEqual({});
  const appRecords = itemsOf(logRequests, "resourceLogs", "scopeLogs", "logRecords", "app");
  expect(attribute(appRecords[0]?.attributes, "gen_ai.evaluation.score.value")).toEqual({
    intValue: 1,
  });

  // R-WIRE-4: metrics.
  const metrics = itemsOf(
    decoded(server, "metrics"),
    "resourceMetrics",
    "scopeMetrics",
    "metrics",
    INSTRUMENTATION_SCOPE_NAME,
  );
  const metric = (name: string): Json => metrics.find((entry: Json) => entry.name === name);
  const usage = metric("gen_ai.client.inference.usage.input_tokens");
  expect(usage.sum.dataPoints[0].asInt).toBe(3);
  expect(usage.sum.dataPoints[0].asDouble).toBeUndefined();
  const duration = metric("gen_ai.client.inference.duration").histogram.dataPoints[0];
  expect(typeof duration.sum).toBe("number");
  expect(duration.explicitBounds).toEqual([...BUCKET_BOUNDARIES.duration]);
  expect(attribute(duration.attributes, "server.port")).toEqual({ intValue: 443 });
  expect(attribute(usage.sum.dataPoints[0].attributes, "server.port")).toEqual({ intValue: 443 });
  const tokens = metric("gen_ai.client.inference.operation.input_tokens").histogram.dataPoints[0];
  expect(tokens.explicitBounds).toEqual([...BUCKET_BOUNDARIES.tokens]);
};

for (const protocol of ["http/protobuf", "http/json"] as const) {
  describe(`${protocol} wire types`, () => {
    test("through the packaged pipeline", async () => {
      const server = await startServer();
      const pipeline = createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol },
        logs: { endpoint: `${server.url}/v1/logs`, protocol },
        metrics: { endpoint: `${server.url}/v1/metrics`, protocol },
        batch: { scheduledDelayMs: 60_000 },
        metricExportIntervalMs: 60_000,
        environment: null,
      });
      cleanups.push(async () => {
        await pipeline.shutdown(1000);
        await server.close();
      });
      const { tracerProvider, meterProvider, loggerProvider } = pipeline;
      if (
        tracerProvider === undefined ||
        meterProvider === undefined ||
        loggerProvider === undefined
      ) {
        throw new Error("expected all providers");
      }
      scenario({ tracerProvider, meterProvider, loggerProvider });
      expect(await pipeline.forceFlush(3000)).toBe(true);
      assertWire(server, protocol);
    });

    test("through the standalone exporters in application-owned providers", async () => {
      const server = await startServer();
      const exporters = createOtlpExporters({
        traces: { endpoint: `${server.url}/v1/traces`, protocol },
        logs: { endpoint: `${server.url}/v1/logs`, protocol },
        metrics: { endpoint: `${server.url}/v1/metrics`, protocol },
        environment: null,
      });
      const { traces, logs, metrics } = exporters;
      if (traces === undefined || logs === undefined || metrics === undefined) {
        throw new Error("expected all exporters");
      }
      const tracerProvider = new TracerProvider({
        spanProcessors: [new SimpleSpanProcessor({ exporter: traces })],
      });
      const loggerProvider = new LoggerProvider({
        processors: [new SimpleLogRecordProcessor({ exporter: logs })],
      });
      const meterProvider = new MeterProvider({
        readers: [
          new PeriodicExportingMetricReader({
            exporter: metrics,
            exportIntervalMillis: 60_000,
            exportTimeoutMillis: 10_000,
          }),
        ],
      });
      cleanups.push(async () => {
        await Promise.all([
          tracerProvider.shutdown(),
          loggerProvider.shutdown(),
          meterProvider.shutdown(),
        ]);
        await server.close();
      });
      scenario({ tracerProvider, meterProvider, loggerProvider });
      await Promise.all([
        tracerProvider.forceFlush(),
        loggerProvider.forceFlush(),
        meterProvider.forceFlush(),
      ]);
      assertWire(server, protocol);
      // Standalone stats count items handed to export() (R-CNT-6).
      expect(traces.stats()).toMatchObject({ dropped: 0, failed: 0, pending: 0 });
      expect(traces.stats().exported).toBe(traces.stats().accepted);
      expect(traces.stats().accepted).toBe(4);
      expect(logs.stats()).toMatchObject({ accepted: 5, exported: 5, pending: 0 });
      expect(metrics.stats().exported).toBeGreaterThan(0);
      expect(metrics.stats().exported).toBe(metrics.stats().accepted);
    });
  });
}
