// Catalog value types on the OTLP wire (D13, core-1 / critic-6): the helper the package's
// exporters apply to OTLP/JSON-shaped requests before serialization.
import { describe, expect, test } from "bun:test";
import {
  JsonLogsSerializer,
  JsonMetricsSerializer,
  JsonTraceSerializer,
} from "@opentelemetry/otlp-transformer";
import { enforceCatalogTypes, numberWireType } from "../src/index.ts";
import { createHarness } from "./support/harness.ts";

type KeyValue = { key: string; value: Record<string, unknown> };
const decode = (bytes: Uint8Array | undefined): Record<string, unknown> =>
  JSON.parse(new TextDecoder().decode(bytes));
const byKey = (attributes: KeyValue[]): Record<string, Record<string, unknown>> =>
  Object.fromEntries(attributes.map((attribute) => [attribute.key, attribute.value]));

const scenario = async () => {
  const harness = createHarness();
  const { observer, clock, tracerProvider } = harness;
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
  clock.advance(1000);
  op.recordChunk();
  op.recordTokenTiming({ timePerOutputTokenS: 2 });
  op.recordResource({
    cpuUtilization: 1,
    cpuPeakUtilization: 1,
    memoryMinAvailableBytes: 1073741824,
    memoryPressure: "warn",
    batteryLevel: 1,
  });
  op.recordUsage({ inputTokens: 3, outputTokens: 4 });
  op.end();
  const ref = op.observationRef();
  if (ref === undefined) throw new Error("expected a reference");
  observer.recordEvaluation(ref, { name: "relevance", scoreValue: 1 });
  // Telemetry of another scope keeps the transformer's encoding.
  tracerProvider
    .getTracer("app")
    .startSpan("other", { attributes: { "gen_ai.request.temperature": 0 } })
    .end();
  return { harness, resourceMetrics: await harness.collectMetrics() };
};

describe("enforceCatalogTypes", () => {
  test("traces: catalog doubles become doubleValue, ints stay intValue, other scopes untouched", async () => {
    const { harness } = await scenario();
    const raw = decode(JsonTraceSerializer.serializeRequest(harness.spans.getFinishedSpans()));
    const before = JSON.stringify(raw);
    expect(before).toContain('"gen_ai.request.temperature","value":{"intValue":0}');
    const request = enforceCatalogTypes(raw) as {
      resourceSpans: {
        scopeSpans: { scope: { name: string }; spans: { attributes: KeyValue[] }[] }[];
      }[];
    };
    const scopes = request.resourceSpans.flatMap((resource) => resource.scopeSpans);
    const ours = scopes.find((scope) => scope.scope.name === "xybrid-observability");
    const attributes = byKey(ours?.spans[0]?.attributes ?? []);
    expect(attributes["gen_ai.request.temperature"]).toEqual({ doubleValue: 0 });
    expect(attributes["gen_ai.request.top_p"]).toEqual({ doubleValue: 1 });
    expect(attributes["gen_ai.response.time_to_first_chunk"]).toEqual({ doubleValue: 1 });
    expect(attributes["xybrid.genai.response.time_per_output_token"]).toEqual({ doubleValue: 2 });
    expect(attributes["xybrid.genai.resource.cpu.utilization"]).toEqual({ doubleValue: 1 });
    expect(attributes["xybrid.genai.resource.cpu.peak_utilization"]).toEqual({ doubleValue: 1 });
    expect(attributes["xybrid.genai.resource.memory.min_available_bytes"]).toEqual({
      intValue: 1073741824,
    });
    expect(attributes["xybrid.genai.resource.memory.pressure"]).toEqual({ stringValue: "warn" });
    expect(attributes["xybrid.genai.resource.battery.level"]).toEqual({ doubleValue: 1 });
    expect(attributes["gen_ai.request.max_tokens"]).toEqual({ intValue: 64 });
    expect(attributes["server.port"]).toEqual({ intValue: 443 });
    expect(attributes["gen_ai.usage.input_tokens"]).toEqual({ intValue: 3 });
    expect(attributes["xybrid.genai.response.chunk_count"]).toEqual({ intValue: 1 });
    const other = scopes.find((scope) => scope.scope.name === "app");
    expect(byKey(other?.spans[0]?.attributes ?? [])["gen_ai.request.temperature"]).toEqual({
      intValue: 0,
    });
  });

  test("logs: the evaluation score is a double", async () => {
    const { harness } = await scenario();
    const request = enforceCatalogTypes(
      decode(JsonLogsSerializer.serializeRequest(harness.logs.getFinishedLogRecords())),
    ) as { resourceLogs: { scopeLogs: { logRecords: { attributes: KeyValue[] }[] }[] }[] };
    const record = request.resourceLogs[0]?.scopeLogs[0]?.logRecords[0];
    expect(byKey(record?.attributes ?? [])["gen_ai.evaluation.score.value"]).toEqual({
      doubleValue: 1,
    });
  });

  test("metrics: counters asInt, dimensions typed, histograms untouched", async () => {
    const { resourceMetrics } = await scenario();
    const request = enforceCatalogTypes(
      decode(JsonMetricsSerializer.serializeRequest(resourceMetrics)),
    ) as {
      resourceMetrics: {
        scopeMetrics: {
          metrics: {
            name: string;
            sum?: { dataPoints: Record<string, unknown>[] };
            histogram?: { dataPoints: Record<string, unknown>[] };
          }[];
        }[];
      }[];
    };
    const metrics = request.resourceMetrics[0]?.scopeMetrics[0]?.metrics ?? [];
    const usage = metrics.find(
      (metric) => metric.name === "gen_ai.client.inference.usage.input_tokens",
    );
    expect(usage?.sum?.dataPoints[0]?.["asInt"]).toBe(3);
    expect(usage?.sum?.dataPoints[0]?.["asDouble"]).toBeUndefined();
    const duration = metrics.find((metric) => metric.name === "gen_ai.client.inference.duration");
    const point = duration?.histogram?.dataPoints[0] as { attributes: KeyValue[]; sum: number };
    expect(byKey(point.attributes)["server.port"]).toEqual({ intValue: 443 });
    expect(point.sum).toBe(1);
  });

  test("int keys turn integral doubles into intValue; Long and string ints convert", () => {
    const request = enforceCatalogTypes({
      resourceSpans: [
        {
          scopeSpans: [
            {
              scope: { name: "xybrid-observability" },
              spans: [
                {
                  attributes: [
                    { key: "server.port", value: { doubleValue: 443 } },
                    { key: "gen_ai.request.seed", value: { doubleValue: 1.5 } },
                    { key: "gen_ai.request.temperature", value: { intValue: "2" } },
                    { key: "gen_ai.request.top_p", value: { intValue: { low: 1, high: 0 } } },
                    { key: "app.custom", value: { intValue: 1 } },
                  ],
                },
              ],
            },
          ],
        },
      ],
    });
    const attributes = byKey(
      (request.resourceSpans[0]?.scopeSpans[0]?.spans[0]?.attributes ?? []) as KeyValue[],
    );
    expect(attributes["server.port"]).toEqual({ intValue: 443 });
    expect(attributes["gen_ai.request.seed"]).toEqual({ doubleValue: 1.5 });
    expect(attributes["gen_ai.request.temperature"]).toEqual({ doubleValue: 2 });
    expect(attributes["gen_ai.request.top_p"]).toEqual({ doubleValue: 1 });
    expect(attributes["app.custom"]).toEqual({ intValue: 1 });
  });

  test("numberWireType follows the catalog, else the transformer's rule", () => {
    expect(numberWireType("gen_ai.request.temperature", 0)).toBe("double");
    expect(numberWireType("gen_ai.evaluation.score.value", 1)).toBe("double");
    expect(numberWireType("server.port", 443)).toBe("int");
    expect(numberWireType("server.port", 1.5)).toBe("double");
    expect(numberWireType("app.custom", 2)).toBe("int");
    expect(numberWireType("app.custom", 2.5)).toBe("double");
  });
});
