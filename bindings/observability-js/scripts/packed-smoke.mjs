// Runs inside an isolated npm consumer; every production import resolves from the tarball.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { LoggerProvider, SimpleLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { SimpleSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import { Observer } from "@xybrid/observability";
import { createOtlpExporters, createOtlpPipeline } from "@xybrid/observability/otlp";
import { attribute, decodeRequest } from "./otlp-decode.mjs";

const bodies = [];
const server = createServer((req, res) => {
  const chunks = [];
  req.on("data", (chunk) => chunks.push(chunk));
  req.on("end", () => {
    const signal = req.url.slice("/v1/".length);
    bodies.push({
      signal,
      value: decodeRequest(signal, req.headers["content-type"], Buffer.concat(chunks)),
    });
    res.end();
  });
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const items = (signal, resource, scope, key) =>
  bodies
    .filter((body) => body.signal === signal)
    .flatMap((body) =>
      body.value[resource].flatMap((entry) => entry[scope].flatMap((group) => group[key])),
    );
try {
  for (const protocol of ["http/json", "http/protobuf"]) {
    for (const standalone of [false, true]) {
      bodies.length = 0;
      const options = {
        environment: null,
        ...Object.fromEntries(
          ["traces", "logs", "metrics"].map((signal) => [
            signal,
            {
              endpoint: `http://127.0.0.1:${server.address().port}/v1/${signal}`,
              protocol,
            },
          ]),
        ),
      };
      let observer;
      let finish;
      let counters;
      if (standalone) {
        const exporters = createOtlpExporters(options);
        const tracerProvider = new TracerProvider({
          spanProcessors: [new SimpleSpanProcessor({ exporter: exporters.traces })],
        });
        const loggerProvider = new LoggerProvider({
          processors: [new SimpleLogRecordProcessor({ exporter: exporters.logs })],
        });
        const meterProvider = new MeterProvider({
          readers: [
            new PeriodicExportingMetricReader({
              exporter: exporters.metrics,
              exportIntervalMillis: 60_000,
            }),
          ],
        });
        observer = new Observer({ tracerProvider, loggerProvider, meterProvider });
        finish = async () => {
          await Promise.all([
            tracerProvider.forceFlush(),
            loggerProvider.forceFlush(),
            meterProvider.forceFlush(),
          ]);
          await Promise.all([
            tracerProvider.shutdown(),
            loggerProvider.shutdown(),
            meterProvider.shutdown(),
          ]);
        };
        counters = () => Object.values(exporters).map((exporter) => exporter.stats());
      } else {
        const pipeline = createOtlpPipeline(options);
        observer = pipeline.observer();
        finish = async () => assert.equal(await pipeline.shutdown(3000), true);
        counters = () => Object.values(pipeline.stats());
      }
      const op = observer.startInference({
        operation: "chat",
        provider: "fake",
        temperature: 1,
        maxTokens: 7,
      });
      op.recordUsage({ inputTokens: 2, outputTokens: 3 });
      op.end();
      const tool = observer.startTool({ name: "lookup" });
      tool.recordResult({ missing: null, empty: "" });
      tool.end();
      assert.equal(
        observer.recordEvaluation(op.observationRef(), { name: "score", scoreValue: 1 }),
        true,
      );
      await finish();
      const spans = items("traces", "resourceSpans", "scopeSpans", "spans");
      const inference = spans.find((span) => span.name === "chat");
      assert.deepEqual(attribute(inference.attributes, "gen_ai.request.temperature"), {
        doubleValue: 1,
      });
      assert.deepEqual(attribute(inference.attributes, "gen_ai.request.max_tokens"), {
        intValue: 7,
      });
      assert.deepEqual(
        attribute(
          spans.find((span) => span.name === "execute_tool lookup").attributes,
          "gen_ai.tool.call.result",
        ),
        { stringValue: '{"missing":null,"empty":""}' },
      );
      const logs = items("logs", "resourceLogs", "scopeLogs", "logRecords");
      const evaluation = logs.find((record) => record.eventName === "gen_ai.evaluation.result");
      assert.deepEqual(attribute(evaluation.attributes, "gen_ai.evaluation.score.value"), {
        doubleValue: 1,
      });
      const metrics = items("metrics", "resourceMetrics", "scopeMetrics", "metrics");
      const duration = metrics.find((metric) => metric.name === "gen_ai.client.inference.duration");
      assert.ok(duration?.histogram.dataPoints.length > 0);
      assert.equal(Number(duration.histogram.dataPoints[0].count), 1);
      assert.ok(Number(duration.histogram.dataPoints[0].sum) >= 0);
      for (const stats of counters()) {
        assert.ok(stats.exported > 0);
        assert.equal(stats.accepted, stats.exported);
        assert.equal(stats.pending + stats.failed + stats.dropped, 0);
      }
      console.log(
        `Packed wire smoke: ${protocol}, ${standalone ? "application-owned providers" : "pipeline"}, all signals`,
      );
    }
  }
} finally {
  await new Promise((resolve) => server.close(resolve));
}
