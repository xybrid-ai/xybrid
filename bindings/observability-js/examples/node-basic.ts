// A standalone example with no Xybrid runtime: a fake streaming chat model that calls a tool,
// recorded inside a workflow, followed by an evaluation and user feedback.
//
//   pnpm example                                                      # prints to the console
//   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 pnpm example    # sends OTLP/HTTP
import {
  ConsoleLogRecordExporter,
  LoggerProvider,
  SimpleLogRecordProcessor,
} from "@opentelemetry/sdk-logs";
import {
  ConsoleMetricExporter,
  MeterProvider,
  PeriodicExportingMetricReader,
} from "@opentelemetry/sdk-metrics";
import { ConsoleSpanExporter, SimpleSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import { type LoggerProviderLike, Observer } from "@xybrid/observability";
import { createOtlpPipeline, otlpConfigFromEnv } from "@xybrid/observability/otlp";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Pretends to be a local model: streams words, timestamping when it decoded each one. */
async function* fakeModel(prompt: string): AsyncGenerator<{ text: string; decodedAtMs: number }> {
  await sleep(30);
  for (const word of `Looking up the weather for: ${prompt}`.split(" ")) {
    const decodedAtMs = Date.now();
    await sleep(5);
    yield { text: `${word} `, decodedAtMs };
  }
}

// Content options: an example declarative rule applied after any redaction callback.
const content = {
  rules: [{ type: "replace_literal", find: "Paris", replace: "[CITY]" }] as const,
};

const setup = (): { observer: Observer; shutdown: () => Promise<unknown> } => {
  const env = otlpConfigFromEnv();
  if (env.traces !== undefined || env.metrics !== undefined || env.logs !== undefined) {
    const pipeline = createOtlpPipeline({
      resource: { serviceName: "observability-example" },
      ...env,
    });
    return {
      observer: pipeline.observer(content),
      shutdown: async () => {
        await pipeline.shutdown();
        console.log("export stats", pipeline.stats());
      },
    };
  }
  // Application-owned console providers; nothing is registered globally.
  const tracerProvider = new TracerProvider({
    spanProcessors: [new SimpleSpanProcessor({ exporter: new ConsoleSpanExporter() })],
  });
  const reader = new PeriodicExportingMetricReader({
    exporter: new ConsoleMetricExporter(),
    exportIntervalMillis: 60_000,
  });
  const meterProvider = new MeterProvider({ readers: [reader] });
  const loggerProvider = new LoggerProvider({
    processors: [new SimpleLogRecordProcessor({ exporter: new ConsoleLogRecordExporter() })],
  });
  return {
    observer: new Observer({
      tracerProvider,
      meterProvider,
      loggerProvider: loggerProvider as LoggerProviderLike,
      content,
    }),
    shutdown: () =>
      Promise.all([tracerProvider.shutdown(), meterProvider.shutdown(), loggerProvider.shutdown()]),
  };
};

const { observer, shutdown } = setup();
const workflow = observer.startWorkflow("weather_assistant");
const question = "Paris";

const chat = observer.startInference(
  {
    operation: "chat",
    provider: "fake",
    model: "fake-chat-1",
    location: "local",
    runtime: "example",
    stream: true,
    maxTokens: 64,
    inputModalities: ["text"],
    outputModalities: ["text"],
  },
  { parent: workflow.context() },
);
chat.recordInputMessages([{ role: "user", parts: [{ type: "text", content: question }] }]);
let text = "";
for await (const delta of fakeModel(question)) {
  // The model measured when it decoded the token; only the first call counts.
  chat.recordFirstToken(delta.decodedAtMs);
  chat.recordChunk();
  text += delta.text;
}
chat.recordOutputMessages([
  {
    role: "assistant",
    parts: [
      { type: "text", content: text.trim() },
      { type: "tool_call", id: "call_1", name: "get_weather", arguments: { location: question } },
    ],
    finish_reason: "tool_call",
  },
]);
chat.recordFinishReasons(["tool_call"]);
chat.recordUsage({ inputTokens: 12, outputTokens: text.split(" ").length });
chat.end();

const tool = observer.startTool(
  {
    name: "get_weather",
    callId: "call_1",
    toolType: "function",
    arguments: { location: question },
  },
  { parent: workflow.context() },
);
await sleep(20);
tool.recordResult({ forecast: "rainy", temperature_c: 14 });
tool.end();
workflow.end();

const target = chat.observationRef();
if (target !== undefined) {
  observer.recordEvaluation(target, { name: "relevance", scoreValue: 0.9, scoreLabel: "relevant" });
  observer.recordFeedback(target, { score: 1, label: "thumbs_up", comment: "helpful" });
  console.log(`trace ${target.trace_id} span ${target.span_id}`);
}
// The reference of each operation by role, as JSON: what an application keeps to find its
// telemetry in a backend later.
const refs = {
  workflow: workflow.observationRef() ?? null,
  inference: target ?? null,
  tool: tool.observationRef() ?? null,
};
console.log(`observation refs: ${JSON.stringify(refs)}`);
await shutdown();
