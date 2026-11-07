import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type AttemptOperation,
  type ContentCapture,
  type InferenceOperation,
  type InferenceRequest,
  JSON_ATTRIBUTES,
  type ModelLoadOperation,
  substringRedactor,
  type ToolOperation,
  type WorkflowOperation,
} from "../../src/index.ts";
import {
  CLOCK_START_MS,
  canonical,
  createHarness,
  type NormalizedMetric,
  normalizeMetrics,
  relativeMs,
  spanKindName,
  statusCodeName,
} from "./harness.ts";

export const FIXTURE_DIR = fileURLToPath(
  new URL("../../../../crates/xybrid-observability/contract/fixtures/", import.meta.url),
);

type Json = Record<string, unknown>;

export type Fixture = {
  name: string;
  description: string;
  config: {
    content_capture: ContentCapture;
    max_content_attribute_bytes: number;
    redact_substrings: string[];
    logs: boolean;
  };
  steps: (Json & { op: string; at_ms: number; id?: string })[];
  expect: NormalizedOutput | null;
};

export type NormalizedSpan = {
  id: string | null;
  name: string;
  kind: string;
  parent: string | null;
  trace: number;
  status: { code: string; description?: string };
  start_ms: number;
  end_ms: number;
  attributes: Record<string, unknown>;
  trace_id?: string;
  tracestate?: string;
  parent_span_id?: string;
};

export type NormalizedLog = {
  event_name: string | null;
  severity_number: number | null;
  span: string | null;
  attributes: Record<string, unknown>;
  body?: unknown;
};

export type NormalizedOutput = {
  spans: NormalizedSpan[];
  metrics: NormalizedMetric[];
  logs: NormalizedLog[];
};

export const loadFixtures = (): Fixture[] =>
  readdirSync(FIXTURE_DIR)
    .filter((file) => file.endsWith(".json"))
    .sort()
    .map((file) => JSON.parse(readFileSync(join(FIXTURE_DIR, file), "utf8")) as Fixture);

type Handle =
  | InferenceOperation
  | AttemptOperation
  | ToolOperation
  | WorkflowOperation
  | ModelLoadOperation;

const str = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);
const num = (value: unknown): number | undefined => (typeof value === "number" ? value : undefined);
const strings = (value: unknown): string[] | undefined =>
  Array.isArray(value) ? (value as string[]) : undefined;

/** Maps contract `request` fields (snake_case) to the JS API. */
export const requestFromFixture = (request: Json): InferenceRequest => {
  const variables = request["prompt_variables"] as Record<string, string> | undefined;
  const promptName = str(request["prompt_name"]);
  return {
    operation: String(request["operation"]),
    provider: String(request["provider"]),
    model: str(request["model"]),
    location: str(request["location"]) as InferenceRequest["location"],
    runtime: str(request["runtime"]),
    runtimeVersion: str(request["runtime_version"]),
    accelerator: str(request["accelerator"]),
    task: str(request["task"]),
    stream: typeof request["stream"] === "boolean" ? request["stream"] : undefined,
    maxTokens: num(request["max_tokens"]),
    temperature: num(request["temperature"]),
    topP: num(request["top_p"]),
    topK: num(request["top_k"]),
    seed: num(request["seed"]),
    stopSequences: strings(request["stop_sequences"]),
    outputType: str(request["output_type"]),
    conversationId: str(request["conversation_id"]),
    prompt:
      promptName === undefined
        ? undefined
        : { name: promptName, version: str(request["prompt_version"]), variables },
    inputModalities: strings(request["input_modalities"]),
    outputModalities: strings(request["output_modalities"]),
    serverAddress: str(request["server_address"]),
    serverPort: num(request["server_port"]),
    embeddingDimensions: num(request["embedding_dimensions"]),
  };
};

/** Runs one fixture against in-memory exporters and a manual clock; returns normalized output. */
export const runFixture = async (fixture: Fixture): Promise<NormalizedOutput> => {
  const config = fixture.config;
  const harness = createHarness({
    logs: config.logs,
    content: {
      capture: config.content_capture,
      maxAttributeBytes: config.max_content_attribute_bytes,
      ...(config.redact_substrings.length > 0
        ? { redact: substringRedactor(config.redact_substrings) }
        : {}),
    },
  });
  const { observer, clock } = harness;
  const handles = new Map<string, Handle>();
  const spanIds = new Map<string, string>();
  const handle = <T extends Handle>(id: unknown): T => {
    const found = handles.get(String(id));
    if (found === undefined) throw new Error(`${fixture.name}: unknown handle ${String(id)}`);
    return found as T;
  };
  const register = (id: string, operation: Handle): void => {
    handles.set(id, operation);
    const spanId = operation.observationRef()?.spanId;
    if (spanId !== undefined) spanIds.set(spanId, id);
  };
  const startOptions = (step: Json) => ({
    ...(step["parent"] === undefined ? {} : { parent: handle(step["parent"]).context() }),
    ...(typeof step["parent_traceparent"] === "string"
      ? {
          parent: step["parent_traceparent"],
          tracestate: str(step["parent_tracestate"]),
        }
      : {}),
    trace: step["trace"] === "suppress" ? ("suppress" as const) : ("default" as const),
  });

  for (const step of fixture.steps) {
    clock.set(CLOCK_START_MS + step.at_ms);
    const id = String(step.id ?? "");
    switch (step.op) {
      case "start_inference":
        register(
          id,
          observer.startInference(requestFromFixture(step["request"] as Json), startOptions(step)),
        );
        break;
      case "start_workflow":
        register(id, observer.startWorkflow(String(step["name"]), startOptions(step)));
        break;
      case "start_tool":
        register(
          id,
          observer.startTool(
            {
              name: String(step["name"]),
              callId: str(step["call_id"]),
              toolType: str(step["tool_type"]),
              description: str(step["description"]),
              arguments: step["arguments"],
            },
            startOptions(step),
          ),
        );
        break;
      case "start_model_load":
        register(
          id,
          observer.startModelLoad(
            {
              model: String(step["model"]),
              provider: str(step["provider"]),
              runtime: str(step["runtime"]),
              format: str(step["format"]),
              sizeBytes: num(step["size_bytes"]),
            },
            startOptions(step),
          ),
        );
        break;
      case "start_attempt":
        register(
          id,
          handle<InferenceOperation>(step["parent"]).startAttempt({
            index: Number(step["index"]),
            reason: String(step["reason"]),
            provider: String(step["provider"]),
            model: str(step["model"]),
            location: step["location"] === "remote" ? "remote" : "local",
            runtime: str(step["runtime"]),
            serverAddress: str(step["server_address"]),
            serverPort: num(step["server_port"]),
          }),
        );
        break;
      case "input_messages":
        handle<InferenceOperation>(id).recordInputMessages(step["value"] as never);
        break;
      case "output_messages":
        handle<InferenceOperation>(id).recordOutputMessages(step["value"] as never);
        break;
      case "system_instructions":
        handle<InferenceOperation>(id).recordSystemInstructions(step["value"] as never);
        break;
      case "tool_definitions":
        handle<InferenceOperation>(id).recordToolDefinitions(step["value"] as unknown[]);
        break;
      case "tool_result":
        handle<ToolOperation>(id).recordResult(step["value"]);
        break;
      case "chunk":
        handle<InferenceOperation>(id).recordChunk();
        break;
      case "first_token":
        handle<InferenceOperation>(id).recordFirstToken();
        break;
      case "token_timing":
        handle<InferenceOperation>(id).recordTokenTiming({
          timePerOutputTokenS: Number(step["time_per_output_token_s"]),
        });
        break;
      case "usage":
        handle<InferenceOperation | AttemptOperation>(id).recordUsage({
          inputTokens: num(step["input_tokens"]),
          outputTokens: num(step["output_tokens"]),
          cacheReadInputTokens: num(step["cache_read_input_tokens"]),
          cacheWriteInputTokens: num(step["cache_write_input_tokens"]),
          reasoningOutputTokens: num(step["reasoning_output_tokens"]),
          textInputTokens: num(step["text_input_tokens"]),
          imageInputTokens: num(step["image_input_tokens"]),
          audioInputTokens: num(step["audio_input_tokens"]),
          textOutputTokens: num(step["text_output_tokens"]),
          imageOutputTokens: num(step["image_output_tokens"]),
          audioOutputTokens: num(step["audio_output_tokens"]),
        });
        break;
      case "finish_reasons":
        handle<InferenceOperation>(id).recordFinishReasons(step["value"] as string[]);
        break;
      case "response":
        handle<InferenceOperation>(id).recordResponse({
          ...(str(step["response_id"]) === undefined
            ? {}
            : { responseId: str(step["response_id"]) as string }),
          ...(str(step["model"]) === undefined ? {} : { model: str(step["model"]) as string }),
        });
        break;
      case "resource":
        handle<InferenceOperation | ModelLoadOperation>(id).recordResource({
          cpuUtilization: num(step["cpu_utilization"]),
          memoryUsedBytes: num(step["memory_used_bytes"]),
          processRssBytes: num(step["process_rss_bytes"]),
          processPeakRssBytes: num(step["process_peak_rss_bytes"]),
          thermalState: str(step["thermal_state"]),
          batteryLevel: num(step["battery_level"]),
          batteryCharging:
            typeof step["battery_charging"] === "boolean" ? step["battery_charging"] : undefined,
          acceleratorUtilization: num(step["accelerator_utilization"]),
          acceleratorMemoryUsedBytes: num(step["accelerator_memory_used_bytes"]),
        });
        break;
      case "model_cache_hit":
        handle<ModelLoadOperation>(id).recordCacheHit(step["value"] === true);
        break;
      case "end":
        handle(id).end();
        break;
      case "end_with_error":
        handle(id).endWithError({
          errorType: String(step["error_type"]),
          message: str(step["message"]),
        });
        break;
      case "cancel":
        handle(id).cancel();
        break;
      case "evaluation": {
        const target = handle(step["target"]).observationRef();
        if (target !== undefined) {
          observer.recordEvaluation(target, {
            name: String(step["name"]),
            scoreValue: num(step["score_value"]),
            scoreLabel: str(step["score_label"]),
            explanation: str(step["explanation"]),
            responseId: str(step["response_id"]),
          });
        }
        break;
      }
      case "feedback": {
        const target = handle(step["target"]).observationRef();
        if (target !== undefined) {
          observer.recordFeedback(target, {
            score: num(step["score"]),
            label: str(step["label"]),
            comment: str(step["comment"]),
          });
        }
        break;
      }
      default:
        throw new Error(`${fixture.name}: unknown step op ${step.op}`);
    }
  }

  const withRemoteParent = fixture.steps.some((step) => step["parent_traceparent"] !== undefined);
  const traceIndex = new Map<string, number>();
  const spans = harness.spans.getFinishedSpans().map((span): NormalizedSpan => {
    const context = span.spanContext();
    if (!traceIndex.has(context.traceId)) traceIndex.set(context.traceId, traceIndex.size);
    const attributes: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(span.attributes)) {
      attributes[key] =
        JSON_ATTRIBUTES.has(key) && typeof value === "string" ? JSON.parse(value) : value;
    }
    const parentSpanId = span.parentSpanContext?.spanId;
    const normalized: NormalizedSpan = {
      id: spanIds.get(context.spanId) ?? null,
      name: span.name,
      kind: spanKindName(span.kind),
      parent: parentSpanId === undefined ? null : (spanIds.get(parentSpanId) ?? null),
      trace: traceIndex.get(context.traceId) ?? 0,
      status:
        span.status.message === undefined || span.status.message === ""
          ? { code: statusCodeName(span.status.code) }
          : { code: statusCodeName(span.status.code), description: span.status.message },
      start_ms: relativeMs(span.startTime),
      end_ms: relativeMs(span.endTime),
      attributes: canonical(attributes) as Record<string, unknown>,
    };
    if (withRemoteParent) {
      normalized.trace_id = context.traceId;
      normalized.tracestate = context.traceState?.serialize() ?? "";
      if (parentSpanId !== undefined && !spanIds.has(parentSpanId)) {
        normalized.parent_span_id = parentSpanId;
      }
    }
    return normalized;
  });
  const logs = harness.logs.getFinishedLogRecords().map((record): NormalizedLog => {
    const spanId = record.spanContext?.spanId;
    const normalized: NormalizedLog = {
      event_name: record.eventName ?? null,
      severity_number: record.severityNumber ?? null,
      span: spanId === undefined ? null : (spanIds.get(spanId) ?? null),
      attributes: canonical(record.attributes) as Record<string, unknown>,
    };
    if (record.body !== undefined) normalized.body = canonical(record.body);
    return normalized;
  });
  const metrics = normalizeMetrics(await harness.collectMetrics());
  return { spans, metrics, logs };
};

const TOLERANCE = 1e-9;

/** Deep comparison with the contract's double tolerance; returns the first difference path. */
export const firstDifference = (actual: unknown, expected: unknown, path = "$"): string | null => {
  if (typeof actual === "number" && typeof expected === "number") {
    return Math.abs(actual - expected) <= TOLERANCE ? null : `${path}: ${actual} != ${expected}`;
  }
  if (Array.isArray(actual) && Array.isArray(expected)) {
    if (actual.length !== expected.length) {
      return `${path}: length ${actual.length} != ${expected.length}`;
    }
    for (let index = 0; index < actual.length; index += 1) {
      const difference = firstDifference(actual[index], expected[index], `${path}[${index}]`);
      if (difference !== null) return difference;
    }
    return null;
  }
  if (
    actual !== null &&
    expected !== null &&
    typeof actual === "object" &&
    typeof expected === "object" &&
    !Array.isArray(actual) &&
    !Array.isArray(expected)
  ) {
    const keys = new Set([...Object.keys(actual), ...Object.keys(expected)]);
    for (const key of [...keys].sort()) {
      const difference = firstDifference(
        (actual as Json)[key],
        (expected as Json)[key],
        `${path}.${key}`,
      );
      if (difference !== null) return difference;
    }
    return null;
  }
  return Object.is(actual, expected)
    ? null
    : `${path}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`;
};
