// Runner for the shared contract fixtures (CONTRACT.md "Fixture format", contract v2). It fails on
// anything it does not understand: unknown config fields, ops or step fields, a null `expect`,
// a failed inline assertion, a metric descriptor that differs from the catalog, or a metric
// collection error.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { DataPointType, type ResourceMetrics } from "@opentelemetry/sdk-metrics";
import {
  type AttemptOperation,
  BUCKET_BOUNDARIES,
  type ContentCaptureInput,
  INSTRUMENTATION_SCOPE_NAME,
  type InferenceOperation,
  type InferenceRequest,
  JSON_ATTRIBUTES,
  type JsonValue,
  METRICS,
  type MemoryPressure,
  type ModelLoadOperation,
  ObservationContext,
  type ObservationRef,
  type Redactor,
  type SpanOperation,
  serializeContext,
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
  type TracerKind,
} from "./harness.ts";

export const FIXTURE_DIR = fileURLToPath(
  new URL("../../../../crates/xybrid-observability/contract/fixtures/", import.meta.url),
);

type Json = Record<string, unknown>;

export type FixtureConfig = {
  content_capture?: ContentCaptureInput;
  max_content_attribute_bytes?: number;
  max_retained_operation_bytes?: number;
  max_traversal_bytes?: number;
  redact_substrings?: string[];
  test_redactor?: { drop_kinds?: string[]; fail_kinds?: string[]; invalid_output_kinds?: string[] };
  redaction_rules?: unknown[];
  logs?: boolean;
  tracer?: TracerKind;
};

export type Fixture = {
  name: string;
  description: string;
  config: FixtureConfig;
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

const FIXTURE_FIELDS = new Set(["name", "description", "config", "steps", "expect"]);
const CONFIG_FIELDS = new Set([
  "content_capture",
  "max_content_attribute_bytes",
  "max_retained_operation_bytes",
  "max_traversal_bytes",
  "redact_substrings",
  "test_redactor",
  "redaction_rules",
  "logs",
  "tracer",
]);
const TEST_REDACTOR_FIELDS = new Set(["drop_kinds", "fail_kinds", "invalid_output_kinds"]);
const PARENT = ["parent", "parent_form", "parent_traceparent", "parent_tracestate"];
const USAGE_FIELDS = [
  "input_tokens",
  "output_tokens",
  "cache_read_input_tokens",
  "cache_write_input_tokens",
  "reasoning_output_tokens",
  "text_input_tokens",
  "image_input_tokens",
  "audio_input_tokens",
  "text_output_tokens",
  "image_output_tokens",
  "audio_output_tokens",
];
const RESOURCE_FIELDS = [
  "cpu_utilization",
  "cpu_peak_utilization",
  "memory_used_bytes",
  "memory_min_available_bytes",
  "memory_pressure",
  "process_rss_bytes",
  "process_peak_rss_bytes",
  "thermal_state",
  "battery_level",
  "battery_charging",
  "accelerator_utilization",
  "accelerator_memory_used_bytes",
];
const CONTENT_OP = ["id", "value"];
/** Fields each op may carry besides `op` and `at_ms`. */
const STEP_FIELDS: Readonly<Record<string, readonly string[]>> = {
  start_inference: ["id", ...PARENT, "trace", "time_ms", "expect_error", "request"],
  start_workflow: ["id", ...PARENT, "trace", "name", "time_ms", "expect_error"],
  start_tool: [
    "id",
    ...PARENT,
    "trace",
    "name",
    "call_id",
    "tool_type",
    "description",
    "arguments",
    "time_ms",
    "expect_error",
  ],
  start_model_load: [
    "id",
    ...PARENT,
    "trace",
    "model",
    "provider",
    "runtime",
    "format",
    "size_bytes",
    "time_ms",
    "expect_error",
  ],
  start_attempt: [
    "id",
    "parent",
    "index",
    "reason",
    "provider",
    "model",
    "location",
    "runtime",
    "server_address",
    "server_port",
    "time_ms",
  ],
  input_messages: CONTENT_OP,
  output_messages: CONTENT_OP,
  system_instructions: CONTENT_OP,
  tool_definitions: CONTENT_OP,
  tool_result: CONTENT_OP,
  chunk: ["id", "time_ms"],
  first_token: ["id", "time_ms"],
  token_timing: ["id", "time_per_output_token_s"],
  usage: ["id", ...USAGE_FIELDS],
  finish_reasons: ["id", "value"],
  response: ["id", "response_id", "model"],
  resource: ["id", ...RESOURCE_FIELDS],
  model_cache_hit: ["id", "value"],
  end: ["id", "time_ms"],
  cancel: ["id", "time_ms"],
  end_with_error: ["id", "error_type", "message", "time_ms"],
  evaluation: [
    "target",
    "target_ref",
    "name",
    "score_value",
    "score_label",
    "explanation",
    "response_id",
    "expect_result",
  ],
  feedback: ["target", "target_ref", "score", "label", "comment", "expect_result"],
  assert_ref: ["target", "present", "trace_id"],
  assert_context: ["target", "present", "sampled", "trace_id", "traceparent", "tracestate"],
  parse_context: ["traceparent", "tracestate", "expect"],
};
const REQUEST_FIELDS = new Set([
  "operation",
  "provider",
  "model",
  "location",
  "runtime",
  "runtime_version",
  "accelerator",
  "task",
  "stream",
  "max_tokens",
  "temperature",
  "top_p",
  "top_k",
  "seed",
  "stop_sequences",
  "frequency_penalty",
  "presence_penalty",
  "output_type",
  "conversation_id",
  "prompt_name",
  "prompt_version",
  "prompt_variables",
  "input_modalities",
  "output_modalities",
  "server_address",
  "server_port",
  "embedding_dimensions",
]);
const PARSE_EXPECT_FIELDS = new Set(["valid", "traceparent", "tracestate", "sampled"]);
const PARENT_FORMS = new Set(["object", "serialized", "cloned", "traceparent"]);

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
  | ModelLoadOperation
  | SpanOperation;

const str = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);
const num = (value: unknown): number | undefined => (typeof value === "number" ? value : undefined);
const strings = (value: unknown): string[] | undefined =>
  Array.isArray(value) ? (value as string[]) : undefined;

const unknownFields = (value: Json, allowed: ReadonlySet<string>): string[] =>
  Object.keys(value).filter((key) => !allowed.has(key));

/** Expands `$nested_array` and `$repeat_string` value macros anywhere inside `value`. */
export const expandMacros = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(expandMacros);
  if (value === null || typeof value !== "object") return value;
  const keys = Object.keys(value);
  const record = value as Json;
  if (keys.length === 1 && keys[0] === "$nested_array") {
    const spec = record["$nested_array"] as { depth: number; leaf: unknown };
    let result: unknown = expandMacros(spec.leaf);
    for (let index = 0; index < spec.depth; index += 1) result = [result];
    return result;
  }
  if (keys.length === 1 && keys[0] === "$repeat_string") {
    const spec = record["$repeat_string"] as { value: string; count: number };
    return spec.value.repeat(spec.count);
  }
  const result: Json = {};
  for (const key of keys) result[key] = expandMacros(record[key]);
  return result;
};

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
    frequencyPenalty: num(request["frequency_penalty"]),
    presencePenalty: num(request["presence_penalty"]),
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

/** The fixture reference callback (CONTRACT.md "Fixture format", `test_redactor`). */
const fixtureRedactor = (config: FixtureConfig): Redactor | undefined => {
  const substrings = config.redact_substrings ?? [];
  const test = config.test_redactor;
  if (substrings.length === 0 && test === undefined) return undefined;
  const replace = (value: JsonValue): JsonValue => {
    if (typeof value === "string") {
      let result = value;
      for (const substring of substrings) {
        if (substring.length > 0) result = result.split(substring).join("[REDACTED]");
      }
      return result;
    }
    if (value === null || typeof value !== "object") return value;
    if (Array.isArray(value)) return value.map(replace);
    const result: Record<string, JsonValue> = {};
    for (const [key, field] of Object.entries(value)) result[key] = replace(field);
    return result;
  };
  return (kind, value) => {
    if (test?.fail_kinds?.includes(kind)) throw new Error("fixture redactor failure");
    if (test?.drop_kinds?.includes(kind)) return undefined;
    if (test?.invalid_output_kinds?.includes(kind)) {
      let nested: JsonValue = "x";
      for (let index = 0; index < 129; index += 1) nested = [nested];
      return nested;
    }
    return replace(value);
  };
};

/** W3C optional whitespace trim, as the parser applies it. */
const trimOws = (value: string): string => value.replace(/^[ \t]+|[ \t]+$/g, "");

const fail = (fixture: Fixture, index: number, message: string): never => {
  throw new Error(`${fixture.name}: step ${index}: ${message}`);
};

/** Checks every metric stream of the package's scope against the catalog. */
const checkDescriptors = (fixture: Fixture, resourceMetrics: ResourceMetrics): void => {
  for (const scope of resourceMetrics.scopeMetrics) {
    if (scope.scope.name !== INSTRUMENTATION_SCOPE_NAME) continue;
    for (const metric of scope.metrics) {
      const name = metric.descriptor.name;
      const spec = (METRICS as Record<string, (typeof METRICS)[keyof typeof METRICS]>)[name];
      if (spec === undefined) throw new Error(`${fixture.name}: unknown metric ${name}`);
      if (metric.descriptor.unit !== spec.unit) throw new Error(`${fixture.name}: ${name} unit`);
      const valueType = spec.value_type === "int" ? 0 : 1;
      if (metric.descriptor.valueType !== valueType) {
        throw new Error(`${fixture.name}: ${name} value type`);
      }
      if (spec.instrument === "histogram") {
        if (metric.dataPointType !== DataPointType.HISTOGRAM) {
          throw new Error(`${fixture.name}: ${name} is not a histogram`);
        }
        const boundaries = BUCKET_BOUNDARIES[spec.buckets];
        for (const point of metric.dataPoints) {
          const buckets = (point.value as { buckets: { boundaries: number[] } }).buckets;
          if (JSON.stringify(buckets.boundaries) !== JSON.stringify(boundaries)) {
            throw new Error(`${fixture.name}: ${name} bucket boundaries`);
          }
        }
      } else if (metric.dataPointType !== DataPointType.SUM || !metric.isMonotonic) {
        throw new Error(`${fixture.name}: ${name} is not a monotonic counter`);
      }
    }
  }
};

/** Runs one fixture against in-memory exporters and a manual clock; returns normalized output. */
export const runFixture = async (fixture: Fixture): Promise<NormalizedOutput> => {
  const extraTop = unknownFields(fixture as unknown as Json, FIXTURE_FIELDS);
  if (extraTop.length > 0) throw new Error(`${fixture.name}: unknown fields ${extraTop}`);
  if (fixture.expect === null) throw new Error(`${fixture.name}: expect is null`);
  const config = fixture.config;
  const extraConfig = unknownFields(config as Json, CONFIG_FIELDS);
  if (extraConfig.length > 0) throw new Error(`${fixture.name}: unknown config ${extraConfig}`);
  if (config.test_redactor !== undefined) {
    const extra = unknownFields(config.test_redactor as Json, TEST_REDACTOR_FIELDS);
    if (extra.length > 0) throw new Error(`${fixture.name}: unknown test_redactor ${extra}`);
  }
  const redact = fixtureRedactor(config);
  const harness = createHarness({
    logs: config.logs ?? true,
    tracer: config.tracer ?? "sdk",
    content: {
      capture: config.content_capture ?? "span_only",
      ...(config.max_content_attribute_bytes === undefined
        ? {}
        : { maxAttributeBytes: config.max_content_attribute_bytes }),
      ...(config.max_retained_operation_bytes === undefined
        ? {}
        : { maxRetainedOperationBytes: config.max_retained_operation_bytes }),
      ...(config.max_traversal_bytes === undefined
        ? {}
        : { maxTraversalBytes: config.max_traversal_bytes }),
      ...(redact === undefined ? {} : { redact }),
      ...(config.redaction_rules === undefined ? {} : { rules: config.redaction_rules as never }),
    },
  });
  const { observer, clock } = harness;
  const handles = new Map<string, Handle>();
  const spanIds = new Map<string, string>();
  const at = (value: unknown): number | undefined =>
    typeof value === "number" ? CLOCK_START_MS + value : undefined;

  for (const [index, step] of fixture.steps.entries()) {
    const allowed = STEP_FIELDS[step.op];
    if (allowed === undefined) fail(fixture, index, `unknown op ${step.op}`);
    const extra = unknownFields(step, new Set(["op", "at_ms", ...(allowed ?? [])]));
    if (extra.length > 0) fail(fixture, index, `unknown fields ${extra} for ${step.op}`);
    if (typeof step.at_ms !== "number") fail(fixture, index, "missing at_ms");
    clock.set(CLOCK_START_MS + step.at_ms);
    const id = String(step.id ?? "");
    const handle = <T extends Handle>(key: unknown): T => {
      const found = handles.get(String(key));
      if (found === undefined) return fail(fixture, index, `unknown handle ${String(key)}`);
      return found as T;
    };
    const register = (operation: Handle): void => {
      handles.set(id, operation);
      // Span ids map to handles only through references, never through pass-through contexts.
      const spanId = operation.observationRef()?.span_id;
      if (spanId !== undefined) spanIds.set(spanId, id);
    };
    const startOptions = () => {
      const options: Record<string, unknown> = {
        trace: step["trace"] === "suppress" ? "suppress" : "default",
      };
      const time = at(step["time_ms"]);
      if (time !== undefined) options["startTimeMs"] = time;
      if (step["parent"] !== undefined) {
        const form = str(step["parent_form"]) ?? "object";
        if (!PARENT_FORMS.has(form)) fail(fixture, index, `unknown parent_form ${form}`);
        const ctx = handle(step["parent"]).context();
        if (ctx !== undefined) {
          if (form === "object") options["parent"] = ctx;
          else if (form === "serialized") options["parent"] = serializeContext(ctx);
          else if (form === "cloned") options["parent"] = structuredClone(ctx);
          else {
            options["parent"] = ctx.toTraceparent();
            if (ctx.tracestate !== undefined) options["tracestate"] = ctx.tracestate;
          }
        }
      } else if (typeof step["parent_traceparent"] === "string") {
        options["parent"] = step["parent_traceparent"];
        const tracestate = str(step["parent_tracestate"]);
        if (tracestate !== undefined) options["tracestate"] = tracestate;
      }
      return options;
    };
    /** Starts an operation, honoring `expect_error: "invalid_parent"`. */
    const start = (create: () => Handle): void => {
      const expected = step["expect_error"];
      if (expected === undefined) {
        register(create());
        return;
      }
      if (expected !== "invalid_parent") fail(fixture, index, `unknown expect_error ${expected}`);
      try {
        create();
      } catch (error) {
        if (error instanceof TypeError) return;
        throw error;
      }
      fail(fixture, index, "expected the start call to reject an invalid parent");
    };

    switch (step.op) {
      case "start_inference": {
        const request = step["request"] as Json;
        const extraRequest = unknownFields(request, REQUEST_FIELDS);
        if (extraRequest.length > 0) fail(fixture, index, `unknown request ${extraRequest}`);
        start(() => observer.startInference(requestFromFixture(request), startOptions()));
        break;
      }
      case "start_workflow":
        start(() => observer.startWorkflow(String(step["name"]), startOptions()));
        break;
      case "start_tool":
        start(() =>
          observer.startTool(
            {
              name: String(step["name"]),
              callId: str(step["call_id"]),
              toolType: str(step["tool_type"]),
              description: str(step["description"]),
              arguments: expandMacros(step["arguments"]),
            },
            startOptions(),
          ),
        );
        break;
      case "start_model_load":
        start(() =>
          observer.startModelLoad(
            {
              model: String(step["model"]),
              provider: str(step["provider"]),
              runtime: str(step["runtime"]),
              format: str(step["format"]),
              sizeBytes: num(step["size_bytes"]),
            },
            startOptions(),
          ),
        );
        break;
      case "start_attempt":
        register(
          handle<InferenceOperation>(step["parent"]).startAttempt(
            {
              index: Number(step["index"]),
              reason: String(step["reason"]),
              provider: String(step["provider"]),
              model: str(step["model"]),
              location: step["location"] === "remote" ? "remote" : "local",
              runtime: str(step["runtime"]),
              serverAddress: str(step["server_address"]),
              serverPort: num(step["server_port"]),
            },
            at(step["time_ms"]),
          ),
        );
        break;
      case "input_messages":
        handle<InferenceOperation>(id).recordInputMessages(expandMacros(step["value"]) as never);
        break;
      case "output_messages":
        handle<InferenceOperation>(id).recordOutputMessages(expandMacros(step["value"]) as never);
        break;
      case "system_instructions":
        handle<InferenceOperation>(id).recordSystemInstructions(
          expandMacros(step["value"]) as never,
        );
        break;
      case "tool_definitions":
        handle<InferenceOperation>(id).recordToolDefinitions(
          expandMacros(step["value"]) as unknown[],
        );
        break;
      case "tool_result":
        handle<ToolOperation>(id).recordResult(expandMacros(step["value"]));
        break;
      case "chunk":
        handle<InferenceOperation>(id).recordChunk(at(step["time_ms"]));
        break;
      case "first_token":
        handle<InferenceOperation>(id).recordFirstToken(at(step["time_ms"]));
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
          cpuPeakUtilization: num(step["cpu_peak_utilization"]),
          memoryUsedBytes: num(step["memory_used_bytes"]),
          memoryMinAvailableBytes: num(step["memory_min_available_bytes"]),
          // The fixture passes whatever string it holds; the package enforces the vocabulary.
          memoryPressure: str(step["memory_pressure"]) as MemoryPressure | undefined,
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
        handle(id).end(at(step["time_ms"]));
        break;
      case "end_with_error":
        handle(id).endWithError(
          { errorType: String(step["error_type"]), message: str(step["message"]) },
          at(step["time_ms"]),
        );
        break;
      case "cancel":
        handle(id).cancel(at(step["time_ms"]));
        break;
      case "evaluation":
      case "feedback": {
        const expected = step["expect_result"] ?? true;
        let target: unknown;
        if (step["target_ref"] !== undefined) {
          target = step["target_ref"];
        } else {
          target = handle(step["target"]).observationRef();
          if (target === undefined) {
            if (expected === false) break;
            fail(fixture, index, `handle ${String(step["target"])} has no reference`);
          }
        }
        const result =
          step.op === "evaluation"
            ? observer.recordEvaluation(target as ObservationRef, {
                name: String(step["name"]),
                scoreValue: num(step["score_value"]),
                scoreLabel: str(step["score_label"]),
                explanation: str(step["explanation"]),
                responseId: str(step["response_id"]),
              })
            : observer.recordFeedback(target as ObservationRef, {
                score: num(step["score"]),
                label: str(step["label"]),
                comment: str(step["comment"]),
              });
        if (result !== expected) fail(fixture, index, `${step.op} returned ${result}`);
        break;
      }
      case "assert_ref": {
        const operation = handle(step["target"]);
        const ref = operation.observationRef();
        if ((ref !== undefined) !== step["present"]) {
          fail(fixture, index, `reference presence is ${ref !== undefined}`);
        }
        if (ref !== undefined) {
          if (!/^[0-9a-f]{32}$/.test(ref.trace_id) || !/^[0-9a-f]{16}$/.test(ref.span_id)) {
            fail(fixture, index, "malformed reference");
          }
          if (step["trace_id"] !== undefined && ref.trace_id !== step["trace_id"]) {
            fail(fixture, index, "reference trace id");
          }
          const ctx = operation.context();
          if (ctx !== undefined && ctx.spanId !== ref.span_id) {
            fail(fixture, index, "reference span id differs from the context");
          }
        }
        break;
      }
      case "assert_context": {
        const ctx = handle(step["target"]).context();
        if ((ctx !== undefined) !== step["present"]) {
          fail(fixture, index, `context presence is ${ctx !== undefined}`);
        }
        if (ctx !== undefined) {
          if (step["sampled"] !== undefined && ctx.sampled !== step["sampled"]) {
            fail(fixture, index, `context sampled is ${ctx.sampled}`);
          }
          if (step["trace_id"] !== undefined && ctx.traceId !== step["trace_id"]) {
            fail(fixture, index, "context trace id");
          }
          if (step["traceparent"] !== undefined && ctx.toTraceparent() !== step["traceparent"]) {
            fail(fixture, index, `context traceparent ${ctx.toTraceparent()}`);
          }
          if (step["tracestate"] !== undefined && (ctx.tracestate ?? null) !== step["tracestate"]) {
            fail(fixture, index, `context tracestate ${ctx.tracestate}`);
          }
        }
        break;
      }
      case "parse_context": {
        const expected = step["expect"] as Json;
        const extraExpect = unknownFields(expected, PARSE_EXPECT_FIELDS);
        if (extraExpect.length > 0) fail(fixture, index, `unknown expect ${extraExpect}`);
        const ctx = ObservationContext.fromTraceparent(
          String(step["traceparent"]),
          str(step["tracestate"]) ?? null,
        );
        if ((ctx !== undefined) !== expected["valid"]) {
          fail(fixture, index, `validity is ${ctx !== undefined}`);
        }
        if (ctx !== undefined) {
          if (
            expected["traceparent"] !== undefined &&
            ctx.toTraceparent() !== expected["traceparent"]
          ) {
            fail(fixture, index, `traceparent ${ctx.toTraceparent()}`);
          }
          if (
            expected["tracestate"] !== undefined &&
            (ctx.tracestate ?? null) !== expected["tracestate"]
          ) {
            fail(fixture, index, `tracestate ${ctx.tracestate}`);
          }
          if (expected["sampled"] !== undefined && ctx.sampled !== expected["sampled"]) {
            fail(fixture, index, `sampled ${ctx.sampled}`);
          }
        }
        break;
      }
      default:
        fail(fixture, index, `unknown op ${step.op}`);
    }
  }

  const remoteTraceIds = new Set(
    fixture.steps
      .map((step) => step["parent_traceparent"])
      .filter((value): value is string => typeof value === "string")
      .map((value) => trimOws(value).slice(3, 35)),
  );
  const withRemoteParent = remoteTraceIds.size > 0;
  const traceIndex = new Map<string, number>();
  const spans = harness.spans.getFinishedSpans().map((span): NormalizedSpan => {
    if (span.events.length > 0) throw new Error(`${fixture.name}: span ${span.name} has events`);
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
    if (remoteTraceIds.has(context.traceId)) {
      normalized.trace_id = context.traceId;
      normalized.tracestate = context.traceState?.serialize() ?? "";
    }
    if (
      withRemoteParent &&
      parentSpanId !== undefined &&
      /^[0-9a-f]{16}$/.test(parentSpanId) &&
      parentSpanId !== "0000000000000000" &&
      !spanIds.has(parentSpanId)
    ) {
      normalized.parent_span_id = parentSpanId;
    }
    return normalized;
  });
  const logs = harness.logs.getFinishedLogRecords().map((record): NormalizedLog => {
    const spanId = record.spanContext?.spanId;
    const normalized: NormalizedLog = {
      event_name: record.eventName ?? null,
      severity_number: record.severityNumber ?? null,
      span: spanId === undefined ? null : (spanIds.get(spanId) ?? spanId),
      attributes: canonical(record.attributes) as Record<string, unknown>,
    };
    if (record.body !== undefined) normalized.body = canonical(record.body);
    return normalized;
  });
  const { resourceMetrics, errors } = await harness.collect();
  if (errors.length > 0) throw new Error(`${fixture.name}: metric collection errors`);
  checkDescriptors(fixture, resourceMetrics);
  const metrics = normalizeMetrics(resourceMetrics);
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
