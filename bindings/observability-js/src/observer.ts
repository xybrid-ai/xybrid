import * as otel from "@opentelemetry/api";
import {
  type AttributeValue,
  type Context,
  type Counter,
  context as contextApi,
  type Histogram,
  type Meter,
  type MeterProvider,
  metrics as metricsApi,
  ROOT_CONTEXT,
  type Span,
  SpanKind,
  SpanStatusCode,
  TraceFlags,
  type TracerProvider,
  trace as traceApi,
  ValueType,
} from "@opentelemetry/api";
import { logs as legacyLogs } from "@opentelemetry/api-logs";
import {
  ATTEMPT_SPAN_NAME,
  ATTR,
  BUCKET_BOUNDARIES,
  type ContentCapture,
  DEFAULTS,
  ENUMS,
  ENV,
  EVENTS,
  INSTRUMENTATION_SCOPE_NAME,
  METRICS,
  type MetricName,
  OPERATIONS,
} from "./catalog.ts";
import {
  boundJson,
  canonicalInputMessages,
  canonicalOutputMessages,
  canonicalParts,
  type InputMessage,
  type JsonValue,
  type MessagePart,
  type OutputMessage,
  type Redactor,
  toJson,
  truncateString,
  utf8Length,
} from "./content.ts";
import {
  ObservationContext,
  type ObservationRef,
  type ParentInput,
  resolveParent,
} from "./trace-context.ts";
import { VERSION } from "./version.ts";

/** Span attributes accepted by `attributes` and `setAttribute`. */
export type Attributes = Readonly<Record<string, AttributeValue | undefined>>;

/** Structured log attribute values (OpenTelemetry `AnyValue`). */
type AnyValue = unknown;

/** A log record as the OpenTelemetry logs API (`@opentelemetry/api` 1.10 or `api-logs`) takes it. */
export type LogRecordLike = {
  eventName?: string;
  timestamp?: otel.TimeInput;
  observedTimestamp?: otel.TimeInput;
  severityNumber?: number;
  severityText?: string;
  body?: AnyValue;
  attributes?: Record<string, AnyValue>;
  context?: Context;
};
export type LoggerLike = { emit(record: LogRecordLike): void };
/** Structural logger provider: `@opentelemetry/sdk-logs` of either the 0.2xx or 3.x line fits. */
export type LoggerProviderLike = {
  getLogger(name: string, version?: string, options?: { schemaUrl?: string }): LoggerLike;
};

export type ContentOptions = {
  /** Where content goes. Explicit value wins over `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`. */
  readonly capture?: ContentCapture;
  /** UTF-8 byte budget per content attribute (default 16384). */
  readonly maxAttributeBytes?: number;
  /** Runs before truncation and serialization; a thrown error omits that content. */
  readonly redact?: Redactor;
};

export type ObserverOptions = {
  /** Application-owned providers. `undefined` reads the API globals; `null` disables the signal. */
  readonly tracerProvider?: TracerProvider | null;
  readonly meterProvider?: MeterProvider | null;
  readonly loggerProvider?: LoggerProviderLike | null;
  readonly content?: ContentOptions;
  /** Epoch milliseconds; tests inject a manual clock. */
  readonly clock?: () => number;
};

export type TraceMode = "default" | "suppress";

export type StartOptions = {
  /** Explicit parent; always wins over the ambient OpenTelemetry context. */
  readonly parent?: ParentInput | undefined;
  /** `tracestate` accompanying a `traceparent` string parent. */
  readonly tracestate?: string | undefined;
  readonly trace?: TraceMode | undefined;
  readonly startTimeMs?: number | undefined;
  readonly attributes?: Attributes | undefined;
};

export type Location = "local" | "remote";
export type InferenceOperationName =
  | "chat"
  | "text_completion"
  | "generate_content"
  | "embeddings"
  | "predict"
  | (string & {});

export type InferenceRequest = {
  readonly operation: InferenceOperationName;
  readonly provider: string;
  readonly model?: string | undefined;
  readonly location?: Location | undefined;
  readonly runtime?: string | undefined;
  readonly runtimeVersion?: string | undefined;
  readonly accelerator?: string | undefined;
  readonly task?: string | undefined;
  readonly stream?: boolean | undefined;
  readonly maxTokens?: number | undefined;
  readonly temperature?: number | undefined;
  readonly topP?: number | undefined;
  readonly topK?: number | undefined;
  readonly seed?: number | undefined;
  readonly stopSequences?: readonly string[] | undefined;
  readonly outputType?: string | undefined;
  readonly conversationId?: string | undefined;
  readonly prompt?:
    | {
        readonly name: string;
        readonly version?: string | undefined;
        readonly variables?: Readonly<Record<string, string>> | undefined;
      }
    | undefined;
  readonly inputModalities?: readonly string[] | undefined;
  readonly outputModalities?: readonly string[] | undefined;
  readonly serverAddress?: string | undefined;
  readonly serverPort?: number | undefined;
  readonly embeddingDimensions?: number | undefined;
};

export type Usage = {
  readonly inputTokens?: number | undefined;
  readonly outputTokens?: number | undefined;
  readonly cacheReadInputTokens?: number | undefined;
  readonly cacheWriteInputTokens?: number | undefined;
  readonly reasoningOutputTokens?: number | undefined;
  readonly textInputTokens?: number | undefined;
  readonly imageInputTokens?: number | undefined;
  readonly audioInputTokens?: number | undefined;
  readonly textOutputTokens?: number | undefined;
  readonly imageOutputTokens?: number | undefined;
  readonly audioOutputTokens?: number | undefined;
};

export type ResourceSample = {
  readonly cpuUtilization?: number | undefined;
  readonly memoryUsedBytes?: number | undefined;
  readonly processRssBytes?: number | undefined;
  readonly processPeakRssBytes?: number | undefined;
  readonly thermalState?: "nominal" | "fair" | "serious" | "critical" | (string & {}) | undefined;
  readonly batteryLevel?: number | undefined;
  readonly batteryCharging?: boolean | undefined;
  readonly acceleratorUtilization?: number | undefined;
  readonly acceleratorMemoryUsedBytes?: number | undefined;
};

export type ErrorInfo = { readonly errorType: string; readonly message?: string | undefined };

export type AttemptInfo = {
  readonly index: number;
  readonly reason: "initial" | "retry" | "fallback" | (string & {});
  readonly provider: string;
  readonly model?: string | undefined;
  readonly location: Location;
  readonly runtime?: string | undefined;
  readonly serverAddress?: string | undefined;
  readonly serverPort?: number | undefined;
};

export type ToolCall = {
  readonly name: string;
  readonly callId?: string | undefined;
  readonly toolType?: string | undefined;
  readonly description?: string | undefined;
  readonly arguments?: unknown;
};

export type ModelLoadInfo = {
  readonly model: string;
  readonly provider?: string | undefined;
  readonly runtime?: string | undefined;
  readonly format?: string | undefined;
  readonly sizeBytes?: number | undefined;
};

export type Evaluation = {
  readonly name: string;
  readonly scoreValue?: number | undefined;
  readonly scoreLabel?: string | undefined;
  readonly explanation?: string | undefined;
  readonly responseId?: string | undefined;
};

export type Feedback = {
  readonly score?: number | undefined;
  readonly label?: string | undefined;
  readonly comment?: string | undefined;
};

const USAGE_ATTRIBUTES: readonly (readonly [keyof Usage, string])[] = [
  ["inputTokens", ATTR.USAGE_INPUT_TOKENS],
  ["outputTokens", ATTR.USAGE_OUTPUT_TOKENS],
  ["cacheReadInputTokens", ATTR.USAGE_CACHE_READ_INPUT_TOKENS],
  ["cacheWriteInputTokens", ATTR.USAGE_CACHE_WRITE_INPUT_TOKENS],
  ["reasoningOutputTokens", ATTR.USAGE_REASONING_OUTPUT_TOKENS],
  ["textInputTokens", ATTR.USAGE_TEXT_INPUT_TOKENS],
  ["imageInputTokens", ATTR.USAGE_IMAGE_INPUT_TOKENS],
  ["audioInputTokens", ATTR.USAGE_AUDIO_INPUT_TOKENS],
  ["textOutputTokens", ATTR.USAGE_TEXT_OUTPUT_TOKENS],
  ["imageOutputTokens", ATTR.USAGE_IMAGE_OUTPUT_TOKENS],
  ["audioOutputTokens", ATTR.USAGE_AUDIO_OUTPUT_TOKENS],
];

const RESOURCE_ATTRIBUTES: readonly (readonly [keyof ResourceSample, string])[] = [
  ["cpuUtilization", ATTR.RESOURCE_CPU_UTILIZATION],
  ["memoryUsedBytes", ATTR.RESOURCE_MEMORY_USED_BYTES],
  ["processRssBytes", ATTR.RESOURCE_PROCESS_RSS_BYTES],
  ["processPeakRssBytes", ATTR.RESOURCE_PROCESS_PEAK_RSS_BYTES],
  ["thermalState", ATTR.RESOURCE_THERMAL_STATE],
  ["batteryLevel", ATTR.RESOURCE_BATTERY_LEVEL],
  ["batteryCharging", ATTR.RESOURCE_BATTERY_CHARGING],
  ["acceleratorUtilization", ATTR.RESOURCE_ACCELERATOR_UTILIZATION],
  ["acceleratorMemoryUsedBytes", ATTR.RESOURCE_ACCELERATOR_MEMORY_USED_BYTES],
];

const TOKEN_MODALITIES = ["text", "image", "audio"] as const;

const msToHrTime = (ms: number): otel.HrTime => {
  let seconds = Math.floor(ms / 1000);
  let nanos = Math.round((ms - seconds * 1000) * 1e6);
  if (nanos >= 1e9) {
    seconds += 1;
    nanos -= 1e9;
  }
  return [seconds, nanos];
};

const defaultClock = (): number =>
  typeof performance !== "undefined" && typeof performance.timeOrigin === "number"
    ? performance.timeOrigin + performance.now()
    : Date.now();

const readEnv = (name: string): string | undefined => {
  const process = (globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process;
  return process?.env?.[name];
};

/** Resolves the capture mode: explicit, then the environment variable, then `span_only`. */
export const resolveContentCapture = (explicit?: ContentCapture): ContentCapture => {
  if (explicit !== undefined) return explicit;
  const fromEnv = readEnv(ENV.content_capture);
  if (fromEnv === undefined) return DEFAULTS.content_capture;
  const normalized = fromEnv.trim().toLowerCase();
  return (ENUMS.content_capture as readonly string[]).includes(normalized)
    ? (normalized as ContentCapture)
    : "no_content";
};

const globalLoggerProvider = (): LoggerProviderLike => {
  // @opentelemetry/api >= 1.10 carries the logs API; older versions use @opentelemetry/api-logs.
  const builtIn = (otel as unknown as { logs?: { getLoggerProvider(): LoggerProviderLike } }).logs;
  return builtIn?.getLoggerProvider() ?? (legacyLogs.getLoggerProvider() as LoggerProviderLike);
};

type Instrument = Histogram | Counter;

// Distinct from every callback result, including an intentional null or undefined.
const REDACTION_FAILED = Symbol("redaction failed");

/** Shared, immutable state behind an {@link Observer} and its operations. */
class ObserverCore {
  readonly capture: ContentCapture;
  readonly budget: number;
  readonly redactor: Redactor | undefined;
  readonly clock: () => number;
  readonly tracer: otel.Tracer | undefined;
  readonly logger: LoggerLike | undefined;
  readonly #meterProvider: MeterProvider | null | undefined;
  #meter:
    | { provider: MeterProvider; meter: Meter; instruments: Map<string, Instrument> }
    | undefined;

  constructor(options: ObserverOptions) {
    const content = options.content ?? {};
    this.capture = resolveContentCapture(content.capture);
    const budget = content.maxAttributeBytes ?? DEFAULTS.max_content_attribute_bytes;
    if (!Number.isSafeInteger(budget) || budget <= 0) {
      throw new RangeError("content.maxAttributeBytes must be a positive integer.");
    }
    this.budget = budget;
    this.redactor = content.redact;
    this.clock = options.clock ?? defaultClock;
    const tracerProvider =
      options.tracerProvider === undefined ? traceApi.getTracerProvider() : options.tracerProvider;
    this.tracer = tracerProvider?.getTracer(INSTRUMENTATION_SCOPE_NAME, VERSION);
    const loggerProvider =
      options.loggerProvider === undefined ? globalLoggerProvider() : options.loggerProvider;
    this.logger = loggerProvider?.getLogger(INSTRUMENTATION_SCOPE_NAME, VERSION);
    this.#meterProvider = options.meterProvider;
  }

  get spanContent(): boolean {
    return this.capture === "span_only" || this.capture === "span_and_event";
  }

  get eventContent(): boolean {
    return this.capture === "event_only" || this.capture === "span_and_event";
  }

  redact(kind: string, value: unknown): unknown {
    try {
      return this.redactor === undefined ? value : this.redactor(kind, value);
    } catch {
      // Never fall back to the original value or report the exception: either may contain content.
      return REDACTION_FAILED;
    }
  }

  /** Redacted, bounded error message (code points, not UTF-16 units). */
  errorMessage(message: string | undefined): string | undefined {
    if (message === undefined) return undefined;
    const redacted = this.redact("error.message", message);
    if (redacted === REDACTION_FAILED) return undefined;
    const text = typeof redacted === "string" ? redacted : JSON.stringify(toJson(redacted));
    const chars = Array.from(text);
    return chars.length <= DEFAULTS.max_error_message_chars
      ? text
      : chars.slice(0, DEFAULTS.max_error_message_chars).join("");
  }

  #instrument(name: MetricName): Instrument | undefined {
    const provider =
      this.#meterProvider === undefined ? metricsApi.getMeterProvider() : this.#meterProvider;
    if (provider === null) return undefined;
    if (this.#meter?.provider !== provider) {
      this.#meter = {
        provider,
        meter: provider.getMeter(INSTRUMENTATION_SCOPE_NAME, VERSION),
        instruments: new Map(),
      };
    }
    const cached = this.#meter.instruments.get(name);
    if (cached !== undefined) return cached;
    const spec = METRICS[name];
    const valueType = spec.value_type === "int" ? ValueType.INT : ValueType.DOUBLE;
    const instrument =
      spec.instrument === "histogram"
        ? this.#meter.meter.createHistogram(name, {
            unit: spec.unit,
            valueType,
            advice: { explicitBucketBoundaries: [...BUCKET_BOUNDARIES[spec.buckets]] },
          })
        : this.#meter.meter.createCounter(name, { unit: spec.unit, valueType });
    this.#meter.instruments.set(name, instrument);
    return instrument;
  }

  record(name: MetricName, value: number, attributes: otel.Attributes): void {
    const instrument = this.#instrument(name);
    if (instrument === undefined) return;
    if ("record" in instrument) instrument.record(value, attributes);
    else instrument.add(value, attributes);
  }

  emit(
    eventName: keyof typeof EVENTS,
    atMs: number,
    attributes: Record<string, AnyValue>,
    ctx: Context | undefined,
  ): boolean {
    if (this.logger === undefined) return false;
    const event = EVENTS[eventName];
    this.logger.emit({
      eventName,
      timestamp: msToHrTime(atMs),
      observedTimestamp: msToHrTime(atMs),
      severityNumber: event.severity_number,
      severityText: event.severity_text,
      attributes,
      ...(ctx === undefined ? {} : { context: ctx }),
    });
    return true;
  }
}

const pick = (
  source: Readonly<Record<string, unknown>>,
  keys: readonly string[],
): otel.Attributes => {
  const result: otel.Attributes = {};
  for (const key of keys) {
    const value = source[key];
    if (value !== undefined) result[key] = value as AttributeValue;
  }
  return result;
};

const COMMON_DIMENSIONS = [
  ATTR.OPERATION_NAME,
  ATTR.PROVIDER_NAME,
  ATTR.REQUEST_MODEL,
  ATTR.RESPONSE_MODEL,
  ATTR.SERVER_ADDRESS,
  ATTR.SERVER_PORT,
  ATTR.EXECUTION_LOCATION,
] as const;

/** Lifecycle shared by every operation handle. */
abstract class Operation {
  protected readonly core: ObserverCore;
  protected readonly span: Span | undefined;
  protected readonly startMs: number;
  protected readonly suppressed: boolean;
  /** Every non-content attribute recorded, mirrored for metrics and events. */
  protected readonly attributes: Record<string, AttributeValue> = {};
  /** Structured, bounded content destined for the details event. */
  protected readonly eventContent: Record<string, AnyValue> = {};
  protected readonly truncated: string[] = [];
  protected errorType: string | undefined;
  #ended = false;
  readonly #context: ObservationContext | undefined;
  readonly #recorded: boolean;

  protected constructor(
    core: ObserverCore,
    name: string,
    kind: SpanKind,
    initial: Record<string, AttributeValue | undefined>,
    options: StartOptions,
    parentOverride?: Context,
  ) {
    this.core = core;
    this.startMs = options.startTimeMs ?? core.clock();
    this.suppressed = options.trace === "suppress";
    for (const [key, value] of Object.entries({ ...initial, ...(options.attributes ?? {}) })) {
      if (value !== undefined) this.attributes[key] = value;
    }
    if (this.suppressed || core.tracer === undefined) {
      this.span = undefined;
      this.#context = undefined;
      this.#recorded = false;
      return;
    }
    const parentContext = parentOverride ?? Operation.parentContext(options);
    this.span = core.tracer.startSpan(
      name,
      { kind, attributes: { ...this.attributes }, startTime: msToHrTime(this.startMs) },
      parentContext,
    );
    this.#context = ObservationContext.fromSpanContext(this.span.spanContext());
    // Captured now: an ended span no longer reports isRecording().
    this.#recorded = this.span.isRecording();
  }

  private static parentContext(options: StartOptions): Context {
    const explicit = resolveParent(options.parent, options.tracestate);
    const active = contextApi.active();
    return explicit === undefined ? active : explicit.toOtelContext(active);
  }

  /** Context to pass explicitly to children across workers, threads and processes. */
  context(): ObservationContext | undefined {
    return this.#context;
  }

  /** Ids of the recorded span; `undefined` when the operation is not traced. */
  observationRef(): ObservationRef | undefined {
    return this.#recorded ? this.#context?.toRef() : undefined;
  }

  get ended(): boolean {
    return this.#ended;
  }

  setAttribute(key: string, value: AttributeValue | undefined): void {
    if (this.#ended || value === undefined) return;
    this.attributes[key] = value;
    this.span?.setAttribute(key, value);
  }

  protected setAttributes(values: Record<string, AttributeValue | undefined>): void {
    for (const [key, value] of Object.entries(values)) this.setAttribute(key, value);
  }

  protected markTruncated(key: string): void {
    if (!this.truncated.includes(key)) this.truncated.push(key);
  }

  /**
   * Records one JSON content attribute: redaction, then truncation, then span (JSON string)
   * and/or details event (structured) per the capture mode. Schema content ignores the mode
   * for spans.
   */
  protected recordJsonContent(key: string, value: JsonValue, governed: boolean): void {
    if (this.#ended) return;
    if (governed && this.core.capture === "no_content") return;
    const toSpan = !governed || this.core.spanContent;
    const toEvent = this.core.eventContent;
    if (!toSpan && !toEvent) return;
    const redacted = this.core.redact(key, value);
    if (redacted === REDACTION_FAILED) return;
    const bounded = boundJson(toJson(redacted), this.core.budget);
    if (bounded.kind === "omitted" || bounded.truncated) this.markTruncated(key);
    if (bounded.kind === "omitted") return;
    if (toSpan) this.span?.setAttribute(key, bounded.json);
    if (toEvent) this.eventContent[key] = bounded.value;
  }

  /** Records a plain-string content attribute with the `K = B` prefix rule. */
  protected recordStringContent(key: string, value: string): string | undefined {
    if (this.core.capture === "no_content") return undefined;
    const redacted = this.core.redact(key, value);
    if (redacted === REDACTION_FAILED) return undefined;
    const text = typeof redacted === "string" ? redacted : JSON.stringify(toJson(redacted));
    if (utf8Length(text) > this.core.budget) this.markTruncated(key);
    const bounded = truncateString(text, this.core.budget);
    if (this.core.spanContent) this.span?.setAttribute(key, bounded);
    if (this.core.eventContent) this.eventContent[key] = bounded;
    return bounded;
  }

  end(atMs?: number): void {
    this.finish(atMs ?? this.core.clock(), undefined);
  }

  endWithError(error: ErrorInfo, atMs?: number): void {
    this.finish(atMs ?? this.core.clock(), error);
  }

  cancel(atMs?: number): void {
    this.endWithError({ errorType: "cancelled" }, atMs);
  }

  /** Hook: final attributes before the span ends (an error, when any, is already recorded). */
  protected beforeEnd(_endMs: number): void {}

  /** Hook: metrics, recorded once whether or not the span is sampled. */
  protected abstract recordMetrics(endMs: number): void;

  /** Hook: log events at the end (inference operations only in contract v1). */
  protected emitEvents(_endMs: number, _error: ErrorInfo | undefined, _message?: string): void {}

  protected spanContextForEvents(): Context | undefined {
    return this.span === undefined ? undefined : traceApi.setSpan(ROOT_CONTEXT, this.span);
  }

  private finish(endMs: number, error: ErrorInfo | undefined): void {
    if (this.#ended) return;
    let message: string | undefined;
    if (error !== undefined) {
      this.errorType = error.errorType;
      message = this.core.errorMessage(error.message);
      this.attributes[ATTR.ERROR_TYPE] = error.errorType;
      this.span?.setAttribute(ATTR.ERROR_TYPE, error.errorType);
      this.span?.setStatus({ code: SpanStatusCode.ERROR, message: message ?? error.errorType });
    }
    this.beforeEnd(endMs);
    if (this.truncated.length > 0) {
      this.attributes[ATTR.CONTENT_TRUNCATED_ATTRIBUTES] = [...this.truncated];
      this.span?.setAttribute(ATTR.CONTENT_TRUNCATED_ATTRIBUTES, [...this.truncated]);
    }
    this.#ended = true;
    this.recordMetrics(endMs);
    this.emitEvents(endMs, error, message);
    this.span?.end(msToHrTime(endMs));
  }

  protected durationSeconds(endMs: number): number {
    return (endMs - this.startMs) / 1000;
  }
}

/** Measurements that streaming inference and attempts share. */
abstract class UsageOperation extends Operation {
  protected usage: Usage = {};

  recordUsage(usage: Usage): void {
    if (this.ended) return;
    const merged: Record<string, number | undefined> = { ...this.usage };
    for (const [field, key] of USAGE_ATTRIBUTES) {
      const value = usage[field];
      if (value === undefined || value === null) continue;
      merged[field] = value as number;
      this.setAttribute(key, value as number);
    }
    this.usage = merged as Usage;
  }

  recordResource(sample: ResourceSample): void {
    for (const [field, key] of RESOURCE_ATTRIBUTES) {
      const value = sample[field];
      if (value !== undefined && value !== null) this.setAttribute(key, value);
    }
  }
}

/** One concrete execution attempt (initial try, retry, or fallback) of a logical request. */
export class AttemptOperation extends UsageOperation {
  readonly #operationName: string;

  /** @internal */
  constructor(
    core: ObserverCore,
    operationName: string,
    info: AttemptInfo,
    startTimeMs: number | undefined,
    suppressed: boolean,
    parentContext: Context | undefined,
  ) {
    super(
      core,
      ATTEMPT_SPAN_NAME.replace("{index}", String(info.index)),
      info.location === "remote" ? SpanKind.CLIENT : SpanKind.INTERNAL,
      {
        [ATTR.ATTEMPT_INDEX]: info.index,
        [ATTR.ATTEMPT_REASON]: info.reason,
        [ATTR.PROVIDER_NAME]: info.provider,
        [ATTR.REQUEST_MODEL]: info.model,
        [ATTR.EXECUTION_LOCATION]: info.location,
        [ATTR.RUNTIME_NAME]: info.runtime,
        [ATTR.SERVER_ADDRESS]: info.serverAddress,
        [ATTR.SERVER_PORT]: info.serverPort,
      },
      {
        trace: suppressed ? "suppress" : "default",
        startTimeMs,
      },
      parentContext,
    );
    this.#operationName = operationName;
  }

  protected recordMetrics(): void {
    this.core.record("xybrid.genai.client.inference.attempts", 1, {
      [ATTR.OPERATION_NAME]: this.#operationName,
      ...pick(this.attributes, [
        ATTR.PROVIDER_NAME,
        ATTR.REQUEST_MODEL,
        ATTR.EXECUTION_LOCATION,
        ATTR.ATTEMPT_REASON,
        ATTR.ERROR_TYPE,
      ]),
    });
  }
}

const spanName = (template: string, value: string | undefined, operation: string): string =>
  value === undefined || value.length === 0 ? operation : template.replace(/\{[a-z]+\}/, value);

/** A logical inference request: chat, completion, content generation, embeddings, predict. */
export class InferenceOperation extends UsageOperation {
  readonly #request: InferenceRequest;
  #chunks: number[] = [];
  #firstTokenMs: number | undefined;
  #attempts = 0;
  #outputRecorded = false;
  #finishReasonsRecorded = false;

  /** @internal */
  constructor(core: ObserverCore, request: InferenceRequest, options: StartOptions) {
    const template =
      (OPERATIONS as Record<string, { span_name: string }>)[request.operation]?.span_name ??
      `${request.operation} {model}`;
    super(
      core,
      spanName(template, request.model, request.operation),
      request.location === "remote" ? SpanKind.CLIENT : SpanKind.INTERNAL,
      {
        [ATTR.OPERATION_NAME]: request.operation,
        [ATTR.PROVIDER_NAME]: request.provider,
        [ATTR.REQUEST_MODEL]: request.model,
        [ATTR.EXECUTION_LOCATION]: request.location,
        [ATTR.RUNTIME_NAME]: request.runtime,
        [ATTR.RUNTIME_VERSION]: request.runtimeVersion,
        [ATTR.ACCELERATOR]: request.accelerator,
        [ATTR.TASK]: request.task,
        [ATTR.REQUEST_STREAM]: request.stream === true ? true : undefined,
        [ATTR.REQUEST_MAX_TOKENS]: request.maxTokens,
        [ATTR.REQUEST_TEMPERATURE]: request.temperature,
        [ATTR.REQUEST_TOP_P]: request.topP,
        [ATTR.REQUEST_TOP_K]: request.topK,
        [ATTR.REQUEST_SEED]: request.seed,
        [ATTR.REQUEST_STOP_SEQUENCES]:
          request.stopSequences === undefined ? undefined : [...request.stopSequences],
        [ATTR.OUTPUT_TYPE]: request.outputType,
        [ATTR.CONVERSATION_ID]: request.conversationId,
        [ATTR.PROMPT_NAME]: request.prompt?.name,
        [ATTR.PROMPT_VERSION]: request.prompt?.version,
        [ATTR.INPUT_MODALITIES]:
          request.inputModalities === undefined ? undefined : [...request.inputModalities],
        [ATTR.OUTPUT_MODALITIES]:
          request.outputModalities === undefined ? undefined : [...request.outputModalities],
        [ATTR.SERVER_ADDRESS]: request.serverAddress,
        [ATTR.SERVER_PORT]: request.serverPort,
        [ATTR.EMBEDDINGS_DIMENSION_COUNT]: request.embeddingDimensions,
      },
      options,
    );
    this.#request = request;
    for (const [name, value] of Object.entries(request.prompt?.variables ?? {})) {
      this.recordStringContent(`${ATTR.PROMPT_VARIABLE_PREFIX}${name}`, value);
    }
  }

  recordSystemInstructions(parts: readonly MessagePart[]): void {
    this.recordJsonContent(ATTR.SYSTEM_INSTRUCTIONS, canonicalParts(parts), true);
  }

  recordInputMessages(messages: readonly InputMessage[]): void {
    this.recordJsonContent(ATTR.INPUT_MESSAGES, canonicalInputMessages(messages), true);
  }

  /** Tool schemas: recorded in every capture mode, still redacted and bounded. */
  recordToolDefinitions(definitions: readonly unknown[]): void {
    this.recordJsonContent(ATTR.TOOL_DEFINITIONS, toJson(definitions), false);
  }

  recordOutputMessages(messages: readonly OutputMessage[]): void {
    if (this.ended) return;
    this.#outputRecorded = true;
    this.recordJsonContent(ATTR.OUTPUT_MESSAGES, canonicalOutputMessages(messages), true);
  }

  /** Marks one decoded output chunk carrying new non-empty output. */
  recordChunk(atMs?: number): void {
    if (this.ended) return;
    this.#chunks.push(atMs ?? this.core.clock());
  }

  /** Marks the runtime's first output token; later calls are ignored. */
  recordFirstToken(atMs?: number): void {
    if (this.ended || this.#firstTokenMs !== undefined) return;
    this.#firstTokenMs = atMs ?? this.core.clock();
  }

  /** A runtime-measured mean inter-token time. */
  recordTokenTiming(timing: { readonly timePerOutputTokenS: number }): void {
    this.setAttribute(ATTR.RESPONSE_TIME_PER_OUTPUT_TOKEN, timing.timePerOutputTokenS);
  }

  recordFinishReasons(reasons: readonly string[]): void {
    if (this.ended) return;
    this.#finishReasonsRecorded = true;
    this.setAttribute(ATTR.RESPONSE_FINISH_REASONS, [...reasons]);
  }

  recordResponse(response: { readonly responseId?: string; readonly model?: string }): void {
    this.setAttributes({
      [ATTR.RESPONSE_ID]: response.responseId,
      [ATTR.RESPONSE_MODEL]: response.model,
    });
  }

  /** Starts a child attempt span; the logical span keeps the request totals. */
  startAttempt(info: AttemptInfo, startTimeMs?: number): AttemptOperation {
    this.#attempts += 1;
    const parentContext =
      this.span === undefined ? undefined : traceApi.setSpan(contextApi.active(), this.span);
    return new AttemptOperation(
      this.core,
      this.#request.operation,
      info,
      startTimeMs,
      this.suppressed || this.span === undefined,
      parentContext,
    );
  }

  protected override beforeEnd(): void {
    if (this.errorType !== undefined && this.#outputRecorded && !this.#finishReasonsRecorded) {
      this.attributes[ATTR.RESPONSE_FINISH_REASONS] = ["error"];
      this.span?.setAttribute(ATTR.RESPONSE_FINISH_REASONS, ["error"]);
    }
    const first = this.#chunks[0];
    const derived: Record<string, AttributeValue | undefined> = {
      [ATTR.RESPONSE_TIME_TO_FIRST_CHUNK]:
        first === undefined ? undefined : (first - this.startMs) / 1000,
      [ATTR.RESPONSE_TIME_TO_FIRST_TOKEN]:
        this.#firstTokenMs === undefined ? undefined : (this.#firstTokenMs - this.startMs) / 1000,
      [ATTR.RESPONSE_CHUNK_COUNT]: this.#chunks.length > 0 ? this.#chunks.length : undefined,
      [ATTR.ATTEMPT_COUNT]: this.#attempts > 0 ? this.#attempts : undefined,
    };
    for (const [key, value] of Object.entries(derived)) {
      if (value === undefined) continue;
      this.attributes[key] = value;
      this.span?.setAttribute(key, value);
    }
  }

  protected recordMetrics(endMs: number): void {
    const operation = this.#request.operation;
    const spec = (OPERATIONS as Record<string, { duration_metric: MetricName; category: string }>)[
      operation
    ];
    const common = pick(this.attributes, COMMON_DIMENSIONS);
    const withError =
      this.errorType === undefined ? common : { ...common, [ATTR.ERROR_TYPE]: this.errorType };
    this.core.record(
      spec?.duration_metric ?? "gen_ai.client.operation.duration",
      this.durationSeconds(endMs),
      withError,
    );
    const [first, ...rest] = this.#chunks;
    if (first !== undefined) {
      this.core.record(
        "gen_ai.client.inference.time_to_first_chunk",
        (first - this.startMs) / 1000,
        common,
      );
      let previous = first;
      for (const chunk of rest) {
        this.core.record(
          "gen_ai.client.inference.time_per_output_chunk",
          (chunk - previous) / 1000,
          common,
        );
        previous = chunk;
      }
    }
    if (this.#firstTokenMs !== undefined) {
      this.core.record(
        "xybrid.genai.client.inference.time_to_first_token",
        (this.#firstTokenMs - this.startMs) / 1000,
        common,
      );
    }
    const category = spec?.category;
    const usage = this.usage;
    if (category === "embeddings" && usage.inputTokens !== undefined) {
      this.core.record(
        "gen_ai.client.embeddings.operation.input_tokens",
        usage.inputTokens,
        withError,
      );
    }
    if (category === "inference") this.recordUsageMetrics(common, withError);
  }

  /** Token histograms (with `error.type`) and per-modality usage counters (without). */
  private recordUsageMetrics(common: otel.Attributes, withError: otel.Attributes): void {
    const usage = this.usage;
    if (usage.inputTokens !== undefined) {
      this.core.record(
        "gen_ai.client.inference.operation.input_tokens",
        usage.inputTokens,
        withError,
      );
    }
    if (usage.outputTokens !== undefined) {
      this.core.record(
        "gen_ai.client.inference.operation.output_tokens",
        usage.outputTokens,
        withError,
      );
    }
    const perModality = (
      metric: MetricName,
      total: number | undefined,
      counts: Record<(typeof TOKEN_MODALITIES)[number], number | undefined>,
    ): void => {
      let covered = 0;
      for (const modality of TOKEN_MODALITIES) {
        const count = counts[modality];
        if (count === undefined || count <= 0) continue;
        covered += count;
        this.core.record(metric, count, { ...common, [ATTR.TOKEN_MODALITY]: modality });
      }
      if (total !== undefined && total - covered > 0) {
        this.core.record(metric, total - covered, { ...common, [ATTR.TOKEN_MODALITY]: "unknown" });
      }
    };
    perModality("gen_ai.client.inference.usage.input_tokens", usage.inputTokens, {
      text: usage.textInputTokens,
      image: usage.imageInputTokens,
      audio: usage.audioInputTokens,
    });
    perModality("gen_ai.client.inference.usage.output_tokens", usage.outputTokens, {
      text: usage.textOutputTokens,
      image: usage.imageOutputTokens,
      audio: usage.audioOutputTokens,
    });
    const single = (modalities: readonly string[] | undefined): string => {
      const only = modalities?.length === 1 ? modalities[0] : undefined;
      return only !== undefined && (TOKEN_MODALITIES as readonly string[]).includes(only)
        ? only
        : "unknown";
    };
    const totals: readonly (readonly [MetricName, number | undefined, string])[] = [
      [
        "gen_ai.client.inference.usage.cache_read.input_tokens",
        usage.cacheReadInputTokens,
        single(this.#request.inputModalities),
      ],
      [
        "gen_ai.client.inference.usage.cache_write.input_tokens",
        usage.cacheWriteInputTokens,
        single(this.#request.inputModalities),
      ],
      [
        "gen_ai.client.inference.usage.reasoning.output_tokens",
        usage.reasoningOutputTokens,
        single(this.#request.outputModalities),
      ],
    ];
    for (const [metric, value, modality] of totals) {
      if (value !== undefined && value > 0) {
        this.core.record(metric, value, { ...common, [ATTR.TOKEN_MODALITY]: modality });
      }
    }
  }

  /**
   * The exception event (not for cancellations) and, when content goes to events, the details
   * event. Both are emitted for suppressed operations too, without a span.
   */
  protected override emitEvents(
    endMs: number,
    error: ErrorInfo | undefined,
    message: string | undefined,
  ): void {
    const ctx = this.spanContextForEvents();
    if (error !== undefined && error.errorType !== "cancelled") {
      this.core.emit(
        "gen_ai.client.operation.exception",
        endMs,
        {
          [ATTR.EXCEPTION_TYPE]: error.errorType,
          ...(message === undefined ? {} : { [ATTR.EXCEPTION_MESSAGE]: message }),
        },
        ctx,
      );
    }
    if (!this.core.eventContent) return;
    const summary = pick(this.attributes, [
      ATTR.OPERATION_NAME,
      ATTR.PROVIDER_NAME,
      ATTR.REQUEST_MODEL,
      ATTR.RESPONSE_MODEL,
      ATTR.RESPONSE_ID,
      ATTR.USAGE_INPUT_TOKENS,
      ATTR.USAGE_OUTPUT_TOKENS,
      ATTR.RESPONSE_FINISH_REASONS,
      ATTR.ERROR_TYPE,
    ]);
    this.core.emit(
      "gen_ai.client.inference.operation.details",
      endMs,
      {
        ...summary,
        ...this.eventContent,
        ...(this.truncated.length > 0
          ? { [ATTR.CONTENT_TRUNCATED_ATTRIBUTES]: [...this.truncated] }
          : {}),
      },
      ctx,
    );
  }
}

/** A tool execution; arguments and results are governed content. */
export class ToolOperation extends Operation {
  /** @internal */
  constructor(core: ObserverCore, call: ToolCall, options: StartOptions) {
    super(
      core,
      spanName(OPERATIONS.execute_tool.span_name, call.name, "execute_tool"),
      SpanKind.INTERNAL,
      {
        [ATTR.OPERATION_NAME]: "execute_tool",
        [ATTR.TOOL_NAME]: call.name,
        [ATTR.TOOL_CALL_ID]: call.callId,
        [ATTR.TOOL_TYPE]: call.toolType,
        [ATTR.TOOL_DESCRIPTION]: call.description,
      },
      options,
    );
    if (call.arguments !== undefined) {
      this.recordJsonContent(ATTR.TOOL_CALL_ARGUMENTS, toJson(call.arguments), true);
    }
  }

  recordResult(value: unknown): void {
    this.recordJsonContent(ATTR.TOOL_CALL_RESULT, toJson(value), true);
  }

  protected recordMetrics(endMs: number): void {
    this.core.record(
      "gen_ai.execute_tool.duration",
      this.durationSeconds(endMs),
      pick(this.attributes, [ATTR.OPERATION_NAME, ATTR.TOOL_NAME, ATTR.TOOL_TYPE, ATTR.ERROR_TYPE]),
    );
  }
}

/** A multi-step workflow (agent loop, pipeline) that parents inferences and tools. */
export class WorkflowOperation extends Operation {
  /** @internal */
  constructor(core: ObserverCore, name: string, options: StartOptions) {
    super(
      core,
      spanName(OPERATIONS.invoke_workflow.span_name, name, "invoke_workflow"),
      SpanKind.INTERNAL,
      { [ATTR.OPERATION_NAME]: "invoke_workflow", [ATTR.WORKFLOW_NAME]: name },
      options,
    );
  }

  protected recordMetrics(endMs: number): void {
    this.core.record(
      "gen_ai.invoke_workflow.duration",
      this.durationSeconds(endMs),
      pick(this.attributes, [ATTR.OPERATION_NAME, ATTR.WORKFLOW_NAME, ATTR.ERROR_TYPE]),
    );
  }
}

/** Loading a model into a runtime. */
export class ModelLoadOperation extends Operation {
  /** @internal */
  constructor(core: ObserverCore, info: ModelLoadInfo, options: StartOptions) {
    super(
      core,
      spanName(OPERATIONS.load_model.span_name, info.model, "load_model"),
      SpanKind.INTERNAL,
      {
        [ATTR.OPERATION_NAME]: "load_model",
        [ATTR.REQUEST_MODEL]: info.model,
        [ATTR.PROVIDER_NAME]: info.provider,
        [ATTR.RUNTIME_NAME]: info.runtime,
        [ATTR.MODEL_FORMAT]: info.format,
        [ATTR.MODEL_SIZE_BYTES]: info.sizeBytes,
      },
      options,
    );
  }

  recordCacheHit(hit: boolean): void {
    this.setAttribute(ATTR.MODEL_CACHE_HIT, hit);
  }

  recordResource(sample: ResourceSample): void {
    for (const [field, key] of RESOURCE_ATTRIBUTES) {
      const value = sample[field];
      if (value !== undefined && value !== null) this.setAttribute(key, value);
    }
  }

  protected recordMetrics(endMs: number): void {
    const common = pick(this.attributes, COMMON_DIMENSIONS);
    this.core.record(
      "gen_ai.client.operation.duration",
      this.durationSeconds(endMs),
      this.errorType === undefined ? common : { ...common, [ATTR.ERROR_TYPE]: this.errorType },
    );
  }
}

/**
 * A generic `INTERNAL` span for runtime-internal work (pre-processing, decoding, routing). It
 * carries only the caller's attributes, never `gen_ai.operation.name`, and records no metrics,
 * so GenAI-aware backends never count it as a model call.
 */
export class SpanOperation extends Operation {
  /** @internal */
  constructor(core: ObserverCore, name: string, attributes: Attributes, options: StartOptions) {
    super(core, name, SpanKind.INTERNAL, { ...attributes }, options);
  }

  protected recordMetrics(): void {}
}

const isRef = (value: unknown): value is ObservationRef =>
  typeof value === "object" &&
  value !== null &&
  typeof (value as ObservationRef).traceId === "string" &&
  /^[0-9a-f]{32}$/.test((value as ObservationRef).traceId) &&
  typeof (value as ObservationRef).spanId === "string" &&
  /^[0-9a-f]{16}$/.test((value as ObservationRef).spanId);

const refContext = (target: ObservationRef): Context =>
  traceApi.setSpanContext(ROOT_CONTEXT, {
    traceId: target.traceId,
    spanId: target.spanId,
    traceFlags: TraceFlags.SAMPLED,
    isRemote: true,
  });

/**
 * Records GenAI operations as OpenTelemetry spans, metrics and events through
 * application-owned providers. Cheap to share; never installs globals; hot paths never do I/O.
 */
export class Observer {
  readonly #core: ObserverCore;

  constructor(options: ObserverOptions = {}) {
    this.#core = new ObserverCore(options);
  }

  /** The effective content capture mode. */
  get contentCapture(): ContentCapture {
    return this.#core.capture;
  }

  startInference(request: InferenceRequest, options: StartOptions = {}): InferenceOperation {
    return new InferenceOperation(this.#core, request, options);
  }

  startWorkflow(name: string, options: StartOptions = {}): WorkflowOperation {
    return new WorkflowOperation(this.#core, name, options);
  }

  startTool(call: ToolCall, options: StartOptions = {}): ToolOperation {
    return new ToolOperation(this.#core, call, options);
  }

  startModelLoad(info: ModelLoadInfo, options: StartOptions = {}): ModelLoadOperation {
    return new ModelLoadOperation(this.#core, info, options);
  }

  /** A generic internal span (no GenAI semantics, no metrics) for runtime stages. */
  startSpan(name: string, attributes: Attributes = {}, options: StartOptions = {}): SpanOperation {
    return new SpanOperation(this.#core, name, attributes, options);
  }

  /** Emits `gen_ai.evaluation.result` parented to `target`; `false` without a logger. */
  recordEvaluation(target: ObservationRef, evaluation: Evaluation): boolean {
    if (this.#core.logger === undefined || !isRef(target)) return false;
    const attributes: Record<string, AnyValue> = { [ATTR.EVALUATION_NAME]: evaluation.name };
    if (evaluation.scoreValue !== undefined) {
      attributes[ATTR.EVALUATION_SCORE_VALUE] = evaluation.scoreValue;
    }
    if (evaluation.scoreLabel !== undefined) {
      attributes[ATTR.EVALUATION_SCORE_LABEL] = evaluation.scoreLabel;
    }
    if (evaluation.explanation !== undefined) {
      attributes[ATTR.EVALUATION_EXPLANATION] = evaluation.explanation;
    }
    if (evaluation.responseId !== undefined) attributes[ATTR.RESPONSE_ID] = evaluation.responseId;
    return this.#core.emit(
      "gen_ai.evaluation.result",
      this.#core.clock(),
      attributes,
      refContext(target),
    );
  }

  /** User feedback as `gen_ai.evaluation.result` named `user_feedback`; the comment is content. */
  recordFeedback(target: ObservationRef, feedback: Feedback): boolean {
    if (this.#core.logger === undefined || !isRef(target)) return false;
    const attributes: Record<string, AnyValue> = { [ATTR.EVALUATION_NAME]: "user_feedback" };
    if (feedback.score !== undefined) attributes[ATTR.EVALUATION_SCORE_VALUE] = feedback.score;
    if (feedback.label !== undefined) attributes[ATTR.EVALUATION_SCORE_LABEL] = feedback.label;
    if (feedback.comment !== undefined && this.#core.capture !== "no_content") {
      const redacted = this.#core.redact(ATTR.FEEDBACK_COMMENT, feedback.comment);
      if (redacted !== REDACTION_FAILED) {
        const text = typeof redacted === "string" ? redacted : JSON.stringify(toJson(redacted));
        attributes[ATTR.FEEDBACK_COMMENT] = truncateString(text, this.#core.budget);
      }
    }
    return this.#core.emit(
      "gen_ai.evaluation.result",
      this.#core.clock(),
      attributes,
      refContext(target),
    );
  }
}
