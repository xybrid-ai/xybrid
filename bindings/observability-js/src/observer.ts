import * as otel from "@opentelemetry/api";
import {
  type AttributeValue,
  type Context,
  type Counter,
  context as contextApi,
  diag,
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
  CUSTOM_INFERENCE_OPERATION,
  DEFAULTS,
  ENUMS,
  ENV,
  EVENTS,
  INFERENCE_OPERATIONS,
  INSTRUMENTATION_SCOPE_NAME,
  METRICS,
  type MemoryPressure,
  type MetricName,
  OPERATIONS,
} from "./catalog.ts";
import {
  boundCodePoints,
  boundJson,
  ContentProcessor,
  type ContentShape,
  type InputMessage,
  type JsonValue,
  jsonCost,
  type MessagePart,
  type OutputMessage,
  type RedactionRule,
  type Redactor,
  toRedactionRules,
  truncateString,
  utf8Length,
} from "./content.ts";
import {
  ObservationContext,
  type ObservationRef,
  type ParentInput,
  resolveParent,
  toObservationRef,
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
/**
 * Structural logger. A logger without `enabled()` counts as enabled; one whose `enabled()`
 * returns `false` receives nothing.
 */
export type LoggerLike = {
  emit(record: LogRecordLike): void;
  enabled?(options?: { context?: Context; severityNumber?: number; eventName?: string }): boolean;
};
/** Structural logger provider: `@opentelemetry/sdk-logs` of either the 0.2xx or 3.x line fits. */
export type LoggerProviderLike = {
  getLogger(name: string, version?: string, options?: { schemaUrl?: string }): LoggerLike;
};

/** A content capture mode; explicit values are trimmed and case-insensitive. */
export type ContentCaptureInput = ContentCapture | Uppercase<ContentCapture>;

export type ContentOptions = {
  /**
   * Where content goes. An explicit value (trimmed, case-insensitive) wins over
   * `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`; any other value throws `RangeError`.
   */
  readonly capture?: ContentCaptureInput | undefined;
  /** UTF-8 byte budget per content attribute (default 16384, integer >= 1). */
  readonly maxAttributeBytes?: number | undefined;
  /** Content retained by one operation until it ends (default 491520 = 480 KiB). */
  readonly maxRetainedOperationBytes?: number | undefined;
  /** Preflight cost bound of one content value (default 491520 = 480 KiB). */
  readonly maxTraversalBytes?: number | undefined;
  /** Callback redactor; runs on a canonical copy before the declarative rules. */
  readonly redact?: Redactor | undefined;
  /** Declarative rules applied after `redact` (validated; invalid rules throw `TypeError`). */
  readonly rules?: readonly RedactionRule[] | undefined;
};

export type ObserverOptions = {
  /** Application-owned providers. `undefined` reads the API globals; `null` disables the signal. */
  readonly tracerProvider?: TracerProvider | null | undefined;
  readonly meterProvider?: MeterProvider | null | undefined;
  readonly loggerProvider?: LoggerProviderLike | null | undefined;
  readonly content?: ContentOptions | undefined;
  /** Epoch milliseconds, used verbatim (tests inject a manual clock). */
  readonly clock?: (() => number) | undefined;
};

export type TraceMode = "default" | "suppress";

export type StartOptions = {
  /** Explicit parent; always wins over the ambient OpenTelemetry context. Invalid throws. */
  readonly parent?: ParentInput;
  /** `tracestate` accompanying a `traceparent` string parent. */
  readonly tracestate?: string | undefined;
  readonly trace?: TraceMode | undefined;
  /** Explicit start time in epoch milliseconds; a non-finite value is ignored. */
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
  readonly frequencyPenalty?: number | undefined;
  readonly presencePenalty?: number | undefined;
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

/** Token counts: non-negative safe integers; anything else is ignored. */
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

/**
 * A host-measured resource sample. Omit what was not measured; peak, minimum and state fields
 * describe the operation's measurement window (CONTRACT.md "Resource samples").
 */
export type ResourceSample = {
  /** CPU utilization, 0–1. */
  readonly cpuUtilization?: number | undefined;
  /** Highest CPU utilization observed in the window, 0–1. */
  readonly cpuPeakUtilization?: number | undefined;
  readonly memoryUsedBytes?: number | undefined;
  /** Lowest system memory available for allocation in the window, bytes. */
  readonly memoryMinAvailableBytes?: number | undefined;
  /** Most severe memory pressure in the window; any other value is ignored. */
  readonly memoryPressure?: MemoryPressure | undefined;
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
  /** Who serves the model; defaults to `runtime`, and is absent when neither is known. */
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

type Check = "count" | "finite" | "string" | "boolean" | "memory_pressure";

const RESOURCE_ATTRIBUTES: readonly (readonly [keyof ResourceSample, string, Check])[] = [
  ["cpuUtilization", ATTR.RESOURCE_CPU_UTILIZATION, "finite"],
  ["cpuPeakUtilization", ATTR.RESOURCE_CPU_PEAK_UTILIZATION, "finite"],
  ["memoryUsedBytes", ATTR.RESOURCE_MEMORY_USED_BYTES, "count"],
  ["memoryMinAvailableBytes", ATTR.RESOURCE_MEMORY_MIN_AVAILABLE_BYTES, "count"],
  ["memoryPressure", ATTR.RESOURCE_MEMORY_PRESSURE, "memory_pressure"],
  ["processRssBytes", ATTR.RESOURCE_PROCESS_RSS_BYTES, "count"],
  ["processPeakRssBytes", ATTR.RESOURCE_PROCESS_PEAK_RSS_BYTES, "count"],
  ["thermalState", ATTR.RESOURCE_THERMAL_STATE, "string"],
  ["batteryLevel", ATTR.RESOURCE_BATTERY_LEVEL, "finite"],
  ["batteryCharging", ATTR.RESOURCE_BATTERY_CHARGING, "boolean"],
  ["acceleratorUtilization", ATTR.RESOURCE_ACCELERATOR_UTILIZATION, "finite"],
  ["acceleratorMemoryUsedBytes", ATTR.RESOURCE_ACCELERATOR_MEMORY_USED_BYTES, "count"],
];

const TOKEN_MODALITIES = ["text", "image", "audio"] as const;

// ---------------------------------------------------------------------------------------------
// Value checks (CONTRACT.md "Usage" and "Timing"): invalid measurements are absent, never zero.

const ignored = (key: string): undefined => {
  diag.debug(`xybrid-observability: ignored an invalid value for ${key}`);
  return undefined;
};

/** A non-negative safe integer (token and byte counts). */
const count = (value: unknown, key: string): number | undefined => {
  if (value === undefined || value === null) return undefined;
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : ignored(key);
};

/** A safe integer (request integers, ports, indexes). */
const integer = (value: unknown, key: string): number | undefined => {
  if (value === undefined || value === null) return undefined;
  return typeof value === "number" && Number.isSafeInteger(value) ? value : ignored(key);
};

/** A finite double. */
const finite = (value: unknown, key: string): number | undefined => {
  if (value === undefined || value === null) return undefined;
  return typeof value === "number" && Number.isFinite(value) ? value : ignored(key);
};

const text = (value: unknown): string | undefined =>
  typeof value === "string" ? value : undefined;

/** A non-empty string array copy; empty or missing arrays are absent. */
const strings = (value: unknown): string[] | undefined =>
  Array.isArray(value) && value.length > 0 ? value.map(String) : undefined;

/** A level of the closed `memory.pressure` vocabulary; an unknown level is not a measurement. */
const memoryPressure = (value: unknown, key: string): MemoryPressure | undefined => {
  if (value === undefined || value === null) return undefined;
  const levels: readonly unknown[] = ENUMS[ATTR.RESOURCE_MEMORY_PRESSURE];
  return levels.includes(value) ? (value as MemoryPressure) : ignored(key);
};

const checked = (value: unknown, check: Check, key: string): AttributeValue | undefined => {
  switch (check) {
    case "count":
      return count(value, key);
    case "finite":
      return finite(value, key);
    case "string":
      return text(value);
    case "memory_pressure":
      return memoryPressure(value, key);
    default:
      return typeof value === "boolean" ? value : undefined;
  }
};

/** An explicit time when it is a finite number, else `undefined`. */
const explicitTime = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

const msToHrTime = (ms: number): otel.HrTime => {
  let seconds = Math.floor(ms / 1000);
  let nanos = Math.round((ms - seconds * 1000) * 1e6);
  if (nanos >= 1e9) {
    seconds += 1;
    nanos -= 1e9;
  }
  return [seconds, nanos];
};

/** Seconds between two epoch-millisecond times, clamped at zero. */
const elapsedSeconds = (fromMs: number, toMs: number): number => Math.max(0, toMs - fromMs) / 1000;

/** Orders strings by Unicode code point (equivalently UTF-8 bytes), not UTF-16 units. */
const compareCodePoints = (a: string, b: string): number => {
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const x = a.codePointAt(i) ?? 0;
    const y = b.codePointAt(j) ?? 0;
    if (x !== y) return x - y;
    i += x > 0xffff ? 2 : 1;
    j += y > 0xffff ? 2 : 1;
  }
  return a.length - i - (b.length - j);
};

// ---------------------------------------------------------------------------------------------
// Configuration.

const ENV_FAILED: unique symbol = Symbol("environment unreadable");

/** An environment variable: trimmed; empty means unset; a throwing lookup is reported. */
const readEnv = (name: string): string | undefined | typeof ENV_FAILED => {
  try {
    const process = (globalThis as { process?: { env?: Record<string, string | undefined> } })
      .process;
    const value = process?.env?.[name];
    if (value === undefined || value === null) return undefined;
    const trimmed = String(value).trim();
    return trimmed.length === 0 ? undefined : trimmed;
  } catch {
    return ENV_FAILED;
  }
};

const captureMode = (value: string): ContentCapture | undefined => {
  const normalized = value.trim().toLowerCase();
  return (ENUMS.content_capture as readonly string[]).includes(normalized)
    ? (normalized as ContentCapture)
    : undefined;
};

/**
 * Resolves the capture mode: an explicit value (trimmed, case-insensitive; anything else throws
 * `RangeError`), else `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`, else the legacy
 * `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` (`true` = `span_only`), else `span_only`.
 * An unknown or unreadable environment value means `no_content`.
 */
export const resolveContentCapture = (explicit?: ContentCaptureInput | string): ContentCapture => {
  if (explicit !== undefined) {
    const mode = typeof explicit === "string" ? captureMode(explicit) : undefined;
    if (mode === undefined) {
      throw new RangeError(
        "content.capture must be no_content, span_only, event_only or span_and_event.",
      );
    }
    return mode;
  }
  const current = readEnv(ENV.content_capture);
  if (current === ENV_FAILED) return "no_content";
  if (current !== undefined) return captureMode(current) ?? "no_content";
  const legacy = readEnv(ENV.content_capture_legacy);
  if (legacy === ENV_FAILED) return "no_content";
  if (legacy !== undefined) {
    const normalized = legacy.toLowerCase();
    if (normalized === "true") return "span_only";
    return captureMode(normalized) ?? "no_content";
  }
  return DEFAULTS.content_capture;
};

const positiveInteger = (value: unknown, fallback: number, name: string): number => {
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be an integer >= 1.`);
  }
  return value;
};

type Instrument = Histogram | Counter;

/** Shared, immutable state behind an {@link Observer} and its operations. */
class ObserverCore {
  readonly capture: ContentCapture;
  readonly budget: number;
  readonly maxRetained: number;
  readonly content: ContentProcessor;
  readonly tracer: otel.Tracer | undefined;
  readonly #clock: (() => number) | undefined;
  readonly #logger: LoggerLike | null | undefined;
  #globalLoggers: readonly LoggerLike[] | undefined;
  readonly #meterProvider: MeterProvider | null | undefined;
  #meter:
    | { provider: MeterProvider; meter: Meter; instruments: Map<string, Instrument> }
    | undefined;

  constructor(options: ObserverOptions) {
    const content = options.content ?? {};
    this.capture = resolveContentCapture(content.capture);
    this.budget = positiveInteger(
      content.maxAttributeBytes,
      DEFAULTS.max_content_attribute_bytes,
      "content.maxAttributeBytes",
    );
    this.maxRetained = positiveInteger(
      content.maxRetainedOperationBytes,
      DEFAULTS.max_retained_operation_bytes,
      "content.maxRetainedOperationBytes",
    );
    const maxTraversalBytes = positiveInteger(
      content.maxTraversalBytes,
      DEFAULTS.max_traversal_bytes,
      "content.maxTraversalBytes",
    );
    if (content.redact !== undefined && typeof content.redact !== "function") {
      throw new TypeError("content.redact must be a function.");
    }
    this.content = new ContentProcessor({
      redactor: content.redact,
      rules: content.rules === undefined ? [] : toRedactionRules(content.rules),
      maxTraversalBytes,
    });
    if (options.clock !== undefined && typeof options.clock !== "function") {
      throw new TypeError("clock must be a function.");
    }
    this.#clock = options.clock;
    const tracerProvider =
      options.tracerProvider === undefined ? traceApi.getTracerProvider() : options.tracerProvider;
    this.tracer = tracerProvider?.getTracer(INSTRUMENTATION_SCOPE_NAME, VERSION);
    this.#logger =
      options.loggerProvider === undefined || options.loggerProvider === null
        ? options.loggerProvider
        : options.loggerProvider.getLogger(INSTRUMENTATION_SCOPE_NAME, VERSION);
    this.#meterProvider = options.meterProvider;
  }

  get spanContent(): boolean {
    return this.capture === "span_only" || this.capture === "span_and_event";
  }

  get eventContent(): boolean {
    return this.capture === "event_only" || this.capture === "span_and_event";
  }

  /** Wall-clock epoch milliseconds for calls without an operation. */
  wallNow(): number {
    return this.#clock === undefined ? Date.now() : this.#clock();
  }

  /**
   * The default clock of one operation: wall-clock anchored at its start and monotonic within
   * it (`Date.now()` at start plus `performance.now()` elapsed). An injected clock wins.
   */
  operationClock(): () => number {
    if (this.#clock !== undefined) return this.#clock;
    const wallStart = Date.now();
    const performance = (globalThis as { performance?: { now?: () => number } }).performance;
    const now = performance?.now;
    if (performance === undefined || typeof now !== "function") return Date.now;
    const monotonicStart = now.call(performance);
    return () => wallStart + Math.max(0, now.call(performance) - monotonicStart);
  }

  /** Redacted, rule-applied error message bounded to 256 code points; `undefined` if dropped. */
  errorMessage(message: string | undefined): string | undefined {
    if (message === undefined) return undefined;
    const outcome = this.content.process(ATTR_ERROR_MESSAGE, message, "string");
    if (outcome.kind !== "value" || typeof outcome.value !== "string") {
      if (outcome.kind === "failed") diag.debug(`xybrid-observability: error.message omitted`);
      return undefined;
    }
    return boundCodePoints(outcome.value, DEFAULTS.max_error_message_chars);
  }

  #loggers(): readonly LoggerLike[] {
    if (this.#logger !== undefined) return this.#logger === null ? [] : [this.#logger];
    if (this.#globalLoggers === undefined) {
      // @opentelemetry/api >= 1.10 carries the logs API; older setups register through api-logs.
      // Both return proxy loggers when nothing is registered yet, so late registration works.
      const builtIn = (otel as unknown as { logs?: LoggerProviderLike }).logs;
      this.#globalLoggers = [
        ...(builtIn === undefined ? [] : [builtIn.getLogger(INSTRUMENTATION_SCOPE_NAME, VERSION)]),
        legacyLogs.getLogger(INSTRUMENTATION_SCOPE_NAME, VERSION) as LoggerLike,
      ];
    }
    return this.#globalLoggers;
  }

  /** The first logger that reports itself enabled for this event, if any. */
  #enabledLogger(
    eventName: keyof typeof EVENTS,
    severityNumber: number,
    ctx: Context,
  ): LoggerLike | undefined {
    for (const logger of this.#loggers()) {
      try {
        if (typeof logger.enabled !== "function") return logger;
        if (logger.enabled({ context: ctx, severityNumber, eventName })) return logger;
      } catch {
        // A throwing enabled() counts as disabled.
      }
    }
    return undefined;
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

  /** Emits one event; `true` only when an enabled logger received it. Never throws. */
  emit(
    eventName: keyof typeof EVENTS,
    atMs: number,
    attributes: Record<string, AnyValue>,
    ctx: Context,
  ): boolean {
    const event = EVENTS[eventName];
    const logger = this.#enabledLogger(eventName, event.severity_number, ctx);
    if (logger === undefined) return false;
    try {
      logger.emit({
        eventName,
        timestamp: msToHrTime(atMs),
        observedTimestamp: msToHrTime(atMs),
        severityNumber: event.severity_number,
        severityText: event.severity_text,
        attributes,
        context: ctx,
      });
      return true;
    } catch {
      return false;
    }
  }

  /** Validated start state; an invalid explicit parent throws before any work. */
  start(options: StartOptions): StartState {
    const parent = resolveParent(options.parent, options.tracestate);
    return {
      parent,
      suppressed: options.trace === "suppress",
      startTimeMs: explicitTime(options.startTimeMs),
      attributes: options.attributes,
    };
  }
}

const ATTR_ERROR_MESSAGE = "error.message";

type StartState = {
  readonly parent: ObservationContext | undefined;
  readonly suppressed: boolean;
  readonly startTimeMs: number | undefined;
  readonly attributes?: Attributes | undefined;
  /** Attempts: the logical span's context (already resolved). */
  readonly parentContext?: Context | undefined;
};

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

const defined = (values: Record<string, AttributeValue | undefined>): otel.Attributes => {
  const result: otel.Attributes = {};
  for (const [key, value] of Object.entries(values)) if (value !== undefined) result[key] = value;
  return result;
};

/** Content retained on an operation until it ends. */
type Retained = {
  readonly span: string | undefined;
  readonly event: JsonValue | undefined;
  readonly bytes: number;
};

/** Lifecycle shared by every operation handle. */
abstract class Operation {
  protected readonly core: ObserverCore;
  /** The operation's own span (sampled or not); `undefined` without one. */
  protected readonly span: Span | undefined;
  protected readonly startMs: number;
  protected readonly clock: () => number;
  protected readonly suppressed: boolean;
  /** Every non-content attribute recorded, mirrored for events. */
  protected readonly attributes: Record<string, AttributeValue> = {};
  /** Metric dimensions from typed inputs only (CONTRACT.md "Metric dimensions"). */
  protected readonly dims: Record<string, AttributeValue>;
  protected errorType: string | undefined;
  readonly #content = new Map<string, Retained>();
  #retained = 0;
  readonly #truncated: string[] = [];
  #ended = false;
  readonly #context: ObservationContext | undefined;
  readonly #ownSpan: boolean;

  protected constructor(
    core: ObserverCore,
    name: string,
    kind: SpanKind,
    initial: Record<string, AttributeValue | undefined>,
    dims: Record<string, AttributeValue | undefined>,
    state: StartState,
  ) {
    this.core = core;
    this.clock = core.operationClock();
    this.startMs = state.startTimeMs ?? this.clock();
    this.suppressed = state.suppressed;
    this.dims = defined(dims) as Record<string, AttributeValue>;
    for (const [key, value] of Object.entries({ ...initial, ...(state.attributes ?? {}) })) {
      if (value !== undefined) this.attributes[key] = value;
    }
    if (this.suppressed || core.tracer === undefined) {
      this.span = undefined;
      this.#context = undefined;
      this.#ownSpan = false;
      return;
    }
    const active = contextApi.active();
    const parentContext =
      state.parentContext ??
      (state.parent === undefined ? active : state.parent.toOtelContext(active));
    const parentSpanId = traceApi.getSpanContext(parentContext)?.spanId;
    const span = core.tracer.startSpan(
      name,
      { kind, attributes: { ...this.attributes }, startTime: msToHrTime(this.startMs) },
      parentContext,
    );
    const context = ObservationContext.fromSpanContext(span.spanContext());
    // API-only tracers return the parent's context: no span of its own, only propagation.
    this.#ownSpan = context !== undefined && context.spanId !== parentSpanId;
    this.span = this.#ownSpan ? span : undefined;
    this.#context = context;
  }

  /** Context to pass explicitly to children across workers, threads and processes. */
  context(): ObservationContext | undefined {
    return this.#context;
  }

  /**
   * `{trace_id, span_id}` of the operation's own span, sampled or not; `undefined` when the
   * operation is suppressed or has no span of its own (no tracer, or an API-only tracer).
   */
  observationRef(): ObservationRef | undefined {
    return this.#ownSpan ? this.#context?.toRef() : undefined;
  }

  get ended(): boolean {
    return this.#ended;
  }

  /** Sets a span attribute; it never changes metric dimensions. */
  setAttribute(key: string, value: AttributeValue | undefined): void {
    if (this.#ended || value === undefined) return;
    this.attributes[key] = value;
    this.span?.setAttribute(key, value);
  }

  protected setAttributes(values: Record<string, AttributeValue | undefined>): void {
    for (const [key, value] of Object.entries(values)) this.setAttribute(key, value);
  }

  /** Whether the operation emits a details event that carries content. */
  protected get hasDetailsEvent(): boolean {
    return false;
  }

  protected get truncated(): readonly string[] {
    return this.#truncated;
  }

  /** Content destined for the details event, by key, in recording order. */
  protected eventContent(): Record<string, AnyValue> {
    const result: Record<string, AnyValue> = {};
    for (const [key, entry] of this.#content) {
      if (entry.event !== undefined) result[key] = entry.event;
    }
    return result;
  }

  #mark(key: string): void {
    if (!this.#truncated.includes(key)) this.#truncated.push(key);
  }

  #release(key: string): void {
    const previous = this.#content.get(key);
    if (previous === undefined) return;
    this.#retained -= previous.bytes;
    this.#content.delete(key);
  }

  /**
   * Records one content attribute through the pipeline (CONTRACT.md "Content pipeline"):
   * capture gate, omit rules, preflight, clone, redactor, rules, truncation, retained budget.
   * The value is written once at the end; a later record of the same key replaces it. Schema
   * content goes to spans in every capture mode.
   */
  protected recordContent(key: string, raw: unknown, shape: ContentShape, governed: boolean): void {
    if (this.#ended) return;
    const toSpan = !governed || this.core.spanContent;
    const toEvent = this.hasDetailsEvent && this.core.eventContent;
    if (!toSpan && !toEvent) return;
    const outcome = this.core.content.process(key, raw, shape);
    if (outcome.kind === "dropped") {
      this.#release(key);
      return;
    }
    if (outcome.kind === "failed") {
      diag.debug(`xybrid-observability: ${key} omitted (${outcome.reason})`);
      this.#release(key);
      this.#mark(key);
      return;
    }
    let spanValue: string | undefined;
    let eventValue: JsonValue | undefined;
    if (shape === "string") {
      const value = outcome.value as string;
      if (utf8Length(value) > this.core.budget) this.#mark(key);
      const bounded = truncateString(value, this.core.budget);
      spanValue = toSpan ? bounded : undefined;
      eventValue = toEvent ? bounded : undefined;
    } else {
      const bounded = boundJson(outcome.value, this.core.budget);
      if (bounded.kind === "omitted") {
        this.#release(key);
        this.#mark(key);
        return;
      }
      if (bounded.truncated) this.#mark(key);
      spanValue = toSpan ? bounded.json : undefined;
      eventValue = toEvent ? bounded.value : undefined;
    }
    const keyBytes = DEFAULTS.preflight_node_bytes + utf8Length(key);
    const bytes =
      (spanValue === undefined ? 0 : keyBytes + utf8Length(spanValue)) +
      (eventValue === undefined ? 0 : keyBytes + jsonCost(eventValue));
    this.#release(key);
    if (this.#retained + bytes > this.core.maxRetained) {
      diag.debug(`xybrid-observability: ${key} omitted (retained budget)`);
      this.#mark(key);
      return;
    }
    this.#content.set(key, { span: spanValue, event: eventValue, bytes });
    this.#retained += bytes;
  }

  /** The time of a timing call: a finite explicit time verbatim, else the operation clock. */
  protected time(atMs: number | undefined): number {
    return explicitTime(atMs) ?? this.clock();
  }

  end(atMs?: number): void {
    this.#finish(atMs, undefined);
  }

  endWithError(error: ErrorInfo, atMs?: number): void {
    this.#finish(atMs, error);
  }

  cancel(atMs?: number): void {
    this.endWithError({ errorType: "cancelled" }, atMs);
  }

  /** Hook: final measurement attributes before content and the end (error already recorded). */
  protected beforeEnd(_endMs: number): void {}

  /** Hook: metrics, recorded once whether or not the span is sampled. */
  protected abstract recordMetrics(endMs: number): void;

  /** Hook: log events at the end. */
  protected emitEvents(_endMs: number, _error: ErrorInfo | undefined, _message?: string): void {}

  /** The record context of this operation's events: its own span, else explicitly none. */
  protected eventContext(): Context {
    return this.span === undefined ? ROOT_CONTEXT : traceApi.setSpan(ROOT_CONTEXT, this.span);
  }

  /** Writes a derived measurement at the end (mirror and span). */
  protected setDerived(key: string, value: AttributeValue | undefined): void {
    if (value === undefined) return;
    this.attributes[key] = value;
    this.span?.setAttribute(key, value);
  }

  #finish(atMs: number | undefined, error: ErrorInfo | undefined): void {
    if (this.#ended) return;
    // A span never ends before it starts; every derived duration is clamped.
    const endMs = Math.max(this.time(atMs), this.startMs);
    let message: string | undefined;
    if (error !== undefined) {
      const errorType = String(error.errorType);
      this.errorType = errorType;
      message = this.core.errorMessage(error.message);
      this.setDerived(ATTR.ERROR_TYPE, errorType);
      this.span?.setStatus({ code: SpanStatusCode.ERROR, message: message ?? errorType });
    }
    this.beforeEnd(endMs);
    if (this.#truncated.length > 0) {
      this.setDerived(ATTR.CONTENT_TRUNCATED_ATTRIBUTES, [...this.#truncated]);
    }
    // Content last, so attribute-count limits drop content before metadata.
    for (const [key, entry] of this.#content) {
      if (entry.span !== undefined) this.span?.setAttribute(key, entry.span);
    }
    this.#ended = true;
    this.recordMetrics(endMs);
    this.emitEvents(endMs, error, message);
    this.span?.end(msToHrTime(endMs));
  }

  protected durationSeconds(endMs: number): number {
    return elapsedSeconds(this.startMs, endMs);
  }

  /** Dimensions plus `error.type` when the operation failed. */
  protected dimsWithError(): otel.Attributes {
    return this.errorType === undefined
      ? { ...this.dims }
      : { ...this.dims, [ATTR.ERROR_TYPE]: this.errorType };
  }
}

/** Measurements that streaming inference and attempts share. */
abstract class UsageOperation extends Operation {
  protected usage: Usage = {};

  /** Replaces usage field by field; invalid counts are ignored. */
  recordUsage(usage: Usage): void {
    if (this.ended || typeof usage !== "object" || usage === null) return;
    const merged: Record<string, number | undefined> = { ...this.usage };
    for (const [field, key] of USAGE_ATTRIBUTES) {
      const value = count(usage[field], key);
      if (value === undefined) continue;
      merged[field] = value;
      this.setAttribute(key, value);
    }
    this.usage = merged as Usage;
  }

  recordResource(sample: ResourceSample): void {
    if (typeof sample !== "object" || sample === null) return;
    for (const [field, key, check] of RESOURCE_ATTRIBUTES) {
      this.setAttribute(key, checked(sample[field], check, key));
    }
  }
}

/** One concrete execution attempt (initial try, retry, or fallback) of a logical request. */
export class AttemptOperation extends UsageOperation {
  /** @internal */
  constructor(
    core: ObserverCore,
    operationName: string,
    info: AttemptInfo,
    startTimeMs: number | undefined,
    suppressed: boolean,
    parentContext: Context | undefined,
  ) {
    const index = integer(info.index, ATTR.ATTEMPT_INDEX);
    const location = text(info.location);
    super(
      core,
      ATTEMPT_SPAN_NAME.replace("{index}", () => String(index ?? info.index)),
      location === "remote" ? SpanKind.CLIENT : SpanKind.INTERNAL,
      {
        [ATTR.ATTEMPT_INDEX]: index,
        [ATTR.ATTEMPT_REASON]: text(info.reason),
        [ATTR.PROVIDER_NAME]: text(info.provider),
        [ATTR.REQUEST_MODEL]: text(info.model),
        [ATTR.EXECUTION_LOCATION]: location,
        [ATTR.RUNTIME_NAME]: text(info.runtime),
        [ATTR.SERVER_ADDRESS]: text(info.serverAddress),
        [ATTR.SERVER_PORT]: integer(info.serverPort, ATTR.SERVER_PORT),
      },
      {
        [ATTR.OPERATION_NAME]: operationName,
        [ATTR.PROVIDER_NAME]: text(info.provider),
        [ATTR.REQUEST_MODEL]: text(info.model),
        [ATTR.EXECUTION_LOCATION]: location,
        [ATTR.ATTEMPT_REASON]: text(info.reason),
      },
      {
        parent: undefined,
        suppressed,
        startTimeMs: explicitTime(startTimeMs),
        parentContext,
      },
    );
  }

  protected recordMetrics(): void {
    this.core.record("xybrid.genai.client.inference.attempts", 1, this.dimsWithError());
  }
}

/** Span name from a catalog template, substituting the subject literally. */
const spanName = (template: string, subject: string | undefined, operation: string): string =>
  subject === undefined || subject.length === 0
    ? operation
    : template.replace(/\{[a-z]+\}/, () => subject);

/** A logical inference request: chat, completion, content generation, embeddings, predict. */
export class InferenceOperation extends UsageOperation {
  readonly #request: InferenceRequest;
  readonly #category: string;
  readonly #durationMetric: MetricName;
  #firstChunkMs: number | undefined;
  #previousChunkMs: number | undefined;
  #chunkCount = 0;
  #firstTokenMs: number | undefined;
  #attempts = 0;
  #outputRecorded = false;
  #finishReasonsRecorded = false;

  /** @internal */
  constructor(core: ObserverCore, request: InferenceRequest, state: StartState) {
    const operation = String(request.operation);
    const catalog = (INFERENCE_OPERATIONS as readonly string[]).includes(operation);
    const spec = catalog
      ? OPERATIONS[operation as (typeof INFERENCE_OPERATIONS)[number]]
      : CUSTOM_INFERENCE_OPERATION;
    const model = text(request.model);
    const location = text(request.location);
    const serverPort = integer(request.serverPort, ATTR.SERVER_PORT);
    super(
      core,
      catalog
        ? spanName(spec.span_name, model, operation)
        : model
          ? `${operation} ${model}`
          : operation,
      location === "remote" ? SpanKind.CLIENT : SpanKind.INTERNAL,
      {
        [ATTR.OPERATION_NAME]: operation,
        [ATTR.PROVIDER_NAME]: text(request.provider),
        [ATTR.REQUEST_MODEL]: model,
        [ATTR.EXECUTION_LOCATION]: location,
        [ATTR.RUNTIME_NAME]: text(request.runtime),
        [ATTR.RUNTIME_VERSION]: text(request.runtimeVersion),
        [ATTR.ACCELERATOR]: text(request.accelerator),
        [ATTR.TASK]: text(request.task),
        [ATTR.REQUEST_STREAM]: request.stream === true ? true : undefined,
        [ATTR.REQUEST_MAX_TOKENS]: integer(request.maxTokens, ATTR.REQUEST_MAX_TOKENS),
        [ATTR.REQUEST_TEMPERATURE]: finite(request.temperature, ATTR.REQUEST_TEMPERATURE),
        [ATTR.REQUEST_TOP_P]: finite(request.topP, ATTR.REQUEST_TOP_P),
        [ATTR.REQUEST_TOP_K]: integer(request.topK, ATTR.REQUEST_TOP_K),
        [ATTR.REQUEST_SEED]: integer(request.seed, ATTR.REQUEST_SEED),
        [ATTR.REQUEST_FREQUENCY_PENALTY]: finite(
          request.frequencyPenalty,
          ATTR.REQUEST_FREQUENCY_PENALTY,
        ),
        [ATTR.REQUEST_PRESENCE_PENALTY]: finite(
          request.presencePenalty,
          ATTR.REQUEST_PRESENCE_PENALTY,
        ),
        [ATTR.REQUEST_STOP_SEQUENCES]: strings(request.stopSequences),
        [ATTR.OUTPUT_TYPE]: text(request.outputType),
        [ATTR.CONVERSATION_ID]: text(request.conversationId),
        [ATTR.PROMPT_NAME]: text(request.prompt?.name),
        [ATTR.PROMPT_VERSION]: text(request.prompt?.version),
        [ATTR.INPUT_MODALITIES]: strings(request.inputModalities),
        [ATTR.OUTPUT_MODALITIES]: strings(request.outputModalities),
        [ATTR.SERVER_ADDRESS]: text(request.serverAddress),
        [ATTR.SERVER_PORT]: serverPort,
        [ATTR.EMBEDDINGS_DIMENSION_COUNT]: integer(
          request.embeddingDimensions,
          ATTR.EMBEDDINGS_DIMENSION_COUNT,
        ),
      },
      {
        [ATTR.OPERATION_NAME]: operation,
        [ATTR.PROVIDER_NAME]: text(request.provider),
        [ATTR.REQUEST_MODEL]: model,
        [ATTR.SERVER_ADDRESS]: text(request.serverAddress),
        [ATTR.SERVER_PORT]: serverPort,
        [ATTR.EXECUTION_LOCATION]: location,
      },
      state,
    );
    this.#request = request;
    this.#category = spec.category;
    this.#durationMetric = spec.duration_metric;
    const variables = request.prompt?.variables;
    if (typeof variables === "object" && variables !== null) {
      for (const name of Object.keys(variables).sort(compareCodePoints)) {
        this.recordContent(
          `${ATTR.PROMPT_VARIABLE_PREFIX}${name}`,
          (variables as Record<string, unknown>)[name],
          "string",
          true,
        );
      }
    }
  }

  protected override get hasDetailsEvent(): boolean {
    return true;
  }

  recordSystemInstructions(parts: readonly MessagePart[]): void {
    this.recordContent(ATTR.SYSTEM_INSTRUCTIONS, parts, "parts", true);
  }

  recordInputMessages(messages: readonly InputMessage[]): void {
    this.recordContent(ATTR.INPUT_MESSAGES, messages, "input_messages", true);
  }

  /** Tool schemas: recorded in every capture mode, still redacted and bounded. */
  recordToolDefinitions(definitions: readonly unknown[]): void {
    this.recordContent(ATTR.TOOL_DEFINITIONS, definitions, "json", false);
  }

  recordOutputMessages(messages: readonly OutputMessage[]): void {
    if (this.ended) return;
    this.#outputRecorded = true;
    this.recordContent(ATTR.OUTPUT_MESSAGES, messages, "output_messages", true);
  }

  /**
   * Marks one delivered output chunk carrying new non-empty output. It never sets or influences
   * first-token time. The first chunk records time to first chunk; each later chunk records the
   * time since the previous call (clamped at zero). Points are recorded immediately.
   */
  recordChunk(atMs?: number): void {
    if (this.ended) return;
    const at = this.time(atMs);
    if (this.#firstChunkMs === undefined) {
      this.#firstChunkMs = at;
      this.core.record(
        "gen_ai.client.inference.time_to_first_chunk",
        elapsedSeconds(this.startMs, at),
        { ...this.dims },
      );
    } else {
      this.core.record(
        "gen_ai.client.inference.time_per_output_chunk",
        elapsedSeconds(this.#previousChunkMs ?? at, at),
        { ...this.dims },
      );
    }
    this.#previousChunkMs = at;
    this.#chunkCount += 1;
  }

  /**
   * Records the runtime-measured time of the first output token. Pass only measured timestamps,
   * never a chunk-callback time used as an estimate. The first call wins.
   */
  recordFirstToken(atMs?: number): void {
    if (this.ended || this.#firstTokenMs !== undefined) return;
    const at = this.time(atMs);
    this.#firstTokenMs = at;
    this.core.record(
      "xybrid.genai.client.inference.time_to_first_token",
      elapsedSeconds(this.startMs, at),
      { ...this.dims },
    );
  }

  /** A runtime-measured mean inter-token time; non-finite values are ignored. */
  recordTokenTiming(timing: { readonly timePerOutputTokenS: number }): void {
    this.setAttribute(
      ATTR.RESPONSE_TIME_PER_OUTPUT_TOKEN,
      finite(timing?.timePerOutputTokenS, ATTR.RESPONSE_TIME_PER_OUTPUT_TOKEN),
    );
  }

  /** Finish reasons; an empty list is absent and does not count as recorded. */
  recordFinishReasons(reasons: readonly string[]): void {
    if (this.ended) return;
    const values = strings(reasons);
    if (values === undefined) return;
    this.#finishReasonsRecorded = true;
    this.setAttribute(ATTR.RESPONSE_FINISH_REASONS, values);
  }

  /** The response id and model; the model joins the metric dimensions (last value wins). */
  recordResponse(response: { readonly responseId?: string; readonly model?: string }): void {
    if (this.ended || typeof response !== "object" || response === null) return;
    const model = text(response.model);
    this.setAttributes({
      [ATTR.RESPONSE_ID]: text(response.responseId),
      [ATTR.RESPONSE_MODEL]: model,
    });
    if (model !== undefined) this.dims[ATTR.RESPONSE_MODEL] = model;
  }

  /** Starts a child attempt span; the logical span keeps the request totals. */
  startAttempt(info: AttemptInfo, startTimeMs?: number): AttemptOperation {
    this.#attempts += 1;
    const parentContext =
      this.span === undefined ? undefined : traceApi.setSpan(contextApi.active(), this.span);
    return new AttemptOperation(
      this.core,
      String(this.#request.operation),
      info,
      startTimeMs,
      this.suppressed || this.span === undefined,
      parentContext,
    );
  }

  protected override beforeEnd(): void {
    if (this.errorType !== undefined && this.#outputRecorded && !this.#finishReasonsRecorded) {
      this.setDerived(ATTR.RESPONSE_FINISH_REASONS, ["error"]);
    }
    if (this.#firstChunkMs !== undefined) {
      this.setDerived(
        ATTR.RESPONSE_TIME_TO_FIRST_CHUNK,
        elapsedSeconds(this.startMs, this.#firstChunkMs),
      );
    }
    if (this.#firstTokenMs !== undefined) {
      this.setDerived(
        ATTR.RESPONSE_TIME_TO_FIRST_TOKEN,
        elapsedSeconds(this.startMs, this.#firstTokenMs),
      );
    }
    if (this.#chunkCount > 0) this.setDerived(ATTR.RESPONSE_CHUNK_COUNT, this.#chunkCount);
    if (this.#attempts > 0) this.setDerived(ATTR.ATTEMPT_COUNT, this.#attempts);
  }

  protected recordMetrics(endMs: number): void {
    const withError = this.dimsWithError();
    this.core.record(this.#durationMetric, this.durationSeconds(endMs), withError);
    const usage = this.usage;
    if (this.#category === "embeddings" && usage.inputTokens !== undefined) {
      this.core.record(
        "gen_ai.client.embeddings.operation.input_tokens",
        usage.inputTokens,
        withError,
      );
    }
    if (this.#category === "inference") this.#recordUsageMetrics({ ...this.dims }, withError);
  }

  /** Token histograms (with `error.type`) and per-modality usage counters (without). */
  #recordUsageMetrics(common: otel.Attributes, withError: otel.Attributes): void {
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
        const value = counts[modality];
        if (value === undefined || value <= 0) continue;
        covered += value;
        this.core.record(metric, value, { ...common, [ATTR.TOKEN_MODALITY]: modality });
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
      const only = Array.isArray(modalities) && modalities.length === 1 ? modalities[0] : undefined;
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
   * event. Both are emitted for suppressed operations too, without a trace context.
   */
  protected override emitEvents(
    endMs: number,
    error: ErrorInfo | undefined,
    message: string | undefined,
  ): void {
    const ctx = this.eventContext();
    if (error !== undefined && this.errorType !== "cancelled") {
      this.core.emit(
        "gen_ai.client.operation.exception",
        endMs,
        {
          [ATTR.EXCEPTION_TYPE]: this.errorType,
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
        ...this.eventContent(),
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
  constructor(core: ObserverCore, call: ToolCall, state: StartState) {
    const name = text(call.name);
    const toolType = text(call.toolType);
    super(
      core,
      spanName(OPERATIONS.execute_tool.span_name, name, "execute_tool"),
      SpanKind.INTERNAL,
      {
        [ATTR.OPERATION_NAME]: "execute_tool",
        [ATTR.TOOL_NAME]: name,
        [ATTR.TOOL_CALL_ID]: text(call.callId),
        [ATTR.TOOL_TYPE]: toolType,
        [ATTR.TOOL_DESCRIPTION]: text(call.description),
      },
      {
        [ATTR.OPERATION_NAME]: "execute_tool",
        [ATTR.TOOL_NAME]: name,
        [ATTR.TOOL_TYPE]: toolType,
      },
      state,
    );
    if (call.arguments !== undefined) {
      this.recordContent(ATTR.TOOL_CALL_ARGUMENTS, call.arguments, "json", true);
    }
  }

  protected override get hasDetailsEvent(): boolean {
    return true;
  }

  recordResult(value: unknown): void {
    this.recordContent(ATTR.TOOL_CALL_RESULT, value, "json", true);
  }

  protected override emitEvents(endMs: number): void {
    const content = this.eventContent();
    if (!this.core.eventContent || Object.keys(content).length === 0) return;
    this.core.emit(
      "gen_ai.execute_tool.operation.details",
      endMs,
      {
        ...pick(this.attributes, [
          ATTR.OPERATION_NAME,
          ATTR.TOOL_NAME,
          ATTR.TOOL_TYPE,
          ATTR.ERROR_TYPE,
        ]),
        ...content,
        ...(this.truncated.length === 0
          ? {}
          : { [ATTR.CONTENT_TRUNCATED_ATTRIBUTES]: [...this.truncated] }),
      },
      this.eventContext(),
    );
  }

  protected recordMetrics(endMs: number): void {
    this.core.record(
      "gen_ai.execute_tool.duration",
      this.durationSeconds(endMs),
      this.dimsWithError(),
    );
  }
}

/** A multi-step workflow (agent loop, pipeline) that parents inferences and tools. */
export class WorkflowOperation extends Operation {
  /** @internal */
  constructor(core: ObserverCore, name: string, state: StartState) {
    const workflow = text(name);
    super(
      core,
      spanName(OPERATIONS.invoke_workflow.span_name, workflow, "invoke_workflow"),
      SpanKind.INTERNAL,
      { [ATTR.OPERATION_NAME]: "invoke_workflow", [ATTR.WORKFLOW_NAME]: workflow },
      { [ATTR.OPERATION_NAME]: "invoke_workflow", [ATTR.WORKFLOW_NAME]: workflow },
      state,
    );
  }

  protected recordMetrics(endMs: number): void {
    this.core.record(
      "gen_ai.invoke_workflow.duration",
      this.durationSeconds(endMs),
      this.dimsWithError(),
    );
  }
}

/** Loading a model into a runtime. */
export class ModelLoadOperation extends Operation {
  /** @internal */
  constructor(core: ObserverCore, info: ModelLoadInfo, state: StartState) {
    const model = text(info.model);
    // The runtime serves a local load when no provider is named; never "unknown".
    const provider = text(info.provider) ?? text(info.runtime);
    super(
      core,
      spanName(OPERATIONS.load_model.span_name, model, "load_model"),
      SpanKind.INTERNAL,
      {
        [ATTR.OPERATION_NAME]: "load_model",
        [ATTR.REQUEST_MODEL]: model,
        [ATTR.PROVIDER_NAME]: provider,
        [ATTR.RUNTIME_NAME]: text(info.runtime),
        [ATTR.MODEL_FORMAT]: text(info.format),
        [ATTR.MODEL_SIZE_BYTES]: count(info.sizeBytes, ATTR.MODEL_SIZE_BYTES),
      },
      {
        [ATTR.OPERATION_NAME]: "load_model",
        [ATTR.PROVIDER_NAME]: provider,
        [ATTR.REQUEST_MODEL]: model,
      },
      state,
    );
  }

  recordCacheHit(hit: boolean): void {
    if (typeof hit === "boolean") this.setAttribute(ATTR.MODEL_CACHE_HIT, hit);
  }

  recordResource(sample: ResourceSample): void {
    if (typeof sample !== "object" || sample === null) return;
    for (const [field, key, check] of RESOURCE_ATTRIBUTES) {
      this.setAttribute(key, checked(sample[field], check, key));
    }
  }

  protected recordMetrics(endMs: number): void {
    this.core.record(
      "gen_ai.client.operation.duration",
      this.durationSeconds(endMs),
      this.dimsWithError(),
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
  constructor(core: ObserverCore, name: string, attributes: Attributes, state: StartState) {
    super(core, String(name), SpanKind.INTERNAL, { ...attributes }, {}, state);
  }

  protected recordMetrics(): void {}
}

/** The record context of an evaluation: the reference's ids with trace flags 0. */
const refContext = (target: ObservationRef): Context =>
  traceApi.setSpanContext(ROOT_CONTEXT, {
    traceId: target.trace_id,
    spanId: target.span_id,
    traceFlags: TraceFlags.NONE,
    isRemote: true,
  });

/**
 * Records GenAI operations as OpenTelemetry spans, metrics and events through
 * application-owned providers. Cheap to share; never installs globals; hot paths never do I/O.
 */
export class Observer {
  readonly #core: ObserverCore;

  /**
   * Validates options: an invalid content capture or budget throws `RangeError`; invalid
   * redaction rules, redactor or clock throw `TypeError`.
   */
  constructor(options: ObserverOptions = {}) {
    this.#core = new ObserverCore(options);
  }

  /** The effective content capture mode (lowercase). */
  get contentCapture(): ContentCapture {
    return this.#core.capture;
  }

  /**
   * Captures the active application context for a later callback invocation.
   * This binds the callback itself; it does not propagate context into native Promise continuations.
   */
  bindContext<Args extends unknown[], Result>(
    callback: (...args: Args) => Result,
  ): (...args: Args) => Result {
    const captured = contextApi.active();
    return (...args: Args) => contextApi.with(captured, callback, undefined, ...args);
  }

  /** Starts an inference. An invalid explicit parent throws `TypeError` before any work. */
  startInference(request: InferenceRequest, options: StartOptions = {}): InferenceOperation {
    return new InferenceOperation(this.#core, request, this.#core.start(options));
  }

  startWorkflow(name: string, options: StartOptions = {}): WorkflowOperation {
    return new WorkflowOperation(this.#core, name, this.#core.start(options));
  }

  startTool(call: ToolCall, options: StartOptions = {}): ToolOperation {
    return new ToolOperation(this.#core, call, this.#core.start(options));
  }

  startModelLoad(info: ModelLoadInfo, options: StartOptions = {}): ModelLoadOperation {
    return new ModelLoadOperation(this.#core, info, this.#core.start(options));
  }

  /** A generic internal span (no GenAI semantics, no metrics) for runtime stages. */
  startSpan(name: string, attributes: Attributes = {}, options: StartOptions = {}): SpanOperation {
    const state = this.#core.start(options);
    return new SpanOperation(this.#core, name, attributes, state);
  }

  /**
   * Emits `gen_ai.evaluation.result` linked to `target` (trace flags 0). Returns `true` only when
   * an enabled logger received it; `false` (recording nothing) without one or for an invalid
   * reference. References of unsampled spans are valid.
   */
  recordEvaluation(target: ObservationRef, evaluation: Evaluation): boolean {
    const ref = toObservationRef(target);
    if (ref === undefined || typeof evaluation !== "object" || evaluation === null) return false;
    const attributes: Record<string, AnyValue> = {
      [ATTR.EVALUATION_NAME]: String(evaluation.name),
    };
    const score = finite(evaluation.scoreValue, ATTR.EVALUATION_SCORE_VALUE);
    if (score !== undefined) attributes[ATTR.EVALUATION_SCORE_VALUE] = score;
    const label = text(evaluation.scoreLabel);
    if (label !== undefined) attributes[ATTR.EVALUATION_SCORE_LABEL] = label;
    const explanation = text(evaluation.explanation);
    if (explanation !== undefined) attributes[ATTR.EVALUATION_EXPLANATION] = explanation;
    const responseId = text(evaluation.responseId);
    if (responseId !== undefined) attributes[ATTR.RESPONSE_ID] = responseId;
    return this.#core.emit(
      "gen_ai.evaluation.result",
      this.#core.wallNow(),
      attributes,
      refContext(ref),
    );
  }

  /**
   * User feedback as `gen_ai.evaluation.result` named `user_feedback`. The comment is governed
   * content (capture mode, redaction, rules, `K = B` truncation). Returns as `recordEvaluation`.
   */
  recordFeedback(target: ObservationRef, feedback: Feedback): boolean {
    const ref = toObservationRef(target);
    if (ref === undefined || typeof feedback !== "object" || feedback === null) return false;
    const core = this.#core;
    const attributes: Record<string, AnyValue> = { [ATTR.EVALUATION_NAME]: "user_feedback" };
    const score = finite(feedback.score, ATTR.EVALUATION_SCORE_VALUE);
    if (score !== undefined) attributes[ATTR.EVALUATION_SCORE_VALUE] = score;
    const label = text(feedback.label);
    if (label !== undefined) attributes[ATTR.EVALUATION_SCORE_LABEL] = label;
    if (feedback.comment !== undefined && core.capture !== "no_content") {
      const outcome = core.content.process(ATTR.FEEDBACK_COMMENT, feedback.comment, "string");
      if (outcome.kind === "value" && typeof outcome.value === "string") {
        attributes[ATTR.FEEDBACK_COMMENT] = truncateString(outcome.value, core.budget);
      }
    }
    return core.emit("gen_ai.evaluation.result", core.wallNow(), attributes, refContext(ref));
  }
}
