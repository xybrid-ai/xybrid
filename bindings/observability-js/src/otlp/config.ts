/**
 * OTLP pipeline and exporter options: types, construction-time validation (CONTRACT.md
 * "Configuration sources") and the standard `OTEL_*` environment variables.
 *
 * Everything here is pure: validation runs before any provider, processor, timer,
 * `AbortController` or connection exists, so an invalid option can never leave work behind.
 */
import { type Attributes, diag } from "@opentelemetry/api";
import { SDK_INFO } from "@opentelemetry/core";
import { DEFAULTS, ENV, PRESETS } from "../catalog.ts";

export type OtlpProtocol = "http/protobuf" | "http/json";

/**
 * Where a signal's headers come from. `merge_environment` (the default) sends
 * `OTEL_EXPORTER_OTLP_HEADERS`, overlaid by `OTEL_EXPORTER_OTLP_{SIGNAL}_HEADERS`, overlaid by
 * the programmatic headers (keys compared case-insensitively, environment read once at
 * construction). `isolated` sends the programmatic headers only.
 */
export type HeaderPolicy = "merge_environment" | "isolated";

/** An environment map. Values are strings; `undefined` and empty values are unset. */
export type OtlpEnvironment = Readonly<Record<string, string | undefined>>;

/** One OTLP/HTTP signal destination. */
export type OtlpSignalOptions = {
  /** Full per-signal URL, e.g. `http://localhost:4318/v1/traces`. */
  readonly endpoint: string;
  readonly headers?: Readonly<Record<string, string>> | undefined;
  readonly protocol?: OtlpProtocol | undefined;
  /** Per-attempt request timeout in milliseconds (default 10000). */
  readonly timeoutMs?: number | undefined;
  /** Default `merge_environment`; the Xybrid preset uses `isolated`. */
  readonly headerPolicy?: HeaderPolicy | undefined;
};

export type RetryOptions = {
  /** Retries after the first attempt (default 2; 0 disables retries). */
  readonly maxRetries?: number | undefined;
  /** First backoff in milliseconds (default 200). */
  readonly initialDelayMs?: number | undefined;
  /** Backoff cap in milliseconds (default 2000; at least `initialDelayMs`). */
  readonly maxDelayMs?: number | undefined;
};

/** Span and log batching, and the per-signal byte budget (defaults from the catalog). */
export type BatchOptions = {
  /** Queued plus in-flight items per signal (default 2048). */
  readonly maxQueueSize?: number | undefined;
  /** Items per export (default 256); the effective value is `min(batch, queue)`. */
  readonly maxExportBatchSize?: number | undefined;
  readonly scheduledDelayMs?: number | undefined;
  /** One absolute deadline per export, shared by splits, retries and backoff (default 10000). */
  readonly exportTimeoutMs?: number | undefined;
  /** Queued plus in-flight bytes per signal, conservatively estimated (default 32 MiB). */
  readonly maxBufferedBytes?: number | undefined;
};

export type ResourceOptions = {
  readonly serviceName: string;
  readonly serviceVersion?: string | undefined;
  readonly attributes?: Readonly<Record<string, string | number | boolean>> | undefined;
};

export type OtlpPipelineOptions = {
  readonly traces?: OtlpSignalOptions | undefined;
  readonly metrics?: OtlpSignalOptions | undefined;
  readonly logs?: OtlpSignalOptions | undefined;
  readonly resource?: ResourceOptions | undefined;
  readonly batch?: BatchOptions | undefined;
  readonly metricExportIntervalMs?: number | undefined;
  readonly retry?: RetryOptions | undefined;
  /** Upper bound of one encoded request body (default 480 KiB). */
  readonly maxRequestBodyBytes?: number | undefined;
  /**
   * Environment used for header merging and resource variables: `undefined` reads
   * `process.env` when it exists, `null` ignores the environment.
   */
  readonly environment?: OtlpEnvironment | null | undefined;
};

export type Signal = "traces" | "metrics" | "logs";

/** A validated signal destination with its final header set. */
export type ResolvedSignal = {
  readonly signal: Signal;
  readonly url: string;
  readonly protocol: OtlpProtocol;
  readonly timeoutMs: number;
  /** Exactly what the final transport sends, `Content-Type` included. */
  readonly headers: Readonly<Record<string, string>>;
};

/** Validated numeric limits. */
export type ResolvedLimits = {
  readonly maxQueueSize: number;
  readonly maxExportBatchSize: number;
  readonly scheduledDelayMs: number;
  readonly exportTimeoutMs: number;
  readonly maxBufferedBytes: number;
  readonly maxRetries: number;
  readonly initialDelayMs: number;
  readonly maxDelayMs: number;
  readonly maxRequestBodyBytes: number;
  readonly maxResponseBodyBytes: number;
  readonly metricExportIntervalMs: number;
  /** `min(exportTimeoutMs, metricExportIntervalMs)`. */
  readonly metricExportTimeoutMs: number;
};

export type ResolvedOptions = {
  readonly traces: ResolvedSignal | undefined;
  readonly metrics: ResolvedSignal | undefined;
  readonly logs: ResolvedSignal | undefined;
  readonly limits: ResolvedLimits;
  /** Resource attributes to merge over the SDK default resource. */
  readonly resourceAttributes: Attributes;
};

/** Largest delay `setTimeout` honours; larger values fire immediately in Node. */
const MAX_TIMER_MS = 2 ** 31 - 1;

const integerOption = (
  name: string,
  value: unknown,
  fallback: number,
  min: number,
  max: number = Number.MAX_SAFE_INTEGER,
): number => {
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(
      `${name} must be an integer between ${min} and ${max}${typeof value === "number" ? "" : " (got a non-number)"}.`,
    );
  }
  return value;
};

const msOption = (name: string, value: unknown, fallback: number, min = 1): number =>
  integerOption(name, value, fallback, min, MAX_TIMER_MS);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Validates an endpoint: an absolute `http`/`https` URL without credentials. */
const endpointUrl = (name: string, value: unknown): string => {
  if (typeof value !== "string") throw new TypeError(`${name} must be a URL string.`);
  let url: URL;
  try {
    url = new URL(value.trim());
  } catch {
    throw new TypeError(`${name} must be an absolute http or https URL.`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new TypeError(`${name} must be an absolute http or https URL.`);
  }
  if (url.username !== "" || url.password !== "") {
    throw new TypeError(`${name} must not carry credentials; use headers.`);
  }
  return url.href;
};

/** `true` when `fetch` accepts the header name and value. */
const validHeader = (name: string, value: string): boolean => {
  if (name.length === 0) return false;
  try {
    new Headers().append(name, value);
    return true;
  } catch {
    return false;
  }
};

const programmaticHeaders = (name: string, value: unknown): [string, string][] => {
  if (value === undefined) return [];
  if (!isRecord(value)) throw new TypeError(`${name} must be an object of strings.`);
  const entries: [string, string][] = [];
  for (const key of Object.keys(value)) {
    const header = value[key];
    if (typeof header !== "string" || !validHeader(key, header)) {
      // Header names are configuration, values may be secrets: name the key only.
      throw new TypeError(`${name}["${key}"] is not a valid HTTP header.`);
    }
    entries.push([key, header]);
  }
  return entries;
};

/**
 * `key=value` pairs separated by commas, trimmed and percent-decoded. A malformed pair or
 * escape skips that pair; diagnostics name the variable, never the value.
 */
export const parseKeyValueList = (
  value: string | undefined,
  variable: string,
): [string, string][] => {
  const pairs: [string, string][] = [];
  if (value === undefined) return pairs;
  for (const raw of value.split(",")) {
    if (raw.trim() === "") continue;
    const equals = raw.indexOf("=");
    if (equals <= 0) {
      diag.warn(`xybrid-observability: skipped a malformed entry in ${variable}`);
      continue;
    }
    try {
      const key = decodeURIComponent(raw.slice(0, equals).trim());
      const decoded = decodeURIComponent(raw.slice(equals + 1).trim());
      if (key.length === 0) {
        diag.warn(`xybrid-observability: skipped a malformed entry in ${variable}`);
        continue;
      }
      pairs.push([key, decoded]);
    } catch {
      diag.warn(`xybrid-observability: skipped a malformed entry in ${variable}`);
    }
  }
  return pairs;
};

/** The process environment, or `undefined` in browsers and restricted runtimes. */
const processEnvironment = (): OtlpEnvironment | undefined => {
  try {
    const env = (globalThis as { process?: { env?: unknown } }).process?.env;
    return isRecord(env) ? (env as OtlpEnvironment) : undefined;
  } catch {
    return undefined;
  }
};

/** Reads one variable: trimmed, empty means unset; a throwing environment reads as unset. */
const readVariable = (env: OtlpEnvironment | undefined, name: string): string | undefined => {
  if (env === undefined) return undefined;
  let value: unknown;
  try {
    value = env[name];
  } catch {
    return undefined;
  }
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed === "" ? undefined : trimmed;
};

const SIGNALS = [
  ["traces", "TRACES", "v1/traces"],
  ["metrics", "METRICS", "v1/metrics"],
  ["logs", "LOGS", "v1/logs"],
] as const;

const signalVariable = (template: string, upper: string): string =>
  template.replace("{SIGNAL}", upper);

const upperOf = (signal: Signal): string =>
  signal === "traces" ? "TRACES" : signal === "metrics" ? "METRICS" : "LOGS";

/** Header names compared case-insensitively; a later source replaces an earlier entry. */
const mergeHeaders = (...sources: [string, string][][]): Map<string, [string, string]> => {
  const merged = new Map<string, [string, string]>();
  for (const source of sources) {
    for (const [name, value] of source) merged.set(name.toLowerCase(), [name, value]);
  }
  return merged;
};

const environmentHeaders = (
  env: OtlpEnvironment | undefined,
  variable: string,
): [string, string][] =>
  parseKeyValueList(readVariable(env, variable), variable).filter(([name, value]) => {
    if (validHeader(name, value)) return true;
    diag.warn(`xybrid-observability: skipped an invalid header in ${variable}`);
    return false;
  });

const resolveSignal = (
  signal: Signal,
  options: unknown,
  env: OtlpEnvironment | undefined,
): ResolvedSignal | undefined => {
  if (options === undefined) return undefined;
  if (!isRecord(options)) throw new TypeError(`${signal} must be an object with an endpoint.`);
  const url = endpointUrl(`${signal}.endpoint`, options["endpoint"]);
  const protocol = options["protocol"];
  if (protocol !== undefined && protocol !== "http/protobuf" && protocol !== "http/json") {
    throw new TypeError(`${signal}.protocol must be "http/protobuf" or "http/json".`);
  }
  const policy = options["headerPolicy"];
  if (policy !== undefined && policy !== "merge_environment" && policy !== "isolated") {
    throw new TypeError(`${signal}.headerPolicy must be "merge_environment" or "isolated".`);
  }
  const timeoutMs = msOption(
    `${signal}.timeoutMs`,
    options["timeoutMs"],
    DEFAULTS.batch.export_timeout_ms,
  );
  const programmatic = programmaticHeaders(`${signal}.headers`, options["headers"]);
  const upper = upperOf(signal);
  const merged =
    policy === "isolated"
      ? mergeHeaders(programmatic)
      : mergeHeaders(
          environmentHeaders(env, ENV.otlp_headers),
          environmentHeaders(env, signalVariable(ENV.otlp_signal_headers, upper)),
          programmatic,
        );
  // Transport-owned headers always win.
  merged.delete("content-type");
  const json = protocol === "http/json";
  const headers: Record<string, string> = {};
  for (const [name, value] of merged.values()) headers[name] = value;
  headers["Content-Type"] = json ? "application/json" : "application/x-protobuf";
  return Object.freeze({
    signal,
    url,
    protocol: json ? "http/json" : "http/protobuf",
    timeoutMs,
    headers: Object.freeze(headers),
  });
};

const resolveLimits = (options: OtlpPipelineOptions): ResolvedLimits => {
  const batch: unknown = options.batch ?? {};
  const retry: unknown = options.retry ?? {};
  if (!isRecord(batch)) throw new TypeError("batch must be an object.");
  if (!isRecord(retry)) throw new TypeError("retry must be an object.");
  const maxQueueSize = integerOption(
    "batch.maxQueueSize",
    batch["maxQueueSize"],
    DEFAULTS.batch.max_queue_size,
    1,
  );
  const maxExportBatchSize = integerOption(
    "batch.maxExportBatchSize",
    batch["maxExportBatchSize"],
    DEFAULTS.batch.max_export_batch_size,
    1,
  );
  const scheduledDelayMs = msOption(
    "batch.scheduledDelayMs",
    batch["scheduledDelayMs"],
    DEFAULTS.batch.scheduled_delay_ms,
  );
  const exportTimeoutMs = msOption(
    "batch.exportTimeoutMs",
    batch["exportTimeoutMs"],
    DEFAULTS.batch.export_timeout_ms,
  );
  const maxBufferedBytes = integerOption(
    "batch.maxBufferedBytes",
    batch["maxBufferedBytes"],
    DEFAULTS.max_buffered_bytes_per_signal,
    1,
  );
  const maxRetries = integerOption(
    "retry.maxRetries",
    retry["maxRetries"],
    DEFAULTS.retry.max_retries,
    0,
  );
  const initialDelayMs = msOption(
    "retry.initialDelayMs",
    retry["initialDelayMs"],
    DEFAULTS.retry.initial_delay_ms,
    0,
  );
  const maxDelayMs = msOption(
    "retry.maxDelayMs",
    retry["maxDelayMs"],
    Math.max(DEFAULTS.retry.max_delay_ms, initialDelayMs),
    0,
  );
  if (maxDelayMs < initialDelayMs) {
    throw new RangeError("retry.maxDelayMs must be at least retry.initialDelayMs.");
  }
  const maxRequestBodyBytes = integerOption(
    "maxRequestBodyBytes",
    options.maxRequestBodyBytes,
    DEFAULTS.max_request_body_bytes,
    1,
  );
  const metricExportIntervalMs = msOption(
    "metricExportIntervalMs",
    options.metricExportIntervalMs,
    DEFAULTS.metric_export_interval_ms,
  );
  return Object.freeze({
    maxQueueSize,
    maxExportBatchSize: Math.min(maxExportBatchSize, maxQueueSize),
    scheduledDelayMs,
    exportTimeoutMs,
    maxBufferedBytes,
    maxRetries,
    initialDelayMs,
    maxDelayMs,
    maxRequestBodyBytes,
    maxResponseBodyBytes: DEFAULTS.max_response_body_bytes,
    metricExportIntervalMs,
    metricExportTimeoutMs: Math.min(exportTimeoutMs, metricExportIntervalMs),
  });
};

const resolveResource = (
  resource: unknown,
  env: OtlpEnvironment | undefined,
): Readonly<Attributes> => {
  const attributes: Attributes = {};
  // Precedence: programmatic > OTEL_SERVICE_NAME > service.name in OTEL_RESOURCE_ATTRIBUTES.
  for (const [key, value] of parseKeyValueList(
    readVariable(env, ENV.resource_attributes),
    ENV.resource_attributes,
  )) {
    attributes[key] = value;
  }
  const serviceName = readVariable(env, ENV.service_name);
  if (serviceName !== undefined) attributes["service.name"] = serviceName;
  if (resource === undefined) return Object.freeze(attributes);
  if (!isRecord(resource) || typeof resource["serviceName"] !== "string") {
    throw new TypeError("resource.serviceName must be a string.");
  }
  const extra = resource["attributes"];
  if (extra !== undefined) {
    if (!isRecord(extra)) throw new TypeError("resource.attributes must be an object.");
    for (const key of Object.keys(extra)) {
      const value = extra[key];
      if (typeof value !== "string" && typeof value !== "number" && typeof value !== "boolean") {
        throw new TypeError(`resource.attributes["${key}"] must be a string, number or boolean.`);
      }
      attributes[key] = value;
    }
  }
  attributes["service.name"] = resource["serviceName"];
  const version = resource["serviceVersion"];
  if (version !== undefined) {
    if (typeof version !== "string")
      throw new TypeError("resource.serviceVersion must be a string.");
    attributes["service.version"] = version;
  }
  return Object.freeze(attributes);
};

/** The environment a build uses: explicit map, `process.env`, or none (`null`). */
const buildEnvironment = (
  environment: OtlpEnvironment | null | undefined,
): OtlpEnvironment | undefined => {
  if (environment === null) return undefined;
  if (environment === undefined) return processEnvironment();
  if (!isRecord(environment)) throw new TypeError("environment must be an object or null.");
  return environment;
};

/**
 * Validates every option and resolves headers and resource attributes. Throws `RangeError`
 * (numeric limits) or `TypeError` (endpoints, protocols, headers, shapes) naming the option.
 * Reads the environment once.
 */
export const resolveOtlpOptions = (options: OtlpPipelineOptions): ResolvedOptions => {
  // npm can accept conflicting optional peers. Reject an unsupported runtime before timers
  // or providers exist, including standalone exporters that do not instantiate a provider.
  if (!/^3\.(?:0|1)\./.test(SDK_INFO["telemetry.sdk.version"])) {
    throw new TypeError("@xybrid/observability/otlp requires OpenTelemetry SDK 3.0 or 3.1.");
  }
  if (!isRecord(options)) throw new TypeError("OTLP options must be an object.");
  const env = buildEnvironment(options.environment);
  const limits = resolveLimits(options);
  const traces = resolveSignal("traces", options.traces, env);
  const metrics = resolveSignal("metrics", options.metrics, env);
  const logs = resolveSignal("logs", options.logs, env);
  const resourceAttributes = resolveResource(options.resource, env);
  return Object.freeze({ traces, metrics, logs, limits, resourceAttributes });
};

const positiveInteger = (
  env: OtlpEnvironment | undefined,
  name: string,
  max: number = MAX_TIMER_MS,
): number | undefined => {
  const value = readVariable(env, name);
  if (value === undefined) return undefined;
  const parsed = /^\d+$/.test(value) ? Number(value) : Number.NaN;
  if (Number.isSafeInteger(parsed) && parsed >= 1 && parsed <= max) return parsed;
  diag.warn(`xybrid-observability: ignored an invalid ${name}`);
  return undefined;
};

const parseProtocol = (value: string | undefined, variable: string): OtlpProtocol | undefined => {
  if (value === undefined) return undefined;
  const protocol = value.toLowerCase();
  if (protocol === "http/json" || protocol === "http/protobuf") return protocol;
  diag.warn(
    `xybrid-observability: ${variable} is not http/json or http/protobuf; using http/protobuf`,
  );
  return "http/protobuf";
};

/**
 * Builds pipeline options from the standard variables: `OTEL_SDK_DISABLED`,
 * `OTEL_EXPORTER_OTLP_[{SIGNAL}_]ENDPOINT|PROTOCOL|TIMEOUT`, `OTEL_{SIGNAL}_EXPORTER=none`,
 * `OTEL_BSP_*` and `OTEL_METRIC_EXPORT_INTERVAL`. Headers (`OTEL_EXPORTER_OTLP_[{SIGNAL}_]HEADERS`)
 * and resource variables are applied when the pipeline is built, from the same environment.
 * Returns `{}` (no signals) in browsers, when `OTEL_SDK_DISABLED=true`, or when nothing is set.
 */
export const otlpConfigFromEnv = (env?: OtlpEnvironment): OtlpPipelineOptions => {
  const source = env ?? processEnvironment();
  if (readVariable(source, ENV.sdk_disabled)?.toLowerCase() === "true") return {};
  const base = readVariable(source, ENV.otlp_endpoint);
  const options: Record<string, unknown> = {};
  for (const [signal, upper, path] of SIGNALS) {
    const exporters = readVariable(source, signalVariable(ENV.signal_exporter, upper));
    if (
      exporters
        ?.split(",")
        .map((name) => name.trim().toLowerCase())
        .includes("none")
    ) {
      continue;
    }
    const specific = readVariable(source, signalVariable(ENV.otlp_signal_endpoint, upper));
    const endpoint =
      specific !== undefined
        ? specific
        : base !== undefined
          ? `${base.replace(/\/+$/, "")}/${path}`
          : undefined;
    if (endpoint === undefined) continue;
    const protocolVariable = signalVariable(ENV.otlp_signal_protocol, upper);
    const protocol =
      parseProtocol(readVariable(source, protocolVariable), protocolVariable) ??
      parseProtocol(readVariable(source, ENV.otlp_protocol), ENV.otlp_protocol);
    const timeoutMs =
      positiveInteger(source, signalVariable(ENV.otlp_signal_timeout, upper)) ??
      positiveInteger(source, ENV.otlp_timeout);
    options[signal] = {
      endpoint,
      protocol: protocol ?? "http/protobuf",
      ...(timeoutMs === undefined ? {} : { timeoutMs }),
    } satisfies OtlpSignalOptions;
  }
  const batch: Record<string, number> = {};
  const bsp = [
    ["maxQueueSize", ENV.bsp_max_queue_size, Number.MAX_SAFE_INTEGER],
    ["maxExportBatchSize", ENV.bsp_max_export_batch_size, Number.MAX_SAFE_INTEGER],
    ["scheduledDelayMs", ENV.bsp_schedule_delay, MAX_TIMER_MS],
    ["exportTimeoutMs", ENV.bsp_export_timeout, MAX_TIMER_MS],
  ] as const;
  for (const [key, name, max] of bsp) {
    const value = positiveInteger(source, name, max);
    if (value !== undefined) batch[key] = value;
  }
  if (Object.keys(batch).length > 0) options["batch"] = batch;
  const interval = positiveInteger(source, ENV.metric_export_interval);
  if (interval !== undefined) options["metricExportIntervalMs"] = interval;
  // Headers and resource variables are read again at build: keep the same source.
  if (env !== undefined) options["environment"] = env;
  return options as OtlpPipelineOptions;
};

/**
 * Traces only, to Xybrid's ingest endpoint with exactly `Authorization: Bearer <apiKey>`
 * (`headerPolicy: "isolated"`: no `OTEL_EXPORTER_OTLP_*` variable is read or merged). Metrics
 * and evaluation events are never sent there; route them through a Collector.
 */
export const xybridTracesPreset = (
  apiKey: string,
  options: { readonly endpoint?: string | undefined; readonly timeoutMs?: number | undefined } = {},
): OtlpPipelineOptions => {
  if (typeof apiKey !== "string" || apiKey.trim().length === 0) {
    throw new TypeError("xybridTracesPreset requires a non-empty API key.");
  }
  const preset = PRESETS.xybrid_traces;
  return {
    traces: {
      endpoint: options.endpoint ?? preset.traces_endpoint,
      headers: { [preset.auth_header]: `${preset.auth_scheme} ${apiKey.trim()}` },
      protocol: preset.protocol,
      headerPolicy: preset.header_policy,
      ...(options.timeoutMs === undefined ? {} : { timeoutMs: options.timeoutMs }),
    },
  };
};

/** Lifecycle call timeouts: a non-finite or negative value means the catalog default. */
export const lifecycleTimeout = (value: unknown, fallback: number): number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.min(value, MAX_TIMER_MS)
    : fallback;
