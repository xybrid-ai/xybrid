/**
 * `@xybrid/observability/otlp`: an application-owned OTLP/HTTP pipeline.
 *
 * Builds tracer, meter and logger providers on the OpenTelemetry JS SDK 3.x line without
 * registering any of them globally. The application creates the pipeline, hands
 * `pipeline.observer()` to instrumented code, and shuts the pipeline down.
 */

import { ExportResultCode } from "@opentelemetry/core";
import {
  JsonLogsSerializer,
  JsonMetricsSerializer,
  JsonTraceSerializer,
  ProtobufLogsSerializer,
  ProtobufMetricsSerializer,
  ProtobufTraceSerializer,
} from "@opentelemetry/otlp-transformer";
import { defaultResource, resourceFromAttributes } from "@opentelemetry/resources";
import {
  BatchLogRecordProcessor,
  LoggerProvider,
  type LogRecordExporter,
  type ReadableLogRecord,
} from "@opentelemetry/sdk-logs";
import {
  MeterProvider,
  PeriodicExportingMetricReader,
  type PushMetricExporter,
  type ResourceMetrics,
} from "@opentelemetry/sdk-metrics";
import type { ReadableSpan } from "@opentelemetry/sdk-trace";
import { AlwaysOnSampler, ParentBasedSampler, TracerProvider } from "@opentelemetry/sdk-trace";
import { DEFAULTS, PRESETS } from "../catalog.ts";
import { type ContentOptions, type LoggerProviderLike, Observer } from "../observer.ts";
import {
  OtlpHttpSender,
  type OtlpProtocol,
  type OtlpSignalOptions,
  type RetryOptions,
} from "./http.ts";
import {
  type BatchOptions,
  CountingBatchSpanProcessor,
  type ExportStats,
} from "./span-processor.ts";

export type { BatchOptions, ExportStats, OtlpProtocol, OtlpSignalOptions, RetryOptions };

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
  /** Span and log batching (defaults from the contract catalog). */
  readonly batch?: BatchOptions | undefined;
  readonly metricExportIntervalMs?: number | undefined;
  readonly retry?: RetryOptions | undefined;
  /** Upper bound of one request body; larger span/log batches are split (default 480 KiB). */
  readonly maxRequestBodyBytes?: number | undefined;
};

/** Resolves with `true` when `work` settled in time; never rejects. */
const bounded = async (work: Promise<unknown>, timeoutMs: number): Promise<boolean> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<false>((resolve) => {
    timer = setTimeout(() => resolve(false), timeoutMs);
    (timer as { unref?: () => void }).unref?.();
  });
  const done = work.then(
    () => true as const,
    () => true as const,
  );
  const result = await Promise.race([done, timeout]);
  clearTimeout(timer);
  return result;
};

class LogExporter implements LogRecordExporter {
  constructor(readonly sender: OtlpHttpSender<ReadableLogRecord[]>) {}

  export(
    logs: ReadableLogRecord[],
    resultCallback: (result: { code: ExportResultCode }) => void,
  ): void {
    void this.sender
      .sendItems(logs, (items) => items)
      .then(
        (outcome) =>
          resultCallback({
            code: outcome.failed > 0 ? ExportResultCode.FAILED : ExportResultCode.SUCCESS,
          }),
        () => resultCallback({ code: ExportResultCode.FAILED }),
      );
  }

  async forceFlush(): Promise<void> {}

  async shutdown(): Promise<void> {
    this.sender.abort();
  }
}

class MetricExporter implements PushMetricExporter {
  constructor(readonly sender: OtlpHttpSender<ResourceMetrics>) {}

  export(
    metrics: ResourceMetrics,
    resultCallback: (result: { code: ExportResultCode }) => void,
  ): void {
    void this.sender.sendWhole(metrics).then(
      (ok) => resultCallback({ code: ok ? ExportResultCode.SUCCESS : ExportResultCode.FAILED }),
      () => resultCallback({ code: ExportResultCode.FAILED }),
    );
  }

  async forceFlush(): Promise<void> {}

  async shutdown(): Promise<void> {
    this.sender.abort();
  }
}

/** Providers and delivery state of one OTLP pipeline. Never registered globally. */
export class OtlpPipeline {
  readonly tracerProvider: TracerProvider | undefined;
  readonly meterProvider: MeterProvider | undefined;
  readonly loggerProvider: LoggerProvider | undefined;
  readonly #spans: CountingBatchSpanProcessor | undefined;
  readonly #senders: OtlpHttpSender<never>[] = [];
  #shutdown: Promise<boolean> | undefined;

  constructor(options: OtlpPipelineOptions) {
    const resource = defaultResource().merge(
      resourceFromAttributes({
        ...(options.resource?.attributes ?? {}),
        ...(options.resource === undefined ? {} : { "service.name": options.resource.serviceName }),
        ...(options.resource?.serviceVersion === undefined
          ? {}
          : { "service.version": options.resource.serviceVersion }),
      }),
    );
    const batch = options.batch ?? {};
    const body = options.maxRequestBodyBytes ?? DEFAULTS.max_request_body_bytes;
    if (options.traces !== undefined) {
      const sender = new OtlpHttpSender<ReadableSpan[]>(
        "traces",
        options.traces,
        { protobuf: ProtobufTraceSerializer, json: JsonTraceSerializer },
        options.retry,
        body,
      );
      this.#senders.push(sender as OtlpHttpSender<never>);
      this.#spans = new CountingBatchSpanProcessor(sender, batch);
      this.tracerProvider = new TracerProvider({
        resource,
        sampler: new ParentBasedSampler({ root: new AlwaysOnSampler() }),
        spanProcessors: [this.#spans],
      });
    }
    if (options.metrics !== undefined) {
      const sender = new OtlpHttpSender<ResourceMetrics>(
        "metrics",
        options.metrics,
        { protobuf: ProtobufMetricsSerializer, json: JsonMetricsSerializer },
        options.retry,
        body,
      );
      this.#senders.push(sender as OtlpHttpSender<never>);
      const interval = options.metricExportIntervalMs ?? DEFAULTS.metric_export_interval_ms;
      this.meterProvider = new MeterProvider({
        resource,
        readers: [
          new PeriodicExportingMetricReader({
            exporter: new MetricExporter(sender),
            exportIntervalMillis: interval,
            exportTimeoutMillis: Math.min(
              batch.exportTimeoutMs ?? DEFAULTS.batch.export_timeout_ms,
              interval,
            ),
          }),
        ],
      });
    }
    if (options.logs !== undefined) {
      const sender = new OtlpHttpSender<ReadableLogRecord[]>(
        "logs",
        options.logs,
        { protobuf: ProtobufLogsSerializer, json: JsonLogsSerializer },
        options.retry,
        body,
      );
      this.#senders.push(sender as OtlpHttpSender<never>);
      const maxQueueSize = batch.maxQueueSize ?? DEFAULTS.batch.max_queue_size;
      this.loggerProvider = new LoggerProvider({
        resource,
        processors: [
          new BatchLogRecordProcessor({
            exporter: new LogExporter(sender),
            maxQueueSize,
            maxExportBatchSize: Math.min(
              batch.maxExportBatchSize ?? DEFAULTS.batch.max_export_batch_size,
              maxQueueSize,
            ),
            scheduledDelayMillis: batch.scheduledDelayMs ?? DEFAULTS.batch.scheduled_delay_ms,
            exportTimeoutMillis: batch.exportTimeoutMs ?? DEFAULTS.batch.export_timeout_ms,
          }),
        ],
      });
    }
  }

  /**
   * An {@link Observer} over this pipeline's providers. Signals the pipeline does not export are
   * disabled rather than read from globals, so evaluation events return `false` on a traces-only
   * pipeline.
   */
  observer(content?: ContentOptions): Observer {
    return new Observer({
      tracerProvider: this.tracerProvider ?? null,
      meterProvider: this.meterProvider ?? null,
      loggerProvider: (this.loggerProvider as LoggerProviderLike | undefined) ?? null,
      ...(content === undefined ? {} : { content }),
    });
  }

  stats(): ExportStats {
    return (
      this.#spans?.stats() ?? {
        spansQueued: 0,
        spansExported: 0,
        spansDropped: 0,
        exportFailures: 0,
      }
    );
  }

  /** Exports what is queued. Bounded; resolves `false` on timeout; immediate after shutdown. */
  async forceFlush(timeoutMs: number = DEFAULTS.flush_timeout_ms): Promise<boolean> {
    if (this.#shutdown !== undefined) return true;
    return bounded(
      Promise.allSettled([
        this.tracerProvider?.forceFlush(),
        this.meterProvider?.forceFlush(),
        this.loggerProvider?.forceFlush(),
      ]),
      timeoutMs,
    );
  }

  /** Flushes and stops every provider once. Bounded; later calls return the first result. */
  shutdown(timeoutMs: number = DEFAULTS.shutdown_timeout_ms): Promise<boolean> {
    this.#shutdown ??= (async () => {
      const completed = await bounded(
        Promise.allSettled([
          this.tracerProvider?.shutdown(),
          this.meterProvider?.shutdown(),
          this.loggerProvider?.shutdown(),
        ]),
        timeoutMs,
      );
      // Whatever is still in flight (retries, slow endpoints) is abandoned, not awaited.
      for (const sender of this.#senders) sender.abort();
      return completed;
    })();
    return this.#shutdown;
  }
}

export const createOtlpPipeline = (options: OtlpPipelineOptions): OtlpPipeline =>
  new OtlpPipeline(options);

/**
 * Traces only, to Xybrid's ingest endpoint with `Authorization: Bearer <apiKey>`. Metrics and
 * evaluation events are never sent there; route them through a Collector.
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
      protocol: "http/protobuf",
      ...(options.timeoutMs === undefined ? {} : { timeoutMs: options.timeoutMs }),
    },
  };
};

type Env = Readonly<Record<string, string | undefined>>;

const parseHeaders = (value: string | undefined): Record<string, string> => {
  const headers: Record<string, string> = {};
  for (const pair of (value ?? "").split(",")) {
    const equals = pair.indexOf("=");
    if (equals <= 0) continue;
    try {
      const key = decodeURIComponent(pair.slice(0, equals).trim());
      if (key.length > 0) headers[key] = decodeURIComponent(pair.slice(equals + 1).trim());
    } catch {
      // A malformed percent-escape drops that header, as the specification allows.
    }
  }
  return headers;
};

const parsePositiveInt = (value: string | undefined): number | undefined => {
  if (value === undefined || value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : undefined;
};

const SIGNALS = [
  ["traces", "TRACES", "v1/traces"],
  ["metrics", "METRICS", "v1/metrics"],
  ["logs", "LOGS", "v1/logs"],
] as const;

/**
 * Reads the standard `OTEL_EXPORTER_OTLP_*`, `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`,
 * `OTEL_BSP_*`, `OTEL_METRIC_EXPORT_INTERVAL` and `OTEL_{TRACES,METRICS,LOGS}_EXPORTER=none`
 * variables. Returns `{}` (no signals) in browsers or when nothing is configured.
 */
export const otlpConfigFromEnv = (env?: Env): OtlpPipelineOptions => {
  const source =
    env ?? (globalThis as { process?: { env?: Env } }).process?.env ?? ({} satisfies Env);
  const base = source["OTEL_EXPORTER_OTLP_ENDPOINT"]?.trim();
  const sharedHeaders = parseHeaders(source["OTEL_EXPORTER_OTLP_HEADERS"]);
  const sharedProtocol = source["OTEL_EXPORTER_OTLP_PROTOCOL"]?.trim();
  const sharedTimeout = parsePositiveInt(source["OTEL_EXPORTER_OTLP_TIMEOUT"]);
  const options: Record<string, unknown> = {};
  for (const [signal, upper, path] of SIGNALS) {
    if (source[`OTEL_${upper}_EXPORTER`]?.trim().toLowerCase() === "none") continue;
    const specific = source[`OTEL_EXPORTER_OTLP_${upper}_ENDPOINT`]?.trim();
    const endpoint =
      specific !== undefined && specific !== ""
        ? specific
        : base !== undefined && base !== ""
          ? `${base.replace(/\/+$/, "")}/${path}`
          : undefined;
    if (endpoint === undefined) continue;
    const protocol = source[`OTEL_EXPORTER_OTLP_${upper}_PROTOCOL`]?.trim() ?? sharedProtocol;
    const timeoutMs =
      parsePositiveInt(source[`OTEL_EXPORTER_OTLP_${upper}_TIMEOUT`]) ?? sharedTimeout;
    options[signal] = {
      endpoint,
      headers: {
        ...sharedHeaders,
        ...parseHeaders(source[`OTEL_EXPORTER_OTLP_${upper}_HEADERS`]),
      },
      // gRPC is not supported by this pipeline; anything but http/json uses http/protobuf.
      protocol: protocol === "http/json" ? "http/json" : "http/protobuf",
      ...(timeoutMs === undefined ? {} : { timeoutMs }),
    } satisfies OtlpSignalOptions;
  }
  const serviceName = source["OTEL_SERVICE_NAME"]?.trim();
  const resourceAttributes = parseHeaders(source["OTEL_RESOURCE_ATTRIBUTES"]);
  if ((serviceName !== undefined && serviceName !== "") || Object.keys(resourceAttributes).length) {
    options["resource"] = {
      serviceName: serviceName || resourceAttributes["service.name"] || "unknown_service",
      attributes: resourceAttributes,
    } satisfies ResourceOptions;
  }
  const batch: Record<string, number> = {};
  const bsp = [
    ["maxQueueSize", "OTEL_BSP_MAX_QUEUE_SIZE"],
    ["maxExportBatchSize", "OTEL_BSP_MAX_EXPORT_BATCH_SIZE"],
    ["scheduledDelayMs", "OTEL_BSP_SCHEDULE_DELAY"],
    ["exportTimeoutMs", "OTEL_BSP_EXPORT_TIMEOUT"],
  ] as const;
  for (const [key, name] of bsp) {
    const value = parsePositiveInt(source[name]);
    if (value !== undefined) batch[key] = value;
  }
  if (Object.keys(batch).length > 0) options["batch"] = batch;
  const interval = parsePositiveInt(source["OTEL_METRIC_EXPORT_INTERVAL"]);
  if (interval !== undefined) options["metricExportIntervalMs"] = interval;
  return options as OtlpPipelineOptions;
};
