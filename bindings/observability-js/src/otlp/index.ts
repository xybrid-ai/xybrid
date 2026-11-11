/**
 * `@xybrid/observability/otlp`: an application-owned OTLP/HTTP pipeline, and standalone
 * exporters for pipelines the application builds itself.
 *
 * Builds tracer, meter and logger providers on the OpenTelemetry JS SDK 3.0/3.1 line without
 * registering any of them globally. The application creates the pipeline, hands
 * `pipeline.observer()` to instrumented code, and shuts the pipeline down.
 */

import { defaultResource, resourceFromAttributes } from "@opentelemetry/resources";
import { LoggerProvider, type ReadableLogRecord } from "@opentelemetry/sdk-logs";
import { MeterProvider } from "@opentelemetry/sdk-metrics";
import type { ReadableSpan } from "@opentelemetry/sdk-trace";
import { AlwaysOnSampler, ParentBasedSampler, TracerProvider } from "@opentelemetry/sdk-trace";
import { DEFAULTS, type SpanContentEncoding } from "../catalog.ts";
import {
  type ContentOptions,
  type LoggerProviderLike,
  Observer,
  spanContentEncodingOf,
} from "../observer.ts";
import {
  BoundedBatchQueue,
  PipelineLogProcessor,
  PipelineSpanProcessor,
  type ShutdownOutcome,
  withTimeout,
} from "./batch-processor.ts";
import {
  type BatchOptions,
  type HeaderPolicy,
  lifecycleTimeout,
  type OtlpEnvironment,
  type OtlpPipelineOptions,
  type OtlpProtocol,
  type OtlpSignalOptions,
  otlpConfigFromEnv,
  type ResourceOptions,
  type RetryOptions,
  resolveOtlpOptions,
  xybridTracesPreset,
} from "./config.ts";
import {
  estimateLogBytes,
  estimateSpanBytes,
  logEncoder,
  metricEncoder,
  traceEncoder,
} from "./encoding.ts";
import {
  createOtlpExporters,
  type OtlpExporters,
  OtlpLogExporter,
  OtlpMetricExporter,
  OtlpTraceExporter,
} from "./exporters.ts";
import { MetricDelivery, PipelineMetricReader } from "./metrics.ts";
import { BatchSender } from "./sender.ts";
import { type PipelineStats, type SignalStats, ZERO_STATS } from "./stats.ts";
import { now, OtlpTransport } from "./transport.ts";

export type {
  BatchOptions,
  HeaderPolicy,
  OtlpEnvironment,
  OtlpExporters,
  OtlpPipelineOptions,
  OtlpProtocol,
  OtlpSignalOptions,
  PipelineStats,
  ResourceOptions,
  RetryOptions,
  SignalStats,
};
export {
  createOtlpExporters,
  OtlpLogExporter,
  OtlpMetricExporter,
  OtlpTraceExporter,
  otlpConfigFromEnv,
  xybridTracesPreset,
};

/** Providers and delivery state of one OTLP pipeline. Never registered globally. */
export class OtlpPipeline {
  readonly tracerProvider: TracerProvider | undefined;
  readonly meterProvider: MeterProvider | undefined;
  readonly loggerProvider: LoggerProvider | undefined;
  readonly #spans: BoundedBatchQueue<ReadableSpan> | undefined;
  readonly #logs: BoundedBatchQueue<ReadableLogRecord> | undefined;
  readonly #metrics: PipelineMetricReader | undefined;
  /** `maxRequestBodyBytes` and content encoding of the traces export, if spans are exported. */
  readonly #spanRequestBound: number | undefined;
  readonly #spanContentEncoding: SpanContentEncoding | undefined;
  #shutdown: Promise<boolean> | undefined;

  /**
   * Validates every option first (`RangeError` for limits, `TypeError` for endpoints, headers
   * and shapes), before any provider, processor, timer or connection is created.
   */
  constructor(options: OtlpPipelineOptions) {
    const resolved = resolveOtlpOptions(options);
    const { limits } = resolved;
    const resource = defaultResource().merge(resourceFromAttributes(resolved.resourceAttributes));
    this.#spanRequestBound = resolved.traces === undefined ? undefined : limits.maxRequestBodyBytes;
    this.#spanContentEncoding =
      resolved.traces === undefined
        ? undefined
        : resolved.traces.protocol === "http/json"
          ? "json"
          : "protobuf";
    if (resolved.traces !== undefined) {
      const sender = new BatchSender<ReadableSpan>(
        "traces",
        new OtlpTransport(resolved.traces, limits),
        traceEncoder(resolved.traces.protocol),
        limits.maxRequestBodyBytes,
      );
      this.#spans = new BoundedBatchQueue(sender, estimateSpanBytes, limits);
      this.tracerProvider = new TracerProvider({
        resource,
        sampler: new ParentBasedSampler({ root: new AlwaysOnSampler() }),
        spanProcessors: [new PipelineSpanProcessor(this.#spans)],
      });
    }
    if (resolved.metrics !== undefined) {
      const delivery = new MetricDelivery(
        new OtlpTransport(resolved.metrics, limits),
        metricEncoder(resolved.metrics.protocol),
        limits,
      );
      this.#metrics = new PipelineMetricReader(delivery, limits);
      this.meterProvider = new MeterProvider({ resource, readers: [this.#metrics] });
    }
    if (resolved.logs !== undefined) {
      const sender = new BatchSender<ReadableLogRecord>(
        "logs",
        new OtlpTransport(resolved.logs, limits),
        logEncoder(resolved.logs.protocol),
        limits.maxRequestBodyBytes,
      );
      this.#logs = new BoundedBatchQueue(sender, estimateLogBytes, limits);
      this.loggerProvider = new LoggerProvider({
        resource,
        processors: [new PipelineLogProcessor(this.#logs)],
      });
    }
  }

  /**
   * An {@link Observer} over this pipeline's providers. Signals the pipeline does not export are
   * disabled rather than read from globals, so evaluation events return `false` on a traces-only
   * pipeline. With a traces signal, the content budgets are clamped to fit this pipeline's
   * `maxRequestBodyBytes` (or `content.maxRequestBodyBytes`, default 480 KiB, if smaller) as its
   * traces protocol writes content: `http/protobuf` carries twice the content of `http/json` in
   * the same request (416 KiB against 208 KiB at the default bound). The pipeline's protocol
   * replaces `content.spanContentEncoding`.
   */
  observer(content?: ContentOptions): Observer {
    const bound = this.#spanRequestBound;
    const encoding = this.#spanContentEncoding;
    const own =
      content?.maxRequestBodyBytes === undefined
        ? DEFAULTS.max_request_body_bytes
        : content.maxRequestBodyBytes;
    // Only a valid bound larger than the pipeline's (the default when absent) is replaced, as in
    // Rust; invalid input still throws.
    const replace =
      bound !== undefined && typeof own === "number" && Number.isSafeInteger(own) && own > bound;
    // The pipeline knows how its spans are written; an invalid `spanContentEncoding` still throws.
    if (encoding !== undefined) spanContentEncodingOf(content?.spanContentEncoding);
    const fitted =
      encoding === undefined
        ? replace
          ? { ...content, maxRequestBodyBytes: bound }
          : content
        : {
            ...content,
            ...(replace ? { maxRequestBodyBytes: bound } : {}),
            spanContentEncoding: encoding,
          };
    return new Observer({
      tracerProvider: this.tracerProvider ?? null,
      meterProvider: this.meterProvider ?? null,
      loggerProvider: (this.loggerProvider as LoggerProviderLike | undefined) ?? null,
      ...(fitted === undefined ? {} : { content: fitted }),
    });
  }

  /** Per-signal counters; every snapshot satisfies `accepted = exported + dropped + failed + pending`. */
  stats(): PipelineStats {
    return Object.freeze({
      traces: this.#spans?.counter.snapshot() ?? ZERO_STATS,
      logs: this.#logs?.counter.snapshot() ?? ZERO_STATS,
      metrics: this.#metrics?.delivery.counter.snapshot() ?? ZERO_STATS,
    });
  }

  /**
   * Exports the items pending at the call (plus a metrics collection it takes). Resolves `true`
   * only when all of them were delivered without loss within `timeoutMs`; a timeout resolves
   * `false` and lets the work continue. After shutdown began, resolves with the shutdown result.
   */
  async forceFlush(timeoutMs: number = DEFAULTS.flush_timeout_ms): Promise<boolean> {
    const timeout = lifecycleTimeout(timeoutMs, DEFAULTS.flush_timeout_ms);
    if (this.#shutdown !== undefined) return withTimeout(this.#shutdown, timeout, false);
    const results = await Promise.all([
      this.#spans?.flush(timeout),
      this.#logs?.flush(timeout),
      this.#metrics?.flush(timeout),
    ]);
    return results.every((result) => result !== false);
  }

  /**
   * Closes admission for every signal, exports what is queued and takes a final metrics
   * collection, cancels whatever still runs at the deadline, settles the counters and shuts the
   * providers down. Resolves `true` only when every item pending at the call was delivered.
   * Idempotent: later calls return the first result (waiting at most their own `timeoutMs`).
   */
  shutdown(timeoutMs: number = DEFAULTS.shutdown_timeout_ms): Promise<boolean> {
    const timeout = lifecycleTimeout(timeoutMs, DEFAULTS.shutdown_timeout_ms);
    if (this.#shutdown !== undefined) return withTimeout(this.#shutdown, timeout, false);
    const deadline = now() + timeout;
    this.#spans?.close();
    this.#logs?.close();
    this.#shutdown = (async () => {
      const outcomes: (ShutdownOutcome | undefined)[] = await Promise.all([
        this.#spans?.shutdown(deadline),
        this.#logs?.shutdown(deadline),
        this.#metrics?.shutdownBy(deadline),
      ]);
      // Every processor and the reader are already shut down, so this only releases the
      // providers; bound it anyway.
      await withTimeout(
        Promise.allSettled([
          this.tracerProvider?.shutdown(),
          this.loggerProvider?.shutdown(),
          this.meterProvider?.shutdown(),
        ]),
        Math.max(50, deadline - now()),
        undefined,
      );
      return outcomes.every((outcome) => outcome === undefined || outcome === "success");
    })();
    return this.#shutdown;
  }
}

/** Validates `options` and builds an {@link OtlpPipeline}. */
export const createOtlpPipeline = (options: OtlpPipelineOptions): OtlpPipeline =>
  new OtlpPipeline(options);
