/**
 * Standalone exporters for application-owned pipelines (CONTRACT.md "Paired observer and
 * exporter adapters").
 *
 * Pair them with the package's `Observer` over your own providers to keep catalog wire types,
 * structured null, transport bounds and counters. Each exporter is the final stage of one
 * provider's pipeline; its deadline (`batch.exportTimeoutMs`) must not exceed the timeout of
 * the processor or reader that calls it. Stats count items handed to `export()`; pending items
 * are those in flight, and `batch.maxBufferedBytes` bounds in-flight bytes.
 */
import { diag } from "@opentelemetry/api";
import { type ExportResult, ExportResultCode } from "@opentelemetry/core";
import type { LogRecordExporter, ReadableLogRecord } from "@opentelemetry/sdk-logs";
import {
  AggregationTemporality,
  type PushMetricExporter,
  type ResourceMetrics,
} from "@opentelemetry/sdk-metrics";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace";
import {
  type OtlpPipelineOptions,
  type ResolvedLimits,
  type ResolvedSignal,
  resolveOtlpOptions,
  type Signal,
} from "./config.ts";
import {
  countDataPoints,
  estimateLogBytes,
  estimateSpanBytes,
  logEncoder,
  metricEncoder,
  traceEncoder,
} from "./encoding.ts";
import { MetricDelivery } from "./metrics.ts";
import { BatchSender, exportScope, failedOutcome, type Outcome } from "./sender.ts";
import { SignalCounter, type SignalStats } from "./stats.ts";
import { OtlpTransport } from "./transport.ts";

const RESOLVED = Symbol("xybrid.otlp.resolved");

type Init = { readonly signal: ResolvedSignal; readonly limits: ResolvedLimits };

/** Validates `options` (or unwraps options already validated by {@link createOtlpExporters}). */
const initFor = (signal: Signal, options: OtlpPipelineOptions): Init => {
  const internal = (options as { readonly [RESOLVED]?: Init })[RESOLVED];
  if (internal !== undefined) return internal;
  const resolved = resolveOtlpOptions(options);
  const target = resolved[signal];
  if (target === undefined) throw new TypeError(`${signal} options with an endpoint are required.`);
  return { signal: target, limits: resolved.limits };
};

const report = (callback: (result: ExportResult) => void, succeeded: boolean): void => {
  try {
    callback({ code: succeeded ? ExportResultCode.SUCCESS : ExportResultCode.FAILED });
  } catch {
    // A throwing SDK callback must not break the exporter.
  }
};

/** Shared state of the trace and log exporters. */
class StandaloneBatchExporter<Item> {
  readonly counter = new SignalCounter();
  readonly #sender: BatchSender<Item>;
  readonly #estimate: (item: Item) => number;
  readonly #limits: ResolvedLimits;
  readonly #inFlight = new Set<Promise<Outcome>>();
  #shutdown: Promise<void> | undefined;

  constructor(sender: BatchSender<Item>, estimate: (item: Item) => number, limits: ResolvedLimits) {
    this.#sender = sender;
    this.#estimate = estimate;
    this.#limits = limits;
  }

  export(items: readonly Item[], callback: (result: ExportResult) => void): void {
    const count = items.length;
    if (this.#shutdown !== undefined) {
      this.counter.fail(count);
      report(callback, false);
      return;
    }
    let bytes = 0;
    try {
      for (const item of items) bytes += this.#estimate(item);
    } catch {
      bytes = Number.POSITIVE_INFINITY;
    }
    if (!(this.counter.bufferedBytes + bytes <= this.#limits.maxBufferedBytes)) {
      diag.warn("xybrid-observability: export refused (in-flight byte budget)");
      this.counter.refuse(count);
      report(callback, false);
      return;
    }
    this.counter.admit(count, bytes);
    const scope = exportScope(this.#limits.exportTimeoutMs, []);
    const work = this.#sender
      .send([...items], scope)
      .catch(() => failedOutcome(count))
      .then((outcome) => {
        scope.release();
        this.counter.settle(outcome, bytes);
        this.#inFlight.delete(work);
        report(callback, outcome.exported === count);
        return outcome;
      });
    this.#inFlight.add(work);
  }

  /** Waits for the exports in flight at the call. */
  async forceFlush(): Promise<void> {
    await Promise.all([...this.#inFlight]);
  }

  /** Rejects new exports, then waits for in-flight ones (each bounded by its deadline). */
  shutdown(): Promise<void> {
    this.#shutdown ??= (async () => {
      await Promise.all([...this.#inFlight]);
    })();
    return this.#shutdown;
  }
}

/** OTLP/HTTP span exporter for an application-owned `TracerProvider`. */
export class OtlpTraceExporter implements SpanExporter {
  readonly #exporter: StandaloneBatchExporter<ReadableSpan>;

  /** Validates `options` (`options.traces` is required); throws `RangeError`/`TypeError`. */
  constructor(options: OtlpPipelineOptions) {
    const { signal, limits } = initFor("traces", options);
    const sender = new BatchSender<ReadableSpan>(
      "traces",
      new OtlpTransport(signal, limits),
      traceEncoder(signal.protocol),
      limits.maxRequestBodyBytes,
    );
    this.#exporter = new StandaloneBatchExporter(sender, estimateSpanBytes, limits);
  }

  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.#exporter.export(spans, resultCallback);
  }

  forceFlush(): Promise<void> {
    return this.#exporter.forceFlush();
  }

  shutdown(): Promise<void> {
    return this.#exporter.shutdown();
  }

  stats(): SignalStats {
    return this.#exporter.counter.snapshot();
  }
}

/** OTLP/HTTP log record exporter for an application-owned `LoggerProvider`. */
export class OtlpLogExporter implements LogRecordExporter {
  readonly #exporter: StandaloneBatchExporter<ReadableLogRecord>;

  /** Validates `options` (`options.logs` is required); throws `RangeError`/`TypeError`. */
  constructor(options: OtlpPipelineOptions) {
    const { signal, limits } = initFor("logs", options);
    const sender = new BatchSender<ReadableLogRecord>(
      "logs",
      new OtlpTransport(signal, limits),
      logEncoder(signal.protocol),
      limits.maxRequestBodyBytes,
    );
    this.#exporter = new StandaloneBatchExporter(sender, estimateLogBytes, limits);
  }

  export(logs: ReadableLogRecord[], resultCallback: (result: ExportResult) => void): void {
    this.#exporter.export(logs, resultCallback);
  }

  forceFlush(): Promise<void> {
    return this.#exporter.forceFlush();
  }

  shutdown(): Promise<void> {
    return this.#exporter.shutdown();
  }

  stats(): SignalStats {
    return this.#exporter.counter.snapshot();
  }
}

/** OTLP/HTTP metric exporter (cumulative) for an application-owned metric reader. */
export class OtlpMetricExporter implements PushMetricExporter {
  readonly #delivery: MetricDelivery;
  readonly #timeoutMs: number;
  #shutdown: Promise<void> | undefined;

  /** Validates `options` (`options.metrics` is required); throws `RangeError`/`TypeError`. */
  constructor(options: OtlpPipelineOptions) {
    const { signal, limits } = initFor("metrics", options);
    this.#delivery = new MetricDelivery(
      new OtlpTransport(signal, limits),
      metricEncoder(signal.protocol),
      limits,
    );
    this.#timeoutMs = limits.metricExportTimeoutMs;
  }

  export(metrics: ResourceMetrics, resultCallback: (result: ExportResult) => void): void {
    if (this.#shutdown !== undefined) {
      this.#delivery.counter.fail(countDataPoints(metrics));
      report(resultCallback, false);
      return;
    }
    void this.#delivery
      .deliver(metrics, this.#timeoutMs)
      .then((outcome) => report(resultCallback, outcome.dropped + outcome.failed === 0));
  }

  async forceFlush(): Promise<void> {
    await Promise.all(this.#delivery.pendingWork());
  }

  shutdown(): Promise<void> {
    this.#shutdown ??= (async () => {
      this.#delivery.close();
      await Promise.all(this.#delivery.pendingWork());
    })();
    return this.#shutdown;
  }

  selectAggregationTemporality(): AggregationTemporality {
    return AggregationTemporality.CUMULATIVE;
  }

  stats(): SignalStats {
    return this.#delivery.counter.snapshot();
  }
}

/** The standalone exporters of {@link createOtlpExporters}, one per configured signal. */
export type OtlpExporters = {
  readonly traces?: OtlpTraceExporter;
  readonly logs?: OtlpLogExporter;
  readonly metrics?: OtlpMetricExporter;
};

/**
 * Builds one standalone exporter per configured signal after validating every option (the
 * environment is read once, for `merge_environment` headers). `resource`, batching queue sizes
 * and the scheduled delay do not apply: the application's providers and processors own them.
 */
export const createOtlpExporters = (options: OtlpPipelineOptions): OtlpExporters => {
  const resolved = resolveOtlpOptions(options);
  const init = (signal: ResolvedSignal): OtlpPipelineOptions =>
    ({ [RESOLVED]: { signal, limits: resolved.limits } }) as unknown as OtlpPipelineOptions;
  return {
    ...(resolved.traces === undefined
      ? {}
      : { traces: new OtlpTraceExporter(init(resolved.traces)) }),
    ...(resolved.logs === undefined ? {} : { logs: new OtlpLogExporter(init(resolved.logs)) }),
    ...(resolved.metrics === undefined
      ? {}
      : { metrics: new OtlpMetricExporter(init(resolved.metrics)) }),
  };
};
