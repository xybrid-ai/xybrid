/**
 * Metrics delivery: one request per collection, never split, serialized per exporter.
 *
 * A collection whose encoded body exceeds `maxRequestBodyBytes` is not sent and its data points
 * fail. Collections wait for the previous request (the wait counts toward their own deadline);
 * waiting plus in-flight encoded bytes never exceed `maxBufferedBytes`.
 */
import { diag } from "@opentelemetry/api";
import { MetricReader, type ResourceMetrics } from "@opentelemetry/sdk-metrics";
import { DEFAULTS } from "../catalog.ts";
import { type ShutdownOutcome, withTimeout } from "./batch-processor.ts";
import type { ResolvedLimits } from "./config.ts";
import { countDataPoints, type Encode } from "./encoding.ts";
import { droppedOutcome, exportScope, failedOutcome, type Outcome, sendWhole } from "./sender.ts";
import { SignalCounter } from "./stats.ts";
import { now, type OtlpTransport } from "./transport.ts";

const NOTHING: Outcome = Object.freeze({ exported: 0, dropped: 0, failed: 0 });

const lossless = (outcome: Outcome | undefined): boolean =>
  outcome !== undefined && outcome.dropped + outcome.failed === 0;

export class MetricDelivery {
  readonly counter = new SignalCounter();
  readonly #transport: OtlpTransport;
  readonly #encode: Encode<ResourceMetrics>;
  readonly #limits: ResolvedLimits;
  readonly #pending = new Set<Promise<Outcome>>();
  /** Aborted by a pipeline shutdown deadline. */
  readonly #cancel = new AbortController();
  #tail: Promise<unknown> = Promise.resolve();
  #closed = false;

  constructor(transport: OtlpTransport, encode: Encode<ResourceMetrics>, limits: ResolvedLimits) {
    this.#transport = transport;
    this.#encode = encode;
    this.#limits = limits;
  }

  /** Collections waiting or in flight right now. */
  pendingWork(): Promise<Outcome>[] {
    return [...this.#pending];
  }

  close(): void {
    this.#closed = true;
  }

  /** Aborts the in-flight request (failed); waiting collections are dropped. */
  cancel(): void {
    this.#cancel.abort();
  }

  /** Delivers one collection with a deadline of `timeoutMs` from now. Never rejects. */
  deliver(metrics: ResourceMetrics, timeoutMs: number): Promise<Outcome> {
    let points: number;
    try {
      points = countDataPoints(metrics);
    } catch {
      return Promise.resolve(NOTHING);
    }
    if (points === 0) return Promise.resolve(NOTHING);
    if (this.#closed) {
      this.counter.refuse(points);
      return Promise.resolve(droppedOutcome(points));
    }
    let body: Uint8Array;
    try {
      body = this.#encode(metrics);
    } catch {
      diag.warn("xybrid-observability: could not encode a metrics collection");
      this.counter.fail(points);
      return Promise.resolve(failedOutcome(points));
    }
    if (body.length > this.#limits.maxRequestBodyBytes) {
      diag.warn("xybrid-observability: a metrics collection above the body bound was not sent");
      this.counter.fail(points);
      return Promise.resolve(failedOutcome(points));
    }
    if (!(this.counter.bufferedBytes + body.length <= this.#limits.maxBufferedBytes)) {
      this.counter.refuse(points);
      return Promise.resolve(droppedOutcome(points));
    }
    const bytes = body.length;
    this.counter.admit(points, bytes);
    const scope = exportScope(timeoutMs, [this.#cancel.signal]);
    const run = this.#tail
      .then(() =>
        this.#cancel.signal.aborted
          ? droppedOutcome(points)
          : sendWhole(this.#transport, body, points, scope),
      )
      .catch(() => failedOutcome(points))
      .then((outcome) => {
        scope.release();
        this.counter.settle(outcome, bytes);
        this.#pending.delete(run);
        return outcome;
      });
    this.#pending.add(run);
    this.#tail = run;
    return run;
  }
}

/**
 * The pipeline's periodic reader: collects every `metricExportIntervalMs` and delivers each
 * collection with a deadline of `min(exportTimeoutMs, metricExportIntervalMs)`. Flush and
 * shutdown results are exact: they report the outcome of the collections they cover.
 */
export class PipelineMetricReader extends MetricReader {
  readonly delivery: MetricDelivery;
  readonly #limits: ResolvedLimits;
  #interval: ReturnType<typeof setInterval> | undefined;
  #shutdown: Promise<ShutdownOutcome> | undefined;

  constructor(delivery: MetricDelivery, limits: ResolvedLimits) {
    super();
    this.delivery = delivery;
    this.#limits = limits;
  }

  protected override onInitialized(): void {
    this.#interval = setInterval(() => {
      void this.#collectAndDeliver();
    }, this.#limits.metricExportIntervalMs);
    (this.#interval as { unref?: () => void }).unref?.();
  }

  async #collectAndDeliver(deadline = Number.POSITIVE_INFINITY): Promise<Outcome | undefined> {
    const timeout = Math.min(this.#limits.metricExportTimeoutMs, Math.max(0, deadline - now()));
    if (timeout <= 0) return undefined;
    try {
      const collected = await withTimeout(
        this.collect({ timeoutMillis: timeout }),
        timeout,
        undefined,
      );
      if (collected === undefined || now() >= deadline) return undefined;
      const failed = collected.errors.length > 0;
      if (failed) diag.debug("xybrid-observability: metrics collection reported errors");
      const outcome = await this.delivery.deliver(
        collected.resourceMetrics,
        Math.min(this.#limits.metricExportTimeoutMs, deadline - now()),
      );
      return failed ? undefined : outcome;
    } catch {
      return undefined;
    }
  }

  /**
   * Covers the collections pending at the call plus a new collection; `true` when all of them
   * were delivered without loss within `timeoutMs`. After shutdown began, the shutdown result.
   */
  flush(timeoutMs: number): Promise<boolean> {
    if (this.#shutdown !== undefined)
      return this.#shutdown.then((outcome) => outcome === "success");
    const covered = this.delivery.pendingWork();
    const work = Promise.all([...covered, this.#collectAndDeliver()]).then((outcomes) =>
      outcomes.every(lossless),
    );
    return withTimeout(work, timeoutMs, false);
  }

  /**
   * Stops periodic collection, takes a final collection, and at `deadline` aborts what is in
   * flight (failed) and drops what still waits. Idempotent.
   */
  shutdownBy(deadline: number): Promise<ShutdownOutcome> {
    this.#shutdown ??= (async () => {
      if (this.#interval !== undefined) clearInterval(this.#interval);
      const covered = this.delivery.pendingWork();
      let cancelled = false;
      const timer = setTimeout(
        () => {
          // Final collection can still be running before it has admitted any points.
          cancelled = true;
          this.delivery.cancel();
        },
        Math.ceil(Math.max(0, deadline - now())),
      );
      (timer as { unref?: () => void }).unref?.();
      try {
        const outcomes = await Promise.all([...covered, this.#collectAndDeliver(deadline)]);
        if (cancelled || now() >= deadline) return "timeout";
        return outcomes.every(lossless) ? "success" : "failure";
      } finally {
        clearTimeout(timer);
        this.delivery.close();
      }
    })();
    return this.#shutdown;
  }

  protected override async onForceFlush(): Promise<void> {
    await this.flush(DEFAULTS.flush_timeout_ms);
  }

  protected override async onShutdown(): Promise<void> {
    await this.shutdownBy(now() + DEFAULTS.shutdown_timeout_ms);
  }
}
