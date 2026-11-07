import { type Context, TraceFlags } from "@opentelemetry/api";
import type { ReadableSpan, Span, SpanProcessor } from "@opentelemetry/sdk-trace";
import { DEFAULTS } from "../catalog.ts";
import type { OtlpHttpSender } from "./http.ts";

export type BatchOptions = {
  readonly maxQueueSize?: number | undefined;
  readonly maxExportBatchSize?: number | undefined;
  readonly scheduledDelayMs?: number | undefined;
  readonly exportTimeoutMs?: number | undefined;
};

/** Delivery counters of the traces pipeline. */
export type ExportStats = {
  /** Spans accepted into the queue. */
  readonly spansQueued: number;
  /** Spans the endpoint acknowledged. */
  readonly spansExported: number;
  /** Spans that will never be delivered: queue full, after shutdown, oversized, or failed export. */
  readonly spansDropped: number;
  /** Export requests (batches) that failed after retries or timed out. */
  readonly exportFailures: number;
};

const unref = (timer: ReturnType<typeof setTimeout>): void => {
  (timer as { unref?: () => void }).unref?.();
};

/**
 * Bounded batching span processor with exact counters. `onEnd` only enqueues: serialization and
 * network I/O always run from a timer, never inside the caller's `span.end()`.
 */
export class CountingBatchSpanProcessor implements SpanProcessor {
  readonly #sender: OtlpHttpSender<ReadableSpan[]>;
  readonly #maxQueueSize: number;
  readonly #maxBatch: number;
  readonly #delayMs: number;
  readonly #exportTimeoutMs: number;
  #queue: ReadableSpan[] = [];
  #timer: ReturnType<typeof setTimeout> | undefined;
  #draining: Promise<void> | undefined;
  #shutdown: Promise<void> | undefined;
  #queued = 0;
  #exported = 0;
  #dropped = 0;
  #failures = 0;

  constructor(sender: OtlpHttpSender<ReadableSpan[]>, options: BatchOptions = {}) {
    this.#sender = sender;
    this.#maxQueueSize = options.maxQueueSize ?? DEFAULTS.batch.max_queue_size;
    this.#maxBatch = Math.min(
      options.maxExportBatchSize ?? DEFAULTS.batch.max_export_batch_size,
      this.#maxQueueSize,
    );
    this.#delayMs = options.scheduledDelayMs ?? DEFAULTS.batch.scheduled_delay_ms;
    this.#exportTimeoutMs = options.exportTimeoutMs ?? DEFAULTS.batch.export_timeout_ms;
  }

  stats(): ExportStats {
    return {
      spansQueued: this.#queued,
      spansExported: this.#exported,
      spansDropped: this.#dropped,
      exportFailures: this.#failures,
    };
  }

  onStart(_span: Span, _parentContext: Context): void {}

  onEnd(span: ReadableSpan): void {
    if ((span.spanContext().traceFlags & TraceFlags.SAMPLED) === 0) return;
    if (this.#shutdown !== undefined || this.#queue.length >= this.#maxQueueSize) {
      this.#dropped += 1;
      return;
    }
    this.#queue.push(span);
    this.#queued += 1;
    this.#schedule(this.#queue.length >= this.#maxBatch ? 0 : this.#delayMs);
  }

  #schedule(delayMs: number): void {
    if (this.#timer !== undefined) {
      if (delayMs > 0) return;
      clearTimeout(this.#timer);
    }
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      void this.#drain();
    }, delayMs);
    unref(this.#timer);
  }

  /** Exports every queued batch; concurrent callers share the in-flight drain. */
  #drain(): Promise<void> {
    if (this.#draining !== undefined) return this.#draining;
    this.#draining = (async () => {
      try {
        while (this.#queue.length > 0) {
          await this.#exportBatch(this.#queue.splice(0, this.#maxBatch));
        }
      } finally {
        this.#draining = undefined;
      }
    })();
    return this.#draining;
  }

  async #exportBatch(batch: ReadableSpan[]): Promise<void> {
    const batchDeadline = new AbortController();
    const timer = setTimeout(() => batchDeadline.abort(), this.#exportTimeoutMs);
    unref(timer);
    // The deadline cancels fetches, retry backoff and remaining split requests. Wait for the
    // transport to settle before the next batch, preserving acknowledged partial exports.
    const outcome = await this.#sender
      .sendItems(batch, (items) => items, batchDeadline.signal)
      .catch(() => undefined)
      .finally(() => clearTimeout(timer));
    if (outcome === undefined) {
      this.#failures += 1;
      this.#dropped += batch.length;
      return;
    }
    this.#exported += outcome.exported;
    this.#dropped += outcome.failed + outcome.oversized;
    if (outcome.failed > 0) this.#failures += 1;
  }

  async forceFlush(): Promise<void> {
    if (this.#timer !== undefined) {
      clearTimeout(this.#timer);
      this.#timer = undefined;
    }
    await this.#drain();
  }

  shutdown(): Promise<void> {
    this.#shutdown ??= (async () => {
      await this.forceFlush();
    })();
    return this.#shutdown;
  }
}
