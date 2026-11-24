/**
 * Bounded batching for spans and log records (CONTRACT.md "Lifecycle and delivery").
 *
 * `offer` only enqueues: estimation and admission are synchronous and constant-work per item,
 * and encoding and network I/O always run later from a timer. Queued plus in-flight items never
 * exceed `maxQueueSize`, and their conservatively estimated bytes never exceed
 * `maxBufferedBytes`; anything over either limit, or offered after shutdown began, is dropped
 * and counted. Batches are exported one at a time in admission order.
 */
import { type Context, TraceFlags } from "@opentelemetry/api";
import type { LogRecordProcessor, ReadableLogRecord } from "@opentelemetry/sdk-logs";
import type { ReadableSpan, Span, SpanProcessor } from "@opentelemetry/sdk-trace";
import { DEFAULTS } from "../catalog.ts";
import type { ResolvedLimits } from "./config.ts";
import {
  type BatchSender,
  droppedOutcome,
  exportScope,
  failedOutcome,
  type Outcome,
} from "./sender.ts";
import { SignalCounter } from "./stats.ts";
import { now } from "./transport.ts";

/** Result of a shutdown: everything delivered, losses, or work cancelled at the deadline. */
export type ShutdownOutcome = "success" | "failure" | "timeout";

type Entry<Item> = { readonly item: Item; readonly bytes: number; readonly seq: number };

/** Waits for every item admitted in `(from, to]` to become terminal; `lossy` records losses. */
type Waiter = {
  readonly from: number;
  readonly to: number;
  lossy: boolean;
  readonly resolve: (succeeded: boolean) => void;
};

const unref = (timer: ReturnType<typeof setTimeout>): void => {
  (timer as { unref?: () => void }).unref?.();
};

/** Resolves with `work`'s value, or `fallback` after `timeoutMs` (then calls `onTimeout`). */
export const withTimeout = <T>(
  work: Promise<T>,
  timeoutMs: number,
  fallback: T,
  onTimeout?: () => void,
): Promise<T> =>
  new Promise((resolve) => {
    const timer = setTimeout(() => {
      onTimeout?.();
      resolve(fallback);
    }, timeoutMs);
    unref(timer);
    work.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      () => {
        clearTimeout(timer);
        resolve(fallback);
      },
    );
  });

export class BoundedBatchQueue<Item> {
  readonly counter = new SignalCounter();
  readonly #sender: BatchSender<Item>;
  readonly #estimate: (item: Item) => number;
  readonly #limits: ResolvedLimits;
  readonly #queue: Entry<Item>[] = [];
  readonly #waiters = new Set<Waiter>();
  /** Aborted at the shutdown deadline: cancels in-flight work and stops the drain. */
  readonly #cancel = new AbortController();
  /** Sequence number of the last admitted item. */
  #admitted = 0;
  /** Every item with a sequence number up to this one is terminal. */
  #completed = 0;
  #closed = false;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #running: Promise<void> | undefined;
  #shutdown: Promise<ShutdownOutcome> | undefined;

  constructor(sender: BatchSender<Item>, estimate: (item: Item) => number, limits: ResolvedLimits) {
    this.#sender = sender;
    this.#estimate = estimate;
    this.#limits = limits;
  }

  /** Admits or drops one item. Never throws, never performs I/O. */
  offer(item: Item): void {
    if (this.#closed) {
      this.counter.refuse(1);
      return;
    }
    let bytes: number;
    try {
      bytes = this.#estimate(item);
    } catch {
      bytes = Number.POSITIVE_INFINITY;
    }
    if (
      this.counter.pending >= this.#limits.maxQueueSize ||
      !(this.counter.bufferedBytes + bytes <= this.#limits.maxBufferedBytes)
    ) {
      this.counter.refuse(1);
      return;
    }
    this.counter.admit(1, bytes);
    this.#admitted += 1;
    this.#queue.push({ item, bytes, seq: this.#admitted });
    this.#schedule(this.#queue.length >= this.#limits.maxExportBatchSize ? 0 : undefined);
  }

  /** Stops admission; later offers are dropped. */
  close(): void {
    this.#closed = true;
  }

  #schedule(delayMs: number | undefined): void {
    if (this.#running !== undefined) return;
    if (this.#timer !== undefined) {
      if (delayMs === undefined) return;
      clearTimeout(this.#timer);
    }
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      void this.#run();
    }, delayMs ?? this.#limits.scheduledDelayMs);
    unref(this.#timer);
  }

  /** Exports queued batches until the queue is empty; concurrent callers share the loop. */
  #run(): Promise<void> {
    if (this.#running !== undefined) return this.#running;
    if (this.#timer !== undefined) {
      clearTimeout(this.#timer);
      this.#timer = undefined;
    }
    if (this.#queue.length === 0 || this.#cancel.signal.aborted) return Promise.resolve();
    const running = (async () => {
      try {
        // The first iteration always awaits, so `#running` is assigned before `finally` runs.
        while (this.#queue.length > 0 && !this.#cancel.signal.aborted) {
          await this.#export(this.#queue.splice(0, this.#limits.maxExportBatchSize));
        }
      } finally {
        this.#running = undefined;
      }
    })();
    this.#running = running;
    return running;
  }

  async #export(batch: Entry<Item>[]): Promise<void> {
    const scope = exportScope(this.#limits.exportTimeoutMs, [this.#cancel.signal]);
    let outcome: Outcome;
    try {
      outcome = await this.#sender.send(
        batch.map((entry) => entry.item),
        scope,
      );
    } catch {
      outcome = failedOutcome(batch.length);
    } finally {
      scope.release();
    }
    this.#complete(batch, outcome);
  }

  #complete(batch: readonly Entry<Item>[], outcome: Outcome): void {
    const first = batch[0];
    const last = batch[batch.length - 1];
    if (first === undefined || last === undefined) return;
    let bytes = 0;
    for (const entry of batch) bytes += entry.bytes;
    this.counter.settle(outcome, bytes);
    this.#completed = last.seq;
    const lossy = outcome.dropped + outcome.failed > 0;
    for (const waiter of this.#waiters) {
      if (lossy && last.seq > waiter.from && first.seq <= waiter.to) waiter.lossy = true;
      if (this.#completed >= waiter.to) {
        this.#waiters.delete(waiter);
        waiter.resolve(!waiter.lossy);
      }
    }
  }

  /**
   * Resolves `true` once every item pending at the call reached a terminal outcome and none
   * failed or was dropped; `false` on such a loss or after `timeoutMs` (work continues). After
   * shutdown began, resolves with the shutdown result.
   */
  flush(timeoutMs: number): Promise<boolean> {
    if (this.#shutdown !== undefined)
      return this.#shutdown.then((outcome) => outcome === "success");
    if (this.#completed >= this.#admitted) return Promise.resolve(true);
    let resolve: (succeeded: boolean) => void = () => undefined;
    const done = new Promise<boolean>((settle) => {
      resolve = settle;
    });
    const waiter: Waiter = { from: this.#completed, to: this.#admitted, lossy: false, resolve };
    this.#waiters.add(waiter);
    void this.#run();
    return withTimeout(done, timeoutMs, false, () => this.#waiters.delete(waiter));
  }

  /**
   * Closes admission, exports everything queued, and at `deadline` (on the transport's
   * monotonic clock) aborts in-flight work (failed) and drops what is still queued. Counters are
   * settled when the promise resolves. Idempotent: later calls return the first result.
   */
  shutdown(deadline: number): Promise<ShutdownOutcome> {
    this.#shutdown ??= this.#shutdownBy(deadline);
    return this.#shutdown;
  }

  async #shutdownBy(deadline: number): Promise<ShutdownOutcome> {
    this.#closed = true;
    const covered: Waiter = {
      from: this.#completed,
      to: this.#admitted,
      lossy: false,
      resolve: () => undefined,
    };
    if (covered.to > covered.from) this.#waiters.add(covered);
    let cancelled = false;
    const timer = setTimeout(
      () => {
        cancelled = this.counter.pending > 0;
        this.#cancel.abort();
      },
      Math.max(0, deadline - now()),
    );
    unref(timer);
    try {
      await this.#run();
    } finally {
      clearTimeout(timer);
    }
    if (this.#queue.length > 0) {
      // Not sent before the deadline.
      const rest = this.#queue.splice(0);
      this.#complete(rest, droppedOutcome(rest.length));
    }
    this.#waiters.delete(covered);
    if (cancelled) return "timeout";
    return covered.lossy ? "failure" : "success";
  }
}

/** The pipeline's span processor: sampled spans only, admitted into a bounded queue. */
export class PipelineSpanProcessor implements SpanProcessor {
  readonly queue: BoundedBatchQueue<ReadableSpan>;

  constructor(queue: BoundedBatchQueue<ReadableSpan>) {
    this.queue = queue;
  }

  onStart(_span: Span, _parentContext: Context): void {}

  onEnd(span: ReadableSpan): void {
    if ((span.spanContext().traceFlags & TraceFlags.SAMPLED) === 0) return;
    this.queue.offer(span);
  }

  async forceFlush(): Promise<void> {
    await this.queue.flush(DEFAULTS.flush_timeout_ms);
  }

  async shutdown(): Promise<void> {
    await this.queue.shutdown(now() + DEFAULTS.shutdown_timeout_ms);
  }
}

/** The pipeline's log record processor (replaces the SDK `BatchLogRecordProcessor`). */
export class PipelineLogProcessor implements LogRecordProcessor {
  readonly queue: BoundedBatchQueue<ReadableLogRecord>;

  constructor(queue: BoundedBatchQueue<ReadableLogRecord>) {
    this.queue = queue;
  }

  onEmit(record: ReadableLogRecord): void {
    this.queue.offer(record);
  }

  async forceFlush(): Promise<void> {
    await this.queue.flush(DEFAULTS.flush_timeout_ms);
  }

  async shutdown(): Promise<void> {
    await this.queue.shutdown(now() + DEFAULTS.shutdown_timeout_ms);
  }
}
