/**
 * Per-signal delivery counters (CONTRACT.md "Lifecycle and delivery").
 *
 * Every snapshot satisfies `accepted == exported + dropped + failed + pending`: each update
 * changes `pending` and exactly one terminal counter by the same amount, synchronously.
 */
import type { Outcome } from "./sender.ts";

/** Delivery counters of one signal. */
export type SignalStats = {
  /** Items offered, including items refused at admission. */
  readonly accepted: number;
  /** Items the endpoint acknowledged (2xx minus partial-success rejections). */
  readonly exported: number;
  /** Items discarded locally without a deciding delivery attempt. */
  readonly dropped: number;
  /** Items whose export ended without acknowledgement (delivery failed or unknown). */
  readonly failed: number;
  /** Accepted items not yet terminal (queued, in flight, or waiting in backoff). */
  readonly pending: number;
  /**
   * Library-owned bytes held for the pending items: the per-item estimate (for metrics, the
   * encoded collection) admitted against `batch.maxBufferedBytes` (32 MiB). Not process memory;
   * zero once nothing is pending, so always zero after shutdown.
   */
  readonly bufferedBytes: number;
};

/** Counters per signal; unconfigured signals are all zero. */
export type PipelineStats = {
  readonly traces: SignalStats;
  readonly logs: SignalStats;
  readonly metrics: SignalStats;
};

export const ZERO_STATS: SignalStats = Object.freeze({
  accepted: 0,
  exported: 0,
  dropped: 0,
  failed: 0,
  pending: 0,
  bufferedBytes: 0,
});

export class SignalCounter {
  #accepted = 0;
  #exported = 0;
  #dropped = 0;
  #failed = 0;
  #pending = 0;
  #bytes = 0;

  get pending(): number {
    return this.#pending;
  }

  /** Estimated bytes held for the pending items (the byte budget's running total). */
  get bufferedBytes(): number {
    return this.#bytes;
  }

  /** Items accepted and now pending, holding `bytes` of the byte budget. */
  admit(items: number, bytes: number): void {
    this.#accepted += items;
    this.#pending += items;
    this.#bytes += bytes;
  }

  /** Items offered and immediately discarded (admission refused or closed). */
  refuse(items: number): void {
    this.#accepted += items;
    this.#dropped += items;
  }

  /** Items offered that failed before becoming pending (for example an oversized collection). */
  fail(items: number): void {
    this.#accepted += items;
    this.#failed += items;
  }

  /** Terminal outcomes of pending items, releasing the `bytes` they held. */
  settle(outcome: Outcome, bytes: number): void {
    this.#exported += outcome.exported;
    this.#dropped += outcome.dropped;
    this.#failed += outcome.failed;
    this.#pending -= outcome.exported + outcome.dropped + outcome.failed;
    this.#bytes -= bytes;
  }

  snapshot(): SignalStats {
    return Object.freeze({
      accepted: this.#accepted,
      exported: this.#exported,
      dropped: this.#dropped,
      failed: this.#failed,
      pending: this.#pending,
      bufferedBytes: this.#bytes,
    });
  }
}
