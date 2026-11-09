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
});

export class SignalCounter {
  #accepted = 0;
  #exported = 0;
  #dropped = 0;
  #failed = 0;
  #pending = 0;

  get pending(): number {
    return this.#pending;
  }

  /** Items accepted and now pending. */
  admit(items: number): void {
    this.#accepted += items;
    this.#pending += items;
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

  /** Terminal outcomes of pending items. */
  settle(outcome: Outcome): void {
    this.#exported += outcome.exported;
    this.#dropped += outcome.dropped;
    this.#failed += outcome.failed;
    this.#pending -= outcome.exported + outcome.dropped + outcome.failed;
  }

  snapshot(): SignalStats {
    return Object.freeze({
      accepted: this.#accepted,
      exported: this.#exported,
      dropped: this.#dropped,
      failed: this.#failed,
      pending: this.#pending,
    });
  }
}
