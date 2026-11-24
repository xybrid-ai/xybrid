/**
 * Splitting and per-export delivery for one signal (CONTRACT.md "Transport").
 *
 * A trace or log batch is encoded with its resource/scope grouping; when the final body exceeds
 * `maxRequestBodyBytes` it is split into `ceil(n/2)` and the rest (original order), each part
 * re-encoded, recursively, and the parts are sent in order. Only a single item whose own request
 * is too large is dropped. A metrics collection is never split: over the bound it fails.
 */
import { diag } from "@opentelemetry/api";
import type { Signal } from "./config.ts";
import type { Encode } from "./encoding.ts";
import { now, type OtlpTransport } from "./transport.ts";

/** Terminal outcomes of the items of one export. */
export type Outcome = {
  readonly exported: number;
  readonly dropped: number;
  readonly failed: number;
};

export const failedOutcome = (items: number): Outcome => ({
  exported: 0,
  dropped: 0,
  failed: items,
});
export const droppedOutcome = (items: number): Outcome => ({
  exported: 0,
  dropped: items,
  failed: 0,
});

const add = (first: Outcome, second: Outcome): Outcome => ({
  exported: first.exported + second.exported,
  dropped: first.dropped + second.dropped,
  failed: first.failed + second.failed,
});

/** One export's absolute deadline (monotonic ms) and cancellation. */
export type ExportScope = { readonly deadline: number; readonly signal: AbortSignal };

/**
 * An `AbortController` aborted at `deadline` and when any parent aborts. `release()` clears the
 * timer and parent listeners.
 */
export const exportScope = (
  timeoutMs: number,
  parents: readonly AbortSignal[],
): ExportScope & { release(): void } => {
  const controller = new AbortController();
  const abort = (): void => controller.abort();
  const deadline = now() + timeoutMs;
  const timer = setTimeout(abort, Math.max(0, timeoutMs));
  (timer as { unref?: () => void }).unref?.();
  for (const parent of parents) {
    if (parent.aborted) controller.abort();
    else parent.addEventListener("abort", abort, { once: true });
  }
  return {
    deadline,
    signal: controller.signal,
    release: () => {
      clearTimeout(timer);
      for (const parent of parents) parent.removeEventListener("abort", abort);
    },
  };
};

export class BatchSender<Item> {
  readonly #signal: Signal;
  readonly #transport: OtlpTransport;
  readonly #encode: Encode<Item[]>;
  readonly #maxBodyBytes: number;

  constructor(signal: Signal, transport: OtlpTransport, encode: Encode<Item[]>, maxBody: number) {
    this.#signal = signal;
    this.#transport = transport;
    this.#encode = encode;
    this.#maxBodyBytes = maxBody;
  }

  /** Delivers `items`, splitting by final encoded size; never throws. */
  async send(items: readonly Item[], scope: ExportScope): Promise<Outcome> {
    if (items.length === 0) return { exported: 0, dropped: 0, failed: 0 };
    // Unsent parts after the deadline or a cancellation fail without a request.
    if (scope.signal.aborted || now() >= scope.deadline) return failedOutcome(items.length);
    let body: Uint8Array | undefined;
    try {
      body = this.#encode([...items]);
    } catch {
      body = undefined;
    }
    if (body !== undefined && body.length <= this.#maxBodyBytes) {
      const delivery = await this.#transport.send(body, items.length, scope.deadline, scope.signal);
      if (!delivery.ok) return failedOutcome(items.length);
      return { exported: items.length - delivery.rejected, dropped: 0, failed: delivery.rejected };
    }
    if (items.length === 1) {
      if (body === undefined) {
        diag.warn(`xybrid-observability: could not encode one ${this.#signal} item`);
        return failedOutcome(1);
      }
      diag.warn(`xybrid-observability: dropped one ${this.#signal} item above the body bound`);
      return droppedOutcome(1);
    }
    const middle = Math.ceil(items.length / 2);
    const first = await this.send(items.slice(0, middle), scope);
    const second = await this.send(items.slice(middle), scope);
    return add(first, second);
  }
}

/** Sends one pre-encoded request whose items succeed or fail together (metrics). */
export const sendWhole = async (
  transport: OtlpTransport,
  body: Uint8Array,
  items: number,
  scope: ExportScope,
): Promise<Outcome> => {
  if (scope.signal.aborted || now() >= scope.deadline) return failedOutcome(items);
  const delivery = await transport.send(body, items, scope.deadline, scope.signal);
  if (!delivery.ok) return failedOutcome(items);
  return { exported: items - delivery.rejected, dropped: 0, failed: delivery.rejected };
};
