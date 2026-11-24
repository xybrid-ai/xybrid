/**
 * The final OTLP/HTTP transport boundary (CONTRACT.md "Transport").
 *
 * Sends exactly the resolved headers, never follows redirects, retries 429/502/503/504 and
 * transport errors with exponential backoff (or `Retry-After`) inside one absolute deadline,
 * reads at most 64 KiB of a 2xx body and decodes OTLP partial success. Diagnostics carry the
 * signal and HTTP status only, never payloads, headers or response bodies.
 */
import { context, diag } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { DEFAULTS } from "../catalog.ts";
import type { ResolvedLimits, ResolvedSignal } from "./config.ts";
import { rejectedItems } from "./encoding.ts";

/** Outcome of one request: acknowledged (with partial-success rejections) or failed. */
export type Delivery =
  | { readonly ok: true; readonly rejected: number }
  | { readonly ok: false; readonly cancelled: boolean };

type Attempt =
  | { readonly kind: "success"; readonly body: Uint8Array | undefined }
  | { readonly kind: "retryable"; readonly status: number; readonly retryAfterMs?: number }
  | { readonly kind: "rejected"; readonly status: number }
  | { readonly kind: "transport" };

const RETRYABLE: ReadonlySet<number> = new Set(DEFAULTS.retry.retryable_status);

/** Monotonic milliseconds; deadlines are expressed on this clock. */
export const now = (): number => performance.now();

/** `Retry-After` as delta-seconds (decimals accepted) or an HTTP-date; `undefined` otherwise. */
export const parseRetryAfter = (header: string | null): number | undefined => {
  if (header === null) return undefined;
  const value = header.trim();
  if (value === "") return undefined;
  if (/^\d+(\.\d+)?$/.test(value)) return Number(value) * 1000;
  const date = Date.parse(value);
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
};

/** Resolves after `ms`, or `false` as soon as `signal` aborts. */
const sleep = (ms: number, signal: AbortSignal): Promise<boolean> =>
  new Promise((resolve) => {
    if (signal.aborted) {
      resolve(false);
      return;
    }
    const timer = setTimeout(() => finish(true), ms);
    const onAbort = (): void => finish(false);
    function finish(completed: boolean): void {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      resolve(completed);
    }
    signal.addEventListener("abort", onAbort, { once: true });
  });

const discard = (response: Response): void => {
  response.body?.cancel().catch(() => undefined);
};

/** Reads at most `limit` bytes; `undefined` on error, abort or truncation (never throws). */
const readCapped = async (response: Response, limit: number): Promise<Uint8Array | undefined> => {
  const body = response.body;
  if (body === null) return new Uint8Array(0);
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  let complete = false;
  try {
    while (total <= limit) {
      const { done, value } = await reader.read();
      if (done) {
        complete = true;
        break;
      }
      chunks.push(value);
      total += value.length;
    }
  } catch {
    return undefined;
  } finally {
    if (!complete) reader.cancel().catch(() => undefined);
  }
  if (!complete || total > limit) return undefined;
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
};

export class OtlpTransport {
  readonly #signal: ResolvedSignal;
  readonly #limits: ResolvedLimits;

  constructor(signal: ResolvedSignal, limits: ResolvedLimits) {
    this.#signal = signal;
    this.#limits = limits;
  }

  /**
   * Sends one request body of `items` items. Every attempt, backoff and response read ends by
   * `deadline` (on {@link now}'s clock) or when `cancel` aborts.
   */
  async send(
    body: Uint8Array,
    items: number,
    deadline: number,
    cancel: AbortSignal,
  ): Promise<Delivery> {
    const name = this.#signal.signal;
    for (let attempt = 0; ; attempt += 1) {
      if (cancel.aborted) return { ok: false, cancelled: true };
      const remaining = deadline - now();
      if (remaining <= 0) return { ok: false, cancelled: true };
      const outcome = await this.#attempt(
        body,
        Math.min(this.#signal.timeoutMs, remaining),
        cancel,
      );
      if (outcome === undefined) return { ok: false, cancelled: true };
      switch (outcome.kind) {
        case "success": {
          const rejected =
            outcome.body === undefined
              ? undefined
              : rejectedItems(name, this.#signal.protocol, outcome.body);
          return { ok: true, rejected: Math.min(rejected ?? 0, items) };
        }
        case "rejected":
          diag.warn(`xybrid-observability: ${name} export rejected (HTTP ${outcome.status})`);
          return { ok: false, cancelled: false };
        case "retryable":
          diag.debug(`xybrid-observability: ${name} export retryable (HTTP ${outcome.status})`);
          break;
        case "transport":
          diag.debug(`xybrid-observability: ${name} export attempt failed (transport)`);
          break;
      }
      if (attempt >= this.#limits.maxRetries) {
        diag.warn(`xybrid-observability: ${name} export failed after retries`);
        return { ok: false, cancelled: false };
      }
      const retryAfter = outcome.kind === "retryable" ? outcome.retryAfterMs : undefined;
      const wait =
        retryAfter ?? Math.min(this.#limits.initialDelayMs * 2 ** attempt, this.#limits.maxDelayMs);
      if (now() + wait >= deadline) {
        // The retry could not start before the deadline: fail now instead of waiting.
        diag.warn(`xybrid-observability: ${name} export failed (retry beyond the deadline)`);
        return { ok: false, cancelled: false };
      }
      if (!(await sleep(wait, cancel))) return { ok: false, cancelled: true };
    }
  }

  /** One request; `undefined` when `cancel` aborted it. */
  async #attempt(
    body: Uint8Array,
    timeoutMs: number,
    cancel: AbortSignal,
  ): Promise<Attempt | undefined> {
    const controller = new AbortController();
    const abort = (): void => controller.abort();
    const timer = setTimeout(abort, Math.ceil(timeoutMs));
    cancel.addEventListener("abort", abort, { once: true });
    try {
      let response: Response;
      try {
        // Synchronous browser context managers restore context at the first await, so
        // suppression is re-entered for every request: exports never trace themselves.
        response = await context.with(suppressTracing(context.active()), () =>
          fetch(this.#signal.url, {
            method: "POST",
            headers: this.#signal.headers,
            body: body as Uint8Array<ArrayBuffer>,
            signal: controller.signal,
            redirect: "manual",
          }),
        );
      } catch {
        return cancel.aborted ? undefined : { kind: "transport" };
      }
      const status = response.status;
      if (response.type === "opaqueredirect" || status === 0 || (status >= 300 && status < 400)) {
        discard(response);
        diag.warn(
          `xybrid-observability: ${this.#signal.signal} export got a redirect (HTTP ${status}); redirects are not followed`,
        );
        return { kind: "rejected", status };
      }
      if (status >= 200 && status < 300) {
        // The status decides; a body problem after a 2xx never changes the outcome.
        return {
          kind: "success",
          body: await readCapped(response, this.#limits.maxResponseBodyBytes),
        };
      }
      discard(response);
      if (RETRYABLE.has(status)) {
        const retryAfterMs = parseRetryAfter(response.headers.get("retry-after"));
        return retryAfterMs === undefined
          ? { kind: "retryable", status }
          : { kind: "retryable", status, retryAfterMs };
      }
      return { kind: "rejected", status };
    } finally {
      clearTimeout(timer);
      cancel.removeEventListener("abort", abort);
    }
  }
}
