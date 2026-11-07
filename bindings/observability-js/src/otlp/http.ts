import { context, diag } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { DEFAULTS } from "../catalog.ts";

export type OtlpProtocol = "http/protobuf" | "http/json";

/** One OTLP/HTTP signal destination. */
export type OtlpSignalOptions = {
  /** Full per-signal URL, e.g. `http://localhost:4318/v1/traces`. */
  readonly endpoint: string;
  readonly headers?: Readonly<Record<string, string>> | undefined;
  readonly protocol?: OtlpProtocol | undefined;
  /** Per-request timeout in milliseconds (default 10000). */
  readonly timeoutMs?: number | undefined;
};

export type RetryOptions = {
  readonly maxRetries?: number | undefined;
  readonly initialDelayMs?: number | undefined;
  readonly maxDelayMs?: number | undefined;
};

type Serializer<T> = { serializeRequest(request: T): Uint8Array | undefined };

export type SendOutcome = {
  readonly exported: number;
  readonly failed: number;
  readonly oversized: number;
};

const RETRYABLE_STATUS = new Set([429, 502, 503, 504]);

const sleep = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const timer = setTimeout(done, ms);
    function done(): void {
      clearTimeout(timer);
      signal.removeEventListener("abort", done);
      resolve();
    }
    signal.addEventListener("abort", done, { once: true });
  });

const retryAfterMs = (header: string | null): number | undefined => {
  if (header === null) return undefined;
  const seconds = Number(header);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  const date = Date.parse(header);
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
};

/**
 * Minimal OTLP/HTTP sender: bounded request bodies (oversized batches are split, an oversized
 * single item is dropped), bounded retries with exponential backoff, per-request timeouts, no
 * persistence. Diagnostics carry the signal and HTTP status only — never payloads or headers.
 */
export class OtlpHttpSender<T> {
  readonly #signal: string;
  readonly #url: string;
  readonly #headers: Record<string, string>;
  readonly #timeoutMs: number;
  readonly #maxRetries: number;
  readonly #initialDelayMs: number;
  readonly #maxDelayMs: number;
  readonly #maxBodyBytes: number;
  readonly #serializer: Serializer<T>;
  readonly #shutdown = new AbortController();

  constructor(
    signal: string,
    options: OtlpSignalOptions,
    serializers: { readonly protobuf: Serializer<T>; readonly json: Serializer<T> },
    retry: RetryOptions = {},
    maxBodyBytes: number = DEFAULTS.max_request_body_bytes,
  ) {
    this.#signal = signal;
    this.#url = new URL(options.endpoint).href;
    const json = options.protocol === "http/json";
    this.#serializer = json ? serializers.json : serializers.protobuf;
    this.#headers = {
      ...(options.headers ?? {}),
      "Content-Type": json ? "application/json" : "application/x-protobuf",
    };
    this.#timeoutMs = options.timeoutMs ?? DEFAULTS.batch.export_timeout_ms;
    this.#maxRetries = retry.maxRetries ?? DEFAULTS.retry.max_retries;
    this.#initialDelayMs = retry.initialDelayMs ?? DEFAULTS.retry.initial_delay_ms;
    this.#maxDelayMs = retry.maxDelayMs ?? DEFAULTS.retry.max_delay_ms;
    this.#maxBodyBytes = maxBodyBytes;
  }

  /** Cancels in-flight requests and pending retries. */
  abort(): void {
    this.#shutdown.abort();
  }

  get aborted(): boolean {
    return this.#shutdown.signal.aborted;
  }

  /** Sends a batch, splitting it until each request fits the body bound. */
  async sendItems<Item>(
    items: readonly Item[],
    wrap: (items: Item[]) => T,
    signal?: AbortSignal,
  ): Promise<SendOutcome> {
    if (this.aborted || signal?.aborted) return { exported: 0, failed: items.length, oversized: 0 };
    if (items.length === 0) return { exported: 0, failed: 0, oversized: 0 };
    const body = this.#serialize(wrap([...items]));
    if (body !== undefined && body.byteLength <= this.#maxBodyBytes) {
      const ok = await this.#post(body, signal);
      return ok
        ? { exported: items.length, failed: 0, oversized: 0 }
        : { exported: 0, failed: items.length, oversized: 0 };
    }
    if (body === undefined) return { exported: 0, failed: items.length, oversized: 0 };
    if (items.length === 1) {
      diag.warn(`xybrid-observability: dropped one ${this.#signal} item above the body bound`);
      return { exported: 0, failed: 0, oversized: 1 };
    }
    const middle = Math.ceil(items.length / 2);
    const first = await this.sendItems(items.slice(0, middle), wrap, signal);
    const second = await this.sendItems(items.slice(middle), wrap, signal);
    return {
      exported: first.exported + second.exported,
      failed: first.failed + second.failed,
      oversized: first.oversized + second.oversized,
    };
  }

  /** Sends one request that cannot be split (a metrics collection). */
  async sendWhole(request: T): Promise<boolean> {
    const body = this.#serialize(request);
    if (body === undefined) return false;
    if (body.byteLength > this.#maxBodyBytes) {
      diag.warn(`xybrid-observability: dropped a ${this.#signal} request above the body bound`);
      return false;
    }
    return this.#post(body);
  }

  #serialize(request: T): Uint8Array | undefined {
    try {
      return this.#serializer.serializeRequest(request);
    } catch {
      diag.warn(`xybrid-observability: could not serialize ${this.#signal}`);
      return undefined;
    }
  }

  #post(body: Uint8Array, signal?: AbortSignal): Promise<boolean> {
    // All signals, split batches and retries share this transport. Keep async work suppressed
    // too, so HTTP instrumentation cannot feed export requests back into the span queue.
    return context.with(suppressTracing(context.active()), () =>
      this.#postSuppressed(body, signal),
    );
  }

  async #postSuppressed(body: Uint8Array, batchSignal?: AbortSignal): Promise<boolean> {
    const signal =
      batchSignal === undefined
        ? this.#shutdown.signal
        : AbortSignal.any([this.#shutdown.signal, batchSignal]);
    for (let attempt = 0; attempt <= this.#maxRetries; attempt += 1) {
      if (signal.aborted) return false;
      let delay = Math.min(this.#initialDelayMs * 2 ** attempt, this.#maxDelayMs);
      const timeout = new AbortController();
      const timer = setTimeout(() => timeout.abort(), this.#timeoutMs);
      const cancelled = (): void => timeout.abort();
      signal.addEventListener("abort", cancelled, { once: true });
      try {
        // Synchronous browser context managers restore context at the first await. Re-enter
        // suppression for every fetch so later retry attempts cannot trace themselves either.
        const response = await context.with(suppressTracing(context.active()), () =>
          fetch(this.#url, {
            method: "POST",
            headers: this.#headers,
            body: body as Uint8Array<ArrayBuffer>,
            signal: timeout.signal,
          }),
        );
        // Drain so the connection can be reused; the response body is never logged.
        await response.arrayBuffer().catch(() => undefined);
        if (signal.aborted) return false;
        if (response.ok) return true;
        if (!RETRYABLE_STATUS.has(response.status)) {
          diag.warn(
            `xybrid-observability: ${this.#signal} export rejected (HTTP ${response.status})`,
          );
          return false;
        }
        delay = Math.min(
          retryAfterMs(response.headers.get("retry-after")) ?? delay,
          this.#maxDelayMs,
        );
        diag.debug(
          `xybrid-observability: ${this.#signal} export retryable (HTTP ${response.status})`,
        );
      } catch (error) {
        const name = error instanceof Error ? error.name : "Error";
        diag.debug(`xybrid-observability: ${this.#signal} export failed (${name})`);
      } finally {
        clearTimeout(timer);
        signal.removeEventListener("abort", cancelled);
      }
      if (attempt < this.#maxRetries) await sleep(delay, signal);
    }
    diag.warn(`xybrid-observability: ${this.#signal} export failed after retries`);
    return false;
  }
}
