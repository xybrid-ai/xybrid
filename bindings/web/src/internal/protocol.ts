import type { ObservationRef } from "@xybrid/observability";

export type Accelerator = "wasm" | "webgpu";
export type Memory = {
  /** Committed WASM linear-memory capacity; it does not shrink on disposal. */
  heapBytes: number;
  /** Live dlmalloc allocations, including Rust and C++ in this module. */
  allocatedBytes: number;
};
export type LoadMetrics = {
  accelerator: Accelerator;
  rustVersion: string;
  adapter: string | null;
  modelBytes: number;
  downloadMs: number;
  initializationMs: number;
  loadMs: number;
  memory: Memory;
  /** The `load_model` span, when an observer recorded one. */
  observation?: ObservationRef;
};
export type RunMetrics = {
  cancelled: boolean;
  promptTokens: number;
  generatedTokens: number;
  firstTokenMs: number | null;
  totalMs: number;
  /** Tokens after the first, divided by time from first token to last token. */
  decodeTokensPerSecond: number | null;
  peakMemory: Memory;
  memory: Memory;
  /** The inference span, when an observer recorded one. */
  observation?: ObservationRef;
};
/** Present only for observed generations: the caller's trace context, as W3C headers. */
export type GenerateObservationContext = { traceparent?: string; tracestate?: string };
export type Request =
  | {
      type: "load";
      id: number;
      accelerator: Accelerator;
      wasmPath: string;
      bytes: ArrayBuffer;
      contextLength: number;
      downloadMs: number;
    }
  | {
      type: "generate";
      id: number;
      prompt: string;
      maxTokens: number;
      /** Asks the worker for epoch-millisecond token and completion times. */
      observation?: GenerateObservationContext;
    }
  | { type: "cancel" }
  | { type: "dispose"; id: number };
export type Response =
  | { type: "loaded"; id: number; metrics: LoadMetrics }
  | { type: "token"; id: number; tokenId: number; text: string; atMs?: number }
  | { type: "complete"; id: number; metrics: RunMetrics; completedAtMs?: number }
  | { type: "disposed"; id: number; memory: Memory }
  | { type: "progress"; loadedBytes: number; totalBytes: number }
  | { type: "error"; id: number; message: string };
