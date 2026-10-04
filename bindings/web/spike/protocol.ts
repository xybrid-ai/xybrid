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
};
export type Request =
  | { type: "load"; id: number; accelerator: Accelerator | "auto" }
  | { type: "generate"; id: number; prompt: string; maxTokens: number }
  | { type: "cancel" }
  | { type: "dispose"; id: number };
export type Response =
  | { type: "loaded"; id: number; metrics: LoadMetrics }
  | { type: "token"; id: number; tokenId: number; text: string }
  | { type: "complete"; id: number; metrics: RunMetrics }
  | { type: "disposed"; id: number; memory: Memory }
  | { type: "progress"; loadedBytes: number; totalBytes: number }
  | { type: "error"; id: number; message: string };
