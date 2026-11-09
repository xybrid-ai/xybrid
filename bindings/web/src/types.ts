import type { Observer, ParentInput, TraceMode } from "@xybrid/observability";

export type AcceleratorPreference = "auto" | "wasm" | "webgpu";
export type SelectedAccelerator = "wasm" | "webgpu";

/** Where an operation joins a trace, and whether it is traced at all. */
export type TraceOptions = {
  /** Explicit parent (`op.context()` or a `traceparent`); wins over the ambient context. */
  readonly parent?: ParentInput;
  /** `tracestate` accompanying a `traceparent` string parent. */
  readonly tracestate?: string;
  /** `suppress` records metrics without a span. */
  readonly trace?: TraceMode;
};

/** Per-generation observation details. */
export type GenerateObservation = TraceOptions & {
  readonly conversationId?: string;
  readonly prompt?: {
    readonly name: string;
    readonly version?: string;
    readonly variables?: Readonly<Record<string, string>>;
  };
};

export type LoadOptions = {
  readonly wasmPath?: string | URL;
  readonly accelerator?: AcceleratorPreference;
  readonly signal?: AbortSignal;
  /**
   * Records the model load and every later generation of this model with an application-owned
   * `@xybrid/observability` observer. Without it nothing is recorded.
   */
  readonly observer?: Observer;
  /** Trace placement of the `load_model` operation. */
  readonly observation?: TraceOptions;
};

export type DownloadProgress = {
  readonly loadedBytes: number;
  readonly totalBytes: number | undefined;
};

export type LlmLoadOptions = LoadOptions & {
  readonly onDownloadProgress?: (progress: DownloadProgress) => void;
};

export type RegistryLoadOptions = LoadOptions & {
  readonly registryUrl?: string | URL;
  readonly version?: string;
  readonly onDownloadProgress?: (progress: DownloadProgress) => void;
};

export type HuggingFaceLoadOptions = LoadOptions & {
  readonly revision?: string;
  readonly file?: string;
  readonly onDownloadProgress?: (progress: DownloadProgress) => void;
};

export type GenerateOptions = {
  readonly maxOutputTokens?: number;
  /** Trace placement and GenAI request details; used only when the model has an observer. */
  readonly observation?: GenerateObservation;
};

/** Load a GGUF URL directly, optionally verifying its declared size and SHA-256. */
export type GgufLoadOptions = LlmLoadOptions & {
  readonly contextLength?: number;
  readonly sizeBytes?: number;
  readonly sha256?: string;
};
