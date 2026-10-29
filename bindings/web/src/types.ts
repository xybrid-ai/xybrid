export type AcceleratorPreference = "auto" | "wasm" | "webgpu";
export type SelectedAccelerator = "wasm" | "webgpu";

export type LoadOptions = {
  readonly wasmPath?: string | URL;
  readonly accelerator?: AcceleratorPreference;
  readonly signal?: AbortSignal;
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
};

/** Load a GGUF URL directly, optionally verifying its declared size and SHA-256. */
export type GgufLoadOptions = LlmLoadOptions & {
  readonly contextLength?: number;
  readonly sizeBytes?: number;
  readonly sha256?: string;
};
