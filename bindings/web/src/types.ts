export type AcceleratorPreference = "auto" | "wasm" | "webgpu";
export type SelectedAccelerator = "wasm" | "webgpu";

export type LoadOptions = {
  readonly wasmPath?: string | URL;
  readonly accelerator?: AcceleratorPreference;
};

export type DownloadProgress = {
  readonly loadedBytes: number;
  readonly totalBytes: number | undefined;
};

export type LlmLoadOptions = LoadOptions & {
  readonly onDownloadProgress?: (progress: DownloadProgress) => void;
};

export type RegistryLoadOptions = {
  readonly wasmPath?: string | URL;
  readonly accelerator?: AcceleratorPreference;
  readonly registryUrl?: string | URL;
  readonly version?: string;
  readonly onDownloadProgress?: (progress: DownloadProgress) => void;
  readonly signal?: AbortSignal;
};

export type HuggingFaceLoadOptions = {
  readonly wasmPath?: string | URL;
  readonly accelerator?: AcceleratorPreference;
  readonly revision?: string;
  readonly file?: string;
  readonly onDownloadProgress?: (progress: DownloadProgress) => void;
  readonly signal?: AbortSignal;
};

export type GenerateOptions = {
  readonly maxOutputTokens?: number;
};

export type TensorDataType = "float32" | "int32" | "uint8";
export type TensorValue =
  | Float32Array<ArrayBufferLike>
  | Int32Array<ArrayBufferLike>
  | Uint8Array<ArrayBufferLike>;

export type TensorDetail = {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dataType: TensorDataType;
};

export type TensorInput =
  | TensorValue
  | {
      readonly data: TensorValue;
      readonly shape: readonly number[];
    };

export type TensorInputs = readonly TensorInput[] | Readonly<Record<string, TensorInput>>;

export type TensorOutput = {
  readonly name: string;
  readonly shape: readonly number[];
  readonly data: TensorValue;
};

export type RunResult = {
  readonly outputs: readonly TensorOutput[];
  readonly byName: Readonly<Record<string, TensorOutput>>;
};
