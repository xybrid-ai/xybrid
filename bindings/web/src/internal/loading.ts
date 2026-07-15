import ky from "ky";

import {
  InvalidMetadataError,
  RuntimeConfigurationError,
  RuntimeInitializationError,
  XybridError,
} from "../errors.ts";
import { type ParsedMetadata, parseMetadata } from "../metadata.ts";
import type {
  HuggingFaceLoadOptions,
  LoadOptions,
  RegistryLoadOptions,
  SelectedAccelerator,
} from "../types.ts";
import type { RuntimeInitializer } from "./initialization.ts";
import { readResponseBytes } from "./response.ts";
import type { BrowserRuntime, RuntimeInitConfig, RuntimeModel } from "./runtime.ts";
import { AcceleratorUnavailableError } from "./runtime.ts";
import { resolveWasmPath } from "./url.ts";

const MAX_METADATA_BYTES = 1024 * 1024;

export type MetadataLoader = (url: URL) => Promise<ParsedMetadata>;

type InitializableRuntime = Parameters<RuntimeInitializer["initialize"]>[0];

type LoadPrelude = {
  readonly metadata: Promise<ParsedMetadata>;
  readonly initialization: Promise<void>;
};

export const normalizeBaseLoadOptions = (
  options: unknown,
  base: string | undefined,
  defaultWasmPath: string,
): NormalizedLoadOptions => {
  if (typeof options !== "object" || options === null) {
    throw new RuntimeConfigurationError("load options must be an object.");
  }
  const values = options as Record<string, unknown>;
  const accelerator = values["accelerator"] ?? "auto";
  validateAcceleratorPreference(accelerator);
  const wasmPath = values["wasmPath"] ?? defaultWasmPath;
  if (typeof wasmPath !== "string" && !(wasmPath instanceof URL)) {
    throw new RuntimeConfigurationError("wasmPath must be a string or URL.");
  }
  return {
    accelerator,
    wasmPath: resolveWasmPath(wasmPath, base),
  };
};

export type NormalizedLoadOptions = {
  readonly accelerator: NonNullable<LoadOptions["accelerator"]>;
  readonly wasmPath: URL;
};

export type NormalizedRegistryLoadOptions = {
  readonly accelerator: NonNullable<LoadOptions["accelerator"]>;
  readonly wasmPath: URL;
  readonly registryUrl: URL | undefined;
  readonly version: string | undefined;
  readonly onDownloadProgress: RegistryLoadOptions["onDownloadProgress"];
  readonly signal: AbortSignal | undefined;
};

export type NormalizedHuggingFaceLoadOptions = {
  readonly accelerator: NonNullable<LoadOptions["accelerator"]>;
  readonly wasmPath: URL;
  readonly revision: string | undefined;
  readonly file: string | undefined;
  readonly onDownloadProgress: HuggingFaceLoadOptions["onDownloadProgress"];
  readonly signal: AbortSignal | undefined;
};

export const normalizeRegistryLoadOptions = (
  options: unknown,
  base: string | undefined,
  defaultWasmPath: string,
): NormalizedRegistryLoadOptions => {
  const normalizedBase = normalizeBaseLoadOptions(
    options === undefined ? {} : options,
    base,
    defaultWasmPath,
  );
  const values = options === undefined ? {} : (options as Record<string, unknown>);
  const onDownloadProgress = values["onDownloadProgress"];
  if (onDownloadProgress !== undefined && typeof onDownloadProgress !== "function") {
    throw new RuntimeConfigurationError("onDownloadProgress must be a function.");
  }
  const rawRegistryUrl = values["registryUrl"];
  let registryUrl: URL | undefined;
  if (rawRegistryUrl !== undefined) {
    if (typeof rawRegistryUrl !== "string" && !(rawRegistryUrl instanceof URL)) {
      throw new RuntimeConfigurationError("registryUrl must be a string or URL.");
    }
    try {
      registryUrl = new URL(rawRegistryUrl);
      if (registryUrl.protocol !== "https:") {
        throw new RuntimeConfigurationError("registryUrl must use HTTPS.");
      }
    } catch (error: unknown) {
      if (error instanceof RuntimeConfigurationError) {
        throw error;
      }
      throw new RuntimeConfigurationError("registryUrl must be a valid HTTPS URL.");
    }
  }
  const version = values["version"];
  if (version !== undefined && typeof version !== "string") {
    throw new RuntimeConfigurationError("version must be a string.");
  }
  const signal = values["signal"];
  if (
    signal !== undefined &&
    (typeof signal !== "object" ||
      signal === null ||
      typeof (signal as AbortSignal).aborted !== "boolean")
  ) {
    throw new RuntimeConfigurationError("signal must be an AbortSignal.");
  }
  return {
    accelerator: normalizedBase.accelerator,
    wasmPath: normalizedBase.wasmPath,
    registryUrl,
    version: version as string | undefined,
    onDownloadProgress: onDownloadProgress as RegistryLoadOptions["onDownloadProgress"],
    signal: signal as AbortSignal | undefined,
  };
};

export const normalizeHuggingFaceLoadOptions = (
  options: unknown,
  base: string | undefined,
  defaultWasmPath: string,
): NormalizedHuggingFaceLoadOptions => {
  const normalizedBase = normalizeBaseLoadOptions(
    options === undefined ? {} : options,
    base,
    defaultWasmPath,
  );
  const values = options === undefined ? {} : (options as Record<string, unknown>);
  const revision = values["revision"];
  if (revision !== undefined && typeof revision !== "string") {
    throw new RuntimeConfigurationError("revision must be a string.");
  }
  const file = values["file"];
  if (file !== undefined && typeof file !== "string") {
    throw new RuntimeConfigurationError("file must be a string.");
  }
  const onDownloadProgress = values["onDownloadProgress"];
  if (onDownloadProgress !== undefined && typeof onDownloadProgress !== "function") {
    throw new RuntimeConfigurationError("onDownloadProgress must be a function.");
  }
  const signal = values["signal"];
  if (
    signal !== undefined &&
    (typeof signal !== "object" ||
      signal === null ||
      typeof (signal as AbortSignal).aborted !== "boolean")
  ) {
    throw new RuntimeConfigurationError("signal must be an AbortSignal.");
  }
  return {
    accelerator: normalizedBase.accelerator,
    wasmPath: normalizedBase.wasmPath,
    revision: revision as string | undefined,
    file: file as string | undefined,
    onDownloadProgress: onDownloadProgress as HuggingFaceLoadOptions["onDownloadProgress"],
    signal: signal as AbortSignal | undefined,
  };
};

export const startLoadPrelude = (
  metadataSource: URL | ParsedMetadata,
  wasmPath: string | URL,
  runtime: InitializableRuntime,
  getMetadata: MetadataLoader,
  initializer: RuntimeInitializer,
): LoadPrelude => {
  const metadata = Promise.resolve()
    .then(() => (metadataSource instanceof URL ? getMetadata(metadataSource) : metadataSource))
    .catch((error: unknown) => {
      if (error instanceof XybridError) {
        throw error;
      }
      throw new InvalidMetadataError("Failed to load model metadata.", error);
    });
  let initialization: Promise<void>;
  try {
    initialization = initializer.initialize(runtime, {
      wasmPath,
      threads: false,
      jspi: false,
    } satisfies RuntimeInitConfig);
  } catch (error: unknown) {
    initialization = Promise.reject(error);
  }
  void Promise.allSettled([initialization]);
  return { metadata, initialization };
};

export const validateAcceleratorPreference: (
  preference: unknown,
) => asserts preference is LoadOptions["accelerator"] = (preference) => {
  if (preference !== "auto" && preference !== "wasm" && preference !== "webgpu") {
    throw new RuntimeConfigurationError("accelerator must be auto, wasm, or webgpu.");
  }
};

export const loadMetadata: MetadataLoader = async (url) => {
  const response = await ky.get(url);
  const bytes = await readResponseBytes(
    response,
    MAX_METADATA_BYTES,
    "Model metadata exceeds the 1 MiB browser limit.",
  );
  const text = new TextDecoder().decode(bytes);
  return parseMetadata(JSON.parse(text));
};

export const selectAccelerated = async <T>(
  preference: LoadOptions["accelerator"],
  create: (accelerator: SelectedAccelerator) => Promise<T>,
): Promise<{ readonly value: T; readonly accelerator: SelectedAccelerator }> => {
  const createOne = async (accelerator: SelectedAccelerator) => {
    try {
      return await create(accelerator);
    } catch (error: unknown) {
      if (error instanceof XybridError || error instanceof AcceleratorUnavailableError) {
        throw error;
      }
      throw new RuntimeInitializationError(error);
    }
  };
  const createExplicit = async (accelerator: SelectedAccelerator) => {
    try {
      const value = await createOne(accelerator);
      return { value, accelerator };
    } catch (error: unknown) {
      if (error instanceof AcceleratorUnavailableError) {
        throw new RuntimeInitializationError(error);
      }
      throw error;
    }
  };
  if (preference === "wasm") {
    return createExplicit("wasm");
  }
  if (preference === "webgpu") {
    return createExplicit("webgpu");
  }
  try {
    const value = await createOne("webgpu");
    return { value, accelerator: "webgpu" };
  } catch (_error: unknown) {
    const value = await createOne("wasm");
    return { value, accelerator: "wasm" };
  }
};

export const compileModel = async (
  runtime: BrowserRuntime,
  modelUrl: URL,
  preference: LoadOptions["accelerator"],
): Promise<{ readonly model: RuntimeModel; readonly accelerator: SelectedAccelerator }> => {
  const { value, accelerator } = await selectAccelerated(preference, (target) =>
    runtime.compile(modelUrl, target),
  );
  return { model: value, accelerator };
};

export const compileModelBytes = async (
  runtime: BrowserRuntime,
  bytes: Uint8Array,
  preference: LoadOptions["accelerator"],
): Promise<{ readonly model: RuntimeModel; readonly accelerator: SelectedAccelerator }> => {
  const { value, accelerator } = await selectAccelerated(preference, (target) =>
    runtime.compileBytes(bytes, target),
  );
  return { model: value, accelerator };
};
