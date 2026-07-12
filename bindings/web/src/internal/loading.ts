import ky from "ky";

import {
  InvalidMetadataError,
  RuntimeConfigurationError,
  RuntimeInitializationError,
  XybridError,
} from "../errors.ts";
import { type ParsedMetadata, parseMetadata } from "../metadata.ts";
import type { LoadOptions, SelectedAccelerator } from "../types.ts";
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
): { readonly accelerator: LoadOptions["accelerator"]; readonly wasmPath: URL } => {
  if (typeof options !== "object" || options === null) {
    throw new RuntimeConfigurationError("load options must be an object.");
  }
  const values = options as Record<string, unknown>;
  validateAcceleratorPreference(values["accelerator"]);
  const wasmPath = values["wasmPath"];
  if (typeof wasmPath !== "string" && !(wasmPath instanceof URL)) {
    throw new RuntimeConfigurationError("wasmPath must be a string or URL.");
  }
  return {
    accelerator: values["accelerator"],
    wasmPath: resolveWasmPath(wasmPath, base),
  };
};

export const startLoadPrelude = (
  metadataUrl: URL,
  wasmPath: string | URL,
  runtime: InitializableRuntime,
  getMetadata: MetadataLoader,
  initializer: RuntimeInitializer,
): LoadPrelude => {
  const metadata = Promise.resolve()
    .then(() => getMetadata(metadataUrl))
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
