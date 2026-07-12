import ky from "ky";

import { RuntimeConfigurationError, RuntimeInitializationError, XybridError } from "../errors.ts";
import { type ParsedMetadata, parseMetadata } from "../metadata.ts";
import type { LoadOptions, SelectedAccelerator } from "../types.ts";
import { readResponseBytes } from "./response.ts";
import type { BrowserRuntime, RuntimeModel } from "./runtime.ts";
import { AcceleratorUnavailableError } from "./runtime.ts";

const MAX_METADATA_BYTES = 1024 * 1024;

export type MetadataLoader = (url: URL) => Promise<ParsedMetadata>;

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
  } catch (error: unknown) {
    if (!(error instanceof AcceleratorUnavailableError)) {
      throw error;
    }
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
