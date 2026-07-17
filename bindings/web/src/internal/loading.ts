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

export type MetadataLoader = (url: URL, signal?: AbortSignal) => Promise<ParsedMetadata>;

type InitializableRuntime = Parameters<RuntimeInitializer["initialize"]>[0];

type LoadPrelude = {
  readonly metadata: Promise<ParsedMetadata>;
};

export type AcceleratorPreflight =
  | { readonly status: "not-requested" | "available" | "unavailable" }
  | { readonly status: "failed"; readonly error: unknown };

export type LoadCancellation = {
  readonly signal: AbortSignal;
  abort(reason?: unknown): void;
  cleanup(): void;
};

const createAbortError = (): DOMException =>
  new DOMException("The operation was aborted.", "AbortError");

export const isAbortError = (error: unknown): boolean =>
  typeof error === "object" &&
  error !== null &&
  "name" in error &&
  (error as { readonly name?: unknown }).name === "AbortError";

export const abortReason = (signal: AbortSignal): unknown => signal.reason ?? createAbortError();

const callerAbortReason = (signal: AbortSignal): unknown =>
  isAbortError(signal.reason) ? signal.reason : createAbortError();

export const throwIfAborted = (signal: AbortSignal | undefined): void => {
  if (signal?.aborted) {
    throw abortReason(signal);
  }
};

export const createLoadCancellation = (callerSignal: AbortSignal | undefined): LoadCancellation => {
  const controller = new AbortController();
  const onCallerAbort = (): void => {
    if (callerSignal !== undefined) {
      controller.abort(callerAbortReason(callerSignal));
    }
  };
  if (callerSignal?.aborted) {
    onCallerAbort();
  } else {
    callerSignal?.addEventListener("abort", onCallerAbort, { once: true });
  }
  return {
    signal: controller.signal,
    abort: (reason) => {
      if (!controller.signal.aborted) {
        controller.abort(reason);
      }
    },
    cleanup: () => callerSignal?.removeEventListener("abort", onCallerAbort),
  };
};

export const runWithLoadCancellation = async <T>(
  callerSignal: AbortSignal | undefined,
  load: (cancellation: LoadCancellation) => Promise<T>,
): Promise<T> => {
  const cancellation = createLoadCancellation(callerSignal);
  try {
    return await load(cancellation);
  } catch (error: unknown) {
    cancellation.abort(error);
    if (callerSignal?.aborted) {
      throw callerAbortReason(callerSignal);
    }
    throw error;
  } finally {
    cancellation.cleanup();
  }
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
    accelerator,
    wasmPath: resolveWasmPath(wasmPath, base),
    signal: signal as AbortSignal | undefined,
  };
};

export type NormalizedLoadOptions = {
  readonly accelerator: NonNullable<LoadOptions["accelerator"]>;
  readonly wasmPath: URL;
  readonly signal: AbortSignal | undefined;
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
  return {
    accelerator: normalizedBase.accelerator,
    wasmPath: normalizedBase.wasmPath,
    registryUrl,
    version: version as string | undefined,
    onDownloadProgress: onDownloadProgress as RegistryLoadOptions["onDownloadProgress"],
    signal: normalizedBase.signal,
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
  return {
    accelerator: normalizedBase.accelerator,
    wasmPath: normalizedBase.wasmPath,
    revision: revision as string | undefined,
    file: file as string | undefined,
    onDownloadProgress: onDownloadProgress as HuggingFaceLoadOptions["onDownloadProgress"],
    signal: normalizedBase.signal,
  };
};

export const startLoadPrelude = (
  metadataSource: URL | ParsedMetadata,
  getMetadata: MetadataLoader,
  signal?: AbortSignal,
): LoadPrelude => {
  const metadata = Promise.resolve()
    .then(() => {
      throwIfAborted(signal);
      return metadataSource instanceof URL ? getMetadata(metadataSource, signal) : metadataSource;
    })
    .catch((error: unknown) => {
      if (signal?.aborted) {
        throw abortReason(signal);
      }
      if (error instanceof XybridError || isAbortError(error)) {
        throw error;
      }
      throw new InvalidMetadataError("Failed to load model metadata.", error);
    });
  return { metadata };
};

export const startRuntimeInitialization = (
  wasmPath: string | URL,
  runtime: InitializableRuntime,
  initializer: RuntimeInitializer,
): Promise<void> => {
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
  return initialization;
};

export const validateAcceleratorPreference: (
  preference: unknown,
) => asserts preference is LoadOptions["accelerator"] = (preference) => {
  if (preference !== "auto" && preference !== "wasm" && preference !== "webgpu") {
    throw new RuntimeConfigurationError("accelerator must be auto, wasm, or webgpu.");
  }
};

export const loadMetadata: MetadataLoader = async (url, signal) => {
  const response = await ky.get(url, {
    credentials: "omit",
    ...(signal === undefined ? {} : { signal }),
  });
  const bytes = await readResponseBytes(
    response,
    MAX_METADATA_BYTES,
    "Model metadata exceeds the 1 MiB browser limit.",
    undefined,
    signal,
  );
  const text = new TextDecoder().decode(bytes);
  return parseMetadata(JSON.parse(text));
};

const asRuntimeInitializationError = (error: unknown): unknown => {
  if (isAbortError(error)) {
    return error;
  }
  if (error instanceof XybridError) {
    return error;
  }
  return new RuntimeInitializationError(error);
};

export const preflightAccelerator = async (
  preference: LoadOptions["accelerator"],
  initialization: Promise<void>,
  probe: () => Promise<void>,
  signal?: AbortSignal,
): Promise<AcceleratorPreflight> => {
  throwIfAborted(signal);
  if (preference === "wasm") {
    return { status: "not-requested" };
  }
  try {
    await initialization;
    throwIfAborted(signal);
    await probe();
    throwIfAborted(signal);
    return { status: "available" };
  } catch (error: unknown) {
    if (signal?.aborted) {
      throw abortReason(signal);
    }
    if (isAbortError(error)) {
      throw error;
    }
    if (preference === "webgpu") {
      throw asRuntimeInitializationError(error);
    }
    if (error instanceof AcceleratorUnavailableError) {
      return { status: "unavailable" };
    }
    return { status: "failed", error };
  }
};

export const loadAfterMetadata = async <T>(
  preference: LoadOptions["accelerator"],
  initialization: Promise<void>,
  probe: () => Promise<void>,
  download: () => Promise<T>,
  cancellation: LoadCancellation,
): Promise<{ readonly value: T; readonly preflight: AcceleratorPreflight }> => {
  let firstFailure: { readonly error: unknown } | undefined;
  const runStage = async <Value>(stage: () => Promise<Value>): Promise<Value> => {
    try {
      throwIfAborted(cancellation.signal);
      return await stage();
    } catch (error: unknown) {
      if (firstFailure === undefined) {
        firstFailure = { error };
      }
      cancellation.abort(error);
      throw error;
    }
  };
  if (preference === "webgpu") {
    const preflight = await runStage(() =>
      preflightAccelerator(preference, initialization, probe, cancellation.signal),
    );
    const value = await runStage(download);
    return { value, preflight };
  }

  if (preference === "auto") {
    const downloadPromise = runStage(download);
    const preflightPromise = runStage(() =>
      preflightAccelerator(preference, initialization, probe, cancellation.signal),
    );
    await Promise.allSettled([downloadPromise, preflightPromise]);
    if (firstFailure !== undefined) {
      throw firstFailure.error;
    }
    const [value, preflight] = await Promise.all([downloadPromise, preflightPromise]);
    return { value, preflight };
  }

  const downloadPromise = runStage(download);
  const initializationPromise = runStage(async () => initialization);
  await Promise.allSettled([downloadPromise, initializationPromise]);
  if (firstFailure !== undefined) {
    throw firstFailure.error;
  }
  return {
    value: await downloadPromise,
    preflight: { status: "not-requested" },
  };
};

export const selectAccelerated = async <T>(
  preference: LoadOptions["accelerator"],
  create: (accelerator: SelectedAccelerator) => Promise<T>,
  preflight: AcceleratorPreflight = { status: "available" },
  signal?: AbortSignal,
  dispose?: (value: T) => void | Promise<void>,
): Promise<{ readonly value: T; readonly accelerator: SelectedAccelerator }> => {
  const createOne = async (accelerator: SelectedAccelerator) => {
    throwIfAborted(signal);
    let value: T | undefined;
    try {
      value = await create(accelerator);
      throwIfAborted(signal);
      return value;
    } catch (error: unknown) {
      if (value !== undefined && signal?.aborted) {
        await Promise.resolve(dispose?.(value)).catch(() => undefined);
      }
      if (signal?.aborted) {
        throw abortReason(signal);
      }
      if (isAbortError(error)) {
        throw error;
      }
      if (error instanceof XybridError || error instanceof AcceleratorUnavailableError) {
        throw error;
      }
      throw new RuntimeInitializationError(error);
    }
  };
  const createWebGpu = async (): Promise<T> => {
    throwIfAborted(signal);
    if (preflight.status === "unavailable") {
      throw new AcceleratorUnavailableError("WebGPU is unavailable.");
    }
    if (preflight.status === "failed") {
      throw asRuntimeInitializationError(preflight.error);
    }
    return createOne("webgpu");
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
    const value = await createWebGpu();
    return { value, accelerator: "webgpu" };
  } catch (webgpuError: unknown) {
    if (signal?.aborted) {
      throw abortReason(signal);
    }
    if (webgpuError instanceof AcceleratorUnavailableError) {
      const value = await createOne("wasm");
      return { value, accelerator: "wasm" };
    }
    try {
      const value = await createOne("wasm");
      return { value, accelerator: "wasm" };
    } catch (wasmError: unknown) {
      throw new RuntimeInitializationError(
        new AggregateError(
          [webgpuError, wasmError],
          "WebGPU and wasm accelerator initialization both failed.",
        ),
      );
    }
  }
};

export const compileModelBytes = async (
  runtime: BrowserRuntime,
  bytes: Uint8Array,
  preference: LoadOptions["accelerator"],
  preflight: AcceleratorPreflight = { status: "available" },
  signal?: AbortSignal,
): Promise<{ readonly model: RuntimeModel; readonly accelerator: SelectedAccelerator }> => {
  const { value, accelerator } = await selectAccelerated(
    preference,
    (target) => runtime.compileBytes(bytes, target),
    preflight,
    signal,
    (model) => model.delete(),
  );
  return { model: value, accelerator };
};
