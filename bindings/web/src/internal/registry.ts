import ky, { isHTTPError, isTimeoutError } from "ky";
import { z } from "zod";

import { RegistryError } from "../errors.ts";
import {
  type ParsedMetadata,
  parseMetadata,
  validateBrowserMetadata,
  validateLlmBrowserMetadata,
} from "../metadata.ts";

const DEFAULT_REGISTRY_URLS = ["https://registry.xybrid.dev", "https://r2.xybrid.dev"] as const;
const REGISTRY_TIMEOUT_MS = 30_000;
const MAX_MODEL_BYTES = 512 * 1024 * 1024;

const resolvedSchema = z
  .object({
    hf_repo: z.string(),
    file: z.string(),
    download_url: z.string(),
    format: z.string(),
    quantization: z.string(),
    size_bytes: z.number(),
    sha256: z.string(),
    passthrough: z.boolean(),
    model_metadata: z.unknown().optional(),
    artifacts: z.array(z.unknown()).optional(),
  })
  .loose();

const registryEnvelopeSchema = z
  .object({
    mask: z.string(),
    platform: z.string(),
    resolved: resolvedSchema,
  })
  .loose();

export type ModelResolution = {
  readonly modelUrl: URL;
  readonly metadata: ParsedMetadata;
  readonly sizeBytes: number;
  readonly sha256: string | undefined;
};

export type RegistryResolution = Omit<ModelResolution, "sha256"> & {
  readonly sha256: string;
};

export type RegistryResolveOptions = {
  readonly registryUrl?: string | URL | undefined;
  readonly version?: string | undefined;
  readonly signal?: AbortSignal | undefined;
};

const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
};

const validateModelId = (id: unknown): string => {
  if (typeof id !== "string" || id.length === 0 || /[/?#%\\]/.test(id) || /\s/.test(id)) {
    throw new RegistryError(
      "Model id must be a non-empty path-safe string without slashes, URL delimiters, percent signs, or whitespace.",
    );
  }
  return id;
};

const registryEndpoint = (
  registryUrl: string | URL,
  id: string,
  expectedFormat: "litertlm" | "tflite",
  version: string | undefined,
): URL => {
  const base = new URL(registryUrl);
  const endpoint = new URL(
    `${base.toString().replace(/\/+$/, "")}/v1/models/${encodeURIComponent(id)}/resolve`,
  );
  endpoint.searchParams.set("platform", "web");
  endpoint.searchParams.set("format", expectedFormat);
  if (version !== undefined) {
    endpoint.searchParams.set("version", version);
  }
  return endpoint;
};

const normalizeRegistryUrl = (value: string | URL): URL => {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:") {
      throw new RegistryError("registryUrl must use HTTPS.");
    }
    return url;
  } catch (error: unknown) {
    if (error instanceof RegistryError) {
      throw error;
    }
    throw new RegistryError("registryUrl must be a valid HTTPS URL.", error);
  }
};

const isNetworkError = (error: unknown): boolean => {
  if (isHTTPError(error) || error instanceof RegistryError || isTimeoutError(error)) {
    return isTimeoutError(error);
  }
  if (error instanceof DOMException) {
    return error.name === "NetworkError";
  }
  return error instanceof TypeError;
};

const validateResolution = (
  input: unknown,
  expectedFormat: "litertlm" | "tflite",
): RegistryResolution => {
  const parsed = registryEnvelopeSchema.safeParse(input);
  if (!parsed.success) {
    throw new RegistryError(
      "Registry response does not contain a valid resolved model.",
      parsed.error,
    );
  }
  const resolved = parsed.data.resolved;
  if (resolved.format !== expectedFormat) {
    throw new RegistryError(
      `Registry resolved format ${resolved.format} instead of the required ${expectedFormat}.`,
    );
  }
  if (
    !Number.isSafeInteger(resolved.size_bytes) ||
    resolved.size_bytes <= 0 ||
    resolved.size_bytes > MAX_MODEL_BYTES
  ) {
    throw new RegistryError(
      `Registry resolved size_bytes must be a positive safe integer no greater than ${MAX_MODEL_BYTES}.`,
    );
  }
  if (!/^[0-9a-f]{64}$/.test(resolved.sha256)) {
    throw new RegistryError(
      "Registry resolved sha256 must be a 64-character lowercase hex string.",
    );
  }
  if (resolved.passthrough !== true) {
    throw new RegistryError("Registry resolved model must have passthrough=true.");
  }
  if (resolved.artifacts !== undefined && resolved.artifacts.length > 0) {
    throw new RegistryError("Registry resolved model must not include artifacts.");
  }
  if (!isPlainObject(resolved.model_metadata)) {
    throw new RegistryError("Registry resolved model_metadata must be a present plain object.");
  }
  let modelUrl: URL;
  try {
    modelUrl = new URL(resolved.download_url);
  } catch (error: unknown) {
    throw new RegistryError("Registry resolved download_url must be a valid HTTPS URL.", error);
  }
  if (modelUrl.protocol !== "https:") {
    throw new RegistryError("Registry resolved download_url must use HTTPS.");
  }

  try {
    const metadata = parseMetadata(resolved.model_metadata);
    const modelFile =
      expectedFormat === "litertlm"
        ? validateLlmBrowserMetadata(metadata).modelFile
        : validateBrowserMetadata(metadata);
    if (resolved.file !== modelFile) {
      throw new RegistryError(
        `Registry resolved file ${resolved.file} does not match execution_template.model_file ${modelFile}.`,
      );
    }
    return {
      modelUrl,
      metadata,
      sizeBytes: resolved.size_bytes,
      sha256: resolved.sha256,
    };
  } catch (error: unknown) {
    if (error instanceof RegistryError) {
      throw error;
    }
    throw new RegistryError("Registry model metadata is not browser-compatible.", error);
  }
};

export const parseRegistryResponse = (
  input: unknown,
  expectedFormat: "litertlm" | "tflite",
): RegistryResolution => validateResolution(input, expectedFormat);

const fetchRegistryResolution = async (
  endpoint: URL,
  id: string,
  expectedFormat: "litertlm" | "tflite",
  signal: AbortSignal | undefined,
): Promise<RegistryResolution> => {
  let response: Response;
  try {
    response = await ky.get(endpoint, {
      retry: 0,
      timeout: REGISTRY_TIMEOUT_MS,
      ...(signal === undefined ? {} : { signal }),
    });
  } catch (error: unknown) {
    if (isHTTPError(error)) {
      if (error.response.status === 404) {
        throw new RegistryError(
          `No browser-compatible variant exists for model id ${id} in ${expectedFormat} format.`,
          error,
        );
      }
      throw new RegistryError(
        `Registry resolve failed with HTTP ${error.response.status} for model id ${id}.`,
        error,
      );
    }
    throw error;
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch (error: unknown) {
    throw new RegistryError("Registry resolve response was not valid JSON.", error);
  }
  return validateResolution(body, expectedFormat);
};

export const resolveRegistryModel = async (
  id: unknown,
  expectedFormat: "litertlm" | "tflite",
  options: RegistryResolveOptions = {},
): Promise<RegistryResolution> => {
  const validatedId = validateModelId(id);
  const registryUrls =
    options.registryUrl === undefined
      ? DEFAULT_REGISTRY_URLS
      : [normalizeRegistryUrl(options.registryUrl)];
  let lastNetworkError: unknown;
  for (const registryUrl of registryUrls) {
    try {
      return await fetchRegistryResolution(
        registryEndpoint(registryUrl, validatedId, expectedFormat, options.version),
        validatedId,
        expectedFormat,
        options.signal,
      );
    } catch (error: unknown) {
      if (!isNetworkError(error)) {
        throw error;
      }
      lastNetworkError = error;
    }
  }
  throw new RegistryError(
    `Registry network resolution failed for model id ${validatedId}.`,
    lastNetworkError,
  );
};
