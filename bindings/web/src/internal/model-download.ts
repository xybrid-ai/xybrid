import ky from "ky";

import { RuntimeInitializationError } from "../errors.ts";
import type { DownloadProgress } from "../types.ts";
import { abortReason, isAbortError } from "./loading.ts";
import { readResponseBytes, readResponseChunks } from "./response.ts";

const MAX_MODEL_BYTES = 512 * 1024 * 1024;
const MODEL_LIMIT_MESSAGE = "Model exceeds the 512 MiB browser limit.";
const MODEL_DOWNLOAD_TIMEOUT_MS = 600_000;

type ResponseReader<Model> = (
  response: Response,
  maximumBytes: number,
  limitMessage: string,
  onProgress?: (loadedBytes: number, totalBytes: number | undefined) => void,
  signal?: AbortSignal,
) => Promise<Model>;

const downloadModel = async <Model>(
  modelUrl: URL,
  readResponse: ResponseReader<Model>,
  onProgress?: (progress: DownloadProgress) => void,
  signal?: AbortSignal,
): Promise<Model> => {
  try {
    const response = await ky.get(modelUrl, {
      credentials: "omit",
      timeout: MODEL_DOWNLOAD_TIMEOUT_MS,
      ...(signal === undefined ? {} : { signal }),
    });
    return await readResponse(
      response,
      MAX_MODEL_BYTES,
      MODEL_LIMIT_MESSAGE,
      onProgress === undefined
        ? undefined
        : (loadedBytes, totalBytes) => onProgress({ loadedBytes, totalBytes }),
      signal,
    );
  } catch (error: unknown) {
    if (signal?.aborted) {
      throw abortReason(signal);
    }
    if (isAbortError(error)) {
      throw error;
    }
    throw new RuntimeInitializationError(error);
  }
};

export const loadModelBytes = (
  modelUrl: URL,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> =>
  downloadModel(modelUrl, readResponseBytes, undefined, signal);

export const loadModelChunks = (
  modelUrl: URL,
  onProgress: ((progress: DownloadProgress) => void) | undefined,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>[]> =>
  downloadModel(
    modelUrl,
    (response, maximumBytes, limitMessage, progress, responseSignal) =>
      readResponseChunks(response, maximumBytes, limitMessage, progress, undefined, responseSignal),
    onProgress,
    signal,
  );
