import ky from "ky";

import { RuntimeInitializationError } from "../errors.ts";
import type { DownloadProgress } from "../types.ts";
import { readResponseBytes, readResponseChunks } from "./response.ts";

const MAX_MODEL_BYTES = 512 * 1024 * 1024;
const MODEL_LIMIT_MESSAGE = "Model exceeds the 512 MiB browser limit.";
const MODEL_DOWNLOAD_TIMEOUT_MS = 600_000;

type ResponseReader<Model> = (
  response: Response,
  maximumBytes: number,
  limitMessage: string,
  onProgress?: (loadedBytes: number, totalBytes: number | undefined) => void,
) => Promise<Model>;

const downloadModel = async <Model>(
  modelUrl: URL,
  readResponse: ResponseReader<Model>,
  onProgress?: (progress: DownloadProgress) => void,
): Promise<Model> => {
  try {
    const response = await ky.get(modelUrl, { timeout: MODEL_DOWNLOAD_TIMEOUT_MS });
    return await readResponse(
      response,
      MAX_MODEL_BYTES,
      MODEL_LIMIT_MESSAGE,
      onProgress === undefined
        ? undefined
        : (loadedBytes, totalBytes) => onProgress({ loadedBytes, totalBytes }),
    );
  } catch (error: unknown) {
    throw new RuntimeInitializationError(error);
  }
};

export const loadModelBytes = (modelUrl: URL): Promise<Uint8Array<ArrayBuffer>> =>
  downloadModel(modelUrl, readResponseBytes);

export const loadModelChunks = (
  modelUrl: URL,
  onProgress: ((progress: DownloadProgress) => void) | undefined,
): Promise<Uint8Array<ArrayBuffer>[]> => downloadModel(modelUrl, readResponseChunks, onProgress);
