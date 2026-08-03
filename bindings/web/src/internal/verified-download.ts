import ky from "ky";

import { IntegrityError, RuntimeInitializationError, XybridError } from "../errors.ts";
import type { DownloadProgress } from "../types.ts";
import { abortReason, isAbortError } from "./loading.ts";
import { readResponseChunks } from "./response.ts";

const MODEL_DOWNLOAD_TIMEOUT_MS = 600_000;

export type VerifiedDownloadOptions = {
  readonly sizeBytes: number;
  readonly sha256: string | undefined;
  readonly onProgress?: ((progress: DownloadProgress) => void) | undefined;
  readonly signal?: AbortSignal | undefined;
};

export const downloadVerifiedModel = async (
  modelUrl: URL,
  options: VerifiedDownloadOptions,
): Promise<Uint8Array<ArrayBuffer>[]> => {
  try {
    const response = await ky.get(modelUrl, {
      credentials: "omit",
      retry: 0,
      timeout: MODEL_DOWNLOAD_TIMEOUT_MS,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
    const hasher =
      options.sha256 === undefined
        ? undefined
        : await (async () => {
            const { createSHA256 } = await import("hash-wasm");
            const value = await createSHA256();
            value.init();
            return value;
          })();
    let totalBytes = 0;
    const chunks = await readResponseChunks(
      response,
      Number.MAX_SAFE_INTEGER,
      "Verified model exceeds the declared size.",
      (loadedBytes) => {
        options.onProgress?.({ loadedBytes, totalBytes: options.sizeBytes });
      },
      (chunk) => {
        totalBytes += chunk.byteLength;
        if (totalBytes > options.sizeBytes) {
          throw new IntegrityError(
            `Verified model exceeds the declared size of ${options.sizeBytes} bytes (received ${totalBytes}).`,
          );
        }
        hasher?.update(chunk);
      },
      options.signal,
    );
    if (totalBytes !== options.sizeBytes) {
      throw new IntegrityError(
        `Verified model size mismatch: expected ${options.sizeBytes} bytes, received ${totalBytes}.`,
      );
    }
    if (options.sha256 !== undefined && hasher !== undefined) {
      const actualSha256 = hasher.digest("hex");
      if (actualSha256 !== options.sha256) {
        throw new IntegrityError(
          `Verified model SHA-256 mismatch: expected ${options.sha256}, received ${actualSha256}.`,
        );
      }
    }
    return chunks;
  } catch (error: unknown) {
    if (options.signal?.aborted) {
      throw abortReason(options.signal);
    }
    if (isAbortError(error)) {
      throw error;
    }
    if (error instanceof XybridError) {
      throw error;
    }
    throw new RuntimeInitializationError(error);
  }
};
