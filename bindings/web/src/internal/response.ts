export const readResponseChunks = async (
  response: Response,
  maximumBytes: number,
  limitMessage: string,
  onProgress?: (loadedBytes: number, totalBytes: number | undefined) => void,
  onChunk?: (chunk: Uint8Array<ArrayBuffer>) => void,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>[]> => {
  if (signal?.aborted) {
    throw signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
  }
  const declaredLength = Number(response.headers.get("content-length"));
  const declaredTotal =
    Number.isFinite(declaredLength) && declaredLength > 0 ? declaredLength : undefined;
  if (declaredTotal !== undefined && declaredTotal > maximumBytes) {
    throw new Error(limitMessage);
  }

  if (response.body === null) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > maximumBytes) {
      throw new Error(limitMessage);
    }
    onChunk?.(bytes);
    onProgress?.(bytes.byteLength, declaredTotal ?? bytes.byteLength);
    return [bytes];
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let totalBytes = 0;
  let aborted = false;
  const abort = (): void => {
    aborted = true;
    void reader.cancel(signal?.reason).catch(() => undefined);
  };
  signal?.addEventListener("abort", abort, { once: true });
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (aborted || signal?.aborted) {
        throw signal?.reason ?? new DOMException("The operation was aborted.", "AbortError");
      }
      if (done) {
        break;
      }
      totalBytes += value.byteLength;
      if (totalBytes > maximumBytes) {
        throw new Error(limitMessage);
      }
      onChunk?.(value);
      chunks.push(value);
      onProgress?.(totalBytes, declaredTotal);
    }
  } catch (error: unknown) {
    // Releasing the lock alone leaves the transfer streaming in the
    // background; stop it before surfacing the failure.
    await reader
      .cancel(error instanceof Error ? error.message : String(error))
      .catch(() => undefined);
    throw error;
  } finally {
    signal?.removeEventListener("abort", abort);
    reader.releaseLock();
  }

  return chunks;
};

export const readResponseBytes = async (
  response: Response,
  maximumBytes: number,
  limitMessage: string,
  onProgress?: (loadedBytes: number, totalBytes: number | undefined) => void,
): Promise<Uint8Array<ArrayBuffer>> => {
  const chunks = await readResponseChunks(response, maximumBytes, limitMessage, onProgress);
  const totalBytes = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);

  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
};
