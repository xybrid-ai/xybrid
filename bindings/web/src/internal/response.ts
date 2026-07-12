export const readResponseBytes = async (
  response: Response,
  maximumBytes: number,
  limitMessage: string,
  onProgress?: (loadedBytes: number, totalBytes: number | undefined) => void,
): Promise<Uint8Array<ArrayBuffer>> => {
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
    onProgress?.(bytes.byteLength, declaredTotal ?? bytes.byteLength);
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let totalBytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      totalBytes += value.byteLength;
      if (totalBytes > maximumBytes) {
        await reader.cancel(limitMessage);
        throw new Error(limitMessage);
      }
      chunks.push(value);
      onProgress?.(totalBytes, declaredTotal);
    }
  } finally {
    reader.releaseLock();
  }

  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
};
