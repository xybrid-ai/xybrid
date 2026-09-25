// Minimal base64 for the one binary payload JS produces itself: live-ASR PCM.
// Hermes has no Buffer, and `btoa` only takes binary strings, so encode the
// bytes directly. Every platform React Native runs on is little-endian, which
// is the byte order the native shims decode Float32 samples in.

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/** Standard (RFC 4648, padded) base64 of `bytes`. */
export function bytesToBase64(bytes: Uint8Array): string {
  const chunks: string[] = [];
  let chunk = '';
  let i = 0;
  for (; i + 2 < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    chunk +=
      ALPHABET[(n >> 18) & 63] +
      ALPHABET[(n >> 12) & 63] +
      ALPHABET[(n >> 6) & 63] +
      ALPHABET[n & 63];
    // Flush periodically so long inputs don't build one enormous rope.
    if (chunk.length >= 8192) {
      chunks.push(chunk);
      chunk = '';
    }
  }
  const rest = bytes.length - i;
  if (rest === 1) {
    const n = bytes[i] << 16;
    chunk += ALPHABET[(n >> 18) & 63] + ALPHABET[(n >> 12) & 63] + '==';
  } else if (rest === 2) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8);
    chunk +=
      ALPHABET[(n >> 18) & 63] + ALPHABET[(n >> 12) & 63] + ALPHABET[(n >> 6) & 63] + '=';
  }
  chunks.push(chunk);
  return chunks.join('');
}

/** Base64 of Float32 PCM samples as little-endian bytes. */
export function float32ToBase64(samples: Float32Array | ArrayLike<number>): string {
  const floats = samples instanceof Float32Array ? samples : Float32Array.from(samples);
  return bytesToBase64(new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength));
}
