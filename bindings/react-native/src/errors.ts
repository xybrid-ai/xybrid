// Every native rejection carries one of these codes as `error.code`. The
// SDK codes map 1:1 onto bolt's `XybridError` variants (same order as
// crates/xybrid-bolt/src/lib.rs); the last three are raised by the React
// Native layer itself.

/** Stable error codes for `XybridError.code`. */
export const XybridErrorCodes = [
  'xybrid_model_not_found',
  'xybrid_directory_not_found',
  'xybrid_metadata_not_found',
  'xybrid_metadata_invalid',
  'xybrid_load_error',
  'xybrid_inference_error',
  'xybrid_aborted_cloud_fallback',
  'xybrid_streaming_unsupported',
  'xybrid_not_loaded',
  'xybrid_config_error',
  'xybrid_network_error',
  'xybrid_offline',
  'xybrid_io_error',
  'xybrid_cache_error',
  'xybrid_pipeline_error',
  'xybrid_circuit_open',
  'xybrid_rate_limited',
  'xybrid_timeout',
  'xybrid_missing_artifact',
  'xybrid_unsupported_model_capability',
  'xybrid_unsupported_backend_capability',
  'xybrid_invalid_image',
  'xybrid_cancelled',
  // React Native layer:
  /** The handle was never issued, was released, or names another kind of object. */
  'xybrid_handle',
  /** A JS argument could not be decoded (bad envelope, options, source, base64…). */
  'xybrid_invalid_argument',
  /** Anything unexpected; the message carries the detail. */
  'xybrid_unknown',
] as const;

export type XybridErrorCode = (typeof XybridErrorCodes)[number];

/** A rejection raised by the Xybrid native module. */
export interface XybridError extends Error {
  code: XybridErrorCode;
}

const CODES: ReadonlySet<string> = new Set(XybridErrorCodes);

/** Narrow an unknown rejection to a {@link XybridError}. */
export function isXybridError(error: unknown): error is XybridError {
  if (!(error instanceof Error)) return false;
  const { code } = error as Error & { code?: unknown };
  return typeof code === 'string' && CODES.has(code);
}

// Mirrors `RetryableError::is_retryable` on the Rust `SdkError`.
const RETRYABLE: ReadonlySet<XybridErrorCode> = new Set<XybridErrorCode>([
  'xybrid_network_error',
  'xybrid_rate_limited',
  'xybrid_timeout',
  'xybrid_offline',
]);

/** Whether retrying the same call later can succeed (network, rate limit, timeout, offline). */
export function isRetryable(error: unknown): boolean {
  return isXybridError(error) && RETRYABLE.has(error.code);
}
