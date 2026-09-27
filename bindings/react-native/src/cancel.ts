import NativeXybrid from './NativeXybrid';

/**
 * The part of the standard `AbortSignal` this SDK needs, so any
 * implementation (React Native's `AbortController`, a polyfill) works.
 */
export interface AbortSignalLike {
  readonly aborted: boolean;
  addEventListener(type: 'abort', listener: () => void): void;
  removeEventListener(type: 'abort', listener: () => void): void;
}

/** A native cancellation token bound to an `AbortSignal` for one call. */
export interface CancelBinding {
  /** Native token handle, or `null` when the call has no signal. */
  readonly token: string | null;
  /** Detach from the signal and free the native token. Never rejects. */
  release(): Promise<void>;
}

const NO_SIGNAL: CancelBinding = { token: null, release: () => Promise.resolve() };

/**
 * Create a native stop button wired to `signal`. Aborting the signal cancels
 * the token, which stops generation at the next token boundary. A signal
 * that is already aborted cancels the token before the call starts, so the
 * call rejects with `xybrid_cancelled` instead of running.
 */
export async function bindCancel(signal: AbortSignalLike | undefined): Promise<CancelBinding> {
  if (!signal) return NO_SIGNAL;
  const token = await NativeXybrid.createCancelToken();
  const onAbort = () => {
    NativeXybrid.cancel(token).catch(() => {});
  };
  if (signal.aborted) {
    await NativeXybrid.cancel(token);
  } else {
    signal.addEventListener('abort', onAbort);
  }
  return {
    token,
    release: async () => {
      signal.removeEventListener('abort', onAbort);
      await NativeXybrid.dispose(token).catch(() => {});
    },
  };
}
