import NativeXybrid from './NativeXybrid';
import { float32ToBase64 } from './base64';
import type { PartialResult } from './types';

/**
 * A live ASR session: feed microphone PCM in, read partial transcripts out.
 * Open one with `model.stream()`.
 *
 * ```ts
 * const session = await model.stream();
 * (async () => {
 *   for await (const partial of session.partials()) setCaption(partial.text);
 * })();
 * // from your audio callback (Float32, mono, 16 kHz):
 * await session.feed(samples);
 * // when the user stops talking:
 * const transcript = await session.flush();
 * await session.release();
 * ```
 */
export class StreamingSession {
  /** @internal Use `model.stream()`. */
  constructor(
    /** Opaque native handle. */
    readonly handle: string,
  ) {}

  /**
   * Queue PCM samples (Float32, mono, 16 kHz). Transcription happens off the
   * JS thread; this only waits when the native queue is full.
   */
  feed(samples: Float32Array | ArrayLike<number>): Promise<void> {
    return NativeXybrid.sessionFeed(this.handle, float32ToBase64(samples));
  }

  /**
   * Partial transcripts as they arrive, ending when the session finishes
   * (after {@link flush} or {@link cancel}). Each partial is cumulative:
   * render it in place of the previous one. Use one iterator per session.
   */
  async *partials(): AsyncGenerator<PartialResult, void, void> {
    for (;;) {
      const partial = (await NativeXybrid.sessionNextPartial(this.handle)) as PartialResult | null;
      if (partial == null) return;
      yield partial;
    }
  }

  /**
   * Drain the buffered audio and return the complete transcript. The session
   * is over afterwards.
   */
  flush(): Promise<string> {
    return NativeXybrid.sessionFlush(this.handle);
  }

  /** Start over on fresh audio without reloading the model. */
  reset(): Promise<void> {
    return NativeXybrid.sessionReset(this.handle);
  }

  /** Stop and discard buffered audio — the "user walked away" path. */
  cancel(): Promise<void> {
    return NativeXybrid.sessionCancel(this.handle);
  }

  /** Whether the session still accepts audio. */
  isRunning(): Promise<boolean> {
    return NativeXybrid.sessionIsRunning(this.handle);
  }

  /** Free the native session (cancelling it if still running). */
  release(): Promise<void> {
    return NativeXybrid.dispose(this.handle);
  }
}
