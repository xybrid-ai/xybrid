import NativeXybrid from './NativeXybrid';
import { bindCancel, type AbortSignalLike } from './cancel';
import type { ConversationContext } from './context';
import { StreamingSession } from './session';
import type {
  AbortSignalKind,
  DownloadStatus,
  Envelope,
  GenerationConfig,
  InferenceResult,
  ModelInfo,
  StreamToken,
  StreamingConfig,
  VoiceInfo,
} from './types';
import {
  fromWireDownloadStatus,
  fromWireGenerationConfig,
  fromWireResult,
  fromWireStreamToken,
  toWireEnvelope,
  toWireRunOptions,
  type WireGenerationConfig,
  type WireStreamEvent,
} from './wire';

/**
 * Per-call options, mirroring bolt's `XybridRunOptions` plus the two handles
 * the other SDKs pass as separate arguments (a conversation context and a
 * stop button).
 */
export interface RunOptions {
  /** Sampling parameters for LLM inference. */
  generationConfig?: GenerationConfig;
  /** Device-stress signals that abort the run early. */
  abortOn?: AbortSignalKind[];
  /** Allow this call to fall back to the cloud gateway under device stress. */
  fallbackToCloud?: boolean;
  /** Tokens to emit after an abort signal before stopping. */
  maxGraceTokens?: number;
  /** Correlation id threaded into telemetry for this call. */
  correlationId?: string;
  /** Cloud provider selected by the caller for an enabled fallback. */
  cloudProvider?: string;
  /** Cloud model selected by the caller for an enabled fallback. */
  cloudModel?: string;
  /** Versioned cloud gateway base URL, validated by the shared facade. */
  cloudGatewayUrl?: string;
  /**
   * Seed the run with this conversation's history (multi-turn chat). The
   * context is not modified: push the turns you want to keep yourself.
   */
  context?: ConversationContext;
  /**
   * Stop button. Aborting stops generation at the next token boundary. A
   * batch `run` is only cancellable before generation starts; use
   * `runStreaming` when a mid-flight stop matters.
   */
  signal?: AbortSignalLike;
}

/** Where a model comes from. Build one with the {@link ModelLoader} factories. */
export type ModelSource =
  | { kind: 'registry'; id: string }
  | { kind: 'registrySpeculative'; id: string }
  | { kind: 'bundle'; path: string }
  | { kind: 'directory'; path: string }
  | { kind: 'huggingFace'; repo: string; revision?: string }
  | { kind: 'modelFile'; path: string };

function toWireSource(source: ModelSource): { kind: string; value: string; revision?: string } {
  switch (source.kind) {
    case 'registry':
    case 'registrySpeculative':
      return { kind: source.kind, value: source.id };
    case 'bundle':
    case 'directory':
    case 'modelFile':
      return { kind: source.kind, value: source.path };
    case 'huggingFace':
      return {
        kind: source.kind,
        value: source.repo,
        ...(source.revision ? { revision: source.revision } : {}),
      };
  }
}

/**
 * A cheap description of a model. Creating one does no I/O; {@link load}
 * resolves, downloads if needed, and loads it.
 */
export class ModelLoader {
  private constructor(readonly source: ModelSource) {}

  /** Describe any source. */
  static from(source: ModelSource): ModelLoader {
    return new ModelLoader(source);
  }

  /** A model from the Xybrid registry. The recommended path. */
  static fromRegistry(id: string): ModelLoader {
    return new ModelLoader({ kind: 'registry', id });
  }

  /**
   * Serve from the cloud gateway while the registry weights download in the
   * background, instead of blocking on the download. `load()` then resolves
   * almost immediately with a model that switches to on-device by itself.
   * Needs an API key and an uncached model — {@link willSpeculate} tells you
   * up front — otherwise it behaves like {@link fromRegistry}. LLM/chat only.
   */
  static fromRegistrySpeculative(id: string): ModelLoader {
    return new ModelLoader({ kind: 'registrySpeculative', id });
  }

  /** A local `.xyb` bundle. */
  static fromBundle(path: string): ModelLoader {
    return new ModelLoader({ kind: 'bundle', path });
  }

  /** A local directory containing `model_metadata.json`. */
  static fromDirectory(path: string): ModelLoader {
    return new ModelLoader({ kind: 'directory', path });
  }

  /** A Hugging Face repository (`org/repo` or `org/repo:variant`), optionally pinned. */
  static fromHuggingFace(repo: string, options: { revision?: string } = {}): ModelLoader {
    return new ModelLoader({ kind: 'huggingFace', repo, ...options });
  }

  /** A raw GGUF file; `model_metadata.json` is generated next to it if absent. */
  static fromModelFile(path: string): ModelLoader {
    return new ModelLoader({ kind: 'modelFile', path });
  }

  /** Resolve, download if needed, and load. Rejects with a typed `xybrid_*` code. */
  async load(): Promise<Model> {
    return new Model(await NativeXybrid.loadModel(toWireSource(this.source)));
  }

  /**
   * Whether {@link load} would actually speculate: a speculative source, an
   * API key, and a model not yet cached. Never touches the network.
   */
  async willSpeculate(): Promise<boolean> {
    if (this.source.kind !== 'registrySpeculative') return false;
    return NativeXybrid.willSpeculate(this.source.id);
  }

  /**
   * Start downloading the weights in the background without loading them —
   * the object to drive a progress bar from. The download fills the cache, so
   * the {@link load} afterwards returns at once. Resolves `null` for sources
   * with nothing to fetch (bundle, directory, Hugging Face, model file).
   */
  async download(options: { platform?: string } = {}): Promise<ModelDownload | null> {
    const { source } = this;
    if (source.kind !== 'registry' && source.kind !== 'registrySpeculative') return null;
    return new ModelDownload(await NativeXybrid.startDownload(source.id, options.platform ?? null));
  }
}

/** A model loaded in native memory. Call {@link release} when done with it. */
export class Model {
  /** @internal Use {@link ModelLoader.load}. */
  constructor(
    /** Opaque native handle — not the model id; see {@link info}. */
    readonly handle: string,
  ) {}

  /** Identity and capabilities, in one native call. */
  async info(): Promise<ModelInfo> {
    const raw = (await NativeXybrid.modelInfo(this.handle)) as Omit<
      ModelInfo,
      'defaultGenerationConfig'
    > & { defaultGenerationConfig: WireGenerationConfig };
    return {
      ...raw,
      defaultGenerationConfig: fromWireGenerationConfig(raw.defaultGenerationConfig),
    };
  }

  /** Whether the weights are resident (false after {@link unload} until the next run). */
  isLoaded(): Promise<boolean> {
    return NativeXybrid.isLoaded(this.handle);
  }

  /**
   * Run one inference. Pass sampling parameters under `generationConfig`,
   * a conversation under `context`, and an `AbortSignal` under `signal`.
   */
  async run(envelope: Envelope, options: RunOptions = {}): Promise<InferenceResult> {
    const wireEnvelope = toWireEnvelope(envelope);
    const cancel = await bindCancel(options.signal);
    try {
      const wire = await NativeXybrid.run(
        this.handle,
        wireEnvelope,
        toWireRunOptions(options, { context: options.context?.handle, cancel: cancel.token }),
      );
      return fromWireResult(wire as Parameters<typeof fromWireResult>[0]);
    } finally {
      await cancel.release();
    }
  }

  /**
   * Stream inference token by token. Yields each {@link StreamToken}; the
   * final {@link InferenceResult} is the generator's return value.
   *
   * ```ts
   * for await (const token of model.runStreaming(Envelope.text('Hi'))) {
   *   setText((prev) => prev + token.token);
   * }
   * ```
   *
   * Generation stops natively when iteration ends — completion, `break`, a
   * thrown error, or an aborted `signal`. A generator that is merely
   * abandoned is never cleaned up (JS runs no `finally` on GC), so on
   * unmount abort the signal or call `gen.return()`.
   *
   * When the turn ends on tool calls, the terminal token carries `toolCalls`
   * and `rawText` (pass it to `Envelope.toolResults`).
   */
  async *runStreaming(
    envelope: Envelope,
    options: RunOptions = {},
  ): AsyncGenerator<StreamToken, InferenceResult | undefined, void> {
    const wireEnvelope = toWireEnvelope(envelope);
    const cancel = await bindCancel(options.signal);
    let stream: string | null = null;
    try {
      stream = await NativeXybrid.streamStart(
        this.handle,
        wireEnvelope,
        toWireRunOptions(options, { context: options.context?.handle, cancel: cancel.token }),
      );
      for (;;) {
        const event = (await NativeXybrid.streamNext(stream)) as WireStreamEvent | null;
        // `null`: the stream was disposed underneath us (model released).
        if (event == null) return undefined;
        switch (event.kind) {
          case 'token':
            yield fromWireStreamToken(event.token);
            break;
          case 'complete':
            return fromWireResult(event.result);
          default:
            throw new Error(
              `Unexpected stream event: ${String((event as { kind?: unknown }).kind)}`,
            );
        }
      }
    } finally {
      // Disposing aborts native generation; idempotent after completion.
      if (stream !== null) await NativeXybrid.dispose(stream).catch(() => {});
      await cancel.release();
    }
  }

  /**
   * Open a live ASR session: feed microphone PCM in, read partial transcripts
   * out. Audio must be Float32, mono, 16 kHz. Rejects with
   * `xybrid_streaming_unsupported` for a model that cannot stream.
   */
  async stream(config: StreamingConfig = {}): Promise<StreamingSession> {
    return new StreamingSession(await NativeXybrid.openStreamingSession(this.handle, config));
  }

  /** Prime the model so first-token latency measures inference, not cold start. */
  warmup(): Promise<void> {
    return NativeXybrid.warmup(this.handle);
  }

  /**
   * Free the weights while keeping this handle valid — the next run reloads
   * them. Contrast with {@link release}, which frees the handle.
   */
  unload(): Promise<void> {
    return NativeXybrid.unload(this.handle);
  }

  /**
   * Whether runs are currently answered by the cloud because the local
   * weights aren't ready (speculative loads). Predicts the next run;
   * `InferenceResult.executionTarget` reports what a finished run did.
   */
  isCloudServing(): Promise<boolean> {
    return NativeXybrid.isCloudServing(this.handle);
  }

  /** Download progress of a speculative load; `ready` at 1.0 for local models. */
  async downloadStatus(): Promise<DownloadStatus> {
    return fromWireDownloadStatus(
      (await NativeXybrid.downloadStatus(this.handle)) as DownloadStatus,
    );
  }

  /**
   * Yield the speculative download's status whenever it changes, ending
   * after a terminal state. A local model yields one `ready` frame.
   */
  downloadProgress(options: { intervalMs?: number } = {}): AsyncGenerator<DownloadStatus, void, void> {
    return pollDownload(() => this.downloadStatus(), options.intervalMs);
  }

  /**
   * Wait natively until the download settles or `timeoutMs` elapses, then
   * report it. `0` makes it a non-blocking read.
   */
  async awaitDownload(timeoutMs: number): Promise<DownloadStatus> {
    return fromWireDownloadStatus(
      (await NativeXybrid.awaitDownload(this.handle, timeoutMs)) as DownloadStatus,
    );
  }

  /** TTS voices; empty for models without voices. */
  async voices(): Promise<VoiceInfo[]> {
    return (await NativeXybrid.voices(this.handle)) as VoiceInfo[];
  }

  /** The default TTS voice, if the model has voices. */
  async defaultVoice(): Promise<VoiceInfo | null> {
    return (await NativeXybrid.defaultVoice(this.handle)) as VoiceInfo | null;
  }

  /** One TTS voice by id. */
  async voice(voiceId: string): Promise<VoiceInfo | null> {
    return (await NativeXybrid.voice(this.handle, voiceId)) as VoiceInfo | null;
  }

  /**
   * Free the native model — weights live in the native heap, so release
   * models you no longer need. Also stops token streams started from it.
   * Later calls on this instance reject with `xybrid_handle`.
   */
  release(): Promise<void> {
    return NativeXybrid.dispose(this.handle);
  }
}

const TERMINAL_STATES = new Set<DownloadStatus['state']>(['ready', 'failed', 'cancelled']);

/**
 * Poll `read` and yield each changed status, ending after a terminal state.
 * The native side throttles its own updates (~10/s), so a 200 ms default
 * loses nothing a progress bar can show.
 */
async function* pollDownload(
  read: () => Promise<DownloadStatus>,
  intervalMs = 200,
): AsyncGenerator<DownloadStatus, void, void> {
  const delay = Math.max(16, intervalMs);
  let previous: DownloadStatus | null = null;
  for (;;) {
    const status = await read();
    if (
      previous === null ||
      status.state !== previous.state ||
      status.downloadedBytes !== previous.downloadedBytes ||
      status.progress !== previous.progress
    ) {
      yield status;
      previous = status;
    }
    if (TERMINAL_STATES.has(status.state)) return;
    await new Promise((resolve) => setTimeout(resolve, delay));
  }
}

/**
 * A model download running in the background, separate from loading. Start
 * one with {@link ModelLoader.download}. Releasing the handle does not stop
 * the transfer — call {@link cancel}.
 */
export class ModelDownload {
  /** @internal */
  constructor(readonly handle: string) {}

  /** Progress, bytes and state in one consistent read. */
  async status(): Promise<DownloadStatus> {
    return fromWireDownloadStatus(
      (await NativeXybrid.downloadHandleStatus(this.handle)) as DownloadStatus,
    );
  }

  /** The failure message once the download failed, else `null`. */
  error(): Promise<string | null> {
    return NativeXybrid.downloadHandleError(this.handle);
  }

  /** Stop the transfer. The status then settles on `cancelled`. */
  cancel(): Promise<void> {
    return NativeXybrid.cancelDownload(this.handle);
  }

  /**
   * Yield the status whenever it changes, ending after a terminal state
   * (`ready`, `failed`, `cancelled`). Polls natively every `intervalMs`.
   *
   * ```ts
   * for await (const s of download.progress()) setProgress(s.progress);
   * ```
   */
  progress(options: { intervalMs?: number } = {}): AsyncGenerator<DownloadStatus, void, void> {
    return pollDownload(() => this.status(), options.intervalMs);
  }

  /** Free the handle (the transfer keeps going unless cancelled). */
  release(): Promise<void> {
    return NativeXybrid.dispose(this.handle);
  }
}
