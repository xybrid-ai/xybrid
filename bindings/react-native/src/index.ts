import NativeXybrid from './NativeXybrid';
import type {
  DownloadStatus,
  Envelope,
  GenerationConfig,
  InferenceResult,
  ModelHandle,
  RunOptions,
  StreamEvent,
  StreamToken,
  ThermalState,
  VoiceInfo,
} from './types';

export type {
  AbortSignalKind,
  AudioEnvelope,
  DownloadState,
  DownloadStatus,
  EmbeddingEnvelope,
  Envelope,
  ExecutionTarget,
  GenerationConfig,
  InferenceResult,
  ModelHandle,
  RunOptions,
  StreamEvent,
  StreamToken,
  TextEnvelope,
  ThermalState,
  VoiceInfo,
} from './types';

export { GenerationConfigs, creative, greedy } from './presets';

/**
 * Convert a JSON Schema (as an object or JSON string) into a GBNF grammar for
 * {@link GenerationConfig.grammar}, so a local LLM emits schema-valid JSON.
 *
 * ```ts
 * const grammar = await jsonSchemaToGbnf({
 *   type: 'object',
 *   properties: { name: { type: 'string' } },
 *   required: ['name'],
 * });
 * const result = await model.run(envelope, { generationConfig: { grammar } });
 * ```
 *
 * Rejects with a config error on invalid JSON or an unsupported schema
 * construct. Conversion runs natively — the same converter every other
 * binding uses.
 */
export function jsonSchemaToGbnf(schema: object | string): Promise<string> {
  const json = typeof schema === 'string' ? schema : JSON.stringify(schema);
  return NativeXybrid.jsonSchemaToGbnf(json);
}

// Keys that only appear on `RunOptions`, never on a bare `GenerationConfig`.
// Used to tell the two apart when a caller passes either form to `run()`.
const RUN_OPTION_KEYS = [
  'generationConfig',
  'abortOn',
  'fallbackToCloud',
  'maxGraceTokens',
  'correlationId',
] as const;

// Accept either the canonical `RunOptions` or a bare `GenerationConfig`
// (the pre-RunOptions shorthand) and produce the wire object the native
// shims decode — or `null` when there's nothing to send.
function normalizeRunOptions(
  options: RunOptions | GenerationConfig | undefined,
): RunOptions | null {
  // Guard the `in` checks below: a JS caller can pass a non-object despite the
  // TS types, and `in` on a primitive throws a TypeError.
  if (!options || typeof options !== 'object') return null;
  const isRunOptions = RUN_OPTION_KEYS.some((k) => k in options);
  return isRunOptions
    ? (options as RunOptions)
    : { generationConfig: options as GenerationConfig };
}

// Cache the in-flight init promise so concurrent callers all await the same
// underlying native call. The native side is documented as idempotent, but
// without this the bare boolean gate lets every caller that arrives before
// the first await resolves re-enter the bridge — wasting work and risking
// observable ordering surprises (e.g. multiple `setBinding` writes against
// the OnceLock, or a load() racing the cache-dir setup).
let initPromise: Promise<void> | null = null;
let initialized = false;

export const Xybrid = {
  /**
   * Initialize the SDK. Must be called once before any model loading.
   *
   * On Android the native module passes the app's files dir as the SDK cache
   * root; on iOS the cache dir is resolved by the platform layer. Safe to
   * call concurrently — every caller receives the same underlying promise.
   */
  initialize(): Promise<void> {
    if (initPromise) return initPromise;
    const p = NativeXybrid.initialize(null).then(
      () => {
        initialized = true;
      },
      (err: unknown) => {
        // Reset on failure so the next caller can retry. Without this, a
        // transient init failure (e.g. cache dir creation) would poison the
        // module for the rest of the JS context's lifetime.
        initPromise = null;
        throw err;
      },
    );
    initPromise = p;
    return p;
  },

  /** True after `initialize()` has resolved at least once in this JS context. */
  get isInitialized(): boolean {
    return initialized;
  },

  /** Push a battery percentage (0..=100) to the routing engine. */
  setBatteryLevel(percent: number): Promise<void> {
    return NativeXybrid.setBatteryLevel(percent);
  },

  clearBatteryLevel(): Promise<void> {
    return NativeXybrid.clearBatteryLevel();
  },

  setThermalState(state: ThermalState): Promise<void> {
    return NativeXybrid.setThermalState(state);
  },

  clearThermalState(): Promise<void> {
    return NativeXybrid.clearThermalState();
  },

  /**
   * Point the cloud gateway at a platform base URL (staging, self-hosted).
   * Pass a bare base URL — the `/v1` suffix is applied internally.
   */
  setPlatformUrl(url: string): Promise<void> {
    return NativeXybrid.setPlatformUrl(url);
  },

  /**
   * Enable speculative cloud fallback globally: a registry model that isn't
   * downloaded yet is served from the gateway while the weights download.
   *
   * Only takes effect when an API key resolves. Speculation is LLM/chat only —
   * prefer {@link ModelLoader.fromRegistrySpeculative} when the app also loads
   * ASR/TTS models, which cannot be served this way.
   */
  setSpeculativeCloud(enabled: boolean): Promise<void> {
    return NativeXybrid.setSpeculativeCloud(enabled);
  },

  isSpeculativeCloudEnabled(): Promise<boolean> {
    return NativeXybrid.isSpeculativeCloudEnabled();
  },
};

export class ModelLoader {
  private constructor(private readonly factory: () => Promise<string>) {}

  static fromRegistry(modelId: string): ModelLoader {
    return new ModelLoader(() => NativeXybrid.loadFromRegistry(modelId));
  }

  /**
   * Serve from the cloud gateway while the registry weights download in the
   * background, instead of blocking on the download.
   *
   * `load()` then resolves almost immediately with a cloud-backed model that
   * switches to on-device by itself once the download lands. Requires a
   * resolvable API key and an uncached model — otherwise this behaves exactly
   * like {@link fromRegistry}. Poll {@link Model.downloadStatus} for progress
   * and {@link Model.isCloudServing} to know which leg is answering.
   * LLM/chat models only.
   */
  static fromRegistrySpeculative(modelId: string): ModelLoader {
    return new ModelLoader(() =>
      NativeXybrid.loadFromRegistrySpeculative(modelId),
    );
  }

  static fromBundle(path: string): ModelLoader {
    return new ModelLoader(() => NativeXybrid.loadFromBundle(path));
  }

  static fromDirectory(path: string): ModelLoader {
    return new ModelLoader(() => NativeXybrid.loadFromDirectory(path));
  }

  static fromHuggingface(repo: string): ModelLoader {
    return new ModelLoader(() => NativeXybrid.loadFromHuggingface(repo));
  }

  async load(): Promise<Model> {
    // initialize() now returns the cached promise on subsequent calls, so
    // unconditionally awaiting it is free after the first resolve and avoids
    // a second TOCTOU window between the check and the call.
    await Xybrid.initialize();
    const handle = await this.factory();
    return new Model(handle);
  }
}

export class Model {
  constructor(private readonly handle: ModelHandle) {}

  get id(): ModelHandle {
    return this.handle;
  }

  /**
   * Run inference. The second argument is a {@link RunOptions} carrying the
   * sampling config plus the platform-plane knobs (cloud fallback,
   * abort-on-stress, telemetry correlation), mirroring the Apple/Kotlin SDKs.
   *
   * A bare {@link GenerationConfig} is also accepted as shorthand for
   * `{ generationConfig }`.
   */
  async run(
    envelope: Envelope,
    options?: RunOptions | GenerationConfig,
  ): Promise<InferenceResult> {
    const result = (await NativeXybrid.run(
      this.handle,
      envelope,
      normalizeRunOptions(options),
    )) as InferenceResult;
    return result;
  }

  /**
   * Stream inference token-by-token. Yields each {@link StreamToken} as it is
   * generated and returns the final {@link InferenceResult} (latency, metrics)
   * as the generator's return value. Errors mid-stream are thrown.
   *
   * Consume with `for await`:
   * ```ts
   * for await (const t of model.runStreaming(envelope)) {
   *   setText((prev) => prev + t.token);
   * }
   * ```
   *
   * The underlying native run is aborted automatically when iteration ends —
   * it completes, you `break`, or an error is thrown; each of these runs this
   * generator's cleanup (`break`/`throw` call `return()` under the hood). A
   * generator that is merely *abandoned* mid-stream — the consumer stops
   * calling `next()` without breaking — is never cleaned up (JS runs no
   * `finally` on GC), leaving the native run alive until its model is
   * released. So on unmount, break out of the loop or call `gen.return()`.
   *
   * The second argument mirrors {@link run}: a {@link RunOptions} (or a bare
   * {@link GenerationConfig} shorthand). Non-LLM models emit a single token
   * carrying the full result, then complete.
   *
   * The final {@link InferenceResult} (latency, metrics) is the generator's
   * *return* value, not a yielded token — capture it via manual iteration:
   * ```ts
   * const gen = model.runStreaming(envelope);
   * let next = await gen.next();
   * while (!next.done) { append(next.value.token); next = await gen.next(); }
   * const result = next.value; // InferenceResult | undefined
   * ```
   *
   * Errors raised mid-stream are thrown from the iterator with the same typed
   * `xybrid_*` codes that {@link run} rejects with (the native `streamNext`
   * promise rejects, and the rejection propagates out of the generator).
   */
  async *runStreaming(
    envelope: Envelope,
    options?: RunOptions | GenerationConfig,
  ): AsyncGenerator<StreamToken, InferenceResult | undefined, void> {
    const streamHandle = await NativeXybrid.streamStart(
      this.handle,
      envelope,
      normalizeRunOptions(options),
    );
    try {
      for (;;) {
        const event = (await NativeXybrid.streamNext(streamHandle)) as StreamEvent | null;
        // `null` means the native stream is exhausted without an explicit
        // terminal event (e.g. it was released out from under us) — stop.
        if (event == null) return undefined;
        switch (event.kind) {
          case 'token':
            yield event.token;
            break;
          case 'complete':
            return event.result;
          default:
            // Exhaustiveness guard: a new native event kind that this switch
            // doesn't handle is a contract/version-skew bug, not a silent stop.
            throw new Error(
              `Unexpected stream event: ${(event as { kind: string }).kind}`,
            );
        }
      }
    } finally {
      // Always release on generator cleanup: covers normal completion, early
      // `break`, thrown errors, and explicit `gen.return()` — but NOT silent
      // abandonment (JS never runs `finally` on GC; see the doc comment).
      // Releasing aborts the native run and is idempotent (releasing an
      // already-finished stream is a no-op).
      await NativeXybrid.streamRelease(streamHandle);
    }
  }

  /**
   * Warm up the model with a priming inference, so first-token latency on the
   * next `run` is attributable to inference rather than cold start.
   */
  warmup(): Promise<void> {
    return NativeXybrid.warmup(this.handle);
  }

  /**
   * Unload the model's weights to free native memory while keeping this handle
   * valid — a later `run` transparently reloads. Use this to shed memory under
   * pressure without discarding the handle (contrast with {@link release}).
   */
  unload(): Promise<void> {
    return NativeXybrid.unload(this.handle);
  }

  /**
   * Whether runs are currently answered by the cloud because the local weights
   * are not ready yet. `false` for ordinary local models.
   *
   * This predicts the *next* run; {@link InferenceResult.executionTarget}
   * reports what a run that already happened actually did. They differ when a
   * cloud leg fails and degrades to local mid-call.
   */
  isCloudServing(): Promise<boolean> {
    return NativeXybrid.isCloudServing(this.handle);
  }

  /**
   * Download progress and state in one consistent read. Reports `ready` at 1.0
   * for an ordinary local model, so the UI needs no special case.
   */
  async downloadStatus(): Promise<DownloadStatus> {
    return (await NativeXybrid.downloadStatus(this.handle)) as DownloadStatus;
  }

  /**
   * Block until the download settles or `timeoutMs` elapses, then report it.
   *
   * The blocking wait happens natively, so a host can `await` this once
   * instead of driving a JS timer. Pass `0` for a non-blocking read.
   */
  async awaitDownload(timeoutMs: number): Promise<DownloadStatus> {
    return (await NativeXybrid.awaitDownload(
      this.handle,
      timeoutMs,
    )) as DownloadStatus;
  }

  async voices(): Promise<VoiceInfo[] | null> {
    const list = await NativeXybrid.voices(this.handle);
    return list as VoiceInfo[] | null;
  }

  defaultVoiceId(): Promise<string | null> {
    return NativeXybrid.defaultVoiceId(this.handle);
  }

  hasVoices(): Promise<boolean> {
    return NativeXybrid.hasVoices(this.handle);
  }

  /**
   * Release the underlying native model handle. Subsequent calls on this
   * instance will reject. Call this when a model is no longer needed —
   * loaded models hold significant memory (weights live in native heap).
   */
  release(): Promise<void> {
    return NativeXybrid.releaseModel(this.handle);
  }
}
