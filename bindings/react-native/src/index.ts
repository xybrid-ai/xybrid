import NativeXybrid from './NativeXybrid';
import type {
  Envelope,
  GenerationConfig,
  InferenceResult,
  ModelHandle,
  ThermalState,
  VoiceInfo,
} from './types';

export type {
  AudioEnvelope,
  EmbeddingEnvelope,
  Envelope,
  GenerationConfig,
  InferenceResult,
  ModelHandle,
  TextEnvelope,
  ThermalState,
  VoiceInfo,
} from './types';

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
};

export class ModelLoader {
  private constructor(private readonly factory: () => Promise<string>) {}

  static fromRegistry(modelId: string): ModelLoader {
    return new ModelLoader(() => NativeXybrid.loadFromRegistry(modelId));
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

  async run(envelope: Envelope, config?: GenerationConfig): Promise<InferenceResult> {
    const result = (await NativeXybrid.run(
      this.handle,
      envelope,
      config ?? null,
    )) as InferenceResult;
    return result;
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
