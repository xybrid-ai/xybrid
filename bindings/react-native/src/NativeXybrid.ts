// TurboModule spec consumed by React Native codegen.
//
// Codegen constraints (see https://reactnative.dev/docs/the-new-architecture/pure-cxx-modules):
// only `string`, `number`, `boolean`, `void`, `Promise<T>`, plain object types,
// and arrays of those are allowed. Discriminated unions cross as plain `Object`
// and are reconstructed in the TS facade (src/index.ts) — keeping the spec flat
// avoids per-platform shim differences.
//
// Model handles are opaque string IDs. The native modules keep a map of
// `id -> XybridModel` (Swift) / `id -> XybridModel` (Kotlin) and clean up
// when `releaseModel` is called.

import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  // -- Lifecycle --
  initialize(cacheDir: string | null): Promise<void>;

  // -- Loaders (return opaque handle ID) --
  loadFromRegistry(modelId: string): Promise<string>;
  // Serve from the cloud gateway while the registry weights download in the
  // background, instead of blocking on the download. Resolves almost
  // immediately; the model switches to on-device by itself once the download
  // lands. Requires a resolvable API key and an uncached model, otherwise it
  // behaves exactly like `loadFromRegistry`. LLM/chat models only.
  loadFromRegistrySpeculative(modelId: string): Promise<string>;
  loadFromBundle(path: string): Promise<string>;
  loadFromDirectory(path: string): Promise<string>;
  loadFromHuggingface(repo: string): Promise<string>;
  releaseModel(handle: string): Promise<void>;

  // -- Model lifecycle --
  // Warm up the model (runs a priming inference so first-token latency is
  // attributable to warmup vs. inference) and unload it (frees native memory
  // while keeping the handle valid for a later reload). Mirror the
  // Apple/Kotlin/Flutter `warmup`/`unload` surface.
  warmup(handle: string): Promise<void>;
  unload(handle: string): Promise<void>;

  // -- Speculative cloud --
  // `isCloudServing` predicts the next run; `InferenceResult.executionTarget`
  // reports what a run that already happened actually did (they differ when a
  // cloud leg fails and degrades to local mid-call). `downloadStatus` returns
  // `{ state, progress }` — poll it for a progress bar. `awaitDownload` blocks
  // natively until the download settles or the timeout elapses, so JS can
  // await it once instead of running a timer.
  isCloudServing(handle: string): Promise<boolean>;
  downloadStatus(handle: string): Promise<Object>;
  awaitDownload(handle: string, timeoutMs: number): Promise<Object>;

  // -- Inference --
  // `envelope` and `options` cross as Objects; the TS facade narrows to the
  // discriminated `Envelope` union and normalizes the second arg to a
  // `RunOptions` shape (`{ generationConfig, abortOn, fallbackToCloud,
  // maxGraceTokens, correlationId }`). Native side validates `kind` and
  // rejects with an Error if it doesn't match a known variant.
  run(handle: string, envelope: Object, options: Object | null): Promise<Object>;

  // -- Streaming (pull-based) --
  // `streamStart` begins a run and returns an opaque stream-handle id. Pull
  // events with `streamNext` until it resolves to `null` (exhausted). Each
  // event Object is the discriminated `StreamEvent` union (narrowed in the TS
  // facade by its `kind` field); mid-stream failures reject the `streamNext`
  // promise with the same typed `xybrid_*` codes as `run`. Always
  // `streamRelease` when stopping early — it aborts the underlying run, which
  // otherwise keeps generating.
  streamStart(handle: string, envelope: Object, options: Object | null): Promise<string>;
  streamNext(streamHandle: string): Promise<Object | null>;
  streamRelease(streamHandle: string): Promise<void>;

  // -- TTS introspection --
  voices(handle: string): Promise<Object[] | null>;
  defaultVoiceId(handle: string): Promise<string | null>;
  hasVoices(handle: string): Promise<boolean>;

  // -- Platform-state push (forwarded to xybrid-sdk) --
  // The Swift wrapper auto-registers UIDevice battery observers and the
  // Kotlin wrapper auto-registers BatteryManager + thermal listeners on
  // `initialize()`, so apps shouldn't need to call these directly. Exposed
  // for tests and for hosts that want to forward their own readings.
  setBatteryLevel(percent: number): Promise<void>;
  clearBatteryLevel(): Promise<void>;
  setThermalState(state: string): Promise<void>;
  clearThermalState(): Promise<void>;

  // -- Utilities --
  // Convert a JSON Schema (as a JSON string) into a GBNF grammar for
  // `GenerationConfig.grammar`. Rejects on invalid JSON or an unsupported
  // schema construct.
  jsonSchemaToGbnf(schemaJson: string): Promise<string>;

  // -- Cloud gateway configuration --
  // `setPlatformUrl` takes a bare base URL (the `/v1` suffix is internal) and
  // is held in memory, not the environment. `setSpeculativeCloud` flips the
  // global default; prefer `loadFromRegistrySpeculative` per model when the app
  // also loads ASR/TTS models, which cannot be served from the cloud.
  setPlatformUrl(url: string): Promise<void>;
  setSpeculativeCloud(enabled: boolean): Promise<void>;
  isSpeculativeCloudEnabled(): Promise<boolean>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('RNXybrid');
