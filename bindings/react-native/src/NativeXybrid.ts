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
  loadFromBundle(path: string): Promise<string>;
  loadFromDirectory(path: string): Promise<string>;
  loadFromHuggingface(repo: string): Promise<string>;
  releaseModel(handle: string): Promise<void>;

  // -- Inference --
  // `envelope` and `config` cross as Objects; the TS facade narrows to the
  // discriminated `Envelope` union. Native side validates `kind` and rejects
  // with an Error if it doesn't match a known variant.
  run(handle: string, envelope: Object, config: Object | null): Promise<Object>;

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
}

export default TurboModuleRegistry.getEnforcing<Spec>('RNXybrid');
