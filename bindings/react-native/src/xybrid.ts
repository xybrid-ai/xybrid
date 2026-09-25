import { AppState } from 'react-native';

import NativeXybrid from './NativeXybrid';
import { ModelLoader } from './model';
import type { CacheEntry, CacheStatus, ThermalState } from './types';

/** Options for {@link Xybrid.initialize}. All optional. */
export interface XybridInitOptions {
  /**
   * Xybrid API key. Enables the platform features on top of local inference:
   * telemetry to the dashboard, cloud fallback, speculative cloud serving.
   * Get one at https://dashboard.xybrid.dev.
   */
  apiKey?: string;
  /** Override the LLM gateway URL (full URL including `/v1`). */
  gatewayUrl?: string;
  /** Override the telemetry ingest URL (self-hosted dashboards). */
  ingestUrl?: string;
  /**
   * Where downloaded models live. Defaults to `<Caches>/xybrid/models` on
   * iOS and `<filesDir>/xybrid/models` on Android. Only honoured when
   * `initialize` is the first Xybrid call of the process.
   */
  cacheDir?: string;
}

let initialized = false;

/**
 * Process-wide SDK entry points. Local inference needs no setup at all —
 * `initialize` only matters for the optional platform features.
 */
export const Xybrid = {
  /**
   * Apply process-wide configuration. The SDK otherwise configures itself
   * anonymously on first use, so call this before anything else when you
   * have options. Configuration is applied once per app process: repeating
   * it with the same options resolves, with different options rejects
   * (`xybrid_config_error`) — restart the app to change them.
   */
  async initialize(options: XybridInitOptions = {}): Promise<void> {
    await NativeXybrid.initialize(options);
    initialized = true;
  },

  /** Whether {@link initialize} has resolved in this JS context. */
  get isInitialized(): boolean {
    return initialized;
  },

  /** Describe a registry model without loading it (same as `ModelLoader.fromRegistry`). */
  model(id: string): ModelLoader {
    return ModelLoader.fromRegistry(id);
  },

  /** The native SDK version, e.g. `0.9.0`. */
  version(): Promise<string> {
    return NativeXybrid.sdkVersion();
  },

  /** Whether an API key resolves, from {@link initialize} or the environment. */
  hasApiKey(): Promise<boolean> {
    return NativeXybrid.hasApiKey();
  },

  /** Set the key for a cloud provider the gateway forwards to (not the Xybrid key). */
  setProviderApiKey(provider: string, apiKey: string): Promise<void> {
    return NativeXybrid.setProviderApiKey(provider, apiKey);
  },

  /** Point the cloud gateway at a platform base URL; the `/v1` suffix is added for you. */
  setPlatformUrl(url: string): Promise<void> {
    return NativeXybrid.setPlatformUrl(url);
  },

  /**
   * Default speculative cloud serving on or off for loads that don't opt in
   * per load (`ModelLoader.fromRegistrySpeculative` always speculates). Needs
   * an API key. LLM/chat only — prefer the per-load form when the app also
   * loads ASR/TTS models.
   */
  setSpeculativeCloud(enabled: boolean): Promise<void> {
    return NativeXybrid.setSpeculativeCloud(enabled);
  },

  isSpeculativeCloudEnabled(): Promise<boolean> {
    return NativeXybrid.isSpeculativeCloudEnabled();
  },

  /**
   * Release every idle model's memory; resolves how many were released. Runs
   * in flight are skipped, and a released model reloads itself on next use.
   */
  releaseMemory(): Promise<number> {
    return NativeXybrid.releaseMemory();
  },

  /**
   * Call {@link releaseMemory} on every OS memory warning (React Native
   * `AppState` `memoryWarning`, iOS and Android). Returns the unsubscribe
   * function.
   */
  releaseMemoryOnWarning(): () => void {
    const subscription = AppState.addEventListener('memoryWarning', () => {
      NativeXybrid.releaseMemory().catch(() => {});
    });
    return () => subscription.remove();
  },

  /** Release least-recently-used idle models before loads under memory pressure. */
  setAutoRelease(enabled: boolean): Promise<void> {
    return NativeXybrid.setAutoRelease(enabled);
  },

  isAutoReleaseEnabled(): Promise<boolean> {
    return NativeXybrid.isAutoReleaseEnabled();
  },

  // -- Device state. Both platforms already observe battery (and Android
  //    thermal) natively; these exist for tests and custom readings. --

  /** Push a battery percentage (0–100) to the routing engine. */
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

  // -- Model cache. Every call walks or deletes the cache on disk, off the
  //    JS thread. --

  /** Aggregate storage usage across every managed cache location. */
  async modelCacheStatus(): Promise<CacheStatus> {
    return (await NativeXybrid.cacheStatus()) as CacheStatus;
  },

  /** Physical entries; a model can appear more than once. */
  async modelCacheEntries(): Promise<CacheEntry[]> {
    return (await NativeXybrid.cacheEntries()) as CacheEntry[];
  },

  /** Whether `modelId` occupies any cache entry. */
  hasCachedModelData(modelId: string): Promise<boolean> {
    return NativeXybrid.cacheIsModelCached(modelId);
  },

  /** A local path for `modelId`, or `null`. Presence does not mean extracted. */
  cachedModelPath(modelId: string): Promise<string | null> {
    return NativeXybrid.cacheModelPath(modelId);
  },

  /** Models extracted, validated and ready to run offline. */
  extractedModelIds(): Promise<string[]> {
    return NativeXybrid.cacheExtractedModelIds();
  },

  /** Remove every cache entry for `modelId`. Don't race a load of the same model. */
  removeCachedModel(modelId: string): Promise<number> {
    return NativeXybrid.cacheRemoveModel(modelId);
  },

  /** Clear all managed cache storage. Don't race any model load. */
  clearModelCache(): Promise<number> {
    return NativeXybrid.cacheClear();
  },
};
