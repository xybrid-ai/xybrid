// TurboModule spec consumed by React Native Codegen — the one contract both
// native shims are compiled against: Android extends the generated
// `NativeXybridSpec` class and iOS conforms to the generated
// `NativeXybridSpec` protocol (tests/spec-conformance.test.mjs checks the
// selectors without a Mac).
//
// Codegen accepts only `string`, `number`, `boolean`, `Object`, arrays and
// `Promise`s of those, and `| null`. Structured values therefore cross as
// plain `Object`s: the TS facade (src/index.ts) builds and narrows them, and
// the native codecs (ios/XybridCodec.swift, android/.../XybridCodec.kt)
// translate them to and from the bolt types.
//
// Native objects (models, conversation contexts, cancellation tokens, token
// streams, downloads, pipelines, live-ASR sessions) are opaque string handles
// held in one registry per platform. `dispose` frees any of them, and freeing
// a model also disposes the token streams started from it.
//
// Nothing here is public API. Keep method names distinct from NSObject /
// Java `Object` members (hence `dispose`, not `release`; `sdkVersion`, not
// `version`).

import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  // -- SDK configuration --
  // `options` is `{ apiKey?, gatewayUrl?, ingestUrl?, cacheDir? }`. The SDK
  // configures itself anonymously on first use; the first `initialize` that
  // carries options applies them, and later ones reject if they differ.
  initialize(options: Object | null): Promise<void>;
  sdkVersion(): Promise<string>;
  hasApiKey(): Promise<boolean>;
  setProviderApiKey(provider: string, apiKey: string): Promise<void>;
  setPlatformUrl(url: string): Promise<void>;
  setSpeculativeCloud(enabled: boolean): Promise<void>;
  isSpeculativeCloudEnabled(): Promise<boolean>;
  willSpeculate(modelId: string): Promise<boolean>;
  releaseMemory(): Promise<number>;
  setAutoRelease(enabled: boolean): Promise<void>;
  isAutoReleaseEnabled(): Promise<boolean>;

  // -- Device state push (both platforms also observe it natively) --
  setBatteryLevel(percent: number): Promise<void>;
  clearBatteryLevel(): Promise<void>;
  setThermalState(state: string): Promise<void>;
  clearThermalState(): Promise<void>;

  // -- Model cache --
  cacheStatus(): Promise<Object>;
  cacheEntries(): Promise<Object[]>;
  cacheIsModelCached(modelId: string): Promise<boolean>;
  cacheModelPath(modelId: string): Promise<string | null>;
  cacheExtractedModelIds(): Promise<string[]>;
  cacheRemoveModel(modelId: string): Promise<number>;
  cacheClear(): Promise<number>;

  // -- Stateless helpers --
  jsonSchemaToGbnf(schemaJson: string): Promise<string>;
  toolResultsEnvelope(
    userText: string,
    priorAssistantText: string,
    results: Object[],
  ): Promise<Object>;

  // -- Handles --
  dispose(handle: string): Promise<void>;

  // -- Models --
  // `source` is `{ kind, value, revision? }`, kind one of `registry`,
  // `registrySpeculative`, `bundle`, `directory`, `huggingFace`, `modelFile`.
  loadModel(source: Object): Promise<string>;
  modelInfo(model: string): Promise<Object>;
  isLoaded(model: string): Promise<boolean>;
  warmup(model: string): Promise<void>;
  unload(model: string): Promise<void>;
  isCloudServing(model: string): Promise<boolean>;
  downloadStatus(model: string): Promise<Object>;
  awaitDownload(model: string, timeoutMs: number): Promise<Object>;
  voices(model: string): Promise<Object[]>;
  defaultVoice(model: string): Promise<Object | null>;
  voice(model: string, voiceId: string): Promise<Object | null>;

  // -- Inference --
  // `options` is the wire form of `RunOptions`: sampling config, platform
  // knobs, optional cloud provider/model/gateway, plus `context` (a
  // conversation handle) and `cancel` (a cancellation-token handle)
  // resolved natively.
  run(model: string, envelope: Object, options: Object | null): Promise<Object>;
  // Pull-based token streaming. `streamNext` resolves each event
  // (`{ kind: 'token', token }` or `{ kind: 'complete', result }`) and `null`
  // once the stream is gone; failures reject with the same codes as `run`.
  // Disposing a stream aborts its generation.
  streamStart(model: string, envelope: Object, options: Object | null): Promise<string>;
  streamNext(stream: string): Promise<Object | null>;

  // -- Cancellation tokens (the stop button behind `RunOptions.signal`) --
  createCancelToken(): Promise<string>;
  cancel(token: string): Promise<void>;

  // -- Conversation contexts --
  createContext(contextId: string | null): Promise<string>;
  contextPush(context: string, envelope: Object): Promise<void>;
  contextSetSystem(context: string, envelope: Object): Promise<void>;
  contextClear(context: string): Promise<void>;
  contextInfo(context: string): Promise<Object>;
  contextHistory(context: string): Promise<Object[]>;
  contextSetMaxHistoryLength(context: string, length: number): Promise<void>;

  // -- Standalone downloads (fill the cache without loading) --
  startDownload(modelId: string, platform: string | null): Promise<string>;
  downloadHandleStatus(download: string): Promise<Object>;
  downloadHandleError(download: string): Promise<string | null>;
  cancelDownload(download: string): Promise<void>;

  // -- Pipelines --
  // `source` is `{ kind: 'yaml' | 'file' | 'bundle', value }`.
  loadPipeline(source: Object): Promise<string>;
  pipelineInfo(pipeline: string): Promise<Object>;
  runPipeline(pipeline: string, envelope: Object, options: Object | null): Promise<Object>;

  // -- Live ASR sessions --
  // `samplesBase64` is little-endian Float32 PCM, mono, 16 kHz.
  openStreamingSession(model: string, config: Object | null): Promise<string>;
  sessionFeed(session: string, samplesBase64: string): Promise<void>;
  sessionNextPartial(session: string): Promise<Object | null>;
  sessionFlush(session: string): Promise<string>;
  sessionReset(session: string): Promise<void>;
  sessionCancel(session: string): Promise<void>;
  sessionIsRunning(session: string): Promise<boolean>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('RNXybrid');
