// @xybrid/react-native — on-device (and optionally cloud) inference for React
// Native, backed by the same Rust SDK as the Swift, Kotlin, Flutter and Unity
// bindings.
//
//   await Xybrid.initialize({ apiKey });            // optional
//   const model = await ModelLoader.fromRegistry('qwen3.5-0.8b').load();
//   const result = await model.run(Envelope.text('Hello'));
//   await model.release();

import NativeXybrid from './NativeXybrid';

export { Xybrid, type XybridInitOptions } from './xybrid';
export {
  Model,
  ModelDownload,
  ModelLoader,
  type ModelSource,
  type RunOptions,
} from './model';
export { ConversationContext } from './context';
export { Pipeline } from './pipeline';
export { StreamingSession } from './session';
export { Envelope } from './envelope';
export { GenerationConfigs, creative, greedy } from './presets';
export {
  XybridErrorCodes,
  isRetryable,
  isXybridError,
  type XybridError,
  type XybridErrorCode,
} from './errors';
export type { AbortSignalLike } from './cancel';
export { bytesToBase64, float32ToBase64 } from './base64';

export type {
  AbortSignalKind,
  AudioEnvelope,
  CacheEntry,
  CacheEntryLocation,
  CacheStatus,
  ConversationInfo,
  DownloadState,
  DownloadStatus,
  EmbeddingEnvelope,
  EnvelopeKind,
  EnvelopeMetadata,
  ExecutionTarget,
  GenerationConfig,
  ImageEnvelope,
  ImageFormat,
  InferenceMetrics,
  InferenceResult,
  MessageRole,
  ModelInfo,
  MultiPartEnvelope,
  OutputType,
  PartialResult,
  PipelineInfo,
  PipelineResult,
  StageLatency,
  StageResult,
  StreamToken,
  StreamingConfig,
  TextEnvelope,
  ThermalState,
  ToolCall,
  ToolDefinition,
  ToolResult,
  VoiceInfo,
} from './types';

/**
 * Convert a JSON Schema (object or JSON string) into a GBNF grammar for
 * `generationConfig.grammar`, so a local LLM emits schema-valid JSON. Runs the
 * same native converter every binding uses; rejects with
 * `xybrid_config_error` on invalid JSON or an unsupported construct.
 *
 * ```ts
 * const grammar = await jsonSchemaToGbnf({
 *   type: 'object',
 *   properties: { name: { type: 'string' } },
 *   required: ['name'],
 * });
 * const result = await model.run(prompt, { generationConfig: { grammar } });
 * ```
 */
export function jsonSchemaToGbnf(schema: object | string): Promise<string> {
  return NativeXybrid.jsonSchemaToGbnf(
    typeof schema === 'string' ? schema : JSON.stringify(schema),
  );
}
