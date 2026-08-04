// Type definitions mirroring the bolt FFI surface in crates/xybrid-bolt/src/lib.rs.
// These cross the codegen boundary, so only TurboModule-supported primitives are
// used here: string, number, boolean, arrays of primitives, and plain object
// records. Binary payloads (audio bytes) ride as base64-encoded strings until
// the JSI variant lands — see README.md for the migration path.

export type ModelHandle = string;

export type ThermalState = 'normal' | 'warm' | 'hot' | 'critical';

export interface AudioEnvelope {
  kind: 'audio';
  /** PCM/WAV bytes, base64-encoded. */
  bytesBase64: string;
  sampleRate: number;
  channels: number;
}

export interface TextEnvelope {
  kind: 'text';
  text: string;
  voiceId?: string;
  speed?: number;
}

export interface EmbeddingEnvelope {
  kind: 'embedding';
  data: number[];
}

export type Envelope = AudioEnvelope | TextEnvelope | EmbeddingEnvelope;

export interface GenerationConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  minP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
}

/**
 * Device-stress signals that abort an in-flight run early. Mirrors the bolt
 * `XybridAbortSignal` enum 1:1; the native shims map each string onto the
 * corresponding enum case.
 */
export type AbortSignalKind =
  | 'memoryPressureWarn'
  | 'memoryPressureCritical'
  | 'thermalHot'
  | 'thermalCritical';

/**
 * Per-call execution policy, mirroring the bolt `XybridRunOptions` surface the
 * Apple/Kotlin SDKs expose. `generationConfig` carries the sampling params;
 * the remaining fields drive the platform plane (cloud fallback, abort-on-stress,
 * telemetry correlation).
 */
export interface RunOptions {
  /** Sampling parameters for LLM inference. */
  generationConfig?: GenerationConfig;
  /** Device-stress signals that abort the run early. */
  abortOn?: AbortSignalKind[];
  /** Allow this call to fall back to the cloud gateway under device stress. */
  fallbackToCloud?: boolean;
  /** Tokens to emit after an abort signal before stopping (grace window). */
  maxGraceTokens?: number;
  /** Correlation ID threaded into telemetry for this call. */
  correlationId?: string;
}

export interface InferenceResult {
  success: boolean;
  text?: string;
  /**
   * The model's chain-of-thought / reasoning text (LLM `<think>` blocks),
   * surfaced separately from `text`, which always excludes it. Absent when
   * the model emitted no reasoning or the backend doesn't surface one.
   */
  reasoningContent?: string;
  /** base64-encoded audio bytes when present. */
  audioBytesBase64?: string;
  embedding?: number[];
  latencyMs: number;
}

/** One token produced during streaming inference. */
export interface StreamToken {
  /** The decoded token text for this step. */
  token: string;
  /** Raw token id, when the backend exposes one. */
  tokenId?: number;
  /** Zero-based index of this token in the generation sequence. */
  index: number;
  /** All text generated so far (every token so far, concatenated). */
  cumulativeText: string;
  /**
   * Set only on the final token: `'stop'` (hit EOS / a stop sequence) or
   * `'length'` (hit the `maxTokens` cap). Absent while generation continues.
   */
  finishReason?: string;
}

/**
 * An event pulled from a streaming run. A stream yields zero or more `token`
 * events, then exactly one terminal `complete`. Mid-stream failures are not
 * an event: the native `streamNext` call rejects with the same typed
 * `xybrid_*` error codes as `run`.
 *
 * Crosses the codegen boundary as a plain object (the TurboModule spec can't
 * express a union — see `NativeXybrid.ts`) and is narrowed by its `kind`
 * discriminant (matching {@link Envelope}). Consumers normally use
 * {@link Model.runStreaming} rather than reading these directly.
 */
export type StreamEvent =
  | { kind: 'token'; token: StreamToken }
  | { kind: 'complete'; result: InferenceResult };

export interface VoiceInfo {
  id: string;
  name: string;
  gender?: string;
  language?: string;
  style?: string;
}
