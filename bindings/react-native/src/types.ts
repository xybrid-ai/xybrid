// Public types. They mirror the bolt `#[data]` records in
// crates/xybrid-bolt/src/lib.rs field for field (camelCase), so a value means
// the same thing here as in the Swift and Kotlin SDKs. tests/parity.test.mjs
// fails when bolt gains a field or export that is neither mapped here nor
// listed as intentionally excluded in parity.json.
//
// Binary payloads (audio, images) cross as base64 strings: TurboModules have
// no ArrayBuffer type. See README "Binary payloads".

/** Thermal band pushed to the routing engine. */
export type ThermalState = 'normal' | 'warm' | 'hot' | 'critical';

/** LLM message role, stored as `xybrid.role` envelope metadata. */
export type MessageRole = 'system' | 'user' | 'assistant';

/** Image encodings accepted by vision-language models. */
export type ImageFormat = 'png' | 'jpeg' | 'webp';

/** Free-form envelope metadata. Well-known keys have typed fields instead. */
export type EnvelopeMetadata = Record<string, string>;

interface EnvelopeBase {
  /** LLM message role (`xybrid.role` metadata). */
  role?: MessageRole;
  /** Raw metadata entries; typed fields win when both are set. */
  metadata?: EnvelopeMetadata;
}

/** Text input: an LLM prompt, or text to speak for TTS models. */
export interface TextEnvelope extends EnvelopeBase {
  kind: 'text';
  text: string;
  /** TTS voice (`voice_id` metadata). */
  voiceId?: string;
  /** TTS speed multiplier (`speed` metadata). */
  speed?: number;
}

/** Audio input or output: PCM/WAV bytes. */
export interface AudioEnvelope extends EnvelopeBase {
  kind: 'audio';
  /** PCM/WAV bytes, base64-encoded. */
  bytesBase64: string;
  /** Sample rate in Hz (`sample_rate` metadata). Defaults to 16000 on input. */
  sampleRate?: number;
  /** Channel count (`channels` metadata). Defaults to 1 on input. */
  channels?: number;
}

/** An embedding vector. */
export interface EmbeddingEnvelope extends EnvelopeBase {
  kind: 'embedding';
  data: number[];
}

/** An encoded image for vision-language models. */
export interface ImageEnvelope extends EnvelopeBase {
  kind: 'image';
  /** PNG, JPEG or WebP bytes, base64-encoded. */
  bytesBase64: string;
  format: ImageFormat;
}

/** A multi-part message, e.g. a prompt plus image attachments. */
export interface MultiPartEnvelope extends EnvelopeBase {
  kind: 'multipart';
  parts: Envelope[];
}

/** Input to (and output of) a model. Narrow on `kind`. */
export type Envelope =
  | TextEnvelope
  | AudioEnvelope
  | EmbeddingEnvelope
  | ImageEnvelope
  | MultiPartEnvelope;

/** The payload kinds an {@link Envelope} can carry. */
export type EnvelopeKind = Envelope['kind'];

/** A tool (function) the model may ask to call. */
export interface ToolDefinition {
  name: string;
  description: string;
  /**
   * JSON Schema of the arguments, as an object or a JSON string. Crosses the
   * boundary as a string (`parametersJson` in the other SDKs).
   */
  parameters: object | string;
}

/** A tool call the model emitted. */
export interface ToolCall {
  /** Echo this back as {@link ToolResult.callId}. */
  id: string;
  name: string;
  /** Arguments as a JSON string, exactly as the model produced them. */
  argumentsJson: string;
}

/** The outcome of running one tool, fed back with `Envelope.toolResults`. */
export interface ToolResult {
  /** The {@link ToolCall.id} this answers. */
  callId: string;
  name: string;
  /**
   * The tool's output: any JSON-serializable value, or an already-encoded
   * JSON string.
   */
  content: unknown;
}

/** Sampling parameters for LLM inference. Unset fields use model defaults. */
export interface GenerationConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  minP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
  /**
   * GBNF grammar constraining output (local llama backend only). Build one
   * from a JSON Schema with `jsonSchemaToGbnf`, or pass raw GBNF.
   */
  grammar?: string;
  /** Tools the model may call this turn. Empty means no tool calling. */
  tools?: ToolDefinition[];
}

/** Device-stress signals that abort a run early (bolt `XybridAbortSignal`). */
export type AbortSignalKind =
  | 'memoryPressureWarn'
  | 'memoryPressureCritical'
  | 'thermalHot'
  | 'thermalCritical';

/** Where a result was produced — observed fact, not a routing preference. */
export type ExecutionTarget = 'local' | 'cloud';

/** What kind of payload a model produces. */
export type OutputType = 'text' | 'audio' | 'embedding' | 'unknown';

/** Latency of one pipeline stage. */
export interface StageLatency {
  stageId: string;
  latencyMs: number;
}

/**
 * Measurements from the native runtime. LLM-only values are absent when the
 * model or backend does not report them.
 */
export interface InferenceMetrics {
  /** Total wall-clock inference time in milliseconds. */
  totalMs: number;
  /** Time to first generated token in milliseconds. */
  ttftMs?: number;
  /** Overall generated-token throughput. */
  tokensPerSecond?: number;
  /** Prompt-processing throughput. */
  prefillTps?: number;
  /** Generated-token decode throughput. */
  decodeTps?: number;
  /** Number of generated tokens. */
  tokensOut?: number;
  /** Per-stage measurements in native execution order. */
  stageLatenciesMs: StageLatency[];
}

/** The result of one inference. */
export interface InferenceResult {
  /** The output, losslessly (payload plus metadata). */
  envelope: Envelope;
  outputType: OutputType;
  /** The model that answered — identical on the local and cloud legs. */
  modelId: string;
  /** Text output, when the model produced text. */
  text?: string;
  /** Audio output (base64), when the model produced audio. */
  audioBytesBase64?: string;
  /** Embedding output, when the model produced one. */
  embedding?: number[];
  /**
   * Chain-of-thought (`<think>` blocks) surfaced separately from `text`,
   * which never includes it. Absent when the model emitted none.
   */
  reasoningContent?: string;
  /** Tool calls the model emitted this turn. Empty unless tools were offered. */
  toolCalls: ToolCall[];
  latencyMs: number;
  metrics: InferenceMetrics;
  /** Where this answer actually came from. */
  executionTarget: ExecutionTarget;
}

/** One token from a streaming run. */
export interface StreamToken {
  /** The decoded text of this step. */
  token: string;
  /** Raw token id, when the backend exposes one. */
  tokenId?: number;
  /** Zero-based position in the generated sequence. */
  index: number;
  /** Every emitted token so far, concatenated (tool-call blocks suppressed). */
  cumulativeText: string;
  /**
   * Set on the final token only: `'stop'`, `'length'`, or `'tool_calls'`
   * when the turn ended on a tool-call block.
   */
  finishReason?: string;
  /** Tool calls parsed from the turn — terminal token only. */
  toolCalls: ToolCall[];
  /**
   * The turn's raw output, tool-call block included — pass it to
   * `Envelope.toolResults` as `priorAssistantText`. Present only alongside
   * `toolCalls`.
   */
  rawText?: string;
}

/** Lifecycle of a model download. */
export type DownloadState = 'downloading' | 'ready' | 'failed' | 'cancelled';

/**
 * Download progress, bytes and state in one consistent read. `progress` is
 * aggregated across every artifact, never moves backwards, and reaches 1.0
 * only alongside `ready`.
 */
export interface DownloadStatus {
  state: DownloadState;
  /** 0.0 to 1.0. */
  progress: number;
  /** Bytes written so far, across every artifact. */
  downloadedBytes: number;
  /**
   * Declared total across every artifact; absent when the source publishes
   * no size (a Hugging Face repo, or a registry entry without one).
   */
  totalBytes?: number;
}

/** A TTS voice. */
export interface VoiceInfo {
  id: string;
  name: string;
  gender?: string;
  language?: string;
  style?: string;
}

/** Identity and capabilities of a loaded model. */
export interface ModelInfo {
  /** Registry / metadata id — not the native handle. */
  modelId: string;
  version: string;
  outputType: OutputType;
  isLlm: boolean;
  supportsStreaming: boolean;
  /** Whether the model emits true token-by-token output. */
  supportsTokenStreaming: boolean;
  /**
   * Whether the bundle declares tool-calling support; `null` when it says
   * nothing. Advisory — enforcement happens at run time.
   */
  supportsToolCalling: boolean | null;
  hasVoices: boolean;
  /** The model's resolved generation defaults. */
  defaultGenerationConfig: GenerationConfig;
}

/** Where a physical cache entry lives. */
export type CacheEntryLocation = 'registry' | 'extracted' | 'huggingFace' | 'huggingFaceHub';

/** One physical model entry in managed cache storage. */
export interface CacheEntry {
  modelId: string;
  location: CacheEntryLocation;
  path: string;
  sizeBytes: number;
}

/** Aggregate storage usage across every managed cache location. */
export interface CacheStatus {
  totalSizeBytes: number;
  entryCount: number;
  modelCount: number;
  extractedModelCount: number;
  cacheRoot: string;
}

/** Snapshot of a conversation context. */
export interface ConversationInfo {
  id: string;
  /** History turns, excluding the system envelope. */
  historyLength: number;
  hasSystem: boolean;
}

/** What one pipeline stage produced. */
export interface StageResult {
  /** The YAML `id:` (or the model id when the stage declares none). */
  stageId: string;
  /** This stage's output — also the next stage's input. */
  envelope: Envelope;
  outputType: OutputType;
  text?: string;
  audioBytesBase64?: string;
  latencyMs: number;
  /** Stages of one pipeline can run in different places. */
  executionTarget: ExecutionTarget;
  metrics: InferenceMetrics;
}

/** Result of a pipeline run: the final output plus every stage's own. */
export interface PipelineResult {
  /** The final stage's output. */
  envelope: Envelope;
  outputType: OutputType;
  text?: string;
  audioBytesBase64?: string;
  /** Wall-clock time of the whole run. */
  latencyMs: number;
  /** Every executed stage, in order. */
  stages: StageResult[];
}

/** Static description of a loaded pipeline. */
export interface PipelineInfo {
  name: string | null;
  stageNames: string[];
  stageCount: number;
}

/** Chunking options for a live ASR session. */
export interface StreamingConfig {
  /** Must be 16000 — the ASR backends are fixed there. Default 16000. */
  sampleRate?: number;
  /**
   * Chunk on speech boundaries with a Silero VAD model in `modelDir`
   * (containing `model.onnx`). Omit for fixed time windows.
   */
  vad?: { modelDir: string } | null;
  /** VAD sensitivity, 0.0–1.0. Default 0.5. Ignored without `vad`. */
  vadThreshold?: number;
  /** Language hint such as `'en'`; omit for the model default. */
  language?: string;
  /** Whisper encoder context in mel frames; omit for the model default. */
  audioCtx?: number;
}

/** A partial transcript from a live ASR session. */
export interface PartialResult {
  /** Transcript so far. Cumulative — replace the previous partial. */
  text: string;
  /** `true` once this span is committed and will not change. */
  isStable: boolean;
  /** Monotonic chunk sequence number. */
  chunkSequence: number;
  /** Audio covered so far, in milliseconds. */
  audioDurationMs: number;
}
