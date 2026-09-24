// Translation between the public types and the plain objects that cross the
// TurboModule boundary. Every convenience field (voiceId, speed, sampleRate,
// channels, role) is folded into / lifted out of envelope metadata HERE, once,
// so the native codecs stay generic and lossless: an envelope is a payload
// kind plus string metadata, exactly like bolt's `XybridEnvelope`.
//
// Pure functions only — no native calls — so tests/wire.test.mjs covers them
// on Node.

import type {
  DownloadStatus,
  Envelope,
  EnvelopeMetadata,
  GenerationConfig,
  InferenceMetrics,
  InferenceResult,
  MessageRole,
  OutputType,
  PipelineResult,
  StageResult,
  StreamToken,
  ToolCall,
  ToolDefinition,
  ToolResult,
} from './types';

// Metadata keys the Rust core reads; see the Swift/Kotlin envelope factories.
const KEY_VOICE_ID = 'voice_id';
const KEY_SPEED = 'speed';
const KEY_SAMPLE_RATE = 'sample_rate';
const KEY_CHANNELS = 'channels';
const KEY_ROLE = 'xybrid.role';

const ROLES: readonly MessageRole[] = ['system', 'user', 'assistant'];

// -- Wire shapes --------------------------------------------------------------

export type WireEnvelope =
  | { kind: 'text'; text: string; metadata: EnvelopeMetadata }
  | { kind: 'audio'; bytesBase64: string; metadata: EnvelopeMetadata }
  | { kind: 'embedding'; data: number[]; metadata: EnvelopeMetadata }
  | { kind: 'image'; bytesBase64: string; format: string; metadata: EnvelopeMetadata }
  | { kind: 'multipart'; parts: WireEnvelope[]; metadata: EnvelopeMetadata };

export interface WireToolDefinition {
  name: string;
  description: string;
  parametersJson: string;
}

export interface WireGenerationConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  minP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
  grammar?: string;
  tools?: WireToolDefinition[];
}

export interface WireRunOptions {
  generationConfig?: WireGenerationConfig;
  abortOn?: string[];
  fallbackToCloud?: boolean;
  maxGraceTokens?: number;
  correlationId?: string;
  /** Conversation-context handle. */
  context?: string;
  /** Cancellation-token handle. */
  cancel?: string;
}

export interface WireResult {
  envelope: WireEnvelope;
  outputType: OutputType;
  modelId: string;
  latencyMs: number;
  executionTarget: 'local' | 'cloud';
  metrics: InferenceMetrics;
  toolCalls: ToolCall[];
  reasoningContent?: string;
}

export interface WireStreamToken {
  token: string;
  tokenId?: number;
  index: number;
  cumulativeText: string;
  finishReason?: string;
  toolCalls: ToolCall[];
  rawText?: string;
}

export type WireStreamEvent =
  | { kind: 'token'; token: WireStreamToken }
  | { kind: 'complete'; result: WireResult };

export interface WireStageResult {
  stageId: string;
  envelope: WireEnvelope;
  outputType: OutputType;
  latencyMs: number;
  executionTarget: 'local' | 'cloud';
  metrics: InferenceMetrics;
}

export interface WirePipelineResult {
  envelope: WireEnvelope;
  outputType: OutputType;
  latencyMs: number;
  stages: WireStageResult[];
}

// -- Envelopes ----------------------------------------------------------------

function setIfDefined(target: EnvelopeMetadata, key: string, value: unknown): void {
  if (value !== undefined && value !== null) target[key] = String(value);
}

/**
 * Fold an envelope's convenience fields into metadata. Typed fields win over
 * raw `metadata` entries for the same key; audio input defaults to 16 kHz
 * mono, like the Swift and Kotlin `Envelope.audio` factories.
 */
export function toWireEnvelope(envelope: Envelope): WireEnvelope {
  if (envelope == null || typeof envelope !== 'object') {
    throw new TypeError('An envelope must be an object with a `kind` field');
  }
  const metadata: EnvelopeMetadata = { ...(envelope.metadata ?? {}) };
  setIfDefined(metadata, KEY_ROLE, envelope.role);

  switch (envelope.kind) {
    case 'text':
      requireType(envelope.text, 'string', 'text envelope: `text`');
      setIfDefined(metadata, KEY_VOICE_ID, envelope.voiceId);
      setIfDefined(metadata, KEY_SPEED, envelope.speed);
      return { kind: 'text', text: envelope.text, metadata };
    case 'audio':
      requireType(envelope.bytesBase64, 'string', 'audio envelope: `bytesBase64`');
      setIfDefined(metadata, KEY_SAMPLE_RATE, envelope.sampleRate ?? metadata[KEY_SAMPLE_RATE] ?? 16000);
      setIfDefined(metadata, KEY_CHANNELS, envelope.channels ?? metadata[KEY_CHANNELS] ?? 1);
      return { kind: 'audio', bytesBase64: envelope.bytesBase64, metadata };
    case 'embedding':
      if (!Array.isArray(envelope.data)) {
        throw new TypeError('embedding envelope: `data` must be a number array');
      }
      return { kind: 'embedding', data: envelope.data, metadata };
    case 'image':
      requireType(envelope.bytesBase64, 'string', 'image envelope: `bytesBase64`');
      return {
        kind: 'image',
        bytesBase64: envelope.bytesBase64,
        format: normalizeImageFormat(envelope.format),
        metadata,
      };
    case 'multipart':
      if (!Array.isArray(envelope.parts)) {
        throw new TypeError('multipart envelope: `parts` must be an array');
      }
      return { kind: 'multipart', parts: envelope.parts.map(toWireEnvelope), metadata };
    default:
      throw new TypeError(
        `Unknown envelope kind: ${String((envelope as { kind?: unknown }).kind)}`,
      );
  }
}

/** Lift well-known metadata keys back into typed fields. Lossless. */
export function fromWireEnvelope(wire: WireEnvelope): Envelope {
  const metadata = { ...(wire.metadata ?? {}) };
  const role = metadata[KEY_ROLE] as MessageRole | undefined;
  const base = {
    ...(role && ROLES.includes(role) ? { role } : {}),
    metadata,
  };
  switch (wire.kind) {
    case 'text': {
      const speed = parseNumber(metadata[KEY_SPEED]);
      return {
        kind: 'text',
        text: wire.text,
        ...(metadata[KEY_VOICE_ID] !== undefined ? { voiceId: metadata[KEY_VOICE_ID] } : {}),
        ...(speed !== undefined ? { speed } : {}),
        ...base,
      };
    }
    case 'audio': {
      const sampleRate = parseNumber(metadata[KEY_SAMPLE_RATE]);
      const channels = parseNumber(metadata[KEY_CHANNELS]);
      return {
        kind: 'audio',
        bytesBase64: wire.bytesBase64,
        ...(sampleRate !== undefined ? { sampleRate } : {}),
        ...(channels !== undefined ? { channels } : {}),
        ...base,
      };
    }
    case 'embedding':
      return { kind: 'embedding', data: wire.data, ...base };
    case 'image':
      return {
        kind: 'image',
        bytesBase64: wire.bytesBase64,
        format: normalizeImageFormat(wire.format),
        ...base,
      };
    case 'multipart':
      return { kind: 'multipart', parts: wire.parts.map(fromWireEnvelope), ...base };
    default:
      throw new Error(
        `Unexpected envelope kind from native: ${String((wire as { kind?: unknown }).kind)}`,
      );
  }
}

/** Accept `jpg` as an alias and reject anything the core cannot decode. */
export function normalizeImageFormat(format: string): 'png' | 'jpeg' | 'webp' {
  const normalized = String(format ?? '').trim().toLowerCase();
  if (normalized === 'jpg' || normalized === 'jpeg') return 'jpeg';
  if (normalized === 'png' || normalized === 'webp') return normalized;
  throw new TypeError(`Unsupported image format '${format}'. Supported: png, jpeg, jpg, webp`);
}

// -- Options ------------------------------------------------------------------

const GENERATION_KEYS = [
  'maxTokens',
  'temperature',
  'topP',
  'minP',
  'topK',
  'repetitionPenalty',
  'stopSequences',
  'grammar',
  'tools',
] as const;

export function toWireToolDefinition(tool: ToolDefinition): WireToolDefinition {
  const parameters = tool.parameters;
  return {
    name: tool.name,
    description: tool.description,
    parametersJson: typeof parameters === 'string' ? parameters : JSON.stringify(parameters),
  };
}

export function toWireGenerationConfig(config: GenerationConfig): WireGenerationConfig {
  const { tools, ...rest } = config;
  return {
    ...rest,
    ...(tools ? { tools: tools.map(toWireToolDefinition) } : {}),
  };
}

export function fromWireGenerationConfig(wire: WireGenerationConfig): GenerationConfig {
  const { tools, ...rest } = wire;
  return {
    ...rest,
    tools: (tools ?? []).map((tool) => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parametersJson,
    })),
  };
}

/** Inputs to {@link toWireRunOptions}: the serializable `RunOptions` fields. */
export interface RunOptionsInput {
  generationConfig?: GenerationConfig;
  abortOn?: string[];
  fallbackToCloud?: boolean;
  maxGraceTokens?: number;
  correlationId?: string;
}

/**
 * Build the wire options. Sampling parameters must sit under
 * `generationConfig`; passing them at the top level is a mistake that would
 * otherwise be silently ignored, so it throws.
 */
export function toWireRunOptions(
  options: RunOptionsInput | undefined,
  handles: { context?: string | null; cancel?: string | null } = {},
): WireRunOptions | null {
  if (options != null && typeof options !== 'object') {
    throw new TypeError('Run options must be an object');
  }
  const misplaced = GENERATION_KEYS.filter((key) => options != null && key in options);
  if (misplaced.length > 0) {
    throw new TypeError(
      `Sampling parameters belong under \`generationConfig\` (got ${misplaced.join(', ')} ` +
        'at the top level of the run options)',
    );
  }
  const wire: WireRunOptions = {};
  if (options?.generationConfig) {
    wire.generationConfig = toWireGenerationConfig(options.generationConfig);
  }
  if (options?.abortOn) wire.abortOn = [...options.abortOn];
  if (options?.fallbackToCloud !== undefined) wire.fallbackToCloud = options.fallbackToCloud;
  if (options?.maxGraceTokens !== undefined) wire.maxGraceTokens = options.maxGraceTokens;
  if (options?.correlationId !== undefined) wire.correlationId = options.correlationId;
  if (handles.context) wire.context = handles.context;
  if (handles.cancel) wire.cancel = handles.cancel;
  return Object.keys(wire).length > 0 ? wire : null;
}

export function toWireToolResult(result: ToolResult): {
  callId: string;
  name: string;
  contentJson: string;
} {
  const { content } = result;
  return {
    callId: result.callId,
    name: result.name,
    contentJson: typeof content === 'string' ? content : JSON.stringify(content ?? null),
  };
}

// -- Results ------------------------------------------------------------------

/** Convenience payload fields derived from an envelope (no copies). */
function payloadFields(envelope: WireEnvelope): {
  text?: string;
  audioBytesBase64?: string;
  embedding?: number[];
} {
  switch (envelope.kind) {
    case 'text':
      return { text: envelope.text };
    case 'audio':
      return { audioBytesBase64: envelope.bytesBase64 };
    case 'embedding':
      return { embedding: envelope.data };
    default:
      return {};
  }
}

export function fromWireResult(wire: WireResult): InferenceResult {
  return {
    envelope: fromWireEnvelope(wire.envelope),
    outputType: wire.outputType,
    modelId: wire.modelId,
    ...payloadFields(wire.envelope),
    ...(wire.reasoningContent !== undefined ? { reasoningContent: wire.reasoningContent } : {}),
    toolCalls: wire.toolCalls ?? [],
    latencyMs: wire.latencyMs,
    metrics: wire.metrics,
    executionTarget: wire.executionTarget,
  };
}

export function fromWireStreamToken(wire: WireStreamToken): StreamToken {
  return { ...wire, toolCalls: wire.toolCalls ?? [] };
}

function fromWireStage(wire: WireStageResult): StageResult {
  const payload = payloadFields(wire.envelope);
  return {
    stageId: wire.stageId,
    envelope: fromWireEnvelope(wire.envelope),
    outputType: wire.outputType,
    ...(payload.text !== undefined ? { text: payload.text } : {}),
    ...(payload.audioBytesBase64 !== undefined
      ? { audioBytesBase64: payload.audioBytesBase64 }
      : {}),
    latencyMs: wire.latencyMs,
    executionTarget: wire.executionTarget,
    metrics: wire.metrics,
  };
}

export function fromWirePipelineResult(wire: WirePipelineResult): PipelineResult {
  const payload = payloadFields(wire.envelope);
  return {
    envelope: fromWireEnvelope(wire.envelope),
    outputType: wire.outputType,
    ...(payload.text !== undefined ? { text: payload.text } : {}),
    ...(payload.audioBytesBase64 !== undefined
      ? { audioBytesBase64: payload.audioBytesBase64 }
      : {}),
    latencyMs: wire.latencyMs,
    stages: wire.stages.map(fromWireStage),
  };
}

export function fromWireDownloadStatus(wire: DownloadStatus): DownloadStatus {
  // Both shims omit `totalBytes` when unknown; drop an explicit null too so
  // callers can rely on `status.totalBytes === undefined`.
  const { totalBytes, ...rest } = wire;
  return totalBytes == null ? rest : { ...rest, totalBytes };
}

// -- Helpers ------------------------------------------------------------------

function requireType(value: unknown, type: 'string', what: string): void {
  if (typeof value !== type) throw new TypeError(`${what} must be a ${type}`);
}

function parseNumber(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}
