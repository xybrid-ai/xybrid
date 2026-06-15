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

export interface InferenceResult {
  success: boolean;
  text?: string;
  /** base64-encoded audio bytes when present. */
  audioBytesBase64?: string;
  embedding?: number[];
  latencyMs: number;
}

export interface VoiceInfo {
  id: string;
  name: string;
  gender?: string;
  language?: string;
  style?: string;
}
