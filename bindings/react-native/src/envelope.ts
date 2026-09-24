import NativeXybrid from './NativeXybrid';
import type {
  AudioEnvelope,
  EmbeddingEnvelope,
  Envelope as EnvelopeType,
  ImageEnvelope,
  ImageFormat,
  MessageRole,
  MultiPartEnvelope,
  TextEnvelope,
  ToolResult,
} from './types';
import {
  fromWireEnvelope,
  normalizeImageFormat,
  toWireToolResult,
  type WireEnvelope,
} from './wire';

/** Input to (and output of) a model. Narrow on `kind`. */
export type Envelope = EnvelopeType;

/**
 * Envelope factories, named like the Swift and Kotlin `Envelope` helpers.
 * Object literals (`{ kind: 'text', text }`) work just as well; these only
 * save typing and apply the same defaults.
 */
export const Envelope = {
  /** A text prompt, or text to speak with an optional TTS voice and speed. */
  text(
    text: string,
    options: { voiceId?: string; speed?: number; role?: MessageRole } = {},
  ): TextEnvelope {
    return { kind: 'text', text, ...options };
  },

  /** A user turn for a conversation. */
  user(text: string): TextEnvelope {
    return { kind: 'text', text, role: 'user' };
  },

  /** An assistant turn for a conversation (e.g. a model reply you keep). */
  assistant(text: string): TextEnvelope {
    return { kind: 'text', text, role: 'assistant' };
  },

  /** A system prompt. */
  system(text: string): TextEnvelope {
    return { kind: 'text', text, role: 'system' };
  },

  /** PCM/WAV audio, base64-encoded. Defaults to 16 kHz mono. */
  audio(
    bytesBase64: string,
    options: { sampleRate?: number; channels?: number } = {},
  ): AudioEnvelope {
    return {
      kind: 'audio',
      bytesBase64,
      sampleRate: options.sampleRate ?? 16000,
      channels: options.channels ?? 1,
    };
  },

  /** An embedding vector. */
  embedding(data: number[]): EmbeddingEnvelope {
    return { kind: 'embedding', data };
  },

  /**
   * An encoded image for vision-language models. The bytes are validated
   * natively at run time (`xybrid_invalid_image` on bad input).
   */
  image(bytesBase64: string, format: ImageFormat | 'jpg'): ImageEnvelope {
    return { kind: 'image', bytesBase64, format: normalizeImageFormat(format) };
  },

  /** A user message with image attachments, for vision-language models. */
  userMessage(text: string, images: ImageEnvelope[] = []): MultiPartEnvelope {
    if (!images.every((image) => image?.kind === 'image')) {
      throw new TypeError('Envelope.userMessage accepts only image envelopes');
    }
    return {
      kind: 'multipart',
      parts: [{ kind: 'text', text }, ...images],
      role: 'user',
    };
  },

  /**
   * The continuation envelope for the turn after the model asked for tools.
   *
   * One `run` is one model turn, so the tool loop lives in your code: run a
   * request carrying `generationConfig.tools`, execute every
   * `result.toolCalls` entry, then run this envelope (with the same tools) to
   * feed the outcomes back.
   *
   * @param userText the original user message of the turn being continued.
   * @param priorAssistantText that turn's raw output — `result.text` from a
   *   batch run, or the terminal token's `rawText` when streaming.
   * @param results tool outcomes, in call order.
   */
  async toolResults(
    userText: string,
    priorAssistantText: string,
    results: ToolResult[],
  ): Promise<Envelope> {
    const wire = (await NativeXybrid.toolResultsEnvelope(
      userText,
      priorAssistantText,
      results.map(toWireToolResult),
    )) as WireEnvelope;
    return fromWireEnvelope(wire);
  },
};
