import NativeXybrid from './NativeXybrid';
import type { ConversationInfo, Envelope } from './types';
import { fromWireEnvelope, toWireEnvelope, type WireEnvelope } from './wire';

/**
 * Multi-turn conversation history, held natively. Pass it to
 * `model.run(envelope, { context })` or `runStreaming`; inference reads the
 * history but never changes it. The prompt you run is appended to the
 * history for that run only, so push both turns *after* the run — pushing the
 * prompt first would send it twice:
 *
 * ```ts
 * const chat = await ConversationContext.create();
 * await chat.setSystem('You are terse.');
 *
 * const question = Envelope.user('What is the capital of France?');
 * const reply = await model.run(question, { context: chat });
 * await chat.push(question);
 * await chat.push(reply.envelope); // already tagged `assistant`
 * ```
 */
export class ConversationContext {
  private constructor(
    /** Opaque native handle. */
    readonly handle: string,
  ) {}

  /** A new, empty conversation; `id` defaults to a fresh one (telemetry correlation). */
  static async create(id?: string): Promise<ConversationContext> {
    return new ConversationContext(await NativeXybrid.createContext(id ?? null));
  }

  /** Append a turn. Give it a `role` (see `Envelope.user` / `Envelope.assistant`). */
  push(envelope: Envelope): Promise<void> {
    return NativeXybrid.contextPush(this.handle, toWireEnvelope(envelope));
  }

  /** Set the persistent system prompt; it survives {@link clear}. */
  setSystem(system: Envelope | string): Promise<void> {
    const envelope: Envelope =
      typeof system === 'string' ? { kind: 'text', text: system, role: 'system' } : system;
    return NativeXybrid.contextSetSystem(this.handle, toWireEnvelope(envelope));
  }

  /** Drop the history; the system prompt is kept. */
  clear(): Promise<void> {
    return NativeXybrid.contextClear(this.handle);
  }

  /** Id, history length and whether a system prompt is set. */
  async info(): Promise<ConversationInfo> {
    return (await NativeXybrid.contextInfo(this.handle)) as ConversationInfo;
  }

  /** History turns in order, excluding the system prompt. */
  async history(): Promise<Envelope[]> {
    const wire = (await NativeXybrid.contextHistory(this.handle)) as WireEnvelope[];
    return wire.map(fromWireEnvelope);
  }

  /** Cap the history; older turns are pruned first, immediately. */
  setMaxHistoryLength(length: number): Promise<void> {
    return NativeXybrid.contextSetMaxHistoryLength(this.handle, length);
  }

  /** Free the native context. Runs already started keep their snapshot. */
  release(): Promise<void> {
    return NativeXybrid.dispose(this.handle);
  }
}
