import NativeXybrid from './NativeXybrid';
import type { ObservationOptions, ObservationRef } from './types';

/** Normalized W3C context; pass these fields as a child's ObservationOptions. */
export interface ObservationContext { traceparent: string; tracestate?: string }
export type ExecutionLocation = 'local' | 'remote';
export interface InferenceRequest {
  operation: string;
  provider: string;
  model?: string;
  location?: ExecutionLocation;
  runtime?: string;
  task?: string;
  stream?: boolean;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  serverAddress?: string;
  serverPort?: number;
}
export interface ToolCallInfo {
  name: string;
  callId?: string;
  toolType?: string;
  description?: string;
  argumentsJson?: string;
}
export interface Usage {
  inputTokens?: number;
  outputTokens?: number;
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
  reasoningOutputTokens?: number;
}
export interface ModelLoadInfo {
  model: string;
  provider?: string;
  runtime?: string;
  format?: string;
  sizeBytes?: number;
}

/** A native operation. Dispose explicitly; module teardown disposes remaining handles.
 * Completion, failure and cancellation are idempotent; the first one wins.
 */
class ManualObservation {
  protected constructor(readonly handle: string) {}
  async context(): Promise<ObservationContext | undefined> {
    return (await NativeXybrid.observationContext(this.handle) ?? undefined) as ObservationContext | undefined;
  }
  async observationRef(): Promise<ObservationRef | undefined> {
    return (await NativeXybrid.observationReference(this.handle) ?? undefined) as ObservationRef | undefined;
  }
  complete(): Promise<void> { return NativeXybrid.observationComplete(this.handle); }
  fail(errorType: string, message?: string): Promise<void> {
    return NativeXybrid.observationFail(this.handle, errorType, message ?? null);
  }
  cancel(): Promise<void> { return NativeXybrid.observationCancel(this.handle); }
  isFinished(): Promise<boolean> { return NativeXybrid.observationIsFinished(this.handle); }
  /** Cancels unfinished work when its final native owner is released. Never flushes. */
  dispose(): Promise<void> { return NativeXybrid.dispose(this.handle); }
  protected record(field: string, value: Object = {}): Promise<void> {
    return NativeXybrid.observationRecord(this.handle, field, value);
  }
}

export class WorkflowObservation extends ManualObservation {
  static async start(name: string, options?: ObservationOptions): Promise<WorkflowObservation> {
    return new WorkflowObservation(await NativeXybrid.observationStart('workflow', { name }, options ?? null));
  }
}
export class ToolObservation extends ManualObservation {
  static async start(call: ToolCallInfo, options?: ObservationOptions): Promise<ToolObservation> {
    return new ToolObservation(await NativeXybrid.observationStart('tool', call, options ?? null));
  }
  recordResult(resultJson: string): Promise<void> { return this.record('result', { json: resultJson }); }
}
export class InferenceObservation extends ManualObservation {
  static async start(request: InferenceRequest, options?: ObservationOptions): Promise<InferenceObservation> {
    return new InferenceObservation(await NativeXybrid.observationStart('inference', request, options ?? null));
  }
  recordSystemInstructions(partsJson: string): Promise<void> { return this.record('systemInstructions', { json: partsJson }); }
  recordInputMessages(messagesJson: string): Promise<void> { return this.record('inputMessages', { json: messagesJson }); }
  recordOutputMessages(messagesJson: string): Promise<void> { return this.record('outputMessages', { json: messagesJson }); }
  recordToolDefinitions(definitionsJson: string): Promise<void> { return this.record('toolDefinitions', { json: definitionsJson }); }
  recordChunk(): Promise<void> { return this.record('chunk'); }
  /** A measured timestamp in epoch milliseconds, or the current instant. */
  recordFirstToken(atEpochMs?: number): Promise<void> { return this.record('firstToken', { atEpochMs }); }
  recordUsage(usage: Usage): Promise<void> { return this.record('usage', usage); }
  recordFinishReasons(reasons: string[]): Promise<void> { return this.record('finishReasons', { reasons }); }
  recordResponse(responseId?: string, model?: string): Promise<void> { return this.record('response', { responseId, model }); }
}
export class ModelLoadObservation extends ManualObservation {
  static async start(info: ModelLoadInfo, options?: ObservationOptions): Promise<ModelLoadObservation> {
    return new ModelLoadObservation(await NativeXybrid.observationStart('modelLoad', info, options ?? null));
  }
  recordCacheHit(hit: boolean): Promise<void> { return this.record('cacheHit', { hit }); }
}
