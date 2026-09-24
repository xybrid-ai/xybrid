import NativeXybrid from './NativeXybrid';
import type { Envelope, PipelineInfo, PipelineResult } from './types';
import {
  fromWirePipelineResult,
  toWireEnvelope,
  type WirePipelineResult,
} from './wire';

/**
 * A multi-stage pipeline (e.g. ASR → LLM → TTS) defined in YAML. A run
 * returns every stage's output, so a voice assistant can show what it heard
 * and what it answered while it plays the audio:
 *
 * ```ts
 * const pipeline = await Pipeline.fromFile(path);
 * const result = await pipeline.run(Envelope.audio(pcmBase64));
 * const heard = result.stages.find((s) => s.stageId === 'asr')?.text;
 * ```
 */
export class Pipeline {
  private constructor(
    /** Opaque native handle. */
    readonly handle: string,
  ) {}

  /** Parse and load a pipeline from YAML content. */
  static async fromYaml(yaml: string): Promise<Pipeline> {
    return new Pipeline(await NativeXybrid.loadPipeline({ kind: 'yaml', value: yaml }));
  }

  /** Read, parse and load a pipeline YAML file. */
  static async fromFile(path: string): Promise<Pipeline> {
    return new Pipeline(await NativeXybrid.loadPipeline({ kind: 'file', value: path }));
  }

  /** Load a pipeline bundle. */
  static async fromBundle(path: string): Promise<Pipeline> {
    return new Pipeline(await NativeXybrid.loadPipeline({ kind: 'bundle', value: path }));
  }

  /** Name and stage ids, in execution order. */
  async info(): Promise<PipelineInfo> {
    return (await NativeXybrid.pipelineInfo(this.handle)) as PipelineInfo;
  }

  /**
   * Run every stage, downloading any missing model first. Per-stage
   * generation settings belong in the YAML; only `correlationId` applies here.
   */
  async run(
    envelope: Envelope,
    options: { correlationId?: string } = {},
  ): Promise<PipelineResult> {
    const wire = await NativeXybrid.runPipeline(
      this.handle,
      toWireEnvelope(envelope),
      options.correlationId !== undefined ? { correlationId: options.correlationId } : null,
    );
    return fromWirePipelineResult(wire as WirePipelineResult);
  }

  /** Free the native pipeline. */
  release(): Promise<void> {
    return NativeXybrid.dispose(this.handle);
  }
}
