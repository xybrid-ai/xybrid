// Compile-time checks of the public API (`npm run typecheck`). Each
// `@ts-expect-error` line must keep failing to type-check; the rest must keep
// compiling.

import {
  Envelope,
  GenerationConfigs,
  Observability,
  ObservabilityConfig,
  type InferenceMetrics,
  type InferenceResult,
  type Model,
  type ObservationRef,
  type RunOptions,
  type StreamToken,
  Xybrid,
  isXybridError,
} from '../../src';

declare const model: Model;
declare const signal: AbortSignal;

async function usage(): Promise<void> {
  const result: InferenceResult = await model.run(Envelope.text('hi'), {
    generationConfig: GenerationConfigs.greedy({ maxTokens: 32 }),
    abortOn: ['thermalCritical'],
    signal,
  });
  const text: string | undefined = result.text;
  void text;

  for await (const token of model.runStreaming({ kind: 'text', text: 'hi' })) {
    const t: StreamToken = token;
    void t;
  }

  const options: RunOptions = {
    // @ts-expect-error sampling parameters live under `generationConfig`.
    maxTokens: 32,
  };
  void options;

  // @ts-expect-error unknown abort signal.
  const bad: RunOptions = { abortOn: ['meltdown'] };
  void bad;

  // @ts-expect-error envelopes are discriminated on `kind`.
  await model.run({ kind: 'text', bytesBase64: '' });

  await ObservabilityConfig.xybrid('key').setService('app', '1.0').setMetadataOnly(true).init();
  const traced = await model.run(Envelope.text('hi'), { traceparent: '00-…-01', suppressTrace: false });
  const observation: ObservationRef | undefined = traced.observation;
  if (observation) await Observability.recordFeedback(observation, { score: 1 });
  await Observability.flush(2000);

  // @ts-expect-error ingestUrl was removed; export is configured with ObservabilityConfig.
  await Xybrid.initialize({ apiKey: 'key', ingestUrl: 'https://ingest.example' });

  // @ts-expect-error an evaluation needs a name.
  await Observability.recordEvaluation({ traceId: 't', spanId: 's' }, { scoreValue: 1 });

  try {
    await model.warmup();
  } catch (error) {
    if (isXybridError(error)) {
      // Narrowed: the code is one of the documented literals.
      const code: 'xybrid_offline' | string = error.code;
      void code;
    }
  }
}
void usage;

// totalMs is the always-available measurement.
// @ts-expect-error totalMs must not become optional.
const missingTotal: InferenceMetrics = { stageLatenciesMs: [] };
void missingTotal;
