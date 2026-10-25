// Compile-time checks of the public API (`npm run typecheck`). Each
// `@ts-expect-error` line must keep failing to type-check; the rest must keep
// compiling.

import {
  Envelope,
  GenerationConfigs,
  type InferenceMetrics,
  type InferenceResult,
  type Model,
  type RunOptions,
  type StreamToken,
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

  const cloudOptions: RunOptions = {
    fallbackToCloud: false,
    cloudProvider: 'openai',
    cloudModel: 'gpt-4o-mini',
    cloudGatewayUrl: 'https://api.xybrid.dev/v1',
  };
  await model.run({ kind: 'text', text: 'hi' }, cloudOptions);

  // @ts-expect-error unknown abort signal.
  const bad: RunOptions = { abortOn: ['meltdown'] };
  void bad;

  // @ts-expect-error envelopes are discriminated on `kind`.
  await model.run({ kind: 'text', bytesBase64: '' });

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
