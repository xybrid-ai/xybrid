import type { InferenceMetrics, InferenceResult } from '../src/types';

const populatedMetrics = {
  totalMs: 120,
  ttftMs: 18,
  tokensPerSecond: 42.5,
  prefillTps: 95.25,
  decodeTps: 39.75,
  tokensOut: 24,
  stageLatenciesMs: [
    { stageId: 'preprocess', latencyMs: 7 },
    { stageId: 'inference', latencyMs: 108 },
  ],
} satisfies InferenceMetrics;

const metricsWithoutLlmFields = {
  totalMs: 33,
  stageLatenciesMs: [{ stageId: 'inference', latencyMs: 31 }],
} satisfies InferenceMetrics;

const result = {
  success: true,
  text: 'ok',
  latencyMs: populatedMetrics.totalMs,
  metrics: populatedMetrics,
  executionTarget: 'local',
} satisfies InferenceResult;

void result;
void metricsWithoutLlmFields;

// totalMs is the always-available wall-clock measurement.
// @ts-expect-error totalMs must not become optional.
const missingTotalMs: InferenceMetrics = { stageLatenciesMs: [] };
void missingTotalMs;
