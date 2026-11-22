/**
 * JSON report written into the app sandbox for the host runner to collect.
 *
 * The host validates schema, required step ids, versions, and that `success`
 * is true only after cleanup. Do not mark success before handles are released.
 */

import { File, Paths } from 'expo-file-system';

export const REPORT_FILE_NAME = 'xybrid-consumer-smoke-report.json';
export const REPORT_SCHEMA_VERSION = 1;

/** Steps the host requires, in order. */
export const REQUIRED_STEP_IDS = [
  'version_match',
  'model_load',
  'model_info',
  'warmup',
  'inference',
  'streaming',
  'cancel_settles',
  'post_cancel_inference',
  'conversation',
  'constrained_json',
  'release',
] as const;

export type StepId = (typeof REQUIRED_STEP_IDS)[number];

export type StepResult = {
  id: StepId;
  ok: boolean;
  detail?: string;
  durationMs?: number;
};

export type SmokeReport = {
  schemaVersion: number;
  runId: string;
  startedAt: string;
  finishedAt?: string;
  success: boolean;
  versions: {
    npmPackage: string;
    nativeSdk: string;
    expo: string;
    reactNative: string;
  };
  results: StepResult[];
  error?: string;
};

export function reportPath(): string {
  return new File(Paths.document, REPORT_FILE_NAME).uri;
}

export function newRunId(): string {
  // Prefer crypto.randomUUID when Hermes provides it; fall back for older runtimes.
  const c = globalThis.crypto as { randomUUID?: () => string } | undefined;
  if (c?.randomUUID) return c.randomUUID();
  return `run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function writeReport(report: SmokeReport): Promise<void> {
  const file = new File(Paths.document, REPORT_FILE_NAME);
  if (!file.exists) {
    file.create();
  }
  await file.write(JSON.stringify(report, null, 2));
}
