import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const read = (relativePath) =>
  readFileSync(new URL(`../${relativePath}`, import.meta.url), 'utf8');

const requiredFields = [
  'totalMs',
  'ttftMs',
  'tokensPerSecond',
  'prefillTps',
  'decodeTps',
  'tokensOut',
  'stageLatenciesMs',
];

test('the public result exposes the complete canonical metrics contract', () => {
  const types = read('src/types.ts');
  assert.match(types, /export interface StageLatency\s*{[^}]*stageId:\s*string;[^}]*latencyMs:\s*number;/s);
  const metrics = types.match(/export interface InferenceMetrics\s*{(?<body>[^}]*)}/s)?.groups?.body;
  assert.ok(metrics, 'InferenceMetrics interface is missing');
  for (const field of requiredFields) assert.match(metrics, new RegExp(`\\b${field}\\??:`));
  assert.match(metrics, /\btotalMs:\s*number;/);
  for (const field of requiredFields.slice(1, 6)) {
    assert.match(metrics, new RegExp(`\\b${field}\\?:\\s*number;`));
  }
  assert.match(metrics, /\bstageLatenciesMs:\s*StageLatency\[\];/);
  assert.match(types, /export interface InferenceResult\s*{[^}]*metrics:\s*InferenceMetrics;/s);
});

test('iOS result encoding delegates to the production metrics converter', () => {
  const swift = read('ios/XybridModuleImpl.swift');
  assert.match(swift, /"metrics":\s*encodeInferenceMetrics\(r\.metrics\)/);
});

test('Android result encoding delegates to the production metrics converter', () => {
  const kotlin = read('android/src/main/java/ai/xybrid/reactnative/XybridModule.kt');
  assert.match(kotlin, /out\.putMap\("metrics",\s*encodeInferenceMetrics\(r\.metrics\)\)/);
});

test('Android metrics converter preserves actual fixture values', () => {
  // Compile the production file, not an extracted/copied implementation. Only
  // canonical records and RN containers are stand-ins; see kotlin/README.md.
  const directory = mkdtempSync(join(tmpdir(), 'xybrid-rn-metrics-'));
  const jar = join(directory, 'metrics-test.jar');
  const sources = [
    'android/src/main/java/ai/xybrid/reactnative/XybridMetricsEncoder.kt',
    'tests/kotlin/MetricsRecords.kt',
    'tests/kotlin/ReactBridge.kt',
    'tests/kotlin/main.kt',
  ].map((path) => fileURLToPath(new URL(`../${path}`, import.meta.url)));

  const run = (command, args, timeout) => {
    const result = spawnSync(command, args, { encoding: 'utf8', timeout });
    assert.ifError(result.error);
    assert.equal(result.status, 0, `${command} failed:\n${result.stdout}\n${result.stderr}`);
  };

  try {
    run('kotlinc', [...sources, '-Werror', '-include-runtime', '-d', jar], 120_000);
    run('java', ['-jar', jar], 30_000);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('batch and streaming terminal values share each platform result encoder', () => {
  const swift = read('ios/XybridModuleImpl.swift');
  const kotlin = read('android/src/main/java/ai/xybrid/reactnative/XybridModule.kt');
  assert.equal((swift.match(/encodeResult\(result\)/g) ?? []).length, 2);
  assert.equal((kotlin.match(/encodeResult\(result\)/g) ?? []).length, 2);
});
