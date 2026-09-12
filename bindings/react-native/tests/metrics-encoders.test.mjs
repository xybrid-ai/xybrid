import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const read = (relativePath) =>
  readFileSync(new URL(`../${relativePath}`, import.meta.url), 'utf8');

const types = read('src/types.ts');
const swift = read('ios/XybridModuleImpl.swift');
const kotlin = read('android/src/main/java/ai/xybrid/reactnative/XybridModule.kt');

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
  assert.match(swift, /"metrics":\s*encodeInferenceMetrics\(r\.metrics\)/);
});

test('Android result encoding wires all fields without fallback defaults', () => {
  assert.match(kotlin, /out\.putMap\("metrics",\s*encodeMetrics\(r\.metrics\)\)/);
  const encoder = kotlin.match(/private fun encodeMetrics[\s\S]*?\n  }/)?.[0];
  assert.ok(encoder, 'Android metrics encoder is missing');
  assert.match(encoder, /stageLatenciesMs\.forEach/);
  for (const field of requiredFields.slice(1, 6)) {
    assert.match(encoder, new RegExp(`m\\.${field}\\?\\.let`));
  }
  assert.doesNotMatch(encoder, /\?:\s*(?:0|0\.0)/);
});

test('batch and streaming terminal values share each platform result encoder', () => {
  assert.equal((swift.match(/encodeResult\(result\)/g) ?? []).length, 2);
  assert.equal((kotlin.match(/encodeResult\(result\)/g) ?? []).length, 2);
});
