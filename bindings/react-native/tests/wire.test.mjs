// The TS half of the wire contract (src/wire.ts). The native halves are
// covered by tests/swift and tests/kotlin against the same shapes.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { loadLibrary, loadWire } from './support/fake-native.mjs';

const wire = loadWire();
const { bytesToBase64, float32ToBase64 } = loadLibrary();

test('text sugar folds into metadata and lifts back out', () => {
  const envelope = {
    kind: 'text',
    text: 'hi',
    voiceId: 'af_bella',
    speed: 1.25,
    role: 'assistant',
    metadata: { custom: 'x', voice_id: 'overridden' },
  };
  const encoded = wire.toWireEnvelope(envelope);
  assert.deepEqual(encoded, {
    kind: 'text',
    text: 'hi',
    metadata: { custom: 'x', voice_id: 'af_bella', speed: '1.25', 'xybrid.role': 'assistant' },
  });
  const decoded = wire.fromWireEnvelope(encoded);
  assert.equal(decoded.voiceId, 'af_bella');
  assert.equal(decoded.speed, 1.25);
  assert.equal(decoded.role, 'assistant');
  assert.equal(decoded.metadata.custom, 'x');
  // Round trip is stable.
  assert.deepEqual(wire.toWireEnvelope(decoded), encoded);
});

test('audio input defaults to 16 kHz mono unless told otherwise', () => {
  assert.deepEqual(wire.toWireEnvelope({ kind: 'audio', bytesBase64: 'AAAA' }).metadata, {
    sample_rate: '16000',
    channels: '1',
  });
  assert.deepEqual(
    wire.toWireEnvelope({ kind: 'audio', bytesBase64: 'AAAA', sampleRate: 24000, channels: 2 }).metadata,
    { sample_rate: '24000', channels: '2' },
  );
  assert.deepEqual(
    wire.toWireEnvelope({ kind: 'audio', bytesBase64: 'AAAA', metadata: { sample_rate: '8000' } }).metadata,
    { sample_rate: '8000', channels: '1' },
  );
});

test('images normalize their format and multipart recurses', () => {
  const encoded = wire.toWireEnvelope({
    kind: 'multipart',
    role: 'user',
    parts: [
      { kind: 'text', text: 'what is this?' },
      { kind: 'image', bytesBase64: 'iVBO', format: 'JPG' },
    ],
  });
  assert.deepEqual(encoded, {
    kind: 'multipart',
    metadata: { 'xybrid.role': 'user' },
    parts: [
      { kind: 'text', text: 'what is this?', metadata: {} },
      { kind: 'image', bytesBase64: 'iVBO', format: 'jpeg', metadata: {} },
    ],
  });
  assert.throws(() => wire.normalizeImageFormat('gif'), /Unsupported image format/);
});

test('malformed envelopes fail in JS, before the bridge', () => {
  assert.throws(() => wire.toWireEnvelope({ kind: 'video' }), /Unknown envelope kind/);
  assert.throws(() => wire.toWireEnvelope({ kind: 'text' }), /text envelope/);
  assert.throws(() => wire.toWireEnvelope({ kind: 'embedding', data: 'nope' }), /number array/);
  assert.throws(() => wire.toWireEnvelope(null), /must be an object/);
});

test('run options carry sampling, platform knobs and handles', () => {
  assert.equal(wire.toWireRunOptions(undefined), null);
  assert.equal(wire.toWireRunOptions({}), null);
  assert.deepEqual(
    wire.toWireRunOptions(
      {
        generationConfig: {
          maxTokens: 64,
          tools: [{ name: 'f', description: 'd', parameters: { type: 'object' } }],
        },
        abortOn: ['thermalHot'],
        fallbackToCloud: true,
        maxGraceTokens: 8,
        correlationId: 'req-1',
        cloudProvider: 'openai',
        cloudModel: 'gpt-4o-mini',
        cloudGatewayUrl: 'https://api.xybrid.dev/v1',
      },
      { context: 'context:1', cancel: 'cancel:1' },
    ),
    {
      generationConfig: {
        maxTokens: 64,
        tools: [{ name: 'f', description: 'd', parametersJson: '{"type":"object"}' }],
      },
      abortOn: ['thermalHot'],
      fallbackToCloud: true,
      maxGraceTokens: 8,
      correlationId: 'req-1',
      cloudProvider: 'openai',
      cloudModel: 'gpt-4o-mini',
      cloudGatewayUrl: 'https://api.xybrid.dev/v1',
      context: 'context:1',
      cancel: 'cancel:1',
    },
  );
});

test('cloud destination does not enable fallback and blank values survive the wire', () => {
  assert.deepEqual(wire.toWireRunOptions({
    fallbackToCloud: false,
    cloudProvider: '',
    cloudModel: ' ',
    cloudGatewayUrl: '',
  }), {
    fallbackToCloud: false,
    cloudProvider: '',
    cloudModel: ' ',
    cloudGatewayUrl: '',
  });
});

test('top-level sampling parameters are rejected instead of ignored', () => {
  assert.throws(
    () => wire.toWireRunOptions({ maxTokens: 64, temperature: 0 }),
    /belong under `generationConfig` \(got maxTokens, temperature/,
  );
});

test('tool results stringify content unless it is already JSON', () => {
  assert.deepEqual(wire.toWireToolResult({ callId: 'c', name: 'n', content: { ok: true } }), {
    callId: 'c',
    name: 'n',
    contentJson: '{"ok":true}',
  });
  assert.equal(wire.toWireToolResult({ callId: 'c', name: 'n', content: '[1]' }).contentJson, '[1]');
  assert.equal(wire.toWireToolResult({ callId: 'c', name: 'n', content: undefined }).contentJson, 'null');
});

test('results expose the lossless envelope plus payload shortcuts', () => {
  const metrics = { totalMs: 5, stageLatenciesMs: [] };
  const text = wire.fromWireResult({
    envelope: { kind: 'text', text: 'hello', metadata: { 'xybrid.role': 'assistant' } },
    outputType: 'text',
    modelId: 'm',
    latencyMs: 5,
    executionTarget: 'cloud',
    metrics,
    toolCalls: [{ id: '1', name: 'f', argumentsJson: '{}' }],
    reasoningContent: 'thinking',
  });
  assert.equal(text.text, 'hello');
  assert.equal(text.envelope.role, 'assistant');
  assert.equal(text.reasoningContent, 'thinking');
  assert.equal(text.executionTarget, 'cloud');
  assert.equal(text.toolCalls.length, 1);
  assert.equal('audioBytesBase64' in text, false);

  const audio = wire.fromWireResult({
    envelope: { kind: 'audio', bytesBase64: 'UklG', metadata: { sample_rate: '24000' } },
    outputType: 'audio',
    modelId: 'tts',
    latencyMs: 1,
    executionTarget: 'local',
    metrics,
    toolCalls: [],
  });
  assert.equal(audio.audioBytesBase64, 'UklG');
  assert.equal(audio.envelope.sampleRate, 24000);
  assert.equal('reasoningContent' in audio, false);
});

test('an unknown download size reads as undefined on every platform', () => {
  const base = { state: 'downloading', progress: 0.5, downloadedBytes: 10 };
  assert.equal('totalBytes' in wire.fromWireDownloadStatus({ ...base, totalBytes: null }), false);
  assert.equal(wire.fromWireDownloadStatus({ ...base, totalBytes: 20 }).totalBytes, 20);
});

test('pipeline results carry each stage with payload shortcuts', () => {
  const metrics = { totalMs: 1, stageLatenciesMs: [] };
  const result = wire.fromWirePipelineResult({
    envelope: { kind: 'audio', bytesBase64: 'AA==', metadata: {} },
    outputType: 'audio',
    latencyMs: 30,
    stages: [
      { stageId: 'asr', envelope: { kind: 'text', text: 'hi', metadata: {} }, outputType: 'text', latencyMs: 10, executionTarget: 'local', metrics },
      { stageId: 'tts', envelope: { kind: 'audio', bytesBase64: 'AA==', metadata: {} }, outputType: 'audio', latencyMs: 20, executionTarget: 'local', metrics },
    ],
  });
  assert.equal(result.audioBytesBase64, 'AA==');
  assert.equal(result.stages[0].text, 'hi');
  assert.equal(result.stages[1].audioBytesBase64, 'AA==');
});

test('base64 matches Node for every length and PCM stays little-endian', () => {
  for (let length = 0; length < 70; length++) {
    const bytes = Uint8Array.from({ length }, (_, i) => (i * 37 + length) & 0xff);
    assert.equal(bytesToBase64(bytes), Buffer.from(bytes).toString('base64'));
  }
  const samples = new Float32Array([0, 1, -1, 0.5]);
  const decoded = Buffer.from(float32ToBase64(samples), 'base64');
  assert.equal(decoded.readFloatLE(4), 1);
  assert.equal(decoded.readFloatLE(8), -1);
  assert.equal(float32ToBase64([0.5]), float32ToBase64(new Float32Array([0.5])));
});
