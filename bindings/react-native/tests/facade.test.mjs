// The public TS facade against a recording fake of the TurboModule: which
// native calls each API makes, in what order, and how handles, stop buttons
// and streams are cleaned up.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { installFakeNative, loadLibrary, loadReactNativeStub } from './support/fake-native.mjs';

const lib = loadLibrary();
const {
  ConversationContext,
  Envelope,
  GenerationConfigs,
  Model,
  ModelLoader,
  Pipeline,
  Xybrid,
  isRetryable,
  isXybridError,
  jsonSchemaToGbnf,
} = lib;

const metrics = { totalMs: 3, stageLatenciesMs: [] };
const wireResult = (text = 'ok') => ({
  envelope: { kind: 'text', text, metadata: {} },
  outputType: 'text',
  modelId: 'qwen',
  latencyMs: 3,
  executionTarget: 'local',
  metrics,
  toolCalls: [],
});
const names = (calls) => calls.map(([name]) => name);

class Signal {
  aborted = false;
  listeners = new Set();
  addEventListener(_type, listener) {
    this.listeners.add(listener);
  }
  removeEventListener(_type, listener) {
    this.listeners.delete(listener);
  }
  abort() {
    this.aborted = true;
    for (const listener of this.listeners) listener();
  }
}

test('initialize forwards options; the SDK needs no init for local use', async () => {
  const calls = installFakeNative();
  await Xybrid.initialize({ apiKey: 'key' });
  assert.deepEqual(calls, [['initialize', { apiKey: 'key' }]]);
  assert.equal(Xybrid.isInitialized, true);

  const loadCalls = installFakeNative({ loadModel: () => 'model:1' });
  await ModelLoader.fromRegistry('qwen').load();
  assert.deepEqual(names(loadCalls), ['loadModel'], 'load() must not call initialize()');
});

test('model sources cross as { kind, value, revision? }', async () => {
  const calls = installFakeNative({ loadModel: () => 'model:1' });
  await ModelLoader.fromRegistry('a').load();
  await ModelLoader.fromRegistrySpeculative('b').load();
  await ModelLoader.fromBundle('/x.xyb').load();
  await ModelLoader.fromDirectory('/dir').load();
  await ModelLoader.fromHuggingFace('org/repo', { revision: 'v2' }).load();
  await ModelLoader.fromHuggingFace('org/repo').load();
  await ModelLoader.fromModelFile('/m.gguf').load();
  assert.deepEqual(
    calls.map(([, source]) => source),
    [
      { kind: 'registry', value: 'a' },
      { kind: 'registrySpeculative', value: 'b' },
      { kind: 'bundle', value: '/x.xyb' },
      { kind: 'directory', value: '/dir' },
      { kind: 'huggingFace', value: 'org/repo', revision: 'v2' },
      { kind: 'huggingFace', value: 'org/repo' },
      { kind: 'modelFile', value: '/m.gguf' },
    ],
  );
  assert.equal(Xybrid.model('z').source.kind, 'registry');
});

test('willSpeculate and download only apply to registry sources', async () => {
  const calls = installFakeNative({ willSpeculate: () => true, startDownload: () => 'download:1' });
  assert.equal(await ModelLoader.fromRegistry('a').willSpeculate(), false);
  assert.equal(await ModelLoader.fromRegistrySpeculative('a').willSpeculate(), true);
  assert.equal(await ModelLoader.fromDirectory('/d').download(), null);
  const download = await ModelLoader.fromRegistry('a').download({ platform: 'ios' });
  assert.equal(download.handle, 'download:1');
  assert.deepEqual(calls, [
    ['willSpeculate', 'a'],
    ['startDownload', 'a', 'ios'],
  ]);
});

test('run sends the wire envelope and options and decorates the result', async () => {
  const calls = installFakeNative({ run: () => wireResult('bonjour') });
  const model = new Model('model:1');
  const result = await model.run(Envelope.text('hi', { voiceId: 'v' }), {
    generationConfig: GenerationConfigs.greedy({ maxTokens: 8 }),
    correlationId: 'req',
  });
  assert.equal(result.text, 'bonjour');
  assert.deepEqual(calls, [
    [
      'run',
      'model:1',
      { kind: 'text', text: 'hi', metadata: { voice_id: 'v' } },
      {
        generationConfig: { temperature: 0, topP: 1, topK: 0, stopSequences: [], maxTokens: 8 },
        correlationId: 'req',
      },
    ],
  ]);
});

test('a signal becomes a native stop button, released after the run', async () => {
  const signal = new Signal();
  let release;
  const calls = installFakeNative({
    run: () => new Promise((resolve) => (release = resolve)),
  });
  const pending = new Model('model:1').run(Envelope.text('hi'), { signal });
  await new Promise((resolve) => setImmediate(resolve));
  signal.abort();
  await new Promise((resolve) => setImmediate(resolve));
  release(wireResult());
  await pending;
  assert.deepEqual(names(calls), ['createCancelToken', 'run', 'cancel', 'dispose']);
  const token = calls[1][3].cancel;
  assert.match(token, /^cancel:/);
  assert.deepEqual(calls[2], ['cancel', token]);
  assert.deepEqual(calls[3], ['dispose', token]);
  assert.equal(signal.listeners.size, 0);
});

test('an already-aborted signal cancels before the run starts', async () => {
  const signal = new Signal();
  signal.abort();
  const calls = installFakeNative({ run: () => wireResult() });
  await new Model('model:1').run(Envelope.text('hi'), { signal });
  assert.deepEqual(names(calls), ['createCancelToken', 'cancel', 'run', 'dispose']);
});

test('a context crosses as its handle', async () => {
  const calls = installFakeNative({ createContext: () => 'context:7', run: () => wireResult() });
  const chat = await ConversationContext.create('chat-1');
  await chat.setSystem('Be brief.');
  await chat.push(Envelope.user('hello'));
  await new Model('model:1').run(Envelope.user('again'), { context: chat });
  assert.deepEqual(calls[0], ['createContext', 'chat-1']);
  assert.deepEqual(calls[1], [
    'contextSetSystem',
    'context:7',
    { kind: 'text', text: 'Be brief.', metadata: { 'xybrid.role': 'system' } },
  ]);
  assert.deepEqual(calls[2][2].metadata, { 'xybrid.role': 'user' });
  assert.equal(calls[3][3].context, 'context:7');
});

test('run forwards cloud destination without changing explicit false fallback', async () => {
  const calls = installFakeNative({ run: () => wireResult() });
  await new Model('model:1').run(Envelope.text('hi'), {
    fallbackToCloud: false,
    cloudProvider: '',
    cloudModel: 'gpt-4o-mini',
    cloudGatewayUrl: 'https://api.xybrid.dev/v1',
  });
  assert.deepEqual(calls[0][3], {
    fallbackToCloud: false,
    cloudProvider: '',
    cloudModel: 'gpt-4o-mini',
    cloudGatewayUrl: 'https://api.xybrid.dev/v1',
  });
});

test('streaming forwards cloud destination through the same options wire', async () => {
  const calls = installFakeNative({
    streamStart: () => 'stream:cloud',
    streamNext: () => ({ kind: 'complete', result: wireResult() }),
  });
  const generator = new Model('model:1').runStreaming(Envelope.text('hi'), {
    fallbackToCloud: true,
    cloudProvider: 'openai',
  });
  await generator.next();
  assert.deepEqual(calls[0][3], { fallbackToCloud: true, cloudProvider: 'openai' });
});

test('streaming yields tokens, returns the result and disposes the stream', async () => {
  const events = [
    { kind: 'token', token: { token: 'Hel', index: 0, cumulativeText: 'Hel', toolCalls: [] } },
    { kind: 'token', token: { token: 'lo', index: 1, cumulativeText: 'Hello', toolCalls: [] } },
    { kind: 'complete', result: wireResult('Hello') },
  ];
  const calls = installFakeNative({ streamStart: () => 'stream:1', streamNext: () => events.shift() });
  const generator = new Model('model:1').runStreaming(Envelope.text('hi'));
  const tokens = [];
  let next = await generator.next();
  while (!next.done) {
    tokens.push(next.value.token);
    next = await generator.next();
  }
  assert.deepEqual(tokens, ['Hel', 'lo']);
  assert.equal(next.value.text, 'Hello');
  assert.deepEqual(names(calls), ['streamStart', 'streamNext', 'streamNext', 'streamNext', 'dispose']);
  assert.deepEqual(calls.at(-1), ['dispose', 'stream:1']);
});

test('breaking out of a stream disposes it (which aborts generation)', async () => {
  const calls = installFakeNative({
    streamStart: () => 'stream:2',
    streamNext: () => ({ kind: 'token', token: { token: 'x', index: 0, cumulativeText: 'x', toolCalls: [] } }),
  });
  for await (const token of new Model('model:1').runStreaming(Envelope.text('hi'))) {
    assert.equal(token.token, 'x');
    break;
  }
  assert.deepEqual(names(calls), ['streamStart', 'streamNext', 'dispose']);
});

test('a failing pull still disposes the stream and rethrows', async () => {
  const failure = Object.assign(new Error('boom'), { code: 'xybrid_inference_error' });
  const calls = installFakeNative({
    streamStart: () => 'stream:3',
    streamNext: () => {
      throw failure;
    },
  });
  await assert.rejects(async () => {
    for await (const _ of new Model('model:1').runStreaming(Envelope.text('hi'))) {
      // unreachable
    }
  }, failure);
  assert.deepEqual(calls.at(-1), ['dispose', 'stream:3']);
});

test('model info decodes the default generation config', async () => {
  installFakeNative({
    modelInfo: () => ({
      modelId: 'qwen',
      version: '1',
      outputType: 'text',
      isLlm: true,
      supportsStreaming: true,
      supportsTokenStreaming: true,
      supportsToolCalling: null,
      hasVoices: false,
      defaultGenerationConfig: {
        maxTokens: 256,
        stopSequences: [],
        tools: [{ name: 'f', description: 'd', parametersJson: '{}' }],
      },
    }),
  });
  const info = await new Model('model:1').info();
  assert.equal(info.supportsToolCalling, null);
  assert.deepEqual(info.defaultGenerationConfig.tools, [{ name: 'f', description: 'd', parameters: '{}' }]);
});

test('download progress polls until a terminal state and yields changes only', async () => {
  const statuses = [
    { state: 'downloading', progress: 0, downloadedBytes: 0 },
    { state: 'downloading', progress: 0, downloadedBytes: 0 },
    { state: 'downloading', progress: 0.5, downloadedBytes: 5, totalBytes: 10 },
    { state: 'ready', progress: 1, downloadedBytes: 10, totalBytes: 10 },
  ];
  installFakeNative({ startDownload: () => 'download:9', downloadHandleStatus: () => statuses.shift() });
  const download = await ModelLoader.fromRegistry('a').download();
  const seen = [];
  for await (const status of download.progress({ intervalMs: 1 })) seen.push(status.progress);
  assert.deepEqual(seen, [0, 0.5, 1]);
});

test('live ASR sessions feed base64 Float32 and pull partials until null', async () => {
  const partials = [
    { text: 'he', isStable: false, chunkSequence: 0, audioDurationMs: 100 },
    { text: 'hello', isStable: true, chunkSequence: 1, audioDurationMs: 200 },
    null,
  ];
  const calls = installFakeNative({
    openStreamingSession: () => 'session:1',
    sessionNextPartial: () => partials.shift(),
    sessionFlush: () => 'hello',
  });
  const session = await new Model('model:1').stream({ language: 'en' });
  await session.feed(new Float32Array([0.25]));
  const texts = [];
  for await (const partial of session.partials()) texts.push(partial.text);
  assert.equal(await session.flush(), 'hello');
  await session.release();
  assert.deepEqual(texts, ['he', 'hello']);
  assert.deepEqual(calls[0], ['openStreamingSession', 'model:1', { language: 'en' }]);
  assert.equal(Buffer.from(calls[1][2], 'base64').readFloatLE(0), 0.25);
  assert.deepEqual(calls.at(-1), ['dispose', 'session:1']);
});

test('pipelines load from any source and only forward correlationId', async () => {
  const calls = installFakeNative({
    loadPipeline: () => 'pipeline:1',
    runPipeline: () => ({
      envelope: { kind: 'text', text: 'done', metadata: {} },
      outputType: 'text',
      latencyMs: 2,
      stages: [],
    }),
  });
  const pipeline = await Pipeline.fromYaml('name: x');
  const result = await pipeline.run(Envelope.text('go'), { correlationId: 'c' });
  await (await Pipeline.fromFile('file:///p.yaml')).run(Envelope.text('go'));
  assert.equal(result.text, 'done');
  assert.deepEqual(calls[0], ['loadPipeline', { kind: 'yaml', value: 'name: x' }]);
  assert.deepEqual(calls[1][3], { correlationId: 'c' });
  assert.equal(calls[3][3], null);
});

test('tool results round-trip through the native envelope builder', async () => {
  const calls = installFakeNative({
    toolResultsEnvelope: () => ({
      kind: 'text',
      text: 'continuation',
      metadata: { 'xybrid.role': 'user', 'xybrid.tool_results': '[]' },
    }),
  });
  const envelope = await Envelope.toolResults('weather?', '<tool_call>…', [
    { callId: '1', name: 'weather', content: { tempC: 21 } },
  ]);
  assert.equal(envelope.role, 'user');
  assert.equal(envelope.metadata['xybrid.tool_results'], '[]');
  assert.deepEqual(calls[0][3], [{ callId: '1', name: 'weather', contentJson: '{"tempC":21}' }]);
});

test('userMessage builds a user multipart and rejects non-images', () => {
  const image = Envelope.image('iVBO', 'jpg');
  assert.equal(image.format, 'jpeg');
  const message = Envelope.userMessage('what is this?', [image]);
  assert.equal(message.kind, 'multipart');
  assert.equal(message.role, 'user');
  assert.throws(() => Envelope.userMessage('x', [Envelope.text('nope')]), /only image envelopes/);
});

test('jsonSchemaToGbnf serializes object schemas', async () => {
  const calls = installFakeNative({ jsonSchemaToGbnf: () => 'root ::= ...' });
  assert.equal(await jsonSchemaToGbnf({ type: 'object' }), 'root ::= ...');
  await jsonSchemaToGbnf('{"type":"string"}');
  assert.deepEqual(calls.map(([, schema]) => schema), ['{"type":"object"}', '{"type":"string"}']);
});

test('releaseMemoryOnWarning follows AppState memory warnings', async () => {
  const calls = installFakeNative({ releaseMemory: () => 2 });
  const appState = loadReactNativeStub().AppState;
  const unsubscribe = Xybrid.releaseMemoryOnWarning();
  appState.__emit('memoryWarning');
  appState.__emit('change');
  unsubscribe();
  appState.__emit('memoryWarning');
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(names(calls), ['releaseMemory']);
  assert.equal(appState.__listenerCount(), 0);
});

test('cache helpers use the Swift/Kotlin names', async () => {
  const calls = installFakeNative();
  await Xybrid.modelCacheStatus();
  await Xybrid.modelCacheEntries();
  await Xybrid.hasCachedModelData('a');
  await Xybrid.cachedModelPath('a');
  await Xybrid.extractedModelIds();
  await Xybrid.removeCachedModel('a');
  await Xybrid.clearModelCache();
  assert.deepEqual(names(calls), [
    'cacheStatus',
    'cacheEntries',
    'cacheIsModelCached',
    'cacheModelPath',
    'cacheExtractedModelIds',
    'cacheRemoveModel',
    'cacheClear',
  ]);
});

test('error helpers recognise native rejections', () => {
  const offline = Object.assign(new Error('no network'), { code: 'xybrid_offline' });
  const handle = Object.assign(new Error('gone'), { code: 'xybrid_handle' });
  assert.equal(isXybridError(offline), true);
  assert.equal(isRetryable(offline), true);
  assert.equal(isRetryable(handle), false);
  assert.equal(isXybridError(Object.assign(new Error('x'), { code: 'EFOO' })), false);
  assert.equal(isXybridError('xybrid_offline'), false);
});

test('every public entry point named in parity.json exists', async () => {
  const { readFile } = await import('node:fs/promises');
  const parity = JSON.parse(await readFile(new URL('../parity.json', import.meta.url), 'utf8'));
  const apis = new Set(
    [...Object.values(parity.functions), ...Object.values(parity.methods)]
      .map((entry) => entry.api)
      .filter(Boolean),
  );
  for (const api of apis) {
    if (api.includes('#')) {
      const [owner, method] = api.split('#');
      assert.equal(typeof lib[owner]?.prototype?.[method], 'function', `${api} is not a method`);
    } else {
      const value = api.split('.').reduce((object, key) => object?.[key], lib);
      assert.equal(typeof value, 'function', `${api} is not exported`);
    }
  }
});
