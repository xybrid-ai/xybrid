const test = require('node:test');
const assert = require('node:assert/strict');

const { decodeModelInfo } = require(process.env.XYBRID_MODEL_INFO_MODULE);

const base = {
  modelId: 'fixture-model',
  version: '1.2.3',
  isLoaded: true,
  supportsStreaming: true,
  supportsTokenStreaming: false,
  isLlm: false,
};

for (const outputType of ['text', 'audio', 'embedding', 'unknown']) {
  test(`decodes every field for the ${outputType} output type`, () => {
    const native = { ...base, outputType };
    assert.deepEqual(decodeModelInfo(native), native);
  });
}

test('rejects unknown output types instead of inferring a capability', () => {
  assert.throws(
    () => decodeModelInfo({ ...base, outputType: 'video' }),
    /invalid outputType/,
  );
});

test('rejects incomplete native snapshots', () => {
  const { supportsStreaming: _, ...incomplete } = base;
  assert.throws(
    () => decodeModelInfo({ ...incomplete, outputType: 'text' }),
    /invalid supportsStreaming/,
  );
});

test('Model keeps its native handle distinct from canonical model metadata', async () => {
  let receivedHandle;
  global.__XYBRID_NATIVE__ = {
    modelInfo: async (handle) => {
      receivedHandle = handle;
      return { ...base, outputType: 'text' };
    },
  };
  const { Model } = require(process.env.XYBRID_MODEL_FACADE_MODULE);
  const model = new Model('opaque-native-handle');

  assert.equal(model.nativeHandle, 'opaque-native-handle');
  assert.equal(model.id, 'opaque-native-handle');
  assert.equal((await model.info()).modelId, 'fixture-model');
  assert.equal(receivedHandle, 'opaque-native-handle');
});

test('Model.info preserves the native xybrid_handle rejection', async () => {
  const nativeError = Object.assign(new Error('Unknown model handle'), {
    code: 'xybrid_handle',
  });
  global.__XYBRID_NATIVE__.modelInfo = async () => {
    throw nativeError;
  };
  const { Model } = require(process.env.XYBRID_MODEL_FACADE_MODULE);

  await assert.rejects(new Model('released-handle').info(), (error) => {
    assert.equal(error, nativeError);
    assert.equal(error.code, 'xybrid_handle');
    return true;
  });
});
