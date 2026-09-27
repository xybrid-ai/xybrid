// A recording fake of the TurboModule. Tests override the methods they care
// about; every call is logged as [name, ...args] for ordering assertions.

import { createRequire } from 'node:module';
import { join } from 'node:path';

const require = createRequire(import.meta.url);

/** The compiled public API (see tests/run.mjs). */
export function loadLibrary() {
  const lib = process.env.XYBRID_RN_LIB;
  if (!lib) throw new Error('run through tests/run.mjs (XYBRID_RN_LIB is unset)');
  return require(join(lib, 'index.js'));
}

/** Pure wire helpers, for the codec tests. */
export function loadWire() {
  return require(join(process.env.XYBRID_RN_LIB, 'wire.js'));
}

/** The stubbed `react-native` module the build resolves. */
export function loadReactNativeStub() {
  return require(join(process.env.XYBRID_RN_LIB, '..', 'node_modules', 'react-native'));
}

let counter = 0;

export function installFakeNative(overrides = {}) {
  const calls = [];
  const fake = new Proxy(
    {},
    {
      get(_target, name) {
        if (name === 'calls') return calls;
        return (...args) => {
          calls.push([name, ...args]);
          if (name in overrides) return Promise.resolve().then(() => overrides[name](...args));
          if (name === 'createCancelToken') return Promise.resolve(`cancel:${++counter}`);
          return Promise.resolve(null);
        };
      },
    },
  );
  globalThis.__xybridNative = fake;
  return calls;
}
