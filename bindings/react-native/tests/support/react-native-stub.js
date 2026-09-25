// Stands in for `react-native` when the compiled facade runs on Node. The
// TurboModule forwards every call to whichever fake the current test
// installed on `globalThis.__xybridNative`.
'use strict';

const nativeModule = new Proxy(
  {},
  {
    get(_target, name) {
      const fake = globalThis.__xybridNative;
      if (!fake) throw new Error('no fake native module installed');
      const method = fake[name];
      if (typeof method !== 'function') {
        throw new Error(`fake native module has no ${String(name)}()`);
      }
      return method.bind(fake);
    },
  },
);

const appStateListeners = new Set();

module.exports = {
  TurboModuleRegistry: {
    getEnforcing: (name) => {
      if (name !== 'RNXybrid') throw new Error(`unexpected module ${name}`);
      return nativeModule;
    },
  },
  AppState: {
    addEventListener(type, listener) {
      const entry = { type, listener };
      appStateListeners.add(entry);
      return { remove: () => appStateListeners.delete(entry) };
    },
    __emit(type) {
      for (const entry of appStateListeners) if (entry.type === type) entry.listener();
    },
    __listenerCount: () => appStateListeners.size,
  },
};
