// Both native shims must implement exactly the Codegen spec. The compilers
// enforce most of it (Kotlin overrides the generated abstract class; the
// podspec turns a missing protocol method into an error), but only a real
// Xcode build sees the iOS side — and on iOS a selector mismatch compiles
// yet crashes at the first call, because the TurboModule builds its
// NSInvocation from the protocol's selector. This checks the selectors on
// any machine, before any native build.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { loadTsSurface } from './support/ts-surface.mjs';

const { specMethods } = loadTsSurface();
const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');

/** The ObjC selector Codegen generates for a promise-returning spec method. */
function selector(name, params) {
  if (params.length === 0) return `${name}:reject:`;
  return `${name}:${params.slice(1).map((param) => `${param}:`).join('')}resolve:reject:`;
}

const expected = new Map([...specMethods].map(([name, params]) => [selector(name, params), name]));

test('Codegen selectors are derived as React Native generates them', () => {
  // Spot checks against a real `generate-codegen-artifacts` run.
  assert.equal(selector('initialize', ['options']), 'initialize:resolve:reject:');
  assert.equal(selector('sdkVersion', []), 'sdkVersion:reject:');
  assert.equal(selector('awaitDownload', ['model', 'timeoutMs']), 'awaitDownload:timeoutMs:resolve:reject:');
  assert.ok(expected.size >= 60, 'the spec parser found too few methods');
});

test('XybridModule.mm implements every spec selector and forwards it unchanged', () => {
  const source = read('ios/XybridModule.mm');
  const implemented = new Map();
  for (const match of source.matchAll(/^- \(void\)([\s\S]*?)\n\{\n\s*\[_impl ([^\]]*)\];/gm)) {
    const declared = [...match[1].matchAll(/(\w+):\(/g)].map((part) => `${part[1]}:`).join('');
    const forwarded = [...match[2].matchAll(/(\w+):\w+/g)].map((part) => `${part[1]}:`).join('');
    // Argument-less lifecycle methods (-invalidate) are not spec methods.
    if (declared) implemented.set(declared, forwarded);
  }
  for (const [sel, name] of expected) {
    assert.ok(implemented.has(sel), `XybridModule.mm lacks -${sel} (spec method ${name})`);
    assert.equal(implemented.get(sel), sel, `-${sel} forwards to a different Swift selector`);
  }
  for (const sel of implemented.keys()) {
    assert.ok(expected.has(sel), `XybridModule.mm implements -${sel}, which the spec doesn't declare`);
  }
});

test('XybridModuleImpl.swift exposes every forwarded selector', () => {
  const source = read('ios/XybridModuleImpl.swift');
  const exposed = new Set([...source.matchAll(/@objc\(([\w:]+)\)/g)].map((match) => match[1]));
  exposed.delete('XybridModuleImpl');
  for (const sel of expected.keys()) {
    assert.ok(exposed.has(sel), `XybridModuleImpl.swift has no @objc(${sel})`);
  }
  for (const sel of exposed) {
    assert.ok(expected.has(sel), `XybridModuleImpl.swift exposes @objc(${sel}), which the spec doesn't declare`);
  }
});

test('XybridModule.kt overrides every spec method', () => {
  const source = read('android/src/main/java/ai/xybrid/reactnative/XybridModule.kt');
  const overridden = new Set([...source.matchAll(/override fun (\w+)\(/g)].map((match) => match[1]));
  for (const name of specMethods.keys()) {
    assert.ok(overridden.has(name), `XybridModule.kt does not override ${name}()`);
  }
});
