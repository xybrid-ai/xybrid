// The drift guard. React Native is hand-bridged, so every capability the
// generated bindings inherit from crates/xybrid-bolt has to be wired here by
// hand — and historically, new ones silently weren't. This test reads bolt's
// exported surface and fails until each new function, method, record field,
// enum variant or error variant is mapped in parity.json (or excluded with a
// reason), and checks every mapping still points at something real.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { readBoltSurface } from './support/bolt-surface.mjs';
import { loadTsSurface } from './support/ts-surface.mjs';

const bolt = readBoltSurface(new URL('../../../crates/xybrid-bolt/src/lib.rs', import.meta.url));
const parity = JSON.parse(readFileSync(new URL('../parity.json', import.meta.url), 'utf8'));
const tsSurface = loadTsSurface();
const errorsSource = readFileSync(new URL('../src/errors.ts', import.meta.url), 'utf8');

const HOW_TO =
  'Wire it through NativeXybrid.ts + both shims and map it in bindings/react-native/parity.json, ' +
  'or add it there as { "excluded": "<reason>" }.';

function checkEntry(where, entry) {
  assert.ok(entry && typeof entry === 'object', `${where}: malformed parity entry`);
  if ('excluded' in entry || 'native' in entry) {
    const reason = entry.excluded ?? entry.native;
    assert.ok(typeof reason === 'string' && reason.length > 10, `${where}: give a real reason`);
    return;
  }
  assert.ok(tsSurface.specMethods.has(entry.spec), `${where}: spec method '${entry.spec}' is not in NativeXybrid.ts`);
  assert.equal(typeof entry.api, 'string', `${where}: name the public TS entry point`);
}

test('the bolt reader still understands lib.rs', () => {
  // Guards the guard: if the reader silently finds nothing, everything passes.
  assert.ok(bolt.functions.length >= 25, `only ${bolt.functions.length} functions found`);
  assert.ok(bolt.methods.length >= 80, `only ${bolt.methods.length} methods found`);
  assert.ok(Object.keys(bolt.records).length >= 25, 'too few #[data] records found');
  assert.ok(bolt.errors.length >= 20, 'too few error variants found');
});

test('every exported bolt function is mapped', () => {
  for (const name of bolt.functions) {
    assert.ok(name in parity.functions, `bolt exports fn ${name}() but React Native doesn't. ${HOW_TO}`);
    checkEntry(`functions.${name}`, parity.functions[name]);
  }
  for (const name of Object.keys(parity.functions)) {
    assert.ok(bolt.functions.includes(name), `parity.json maps ${name}(), which bolt no longer exports`);
  }
});

test('every exported bolt method is mapped', () => {
  for (const name of bolt.methods) {
    assert.ok(name in parity.methods, `bolt exports ${name}() but React Native doesn't. ${HOW_TO}`);
    checkEntry(`methods.${name}`, parity.methods[name]);
  }
  for (const name of Object.keys(parity.methods)) {
    assert.ok(bolt.methods.includes(name), `parity.json maps ${name}(), which bolt no longer exports`);
  }
});

test('every bolt record field and enum variant reaches a TS type', () => {
  for (const [name, record] of Object.entries(bolt.records)) {
    const entry = parity.records[name];
    assert.ok(entry, `bolt has #[data] ${name} but parity.json doesn't map it. ${HOW_TO}`);
    if ('native' in entry || 'excluded' in entry) continue;

    if (record.kind === 'struct') {
      const properties = tsSurface.propertiesOf(entry.ts);
      assert.ok(properties, `${name}: TS type ${entry.ts} is not exported from src/index.ts`);
      for (const field of record.members) {
        const mapped = entry.fields?.[field];
        assert.ok(mapped, `${name}.${field} is new in bolt; map it in parity.json. ${HOW_TO}`);
        assert.ok(properties.has(mapped), `${name}.${field} maps to ${entry.ts}.${mapped}, which doesn't exist`);
      }
      for (const field of Object.keys(entry.fields ?? {})) {
        assert.ok(record.members.includes(field), `parity.json maps ${name}.${field}, which bolt no longer has`);
      }
    } else {
      const literals = tsSurface.literalsOf(entry.ts);
      assert.ok(literals, `${name}: TS type ${entry.ts} is not exported from src/index.ts`);
      for (const variant of record.members) {
        const mapped = entry.variants?.[variant];
        assert.ok(mapped, `${name}::${variant} is new in bolt; map it in parity.json. ${HOW_TO}`);
        assert.ok(literals.has(mapped), `${name}::${variant} maps to '${mapped}', not a member of ${entry.ts}`);
      }
    }
  }
});

test('every bolt error variant has a stable xybrid_* code', () => {
  for (const variant of bolt.errors) {
    const code = parity.errors[variant];
    assert.ok(code, `XybridError::${variant} is new; give it a code in parity.json, errors.ts and both shims`);
    assert.ok(errorsSource.includes(`'${code}'`), `${code} is missing from src/errors.ts`);
  }
});

test('the native shims map every error code parity.json declares', () => {
  const swift = readFileSync(new URL('../ios/XybridCodec.swift', import.meta.url), 'utf8');
  const kotlin = readFileSync(
    new URL('../android/src/main/java/ai/xybrid/reactnative/XybridCodec.kt', import.meta.url),
    'utf8',
  );
  for (const code of Object.values(parity.errors)) {
    assert.ok(swift.includes(`"${code}"`), `ios/XybridCodec.swift never produces ${code}`);
    assert.ok(kotlin.includes(`"${code}"`), `android XybridCodec.kt never produces ${code}`);
  }
});
