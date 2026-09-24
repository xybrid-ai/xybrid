// Reads the exported surface of crates/xybrid-bolt/src/lib.rs — the source
// every other binding is generated from — without a Rust toolchain: free
// `#[export]` functions, methods of `#[export] impl` blocks, and the fields /
// variants of `#[data]` records. The file follows a regular layout (rustfmt'd,
// one item per attribute), which is what makes a line-level reader enough.

import { readFileSync } from 'node:fs';

export function readBoltSurface(path) {
  const lines = readFileSync(path, 'utf8').split('\n');
  const functions = [];
  const methods = [];
  const records = {};
  let errors = [];

  let pendingExport = false;
  let pendingData = false;
  let exportImpl = null; // type name while inside `#[export] impl Type {`
  let implDepth = 0;
  let record = null; // { name, kind, members, depth }

  const depthDelta = (line) =>
    (line.match(/{/g) ?? []).length - (line.match(/}/g) ?? []).length;

  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith('//')) continue;

    if (exportImpl) {
      if (implDepth === 1) {
        const fn = line.match(/^pub fn (\w+)/);
        if (fn) methods.push(`${exportImpl}::${fn[1]}`);
      }
      implDepth += depthDelta(line);
      if (implDepth === 0) exportImpl = null;
      continue;
    }

    if (record) {
      if (record.depth === 1) {
        const field = line.match(/^pub (\w+):/);
        const variant = line.match(/^([A-Z]\w*)\s*[,{(]?/);
        if (record.kind === 'struct' && field) record.members.push(field[1]);
        if (record.kind === 'enum' && variant && !line.startsWith('#')) record.members.push(variant[1]);
      }
      record.depth += depthDelta(line);
      if (record.depth === 0) {
        records[record.name] = { kind: record.kind, members: record.members };
        record = null;
      }
      continue;
    }

    if (line === '#[export]') {
      pendingExport = true;
      continue;
    }
    if (line === '#[data]' || line === '#[error]') {
      pendingData = true;
      continue;
    }
    if (line.startsWith('#[')) continue; // derives, allows, ffi_stream…

    if (pendingExport) {
      const impl = line.match(/^impl (\w+)\s*{/);
      const fn = line.match(/^pub fn (\w+)/);
      if (impl) {
        exportImpl = impl[1];
        implDepth = 1;
      } else if (fn) {
        functions.push(fn[1]);
      }
      pendingExport = false;
      continue;
    }
    if (pendingData) {
      const item = line.match(/^pub (struct|enum) (\w+)/);
      if (item) {
        record = { name: item[2], kind: item[1], members: [], depth: depthDelta(line) };
        if (record.depth === 0) {
          records[record.name] = { kind: record.kind, members: [] };
          record = null;
        }
      }
      pendingData = false;
    }
  }
  // `#[error]` enums are read like records; split the error enum out.
  if (records.XybridError) {
    errors = records.XybridError.members;
    delete records.XybridError;
  }
  return { functions, methods, records, errors };
}
