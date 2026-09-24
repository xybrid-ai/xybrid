// `pod install` runs ios/xybrid_natives.rb to stage the XCFramework. With a
// local override (XYBRID_XCFRAMEWORK_PATH), a rebuild must replace the staged
// copy even when only files deep inside the framework changed — otherwise the
// app links stale native code against updated Swift glue. Needs Ruby, which
// every CocoaPods machine has; skipped where there is none.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const natives = fileURLToPath(new URL('../ios/xybrid_natives.rb', import.meta.url));
// `ruby` can be a version-manager shim with no version selected; fall back to
// the system Ruby CocoaPods uses on macOS.
const ruby = ['ruby', '/usr/bin/ruby'].find((bin) => spawnSync(bin, ['-e', 'exit 0']).status === 0);

function prepare(root, override) {
  const script = `require ${JSON.stringify(natives)}; XybridNatives.prepare!(ARGV[0])`;
  const run = spawnSync(ruby, ['-e', script, root], {
    encoding: 'utf8',
    env: { ...process.env, XYBRID_XCFRAMEWORK_PATH: override },
  });
  assert.equal(run.status, 0, run.stderr);
}

test('a local XCFramework rebuilt in place replaces the staged copy', { skip: !ruby && 'no Ruby' }, () => {
  const root = mkdtempSync(join(tmpdir(), 'xybrid-natives-'));
  try {
    writeFileSync(join(root, 'package.json'), JSON.stringify({ version: '0.0.0' }));
    mkdirSync(join(root, 'ios', 'XybridSwift'), { recursive: true });
    for (const name of ['Xybrid.swift', 'xybrid_bolt.swift']) {
      writeFileSync(join(root, 'ios', 'XybridSwift', name), '');
    }
    const framework = join(root, 'build', 'XybridFFI.xcframework');
    const library = join(framework, 'ios-arm64', 'libxybrid_bolt.a');
    mkdirSync(dirname(library), { recursive: true });
    writeFileSync(library, 'first build');
    const staged = join(root, 'ios', 'Frameworks', 'XybridFFI.xcframework', 'ios-arm64', 'libxybrid_bolt.a');

    prepare(root, framework);
    assert.equal(readFileSync(staged, 'utf8'), 'first build');

    // Unchanged source: the staged copy is reused, not copied again.
    const copiedAt = statSync(staged).mtimeMs;
    prepare(root, framework);
    assert.equal(statSync(staged).mtimeMs, copiedAt);

    // Rewriting a file inside the framework leaves the framework
    // directory's own mtime untouched.
    const frameworkMtime = statSync(framework).mtimeMs;
    writeFileSync(library, 'second build');
    assert.equal(statSync(framework).mtimeMs, frameworkMtime);

    prepare(root, framework);
    assert.equal(readFileSync(staged, 'utf8'), 'second build');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
