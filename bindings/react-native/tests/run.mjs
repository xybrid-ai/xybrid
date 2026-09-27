// Runs the Node test suite: compiles src/ to a temp directory as CommonJS,
// puts a stub `react-native` next to it (tests/support/react-native-stub.js),
// then runs every tests/*.test.mjs with XYBRID_RN_LIB pointing at the build.
// Pass test files to run a subset.

import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const tsc = require.resolve('typescript/bin/tsc');

const out = mkdtempSync(join(tmpdir(), 'xybrid-rn-tests-'));
try {
  const lib = join(out, 'lib');
  const build = spawnSync(
    process.execPath,
    [tsc, '-p', join(here, 'tsconfig.json'), '--outDir', lib],
    { stdio: 'inherit' },
  );
  if (build.status !== 0) process.exit(build.status ?? 1);

  const stub = join(out, 'node_modules', 'react-native');
  mkdirSync(stub, { recursive: true });
  copyFileSync(join(here, 'support', 'react-native-stub.js'), join(stub, 'index.js'));

  const requested = process.argv.slice(2);
  const files = requested.length
    ? requested
    : readdirSync(here)
        .filter((name) => name.endsWith('.test.mjs'))
        .map((name) => join(here, name));
  const run = spawnSync(process.execPath, ['--test', ...files], {
    stdio: 'inherit',
    env: { ...process.env, XYBRID_RN_LIB: lib },
  });
  process.exitCode = run.status ?? 1;
} finally {
  rmSync(out, { recursive: true, force: true });
}
