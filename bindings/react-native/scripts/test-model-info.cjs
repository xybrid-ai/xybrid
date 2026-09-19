#!/usr/bin/env node

const { mkdirSync, mkdtempSync, rmSync, writeFileSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const { spawnSync } = require('node:child_process');

const root = join(__dirname, '..');
const outputDir = mkdtempSync(join(tmpdir(), 'xybrid-rn-model-info-'));
const stubModules = join(outputDir, 'node_modules');

try {
  const tsc = process.platform === 'win32'
    ? join(root, 'node_modules', '.bin', 'tsc.cmd')
    : join(root, 'node_modules', '.bin', 'tsc');
  const compile = spawnSync(
    tsc,
    [
      'src/model-info.ts',
      'src/index.ts',
      'src/NativeXybrid.ts',
      'src/presets.ts',
      'tests/model-info-types.ts',
      '--target', 'ES2020',
      '--module', 'commonjs',
      '--moduleResolution', 'node',
      '--strict',
      '--esModuleInterop',
      '--skipLibCheck',
      '--outDir', outputDir,
    ],
    { cwd: root, encoding: 'utf8' },
  );
  process.stdout.write(compile.stdout ?? '');
  process.stderr.write(compile.stderr ?? '');
  if (compile.error) throw compile.error;
  if (compile.status !== 0) {
    throw new Error(`Model-info test compilation failed (exit ${compile.status})`);
  }

  const reactNativeStub = join(stubModules, 'react-native');
  mkdirSync(reactNativeStub, { recursive: true });
  writeFileSync(
    join(reactNativeStub, 'index.js'),
    'exports.TurboModuleRegistry = { getEnforcing: () => global.__XYBRID_NATIVE__ };\n',
  );

  const test = spawnSync(
    process.execPath,
    ['--test', join(root, 'tests', 'model-info.test.cjs')],
    {
      cwd: root,
      encoding: 'utf8',
      env: {
        ...process.env,
        XYBRID_MODEL_INFO_MODULE: join(outputDir, 'src', 'model-info.js'),
        XYBRID_MODEL_FACADE_MODULE: join(outputDir, 'src', 'index.js'),
        NODE_PATH: stubModules,
      },
    },
  );
  process.stdout.write(test.stdout ?? '');
  process.stderr.write(test.stderr ?? '');
  if (test.error) throw test.error;
  process.exitCode = test.status ?? 1;
} finally {
  rmSync(outputDir, { recursive: true, force: true });
}
