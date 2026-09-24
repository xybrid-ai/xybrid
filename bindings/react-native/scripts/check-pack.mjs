// Checks what `npm pack` would publish, before anything is published: every
// file a consumer's Metro, Jest, TypeScript, CocoaPods and Gradle need is in
// the tarball, and nothing that must not ship is (the XCFramework is fetched
// at `pod install`; tests, the example and local build state stay home).
//
// Usage: node scripts/check-pack.mjs   (runs prepack, like the real publish)

import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const [report] = JSON.parse(
  execFileSync('npm', ['pack', '--dry-run', '--json'], { cwd: root, encoding: 'utf8' }),
);
const files = new Set(report.files.map((file) => file.path));

const required = [
  'package.json',
  'README.md',
  'LICENSE',
  'react-native-xybrid.podspec',
  'src/index.ts',
  'src/NativeXybrid.ts',
  'lib/commonjs/index.js',
  'lib/module/index.js',
  'lib/typescript/index.d.ts',
  'ios/XybridModule.mm',
  'ios/XybridModuleImpl.swift',
  'ios/XybridCodec.swift',
  'ios/XybridHandles.swift',
  'ios/xybrid_natives.rb',
  'ios/XybridSwift/Xybrid.swift',
  'ios/XybridSwift/xybrid_bolt.swift',
  'android/build.gradle',
  'android/src/main/AndroidManifest.xml',
  'android/src/main/java/ai/xybrid/reactnative/XybridModule.kt',
  'android/src/main/java/ai/xybrid/reactnative/XybridPackage.kt',
];
const forbidden = [/^ios\/Frameworks\//, /^example\//, /^tests\//, /node_modules\//, /\.xybrid-natives$/, /^android\/build\//];

const missing = required.filter((path) => !files.has(path));
const leaked = [...files].filter((path) => forbidden.some((pattern) => pattern.test(path)));

console.log(`${report.name}@${report.version}: ${files.size} files, ${(report.size / 1024).toFixed(0)} KiB packed`);
if (missing.length || leaked.length) {
  for (const path of missing) console.error(`missing from the tarball: ${path}`);
  for (const path of leaked) console.error(`must not ship: ${path}`);
  process.exit(1);
}
