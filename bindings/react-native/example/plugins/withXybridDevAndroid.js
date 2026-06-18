// Expo config plugin for building react-native-xybrid from LOCAL repo source
// in this monorepo. Applied via app.json `plugins`. Because `expo prebuild`
// regenerates android/, these edits can't live in the generated files — they
// have to be reapplied on every prebuild, which is what a config plugin is for.
//
// It does two things:
//
//  1. keepDebugSymbols — stop AGP from stripping the xybrid native libs.
//     AGP's stripDebugSymbols mangles the large 16KB-page Rust .so (breaks
//     DT_GNU_HASH → dlopen "empty/missing DT_HASH ... new hash type from the
//     future"). Shipping them unstripped is what makes them load. (This is a
//     workaround for a .so-packaging bug that affects ALL consumers — see the
//     binding README "Known issues".)
//
//  2. mavenLocal() first — resolve `ai.xybrid:xybrid-kotlin` from the locally
//     published AAR (built from working-tree source) ahead of Maven Central,
//     so the example tracks current source instead of the last release. A
//     composite/project build can't be used here because bindings/kotlin pins
//     Gradle 8.13 / Kotlin 1.9.22 while the Expo app is Gradle 9.3.1 / Kotlin
//     2.x — incompatible in one build. Publishing a local AAR keeps the kotlin
//     module on its own toolchain. Dev flow:
//       cargo xtask build-android --release
//       (cd bindings/kotlin && ./gradlew publishToMavenLocal)
//       (cd bindings/react-native/example && npx expo run:android)
const { withProjectBuildGradle, withAppBuildGradle } = require('@expo/config-plugins');

const KEEP = [
  '**/libxybrid-bolt.so',
  '**/libonnxruntime.so',
  '**/libc++_shared.so',
];

function addMavenLocal(config) {
  return withProjectBuildGradle(config, (cfg) => {
    if (cfg.modResults.language !== 'groovy') return cfg;
    if (cfg.modResults.contents.includes('mavenLocal()')) return cfg;
    // Insert mavenLocal() as the first repository in allprojects so the
    // locally-published xybrid-kotlin AAR wins on identical versions.
    cfg.modResults.contents = cfg.modResults.contents.replace(
      /allprojects\s*\{\s*repositories\s*\{/,
      (m) => `${m}\n      mavenLocal()`,
    );
    return cfg;
  });
}

function addKeepDebugSymbols(config) {
  return withAppBuildGradle(config, (cfg) => {
    if (cfg.modResults.language !== 'groovy') return cfg;
    if (cfg.modResults.contents.includes('keepDebugSymbols += "**/libxybrid-bolt.so"')) {
      return cfg;
    }
    const lines = KEEP.map((g) => `            keepDebugSymbols += "${g}"`).join('\n');
    cfg.modResults.contents = cfg.modResults.contents.replace(
      /packagingOptions\s*\{\s*jniLibs\s*\{/,
      (m) => `${m}\n${lines}`,
    );
    return cfg;
  });
}

module.exports = function withXybridDevAndroid(config) {
  return addKeepDebugSymbols(addMavenLocal(config));
};
