// Expo config plugin for building @xybrid/react-native from LOCAL repo source
// in this monorepo. Applied via app.json `plugins`. Because `expo prebuild`
// regenerates android/, these edits can't live in the generated files — they
// have to be reapplied on every prebuild, which is what a config plugin is for.
//
// It makes two edits:
//
// 1. Sets `android.minSdkVersion=29` in gradle.properties. The Kotlin SDK
// requires Android 10 (API 29), and an app below that fails the manifest
// merge. Expo feeds this property into the app's minSdk (it is what
// expo-build-properties' `android.minSdkVersion` writes); doing it here keeps
// the example free of an extra dependency. A higher value is left alone.
//
// 2. Inserts mavenLocal() first in the android repositories so the example
// resolves `ai.xybrid:xybrid-kotlin` from the locally published AAR (built
// from working-tree source) ahead of Maven Central — the example tracks
// current source instead of the last release. A composite/project build can't
// be used here because bindings/kotlin pins Gradle 8.13 / Kotlin 1.9.22 while
// the Expo app is Gradle 9.x / Kotlin 2.x — incompatible in one build.
// Publishing a local AAR keeps the kotlin module on its own toolchain.
//
// Dev flow:
//   bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar
//   (stage its jni/* into bindings/kotlin/libs/ — see bindings/kotlin/README.md)
//   (cd bindings/kotlin && ./gradlew publishToMavenLocal)
//   (cd bindings/react-native/example && npx expo run:android)
//
// NOTE: no keepDebugSymbols workaround is needed — as of #287 the bolt .so is
// a clean, 16KB-aligned linker output (no patchelf rewrite), so it survives
// AGP's stripDebugSymbols. If a future change reintroduces a strip-fragile
// .so, the symptom is a launch-time `dlopen failed: empty/missing
// DT_HASH/DT_GNU_HASH` and the fix belongs in the bolt Android build, not here.
const { withGradleProperties, withProjectBuildGradle } = require('@expo/config-plugins');

// Android 10, the floor of ai.xybrid:xybrid-kotlin (and so @xybrid/react-native).
const XYBRID_MIN_SDK = 29;
const MIN_SDK_PROPERTY = 'android.minSdkVersion';

function withXybridMinSdk(config) {
  return withGradleProperties(config, (cfg) => {
    const existing = cfg.modResults.find(
      (item) => item.type === 'property' && item.key === MIN_SDK_PROPERTY,
    );
    if (!existing) {
      cfg.modResults.push({
        type: 'property',
        key: MIN_SDK_PROPERTY,
        value: String(XYBRID_MIN_SDK),
      });
    } else if (!(Number.parseInt(existing.value, 10) >= XYBRID_MIN_SDK)) {
      existing.value = String(XYBRID_MIN_SDK);
    }
    return cfg;
  });
}

module.exports = function withXybridDevAndroid(config) {
  config = withXybridMinSdk(config);
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
};
