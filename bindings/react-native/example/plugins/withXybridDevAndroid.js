// Expo config plugin for building react-native-xybrid from LOCAL repo source
// in this monorepo. Applied via app.json `plugins`. Because `expo prebuild`
// regenerates android/, this edit can't live in the generated files — it has
// to be reapplied on every prebuild, which is what a config plugin is for.
//
// It inserts mavenLocal() first in the android repositories so the example
// resolves `ai.xybrid:xybrid-kotlin` from the locally published AAR (built
// from working-tree source) ahead of Maven Central — the example tracks
// current source instead of the last release. A composite/project build can't
// be used here because bindings/kotlin pins Gradle 8.13 / Kotlin 1.9.22 while
// the Expo app is Gradle 9.x / Kotlin 2.x — incompatible in one build.
// Publishing a local AAR keeps the kotlin module on its own toolchain.
//
// Dev flow:
//   cargo xtask build-android --release
//   (cd bindings/kotlin && ./gradlew publishToMavenLocal)
//   (cd bindings/react-native/example && npx expo run:android)
//
// NOTE: no keepDebugSymbols workaround is needed — as of #287 the bolt .so is
// a clean, 16KB-aligned linker output (no patchelf rewrite), so it survives
// AGP's stripDebugSymbols. If a future change reintroduces a strip-fragile
// .so, the symptom is a launch-time `dlopen failed: empty/missing
// DT_HASH/DT_GNU_HASH` and the fix belongs in the bolt Android build, not here.
const { withProjectBuildGradle } = require('@expo/config-plugins');

module.exports = function withXybridDevAndroid(config) {
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
