"""Flutter native build script regressions (no model or native compilation).

The pod test runs with Python alone. To also run the real Android Gradle plugin
fixture, set XYBRID_TEST_GRADLE to a Gradle executable and XYBRID_TEST_AGP to its
compatible AGP version, on Linux with ANDROID_HOME pointing to an Android SDK::

    XYBRID_TEST_GRADLE=/path/to/gradle-9.1.0/bin/gradle XYBRID_TEST_AGP=9.0.0 \
      python3 -m unittest discover -s tools/scripts/tests -p test_flutter_build_scripts.py -v

Repeat with Gradle 8.14 and AGP 8.11.1. Fixtures live in temporary directories;
Flutter discovery and Cargo execution are stubbed, while AGP/Gradle are real.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
FLUTTER = REPO / "bindings/flutter"
GRADLE = os.environ.get("XYBRID_TEST_GRADLE")
AGP = os.environ.get("XYBRID_TEST_AGP")


def write(root, path, contents):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents)
    return target


class PodBuildTests(unittest.TestCase):
    def test_build_does_not_dump_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy(FLUTTER / "cargokit/build_pod.sh", root / "build_pod.sh")
            write(root, "run_build_tool.sh", "#!/bin/sh\n[ \"$1\" = build-pod ]\n")
            env = dict(os.environ, PLATFORM_NAME="macosx", ARCHS="arm64",
                       CONFIGURATION="Debug", PODS_TARGET_SRCROOT=str(root),
                       TARGET_TEMP_DIR=str(root), PODS_CONFIGURATION_BUILD_DIR=str(root),
                       PRODUCT_NAME="fixture", SRCROOT=str(root), PODS_ROOT=str(root),
                       OBJROOT=str(root), BUILT_PRODUCTS_DIR=str(root),
                       EXECUTABLE_PATH="fixture", XYBRID_TEST_SECRET="must-not-appear-in-logs")
            result = subprocess.run(["bash", str(root / "build_pod.sh"), "rust", "fixture"],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("XYBRID_TEST_SECRET", result.stdout + result.stderr)
            self.assertNotIn(env["XYBRID_TEST_SECRET"], result.stdout + result.stderr)


@unittest.skipUnless(GRADLE and AGP, "set XYBRID_TEST_GRADLE and XYBRID_TEST_AGP for real Gradle/AGP")
class GradleBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="flutter-gradle-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = root = Path(cls.temp.name)
        write(root, "settings.gradle", """rootProject.name = 'flutter-build-fixture'
include ':app', ':xybrid_flutter'
project(':xybrid_flutter').projectDir = file('plugin/android')
""")
        write(root, "build.gradle", """buildscript {
    repositories { google(); mavenCentral() }
    dependencies { classpath 'com.android.tools.build:gradle:AGP_VERSION' }
}
allprojects { repositories { google(); mavenCentral() } }
project(':xybrid_flutter') { evaluationDependsOn(':app') }
""".replace("AGP_VERSION", AGP))
        write(root, "gradle.properties", "org.gradle.jvmargs=-Xmx512m\norg.gradle.workers.max=1\nandroid.useAndroidX=true\n")
        write(root, "buildSrc/build.gradle", "plugins { id 'groovy' }\n")
        write(root, "buildSrc/src/main/groovy/com/flutter/gradle/FlutterPlugin.groovy", """package com.flutter.gradle
import org.gradle.api.*
class FlutterPlugin implements Plugin<Project> {
    Project project
    void apply(Project project) { this.project = project }
}
""")
        write(root, "buildSrc/src/main/groovy/com/flutter/gradle/FlutterPluginUtils.groovy", """package com.flutter.gradle
import org.gradle.api.Project
class FlutterPluginUtils {
    static List<String> getTargetPlatforms(Project project) {
        ['android-arm', 'android-arm64', 'android-x64']
    }
}
""")
        write(root, "app/build.gradle", """apply plugin: 'com.android.application'
apply plugin: com.flutter.gradle.FlutterPlugin
android {
    namespace = 'fixture.app'
    compileSdk = 34
    ndkVersion = '27.0.12077973'
    if (project.hasProperty('customNdk')) { ndkPath = rootProject.file('custom-ndk').path }
    defaultConfig { minSdk = 23 }
    buildTypes { profile { initWith debug } }
    flavorDimensions += 'edition'
    productFlavors { demo { dimension 'edition' }; full { dimension 'edition' } }
}
""")
        for name in ["android/build.gradle", "cargokit/gradle/plugin.gradle"]:
            target = root / "plugin" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(FLUTTER / name, target)
        with (root / "plugin/android/build.gradle").open("a") as script:
            script.write("""
android {
    buildTypes { profile { initWith debug } }
    flavorDimensions += 'edition'
    productFlavors { demo { dimension 'edition' }; full { dimension 'edition' } }
}
// Check both already-created and lazily-created flavored merge tasks.
tasks.register('verifyWiring') {
    doLast {
        ['DemoDebug', 'FullDebug', 'DemoProfile', 'FullProfile', 'DemoRelease', 'FullRelease'].each { variant ->
            def merge = tasks.named("merge${variant}NativeLibs").get()
            assert merge.taskDependencies.getDependencies(merge)*.name.any { it.startsWith('cargokitCargoBuild') }
            def jni = tasks.named("merge${variant}JniLibFolders").get()
            assert jni.taskDependencies.getDependencies(jni)*.name.contains('copyNdkLibs')
        }
        assert tasks.names.count { it == 'cargokitCargoBuildXybrid_flutterDebug' } == 1
        assert android.packagingOptions.jniLibs.pickFirsts.contains('**/libc++_shared.so')
    }
}
""")
        write(root, "plugin/cargokit/run_build_tool.sh", """#!/bin/sh
set -eu
[ "$1" = build-gradle ]
if [ "${XYBRID_TEST_CARGO_FAIL:-}" = 1 ]; then exit 19; fi
mkdir -p "$CARGOKIT_OUTPUT_DIR"
for abi in arm64-v8a x86_64; do
    mkdir -p "$CARGOKIT_OUTPUT_DIR/$abi"
    printf '%s' "$CARGOKIT_CONFIGURATION" > "$CARGOKIT_OUTPUT_DIR/$abi/libxybrid_flutter.so"
done
python3 -c 'import json,os; from pathlib import Path; Path(os.environ["CARGOKIT_OUTPUT_DIR"], "environment.json").write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith("CARGOKIT_")}))'
""")
        write(root, "plugin/rust/Cargo.toml", '[package]\nname = "fixture"\nversion = "0.1.0"\n')
        # Minimal NDK metadata plus distinguishable payloads suffice for AGP's
        # real NDK locator. No compiler is needed for the tasks under test.
        for ndk, marker in [("sdk/ndk/27.0.12077973", "host-version"), ("custom-ndk", "host-path")]:
            write(root, ndk + "/source.properties", "Pkg.Desc = Android NDK\nPkg.Revision = 27.0.12077973\n")
            for triple in ["aarch64-linux-android", "x86_64-linux-android"]:
                write(root, ndk + "/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/" + triple + "/libc++_shared.so", marker)
        sdk = Path(os.environ["ANDROID_HOME"])
        for directory in ["platforms", "build-tools", "licenses", "platform-tools"]:
            (root / "sdk" / directory).symlink_to(sdk / directory, target_is_directory=True)
        write(root, "local.properties", "sdk.dir=" + str(root / "sdk") + "\n")

    def gradle(self, *tasks, fail=False):
        env = dict(os.environ, XYBRID_TEST_CARGO_FAIL="1" if fail else "0")
        # local.properties deliberately selects an isolated SDK, not ANDROID_HOME.
        env.pop("ANDROID_HOME", None)
        env.pop("ANDROID_SDK_ROOT", None)
        log_dir = os.environ.get("XYBRID_TEST_LOG_DIR")
        if log_dir:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            output = (path / ("gradle-" + AGP + ".log")).open("a+")
        else:
            output = tempfile.TemporaryFile(mode="w+")
        with output:
            start = output.tell()
            result = subprocess.run([GRADLE, "-p", str(self.root), "--no-daemon", "--console=plain", "--stacktrace", *tasks],
                                    env=env, text=True, stdout=output, stderr=subprocess.STDOUT)
            output.seek(start)
            result.stdout = output.read()
        if not fail:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def test_host_ndk_and_task_execution(self):
        self.gradle(*[":xybrid_flutter:merge" + flavor + mode + "NativeLibs"
                      for flavor in ["Demo", "Full"] for mode in ["Debug", "Profile", "Release"]],
                    ":xybrid_flutter:verifyWiring")
        for mode in ["debug", "profile", "release"]:
            for abi in ["arm64-v8a", "x86_64"]:
                merged = list((self.root / "plugin/android/build/intermediates/merged_native_libs").glob(
                    "*/merge*NativeLibs/out/lib/" + abi + "/libxybrid_flutter.so"))
                self.assertTrue(merged)
                self.assertEqual(sum(path.read_text() == mode for path in merged), 2)
                for path in merged:
                    self.assertEqual(path.with_name("libc++_shared.so").read_text(), "host-version")
        for abi in ["arm64-v8a", "x86_64"]:
            self.assertEqual((self.root / "plugin/android/build/ndk-libs" / abi / "libc++_shared.so").read_text(), "host-version")
        for mode in ["debug", "profile", "release"]:
            data = json.loads((self.root / "plugin/android/build/jniLibs" / mode / "environment.json").read_text())
            self.assertEqual(data["CARGOKIT_NDK_VERSION"], "27.0.12077973")
            self.assertEqual(data["CARGOKIT_COMPILE_SDK_VERSION"], "34")
            self.assertEqual(data["CARGOKIT_MIN_SDK_VERSION"], "23")
            self.assertEqual(data["CARGOKIT_RUST_MIN_SDK_VERSION"], "28")
            self.assertEqual(data["CARGOKIT_CONFIGURATION"], mode)
            self.assertEqual(data["CARGOKIT_MANIFEST_DIR"], str(self.root / "plugin/android/../rust"))
            self.assertNotIn("android-arm", data["CARGOKIT_TARGET_PLATFORMS"].split(","))
            self.assertIn("android-arm64", data["CARGOKIT_TARGET_PLATFORMS"].split(","))
        # A path override must invalidate the Copy task, without --rerun-tasks.
        self.gradle(":xybrid_flutter:copyNdkLibs",
                    ":xybrid_flutter:cargokitCargoBuildXybrid_flutterDebug", "-PcustomNdk")
        # The existing Dart AndroidEnvironment reconstructs this path. ndkPath
        # affects the runtime copy, but its source-build support is separate work.
        custom_env = json.loads((self.root / "plugin/android/build/jniLibs/debug/environment.json").read_text())
        cargo_ndk = Path(custom_env["CARGOKIT_SDK_DIR"]) / "ndk" / custom_env["CARGOKIT_NDK_VERSION"]
        self.assertEqual(cargo_ndk, self.root / "sdk/ndk/27.0.12077973")
        self.assertNotEqual(cargo_ndk, self.root / "custom-ndk")
        for abi in ["arm64-v8a", "x86_64"]:
            self.assertEqual((self.root / "plugin/android/build/ndk-libs" / abi / "libc++_shared.so").read_text(), "host-path")
        result = self.gradle(":xybrid_flutter:cargokitCargoBuildXybrid_flutterDebug", fail=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("non-zero exit value 19", result.stdout)
        missing = self.root / "sdk/ndk/27.0.12077973/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/x86_64-linux-android/libc++_shared.so"
        missing.unlink()
        result = self.gradle(":xybrid_flutter:copyNdkLibs", fail=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("libc++_shared.so for x86_64 is missing from the host NDK", result.stdout)


if __name__ == "__main__":
    unittest.main()
