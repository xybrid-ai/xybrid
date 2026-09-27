#!/usr/bin/env bash
# Runs android/.../XybridCodec.kt against the real bolt Kotlin records on a
# plain JVM — no Android SDK, emulator or Rust build. The Kotlin module as a
# whole is compiled by the Android example build (build-react-native.yml).
#
# Needs `kotlinc` and `java` on PATH. Usage: scripts/test-android.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KOTLIN_SDK="$HERE/../kotlin/src/main/kotlin/ai/xybrid"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# XybridBolt.kt's Flow helpers need kotlinx-coroutines on the classpath.
COROUTINES="$WORK/kotlinx-coroutines-core-jvm.jar"
curl -sfL -o "$COROUTINES" \
  https://repo1.maven.org/maven2/org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.7.3/kotlinx-coroutines-core-jvm-1.7.3.jar

kotlinc -nowarn -cp "$COROUTINES" -include-runtime -d "$WORK/codec-tests.jar" \
  "$KOTLIN_SDK/XybridBolt.kt" \
  "$HERE/android/src/main/java/ai/xybrid/reactnative/XybridCodec.kt" \
  "$HERE/tests/kotlin/android/util/Base64.kt" \
  "$HERE/tests/kotlin/CodecTest.kt"
java -cp "$WORK/codec-tests.jar:$COROUTINES" CodecTestKt
