#!/usr/bin/env bash
# build-android-bolt.sh
#
# Re-pack the Android bolt artifact with the `platform-android` feature
# (pulls in xybrid-core/ort-dynamic, llm-llamacpp, candle — the feature
# set every TTS / ASR / LLM model on Android needs), then refresh the
# jniLibs the Kotlin module ships.
#
# Why this exists:
#
# - `boltffi pack android --release` on its own builds with no Cargo
#   features. That's enough to link the SDK skeleton but produces a
#   `libxybrid-bolt.so` that panics at runtime ("requires the
#   llm-mistral or llm-llamacpp feature" for LLM models, and a SIGABRT
#   inside `ort::setup_api` when the phonemizer tries to load the ORT
#   runtime that wasn't compiled in for ASR / TTS).
# - The `platform-android` feature compiles llama.cpp from source via
#   CMake. CMake invokes `cc-rs` which defaults to the legacy unsuffixed
#   `aarch64-linux-android-clang` toolchain name. NDK r27+ only ships
#   API-suffixed binaries (`aarch64-linux-android24-clang`, etc.). The
#   env-var block below points cc-rs at the right binaries so the
#   llama.cpp / cpp-httplib / candle native builds succeed for every
#   ABI.
# - The bolt artifact uses `ort-dynamic` mode and dlopens
#   `libonnxruntime.so` at runtime; bundle the prebuilt
#   libonnxruntime.so + libc++_shared.so from vendor/ort-android/
#   alongside `libxybrid-bolt.so` so ORT can find them on device.
#
# Usage: ./tools/scripts/build-android-bolt.sh
# Optional env overrides:
#   ANDROID_NDK_HOME, ANDROID_HOME, ANDROID_API (default 28)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOLT_CRATE="$REPO_ROOT/crates/xybrid-bolt"
KOTLIN_LIBS="$REPO_ROOT/bindings/kotlin/libs"
ORT_VENDOR="$REPO_ROOT/vendor/ort-android"

# Resolve NDK. Honor the user's ANDROID_NDK_HOME if set; otherwise look
# under the canonical SDK install location.
: "${ANDROID_HOME:=$HOME/Library/Android/sdk}"
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    # Pick the highest version directory under sdk/ndk/ — works regardless
    # of whether the user has r26 / r27 / r29 installed.
    ANDROID_NDK_HOME="$(ls -d "$ANDROID_HOME"/ndk/*/ 2>/dev/null | sort -V | tail -n1 | sed 's:/$::')"
fi
if [ -z "$ANDROID_NDK_HOME" ] || [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "error: ANDROID_NDK_HOME not set and no NDK found under $ANDROID_HOME/ndk/" >&2
    exit 1
fi

# Host platform inside the NDK. Apple Silicon Macs still install the
# `darwin-x86_64` toolchain (NDK doesn't ship a separate arm64 toolchain).
HOST=darwin-x86_64
BIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST/bin"
if [ ! -d "$BIN" ]; then
    echo "error: NDK toolchain bin not found at $BIN" >&2
    exit 1
fi

# Minimum Android API. Pinned to 28 to match
# bindings/kotlin/build.gradle.kts (`defaultConfig.minSdk = 24` is the
# library floor, but xybrid-sdk's platform-android preset / candle / ort
# require 24+; 28 keeps Vulkan / NNAPI APIs reachable for backends that
# need them).
: "${ANDROID_API:=28}"

export ANDROID_NDK_HOME ANDROID_HOME

# cc-rs picks up CC_/CXX_/AR_<target> env vars to skip its legacy
# toolchain-name search. cargo picks up CARGO_TARGET_<TARGET>_LINKER.
export CC_aarch64_linux_android="$BIN/aarch64-linux-android${ANDROID_API}-clang"
export CXX_aarch64_linux_android="$BIN/aarch64-linux-android${ANDROID_API}-clang++"
export AR_aarch64_linux_android="$BIN/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"

export CC_armv7_linux_androideabi="$BIN/armv7a-linux-androideabi${ANDROID_API}-clang"
export CXX_armv7_linux_androideabi="$BIN/armv7a-linux-androideabi${ANDROID_API}-clang++"
export AR_armv7_linux_androideabi="$BIN/llvm-ar"
export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$CC_armv7_linux_androideabi"

export CC_x86_64_linux_android="$BIN/x86_64-linux-android${ANDROID_API}-clang"
export CXX_x86_64_linux_android="$BIN/x86_64-linux-android${ANDROID_API}-clang++"
export AR_x86_64_linux_android="$BIN/llvm-ar"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$CC_x86_64_linux_android"

export CC_i686_linux_android="$BIN/i686-linux-android${ANDROID_API}-clang"
export CXX_i686_linux_android="$BIN/i686-linux-android${ANDROID_API}-clang++"
export AR_i686_linux_android="$BIN/llvm-ar"
export CARGO_TARGET_I686_LINUX_ANDROID_LINKER="$CC_i686_linux_android"

echo "==> Packing Android bolt artifact"
echo "    NDK:      $ANDROID_NDK_HOME"
echo "    API:      $ANDROID_API"
echo "    Features: platform-android"

cd "$BOLT_CRATE"
rm -rf dist/android
boltffi pack android --release \
    --cargo-arg=--features --cargo-arg=platform-android

echo "==> Copying libxybrid-bolt.so into bindings/kotlin/libs/"
for abi in arm64-v8a armeabi-v7a x86 x86_64; do
    src="$BOLT_CRATE/dist/android/jniLibs/$abi/libxybrid-bolt.so"
    dst_dir="$KOTLIN_LIBS/$abi"
    if [ -f "$src" ]; then
        mkdir -p "$dst_dir"
        cp "$src" "$dst_dir/"
        echo "    [$abi] $(du -h "$dst_dir/libxybrid-bolt.so" | cut -f1)"
    else
        echo "    [$abi] skipped (no artifact)"
    fi
done

echo "==> Bundling ORT runtime from vendor/ort-android/"
# `ort-dynamic` mode dlopen's libonnxruntime.so at runtime. Only the
# arm64-v8a and x86_64 ORT slices are vendored (matches the historical
# coverage — armeabi-v7a and x86 are not first-class Android targets
# for ORT).
for abi in arm64-v8a x86_64; do
    for lib in libonnxruntime.so libc++_shared.so; do
        src="$ORT_VENDOR/$abi/$lib"
        dst="$KOTLIN_LIBS/$abi/$lib"
        if [ -f "$src" ]; then
            cp "$src" "$dst"
        else
            echo "    [$abi] missing: $src" >&2
        fi
    done
done

echo "==> Done. Rebuild the example with:"
echo "    cd examples/android && ./gradlew :app:assembleDebug"
