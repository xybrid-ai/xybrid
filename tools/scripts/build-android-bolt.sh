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
#
# Pin to NDK r27 by default. r29 (latest as of this script) has linker /
# toolchain differences that surface as build failures on x86_64 with
# our llama.cpp + cpp-httplib + candle native deps. r27 is the version
# the project's existing flutter binding and uniffi build flow target;
# stick with it until r29 is explicitly validated.
: "${ANDROID_HOME:=$HOME/Library/Android/sdk}"
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    # Prefer an explicit r27.x install; fall back to the highest version
    # available so the script still does *something* on hosts that only
    # have r28+.
    if compgen -G "$ANDROID_HOME/ndk/27.*" > /dev/null; then
        ANDROID_NDK_HOME="$(ls -d "$ANDROID_HOME"/ndk/27.*/ 2>/dev/null | sort -V | tail -n1 | sed 's:/$::')"
    else
        ANDROID_NDK_HOME="$(ls -d "$ANDROID_HOME"/ndk/*/ 2>/dev/null | sort -V | tail -n1 | sed 's:/$::')"
    fi
fi
if [ -z "$ANDROID_NDK_HOME" ] || [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "error: ANDROID_NDK_HOME not set and no NDK found under $ANDROID_HOME/ndk/" >&2
    exit 1
fi

# Host platform inside the NDK. Detect it from the OS rather than pinning
# to a single value: macOS (incl. Apple Silicon, which still installs the
# `darwin-x86_64` toolchain) builds locally, but CI runs this script on
# Linux (`linux-x86_64`) via `cargo xtask build-android`.
case "$(uname -s)" in
    Darwin) HOST=darwin-x86_64 ;;
    Linux)  HOST=linux-x86_64 ;;
    *)
        echo "error: unsupported host OS '$(uname -s)'; expected Darwin or Linux" >&2
        exit 1
        ;;
esac
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
# Release by default; set DEBUG=1 for a faster, unoptimized build when
# debugging native Android issues (symbols, asserts, quicker compile).
PROFILE_FLAG="--release"
if [ "${DEBUG:-0}" = "1" ]; then
    PROFILE_FLAG=""
    echo "    Profile:  debug (DEBUG=1)"
fi
# shellcheck disable=SC2086  # deliberate word-split: empty PROFILE_FLAG = debug
boltffi pack android $PROFILE_FLAG \
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

echo "==> Patching DT_NEEDED to include libc++_shared.so"
# boltffi 0.25's android pack does a second link step
# (pack/android/link.rs::android_shared_link_args) with hardcoded link
# args — only `-lm -llog -ldl`, no `-lc++_shared`. The cargo-staticlib
# pulled in c++_shared via rustflags, but the final clang `-shared` link
# strips that dep out. Result: dlopen on device fails with
#   cannot locate symbol "_ZTISt13runtime_error" referenced by libxybrid-bolt.so
# Fix: add DT_NEEDED post-link via patchelf. Works regardless of which
# linker boltffi invoked; treats the .so as an opaque ELF artifact.
if command -v patchelf > /dev/null 2>&1; then
    for abi in arm64-v8a armeabi-v7a x86 x86_64; do
        so="$KOTLIN_LIBS/$abi/libxybrid-bolt.so"
        if [ -f "$so" ]; then
            patchelf --add-needed libc++_shared.so "$so"
        fi
    done
else
    echo "    error: patchelf not on PATH (brew install patchelf). Aborting" >&2
    exit 1
fi

echo "==> Bundling ORT runtime from vendor/ort-android/"
# `ort-dynamic` mode dlopen's libonnxruntime.so at runtime. Only the
# arm64-v8a and x86_64 ORT slices are vendored (matches the historical
# coverage — armeabi-v7a and x86 are not first-class Android targets
# for ORT).
for abi in arm64-v8a x86_64; do
    src="$ORT_VENDOR/$abi/libonnxruntime.so"
    if [ -f "$src" ]; then
        cp "$src" "$KOTLIN_LIBS/$abi/libonnxruntime.so"
    else
        echo "    [$abi] missing: $src" >&2
    fi
done

echo "==> Bundling libc++_shared.so from NDK for every ABI"
# `libxybrid-bolt.so` is linked against libc++_shared.so (CMake builds
# llama.cpp / cpp-httplib / candle's native deps with
# `-DANDROID_STL=c++_shared`). Without c++_shared next to the bolt lib,
# the loader fails at process start:
#   dlopen failed: cannot locate symbol "_ZTISt13runtime_error"
#       referenced by "libxybrid-bolt.so"
# Source the .so from the NDK sysroot — same lib the toolchain linked
# the .so against, so versions match exactly.
#
# A `case` map (not an associative array) keeps this working on the stock
# macOS /bin/bash 3.2 that local dev may pick up, where `declare -A` errors.
NDK_SYSROOT_LIB="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST/sysroot/usr/lib"
for abi in arm64-v8a armeabi-v7a x86 x86_64; do
    case "$abi" in
        arm64-v8a)   triple=aarch64-linux-android ;;
        armeabi-v7a) triple=arm-linux-androideabi ;;
        x86)         triple=i686-linux-android ;;
        x86_64)      triple=x86_64-linux-android ;;
    esac
    src="$NDK_SYSROOT_LIB/$triple/libc++_shared.so"
    dst_dir="$KOTLIN_LIBS/$abi"
    if [ -f "$src" ] && [ -d "$dst_dir" ]; then
        cp "$src" "$dst_dir/libc++_shared.so"
    else
        echo "    [$abi] skipped (src=$src exists?=$([ -f "$src" ] && echo yes || echo no))"
    fi
done

echo "==> Done. Rebuild the example with:"
echo "    cd examples/android && ./gradlew :app:assembleDebug"
