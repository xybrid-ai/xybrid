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

# NOTE: 32-bit x86 (i686-linux-android) is intentionally not built — see the
# `architectures` note in crates/xybrid-bolt/boltffi.toml. Emulator-only,
# ORT-less, and the 0.2.0 native vision backend doesn't compile for it.

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

# Why a two-phase build with a clang shim (instead of one `boltffi pack`):
#
# `boltffi pack android` builds the Rust *staticlib* (libxybrid_bolt.a) and
# then relinks it into the shipped .so with a hardcoded
# `clang -shared … -lm -llog -ldl` (boltffi_cli 0.25.3
# pack/android/link.rs::android_shared_link_args). That relink passes
# neither `-lc++_shared` nor `-Wl,-z,max-page-size=16384`. And because the
# artifact boltffi consumes is a *staticlib*, the `link-arg` rustflags in the
# repo-root .cargo/config.toml never reach the shipped .so (rustc doesn't link
# a staticlib). So the shipped .so used to come out with:
#   - undefined C++ ABI symbols (papered over by a post-link
#     `patchelf --add-needed libc++_shared.so`), and
#   - 4 KB LOAD-segment alignment instead of 16 KB.
#
# The patchelf rewrite appended a fresh high-alignment LOAD segment. A
# consumer's AGP `stripDebugSymbols` then re-compacted the file and left that
# segment at a non-congruent file offset (p_offset % p_align != p_vaddr %
# p_align), so bionic couldn't find .gnu.hash → `dlopen failed: empty/missing
# DT_HASH/DT_GNU_HASH … (new hash type from the future?)`. The 4 KB alignment
# separately broke loading on 16 KB-page devices (Android 15+).
#
# Fix: inject `-lc++_shared` + `-Wl,-z,max-page-size=16384` into boltffi's
# final relink and drop patchelf, so the .so is a clean linker output (no
# post-link ELF rewrite → survives any downstream strip; 16 KB-aligned →
# loads on 16 KB-page devices). boltffi resolves its relink clang by absolute
# path from ANDROID_NDK_HOME, and that same var drives the llama.cpp CMake
# build — so we can't just point it at a wrapper for the whole build. Instead:
#   1. `boltffi build android` with the REAL NDK (compiles the native deps).
#   2. `boltffi pack android --no-build` with ANDROID_NDK_HOME pointed at a
#      minimal clang *shim* that execs the real clang and appends the two link
#      args only on the `-shared` relink. No CMake runs in this phase, so the
#      shim NDK needs nothing but the per-ABI clang wrappers.

# boltffi resolves its relink clang as `<prefix><min_sdk>-clang`; keep the
# shim wrapper names in sync with boltffi.toml's [targets.android] min_sdk.
BOLT_MIN_SDK="$(grep -A40 '^\[targets.android\]' "$BOLT_CRATE/boltffi.toml" \
    | grep -m1 '^min_sdk' | grep -oE '[0-9]+' || true)"
: "${BOLT_MIN_SDK:=24}"

echo "==> Phase 1/2: building Android staticlibs (real NDK)"
# shellcheck disable=SC2086  # deliberate word-split: empty PROFILE_FLAG = debug
boltffi build android $PROFILE_FLAG \
    --cargo-arg=--features --cargo-arg=platform-android

echo "==> Building clang shim (adds -lc++_shared + 16 KB alignment at link)"
# Minimal fake-NDK whose only contents are per-ABI clang wrappers. Each wrapper
# execs the real NDK clang (so it finds its real sysroot) and, *only when
# linking* (`-shared` present), appends the link args boltffi's relink omits.
# A compile invocation (`-c`, e.g. jni_glue.c) passes through untouched.
SHIM_NDK="$REPO_ROOT/target/android-bolt-clang-shim/ndk"
SHIM_BIN="$SHIM_NDK/toolchains/llvm/prebuilt/$HOST/bin"
rm -rf "$REPO_ROOT/target/android-bolt-clang-shim"
mkdir -p "$SHIM_BIN"
# llvm-ar isn't used in the --no-build pack, but symlink it so any incidental
# toolchain lookup resolves.
ln -sf "$BIN/llvm-ar" "$SHIM_BIN/llvm-ar"
# ABI -> clang-wrapper prefix, mirroring boltffi's AndroidAbi::clang_prefix.
# 16 KB page alignment applies to the 64-bit ABIs that ship on Android 15+
# devices (arm64, x86_64); armv7/i686 are pre-16 KB-page and only need the
# c++_shared dep (matches the .cargo/config.toml policy for the cdylib paths).
emit_shim() {
    local prefix="$1" align16k="$2"
    local wrapper="$SHIM_BIN/${prefix}${BOLT_MIN_SDK}-clang"
    local extra='-Wl,--no-as-needed -lc++_shared -Wl,--as-needed'
    if [ "$align16k" = "1" ]; then
        extra="$extra -Wl,-z,max-page-size=16384"
    fi
    cat > "$wrapper" <<EOF
#!/usr/bin/env bash
# Generated by build-android-bolt.sh — do not edit. Execs the real NDK clang,
# appending C++-runtime + page-alignment link args only on the -shared relink.
set -euo pipefail
real="$BIN/${prefix}${BOLT_MIN_SDK}-clang"
for a in "\$@"; do
    if [ "\$a" = "-shared" ]; then
        exec "\$real" "\$@" $extra
    fi
done
exec "\$real" "\$@"
EOF
    chmod +x "$wrapper"
}
emit_shim aarch64-linux-android      1
emit_shim x86_64-linux-android       1
emit_shim armv7a-linux-androideabi   0
emit_shim i686-linux-android         0

echo "==> Phase 2/2: re-packing jniLibs through the shim (no rebuild)"
# shellcheck disable=SC2086  # deliberate word-split: empty PROFILE_FLAG = debug
ANDROID_NDK_HOME="$SHIM_NDK" boltffi pack android $PROFILE_FLAG --no-build

echo "==> Copying libxybrid-bolt.so into bindings/kotlin/libs/"
for abi in arm64-v8a armeabi-v7a x86_64; do
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

# NOTE: no patchelf step. `libc++_shared.so` is now a DT_NEEDED added by the
# shim at link time (see the two-phase build above), so there is no post-link
# ELF rewrite — the .so survives a consumer's AGP strip.

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
for abi in arm64-v8a armeabi-v7a x86_64; do
    case "$abi" in
        arm64-v8a)   triple=aarch64-linux-android ;;
        armeabi-v7a) triple=arm-linux-androideabi ;;
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

echo "==> Validating linked .so (catches a shim regression before it ships)"
# Static tripwire for the two failure modes this build guards against:
#   - c++_shared must be a DT_NEEDED (so dlopen resolves the C++ runtime),
#   - the 64-bit ABIs must be 16 KB-aligned (so they load on Android 15+),
#   - a strip must preserve PT_LOAD congruence (p_offset % p_align ==
#     p_vaddr % p_align) and the GNU_HASH (so a consumer's AGP strip can't
#     corrupt the loader's view — the bug this whole script works around).
# readelf can't *prove* a load (the loader reads program headers, readelf
# reads section headers) — the authoritative gate is the emulator dlopen in
# .github/workflows/build-android.yml. This is the cheap local pre-check.
READELF="$BIN/llvm-readelf"
STRIP="$BIN/llvm-strip"
validate_so() {
    local so="$1" want16k="$2" abi="$3"
    if ! "$READELF" -d "$so" | grep -q 'libc++_shared.so'; then
        echo "    [$abi] FAIL: libc++_shared.so not in DT_NEEDED" >&2
        return 1
    fi
    if [ "$want16k" = "1" ]; then
        while read -r align; do
            if [ "$((align))" -lt "$((0x4000))" ]; then
                echo "    [$abi] FAIL: LOAD segment align $align < 16 KB (0x4000)" >&2
                return 1
            fi
        done < <("$READELF" -l "$so" | awk '$1=="LOAD"{if(NF>4){print $NF}else{getline;print $NF}}')
    fi
    # Strip a copy exactly as a consumer's AGP would, then re-check the loader
    # invariants. A patchelf-style appended segment would break congruence here.
    local stripped="$so.stripcheck"
    cp "$so" "$stripped"
    "$STRIP" --strip-all "$stripped"
    if ! "$READELF" -d "$stripped" | grep -q 'GNU_HASH'; then
        echo "    [$abi] FAIL: GNU_HASH gone after strip" >&2
        rm -f "$stripped"; return 1
    fi
    while read -r off vaddr align; do
        if [ "$((off % align))" -ne "$((vaddr % align))" ]; then
            echo "    [$abi] FAIL: PT_LOAD congruence broken after strip" \
                 "(off=$off vaddr=$vaddr align=$align)" >&2
            rm -f "$stripped"; return 1
        fi
    done < <("$READELF" -l "$stripped" | awk '$1=="LOAD"{if(NF>4){print $2,$3,$NF}else{off=$2;vaddr=$3;getline;print off,vaddr,$NF}}')
    rm -f "$stripped"
    local align_note=""
    [ "$want16k" = "1" ] && align_note="16 KB align, "
    echo "    [$abi] OK (c++_shared dep, ${align_note}strip-safe)"
}
VALIDATED=0
for abi in arm64-v8a armeabi-v7a x86_64; do
    so="$KOTLIN_LIBS/$abi/libxybrid-bolt.so"
    [ -f "$so" ] || continue
    case "$abi" in
        arm64-v8a | x86_64) want16k=1 ;;
        *)                  want16k=0 ;;
    esac
    validate_so "$so" "$want16k" "$abi"
    VALIDATED=$((VALIDATED + 1))
done
if [ "$VALIDATED" -eq 0 ]; then
    echo "error: no libxybrid-bolt.so was produced to validate" >&2
    exit 1
fi

echo "==> Done. Rebuild the example with:"
echo "    cd examples/android && ./gradlew :app:assembleDebug"
