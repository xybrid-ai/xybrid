#!/usr/bin/env bash
# android-dlopen-gate.sh — push the staged bolt .so + siblings + probe to a
# booted emulator and assert libxybrid-bolt.so dlopens, both as built AND
# after an AGP-style strip. This is the authoritative gate for the
# strip-fragility / 16 KB-alignment bug: readelf can't prove a lib loads
# (it reads section headers; the loader reads program headers), only an
# on-device dlopen can.
#
# Expects a directory (default ./dlopen-gate) staged by build-android.yml
# containing: libxybrid-bolt.so, libxybrid-bolt.stripped.so, libc++_shared.so,
# libonnxruntime.so, android-dlopen-probe. Run after the emulator is booted
# (the emulator-runner action waits for boot before invoking this).
set -euo pipefail

GATE_DIR="${1:-dlopen-gate}"
DEVICE_DIR=/data/local/tmp/xybrid-gate
ADB="${ADB:-adb}"

echo "==> Pushing gate artifacts to $DEVICE_DIR"
"$ADB" shell "rm -rf $DEVICE_DIR && mkdir -p $DEVICE_DIR"
for f in libxybrid-bolt.so libxybrid-bolt.stripped.so libc++_shared.so \
         libonnxruntime.so android-dlopen-probe; do
    "$ADB" push "$GATE_DIR/$f" "$DEVICE_DIR/$f" > /dev/null
done
"$ADB" shell "chmod 755 $DEVICE_DIR/android-dlopen-probe"

# dlopen each candidate. RTLD_NOW + the siblings in LD_LIBRARY_PATH mirror an
# app's per-ABI lib dir. `libxybrid-bolt.so` is renamed in place so the probe
# always loads the same SONAME (the stripped copy is the same lib, AGP-stripped).
probe() {
    local lib="$1" label="$2"
    "$ADB" push "$GATE_DIR/$lib" "$DEVICE_DIR/libxybrid-bolt.so" > /dev/null
    # Capture stderr too (the probe prints DLOPEN-FAIL there) and `|| true` so
    # a failing probe doesn't trip `set -e` before we can report it — we want
    # the formatted message + an explicit `return 1`, not an abrupt exit.
    local out
    out="$("$ADB" shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./android-dlopen-probe $DEVICE_DIR/libxybrid-bolt.so" 2>&1 | tr -d '\r')" || true
    echo "    [$label] $out"
    case "$out" in
        DLOPEN-OK*) return 0 ;;
        *) echo "    [$label] FAILED" >&2; return 1 ;;
    esac
}

echo "==> dlopen as built"
probe libxybrid-bolt.so "as-built"
echo "==> dlopen after AGP-style strip (the regression this gate guards)"
probe libxybrid-bolt.stripped.so "stripped"

echo "==> dlopen gate passed"
