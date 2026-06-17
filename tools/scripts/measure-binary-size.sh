#!/usr/bin/env bash
# measure-binary-size.sh — size the native mtmd/vision delta on the host.
#
# The native vision backend (`llm-llamacpp-vision`, llama.cpp's mtmd/clip) is
# OPT-IN and ships in NO mobile artifact. This script answers "how many bytes
# would it add?" by building the xybrid-bolt staticlib + cdylib for the host
# platform twice and diffing them:
#
#   baseline : --features <PRESET>                     (platform preset; no vision)
#   vision   : --features <PRESET>,llm-llamacpp-vision (adds the native mtmd backend)
#
# This is a LOCAL PROXY for the per-platform shipped-artifact delta (iOS .a /
# Android .so), not a substitute: the host is aarch64-apple-darwin with
# platform-macos (CoreML/Metal ORT + Candle-metal), so the mtmd/llama.cpp delta
# is directionally representative but absolute byte counts differ per target.
# The staticlib (.a) is the closest proxy for the shipped iOS .a; the cdylib
# (.dylib) for the Android .so. Prefer the STRIPPED delta as the meaningful
# figure. For the true shipped numbers, read the per-ABI / per-slice sizes the
# build-android.yml / build-apple.yml jobs print.
#
# Each variant builds llama.cpp from source via cmake (multi-minute, cold), so
# this is invoked explicitly — it is NOT wired into CI.
#
# Usage:
#   tools/scripts/measure-binary-size.sh                    # platform-macos (default on this host)
#   PRESET=platform-desktop tools/scripts/measure-binary-size.sh   # CPU-only preset (no Metal/CoreML)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRESET="${PRESET:-platform-macos}"
PKG="xybrid-bolt"
LIB="libxybrid_bolt"   # crate lib name (xybrid-bolt -> underscored)
# cdylib extension is .dylib on macOS, .so on Linux (e.g. PRESET=platform-desktop).
SO_EXT="so"; [ "$(uname)" = "Darwin" ] && SO_EXT="dylib"

# Separate target dirs: the two builds differ only by the mtmd cmake target;
# a shared dir would let cargo reuse the wrong native objects across the flip.
BASE_TGT="$REPO_ROOT/target/size-base"
VIS_TGT="$REPO_ROOT/target/size-vision"

build () {  # $1=target-dir  $2=feature-list
  echo "==> building $PKG ($2) [release] -> $1" >&2
  CARGO_TARGET_DIR="$1" cargo build --release -p "$PKG" --features "$2"
}

build "$BASE_TGT" "$PRESET"
build "$VIS_TGT"  "$PRESET,llm-llamacpp-vision"

# Correctness guard: the vision build must actually link the native mtmd code,
# otherwise a broken sdk->core->llama-sys/vision chain would silently report a
# ~0 delta and look like "vision is free". `nm` on the produced staticlib is
# robust to build caching (it inspects the artifact, not the build log).
VIS_A="$VIS_TGT/release/$LIB.a"
if command -v nm >/dev/null 2>&1 && [ -f "$VIS_A" ]; then
  # Use `grep -c`, NOT `grep -q`: under `set -o pipefail`, `grep -q` closes the
  # pipe on the first match, which makes the (still-running) `nm` die with
  # SIGPIPE and reports the whole pipeline as failed — a false FATAL even when
  # mtmd IS present. `grep -c` consumes all of nm's output, so nm exits 0.
  mtmd_syms="$(nm "$VIS_A" 2>/dev/null | grep -ci 'mtmd' || true)"
  if [ "${mtmd_syms:-0}" -eq 0 ]; then
    echo "FATAL: vision build linked no mtmd symbols — the llm-llamacpp-vision" >&2
    echo "       feature chain is broken; the measured delta is meaningless." >&2
    exit 1
  fi
fi

bytes () { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }   # macOS | Linux
stripped_bytes () {
  local src="$1" tmp; tmp="$(mktemp "${TMPDIR:-/tmp}/binary-size.XXXXXX")"; cp "$src" "$tmp"
  if   command -v llvm-strip >/dev/null 2>&1; then llvm-strip -x "$tmp" 2>/dev/null || true
  elif command -v strip      >/dev/null 2>&1; then strip      -x "$tmp" 2>/dev/null || true; fi
  bytes "$tmp"; rm -f "$tmp"
}
mib () { awk "BEGIN{printf \"%.1f\", $1/1048576}"; }

row () {  # $1=label $2=base-file $3=vision-file
  if [ ! -f "$2" ] || [ ! -f "$3" ]; then echo "skip $1 (missing artifact)" >&2; return; fi
  local b v bs vs
  b=$(bytes "$2");           v=$(bytes "$3")
  bs=$(stripped_bytes "$2"); vs=$(stripped_bytes "$3")
  printf '%-30s %10s %10s %+9s %12s %12s %+10s\n' \
    "$1" "$(mib "$b")" "$(mib "$v")" "$(mib "$((v-b))")" \
    "$(mib "$bs")" "$(mib "$vs")" "$(mib "$((vs-bs))")"
}

echo
echo "Native vision (mtmd) size delta — all figures MiB"
printf '%-30s %10s %10s %9s %12s %12s %10s\n' \
  artifact base vision delta base-strip vision-strip delta-strip
row "$LIB.a  (staticlib ~ iOS .a)"   "$BASE_TGT/release/$LIB.a"     "$VIS_TGT/release/$LIB.a"
row "$LIB.$SO_EXT (cdylib ~ Android .so)" "$BASE_TGT/release/$LIB.$SO_EXT" "$VIS_TGT/release/$LIB.$SO_EXT"
echo
echo "Preset: $PRESET   Host: $(rustc -vV | sed -n 's/^host: //p')"
echo "Note: host proxy, not the shipped per-platform delta (see header)."
