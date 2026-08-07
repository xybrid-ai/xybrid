#!/usr/bin/env bash
# measure-binary-size.sh — size a native backend's contribution on the host.
#
# Answers "how many bytes does this backend add?" by building the xybrid-bolt
# staticlib + cdylib twice and diffing them:
#
#   baseline : --features "$BASELINE_FEATURES"
#   delta    : --features "$DELTA_FEATURES"
#
# Anything the two feature sets share (ORT, Candle, the rest of llama.cpp)
# is constant on both sides and cancels out, so the figure printed is the
# isolated cost of what the delta adds.
#
# This is a LOCAL PROXY for the per-platform shipped-artifact delta (iOS .a /
# Android .so), not a substitute: it builds for the host triple, so the number
# is directionally representative but absolute byte counts differ per target.
# The staticlib (.a) is the closest proxy for the shipped iOS .a; the cdylib
# (.dylib/.so) for the Android .so. Prefer the STRIPPED delta as the meaningful
# figure. For the true shipped numbers, read the per-ABI / per-slice sizes the
# build-android.yml / build-apple.yml jobs print.
#
# Each variant builds llama.cpp from source via cmake (multi-minute, cold), so
# this is invoked explicitly — it is NOT wired into CI.
#
# Usage:
#   tools/scripts/measure-binary-size.sh                 # native vision (mtmd)
#   tools/scripts/measure-binary-size.sh whispercpp      # whisper.cpp ASR
#   BASELINE_FEATURES=... DELTA_FEATURES=... REQUIRED_SYMBOL=... \
#     LABEL="..." tools/scripts/measure-binary-size.sh   # anything else
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="xybrid-bolt"
LIB="libxybrid_bolt"   # crate lib name (xybrid-bolt -> underscored)
# cdylib extension is .dylib on macOS, .so on Linux.
SO_EXT="so"; [ "$(uname)" = "Darwin" ] && SO_EXT="dylib"

# Named presets. As of 0.2.1 every `platform-*` preset bundles
# `llm-llamacpp-vision`, so a preset can no longer serve as a lean baseline —
# each pair below isolates one backend against the llama build it sits on.
case "${1:-vision}" in
  vision)
    : "${LABEL:=Native vision (mtmd)}"
    : "${BASELINE_FEATURES:=llm-llamacpp}"
    : "${DELTA_FEATURES:=llm-llamacpp-vision}"
    : "${REQUIRED_SYMBOL:=mtmd}"
    ;;
  whispercpp)
    # whisper.cpp links the ggml llama.cpp already built, so the baseline must
    # include the llama backend — otherwise the "delta" would wrongly include
    # ggml itself, which every shipped artifact already carries.
    : "${LABEL:=whisper.cpp ASR}"
    : "${BASELINE_FEATURES:=llm-llamacpp-vision}"
    : "${DELTA_FEATURES:=llm-llamacpp-vision,asr-whispercpp}"
    : "${REQUIRED_SYMBOL:=whisper_full}"
    ;;
  custom)
    : "${LABEL:?set LABEL for a custom measurement}"
    : "${BASELINE_FEATURES:?set BASELINE_FEATURES for a custom measurement}"
    : "${DELTA_FEATURES:?set DELTA_FEATURES for a custom measurement}"
    : "${REQUIRED_SYMBOL:=}"
    ;;
  *)
    echo "usage: $(basename "$0") [vision|whispercpp|custom]" >&2
    exit 2
    ;;
esac

# Separate target dirs: the two builds differ by a native cmake/cc target, and a
# shared dir would let cargo reuse the wrong native objects across the flip.
BASE_TGT="$REPO_ROOT/target/size-base"
DELTA_TGT="$REPO_ROOT/target/size-delta"

build () {  # $1=target-dir  $2=feature-list
  echo "==> building $PKG ($2) [release] -> $1" >&2
  CARGO_TARGET_DIR="$1" cargo build --release -p "$PKG" --features "$2"
}

build "$BASE_TGT"  "$BASELINE_FEATURES"
build "$DELTA_TGT" "$DELTA_FEATURES"

# Correctness guard: the delta build must actually link the native code under
# test, otherwise a broken feature chain would silently report a ~0 delta and
# look like "this backend is free". `nm` on the produced staticlib is robust to
# build caching (it inspects the artifact, not the build log).
DELTA_A="$DELTA_TGT/release/$LIB.a"
if [ -n "$REQUIRED_SYMBOL" ] && command -v nm >/dev/null 2>&1 && [ -f "$DELTA_A" ]; then
  # Use `grep -c`, NOT `grep -q`: under `set -o pipefail`, `grep -q` closes the
  # pipe on the first match, which makes the (still-running) `nm` die with
  # SIGPIPE and reports the whole pipeline as failed — a false FATAL even when
  # the symbol IS present. `grep -c` consumes all of nm's output, so nm exits 0.
  found="$(nm "$DELTA_A" 2>/dev/null | grep -ci "$REQUIRED_SYMBOL" || true)"
  if [ "${found:-0}" -eq 0 ]; then
    echo "FATAL: the delta build linked no '$REQUIRED_SYMBOL' symbols — the" >&2
    echo "       '$DELTA_FEATURES' feature chain is broken; the measured" >&2
    echo "       delta is meaningless." >&2
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

row () {  # $1=label $2=base-file $3=delta-file
  if [ ! -f "$2" ] || [ ! -f "$3" ]; then echo "skip $1 (missing artifact)" >&2; return; fi
  local b v bs vs
  b=$(bytes "$2");           v=$(bytes "$3")
  bs=$(stripped_bytes "$2"); vs=$(stripped_bytes "$3")
  printf '%-30s %10s %10s %+9s %12s %12s %+10s\n' \
    "$1" "$(mib "$b")" "$(mib "$v")" "$(mib "$((v-b))")" \
    "$(mib "$bs")" "$(mib "$vs")" "$(mib "$((vs-bs))")"
}

echo
echo "$LABEL size delta — all figures MiB"
echo "  baseline: $BASELINE_FEATURES"
echo "  delta:    $DELTA_FEATURES"
printf '%-30s %10s %10s %9s %12s %12s %10s\n' \
  artifact base with delta base-strip with-strip delta-strip
row "$LIB.a  (staticlib ~ iOS .a)"   "$BASE_TGT/release/$LIB.a"     "$DELTA_TGT/release/$LIB.a"
row "$LIB.$SO_EXT (cdylib ~ Android .so)" "$BASE_TGT/release/$LIB.$SO_EXT" "$DELTA_TGT/release/$LIB.$SO_EXT"
echo
echo "Host: $(rustc -vV | sed -n 's/^host: //p')"
echo "Note: host proxy, not the shipped per-platform delta (see header)."
