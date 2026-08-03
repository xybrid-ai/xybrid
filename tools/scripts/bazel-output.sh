#!/usr/bin/env bash
# Resolve a Bazel target's output file(s) to real paths.
#
# Why this exists: `bazel-bin/` is a symlink that tracks the LAST invocation,
# so a job that builds several configs in sequence and then reads `bazel-bin`
# gets whichever config ran last (this actually shipped once — a Mach-O binary
# under a Linux name, "Exec format error"). The fix is to resolve each artifact
# with `cquery --output=files` immediately after its build, which every job did
# by hand, 25 times, inconsistently: mostly `| head -1`, sometimes
# `| grep '\.dll$'`.
#
# `head -1` is a guess — it silently returns whichever output sorts first if a
# target ever grows a second one. This asserts instead:
#
#   (default)     exactly one output, or fail with what it actually found
#   --ext EXT     exactly one output ending in .EXT (e.g. --ext dll)
#   --all         every output, one per line (for callers that want them all)
#
# Verified output counts at the time of writing: the rust binaries, cdylibs and
# staticlibs all resolve to exactly 1 file on macos-metal / windows-gnu / ios /
# android / linux; only the windows-MSVC cdylib emits 2 (`.dll` + `.dll.lib`),
# which is what --ext and --all are for.
#
# Usage:
#   BIN=$(tools/scripts/bazel-output.sh //crates/xybrid-cli:xybrid --config=remote -c opt)
#   DLL=$(tools/scripts/bazel-output.sh --ext dll //crates/xybrid-bolt:xybrid_bolt_cdylib --config=windows-msvc -c opt)
#   tools/scripts/bazel-output.sh --all //bindings/flutter/rust:xybrid_flutter_ffi_cdylib "$@"
#
# This only RESOLVES paths; build the target first. Pass the same config flags
# to both, or cquery reports a different configuration than the one you built.
set -euo pipefail

MODE=one
EXT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --all) MODE=all; shift ;;
    --ext) MODE=ext; EXT="${2:?--ext needs an extension, e.g. --ext dll}"; shift 2 ;;
    --) shift; break ;;
    -*) echo "bazel-output.sh: unknown option '$1'" >&2; exit 2 ;;
    *) break ;;
  esac
done

TARGET="${1:?usage: bazel-output.sh [--ext EXT | --all] TARGET [BAZEL_ARGS...]}"
shift

# cquery chatter goes to stderr; the file list is stdout. Let stderr through so
# a real analysis failure is visible in the job log instead of an empty result.
FILES=$("${BAZEL:-bazelisk}" cquery "$@" --output=files "$TARGET")

if [ "$MODE" = ext ]; then
  # Literal suffix match — an --ext of `dll.lib` must not have its dot treated
  # as a regex wildcard, and `--ext dll` must not swallow `.dll.lib`.
  MATCHED=""
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in
      *".$EXT") MATCHED="${MATCHED}${f}"$'\n' ;;
    esac
  done <<< "$FILES"
  FILES="${MATCHED%$'\n'}"
fi

if [ "$MODE" = all ]; then
  printf '%s\n' "$FILES"
  exit 0
fi

COUNT=$(printf '%s' "$FILES" | grep -c . || true)
if [ "$COUNT" != "1" ]; then
  {
    echo "bazel-output.sh: expected exactly 1 output for $TARGET"
    [ -n "$EXT" ] && echo "  (filtered to .$EXT)"
    echo "  bazel args: $*"
    echo "  got $COUNT:"
    printf '%s\n' "$FILES" | sed 's/^/    /'
  } >&2
  exit 1
fi

printf '%s\n' "$FILES"
