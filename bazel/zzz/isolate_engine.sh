#!/bin/bash
# Isolates a static engine archive: one relocatable object whose only global
# definitions are the engine's C API (see //bazel/zzz:isolate.bzl).
#
#   isolate_engine.sh LD OBJCOPY NM AR ENGINE_ARCHIVE API_LIST OUTPUT_ARCHIVE
#
# LD, OBJCOPY, NM and AR are the target C++ toolchain's ld.lld, llvm-objcopy,
# llvm-nm and llvm-ar. API_LIST holds one symbol name per line.
#
# Only counts and the API names (public: crates/zzz-sys/src/api.rs declares
# them) are printed. Tool diagnostics can name private engine symbols, so a
# failing tool is reported by name and line count only. Besides the four tools
# it needs only bash, coreutils, diffutils, grep and sed, which the bare RBE
# image (ubuntu:22.04) has; no Python.
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: isolate_engine.sh LD OBJCOPY NM AR ENGINE_ARCHIVE API_LIST OUTPUT_ARCHIVE" >&2
  exit 2
fi
ld=$1 objcopy=$2 nm=$3 ar=$4 archive=$5 api=$6 output=$7

work=$(mktemp -d "${TMPDIR:-/tmp}/zzz-isolate.XXXXXX")
trap 'rm -rf "$work"' EXIT

fail() {
  echo "zzz engine isolation: $*" >&2
  exit 1
}

count() {
  wc -l < "$1" | tr -d ' '
}

# run OUTFILE TOOL ARGS...: TOOL's stdout to OUTFILE; on failure, withhold
# its diagnostics.
run() {
  local into=$1
  shift
  if ! "$@" > "$into" 2> "$work/diagnostics"; then
    fail "$(basename "$1") failed; $(count "$work/diagnostics") line(s) of its diagnostics withheld (they can name private engine symbols)"
  fi
}

# symbols OUTFILE FILE NM_FLAGS...: the sorted, unique symbol names of an
# object or a whole archive. -A puts "file: " (an archive member as
# "archive:member: ") before every name instead of printing a header per
# member; symbol names hold no ": ", so everything up to the last one goes.
symbols() {
  local into=$1 file=$2
  shift 2
  run "$work/nm.out" "$nm" -A -j "$@" "$file"
  sed -e 's/^.*: //' "$work/nm.out" | LC_ALL=C sort -u > "$into"
}

for tool in "$ld" "$objcopy" "$nm" "$ar"; do
  [ -n "$tool" ] && [ -x "$tool" ] \
    || fail "the C++ toolchain must declare ld, objcopy, nm and ar; got '$tool'"
done
LC_ALL=C sort -u "$api" > "$work/api"
[ "$(count "$work/api")" -gt 0 ] || fail "the API list is empty"

# What the archive itself imports: names its members reference (undefined)
# that no member defines globally.
symbols "$work/archive.globals" "$archive" -g --defined-only
symbols "$work/archive.undefined" "$archive" -u
LC_ALL=C comm -23 "$work/archive.undefined" "$work/archive.globals" > "$work/archive.imports"

# 1. Every member, prelinked. -r leaves relocations against global symbols in
#    place, so the engine's references to its own runtime still point at its
#    own definitions once those turn local below.
prelinked="$work/prelinked.o"
run "$work/ld.out" "$ld" -r --whole-archive "$archive" --no-whole-archive -o "$prelinked"

# A common (tentative) symbol stays common through -r (ld.lld cannot allocate
# one there), and llvm-objcopy cannot make one local. llvm-nm -P prints its
# type as C.
run "$work/all" "$nm" -P "$prelinked"
commons=$(grep -c -E '^[^ ]+ C( |$)' "$work/all" || true)
[ "$commons" = 0 ] || fail "$commons common symbol(s), which a relocatable link cannot allocate; build the engine with -fno-common"

symbols "$work/globals.before" "$prelinked" -g --defined-only
symbols "$work/undefined.before" "$prelinked" -g -u
symbols "$work/defined.before" "$prelinked" --defined-only
missing=$(LC_ALL=C comm -23 "$work/api" "$work/globals.before" | tr '\n' ' ')
[ -z "$missing" ] || fail "the engine does not define its API: ${missing% }"
# Whatever the prelinked object imports, the archive imported: no global
# definition of the engine's may be left undefined (it would bind outside the
# engine), and ld -r adds no import. An import that went would be harmless,
# so this is a subset check.
LC_ALL=C comm -23 "$work/undefined.before" "$work/archive.imports" > "$work/unexpected"
[ ! -s "$work/unexpected" ] \
  || fail "$(count "$work/unexpected") symbol(s) left undefined by ld -r that the archive does not import"

# 2. Only the API stays global; every other definition turns local.
#    llvm-objcopy leaves undefined symbols global.
isolated="$work/zzz_embed_isolated.o"
run "$work/objcopy.out" "$objcopy" --keep-global-symbols="$work/api" "$prelinked" "$isolated"

# 3. The checks the link relies on. -g -u lists only global undefined symbols,
#    so an import made local (which no linker could resolve) counts as gone.
symbols "$work/globals.after" "$isolated" -g --defined-only
symbols "$work/undefined.after" "$isolated" -g -u
symbols "$work/defined.after" "$isolated" --defined-only
if ! cmp -s "$work/globals.after" "$work/api"; then
  fail "$(count "$work/globals.after") global or weak definition(s) left; expected exactly the $(count "$work/api") API symbols"
fi
cmp -s "$work/undefined.after" "$work/undefined.before" \
  || fail "the undefined imports changed ($(count "$work/undefined.before") before, $(count "$work/undefined.after") after)"
cmp -s "$work/defined.after" "$work/defined.before" \
  || fail "the set of defined symbols changed ($(count "$work/defined.before") before, $(count "$work/defined.after") after)"

# 4. One-member archive; D: zero timestamps, uids and modes.
rm -f "$output"
run "$work/ar.out" "$ar" rcsD "$output" "$isolated"

echo "zzz engine isolation: $(count "$work/api") API symbol(s) kept global," \
  "$(( $(count "$work/globals.before") - $(count "$work/api") )) other global or weak definition(s) made local," \
  "$(count "$work/undefined.before") undefined import(s) unchanged"
