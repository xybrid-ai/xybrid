#!/bin/bash
# The isolation self-test's assertions (check.bzl). Everything here is a fake
# engine, so every name may be printed.
#
#   check.sh NM OBJDUMP RAW ISOLATED LIBRARY API LOCAL UNDEFINED IMPORTS REPORT
#
# RAW and ISOLATED are the fake engine's archive before and after
# //bazel/zzz:isolate.bzl; LIBRARY is a shared library linked from host.c and
# ISOLATED with the API-29 NDK toolchain. API, LOCAL, UNDEFINED and IMPORTS are
# files of names, one per line.
set -euo pipefail

if [ "$#" -ne 10 ]; then
  echo "usage: check.sh NM OBJDUMP RAW ISOLATED LIBRARY API LOCAL UNDEFINED IMPORTS REPORT" >&2
  exit 2
fi
nm=$1 objdump=$2 raw=$3 isolated=$4 library=$5 api=$6 local_names=$7 undefined=$8 imports=$9 report=${10}

work=$(mktemp -d "${TMPDIR:-/tmp}/zzz-selftest.XXXXXX")
trap 'rm -rf "$work"' EXIT
failures=0
fail() {
  echo "zzz isolation self-test: $*" >&2
  failures=$((failures + 1))
}
names() { LC_ALL=C sort -u "$1"; }
joined() { LC_ALL=C sort -u "$1" | paste -sd " " -; }
# -A puts "archive:member: " before every line instead of a header per member
# (as in isolate_engine.sh); symbol names hold no ": ".
symbols() { "$nm" -A "$@" | sed -e 's/^.*: //' | LC_ALL=C sort -u; }

symbols -g --defined-only -j "$raw" > "$work/raw.globals"
symbols -g --defined-only -j "$isolated" > "$work/globals"
symbols -P --defined-only "$isolated" > "$work/defined"
symbols -u -j "$isolated" > "$work/undefined"
"$objdump" -h "$isolated" > "$work/sections"
"$objdump" -T "$library" > "$work/dynsym"
"$objdump" -p "$library" > "$work/headers"
"$objdump" -R "$library" > "$work/relocations"

# The fixture must exercise the property: before isolation it exports what
# the engine's runtime does.
for name in $(names "$local_names"); do
  grep -qxF "$name" "$work/raw.globals" || fail "$name is not a global definition of the raw fake engine"
done

# Isolated archive: exactly the API global, the rest local, imports kept.
names "$api" > "$work/api"
cmp -s "$work/globals" "$work/api" \
  || fail "global definitions are [$(joined "$work/globals")], expected the API [$(joined "$work/api")]"
for name in $(names "$local_names"); do
  grep -qE "^$name [a-z] " "$work/defined" || fail "$name is not a local definition of the isolated engine"
  if grep -qE "^$name [A-Z] " "$work/defined"; then fail "$name is still a global definition"; fi
done
for name in $(names "$undefined"); do
  grep -qxF "$name" "$work/undefined" || fail "$name is no longer an undefined import of the isolated engine"
done
# -fdata-sections names the TLS section .tbss.<symbol>.
for section in .tbss .init_array .eh_frame; do
  grep -qE "[[:space:]]${section//./\\.}(\\.[^[:space:]]+)?[[:space:]]" "$work/sections" \
    || fail "the isolated engine lost its $section section"
done

# Linked library: libc and libm imported, nothing of the engine's runtime
# exported, TLS through the dynamic (TLSDESC) model only.
# (No grep -q at the end of a pipe: with pipefail an early exit can fail it.)
dynamic() { grep -E "[[:space:]]$1\$" "$work/dynsym" || true; }
for name in $(names "$imports"); do
  grep -qF '*UND*' <<<"$(dynamic "$name")" || fail "the library does not import $name"
done
for name in $(names "$local_names"); do
  defined=$(grep -vF '*UND*' <<<"$(dynamic "$name")" || true)
  [ -z "$defined" ] || fail "the library defines $name in its dynamic symbol table"
done
grep -qE '^[[:space:]]+TLS[[:space:]]' "$work/headers" || fail "the library has no PT_TLS segment"
flags=$(sed -nE 's/^[[:space:]]+FLAGS[[:space:]]+0x([0-9a-fA-F]+)$/\1/p' "$work/headers")
if [ -n "$flags" ] && (( 0x$flags & 0x10 )); then
  fail "the library's DT_FLAGS has STATIC_TLS"
fi
if grep -q 'TPREL' "$work/relocations"; then fail "the library has initial-exec (TPREL) TLS relocations"; fi
grep -q 'R_AARCH64_TLSDESC' "$work/relocations" || fail "the library has no TLSDESC relocation for the engine's thread-local"

if [ "$failures" -ne 0 ]; then
  echo "zzz isolation self-test: $failures failure(s)" >&2
  exit 1
fi
{
  echo "raw fake engine: $(wc -l < "$work/raw.globals" | tr -d ' ') global definition(s): $(joined "$work/raw.globals")"
  echo "isolated: $(wc -l < "$work/globals" | tr -d ' ') global definition(s), the API; now local: $(joined "$local_names")"
  echo "isolated: still undefined: $(joined "$undefined"); .tbss, .init_array and .eh_frame kept"
  echo "library: imports $(joined "$imports"); PT_TLS, TLSDESC, no STATIC_TLS or TPREL"
} > "$report"
cat "$report"
