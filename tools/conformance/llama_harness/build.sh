#!/usr/bin/env bash
# Build the same-artifact (G2) scoring harness against the llama.cpp libraries
# cargo links into xybrid-llama's choice_logits test.
#
# Usage: tools/conformance/llama_harness/build.sh [FEATURES]
#
# FEATURES are xybrid-llama's cargo features for that test, `bindings` by
# default; run the test with the same ones. The script builds the test without
# running it, reads which archives and system libraries cargo linked for
# llama.cpp (a downloaded or staged natives slice, or a source build) from the
# build script's report, and links the harness against exactly those.
#
# Prints the path of the built binary, target/llama-harness/llama_harness.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
die() { echo "build.sh: $*" >&2; exit 1; }
features="${1:-bindings}"

# One line per search path ("path DIR") and library ("lib SPEC") cargo linked.
report="$(cd "$repo" && cargo test -p xybrid-llama --features "$features" --test choice_logits \
    --no-run --message-format=json | python3 -c '
import json, sys
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    if message.get("reason") == "build-script-executed" and "xybrid-llama-sys" in message["package_id"]:
        for path in message["linked_paths"]:
            print("path", path.split("=", 1)[-1])
        for lib in message["linked_libs"]:
            print("lib", lib)
')"
[[ -n "$report" ]] || die "cargo reported no llama.cpp link step for features '$features'"

paths=()
libs=()
while read -r kind value; do
    case "$kind" in
        path) paths+=("$value") ;;
        lib) libs+=("$value") ;;
    esac
done <<< "$report"

archives=()
add_archive() {
    local dir
    for dir in "${paths[@]}"; do
        if [[ -f "$dir/lib$1.a" ]]; then
            archives+=("$dir/lib$1.a")
            return
        fi
    done
    die "lib$1.a is in none of cargo's search paths"
}

system=()
seen=" "
for lib in "${libs[@]}"; do
    case "$seen" in *" $lib "*) continue ;; esac
    seen+="$lib "
    case "$lib" in
        static=llama_wrapper) ;; # xybrid's C shim; the harness calls llama.h directly
        c++ | stdc++) ;;         # the C++ compiler links its standard library itself
        static=*) add_archive "${lib#static=}" ;;
        framework=*) system+=(-framework "${lib#framework=}") ;;
        dylib=*) system+=("-l${lib#dylib=}") ;;
        *=*) die "unexpected link directive: $lib" ;;
        *) system+=("-l$lib") ;;
    esac
done

# libllama.a's install prefix holds the matching headers.
llama=""
for archive in ${archives[@]+"${archives[@]}"}; do
    [[ "$archive" == */libllama.a ]] && llama="$archive"
done
[[ -n "$llama" ]] || die "cargo did not link libllama.a"
lib_dir="$(dirname "$llama")"
case "$(basename "$lib_dir")" in
    lib | lib64) prefix="$(dirname "$lib_dir")" ;;
    *) prefix="$lib_dir" ;;
esac
[[ -f "$prefix/include/llama.h" ]] || die "no include/llama.h next to $lib_dir"

# Static archives resolve in one pass with GNU ld; a group lets them refer to
# each other in any order. Apple's linker needs no group.
if [[ "$(uname -s)" == Linux ]]; then
    archives=("-Wl,--start-group" "${archives[@]}" "-Wl,--end-group")
fi

out="$repo/target/llama-harness"
mkdir -p "$out"
"${CXX:-c++}" -std=c++17 -O2 -I"$prefix/include" "$here/main.cpp" "${archives[@]}" \
    ${system[@]+"${system[@]}"} -o "$out/llama_harness"
echo "$out/llama_harness"
