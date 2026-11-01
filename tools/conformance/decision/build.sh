#!/usr/bin/env bash
# Build the upstream decision oracle and the stock llama-server from the pinned
# llama.cpp (vendor/llama-cpp), CPU only, into tools/conformance/decision/build.
#
# Usage: tools/conformance/decision/build.sh [extra CMake configure arguments]
#
# Needs cmake (3.14+), a C++17 compiler and ninja (or make), and the submodule:
#   git submodule update --init vendor/llama-cpp
# Prints the bin directory holding decision-oracle and llama-server.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
build="$here/build"
die() { echo "build.sh: $*" >&2; exit 1; }

[[ -f "$repo/vendor/llama-cpp/tools/server/server-decision.cpp" ]] \
    || die "no vendor/llama-cpp checkout: git submodule update --init vendor/llama-cpp"

generator=()
if [[ ! -f "$build/CMakeCache.txt" ]] && command -v ninja > /dev/null; then
    generator=(-G Ninja)
fi
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2> /dev/null || echo 4)}"

start=$SECONDS
cmake -S "$here" -B "$build" ${generator[@]+"${generator[@]}"} -DCMAKE_BUILD_TYPE=Release "$@" > /dev/null
cmake --build "$build" --target decision-oracle llama-server -j "$jobs"
echo "build.sh: built in $((SECONDS - start)) s" >&2
echo "$build/bin"
