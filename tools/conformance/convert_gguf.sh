#!/usr/bin/env bash
# Build the Qwen3.5-0.8B conformance GGUFs with the llama.cpp xybrid links.
#
#   convert_gguf.sh f16    <hf-snapshot-dir> <out.gguf>
#   convert_gguf.sh q4_k_m <f16.gguf>        <out.gguf>
#
# Both refuse unless vendor/llama-cpp is checked out, unmodified, at the
# revision provenance.json pins, and both fail unless the output embeds exactly
# templates/qwen3.5.jinja (G0's template). README.md explains why the bytes do
# not depend on the host. XYBRID_CONFORMANCE_CACHE holds the llama-quantize
# build and logs (default: <repo>/target/choice-conformance).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
LLAMA="$REPO_ROOT/vendor/llama-cpp"
FIXTURES="$REPO_ROOT/integration-tests/fixtures/choice"
CACHE="${XYBRID_CONFORMANCE_CACHE:-$REPO_ROOT/target/choice-conformance}"
MODEL_NAME="Qwen3.5-0.8B"

die() {
    echo "convert_gguf: $*" >&2
    exit 1
}

# Paths to remove on exit. A script-level list, not an EXIT trap over function
# locals: bash 5 runs the trap after errexit has unwound them.
CLEANUP=()
cleanup() {
    local path
    for path in "${CLEANUP[@]+"${CLEANUP[@]}"}"; do
        rm -rf "$path"
    done
}
trap cleanup EXIT

usage() {
    sed -n '2,5p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

require_pinned_llama() {
    local pinned actual
    pinned="$(jq -er '.sources["llama.cpp"].revision' "$FIXTURES/provenance.json")" ||
        die "provenance.json pins no llama.cpp revision"
    [ -f "$LLAMA/convert_hf_to_gguf.py" ] ||
        die "vendor/llama-cpp is not checked out (git submodule update --init vendor/llama-cpp)"
    actual="$(git -C "$LLAMA" rev-parse HEAD)"
    [ "$actual" = "$pinned" ] || die "vendor/llama-cpp is at $actual, provenance pins $pinned"
    [ -z "$(git -C "$LLAMA" status --porcelain --untracked-files=no)" ] ||
        die "vendor/llama-cpp has local modifications"
    echo "$pinned"
}

convert_env() {
    uv run --no-project --python 3.12 \
        --with-requirements "$HERE/requirements-convert.txt" "$@"
}

# Move $1 to $2 only if it embeds exactly the committed chat template.
publish() {
    PYTHONPATH="$LLAMA/gguf-py" convert_env python - "$1" "$FIXTURES/templates/qwen3.5.jinja" <<'PY'
import sys
from pathlib import Path

from gguf import GGUFReader

field = GGUFReader(sys.argv[1]).get_field("tokenizer.chat_template")
if field is None:
    sys.exit(f"{sys.argv[1]}: no embedded chat template")
if bytes(field.parts[field.data[0]]) != Path(sys.argv[2]).read_bytes():
    sys.exit(f"{sys.argv[1]}: embedded chat template differs from {sys.argv[2]}")
PY
    mv -f "$1" "$2"
    echo "wrote $2"
}

convert_f16() {
    local snapshot="$1" out="$2" view tmp log
    [ -f "$snapshot/config.json" ] || die "$snapshot is not a Hugging Face snapshot"
    snapshot="$(cd "$snapshot" && pwd)"
    require_pinned_llama >/dev/null
    mkdir -p "$(dirname "$out")" "$CACHE/logs"
    # The converter names the model after the snapshot directory
    # (general.basename/finetune/size_label), so convert from a view named after
    # the upstream repository: the bytes must not depend on the staging path.
    view="$(mktemp -d)"
    tmp="$out.tmp.$$"
    CLEANUP+=("$view" "$tmp")
    mkdir -p "$view/$MODEL_NAME"
    ln -s "$snapshot"/* "$view/$MODEL_NAME"/
    log="$CACHE/logs/convert-f16.log"
    if ! convert_env python "$LLAMA/convert_hf_to_gguf.py" "$view/$MODEL_NAME" \
        --outtype f16 --model-name "$MODEL_NAME" --outfile "$tmp" >"$log" 2>&1; then
        tail -n 40 "$log" >&2
        die "conversion failed; full log: $log"
    fi
    publish "$tmp" "$out"
}

# CPU only, no -march=native, and no fused multiply-add contraction: GCC
# contracts across statements by default and clang only within one, which alone
# changes the Q4_K_M bytes on arm64.
QUANTIZE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_FLAGS=-ffp-contract=off
    -DCMAKE_CXX_FLAGS=-ffp-contract=off
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_NATIVE=OFF
    -DGGML_METAL=OFF
    -DGGML_BLAS=OFF
    -DGGML_OPENMP=OFF
    -DLLAMA_OPENSSL=OFF
    -DLLAMA_BUILD_COMMON=ON
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=OFF
)

build_quantize() {
    local revision="$1" flags build
    # Key the cached build by its flags, so a flag change never reuses it.
    if command -v sha256sum >/dev/null 2>&1; then
        flags="$(printf '%s\n' "${QUANTIZE_FLAGS[@]}" | sha256sum | cut -c1-12)"
    else
        flags="$(printf '%s\n' "${QUANTIZE_FLAGS[@]}" | shasum -a 256 | cut -c1-12)"
    fi
    build="$CACHE/build/llama-quantize-$revision-$flags"
    if [ ! -x "$build/bin/llama-quantize" ]; then
        cmake -S "$LLAMA" -B "$build" "${QUANTIZE_FLAGS[@]}" >&2
        # One job per core. A bare --parallel is an unbounded `make -j`: on a
        # 3-core, 7 GB CI runner it starts ~180 compiles at once and thrashes
        # for most of an hour.
        cmake --build "$build" --target llama-quantize \
            --parallel "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" >&2
    fi
    echo "$build/bin/llama-quantize"
}

convert_q4_k_m() {
    local f16="$1" out="$2" revision quantize tmp log
    [ -f "$f16" ] || die "$f16 does not exist"
    revision="$(require_pinned_llama)"
    quantize="$(build_quantize "$revision")"
    mkdir -p "$(dirname "$out")" "$CACHE/logs"
    tmp="$out.tmp.$$"
    CLEANUP+=("$tmp")
    log="$CACHE/logs/quantize-q4_k_m.log"
    if ! "$quantize" "$f16" "$tmp" Q4_K_M >"$log" 2>&1; then
        tail -n 40 "$log" >&2
        die "quantization failed; full log: $log"
    fi
    publish "$tmp" "$out"
}

[ $# -eq 3 ] || usage
command -v jq >/dev/null || die "jq is required"
command -v uv >/dev/null || die "uv is required (https://docs.astral.sh/uv/)"
case "$1" in
    f16) convert_f16 "$2" "$3" ;;
    q4_k_m)
        command -v cmake >/dev/null || die "cmake is required"
        convert_q4_k_m "$2" "$3"
        ;;
    *) usage ;;
esac
