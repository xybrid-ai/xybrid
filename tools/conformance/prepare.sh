#!/usr/bin/env bash
# Stage the choice-scoring conformance models, verified against their pins.
#
#   tools/conformance/prepare.sh [--inputs] <model-id>...
#
# Ids are the "derived" entries of integration-tests/fixtures/models/models.json.
# A staged artifact that matches its sha256 is kept. Otherwise every input is
# fetched from its immutable URL and sha256-checked, the artifact is rebuilt,
# and it is staged only if it matches the pin. --inputs also stages the inputs
# (in $XYBRID_CONFORMANCE_CACHE/inputs/<id>/) for regenerating goldens.
# Needs curl, jq and uv; the GGUFs also need cmake and vendor/llama-cpp.
# XYBRID_CONFORMANCE_CACHE holds inputs, builds and logs (default:
# <repo>/target/choice-conformance). See README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
MODELS_DIR="$REPO_ROOT/integration-tests/fixtures/models"
MANIFEST="$MODELS_DIR/models.json"
FIXTURES="$REPO_ROOT/integration-tests/fixtures/choice"
CACHE="${XYBRID_CONFORMANCE_CACHE:-$REPO_ROOT/target/choice-conformance}"

die() {
    echo "prepare: $*" >&2
    exit 1
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

entry() {
    jq -ec --arg id "$1" '.models[$id] | select(.source == "derived")' "$MANIFEST" ||
        die "$1 is not a derived model in models.json"
}

# "<output>\t<sha256>" for every declared output of $1.
outputs() {
    entry "$1" | jq -r '.files[] | [.output, .sha256] | @tsv'
}

# Refuse paths that could escape the directory they are joined to.
safe_relative() {
    case "$1" in
        "" | /* | ..* | */../* | */..) die "unsafe path in models.json: $1" ;;
    esac
}

staged() {
    local id="$1" output sha
    while IFS=$'\t' read -r output sha; do
        [ -f "$MODELS_DIR/$id/$output" ] || return 1
        [ "$(sha256_file "$MODELS_DIR/$id/$output")" = "$sha" ] || return 1
    done < <(outputs "$id")
}

fetch() {
    local url="$1" path="$2" sha="$3" actual
    if [ -f "$path" ] && [ "$(sha256_file "$path")" = "$sha" ]; then
        return
    fi
    mkdir -p "$(dirname "$path")"
    echo "  fetching $url"
    # HTTPS only, redirects included; the sha256 below is the integrity check.
    curl -fL --proto '=https' --proto-redir '=https' --connect-timeout 30 --retry 3 -sS \
        -o "$path.part" "$url" || die "download failed: $url"
    actual="$(sha256_file "$path.part")"
    if [ "$actual" != "$sha" ]; then
        rm -f "$path.part"
        die "sha256 mismatch for $url: expected $sha, got $actual"
    fi
    mv -f "$path.part" "$path"
}

# Stage the pinned inputs of $1; an input may be another derived model's output.
stage_inputs() {
    local id="$1" kind ref output sha
    # read merges empty tab-separated columns, so every row names its kind and
    # has no empty field.
    while IFS=$'\t' read -r kind ref output sha; do
        safe_relative "$output"
        if [ "$kind" = url ]; then
            fetch "$ref" "$CACHE/inputs/$id/$output" "$sha"
        else
            prepare "$ref"
            [ "$(sha256_file "$MODELS_DIR/$ref/$output")" = "$sha" ] ||
                die "$ref/$output does not match the digest $id pins"
        fi
    done < <(entry "$id" | jq -r '.inputs[]
        | [(if .url then "url" else "model" end), (.url // .model), .output, .sha256]
        | @tsv')
}

# Rebuild the outputs of $1 into $2.
build() {
    local id="$1" out="$2" inputs="$CACHE/inputs/$1"
    case "$id" in
        cua-s1-forms)
            mkdir -p "$CACHE/reports"
            uv run --no-project --python 3.12 --with-requirements "$HERE/requirements-forms.txt" \
                python "$HERE/export_forms_onnx.py" \
                --cua-src "$inputs" --checkpoint "$inputs/cua-s1-forms.safetensors" \
                --cases "$FIXTURES/cases/forms.json" --out "$out/cua-s1-forms.onnx" \
                --report "$CACHE/reports/cua-s1-forms-export.json" >/dev/null
            echo "  export gates (max |Δ|): $(jq -c .max_abs "$CACHE/reports/cua-s1-forms-export.json")"
            ;;
        qwen3.5-0.8b-f16-cf)
            "$HERE/convert_gguf.sh" f16 "$inputs" "$out/Qwen3.5-0.8B-F16.gguf"
            ;;
        qwen3.5-0.8b-q4km-cf)
            "$HERE/convert_gguf.sh" q4_k_m \
                "$MODELS_DIR/qwen3.5-0.8b-f16-cf/Qwen3.5-0.8B-F16.gguf" \
                "$out/Qwen3.5-0.8B-Q4_K_M.gguf"
            ;;
        *) die "no build recipe for $id" ;;
    esac
}

prepare() {
    local id="$1" out output sha actual dest
    entry "$id" >/dev/null
    if staged "$id"; then
        echo "$id: staged artifacts verified"
        [ "$STAGE_INPUTS" = 0 ] || stage_inputs "$id"
        return 0
    fi
    echo "$id: rebuilding from pinned inputs"
    stage_inputs "$id"
    out="$CACHE/staging/$id"
    rm -rf "$out"
    mkdir -p "$out"
    build "$id" "$out"
    # Verify every output before any of them replaces a staged file.
    while IFS=$'\t' read -r output sha; do
        safe_relative "$output"
        [ -f "$out/$output" ] || die "$id: the build did not produce $output"
        actual="$(sha256_file "$out/$output")"
        [ "$actual" = "$sha" ] ||
            die "$id: rebuilt $output has sha256 $actual, models.json pins $sha"
    done < <(outputs "$id")
    # The cache may be on another filesystem, where mv copies: land each output
    # beside its target, verify that copy, then rename it into place.
    while IFS=$'\t' read -r output sha; do
        dest="$MODELS_DIR/$id/$output"
        mkdir -p "$(dirname "$dest")"
        mv -f "$out/$output" "$dest.part"
        [ "$(sha256_file "$dest.part")" = "$sha" ] || die "$id: $output changed while staging"
        mv -f "$dest.part" "$dest"
    done < <(outputs "$id")
    rm -rf "$out"
    echo "$id: staged and verified"
}

STAGE_INPUTS=0
if [ "${1:-}" = "--inputs" ]; then
    STAGE_INPUTS=1
    shift
fi
[ $# -ge 1 ] || die "usage: $0 [--inputs] <model-id>..."
command -v curl >/dev/null || die "curl is required"
command -v jq >/dev/null || die "jq is required"
for id in "$@"; do
    prepare "$id"
done
