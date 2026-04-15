#!/usr/bin/env bash
set -euo pipefail

# Fetch all assets needed to run the NeuTTS Nano Q4 fixture end-to-end.
#
# Downloads:
#   1. GGUF backbone   - neuphonic/neutts-nano-q4-gguf
#   2. ONNX decoder    - neuphonic/neucodec-onnx-decoder-int8 (renamed)
#   3. Reference voices - github.com/neuphonic/neutts/samples (jo, dave)
#
# Voice .pt files are converted to the raw u32+i32*N binary format that
# VoiceFormat::PrecomputedCodes expects. Conversion requires PyTorch
# (one-time, not a runtime dependency).
#
# Usage: ./integration-tests/fetch-neutts-nano.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR="$SCRIPT_DIR/fixtures/models/neutts-nano-q4"
VOICES_DIR="$FIXTURE_DIR/voices"

GGUF_URL="https://huggingface.co/neuphonic/neutts-nano-q4-gguf/resolve/main/neutts-nano-Q4_0.gguf"
DECODER_URL="https://huggingface.co/neuphonic/neucodec-onnx-decoder-int8/resolve/main/model.onnx"
VOICES_BASE="https://raw.githubusercontent.com/neuphonic/neutts/main/samples"
VOICES=(jo dave)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}==>${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC}  $*"; }
warn()  { echo -e "${YELLOW}!${NC}  $*"; }
fatal() { echo -e "${RED}✗  $*${NC}" >&2; exit 1; }

command -v curl >/dev/null || fatal "curl is required"
command -v python3 >/dev/null || fatal "python3 is required"

# Pick a runner for the .pt -> .bin conversion. PyTorch is needed only to
# read the .pt files; we isolate it so we don't fight PEP 668 on macOS.
#
# Preference order:
#   1. uv (fast, ephemeral env, no on-disk state)
#   2. local venv at integration-tests/.venv-neutts (one-time setup, then reused)
VENV_DIR="$SCRIPT_DIR/.venv-neutts"
PY_RUNNER=""

if command -v uv >/dev/null 2>&1; then
    PY_RUNNER="uv-run"
elif python3 -c "import torch" 2>/dev/null; then
    PY_RUNNER="system"
else
    info "PyTorch not found in system Python — setting up local venv"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR" || fatal "failed to create venv at $VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    if ! python3 -c "import torch" 2>/dev/null; then
        info "Installing torch (CPU build) into $VENV_DIR — one-time, ~250 MB"
        python3 -m pip install --quiet --upgrade pip
        python3 -m pip install --quiet torch \
            || fatal "torch install failed. Try:  brew install uv  &&  re-run this script"
    fi
    PY_RUNNER="venv"
fi

run_python() {
    case "$PY_RUNNER" in
        uv-run) uv run --quiet --with torch python "$@" ;;
        venv|system) python3 "$@" ;;
        *) fatal "no python runner configured" ;;
    esac
}

# Resume-friendly download with sanity check.
# Args: url, dest, label, [min_bytes=1024]
download() {
    local url="$1"
    local dest="$2"
    local label="$3"
    local min_bytes="${4:-1024}"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        ok "$label already present: $(basename "$dest")"
        return
    fi

    info "Downloading $label"
    curl --fail --location --progress-bar -C - -o "$dest" "$url" \
        || fatal "Download failed: $url"

    local size
    size=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
    if [ "$size" -lt "$min_bytes" ]; then
        rm -f "$dest"
        fatal "Downloaded file too small ($size bytes < $min_bytes expected), likely an error page: $url"
    fi
    # Catch HTML/error pages even when above min_bytes (rare for tiny txt files).
    if head -c 16 "$dest" 2>/dev/null | grep -qE '^(<!DOCTYPE|<html|404:)'; then
        rm -f "$dest"
        fatal "Downloaded file looks like an HTML error page: $url"
    fi
    ok "$label: $size bytes"
}

mkdir -p "$VOICES_DIR"

[ -f "$FIXTURE_DIR/model_metadata.json" ] \
    || fatal "Fixture metadata missing: $FIXTURE_DIR/model_metadata.json
       This script only fetches the model files. The metadata is committed."

info "Target: $FIXTURE_DIR"
echo

download "$GGUF_URL"    "$FIXTURE_DIR/neutts-nano-Q4_0.gguf"      "GGUF backbone (~195 MB)"
download "$DECODER_URL" "$FIXTURE_DIR/neucodec-decoder-int8.onnx" "NeuCodec decoder (~312 MB)"

echo
info "Fetching reference voices: ${VOICES[*]}"
for voice in "${VOICES[@]}"; do
    download "$VOICES_BASE/$voice.pt"  "$VOICES_DIR/$voice.pt"  "$voice.pt"  1024
    download "$VOICES_BASE/$voice.txt" "$VOICES_DIR/$voice.txt" "$voice.txt" 16
done

echo
info "Converting .pt voice codes to binary format (u32 count + i32 LE values)"
for voice in "${VOICES[@]}"; do
    pt="$VOICES_DIR/$voice.pt"
    bin="$VOICES_DIR/$voice.bin"
    if [ -f "$bin" ] && [ "$bin" -nt "$pt" ]; then
        ok "$voice.bin already up-to-date"
        continue
    fi
    run_python "$SCRIPT_DIR/convert_neutts_voice.py" "$pt" "$bin"
done

echo
info "Verifying fixture contents"
all_present=1
for f in \
    "model_metadata.json" \
    "neutts-nano-Q4_0.gguf" \
    "neucodec-decoder-int8.onnx" \
    "voices/jo.bin" "voices/jo.txt" \
    "voices/dave.bin" "voices/dave.txt"; do
    if [ -f "$FIXTURE_DIR/$f" ]; then
        ok "$f"
    else
        warn "missing: $f"
        all_present=0
    fi
done

echo
if [ $all_present -eq 1 ]; then
    ok "Fixture ready at $FIXTURE_DIR"
    echo
    echo "Run end-to-end:"
    echo "  cargo run --example neutts_tts -p xybrid-core --features llm-llamacpp -- \\"
    echo "    --model-dir integration-tests/fixtures/models/neutts-nano-q4 \\"
    echo "    --text \"Hello from xybrid\" \\"
    echo "    --voice jo \\"
    echo "    --output /tmp/neutts.wav"
else
    fatal "Some files are missing — see warnings above."
fi
