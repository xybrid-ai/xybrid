#!/usr/bin/env bash
set -euo pipefail

# Xybrid Integration Test Model Downloader
# Downloads models from HuggingFace for integration testing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/fixtures/models"
MANIFEST="$MODELS_DIR/models.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_dependencies() {
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: curl is required but not installed${NC}"
        exit 1
    fi
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}Warning: jq not found, using basic parsing${NC}"
        HAS_JQ=false
    else
        HAS_JQ=true
    fi
}

# List available models from manifest
list_models() {
    echo "Available models:"
    echo ""
    if $HAS_JQ; then
        jq -r '.models | to_entries[] | "  \(.key) (\(.value.size_mb)MB) - \(.value.description)"' "$MANIFEST"
    else
        echo "  wav2vec2-base-960h (380MB) - ASR model"
        echo "  kitten-tts (85MB) - Lightweight TTS"
        echo "  kokoro-82m (330MB) - High-quality TTS"
        echo "  whisper-tiny (150MB) - Whisper ASR"
        echo "  mnist (1MB) - MNIST classification"
    fi
    echo ""
    echo "Usage: $0 [model-name|--all|--list|--check]"
}

# Check which models are present
check_models() {
    echo "Checking models..."
    echo ""

    local models=("wav2vec2-base-960h" "kitten-tts" "kokoro-82m" "whisper-tiny" "mnist")
    local missing=0

    for model in "${models[@]}"; do
        local model_dir="$MODELS_DIR/$model"
        if [ -d "$model_dir" ] && [ -f "$model_dir/model_metadata.json" ]; then
            echo -e "  ${GREEN}✓${NC} $model"
        else
            echo -e "  ${RED}✗${NC} $model (missing)"
            ((missing++))
        fi
    done

    echo ""
    if [ $missing -eq 0 ]; then
        echo -e "${GREEN}All models present!${NC}"
    else
        echo -e "${YELLOW}$missing model(s) missing. Run '$0 --all' to download.${NC}"
    fi
}

# Download a single model from HuggingFace
download_model() {
    local model_name="$1"
    local model_dir="$MODELS_DIR/$model_name"

    # Get model info from manifest
    if $HAS_JQ; then
        local repo=$(jq -r ".models[\"$model_name\"].repo // empty" "$MANIFEST")
        local files=$(jq -r ".models[\"$model_name\"].files[]? // empty" "$MANIFEST")
    else
        # Fallback: hardcoded repos
        case "$model_name" in
            wav2vec2-base-960h) repo="xybrid-ai/wav2vec2-base-960h" ;;
            kitten-tts) repo="xybrid-ai/kitten-tts" ;;
            kokoro-82m) repo="xybrid-ai/kokoro-82m" ;;
            whisper-tiny) repo="xybrid-ai/whisper-tiny" ;;
            mnist) repo="xybrid-ai/mnist" ;;
            *)
                echo -e "${RED}Unknown model: $model_name${NC}"
                return 1
                ;;
        esac
        files=""
    fi

    if [ -z "$repo" ]; then
        echo -e "${RED}Model '$model_name' not found in manifest${NC}"
        return 1
    fi

    echo -e "${YELLOW}Downloading $model_name from $repo...${NC}"
    mkdir -p "$model_dir"

    # Download using HuggingFace Hub URL pattern
    local base_url="https://huggingface.co/$repo/resolve/main"

    # If we have file list from jq, use it; otherwise download common files
    if [ -n "$files" ]; then
        for file in $files; do
            echo "  Downloading $file..."
            curl -L -# -o "$model_dir/$file" "$base_url/$file" || {
                echo -e "${RED}Failed to download $file${NC}"
            }
        done
    else
        # Fallback: try common files
        for file in model.onnx model_metadata.json vocab.json tokens.txt voices.bin; do
            if curl -L -s -f -o "$model_dir/$file" "$base_url/$file" 2>/dev/null; then
                echo "  Downloaded $file"
            fi
        done
    fi

    # Verify download
    if [ -f "$model_dir/model_metadata.json" ]; then
        echo -e "${GREEN}✓ $model_name downloaded successfully${NC}"
    else
        echo -e "${RED}✗ $model_name download incomplete (missing model_metadata.json)${NC}"
        return 1
    fi
}

# Download all models
download_all() {
    echo "Downloading all models..."
    echo ""

    local models=("mnist" "wav2vec2-base-960h" "kitten-tts" "kokoro-82m" "whisper-tiny")
    local failed=0

    for model in "${models[@]}"; do
        download_model "$model" || ((failed++))
        echo ""
    done

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All models downloaded successfully!${NC}"
    else
        echo -e "${YELLOW}$failed model(s) failed to download${NC}"
        exit 1
    fi
}

# Main
check_dependencies

case "${1:-}" in
    --list|-l)
        list_models
        ;;
    --check|-c)
        check_models
        ;;
    --all|-a)
        download_all
        ;;
    --help|-h|"")
        echo "Xybrid Test Model Downloader"
        echo ""
        echo "Usage: $0 [OPTIONS] [model-name]"
        echo ""
        echo "Options:"
        echo "  --list, -l     List available models"
        echo "  --check, -c    Check which models are present"
        echo "  --all, -a      Download all models"
        echo "  --help, -h     Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 --list           # List available models"
        echo "  $0 kitten-tts       # Download specific model"
        echo "  $0 --all            # Download all models"
        ;;
    *)
        download_model "$1"
        ;;
esac
