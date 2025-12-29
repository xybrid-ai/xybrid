#!/usr/bin/env bash
set -euo pipefail

# Xybrid Integration Test Model Downloader
# Supports two download sources:
#   - registry: Downloads from xybrid registry (api.xybrid.dev)
#   - url: Downloads directly from URLs (GitHub, HuggingFace, etc.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/fixtures/models"
MANIFEST="$MODELS_DIR/models.json"
REGISTRY_API="https://api.xybrid.dev"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for required tools
check_dependencies() {
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: curl is required but not installed${NC}"
        exit 1
    fi
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: jq is required but not installed${NC}"
        echo "Install with: brew install jq (macOS) or apt install jq (Linux)"
        exit 1
    fi
}

# List available models from manifest
list_models() {
    echo "Available models:"
    echo ""
    echo -e "${BLUE}Registry models (via xybrid registry):${NC}"
    jq -r '.models | to_entries[] | select(.value.source == "registry") | "  \(.key) (\(.value.size_mb)MB) - \(.value.description)"' "$MANIFEST"
    echo ""
    echo -e "${BLUE}Direct URL models:${NC}"
    jq -r '.models | to_entries[] | select(.value.source == "url") | "  \(.key) (\(.value.size_mb)MB) - \(.value.description)"' "$MANIFEST"
    echo ""
    echo "Usage: $0 [model-name|--all|--list|--check]"
}

# Check which models are present
check_models() {
    echo "Checking models..."
    echo ""

    local models
    models=$(jq -r '.models | keys[]' "$MANIFEST")
    local missing=0
    local present=0

    for model in $models; do
        local model_dir="$MODELS_DIR/$model"
        local source
        source=$(jq -r ".models[\"$model\"].source" "$MANIFEST")

        # Check for model.onnx or model_metadata.json
        if [ -d "$model_dir" ] && { [ -f "$model_dir/model_metadata.json" ] || [ -f "$model_dir/model.onnx" ]; }; then
            echo -e "  ${GREEN}✓${NC} $model [$source]"
            ((present++))
        else
            echo -e "  ${RED}✗${NC} $model [$source] (missing)"
            ((missing++))
        fi
    done

    echo ""
    if [ $missing -eq 0 ]; then
        echo -e "${GREEN}All $present models present!${NC}"
    else
        echo -e "${YELLOW}$missing model(s) missing, $present present. Run '$0 --all' to download.${NC}"
    fi
}

# Validate that a downloaded file is not an error page
validate_file() {
    local file="$1"
    local min_size="${2:-100}"

    if [ ! -f "$file" ]; then
        return 1
    fi

    local size
    size=$(wc -c < "$file" | tr -d ' ')

    if [ "$size" -lt "$min_size" ]; then
        # Check if it's an error message
        if grep -qi "invalid\|error\|not found\|404\|unauthorized" "$file" 2>/dev/null; then
            return 1
        fi
    fi
    return 0
}

# Download model from xybrid registry
download_from_registry() {
    local model_name="$1"
    local model_dir="$MODELS_DIR/$model_name"

    echo -e "${YELLOW}Downloading $model_name from xybrid registry...${NC}"

    # Try to resolve from registry
    local resolve_url="$REGISTRY_API/v1/models/$model_name/resolve"
    local resolve_response

    if ! resolve_response=$(curl -sf "$resolve_url" 2>/dev/null); then
        echo -e "${RED}✗ Failed to resolve $model_name from registry${NC}"
        echo "  Registry may be unavailable or model not found"
        return 1
    fi

    # Extract bundle URL from response
    local bundle_url
    bundle_url=$(echo "$resolve_response" | jq -r '.bundle_url // .url // empty')

    if [ -z "$bundle_url" ]; then
        echo -e "${RED}✗ No bundle URL found for $model_name${NC}"
        return 1
    fi

    mkdir -p "$model_dir"

    # Download and extract bundle
    local bundle_file="$model_dir/bundle.xyb"
    echo "  Downloading bundle..."

    if curl -L -# -f -o "$bundle_file" "$bundle_url"; then
        echo "  Extracting bundle..."
        # .xyb files are zip archives
        if unzip -q -o "$bundle_file" -d "$model_dir" 2>/dev/null; then
            rm -f "$bundle_file"
            echo -e "${GREEN}✓ $model_name downloaded successfully${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to extract bundle${NC}"
            rm -rf "$model_dir"
            return 1
        fi
    else
        echo -e "${RED}✗ Failed to download bundle${NC}"
        rm -rf "$model_dir"
        return 1
    fi
}

# Download model from direct URLs
download_from_url() {
    local model_name="$1"
    local model_dir="$MODELS_DIR/$model_name"

    echo -e "${YELLOW}Downloading $model_name from direct URLs...${NC}"
    mkdir -p "$model_dir"

    local download_failed=false

    # Download each file
    local files
    files=$(jq -c ".models[\"$model_name\"].files[]" "$MANIFEST")

    while IFS= read -r file_entry; do
        local url
        local output
        url=$(echo "$file_entry" | jq -r '.url')
        output=$(echo "$file_entry" | jq -r '.output')

        echo "  Downloading $output..."
        if curl -L -# -f -o "$model_dir/$output" "$url" 2>/dev/null; then
            if validate_file "$model_dir/$output"; then
                echo -e "  ${GREEN}✓${NC} $output"
            else
                echo -e "  ${RED}✗${NC} $output (invalid response)"
                rm -f "$model_dir/$output"
                download_failed=true
            fi
        else
            echo -e "  ${RED}✗${NC} $output (download failed)"
            download_failed=true
        fi
    done <<< "$files"

    # Generate model_metadata.json if defined in manifest
    local has_metadata
    has_metadata=$(jq -r ".models[\"$model_name\"].model_metadata // empty" "$MANIFEST")

    if [ -n "$has_metadata" ]; then
        echo "  Generating model_metadata.json..."
        jq ".models[\"$model_name\"].model_metadata" "$MANIFEST" > "$model_dir/model_metadata.json"
        echo -e "  ${GREEN}✓${NC} model_metadata.json"
    fi

    # Verify download
    if [ "$download_failed" = true ]; then
        echo -e "${RED}✗ $model_name download incomplete${NC}"
        return 1
    fi

    # Check we have at least model.onnx
    if [ -f "$model_dir/model.onnx" ]; then
        echo -e "${GREEN}✓ $model_name downloaded successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ $model_name missing model.onnx${NC}"
        return 1
    fi
}

# Download a single model (auto-detect source)
download_model() {
    local model_name="$1"

    # Check if model exists in manifest
    local model_entry
    model_entry=$(jq -r ".models[\"$model_name\"] // empty" "$MANIFEST")

    if [ -z "$model_entry" ]; then
        echo -e "${RED}Unknown model: $model_name${NC}"
        echo "Run '$0 --list' to see available models"
        return 1
    fi

    # Get source type
    local source
    source=$(jq -r ".models[\"$model_name\"].source" "$MANIFEST")

    case "$source" in
        registry)
            download_from_registry "$model_name"
            ;;
        url)
            download_from_url "$model_name"
            ;;
        *)
            echo -e "${RED}Unknown source type: $source${NC}"
            return 1
            ;;
    esac
}

# Download all models
download_all() {
    echo "Downloading all models..."
    echo ""

    local models
    models=$(jq -r '.models | keys[]' "$MANIFEST")
    local failed=0
    local succeeded=0

    for model in $models; do
        download_model "$model" && ((succeeded++)) || ((failed++))
        echo ""
    done

    echo "========================================"
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All $succeeded models downloaded successfully!${NC}"
    else
        echo -e "${YELLOW}$succeeded succeeded, $failed failed${NC}"
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
        echo "Download sources:"
        echo "  registry  - Downloads from xybrid registry (api.xybrid.dev)"
        echo "  url       - Downloads directly from URLs"
        echo ""
        echo "Examples:"
        echo "  $0 --list           # List available models"
        echo "  $0 mnist            # Download MNIST model (direct URL)"
        echo "  $0 kitten-tts       # Download from registry"
        echo "  $0 --all            # Download all models"
        ;;
    *)
        download_model "$1"
        ;;
esac
