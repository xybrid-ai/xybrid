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

# Detect current platform
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$os" in
        darwin)
            case "$arch" in
                arm64) echo "macos-arm64" ;;
                x86_64) echo "macos-x64" ;;
                *) echo "macos" ;;
            esac
            ;;
        linux)
            case "$arch" in
                aarch64) echo "linux-arm64" ;;
                x86_64) echo "linux-x64" ;;
                *) echo "linux" ;;
            esac
            ;;
        *)
            echo "universal"
            ;;
    esac
}

# Download model from xybrid registry
download_from_registry() {
    local model_name="$1"
    local model_dir="$MODELS_DIR/$model_name"
    local platform
    platform=$(detect_platform)

    echo -e "${YELLOW}Downloading $model_name from xybrid registry...${NC}"
    echo "  Platform: $platform"

    # Resolve model from registry
    local resolve_url="$REGISTRY_API/v1/models/registry/$model_name/resolve?platform=$platform"
    local resolve_response

    if ! resolve_response=$(curl -sf "$resolve_url" 2>/dev/null); then
        echo -e "${RED}✗ Failed to resolve $model_name from registry${NC}"
        echo "  Registry may be unavailable or model not found"
        echo "  Tried: $resolve_url"
        return 1
    fi

    # Extract download URL from response
    local download_url
    local file_name
    local size_bytes
    download_url=$(echo "$resolve_response" | jq -r '.resolved.download_url // empty')
    file_name=$(echo "$resolve_response" | jq -r '.resolved.file // empty')
    size_bytes=$(echo "$resolve_response" | jq -r '.resolved.size_bytes // 0')

    if [ -z "$download_url" ]; then
        echo -e "${RED}✗ No download URL found for $model_name${NC}"
        echo "  Response: $resolve_response"
        return 1
    fi

    local size_mb=$((size_bytes / 1024 / 1024))
    echo "  File: $file_name (~${size_mb}MB)"

    mkdir -p "$model_dir"

    # Download the bundle
    local bundle_file="$model_dir/$file_name"
    echo "  Downloading..."

    if curl -L -# -f -o "$bundle_file" "$download_url"; then
        echo "  Extracting bundle..."

        # Detect archive type by magic bytes
        local magic
        magic=$(xxd -l 4 -p "$bundle_file" 2>/dev/null)

        local extract_success=false

        case "$magic" in
            28b52ffd)
                # Zstandard compressed tar archive
                if command -v zstd &> /dev/null; then
                    if zstd -d "$bundle_file" -o "$model_dir/bundle.tar" 2>/dev/null && \
                       tar -xf "$model_dir/bundle.tar" -C "$model_dir" 2>/dev/null; then
                        rm -f "$model_dir/bundle.tar"
                        extract_success=true
                    fi
                else
                    echo -e "${RED}✗ zstd not installed. Install with: brew install zstd${NC}"
                    rm -rf "$model_dir"
                    return 1
                fi
                ;;
            504b0304)
                # ZIP archive
                if unzip -q -o "$bundle_file" -d "$model_dir" 2>/dev/null; then
                    extract_success=true
                fi
                ;;
            1f8b08*)
                # Gzip compressed tar archive
                if tar -xzf "$bundle_file" -C "$model_dir" 2>/dev/null; then
                    extract_success=true
                fi
                ;;
            *)
                echo -e "${RED}✗ Unknown archive format (magic: $magic)${NC}"
                rm -rf "$model_dir"
                return 1
                ;;
        esac

        rm -f "$bundle_file"

        if $extract_success; then
            # Verify extraction
            if [ -f "$model_dir/model_metadata.json" ] || [ -f "$model_dir/model.onnx" ]; then
                echo -e "${GREEN}✓ $model_name downloaded successfully${NC}"
                return 0
            else
                echo -e "${RED}✗ Bundle extracted but missing expected files${NC}"
                ls -la "$model_dir"
                return 1
            fi
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

    # Check if it's an archive download or file list
    local is_archive
    is_archive=$(jq -r ".models[\"$model_name\"].archive // false" "$MANIFEST")

    if [ "$is_archive" = "true" ]; then
        local archive_url
        local archive_strip
        archive_url=$(jq -r ".models[\"$model_name\"].archive_url" "$MANIFEST")
        archive_strip=$(jq -r ".models[\"$model_name\"].archive_strip // empty" "$MANIFEST")

        echo "  Downloading archive from $archive_url..."
        local archive_file="$model_dir/archive.tar.bz2"
        
        if curl -L -# -f -o "$archive_file" "$archive_url"; then
            echo "  Extracting..."
            if tar -xf "$archive_file" -C "$model_dir" 2>/dev/null; then
                rm -f "$archive_file"
                
                # If we need to strip a directory (mv subdir/* .)
                if [ -n "$archive_strip" ] && [ -d "$model_dir/$archive_strip" ]; then
                    mv "$model_dir/$archive_strip"/* "$model_dir/" 2>/dev/null
                    rmdir "$model_dir/$archive_strip" 2>/dev/null
                fi
                echo -e "  ${GREEN}✓${NC} Archive extracted"
            else
                echo -e "  ${RED}✗${NC} Extraction failed"
                rm -f "$archive_file"
                return 1
            fi
        else
            echo -e "  ${RED}✗${NC} Archive download failed"
            rm -f "$archive_file"
            rm -rf "$model_dir"
            return 1
        fi
    else
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
    fi

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
