#!/usr/bin/env bash
set -euo pipefail

# Xybrid Integration Test Model Downloader
# Supports two download sources:
#   - registry: Downloads from xybrid registry (registry.xybrid.dev)
#   - url: Downloads directly from URLs (GitHub, HuggingFace, etc.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${XYBRID_MODELS_DIR:-$SCRIPT_DIR/fixtures/models}"
MANIFEST="$MODELS_DIR/models.json"
REGISTRY_API="https://registry.xybrid.dev"

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
    echo -e "${BLUE}Staged models (not downloaded by --all):${NC}"
    jq -r '
      .models
      | to_entries[]
      | select(.value.source == "staged")
      | "  \(.key) (\(.value.status // "staged"))"
        + (if .value.test_env_var then " env: \(.value.test_env_var)" else "" end)
        + " - \(.value.description)"
    ' "$MANIFEST"
    echo ""
    echo "Usage: $0 [model-name|--all|--list|--check]"
}

staged_required_files() {
    local model="$1"
    jq -r ".models[\"$model\"].required_files[]?" "$MANIFEST"
}

staged_missing_files() {
    local model="$1"
    local model_dir="$2"
    local missing=""
    local file

    append_missing() {
        local name="$1"
        if [ -n "$missing" ]; then
            missing="$missing, $name"
        else
            missing="$name"
        fi
    }

    while IFS= read -r file; do
        if [ -n "$file" ] && [ ! -f "$model_dir/$file" ]; then
            append_missing "$file"
        fi
    done < <(staged_required_files "$model")

    local index="$model_dir/model.safetensors.index.json"
    if [ -f "$index" ]; then
        local shards
        if ! shards=$(jq -r '.weight_map // empty | values[]?' "$index" 2>/dev/null); then
            append_missing "model.safetensors.index.json (invalid)"
            echo "$missing"
            return
        fi
        if [ -z "$shards" ]; then
            append_missing "model.safetensors.index.json weight_map"
        fi
        while IFS= read -r file; do
            [ -z "$file" ] && continue
            case "$file" in
                /*|..|../*|*/..|*/../*)
                    append_missing "unsafe shard path: $file"
                    continue
                    ;;
            esac
            if [ ! -f "$model_dir/$file" ]; then
                append_missing "$file"
            fi
        done <<< "$shards"
    fi

    echo "$missing"
}

print_staged_fixture_hint() {
    local model="$1"
    local env_var
    local required
    env_var=$(jq -r ".models[\"$model\"].test_env_var // empty" "$MANIFEST")
    required=$(jq -r ".models[\"$model\"].required_files // [] | join(\", \")" "$MANIFEST")

    if [ -n "$env_var" ]; then
        echo "  Test env: $env_var=/path/to/$model"
    fi
    if [ -n "$required" ]; then
        echo "  Required files: $required"
    fi
}

# Check which models are present
check_models() {
    echo "Checking models..."
    echo ""

    local models
    models=$(jq -r '.models | keys[]' "$MANIFEST")
    local missing=0
    local present=0
    local staged_present=0
    local staged_unset=0
    local staged_incomplete=0

    for model in $models; do
        local model_dir="$MODELS_DIR/$model"
        local source
        source=$(jq -r ".models[\"$model\"].source" "$MANIFEST")

        if [ "$source" = "staged" ]; then
            local env_var
            env_var=$(jq -r ".models[\"$model\"].test_env_var // empty" "$MANIFEST")
            if [ -n "$env_var" ]; then
                local staged_dir="${!env_var:-}"
                if [ -n "$staged_dir" ]; then
                    local missing_files
                    missing_files=$(staged_missing_files "$model" "$staged_dir")
                    if [ -z "$missing_files" ]; then
                        echo -e "  ${GREEN}✓${NC} $model [staged] via $env_var=$staged_dir"
                        staged_present=$((staged_present + 1))
                    else
                        echo -e "  ${YELLOW}•${NC} $model [staged] via $env_var=$staged_dir (missing: $missing_files)"
                        staged_incomplete=$((staged_incomplete + 1))
                    fi
                elif [ -d "$model_dir" ]; then
                    local missing_files
                    missing_files=$(staged_missing_files "$model" "$model_dir")
                    if [ -z "$missing_files" ]; then
                        echo -e "  ${GREEN}✓${NC} $model [staged] via local fixture $model_dir"
                        staged_present=$((staged_present + 1))
                    else
                        echo -e "  ${YELLOW}•${NC} $model [staged] via local fixture $model_dir (missing: $missing_files)"
                        staged_incomplete=$((staged_incomplete + 1))
                    fi
                else
                    echo -e "  ${YELLOW}•${NC} $model [staged] set $env_var=/path/to/$model"
                    staged_unset=$((staged_unset + 1))
                fi
            else
                echo -e "  ${YELLOW}•${NC} $model [staged] (not downloaded by --all)"
                staged_unset=$((staged_unset + 1))
            fi
            continue
        fi

        # Check for model.onnx or model_metadata.json
        if [ -d "$model_dir" ] && { [ -f "$model_dir/model_metadata.json" ] || [ -f "$model_dir/model.onnx" ]; }; then
            echo -e "  ${GREEN}✓${NC} $model [$source]"
            present=$((present + 1))
        else
            echo -e "  ${RED}✗${NC} $model [$source] (missing)"
            missing=$((missing + 1))
        fi
    done

    echo ""
    if [ $missing -eq 0 ]; then
        echo -e "${GREEN}All $present downloadable models present!${NC}"
    else
        echo -e "${YELLOW}$missing model(s) missing, $present present. Run '$0 --all' to download.${NC}"
    fi
    if [ $staged_present -gt 0 ] || [ $staged_unset -gt 0 ] || [ $staged_incomplete -gt 0 ]; then
        echo -e "${BLUE}Staged fixtures:${NC} $staged_present ready, $staged_unset unset, $staged_incomplete incomplete"
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
    local resolve_url="$REGISTRY_API/v1/models/$model_name/resolve?platform=$platform"
    local resolve_response
    local http_code
    local temp_file
    temp_file=$(mktemp)

    # Capture both response body and HTTP status code
    http_code=$(curl -s -w "%{http_code}" -o "$temp_file" "$resolve_url" 2>/dev/null)
    resolve_response=$(cat "$temp_file")
    rm -f "$temp_file"

    if [ "$http_code" != "200" ]; then
        echo -e "${RED}✗ Failed to resolve $model_name from registry${NC}"
        echo "  HTTP Status: $http_code"
        case "$http_code" in
            000)
                echo "  Error: Could not connect to registry (network error or DNS failure)"
                ;;
            404)
                echo "  Error: Model '$model_name' not found in registry"
                ;;
            401|403)
                echo "  Error: Unauthorized - check authentication"
                ;;
            500|502|503)
                echo "  Error: Registry server error"
                ;;
            *)
                echo "  Error: Unexpected status code"
                ;;
        esac
        if [ -n "$resolve_response" ]; then
            # Try to extract error message from JSON response
            local error_msg
            error_msg=$(echo "$resolve_response" | jq -r '.error // .message // empty' 2>/dev/null)
            if [ -n "$error_msg" ]; then
                echo "  Message: $error_msg"
            else
                echo "  Response: $resolve_response"
            fi
        fi
        echo "  URL: $resolve_url"
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
            # Fix permissions (some archives have restrictive permissions)
            chmod -R u+rw "$model_dir" 2>/dev/null || true

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

    # Check we have at least one model file (onnx, gguf, safetensors, etc.)
    local model_file_found=false
    for ext in onnx gguf safetensors bin; do
        if ls "$model_dir"/*."$ext" 1>/dev/null 2>&1; then
            model_file_found=true
            break
        fi
    done

    if $model_file_found; then
        echo -e "${GREEN}✓ $model_name downloaded successfully${NC}"
        return 0
    elif [ -f "$model_dir/model_metadata.json" ]; then
        # Some models might only have metadata + other files
        echo -e "${GREEN}✓ $model_name downloaded successfully (metadata only)${NC}"
        return 0
    else
        echo -e "${RED}✗ $model_name missing model file (no .onnx, .gguf, .safetensors, or .bin found)${NC}"
        ls -la "$model_dir" 2>/dev/null || true
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
        staged)
            local notes
            notes=$(jq -r ".models[\"$model_name\"].notes // \"This fixture is staged and not downloadable yet.\"" "$MANIFEST")
            echo -e "${YELLOW}$model_name is staged and is not downloaded by this script.${NC}"
            print_staged_fixture_hint "$model_name"
            echo "  $notes"
            return 1
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
    models=$(jq -r '.models | to_entries[] | select(.value.source != "staged") | .key' "$MANIFEST")
    local failed=0
    local succeeded=0

    for model in $models; do
        if download_model "$model"; then
            succeeded=$((succeeded + 1))
        else
            failed=$((failed + 1))
        fi
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
        echo "  registry  - Downloads from xybrid registry (registry.xybrid.dev)"
        echo "  url       - Downloads directly from URLs"
        echo "  staged    - Local/manual fixture; skipped by --all and surfaced through its test env var"
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
