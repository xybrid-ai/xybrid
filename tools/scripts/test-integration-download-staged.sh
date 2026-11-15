#!/usr/bin/env bash
# Regression tests for integration-tests/download.sh staged MLX fixture checks.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
DOWNLOAD_SCRIPT="$REPO_ROOT/integration-tests/download.sh"

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

assert_file_contains() {
    local file="$1"
    local pattern="$2"
    if ! grep -Eq "$pattern" "$file"; then
        echo "expected $file to contain pattern: $pattern" >&2
        echo "--- $file ---" >&2
        cat "$file" >&2
        exit 1
    fi
}

gemma_dir="$TMP_ROOT/gemma4-2b"
mkdir -p "$gemma_dir"
printf '{}\n' > "$gemma_dir/config.json"
printf '{}\n' > "$gemma_dir/tokenizer.json"
printf '{}\n' > "$gemma_dir/tokenizer_config.json"
printf '{{ messages[0].content }}\n' > "$gemma_dir/chat_template.jinja"
cat > "$gemma_dir/model.safetensors.index.json" <<'JSON'
{
  "metadata": {},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00001.safetensors"
  }
}
JSON

check_output="$TMP_ROOT/check.out"
XYBRID_MLX_GEMMA_DIR="$gemma_dir" "$DOWNLOAD_SCRIPT" --check > "$check_output"
assert_file_contains \
    "$check_output" \
    'gemma4-2b \[staged\].*missing: model-00001-of-00001\.safetensors'

local_models="$TMP_ROOT/local-models"
mkdir -p "$local_models/local-mlx-fixture"
cat > "$local_models/models.json" <<'JSON'
{
  "models": {
    "local-mlx-fixture": {
      "source": "staged",
      "test_env_var": "XYBRID_MLX_LOCAL_FIXTURE_DIR",
      "required_files": ["config.json", "tokenizer.json", "model.safetensors"]
    }
  }
}
JSON
printf '{}\n' > "$local_models/local-mlx-fixture/config.json"
printf '{}\n' > "$local_models/local-mlx-fixture/tokenizer.json"
printf 'weights\n' > "$local_models/local-mlx-fixture/model.safetensors"

local_output="$TMP_ROOT/local.out"
XYBRID_MODELS_DIR="$local_models" "$DOWNLOAD_SCRIPT" --check > "$local_output"
assert_file_contains \
    "$local_output" \
    'local-mlx-fixture \[staged\] via local fixture .*local-mlx-fixture'
assert_file_contains \
    "$local_output" \
    'Staged fixtures:.*1 ready, 0 unset, 0 incomplete'

nomic_dir="$TMP_ROOT/nomic-embed-text-v1.5"
mkdir -p "$nomic_dir"
printf '{}\n' > "$nomic_dir/config.json"
printf '{}\n' > "$nomic_dir/tokenizer.json"

nomic_output="$TMP_ROOT/nomic.out"
XYBRID_MLX_NOMIC_DIR="$nomic_dir" "$DOWNLOAD_SCRIPT" --check > "$nomic_output"
assert_file_contains \
    "$nomic_output" \
    'nomic-embed-text-v1\.5 \[staged\].*missing: model\.safetensors'

hint_output="$TMP_ROOT/nomic-hint.out"
if "$DOWNLOAD_SCRIPT" nomic-embed-text-v1.5 > "$hint_output" 2>&1; then
    echo "expected staged Nomic download request to return non-zero" >&2
    cat "$hint_output" >&2
    exit 1
fi
assert_file_contains \
    "$hint_output" \
    'Test env: XYBRID_MLX_NOMIC_DIR=/path/to/nomic-embed-text-v1\.5'
assert_file_contains \
    "$hint_output" \
    'Required files: config\.json, tokenizer\.json, model\.safetensors'

printf 'integration-tests/download.sh staged fixture tests passed\n'
