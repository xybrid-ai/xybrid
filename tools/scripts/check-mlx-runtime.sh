#!/usr/bin/env bash
# check-mlx-runtime.sh — lint and test the MLX runtime tier on Apple Silicon.
#
# One list of steps for the two places that run it: ci.yml, on a pull request
# that touches the MLX backend, and test-mlx.yml, every night on master. It
# needs vendor/mlx-apple/mlx.xcframework in place (see docs/backends/mlx.md).
#
# The bundles here are small and carry random weights: they prove the
# plumbing. tools/scripts/run-mlx-real-tests.sh runs the real ones.
#
# One failure is retried, and only where XYBRID_MLX_GPU_RETRIES asks for it:
# the GPU rejecting a command buffer, which MLX reports as
# "[METAL] Command buffer execution failed: <Metal's reason>". A hosted
# runner's virtual GPU does that now and then, to any process, and it says
# nothing about the code under test. Every occurrence is printed and counted.
# Any other failure is final, and stops the script at that step.
#
# Usage:
#   tools/scripts/check-mlx-runtime.sh [--benches]
#
# --benches also builds the two MLX benchmarks in the release profile. That
# compiles the whole graph a second time, so only the nightly run asks for it;
# clippy's --all-targets already type-checks them on every run.
#
# XYBRID_MLX_GPU_RETRIES is how often a GPU rejection is retried (default 0).
# XYBRID_MLX_CARGO replaces `cargo` (the script's own test uses it).

set -uo pipefail

BENCHES=false
for arg in "$@"; do
    case "$arg" in
        --benches) BENCHES=true ;;
        -h | --help)
            sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT" || exit 2

CARGO="${XYBRID_MLX_CARGO:-cargo}"
GPU_RETRIES="${XYBRID_MLX_GPU_RETRIES:-0}"
GPU_REJECTION="[METAL] Command buffer execution failed"
RUNTIME=(--no-default-features --features "llm-mlx-runtime,llm-llamacpp")

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT
SUMMARY=()
REJECTIONS=0

group() {
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "::group::$1"
    else
        echo "== $1"
    fi
}

endgroup() {
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "::endgroup::"
    fi
}

# step <retry|once> <name> <command...>
step() {
    local policy="$1" name="$2"
    shift 2
    local attempt=0 status start
    start=$(date +%s)
    while :; do
        group "$name"
        "$@" 2>&1 | tee "$LOG"
        status=${PIPESTATUS[0]}
        endgroup
        if [ "$status" -eq 0 ]; then
            break
        fi
        if [ "$policy" = retry ] && grep -qF -- "$GPU_REJECTION" "$LOG"; then
            REJECTIONS=$((REJECTIONS + 1))
            echo "-- the GPU rejected work during: $name"
            grep -F -- "$GPU_REJECTION" "$LOG" | head -3 | sed 's/^/   /'
            if [ "$attempt" -lt "$GPU_RETRIES" ]; then
                attempt=$((attempt + 1))
                echo "-- retry $attempt of $GPU_RETRIES"
                continue
            fi
        fi
        echo "FAILED: $name (exit $status)" >&2
        exit "$status"
    done
    SUMMARY+=("$(printf '%4ds  %s' "$(($(date +%s) - start))" "$name")")
}

# The two MLX crates on their own, then everything that links them.
step retry "Test xybrid-mlx-sys" "$CARGO" test -p xybrid-mlx-sys --features bindings
step retry "Test xybrid-mlx" "$CARGO" test -p xybrid-mlx --features bindings
step once "Clippy xybrid-mlx-sys" \
    "$CARGO" clippy -p xybrid-mlx-sys --features bindings --all-targets -- -D warnings
step once "Clippy xybrid-mlx" \
    "$CARGO" clippy -p xybrid-mlx --features bindings --all-targets -- -D warnings
step once "Clippy xybrid-core, MLX runtime" \
    "$CARGO" clippy -p xybrid-core "${RUNTIME[@]}" --all-targets -- -D warnings
step retry "Test xybrid-core, MLX runtime" \
    "$CARGO" test -p xybrid-core "${RUNTIME[@]}" --lib
step retry "Test MLX chat" "$CARGO" test -p xybrid-core "${RUNTIME[@]}" --test mlx_llm_chat
step retry "Test MLX embeddings" \
    "$CARGO" test -p xybrid-core "${RUNTIME[@]}" --test mlx_embedding
step retry "Test xybrid-sdk, MLX runtime" "$CARGO" test -p xybrid-sdk "${RUNTIME[@]}" --lib
step retry "Test xybrid-cli, MLX runtime" \
    "$CARGO" test -p xybrid-cli "${RUNTIME[@]}" --bin xybrid

if [ "$BENCHES" = true ]; then
    step once "Build the MLX vs llama.cpp benchmark" \
        "$CARGO" bench -p xybrid-core "${RUNTIME[@]}" --bench llm_backend_compare --no-run
    step once "Build the MLX embedding benchmark" \
        "$CARGO" bench -p xybrid-core "${RUNTIME[@]}" --bench mlx_embedding --no-run
fi

echo "== summary"
printf '%s\n' "${SUMMARY[@]}"
if [ "$REJECTIONS" -gt 0 ]; then
    echo "the GPU rejected work $REJECTIONS time(s); each was retried and passed"
fi
echo "the MLX runtime tier passed"
