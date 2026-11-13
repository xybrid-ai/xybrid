#!/usr/bin/env bash
# capture-golden-tokens.sh — write the Tier A ("self") goldens that
# crates/xybrid-core/tests/mlx_golden_parity.rs compares against.
#
# A golden is the greedy token stream this implementation produces for a
# canonical prompt, on this machine, at this commit. Capture them at a commit
# you trust, then run the parity test while changing decode code: any
# difference means the change altered decode numerics.
#
# Goldens are written to <bundle>/.golden/ next to the staged model, are
# specific to the machine and the linked MLX version, and are never committed.
#
# Tier B goldens (`upstream-greedy-*.json`, captured from Python mlx-lm and
# read by the ignored `golden_upstream_audit` test) are not produced by this
# script. They use the same JSON shape (`GoldenFile` in the test) with
# `tier: "upstream"` and a `margins_top2` list.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  XYBRID_MLX_QWEN_DIR=/path/to/bundle ./tools/scripts/capture-golden-tokens.sh --self

Options:
  --self      Capture Tier A goldens for every staged bundle.
  -h, --help  Show this help.

Bundles are staged through the variables the MLX tests read:
  XYBRID_MLX_QWEN_DIR, XYBRID_MLX_QWEN_4B_DIR, XYBRID_MLX_LFM25_DIR,
  XYBRID_MLX_GEMMA_DIR

Requires an MLX runtime build: Apple Silicon macOS and
vendor/mlx-apple/mlx.xcframework (tools/scripts/build-local-mlx-xcframework.sh).
USAGE
}

MODE=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --self)    MODE=self; shift ;;
        -h|--help) usage; exit 0 ;;
        *)         echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$MODE" != self ]; then
    usage >&2
    exit 2
fi

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "error: goldens are captured with the MLX runtime, which needs Apple Silicon macOS" >&2
    exit 1
fi

staged=0
for var in XYBRID_MLX_QWEN_DIR XYBRID_MLX_QWEN_4B_DIR XYBRID_MLX_LFM25_DIR XYBRID_MLX_GEMMA_DIR; do
    if [ -n "${!var:-}" ]; then
        staged=$((staged + 1))
    fi
done
if [ "$staged" -eq 0 ]; then
    echo "error: no bundle staged; set one of the XYBRID_MLX_*_DIR variables (see --help)" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

commit=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    commit="${commit}-dirty"
    echo "warning: the worktree has uncommitted changes; goldens will record ${commit}" >&2
fi

XYBRID_GOLDEN_COMMIT="$commit" \
XYBRID_GOLDEN_MACHINE="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)" \
XYBRID_GOLDEN_MACOS="$(sw_vers -productVersion 2>/dev/null || echo unknown)" \
XYBRID_GOLDEN_DATE="$(date -u +%Y-%m-%d)" \
    cargo test -p xybrid-core --no-default-features --features llm-mlx-runtime,llm-llamacpp \
    --test mlx_golden_parity -- --ignored --exact capture_self_goldens --nocapture

echo "Goldens written. Verify with:"
echo "  cargo test -p xybrid-core --no-default-features --features llm-mlx-runtime,llm-llamacpp --test mlx_golden_parity"
