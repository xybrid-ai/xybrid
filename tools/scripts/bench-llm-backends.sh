#!/usr/bin/env bash
# bench-llm-backends.sh — Run the MLX vs llama.cpp decode-throughput
# comparison benchmark with a warm-up pass + measurement pass + threshold
# check.
#
# Requires: macOS + both `llm-mlx-runtime` and `llm-llamacpp` features
# available (i.e. vendor/mlx-apple/mlx.xcframework fetched and the
# llama.cpp build toolchain usable). Missing fixtures are treated as
# "skip" and the corresponding model's threshold assertion is bypassed
# unless XYBRID_BENCH_STRICT=1 is set.
#
# Usage:
#   tools/scripts/bench-llm-backends.sh [--warmup-only|--measure-only] [--strict]
#
# Options:
#   --warmup-only   Run only the throwaway warm-up pass (useful for
#                   populating Metal's kernel cache ahead of a manual run).
#   --measure-only  Skip the warm-up pass; go straight to the measurement.
#   --strict        Pass XYBRID_BENCH_STRICT=1 to the bench — any skipped
#                   or failed cell becomes a non-zero exit.
#   -h, --help      Show this help.

set -euo pipefail

usage() {
    cat <<'EOF'
bench-llm-backends.sh — run the MLX vs llama.cpp decode-throughput
benchmark. Warm-up pass first (discarded), then measurement pass.

Usage:
  tools/scripts/bench-llm-backends.sh [--warmup-only|--measure-only] [--strict]

Options:
  --warmup-only   Run only the throwaway warm-up pass.
  --measure-only  Skip the warm-up pass.
  --strict        Exit non-zero on any skipped or failed cell.
  -h, --help      Show this help.
EOF
}

WARMUP=true
MEASURE=true
STRICT=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --warmup-only)   MEASURE=false; shift ;;
        --measure-only)  WARMUP=false; shift ;;
        --strict)        STRICT=true; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Resolve the repo root (this script lives at tools/scripts/).
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

cd "$REPO_ROOT"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "bench-llm-backends: host is $(uname -s), not Darwin — skipping." >&2
    exit 0
fi

FEATURES="llm-mlx-runtime llm-llamacpp"
BENCH_BIN="llm_backend_compare"

if [ "$WARMUP" = true ]; then
    echo "==> warm-up pass (output discarded)"
    # stderr carries the per-round progress lines; stdout is cargo output
    # that we also want to keep visible. We do NOT capture either because
    # the user should see what's happening.
    cargo bench -p xybrid-core --features "$FEATURES" --bench "$BENCH_BIN" \
        || echo "warm-up exited non-zero (ignored)" >&2
    echo ""
fi

if [ "$MEASURE" = true ]; then
    echo "==> measurement pass"
    if [ "$STRICT" = true ]; then
        export XYBRID_BENCH_STRICT=1
    fi
    cargo bench -p xybrid-core --features "$FEATURES" --bench "$BENCH_BIN"

    REPORT="$REPO_ROOT/target/benchmark-results/llm_backend_compare.md"
    if [ -f "$REPORT" ]; then
        echo ""
        echo "report: $REPORT"
    else
        echo "warning: bench did not produce $REPORT" >&2
    fi
fi
