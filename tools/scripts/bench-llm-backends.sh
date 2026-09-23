#!/usr/bin/env bash
# bench-llm-backends.sh — Run the MLX vs llama.cpp decode-throughput
# comparison benchmark with a warm-up pass + measurement pass + parity
# check.
#
# Requires: Apple Silicon macOS + both `llm-mlx-runtime` and `llm-llamacpp`
# features available (i.e. vendor/mlx-apple/mlx.xcframework fetched and
# the llama.cpp build toolchain usable). Unsupported hosts and missing fixtures
# are treated as "skip"; informational pairs bypass parity assertions, while
# strict mode still turns skipped/failed cells into a non-zero exit.
#
# Usage:
#   tools/scripts/bench-llm-backends.sh [--warmup-only|--measure-only] [--strict]
#
# Options:
#   --warmup-only   Run only the throwaway, non-reporting warm-up pass
#                   (useful for populating Metal's kernel cache ahead of
#                   a manual run).
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
  --warmup-only   Run only the throwaway, non-reporting warm-up pass.
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

skip() {
    echo "$1" >&2
    if [ "$STRICT" = true ]; then
        exit 1
    fi
    exit 0
}

if [ "$(uname -s)" != "Darwin" ]; then
    skip "bench-llm-backends: host is $(uname -s), not Darwin — skipping."
fi

if [ "$(uname -m)" != "arm64" ]; then
    skip "bench-llm-backends: host is $(uname -m), not arm64 Apple Silicon — skipping."
fi

FEATURES="llm-mlx-runtime llm-llamacpp"
BENCH_BIN="llm_backend_compare"
CARGO_BENCH=(cargo bench -p xybrid-core --no-default-features --features "$FEATURES" --bench "$BENCH_BIN")

if [ "$WARMUP" = true ]; then
    echo "==> warm-up pass (output discarded)"
    # stderr carries the per-round progress lines; stdout is cargo output
    # that we also want to keep visible. We do NOT capture either because
    # the user should see what's happening.
    env XYBRID_BENCH_WARMUP_ONLY=1 \
        "${CARGO_BENCH[@]}" \
        || echo "warm-up exited non-zero (ignored)" >&2
    echo ""
fi

if [ "$MEASURE" = true ]; then
    echo "==> measurement pass"
    if [ "$STRICT" = true ]; then
        export XYBRID_BENCH_STRICT=1
    fi
    "${CARGO_BENCH[@]}"

    REPORT="$REPO_ROOT/target/benchmark-results/llm_backend_compare.md"
    if [ -f "$REPORT" ]; then
        echo ""
        echo "report: $REPORT"
    else
        echo "warning: bench did not produce $REPORT" >&2
    fi
fi
