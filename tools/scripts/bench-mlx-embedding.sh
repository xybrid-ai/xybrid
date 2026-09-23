#!/usr/bin/env bash
# bench-mlx-embedding.sh — Run the MLX SafeTensors embedding benchmark with a
# warm-up pass + measurement pass.
#
# Requires: Apple Silicon macOS with `llm-mlx-runtime` available. The Nomic
# fixture is resolved from integration-tests/fixtures/models/nomic-embed-text-v1.5
# or from XYBRID_MLX_NOMIC_DIR. Unsupported hosts and missing fixtures are
# treated as "skip"; strict mode turns skipped/failed runs into a non-zero exit.
#
# Usage:
#   tools/scripts/bench-mlx-embedding.sh [--warmup-only|--measure-only] [--strict]

set -euo pipefail

usage() {
    cat <<'EOF'
bench-mlx-embedding.sh — run the MLX SafeTensors embedding benchmark.
Warm-up pass first (discarded), then measurement pass.

Usage:
  tools/scripts/bench-mlx-embedding.sh [--warmup-only|--measure-only] [--strict]

Options:
  --warmup-only   Run only the throwaway, non-reporting warm-up pass.
  --measure-only  Skip the warm-up pass.
  --strict        Exit non-zero on a skipped or failed benchmark.
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
    skip "bench-mlx-embedding: host is $(uname -s), not Darwin — skipping."
fi

if [ "$(uname -m)" != "arm64" ]; then
    skip "bench-mlx-embedding: host is $(uname -m), not arm64 Apple Silicon — skipping."
fi

FEATURES="llm-mlx-runtime"
BENCH_BIN="mlx_embedding"
CARGO_BENCH=(cargo bench -p xybrid-core --no-default-features --features "$FEATURES" --bench "$BENCH_BIN")

if [ "$WARMUP" = true ]; then
    echo "==> warm-up pass (output discarded)"
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

    REPORT="$REPO_ROOT/target/benchmark-results/mlx_embedding.md"
    if [ -f "$REPORT" ]; then
        echo ""
        echo "report: $REPORT"
    else
        echo "warning: bench did not produce $REPORT" >&2
    fi
fi
