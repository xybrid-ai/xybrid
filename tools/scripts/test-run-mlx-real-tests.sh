#!/usr/bin/env bash
# Regression tests for run-mlx-real-tests.sh: one process per test and then
# all in one process, a crash attributed to its test, the remaining tests
# still run, stale filters fail, and only a GPU rejection is retried.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/run-mlx-real-tests.sh"

if ! command -v jq >/dev/null 2>&1; then
    echo "skip: jq is not installed"
    exit 0
fi

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

# A stand-in for the libtest binary: lists its tests, and runs the ones named
# after `--exact`. `aborts_like_mlx` dies on SIGABRT, as an uncaught MLX
# exception does; `fails_an_assertion` exits the way a failed test does;
# `gpu_rejects` fails its first $FAKE_GPU_REJECTIONS runs with the error MLX
# returns when the GPU fails a command buffer.
FAKE_BIN="$TMP_ROOT/mlx_llm_chat-fake"
cat > "$FAKE_BIN" <<'EOF'
#!/usr/bin/env bash
names=(real_one_passes real_two_aborts_like_mlx real_three_fails_an_assertion real_four_passes
    flaky_gpu_rejects)
if [ "${1:-}" = "--list" ]; then
    filter="${4:-}"
    for name in "${names[@]}"; do
        case "$name" in *"$filter"*) echo "$name: test" ;; esac
    done
    exit 0
fi
if [ "${1:-}" != "--exact" ]; then
    echo "unexpected arguments: $*" >&2
    exit 64
fi
shift
selected=()
while [ "$#" -gt 0 ] && [ "${1#--}" = "$1" ]; do
    selected+=("$1")
    shift
done
echo "ran ${selected[*]}" >> "$FAKE_RAN"
status=0
for name in "${selected[@]}"; do
    case "$name" in
        *aborts_like_mlx) kill -ABRT $$ ;;
        *fails_an_assertion) status=101 ;;
        *gpu_rejects)
            rejected=$(cat "$FAKE_REJECTED" 2>/dev/null || echo 0)
            if [ "$rejected" -lt "${FAKE_GPU_REJECTIONS:-0}" ]; then
                echo $((rejected + 1)) > "$FAKE_REJECTED"
                echo "generation failed: [METAL] Command buffer execution failed:" \
                    "Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)." >&2
                status=101
            fi
            ;;
    esac
done
exit "$status"
EOF
chmod +x "$FAKE_BIN"

export FAKE_RAN="$TMP_ROOT/ran.txt"
export FAKE_REJECTED="$TMP_ROOT/rejected.txt"
export XYBRID_MLX_TEST_BIN="$FAKE_BIN"
# No crash report exists for the stand-in: an empty directory keeps the
# lookup from finding someone else's, and there is nothing to wait for.
export XYBRID_MLX_CRASH_REPORTS="$TMP_ROOT/no-reports"
export XYBRID_MLX_CRASH_REPORT_WAIT=0
mkdir -p "$XYBRID_MLX_CRASH_REPORTS"

fail() {
    echo "FAIL: $1" >&2
    echo "--- output" >&2
    cat "$TMP_ROOT/out.txt" >&2
    exit 1
}

expect_output() {
    grep -qF -- "$1" "$TMP_ROOT/out.txt" || fail "expected output to contain: $1"
}

# 1. Passing tests only: exit 0; each runs in its own process, then both in one.
: > "$FAKE_RAN"
"$RUNNER" llm-mlx-runtime passes > "$TMP_ROOT/out.txt" 2>&1 || fail "passing tests must exit 0"
expect_output "all 2 real-weight tests passed, alone and in one process"
[ "$(cat "$FAKE_RAN")" = $'ran real_one_passes\nran real_four_passes\nran real_one_passes real_four_passes' ] \
    || fail "expected each test alone and then both together, got: $(cat "$FAKE_RAN")"

# 2. An abort and a failure: both are named, the tests after them still run,
#    and the one-process pass reports its own death.
: > "$FAKE_RAN"
if "$RUNNER" llm-mlx-runtime real_ > "$TMP_ROOT/out.txt" 2>&1; then
    fail "a crashing test must fail the run"
fi
[ "$(wc -l < "$FAKE_RAN" | tr -d ' ')" = "5" ] || fail "every test must run despite the abort"
expect_output "real_two_aborts_like_mlx died with signal 6"
expect_output "3 failure(s):"
expect_output "real_two_aborts_like_mlx (signal 6)"
expect_output "real_three_fails_an_assertion (exit 101)"
expect_output "all in one process, run 1 (signal 6)"
expect_output "(no crash report appeared in $XYBRID_MLX_CRASH_REPORTS)"

# 3. Overlapping filters run a test once.
: > "$FAKE_RAN"
"$RUNNER" llm-mlx-runtime real_one passes > "$TMP_ROOT/out.txt" 2>&1 || fail "overlapping filters must pass"
[ "$(head -2 "$FAKE_RAN")" = $'ran real_one_passes\nran real_four_passes' ] \
    || fail "a test matched by two filters must run once, got: $(cat "$FAKE_RAN")"

# 4. The one-process pass repeats on request.
: > "$FAKE_RAN"
XYBRID_MLX_TOGETHER_RUNS=3 "$RUNNER" llm-mlx-runtime passes > "$TMP_ROOT/out.txt" 2>&1 \
    || fail "repeated one-process passes must pass"
[ "$(grep -c '^ran real_one_passes real_four_passes$' "$FAKE_RAN")" = "3" ] \
    || fail "expected three one-process passes, got: $(cat "$FAKE_RAN")"

# 5. A GPU rejection is retried as often as asked, and counted.
: > "$FAKE_RAN"
rm -f "$FAKE_REJECTED"
FAKE_GPU_REJECTIONS=2 XYBRID_MLX_GPU_RETRIES=2 XYBRID_MLX_TOGETHER_RUNS=0 \
    "$RUNNER" llm-mlx-runtime flaky_ > "$TMP_ROOT/out.txt" 2>&1 \
    || fail "a GPU rejection within the retry budget must pass"
[ "$(grep -c '^ran flaky_gpu_rejects$' "$FAKE_RAN")" = "3" ] \
    || fail "expected two retries after two rejections, got: $(cat "$FAKE_RAN")"
expect_output "kIOGPUCommandBufferCallbackErrorOutOfMemory"
expect_output "retry 2 of 2"
expect_output "the GPU rejected a command buffer 2 time(s)"

# 6. Past the budget, and by default, a GPU rejection fails the run.
: > "$FAKE_RAN"
rm -f "$FAKE_REJECTED"
if FAKE_GPU_REJECTIONS=2 XYBRID_MLX_GPU_RETRIES=1 XYBRID_MLX_TOGETHER_RUNS=0 \
    "$RUNNER" llm-mlx-runtime flaky_ > "$TMP_ROOT/out.txt" 2>&1; then
    fail "a GPU rejection past the retry budget must fail the run"
fi
expect_output "flaky_gpu_rejects (exit 101)"
: > "$FAKE_RAN"
rm -f "$FAKE_REJECTED"
if FAKE_GPU_REJECTIONS=1 XYBRID_MLX_TOGETHER_RUNS=0 \
    "$RUNNER" llm-mlx-runtime flaky_ > "$TMP_ROOT/out.txt" 2>&1; then
    fail "without XYBRID_MLX_GPU_RETRIES a GPU rejection must fail the run"
fi
[ "$(grep -c '^ran flaky_gpu_rejects$' "$FAKE_RAN")" = "1" ] \
    || fail "no retry is expected by default, got: $(cat "$FAKE_RAN")"

# 7. Nothing else is retried, whatever the budget: not a crash, not a failure.
for other in aborts_like_mlx fails_an_assertion; do
    : > "$FAKE_RAN"
    if XYBRID_MLX_GPU_RETRIES=5 XYBRID_MLX_TOGETHER_RUNS=0 \
        "$RUNNER" llm-mlx-runtime "$other" > "$TMP_ROOT/out.txt" 2>&1; then
        fail "$other must fail the run"
    fi
    [ "$(grep -c "^ran real_.*_$other\$" "$FAKE_RAN")" = "1" ] \
        || fail "$other must not be retried, got: $(cat "$FAKE_RAN")"
done

# 8. A filter that matches nothing (a renamed test) is an error, not a pass.
: > "$FAKE_RAN"
if "$RUNNER" llm-mlx-runtime passes renamed_away > "$TMP_ROOT/out.txt" 2>&1; then
    fail "a filter that matches no test must fail the run"
fi
expect_output "no test matches the filter 'renamed_away'"
[ ! -s "$FAKE_RAN" ] || fail "no test may run when a filter is stale"

echo "run-mlx-real-tests.sh: all checks passed"
