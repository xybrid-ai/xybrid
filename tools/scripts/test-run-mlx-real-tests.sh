#!/usr/bin/env bash
# Regression tests for run-mlx-real-tests.sh: one process per test, a crash
# attributed to its test, the remaining tests still run, stale filters fail.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/run-mlx-real-tests.sh"

if ! command -v jq >/dev/null 2>&1; then
    echo "skip: jq is not installed"
    exit 0
fi

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

# A stand-in for the libtest binary: lists three tests, and runs one of them
# by exact name. `aborts_like_mlx` dies on SIGABRT, as an uncaught MLX
# exception does; `fails_an_assertion` exits the way a failed test does.
FAKE_BIN="$TMP_ROOT/mlx_llm_chat-fake"
cat > "$FAKE_BIN" <<'EOF'
#!/usr/bin/env bash
names=(real_one_passes real_two_aborts_like_mlx real_three_fails_an_assertion real_four_passes)
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
echo "ran $2" >> "$FAKE_RAN"
case "$2" in
    *aborts_like_mlx) kill -ABRT $$ ;;
    *fails_an_assertion) exit 101 ;;
    *) exit 0 ;;
esac
EOF
chmod +x "$FAKE_BIN"

export FAKE_RAN="$TMP_ROOT/ran.txt"
export XYBRID_MLX_TEST_BIN="$FAKE_BIN"
# No crash report exists for the stand-in; an empty directory keeps the
# lookup from finding someone else's.
export XYBRID_MLX_CRASH_REPORTS="$TMP_ROOT/no-reports"
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

# 1. Passing tests only: exit 0, each run once, in its own process.
: > "$FAKE_RAN"
"$RUNNER" llm-mlx-runtime passes > "$TMP_ROOT/out.txt" 2>&1 || fail "passing tests must exit 0"
expect_output "all 2 real-weight tests passed"
[ "$(cat "$FAKE_RAN")" = $'ran real_one_passes\nran real_four_passes' ] \
    || fail "expected exactly the two passing tests to run, got: $(cat "$FAKE_RAN")"

# 2. An abort and a failure: both are named, and the tests after them still run.
: > "$FAKE_RAN"
if "$RUNNER" llm-mlx-runtime real_ > "$TMP_ROOT/out.txt" 2>&1; then
    fail "a crashing test must fail the run"
fi
[ "$(wc -l < "$FAKE_RAN" | tr -d ' ')" = "4" ] || fail "every test must run despite the abort"
expect_output "real_two_aborts_like_mlx died with signal 6"
expect_output "2 of 4 real-weight tests failed:"
expect_output "real_two_aborts_like_mlx (signal 6)"
expect_output "real_three_fails_an_assertion (exit 101)"

# 3. Overlapping filters run a test once.
: > "$FAKE_RAN"
"$RUNNER" llm-mlx-runtime real_one passes > "$TMP_ROOT/out.txt" 2>&1 || fail "overlapping filters must pass"
[ "$(cat "$FAKE_RAN")" = $'ran real_one_passes\nran real_four_passes' ] \
    || fail "a test matched by two filters must run once, got: $(cat "$FAKE_RAN")"

# 4. A filter that matches nothing (a renamed test) is an error, not a pass.
: > "$FAKE_RAN"
if "$RUNNER" llm-mlx-runtime passes renamed_away > "$TMP_ROOT/out.txt" 2>&1; then
    fail "a filter that matches no test must fail the run"
fi
expect_output "no test matches the filter 'renamed_away'"
[ ! -s "$FAKE_RAN" ] || fail "no test may run when a filter is stale"

echo "run-mlx-real-tests.sh: all checks passed"
