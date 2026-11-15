#!/usr/bin/env bash
# Tests check-mlx-runtime.sh with a stand-in for cargo: which steps run, that
# only a GPU rejection is retried, and that any other failure stops the run.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/check-mlx-runtime.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

# The stand-in records each invocation. FAKE_FAIL names the step to fail (a
# substring of its arguments), FAKE_MESSAGE is what it prints when it does,
# and FAKE_FAILURES is how many times it fails before it passes.
FAKE_CARGO="$TMP_ROOT/cargo"
cat > "$FAKE_CARGO" <<'FAKE'
#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS"
if [ -n "${FAKE_FAIL:-}" ] && [[ "$*" == *"$FAKE_FAIL"* ]]; then
    count=0
    [ -f "$FAKE_COUNT" ] && count=$(cat "$FAKE_COUNT")
    if [ "$count" -lt "${FAKE_FAILURES:-1}" ]; then
        echo $((count + 1)) > "$FAKE_COUNT"
        echo "${FAKE_MESSAGE:-boom}"
        exit 101
    fi
fi
echo "ok: $*"
FAKE
chmod +x "$FAKE_CARGO"

run_case() { # name, expected exit, then env assignments and runner arguments
    local name="$1" expected="$2"
    shift 2
    export FAKE_CALLS="$TMP_ROOT/$name.calls" FAKE_COUNT="$TMP_ROOT/$name.count"
    OUT="$TMP_ROOT/$name.out"
    set +e
    env -u GITHUB_ACTIONS XYBRID_MLX_CARGO="$FAKE_CARGO" "$@" > "$OUT" 2>&1
    local status=$?
    set -e
    if [ "$status" -ne "$expected" ]; then
        echo "$name: expected exit $expected, got $status" >&2
        cat "$OUT" >&2
        exit 1
    fi
}

expect() { # name, pattern, file
    if ! grep -qF -- "$2" "$3"; then
        echo "$1: expected to find: $2" >&2
        cat "$3" >&2
        exit 1
    fi
}

refuse() { # name, pattern, file
    if grep -qF -- "$2" "$3"; then
        echo "$1: did not expect: $2" >&2
        cat "$3" >&2
        exit 1
    fi
}

REJECTION="[METAL] Command buffer execution failed: Caused GPU Timeout Error"

# 1. Every step runs, in the runtime tier's feature set; no benchmarks.
run_case all 0 "$RUNNER"
expect all "test -p xybrid-mlx --features bindings" "$FAKE_CALLS"
expect all "clippy -p xybrid-core --no-default-features --features llm-mlx-runtime,llm-llamacpp --all-targets -- -D warnings" "$FAKE_CALLS"
expect all "--test mlx_llm_chat" "$FAKE_CALLS"
expect all "test -p xybrid-cli" "$FAKE_CALLS"
refuse all "bench" "$FAKE_CALLS"
expect all "the MLX runtime tier passed" "$OUT"

# 2. --benches adds the two release builds.
run_case benches 0 "$RUNNER" --benches
expect benches "--bench llm_backend_compare --no-run" "$FAKE_CALLS"
expect benches "--bench mlx_embedding --no-run" "$FAKE_CALLS"

# 3. A GPU rejection within the budget is retried, and counted.
run_case retried 0 env FAKE_FAIL="--test mlx_llm_chat" FAKE_MESSAGE="$REJECTION" \
    FAKE_FAILURES=2 XYBRID_MLX_GPU_RETRIES=2 "$RUNNER"
expect retried "retry 2 of 2" "$OUT"
expect retried "the GPU rejected work 2 time(s)" "$OUT"
expect retried "test -p xybrid-cli" "$FAKE_CALLS"

# 4. Past the budget it fails, at that step.
run_case exhausted 101 env FAKE_FAIL="--test mlx_llm_chat" FAKE_MESSAGE="$REJECTION" \
    FAKE_FAILURES=3 XYBRID_MLX_GPU_RETRIES=2 "$RUNNER"
expect exhausted "FAILED: Test MLX chat" "$OUT"
refuse exhausted "test -p xybrid-cli" "$FAKE_CALLS"

# 5. No retry unless it is asked for.
run_case no_budget 101 env FAKE_FAIL="--test mlx_llm_chat" FAKE_MESSAGE="$REJECTION" "$RUNNER"
refuse no_budget "retry 1" "$OUT"

# 6. Any other failure is final, whatever the budget.
run_case other 101 env FAKE_FAIL="--test mlx_llm_chat" FAKE_MESSAGE="assertion failed" \
    XYBRID_MLX_GPU_RETRIES=2 "$RUNNER"
refuse other "retry 1" "$OUT"
expect other "FAILED: Test MLX chat" "$OUT"

# 7. A lint failure is never retried, even if its output names the rejection.
run_case lint 101 env FAKE_FAIL="clippy -p xybrid-core" FAKE_MESSAGE="$REJECTION" \
    XYBRID_MLX_GPU_RETRIES=2 "$RUNNER"
refuse lint "retry 1" "$OUT"

# 8. An unknown argument is refused before anything runs.
run_case usage 2 "$RUNNER" --nope
[ ! -s "$FAKE_CALLS" ] || { echo "usage: nothing should have run" >&2; exit 1; }

echo "check-mlx-runtime.sh behaves as documented"
