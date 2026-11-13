#!/usr/bin/env bash
# run-mlx-real-tests.sh — run the real-weight MLX tests, one process per test.
#
# MLX reports some failures by throwing where no caller can catch (a Metal
# command buffer that fails throws from its completion handler), and the
# process aborts. Run in one process, the first such test takes the others
# down with it and the log names none of them. Here every test gets its own
# process: a crash is attributed to its test, the remaining tests still run,
# and the backtrace from the macOS crash report is printed so the log shows
# where MLX threw.
#
# Usage:
#   XYBRID_MLX_QWEN_DIR=... XYBRID_MLX_QWEN_4BIT_DIR=... \
#     tools/scripts/run-mlx-real-tests.sh [features [filter...]]
#
# `features` is the cargo feature list (default `llm-mlx-runtime`). Each
# filter is a libtest name filter over crates/xybrid-core/tests/mlx_llm_chat.rs;
# the default set selects the tests that need a Qwen bundle. Set
# XYBRID_MLX_REQUIRE, as test-mlx.yml does, to make an unstaged bundle a
# failure instead of a skip.
#
# XYBRID_MLX_TEST_BIN runs an already-built test binary instead of building.

set -uo pipefail

FEATURES="${1:-llm-mlx-runtime}"
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "$#" -gt 0 ]; then
    FILTERS=("$@")
else
    FILTERS=(real_qwen four_turn_chat streaming_matches_non_streaming qwen35_quantized)
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# Cargo runs a package's tests from the package directory.
TEST_CWD="$REPO_ROOT/crates/xybrid-core"
CRASH_REPORTS="${XYBRID_MLX_CRASH_REPORTS:-$HOME/Library/Logs/DiagnosticReports}"

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required" >&2
    exit 2
fi

SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

echo "== host"
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
echo "memory: $(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576)) MB"
# A hosted runner reports "Apple Paravirtual device" here.
system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Metal" || true

BIN="${XYBRID_MLX_TEST_BIN:-}"
if [ -z "$BIN" ]; then
    echo "== build"
    if ! (cd "$REPO_ROOT" && cargo test -p xybrid-core --no-default-features \
        --features "$FEATURES" --test mlx_llm_chat --no-run \
        --message-format=json-render-diagnostics) > "$SCRATCH/artifacts.json"; then
        echo "error: the mlx_llm_chat test binary did not build" >&2
        exit 1
    fi
    BIN=$(jq -r 'select(.reason == "compiler-artifact" and .target.name == "mlx_llm_chat"
        and .executable != null) | .executable' "$SCRATCH/artifacts.json" | tail -1)
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "error: could not locate the mlx_llm_chat test binary (${BIN:-none})" >&2
    exit 1
fi

# The names come from the binary, so a renamed test cannot silently drop out:
# a filter that matches nothing is an error.
TESTS=()
for filter in "${FILTERS[@]}"; do
    matched=0
    while IFS= read -r line; do
        name="${line%: test}"
        matched=1
        case " ${TESTS[*]:-} " in
            *" $name "*) ;;
            *) TESTS+=("$name") ;;
        esac
    done < <(cd "$TEST_CWD" && "$BIN" --list --format terse "$filter" 2>/dev/null | grep ': test$')
    if [ "$matched" -eq 0 ]; then
        echo "error: no test matches the filter '$filter'" >&2
        exit 1
    fi
done

# Print the crash report the system wrote for a process that died after
# `$1` was created: what it died of, and the crashed thread's backtrace. For
# an uncaught C++ exception that thread still holds the frames that threw.
print_crash_report() {
    local marker="$1" report="" _
    # The report is written a moment after the process dies.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        report=$(find "$CRASH_REPORTS" -maxdepth 1 -name "$(basename "$BIN")*.ips" \
            -newer "$marker" 2>/dev/null | sort | tail -1)
        if [ -n "$report" ]; then
            break
        fi
        sleep 2
    done
    if [ -z "$report" ]; then
        echo "   (no crash report appeared in $CRASH_REPORTS)"
        return
    fi
    echo "   crash report: $report"
    # The first line of an .ips file is a separate JSON header.
    tail -n +2 "$report" | jq -r '
        def frame($images):
            "     " + ($images[.imageIndex].name // "?") + "  "
            + (.symbol // ("+" + (.imageOffset | tostring)));
        .usedImages as $images
        | "   exception: " + ([.exception.type, .exception.signal, .exception.subtype]
                              | map(select(. != null)) | join(" ")),
          (if .asi then "   message: " + ([.asi[] | .[]] | join(" | ")) else empty end),
          (if .lastExceptionBacktrace then
              "   thrown from:", (.lastExceptionBacktrace[:40][] | frame($images))
           else empty end),
          "   crashed thread:",
          ([.threads[] | select(.triggered)][0].frames[:40][] | frame($images))
    ' || echo "   (the crash report could not be read)"
}

failed=()
for test in "${TESTS[@]}"; do
    echo "== $test"
    marker="$SCRATCH/marker"
    : > "$marker"
    (cd "$TEST_CWD" && "$BIN" --exact "$test" --nocapture --test-threads=1)
    status=$?
    if [ "$status" -eq 0 ]; then
        continue
    fi
    # 128 + signal: the process died instead of failing an assertion.
    if [ "$status" -gt 128 ]; then
        failed+=("$test (signal $((status - 128)))")
        echo "-- $test died with signal $((status - 128))"
        print_crash_report "$marker"
    else
        failed+=("$test (exit $status)")
    fi
done

echo "== summary"
if [ "${#failed[@]}" -eq 0 ]; then
    echo "all ${#TESTS[@]} real-weight tests passed"
    exit 0
fi
echo "${#failed[@]} of ${#TESTS[@]} real-weight tests failed:"
printf '  %s\n' "${failed[@]}"
exit 1
