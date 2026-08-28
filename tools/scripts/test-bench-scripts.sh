#!/usr/bin/env bash
# Regression tests for benchmark wrapper host-skip and strict-mode behavior.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

FAKE_BIN="$TMP_ROOT/bin"
mkdir -p "$FAKE_BIN"

cat > "$FAKE_BIN/uname" <<'SH'
#!/usr/bin/env bash
case "$1" in
    -s) printf '%s\n' "${FAKE_UNAME_S:?}" ;;
    -m) printf '%s\n' "${FAKE_UNAME_M:?}" ;;
    *) /usr/bin/uname "$@" ;;
esac
SH
chmod +x "$FAKE_BIN/uname"

run_script() {
    local script="$1"
    local os="$2"
    local arch="$3"
    local mode="$4"
    local out="$TMP_ROOT/out.txt"
    shift 4

    set +e
    PATH="$FAKE_BIN:$PATH" \
        FAKE_UNAME_S="$os" \
        FAKE_UNAME_M="$arch" \
        "$REPO_ROOT/$script" "$@" >"$out" 2>&1
    local status=$?
    set -e

    case "$mode" in
        pass)
            if [ "$status" -ne 0 ]; then
                echo "expected $script $* to pass, got $status" >&2
                cat "$out" >&2
                exit 1
            fi
            ;;
        fail)
            if [ "$status" -eq 0 ]; then
                echo "expected $script $* to fail" >&2
                cat "$out" >&2
                exit 1
            fi
            ;;
        *)
            echo "unknown mode: $mode" >&2
            exit 2
            ;;
    esac

    cat "$out"
}

assert_output_contains() {
    local output="$1"
    local pattern="$2"
    if ! grep -Eq "$pattern" <<<"$output"; then
        echo "expected output to contain pattern: $pattern" >&2
        echo "--- output ---" >&2
        printf '%s\n' "$output" >&2
        exit 1
    fi
}

for script in tools/scripts/bench-llm-backends.sh tools/scripts/bench-mlx-embedding.sh; do
    output=$(run_script "$script" Linux x86_64 pass)
    assert_output_contains "$output" 'not Darwin'

    output=$(run_script "$script" Linux x86_64 fail --strict)
    assert_output_contains "$output" 'not Darwin'

    output=$(run_script "$script" Darwin x86_64 pass)
    assert_output_contains "$output" 'not arm64'

    output=$(run_script "$script" Darwin x86_64 fail --strict)
    assert_output_contains "$output" 'not arm64'
done

printf 'benchmark wrapper host-gate tests passed\n'
