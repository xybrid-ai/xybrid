#!/usr/bin/env bash
# lanes.sh: one subcommand per staging lane (README, "Running a lane"). Each
# builds or stages the candidate for its runtime, then hands run_staging.py the
# command that copies config.json to the runtime, launches the scenario and
# copies expectations.json back. Run from anywhere in the checkout, on the Mac:
#
#   tools/scripts/observability_staging/lanes.sh <lane> [run_staging.py run options]
#   tools/scripts/observability_staging/lanes.sh list
#
# Environment:
#   XYBRID_STEP6_EVIDENCE   evidence root (default ~/xybrid-acceptance-evidence/2026-11-11/step6);
#                           the frozen CI artifacts are in its artifacts/ directory
#   XYBRID_STEP6_CANDIDATE  the SDK commit under test (default 00e61710...)
#   XYBRID_IPHONE_UDID, XYBRID_APPLE_TEAM, XYBRID_MAC_LAN_IP  device lanes
#
# The staging key never passes through this script: run_staging.py reads it
# from ~/.xybrid-step6/key and writes it only into the private config.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
EV="${XYBRID_STEP6_EVIDENCE:-$HOME/xybrid-acceptance-evidence/2026-11-11/step6}"
ART="$EV/artifacts"
CANDIDATE="${XYBRID_STEP6_CANDIDATE:-00e617102bc77ca0602115669cf62a68e56a92bb}"
UDID="${XYBRID_IPHONE_UDID:-00008130-001C446E1408001C}"
TEAM="${XYBRID_APPLE_TEAM:-CPVMT4DS8N}"
LAN_IP="${XYBRID_MAC_LAN_IP:-$(ipconfig getifaddr en0 2>/dev/null || true)}"
RUN=(python3 "$HERE/run_staging.py" run --commit "$CANDIDATE" --out-root "$EV/lanes")
cd "$ROOT"

lane_python_mac() {
    # The CPython bridge and cdylib built from this checkout
    # (XYBRID_FEATURES=platform-macos tools/scripts/build-python-bolt.sh).
    test -n "$(ls bindings/python/xybrid/_bolt/_native*.so 2>/dev/null)" \
        || { echo "error: build the Python bridge first (tools/scripts/build-python-bolt.sh)" >&2; exit 2; }
    PYTHONPATH="$ROOT/bindings/python" "${RUN[@]}" --lane python-mac "$@" -- \
        python3 bindings/python/tests/observability_staging.py --config {config} --expectations {expectations}
}

lane_rust_mac() {
    cargo build --release -p xybrid-sdk --features platform-macos,dev-tools --example observability_acceptance
    "${RUN[@]}" --lane rust-mac "$@" -- \
        target/release/examples/observability_acceptance --config {config} --expectations {expectations}
}

lanes() {
    declare -F | awk '{print $3}' | sed -n 's/^lane_//p' | tr '_' '-'
}

case "${1:-}" in
    ""|-h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    list) lanes; exit 0 ;;
esac
lane="$1"; shift
function="lane_${lane//-/_}"
if ! declare -F "$function" >/dev/null; then
    echo "error: unknown lane $lane; lanes: $(lanes | tr '\n' ' ')" >&2
    exit 2
fi
"$function" "$@"
