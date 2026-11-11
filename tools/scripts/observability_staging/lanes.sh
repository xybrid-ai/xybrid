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

# A.5: the CLI as customers run it (key in XYBRID_API_KEY, staging platform):
# one local LLM run, one local pipeline, one cloud stage through the staging
# gateway, each with --trace-export; export-check reads every exported trace
# back from staging with the export as the ground truth.
lane_cli_mac() {
    cargo build --release -p xybrid-cli --features platform-macos
    cli_lane cli-mac "$ROOT/target/release/xybrid" https://staging.api.xybrid.dev --trace-export
}

# A.6: the released v0.17.0 CLI (no OTLP export; legacy telemetry) against
# staging, read back through the legacy trace list and the aggregate. It posts
# its events to <platform-url>/v1/events/batch, which staging serves on the
# ingest host only (the API host answers 404), so the ingest host is its
# platform URL here; its cloud stage then has no gateway and fails.
lane_cli_legacy_mac() {
    local dir="$EV/released-cli-v0.17.0"
    if [ ! -x "$dir/xybrid" ]; then
        mkdir -p "$dir"
        gh release download v0.17.0 -R xybrid-ai/xybrid -p 'xybrid-cli-*macos*arm64*' -p 'xybrid-*-aarch64-apple-darwin*' -D "$dir" --clobber
        (cd "$dir" && for archive in *.tar.gz *.zip; do
            [ -e "$archive" ] || continue
            case "$archive" in *.zip) unzip -o -q "$archive" ;; *) tar -xzf "$archive" ;; esac
        done)
        find "$dir" -name xybrid -type f -perm -u+x -exec cp {} "$dir/xybrid" \; -quit
        shasum -a 256 "$dir"/* > "$dir/sha256.txt" 2>/dev/null || true
    fi
    cli_lane cli-legacy-mac "$dir/xybrid" https://staging.ingest.xybrid.dev
}

cli_lane() {
    local lane="$1" cli="$2" platform="$3" export_flag="${4:-}" out key_file="$HOME/.xybrid-step6/key"
    out="$EV/lanes/$lane/s6-$(date -u +%Y%m%dt%H%M%Sz)-$lane"
    mkdir -p "$out"
    "$cli" --version > "$out/cli-version.txt" 2>&1 || true
    python3 "$HERE/run_staging.py" stats --out "$out/stats-before.json"
    printf 'name: staging-cli-pipeline\nstages:\n  - %s\n' "lfm2.5-230m" > "$out/pipeline.yaml"
    printf 'name: staging-cli-cloud\nstages:\n  - model: %s\n    target: cloud\n    provider: openai\n    max_tokens: 24\n' \
        "gpt-4o-mini" > "$out/cloud.yaml"
    local exports=()
    run_cli() {
        local name="$1" status=0; shift
        local trace=()
        if [ -n "$export_flag" ]; then
            trace=("$export_flag" "$out/$name.json")
            exports+=(--export "$out/$name.json")
        fi
        # The key reaches the CLI as customers pass it (XYBRID_API_KEY), never argv.
        XYBRID_API_KEY="$(cat "$key_file")" "$cli" --platform-url "$platform" \
            ${export_flag:+--observability} run "$@" ${trace[@]+"${trace[@]}"} \
            > "$out/$name.log" 2>&1 || status=$?
        echo "$name exit $status" | tee -a "$out/exits.txt"
    }
    run_cli cli-llm --model lfm2.5-230m --input-text "In one short sentence: what colour is a clear daytime sky?" --max-tokens 24
    run_cli cli-pipeline --config "$out/pipeline.yaml" --input-text "In one short sentence: why is the sea salty?"
    run_cli cli-cloud --config "$out/cloud.yaml" --input-text "In one short sentence: what is a lighthouse?"
    if [ -n "$export_flag" ]; then
        python3 "$HERE/run_staging.py" export-check --out "$out" --stats-before "$out/stats-before.json" \
            ${exports[@]+"${exports[@]}"}
    else
        sleep 30
        python3 "$HERE/run_staging.py" stats --out "$out/stats-after.json" --traces 20
    fi
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
