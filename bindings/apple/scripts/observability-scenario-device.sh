#!/usr/bin/env bash
# observability-scenario-device.sh: run ObservabilityRuntimeTests, the shared
# observability scenario (tools/scripts/observability_capture/scenario.md), on a
# physical iOS device and write the expectations.json it reports to the host.
#
#   observability-scenario-device.sh <endpoint> <expectations-out> <device-udid> <derived-data>
#
# <endpoint> is the capture receiver as the device reaches it: the host's LAN
# address (the device must be on the same network; iOS may ask once for local
# network access). The tests must already be built for the device:
#
#   ./bindings/apple/scripts/set-natives-mode.sh --set-local   # with the xcframework in place; never commit
#   xcodebuild build-for-testing -scheme Xybrid \
#     -destination "platform=iOS,id=<udid>" -derivedDataPath <derived-data> \
#     -allowProvisioningUpdates DEVELOPMENT_TEAM=<team> -skipPackagePluginValidation
#
# then, from the repository root:
#
#   python3 tools/scripts/observability_capture/run_scenario.py --binding swift \
#     --endpoint-host <host LAN address> --out-dir out/swift-device -- \
#     bash bindings/apple/scripts/observability-scenario-device.sh {endpoint} {expectations} <udid> <derived-data>
#
# A device cannot read or write host paths, so the spec goes to the test inline
# (base64) and the test prints expectations.json as one marked line, which this
# script extracts from the xcodebuild output.
set -euo pipefail

ENDPOINT="${1:?endpoint}"
EXPECTATIONS="${2:?expectations-out}"
DEVICE="${3:?device udid}"
DERIVED="${4:?derived data path}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SPEC="$ROOT/tools/scripts/observability_capture/scenario.json"
MARKER="XYBRID_SCENARIO_EXPECTATIONS_BASE64="
OUT_DIR="$(dirname "$EXPECTATIONS")"
LOG="$OUT_DIR/xcodebuild-test.log"
mkdir -p "$OUT_DIR"

status=0
(cd "$ROOT/bindings/apple" && xcodebuild test-without-building \
    -scheme Xybrid \
    -destination "platform=iOS,id=$DEVICE" \
    -derivedDataPath "$DERIVED" \
    -only-testing:XybridTests/ObservabilityRuntimeTests \
    -skipPackagePluginValidation \
    TEST_RUNNER_XYBRID_SCENARIO_DEVICE=1 \
    "TEST_RUNNER_XYBRID_SCENARIO_ENDPOINT=$ENDPOINT" \
    "TEST_RUNNER_XYBRID_SCENARIO_SPEC_BASE64=$(base64 < "$SPEC" | tr -d '\n')") \
    > "$LOG" 2>&1 || status=$?
grep -E "Test Case|Executed|error:|XCTSkip|skipped" "$LOG" | tail -20 || true

line="$(grep -o "${MARKER}[A-Za-z0-9+/=]*" "$LOG" | tail -1 || true)"
if [ -z "$line" ]; then
    echo "error: the test printed no ${MARKER} line (see $LOG)" >&2
    exit 1
fi
printf '%s' "${line#"$MARKER"}" | base64 --decode > "$EXPECTATIONS"
echo "==> Wrote $EXPECTATIONS from the device's test output"
exit "$status"
