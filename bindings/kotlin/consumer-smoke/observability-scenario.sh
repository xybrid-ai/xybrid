#!/usr/bin/env bash
# observability-scenario.sh: run ObservabilityScenarioTest, the shared
# observability scenario (tools/scripts/observability_capture/scenario.md)
# through the Kotlin wrapper, on the connected device or emulator, and copy the
# expectations.json it wrote back to the host.
#
#   observability-scenario.sh <endpoint> <expectations-out>
#
# <endpoint> is the capture receiver as the device reaches it
# (http://10.0.2.2:<port> from an emulator). Both arguments default to the
# XYBRID_SCENARIO_ENDPOINT / _EXPECTATIONS variables run_scenario.py sets, which
# fills the placeholders below and then runs check_capture.py:
#
#   python3 tools/scripts/observability_capture/run_scenario.py --binding kotlin \
#     --endpoint-host 10.0.2.2 --out-dir out/kotlin -- \
#     bash bindings/kotlin/consumer-smoke/observability-scenario.sh {endpoint} {expectations}
#
# Installs the APKs built beforehand against the AAR in app/libs/:
#
#   bindings/kotlin/gradlew -p bindings/kotlin/consumer-smoke \
#     :app:assembleDebug :app:assembleDebugAndroidTest
#
# The instrumentation output and a logcat dump land next to <expectations-out>.
# Exits non-zero unless the test passed and its expectations.json was copied.
set -euo pipefail

ENDPOINT="${1:-${XYBRID_SCENARIO_ENDPOINT:-}}"
EXPECTATIONS="${2:-${XYBRID_SCENARIO_EXPECTATIONS:-}}"
if [ -z "$ENDPOINT" ] || [ -z "$EXPECTATIONS" ]; then
    echo "usage: $0 <endpoint> <expectations-out>" >&2
    exit 2
fi

ADB="${ADB:-adb}"
HERE="$(cd "$(dirname "$0")" && pwd)"
APP_ID=ai.xybrid.smoke
RUNNER="$APP_ID.test/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS=ai.xybrid.smoke.ObservabilityScenarioTest
# Relative to the app's data directory, where `run-as` starts.
DEVICE_EXPECTATIONS=files/observability-scenario/expectations.json
APK="$HERE/app/build/outputs/apk/debug/app-debug.apk"
TEST_APK="$HERE/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
OUT_DIR="$(dirname "$EXPECTATIONS")"
RESULTS="$OUT_DIR/instrumentation.txt"
mkdir -p "$OUT_DIR"

for apk in "$APK" "$TEST_APK"; do
    if [ ! -f "$apk" ]; then
        echo "error: $apk is missing; build both APKs first (see the header)" >&2
        exit 2
    fi
done

echo "==> Installing the consumer and its instrumentation APK"
"$ADB" wait-for-device
"$ADB" install -r -t "$APK"
"$ADB" install -r -t "$TEST_APK"
# A reinstall keeps app data: drop a previous run's expectations.json.
"$ADB" shell run-as "$APP_ID" rm -rf "$(dirname "$DEVICE_EXPECTATIONS")" || true
"$ADB" logcat -c || true

echo "==> Running $TEST_CLASS against $ENDPOINT"
status=0
"$ADB" shell am instrument -w -r -e class "$TEST_CLASS" -e xybridScenarioEndpoint "$ENDPOINT" "$RUNNER" \
    | tee "$RESULTS" || status=$?

# The test writes expectations.json even when a step failed, so copy it
# whatever the instrumentation reported: the checker then names the failure.
rm -f "$EXPECTATIONS"
if "$ADB" exec-out run-as "$APP_ID" cat "$DEVICE_EXPECTATIONS" > "$EXPECTATIONS.tmp" \
    && [ -s "$EXPECTATIONS.tmp" ]; then
    mv "$EXPECTATIONS.tmp" "$EXPECTATIONS"
    echo "==> Copied $DEVICE_EXPECTATIONS to $EXPECTATIONS"
else
    rm -f "$EXPECTATIONS.tmp"
    echo "error: could not copy $DEVICE_EXPECTATIONS off the device" >&2
    status=1
fi
"$ADB" logcat -d > "$OUT_DIR/logcat.txt" || true

# `am instrument` exits 0 whatever the test did. Require the one test to pass:
# a negative status code is a failure, an error or a skipped assumption (for
# example a missing endpoint argument), and a crashed process prints no OK line.
if ! grep -q '^OK (1 test)' "$RESULTS" || grep -q 'INSTRUMENTATION_STATUS_CODE: -' "$RESULTS"; then
    echo "error: $TEST_CLASS did not pass; see $RESULTS and $OUT_DIR/logcat.txt" >&2
    status=1
fi
exit "$status"
