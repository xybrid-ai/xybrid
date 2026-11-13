#!/usr/bin/env bash
# observability-scenario.sh: run an observability scenario through the Kotlin
# wrapper on the connected device or emulator, and copy the expectations.json
# it wrote back to the host.
#
#   observability-scenario.sh <endpoint> <expectations-out>
#   observability-scenario.sh staging <config> <expectations-out>
#
# The first form runs ObservabilityScenarioTest, the shared observability
# scenario (tools/scripts/observability_capture/scenario.md). <endpoint> is the
# capture receiver as the device reaches it (http://10.0.2.2:<port> from an
# emulator). Both arguments default to the XYBRID_SCENARIO_ENDPOINT /
# _EXPECTATIONS variables run_scenario.py sets, which fills the placeholders
# below and then runs check_capture.py:
#
#   python3 tools/scripts/observability_capture/run_scenario.py --binding kotlin \
#     --endpoint-host 10.0.2.2 --out-dir out/kotlin -- \
#     bash bindings/kotlin/consumer-smoke/observability-scenario.sh {endpoint} {expectations}
#
# The second form runs ObservabilityStagingScenarioTest, the step 6 staging
# scenario (tools/scripts/observability_staging/README.md) with a registry
# model, on a physical device. <config> is the config.json run_staging.py
# writes. It holds the staging key, so it is streamed straight into the app's
# private files with `adb exec-in` and `run-as` (no copy on shared storage);
# the test deletes it once read, and this script deletes any copy a failed
# launch left. When XYBRID_STAGING_COLLECTOR_PORT is set (run_staging.py sets
# it), `adb reverse` forwards that port for the run, so the collector session's
# http://127.0.0.1:<port> reaches the host's Collector. Both arguments default
# to the XYBRID_STAGING_CONFIG / _EXPECTATIONS variables run_staging.py sets:
#
#   python3 tools/scripts/observability_staging/run_staging.py run --lane kotlin-pixel -- \
#     bash bindings/kotlin/consumer-smoke/observability-scenario.sh staging {config} {expectations}
#
# Installs the APKs built beforehand against the AAR in app/libs/:
#
#   bindings/kotlin/gradlew -p bindings/kotlin/consumer-smoke \
#     :app:assembleDebug :app:assembleDebugAndroidTest
#
# The instrumentation output and a logcat dump (for the staging scenario, the
# app's own process only) land next to <expectations-out>. Exits non-zero
# unless the test passed and its expectations.json was copied.
set -euo pipefail

usage() {
    echo "usage: $0 <endpoint> <expectations-out>" >&2
    echo "       $0 staging <config> <expectations-out>" >&2
    exit 2
}

ADB="${ADB:-adb}"
HERE="$(cd "$(dirname "$0")" && pwd)"
APP_ID=ai.xybrid.smoke
RUNNER="$APP_ID.test/androidx.test.runner.AndroidJUnitRunner"
if [ "${1:-}" = staging ]; then
    STAGING=1
    CONFIG="${2:-${XYBRID_STAGING_CONFIG:-}}"
    EXPECTATIONS="${3:-${XYBRID_STAGING_EXPECTATIONS:-}}"
    if [ -z "$CONFIG" ] || [ -z "$EXPECTATIONS" ]; then
        usage
    fi
    if [ ! -f "$CONFIG" ]; then
        echo "error: $CONFIG is missing" >&2
        exit 2
    fi
    TEST_CLASS=ai.xybrid.smoke.ObservabilityStagingScenarioTest
    # Relative to the app's data directory, where `run-as` starts.
    DEVICE_DIR=files/xybrid-staging
else
    STAGING=
    ENDPOINT="${1:-${XYBRID_SCENARIO_ENDPOINT:-}}"
    EXPECTATIONS="${2:-${XYBRID_SCENARIO_EXPECTATIONS:-}}"
    if [ -z "$ENDPOINT" ] || [ -z "$EXPECTATIONS" ]; then
        usage
    fi
    TEST_CLASS=ai.xybrid.smoke.ObservabilityScenarioTest
    DEVICE_DIR=files/observability-scenario
fi
DEVICE_EXPECTATIONS="$DEVICE_DIR/expectations.json"
DEVICE_CONFIG="$DEVICE_DIR/config.json"
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
# A reinstall keeps app data: drop a previous run's expectations.json (and,
# for the staging scenario, a config a failed launch left).
"$ADB" shell run-as "$APP_ID" rm -rf "$DEVICE_DIR" || true
"$ADB" logcat -c || true

if [ -n "$STAGING" ]; then
    PORT="${XYBRID_STAGING_COLLECTOR_PORT:-}"
    # However this script ends, the key does not stay on the device and the
    # forward does not outlive the run.
    cleanup() {
        "$ADB" shell run-as "$APP_ID" rm -f "$DEVICE_CONFIG" || true
        if [ -n "$PORT" ]; then
            "$ADB" reverse --remove "tcp:$PORT" || true
        fi
    }
    trap cleanup EXIT
    if [ -n "$PORT" ]; then
        echo "==> Forwarding the device's 127.0.0.1:$PORT to the host's Collector"
        "$ADB" reverse "tcp:$PORT" "tcp:$PORT"
    fi

    echo "==> Copying the config into the app's private files"
    # exec-in pipes stdin to the device command, raw, and single-quotes every
    # word after `run-as`, so `sh -c` gets the script as one argument.
    "$ADB" exec-in run-as "$APP_ID" sh -c "umask 077 && mkdir -p $DEVICE_DIR && cat > $DEVICE_CONFIG" < "$CONFIG"
    # exec-in reports no remote status and can return before `cat` is done:
    # wait for the whole file.
    want=$(( $(wc -c < "$CONFIG") ))
    got=
    for _ in 1 2 3 4 5; do
        got="$("$ADB" shell run-as "$APP_ID" stat -c %s "$DEVICE_CONFIG" 2>/dev/null | tr -d '\r' || true)"
        [ "$got" = "$want" ] && break
        sleep 1
    done
    if [ "$got" != "$want" ]; then
        echo "error: $DEVICE_CONFIG has ${got:-no} bytes on the device, expected $want" >&2
        exit 1
    fi

    echo "==> Running $TEST_CLASS"
    status=0
    "$ADB" shell am instrument -w -r -e class "$TEST_CLASS" "$RUNNER" | tee "$RESULTS" || status=$?
else
    echo "==> Running $TEST_CLASS against $ENDPOINT"
    status=0
    "$ADB" shell am instrument -w -r -e class "$TEST_CLASS" -e xybridScenarioEndpoint "$ENDPOINT" "$RUNNER" \
        | tee "$RESULTS" || status=$?
fi

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
if [ -n "$STAGING" ]; then
    # Only the app's own process: run_staging.py scans this directory for key
    # material, and other apps' logs have no place in the evidence.
    APP_UID="$("$ADB" shell run-as "$APP_ID" id -u | tr -d '\r' || true)"
    "$ADB" logcat -d --uid="$APP_UID" > "$OUT_DIR/logcat.txt" || true
else
    "$ADB" logcat -d > "$OUT_DIR/logcat.txt" || true
fi

# `am instrument` exits 0 whatever the test did. Require the one test to pass:
# a negative status code is a failure, an error or a skipped assumption (for
# example a missing endpoint argument or config), and a crashed process prints
# no OK line.
if ! grep -q '^OK (1 test)' "$RESULTS" || grep -q 'INSTRUMENTATION_STATUS_CODE: -' "$RESULTS"; then
    echo "error: $TEST_CLASS did not pass; see $RESULTS and $OUT_DIR/logcat.txt" >&2
    status=1
fi
exit "$status"
