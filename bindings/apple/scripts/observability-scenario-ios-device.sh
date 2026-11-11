#!/usr/bin/env bash
# observability-scenario-ios-device.sh: run the shared observability scenario
# (tools/scripts/observability_capture/scenario.md) through the public Swift API
# on a physical iPhone, inside the example app.
#
# A Swift package's test target cannot run on a physical device (Xcode needs a
# host app), so this script temporarily appends the scenario implementation from
# bindings/apple/Tests/XybridTests/ObservabilityRuntimeTests.swift (the single
# source of the Swift scenario) to examples/ios, with a launch hook, builds and
# installs that app, and restores the example's files. A run then launches the
# app with the scenario's environment and reads expectations.json back from the
# app's console output.
#
# From the repository root, with the xcframework in place and the package in
# local mode (./bindings/apple/scripts/set-natives-mode.sh --set-local; never
# commit that):
#
#   bindings/apple/scripts/observability-scenario-ios-device.sh build <udid> [team]
#   python3 tools/scripts/observability_capture/run_scenario.py --binding swift \
#     --endpoint-host <this Mac's LAN address> --out-dir out/swift-iphone -- \
#     bash bindings/apple/scripts/observability-scenario-ios-device.sh run {endpoint} {expectations} <udid>
#
# The iPhone must be unlocked and on the same network as the Mac. The first run
# triggers iOS's local network prompt and fails to export until it is allowed:
# allow it and run again.
#
# The step 6 staging scenario (tools/scripts/observability_staging/README.md)
# runs the same way from ObservabilityStagingTests.swift, in its own derived
# data. Its config.json holds the staging key, so it never goes into the launch
# environment: run-staging copies it into the app's data container, where the
# scenario deletes it once read, then launches the app and decodes
# expectations.json from the console:
#
#   bindings/apple/scripts/observability-scenario-ios-device.sh build-staging <udid> [team]
#   python3 tools/scripts/observability_staging/run_staging.py run --lane swift-iphone \
#     --collector-host <this Mac's LAN address> -- \
#     bash bindings/apple/scripts/observability-scenario-ios-device.sh run-staging {config} {expectations} <udid>
#
# Both builds install the same app (bundle id), so run the lane built last.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
APP_DIR="$ROOT/examples/ios/XybridExample"
PROJECT="$ROOT/examples/ios/XybridExample.xcodeproj"
TEST="$ROOT/bindings/apple/Tests/XybridTests/ObservabilityRuntimeTests.swift"
SPEC="$ROOT/tools/scripts/observability_capture/scenario.json"
DERIVED="${XYBRID_IOS_DEVICE_DERIVED_DATA:-${TMPDIR:-/tmp}/xybrid-observability-ios-device}"
MARKER="XYBRID_SCENARIO_EXPECTATIONS_BASE64="
STAGING_TEST="$ROOT/bindings/apple/Tests/XybridTests/ObservabilityStagingTests.swift"
STAGING_DERIVED="${XYBRID_IOS_DEVICE_STAGING_DERIVED_DATA:-${TMPDIR:-/tmp}/xybrid-observability-staging-ios-device}"
STAGING_MARKER="XYBRID_STAGING_EXPECTATIONS_BASE64="
# Inside the app's data container; the launch hook reads it from there.
STAGING_CONFIG="Documents/xybrid-staging/config.json"

build() {
    local udid="${1:?udid}" team="${2:-}"
    local entry="$APP_DIR/XybridExampleApp.swift" plist="$APP_DIR/Info.plist"
    # One build at a time: a second build would back up the injected files.
    local lock="$APP_DIR/.observability-scenario.lock" backup
    if ! mkdir "$lock" 2>/dev/null; then
        echo "error: another device build is using examples/ios (remove $lock if it is stale)" >&2
        exit 1
    fi
    backup="$(mktemp -d)"
    cp "$entry" "$backup/XybridExampleApp.swift"; cp "$plist" "$backup/Info.plist"
    # Expanded now: the trap runs after this function's locals are gone.
    # shellcheck disable=SC2064
    trap "cp -f '$backup/XybridExampleApp.swift' '$entry'; cp -f '$backup/Info.plist' '$plist'; rm -rf '$backup' '$lock'" EXIT
    python3 - "$TEST" "$entry" "$plist" <<'PY'
import sys
test, entry, plist = sys.argv[1:]
source = open(test).read()
scenario = source[source.index("// MARK: - Scenario"):]
hook = '''

// Injected by bindings/apple/scripts/observability-scenario-ios-device.sh, never
// committed: runs the shared observability scenario when launched with
// XYBRID_SCENARIO_DEVICE=1, prints expectations.json as one marked line, exits.
enum XybridScenarioDeviceHook {
    static func runIfRequested() {
        let env = ProcessInfo.processInfo.environment
        guard env["XYBRID_SCENARIO_DEVICE"] == "1", let endpoint = env["XYBRID_SCENARIO_ENDPOINT"],
              let encoded = env["XYBRID_SCENARIO_SPEC_BASE64"], let spec = Data(base64Encoded: encoded)
        else { return }
        let local = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("xybrid-observability-scenario", isDirectory: true)
        try? FileManager.default.removeItem(at: local)
        try? FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        let specURL = local.appendingPathComponent("scenario.json")
        try? spec.write(to: specURL)
        let expectations = local.appendingPathComponent("expectations.json")
        let scenario = ObservabilityScenario(
            endpoint: endpoint, workdir: local.appendingPathComponent("work", isDirectory: true))
        let error = scenario.run(spec: specURL)
        try? scenario.write(to: expectations)
        if let data = try? Data(contentsOf: expectations) {
            print("XYBRID_SCENARIO_EXPECTATIONS_BASE64=" + data.base64EncodedString())
        }
        if let error { print("XYBRID_SCENARIO_ERROR=\\(error)") }
        fflush(stdout)
        exit(0)
    }
}
'''
s = open(entry).read()
s = s.replace("import SwiftUI\n", "import Foundation\nimport SwiftUI\nimport Xybrid\n", 1)
s = s.replace("struct XybridExampleApp: App {\n",
              "struct XybridExampleApp: App {\n    init() { XybridScenarioDeviceHook.runIfRequested() }\n\n", 1)
open(entry, "w").write(s + hook + "\n" + scenario)
p = open(plist).read()
p = p.replace("\t<key>NSCameraUsageDescription</key>",
              "\t<key>NSLocalNetworkUsageDescription</key>\n"
              "\t<string>Sends test telemetry to a receiver on this network.</string>\n"
              "\t<key>NSCameraUsageDescription</key>", 1)
open(plist, "w").write(p)
PY
    local args=(-project "$PROJECT" -scheme XybridExample -destination "platform=iOS,id=$udid"
                -derivedDataPath "$DERIVED" -allowProvisioningUpdates -skipPackagePluginValidation)
    [ -n "$team" ] && args+=("DEVELOPMENT_TEAM=$team")
    xcodebuild "${args[@]}" build
    local app
    app="$(find "$DERIVED/Build/Products" -maxdepth 2 -name XybridExample.app -path '*iphoneos*' | head -1)"
    xcrun devicectl device install app --device "$udid" "$app"
    /usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$app/Info.plist" > "$DERIVED/bundle-id"
}

run() {
    local endpoint="${1:?endpoint}" expectations="${2:?expectations}" udid="${3:?udid}"
    local out log bundle env_json line status=0
    out="$(dirname "$expectations")"; mkdir -p "$out"; log="$out/device-console.log"
    bundle="$(cat "$DERIVED/bundle-id")"
    env_json="$(python3 -c 'import base64, json, sys
print(json.dumps({"XYBRID_SCENARIO_DEVICE": "1", "XYBRID_SCENARIO_ENDPOINT": sys.argv[1],
                  "XYBRID_SCENARIO_SPEC_BASE64": base64.b64encode(open(sys.argv[2], "rb").read()).decode()}))' \
        "$endpoint" "$SPEC")"
    xcrun devicectl device process launch --device "$udid" --console --terminate-existing \
        --environment-variables "$env_json" "$bundle" > "$log" 2>&1 || status=$?
    grep -E "XYBRID_SCENARIO_ERROR" "$log" || true
    line="$(grep -o "${MARKER}[A-Za-z0-9+/=]*" "$log" | tail -1 || true)"
    if [ -z "$line" ]; then
        echo "error: the app printed no ${MARKER} line (exit $status; see $log)" >&2
        tail -15 "$log" >&2
        exit 1
    fi
    printf '%s' "${line#"$MARKER"}" | base64 --decode > "$expectations"
    echo "==> Wrote $expectations from the device console"
    # The app always exits 0 (devicectl does not report its status); a scenario
    # step that failed is reported on the console, so fail here as well.
    if grep -q "XYBRID_SCENARIO_ERROR" "$log"; then
        echo "error: the scenario reported a failed step (see $log)" >&2
        exit 1
    fi
    if [ "$status" -ne 0 ]; then
        echo "error: devicectl exited with $status (see $log)" >&2
        exit "$status"
    fi
}

build_staging() {
    local udid="${1:?udid}" team="${2:-}"
    local entry="$APP_DIR/XybridExampleApp.swift" plist="$APP_DIR/Info.plist"
    # The same lock as build: both inject into examples/ios.
    local lock="$APP_DIR/.observability-scenario.lock" backup
    if ! mkdir "$lock" 2>/dev/null; then
        echo "error: another device build is using examples/ios (remove $lock if it is stale)" >&2
        exit 1
    fi
    backup="$(mktemp -d)"
    cp "$entry" "$backup/XybridExampleApp.swift"; cp "$plist" "$backup/Info.plist"
    # Expanded now: the trap runs after this function's locals are gone.
    # shellcheck disable=SC2064
    trap "cp -f '$backup/XybridExampleApp.swift' '$entry'; cp -f '$backup/Info.plist' '$plist'; rm -rf '$backup' '$lock'" EXIT
    python3 - "$STAGING_TEST" "$entry" "$plist" <<'PY'
import sys
test, entry, plist = sys.argv[1:]
source = open(test).read()
scenario = source[source.index("// MARK: - Staging scenario"):]
hook = r'''

// Injected by bindings/apple/scripts/observability-scenario-ios-device.sh
// build-staging, never committed. Launched with XYBRID_STAGING_DEVICE=1, it
// reads the config run-staging copied to Documents/xybrid-staging/config.json
// (the scenario deletes it once read), runs the staging scenario, prints
// expectations.json as one marked line and exits. XYBRID_STAGING_DEVICE=prepare
// only creates that folder, which devicectl may not create, and exits.
enum XybridStagingDeviceHook {
    static func runIfRequested() {
        guard let mode = ProcessInfo.processInfo.environment["XYBRID_STAGING_DEVICE"] else { return }
        let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("xybrid-staging", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        guard mode == "1" else { exit(0) }
        // Minutes of download, inference and export: off the main thread so the
        // launch completes, with the screen kept on (a locked phone suspends the app).
        DispatchQueue.main.async { UIApplication.shared.isIdleTimerDisabled = true }
        let worker = Thread {
            var failure: String?
            do {
                let scenario = try ObservabilityStagingScenario(configAt: folder.appendingPathComponent("config.json"))
                failure = scenario.run()
                let expectations = try scenario.expectations()
                print("XYBRID_STAGING_EXPECTATIONS_BASE64=" + expectations.base64EncodedString())
            } catch {
                failure = "\(type(of: error)): \(error)"
            }
            if let failure { print("XYBRID_STAGING_ERROR=\(failure)") }
            fflush(stdout)
            exit(0)
        }
        worker.stackSize = 8 << 20  // a model loads and runs on this thread
        worker.start()
    }
}
'''
s = open(entry).read()
for old, new in (("import SwiftUI\n", "import Foundation\nimport SwiftUI\nimport UIKit\nimport Xybrid\n"),
                 ("struct XybridExampleApp: App {\n",
                  "struct XybridExampleApp: App {\n    init() { XybridStagingDeviceHook.runIfRequested() }\n\n")):
    if old not in s:
        sys.exit(f"error: {entry} has no {old!r} to inject the staging hook at")
    s = s.replace(old, new, 1)
open(entry, "w").write(s + hook + "\n" + scenario)
p = open(plist).read()
p = p.replace("\t<key>NSCameraUsageDescription</key>",
              "\t<key>NSLocalNetworkUsageDescription</key>\n"
              "\t<string>Sends test telemetry to a Collector on this network.</string>\n"
              "\t<key>NSCameraUsageDescription</key>", 1)
open(plist, "w").write(p)
PY
    local args=(-project "$PROJECT" -scheme XybridExample -destination "platform=iOS,id=$udid"
                -derivedDataPath "$STAGING_DERIVED" -allowProvisioningUpdates -skipPackagePluginValidation)
    [ -n "$team" ] && args+=("DEVELOPMENT_TEAM=$team")
    xcodebuild "${args[@]}" build
    local app
    app="$(find "$STAGING_DERIVED/Build/Products" -maxdepth 2 -name XybridExample.app -path '*iphoneos*' | head -1)"
    xcrun devicectl device install app --device "$udid" "$app"
    /usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$app/Info.plist" > "$STAGING_DERIVED/bundle-id"
}

# Copies a file to STAGING_CONFIG in the app's data container.
stage_config() {
    local udid="$1" bundle="$2" source="$3"
    xcrun devicectl device copy to --device "$udid" --domain-type appDataContainer \
        --domain-identifier "$bundle" --source "$source" --destination "$STAGING_CONFIG"
}

run_staging() {
    local config="${1:?config}" expectations="${2:?expectations}" udid="${3:?udid}"
    local out log bundle line empty status=0
    out="$(dirname "$expectations")"; mkdir -p "$out"; log="$out/device-console.log"
    bundle="$(cat "$STAGING_DERIVED/bundle-id")"
    if ! stage_config "$udid" "$bundle" "$config"; then
        echo "==> Creating Documents/xybrid-staging in the app's container, then copying again"
        xcrun devicectl device process launch --device "$udid" --console --terminate-existing \
            --environment-variables '{"XYBRID_STAGING_DEVICE":"prepare"}' "$bundle"
        if ! stage_config "$udid" "$bundle" "$config"; then
            echo "error: could not copy the config into $bundle's container on $udid" >&2
            exit 1
        fi
    fi
    # No secret in the environment: the hook reads the config from the container.
    xcrun devicectl device process launch --device "$udid" --console --terminate-existing \
        --environment-variables '{"XYBRID_STAGING_DEVICE":"1"}' "$bundle" > "$log" 2>&1 || status=$?
    grep -E "XYBRID_STAGING_ERROR" "$log" || true
    line="$(grep -o "${STAGING_MARKER}[A-Za-z0-9+/=]*" "$log" | tail -1 || true)"
    if [ -z "$line" ]; then
        # The app may not have read (and deleted) the config: overwrite it so
        # the key does not stay on the phone.
        empty="$(mktemp)"
        stage_config "$udid" "$bundle" "$empty" > /dev/null 2>&1 \
            || echo "warning: could not overwrite $STAGING_CONFIG in $bundle's container" >&2
        rm -f "$empty"
        echo "error: the app printed no ${STAGING_MARKER} line (exit $status; see $log)" >&2
        tail -15 "$log" >&2
        exit 1
    fi
    printf '%s' "${line#"$STAGING_MARKER"}" | base64 --decode > "$expectations"
    echo "==> Wrote $expectations from the device console"
    # As in run: the app always exits 0, so a failed step only shows on the console.
    if grep -q "XYBRID_STAGING_ERROR" "$log"; then
        echo "error: the scenario reported a failed step (see $log)" >&2
        exit 1
    fi
    if [ "$status" -ne 0 ]; then
        echo "error: devicectl exited with $status (see $log)" >&2
        exit "$status"
    fi
}

case "${1:-}" in
    build) shift; build "$@" ;;
    run) shift; run "$@" ;;
    build-staging) shift; build_staging "$@" ;;
    run-staging) shift; run_staging "$@" ;;
    *) echo "usage: $0 build <udid> [team] | run <endpoint> <expectations> <udid>" \
            "| build-staging <udid> [team] | run-staging <config> <expectations> <udid>" >&2; exit 2 ;;
esac
