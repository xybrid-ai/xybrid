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
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
APP_DIR="$ROOT/examples/ios/XybridExample"
PROJECT="$ROOT/examples/ios/XybridExample.xcodeproj"
TEST="$ROOT/bindings/apple/Tests/XybridTests/ObservabilityRuntimeTests.swift"
SPEC="$ROOT/tools/scripts/observability_capture/scenario.json"
DERIVED="${XYBRID_IOS_DEVICE_DERIVED_DATA:-${TMPDIR:-/tmp}/xybrid-observability-ios-device}"
MARKER="XYBRID_SCENARIO_EXPECTATIONS_BASE64="

build() {
    local udid="${1:?udid}" team="${2:-}"
    local entry="$APP_DIR/XybridExampleApp.swift" plist="$APP_DIR/Info.plist"
    cp "$entry" "$entry.orig"; cp "$plist" "$plist.orig"
    # Expanded now: the trap runs after this function's locals are gone.
    # shellcheck disable=SC2064
    trap "mv -f '$entry.orig' '$entry'; mv -f '$plist.orig' '$plist'" EXIT
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
}

case "${1:-}" in
    build) shift; build "$@" ;;
    run) shift; run "$@" ;;
    *) echo "usage: $0 build <udid> [team] | run <endpoint> <expectations> <udid>" >&2; exit 2 ;;
esac
