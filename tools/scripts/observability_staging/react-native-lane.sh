#!/usr/bin/env bash
# React Native staging lanes (step 6), called by lanes.sh: the example app
# (bindings/react-native/example) on the Pixel (the CI debug APK's natives,
# JavaScript from Metro on the Mac) and on the iPhone (a Debug build against
# the candidate XCFramework). No tapping: the app runs the staging scenario
# when the copied config is in its documents directory (example/staging.ts).
#   react-native-lane.sh pixel-build  <repo> <apk>
#   react-native-lane.sh pixel-run    <repo> <config> <expectations> <out_dir> <collector_port> <run>
#   react-native-lane.sh iphone-build <repo> [xcframework dir or .zip]
#   react-native-lane.sh iphone-run   <repo> <config> <expectations> <out_dir> <run>
set -euo pipefail

PKG=com.anonymous.example                 # app.json android.package
IOS_BUNDLE=dev.xybrid.rnexample           # set in the generated Xcode project only (team CPVMT4DS8N)
IOS_UDID="${XYBRID_IPHONE_UDID:-00008130-001C446E1408001C}"
IOS_TEAM=CPVMT4DS8N
IOS_DERIVED="${TMPDIR:-/tmp}/xybrid-rn-staging-ios"
# A Debug build without an embedded bundle looks for Metro on the phone
# itself unless told the Mac's address: the RCT_jsLocation user default,
# passed as a launch argument.
METRO_HOST="${XYBRID_MAC_LAN_IP:-$(ipconfig getifaddr en0 2>/dev/null || true)}"
POLL_SECONDS=1200
metro="" copied="" received="" part="" port=""

# Metro for the example (debug apps load their JavaScript from it), with no
# prompts (CI=1), and the platform bundle built once so a bundling error fails
# here instead of on the phone.
start_metro() {
    local example="$1" out="$2" platform="$3"
    [ -d "$example/node_modules" ] || (cd "$example" && npm ci --no-audit --no-fund)
    if curl -sf --max-time 2 http://127.0.0.1:8081/status >/dev/null; then
        echo "error: something already serves :8081 (another Metro?); stop it first" >&2; exit 1
    fi
    (cd "$example" && CI=1 EXPO_NO_TELEMETRY=1 exec ./node_modules/.bin/expo start --dev-client --offline \
        --port 8081) > "$out/metro.log" 2>&1 &
    metro=$!
    for _ in $(seq 120); do
        curl -sf --max-time 2 http://127.0.0.1:8081/status | grep -q running && break
        sleep 1
    done
    curl -sf --max-time 600 -o /dev/null \
        "http://127.0.0.1:8081/.expo/.virtual-metro-entry.bundle?platform=$platform&dev=true&minify=false" \
        || { echo "error: Metro did not serve the $platform bundle (see $out/metro.log)" >&2; exit 1; }
}

stop_metro() {
    [ -n "$metro" ] || return 0
    pkill -TERM -P "$metro" 2>/dev/null || true
    kill -TERM "$metro" 2>/dev/null || true
}

# The copied-back file counts once it parses and names this launch's run.
is_this_run() {
    python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("run") != sys.argv[2])' "$1" "$2" \
        2>/dev/null
}

# The app cannot report a failed step through an exit status: read it here.
report_scenario() {
    python3 -c 'import json, sys
error = json.load(open(sys.argv[1])).get("scenario_error")
print("==> scenario_error: " + error.splitlines()[0] if error else "==> scenario ok")
sys.exit(1 if error else 0)' "$1"
}

pixel_build() {
    local repo="$1" apk="$2"
    adb wait-for-device
    adb install -r "$apk"
    (cd "$repo/bindings/react-native/example" && npm ci --no-audit --no-fund)
}

pixel_cleanup() {
    # The app deletes the config once read; if it never got that far, remove it.
    if [ -n "$copied" ] && [ -z "$received" ]; then
        adb exec-out run-as "$PKG" rm -f files/xybrid-staging/config.json >/dev/null 2>&1 || true
    fi
    [ -z "${logcat:-}" ] || kill "$logcat" 2>/dev/null || true
    rm -f "$part"
    stop_metro
    adb reverse --remove "tcp:$port" >/dev/null 2>&1 || true
    adb reverse --remove tcp:8081 >/dev/null 2>&1 || true
}

pixel_run() {
    local repo="$1" config="$2" expectations="$3" out="$4" run="$6"
    local example="$repo/bindings/react-native/example" deadline
    port="$5" part="$expectations.part"
    trap pixel_cleanup EXIT
    adb wait-for-device
    start_metro "$example" "$out" android
    adb reverse tcp:8081 tcp:8081                 # Metro
    adb reverse "tcp:$port" "tcp:$port"           # the Collector session's http://127.0.0.1:<port>
    adb shell am force-stop "$PKG"
    # <files> is expo-file-system's Paths.document (Context.getFilesDir()).
    # `adb exec-in` returns once stdin is sent, not when the device has written
    # it (and exits 0 either way): wait for the file to reach the config's size.
    local size tries=0
    size="$(wc -c < "$config" | tr -d ' ')"
    until [ "$(adb exec-out run-as "$PKG" sh -c 'wc -c < files/xybrid-staging/config.json' 2>/dev/null \
            | tr -d ' \r\n')" = "$size" ]; do
        tries=$((tries + 1))
        [ "$tries" -le 5 ] || { echo "error: the config did not reach $PKG's files/xybrid-staging" >&2; exit 1; }
        adb exec-in run-as "$PKG" sh -c \
            'umask 077 && rm -rf files/xybrid-staging && mkdir -p files/xybrid-staging && cat > files/xybrid-staging/config.json' \
            < "$config"
        copied=1
        sleep 2
    done
    adb logcat -c
    adb logcat -v time ReactNativeJS:V AndroidRuntime:E '*:S' > "$out/logcat.txt" 2>&1 &
    logcat=$!
    adb shell input keyevent KEYCODE_WAKEUP
    adb shell am start -W -n "$PKG/.MainActivity"
    deadline=$((SECONDS + POLL_SECONDS))
    until adb exec-out run-as "$PKG" cat files/xybrid-staging/expectations.json > "$part" 2>/dev/null \
            && is_this_run "$part" "$run"; do
        [ "$SECONDS" -lt "$deadline" ] || { echo "error: no expectations.json for $run after ${POLL_SECONDS}s" >&2; exit 1; }
        sleep 5
    done
    mv "$part" "$expectations"
    received=1
    adb shell am force-stop "$PKG"
    report_scenario "$expectations"
}

iphone_build() {
    local repo="$1" xcframework="${2:-$1/bazel-bin/bindings/apple/XybridFFI.xcframework.zip}"
    local example="$repo/bindings/react-native/example" jsi
    cd "$example"
    npm ci --no-audit --no-fund
    CI=1 EXPO_NO_GIT_STATUS=1 npx expo prebuild --platform ios --clean --no-install
    # Generated (gitignored) project only: a bundle id team CPVMT4DS8N can sign, and
    # the usage string of the local network prompt (Metro and the Collector are on the LAN).
    sed -i '' -E "s/PRODUCT_BUNDLE_IDENTIFIER = \"?com\.anonymous\.example\"?;/PRODUCT_BUNDLE_IDENTIFIER = $IOS_BUNDLE;/" \
        ios/example.xcodeproj/project.pbxproj
    grep -q "PRODUCT_BUNDLE_IDENTIFIER = $IOS_BUNDLE;" ios/example.xcodeproj/project.pbxproj
    plutil -replace NSLocalNetworkUsageDescription \
        -string "Loads JavaScript from Metro and sends staging telemetry to a Collector on this network." \
        ios/example/Info.plist
    # The pod's own override (ios/xybrid_natives.rb): the candidate XCFramework, never a release download.
    (cd ios && XYBRID_XCFRAMEWORK_PATH="$xcframework" pod install)
    patch_expo_jsi_for_xcode27 "$example"
    # As in build-react-native.yml: prebuild ExpoModulesJSI so its nested xcodebuild phase is a no-op.
    jsi=node_modules/expo-modules-jsi/apple/scripts/build-xcframework.sh
    if [ -x "$jsi" ]; then
        (cd ios && PODS_ROOT="$PWD/Pods" RN_ROOT="$PWD/../node_modules/react-native" PLATFORM_NAME=iphoneos "../$jsi")
    fi
    xcodebuild build -workspace ios/example.xcworkspace -scheme example -configuration Debug \
        -destination "platform=iOS,id=$IOS_UDID" -derivedDataPath "$IOS_DERIVED" -allowProvisioningUpdates \
        DEVELOPMENT_TEAM="$IOS_TEAM"
    xcrun devicectl device install app --device "$IOS_UDID" "$IOS_DERIVED/Build/Products/Debug-iphoneos/example.app"
}

# Expo 56's expo-modules-jsi forms a C function pointer from a ternary, which
# Xcode 27's Swift rejects ("a C function pointer can only be formed from a
# reference to a 'func' or a literal closure"). Rewrite that one expression in
# the installed node_modules copy (never committed); a no-op once Expo has it.
patch_expo_jsi_for_xcode27() {
    local file="$1/node_modules/expo-modules-jsi/apple/Sources/ExpoModulesJSI/Runtime/JavaScriptRuntime.swift"
    [ -f "$file" ] || return 0
    python3 - "$file" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = """    let callbacks = expo.HostObjectCallbacks(
      context, getter, set == nil ? nil : setter, propertyNamesGetter, deallocate)"""
new = """    // Step 6 lane patch (local node_modules only): Xcode 27's Swift forms a C
    // function pointer only from a direct reference, not from a ternary.
    let callbacks = set == nil
      ? expo.HostObjectCallbacks(context, getter, nil, propertyNamesGetter, deallocate)
      : expo.HostObjectCallbacks(context, getter, setter, propertyNamesGetter, deallocate)"""
if old in text:
    open(path, "w").write(text.replace(old, new))
    print("==> patched expo-modules-jsi for Xcode 27 (local node_modules only)")
PY
}

ios_copy_config() {
    xcrun devicectl device copy to --device "$IOS_UDID" --domain-type appDataContainer \
        --domain-identifier "$IOS_BUNDLE" --source "$1" --destination Documents/xybrid-staging/config.json
}

# devicectl may not create Documents/xybrid-staging on a fresh install: then
# launch the app once with -XybridStagingPrepare YES (it creates the folder,
# which needs its JavaScript from Metro) and copy again.
ios_stage_config() {
    ios_copy_config "$1" && return 0
    echo "==> creating Documents/xybrid-staging in the app's container, then copying again"
    xcrun devicectl device process launch --device "$IOS_UDID" --terminate-existing "$IOS_BUNDLE" \
        -XybridStagingPrepare YES -RCT_jsLocation "$METRO_HOST:8081"
    for _ in $(seq 12); do
        sleep 5
        ios_copy_config "$1" > /dev/null 2>&1 && return 0
    done
    echo "error: could not copy the config to the phone; on a first run allow Local Network for the app" \
        "on the phone (it loads its JavaScript from Metro over the LAN), then run again" >&2
    return 1
}

iphone_cleanup() {
    local empty
    # The app deletes the config once read; if it never got that far, overwrite it.
    if [ -n "$copied" ] && [ -z "$received" ]; then
        empty="$(mktemp)"
        ios_copy_config "$empty" >/dev/null 2>&1 || echo "warning: could not overwrite the config on the phone" >&2
        rm -f "$empty"
    fi
    rm -f "$part"
    stop_metro
}

iphone_run() {
    local repo="$1" config="$2" expectations="$3" out="$4" run="$5"
    local example="$repo/bindings/react-native/example" deadline
    part="$expectations.part"
    trap iphone_cleanup EXIT
    start_metro "$example" "$out" ios
    ios_stage_config "$config"
    copied=1
    [ -n "$METRO_HOST" ] || { echo "error: no LAN address for Metro (set XYBRID_MAC_LAN_IP)" >&2; exit 1; }
    xcrun devicectl device process launch --device "$IOS_UDID" --terminate-existing "$IOS_BUNDLE" \
        -RCT_jsLocation "$METRO_HOST:8081"
    deadline=$((SECONDS + POLL_SECONDS))
    until rm -f "$part" && xcrun devicectl device copy from --device "$IOS_UDID" --domain-type appDataContainer \
            --domain-identifier "$IOS_BUNDLE" --source Documents/xybrid-staging/expectations.json \
            --destination "$part" >/dev/null 2>&1 && is_this_run "$part" "$run"; do
        [ "$SECONDS" -lt "$deadline" ] || { echo "error: no expectations.json for $run after ${POLL_SECONDS}s" >&2; exit 1; }
        sleep 10
    done
    mv "$part" "$expectations"
    received=1
    report_scenario "$expectations"
}

case "${1:-}" in
    pixel-build) shift; pixel_build "$@" ;;
    pixel-run) shift; pixel_run "$@" ;;
    iphone-build) shift; iphone_build "$@" ;;
    iphone-run) shift; iphone_run "$@" ;;
    *) echo "usage: $0 pixel-build|pixel-run|iphone-build|iphone-run ..." >&2; exit 2 ;;
esac
