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

lane_swift_iphone() {
    # The CI XCFramework (artifact XybridFFI-xcframework) in local mode, with
    # the staging scenario injected into examples/ios. Return Package.swift to
    # remote mode afterwards (set-natives-mode.sh --set-remote); never commit it.
    rm -rf bindings/apple/XCFrameworks/XybridFFI.xcframework
    mkdir -p bindings/apple/XCFrameworks/XybridFFI.xcframework
    cp -R "$ART/XybridFFI-xcframework/." bindings/apple/XCFrameworks/XybridFFI.xcframework/
    ./bindings/apple/scripts/set-natives-mode.sh --set-local
    bash bindings/apple/scripts/observability-scenario-ios-device.sh build-staging "$UDID" "$TEAM"
    "${RUN[@]}" --lane swift-iphone --collector-host "$LAN_IP" "$@" -- \
        bash bindings/apple/scripts/observability-scenario-ios-device.sh run-staging {config} {expectations} "$UDID"
}

lane_kotlin_pixel() {
    # The consumer-smoke APKs against the CI AAR (artifact android-dlopen-gate).
    [ -x jdk17/Contents/Home/bin/java ] && export JAVA_HOME="$ROOT/jdk17/Contents/Home"
    export ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}" PATH="${ANDROID_HOME:-$HOME/Library/Android/sdk}/platform-tools:$PATH"
    mkdir -p bindings/kotlin/consumer-smoke/app/libs
    cp "$ART/android-dlopen-gate/xybrid-kotlin.aar" bindings/kotlin/consumer-smoke/app/libs/xybrid-kotlin.aar
    bindings/kotlin/gradlew -p bindings/kotlin/consumer-smoke :app:assembleDebug :app:assembleDebugAndroidTest
    "${RUN[@]}" --lane kotlin-pixel "$@" -- \
        bash bindings/kotlin/consumer-smoke/observability-scenario.sh staging {config} {expectations}
}

lane_unity_mac() {
    # The macOS arm64 cdylib built from this checkout, staged into the smoke
    # project; the marker stops the Editor fetching the release dylib.
    local plugins=tools/unity-runtime-smoke/Assets/Xybrid/Plugins
    local editor="${XYBRID_UNITY_EDITOR:-/Applications/Unity/Hub/Editor/6000.3.6f1/Unity.app/Contents/MacOS/Unity}"
    cargo xtask build-ffi --target aarch64-apple-darwin --release --deploy-unity
    rm -rf "$plugins/macOS" "$plugins/macOS.meta"
    mkdir -p "$plugins"
    python3 -c "import json;print(json.load(open('bindings/unity/package.json'))['version'])" \
        > "$plugins/.xybrid-native-macos-version"
    "${RUN[@]}" --lane unity-mac "$@" -- \
        "$editor" -batchmode -projectPath "$ROOT/tools/unity-runtime-smoke" -runTests -testPlatform EditMode \
        -testCategory ObservabilityStagingScenario -testResults {out_dir}/unity-editmode-results.xml -logFile - \
        -xybridStagingConfig {config} -xybridStagingExpectations {expectations}
}

lane_react_native_pixel() {
    # The CI debug APK (artifact xybrid-example-android-debug) carries the
    # candidate's natives; Metro serves this checkout's JavaScript.
    bash "$HERE/react-native-lane.sh" pixel-build "$ROOT" "$ART/xybrid-example-android-debug/app-debug.apk"
    "${RUN[@]}" --lane react-native-pixel "$@" -- bash "$HERE/react-native-lane.sh" pixel-run \
        {repo} {config} {expectations} {out_dir} {collector_port} {run}
}

lane_react_native_iphone() {
    # The example's pod takes the CI XCFramework through XYBRID_XCFRAMEWORK_PATH.
    bash "$HERE/react-native-lane.sh" iphone-build "$ROOT" "$ART/XybridFFI-xcframework"
    "${RUN[@]}" --lane react-native-iphone --collector-host "$LAN_IP" "$@" -- \
        bash "$HERE/react-native-lane.sh" iphone-run {repo} {config} {expectations} {out_dir} {run}
}

lane_web_mac() {
    # The CI web artifact (xybrid-web-wasm) served by the example as /sdk. The
    # example's asset step (copy the package, fetch and verify the GGUF) uses
    # Bun APIs: run it with CI's Bun (1.3.9) through npx when bun is not
    # installed, then build with vite directly (no Bun pre-script).
    local web="$ROOT/bindings/web" package="$ART/xybrid-web-wasm" bun=(bun)
    command -v bun >/dev/null || bun=(npx -y bun@1.3.9)
    (cd "$ROOT/bindings/observability-js" && pnpm install --frozen-lockfile && node scripts/build.ts)
    (cd "$web" && pnpm install --frozen-lockfile && pnpm run observability:prepare \
        && pnpm exec playwright install chromium && rm -rf example/public/sdk \
        && XYBRID_WEB_PACKAGE="$package" "${bun[@]}" run example/scripts/prepare-assets.ts \
        && XYBRID_WEB_PACKAGE="$package" pnpm exec vite build --config example/vite.config.ts \
        && diff -rq "$package/dist" example/dist/sdk)
    "${RUN[@]}" --lane web-mac "$@" -- \
        pnpm --dir "$web" exec playwright test browser-test/staging.spec.ts
}

lane_flutter_mac() {
    # The Dart VM (flutter test) on the release FFI library built from this checkout.
    cargo build --release -p xybrid_flutter --features platform-macos
    (cd bindings/flutter && flutter pub get)
    XYBRID_FLUTTER_NATIVE_LIB="$ROOT/target/release/libxybrid_flutter_ffi.dylib" "${RUN[@]}" --lane flutter-mac "$@" -- \
        bash -c 'cd bindings/flutter && flutter test --no-pub test/observability_staging_test.dart'
}

lane_flutter_pixel() {
    # flutter drive of examples/flutter (cargokit builds the native from this
    # checkout); the config is copied once the app runs (README, Flutter).
    (cd examples/flutter && flutter pub get)
    "${RUN[@]}" --lane flutter-pixel --timeout 3600 "$@" -- bash "$HERE/lanes.sh" _flutter-drive-android
}

lane_flutter_iphone() {
    (cd examples/flutter && flutter pub get)
    "${RUN[@]}" --lane flutter-iphone --collector-host "$LAN_IP" --timeout 3600 "$@" -- \
        bash "$HERE/lanes.sh" _flutter-drive-ios
}

# Internal: run by run_staging.py, with XYBRID_STAGING_CONFIG, _RUN,
# _COLLECTOR_PORT and _EXPECTATIONS in the environment.
flutter_drive() {
    flutter drive --no-pub --keep-app-running -d "$1" --driver test_driver/integration_test.dart \
        --target integration_test/observability_staging_test.dart --dart-define=XYBRID_STAGING_RUN="$XYBRID_STAGING_RUN"
}

_flutter_drive_android() {
    local serial app=com.example.xybrid_example stage=app_flutter/xybrid-staging drive status=0
    local port="$XYBRID_STAGING_COLLECTOR_PORT"
    serial="${ANDROID_SERIAL:-$(adb get-serialno)}"
    cd "$ROOT/examples/flutter"
    adb -s "$serial" reverse "tcp:$port" "tcp:$port"
    adb -s "$serial" shell am force-stop "$app" || true
    adb -s "$serial" shell "run-as $app rm -rf $stage" 2>/dev/null || true
    flutter_drive "$serial" &
    drive=$!
    # A temporary name and a rename: the app never reads a half-copied config.
    until adb -s "$serial" shell pidof "$app" >/dev/null 2>&1 && adb -s "$serial" shell -T \
        "run-as $app sh -c 'umask 077 && mkdir -p $stage && cat > $stage/config.json.tmp && mv $stage/config.json.tmp $stage/config.json'" \
        < "$XYBRID_STAGING_CONFIG"; do
        kill -0 "$drive" 2>/dev/null || break
        sleep 1
    done
    wait "$drive" || status=$?
    adb -s "$serial" shell "run-as $app rm -rf $stage" 2>/dev/null || true
    adb -s "$serial" reverse --remove "tcp:$port" || true
    return "$status"
}

_flutter_drive_ios() {
    local bundle=dev.xybrid.flutterexample drive status=0 probe
    cd "$ROOT/examples/flutter"
    flutter_drive "$UDID" &
    drive=$!
    until xcrun devicectl device info processes --device "$UDID" 2>/dev/null | grep -q '/Runner.app/Runner' \
        && xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer --domain-identifier "$bundle" \
            --source "$XYBRID_STAGING_CONFIG" --destination Documents/xybrid-staging/config.json >/dev/null 2>&1; do
        kill -0 "$drive" 2>/dev/null || break
        sleep 1
    done
    wait "$drive" || status=$?
    # A config the app never read must not stay on the phone: remove the app.
    probe="$(mktemp -d)"
    if xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer --domain-identifier "$bundle" \
        --source Documents/xybrid-staging/config.json --destination "$probe/c" >/dev/null 2>&1; then
        xcrun devicectl device uninstall app --device "$UDID" "$bundle" || true
    fi
    rm -rf "$probe"
    return "$status"
}

# A.3: the Collector interop harness (Rust, JS, JS metadata-only and Python
# producers through the pinned Collector) against staging, with the launch key.
lane_interop_xybrid() {
    (cd bindings/observability-js && pnpm install --frozen-lockfile && node scripts/build.ts)
    cargo build -p xybrid-observability --example standalone
    XYBRID_STAGING_API_KEY="$(cat "$HOME/.xybrid-step6/key")" python3 tools/scripts/observability_interop/run_interop.py \
        --destination xybrid --python "$(command -v python3)" --out-dir "$EV/a3-interop" "$@"
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
        # The release ships the macOS CLI as a bare binary.
        gh release download v0.17.0 -R xybrid-ai/xybrid -p 'xybrid-v0.17.0-macos-arm64' -D "$dir" --clobber
        mv "$dir/xybrid-v0.17.0-macos-arm64" "$dir/xybrid"
        chmod +x "$dir/xybrid"
        xattr -d com.apple.quarantine "$dir/xybrid" 2>/dev/null || true
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
    _flutter-drive-android) _flutter_drive_android; exit $? ;;
    _flutter-drive-ios) _flutter_drive_ios; exit $? ;;
esac
lane="$1"; shift
function="lane_${lane//-/_}"
if ! declare -F "$function" >/dev/null; then
    echo "error: unknown lane $lane; lanes: $(lanes | tr '\n' ' ')" >&2
    exit 2
fi
"$function" "$@"
