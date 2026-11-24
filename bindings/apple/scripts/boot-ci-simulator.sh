#!/usr/bin/env bash
# =============================================================================
# boot-ci-simulator.sh
# =============================================================================
# Creates and boots an iPhone simulator on a CI macOS runner, then prints its
# UDID on stdout. Everything else goes to stderr.
#
# Hosted macOS runners sometimes report none of the image's simulators while
# CoreSimulator rebuilds its caches (actions/runner-images#12862, #12948), and
# a fixed device/runtime pair breaks whenever the image moves its runtimes. So
# this waits for an available iOS runtime, creates the newest iPhone that
# runtime supports, and boots it.
#
# Usage:
#   SIM_ID="$(bindings/apple/scripts/boot-ci-simulator.sh [device name])"
# =============================================================================

set -euo pipefail

NAME="${1:-Xybrid CI iPhone}"

# Prints "<runtime id> <device type id>": the newest available iOS runtime and
# the newest iPhone it supports.
pick_simulator() {
    python3 - <<'PY'
import json, subprocess, sys
listing = json.loads(subprocess.check_output(["xcrun", "simctl", "list", "runtimes", "--json"]))
runtimes = [r for r in listing["runtimes"] if r.get("platform") == "iOS" and r.get("isAvailable")]
runtimes.sort(key=lambda r: [int(part) for part in r["version"].split(".")], reverse=True)
for runtime in runtimes:
    # supportedDeviceTypes lists the newest models first.
    iphones = [d for d in runtime.get("supportedDeviceTypes", []) if d.get("productFamily") == "iPhone"]
    if iphones:
        print(f"Using {iphones[0]['name']} on {runtime['name']}", file=sys.stderr)
        print(runtime["identifier"], iphones[0]["identifier"])
        sys.exit()
sys.exit("no available iOS simulator runtime with an iPhone device type")
PY
}

for attempt in 1 2 3 4 5 6; do
    if PICK="$(pick_simulator)"; then break; fi
    if [ "$attempt" -eq 6 ]; then xcrun simctl list runtimes >&2 || true; exit 1; fi
    echo "Retrying in 20s (attempt $attempt/6)" >&2
    sleep 20
done
read -r RUNTIME_ID DEVICE_TYPE <<<"$PICK"
SIM_ID="$(xcrun simctl create "$NAME" "$DEVICE_TYPE" "$RUNTIME_ID")"
xcrun simctl bootstatus "$SIM_ID" -b >&2
echo "$SIM_ID"
