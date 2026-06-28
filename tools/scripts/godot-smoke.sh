#!/usr/bin/env bash
set -euo pipefail

GODOT_BIN="${GODOT_BIN:-godot}"
if ! command -v "$GODOT_BIN" >/dev/null 2>&1; then
  echo "godot not found; skipping Godot smoke test"
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" == "Darwin" && -z "${XYBRID_GODOT_MACOS_TARGETS:-}" ]]; then
  export XYBRID_GODOT_MACOS_TARGETS
  XYBRID_GODOT_MACOS_TARGETS="$(rustc -vV | sed -n 's/^host: //p')"
fi

tools/scripts/build-godot.sh debug

TMPDIR="${TMPDIR:-/tmp}/xybrid-godot-smoke"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"
cp -R examples/godot/starter/. "$TMPDIR/"
mkdir -p "$TMPDIR/addons"
cp -R bindings/godot/addons/xybrid "$TMPDIR/addons/xybrid"

"$GODOT_BIN" --headless --path "$TMPDIR" -s res://scripts/smoke.gd
