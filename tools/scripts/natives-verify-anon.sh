#!/usr/bin/env bash
# natives-verify-anon.sh
#
# Verifies that every slice in crates/llama-cpp-sys/natives-manifest.txt can be
# fetched ANONYMOUSLY — no ghcr login, no GITHUB_TOKEN, nothing an external
# `cargo build` would not have.
#
# This exists because the failure it catches is invisible everywhere else: our
# own CI jobs `oras login` before pulling, so a PRIVATE package works perfectly
# in CI and silently 401s for every outside consumer. build.rs then degrades to
# a source build and nobody notices the fast path never fires.
#
# If this fails with 401/UNAUTHORIZED, the fix is a GitHub setting, not code:
#   github.com/orgs/xybrid-ai/packages/container/llama-natives/settings
#   → Danger Zone → Change visibility → Public
#
# Usage: natives-verify-anon.sh [manifest-file]
# Requires: curl. Deliberately does NOT use oras — the point is to exercise the
# same plain-HTTPS path build.rs takes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${1:-$ROOT/crates/llama-cpp-sys/natives-manifest.txt}"

[ -f "$MANIFEST" ] || { echo "natives-verify-anon: no manifest at $MANIFEST" >&2; exit 1; }

REGISTRY="$(awk '$1 == "registry" { print $2; exit }' "$MANIFEST")"
[ -n "$REGISTRY" ] || { echo "natives-verify-anon: manifest has no registry line" >&2; exit 1; }
HOST="${REGISTRY%%/*}"
REPO="${REGISTRY#*/}"

TOKEN="$(curl -sf "https://$HOST/token?scope=repository:$REPO:pull&service=$HOST" \
  | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')"
if [ -z "$TOKEN" ]; then
  echo "FAIL: $REGISTRY issues no anonymous pull token — the package is private." >&2
  echo "      Every external \`cargo build\` will fall back to compiling llama.cpp." >&2
  echo "      Fix: make the package public (see the header of this script)." >&2
  exit 1
fi

fail=0
checked=0
while read -r _ target features digest _; do
  checked=$((checked + 1))
  code="$(curl -s -o /dev/null -w '%{http_code}' -I -L \
    -H "Authorization: Bearer $TOKEN" \
    "https://$HOST/v2/$REPO/blobs/$digest")"
  if [ "$code" = "200" ]; then
    echo "ok   $target $features"
  else
    echo "FAIL $target $features — HTTP $code for $digest" >&2
    fail=$((fail + 1))
  fi
done < <(awk '$1 == "slice"' "$MANIFEST")

if [ "$checked" -eq 0 ]; then
  echo "natives-verify-anon: manifest has no slice rows — nothing to verify"
  exit 0
fi
[ "$fail" -eq 0 ] || { echo "natives-verify-anon: $fail/$checked slices unreachable anonymously" >&2; exit 1; }
echo "natives-verify-anon: all $checked slices reachable anonymously"
