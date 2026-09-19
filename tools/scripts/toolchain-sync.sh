#!/usr/bin/env bash
# toolchain-sync.sh — keep the CI Rust pin in step with rust-toolchain.toml
#
# rust-toolchain.toml is the canonical toolchain for this repo. GitHub Actions
# cannot read it directly, so every workflow carries a `RUST_VERSION` env pin
# that is handed to dtolnay/rust-toolchain. This script keeps the two in sync.
#
# Usage:
#   ./tools/scripts/toolchain-sync.sh --check    # verify (CI gate)
#   ./tools/scripts/toolchain-sync.sh --write    # rewrite workflows from the toml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOOLCHAIN_FILE="$REPO_ROOT/rust-toolchain.toml"
WORKFLOW_DIR="$REPO_ROOT/.github/workflows"

MODE="${1:---check}"
case "$MODE" in
  --check|--write) ;;
  *) echo "usage: $0 [--check|--write]" >&2; exit 2 ;;
esac

if [[ ! -f "$TOOLCHAIN_FILE" ]]; then
  echo "error: $TOOLCHAIN_FILE not found" >&2
  exit 1
fi

# `channel = "1.98.0"` -> 1.98.0
CHANNEL="$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$TOOLCHAIN_FILE" | head -1)"
if [[ -z "$CHANNEL" ]]; then
  echo "error: could not read [toolchain] channel from $TOOLCHAIN_FILE" >&2
  exit 1
fi

status=0
found=0

while IFS= read -r wf; do
  # Only workflows that pin a toolchain participate.
  grep -q '^  RUST_VERSION:' "$wf" || continue
  found=$((found + 1))
  current="$(sed -n 's/^  RUST_VERSION:[[:space:]]*"\([^"]*\)".*/\1/p' "$wf" | head -1)"
  rel="${wf#"$REPO_ROOT"/}"

  if [[ "$current" == "$CHANNEL" ]]; then
    [[ "$MODE" == "--check" ]] && echo "ok    $rel ($current)"
    continue
  fi

  if [[ "$MODE" == "--write" ]]; then
    # Portable in-place edit (BSD and GNU sed disagree on -i).
    tmp="$(mktemp)"
    sed "s/^  RUST_VERSION:.*/  RUST_VERSION: \"$CHANNEL\"/" "$wf" >"$tmp"
    mv "$tmp" "$wf"
    echo "wrote $rel ($current -> $CHANNEL)"
  else
    echo "DRIFT $rel: RUST_VERSION=$current but rust-toolchain.toml=$CHANNEL" >&2
    status=1
  fi
done < <(find "$WORKFLOW_DIR" -name '*.yml' | sort)

# A workflow that installs Rust but forgot the pin would silently float.
while IFS= read -r wf; do
  grep -q 'dtolnay/rust-toolchain' "$wf" || continue
  grep -q '^  RUST_VERSION:' "$wf" && continue
  echo "DRIFT ${wf#"$REPO_ROOT"/}: installs Rust but declares no RUST_VERSION pin" >&2
  status=1
done < <(find "$WORKFLOW_DIR" -name '*.yml' | sort)

# So does a step that asks for a floating channel by name.
if grep -rnE 'toolchain:[[:space:]]*(stable|beta|nightly)[[:space:]]*$' "$WORKFLOW_DIR" >/dev/null 2>&1; then
  echo "DRIFT: a workflow still requests a floating toolchain channel:" >&2
  grep -rnE 'toolchain:[[:space:]]*(stable|beta|nightly)[[:space:]]*$' "$WORKFLOW_DIR" >&2
  status=1
fi

if [[ "$found" -eq 0 ]]; then
  echo "error: no workflow declares RUST_VERSION — is the pin wired up?" >&2
  exit 1
fi

if [[ "$MODE" == "--check" ]]; then
  if [[ "$status" -eq 0 ]]; then
    echo "toolchain pin is in sync across $found workflow(s): $CHANNEL"
  else
    echo "" >&2
    echo "Run ./tools/scripts/toolchain-sync.sh --write to fix." >&2
  fi
fi

exit "$status"
