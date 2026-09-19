#!/bin/sh
# xybrid addition — not part of upstream cargokit.
#
# Usage: source_build_possible.sh <manifest-dir>
# Exits 0 when the crate in <manifest-dir> can be built from source where it
# sits, 1 when it cannot (the reason is printed).
#
# The shell twin of `_sourceBuildBlocker` in
# build_tool/lib/src/artifacts_provider.dart — keep the two in step. The
# package published to pub.dev ships `rust/` without the workspace root it
# inherits from and without the sibling crates its path dependencies point at,
# so there it is precompiled-only. Native build scripts use this to skip work
# that only a source build needs.
#
# POSIX sh on purpose: CI runs it under dash, consumers under macOS /bin/sh.

manifest_dir="$1"
manifest="$manifest_dir/Cargo.toml"

if [ ! -f "$manifest" ]; then
  echo "no Cargo.toml in $manifest_dir"
  exit 1
fi

# Every `path = "..."` dependency has to exist.
missing=""
for dep in $(sed -nE 's/.*path[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$manifest" | sort -u); do
  if [ ! -d "$manifest_dir/$dep" ]; then
    missing="$missing $dep"
  fi
done
if [ -n "$missing" ]; then
  echo "path dependencies are not present:$missing"
  exit 1
fi

# `workspace = true` needs a workspace root somewhere above.
if grep -qE 'workspace[[:space:]]*=[[:space:]]*true' "$manifest"; then
  dir=$(cd "$manifest_dir" && pwd -P)
  while :; do
    if [ -f "$dir/Cargo.toml" ] &&
      grep -qE '^[[:space:]]*\[workspace[].]' "$dir/Cargo.toml"; then
      exit 0
    fi
    parent=$(dirname "$dir")
    if [ "$parent" = "$dir" ]; then
      echo "Cargo.toml inherits from a workspace root, but none exists above $manifest_dir"
      exit 1
    fi
    dir="$parent"
  done
fi

exit 0
