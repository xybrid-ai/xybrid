#!/usr/bin/env bash
#
# Materialize the host RN app shell into ./ios/ and ./android/ so the example
# can be built. The committed files (App.tsx, package.json, metro/babel config)
# are everything that's actually unique to this example — the Xcode project
# blob and gradle wrappers come from the upstream RN template via this script.
#
# Idempotent: re-running won't blow away your modifications. Pass --force to
# wipe the generated tree and re-bootstrap from scratch.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="XybridExample"
RN_VERSION="0.76.0"

FORCE=false
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=true ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [[ -d "$EXAMPLE_DIR/ios" || -d "$EXAMPLE_DIR/android" ]]; then
  if [[ "$FORCE" == "true" ]]; then
    echo "→ Removing existing host shell..."
    rm -rf "$EXAMPLE_DIR/ios" "$EXAMPLE_DIR/android"
  else
    echo "✓ Host shell already exists at $EXAMPLE_DIR/{ios,android} — skipping bootstrap."
    echo "  Pass --force to recreate."
    exit 0
  fi
fi

# Stage the upstream template into a tempdir, then move only ios/ and
# android/ into the example. We can't run `init` directly into a directory
# that already has package.json — the CLI bails out — so this two-step is
# the path of least resistance.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "→ Generating $APP_NAME from RN $RN_VERSION template..."
npx --yes @react-native-community/cli@latest init "$APP_NAME" \
  --version "$RN_VERSION" \
  --directory "$TMP_DIR/$APP_NAME" \
  --skip-install \
  --pm npm \
  --install-pods false

echo "→ Moving ios/ and android/ into example..."
mv "$TMP_DIR/$APP_NAME/ios" "$EXAMPLE_DIR/ios"
mv "$TMP_DIR/$APP_NAME/android" "$EXAMPLE_DIR/android"

# Patches:
#   1. Enable the New Architecture (TurboModules) — react-native-xybrid is
#      new-arch only. The CLI defaults differ across RN versions, so we set
#      it explicitly rather than trusting the template.
echo "→ Enabling New Architecture..."
GRADLE_PROPS="$EXAMPLE_DIR/android/gradle.properties"
if ! grep -q "^newArchEnabled=" "$GRADLE_PROPS"; then
  echo "newArchEnabled=true" >> "$GRADLE_PROPS"
else
  # macOS sed needs the empty -i extension; GNU sed accepts it too.
  sed -i.bak 's/^newArchEnabled=.*/newArchEnabled=true/' "$GRADLE_PROPS" \
    && rm -f "$GRADLE_PROPS.bak"
fi

PODFILE="$EXAMPLE_DIR/ios/Podfile"
if ! grep -q "ENV\['RCT_NEW_ARCH_ENABLED'\]" "$PODFILE"; then
  # Insert at the top — env vars must be set before `use_react_native!` is called.
  printf "ENV['RCT_NEW_ARCH_ENABLED'] = '1'\n%s" "$(cat "$PODFILE")" > "$PODFILE.tmp"
  mv "$PODFILE.tmp" "$PODFILE"
fi

echo
echo "✓ Bootstrap complete."
echo
echo "  Next:"
echo "    cd $EXAMPLE_DIR"
echo "    npm install"
echo "    cd ios && pod install && cd ..       # iOS only, on macOS"
echo "    npm run android                       # or 'npm run ios'"
