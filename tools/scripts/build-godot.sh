#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-release}"
FEATURES="${XYBRID_GODOT_FEATURES:-}"
FRAMEWORK_NAME="Xybrid Godot.framework"
FRAMEWORK_EXECUTABLE="Xybrid Godot"
FRAMEWORK_COMPAT_DYLIB="libxybrid_godot.dylib"

case "$PROFILE" in
  debug)
    CARGO_PROFILE_ARGS=()
    PROFILE_DIR="debug"
    ;;
  release)
    CARGO_PROFILE_ARGS=(--release)
    PROFILE_DIR="release"
    ;;
  *)
    echo "usage: $0 [debug|release]" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TARGET_ROOT="${CARGO_TARGET_DIR:-target}"

run_cargo_build() {
  cargo build -p xybrid-godot "${CARGO_PROFILE_ARGS[@]}" --features "$FEATURES" "$@"
}

write_macos_info_plist() {
  local resources_dir="$1"
  local version="${XYBRID_GODOT_BUNDLE_VERSION:-1.0.0}"

  cat > "$resources_dir/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>$FRAMEWORK_EXECUTABLE</string>
  <key>CFBundleIdentifier</key>
  <string>ai.xybrid.godot</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>Xybrid Godot</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleShortVersionString</key>
  <string>$version</string>
  <key>CFBundleSignature</key>
  <string>????</string>
  <key>CFBundleVersion</key>
  <string>$version</string>
  <key>LSMinimumSystemVersion</key>
  <string>11.0</string>
</dict>
</plist>
EOF
}

create_macos_framework() {
  local source_dylib="$1"
  local dest_dir="$2"
  local framework_dir="$dest_dir/$FRAMEWORK_NAME"
  local version_dir="$framework_dir/Versions/A"
  local resources_dir="$version_dir/Resources"
  local binary_path="$version_dir/$FRAMEWORK_EXECUTABLE"

  rm -rf "$framework_dir"
  mkdir -p "$resources_dir"

  ln -s A "$framework_dir/Versions/Current"
  ln -s "Versions/Current/$FRAMEWORK_EXECUTABLE" "$framework_dir/$FRAMEWORK_EXECUTABLE"
  ln -s "Versions/Current/Resources" "$framework_dir/Resources"
  ln -s "Versions/Current/$FRAMEWORK_EXECUTABLE" "$framework_dir/$FRAMEWORK_COMPAT_DYLIB"

  cp "$source_dylib" "$binary_path"
  write_macos_info_plist "$resources_dir"

  install_name_tool -id "@rpath/$FRAMEWORK_NAME/Versions/A/$FRAMEWORK_EXECUTABLE" "$binary_path"

  if [[ "${XYBRID_GODOT_SKIP_CODESIGN:-0}" != "1" ]] && command -v codesign >/dev/null 2>&1; then
    codesign --force --sign - "$framework_dir"
  fi
}

package_macos_framework() {
  local macos_targets="${XYBRID_GODOT_MACOS_TARGETS:-aarch64-apple-darwin x86_64-apple-darwin}"
  local target_libs=()

  for target in $macos_targets; do
    run_cargo_build --target "$target"
    target_libs+=("$TARGET_ROOT/$target/$PROFILE_DIR/libxybrid_godot.dylib")
  done

  local universal_lib="$TARGET_ROOT/$PROFILE_DIR/libxybrid_godot_universal.dylib"
  mkdir -p "$TARGET_ROOT/$PROFILE_DIR"

  if [[ "${#target_libs[@]}" -eq 1 ]]; then
    cp "${target_libs[0]}" "$universal_lib"
  else
    lipo -create "${target_libs[@]}" -output "$universal_lib"
  fi

  local dest="bindings/godot/addons/xybrid/bin/universal-apple-darwin/$PROFILE_DIR"
  mkdir -p "$dest"
  create_macos_framework "$universal_lib" "$dest"

  echo "packaged $dest/$FRAMEWORK_NAME"
}

case "$(uname -s)" in
  Darwin)
    DEFAULT_FEATURES="platform-macos"
    ;;
  Linux)
    PLATFORM="linux"
    LIB_NAME="libxybrid_godot.so"
    DEFAULT_FEATURES="platform-desktop"
    ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    PLATFORM="windows"
    LIB_NAME="xybrid_godot.dll"
    DEFAULT_FEATURES="platform-desktop"
    ;;
  *)
    echo "unsupported host platform: $(uname -s)" >&2
    exit 2
    ;;
esac

if [[ -z "$FEATURES" ]]; then
  FEATURES="$DEFAULT_FEATURES"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  package_macos_framework
  exit 0
fi

run_cargo_build

DEST="bindings/godot/addons/xybrid/bin/$PLATFORM/$PROFILE_DIR"
mkdir -p "$DEST"
cp "$TARGET_ROOT/$PROFILE_DIR/$LIB_NAME" "$DEST/$LIB_NAME"

echo "packaged $DEST/$LIB_NAME"
