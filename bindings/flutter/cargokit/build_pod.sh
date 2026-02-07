#!/bin/sh
set -e

BASEDIR=$(dirname "$0")

# Workaround for https://github.com/dart-lang/pub/issues/4010
BASEDIR=$(cd "$BASEDIR" ; pwd -P)

# Remove XCode SDK from path. Otherwise this breaks tool compilation when building iOS project
NEW_PATH=`echo $PATH | tr ":" "\n" | grep -v "Contents/Developer/" | tr "\n" ":"`

export PATH=${NEW_PATH%?} # remove trailing :

env

# Platform name (macosx, iphoneos, iphonesimulator)
export CARGOKIT_DARWIN_PLATFORM_NAME=$PLATFORM_NAME

# Arctive architectures (arm64, armv7, x86_64), space separated.
export CARGOKIT_DARWIN_ARCHS=$ARCHS

# Current build configuration (Debug, Release)
export CARGOKIT_CONFIGURATION=$CONFIGURATION

# Path to directory containing Cargo.toml.
export CARGOKIT_MANIFEST_DIR=$PODS_TARGET_SRCROOT/$1

# Temporary directory for build artifacts.
export CARGOKIT_TARGET_TEMP_DIR=$TARGET_TEMP_DIR

# Output directory for final artifacts.
export CARGOKIT_OUTPUT_DIR=$PODS_CONFIGURATION_BUILD_DIR/$PRODUCT_NAME

# Directory to store built tool artifacts.
export CARGOKIT_TOOL_TEMP_DIR=$TARGET_TEMP_DIR/build_tool

# Directory inside root project. Not necessarily the top level directory of root project.
export CARGOKIT_ROOT_PROJECT_DIR=$SRCROOT

# Set rustflags for iOS builds to enable fp16 CPU feature
# Required by gemm-f16 crate (used by Candle for Whisper)
# See: https://github.com/sarah-quinones/gemm/issues/31
if [[ "$PLATFORM_NAME" == "iphoneos" ]]; then
  export CARGO_TARGET_AARCH64_APPLE_IOS_RUSTFLAGS="-Ctarget-feature=+fp16"
  echo "=== Setting CARGO_TARGET_AARCH64_APPLE_IOS_RUSTFLAGS for fp16 support ==="
elif [[ "$PLATFORM_NAME" == "iphonesimulator" ]]; then
  export CARGO_TARGET_AARCH64_APPLE_IOS_SIM_RUSTFLAGS="-Ctarget-feature=+fp16"
  echo "=== Setting CARGO_TARGET_AARCH64_APPLE_IOS_SIM_RUSTFLAGS for fp16 support ==="
fi

# Set ORT_LIB_LOCATION for iOS builds to point to custom ONNX Runtime static library
# This allows ort crate to link against our custom ORT 1.23.2 build with CoreML EP
# NOTE: We use a custom build because the onnxruntime-c CocoaPod is stuck at 1.20.0
#
# ort-sys linking modes:
# 1. ORT_IOS_XCFWK_LOCATION: Expects ios-arm64/onnxruntime.framework (dynamic framework)
# 2. ORT_LIB_LOCATION: Expects directory with libonnxruntime.a (static library)
# We use mode 2 since we built a static library, not a framework.
if [[ "$PLATFORM_NAME" == "iphoneos" || "$PLATFORM_NAME" == "iphonesimulator" ]]; then
  # Custom xcframework is bundled in the pod's Frameworks directory (may be a symlink)
  # Structure: ios/Frameworks/onnxruntime.xcframework/ios-arm64/libonnxruntime.a
  ORT_XCFRAMEWORK_BASE="$PODS_TARGET_SRCROOT/Frameworks/onnxruntime.xcframework"

  # If the xcframework is a symlink, resolve it to the real path
  # This handles cases where CocoaPods accesses the plugin through .symlinks/ which
  # breaks relative symlink resolution (.. navigates the logical symlink path, not physical)
  # Using cd -P ensures we resolve to the physical filesystem path first
  if [[ -L "$ORT_XCFRAMEWORK_BASE" ]]; then
    ORT_XCFRAMEWORK_REAL=$(cd -P "$ORT_XCFRAMEWORK_BASE" 2>/dev/null && pwd -P)
    if [[ -d "$ORT_XCFRAMEWORK_REAL" ]]; then
      echo "=== Resolved xcframework symlink: $ORT_XCFRAMEWORK_BASE -> $ORT_XCFRAMEWORK_REAL ==="
      ORT_XCFRAMEWORK_BASE="$ORT_XCFRAMEWORK_REAL"
    fi
  fi

  # Determine the correct library path based on platform
  if [[ "$PLATFORM_NAME" == "iphoneos" ]]; then
    ORT_LIB_PATH="$ORT_XCFRAMEWORK_BASE/ios-arm64"
  else
    # Simulator build - we only have arm64 for now (M1+ Macs)
    ORT_LIB_PATH="$ORT_XCFRAMEWORK_BASE/ios-arm64"
  fi

  if [[ -f "$ORT_LIB_PATH/libonnxruntime.a" ]]; then
    # Only set ORT_LIB_LOCATION for static library linking
    # Do NOT set ORT_IOS_XCFWK_LOCATION since we don't have a .framework bundle
    export ORT_LIB_LOCATION="$ORT_LIB_PATH"
    echo "=== Setting ORT_LIB_LOCATION=$ORT_LIB_LOCATION ==="
    echo "=== Found libonnxruntime.a for static linking ==="
    echo "=== Using custom ONNX Runtime 1.23.2 with CoreML EP ==="
    ls -la "$ORT_LIB_PATH/libonnxruntime.a"
  else
    echo "ERROR: Custom ONNX Runtime static library not found"
    echo "       Expected: $ORT_LIB_PATH/libonnxruntime.a"
    ls -la "$PODS_TARGET_SRCROOT/Frameworks/" 2>/dev/null || echo "       Frameworks directory not found"
    ls -la "$ORT_XCFRAMEWORK_BASE/" 2>/dev/null || echo "       xcframework directory not found"
    exit 1
  fi
fi

FLUTTER_EXPORT_BUILD_ENVIRONMENT=(
  "$PODS_ROOT/../Flutter/ephemeral/flutter_export_environment.sh" # macOS
  "$PODS_ROOT/../Flutter/flutter_export_environment.sh" # iOS
)

for path in "${FLUTTER_EXPORT_BUILD_ENVIRONMENT[@]}"
do
  if [[ -f "$path" ]]; then
    source "$path"
  fi
done

sh "$BASEDIR/run_build_tool.sh" build-pod "$@"

# Copy the built static library to BUILT_PRODUCTS_DIR so it can be found by -force_load
# The library is built by cargo in TARGET_TEMP_DIR, we need it in BUILT_PRODUCTS_DIR
RUST_LIB_NAME="lib$2.a"
if [[ -f "${TARGET_TEMP_DIR}/${RUST_LIB_NAME}" ]]; then
  echo "=== Copying ${RUST_LIB_NAME} to ${BUILT_PRODUCTS_DIR} ==="
  cp "${TARGET_TEMP_DIR}/${RUST_LIB_NAME}" "${BUILT_PRODUCTS_DIR}/${RUST_LIB_NAME}"
else
  # Check in the cargo target directory
  CARGO_OUT="${TARGET_TEMP_DIR}/aarch64-apple-ios-sim/debug/${RUST_LIB_NAME}"
  if [[ -f "${CARGO_OUT}" ]]; then
    echo "=== Copying ${RUST_LIB_NAME} from cargo target to ${BUILT_PRODUCTS_DIR} ==="
    cp "${CARGO_OUT}" "${BUILT_PRODUCTS_DIR}/${RUST_LIB_NAME}"
  else
    echo "=== Looking for ${RUST_LIB_NAME} in ${TARGET_TEMP_DIR}... ==="
    find "${TARGET_TEMP_DIR}" -name "${RUST_LIB_NAME}" -type f 2>/dev/null | head -5
  fi
fi

# Make a symlink from built framework to phony file, which will be used as input to
# build script. This should force rebuild (podspec currently doesn't support alwaysOutOfDate
# attribute on custom build phase)
ln -fs "$OBJROOT/XCBuildData/build.db" "${BUILT_PRODUCTS_DIR}/cargokit_phony"
ln -fs "${BUILT_PRODUCTS_DIR}/${EXECUTABLE_PATH}" "${BUILT_PRODUCTS_DIR}/cargokit_phony_out"
