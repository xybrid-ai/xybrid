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

# ============================================================================
# ORT Resolution: Find or download ONNX Runtime 1.23.2 static library
# ============================================================================
# ort-sys linking modes:
# 1. ORT_IOS_XCFWK_LOCATION: Expects ios-arm64/onnxruntime.framework (dynamic)
# 2. ORT_LIB_LOCATION: Expects directory with libonnxruntime.a (static)
# We use mode 2 since we built a static library, not a framework.
#
# Resolution order:
# 1. Vendored path (monorepo dev — symlink to vendor/ort-ios/)
# 2. Cached download (~/.xybrid/cache/ort-ios/onnxruntime.xcframework)
# 3. Download from HuggingFace and cache
# ============================================================================

ORT_VERSION="1.23.2"
ORT_HF_URL="https://huggingface.co/xybrid-ai/ios-onnxruntime/resolve/main/${ORT_VERSION}/onnxruntime.xcframework.tar.bz2"
ORT_CACHE_DIR="$HOME/.xybrid/cache/ort-ios/${ORT_VERSION}"

if [[ "$PLATFORM_NAME" == "iphoneos" || "$PLATFORM_NAME" == "iphonesimulator" ]]; then
  ORT_XCFRAMEWORK_BASE=""

  # --- Path 1: Vendored xcframework (monorepo dev with symlink) ---
  VENDORED_PATH="$PODS_TARGET_SRCROOT/Frameworks/onnxruntime.xcframework"
  if [[ -e "$VENDORED_PATH" ]]; then
    # Resolve symlinks (CocoaPods .symlinks/ breaks relative symlink resolution)
    if [[ -L "$VENDORED_PATH" ]]; then
      ORT_XCFRAMEWORK_REAL=$(cd -P "$VENDORED_PATH" 2>/dev/null && pwd -P)
      if [[ -d "$ORT_XCFRAMEWORK_REAL" ]]; then
        echo "=== ORT: Using vendored xcframework (resolved symlink) ==="
        echo "=== ORT: $VENDORED_PATH -> $ORT_XCFRAMEWORK_REAL ==="
        ORT_XCFRAMEWORK_BASE="$ORT_XCFRAMEWORK_REAL"
      fi
    else
      echo "=== ORT: Using vendored xcframework (direct) ==="
      ORT_XCFRAMEWORK_BASE="$VENDORED_PATH"
    fi
  fi

  # --- Path 2: Cached download ---
  if [[ -z "$ORT_XCFRAMEWORK_BASE" && -d "$ORT_CACHE_DIR/onnxruntime.xcframework" ]]; then
    echo "=== ORT: Using cached download at $ORT_CACHE_DIR ==="
    ORT_XCFRAMEWORK_BASE="$ORT_CACHE_DIR/onnxruntime.xcframework"
  fi

  # --- Path 3: Download from HuggingFace ---
  if [[ -z "$ORT_XCFRAMEWORK_BASE" ]]; then
    echo "=== ORT: Downloading ONNX Runtime ${ORT_VERSION} from HuggingFace ==="
    echo "=== ORT: URL: $ORT_HF_URL ==="
    mkdir -p "$ORT_CACHE_DIR"
    ARCHIVE_PATH="$ORT_CACHE_DIR/onnxruntime.xcframework.tar.bz2"

    if ! curl -fSL --progress-bar -o "$ARCHIVE_PATH" "$ORT_HF_URL"; then
      echo "ERROR: Failed to download ORT ${ORT_VERSION} from HuggingFace"
      echo "       URL: $ORT_HF_URL"
      echo "       Check your network connection and that the file exists"
      rm -f "$ARCHIVE_PATH"
      exit 1
    fi

    echo "=== ORT: Extracting xcframework ==="
    tar -xjf "$ARCHIVE_PATH" -C "$ORT_CACHE_DIR"
    rm -f "$ARCHIVE_PATH"

    if [[ -d "$ORT_CACHE_DIR/onnxruntime.xcframework" ]]; then
      echo "=== ORT: Successfully downloaded and cached ==="
      ORT_XCFRAMEWORK_BASE="$ORT_CACHE_DIR/onnxruntime.xcframework"
    else
      echo "ERROR: Extracted archive but onnxruntime.xcframework directory not found"
      echo "       Contents of $ORT_CACHE_DIR:"
      ls -la "$ORT_CACHE_DIR/" 2>/dev/null
      exit 1
    fi
  fi

  # --- Set ORT_LIB_LOCATION from resolved xcframework ---
  if [[ "$PLATFORM_NAME" == "iphoneos" ]]; then
    ORT_LIB_PATH="$ORT_XCFRAMEWORK_BASE/ios-arm64"
  else
    # Simulator builds need the simulator-ABI slice — the device `ios-arm64`
    # slice fails the link with "built for 'iOS'" (arm64 alone is not enough;
    # the object files carry a device platform marker). Newer xcframeworks
    # ship an `ios-arm64-simulator` slice; older archives are device-only, so
    # fall back to fetching the pyke `aarch64-apple-ios-sim` static lib — the
    # SAME artifact Bazel's //bazel:ort_ios_sim.bzl fetches (raw-LZMA2 tar,
    # decoded with `xz --format=raw`).
    ORT_LIB_PATH="$ORT_XCFRAMEWORK_BASE/ios-arm64-simulator"
    if [[ ! -f "$ORT_LIB_PATH/libonnxruntime.a" ]]; then
      SIM_CACHE_DIR="$ORT_CACHE_DIR/ios-arm64-simulator"
      if [[ ! -f "$SIM_CACHE_DIR/libonnxruntime.a" ]]; then
        SIM_URL="https://cdn.pyke.io/0/pyke:ort-rs/ms@${ORT_VERSION}/aarch64-apple-ios-sim.tar.lzma2"
        SIM_SHA256="8ae2dfc164b21d0a9f7b8ad046193eb20b4e9c198a21ba8dba0a71e962e15776"
        if ! command -v xz >/dev/null 2>&1; then
          echo "ERROR: xz is required to decode the ONNX Runtime simulator slice"
          echo "       Run: brew install xz"
          exit 1
        fi
        echo "=== ORT: Downloading iOS-simulator slice from pyke CDN ==="
        mkdir -p "$SIM_CACHE_DIR"
        SIM_ARCHIVE="$SIM_CACHE_DIR/ort.tar.lzma2"
        if ! curl -fSL --progress-bar -o "$SIM_ARCHIVE" "$SIM_URL"; then
          echo "ERROR: Failed to download ORT simulator slice from $SIM_URL"
          rm -f "$SIM_ARCHIVE"
          exit 1
        fi
        echo "$SIM_SHA256  $SIM_ARCHIVE" | shasum -a 256 -c - || {
          echo "ERROR: checksum mismatch for ORT simulator slice"
          rm -f "$SIM_ARCHIVE"
          exit 1
        }
        # pyke ships a raw LZMA2 stream (not an xz container); 64 MiB dict
        # matches ort-sys's own decoder settings.
        xz -dc --format=raw --lzma2=dict=64MiB "$SIM_ARCHIVE" | tar -x -C "$SIM_CACHE_DIR"
        rm -f "$SIM_ARCHIVE"
        if [[ ! -f "$SIM_CACHE_DIR/libonnxruntime.a" ]]; then
          echo "ERROR: libonnxruntime.a missing after extracting the simulator slice"
          exit 1
        fi
        echo "=== ORT: Simulator slice cached at $SIM_CACHE_DIR ==="
      fi
      ORT_LIB_PATH="$SIM_CACHE_DIR"
    fi
  fi

  if [[ -f "$ORT_LIB_PATH/libonnxruntime.a" ]]; then
    export ORT_LIB_LOCATION="$ORT_LIB_PATH"
    echo "=== ORT: ORT_LIB_LOCATION=$ORT_LIB_LOCATION ==="
    echo "=== ORT: Using ONNX Runtime ${ORT_VERSION} with CoreML EP ==="
    ls -la "$ORT_LIB_PATH/libonnxruntime.a"
  else
    echo "ERROR: libonnxruntime.a not found after ORT resolution"
    echo "       Expected: $ORT_LIB_PATH/libonnxruntime.a"
    echo "       xcframework base: $ORT_XCFRAMEWORK_BASE"
    ls -la "$ORT_XCFRAMEWORK_BASE/" 2>/dev/null || echo "       xcframework directory not found"
    exit 1
  fi

  # Export the resolved xcframework base for podspec xcconfig path references
  export XYBRID_ORT_XCFRAMEWORK="$ORT_XCFRAMEWORK_BASE"
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
