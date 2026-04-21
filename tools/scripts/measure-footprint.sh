#!/usr/bin/env bash
# measure-footprint.sh - Measure shipping binary footprint across platforms.
#
# Builds xybrid-ffi (the C ABI shared library consumers actually link against)
# in release mode for selected targets, then reports stripped size of our
# library plus any bundled runtime dependencies (ORT, llama.cpp, MLX).
#
# Usage:
#   ./tools/scripts/measure-footprint.sh android-arm64
#   ./tools/scripts/measure-footprint.sh ios-arm64
#   ./tools/scripts/measure-footprint.sh macos-arm64
#   ./tools/scripts/measure-footprint.sh all
#
# Output: prints a markdown table to stdout and writes docs/footprint.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

REPORT="$REPO_ROOT/docs/footprint.md"
mkdir -p "$(dirname "$REPORT")"

TARGET="${1:-all}"

human_size() {
    # bytes -> MB with one decimal
    local bytes="$1"
    awk -v b="$bytes" 'BEGIN { printf "%.1f MB", b / 1024 / 1024 }'
}

file_size_bytes() {
    local path="$1"
    if [ -f "$path" ]; then
        stat -f%z "$path" 2>/dev/null || stat -c%s "$path" 2>/dev/null
    else
        echo 0
    fi
}

dir_size_bytes() {
    local path="$1"
    if [ -d "$path" ]; then
        du -sk "$path" | awk '{print $1 * 1024}'
    else
        echo 0
    fi
}

measure_android_arm64() {
    echo "### Android (arm64-v8a)" | tee -a "$REPORT"
    echo "" | tee -a "$REPORT"

    echo "Building xybrid-ffi for aarch64-linux-android (release, platform-android)..."
    cargo xtask build-ffi --release --target aarch64-linux-android --platform-preset platform-android

    local xybrid_so="$REPO_ROOT/target/aarch64-linux-android/release/libxybrid_ffi.so"
    local xybrid_stripped="$REPO_ROOT/target/aarch64-linux-android/release/libxybrid_ffi.stripped.so"

    if [ ! -f "$xybrid_so" ]; then
        echo "ERROR: expected $xybrid_so not found"
        return 1
    fi

    # Strip using NDK's llvm-strip
    local ndk_strip
    ndk_strip=$(find ~/Library/Android/sdk/ndk -name 'llvm-strip' -path '*aarch64*' 2>/dev/null | head -1)
    if [ -z "$ndk_strip" ]; then
        ndk_strip=$(find ~/Library/Android/sdk/ndk -name 'llvm-strip' 2>/dev/null | head -1)
    fi
    cp "$xybrid_so" "$xybrid_stripped"
    if [ -n "$ndk_strip" ]; then
        "$ndk_strip" "$xybrid_stripped" || true
    fi

    local xybrid_bytes
    xybrid_bytes=$(file_size_bytes "$xybrid_stripped")

    # ORT Android .so (dynamic, shipped separately)
    local ort_so="$REPO_ROOT/vendor/ort-android/arm64-v8a/libonnxruntime.so"
    local ort_bytes
    ort_bytes=$(file_size_bytes "$ort_so")

    # libc++_shared
    local libcxx_so="$REPO_ROOT/vendor/ort-android/arm64-v8a/libc++_shared.so"
    local libcxx_bytes
    libcxx_bytes=$(file_size_bytes "$libcxx_so")

    local total=$((xybrid_bytes + ort_bytes + libcxx_bytes))

    {
        echo "| Component | Size |"
        echo "|---|---|"
        echo "| libxybrid_ffi.so (stripped) | $(human_size "$xybrid_bytes") |"
        echo "| libonnxruntime.so (ORT, dynamic) | $(human_size "$ort_bytes") |"
        echo "| libc++_shared.so | $(human_size "$libcxx_bytes") |"
        echo "| **Total shipped (arm64 slice)** | **$(human_size "$total")** |"
        echo ""
        echo "Notes:"
        echo "- \`libxybrid_ffi.so\` includes llama.cpp statically linked (platform-android preset)."
        echo "- ORT is dynamically loaded on Android (ort-dynamic feature)."
        echo "- MLX is not built for Android."
        echo ""
    } | tee -a "$REPORT"
}

measure_ios_arm64() {
    echo "### iOS (arm64 device)" | tee -a "$REPORT"
    echo "" | tee -a "$REPORT"

    echo "Building xybrid-ffi for aarch64-apple-ios (release, platform-ios)..."
    cargo xtask build-ffi --release --target aarch64-apple-ios --platform-preset platform-ios

    local xybrid_a="$REPO_ROOT/target/aarch64-apple-ios/release/libxybrid_ffi.a"
    if [ ! -f "$xybrid_a" ]; then
        echo "ERROR: expected $xybrid_a not found"
        return 1
    fi
    local xybrid_bytes
    xybrid_bytes=$(file_size_bytes "$xybrid_a")

    # ORT iOS static library (bundled for iOS)
    local ort_a="$REPO_ROOT/vendor/ort-ios/onnxruntime.xcframework/ios-arm64/libonnxruntime.a"
    local ort_bytes
    ort_bytes=$(file_size_bytes "$ort_a")

    # MLX xcframework arm64 slice
    local mlx_dir="$REPO_ROOT/vendor/mlx-apple/mlx.xcframework/ios-arm64"
    local mlx_bytes
    mlx_bytes=$(dir_size_bytes "$mlx_dir")

    local total_with_mlx=$((xybrid_bytes + ort_bytes + mlx_bytes))
    local total_without_mlx=$((xybrid_bytes + ort_bytes))

    {
        echo "| Component | Size |"
        echo "|---|---|"
        echo "| libxybrid_ffi.a (static archive — includes llama.cpp) | $(human_size "$xybrid_bytes") |"
        echo "| libonnxruntime.a (ORT static, ios-arm64 slice) | $(human_size "$ort_bytes") |"
        echo "| mlx.xcframework/ios-arm64 | $(human_size "$mlx_bytes") |"
        echo "| **Total archive size (arm64, with MLX)** | **$(human_size "$total_with_mlx")** |"
        echo "| **Total archive size (arm64, no MLX)** | **$(human_size "$total_without_mlx")** |"
        echo ""
        echo "Notes:"
        echo "- Static archive sizes are PRE-LINK and include unreferenced symbols that dead-code-elimination strips."
        echo "- Actual app binary growth is typically 40-60% of archive size after linker optimization."
        echo "- MLX linked only when \`llm-mlx-runtime\` feature enabled; skeleton \`llm-mlx\` adds no MLX bytes."
        echo ""
    } | tee -a "$REPORT"
}

measure_macos_arm64() {
    echo "### macOS (arm64)" | tee -a "$REPORT"
    echo "" | tee -a "$REPORT"

    echo "Building xybrid-ffi for aarch64-apple-darwin (release, platform-macos)..."
    cargo xtask build-ffi --release --target aarch64-apple-darwin --platform-preset platform-macos

    local xybrid_dylib="$REPO_ROOT/target/aarch64-apple-darwin/release/libxybrid_ffi.dylib"
    local xybrid_stripped="$REPO_ROOT/target/aarch64-apple-darwin/release/libxybrid_ffi.stripped.dylib"

    if [ ! -f "$xybrid_dylib" ]; then
        echo "ERROR: expected $xybrid_dylib not found"
        return 1
    fi

    cp "$xybrid_dylib" "$xybrid_stripped"
    strip -x "$xybrid_stripped" || true

    local xybrid_bytes
    xybrid_bytes=$(file_size_bytes "$xybrid_stripped")

    {
        echo "| Component | Size |"
        echo "|---|---|"
        echo "| libxybrid_ffi.dylib (stripped, includes llama.cpp + ORT static + MLX link) | $(human_size "$xybrid_bytes") |"
        echo ""
        echo "Notes:"
        echo "- macOS build uses \`platform-macos\` preset: ORT + CoreML + Candle-Metal + llama.cpp + MLX."
        echo "- ORT is downloaded dynamically on macOS (not bundled into dylib)."
        echo ""
    } | tee -a "$REPORT"
}

# Initialize report
{
    echo "# Xybrid Footprint Measurements"
    echo ""
    echo "Generated: $(date -u +"%Y-%m-%d %H:%M UTC")"
    echo ""
    echo "Release builds with \`strip\` applied. All sizes are for a single architecture slice."
    echo ""
} > "$REPORT"

case "$TARGET" in
    android-arm64) measure_android_arm64 ;;
    ios-arm64) measure_ios_arm64 ;;
    macos-arm64) measure_macos_arm64 ;;
    all)
        measure_macos_arm64
        measure_android_arm64
        measure_ios_arm64
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Valid: android-arm64 | ios-arm64 | macos-arm64 | all"
        exit 1
        ;;
esac

echo ""
echo "Report written to: $REPORT"
