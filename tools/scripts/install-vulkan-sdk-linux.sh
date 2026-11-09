#!/usr/bin/env bash
# Install the x86_64 LunarG SDK for the shared setup-vulkan-sdk action.
# Usage: bash install-vulkan-sdk-linux.sh VERSION SHA256 DIRECTORY [URL]
set -euo pipefail

version=${1:?Vulkan SDK version is required}
expected_sha=${2:?Linux archive SHA-256 is required}
directory=${3:?SDK directory is required}
url=${4:-https://sdk.lunarg.com/sdk/download/$version/linux/vulkansdk-linux-x86_64-$version.tar.xz}

if [[ ! "$version" =~ ^[0-9]+(\.[0-9]+){3}$ || ! "$expected_sha" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Invalid Vulkan SDK version or SHA-256" >&2
  exit 1
fi

mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
archive="$directory/vulkansdk-$version.tar.xz"
sdk="$directory/$version/x86_64"
trap 'rm -f "$archive.part"' EXIT

verify_archive() {
  printf '%s  %s\n' "$expected_sha" "$1" | sha256sum --check --status
}

if ! verify_archive "$archive" 2>/dev/null; then
  echo "Downloading Vulkan SDK $version (up to 3 attempts, 5 minutes total)"
  # Recover HTTP, connection and interrupted-transfer errors. A slow/stalled
  # body times out after 30s; each attempt is capped at 90s. The outer timeout
  # also bounds retries and backoff, leaving time for apt and extraction.
  timeout --kill-after=5 300 curl --fail --location --silent --show-error \
    --connect-timeout 15 --max-time 90 --speed-limit 1024 --speed-time 30 \
    --retry 2 --retry-all-errors --retry-delay 5 \
    --output "$archive.part" "$url"
  if ! verify_archive "$archive.part"; then
    echo "Vulkan SDK archive SHA-256 mismatch" >&2
    exit 1
  fi
  mv "$archive.part" "$archive"
else
  echo "Using verified Vulkan SDK $version archive from cache"
fi

# Only the prebuilt x86_64 tree is needed; exclude SDK sources and examples.
# Always extract the verified archive so a previous interrupted install heals.
timeout --kill-after=5 90 tar -xJf "$archive" -C "$directory" "$version/x86_64"

# LunarG moved the loader into VulkanLoader/lib in SDK 1.4.350.0. Keep the
# top-level lib paths used by FindVulkan and llama-cpp-sys/build.rs working.
if [[ -f "$sdk/lib/VulkanLoader/lib/libvulkan.so.1" ]]; then
  ln -sfn VulkanLoader/lib/libvulkan.so.1 "$sdk/lib/libvulkan.so.1"
  ln -sfn libvulkan.so.1 "$sdk/lib/libvulkan.so"
fi

for file in bin/glslc include/vulkan/vulkan_core.h include/vk_video/vulkan_video_codecs_common.h include/spirv/unified1/spirv.hpp lib/libvulkan.so; do
  if [[ ! -f "$sdk/$file" ]]; then
    echo "Incomplete Vulkan SDK: missing $sdk/$file" >&2
    exit 1
  fi
done
if [[ ! -x "$sdk/bin/glslc" ]]; then
  echo "Vulkan SDK glslc is not executable" >&2
  exit 1
fi

# Match the upstream action's Linux environment, including the relocated loader.
{
  printf 'VULKAN_SDK=%s\n' "$sdk"
  printf 'VULKAN_VERSION=%s\n' "$version"
  printf 'VK_LAYER_PATH=%s/share/vulkan/explicit_layer.d\n' "$sdk"
  printf 'LD_LIBRARY_PATH=%s/lib/VulkanLoader/lib:%s/lib%s\n' "$sdk" "$sdk" "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
} >> "${GITHUB_ENV:?GITHUB_ENV is required}"
printf '%s/bin\n' "$sdk" >> "${GITHUB_PATH:?GITHUB_PATH is required}"
echo "Installed Vulkan SDK $version at $sdk"
