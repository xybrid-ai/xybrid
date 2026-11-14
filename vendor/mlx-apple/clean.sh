#!/usr/bin/env bash
# vendor/mlx-apple: the MLX xcframework a download or a source build installs,
# and the markers that record which one is installed.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths mlx.xcframework .installed-sha256 .installed-source-pin
clean_run "$@"
