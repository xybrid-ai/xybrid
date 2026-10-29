#!/usr/bin/env bash
# Apple binding: SwiftPM build state and the XCFrameworks unzipped from Bazel.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths .build 'XCFrameworks/*.xcframework'
clean_run "$@"
