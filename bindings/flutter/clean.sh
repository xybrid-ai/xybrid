#!/usr/bin/env bash
# Flutter plugin: `flutter clean`, plus Gradle state and the cargokit target.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_command flutter clean
clean_paths build .dart_tool android/.gradle android/build
clean_paths rust/target cargokit/build_tool/.dart_tool
clean_run "$@"
