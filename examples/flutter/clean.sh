#!/usr/bin/env bash
# Flutter example app: `flutter clean`, plus the CocoaPods and Gradle state it
# leaves behind.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_command flutter clean
clean_paths build .dart_tool ios/Pods ios/.symlinks macos/Pods
clean_paths android/.gradle android/.kotlin
clean_run "$@"
