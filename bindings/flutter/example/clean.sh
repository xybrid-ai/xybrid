#!/usr/bin/env bash
# Flutter plugin example app: `flutter clean`, plus the CocoaPods and Gradle
# state it leaves behind.
. "$(dirname "$0")/../../../tools/scripts/clean-lib.sh"
clean_command flutter clean
clean_paths build .dart_tool macos/Pods android/.gradle
clean_run "$@"
