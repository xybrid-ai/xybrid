#!/usr/bin/env bash
# React Native package: npm install, the bob build, Gradle output, and what
# pod install resolves into ios/.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths node_modules lib android/build android/.gradle .kotlin
clean_paths ios/Frameworks ios/XybridSwift
clean_run "$@"
