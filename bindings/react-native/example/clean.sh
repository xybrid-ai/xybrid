#!/usr/bin/env bash
# React Native example app: npm install, Expo state, and the native build
# output inside the prebuilt ios/ and android/ projects.
. "$(dirname "$0")/../../../tools/scripts/clean-lib.sh"
clean_paths node_modules .expo dist
clean_paths ios/build ios/Pods
clean_paths android/build android/app/build android/app/.cxx
clean_paths android/.gradle android/.kotlin
clean_run "$@"
