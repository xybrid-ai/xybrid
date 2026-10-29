#!/usr/bin/env bash
# Android example app: Gradle output.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths .gradle build app/build
clean_run "$@"
