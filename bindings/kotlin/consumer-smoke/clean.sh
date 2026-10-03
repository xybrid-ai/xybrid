#!/usr/bin/env bash
# Kotlin AAR consumer smoke: Gradle output and the AAR copied into app/libs.
. "$(dirname "$0")/../../../tools/scripts/clean-lib.sh"
clean_paths .gradle build app/build app/libs
clean_run "$@"
