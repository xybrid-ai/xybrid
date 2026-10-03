#!/usr/bin/env bash
# Kotlin binding: Gradle output and the natives staged into libs/.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths .gradle build 'libs/*/*.so'
clean_run "$@"
