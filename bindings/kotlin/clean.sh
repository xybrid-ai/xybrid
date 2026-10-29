#!/usr/bin/env bash
# Kotlin binding: Gradle output, the natives staged into libs/ and the
# speech notices staged into src/main/assets/xybrid/zzz/.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths .gradle build 'libs/*/*.so' src/main/assets/xybrid/zzz
clean_run "$@"
