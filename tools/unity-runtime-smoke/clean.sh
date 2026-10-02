#!/usr/bin/env bash
# Unity runtime smoke project: the Library import cache and build output.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths Library Temp Logs Build
clean_run "$@"
