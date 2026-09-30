#!/usr/bin/env bash
# Unity bolt compile check: dotnet build output.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths bin obj
clean_run "$@"
