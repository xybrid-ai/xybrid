#!/usr/bin/env bash
# Windows bolt DLL smoke: dotnet build output and the staged xybrid_bolt.dll.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths bin obj xybrid_bolt.dll
clean_run "$@"
