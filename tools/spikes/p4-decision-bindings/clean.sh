#!/usr/bin/env bash
# P4 decision-binding spike: the C# harness's build output.
. "$(dirname "$0")/../../scripts/clean-lib.sh"
clean_paths csharp/bin csharp/obj
clean_run "$@"
