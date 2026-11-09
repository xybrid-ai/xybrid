#!/usr/bin/env bash
# Bazel-built browser runtime copies staged for the JavaScript package.
. "$(dirname "$0")/../../../tools/scripts/clean-lib.sh"
clean_paths artifacts
clean_run "$@"
