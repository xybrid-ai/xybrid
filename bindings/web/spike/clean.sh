#!/usr/bin/env bash
# Browser runtime spike: generated runtime copies, model and Vite output.
. "$(dirname "$0")/../../../tools/scripts/clean-lib.sh"
clean_paths public dist
clean_run "$@"
