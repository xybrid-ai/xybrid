#!/usr/bin/env bash
# @xybrid/observability: pnpm install, bundles, and the packed archive.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths node_modules dist test-results
clean_paths LICENSE NOTICE
clean_run "$@"
