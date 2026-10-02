#!/usr/bin/env bash
# Web binding: pnpm install, bundles, and Playwright reports.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths node_modules dist playwright-report test-results
clean_paths example/node_modules example/dist
clean_run "$@"
