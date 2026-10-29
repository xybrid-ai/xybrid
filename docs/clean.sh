#!/usr/bin/env bash
# Docs site: pnpm install and Next.js output.
. "$(dirname "$0")/../tools/scripts/clean-lib.sh"
clean_paths node_modules .next .source out build coverage
clean_run "$@"
