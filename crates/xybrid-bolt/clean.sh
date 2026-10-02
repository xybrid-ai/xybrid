#!/usr/bin/env bash
# xybrid-bolt: boltffi pack output and its own cargo target.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths dist target
clean_run "$@"
