#!/usr/bin/env bash
# G1 comparator: its reports under out/ (compare.py gate, drift, check), and
# any Python bytecode.
. "$(dirname "$0")/../../scripts/clean-lib.sh"
clean_paths out __pycache__ tests/__pycache__
clean_run "$@"
