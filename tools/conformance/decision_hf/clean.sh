#!/usr/bin/env bash
# HF reference for the decision gates: the pinned snapshot (3.9 GB), the
# remote-code module cache, the cross-checks' GGUFs and llama-server build, and
# the run logs, all under .cache/, and any Python bytecode (a different
# XYBRID_DECISION_HF_CACHE is the caller's to clean).
. "$(dirname "$0")/../../scripts/clean-lib.sh"
clean_paths .cache __pycache__ tests/__pycache__
clean_run "$@"
