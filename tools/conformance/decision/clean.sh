#!/usr/bin/env bash
# Upstream decision oracle: the CMake build (llama.cpp, decision-oracle,
# llama-server) and work/ (the downloaded GGUF, metadata variants, run logs).
. "$(dirname "$0")/../../scripts/clean-lib.sh"
clean_paths build work
clean_run "$@"
