#!/usr/bin/env bash
# Managed TTS native-host test outputs.
. "$(dirname "$0")/../scripts/clean-lib.sh"
clean_paths bin obj
clean_run "$@"
