#!/usr/bin/env bash
# Companion Bazel outputs are owned by root clean.sh --bazel.
# This folder creates no local build products.
. "$(dirname "$0")/../../../tools/scripts/clean-lib.sh"
clean_run "$@"
