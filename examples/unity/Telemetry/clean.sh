#!/usr/bin/env bash
# Unity telemetry project: build output and the Library import cache, which
# Unity rebuilds on open.
. "$(dirname "$0")/../../../tools/scripts/clean-lib.sh"
clean_paths '[Ll]ibrary' '[Tt]emp' '[Oo]bj' '[Bb]uild' '[Bb]uilds' '[Ll]ogs'
clean_run "$@"
