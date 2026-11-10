#!/usr/bin/env bash
# down.sh — stop the interop destinations.
#
#   ./down.sh             # stop and remove the containers, keep Langfuse's volumes
#   ./down.sh --volumes   # also delete the volumes: the next ./up.sh starts empty
#
# Phoenix keeps nothing between runs either way. Use --volumes between
# acceptance runs: the Python scenario reuses a fixed trace id, and a fresh
# backend keeps the readback limited to the run being checked.
#
# Bash 3.2 compatible (the macOS /bin/bash).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA=""
case "${1:-}" in
    "") ;;
    --volumes|-v) EXTRA="--volumes" ;;
    *) echo "usage: $0 [--volumes]" >&2; exit 2 ;;
esac

# shellcheck disable=SC2086  # EXTRA is empty or one flag
docker compose -f "$HERE/compose.yaml" --env-file "$HERE/interop.env" \
    --profile phoenix --profile langfuse down --remove-orphans $EXTRA
