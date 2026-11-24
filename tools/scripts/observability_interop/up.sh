#!/usr/bin/env bash
# up.sh — start the self-hosted interop destinations and wait until they answer.
#
#   ./up.sh phoenix | langfuse | all
#
# Uses `docker compose` with compose.yaml and interop.env (fixed loopback
# ports, local-only credentials). On a Mac with OrbStack run `orbctl start`
# first. Langfuse needs about 4 GB of the VM's memory; Phoenix well under 1 GB.
# Pulls only the pinned image digests named in compose.yaml.
#
# Bash 3.2 compatible (the macOS /bin/bash).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$HERE/compose.yaml"
ENV_FILE="$HERE/interop.env"
TARGET="${1:-}"

usage() {
    echo "usage: $0 phoenix|langfuse|all" >&2
    exit 2
}

setting() {
    # The value of KEY in interop.env, unless the environment overrides it.
    local key="$1" value
    value="$(printenv "$key" 2>/dev/null || true)"
    if [ -z "$value" ]; then
        value="$(sed -n "s/^${key}=//p" "$ENV_FILE" | head -n 1)"
    fi
    printf '%s' "$value"
}

wait_for() {
    # wait_for NAME URL SECONDS: polls URL until it answers 2xx.
    local name="$1" url="$2" limit="$3" waited=0
    printf '==> waiting for %s at %s ' "$name" "$url"
    while [ "$waited" -lt "$limit" ]; do
        if curl -fsS -o /dev/null --max-time 5 "$url" 2>/dev/null; then
            echo "ready (${waited}s)"
            return 0
        fi
        printf '.'
        sleep 3
        waited=$((waited + 3))
    done
    echo " not ready after ${limit}s" >&2
    return 1
}

case "$TARGET" in
    phoenix) PROFILES="--profile phoenix" ;;
    langfuse) PROFILES="--profile langfuse" ;;
    all) PROFILES="--profile phoenix --profile langfuse" ;;
    *) usage ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is not on PATH (on macOS: install OrbStack, then 'orbctl start')" >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "error: the Docker daemon is not running (OrbStack: 'orbctl start')" >&2
    exit 1
fi

# shellcheck disable=SC2086  # PROFILES is a deliberate word list
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $PROFILES up -d --wait --wait-timeout 300 || {
    echo "error: docker compose up failed; 'docker compose -f $COMPOSE_FILE --env-file $ENV_FILE logs' shows why" >&2
    exit 1
}

if [ "$TARGET" = "phoenix" ] || [ "$TARGET" = "all" ]; then
    wait_for "Phoenix" "http://127.0.0.1:$(setting PHOENIX_HOST_PORT)/healthz" 120
    echo "    Phoenix $(curl -fsS "http://127.0.0.1:$(setting PHOENIX_HOST_PORT)/arize_phoenix_version" || echo '?')" \
        "UI: http://127.0.0.1:$(setting PHOENIX_HOST_PORT)"
fi
if [ "$TARGET" = "langfuse" ] || [ "$TARGET" = "all" ]; then
    # The first start runs the Postgres and ClickHouse migrations, then the
    # headless initialization creates the project and its API keys.
    wait_for "Langfuse" "http://127.0.0.1:$(setting LANGFUSE_HOST_PORT)/api/public/health" 300
    echo "    Langfuse $(curl -fsS "http://127.0.0.1:$(setting LANGFUSE_HOST_PORT)/api/public/health" || echo '?')" \
        "UI: http://127.0.0.1:$(setting LANGFUSE_HOST_PORT) (login $(setting LANGFUSE_INIT_USER_EMAIL) /" \
        "$(setting LANGFUSE_INIT_USER_PASSWORD))"
fi
