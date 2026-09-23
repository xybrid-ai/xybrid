#!/usr/bin/env bash
# Build and stage native artifacts for the selected Python interpreter.
# PYTHON defaults to python3; XYBRID_FEATURES and DEBUG=1 are optional.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/build_python_bolt.py" "$@"
