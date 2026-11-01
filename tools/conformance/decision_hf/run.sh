#!/usr/bin/env bash
# Run the HF reference runner (run_reference.py) in its pinned environment.
#
#   tools/conformance/decision_hf/run.sh [run_reference.py options] CASE...
#   tools/conformance/decision_hf/run.sh synthetic_median.py [options]
#   tools/conformance/decision_hf/run.sh tests [unittest options]
#
# The environment is Python 3.12 with requirements.txt, built by uv; nothing is
# installed into the system Python. The lock targets Linux x86_64 with CPU-only
# torch, so this sets the PyTorch CPU index for uv there (unless UV_INDEX is
# already set). The snapshot must be downloaded first (download.sh); the run is
# offline: HF_HUB_OFFLINE is set, and the snapshot's code is imported from
# $XYBRID_DECISION_HF_CACHE/modules (default: .cache/ in this folder). `tests`
# runs tests/ (torch, no weights). See README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${XYBRID_DECISION_HF_CACHE:-$HERE/.cache}"

command -v uv >/dev/null || { echo "run: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
if [ "$(uname -s)" = Linux ]; then
    export UV_INDEX="${UV_INDEX:-https://download.pytorch.org/whl/cpu}"
    export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_MODULES_CACHE="$CACHE/modules"
export XYBRID_DECISION_HF_CACHE="$CACHE"
# -I: neither the working directory nor PYTHON* variables reach the import path;
# -B: no bytecode is written next to these scripts.
python=(uv run --no-project --python 3.12 --with-requirements "$HERE/requirements.txt" python -I -B)
case "${1:-}" in
    synthetic_median.py)
        shift
        exec "${python[@]}" "$HERE/synthetic_median.py" "$@" </dev/null
        ;;
    tests)
        shift
        exec "${python[@]}" -m unittest discover -s "$HERE/tests" "$@" </dev/null
        ;;
esac
exec "${python[@]}" "$HERE/run_reference.py" "$@" </dev/null
