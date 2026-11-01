#!/usr/bin/env bash
# Resolve requirements.in (the direct pins) into the fully pinned
# requirements.txt the HF reference runner installs. The lock targets Linux
# x86_64 with CPU-only torch from the PyTorch CPU index, the platform the
# committed observations come from, so torch and torchvision carry their +cpu
# local versions. After relocking, regenerate the observations: their
# provenance.json pins this lock's sha256.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
uv pip compile requirements.in \
    --output-file requirements.txt \
    --python-version 3.12 \
    --python-platform x86_64-manylinux_2_28 \
    --index https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --custom-compile-command "tools/conformance/decision_hf/lock.sh" \
    --no-annotate \
    --quiet
