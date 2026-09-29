#!/usr/bin/env bash
# Resolve each requirements-<env>.in (the direct pins) into the fully pinned
# requirements-<env>.txt the conformance scripts install. The resolution targets
# macOS arm64; Linux x86_64 with the CPU-only torch index (UV_INDEX, as in CI)
# resolves the same versions, torch as the +cpu build the == pin accepts. After
# relocking, update the environment digests in provenance.json and regenerate
# what the changed environment builds.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
for env in forms labels convert; do
    # macOS 14 is the oldest target torch 2.14 ships arm64 wheels for.
    MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile "requirements-$env.in" \
        --output-file "requirements-$env.txt" \
        --python-version 3.12 \
        --python-platform aarch64-apple-darwin \
        --custom-compile-command "tools/conformance/lock.sh" \
        --no-annotate \
        --quiet
done
