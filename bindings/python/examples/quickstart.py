#!/usr/bin/env python3
"""Quickstart: on-device LLM text generation with the xybrid Python SDK.

Loads an instruction-tuned LLM and generates a bounded, deterministic
completion — everything runs locally, no API key required.

Usage:
    python examples/quickstart.py [MODEL_DIR]

MODEL_DIR is a directory holding a `model_metadata.json` and its weights.
It defaults to the LFM2.5-1.2B GGUF fixture shipped in this repo, so from the
repo root you can just run:

    python bindings/python/examples/quickstart.py

To use a registry model instead of a local directory, swap
`XybridModel.from_directory(...)` for `XybridModel.from_registry("<id>")`
(e.g. "kokoro-82m"), and read `result.audio_bytes` for speech models.

Before running, build and bundle the native library once:

    ./tools/scripts/build-python-bolt.sh
    pip install -e bindings/python      # in a Python >= 3.10 environment
"""

from __future__ import annotations

import sys
import time

import xybrid

DEFAULT_MODEL_DIR = "integration-tests/fixtures/models/lfm2.5-1.2b-instruct-bf16-gguf"


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_DIR

    # Anonymous init: local inference, telemetry disabled. Pass an api_key to
    # light up the platform features (see xybrid.init docstring).
    xybrid.init()

    print(f"loading {model_dir} ...")
    model = xybrid.XybridModel.from_directory(model_dir)
    print(f"loaded: {model.model_id} v{model.version} | is_llm={model.is_llm}")

    # Bounded, deterministic decoding so the example finishes quickly.
    # GenerationConfigs.greedy()/creative() are ready-made presets; here we
    # spell out a config to also cap the token count.
    config = xybrid.XybridGenerationConfig(
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        min_p=None,
        top_k=0,
        repetition_penalty=None,
        stop_sequences=[],
        grammar=None,
    )
    options = xybrid.XybridRunOptions(
        generation_config=config,
        abort_on=[],
        fallback_to_cloud=False,
        max_grace_tokens=0,
        correlation_id=None,
    )

    prompt = "What is the capital of France? Answer in one short sentence."
    print(f"\nprompt: {prompt}")

    t0 = time.time()
    result = model.run(xybrid.XybridEnvelope.text(prompt), options)
    wall = time.time() - t0

    print(f"\n--- generated ---\n{result.text}\n-----------------")
    metrics = result.metrics
    print(
        f"tokens_out={metrics.tokens_out} "
        f"tok/s={metrics.tokens_per_second:.1f} "
        f"latency={result.latency_seconds:.2f}s (wall {wall:.1f}s)"
    )

    model.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
