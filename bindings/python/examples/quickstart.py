#!/usr/bin/env python3
"""Quickstart: on-device LLM text generation with the xybrid Python SDK.

Loads an instruction-tuned LLM and generates a bounded, deterministic
completion — everything runs locally, no API key required.

Usage:
    python examples/quickstart.py [MODEL]

MODEL selects what to load and may be:
  - a registry id            e.g. "qwen2.5-0.5b-instruct" (downloaded on first run)
  - a Hugging Face repo      e.g. "xybrid-ai/kokoro-82m"
  - a local model directory  a folder holding model_metadata.json + weights

With no argument it defaults to a small registry LLM, so it runs from a fresh
checkout with nothing to fetch by hand (the model downloads on first run):

    python bindings/python/examples/quickstart.py

To run the local LFM2.5-1.2B GGUF example instead, point it at a directory you
have downloaded — these large fixtures are NOT committed to the repo:

    python bindings/python/examples/quickstart.py \
        integration-tests/fixtures/models/lfm2.5-1.2b-instruct-bf16-gguf

Before running, build and bundle the native library once:

    ./tools/scripts/build-python-bolt.sh
    pip install -e bindings/python      # in a Python >= 3.10 environment
"""

from __future__ import annotations

import os
import sys
import time

import xybrid

DEFAULT_MODEL = "qwen2.5-0.5b-instruct"


def load_model(spec: str) -> xybrid.XybridModel:
    """Load a model from a local directory, a Hugging Face repo, or the registry."""

    if os.path.isdir(spec):
        return xybrid.XybridModel.from_directory(spec)
    if "/" in spec:
        return xybrid.XybridModel.from_huggingface(spec)
    return xybrid.XybridModel.from_registry(spec)


def main() -> int:
    spec = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    # Anonymous init: local inference, telemetry disabled. Pass an api_key to
    # light up the platform features (see xybrid.init docstring).
    xybrid.init()

    print(f"loading {spec} (registry/HF models download on first run) ...")
    t0 = time.time()
    try:
        model = load_model(spec)
    except xybrid.XybridError as exc:
        print(f"could not load '{spec}': {exc}", file=sys.stderr)
        print("pass a registry id, a Hugging Face repo, or a local model directory.", file=sys.stderr)
        return 1
    print(f"loaded: {model.model_id} v{model.version} | is_llm={model.is_llm} ({time.time() - t0:.1f}s)")

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
        tools=[],
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
    try:
        result = model.run(xybrid.XybridEnvelope.text(prompt), options)
    except xybrid.XybridError as exc:
        print(f"inference failed: {exc}", file=sys.stderr)
        model.close()
        return 1
    wall = time.time() - t0

    # LLMs return text; a speech model would surface bytes in audio_bytes.
    if result.text is not None:
        print(f"\n--- generated ---\n{result.text}\n-----------------")
    elif result.audio_bytes is not None:
        print(f"\n(model returned {len(result.audio_bytes)} bytes of audio, not text)")
    else:
        print(f"\n(no text output; output_type={result.output_type})")

    metrics = result.metrics
    tps = metrics.tokens_per_second or 0.0
    print(
        f"tokens_out={metrics.tokens_out} tok/s={tps:.1f} "
        f"latency={result.latency_seconds:.2f}s (wall {wall:.1f}s)"
    )

    model.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
