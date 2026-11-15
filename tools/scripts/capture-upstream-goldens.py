#!/usr/bin/env python3
"""capture-upstream-goldens.py: write the Tier B ("upstream") goldens that
crates/xybrid-core/tests/mlx_golden_parity.rs audits the self goldens against.

A Tier B golden is the greedy token stream Python mlx-lm produces for the
prompt ids of a self golden, with mlx-lm's top-2 logprob margin at every step.
The ignored `golden_upstream_audit` test then compares the two stacks: equal
streams pass, a divergence where upstream's margin is a near-tie is noise, and
a divergence at index 0 or at a wide margin fails.

Run it after `capture-golden-tokens.sh --self`, with the same bundles staged:

    python3 -m venv /tmp/mlx-ref && /tmp/mlx-ref/bin/pip install mlx==<pin> mlx-lm
    XYBRID_MLX_QWEN_DIR=/path/to/bundle \
        /tmp/mlx-ref/bin/python tools/scripts/capture-upstream-goldens.py
    XYBRID_MLX_QWEN_DIR=/path/to/bundle cargo test -p xybrid-core \
        --no-default-features --features llm-mlx-runtime,llm-llamacpp \
        --test mlx_golden_parity -- --ignored golden_upstream_audit --nocapture

Install the `mlx` release vendor/mlx-apple/UPSTREAM_VERSIONS.txt pins, so both
stacks run the same kernels.

The reference is computed in float32, to measure how close the two candidates
really are at the step where the streams part. Both stacks themselves compute
in the checkpoint's own dtype (xybrid's Qwen and LFM adapters do, as mlx-lm
does by default), and for a bfloat16 bundle that rounds every logit to steps
of about 0.125: each stack is then off by a few such steps, in its own way,
and the two part wherever the true margin is smaller than that. Margins read
from a bfloat16 run are quantized the same way and cannot tell a near-tie from
a real difference. `--native-dtype` keeps mlx-lm's default.

Goldens are written to <bundle>/.golden/ and are never committed.
"""

import argparse
import datetime
import json
import os
import pathlib
import platform
import sys

BUNDLE_VARS = (
    "XYBRID_MLX_QWEN_DIR",
    "XYBRID_MLX_QWEN_4B_DIR",
    "XYBRID_MLX_LFM25_DIR",
    "XYBRID_MLX_GEMMA_DIR",
)


def capture(bundle: pathlib.Path, native_dtype: bool) -> int:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    self_goldens = sorted((bundle / ".golden").glob("self-greedy-*.json"))
    if not self_goldens:
        print(
            f"skip: {bundle} has no self goldens (run capture-golden-tokens.sh --self)",
            file=sys.stderr,
        )
        return 0

    model, tokenizer = load(str(bundle))
    if not native_dtype:
        model.set_dtype(mx.float32)
    eos_ids = sorted(int(token) for token in tokenizer.eos_token_ids)

    for self_path in self_goldens:
        own = json.loads(self_path.read_text())
        generated, margins, terminated_by = [], [], "length"
        steps = generate_step(
            mx.array(own["prompt_ids"]),
            model,
            max_tokens=own["params"]["max_tokens"],
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        )
        for token, logprobs in steps:
            top2 = mx.sort(logprobs.reshape(-1))[-2:]
            margins.append(float(top2[1] - top2[0]))
            if int(token) in eos_ids:
                terminated_by = "eos"
                break
            generated.append(int(token))

        tag = self_path.name[len("self-greedy-") :]
        out_path = bundle / ".golden" / f"upstream-greedy-{tag}"
        out_path.write_text(
            json.dumps(
                {
                    "tier": "upstream",
                    "prompt": own["prompt"],
                    "prompt_ids": own["prompt_ids"],
                    "generated_ids": generated,
                    "eos_ids": eos_ids,
                    "terminated_by": terminated_by,
                    "params": own["params"],
                    "margins_top2": margins,
                    "provenance": {
                        "commit": "n/a",
                        "mlx_lm": mlx_lm.__version__,
                        "mlx_pip": mx.__version__,
                        "machine": platform.processor() or platform.machine(),
                        "macos": platform.mac_ver()[0],
                        "date": datetime.date.today().isoformat(),
                        "prefill_mode": "generate_step, "
                        + ("checkpoint dtype" if native_dtype else "float32"),
                    },
                },
                indent=2,
            )
        )
        ours = own["generated_ids"]
        diverges = next(
            (i for i, (up, mine) in enumerate(zip(generated, ours)) if up != mine), None
        )
        verdict = (
            "identical"
            if diverges is None and len(generated) == len(ours)
            else f"first divergence at {diverges}, upstream margin there "
            f"{margins[diverges]:.4f}"
            if diverges is not None
            else f"lengths differ ({len(generated)} vs {len(ours)})"
        )
        print(
            f"captured {out_path} (mlx {mx.__version__}, mlx-lm {mlx_lm.__version__}; "
            f"self golden on MLX {own['provenance'].get('mlx_runtime')}): {verdict}"
        )
    return len(self_goldens)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--native-dtype",
        action="store_true",
        help="keep the checkpoint's dtype instead of computing in float32",
    )
    args = parser.parse_args()

    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError as error:
        print(
            f"error: {error}. Install the reference stack first: "
            "pip install mlx==<pinned release> mlx-lm",
            file=sys.stderr,
        )
        return 1

    bundles = [
        (var, pathlib.Path(os.environ[var])) for var in BUNDLE_VARS if os.environ.get(var)
    ]
    if not bundles:
        print(
            "error: no bundle staged; set one of " + ", ".join(BUNDLE_VARS),
            file=sys.stderr,
        )
        return 1

    captured = 0
    seen = set()
    for _, bundle in bundles:
        if bundle in seen:
            continue
        seen.add(bundle)
        captured += capture(bundle, args.native_dtype)
    if captured == 0:
        print("error: nothing captured", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
