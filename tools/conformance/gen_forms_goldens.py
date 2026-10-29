#!/usr/bin/env python3
"""Write the CUA-S1-FORMS reference goldens for the conformance cases.

Candidates are the case's choices followed by the spec's fixed choices. Per
case the golden records the G0 byte ids exactly as the scorer must build them
(raw UTF-8 bytes truncated at ``max_len``, plus the offset; padding and masks
are implied) and the pinned PyTorch implementation's logits through its own
collator, with the softmax scores, the best candidate (first on ties) and the
top-1 minus top-2 margin. Inputs must match their pins in provenance.json.
Usage: README.md, "Regenerate the committed fixtures".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from common import (
    NUMERICS,
    best_and_margin,
    deterministic_torch,
    dumps,
    sha256_file,
    softmax,
    verified_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "integration-tests/fixtures/choice"
REQUIREMENTS = Path(__file__).resolve().parent / "requirements-forms.txt"


def encode(text: str, field: dict[str, Any]) -> tuple[list[int], bool]:
    """Unpadded byte ids for one field, and whether the text was truncated."""
    if field["overflow"] != "Truncate":
        raise SystemExit(f"unsupported overflow policy {field['overflow']!r}")
    raw = text.encode("utf-8")
    kept = raw[: field["max_len"]]
    return [byte + field["offset"] for byte in kept], len(kept) < len(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cua-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=FIXTURES / "cases/forms.json")
    parser.add_argument("--spec", type=Path, default=FIXTURES / "specs/cua-s1-forms.json")
    parser.add_argument("--provenance", type=Path, default=FIXTURES / "provenance.json")
    parser.add_argument("--out", type=Path, default=FIXTURES / "goldens/forms.json")
    args = parser.parse_args(argv)

    sources = json.loads(args.provenance.read_text(encoding="utf-8"))["sources"]
    package = "libs/cua-s1/python/src/cua_s1"
    inputs = verified_inputs(
        sources,
        [
            ("cua-s1-forms", "cua-s1-forms.safetensors", args.checkpoint),
            ("cua-s1-forms", "cua-s1-forms.json", args.checkpoint.with_suffix(".json")),
            *[
                ("trycua-cua", f"{package}/{name}", args.cua_src / "cua_s1" / name)
                for name in ("__init__.py", "checkpoint.py", "model.py")
            ],
        ],
    )

    deterministic_torch(0)
    sys.path.insert(0, str(args.cua_src.resolve()))
    from cua_s1.model import ChoiceExample, load_checkpoint

    model, collator, _config = load_checkpoint(args.checkpoint, "cpu")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))

    goldens = []
    for case in cases["cases"]:
        candidates = [{**choice, "fixed": False} for choice in case["choices"]]
        candidates += [{**choice, "fixed": True} for choice in spec["fixed_choices"]]
        if not 2 <= len(candidates) <= spec["max_choices"]:
            raise SystemExit(f"{case['id']}: {len(candidates)} candidates is out of range")
        texts = [candidate["text"] for candidate in candidates]
        batch = collator([ChoiceExample(context=case["context"], options=tuple(texts), label=0)])
        with torch.no_grad():
            logits = model(batch).numpy()[0]
        if logits.shape != (len(candidates),) or not np.all(np.isfinite(logits)):
            raise SystemExit(f"{case['id']}: unexpected reference output {logits}")
        scores = softmax(logits)
        best, margin = best_and_margin(scores)
        context_ids, context_truncated = encode(case["context"], spec["context"])
        encoded = [encode(text, spec["choice"]) for text in texts]
        goldens.append(
            {
                "id": case["id"],
                "candidates": [{"id": c["id"], "fixed": c["fixed"]} for c in candidates],
                "encoded": {
                    "context_ids": context_ids,
                    "context_truncated": context_truncated,
                    "choice_ids": [ids for ids, _ in encoded],
                    "choice_truncated": [truncated for _, truncated in encoded],
                },
                # float32 logits written as the exact doubles they widen to.
                "logits": [float(value) for value in logits],
                "scores": [float(value) for value in scores],
                "best": best,
                "margin": margin,
            }
        )

    document = {
        "description": (
            "CUA-S1-FORMS reference outputs for cases/forms.json: the pinned PyTorch "
            "implementation through its own collator. Scores are the softmax of the "
            "logits over the candidates, computed in float64."
        ),
        "provenance": {
            "sources": {name: sources[name]["revision"] for name in ("cua-s1-forms", "trycua-cua")},
            "generator": "tools/conformance/gen_forms_goldens.py",
            "environment": "forms",
            "requirements_sha256": sha256_file(REQUIREMENTS),
            "cases_sha256": sha256_file(args.cases),
            "spec_sha256": sha256_file(args.spec),
            "inputs": inputs,
            "numerics": NUMERICS,
            "versions": {"torch": torch.__version__, "numpy": np.__version__},
        },
        "cases": goldens,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(document), encoding="utf-8")
    for golden in goldens:
        winner = golden["candidates"][golden["best"]]["id"]
        print(f"{golden['id']:24s} best={winner:22s} margin={golden['margin']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
