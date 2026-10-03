#!/usr/bin/env python3
"""Write the Qwen3.5-0.8B label-scoring reference goldens.

Each case becomes chat messages (an optional system message and one user
message built from the case's prompt templates), rendered by the pinned
tokenizer's ``apply_chat_template`` with ``add_generation_prompt=True`` and no
other keyword arguments, then one float32 CPU forward pass. Per case the golden
records the G0 inputs (messages, rendered bytes and their sha256, token ids,
each label's single token id), the final-position label logits and the
full-row logsumexp, and the derived scores, ``label_mass`` (full-vocabulary
probability of the labels), best label (first on ties) and margin.

Refuses to run unless every snapshot file matches its pin in provenance.json,
``templates/qwen3.5.jinja`` equals the tokenizer's chat template, and the
text-only model loads with no missing, unexpected or mismatched weights (the
vision and multi-token-prediction weights are never used by a causal-LM
forward). Usage: README.md, "Regenerate the committed fixtures".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import tokenizers
import torch
import transformers
from common import (
    NUMERICS,
    best_and_margin,
    deterministic_torch,
    dumps,
    sha256_file,
    verified_inputs,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "integration-tests/fixtures/choice"
REQUIREMENTS = Path(__file__).resolve().parent / "requirements-labels.txt"
# Every snapshot file Transformers reads to build the tokenizer and the model.
SNAPSHOT_FILES = (
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors-00001-of-00001.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def messages_for(case: dict[str, Any], prompt: dict[str, str]) -> list[dict[str, str]]:
    lines = [
        prompt["choice_template"].format(label=label, text=choice["text"])
        for label, choice in zip(case["labels"], case["choices"], strict=True)
    ]
    user = prompt["user_template"].format(
        context=case["context"], choices=prompt["choice_separator"].join(lines)
    )
    system = [{"role": "system", "content": case["system"]}] if case.get("system") else []
    return [*system, {"role": "user", "content": user}]


def label_ids(tokenizer: Any, labels: list[str]) -> list[int]:
    ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise SystemExit(f"label {label!r} is {len(encoded)} tokens, not one")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise SystemExit(f"labels {labels} do not map to distinct tokens")
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True, help="pinned HF snapshot directory")
    parser.add_argument("--cases", type=Path, default=FIXTURES / "cases/labels.json")
    parser.add_argument("--template", type=Path, default=FIXTURES / "templates/qwen3.5.jinja")
    parser.add_argument("--provenance", type=Path, default=FIXTURES / "provenance.json")
    parser.add_argument("--out", type=Path, default=FIXTURES / "goldens/labels.json")
    args = parser.parse_args(argv)

    sources = json.loads(args.provenance.read_text(encoding="utf-8"))["sources"]
    inputs = verified_inputs(
        sources, [("qwen3.5-0.8b", name, args.snapshot / name) for name in SNAPSHOT_FILES]
    )

    deterministic_torch(0)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    template = args.template.read_text(encoding="utf-8")
    if tokenizer.chat_template != template:
        raise SystemExit(f"{args.template} differs from the snapshot's chat template")
    model, loading = AutoModelForCausalLM.from_pretrained(
        args.snapshot, dtype=torch.float32, output_loading_info=True
    )
    problems = {key: sorted(map(str, value)) for key, value in loading.items() if value}
    if problems:
        raise SystemExit(f"the text model did not load cleanly: {problems}")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise SystemExit("expected lm_head tied to the input embeddings")
    model.eval()
    vocab_size = model.config.vocab_size

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    goldens = []
    for case in cases["cases"]:
        messages = messages_for(case, cases["prompt"])
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"]
        if list(templated) != token_ids:
            raise SystemExit(f"{case['id']}: rendering then tokenizing disagrees with the template")
        ids = label_ids(tokenizer, case["labels"])
        with torch.no_grad():
            row = model(input_ids=torch.tensor([token_ids])).logits[0, -1]
        if (
            row.dtype != torch.float32
            or row.shape != (vocab_size,)
            or not bool(torch.isfinite(row).all())
        ):
            raise SystemExit(f"{case['id']}: unexpected final row {row.dtype} {tuple(row.shape)}")
        logsumexp = float(torch.logsumexp(row.double(), dim=-1))
        label_logits = row[ids]
        scores = torch.softmax(label_logits.double(), dim=-1).numpy()
        best, margin = best_and_margin(scores)
        goldens.append(
            {
                "id": case["id"],
                "messages": messages,
                "rendered": rendered,
                "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "token_ids": token_ids,
                "labels": case["labels"],
                "label_ids": ids,
                # float32 logits written as the exact doubles they widen to.
                "label_logits": [float(value) for value in label_logits],
                "logsumexp": logsumexp,
                "scores": [float(value) for value in scores],
                "label_mass": float(np.exp(label_logits.double().numpy() - logsumexp).sum()),
                "best": best,
                "margin": margin,
            }
        )

    document = {
        "description": (
            "Qwen3.5-0.8B label-token reference for cases/labels.json: Hugging Face "
            "apply_chat_template with no template keyword arguments, then one float32 "
            "CPU forward pass. logsumexp, scores and label_mass are computed in float64 "
            "from the float32 final-position logits."
        ),
        "provenance": {
            "sources": {"qwen3.5-0.8b": sources["qwen3.5-0.8b"]["revision"]},
            "generator": "tools/conformance/gen_label_goldens.py",
            "environment": "labels",
            "requirements_sha256": sha256_file(REQUIREMENTS),
            "cases_sha256": sha256_file(args.cases),
            "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "inputs": inputs,
            "loading": {
                "model_class": type(model).__name__,
                "missing_keys": 0,
                "unexpected_keys": 0,
                "mismatched_keys": 0,
                "tied_embeddings": True,
                "vocab_size": vocab_size,
            },
            "generation_prompt": "add_generation_prompt=True, no enable_thinking",
            "numerics": NUMERICS,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
                "jinja2": jinja2.__version__,
            },
        },
        "cases": goldens,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(document), encoding="utf-8")
    for golden in goldens:
        print(
            f"{golden['id']:18s} tokens={len(golden['token_ids']):4d} "
            f"best={golden['labels'][golden['best']]} margin={golden['margin']:.3g} "
            f"label_mass={golden['label_mass']:.3g}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
