#!/usr/bin/env python3
"""Synthetic scores where the two references' score medians disagree (no model, no weights).

The reference's median (`api.py` `score_level`) renormalizes as it accumulates, `mass += p / sum(p)`; llama.cpp's
(`server-decision.cpp` `format_answer`, lines 1731-1740) accumulates `p` as it is. The reference also divides each
score by its float64 temperature before the max (`prompt.py` `readouts`), where llama.cpp divides the difference to
the max by the float32 it parses from the GGUF's string (`exp((s - max) / T)`). Starting from three-level float32
scores whose first two levels hold half the mass, the script walks the third score bit by bit and keeps:

- `end_to_end`: float32 scores where the reference's answer differs from what llama.cpp's rule gives on the same
  scores (the temperature and the operation order move a level's mass across one half);
- `renormalization`: the median rules alone. Float32 scores move the cumulative mass by about 1e-7 a step, far more
  than the 1e-16 where renormalizing matters, so the search walks float64 scores through the reference's own
  `readouts` (which accepts a float64 tensor) and keeps distributions where the reference's renormalized rule and
  the plain cumulative rule, on that same distribution, give different levels; `rule` adds two probability vectors
  built by hand that fall on either side.

The reference side runs the snapshot's own code (`prompt.readouts`, the runner's path, and `api.score_level`). The
llama.cpp side is computed here from the lines cited above, with T = strtof("1.337398"): read from the code, not
observed; the upstream oracle's answer-math cases confirm it. Writes one JSON document (f32 as bit patterns, f64 with
17 significant digits). Run it through run.sh: `run.sh synthetic_median.py --out FILE`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this folder's own run_reference.py

from run_reference import (  # noqa: E402
    DEFAULT_CACHE,
    FIXTURES,
    PINS,
    dumps,
    f32_hex,
    f64,
    loaded_code,
    verify_snapshot,
)

# llama.cpp's effective lfm3-d1 temperature: the GGUF's f32, printed by std::to_string (%f), read back by strtof.
GGUF_TEMPERATURE_TEXT = "1.337398"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def next_f32(value: float, steps: int) -> float:
    bits = struct.unpack("<i", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<i", bits + steps))[0]


def llamacpp_answer(scores: list[float], temperature: float) -> tuple[list[float], int]:
    """format_answer for lfm3-d1, score type, one variant: server-decision.cpp:1675-1740."""
    logits = [float(s) for s in scores]
    top = max(logits)
    p = [math.exp((x - top) / temperature) for x in logits]
    total = 0.0
    for v in p:
        total += v
    probs = [v / total / 1 for v in p]
    mass, level = 0.0, len(probs) - 1
    i = 0
    while i < len(probs) and mass < 0.5:
        mass += probs[i]
        level = i
        i += 1
    return probs, level


def plain_median(probs: list[float]) -> int:
    mass = 0.0
    for i, p in enumerate(probs):
        mass += p
        if mass >= 0.5:
            return i
    return len(probs) - 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get("XYBRID_DECISION_HF_CACHE", DEFAULT_CACHE)))
    parser.add_argument("--out", type=Path, default=FIXTURES / "synthetic/median.json")
    parser.add_argument("--seeds", type=int, default=400, help="second-level scores to start from")
    parser.add_argument("--span", type=int, default=64, help="float32 steps walked each side of the boundary")
    args = parser.parse_args(argv)

    pins = json.loads(PINS.read_text(encoding="utf-8"))
    snapshot = args.cache / "snapshots" / pins["revision"] / "d1-2B"
    os.environ.setdefault("HF_MODULES_CACHE", str(args.cache / "modules"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    verify_snapshot(snapshot, pins)

    import torch
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module("modeling_d1_lfm3_vl.D1Model", str(snapshot))
    package = model_class.__module__.rsplit(".", 1)[0]
    prompt, api, calibration = (importlib.import_module(f"{package}.{m}") for m in ("prompt", "api", "calibration"))
    code = loaded_code(".".join(package.split(".")[:2]), pins)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
    cal = calibration.Calibration.load(snapshot / "calibration.json")
    question = prompt.Score("synthetic", ["low", "medium", "high"])
    ids = [g[0] for g in prompt.readout_ids(tokenizer, question)]
    t_ref = cal.temperature(question)
    t_cpp = float(f32(float(GGUF_TEMPERATURE_TEXT)))

    def reference(scores: list[float], dtype) -> list[float]:
        logz = torch.full((1, len(tokenizer)), -1.0e4, dtype=dtype)
        logz[0, ids] = torch.tensor(scores, dtype=dtype)
        return prompt.readouts(tokenizer, [question], logz, [cal])[0]

    end_to_end, tried = [], 0
    for k in range(args.seeds):
        x = f32(-6.0 + 12.0 * k / args.seeds)
        # p0 + p1 = 1/2 when exp(0) + exp(x/T) = exp(y/T)
        y0 = f32(t_ref * math.log1p(math.exp(x / t_ref)))
        for step in range(-args.span, args.span + 1):
            scores = [0.0, x, next_f32(y0, step)]
            tried += 1
            probs = reference(scores, torch.float32)
            level = api.score_level(probs, "median")
            cpp_probs, cpp_level = llamacpp_answer(scores, t_cpp)
            if level != cpp_level and len(end_to_end) < 3:
                end_to_end.append(
                    {
                        "scores": [f32_hex(s) for s in scores],
                        "reference": {"distribution": [f64(p) for p in probs], "median": level},
                        "llamacpp_rule": {"distribution": [f64(p) for p in cpp_probs], "median": cpp_level},
                    }
                )
        if len(end_to_end) >= 3:
            break

    renormalization, tried64 = [], 0
    for k in range(args.seeds):
        x = -6.0 + 12.0 * k / args.seeds
        y0 = t_ref * math.log1p(math.exp(x / t_ref))
        for step in range(-args.span, args.span + 1):
            scores = [0.0, x, y0 + step * math.ulp(y0)]
            tried64 += 1
            probs = reference(scores, torch.float64)
            level, plain = api.score_level(probs, "median"), plain_median(probs)
            if level != plain and len(renormalization) < 3:
                renormalization.append(
                    {
                        "scores_f64": [f64(v) for v in scores],
                        "distribution": [f64(p) for p in probs],
                        "sum": f64(sum(probs)),
                        "reference_median": level,
                        "plain_cumulative_median": plain,
                    }
                )
        if len(renormalization) >= 3:
            break

    rule = []
    for probs in ([0.25, 0.25, 0.5 + 2.0**-52], [0.25 - 2.0**-55, 0.25 - 2.0**-55, 0.5 - 2.0**-54]):
        rule.append(
            {
                "distribution": [f64(p) for p in probs],
                "sum": f64(sum(probs)),
                "reference_median": api.score_level(probs, "median"),
                "plain_cumulative_median": plain_median(probs),
            }
        )

    document = {
        "description": (
            "Three-level score questions where the reference's median (api.py score_level: renormalized "
            "cumulative mass, float64 temperature from calibration.json) and llama.cpp's (server-decision.cpp "
            "format_answer: plain cumulative mass, temperature strtof of the GGUF's string) disagree. Reference "
            "columns come from the pinned snapshot's own prompt.readouts and api.score_level; llamacpp_rule columns "
            "are computed from server-decision.cpp:1675-1740 and are to be confirmed by the upstream oracle."
        ),
        "generator": "tools/conformance/decision_hf/synthetic_median.py",
        "reference": {"artifact": pins["artifact"], "revision": pins["revision"], "code": code},
        "temperature": {"reference": f64(t_ref), "llamacpp": f64(t_cpp), "llamacpp_text": GGUF_TEMPERATURE_TEXT},
        "levels": ["low", "medium", "high"],
        "end_to_end": {"float32_triples_tried": tried, "cases": end_to_end},
        "renormalization": {"float64_triples_tried": tried64, "cases": renormalization, "rule": rule},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(document), encoding="utf-8")
    print(
        f"end to end: {len(end_to_end)} of {tried} float32 triples; renormalization: {len(renormalization)} of "
        f"{tried64} float64 triples; rule: {[(r['reference_median'], r['plain_cumulative_median']) for r in rule]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
