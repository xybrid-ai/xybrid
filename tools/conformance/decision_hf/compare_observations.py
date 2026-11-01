#!/usr/bin/env python3
"""Compare two sets of HF reference observations case by case (stdlib only).

    compare_observations.py A B [--json OUT]

For every case observed in both directories: whether the prompts and token ids are identical, how many selected
logits and log-probabilities are bit-identical, the largest absolute difference of logits, log-probabilities and
probabilities, and whether the answers agree (choice, median level, and yes/no within 0.5). Packed observations
(`--set-size default`) compare per question against lone ones. Exit status 0 always; the report is the result.

`--json` writes the rows with what they were computed from: this script's sha256 and the sha256 of every observation
file on each side, which provenance.py checks against the committed files and the runs' logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digests(directory: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(directory.glob("*.json"))}


def f32(bits: str) -> float:
    return struct.unpack(">f", bytes.fromhex(bits[2:]))[0]


def number(value: Any) -> float:
    return float(value) if not isinstance(value, str) else float(value.replace("Infinity", "inf"))


def per_question(observation: dict) -> dict[str, dict[str, Any]]:
    """Each question's prompt, ids, selected logits and log-probabilities, distribution and answer, packed or not."""
    hf, detail = observation.get("hf", {}), observation.get("detail", {})
    finalize = hf.get("finalize")
    if not finalize or "answers" not in finalize:
        return {}
    out: dict[str, dict[str, Any]] = {}
    if "packed" in detail:
        for entry in detail["packed"]["evaluate"]:
            out[entry["question"]] = {
                "logits": entry["raw_scores"],
                "logprobs": entry["raw_logprobs"],
                "distribution": entry["distribution"],
                "answer": finalize["answers"][entry["question"]],
                "token_ids": None,
                "prompt": None,
            }
        return out
    passes = hf["prepare"]["passes"]
    extras = {e["question"]: e for e in detail.get("evaluate", [])}
    for i, passed in enumerate(passes):
        name = passed["question"]
        extra = extras.get(name, {})
        out[name] = {
            "logits": hf["evaluate"]["raw_scores"][i],
            "logprobs": extra.get("raw_logprobs"),
            "distribution": extra.get("distribution"),
            "answer": finalize["answers"][name],
            "token_ids": passed["token_ids"],
            "prompt": passed["prompt"],
        }
    return out


def same_answer(a: dict, b: dict) -> bool:
    if a.get("type") != b.get("type"):
        return False
    if a["type"] == "noul":
        return (number(a["noul"]) >= 0.5) == (number(b["noul"]) >= 0.5)
    if a["type"] == "choice" and "choice" in a:
        return a["choice"] == b["choice"]
    if a["type"] == "score":
        return number(a["score"]) == number(b["score"])
    return a == b


def compare(a_dir: Path, b_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for a_path in sorted(a_dir.glob("*.json")):
        b_path = b_dir / a_path.name
        if not b_path.is_file():
            continue
        a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
        if "case" not in a or "case" not in b:
            continue
        qa, qb = per_question(a), per_question(b)
        for name in qa.keys() & qb.keys():
            x, y = qa[name], qb[name]
            row: dict[str, Any] = {"case": a["case"], "question": name}
            if x["token_ids"] is not None and y["token_ids"] is not None:
                row["same_tokens"] = x["token_ids"] == y["token_ids"] and x["prompt"] == y["prompt"]
            if x["logits"] and y["logits"]:
                row["bits_equal"] = sum(p == q for p, q in zip(x["logits"], y["logits"]))
                row["n_selected"] = len(x["logits"])
                row["max_abs_logit"] = max(abs(f32(p) - f32(q)) for p, q in zip(x["logits"], y["logits"]))
                if x["logprobs"] and y["logprobs"]:
                    row["max_abs_logprob"] = max(
                        abs(f32(p) - f32(q)) for p, q in zip(x["logprobs"], y["logprobs"])
                    )
            if x["distribution"] and y["distribution"]:
                row["max_abs_p"] = max(
                    abs(number(p) - number(q)) for p, q in zip(x["distribution"], y["distribution"])
                )
            row["same_answer"] = same_answer(x["answer"], y["answer"])
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--json", type=Path, help="write the rows here")
    args = parser.parse_args(argv)
    rows = compare(args.a, args.b)
    for row in rows:
        bits = f"{row['bits_equal']}/{row['n_selected']}" if "bits_equal" in row else "-"
        print(
            f"{row['case']:34s} {row['question']:12s} tokens={'same' if row.get('same_tokens', True) else 'DIFF':4s} "
            f"bits={bits:7s} dlogit={row.get('max_abs_logit', 0):.3g} dlogprob={row.get('max_abs_logprob', 0):.3g} "
            f"dp={row.get('max_abs_p', 0):.3g} answer={'same' if row['same_answer'] else 'CHANGED'}"
        )
    scored = [r for r in rows if "bits_equal" in r]
    if scored:
        print(
            f"{len(rows)} questions; bit-identical logits in {sum(r['bits_equal'] == r['n_selected'] for r in scored)}"
            f"/{len(scored)}; max |dlogit| {max(r['max_abs_logit'] for r in scored):.3g}; max |dp| "
            f"{max(r.get('max_abs_p', 0) for r in scored):.3g}; answers changed "
            f"{sum(not r['same_answer'] for r in rows)}"
        )
    if args.json:
        document = {
            "recipe": {"compare_observations.py": sha256_file(Path(__file__).resolve())},
            "a": {"path": str(args.a), "files": digests(args.a)},
            "b": {"path": str(args.b), "files": digests(args.b)},
            "rows": rows,
        }
        args.json.write_text(json.dumps(document, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
