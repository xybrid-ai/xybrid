#!/usr/bin/env python3
"""Report HF/GGUF semantic and numerical differences without inventing tolerances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

from hf_reference import digest


def load(path: Path):
    return json.loads(path.read_text())


def f32(bits: str) -> float:
    return struct.unpack(">f", bytes.fromhex(bits[2:]))[0]


def compare(hf_dir: Path, native_dir: Path, count: int, only: set[str] | None = None) -> dict:
    hf_receipt = load(hf_dir / "collection.json")
    native_receipt = load(native_dir / "collection.json")
    hf_count = hf_receipt["counts"]["selected"]
    native_count = native_receipt["corpus"]["selected"]
    if hf_receipt["counts"]["executed"] != hf_count or len(hf_receipt["cases"]) != hf_count:
        raise ValueError("HF collection did not execute the expected case count")
    if only is None and (hf_count != count or native_count != count):
        raise ValueError("reference collections do not have the expected case count")
    if only is not None and len(only) != count:
        raise ValueError("explicit comparison selection differs from the expected case count")
    if native_receipt["status"] != "complete":
        raise ValueError("upstream collection is incomplete or has the wrong case count")
    if not native_receipt["collector"]["official"]:
        raise ValueError("upstream observations are not official")
    if hf_receipt["settings"]["set_size"] != 1 or hf_receipt["settings"]["batch_rows"] != 1:
        raise ValueError("HF collection did not evaluate independent whole prompts")
    native_cases = {case["id"]: case for case in native_receipt["cases"] if case.get("status") == "observed"}
    hf_ids = {case["id"] for case in hf_receipt["cases"]}
    if len(native_cases) != native_count or len(hf_ids) != hf_count:
        raise ValueError("reference collection has omitted or duplicate cases")
    if only is not None and (not only.issubset(hf_ids) or not only.issubset(native_cases)):
        raise ValueError("explicit comparison selection is absent from a reference")
    reports = []
    worst_logit = 0.0
    worst_probability = 0.0
    for entry in hf_receipt["cases"]:
        if only is not None and entry["id"] not in only:
            continue
        path = hf_dir / (entry["id"] + ".json")
        if digest(path) != entry["sha256"]:
            raise ValueError("HF observation changed after collection")
        hf = load(path)
        if entry["id"] not in native_cases:
            raise ValueError("native collection omitted an HF case")
        native = load(native_dir / native_cases[entry["id"]]["observation"])
        if native["request_sha256"] != hf["request_sha256"]:
            raise ValueError("reference requests differ")
        report = {"id": entry["id"], "split": hf["split"], "status": "matching_preparation",
                  "hf_sha256": digest(path), "native_sha256": digest(native_dir / native_cases[entry["id"]]["observation"]),
                  "questions": []}
        if hf["status"] != "complete":
            report.update(status="hf_refused", error_type=hf.get("error_type"), error=hf.get("error"))
            reports.append(report)
            continue
        upstream = native["upstream"]
        if len(hf["passes"]) != len(upstream["prepare"]["passes"]):
            raise ValueError("reference question counts differ")
        for index, passed in enumerate(hf["passes"]):
            if "log_probabilities" not in passed["selected"]:
                raise ValueError("HF observation lacks the separate raw-head and normalized readouts")
            prepared = upstream["prepare"]["passes"][index]
            name = passed["question"]
            if prepared["question"] != name:
                raise ValueError("reference question order differs")
            same_prompt = prepared["prompt"] == passed["prompt"]
            same_tokens = prepared["token_ids"] == passed["token_ids"]
            same_labels = prepared["select"] == [token for group in passed["selected"]["groups"] for token in group]
            question = {"question": name, "same_prompt": same_prompt, "same_tokens": same_tokens,
                        "same_selected_labels": same_labels}
            hf_answer = passed["result"]["answers"][name]
            native_answer = upstream["finalize"]["answers"][name]
            kind = native_answer["type"]
            if kind in ("choice", "score"):
                question["same_selected_answer"] = hf_answer[kind] == native_answer[kind]
                question["hf_selected_answer"] = hf_answer[kind]
                question["native_selected_answer"] = native_answer[kind]
            if same_prompt and same_tokens and same_labels:
                hf_logits = [f32(value) for value in passed["selected"]["raw_scores"]]
                native_logits = [f32(value) for value in upstream["evaluate"]["raw_scores"][index]]
                if len(hf_logits) != len(native_logits):
                    raise ValueError("selected-logit counts differ")
                question["max_observed_abs_logit"] = max(abs(a - b) for a, b in zip(hf_logits, native_logits))
                if kind == "noul":
                    difference = abs(hf_answer["noul"] - native_answer["noul"])
                else:
                    if list(hf_answer["probabilities"]) != list(native_answer["probabilities"]):
                        raise ValueError("distribution order differs")
                    difference = max(abs(value - native_answer["probabilities"][key]) for key, value in hf_answer["probabilities"].items())
                question["max_observed_abs_probability"] = difference
                worst_logit = max(worst_logit, question["max_observed_abs_logit"])
                worst_probability = max(worst_probability, difference)
            else:
                report["status"] = "preparation_difference"
            report["questions"].append(question)
        reports.append(report)
    if len(reports) != count:
        raise ValueError("reference reports do not cover the selected corpus")
    return {
        "schema_version": 1, "purpose": "observed reference differences; no promotion bounds",
        "promotion_qualified": False,
        "comparison_recipe_sha256": digest(Path(__file__)),
        "hf_collection_sha256": digest(hf_dir / "collection.json"),
        "native_collection_sha256": digest(native_dir / "collection.json"),
        "native_artifact": native_receipt["artifact"],
        "source_case_counts": {"hf": hf_count, "native": native_count},
        "selection": sorted(only) if only is not None else None,
        "counts": {"cases": len(reports), "hf_refused": sum(case["status"] == "hf_refused" for case in reports),
                   "preparation_differences": sum(case["status"] == "preparation_difference" for case in reports),
                   "changed_selected_answers": sum(question.get("same_selected_answer") is False for case in reports for question in case["questions"])},
        "matching_preparation_maxima": {"absolute_selected_logit": worst_logit, "absolute_probability": worst_probability},
        "metric_units": {"absolute_selected_logit": "unscaled output-head logit", "absolute_probability": "probability, dimensionless"},
        "cases": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--expect-cases", type=int, default=63)
    parser.add_argument("--only", action="append", help="explicit partial comparison; source receipts retain their original full counts")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.hf_dir, args.native_dir, args.expect_cases, set(args.only) if args.only else None)
        args.out.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    except (ValueError, OSError, KeyError) as error:
        print(f"compare_reference: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
