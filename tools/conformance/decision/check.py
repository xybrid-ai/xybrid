#!/usr/bin/env python3
"""Verify immutable decision evidence inputs without claiming release qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import hf_reference

ROOT = hf_reference.ROOT
FIXTURES = hf_reference.FIXTURES
PIN = "8642541b255bfb5dc39cd351a178be09013a55d6"


def check_recorded_references() -> None:
    observations = FIXTURES / "hf/observations"
    read = lambda name: json.loads((observations / name).read_text())
    hf = read("hf-f32-linux-x86_64/collection.json")
    if hf["counts"]["selected"] != 63 or hf["counts"]["executed"] != 63 or len(hf["cases"]) != 63:
        raise ValueError("recorded HF collection did not attempt all 63 cases")
    for field, path in {
        "assets_manifest_sha256": FIXTURES / "hf/assets.json",
        "corpus_manifest_sha256": FIXTURES / "hf/corpus.json",
        "collector_sha256": Path(__file__).parent / "hf_reference.py",
        "requirements_sha256": Path(__file__).parent / "requirements-hf.txt",
    }.items():
        if hf[field] != hf_reference.digest(path):
            raise ValueError(f"recorded HF reference source changed: {field}")
    q4 = read("q4km-linux-x86_64/collection.json")
    f16 = read("f16-smoke-linux-x86_64/collection.json")
    for receipt, count in ((q4, 63), (f16, 1)):
        if receipt["status"] != "complete" or receipt["corpus"]["selected"] != count or not receipt["collector"]["official"]:
            raise ValueError("recorded upstream collection coverage/identity changed")
    for filename, native, count in (("hf-q4km.json", "q4km-linux-x86_64", 63), ("hf-f16-smoke.json", "f16-smoke-linux-x86_64", 1)):
        report = read(filename)
        if report["promotion_qualified"] is not False or report["counts"]["cases"] != count:
            raise ValueError("reference comparison must retain its unqualified coverage")
        for field, path in {
            "comparison_recipe_sha256": Path(__file__).parent / "compare_reference.py",
            "hf_collection_sha256": observations / "hf-f32-linux-x86_64/collection.json",
            "native_collection_sha256": observations / native / "collection.json",
        }.items():
            if report[field] != hf_reference.digest(path):
                raise ValueError(f"recorded comparison source changed: {field}")
    stage = read("xybrid-linux-x86_64/native-stage.json")
    if stage["stager_sha256"] != hf_reference.digest(Path(__file__).parent / "stage_native.py"):
        raise ValueError("matched-engine staging recipe changed")
    if stage["oracle_identity"]["fingerprint"] != q4["build"]["fingerprint"]:
        raise ValueError("staged engine and reference build identities differ")
    g2 = read("xybrid-linux-x86_64/g2.json")
    if g2["status"] != "passed" or (g2["model_cases_executed"], g2["whole_prompt_passes_executed"]) != (63, 78):
        raise ValueError("recorded G2 coverage changed")
    if g2["reference_collection_sha256"] != hf_reference.digest(observations / "q4km-linux-x86_64/collection.json"):
        raise ValueError("recorded G2 reference identity changed")


def check(model: Path | None = None) -> None:
    manifest = json.loads((FIXTURES / "integrity.json").read_text())
    if manifest["llama_cpp_commit"] != PIN:
        raise ValueError("decision native pin differs from the frozen corpus")
    gitlink = subprocess.check_output(["git", "ls-files", "--stage", "vendor/llama-cpp"], cwd=ROOT, text=True).split()[1]
    if gitlink != PIN:
        raise ValueError("vendor/llama-cpp changed: regenerate and review decision evidence")
    actual = {path.name for path in (FIXTURES / "cases").glob("*.json")}
    pinned = {Path(name).name for name in manifest["files"] if name.startswith("cases/")}
    if actual != pinned or len(actual) != 143:
        raise ValueError("decision corpus differs from its 143-case manifest")
    observations = {path.relative_to(FIXTURES).as_posix() for path in (FIXTURES / "hf/observations").rglob("*") if path.is_file() and path.name != "README.md"}
    pinned_observations = {name for name in manifest["files"] if name.startswith("hf/observations/")}
    if observations != pinned_observations:
        raise ValueError("reviewed reference observations differ from their integrity manifest")
    for name, expected in manifest["files"].items():
        if hf_reference.digest(FIXTURES / name) != expected:
            raise ValueError(f"decision fixture digest mismatch: {name}")
    oracle = json.loads((FIXTURES / "oracle.json").read_text())
    if oracle["collector"]["official"] is not True or oracle["llama_cpp"]["commit"] != PIN:
        raise ValueError("committed upstream observations are not official at the native pin")
    for name, expected in oracle["oracle"]["sources"].items():
        if hf_reference.digest(Path(__file__).parent / name) != expected:
            raise ValueError(f"upstream oracle recipe changed: {name}")
    artifacts = json.loads((FIXTURES / "artifacts.json").read_text())["artifacts"]
    hf = json.loads((FIXTURES / "hf/assets.json").read_text())
    if (hf["repository"], hf["revision"]) != (artifacts["d1-2b-hf"]["repo"], artifacts["d1-2b-hf"]["revision"]):
        raise ValueError("independent HF source pin differs from the artifact pin")
    hf_reference.corpus(None)
    check_recorded_references()
    if model is not None:
        artifact = artifacts["d1-2b-q4km"]
        if model.stat().st_size != artifact["bytes"] or hf_reference.digest(model) != artifact["sha256"]:
            raise ValueError("Q4_K_M artifact identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    try:
        check(args.model)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f"decision evidence: {error}", file=sys.stderr)
        return 1
    print("Verified 143 upstream fixtures, 63 independent-reference inputs, recorded observations and source/artifact identities.")
    print("No HF/GGUF or device numerical promotion threshold is implied by this input check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
