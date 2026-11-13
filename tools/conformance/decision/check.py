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
    print("Verified 143 upstream fixtures, 63 independent-reference inputs and source/artifact identities.")
    print("No HF/GGUF or device numerical promotion threshold is implied by this input check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
