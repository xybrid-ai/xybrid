#!/usr/bin/env python3
"""Write or check integration-tests/fixtures/decision/hf/provenance.json (stdlib only).

    provenance.py --update --runs DIR [--llama-build DIR]   # after regenerating the fixtures
    provenance.py --check               # every fixture file matches its digest, and the pins match

`--update` assembles what the observations were made from and under: the pinned snapshot (pins.json and every file
the reference imported), the environment (requirements.txt digest, library versions), the settings ids, the hardware,
timings and memory, the verification runs (float32 parity, thread count, repeatability) and a summary of each
cross-check file with the llama.cpp build behind it (`--llama-build`, its CMake build directory), from the run logs
run_reference.py wrote with `--run-log` (observations.run.json, packed.run.json, parity.run.json and threads.run.json
in DIR; the last two are optional) and the comparisons in DIR (threads.compare.json, repeat.compare.json,
bf16.compare.json, packed.compare.json, optional). Then it records the sha256 of every other file under the fixtures
folder. `--check` recomputes those digests and the pins' digest, and fails on any difference or unlisted file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
FIXTURES = REPO_ROOT / "integration-tests/fixtures/decision/hf"
PROVENANCE = FIXTURES / "provenance.json"
PINS = HERE / "pins.json"
REQUIREMENTS = HERE / "requirements.txt"

SETTINGS = {
    "hf-cpu-f32-t1": {
        "backend": "cpu (PyTorch, MKL)",
        "threads": 1,
        "precision": (
            "float32 forward over the checkpoint's bfloat16 weights: each decoder layer, the embeddings, the final "
            "norm and the LM head compute with exact float32 widenings of their own parameters (run_reference.py "
            "Float32Compute), bit-identical to a float32 load module by module (verification.float32_parity)"
        ),
        "set_size": 1,
        "batch": 1,
        "padding": False,
        "attention": "the checkpoint's d1 attention (set): full-attention layers read the whole lone prompt both "
        "ways; sliding-window (256) and short-convolution layers stay causal",
        "deterministic_algorithms": True,
        "mode": "eval, inference_mode",
    },
    "hf-cpu-f32-t1-packed": {
        "backend": "cpu (PyTorch, MKL)",
        "threads": 1,
        "precision": "as hf-cpu-f32-t1",
        "set_size": "the checkpoint's d1_set_size (16): every question of a request packed after one state",
        "batch": 1,
        "padding": False,
        "deterministic_algorithms": True,
        "mode": "eval, inference_mode",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_digests() -> dict[str, str]:
    return {
        str(path.relative_to(FIXTURES)): sha256_file(path)
        for path in sorted(FIXTURES.rglob("*"))
        if path.is_file() and path != PROVENANCE
    }


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_summary(log: dict | None) -> dict | None:
    if log is None:
        return None
    return {
        "settings": log["numerics"]["settings"],
        "model_load_seconds": log["model_load_seconds"],
        "peak_rss_gib": log["peak_rss_gib"],
        "peak_anon_gib_between_calls": log["peak_anon_gib_between_calls"],
        "seconds_total": round(sum(c["seconds"] for c in log["cases"]), 1),
        "cases": {c["case"]: {"seconds": c["seconds"], "input_tokens": c["input_tokens"]} for c in log["cases"]},
    }


def comparison(rows: list | None, what: str) -> dict | None:
    if rows is None:
        return None
    scored = [r for r in rows if "bits_equal" in r]
    return {
        "what": what,
        "questions": len(rows),
        "bit_identical": sum(r["bits_equal"] == r["n_selected"] for r in scored),
        "max_abs_logit": max((r["max_abs_logit"] for r in scored), default=0.0),
        "max_abs_probability": max((r.get("max_abs_p", 0.0) for r in scored), default=0.0),
        "answers_changed": sum(not r["same_answer"] for r in rows),
    }


def crosscheck_summary(path: Path) -> dict:
    """The shared-domain numbers of one cross-check file (hf-* cases without hf-diff-*)."""
    document = load(path)
    cases = [c for c in document["cases"] if not c["case"].startswith("hf-diff-")]
    questions = [q for c in cases for q in c.get("questions", {}).values()]
    return {
        "server": document["server"],
        "temperature": document["temperature"],
        "shared_cases": len(cases),
        "shared_questions": len(questions),
        "equal_token_counts": sum(bool(c.get("same_usage")) for c in cases),
        "answers_that_differ": sum(not q["same_answer"] for q in questions),
        "max_abs_probability": max((q["max_abs_p"] for q in questions), default=0.0),
        "max_abs_logit_gap": max((q["max_abs_gap"] or 0.0 for q in questions), default=0.0),
    }


def llama_build(directory: Path | None) -> dict | None:
    """What the cross-checks' llama-server was built with, read from its CMake build directory."""
    if directory is None or not (directory / "CMakeCache.txt").is_file():
        return None
    cache = {}
    for line in (directory / "CMakeCache.txt").read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    compiler = None
    for path in sorted(directory.glob("CMakeFiles/*/CMakeCXXCompiler.cmake")):
        text = path.read_text()
        ident = next((l.split('"')[1] for l in text.splitlines() if l.startswith("set(CMAKE_CXX_COMPILER_ID ")), "")
        version = next(
            (l.split('"')[1] for l in text.splitlines() if l.startswith("set(CMAKE_CXX_COMPILER_VERSION ")), ""
        )
        compiler = f"{ident} {version}".strip()
    return {
        "source": "vendor/llama-cpp",
        "build_type": cache.get("CMAKE_BUILD_TYPE"),
        "ggml_native": cache.get("GGML_NATIVE"),
        "llama_curl": cache.get("LLAMA_CURL"),
        "compiler": compiler,
        "target": "llama-server",
    }


def update(runs: Path, build: Path | None) -> None:
    observations = load(runs / "observations.run.json")
    if observations is None:
        raise SystemExit(f"{runs}/observations.run.json is missing")
    parity = load(runs / "parity.run.json")
    document = {
        "description": (
            "Provenance of the HF reference observations in this folder: LiquidAI's own d1 code and weights "
            "(contract artifact d1-2b-hf) run by tools/conformance/decision_hf/run_reference.py, one question per "
            "sequence. Written by tools/conformance/decision_hf/provenance.py --update; checked by --check."
        ),
        "reference": {
            "artifact": json.loads(PINS.read_text())["artifact"],
            "repo": observations["reference"]["repo"],
            "revision": observations["reference"]["revision"],
            "pins": "tools/conformance/decision_hf/pins.json",
            "pins_sha256": sha256_file(PINS),
            "code_imported": observations["code_imported"],
        },
        "environment": {
            "requirements": "tools/conformance/decision_hf/requirements.txt",
            "requirements_sha256": sha256_file(REQUIREMENTS),
            "versions": observations["environment"]["versions"],
            "cpu_capability": observations["environment"]["cpu_capability"],
            "blas": next(
                (line.strip() for line in observations["environment"]["torch_parallel"] if "Math Kernel" in line),
                None,
            ),
        },
        "settings": SETTINGS,
        "hardware": observations["hardware"],
        "runs": {
            "observations": run_summary(observations),
            "packed": run_summary(load(runs / "packed.run.json")),
        },
        "verification": {
            "float32_parity": None
            if parity is None
            else {
                "what": (
                    "every module call of every forward replayed on a float32 deep copy of the module, the "
                    "parameters a float32 load holds: bit-identical outputs, float32 inputs throughout"
                ),
                "cases": [c["case"] for c in parity["cases"]],
                **parity["float32_parity"],
            },
            "repeatability": comparison(
                load(runs / "repeat.compare.json"), "a second hf-cpu-f32-t1 run of the parity cases"
            ),
            "threads": comparison(load(runs / "threads.compare.json"), "the same cases at 4 threads instead of 1"),
            "bf16": comparison(
                load(runs / "bf16.compare.json"), "the shared-domain cases with the bfloat16 forward instead"
            ),
            "packed": comparison(
                load(runs / "packed.compare.json"), "packed at the checkpoint's set size instead of alone"
            ),
        },
        "crosscheck": {
            "what": (
                "the stock llama-server answering the same requests, compared with the observations: a black-box "
                "check of DIFFERENCES.md and a G1 feasibility measurement, not the upstream oracle"
            ),
            "build": llama_build(build),
            "artifacts": "integration-tests/fixtures/decision/artifacts.json (the coordinator's pins: d1-2b-bf16, "
            "d1-2b-f16, d1-2b-q4km, each checked against its sha256 before use)",
            "files": {
                path.name: crosscheck_summary(path) for path in sorted((FIXTURES / "crosscheck").glob("*.json"))
            },
        },
        "files": fixture_digests(),
    }
    PROVENANCE.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {PROVENANCE.relative_to(REPO_ROOT)}: {len(document['files'])} files")


def check() -> int:
    document = load(PROVENANCE)
    if document is None:
        print(f"{PROVENANCE} is missing", file=sys.stderr)
        return 1
    problems = []
    if document["reference"]["pins_sha256"] != sha256_file(PINS):
        problems.append("pins.json changed since the observations were made")
    if document["environment"]["requirements_sha256"] != sha256_file(REQUIREMENTS):
        problems.append("requirements.txt changed since the observations were made")
    actual = fixture_digests()
    for name in sorted(actual.keys() | document["files"].keys()):
        if name not in document["files"]:
            problems.append(f"{name}: not in provenance.json")
        elif name not in actual:
            problems.append(f"{name}: missing")
        elif actual[name] != document["files"][name]:
            problems.append(f"{name}: sha256 differs")
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"provenance ok: {len(actual)} files")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--update", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--runs", type=Path, help="with --update: the directory of run logs and comparisons")
    parser.add_argument("--llama-build", type=Path, help="with --update: the cross-checks' llama.cpp build directory")
    args = parser.parse_args(argv)
    if args.update:
        if args.runs is None:
            parser.error("--update needs --runs")
        update(args.runs, args.llama_build)
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
