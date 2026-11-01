#!/usr/bin/env python3
"""Write or check integration-tests/fixtures/decision/hf/provenance.json (stdlib only).

    provenance.py --update --runs DIR [--llama-build DIR]   # after regenerating the fixtures
    provenance.py --check

Every committed file here comes from a recipe: the sources that produced it, the pins and the lock it ran under, and
the requests it read. The producers record those identities when they run: run_reference.py's and
synthetic_median.py's `--run-log`, and the outputs of crosscheck_server.py and compare_observations.py --json.

`--update` carries the recorded identities into provenance.json, taking nothing from disk that a run should have
recorded, and refuses (exit 1, nothing written) when any of them no longer holds. It refuses if:

- the pins, the snapshot files they pin, the lock or the recipe sources differ from what a run recorded;
- a run's numerics are not its role's complete settings: the settings id, every value the id stands for (`SETTINGS`,
  `VERIFICATION_SETTINGS`: precision, threads, set size, deterministic algorithms, mode), and the parity gate where
  the role needs a proof;
- a run imported other package versions than the lock's, or another Python than run.sh's, or ran in another
  environment (versions, CPU capability, hardware) than the observation run;
- a committed observation is not the bytes its run wrote, or was made from a different request than its case's;
- an observation or packed run's float32 parity proof is missing, failed or verified nothing;
- a cross-check or comparison was computed from other observations or by another version of its script;
- the cases do not regenerate, validate or name known artifacts and settings (`make_cases.py --check`).

`--check` needs no run log. It recomputes the digests of every committed file, of the pins, the lock and every recipe
source, regenerates and validates the cases, re-binds each observation to its case's request and settings and each
cross-check to the observations, checks that the recorded parity proofs passed, and checks the published settings
against `SETTINGS`, every published run's numerics against its role's, and the published versions against the lock.

Run logs read from DIR:

- observations.run.json and packed.run.json: run_reference.py with --check-f32-parity;
- synthetic.run.json: synthetic_median.py;
- for the verification block, when present: repeat.run.json, threads.run.json and bf16.run.json, with
  repeat.compare.json, threads.compare.json, bf16.compare.json and packed.compare.json (compare_observations.py --json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True  # importing make_cases.py must leave no __pycache__ in this folder
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_cases  # noqa: E402

REPO_ROOT = HERE.parents[2]
FIXTURES = REPO_ROOT / "integration-tests/fixtures/decision/hf"
PROVENANCE = FIXTURES / "provenance.json"
PINS = HERE / "pins.json"
REQUIREMENTS = HERE / "requirements.txt"
WEIGHTS = "model.safetensors"

# The sources whose bytes decide each kind of committed file, besides the pins and the lock.
RECIPES = {
    "observations": ("run_reference.py", "run.sh"),
    "synthetic": ("synthetic_median.py", "run_reference.py", "run.sh"),
    "cases": ("make_cases.py",),
    "crosscheck": ("crosscheck_server.py",),
    "comparisons": ("compare_observations.py",),
    "provenance": ("provenance.py",),  # the writer of provenance.json and the checker: recorded at --update
}
# run_reference.py's run logs: the settings each must have run under, and whether its parity proof is required.
RUNS = {
    "observations": ("hf-cpu-f32-t1", True),
    "packed": ("hf-cpu-f32-t1-packed", True),
    "repeat": ("hf-cpu-f32-t1", False),
    "threads": ("hf-cpu-f32-t4", False),
    "bf16": ("hf-cpu-bf16-t1", False),
}
# The Python run.sh runs every recipe under (`uv run --python 3.12`, as lock.sh locks for it), and what a run log
# records under environment.versions (run_reference.py `versions`): Python and packages the lock pins.
PYTHON = "3.12"
VERSIONED = ("python", "torch", "transformers", "tokenizers", "numpy", "safetensors", "torchvision")
COMPARISONS = {
    "repeat": "a second hf-cpu-f32-t1 run of a subset of the cases",
    "threads": "a subset of the cases at 4 threads instead of 1",
    "bf16": "the pilot cases (hf-* without hf-diff-*) except hf-state-long, with the bfloat16 forward instead",
    "packed": "the multi-question cases packed at the checkpoint's set size instead of alone",
}

# The settings ids run_reference.py records. `numerics` is every value of a run log's `numerics` block but the id and
# the parity flag, as the runner writes it: check_numerics holds each run to its id's, so an id stands for all of
# them. The committed runs' ids are published in provenance.json with their meaning (and --check holds the published
# definitions to these); the verification runs' ids are VERIFICATION_SETTINGS.
SETTINGS = {
    "hf-cpu-f32-t1": {
        "numerics": {
            "precision": "f32",
            "threads": 1,
            "set_size": 1,
            "deterministic_algorithms": True,
            "mode": "eval, inference_mode",
        },
        "backend": "cpu (PyTorch, MKL)",
        "compute": (
            "float32 forward over the checkpoint's bfloat16 weights: each decoder layer, the embeddings, the final "
            "norm and the LM head compute with exact float32 widenings of their own parameters (run_reference.py "
            "Float32Compute), proven bit-identical to a float32 load during the run (verification.float32_parity)"
        ),
        "batch": 1,
        "padding": False,
        "attention": "the checkpoint's d1 attention (set): full-attention layers read the whole lone prompt both "
        "ways; sliding-window (256) and short-convolution layers stay causal",
    },
    "hf-cpu-f32-t1-packed": {
        "numerics": {
            "precision": "f32",
            "threads": 1,
            "set_size": 16,
            "deterministic_algorithms": True,
            "mode": "eval, inference_mode",
        },
        "backend": "cpu (PyTorch, MKL)",
        "compute": "as hf-cpu-f32-t1",
        "packing": "every question of a request after one state, up to the checkpoint's d1_set_size (16) a sequence",
        "batch": 1,
        "padding": False,
    },
}
VERIFICATION_SETTINGS = {
    "hf-cpu-f32-t4": {
        "precision": "f32",
        "threads": 4,
        "set_size": 1,
        "deterministic_algorithms": True,
        "mode": "eval, inference_mode",
    },
    "hf-cpu-bf16-t1": {
        "precision": "bf16",
        "threads": 1,
        "set_size": 1,
        "deterministic_algorithms": True,
        "mode": "eval, inference_mode",
    },
}
NUMERICS = {**{name: settings["numerics"] for name, settings in SETTINGS.items()}, **VERIFICATION_SETTINGS}


class ProvenanceError(Exception):
    """An identity that no longer holds: the fixtures must be regenerated, not re-blessed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recipe(names: tuple[str, ...], base: Path = HERE) -> dict[str, str]:
    return {name: sha256_file(base / name) for name in names}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceError(message)


def identical(a: Any, b: Any) -> bool:
    """The same JSON: unlike ==, true is not 1 and 1.0 is not 1."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions() -> dict[str, str]:
    """Every `name==version` requirements.txt pins, by normalized name."""
    lock = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", line)
        if match:
            lock[normalized(match[1])] = match[2]
    return lock


def check_numerics(role: str, numerics: Any) -> None:
    """A run's numerics are its role's complete settings: the id, every value the id stands for (`NUMERICS`), and the
    parity gate where the role needs a proof. The id alone proves nothing; this check is what makes it stand for
    those values."""
    settings, gate = RUNS[role]
    require(isinstance(numerics, dict), f"{role}: no numerics recorded; rerun it")
    expected = {"settings": settings, **NUMERICS[settings]}
    recorded = {key: value for key, value in numerics.items() if key != "float32_parity_gate"}
    require(identical(recorded, expected), f"{role}: ran under {recorded}, not {settings}'s {expected}; rerun it")
    ran = numerics.get("float32_parity_gate")
    require(isinstance(ran, bool), f"{role}: no float32_parity_gate flag; rerun it")
    require(ran or not gate, f"{role}: ran without --check-f32-parity, which its proof needs; rerun it")


def check_versions(role: str, versions: Any) -> None:
    """The versions a run recorded are the lock's, package by package, and its Python is run.sh's."""
    require(isinstance(versions, dict) and set(versions) == set(VERSIONED),
            f"{role}: recorded the versions of {sorted(versions or {})}, not of {sorted(VERSIONED)}; rerun it")
    lock = locked_versions()
    wrong = [f"{name} {version} (locked: {lock.get(normalized(name))})" for name, version in versions.items()
             if name != "python" and lock.get(normalized(name)) != version]
    if not (versions["python"] == PYTHON or str(versions["python"]).startswith(PYTHON + ".")):
        wrong.append(f"Python {versions['python']} (run.sh: {PYTHON})")
    require(not wrong, f"{role}: ran on {', '.join(wrong)}; rerun it through run.sh")


def check_environment(role: str, log: dict, observations: dict, model_run: bool = True) -> None:
    """A contributing run ran where the observation run did: the same versions and, for a run of the model, the same
    CPU capability and hardware (synthetic_median.py records its versions only)."""
    keys = ("versions", "cpu_capability") if model_run else ("versions",)
    differs = [key for key in keys if log.get("environment", {}).get(key) != observations["environment"][key]]
    if model_run and log.get("hardware") != observations["hardware"]:
        differs.append("hardware")
    require(not differs, f"{role}: ran with other {', '.join(differs)} than the observation run; rerun it")


def current_inputs() -> dict[str, Any]:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    return {
        "pins_sha256": sha256_file(PINS),
        "requirements_sha256": sha256_file(REQUIREMENTS),
        "snapshot": {name: pin["sha256"] for name, pin in pins["files"].items()},
    }


def check_inputs(name: str, recorded: dict, current: dict, sources: tuple[str, ...], without=(), base=HERE) -> None:
    """The pins, the snapshot files, the lock and the recipe sources a run recorded are the current ones."""
    problems = []
    if recorded.get("pins_sha256") != current["pins_sha256"]:
        problems.append("pins.json differs")
    if recorded.get("requirements_sha256") != current["requirements_sha256"]:
        problems.append("requirements.txt differs")
    expected = {n: digest for n, digest in current["snapshot"].items() if n not in without}
    if {f["file"]: f["sha256"] for f in recorded.get("snapshot_files", [])} != expected:
        problems.append("the snapshot files it verified are not the pinned ones")
    now = recipe(sources, base)
    changed = sorted(n for n in sources if recorded.get("recipe", {}).get(n) != now[n])
    if changed:
        problems.append(f"recipe sources differ: {', '.join(changed)}")
    require(not problems, f"{name}: {'; '.join(problems)}; rerun it")


def check_parity(name: str, parity: Any) -> None:
    """A passed proof: calls verified, all bit-identical with float32 inputs, every hooked module verified."""
    require(bool(parity), f"{name}: no float32 parity proof; rerun it with --check-f32-parity")
    passed = (
        parity.get("calls", 0) > 0
        and parity.get("bit_identical") == parity["calls"]
        and parity.get("non_float32_inputs") == 0
        and 0 < parity.get("modules_verified", 0) == parity.get("modules_hooked")
    )
    require(passed, f"{name}: the float32 parity proof failed or verified nothing: {parity}")


def case_requests(cases: Path) -> dict[str, str]:
    """Each case's request digest: what an observation of it must have been made from."""
    out = {}
    for path in sorted(cases.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        out[document["id"]] = sha256_text(document["request_text"])
    return out


def check_run(role: str, log: dict, current: dict, requests: dict[str, str]) -> None:
    check_inputs(role, log["inputs"], current, RECIPES["observations"])
    check_numerics(role, log.get("numerics"))
    check_versions(role, log.get("environment", {}).get("versions"))
    if log["numerics"]["float32_parity_gate"]:  # required for a proof role; checked wherever it ran
        check_parity(role, log.get("float32_parity"))
    for case, digest in log["inputs"]["requests"].items():
        require(case in requests, f"{role}: {case} is no longer a case")
        require(requests[case] == digest, f"{role}: {case}'s request changed since the run; rerun it")


def check_outputs(role: str, log: dict, directory: Path, cases: set[str] | None) -> None:
    """The committed files of a run are exactly the bytes it wrote (and, for `cases`, one per case)."""
    outputs = log["outputs"]
    if cases is not None:
        require(set(outputs) == cases, f"{role}: observed {sorted(set(outputs) ^ cases)} do not match the cases")
    files = {path.stem for path in directory.glob("*.json")}
    require(files == set(outputs), f"{role}: {directory.name}/ holds {sorted(files ^ set(outputs))} beyond the run")
    for case, digest in outputs.items():
        actual = sha256_file(directory / f"{case}.json")
        require(actual == digest, f"{role}: {directory.name}/{case}.json is not what the run wrote")


def check_comparison(role: str, comparison: dict, b_files: dict[str, str] | None) -> None:
    script = sha256_file(HERE / "compare_observations.py")
    require(comparison.get("recipe", {}).get("compare_observations.py") == script,
            f"{role}.compare.json: made by another compare_observations.py; recompute it")
    for name, digest in comparison["a"]["files"].items():
        path = FIXTURES / "observations" / name
        require(path.is_file() and sha256_file(path) == digest,
                f"{role}.compare.json: computed from other observations than the committed {name}")
    if b_files is not None:
        require(comparison["b"]["files"] == b_files, f"{role}.compare.json: its other side is not the run's output")


def check_crosscheck(path: Path) -> dict[str, Any]:
    document = load(path)
    script = sha256_file(HERE / "crosscheck_server.py")
    require(document.get("recipe", {}).get("crosscheck_server.py") == script,
            f"crosscheck/{path.name}: made by another crosscheck_server.py; rerun it")
    require(bool(document.get("artifact")) and bool(document.get("settings")), f"crosscheck/{path.name}: no ids")
    for case, digest in document["observations"].items():
        observed = FIXTURES / "observations" / f"{case}.json"
        require(observed.is_file() and sha256_file(observed) == digest,
                f"crosscheck/{path.name}: compared other observations than the committed {case}.json; rerun it")
    return document


def crosscheck_summary(document: dict) -> dict[str, Any]:
    """One cross-check's numbers on the pilot cases (hf-* without hf-diff-*): the historical subset these summaries
    were first made on, kept as it was, not P11's G1 allowlist (contract C16)."""
    cases = [c for c in document["cases"] if not c["case"].startswith("hf-diff-")]
    questions = [q for c in cases for q in c.get("questions", {}).values()]
    return {
        "server": document["server"],
        "artifact": document["artifact"],
        "settings": document["settings"],
        "temperature": document["temperature"],
        "recipe": document["recipe"],
        "cases": len(document["cases"]),
        "pilot": "the numbers below: hf-* without hf-diff-*, a historical selection by name, not P11's G1 allowlist "
        "(contract C16)",
        "pilot_cases": len(cases),
        "pilot_questions": len(questions),
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
        lines = path.read_text().splitlines()
        ident = next((l.split('"')[1] for l in lines if l.startswith("set(CMAKE_CXX_COMPILER_ID ")), "")
        version = next((l.split('"')[1] for l in lines if l.startswith("set(CMAKE_CXX_COMPILER_VERSION ")), "")
        compiler = f"{ident} {version}".strip()
    return {
        "source": "vendor/llama-cpp",
        "build_type": cache.get("CMAKE_BUILD_TYPE"),
        "ggml_native": cache.get("GGML_NATIVE"),
        "llama_curl": cache.get("LLAMA_CURL"),
        "compiler": compiler,
        "target": "llama-server",
    }


def run_summary(log: dict | None) -> dict | None:
    if log is None:
        return None
    return {
        "numerics": log["numerics"],
        "model_load_seconds": log["model_load_seconds"],
        "peak_rss_gib": log["peak_rss_gib"],
        "peak_anon_gib_between_calls": log["peak_anon_gib_between_calls"],
        "seconds_total": round(sum(c["seconds"] for c in log["cases"]), 1),
        "cases": {c["case"]: {"seconds": c["seconds"], "input_tokens": c["input_tokens"]} for c in log["cases"]},
    }


def comparison_summary(role: str, comparison: dict | None, log: dict | None) -> dict | None:
    if comparison is None:
        return None
    rows = comparison["rows"]
    scored = [r for r in rows if "bits_equal" in r]
    return {
        "what": COMPARISONS[role],
        "run": log["numerics"] if log is not None else None,
        "questions": len(rows),
        "bit_identical": sum(r["bits_equal"] == r["n_selected"] for r in scored),
        "max_abs_logit": max((r["max_abs_logit"] for r in scored), default=0.0),
        "max_abs_probability": max((r.get("max_abs_p", 0.0) for r in scored), default=0.0),
        "answers_changed": sum(not r["same_answer"] for r in rows),
        "recipe": comparison["recipe"],
    }


def fixture_digests() -> dict[str, str]:
    return {
        str(path.relative_to(FIXTURES)): sha256_file(path)
        for path in sorted(FIXTURES.rglob("*"))
        if path.is_file() and path != PROVENANCE
    }


def check_cases() -> None:
    problems = make_cases.check()
    more = f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""
    require(not problems, "cases: " + "; ".join(problems[:10]) + more)


def update(runs: Path, build: Path | None) -> None:
    current = current_inputs()
    requests = case_requests(FIXTURES / "cases")
    check_cases()
    logs: dict[str, dict] = {}
    for role in RUNS:
        log = load(runs / f"{role}.run.json")
        if log is None:
            require(role not in ("observations", "packed"), f"{runs}/{role}.run.json is missing")
            continue
        check_run(role, log, current, requests)
        logs[role] = log
    for role, log in logs.items():
        check_environment(role, log, logs["observations"])
    check_outputs("observations", logs["observations"], FIXTURES / "observations", set(requests))
    check_outputs("packed", logs["packed"], FIXTURES / "packed", None)

    synthetic = load(runs / "synthetic.run.json")
    require(synthetic is not None, f"{runs}/synthetic.run.json is missing")
    check_inputs("synthetic", synthetic["inputs"], current, RECIPES["synthetic"], without=(WEIGHTS,))
    check_versions("synthetic", synthetic.get("environment", {}).get("versions"))
    check_environment("synthetic", synthetic, logs["observations"], model_run=False)
    median = FIXTURES / "synthetic/median.json"
    require(synthetic["outputs"] == {"synthetic/median.json": sha256_file(median)},
            "synthetic: synthetic/median.json is not what the run wrote")

    comparisons: dict[str, dict] = {}
    for role in COMPARISONS:
        comparison = load(runs / f"{role}.compare.json")
        if comparison is None:
            continue
        if role == "packed":
            b_files = {f"{case}.json": digest for case, digest in logs["packed"]["outputs"].items()}
        else:
            require(role in logs, f"{role}.compare.json: its run log {role}.run.json is missing")
            b_files = {f"{case}.json": digest for case, digest in logs[role]["outputs"].items()}
        check_comparison(role, comparison, b_files)
        comparisons[role] = comparison

    crosschecks = {path.name: check_crosscheck(path) for path in sorted((FIXTURES / "crosscheck").glob("*.json"))}

    observations = logs["observations"]
    recorded = {
        **observations["inputs"]["recipe"],
        **synthetic["inputs"]["recipe"],
        **{k: v for c in comparisons.values() for k, v in c["recipe"].items()},
        **{k: v for c in crosschecks.values() for k, v in c["recipe"].items()},
        "make_cases.py": sha256_file(HERE / "make_cases.py"),  # also checked by regenerating the cases
        "provenance.py": sha256_file(HERE / "provenance.py"),  # the writer of this file
    }
    unrecorded = sorted({name for names in RECIPES.values() for name in names} - set(recorded))
    require(not unrecorded, f"no committed output records {unrecorded}: run the comparisons and cross-checks")
    document = {
        "description": (
            "Provenance of the HF reference observations in this folder: LiquidAI's own d1 code and weights "
            "(contract artifact d1-2b-hf) run by tools/conformance/decision_hf/run_reference.py, one question per "
            "sequence. Every identity below is the one its run recorded when it ran, checked equal to the current "
            "inputs. Written by tools/conformance/decision_hf/provenance.py --update; checked by --check."
        ),
        "reference": {
            "artifact": observations["reference"]["artifact"],
            "repo": observations["reference"]["repo"],
            "revision": observations["reference"]["revision"],
            "pins": "tools/conformance/decision_hf/pins.json",
            "pins_sha256": observations["inputs"]["pins_sha256"],
            "snapshot_files": observations["inputs"]["snapshot_files"],
            "code_imported": observations["code_imported"],
        },
        "environment": {
            "requirements": "tools/conformance/decision_hf/requirements.txt",
            "requirements_sha256": observations["inputs"]["requirements_sha256"],
            # every run below recorded these versions, each checked against the lock; the runs of the model, this
            # CPU capability and hardware too
            "shared_by": [*logs, "synthetic"],
            "versions": observations["environment"]["versions"],
            "cpu_capability": observations["environment"]["cpu_capability"],
            "blas": next(
                (line.strip() for line in observations["environment"]["torch_parallel"] if "Math Kernel" in line),
                None,
            ),
        },
        "recipe": dict(sorted(recorded.items())),
        "settings": SETTINGS,
        "hardware": observations["hardware"],
        "runs": {"observations": run_summary(observations), "packed": run_summary(logs["packed"])},
        "verification": {
            "float32_parity": {
                "what": (
                    "during the observation and packed runs, every module call of every forward replayed on a float32 "
                    "deep copy of the module (the parameters a float32 load holds): bit-identical outputs from "
                    "float32 inputs, every hooked module verified; any failure would have ended the run unwritten"
                ),
                "observations": observations["float32_parity"],
                "packed": logs["packed"]["float32_parity"],
            },
            **{role: comparison_summary(role, comparisons.get(role), logs.get(role)) for role in COMPARISONS},
        },
        "crosscheck": {
            "what": (
                "the stock llama-server answering the same requests, compared with the observations: a black-box "
                "check of DIFFERENCES.md and a G1 feasibility measurement, not the upstream oracle. The G1 "
                "counterpart is d1-2b-f16; the BF16 and Q4_K_M files are reports. Each summary counts the historical "
                "pilot cases, not P11's G1 allowlist (contract C16)"
            ),
            "build": llama_build(build),
            "artifacts": "integration-tests/fixtures/decision/artifacts.json (each GGUF checked against its sha256)",
            "files": {name: crosscheck_summary(document) for name, document in crosschecks.items()},
        },
        "files": fixture_digests(),
    }
    PROVENANCE.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {PROVENANCE.relative_to(REPO_ROOT)}: {len(document['files'])} files")


def check() -> list[str]:
    document = load(PROVENANCE)
    if document is None:
        return [f"{PROVENANCE} is missing"]
    problems = []
    current = current_inputs()
    if document["reference"]["pins_sha256"] != current["pins_sha256"]:
        problems.append("pins.json changed since the observations were made")
    recorded_snapshot = {f["file"]: f["sha256"] for f in document["reference"]["snapshot_files"]}
    if recorded_snapshot != current["snapshot"]:
        problems.append("the pinned snapshot files changed since the observations were made")
    if document["environment"]["requirements_sha256"] != current["requirements_sha256"]:
        problems.append("requirements.txt changed since the observations were made")
    expected_sources = sorted({name for names in RECIPES.values() for name in names})
    if sorted(document["recipe"]) != expected_sources:
        problems.append(f"the recorded recipe names {sorted(document['recipe'])}, not {expected_sources}")
    for name, digest in document["recipe"].items():
        if not (HERE / name).is_file() or sha256_file(HERE / name) != digest:
            problems.append(f"{name} changed since the fixtures were made: regenerate them (README.md)")
    actual = fixture_digests()
    for name in sorted(actual.keys() | document["files"].keys()):
        if name not in document["files"]:
            problems.append(f"{name}: not in provenance.json")
        elif name not in actual:
            problems.append(f"{name}: missing")
        elif actual[name] != document["files"][name]:
            problems.append(f"{name}: sha256 differs")
    for role in ("observations", "packed"):
        try:
            check_parity(role, document["verification"]["float32_parity"].get(role))
        except ProvenanceError as error:
            problems.append(str(error))
    requests = case_requests(FIXTURES / "cases")
    for directory in ("observations", "packed"):
        for path in sorted((FIXTURES / directory).glob("*.json")):
            observed = json.loads(path.read_text(encoding="utf-8"))
            if requests.get(observed["case"]) != observed["request_text_sha256"]:
                problems.append(f"{directory}/{path.name}: made from another request than its case's")
            if observed.get("settings") != RUNS[directory][0]:
                problems.append(f"{directory}/{path.name}: made under {observed.get('settings')}, not {RUNS[directory][0]}")
    problems.extend(check_published(document))
    for path in sorted((FIXTURES / "crosscheck").glob("*.json")):
        try:
            check_crosscheck(path)
        except ProvenanceError as error:
            problems.append(str(error))
    problems.extend(f"cases: {problem}" for problem in make_cases.check())
    return problems


def check_published(document: dict) -> list[str]:
    """The published settings are `SETTINGS`, and every published run's numerics and the published versions hold
    against them and the lock, as --update checked them in the run logs."""
    problems = []
    if not identical(document.get("settings"), SETTINGS):
        problems.append("the published settings are not provenance.py's SETTINGS")
    published = {role: (document.get("runs", {}).get(role) or {}).get("numerics") for role in ("observations", "packed")}
    for role in ("repeat", "threads", "bf16"):
        summary = document.get("verification", {}).get(role)
        if summary is not None:
            published[role] = summary.get("run")
    for role, numerics in published.items():
        try:
            check_numerics(role, numerics)
        except ProvenanceError as error:
            problems.append(f"published {error}")
    try:
        check_versions("the environment", document.get("environment", {}).get("versions"))
    except ProvenanceError as error:
        problems.append(f"published {error}")
    return problems


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
        try:
            update(args.runs, args.llama_build)
        except ProvenanceError as error:
            print(f"provenance: {error}; provenance.json not written", file=sys.stderr)
            return 1
        return 0
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"provenance ok: {len(fixture_digests())} files, {len(load(PROVENANCE)['recipe'])} recipe sources")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
