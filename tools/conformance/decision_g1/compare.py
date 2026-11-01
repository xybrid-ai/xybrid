#!/usr/bin/env python3
"""Gate G1: the d1-2b-f16 GGUF through llama.cpp against LiquidAI's own d1 forward.

G1 compares two independent evaluations of the same decision prompts:

- the upstream oracle (tools/conformance/decision): llama.cpp's own decision code on the `d1-2b-f16` GGUF under
  settings `cpu-t4-fa-off` and `cpu-t4-fa-on` (observations under integration-tests/fixtures/decision/g1/);
- the HF reference (tools/conformance/decision_hf): LiquidAI's pinned d1 code and weights, a float32 forward, one
  question per sequence (settings `hf-cpu-f32-t1`).

Nothing numeric is compared until both sides are shown to have evaluated the same thing. Envelopes are parsed with
duplicate keys refused; cases, HF observations and oracle observations are joined by case id and the sha256 recomputed
from the exact request bytes; each side's run identity (HF provenance, the oracle's identity record) is checked and
fingerprinted; each case's questions, answers, passes, readouts and final answers must form complete ordered
bijections; and for every question of the shared domain (g1/allowlist.json) the parsed question, prompt bytes, token
ids, output index, selected tokens and label groups must be identical on both sides. Excluded cases carry checked
expectations (the refusal stage and class, or the exact fields that differ), so the allowlist cannot go stale. Any
missing, extra or mismatched record fails the run.

A generated corpus (the held-out set) is also bound to its generation record, make_heldout.py's generation.json:
- the recipe and seed frozen with the bounds;
- these request bytes;
- a measuring tool and vocabulary recorded without local state: the tool is the build that collected every oracle
  observation, and the vocabulary is a pinned artifact's;
- an allowlist that is exactly its cases and questions;
- every pass of every oracle collection as long as measured, each targeted slot exactly its target.
Once bounds are frozen, the comparator and the recipe must be the ones they were frozen with. These checks compare no
number: they are `compare.py integrity`. Every subcommand that computes a number runs them first and refuses, comparing
nothing, unless they pass.

The metric is the anchored gap error (AGE), in logits. For a question with groups g, s_g is a group's maximum selected
raw logit on each side (upstream's reduction), w the reference winner (first maximum), and

    AGE = max_g |(s_g^candidate - s_w^candidate) - (s_g^reference - s_w^reference)|

The gate: AGE <= B_s for every question of the domain, B_s the frozen bound of setting s (g1/bounds.json). All groups
count. Raw selected-logit differences, probability differences and margins are reported, never bounded.

Discrete answers use upstream's arithmetic on both sides (format_answer: f64, exp((s - max) / T) in order, sequential
sums, first-maximum choice, median at the first cumulative >= 0.5) with upstream's effective temperature. A changed
answer is checked against a necessary condition only: with AGE <= B, a choice flip needs a reference margin <= B, and
a median change needs every separating cumulative inside its one-sided band (sigma(+-2B/T)). A flip that passes is
"consistent with the bound"; a numeric pass with an inconsistent flip is a failure (an arithmetic defect). HF's own
answers (its temperature, log-prob readout and median rule) against upstream's arithmetic on HF's scores are reported
apart and never gated. The oracle's own answers must be reproduced from its scores (a check of this tool).

Subcommands:

  compare.py integrity  every check that compares no number, over every corpus and side; prints identities and
                      failures only
  compare.py gate     every corpus present against g1/bounds.json; exit 1 on any failure
  compare.py freeze   g1/bounds.json from the representative set (refused once it, or held-out data, exists). It
                      records this comparator's sha256 and the held-out recipe's file, sha256 and seed, which later
                      runs must match
  compare.py revalidate  after a regeneration that changed an identity: every corpus must pass under the frozen
                      bounds, then the new identities are recorded in g1/bounds.json; nothing is refitted
  compare.py drift    G3's report: d1-2b-q4km against d1-2b-f16 per setting, and Flash Attention drift
  compare.py check    recompute the committed g1/report.json and g1/q4-drift.json; exit 1 on any difference

Reports go to tools/conformance/decision_g1/out/ unless --out says otherwise. Standard library only; Python 3.9+.
"""

from __future__ import annotations

import argparse
import decimal
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES = REPO / "integration-tests" / "fixtures" / "decision"
OUT = HERE / "out"

SCHEMA = "decision-g1/v1"
PROFILE = "lfm3-d1/text/v1"

F16 = "d1-2b-f16"
Q4 = "d1-2b-q4km"
HF_ARTIFACT = "d1-2b-hf"
HF_SETTINGS = "hf-cpu-f32-t1"
SETTINGS = ("cpu-t4-fa-off", "cpu-t4-fa-on")

# Predeclared in the acceptance plan (v2, item b) before any F16 oracle observation existed. They change only on
# Sami's decision, never to make an observed set pass.
CAP = decimal.Decimal("0.020")
HEADROOM = decimal.Decimal(3)
FLOOR = decimal.Decimal("0.005")
QUANTUM = decimal.Decimal("0.001")

# Each reference's own temperature. llama.cpp reads the GGUF's f32 through a 6-decimal string: strtof("1.337398").
# The HF reference computes with calibration.json's float64; its observations also record the nearest f32.
NATIVE_TEMPERATURE = ("1.337398", "0x3fab2fdc")
HF_TEMPERATURE = ("1.337398453966105", "0x3fab2fdf", 1.337398453966105)

# The certificate's outward allowance, a diagnostic one: it never touches AGE <= B. A cumulative is a sequential f64
# sum of p_i = e_i / S; with n <= 255 terms its relative error is about gamma_255 = 255u / (1 - 255u) = 2.83e-14
# (u = 2^-53), so 1e-12 is about 35 times that bound, and the band edges sigma(+-2B/T) add a few ulps. A logit margin
# s_w - s_g of two f32 values is exact in f64; only B's binary representation (<= u relative) needs room there.
CDF_ALLOWANCE = 1e-12
MARGIN_ALLOWANCE = 1e-12

# How far this tool's f64 arithmetic may sit from the oracle's own answers before its self-check fails. Exact on the
# reference platform (glibc's exp on both sides); the tolerance keeps the check meaningful on another libm.
SELF_CHECK_TOLERANCE = 1e-12

STAGES = ("load", "parse", "prepare", "evaluate", "finalize")
EXCLUSION_KINDS = ("prompt_differs", "upstream_refuses", "hf_refuses", "hf_no_groups")
DIFFERENCE_FIELDS = ("parsed question", "type", "pass index", "prompt bytes", "token ids", "output index",
                     "selected tokens", "label groups")


class Failure(Exception):
    """A check that failed; the message names the case, question and check."""


# ---------------------------------------------------------------------------------------------------------------------
# Values and files
# ---------------------------------------------------------------------------------------------------------------------


def f32(bits: Any) -> float:
    """An f32 written as its bit pattern ("0x3fab2fdc"), as a Python float (exact)."""
    if not (isinstance(bits, str) and len(bits) == 10 and bits.startswith("0x")):
        raise Failure(f"not an f32 bit pattern: {bits!r}")
    try:
        return struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0]
    except ValueError:
        raise Failure(f"not an f32 bit pattern: {bits!r}") from None


def f32_bits(x: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', x))[0]:08x}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_digest(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _no_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        duplicated = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"duplicate keys {duplicated}")
    return dict(pairs)


def load_json(path: Path) -> Any:
    """A JSON file, with duplicate keys refused anywhere in it (a request's own duplicates live inside its
    request_text string and are kept)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except (OSError, ValueError) as error:
        raise Failure(f"{rel(path)}: {error}") from None


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def rounded(x: Optional[float], digits: int = 6) -> Optional[float]:
    """A derived value for a committed report, at 6 significant digits, so that a report does not depend on the last
    bits of a platform's libm."""
    if x is None:
        return None
    if not math.isfinite(x):
        raise Failure(f"a non-finite value reached a report: {x!r}")
    return float(f"{x:.{digits}g}")


# ---------------------------------------------------------------------------------------------------------------------
# Upstream's answer arithmetic (server-decision.cpp format_answer: lfm3-d1, one variant, no media, no score bias)
# ---------------------------------------------------------------------------------------------------------------------


def upstream_probabilities(scores: Sequence[float], temperature: float) -> List[float]:
    """p_i = exp((s_i - max) / T) / sum / 1, in float64, in upstream's operation order."""
    logit_max = scores[0]
    for s in scores[1:]:
        if logit_max < s:
            logit_max = s
    p = []
    total = 0.0
    for s in scores:
        value = math.exp((s - logit_max) / temperature)
        p.append(value)
        total += value
    return [value / total / 1 for value in p]


def upstream_choice(probs: Sequence[float]) -> int:
    """std::max_element: the first maximum."""
    best = 0
    for i in range(1, len(probs)):
        if probs[best] < probs[i]:
            best = i
    return best


def upstream_median(probs: Sequence[float]) -> int:
    """The first level whose sequential cumulative probability reaches one half (the last level if none does)."""
    mass = 0.0
    level = len(probs) - 1
    i = 0
    while i < len(probs) and mass < 0.5:
        mass += probs[i]
        level = i
        i += 1
    return level


def cumulative(probs: Sequence[float]) -> List[float]:
    out = []
    mass = 0.0
    for value in probs:
        mass += value
        out.append(mass)
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def upstream_temperature() -> float:
    return f32(NATIVE_TEMPERATURE[1])


def certificate(kind: str, ref_scores: Sequence[float], cand_scores: Sequence[float], bound: float,
                temperature: float) -> Dict[str, Any]:
    """Whether a changed discrete answer is consistent with `bound` (a necessary condition when AGE <= bound)."""
    p_r = upstream_probabilities(ref_scores, temperature)
    p_c = upstream_probabilities(cand_scores, temperature)
    if kind == "choice":
        r_best, c_best = upstream_choice(p_r), upstream_choice(p_c)
        margin = ref_scores[r_best] - ref_scores[c_best]
        return {"reference_margin": margin, "consistent": margin <= bound + MARGIN_ALLOWANCE}
    c_r, c_c = cumulative(p_r), cumulative(p_c)
    low, high = sigmoid(-2 * bound / temperature), sigmoid(2 * bound / temperature)
    prefixes = []
    consistent = True
    for k in range(len(p_r) - 1):
        r_above, c_above = c_r[k] >= 0.5, c_c[k] >= 0.5
        if r_above == c_above:
            continue
        if r_above:  # the reference reached one half here, the candidate did not
            ok = c_r[k] < high + CDF_ALLOWANCE
        else:
            ok = c_r[k] >= low - CDF_ALLOWANCE
        prefixes.append({"level": k, "reference_cumulative": c_r[k], "direction": "down" if r_above else "up",
                         "consistent": ok})
        consistent = consistent and ok
    return {"band": [low, high], "separating_prefixes": prefixes, "consistent": consistent}


# ---------------------------------------------------------------------------------------------------------------------
# Cases and observations
# ---------------------------------------------------------------------------------------------------------------------


class Case:
    def __init__(self, path: Path):
        doc = load_json(path)
        self.path = path
        if doc.get("schema_version") != 1 or doc.get("profile") != PROFILE:
            raise Failure(f"{rel(path)}: not a schema-v1 {PROFILE} case")
        self.id = doc.get("id")
        if self.id != path.stem:
            raise Failure(f"{rel(path)}: id {self.id!r} is not the file name")
        text = doc.get("request_text")
        if not isinstance(text, str):
            raise Failure(f"{self.id}: no request_text")
        self.request_text = text
        self.sha = sha256_text(text)


# The upstream oracle's collection record (tools/conformance/decision/oracle.py --collect), next to its observations.
COLLECTION_FILE = "collection.json"
# The HF reference's run log (run_reference.py --run-log), next to held-out observations; the representative
# observations are bound by tools/conformance/decision_hf's provenance.json instead.
HF_RUN_LOG = "run.json"
# A generated corpus's record (make_heldout.py), next to its cases directory; and the two files the freeze binds.
GENERATION_FILE = "generation.json"
COMPARATOR = Path(__file__).resolve()
HELDOUT_RECIPE = HERE / "make_heldout.py"


class Observation:
    """One side's stages for one case: an oracle case document (`upstream`) or an HF observation (`hf`)."""

    def __init__(self, path: Path, side: str):
        doc = load_json(path)
        self.path = path
        self.side = side
        if doc.get("schema_version") != 1:
            raise Failure(f"{rel(path)}: schema_version must be 1")
        if side == "hf":
            self.case_id = doc.get("case")
            self.sha = doc.get("request_text_sha256")
            stages = doc.get("hf")
            self.detail = doc.get("detail") or {}
        else:
            # oracle.py --collect: <id>.<sha256 of the request bytes>.json
            self.case_id = doc.get("id")
            self.sha = doc.get("request_sha256")
            stages = doc.get("upstream")
            self.detail = {}
            self.receipt = {k: doc.get(k) for k in ("case_file_sha256", "inputs", "inputs_sha256", "collector",
                                                    "build_fingerprint", "recipe_sha256")}
            if path.name != f"{self.case_id}.{self.sha}.json" or doc.get("source_file") != f"{self.case_id}.json":
                raise Failure(f"{rel(path)}: not the collection file of case {self.case_id!r}")
        self.artifact = doc.get("artifact")
        self.settings = doc.get("settings")
        if side == "hf" and self.case_id != path.stem:
            raise Failure(f"{rel(path)}: case {self.case_id!r} is not the file name")
        if not isinstance(self.sha, str) or len(self.sha) != 64:
            raise Failure(f"{rel(path)}: no request digest")
        if not isinstance(stages, dict) or set(stages) - set(STAGES) - {"server"}:
            raise Failure(f"{rel(path)}: no {side} stages, or unknown stages")
        self.stages = stages

    def refusal(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        for stage in STAGES:
            value = self.stages.get(stage)
            if isinstance(value, dict) and "refusal" in value:
                return stage, value["refusal"]
        return None

    def evaluated(self) -> bool:
        ev = self.stages.get("evaluate")
        return isinstance(ev, dict) and ev.get("source") == "model"

    def question_ids(self) -> List[str]:
        parse = self.stages.get("parse")
        if not isinstance(parse, dict) or not isinstance(parse.get("questions"), list):
            return []
        return [q.get("id") for q in parse["questions"]]


def load_dir(directory: Path, side: str) -> Dict[str, Observation]:
    if not directory.is_dir():
        raise Failure(f"{rel(directory)}: missing")
    out: Dict[str, Observation] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in (COLLECTION_FILE, HF_RUN_LOG):
            continue
        obs = Observation(path, side)
        if obs.case_id in out:
            raise Failure(f"{rel(directory)}: two observations of case {obs.case_id}")
        out[obs.case_id] = obs
    return out


def load_cases(directory: Path) -> Dict[str, Case]:
    if not directory.is_dir():
        raise Failure(f"{rel(directory)}: missing")
    return {case.id: case for case in (Case(path) for path in sorted(directory.glob("*.json")))}


def check_structure(obs: Observation) -> List[str]:
    """Complete, ordered bijections inside one fully evaluated observation."""
    where = f"{obs.case_id} ({rel(obs.path)})"
    problems = []
    parse, prepare = obs.stages.get("parse"), obs.stages.get("prepare")
    evaluate, finalize = obs.stages.get("evaluate"), obs.stages.get("finalize")
    try:
        ids = [q["id"] for q in parse["questions"]]
        answers, passes = prepare["answers"], prepare["passes"]
        raw, scores = evaluate["raw_scores"], evaluate["scores"]
        final_answers, usage = finalize["answers"], finalize["usage"]
    except (KeyError, TypeError):
        return [f"{where}: an incomplete evaluation"]
    if len(set(ids)) != len(ids):
        problems.append(f"{where}: repeated question ids")
    if [a.get("question") for a in answers] != ids:
        problems.append(f"{where}: answer mappings are not one per question, in order")
    if [p.get("question") for p in passes] != ids:
        problems.append(f"{where}: passes are not one per question, in order")
    if [a.get("pass") for a in answers] != list(range(len(answers))):
        problems.append(f"{where}: answer j does not read pass j")
    if len(raw) != len(passes) or len(scores) != len(passes):
        problems.append(f"{where}: {len(raw)} readouts and {len(scores)} score rows for {len(passes)} passes")
    if list(final_answers) != ids:
        problems.append(f"{where}: final answers are not one per question, in order")
    tokens = sum(len(p.get("token_ids", [])) for p in passes)
    if usage != {"input_tokens": tokens, "output_tokens": 0}:
        problems.append(f"{where}: usage {usage} is not the {tokens} evaluated tokens")
    return problems


class QuestionView:
    """Everything one side says about one question, checked for shape and for its reference's temperature."""

    def __init__(self, obs: Observation, qid: str, temperature: Tuple):
        where = f"{obs.case_id}/{qid} ({rel(obs.path)})"
        parse, prepare, evaluate = obs.stages.get("parse"), obs.stages.get("prepare"), obs.stages.get("evaluate")
        if not (isinstance(parse, dict) and isinstance(parse.get("questions"), list)):
            raise Failure(f"{where}: no parsed questions")
        if not (isinstance(prepare, dict) and "passes" in prepare and "answers" in prepare):
            raise Failure(f"{where}: not prepared")
        matches = [q for q in parse["questions"] if q.get("id") == qid]
        if len(matches) != 1:
            raise Failure(f"{where}: {len(matches)} parsed questions with this id")
        self.parsed = (matches[0].get("id"), matches[0].get("type"), list(matches[0].get("keys", [])))
        index = [a.get("question") for a in prepare["answers"]].index(qid) \
            if qid in [a.get("question") for a in prepare["answers"]] else None
        if index is None:
            raise Failure(f"{where}: no answer mapping")
        answer = prepare["answers"][index]
        self.type = answer.get("type")
        self.pass_index = answer.get("pass")
        self.groups = [(g.get("key"), g.get("code"), tuple(g.get("range", ()))) for g in answer.get("groups", [])]
        self.temperature = (answer.get("temperature"), answer.get("temperature_f32"))
        if self.temperature != tuple(temperature[:2]):
            raise Failure(f"{where}: temperature {self.temperature}, not this reference's {tuple(temperature[:2])}")
        if len(temperature) > 2:
            detail = obs.detail.get("prepare")
            entry = detail[index] if isinstance(detail, list) and index < len(detail) else {}
            if entry.get("question") != qid or entry.get("temperature_f64") != temperature[2]:
                raise Failure(f"{where}: float64 temperature {entry.get('temperature_f64')!r}, not {temperature[2]!r}")
        passes = prepare["passes"]
        if not (isinstance(self.pass_index, int) and 0 <= self.pass_index < len(passes)):
            raise Failure(f"{where}: pass index out of range")
        p = passes[self.pass_index]
        self.prompt = p.get("prompt")
        self.token_ids = list(p.get("token_ids", []))
        self.output_index = p.get("output_index")
        self.select = list(p.get("select", []))
        if p.get("question") != qid:
            raise Failure(f"{where}: its pass belongs to another question")
        if not self.token_ids or self.output_index != len(self.token_ids) - 1:
            raise Failure(f"{where}: the output index is not the last token")
        position = 0
        for _, _, span in self.groups:
            if len(span) != 2 or span[0] != position or span[1] <= span[0]:
                raise Failure(f"{where}: group ranges do not partition the selected tokens")
            position = span[1]
        if not self.groups or position != len(self.select):
            raise Failure(f"{where}: group ranges do not partition the selected tokens")
        if self.parsed[1] != self.type or self.parsed[2] != [key for key, _, _ in self.groups]:
            raise Failure(f"{where}: the answer mapping does not match the parsed question")
        if not (isinstance(evaluate, dict) and evaluate.get("source") == "model"):
            raise Failure(f"{where}: no model evaluation")
        try:
            raw = evaluate["raw_scores"][self.pass_index]
            reduced = evaluate["scores"][self.pass_index]
        except (KeyError, IndexError, TypeError):
            raise Failure(f"{where}: no readout for its pass") from None
        if len(raw) != len(self.select) or len(reduced) != len(self.groups):
            raise Failure(f"{where}: {len(raw)} raw scores for {len(self.select)} selected tokens, "
                          f"{len(reduced)} group scores for {len(self.groups)} groups")
        self.raw = [f32(bits) for bits in raw]
        if not all(math.isfinite(x) for x in self.raw):
            raise Failure(f"{where}: a non-finite raw score")
        self.scores = [max(self.raw[a:b]) for _, _, (a, b) in self.groups]
        if [f32_bits(x) for x in self.scores] != list(reduced):
            raise Failure(f"{where}: its group scores are not the maxima of its raw scores")
        finalize = obs.stages.get("finalize")
        self.answer = None
        if isinstance(finalize, dict) and isinstance(finalize.get("answers"), dict):
            self.answer = finalize["answers"].get(qid)

    def differences(self, other: "QuestionView") -> List[str]:
        """The identity checks: the names of the parts that differ, in DIFFERENCE_FIELDS order."""
        pairs = {
            "parsed question": (self.parsed, other.parsed),
            "type": (self.type, other.type),
            "pass index": (self.pass_index, other.pass_index),
            "prompt bytes": (self.prompt, other.prompt),
            "token ids": (self.token_ids, other.token_ids),
            "output index": (self.output_index, other.output_index),
            "selected tokens": (self.select, other.select),
            "label groups": (self.groups, other.groups),
        }
        return [name for name in DIFFERENCE_FIELDS if pairs[name][0] != pairs[name][1]]


def loose_differences(a: Observation, b: Observation) -> Dict[str, List[str]]:
    """Per question, the fields that differ between two prepared observations, temperatures and readouts aside."""
    out: Dict[str, List[str]] = {}
    pa, pb = a.stages.get("prepare") or {}, b.stages.get("prepare") or {}
    ids_a, ids_b = a.question_ids(), b.question_ids()
    if ids_a != ids_b:
        return {"*": ["parsed question"]}
    for j, qid in enumerate(ids_a):
        try:
            qa = [q for q in a.stages["parse"]["questions"] if q["id"] == qid][0]
            qb = [q for q in b.stages["parse"]["questions"] if q["id"] == qid][0]
            aa, ab = pa["answers"][j], pb["answers"][j]
            sa, sb = pa["passes"][aa["pass"]], pb["passes"][ab["pass"]]
        except (KeyError, IndexError, TypeError):
            out[qid] = ["structure"]
            continue
        fields = {
            "parsed question": (qa, qb),
            "type": (aa.get("type"), ab.get("type")),
            "pass index": (aa.get("pass"), ab.get("pass")),
            "prompt bytes": (sa.get("prompt"), sb.get("prompt")),
            "token ids": (sa.get("token_ids"), sb.get("token_ids")),
            "output index": (sa.get("output_index"), sb.get("output_index")),
            "selected tokens": (sa.get("select"), sb.get("select")),
            "label groups": ([(g["key"], g["code"], g["range"]) for g in aa.get("groups", [])],
                             [(g["key"], g["code"], g["range"]) for g in ab.get("groups", [])]),
        }
        differ = [name for name in DIFFERENCE_FIELDS if fields[name][0] != fields[name][1]]
        if differ:
            out[qid] = differ
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Run identities
# ---------------------------------------------------------------------------------------------------------------------


def pinned_llama_cpp() -> Optional[str]:
    """The vendor/llama-cpp gitlink of the checked-out tree."""
    try:
        done = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD:vendor/llama-cpp"], capture_output=True,
                              text=True, check=True)
        return done.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


ORACLE_TOOL = REPO / "tools" / "conformance" / "decision"


def expected_inputs(case: "Case", artifact: str, pinned: Dict[str, Any]) -> Dict[str, Any]:
    """The canonical inputs record of a G1 case (oracle.py case_inputs): its request, no media, no metadata edits,
    no synthetic scores, the pinned artifact with no metadata variant."""
    return {"request_sha256": case.sha, "has_media": False, "metadata": None, "select_logits": None,
            "model": {"artifact": artifact, "sha256": pinned.get("sha256"), "variant_sha256": None}}


def native_identity(directory: Path, artifact: str, settings: str, artifacts: Dict[str, Any],
                    cases: Dict[str, "Case"], cases_dir: Path, observations: Dict[str, "Observation"]) \
        -> Tuple[List[str], Optional[str]]:
    """The oracle's collection receipt next to its observations, checked, and every observation bound to it; and the
    computation fingerprint: what determines the numbers (recipe, build, settings as they ran, devices, platform,
    artifact bytes), not where or when the receipt was written."""
    path = directory / COLLECTION_FILE
    if not path.is_file():
        return [f"{rel(directory)}: no {COLLECTION_FILE} (oracle.py --collect writes it)"], None
    record = load_json(path)
    where = rel(path)
    failures = []
    if record.get("schema_version") != 1 or record.get("status") != "complete":
        failures.append(f"{where}: not a complete schema-v1 collection ({record.get('status')!r})")
    collector = record.get("collector") or {}
    if collector.get("official") is not True or collector.get("problems") or collector.get("developer_run"):
        failures.append(f"{where}: not an official collection: {collector}")
    corpus = record.get("corpus") or {}
    recorded_dir = corpus.get("dir")
    if not recorded_dir or (REPO / recorded_dir).resolve() != cases_dir.resolve():
        failures.append(f"{where}: collected {recorded_dir!r}, not {rel(cases_dir)}")
    want_counts = {"files": len(cases), "selected": len(cases), "expected": len(cases), "only": None}
    got_counts = {k: corpus.get(k) for k in want_counts}
    if got_counts != want_counts:
        failures.append(f"{where}: corpus counts {got_counts}, not {want_counts}")
    recipe = record.get("recipe") or {}
    if not recipe or record.get("recipe_sha256") != canonical_digest(recipe):
        failures.append(f"{where}: no recipe, or a recipe digest that is not its recipe's")
    for name, digest in sorted(recipe.items()):
        current = ORACLE_TOOL / name
        if not current.is_file() or sha256_file(current) != digest:
            failures.append(f"{where}: made by another tools/conformance/decision/{name} than the tree's (the tool "
                            f"changed: collect again)")
    pinned = artifacts["artifacts"].get(artifact) or {}
    want_artifact = {"id": artifact, "file": pinned.get("file"), "bytes": pinned.get("bytes"),
                     "sha256": pinned.get("sha256")}
    if record.get("artifact") != want_artifact:
        failures.append(f"{where}: artifact {record.get('artifact')}, not {want_artifact}")
    build = record.get("build") or {}
    source = build.get("llama_cpp") or {}
    tree = pinned_llama_cpp()
    if source.get("commit") is None or (tree and source.get("commit") != tree) or source.get("dirty") is not False:
        failures.append(f"{where}: llama.cpp {source}, the tree pins {tree!r} (a llama.cpp update makes these "
                        f"observations stale: regenerate them with oracle.py --collect)")
    if not build.get("fingerprint"):
        failures.append(f"{where}: no build fingerprint")
    compiled = build.get("build") or {}
    if compiled.get("build_type") != "Release" or compiled.get("backends") != ["cpu"]:
        failures.append(f"{where}: a {compiled.get('build_type')!r} build with backends {compiled.get('backends')}")
    ran = record.get("settings") or {}
    expected = artifacts["settings"].get(settings) or {}
    if set(ran) != {settings}:
        failures.append(f"{where}: settings {sorted(ran)}, not {settings}")
    entry = ran.get(settings) or {}
    context = entry.get("context") or {}
    want_context = {
        "n_ctx": expected.get("n_ctx"), "n_batch": expected.get("n_batch"), "n_ubatch": expected.get("n_ubatch"),
        "n_seq_max": expected.get("n_seq_max"), "n_outputs_max": expected.get("n_outputs_max"),
        "n_threads": expected.get("threads"), "n_threads_batch": expected.get("threads"),
        "flash_attn": "on" if expected.get("flash_attn") else "off", "causal_attn": False, "memory": False,
        "n_gpu_layers": 0, "offload_devices": "none",
    }
    wrong = {k: (context.get(k), v) for k, v in want_context.items() if context.get(k) != v}
    if wrong:
        failures.append(f"{where}: {settings} ran with {wrong} (recorded, expected)")
    devices = entry.get("oracle_devices")
    if not devices or any(str(d.get("type")).lower() != "cpu" for d in devices):
        failures.append(f"{where}: the oracle's devices are {devices}, not the CPU only")
    if entry.get("model_devices"):
        failures.append(f"{where}: the model was offloaded to {entry.get('model_devices')}")
    entries = record.get("cases") or []
    by_id = {e.get("id"): e for e in entries}
    if len(by_id) != len(entries):
        failures.append(f"{where}: repeated or unnamed case entries")
    for case_id, case in sorted(cases.items()):
        e = by_id.get(case_id)
        if e is None:
            failures.append(f"{where}: case {case_id} was not collected")
            continue
        inputs = expected_inputs(case, artifact, pinned)
        if e.get("status") != "observed" or e.get("file") != case.path.name or \
                e.get("file_sha256") != sha256_file(case.path) or e.get("request_sha256") != case.sha or \
                e.get("inputs") != inputs or e.get("inputs_sha256") != canonical_digest(inputs) or \
                e.get("observation") != f"{case_id}.{case.sha}.json":
            failures.append(f"{where}: case {case_id}: {e.get('status')!r}, or not this case file, request and "
                            f"inputs {e.get('error', '')}")
        if e.get("server_agrees") is False:
            failures.append(f"{where}: the stock llama-server disagrees with the oracle on {case_id}")
        obs = observations.get(case_id)
        if obs is not None:
            receipt = obs.receipt
            if receipt.get("case_file_sha256") != e.get("file_sha256") or receipt.get("inputs") != inputs or \
                    receipt.get("inputs_sha256") != canonical_digest(inputs) or \
                    receipt.get("build_fingerprint") != build.get("fingerprint") or \
                    receipt.get("recipe_sha256") != record.get("recipe_sha256") or \
                    receipt.get("collector") != collector:
                failures.append(f"{rel(obs.path)}: not bound to its collection's case file, inputs, build, recipe "
                                f"and collector")
    for case_id in sorted(set(by_id) - set(cases)):
        failures.append(f"{where}: collected a case that is not in the corpus: {case_id}")
    fingerprint = canonical_digest({
        "recipe_sha256": record.get("recipe_sha256"),
        "build": {k: v for k, v in build.items() if k != "llama_cpp_dir"},
        "artifact": record.get("artifact"),
        "settings": {k: entry.get(k) for k in ("llama_server_args", "context", "oracle_devices", "model_devices")},
        "platform": record.get("platform"),
    })
    return failures, fingerprint


HF_TOOL = REPO / "tools" / "conformance" / "decision_hf"
HF_OBSERVATION_RECIPE = ("run_reference.py", "run.sh")


# The HF numerics every G1 observation must have been made with (the recorded tuple, not its settings label).
HF_NUMERICS = {"settings": HF_SETTINGS, "precision": "f32", "threads": 1, "set_size": 1,
               "deterministic_algorithms": True, "mode": "eval, inference_mode", "float32_parity_gate": True}


def hf_numerics_from_provenance(prov: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    """The numerics tuple a provenance.json records for one of its runs (the same tuple a run log records)."""
    numerics = ((prov.get("runs") or {}).get(run_name) or {}).get("numerics") or {}
    return {k: numerics.get(k) for k in HF_NUMERICS}


def hf_fingerprint(reference: Dict[str, Any], pins_sha256: Any, requirements_sha256: Any, recipe: Dict[str, Any],
                   code_imported: Any, environment: Dict[str, Any], numerics: Dict[str, Any]) -> str:
    """The HF reference's computation: snapshot pins, lock, observation recipe, the code it imported (module names
    without the per-load hash prefix), runtime versions, CPU capability and the full numerics tuple."""
    code = sorted({(str(m.get("module", "")).split(".")[-1], m.get("sha256")) for m in (code_imported or [])})
    return canonical_digest({
        "reference": {k: reference.get(k) for k in ("artifact", "repo", "revision")},
        "pins_sha256": pins_sha256,
        "requirements_sha256": requirements_sha256,
        "recipe": {name: recipe.get(name) for name in HF_OBSERVATION_RECIPE},
        "code_imported": code,
        "versions": environment.get("versions"),
        "cpu_capability": environment.get("cpu_capability"),
        "numerics": {k: numerics.get(k) for k in sorted(HF_NUMERICS)},
    })


def hf_identity(directory: Path, cases: Dict[str, "Case"], hf: Dict[str, Observation]) \
        -> Tuple[List[str], Optional[str]]:
    """The HF reference's identity for these observations; and its computation fingerprint.

    Representative observations (tools/conformance/decision_hf's own folder) are bound by its provenance.json:
    provenance.py --check must pass, and every observation file must be one it records. Held-out observations carry
    their run log (run_reference.py --run-log --check-f32-parity), checked against the current pins, lock and
    recipe, with every request and output digest bound."""
    failures: List[str] = []
    run_log = directory / HF_RUN_LOG
    provenance = directory.parent / "provenance.json"
    if run_log.is_file():
        log = load_json(run_log)
        where = rel(run_log)
        inputs = log.get("inputs") or {}
        numerics = log.get("numerics") or {}
        if inputs.get("pins_sha256") != sha256_file(HF_TOOL / "pins.json"):
            failures.append(f"{where}: other snapshot pins than tools/conformance/decision_hf/pins.json")
        if inputs.get("requirements_sha256") != sha256_file(HF_TOOL / "requirements.txt"):
            failures.append(f"{where}: another lock than tools/conformance/decision_hf/requirements.txt")
        recipe = inputs.get("recipe") or {}
        for name in HF_OBSERVATION_RECIPE:
            if recipe.get(name) != sha256_file(HF_TOOL / name):
                failures.append(f"{where}: another {name} than tools/conformance/decision_hf/{name}")
        tuple_ = {k: numerics.get(k) for k in HF_NUMERICS}
        if tuple_ != HF_NUMERICS:
            failures.append(f"{where}: numerics {tuple_}, not {HF_NUMERICS}")
        parity = log.get("float32_parity") or {}
        if not parity.get("calls") or parity.get("bit_identical") != parity.get("calls") or \
                parity.get("non_float32_inputs") != 0 or parity.get("modules_verified") != parity.get("modules_hooked"):
            failures.append(f"{where}: no passing float32 parity proof: {parity}")
        requests, outputs = inputs.get("requests") or {}, log.get("outputs") or {}
        for case_id, case in sorted(cases.items()):
            if requests.get(case_id) != case.sha:
                failures.append(f"{where}: {case_id} was run from other request bytes")
            obs = hf.get(case_id)
            if obs is not None and outputs.get(case_id) != sha256_file(obs.path):
                failures.append(f"{where}: {rel(obs.path)} is not the file the run wrote")
        if set(requests) != set(cases) or set(outputs) != set(cases):
            failures.append(f"{where}: the run's cases are not the corpus's")
        fingerprint = hf_fingerprint(log.get("reference") or {}, inputs.get("pins_sha256"),
                                     inputs.get("requirements_sha256"), recipe, log.get("code_imported"),
                                     log.get("environment") or {}, numerics)
        return failures, fingerprint
    if not provenance.is_file():
        return [f"{rel(directory)}: neither a run log ({HF_RUN_LOG}) nor a provenance record"], None
    where = rel(provenance)
    done = subprocess.run([sys.executable, "-I", str(HF_TOOL / "provenance.py"), "--check"], capture_output=True,
                          text=True)
    if done.returncode != 0:
        failures.append(f"{where}: provenance.py --check fails: {(done.stdout + done.stderr).strip()[-600:]}")
    prov = load_json(provenance)
    files = prov.get("files") or {}
    for case_id, obs in sorted(hf.items()):
        key = f"{obs.path.parent.name}/{obs.path.name}"
        if files.get(key) != sha256_file(obs.path):
            failures.append(f"{where}: {key} is not a file the provenance records")
    numerics = hf_numerics_from_provenance(prov, directory.name)
    if numerics != HF_NUMERICS:
        failures.append(f"{where}: {directory.name} ran with numerics {numerics}, not {HF_NUMERICS}")
    recipe_block = ((prov.get("settings") or {}).get(HF_SETTINGS) or {})
    declared = recipe_block.get("numerics") or {}
    if {k: declared.get(k) for k in declared} != {k: HF_NUMERICS[k] for k in declared} or not declared or \
            recipe_block.get("batch") != 1 or recipe_block.get("padding") is not False:
        failures.append(f"{where}: {HF_SETTINGS} is not declared as {HF_NUMERICS}, a batch of one without padding")
    parity = ((prov.get("verification") or {}).get("float32_parity") or {}).get(directory.name) or {}
    if not parity.get("calls") or parity.get("bit_identical") != parity.get("calls") or \
            parity.get("non_float32_inputs") != 0 or parity.get("modules_verified") != parity.get("modules_hooked"):
        failures.append(f"{where}: {directory.name} has no passing float32 parity proof: {parity}")
    reference = prov.get("reference") or {}
    environment = prov.get("environment") or {}
    fingerprint = hf_fingerprint(reference, reference.get("pins_sha256"), environment.get("requirements_sha256"),
                                 prov.get("recipe") or {}, reference.get("code_imported"), environment, numerics)
    return failures, fingerprint


# ---------------------------------------------------------------------------------------------------------------------
# Corpora and the allowlist
# ---------------------------------------------------------------------------------------------------------------------


class Corpus:
    """A set of case files, its HF observations, its oracle observation root and its allowlist entry."""

    def __init__(self, name: str, spec: Dict[str, Any], root: Path):
        self.name = name
        self.spec = spec
        self.cases_dir = root / spec["cases"]
        self.hf_dir = root / spec["hf"]
        self.oracle_dir = root / spec["oracle"]
        self.included: Dict[str, List[str]] = spec.get("included", {})
        self.excluded: Dict[str, Dict[str, Any]] = spec.get("excluded", {})
        self.counts: Dict[str, int] = spec.get("counts", {})

    def oracle_path(self, artifact: str, settings: str) -> Path:
        return self.oracle_dir / artifact / settings


def check_membership(corpus: Corpus, cases: Dict[str, Case]) -> List[str]:
    """Every case in the allowlist, every entry a case, declared counts equal, exclusions well formed."""
    failures = []
    inc, exc = set(corpus.included), set(corpus.excluded)
    for both in sorted(inc & exc):
        failures.append(f"{corpus.name}: {both} is both included and excluded")
    for missing in sorted(set(cases) - inc - exc):
        failures.append(f"{corpus.name}: {missing} is neither included nor excluded")
    for stale in sorted((inc | exc) - set(cases)):
        failures.append(f"{corpus.name}: {stale} is in the allowlist but has no case file")
    for case_id, questions in sorted(corpus.included.items()):
        if not questions or len(set(questions)) != len(questions):
            failures.append(f"{corpus.name}: {case_id} needs distinct included questions")
    for case_id, entry in sorted(corpus.excluded.items()):
        if entry.get("kind") not in EXCLUSION_KINDS or not entry.get("reason"):
            failures.append(f"{corpus.name}: {case_id} needs a known exclusion kind and a reason")
    observed = {
        "cases": len(cases),
        "included_cases": len(corpus.included),
        "included_questions": sum(len(q) for q in corpus.included.values()),
        "excluded_cases": len(corpus.excluded),
    }
    for key, value in observed.items():
        if corpus.counts.get(key) != value:
            failures.append(f"{corpus.name}: {key} is {value}, the allowlist declares {corpus.counts.get(key)}")
    for key in sorted(set(corpus.counts) - set(observed)):
        failures.append(f"{corpus.name}: unknown count {key}")
    return failures


def check_join(corpus: Corpus, cases: Dict[str, Case], side: Dict[str, Observation], what: str, artifact: str,
               settings: str) -> List[str]:
    failures = []
    for missing in sorted(set(cases) - set(side)):
        failures.append(f"{corpus.name}/{what}: no observation for {missing}")
    for extra in sorted(set(side) - set(cases)):
        failures.append(f"{corpus.name}/{what}: an observation without a case: {extra}")
    for case_id in sorted(set(cases) & set(side)):
        obs = side[case_id]
        if obs.sha != cases[case_id].sha:
            failures.append(f"{corpus.name}/{what}: {case_id} was made from other request bytes")
        if obs.artifact != artifact:
            failures.append(f"{corpus.name}/{what}: {case_id} names artifact {obs.artifact!r}, not {artifact!r}")
        ev = obs.stages.get("evaluate")
        ran = isinstance(ev, dict) and (ev.get("source") == "model" or "refusal" in ev)
        expected = settings if (ran or obs.side == "hf") else None
        if obs.settings != expected:
            failures.append(f"{corpus.name}/{what}: {case_id} names settings {obs.settings!r}, not {expected!r}")
    return failures


def verify_exclusion(entry: Dict[str, Any], oracle: Observation, hf: Observation) -> Optional[str]:
    """None when the declared expectation holds exactly; otherwise what is stale."""
    kind = entry["kind"]
    if kind in ("upstream_refuses", "hf_refuses"):
        obs = oracle if kind == "upstream_refuses" else hf
        refusal = obs.refusal()
        if refusal is None:
            return "it does not refuse"
        stage, record = refusal
        want = {"stage": entry.get("stage"), "exception": entry.get("exception")}
        got = {"stage": stage, "exception": record.get("exception")}
        if "status" in entry:
            want["status"], got["status"] = entry["status"], record.get("status")
        if "question" in entry:
            want["question"], got["question"] = entry["question"], record.get("question")
        if got != want:
            return f"it refuses with {got}, not {want}"
        if kind == "upstream_refuses" and oracle.evaluated():
            return "upstream evaluated the model"
        return None
    if kind == "hf_no_groups":
        typed = hf.detail.get("typed")
        if hf.stages or not isinstance(typed, dict) or entry.get("question") not in (typed.get("answers") or {}):
            return "HF did not answer it as a typed question without option groups"
        return None
    if kind == "prompt_differs":
        if oracle.refusal() is not None or hf.refusal() is not None:
            return "a side refuses: the prompts are not what differs"
        got = loose_differences(oracle, hf)
        if got != entry.get("questions"):
            return f"the differing fields are {got}, not {entry.get('questions')}"
        return None
    return f"unknown kind {kind!r}"


# ---------------------------------------------------------------------------------------------------------------------
# One comparison: a candidate side against a reference side, question by question
# ---------------------------------------------------------------------------------------------------------------------


class Run:
    def __init__(self):
        self.failures: List[str] = []
        self.inputs: Dict[str, str] = {}
        self.identities: Dict[str, Optional[str]] = {}

    def digest(self, path: Path):
        self.inputs[rel(path)] = sha256_file(path)


def compare_question(cand: QuestionView, ref: QuestionView, bound: Optional[float], temperature: float) \
        -> Dict[str, Any]:
    s_c, s_r = cand.scores, ref.scores
    n = len(s_r)
    w = upstream_choice(s_r)
    deltas = [(s_c[g] - s_c[w]) - (s_r[g] - s_r[w]) for g in range(n)]
    worst = max(range(n), key=lambda g: (abs(deltas[g]), -g))
    age = abs(deltas[worst])
    raw_diff = [c - r for c, r in zip(cand.raw, ref.raw)]
    ordered = sorted(raw_diff)
    mid = len(ordered) // 2
    shift = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    p_c = upstream_probabilities(s_c, temperature)
    p_r = upstream_probabilities(s_r, temperature)
    row: Dict[str, Any] = {
        "type": ref.type,
        "groups": n,
        "tokens": len(ref.token_ids),
        "age": age,
        "age_over_t": age / temperature,
        "age_group": ref.groups[worst][0] if n > 1 else None,
        "raw_max_abs": max(abs(d) for d in raw_diff),
        "raw_shift": shift,
        "raw_max_abs_unshifted": max(abs(d - shift) for d in raw_diff),
        "p_max_abs": max(abs(a - b) for a, b in zip(p_c, p_r)),
        "reference_winner": ref.groups[w][0],
        "one_group": n == 1,
        "flip": None,
    }
    if n > 1:
        row["reference_margin"] = s_r[w] - max(s_r[g] for g in range(n) if g != w)
    if ref.type == "choice":
        c_best, r_best = upstream_choice(p_c), upstream_choice(p_r)
        row["answer"] = {"candidate": cand.groups[c_best][0], "reference": ref.groups[r_best][0]}
        changed = c_best != r_best
    elif ref.type == "score":
        c_med, r_med = upstream_median(p_c), upstream_median(p_r)
        row["answer"] = {"candidate": c_med, "reference": r_med}
        changed = c_med != r_med
    else:
        t = [key for key, _, _ in ref.groups].index("true")
        row["answer"] = {"candidate": p_c[t], "reference": p_r[t]}
        row["p_yes_crosses_half"] = (p_c[t] >= 0.5) != (p_r[t] >= 0.5)
        changed = False
    if changed:
        row["flip"] = {"kind": "choice winner" if ref.type == "choice" else "score median"}
        if bound is not None:
            row["flip"].update(certificate(ref.type, s_r, s_c, bound, temperature))
    return row


def reproduce_answer(view: QuestionView, temperature: float) -> List[str]:
    """Upstream's arithmetic on a side's own scores against that side's own answer (the oracle's)."""
    answer = view.answer
    if not isinstance(answer, dict):
        return ["no final answer"]
    problems = []
    p = upstream_probabilities(view.scores, temperature)
    keys = [key for key, _, _ in view.groups]

    def close(value: Any, expected: float) -> bool:
        return isinstance(value, float) and abs(value - expected) <= SELF_CHECK_TOLERANCE

    def probabilities_ok() -> bool:
        probabilities = answer.get("probabilities")
        return isinstance(probabilities, dict) and list(probabilities) == keys and \
            all(close(probabilities[k], v) for k, v in zip(keys, p))

    if answer.get("type") != view.type:
        problems.append("type")
    if view.type == "choice":
        if answer.get("choice") != keys[upstream_choice(p)]:
            problems.append("choice")
        if not close(answer.get("confidence"), max(p)):
            problems.append("confidence")
        if not probabilities_ok():
            problems.append("probabilities")
    elif view.type == "score":
        if answer.get("score") != float(upstream_median(p)):
            problems.append("score")
        if not close(answer.get("confidence"), max(p)):
            problems.append("confidence")
        if not probabilities_ok():
            problems.append("probabilities")
    elif view.type == "noul":
        if not close(answer.get("noul"), p[keys.index("true")]):
            problems.append("noul")
    return problems


def hf_reference_math(view: QuestionView, temperature: float) -> Optional[Dict[str, Any]]:
    """HF's own discrete answer against upstream's arithmetic on HF's scores (reported, never gated)."""
    answer = view.answer
    if not isinstance(answer, dict):
        return {"problem": "no HF answer"}
    p = upstream_probabilities(view.scores, temperature)
    keys = [key for key, _, _ in view.groups]
    if view.type == "choice":
        mine, theirs = keys[upstream_choice(p)], answer.get("choice")
    elif view.type == "score":
        mine, theirs = float(upstream_median(p)), answer.get("score")
    else:
        mine, theirs = p[keys.index("true")], answer.get("noul")
        if isinstance(theirs, float) and abs(theirs - mine) < 1e-6:
            return None
    if mine == theirs:
        return None
    return {"upstream_math": mine, "hf": theirs}


def compare_corpus(corpus: Corpus, cand: Dict[str, Observation], ref: Dict[str, Observation],
                   cand_temperature: Tuple, ref_temperature: Tuple, bound: Optional[float], run: Run, label: str,
                   self_check: bool = False, reference_math: bool = False, metrics: bool = True) -> Dict[str, Any]:
    """Question by question, the checks and then the comparison. Without `metrics` only the checks run (structure,
    the identity of what both sides evaluated, the self-check): no gap, probability or answer of the two sides is
    compared."""
    T = upstream_temperature()
    questions, reference_math_rows = [], []
    checked = 0
    for case_id in sorted(corpus.included):
        if case_id not in cand or case_id not in ref:
            continue  # reported by the join
        c_obs, r_obs = cand[case_id], ref[case_id]
        problems = check_structure(c_obs) + check_structure(r_obs)
        if problems:
            run.failures.extend(f"{label}: {p}" for p in problems)
            continue
        if c_obs.question_ids() != r_obs.question_ids():
            run.failures.append(f"{label}: {case_id}: the two sides parse different questions")
            continue
        declared = corpus.included[case_id]
        if sorted(declared) != sorted(c_obs.question_ids()):
            run.failures.append(f"{label}: {case_id}: the allowlist does not name exactly its questions")
            continue
        for qid in c_obs.question_ids():
            try:
                cv = QuestionView(c_obs, qid, cand_temperature)
                rv = QuestionView(r_obs, qid, ref_temperature)
            except Failure as error:
                run.failures.append(f"{label}: {error}")
                continue
            differ = cv.differences(rv)
            if differ:
                run.failures.append(f"{label}: {case_id}/{qid}: the two sides evaluated different things: "
                                    + ", ".join(differ))
                continue
            checked += 1
            row = {"case": case_id, "question": qid, **compare_question(cv, rv, bound, T)} if metrics else None
            if self_check:
                for view, side in ((cv, "candidate"), (rv, "reference")):
                    if side == "reference" and reference_math:
                        continue
                    problems = reproduce_answer(view, T)
                    if problems:
                        run.failures.append(f"{label}: {case_id}/{qid}: upstream's arithmetic on the {side}'s "
                                            f"scores does not reproduce its answer: {', '.join(problems)}")
            if not metrics:
                continue
            if reference_math:
                diff = hf_reference_math(rv, T)
                if diff is not None:
                    reference_math_rows.append({"case": case_id, "question": qid, "type": rv.type, **diff})
            questions.append(row)
    expected = sum(len(q) for q in corpus.included.values())
    if checked != expected:
        run.failures.append(f"{label}: compared {checked} questions, the allowlist includes {expected}")
    return {"questions": questions, "reference_math": reference_math_rows}


def summarize(questions: List[Dict[str, Any]], bound: Optional[float]) -> Dict[str, Any]:
    if not questions:
        return {"questions": 0}
    multi = [q for q in questions if not q["one_group"]]
    worst = max(multi or questions, key=lambda q: q["age"])
    flips = [q for q in questions if q["flip"]]
    out: Dict[str, Any] = {
        "questions": len(questions),
        "cases": len({q["case"] for q in questions}),
        "one_group_questions": len(questions) - len(multi),
        "max_age": worst["age"],
        "max_age_at": f"{worst['case']}/{worst['question']}",
        "max_raw_abs": max(q["raw_max_abs"] for q in questions),
        "max_raw_abs_unshifted": max(q["raw_max_abs_unshifted"] for q in questions),
        "max_p_abs": max(q["p_max_abs"] for q in questions),
        "discrete_changes": len(flips),
    }
    if bound is not None:
        out["bound"] = bound
        out["over_bound"] = [f"{q['case']}/{q['question']}" for q in questions if q["age"] > bound]
        out["inconsistent_flips"] = [f"{q['case']}/{q['question']}" for q in flips
                                     if q["age"] <= bound and not q["flip"].get("consistent")]
    return out


def report_rows(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-question rows for a committed report, derived floats rounded."""
    rows = []
    for q in questions:
        row = dict(q)
        for key in ("age", "age_over_t", "raw_max_abs", "raw_shift", "raw_max_abs_unshifted", "p_max_abs",
                    "reference_margin"):
            if key in row:
                row[key] = rounded(row[key])
        if row["type"] == "noul":
            row["answer"] = {k: rounded(v) for k, v in row["answer"].items()}
        if row.get("flip"):
            flip = json.loads(json.dumps(row["flip"]))
            if "reference_margin" in flip:
                flip["reference_margin"] = rounded(flip["reference_margin"])
            if "band" in flip:
                flip["band"] = [rounded(v, 9) for v in flip["band"]]
            for prefix in flip.get("separating_prefixes", []):
                prefix["reference_cumulative"] = rounded(prefix["reference_cumulative"], 9)
            row["flip"] = flip
        rows.append(row)
    return rows


def round_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(summary)
    for key in ("max_age", "max_raw_abs", "max_raw_abs_unshifted", "max_p_abs"):
        if key in out:
            out[key] = rounded(out[key])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The gate, the freeze, the drift report
# ---------------------------------------------------------------------------------------------------------------------


class Context:
    """Where everything lives; tests point it at a temporary tree."""

    def __init__(self, fixtures: Path = FIXTURES, root: Path = REPO):
        self.fixtures = fixtures
        self.g1 = fixtures / "g1"
        self.root = root
        self.allowlist_paths = [self.g1 / "allowlist.json", self.g1 / "heldout" / "allowlist.json"]
        self.bounds_path = self.g1 / "bounds.json"
        self.artifacts_path = fixtures / "artifacts.json"

    def corpora(self) -> List[Corpus]:
        corpora = []
        for path in self.allowlist_paths:
            if not path.is_file():
                continue
            allow = load_json(path)
            if allow.get("schema_version") != 1:
                raise Failure(f"{rel(path)}: schema_version must be 1")
            for name, spec in allow["corpora"].items():
                corpora.append(Corpus(name, spec, self.root))
        names = [c.name for c in corpora]
        if "representative" not in names or len(set(names)) != len(names):
            raise Failure("the allowlists must define the representative corpus once, and no corpus twice")
        return corpora

    def artifacts(self) -> Dict[str, Any]:
        return load_json(self.artifacts_path)


def evaluate_g1(ctx: Context, bounds: Optional[Dict[str, float]], run: Run,
                only: Optional[Iterable[str]] = None, metrics: bool = True) -> Dict[str, Any]:
    artifacts = ctx.artifacts()
    for path in ctx.allowlist_paths + [ctx.artifacts_path]:
        if path.is_file():
            run.digest(path)
    results: Dict[str, Any] = {}
    for corpus in ctx.corpora():
        if only is not None and corpus.name not in only:
            continue
        cases = load_cases(corpus.cases_dir)
        run.failures.extend(check_membership(corpus, cases))
        hf = load_dir(corpus.hf_dir, "hf")
        run.failures.extend(check_join(corpus, cases, hf, "hf", HF_ARTIFACT, HF_SETTINGS))
        failures, fingerprint = hf_identity(corpus.hf_dir, cases, hf)
        run.failures.extend(f"{corpus.name}: {f}" for f in failures)
        run.identities[f"{corpus.name}/hf"] = fingerprint
        for path in sorted(corpus.cases_dir.glob("*.json")) + sorted(corpus.hf_dir.glob("*.json")):
            run.digest(path)
        generation = corpus.cases_dir.parent / GENERATION_FILE
        if corpus.name != "representative" and generation.is_file():
            run.digest(generation)
        per_setting = {}
        for settings in SETTINGS:
            directory = corpus.oracle_path(F16, settings)
            oracle = load_dir(directory, "oracle")
            run.failures.extend(check_join(corpus, cases, oracle, f"{F16}/{settings}", F16, settings))
            failures, fingerprint = native_identity(directory, F16, settings, artifacts, cases, corpus.cases_dir,
                                                    oracle)
            run.failures.extend(f"{corpus.name}: {f}" for f in failures)
            run.identities[f"{corpus.name}/{F16}/{settings}"] = fingerprint
            for path in sorted(directory.glob("*.json")):
                run.digest(path)
            for case_id, entry in sorted(corpus.excluded.items()):
                if case_id in oracle and case_id in hf:
                    stale = verify_exclusion(entry, oracle[case_id], hf[case_id])
                    if stale:
                        run.failures.append(f"{corpus.name} {settings}: {case_id} is excluded as {entry['kind']}, "
                                            f"but {stale}")
            bound = None if bounds is None else bounds[settings]
            label = f"{corpus.name} {F16}/{settings} vs HF"
            result = compare_corpus(corpus, oracle, hf, NATIVE_TEMPERATURE, HF_TEMPERATURE, bound, run, label,
                                    self_check=True, reference_math=True, metrics=metrics)
            if not metrics:
                continue
            summary = summarize(result["questions"], bound)
            if bound is not None:
                for q in summary["over_bound"]:
                    run.failures.append(f"{label}: {q}: AGE above the bound {bound}")
                for q in summary["inconsistent_flips"]:
                    run.failures.append(f"{label}: {q}: within the bound, but its changed answer is not consistent "
                                        f"with it (an arithmetic defect)")
            per_setting[settings] = {"summary": summary, "questions": result["questions"],
                                     "reference_math": result["reference_math"]}
        results[corpus.name] = per_setting
    return results


def frozen_identities(run: Run) -> Dict[str, Optional[str]]:
    return {key: value for key, value in run.identities.items() if key.startswith("representative/")}


def g1_identities(identities: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """The identities G1's numbers depend on: the HF reference's and d1-2b-f16's (not the drift report's Q4)."""
    return {key: value for key, value in identities.items() if f"/{Q4}/" not in key}


# ---------------------------------------------------------------------------------------------------------------------
# Integrity before numbers
# ---------------------------------------------------------------------------------------------------------------------


REFUSED = "the integrity checks failed; nothing was compared"


def recipe_seed(path: Path) -> int:
    """The held-out recipe's literal seed (`SEED = 0x...` in make_heldout.py), read without running the recipe."""
    match = re.search(r"(?m)^SEED = (0x[0-9A-Fa-f]+)$", path.read_text(encoding="utf-8"))
    if match is None:
        raise Failure(f"{rel(path)}: no literal SEED")
    return int(match.group(1), 16)


def frozen_with() -> Dict[str, Any]:
    """What the freeze binds every later run to: this comparator, and the held-out recipe with its seed."""
    return {
        "comparator": {"file": rel(COMPARATOR), "sha256": sha256_file(COMPARATOR)},
        "heldout_recipe": {"file": rel(HELDOUT_RECIPE), "sha256": sha256_file(HELDOUT_RECIPE),
                           "seed": recipe_seed(HELDOUT_RECIPE)},
    }


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def check_generation(corpus: Corpus, cases: Dict[str, Case], recipe: Optional[Dict[str, Any]],
                     artifacts: Dict[str, Any], run: Run) -> List[str]:
    """A generated corpus against its generation record (make_heldout.py's generation.json, next to its cases). The
    record must name the recipe and seed frozen with the bounds, and these request bytes. Its measuring tool and
    vocabulary must be recorded without local state: the tool is the build that collected every oracle observation,
    and the vocabulary is a pinned artifact's. The allowlist must be exactly its cases and questions. Every pass of
    every oracle collection must be as long as measured, and each targeted slot exactly its target."""
    path = corpus.cases_dir.parent / GENERATION_FILE
    where = rel(path)
    if not path.is_file():
        return [f"{where}: missing (make_heldout.py writes it next to the cases)"]
    run.digest(path)
    record = load_json(path)
    failures = []
    if record.get("schema_version") != 1:
        failures.append(f"{where}: schema_version must be 1")
    if recipe is None:
        failures.append(f"{where}: held-out data without a recipe frozen with the bounds (compare.py freeze first)")
    elif record.get("recipe") != {k: recipe.get(k) for k in ("file", "sha256", "seed")}:
        failures.append(f"{where}: made by recipe {record.get('recipe')}, not the one frozen with the bounds {recipe}")
    measured = record.get("measured_with") or {}
    tool = measured.get("tool")
    if not isinstance(tool, str) or not tool or "\\" in tool or Path(tool).is_absolute() or ".." in Path(tool).parts:
        failures.append(f"{where}: the measuring tool is recorded as {tool!r}, not a repository path or a file name")
    if not is_sha256(measured.get("tool_sha256")):
        failures.append(f"{where}: no sha256 of the measuring tool")
    identity = measured.get("tool_identity")
    if not isinstance(identity, dict) or not identity.get("fingerprint") or "llama_cpp_dir" in identity:
        failures.append(f"{where}: no stamped identity of the measuring tool without local state")
        identity = None
    pinned = {artifacts["artifacts"][a].get("file"): artifacts["artifacts"][a].get("sha256") for a in (F16, Q4)
              if a in artifacts["artifacts"]}
    vocab = measured.get("vocab")
    if vocab not in pinned or measured.get("vocab_sha256") != pinned[vocab]:
        failures.append(f"{where}: prompt lengths measured on {vocab!r} ({measured.get('vocab_sha256')}), not a "
                        f"pinned {F16} or {Q4} file")
    calls = measured.get("calls")
    if not isinstance(calls, int) or isinstance(calls, bool) or calls < 1:
        failures.append(f"{where}: no measurement count")
    entries = record.get("cases")
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return failures + [f"{where}: no case records"]
    if [e.get("id") for e in entries] != sorted(cases):
        failures.append(f"{where}: the generated cases are not the corpus's case files")
    if corpus.excluded or corpus.included != {e.get("id"): e.get("questions") for e in entries}:
        failures.append(f"{corpus.name}: the allowlist is not the generation's (every generated case and question "
                        f"included, none excluded)")
    lengths: Dict[str, List[int]] = {}
    for e in entries:
        case_id, passes, target = e.get("id"), e.get("pass_tokens"), e.get("target_tokens")
        case = cases.get(case_id)
        if case is not None and e.get("request_text_sha256") != case.sha:
            failures.append(f"{where}: {case_id}: generated other request bytes than its case file holds")
        questions = e.get("questions")
        if not (isinstance(passes, list) and isinstance(questions, list) and passes and len(passes) == len(questions)
                and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in passes)):
            failures.append(f"{where}: {case_id}: pass lengths {passes!r} for questions {questions!r}")
            continue
        lengths[case_id] = passes
        if target is None:
            if e.get("measured_tokens") is not None:
                failures.append(f"{where}: {case_id}: a measured length without a target")
        elif e.get("measured_tokens") != target or passes[0] != target:
            failures.append(f"{where}: {case_id}: measured {e.get('measured_tokens')} and {passes[0]} tokens, the "
                            f"target is {target}")
    for artifact in (F16, Q4):
        for settings in SETTINGS:
            directory = corpus.oracle_path(artifact, settings)
            if not (directory / COLLECTION_FILE).is_file():
                continue  # the collection checks report it
            label = f"{corpus.name} {artifact}/{settings}"
            build = load_json(directory / COLLECTION_FILE).get("build") or {}
            if identity is not None and build.get("fingerprint") != identity.get("fingerprint"):
                failures.append(f"{label}: collected by build {build.get('fingerprint')}, but the prompt lengths were "
                                f"measured by build {identity.get('fingerprint')}")
            side = load_dir(directory, "oracle")
            for case_id, want in sorted(lengths.items()):
                obs = side.get(case_id)
                if obs is None:
                    continue  # the join reports it
                prepare, finalize = obs.stages.get("prepare"), obs.stages.get("finalize")
                passes = prepare.get("passes") if isinstance(prepare, dict) else None
                got = [len(p.get("token_ids") or []) for p in passes] if isinstance(passes, list) else None
                usage = ((finalize.get("usage") or {}) if isinstance(finalize, dict) else {}).get("input_tokens")
                if got != want or usage != sum(want):
                    failures.append(f"{label}: {case_id}: passes of {got} tokens (usage {usage}), the generation "
                                    f"measured {want}")
    return failures


def check_integrity(ctx: Context) -> Run:
    """Every check that compares no number, over every corpus and side present. They cover:
    - the allowlists and the case files;
    - the HF run identity: its provenance, or a held-out run log with its float32 parity proof and numerics tuple;
    - each oracle collection, d1-2b-f16 and d1-2b-q4km under both settings;
    - the identity of what both sides evaluated, and the oracle's own answers;
    - a generated corpus's generation record;
    - once bounds are frozen, the comparator and recipe they were frozen with.
    Everything that computes a number runs this first and refuses unless it passes."""
    run = Run()
    bounds_doc: Optional[Dict[str, Any]] = None
    try:
        if ctx.bounds_path.is_file():
            bounds_doc = load_json(ctx.bounds_path)
            run.digest(ctx.bounds_path)
    except Failure as error:
        run.failures.append(str(error))
    for evaluate in (lambda: evaluate_g1(ctx, None, run, metrics=False),
                     lambda: evaluate_drift(ctx, run, metrics=False)):
        try:
            evaluate()
        except Failure as error:
            run.failures.append(str(error))
    recipe = None
    if bounds_doc is not None:
        where = rel(ctx.bounds_path)
        frozen = bounds_doc.get("frozen_with") or {}
        comparator, recipe = frozen.get("comparator") or {}, frozen.get("heldout_recipe") or {}
        if comparator.get("sha256") != sha256_file(COMPARATOR):
            run.failures.append(f"{where}: frozen with compare.py {comparator.get('sha256')}, this one is "
                                f"{sha256_file(COMPARATOR)} (the comparator that evaluates must be the frozen one)")
        if recipe.get("sha256") != sha256_file(HELDOUT_RECIPE):
            run.failures.append(f"{where}: frozen with make_heldout.py {recipe.get('sha256')}, the tree's is "
                                f"{sha256_file(HELDOUT_RECIPE)} (a change to the recipe is a new held-out set)")
    try:
        artifacts = ctx.artifacts()
        for corpus in ctx.corpora():
            if corpus.name != "representative":
                run.failures.extend(check_generation(corpus, load_cases(corpus.cases_dir), recipe, artifacts, run))
    except Failure as error:
        run.failures.append(str(error))
    run.failures = list(dict.fromkeys(run.failures))  # both evaluations check the d1-2b-f16 sides
    return run


def cmd_integrity(ctx: Context, out: Path) -> int:
    """The integrity checks alone, run before any number is read. It prints identities and failures, never a gap, a
    probability or an answer."""
    run = check_integrity(ctx)
    for key, value in sorted(run.identities.items()):
        print(f"identity {key}: {value}")
    if ctx.bounds_path.is_file():
        try:
            refused = identity_mismatches(load_json(ctx.bounds_path), g1_identities(run.identities))
            print("identities the frozen bounds accept:", "all of G1's" if not refused else
                  f"all of G1's but {', '.join(refused)} (the gate refuses those; see compare.py revalidate)")
        except Failure:
            pass  # already a failure
    for failure in run.failures:
        print(f"FAIL {failure}")
    print("integrity:", "passed" if not run.failures else "FAILED")
    return 1 if run.failures else 0


def cmd_gate(ctx: Context, out: Path) -> int:
    run = Run()
    try:
        bounds_doc = load_json(ctx.bounds_path)
        bounds = {s: float(bounds_doc["bounds"][s]) for s in SETTINGS}
    except (Failure, KeyError, TypeError, ValueError) as error:
        print(f"compare.py gate: no frozen bounds ({error}); freeze them first", file=sys.stderr)
        return 1
    # Nothing is compared unless every integrity check passes on computations the bounds hold for: those they were
    # frozen on, or revalidated on without a refit.
    checked = check_integrity(ctx)
    checked.identities = g1_identities(checked.identities)
    for key in identity_mismatches(bounds_doc, checked.identities):
        checked.failures.append(f"{key}: a computation identity other than the one the bounds were frozen on "
                                f"(regenerate and run compare.py revalidate; the bounds are never refitted)")
    if checked.failures:
        report = gate_report({}, bounds, checked)
        report["refused"] = REFUSED
        write_report(out / "report.json", report)
        print_summary(report)
        return 1
    run.digest(ctx.bounds_path)
    try:
        results = evaluate_g1(ctx, bounds, run)
    except Failure as error:
        run.failures.append(str(error))
        results = {}
    report = gate_report(results, bounds, run)
    write_report(out / "report.json", report)
    print_summary(report)
    return 1 if run.failures else 0


def gate_report(results: Dict[str, Any], bounds: Dict[str, float], run: Run) -> Dict[str, Any]:
    corpora = {}
    for name, per_setting in results.items():
        corpora[name] = {
            s: {
                "summary": round_summary(v["summary"]),
                "questions": report_rows(v["questions"]),
                "reference_math_differences": [
                    {k: (rounded(x) if isinstance(x, float) else x) for k, x in row.items()}
                    for row in v["reference_math"]
                ],
            }
            for s, v in per_setting.items()
        }
    return {
        "schema": SCHEMA,
        "what": "G1: d1-2b-f16 through the upstream oracle against the HF reference (hf-cpu-f32-t1), anchored gap "
                "error in logits",
        "metric": "AGE = max_g |(s_g^gguf - s_w^gguf) - (s_g^hf - s_w^hf)|, w the HF winner, s_g a group's maximum "
                  "selected raw logit",
        "temperature_f32": NATIVE_TEMPERATURE[1],
        "bounds": bounds,
        "identity": dict(sorted(run.identities.items())),
        "tool": {"compare.py": sha256_file(Path(__file__))},
        "inputs": dict(sorted(run.inputs.items())),
        "passed": not run.failures,
        "failures": run.failures,
        "corpora": corpora,
    }


def write_report(path: Path, report: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def print_summary(report: Dict[str, Any]):
    for name, per_setting in report.get("corpora", {}).items():
        for settings, entry in per_setting.items():
            s = entry["summary"]
            if not s.get("questions"):
                print(f"{name} {settings}: no questions compared")
                continue
            print(f"{name} {settings}: {s['questions']} questions ({s['one_group_questions']} one-group) in "
                  f"{s['cases']} cases, max AGE {s['max_age']} ({s['max_age_at']}), bound {s.get('bound')}, "
                  f"discrete changes {s['discrete_changes']}, max |raw| {s['max_raw_abs']}, max |dp| {s['max_p_abs']}")
            diffs = entry.get("reference_math_differences") or []
            if diffs:
                print(f"  HF's own answers vs upstream arithmetic on HF's scores: {len(diffs)} differences")
    for failure in report.get("failures", []):
        print(f"FAIL {failure}")
    print("G1 gate:", "passed" if report.get("passed") else
          f"refused: {report['refused']}" if report.get("refused") else "FAILED")


def clipped_bound(max_age: float) -> decimal.Decimal:
    """min(C, max(floor, ceil(k * M, quantum))): the clipped-headroom rule."""
    fitted = (decimal.Decimal(repr(max_age)) * HEADROOM).quantize(QUANTUM, rounding=decimal.ROUND_CEILING)
    return min(CAP, max(FLOOR, fitted))


def headroom_record(max_age: float, bound: decimal.Decimal) -> Dict[str, Any]:
    if max_age == 0.0:
        return {"ratio": None, "status": "zero_error"}
    return {"ratio": float(f"{float(bound) / max_age:.6g}"), "status": "measured"}


def cmd_freeze(ctx: Context, out: Path) -> int:
    if ctx.bounds_path.exists():
        print(f"compare.py freeze: {rel(ctx.bounds_path)} exists; bounds are frozen once", file=sys.stderr)
        return 1
    if (ctx.g1 / "heldout").exists():
        print("compare.py freeze: held-out data exists; the bounds must be frozen before it", file=sys.stderr)
        return 1
    checked = check_integrity(ctx)
    if checked.failures:
        for failure in checked.failures:
            print(f"FAIL {failure}")
        print(f"compare.py freeze: refused: {REFUSED}")
        return 1
    run = Run()
    try:
        results = evaluate_g1(ctx, None, run, only=["representative"])
    except Failure as error:
        run.failures.append(str(error))
        results = {}
    if run.failures:
        for failure in run.failures:
            print(f"FAIL {failure}")
        print("compare.py freeze: refused: the representative set fails its integrity checks")
        return 1
    bounds, observed, refusals = {}, {}, []
    for settings in SETTINGS:
        summary = summarize(results["representative"][settings]["questions"], None)
        max_age = summary["max_age"]
        bound = clipped_bound(max_age)
        if decimal.Decimal(repr(max_age)) > CAP / 2:
            refusals.append(f"{settings}: max AGE {max_age!r} is above half the cap ({CAP / 2})")
        bounds[settings] = float(bound)
        observed[settings] = {
            "questions": summary["questions"],
            "one_group_questions": summary["one_group_questions"],
            "cases": summary["cases"],
            "max_age": max_age,
            "max_age_at": summary["max_age_at"],
            "discrete_changes": summary["discrete_changes"],
            "headroom": headroom_record(max_age, bound),
        }
    # Every changed answer must be consistent with the bound this set would freeze (all its AGEs are within it).
    recheck = Run()
    rechecked = evaluate_g1(ctx, bounds, recheck, only=["representative"])
    if recheck.failures:
        refusals.extend(recheck.failures)
    for settings in SETTINGS:
        summary = summarize(rechecked["representative"][settings]["questions"], bounds[settings])
        if summary["over_bound"] or summary["inconsistent_flips"]:
            refusals.append(f"{settings}: over the bound {summary['over_bound']}, changed answers not consistent "
                            f"with it {summary['inconsistent_flips']}")
    if refusals:
        for refusal in refusals:
            print(f"FAIL {refusal}")
        print("compare.py freeze: refused; the gate fails and the cause is investigated (the cap is never raised)")
        return 1
    doc = {
        "schema_version": 1,
        "what": "G1's frozen bounds: the anchored gap error (AGE, logits) of d1-2b-f16 through the upstream oracle "
                "against the HF reference, per settings id. compare.py gate fails on any question above them, and on "
                "observations from another computation than the identities below.",
        "rule": {
            "name": "clipped headroom",
            "cap": str(CAP),
            "headroom_factor": str(HEADROOM),
            "floor": str(FLOOR),
            "quantum": str(QUANTUM),
            "bound": "min(cap, max(floor, ceil(headroom_factor * M, quantum))), M the representative set's max AGE",
            "preconditions": "M <= cap / 2, and every changed discrete answer consistent with the bound",
            "predeclared": "P11 acceptance plan v2, item b, before any F16 oracle observation existed",
        },
        "rationale": [
            "The cap, 0.020 logits for both settings, is a predeclared quality requirement under test, not an "
            "established noise bound: no F16 Flash Attention data existed when it was set. Flash Attention gets no "
            "allowance of its own; a failure of cpu-t4-fa-on blocks the product default.",
            "The bound is absolute and covers every offered group: the error of a gap is the hidden state's error "
            "projected on the difference of two label rows of the LM head, which does not scale with the gap.",
            "The headroom factor and the floor are engineering margins, not statistical guarantees; the clipped rule "
            "guarantees at least twice the representative maximum (recorded as headroom.ratio).",
            "If a set exceeds its bound the gate fails and the cause is investigated; the bound is never widened to "
            "make it pass. A changed target or default is a new experiment.",
        ],
        "bounds": bounds,
        "observed": observed,
        "identity": frozen_identities(run),
        "inputs": dict(sorted(run.inputs.items())),
        "tool": {"compare.py": sha256_file(Path(__file__))},
        "frozen_with": frozen_with(),
    }
    ctx.bounds_path.write_text(json.dumps(doc, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                               encoding="utf-8")
    for settings in SETTINGS:
        o = observed[settings]
        print(f"{settings}: max AGE {o['max_age']!r} at {o['max_age_at']} -> bound {bounds[settings]} "
              f"(headroom {o['headroom']})")
    stamp = doc["frozen_with"]
    print(f"frozen with compare.py {stamp['comparator']['sha256']} and make_heldout.py "
          f"{stamp['heldout_recipe']['sha256']} (seed {stamp['heldout_recipe']['seed']:#x})")
    print(f"wrote {rel(ctx.bounds_path)} (sha256 {sha256_file(ctx.bounds_path)})")
    return 0


def evaluate_drift(ctx: Context, run: Run, metrics: bool = True) -> Dict[str, Any]:
    """G3's report: Q4 against F16 per setting, and Flash Attention drift per artifact. Report only."""
    artifacts = ctx.artifacts()
    out: Dict[str, Any] = {}
    for corpus in ctx.corpora():
        cases = load_cases(corpus.cases_dir)
        run.failures.extend(check_membership(corpus, cases))
        sides: Dict[Tuple[str, str], Dict[str, Observation]] = {}
        for artifact in (F16, Q4):
            for settings in SETTINGS:
                directory = corpus.oracle_path(artifact, settings)
                side = load_dir(directory, "oracle")
                run.failures.extend(check_join(corpus, cases, side, f"{artifact}/{settings}", artifact, settings))
                failures, fingerprint = native_identity(directory, artifact, settings, artifacts, cases,
                                                        corpus.cases_dir, side)
                run.failures.extend(f"{corpus.name}: {f}" for f in failures)
                run.identities[f"{corpus.name}/{artifact}/{settings}"] = fingerprint
                for path in sorted(directory.glob("*.json")):
                    run.digest(path)
                sides[(artifact, settings)] = side
        comparisons = {}
        pairs = [(f"quantization {s}", (Q4, s), (F16, s)) for s in SETTINGS] + \
                [(f"flash attention {a}", (a, "cpu-t4-fa-on"), (a, "cpu-t4-fa-off")) for a in (F16, Q4)]
        for what, cand, ref in pairs:
            label = f"{corpus.name} {cand[0]}/{cand[1]} vs {ref[0]}/{ref[1]}"
            result = compare_corpus(corpus, sides[cand], sides[ref], NATIVE_TEMPERATURE, NATIVE_TEMPERATURE, None,
                                    run, label, self_check=True, metrics=metrics)
            if not metrics:
                continue
            comparisons[what] = {
                "candidate": f"{cand[0]}/{cand[1]}", "reference": f"{ref[0]}/{ref[1]}",
                "summary": round_summary(summarize(result["questions"], None)),
                "questions": report_rows(result["questions"]),
            }
        out[corpus.name] = comparisons
    return out


def cmd_drift(ctx: Context, out: Path) -> int:
    run = check_integrity(ctx)
    refused = bool(run.failures)
    results: Dict[str, Any] = {}
    if refused:
        run.identities = {key: value for key, value in run.identities.items() if not key.endswith("/hf")}
    else:
        run = Run()
        try:
            results = evaluate_drift(ctx, run)
        except Failure as error:
            run.failures.append(str(error))
            results = {}
    report = {
        "schema": SCHEMA,
        "what": "G3's drift report: d1-2b-q4km against d1-2b-f16 through the upstream oracle under the same settings, "
                "and Flash Attention on against off for each artifact. A report, not a gate: no bound applies. "
                "Discrete changes use upstream's arithmetic on both sides; margins are the reference side's.",
        "metric": "AGE anchored on the reference side's winner (the F16 run, or the fa-off run)",
        "temperature_f32": NATIVE_TEMPERATURE[1],
        "identity": dict(sorted(run.identities.items())),
        "tool": {"compare.py": sha256_file(Path(__file__))},
        "inputs": dict(sorted(run.inputs.items())),
        "complete": not run.failures,
        "failures": run.failures,
        "corpora": results,
    }
    if refused:
        report["refused"] = REFUSED
    write_report(out / "q4-drift.json", report)
    for name, comparisons in results.items():
        for what, entry in comparisons.items():
            s = entry["summary"]
            print(f"{name} {what}: {s.get('questions')} questions, max AGE {s.get('max_age')} "
                  f"({s.get('max_age_at')}), discrete changes {s.get('discrete_changes')}, max |dp| {s.get('max_p_abs')}")
    for failure in run.failures:
        print(f"FAIL {failure}")
    if refused:
        print(f"G3 drift report: refused: {REFUSED}")
    return 1 if run.failures else 0


def accepted_identities(bounds_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The identity sets the bounds hold for: the frozen one, then every recorded revalidation's."""
    sets = [bounds_doc.get("identity") or {}]
    sets += [entry.get("identity") or {} for entry in bounds_doc.get("revalidations") or []]
    return sets


def identity_mismatches(bounds_doc: Dict[str, Any], identities: Dict[str, Optional[str]]) -> List[str]:
    """The run's identity keys (corpus/side) whose computation no accepted identity set names."""
    out = []
    for key, value in sorted(identities.items()):
        _, _, rest = key.partition("/")
        accepted = {s.get(f"representative/{rest}") for s in accepted_identities(bounds_doc)}
        if value is None or value not in accepted:
            out.append(key)
    return out


def cmd_revalidate(ctx: Context, out: Path) -> int:
    """After a regeneration that changed a computation identity (a llama.cpp update, a rebuilt oracle, another
    reference environment): every corpus must pass under the frozen bounds, unchanged; then the new identities are
    appended to bounds.json as a revalidation. Nothing is refitted."""
    try:
        bounds_doc = load_json(ctx.bounds_path)
        bounds = {s: float(bounds_doc["bounds"][s]) for s in SETTINGS}
    except (Failure, KeyError, TypeError, ValueError) as error:
        print(f"compare.py revalidate: no frozen bounds ({error})", file=sys.stderr)
        return 1
    checked = check_integrity(ctx)
    if checked.failures:
        for failure in checked.failures:
            print(f"FAIL {failure}")
        print(f"compare.py revalidate: refused: {REFUSED}")
        return 1
    run = Run()
    try:
        results = evaluate_g1(ctx, bounds, run)
    except Failure as error:
        run.failures.append(str(error))
        results = {}
    changed = identity_mismatches(bounds_doc, run.identities)
    if run.failures:
        for failure in run.failures:
            print(f"FAIL {failure}")
        print("compare.py revalidate: refused: the regenerated observations fail the gate under the frozen bounds")
        return 1
    if not changed:
        print("compare.py revalidate: every identity is already accepted; nothing to revalidate")
        return 1
    if set(results) != {c.name for c in ctx.corpora()} or "heldout" not in results:
        print("compare.py revalidate: refused: a revalidation needs the representative and held-out corpora")
        return 1
    entry = {
        "identity": {f"representative/{key.partition('/')[2]}": value for key, value in sorted(run.identities.items())
                     if key.startswith("representative/")},
        "changed": changed,
        "summary": {name: {s: round_summary(summarize(v["questions"], bounds[s])) for s, v in per.items()}
                    for name, per in results.items()},
        "inputs": canonical_digest(run.inputs),
        "tool": {"compare.py": sha256_file(Path(__file__))},
    }
    held = {key.partition("/")[2]: value for key, value in run.identities.items() if key.startswith("heldout/")}
    rep = {key.partition("/")[2]: value for key, value in run.identities.items() if key.startswith("representative/")}
    if held != rep:
        print("compare.py revalidate: refused: the corpora were evaluated by different computations")
        return 1
    bounds_doc.setdefault("revalidations", []).append(entry)
    ctx.bounds_path.write_text(json.dumps(bounds_doc, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                               encoding="utf-8")
    print(f"revalidated {', '.join(changed)} under the frozen bounds {bounds}; wrote {rel(ctx.bounds_path)}")
    return 0


def cmd_check(ctx: Context, out: Path) -> int:
    """Recompute the committed reports from the committed data; fail on any difference."""
    failures = []
    for name, command in (("report.json", cmd_gate), ("q4-drift.json", cmd_drift)):
        committed = ctx.g1 / name
        if not committed.is_file():
            failures.append(f"{rel(committed)}: missing")
            continue
        if command(ctx, out) != 0:
            failures.append(f"{name}: the recomputation failed")
        fresh = out / name
        if not fresh.is_file() or load_json(fresh) != load_json(committed):
            failures.append(f"{rel(committed)} differs from a fresh run ({rel(fresh)})")
    for failure in failures:
        print(f"FAIL {failure}")
    print("check:", "the committed reports match a fresh run" if not failures else "FAILED")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("integrity", "gate", "freeze", "revalidate", "drift", "check"))
    parser.add_argument("--out", default=str(OUT), help="where reports are written (default: %(default)s)")
    args = parser.parse_args(argv)
    ctx = Context()
    command = {"integrity": cmd_integrity, "gate": cmd_gate, "freeze": cmd_freeze, "revalidate": cmd_revalidate,
               "drift": cmd_drift, "check": cmd_check}[args.command]
    try:
        return command(ctx, Path(args.out))
    except Failure as error:
        print(f"FAIL {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
