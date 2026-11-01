"""Tests for compare.py: the metric, the certificate, the freeze, and every integrity rule as a mutation test.

A synthetic corpus is written to a temporary tree in the layouts compare.py reads (case files, HF observations with
their run log, both artifacts' oracle collections with their collection.json, and for the held-out tests a generated
corpus with its generation record), with real pins, lock and recipe digests so that the identity checks run for real.
Each mutation test breaks one thing and checks that the gate fails, or refuses, for that reason.

Run: python3 -m unittest discover -s tools/conformance/decision_g1/tests
"""

import copy
import importlib.util
import json
import math
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare", HERE.parent / "compare.py")
cmp = importlib.util.module_from_spec(spec)
sys.modules["compare"] = cmp
spec.loader.exec_module(cmp)

T = cmp.upstream_temperature()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return cmp.f32_bits(f32(x))


def logits_for(probabilities):
    """f32 logits whose upstream distribution is close to `probabilities`."""
    return [f32(T * math.log(p)) for p in probabilities]


# ---------------------------------------------------------------------------------------------------------------------
# The synthetic corpus
# ---------------------------------------------------------------------------------------------------------------------

QUESTIONS = {
    # case id -> list of (question id, type, keys, codes, aliases per group, logits per selected token)
    "syn-choice": [("team", "choice", ["billing", "shipping", "technical"], ["A", "B", "C"], 2,
                    [5.0, 1.5, 7.25, 2.0, 3.5, 0.25])],
    "syn-score": [("level", "score", ["0", "1", "2", "3"], ["0", "1", "2", "3"], 1, [1.0, 2.5, 2.25, -1.0])],
    "syn-multi": [("yes", "noul", ["true", "false"], ["true", "false"], 3, [4.0, 3.5, 3.0, 2.0, 1.75, 1.5]),
                  ("one", "choice", ["only"], ["A"], 2, [3.0, 1.0])],
}
EXCLUDED = {"syn-refused": "upstream_refuses", "syn-different": "prompt_differs"}
INCLUDED = {"syn-choice": ["team"], "syn-multi": ["yes", "one"], "syn-score": ["level"]}
PASS_TOKENS = 3  # every synthetic pass: prepare_block's three token ids
RECIPE_FILES = ("CMakeLists.txt", "identity.cmake", "oracle.cpp", "test_access.h", "build.sh", "cases.py", "oracle.py")


def request_text(case_id):
    return json.dumps({"state": f"state of {case_id}", "questions": {q[0]: {"type": q[1]} for q in
                                                                       QUESTIONS.get(case_id, [("q", "noul")])}})


def prepare_block(case_id, temperature, prompt_suffix=""):
    passes, answers, parse = [], [], []
    for j, (qid, qtype, keys, codes, aliases, _) in enumerate(QUESTIONS[case_id]):
        tokens = [100 + j, 200 + j, 300 + j] + ([7] if prompt_suffix else [])
        select = [1000 + 10 * g + a for g in range(len(keys)) for a in range(aliases)]
        passes.append({"question": qid, "prompt": f"<prompt {case_id} {qid}>{prompt_suffix}", "token_ids": tokens,
                       "output_index": len(tokens) - 1, "select": select})
        answers.append({"question": qid, "type": qtype, "pass": j,
                        "groups": [{"key": k, "code": c, "range": [g * aliases, (g + 1) * aliases]}
                                   for g, (k, c) in enumerate(zip(keys, codes))],
                        "temperature": temperature[0], "temperature_f32": temperature[1]})
        parse.append({"id": qid, "type": qtype, "keys": keys})
    return parse, passes, answers


def answer_for(qtype, keys, scores):
    p = cmp.upstream_probabilities(scores, T)
    if qtype == "choice":
        return {"type": "choice", "choice": keys[cmp.upstream_choice(p)],
                "probabilities": dict(zip(keys, p)), "confidence": max(p)}
    if qtype == "score":
        return {"type": "score", "score": float(cmp.upstream_median(p)), "legend": {k: k for k in keys},
                "probabilities": dict(zip(keys, p)), "confidence": max(p)}
    return {"type": "noul", "noul": p[keys.index("true")]}


def stages(case_id, side, perturb=0.0, temperature=None):
    temperature = temperature or (cmp.HF_TEMPERATURE if side == "hf" else cmp.NATIVE_TEMPERATURE)
    parse, passes, answers = prepare_block(case_id, temperature)
    raw, scores, final = [], [], {}
    for (qid, qtype, keys, _, aliases, logits), answer in zip(QUESTIONS[case_id], answers):
        values = [f32(x + (perturb * (i % 3 - 1) if side == "oracle" else 0.0)) for i, x in enumerate(logits)]
        group = [max(values[g * aliases:(g + 1) * aliases]) for g in range(len(keys))]
        raw.append([cmp.f32_bits(v) for v in values])
        scores.append([cmp.f32_bits(v) for v in group])
        final[qid] = answer_for(qtype, keys, group)
    tokens = sum(len(p["token_ids"]) for p in passes)
    return {
        "parse": {"questions": parse},
        "prepare": {"passes": passes, "answers": answers},
        "evaluate": {"source": "model", "raw_scores": raw, "scores": scores},
        "finalize": {"answers": final, "usage": {"input_tokens": tokens, "output_tokens": 0}},
    }


class Tree:
    """A temporary repository-shaped tree with a synthetic representative corpus."""

    def __init__(self, perturb=0.001, equal=False):
        self.tmp = Path(tempfile.mkdtemp(prefix="g1-test-"))
        self.fixtures = self.tmp / "integration-tests" / "fixtures" / "decision"
        self.g1 = self.fixtures / "g1"
        self.cases = self.fixtures / "syn" / "cases"
        self.hf = self.fixtures / "syn" / "observations"
        self.oracle = self.g1 / "oracle"
        for d in (self.cases, self.hf, self.g1):
            d.mkdir(parents=True)
        shutil.copy(cmp.FIXTURES / "artifacts.json", self.fixtures / "artifacts.json")
        self.ids = sorted(list(QUESTIONS) + list(EXCLUDED))
        for case_id in self.ids:
            doc = {"schema_version": 1, "id": case_id, "profile": cmp.PROFILE, "title": case_id, "covers": ["x"],
                   "artifact": cmp.F16, "settings": "cpu-t4-fa-off", "request_text": request_text(case_id),
                   "upstream": {}, "xybrid": {}}
            self.write(self.cases / f"{case_id}.json", doc)
        self.write_hf()
        for artifact in (cmp.F16, cmp.Q4):
            for settings in cmp.SETTINGS:
                self.write_oracle(settings, 0.0 if equal else perturb, artifact)
        self.write_allowlist()
        self.ctx = cmp.Context(fixtures=self.fixtures, root=self.tmp)
        self.out = self.tmp / "out"

    def write(self, path, doc):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    def sha(self, case_id):
        return cmp.sha256_text(request_text(case_id))

    def write_hf(self):
        outputs = {}
        for case_id in self.ids:
            if case_id == "syn-refused":
                hf = stages("syn-choice", "hf")
            elif case_id == "syn-different":
                hf = stages("syn-score", "hf")
            else:
                hf = stages(case_id, "hf")
            detail = {"prepare": [{"question": a["question"], "temperature_f64": cmp.HF_TEMPERATURE[2]}
                                  for a in hf["prepare"]["answers"]]}
            doc = {"schema_version": 1, "case": case_id, "source": "hf-reference", "artifact": cmp.HF_ARTIFACT,
                   "settings": cmp.HF_SETTINGS, "request_text_sha256": self.sha(case_id), "hf": hf, "detail": detail}
            self.write(self.hf / f"{case_id}.json", doc)
            outputs[case_id] = cmp.sha256_file(self.hf / f"{case_id}.json")
        self.run_log = {
            "reference": {"artifact": "d1-2b-hf", "repo": "LiquidAI/d1-2B",
                          "revision": "10e7671d44e4b37700020b1eed3f594c5f2af0b4"},
            "inputs": {"pins_sha256": cmp.sha256_file(cmp.HF_TOOL / "pins.json"),
                       "requirements_sha256": cmp.sha256_file(cmp.HF_TOOL / "requirements.txt"),
                       "recipe": {n: cmp.sha256_file(cmp.HF_TOOL / n) for n in cmp.HF_OBSERVATION_RECIPE},
                       "requests": {c: self.sha(c) for c in self.ids}},
            "outputs": outputs,
            "code_imported": [{"module": "abc.api", "sha256": "0" * 64}],
            "environment": {"versions": {"torch": "2.14.0+cpu"}, "cpu_capability": "AVX512"},
            "numerics": dict(cmp.HF_NUMERICS),
            "float32_parity": {"calls": 10, "bit_identical": 10, "non_float32_inputs": 0, "modules_hooked": 27,
                               "modules_verified": 27},
        }
        self.write(self.hf / "run.json", self.run_log)

    def write_oracle(self, settings, perturb, artifact=cmp.F16):
        directory = self.oracle / artifact / settings
        artifacts = json.loads((self.fixtures / "artifacts.json").read_text())
        art, st = artifacts["artifacts"][artifact], artifacts["settings"][settings]
        recipe = {name: cmp.sha256_file(cmp.ORACLE_TOOL / name) for name in RECIPE_FILES}
        collector = {"official": True, "problems": []}
        build = {"fingerprint": "f" * 64, "llama_cpp_dir": "/x",
                 "llama_cpp": {"checkout": "git", "commit": cmp.pinned_llama_cpp(), "dirty": False},
                 "build": {"build_type": "Release", "backends": ["cpu"], "cxx_compiler": "GNU 11.5.0"}}
        entries = []
        for case_id in self.ids:
            sha = self.sha(case_id)
            if case_id == "syn-refused":
                up = {"parse": {"refusal": {"exception": "invalid_argument", "status": 400,
                                            "type": "invalid_request_error", "message": "no"}}}
                ran = None
            elif case_id == "syn-different":
                up = stages("syn-score", "oracle", perturb)
                up["prepare"]["passes"][0]["prompt"] += " (rendered otherwise)"
                ran = settings
            else:
                up = stages(case_id, "oracle", perturb)
                ran = settings
            inputs = {"request_sha256": sha, "has_media": False, "metadata": None, "select_logits": None,
                      "model": {"artifact": artifact, "sha256": art["sha256"], "variant_sha256": None}}
            file_sha = cmp.sha256_file(self.cases / f"{case_id}.json")
            doc = {"schema_version": 1, "id": case_id, "source_file": f"{case_id}.json",
                   "case_file_sha256": file_sha, "request_sha256": sha, "inputs_sha256": cmp.canonical_digest(inputs),
                   "inputs": inputs, "artifact": artifact, "settings": ran, "collector": collector,
                   "build_fingerprint": build["fingerprint"], "recipe_sha256": cmp.canonical_digest(recipe),
                   "upstream": up}
            name = f"{case_id}.{sha}.json"
            self.write(directory / name, doc)
            entries.append({"file": f"{case_id}.json", "id": case_id, "file_sha256": file_sha, "request_sha256": sha,
                            "inputs_sha256": cmp.canonical_digest(inputs), "inputs": inputs, "status": "observed",
                            "observation": name})
        record = {
            "schema_version": 1,
            "status": "complete",
            "corpus": {"dir": str(self.cases), "files": len(self.ids), "selected": len(self.ids),
                       "expected": len(self.ids), "only": None},
            "artifact": {"id": artifact, "file": art["file"], "bytes": art["bytes"], "sha256": art["sha256"]},
            "recipe": recipe,
            "recipe_sha256": cmp.canonical_digest(recipe),
            "collector": collector,
            "build": build,
            "settings": {settings: {
                "llama_server_args": ["-t", "4"],
                "context": {"n_ctx": 4096, "n_batch": 4096, "n_ubatch": 4096, "n_seq_max": 1, "n_outputs_max": 1,
                            "n_threads": 4, "n_threads_batch": 4, "flash_attn": "on" if st["flash_attn"] else "off",
                            "causal_attn": False, "memory": False, "n_gpu_layers": 0, "offload_devices": "none"},
                "oracle_devices": [{"name": "CPU", "type": "cpu"}],
                "model_devices": [],
            }},
            "platform": {"system": "Linux", "machine": "x86_64", "cpu": "test", "libc": "glibc 2.34"},
            "cases": entries,
        }
        self.write(directory / "collection.json", record)

    def write_allowlist(self):
        doc = {"schema_version": 1, "corpora": {"representative": {
            "cases": str(self.cases.relative_to(self.tmp)),
            "hf": str(self.hf.relative_to(self.tmp)),
            "oracle": str(self.oracle.relative_to(self.tmp)),
            "counts": {"cases": 5, "included_cases": 3, "included_questions": 4, "excluded_cases": 2},
            "included": {"syn-choice": ["team"], "syn-multi": ["yes", "one"], "syn-score": ["level"]},
            "excluded": {
                "syn-refused": {"kind": "upstream_refuses", "stage": "parse", "exception": "invalid_argument",
                                "status": 400, "reason": "test"},
                "syn-different": {"kind": "prompt_differs", "questions": {"level": ["prompt bytes"]},
                                  "reason": "test"},
            },
        }}}
        self.write(self.g1 / "allowlist.json", doc)

    def freeze(self):
        return cmp.cmd_freeze(self.ctx, self.out)

    def gate(self):
        return cmp.cmd_gate(self.ctx, self.out)

    def report(self):
        return json.loads((self.out / "report.json").read_text())

    def edit(self, path, change):
        doc = json.loads(path.read_text())
        change(doc)
        self.write(path, doc)

    def oracle_file(self, case_id, settings="cpu-t4-fa-off"):
        return self.oracle / cmp.F16 / settings / f"{case_id}.{self.sha(case_id)}.json"

    def cleanup(self):
        shutil.rmtree(self.tmp)


def add_held_out(t, targets=None):
    """A generated corpus as make_heldout.py and the collections leave it next to frozen bounds. It holds:
    - the included synthetic cases only, every question included and none excluded;
    - a generation record made by the frozen recipe;
    - the HF run log;
    - both artifacts' collections, copied from the representative ones, which hold the same requests."""
    heldout = t.g1 / "heldout"
    ids = sorted(INCLUDED)
    for name, src in (("cases", t.cases), ("hf", t.hf)):
        (heldout / name).mkdir(parents=True)
        for case_id in ids:
            shutil.copy(src / f"{case_id}.json", heldout / name / f"{case_id}.json")
    log = copy.deepcopy(t.run_log)
    log["inputs"]["requests"] = {c: t.sha(c) for c in ids}
    log["outputs"] = {c: cmp.sha256_file(heldout / "hf" / f"{c}.json") for c in ids}
    t.write(heldout / "hf" / "run.json", log)
    for artifact in (cmp.F16, cmp.Q4):
        for settings in cmp.SETTINGS:
            src, dst = t.oracle / artifact / settings, heldout / "oracle" / artifact / settings
            dst.mkdir(parents=True)
            for case_id in ids:
                shutil.copy(src / f"{case_id}.{t.sha(case_id)}.json", dst)
            record = json.loads((src / "collection.json").read_text())
            record["corpus"].update({"dir": str(heldout / "cases"), "files": len(ids), "selected": len(ids),
                                     "expected": len(ids)})
            record["cases"] = [e for e in record["cases"] if e["id"] in ids]
            t.write(dst / "collection.json", record)
    recipe = json.loads(t.ctx.bounds_path.read_text())["frozen_with"]["heldout_recipe"]
    vocab = json.loads((t.fixtures / "artifacts.json").read_text())["artifacts"][cmp.Q4]
    targets = targets or {}
    t.write(heldout / "generation.json", {
        "schema_version": 1,
        "what": "a synthetic generation record",
        "recipe": dict(recipe),
        "cases": [{"id": c, "topic": "test", "questions": INCLUDED[c], "request_text_sha256": t.sha(c),
                   "target_tokens": targets.get(c), "measured_tokens": targets.get(c),
                   "pass_tokens": [PASS_TOKENS] * len(INCLUDED[c])} for c in ids],
        "measured_with": {"tool": "tools/conformance/decision/build/bin/decision-oracle", "tool_sha256": "e" * 64,
                          "tool_identity": {"identity_version": 1, "fingerprint": "f" * 64},
                          "vocab": vocab["file"], "vocab_sha256": vocab["sha256"], "calls": 7},
    })
    spec = {"cases": str((heldout / "cases").relative_to(t.tmp)), "hf": str((heldout / "hf").relative_to(t.tmp)),
            "oracle": str((heldout / "oracle").relative_to(t.tmp)),
            "counts": {"cases": len(ids), "included_cases": len(ids),
                       "included_questions": sum(len(q) for q in INCLUDED.values()), "excluded_cases": 0},
            "included": {c: INCLUDED[c] for c in ids}, "excluded": {}}
    t.write(heldout / "allowlist.json", {"schema_version": 1, "corpora": {"heldout": spec}})
    return heldout


class Quiet(unittest.TestCase):
    """Runs the comparator with its prints captured."""

    def setUp(self):
        import io
        import contextlib
        self.buffer = io.StringIO()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(self.buffer))
        self.stack.enter_context(contextlib.redirect_stderr(self.buffer))
        self.trees = []

    def tearDown(self):
        self.stack.close()
        for tree in self.trees:
            tree.cleanup()

    def tree(self, **kwargs):
        t = Tree(**kwargs)
        self.trees.append(t)
        return t

    def frozen_tree(self, **kwargs):
        t = self.tree(**kwargs)
        self.assertEqual(t.freeze(), 0, self.buffer.getvalue())
        return t

    def assert_gate_fails(self, tree, fragment):
        status = tree.gate()
        failures = tree.report()["failures"]
        self.assertEqual(status, 1, f"the gate passed; expected a failure about {fragment!r}")
        self.assertTrue(any(fragment in f for f in failures), f"no failure mentions {fragment!r}: {failures}")


# ---------------------------------------------------------------------------------------------------------------------
# The freeze and the gate
# ---------------------------------------------------------------------------------------------------------------------


class FreezeTest(Quiet):
    def test_a_clean_corpus_freezes_and_passes(self):
        t = self.frozen_tree()
        bounds = json.loads(t.ctx.bounds_path.read_text())
        for settings in cmp.SETTINGS:
            observed = bounds["observed"][settings]
            self.assertGreater(observed["max_age"], 0)
            self.assertEqual(observed["one_group_questions"], 1)
            self.assertGreaterEqual(observed["headroom"]["ratio"], 2.0)
            self.assertEqual(bounds["bounds"][settings], float(cmp.clipped_bound(observed["max_age"])))
        self.assertEqual(t.gate(), 0, t.report()["failures"])
        self.assertTrue(t.report()["passed"])

    def test_bounds_are_frozen_once_and_before_held_out_data(self):
        t = self.frozen_tree()
        self.assertEqual(t.freeze(), 1)
        u = self.tree()
        (u.g1 / "heldout").mkdir()
        self.assertEqual(u.freeze(), 1)

    def test_zero_error_records_a_null_ratio(self):
        t = self.frozen_tree(equal=True)
        bounds = json.loads(t.ctx.bounds_path.read_text())
        for settings in cmp.SETTINGS:
            self.assertEqual(bounds["observed"][settings]["max_age"], 0.0)
            self.assertEqual(bounds["observed"][settings]["headroom"], {"ratio": None, "status": "zero_error"})
            self.assertEqual(bounds["bounds"][settings], 0.005)
        self.assertNotIn("Infinity", t.ctx.bounds_path.read_text())
        self.assertNotIn("NaN", t.ctx.bounds_path.read_text())

    def test_a_set_above_half_the_cap_is_refused(self):
        t = self.tree(perturb=0.006)  # anchored gaps move by up to 2 x 0.006 > 0.010
        self.assertEqual(t.freeze(), 1)
        self.assertFalse(t.ctx.bounds_path.exists())
        self.assertIn("above half the cap", self.buffer.getvalue())

    def test_clipped_headroom_rule(self):
        self.assertEqual(cmp.clipped_bound(0.0), cmp.FLOOR)
        self.assertEqual(float(cmp.clipped_bound(0.0034)), 0.011)
        self.assertEqual(float(cmp.clipped_bound(0.009)), 0.02)  # clipped: 2.22x
        self.assertEqual(float(cmp.clipped_bound(0.0100)), 0.02)  # clipped: 2x
        self.assertEqual(cmp.headroom_record(0.009, cmp.clipped_bound(0.009))["ratio"], 2.22222)

    def test_the_freeze_records_the_comparator_and_the_held_out_recipe(self):
        t = self.frozen_tree()
        stamp = json.loads(t.ctx.bounds_path.read_text())["frozen_with"]
        self.assertEqual(stamp, {
            "comparator": {"file": "tools/conformance/decision_g1/compare.py",
                           "sha256": cmp.sha256_file(cmp.COMPARATOR)},
            "heldout_recipe": {"file": "tools/conformance/decision_g1/make_heldout.py",
                               "sha256": cmp.sha256_file(cmp.HELDOUT_RECIPE), "seed": 0x5EED6011D1},
        })

    def test_the_freeze_refuses_unless_every_integrity_check_passes(self):
        t = self.tree()
        shutil.rmtree(t.oracle / cmp.Q4 / "cpu-t4-fa-on")
        self.assertEqual(t.freeze(), 1)
        self.assertFalse(t.ctx.bounds_path.exists())
        self.assertIn(cmp.REFUSED, self.buffer.getvalue())
        self.assertNotIn("max AGE", self.buffer.getvalue())

    def test_a_question_above_the_bound_fails_the_gate(self):
        t = self.frozen_tree()
        for settings in cmp.SETTINGS:
            path = t.oracle_file("syn-choice", settings)

            def lift(doc):
                ev = doc["upstream"]["evaluate"]
                raw = [cmp.f32(b) for b in ev["raw_scores"][0]]
                raw[4] = f32(raw[4] + 0.5)
                ev["raw_scores"][0] = [cmp.f32_bits(x) for x in raw]
                ev["scores"][0] = [cmp.f32_bits(max(raw[i:i + 2])) for i in (0, 2, 4)]
                final = answer_for("choice", ["billing", "shipping", "technical"],
                                   [max(raw[i:i + 2]) for i in (0, 2, 4)])
                doc["upstream"]["finalize"]["answers"]["team"] = final
            t.edit(path, lift)
            # The collection record still names the same file; the identity is unchanged.
        self.assert_gate_fails(t, "AGE above the bound")


class RevalidationTest(Quiet):
    def held_out(self, t):
        add_held_out(t)

    def rebuilt(self, t, compiler):
        for corpus_dir in (t.oracle, t.g1 / "heldout" / "oracle"):
            for settings in cmp.SETTINGS:
                t.edit(corpus_dir / cmp.F16 / settings / "collection.json",
                       lambda d: d["build"]["build"].update({"cxx_compiler": compiler}))

    def test_a_rebuilt_oracle_fails_until_revalidated_and_the_bounds_stay(self):
        t = self.frozen_tree()
        self.held_out(t)
        self.assertEqual(t.gate(), 0, t.report()["failures"])
        frozen = json.loads(t.ctx.bounds_path.read_text())["bounds"]
        self.rebuilt(t, "GNU 12.1.0")
        self.assert_gate_fails(t, "a computation identity other than the one the bounds were frozen on")
        self.assertEqual(cmp.cmd_revalidate(t.ctx, t.out), 0, self.buffer.getvalue())
        doc = json.loads(t.ctx.bounds_path.read_text())
        self.assertEqual(doc["bounds"], frozen, "a revalidation never refits")
        self.assertEqual(len(doc["revalidations"]), 1)
        self.assertEqual(t.gate(), 0, t.report()["failures"])
        self.assertEqual(cmp.cmd_revalidate(t.ctx, t.out), 1, "nothing left to revalidate")

    def test_a_revalidation_that_fails_the_bounds_is_refused(self):
        t = self.frozen_tree()
        self.held_out(t)
        self.rebuilt(t, "GNU 12.1.0")
        bound = json.loads(t.ctx.bounds_path.read_text())["bounds"]["cpu-t4-fa-off"]

        def lift(doc):
            ev = doc["upstream"]["evaluate"]
            raw = [cmp.f32(b) for b in ev["raw_scores"][0]]
            raw[4] = f32(raw[4] + 10 * bound)
            ev["raw_scores"][0] = [cmp.f32_bits(x) for x in raw]
            ev["scores"][0] = [cmp.f32_bits(max(raw[i:i + 2])) for i in (0, 2, 4)]
            doc["upstream"]["finalize"]["answers"]["team"] = answer_for(
                "choice", ["billing", "shipping", "technical"], [max(raw[i:i + 2]) for i in (0, 2, 4)])
        t.edit(t.g1 / "heldout" / "oracle" / cmp.F16 / "cpu-t4-fa-off" / f"syn-choice.{t.sha('syn-choice')}.json",
               lift)
        before = t.ctx.bounds_path.read_text()
        self.assertEqual(cmp.cmd_revalidate(t.ctx, t.out), 1)
        self.assertEqual(t.ctx.bounds_path.read_text(), before)

    def test_a_revalidation_needs_the_held_out_corpus(self):
        t = self.frozen_tree()
        for settings in cmp.SETTINGS:
            t.edit(t.oracle / cmp.F16 / settings / "collection.json",
                   lambda d: d["build"]["build"].update({"cxx_compiler": "GNU 12.1.0"}))
        self.assertEqual(cmp.cmd_revalidate(t.ctx, t.out), 1)


# ---------------------------------------------------------------------------------------------------------------------
# A generated corpus: integrity before numbers
# ---------------------------------------------------------------------------------------------------------------------


class HeldOutTest(Quiet):
    """A generated corpus's record is checked before anything is compared. Every command that computes a number
    refuses, comparing nothing, unless the integrity checks pass."""

    def held_out_tree(self, targets=None):
        t = self.frozen_tree()
        add_held_out(t, targets)
        return t

    def generation(self, t):
        return t.g1 / "heldout" / cmp.GENERATION_FILE

    def assert_refused(self, t, fragment):
        """integrity fails for this reason; gate, drift and revalidate refuse it and compare nothing."""
        start = len(self.buffer.getvalue())
        self.assertEqual(cmp.cmd_integrity(t.ctx, t.out), 1)
        self.assertIn(fragment, self.buffer.getvalue()[start:])
        self.assert_gate_fails(t, fragment)
        report = t.report()
        self.assertEqual((report["corpora"], report["refused"]), ({}, cmp.REFUSED))
        self.assertEqual(cmp.cmd_drift(t.ctx, t.out), 1)
        drift = json.loads((t.out / "q4-drift.json").read_text())
        self.assertEqual((drift["corpora"], drift["refused"]), ({}, cmp.REFUSED))
        before = t.ctx.bounds_path.read_text()
        self.assertEqual(cmp.cmd_revalidate(t.ctx, t.out), 1)
        self.assertEqual(t.ctx.bounds_path.read_text(), before)
        self.assertNotIn("max AGE", self.buffer.getvalue()[start:])

    def test_a_generated_corpus_passes_integrity_the_gate_and_the_drift_report(self):
        t = self.held_out_tree({"syn-choice": PASS_TOKENS})
        self.assertEqual(cmp.cmd_integrity(t.ctx, t.out), 0, self.buffer.getvalue())
        self.assertEqual(t.gate(), 0, t.report()["failures"])
        self.assertEqual(set(t.report()["corpora"]), {"representative", "heldout"})
        inputs = {k.split(f"{t.tmp}/")[-1] for k in t.report()["inputs"]}
        self.assertIn("integration-tests/fixtures/decision/g1/heldout/generation.json", inputs)
        self.assertEqual(cmp.cmd_drift(t.ctx, t.out), 0, self.buffer.getvalue())

    def test_integrity_prints_identities_and_no_result(self):
        t = self.held_out_tree()
        start = len(self.buffer.getvalue())
        self.assertEqual(cmp.cmd_integrity(t.ctx, t.out), 0)
        printed = self.buffer.getvalue()[start:]
        for key in ("heldout/hf", f"heldout/{cmp.F16}/cpu-t4-fa-on", f"heldout/{cmp.Q4}/cpu-t4-fa-off",
                    "representative/hf"):
            self.assertIn(f"identity {key}: ", printed)
        self.assertIn("identities the frozen bounds accept: all of G1's", printed)
        self.assertTrue(printed.endswith("integrity: passed\n"), printed)
        # No gap, probability or answer: none of the gate's numbers for the same tree, and no decimal number at all.
        self.assertEqual(t.gate(), 0)
        numbers = set()
        for per_setting in t.report()["corpora"].values():
            for entry in per_setting.values():
                numbers.update(v for v in entry["summary"].values() if isinstance(v, float))
                for row in entry["questions"]:
                    numbers.update(row[k] for k in ("age", "p_max_abs", "raw_max_abs") if k in row)
                    numbers.update(v for v in row["answer"].values() if isinstance(v, float))
        self.assertTrue(numbers)
        for number in numbers:
            self.assertNotIn(repr(number), printed)
        self.assertIsNone(re.search(r"\d\.\d", printed), printed)
        for word in ("AGE", "probabilit", "answer", "max"):
            self.assertNotIn(word, printed)

    def test_a_failed_parity_proof_refuses_before_any_comparison(self):
        t = self.held_out_tree()
        t.edit(t.g1 / "heldout" / "hf" / "run.json", lambda d: d["float32_parity"].update({"bit_identical": 9}))
        self.assert_refused(t, "no passing float32 parity proof")

    def test_another_numerics_tuple_refuses_before_any_comparison(self):
        t = self.held_out_tree()
        t.edit(t.g1 / "heldout" / "hf" / "run.json", lambda d: d["numerics"].update({"precision": "bf16"}))
        self.assert_refused(t, "numerics")

    def test_a_comparator_other_than_the_frozen_one(self):
        t = self.held_out_tree()
        t.edit(t.ctx.bounds_path, lambda d: d["frozen_with"]["comparator"].update({"sha256": "0" * 64}))
        self.assert_refused(t, "the comparator that evaluates must be the frozen one")

    def test_a_recipe_other_than_the_frozen_one(self):
        t = self.held_out_tree()
        t.edit(t.ctx.bounds_path, lambda d: d["frozen_with"]["heldout_recipe"].update({"sha256": "0" * 64}))
        self.assert_refused(t, "a change to the recipe is a new held-out set")

    def test_a_generation_by_another_recipe(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["recipe"].update({"sha256": "0" * 64}))
        self.assert_refused(t, "not the one frozen with the bounds")

    def test_a_generation_with_another_seed(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["recipe"].update({"seed": d["recipe"]["seed"] + 1}))
        self.assert_refused(t, "not the one frozen with the bounds")

    def test_a_measuring_tool_recorded_with_a_local_path(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["measured_with"].update(
            {"tool": "/home/someone/xybrid/tools/conformance/decision/build/bin/decision-oracle"}))
        self.assert_refused(t, "not a repository path or a file name")

    def test_a_tool_identity_that_keeps_its_local_directory(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["measured_with"]["tool_identity"].update(
            {"llama_cpp_dir": "/home/someone/xybrid/vendor/llama-cpp"}))
        self.assert_refused(t, "no stamped identity of the measuring tool without local state")

    def test_a_measuring_tool_without_an_identity(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["measured_with"].pop("tool_identity"))
        self.assert_refused(t, "no stamped identity of the measuring tool without local state")

    def test_lengths_measured_by_another_build(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["measured_with"]["tool_identity"].update({"fingerprint": "0" * 64}))
        self.assert_refused(t, "but the prompt lengths were measured by build")

    def test_lengths_measured_on_another_vocabulary(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["measured_with"].update({"vocab_sha256": "0" * 64}))
        self.assert_refused(t, "not a pinned")

    def test_a_record_without_a_measurement_count(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["measured_with"].update({"calls": 0}))
        self.assert_refused(t, "no measurement count")

    def test_a_generation_of_other_request_bytes(self):
        t = self.held_out_tree()
        t.edit(self.generation(t), lambda d: d["cases"][2].update({"request_text_sha256": "0" * 64}))
        self.assert_refused(t, "syn-score: generated other request bytes")

    def test_a_target_the_generation_missed(self):
        t = self.held_out_tree({"syn-choice": PASS_TOKENS})
        t.edit(self.generation(t), lambda d: d["cases"][0].update({"target_tokens": PASS_TOKENS + 1}))
        self.assert_refused(t, f"syn-choice: measured {PASS_TOKENS} and {PASS_TOKENS} tokens, the target is "
                               f"{PASS_TOKENS + 1}")

    def test_a_target_the_collections_do_not_reach(self):
        t = self.held_out_tree({"syn-choice": PASS_TOKENS + 1})  # generated at 4 tokens, observed at 3
        t.edit(self.generation(t), lambda d: d["cases"][0].update({"pass_tokens": [PASS_TOKENS + 1]}))
        self.assert_refused(t, f"syn-choice: passes of [{PASS_TOKENS}] tokens (usage {PASS_TOKENS}), the generation "
                               f"measured [{PASS_TOKENS + 1}]")

    def test_an_exclusion_in_a_generated_corpus(self):
        t = self.held_out_tree()

        def exclude(doc):
            spec = doc["corpora"]["heldout"]
            spec["included"].pop("syn-score")
            spec["excluded"]["syn-score"] = {"kind": "hf_refuses", "stage": "parse", "exception": "x", "reason": "x"}
            spec["counts"].update({"included_cases": 2, "included_questions": 3, "excluded_cases": 1})
        t.edit(t.g1 / "heldout" / "allowlist.json", exclude)
        self.assert_refused(t, "the allowlist is not the generation's")

    def test_a_generated_corpus_without_its_record(self):
        t = self.held_out_tree()
        self.generation(t).unlink()
        self.assert_refused(t, "missing (make_heldout.py writes it next to the cases)")

    def test_held_out_data_without_frozen_bounds(self):
        t = self.held_out_tree()
        t.ctx.bounds_path.unlink()
        self.assertEqual(cmp.cmd_integrity(t.ctx, t.out), 1)
        self.assertIn("held-out data without a recipe frozen with the bounds", self.buffer.getvalue())
        self.assertEqual(t.gate(), 1)


# ---------------------------------------------------------------------------------------------------------------------
# Mutations: each integrity rule fails for its own reason
# ---------------------------------------------------------------------------------------------------------------------


class MutationTest(Quiet):
    def test_a_deleted_observation(self):
        t = self.frozen_tree()
        t.oracle_file("syn-score").unlink()
        self.assert_gate_fails(t, "no observation for syn-score")

    def test_a_duplicated_observation(self):
        t = self.frozen_tree()
        src = t.oracle_file("syn-score")
        dup = src.with_name(f"syn-score.{'0' * 64}.json")
        doc = json.loads(src.read_text())
        doc["request_sha256"] = "0" * 64
        t.write(dup, doc)
        self.assert_gate_fails(t, "two observations of case syn-score")

    def test_reordered_answers(self):
        t = self.frozen_tree()
        t.edit(t.oracle_file("syn-multi"), lambda d: d["upstream"]["prepare"]["answers"].reverse())
        self.assert_gate_fails(t, "answer mappings are not one per question, in order")

    def test_a_relabelled_observation(self):
        t = self.frozen_tree()
        src = t.oracle_file("syn-score")
        doc = json.loads(src.read_text())
        doc["id"] = "syn-other"
        src.unlink()
        t.write(src.with_name(f"syn-other.{t.sha('syn-score')}.json"), doc)
        self.assert_gate_fails(t, "not the collection file of case")

    def test_changed_request_bytes_without_the_digest(self):
        t = self.frozen_tree()
        t.edit(t.cases / "syn-choice.json", lambda d: d.update({"request_text": d["request_text"] + " "}))
        self.assert_gate_fails(t, "syn-choice was made from other request bytes")

    def test_swapped_settings(self):
        t = self.frozen_tree()
        t.edit(t.oracle_file("syn-choice"), lambda d: d.update({"settings": "cpu-t4-fa-on"}))
        self.assert_gate_fails(t, "names settings 'cpu-t4-fa-on', not 'cpu-t4-fa-off'")

    def test_another_build_identity(self):
        t = self.frozen_tree()
        path = t.oracle / cmp.F16 / "cpu-t4-fa-on" / "collection.json"
        t.edit(path, lambda d: d["build"]["build"].update({"cxx_compiler": "Clang 15"}))
        self.assert_gate_fails(t, "a computation identity other than the one the bounds were frozen on")

    def test_an_incomplete_collection(self):
        t = self.frozen_tree()
        path = t.oracle / cmp.F16 / "cpu-t4-fa-off" / "collection.json"
        t.edit(path, lambda d: d.update({"status": "incomplete"}))
        self.assert_gate_fails(t, "not a complete schema-v1 collection")

    def test_a_collection_by_another_tool(self):
        t = self.frozen_tree()
        path = t.oracle / cmp.F16 / "cpu-t4-fa-off" / "collection.json"

        def other(doc):
            doc["recipe"]["oracle.py"] = "0" * 64
            doc["recipe_sha256"] = cmp.canonical_digest(doc["recipe"])
        t.edit(path, other)
        self.assert_gate_fails(t, "made by another tools/conformance/decision/oracle.py")

    def test_a_case_file_edited_after_collection(self):
        t = self.frozen_tree()
        t.edit(t.cases / "syn-score.json", lambda d: d.update({"title": "edited"}))
        self.assert_gate_fails(t, "case syn-score: 'observed', or not this case file")

    def test_inputs_with_media(self):
        t = self.frozen_tree()

        def media(doc):
            doc["inputs"]["has_media"] = True
            doc["inputs_sha256"] = cmp.canonical_digest(doc["inputs"])
        t.edit(t.oracle_file("syn-choice"), media)
        self.assert_gate_fails(t, "not bound to its collection's case file, inputs, build, recipe")

    def test_an_observation_from_another_build(self):
        t = self.frozen_tree()
        t.edit(t.oracle_file("syn-choice"), lambda d: d.update({"build_fingerprint": "0" * 64}))
        self.assert_gate_fails(t, "not bound to its collection's case file, inputs, build, recipe")

    def test_a_partial_collection(self):
        t = self.frozen_tree()
        path = t.oracle / cmp.F16 / "cpu-t4-fa-off" / "collection.json"
        t.edit(path, lambda d: d["corpus"].update({"selected": 4}))
        self.assert_gate_fails(t, "corpus counts")

    def test_an_unofficial_collection(self):
        t = self.frozen_tree()
        path = t.oracle / cmp.F16 / "cpu-t4-fa-off" / "collection.json"
        t.edit(path, lambda d: d.update({"collector": {"official": False, "problems": ["dirty"]}}))
        self.assert_gate_fails(t, "not an official collection")

    def test_a_gpu_device(self):
        t = self.frozen_tree()
        path = t.oracle / cmp.F16 / "cpu-t4-fa-off" / "collection.json"
        t.edit(path, lambda d: d["settings"]["cpu-t4-fa-off"].update(
            {"oracle_devices": [{"name": "CUDA0", "type": "gpu"}]}))
        self.assert_gate_fails(t, "not the CPU only")

    def test_an_altered_native_temperature(self):
        t = self.frozen_tree()
        t.edit(t.oracle_file("syn-choice"),
               lambda d: d["upstream"]["prepare"]["answers"][0].update({"temperature_f32": "0x3fab2fdf"}))
        self.assert_gate_fails(t, "not this reference's")

    def test_an_altered_hf_temperature(self):
        t = self.frozen_tree()
        t.edit(t.hf / "syn-choice.json", lambda d: d["detail"]["prepare"][0].update({"temperature_f64": 1.25}))
        self.assert_gate_fails(t, "float64 temperature")

    def test_an_expected_refusal_replaced_by_a_runtime_failure(self):
        t = self.frozen_tree()

        def runtime(doc):
            doc["upstream"] = {"parse": {"questions": []}, "prepare": {"passes": [], "answers": []},
                               "evaluate": {"refusal": {"exception": "runtime_error", "status": 500}}}
        t.edit(t.oracle_file("syn-refused"), runtime)
        self.assert_gate_fails(t, "is excluded as upstream_refuses, but it refuses with")

    def test_a_stale_prompt_difference(self):
        t = self.frozen_tree()

        def same(doc):
            doc["upstream"]["prepare"]["passes"][0]["prompt"] = "<prompt syn-score level>"
        for settings in cmp.SETTINGS:
            t.edit(t.oracle_file("syn-different", settings), same)
        self.assert_gate_fails(t, "the differing fields are {}")

    def test_an_included_prompt_that_differs(self):
        t = self.frozen_tree()
        t.edit(t.oracle_file("syn-choice"),
               lambda d: d["upstream"]["prepare"]["passes"][0].update({"prompt": "<another prompt, same tokens>"}))
        self.assert_gate_fails(t, "evaluated different things: prompt bytes")

    def test_group_scores_that_are_not_the_maxima(self):
        t = self.frozen_tree()
        t.edit(t.oracle_file("syn-choice"),
               lambda d: d["upstream"]["evaluate"]["scores"][0].reverse())
        self.assert_gate_fails(t, "not the maxima of its raw scores")

    def test_wrong_usage(self):
        t = self.frozen_tree()
        t.edit(t.oracle_file("syn-score"),
               lambda d: d["upstream"]["finalize"]["usage"].update({"input_tokens": 99}))
        self.assert_gate_fails(t, "usage")

    def test_an_hf_observation_that_is_not_the_run_output(self):
        t = self.frozen_tree()
        t.edit(t.hf / "syn-score.json", lambda d: d.update({"source": "edited"}))
        self.assert_gate_fails(t, "is not the file the run wrote")

    def test_another_hf_numerics_tuple(self):
        t = self.frozen_tree()
        t.edit(t.hf / "run.json", lambda d: d["numerics"].update({"threads": 4}))
        self.assert_gate_fails(t, "numerics")

    def test_a_failed_parity_proof(self):
        t = self.frozen_tree()
        t.edit(t.hf / "run.json", lambda d: d["float32_parity"].update({"bit_identical": 9}))
        self.assert_gate_fails(t, "no passing float32 parity proof")

    def test_a_duplicate_key_in_an_envelope(self):
        t = self.frozen_tree()
        path = t.oracle_file("syn-score")
        text = path.read_text()
        path.write_text(text.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1))
        self.assertEqual(t.gate(), 1)

    def test_a_count_that_does_not_match(self):
        t = self.frozen_tree()
        t.edit(t.g1 / "allowlist.json",
               lambda d: d["corpora"]["representative"]["counts"].update({"included_questions": 5}))
        self.assert_gate_fails(t, "included_questions is 4, the allowlist declares 5")

    def test_a_case_missing_from_the_allowlist(self):
        t = self.frozen_tree()
        t.edit(t.g1 / "allowlist.json",
               lambda d: d["corpora"]["representative"]["included"].pop("syn-score"))
        self.assert_gate_fails(t, "syn-score is neither included nor excluded")

    def test_an_observation_without_a_case(self):
        t = self.frozen_tree()
        (t.cases / "syn-score.json").unlink()
        self.assert_gate_fails(t, "an observation without a case: syn-score")

    def test_an_answer_the_comparator_cannot_reproduce(self):
        t = self.frozen_tree()
        t.edit(t.oracle_file("syn-choice"),
               lambda d: d["upstream"]["finalize"]["answers"]["team"].update({"choice": "billing"}))
        self.assert_gate_fails(t, "does not reproduce its answer: choice")


# ---------------------------------------------------------------------------------------------------------------------
# The metric and the certificate
# ---------------------------------------------------------------------------------------------------------------------


class CertificateTest(unittest.TestCase):
    def test_a_choice_flip_within_the_margin_is_consistent_even_when_age_fails(self):
        # The review's counterexample: the flip is explained by its 0.001 margin, the third group fails AGE.
        ref = [f32(0.0), f32(-0.001), f32(-100.0)]
        cand = [f32(0.0), f32(0.001), f32(-99.0)]
        w = cmp.upstream_choice(ref)
        age = max(abs((cand[g] - cand[w]) - (ref[g] - ref[w])) for g in range(3))
        self.assertAlmostEqual(age, 1.0, places=5)
        cert = cmp.certificate("choice", ref, cand, 0.020, T)
        self.assertTrue(cert["consistent"])
        self.assertAlmostEqual(cert["reference_margin"], 0.001, places=6)

    def test_a_choice_flip_beyond_the_bound_is_inconsistent(self):
        ref, cand = [f32(1.0), f32(0.9)], [f32(0.9), f32(1.0)]
        self.assertFalse(cmp.certificate("choice", ref, cand, 0.020, T)["consistent"])

    def test_bound_equality_is_consistent(self):
        ref = [f32(0.5), f32(0.5 - 0.015625)]
        cand = [f32(0.5), f32(0.5)]
        cand[1] = f32(0.5 + 2 ** -20)
        self.assertTrue(cmp.certificate("choice", ref, cand, 0.015625, T)["consistent"])

    def test_every_separating_prefix_is_checked_with_its_direction(self):
        # The review's score counterexample: the first separating prefix (.505) is inside the band, the second
        # (.99) is not.
        ref = logits_for([0.49, 0.015, 0.485, 0.01])
        cand = logits_for([0.20, 0.20, 0.09, 0.51])
        p_r, p_c = cmp.upstream_probabilities(ref, T), cmp.upstream_probabilities(cand, T)
        self.assertEqual((cmp.upstream_median(p_r), cmp.upstream_median(p_c)), (1, 3))
        cert = cmp.certificate("score", ref, cand, 0.020, T)
        self.assertEqual([p["level"] for p in cert["separating_prefixes"]], [1, 2])
        self.assertEqual([p["direction"] for p in cert["separating_prefixes"]], ["down", "down"])
        self.assertEqual([p["consistent"] for p in cert["separating_prefixes"]], [True, False])
        self.assertFalse(cert["consistent"])

    def test_an_upward_median_change_inside_the_band_is_consistent(self):
        ref = logits_for([0.499, 0.002, 0.499])
        cand = logits_for([0.501, 0.002, 0.497])
        cert = cmp.certificate("score", ref, cand, 0.020, T)
        self.assertEqual([p["direction"] for p in cert["separating_prefixes"]], ["up"])
        self.assertTrue(cert["consistent"])

    def test_exact_ties_take_the_first_maximum(self):
        self.assertEqual(cmp.upstream_choice([0.25, 0.5, 0.5, 0.25]), 1)
        p = cmp.upstream_probabilities([f32(1.0), f32(1.0)], T)
        self.assertEqual(p, [0.5, 0.5])
        self.assertEqual(cmp.upstream_median(p), 0)

    def test_a_perturbation_within_the_bound_never_breaks_the_certificate(self):
        # Necessary condition: when AGE <= B, every changed answer is consistent.
        bound = 0.02
        cases = 0
        for seed in range(400):
            n = 2 + seed % 9
            base = [f32(((seed * 7919 + 31 * i * i) % 997) / 997.0 * 0.06 - 0.03) for i in range(n)]
            delta = [f32(b + bound * (((seed + i * 13) % 5) - 2) / 2.0) for i, b in enumerate(base)]
            w = cmp.upstream_choice(base)
            age = max(abs((delta[g] - delta[w]) - (base[g] - base[w])) for g in range(n))
            if age > bound:
                continue
            for kind in ("choice", "score"):
                p_r, p_c = cmp.upstream_probabilities(base, T), cmp.upstream_probabilities(delta, T)
                changed = (cmp.upstream_choice(p_r) != cmp.upstream_choice(p_c)) if kind == "choice" else \
                    (cmp.upstream_median(p_r) != cmp.upstream_median(p_c))
                if changed:
                    cases += 1
                    self.assertTrue(cmp.certificate(kind, base, delta, bound, T)["consistent"], (seed, kind))
        self.assertGreater(cases, 0, "the sweep exercised some changed answers")


class UpstreamArithmeticTest(unittest.TestCase):
    """compare.py's arithmetic against upstream's own answers on every P3a case it can replay."""

    def test_p3a_finalize_answers_are_reproduced(self):
        cases_dir = cmp.FIXTURES / "cases"
        replayed, exact = 0, 0
        for path in sorted(cases_dir.glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            up = case["upstream"]
            fin, ev = up.get("finalize"), up.get("evaluate")
            if case.get("has_media") or "metadata" in case and "score_bias" in json.dumps(case["metadata"]):
                continue
            if not isinstance(fin, dict) or "answers" not in fin or not isinstance(ev, dict) or "scores" not in ev:
                continue
            for answer_plan, scores in zip(up["prepare"]["answers"], ev["scores"]):
                values = [cmp.f32(b) for b in scores]
                if not all(math.isfinite(v) or v == -math.inf for v in values):
                    continue
                if all(v == -math.inf for v in values):
                    continue
                t = cmp.f32(answer_plan["temperature_f32"])
                keys = [g["key"] for g in answer_plan["groups"]]
                expected = fin["answers"][answer_plan["question"]]
                p = cmp.upstream_probabilities(values, t)
                if answer_plan["type"] == "choice":
                    self.assertEqual(expected["choice"], keys[cmp.upstream_choice(p)], case["id"])
                    got = [expected["probabilities"][k] for k in keys]
                elif answer_plan["type"] == "score":
                    self.assertEqual(expected["score"], float(cmp.upstream_median(p)), case["id"])
                    got = [expected["probabilities"][k] for k in keys]
                else:
                    got = [expected["noul"]]
                    p = [p[keys.index("true")]]
                for a, b in zip(got, p):
                    a = float(a) if not isinstance(a, str) else float(a.replace("Infinity", "inf"))
                    self.assertLessEqual(abs(a - b), cmp.SELF_CHECK_TOLERANCE, case["id"])
                    exact += a == b
                replayed += 1
        self.assertGreaterEqual(replayed, 80, "most of P3a's finalized questions were replayed")
        print(f"\n  replayed {replayed} finalized questions of P3a's cases; {exact} probabilities bit-exact")


if __name__ == "__main__":
    unittest.main()
