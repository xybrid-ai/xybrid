"""Tests for compare.py: the metric, the certificate, the freeze, and every integrity rule as a mutation test.

A synthetic corpus is written to a temporary tree in the layouts compare.py reads (case files, HF observations with
their run log, oracle collections with their collection.json), with real pins, lock and recipe digests so that the
identity checks run for real. Each mutation test breaks one thing and checks that the gate fails for that reason.

Run: python3 -m unittest discover -s tools/conformance/decision_g1/tests
"""

import copy
import importlib.util
import json
import math
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
        for settings in cmp.SETTINGS:
            self.write_oracle(settings, 0.0 if equal else perturb)
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

    def write_oracle(self, settings, perturb):
        directory = self.oracle / cmp.F16 / settings
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
            doc = {"schema_version": 1, "id": case_id, "request_sha256": sha, "source_file": f"{case_id}.json",
                   "artifact": cmp.F16, "settings": ran, "upstream": up}
            name = f"{case_id}.{sha}.json"
            self.write(directory / name, doc)
            entries.append({"file": f"{case_id}.json", "id": case_id, "request_sha256": sha, "status": "observed",
                            "observation": name})
        art = json.loads((self.fixtures / "artifacts.json").read_text())["artifacts"][cmp.F16]
        st = json.loads((self.fixtures / "artifacts.json").read_text())["settings"][settings]
        record = {
            "schema_version": 1,
            "cases_dir": str(self.cases),
            "artifact": {"id": cmp.F16, "file": art["file"], "bytes": art["bytes"], "sha256": art["sha256"]},
            "collector": {"official": True, "problems": []},
            "build": {"fingerprint": "f" * 64, "llama_cpp_dir": "/x",
                      "llama_cpp": {"checkout": "git", "commit": cmp.pinned_llama_cpp(), "dirty": False},
                      "build": {"build_type": "Release", "backends": ["cpu"], "cxx_compiler": "GNU 11.5.0"}},
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
        """A second corpus shaped like the first, as the held-out set is."""
        allow = json.loads((t.g1 / "allowlist.json").read_text())
        spec = copy.deepcopy(allow["corpora"]["representative"])
        heldout = t.g1 / "heldout"
        for name in ("cases", "hf", "oracle"):
            src = t.tmp / spec[name]
            shutil.copytree(src, heldout / name)
            spec[name] = str((heldout / name).relative_to(t.tmp))
        record = heldout / "oracle" / cmp.F16
        for settings in cmp.SETTINGS:
            t.edit(record / settings / "collection.json",
                   lambda d: d.update({"cases_dir": str(heldout / "cases")}))
        t.write(heldout / "allowlist.json", {"schema_version": 1, "corpora": {"heldout": spec}})

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
