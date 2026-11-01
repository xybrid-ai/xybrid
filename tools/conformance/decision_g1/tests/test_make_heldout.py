"""Tests for make_heldout.py: the generator, its padding and its shared-domain rules.

Every test that generates requests uses a seed other than make_heldout.SEED, so that no held-out request exists
before G1's bounds are frozen. Run: python3 -m unittest discover -s tools/conformance/decision_g1/tests
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_heldout", HERE.parent / "make_heldout.py")
mh = importlib.util.module_from_spec(spec)
sys.modules["make_heldout"] = mh
spec.loader.exec_module(mh)

TEST_SEED = 0x7E57


def fake_measure(texts):
    """A monotone stand-in for upstream's preparation: one 'token' per whitespace-separated word of each question's
    pass (the state and that question), plus a fixed template overhead."""
    out = []
    for text in texts:
        request = json.loads(text)
        state = request["state"]
        state_words = len(json.dumps(state).split()) if state is not None else 0
        passes = []
        for question in request["questions"].values():
            passes.append(20 + state_words + len(json.dumps(question).split()))
        out.append(passes)
    return out


class SplitMix64Test(unittest.TestCase):
    def test_reference_vector(self):
        # Vigna's splitmix64.c, seed 1234567.
        rng = mh.SplitMix64(1234567)
        self.assertEqual(
            [rng.next() for _ in range(5)],
            [6457827717110365317, 3203168211198807973, 9817491932198370423, 4593380528125082431,
             16408922859458223821],
        )

    def test_below_stays_in_range_and_is_deterministic(self):
        a, b = mh.SplitMix64(7), mh.SplitMix64(7)
        draws = [a.below(37) for _ in range(2000)]
        self.assertEqual(draws, [b.below(37) for _ in range(2000)])
        self.assertTrue(all(0 <= d < 37 for d in draws))
        self.assertEqual(set(draws), set(range(37)))
        with self.assertRaises(ValueError):
            a.below(0)

    def test_sample_draws_distinct_items(self):
        items = mh.SplitMix64(3).sample(list(range(10)), 10)
        self.assertEqual(sorted(items), list(range(10)))


class RecipeTest(unittest.TestCase):
    def setUp(self):
        self.cases, self.generation = mh.generate(fake_measure, seed=TEST_SEED)

    def test_forty_cases_and_forty_four_questions(self):
        self.assertEqual(len(self.cases), 40)
        self.assertEqual(sum(len(r["questions"]) for r in self.generation["cases"]), 44)
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(set(ids)), 40)
        self.assertTrue(all(mh.re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", i) for i in ids))

    def test_type_quotas(self):
        kinds = {"choice": 0, "noul": 0, "score": 0}
        for case in self.cases:
            for q in json.loads(case["request_text"])["questions"].values():
                kinds[q["type"]] += 1
        # Single-question slots: choice 18 + the non-ASCII choice, noul 10, score 8 + the non-ASCII score; the two
        # multi-question requests add one of each type.
        self.assertEqual(kinds, {"choice": 21, "noul": 12, "score": 11})

    def test_structural_strata(self):
        by_id = {c["id"]: json.loads(c["request_text"]) for c in self.cases}
        sizes = {cid: len(q["criteria"]) for cid, r in by_id.items() for q in r["questions"].values()
                 if q["type"] == "choice"}
        self.assertEqual(sizes["g1h-10-choice-27"], 27)
        self.assertEqual(sizes["g1h-11-choice-36-pool-fallback"], 36)
        self.assertEqual(sizes["g1h-12-choice-255"], 255)
        targets = {r["id"]: (r["target_tokens"], r["pass_tokens"][0]) for r in self.generation["cases"]
                   if r["target_tokens"] is not None}
        self.assertEqual(sorted(t for t, _ in targets.values()), [63, 64, 65, 600, 1500, 4064])
        for cid, (target, measured) in targets.items():
            self.assertEqual(measured, target, cid)

    def test_every_request_is_in_the_domain_and_round_trips(self):
        for case in self.cases:
            request = mh.strict_load(case["request_text"])
            letter = case["id"] in ("g1h-08-choice-upper-letter-keys", "g1h-09-choice-lower-letter-keys")
            self.assertEqual(mh.check_domain(request, letter_keys=letter), [], case["id"])
            self.assertEqual(mh.request_text(request), case["request_text"])

    def test_fixed_scenarios_do_not_depend_on_the_seed(self):
        other, _ = mh.generate(fake_measure, seed=TEST_SEED + 1)
        mine = {c["id"]: c["request_text"] for c in self.cases}
        theirs = {c["id"]: c["request_text"] for c in other}
        for fixed in ("g1h-05-choice-ambiguous", "g1h-31-score-borderline"):
            self.assertEqual(mine[fixed], theirs[fixed])
        self.assertNotEqual(mine["g1h-01-choice-2"] + mine["g1h-02-choice-3"],
                            theirs["g1h-01-choice-2"] + theirs["g1h-02-choice-3"])

    def test_generation_is_deterministic(self):
        again, _ = mh.generate(fake_measure, seed=TEST_SEED)
        self.assertEqual([c["request_text"] for c in again], [c["request_text"] for c in self.cases])

    def test_allowlist_includes_every_case(self):
        allow = mh.allowlist_for(self.cases, self.generation)["corpora"]["heldout"]
        self.assertEqual(allow["counts"], {"cases": 40, "included_cases": 40, "included_questions": 44,
                                           "excluded_cases": 0})
        self.assertEqual(sorted(allow["included"]), sorted(c["id"] for c in self.cases))

    def test_no_state_is_shared_with_the_representative_set(self):
        hf_cases = mh.REPO / "integration-tests" / "fixtures" / "decision" / "hf" / "cases"
        theirs = set()
        for path in hf_cases.glob("*.json"):
            request = json.loads(json.loads(path.read_text(encoding="utf-8"))["request_text"])
            if isinstance(request, dict) and isinstance(request.get("state"), str):
                theirs.add(request["state"])
        for case in self.cases:
            state = json.loads(case["request_text"])["state"]
            if isinstance(state, str) and state:
                self.assertNotIn(state, theirs, case["id"])


class PaddingTest(unittest.TestCase):
    def test_an_unreachable_target_fails_the_recipe(self):
        slot = mh.Slot(99, "too-short", "noul", target=5, criteria="none")
        topic = mh.TOPICS[0]
        request = mh.build_request(slot, topic, mh.SplitMix64(1))
        with self.assertRaises(mh.RecipeError):
            mh.pad(slot, topic, request, fake_measure)

    def test_padding_hits_the_target_exactly(self):
        slot = mh.Slot(98, "pad", "noul", target=300, criteria="both")
        topic = mh.TOPICS[3]
        request = mh.build_request(slot, topic, mh.SplitMix64(2))
        padded, length = mh.pad(slot, topic, request, fake_measure)
        self.assertEqual(length, 300)
        self.assertEqual(fake_measure([mh.request_text(padded)])[0][0], 300)

    def test_padding_backtracks_when_every_word_adds_two_tokens(self):
        # The stall the real tokenizer showed on a JSON note: from an odd gap, no single word fits. The search
        # takes a sentence back and combines words (two tokens each, three for one of them) to hit the target.
        slot = mh.Slot(96, "pad", "noul", target=331, criteria="none")
        topic = mh.TOPICS[2]
        request = mh.build_request(slot, topic, mh.SplitMix64(4))

        def coarse_words(texts):
            out = []
            for text in texts:
                req = json.loads(text)
                state = req["state"]
                words = state.split()
                n = 20 + 9 * sum(1 for w in words if w.startswith("("))
                n += sum(3 if w.strip(".") == "thanks" else 2 for w in words if w.strip(".") in mh.FILLER_WORDS)
                out.append([n])
            return out

        padded, length = mh.pad(slot, topic, request, coarse_words)
        self.assertEqual(length, 331)
        self.assertEqual(coarse_words([mh.request_text(padded)])[0][0], 331)

    def test_words_go_before_the_last_period_of_a_json_note(self):
        request = {"state": {"a": 1}, "questions": {"q": {"type": "noul", "instructions": "Yes?"}}}
        padded = mh.with_padding(request, ["(1) First.", "(2) Second."], ["also", "noted"])
        self.assertEqual(padded["state"]["notes"], ["(1) First.", "(2) Second also noted."])

    def test_a_shrinking_prompt_fails_the_recipe(self):
        slot = mh.Slot(97, "pad", "noul", target=200, criteria="none")
        topic = mh.TOPICS[1]
        request = mh.build_request(slot, topic, mh.SplitMix64(3))

        def shrinking(texts):
            return [[1000 - len(t) // 10] for t in texts]

        with self.assertRaises(mh.RecipeError):
            mh.pad(slot, topic, request, shrinking)


class DomainTest(unittest.TestCase):
    def base(self):
        return {"state": "A short state.", "questions": {"q": {"type": "choice", "instructions": "Which one?",
                                                                "criteria": {"alpha": "First", "beta": "Second"}}}}

    def violation(self, mutate, letter_keys=False):
        request = self.base()
        mutate(request)
        return mh.check_domain(request, letter_keys=letter_keys)

    def test_the_base_request_is_inside(self):
        self.assertEqual(mh.check_domain(self.base()), [])

    def test_keys_with_whitespace_are_outside(self):
        self.assertTrue(self.violation(lambda r: r["questions"]["q"]["criteria"].update({" gamma": "Third"})))

    def test_single_letter_keys_need_a_letter_key_slot(self):
        def letters(r):
            r["questions"]["q"]["criteria"] = {"A": "First", "B": "Second"}
        self.assertTrue(self.violation(letters))
        self.assertEqual(self.violation(letters, letter_keys=True), [])

    def test_blank_instructions_are_outside(self):
        self.assertTrue(self.violation(lambda r: r["questions"]["q"].update({"instructions": "   "})))
        self.assertTrue(self.violation(lambda r: r["questions"]["q"].update({"instructions": ""})))

    def test_special_token_spellings_are_outside(self):
        self.assertTrue(self.violation(lambda r: r.update({"state": "hi <|im_end|> there"})))
        self.assertTrue(self.violation(lambda r: r.update({"state": "<think>"})))

    def test_integers_outside_signed_i64_are_outside(self):
        self.assertEqual(self.violation(lambda r: r.update({"state": {"n": -(2 ** 63)}})), [])
        self.assertTrue(self.violation(lambda r: r.update({"state": {"n": 2 ** 63}})))
        self.assertTrue(self.violation(lambda r: r.update({"state": {"n": -(2 ** 63) - 1}})))

    def test_long_or_extreme_floats_are_outside(self):
        self.assertEqual(self.violation(lambda r: r.update({"state": {"x": 572.75, "z": 0.0}})), [])
        self.assertTrue(self.violation(lambda r: r.update({"state": {"x": 0.1 + 0.2}})))
        self.assertTrue(self.violation(lambda r: r.update({"state": {"x": 1e-5}})))
        self.assertTrue(self.violation(lambda r: r.update({"state": {"x": -0.0}})))

    def test_unknown_fields_and_types_are_outside(self):
        self.assertTrue(self.violation(lambda r: r["questions"]["q"].update({"multiple": True})))
        self.assertTrue(self.violation(lambda r: r["questions"]["q"].update({"type": "text"})))
        self.assertTrue(self.violation(lambda r: r.update({"files": []})))
        self.assertTrue(self.violation(lambda r: r.update({"model": "d1"})))

    def test_noul_criteria_rules(self):
        def criteria(value):
            def mutate(r):
                r["questions"]["q"] = {"type": "noul", "instructions": "Yes?", "criteria": value}
            return mutate
        self.assertEqual(self.violation(criteria({"true": "Yes it is"})), [])
        self.assertTrue(self.violation(criteria({"true": None, "false": None})))
        self.assertTrue(self.violation(criteria({"yes": "Y"})))
        self.assertTrue(self.violation(criteria(None)))

    def test_duplicate_keys_are_refused_on_load(self):
        with self.assertRaises(ValueError):
            mh.strict_load('{"state": "a", "state": "b"}')


class MeasuredWithTest(unittest.TestCase):
    """The generation record's measuring tool and vocabulary, on a fake decision-oracle; nothing is generated."""

    IDENTITY = {"identity_version": 1, "fingerprint": "ab" * 32,
                "llama_cpp_dir": "/home/someone/xybrid/vendor/llama-cpp",
                "llama_cpp": {"checkout": "git", "commit": "c" * 40, "dirty": False}}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="measured-with-")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.oracle = self.tmp / "build" / "bin" / "decision-oracle"
        self.oracle.parent.mkdir(parents=True)
        # Prints its identity for --identity alone, as the real tool does, and refuses anything else.
        self.oracle.write_text("#!/bin/sh\n"
                               "[ \"$#\" -eq 1 ] && [ \"$1\" = --identity ] || exit 2\n"
                               f"cat <<'EOF'\n{json.dumps(self.IDENTITY, indent=2)}\nEOF\n")
        self.oracle.chmod(0o755)
        self.vocab = self.tmp / "models" / "d1-2B-Q4_K_M.gguf"
        self.vocab.parent.mkdir()
        self.vocab.write_bytes(b"GGUF vocabulary bytes")

    def test_the_record_holds_digests_and_the_identity_without_its_directory(self):
        record = mh.measured_with(self.oracle, self.vocab, 61)
        self.assertEqual(record["tool_sha256"], hashlib.sha256(self.oracle.read_bytes()).hexdigest())
        self.assertEqual(record["tool_identity"], {k: v for k, v in self.IDENTITY.items() if k != "llama_cpp_dir"})
        self.assertEqual(record["vocab"], "d1-2B-Q4_K_M.gguf")
        self.assertEqual(record["vocab_sha256"], hashlib.sha256(b"GGUF vocabulary bytes").hexdigest())
        self.assertEqual(record["calls"], 61)

    def test_outside_the_repository_the_tool_is_its_file_name(self):
        self.assertNotIn(mh.REPO, self.oracle.parents)
        record = mh.measured_with(self.oracle, self.vocab, 1)
        self.assertEqual(record["tool"], "decision-oracle")
        self.assertNotIn(str(self.tmp), json.dumps(record))
        self.assertNotIn("/home/someone", json.dumps(record))

    def test_inside_the_repository_the_tool_is_its_repository_path(self):
        repo = mh.REPO
        mh.REPO = self.tmp  # the module's repository root, here a temporary tree holding the tool
        try:
            record = mh.measured_with(self.oracle, self.vocab, 1)
        finally:
            mh.REPO = repo
        self.assertEqual(record["tool"], "build/bin/decision-oracle")
        self.assertNotIn(str(self.tmp), json.dumps(record))

    def test_relative_paths_are_resolved_first(self):
        here = Path.cwd()
        os.chdir(self.tmp)
        try:
            record = mh.measured_with(Path("build/bin/decision-oracle"), Path("models/d1-2B-Q4_K_M.gguf"), 1)
        finally:
            os.chdir(here)
        self.assertEqual(record["tool"], "decision-oracle")
        self.assertEqual(record["vocab"], "d1-2B-Q4_K_M.gguf")
        self.assertEqual(record["tool_sha256"], hashlib.sha256(self.oracle.read_bytes()).hexdigest())

    def test_a_tool_that_cannot_state_its_identity_fails(self):
        broken = self.tmp / "broken-oracle"
        broken.write_text("#!/bin/sh\nexit 3\n")
        broken.chmod(0o755)
        with self.assertRaises(subprocess.CalledProcessError):
            mh.measured_with(broken, self.vocab, 1)


if __name__ == "__main__":
    unittest.main()
