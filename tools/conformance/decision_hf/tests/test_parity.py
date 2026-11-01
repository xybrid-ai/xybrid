"""Unit tests for run_reference.py's float32 parity gate and its failure handling (the pinned environment's torch; no
weights).

`ProductionPathTests` run the production `main`, `Reference.instrument`, `run_case`, `call`, `Float32Compute` and the
immediate `ParityCheck` around a tiny stand-in for the reference (`TinyD1`: 27 hooked modules, as the real one).

Run with `tools/conformance/decision_hf/run.sh tests`.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_reference  # noqa: E402
from run_reference import Float32Compute, ParityCheck, RunnerError  # noqa: E402


class Pair(torch.nn.Module):
    """Two chained modules stored in bfloat16, as the reference's layers are."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first = torch.nn.Linear(16, 16)
        self.second = torch.nn.Linear(16, 8)
        self.to(torch.bfloat16)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))


class ParityGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = Pair()
        self.hooked = [self.model.first, self.model.second]
        Float32Compute(torch, self.hooked)  # registered first, as in run_reference.Reference
        self.parity = ParityCheck(torch, self.hooked)

    def forward(self):
        x = torch.randn(4, 16, generator=torch.Generator().manual_seed(1))
        with torch.inference_mode():
            return self.model(x)

    def replace(self, index, args=None, output=None):
        module, recorded_args, kwargs, recorded_output = self.parity.records[index]
        self.parity.records[index] = (
            module,
            recorded_args if args is None else args,
            kwargs,
            recorded_output if output is None else output,
        )

    def test_widened_forward_passes_with_full_coverage(self):
        out = self.forward()
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in self.model.parameters()))
        self.parity.verify()
        self.assertEqual(
            self.parity.require_coverage(),
            {"calls": 2, "bit_identical": 2, "non_float32_inputs": 0, "modules_hooked": 2, "modules_verified": 2},
        )

    def test_output_plus_one_fails(self):
        # the review's reproduction: a recorded output that no longer matches its replay
        self.forward()
        self.replace(0, output=self.parity.records[0][3] + 1)
        with self.assertRaisesRegex(RunnerError, "bytes differ"):
            self.parity.verify()

    def test_one_flipped_bit_fails(self):
        self.forward()
        bits = self.parity.records[1][3].clone().view(torch.int32)
        bits.view(-1)[0] ^= 1  # the lowest mantissa bit: within any float tolerance
        self.replace(1, output=bits.view(torch.float32))
        with self.assertRaisesRegex(RunnerError, "1 of 128 bytes differ"):
            self.parity.verify()

    def test_extra_output_fails(self):
        self.forward()
        output = self.parity.records[0][3]
        self.replace(0, output=(output, output))
        with self.assertRaisesRegex(RunnerError, "tuple recorded, Tensor replayed"):
            self.parity.verify()

    def test_changed_shape_fails(self):
        self.forward()
        self.replace(0, output=self.parity.records[0][3][:2])
        with self.assertRaisesRegex(RunnerError, r"\(2, 16\) recorded"):
            self.parity.verify()

    def test_bfloat16_input_fails(self):
        self.forward()
        x = self.parity.records[0][1][0]
        self.replace(0, args=(x.to(torch.bfloat16),))
        with self.assertRaisesRegex(RunnerError, "bfloat16, not float32"):
            self.parity.verify()

    def test_empty_proof_fails(self):
        with self.assertRaisesRegex(RunnerError, "no module call"):
            self.parity.require_coverage()

    def test_module_never_verified_fails(self):
        with torch.inference_mode():
            self.model.first(torch.randn(2, 16))
        self.parity.verify()
        with self.assertRaisesRegex(RunnerError, "1 of 2 hooked modules never ran"):
            self.parity.require_coverage()

    def test_unverified_records_fail(self):
        self.forward()
        with self.assertRaisesRegex(RunnerError, "never verified"):
            self.parity.require_coverage()

    def test_discarded_forward_proves_nothing(self):
        self.forward()
        self.parity.discard()
        self.parity.verify()
        with self.assertRaisesRegex(RunnerError, "no module call"):
            self.parity.require_coverage()

    def test_incompatible_structure_is_a_mismatch(self):
        # the review's second path: formatting len(None) raised TypeError, which was no RunnerError
        with self.assertRaisesRegex(RunnerError, "tuple recorded, NoneType replayed"):
            self.parity.compare((1,), None, "Linear output")

    def test_value_it_cannot_compare_fails(self):
        with self.assertRaisesRegex(RunnerError, "cannot compare"):
            self.parity.compare(object(), object(), "Linear output")

    def test_input_it_cannot_read_fails(self):
        self.forward()
        self.replace(0, args=(self.parity.records[0][1][0], object()))
        with self.assertRaisesRegex(RunnerError, r"Linear args\[1\]: a object value, which this check cannot read"):
            self.parity.verify()

    def test_failure_while_comparing_is_a_runner_error_that_stands(self):
        self.forward()
        with mock.patch.object(self.parity, "bits", side_effect=MemoryError("no room for the byte view")):
            with self.assertRaisesRegex(RunnerError, "MemoryError"):
                self.parity.verify()
        self.forward()  # a later forward proves nothing: the first failure stands
        with self.assertRaisesRegex(RunnerError, "an earlier failure stands: .*MemoryError"):
            self.parity.verify()
        with self.assertRaisesRegex(RunnerError, "an earlier failure stands"):
            self.parity.require_coverage()


class ImmediateParityTests(unittest.TestCase):
    """The mode the runner uses: each call verified as its module returns."""

    def setUp(self):
        torch.set_num_threads(1)
        self.model = Pair()
        self.hooked = [self.model.first, self.model.second]
        Float32Compute(torch, self.hooked)

    def forward(self):
        with torch.inference_mode():
            return self.model(torch.randn(4, 16, generator=torch.Generator().manual_seed(1)))

    def test_widened_forward_passes_holding_nothing(self):
        parity = ParityCheck(torch, self.hooked, immediate=True)
        self.forward()
        self.assertEqual(parity.records, [])
        self.assertEqual(parity.require_coverage()["calls"], 2)

    def test_changed_output_stops_the_forward(self):
        # a hook between the widening and the proof changes what the module returned
        self.model.second.register_forward_hook(lambda module, args, output: output + 1)
        ParityCheck(torch, self.hooked, immediate=True)
        with self.assertRaisesRegex(RunnerError, "Linear output: .* bytes differ"):
            self.forward()

    def test_failing_replay_is_a_runner_error(self):
        parity = ParityCheck(torch, self.hooked, immediate=True)
        with mock.patch.object(parity, "float32_copy", side_effect=MemoryError("no room")):
            with self.assertRaisesRegex(RunnerError, "replaying Linear failed"):
                self.forward()

    def test_failure_inside_the_comparison_leaves_the_hook_as_a_runner_error(self):
        # the review's reproduction: an ordinary exception in the immediate comparison left the forward as itself
        parity = ParityCheck(torch, self.hooked, immediate=True)
        with mock.patch.object(parity, "compare", side_effect=MemoryError("comparison failed")):
            with self.assertRaises(RunnerError) as raised:
                self.forward()
        self.assertIsInstance(raised.exception.__cause__, MemoryError)

    def test_failure_the_model_swallows_still_fails_the_proof(self):
        parity = ParityCheck(torch, self.hooked, immediate=True)

        def swallowing(x):
            try:
                return self.model.second(torch.relu(self.model.first(x)))
            except Exception:  # a model that catches what its submodules raise
                return torch.zeros(4, 8)

        with mock.patch.object(parity, "compare", side_effect=MemoryError("comparison failed")):
            with torch.inference_mode():
                swallowing(torch.randn(4, 16))
        with self.assertRaisesRegex(RunnerError, "an earlier failure stands"):
            parity.require_coverage()


class FailingProof:
    def __init__(self, fail_at_end):
        self.fail_at_end = fail_at_end

    def require_coverage(self):
        if self.fail_at_end:
            raise RunnerError("float32 parity: 1 of 27 hooked modules never ran")
        return {"calls": 1, "bit_identical": 1, "non_float32_inputs": 0, "modules_hooked": 1, "modules_verified": 1}


class StubReference:
    """Stands in for the model: answers every case but `bad`, whose proof fails mid-run."""

    fail_at_end = False

    def __init__(self, args, pins):
        self.parity = FailingProof(self.fail_at_end)
        self.faults = run_reference.Faults()
        self.load_seconds = self.peak_anon_gib = 0.0
        self.code, self.versions, self.torch = [], {}, torch

    def note_memory(self):
        pass

    def run_case(self, case):
        if case["id"] == "bad":
            raise RunnerError("float32 parity: LlamaDecoderLayer output: 3 of 8192 bytes differ")
        return {"case": case["id"], "request_text_sha256": "0" * 64, "hf": {}, "detail": {}}


class MainWritesNothingOnFailureTests(unittest.TestCase):
    def run_main(self, case_ids, fail_at_end):
        temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp, ignore_errors=True)
        for case_id in case_ids:
            (temp / f"{case_id}.json").write_text(json.dumps({"id": case_id, "request_text": "{}"}))
        out, log = temp / "out", temp / "run.json"
        stub = type("Stub", (StubReference,), {"fail_at_end": fail_at_end})
        with mock.patch.object(run_reference, "Reference", stub), mock.patch.object(
            run_reference, "verify_snapshot", lambda *a, **k: []
        ), mock.patch.dict(os.environ, {}), mock.patch("sys.stderr"), mock.patch("sys.stdout"):
            code = run_reference.main(
                [str(temp / f"{c}.json") for c in case_ids]
                + ["--out", str(out), "--run-log", str(log), "--check-f32-parity", "--cache", str(temp)]
            )
        return code, out, log

    def test_failure_mid_run_writes_nothing(self):
        code, out, log = self.run_main(["good", "bad"], fail_at_end=False)
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())
        self.assertFalse(log.exists())

    def test_failed_coverage_writes_nothing(self):
        code, out, log = self.run_main(["good"], fail_at_end=True)
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())
        self.assertFalse(log.exists())

    def test_passing_run_writes_observations_and_their_digests(self):
        code, out, log = self.run_main(["good"], fail_at_end=False)
        self.assertEqual(code, 0)
        text = (out / "good.json").read_text()
        recorded = json.loads(log.read_text())
        self.assertEqual(recorded["outputs"], {"good": run_reference.sha256_bytes(text.encode("utf-8"))})
        self.assertEqual(recorded["inputs"]["recipe"], run_reference.recipe(run_reference.RECIPE))
        self.assertTrue(recorded["numerics"]["float32_parity_gate"])


# --------------------------------------------------------------------------- #
# the production path, around a tiny stand-in for the reference
# --------------------------------------------------------------------------- #

VOCAB, WIDTH, LAYERS = 16, 8, 24  # embeddings, 24 decoder layers, final norm, LM head: 27 hooked modules


class TinyText(torch.nn.Module):
    """The text model as `Reference.instrument` reads it: `embed_tokens`, `layers`, `embedding_norm`."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, WIDTH)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(WIDTH, WIDTH) for _ in range(LAYERS))
        self.embedding_norm = torch.nn.LayerNorm(WIDTH)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return self.embedding_norm(hidden)


class TinyBody(torch.nn.Module):
    """The text backbone, whose input ids `Reference.instrument` records."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def forward(self, input_ids):
        return self.text(input_ids)


class TinyQuestion:
    def __init__(self, raw):
        self.type = raw.get("type", "choice")
        self.instructions = raw["instructions"]
        self.criteria = {"a": "first", "b": "second"}


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % (VOCAB - 3) for c in text]

    def convert_ids_to_tokens(self, ids):
        return [f"<{i}>" for i in ids]


class TinyPrompt:
    """The snapshot's `prompt` module, as far as the runner calls it."""

    @staticmethod
    def is_media(instructions):
        return False

    @staticmethod
    def readout_ids(tokenizer, question):
        return [[1], [2]]

    @staticmethod
    def aliases(tokenizer, criteria):
        return [("A", None), ("B", None)]


class TinyEngine:
    """SystemOne's question checks, prompt building and readout, as far as the runner calls them."""

    set_size = 16
    calibration = None

    def __init__(self):
        self.tokenizer = TinyTokenizer()

    def ask(self, raw):
        if raw.get("refuse"):
            raise ValueError("the reference's own check refuses this question")
        if raw.get("exhaust"):
            raise MemoryError("the reference ran out of memory before its model ran")
        return TinyQuestion(raw)

    def _prefix(self, state, others):
        return f"S:{state}|", None

    def _suffixes(self, questions):
        return [f"Q:{q.instructions}" for q in questions], None, None

    def _sequences(self, rows, set_size=None):
        out = []
        for state, questions, _ in rows:
            ids = self.tokenizer.encode(self._prefix(state, [])[0] + "".join(self._suffixes(questions)[0]))
            out.append((None, None, ids, 2, None, [[len(ids) - 1]], None))
        return out

    def _readouts(self, qs, logz, tempered):
        return [torch.softmax(logz[i, [1, 2]].double(), -1).tolist() for i in range(len(qs))]


class TinyD1(torch.nn.Module):
    """`system_one` as the reference shapes it: its own checks, then one forward and its readout.

    `then(call)` may change what call number `call` does: return "fail" to raise ValueError once the model began,
    "swallow" to catch whatever the forward raises and answer anyway, or "refuse" for a typed refusal before it."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.text = TinyText()
        self.body = TinyBody(self.text)
        self.head = torch.nn.Linear(WIDTH, VOCAB, bias=False)
        self.to(torch.bfloat16)
        self.engine = TinyEngine()
        self.calls = 0
        self.then = lambda call: None

    def system_one(self, state, questions, set_size=None):
        self.calls += 1
        behaviour = self.then(self.calls)
        engine = self.engine
        qs = [engine.ask(raw) for raw in questions.values()]
        if behaviour == "refuse":
            raise ValueError("the reference refuses this typed question")
        ((_, _, ids, _, _, ends, _),) = engine._sequences([(state, qs, [])], set_size=set_size)
        try:
            with torch.inference_mode():
                hidden = self.body(input_ids=torch.tensor([ids]))
                if behaviour == "fail":
                    raise ValueError("a failure once the model ran")
                logits = self.head(hidden[0, [end[0] for end in ends]])
        except Exception:
            if behaviour != "swallow":
                raise
            logits = torch.zeros(len(qs), VOCAB)
        probs = engine._readouts(qs, torch.log_softmax(logits.float(), -1), True)
        answers = {n: {"answer": "a" if p[0] >= p[1] else "b", "confidence": max(p)} for n, p in zip(questions, probs)}
        return {"answers": answers, "usage": {"input_tokens": len(ids), "output_tokens": 0}}


class TinyReference(run_reference.Reference):
    """The production Reference (instrument, call, parse, prepare_one, evaluate_one, run_case) around `TinyD1`."""

    last = None

    def __init__(self, args, pins):
        self.torch, self.args, self.pins = torch, args, pins
        self.peak_anon_gib = self.load_seconds = 0.0
        self.code, self.versions = [], {}
        self.model = TinyD1()
        self.engine = self.model.engine
        self.tokenizer = self.engine.tokenizer
        self.modules = {"prompt": TinyPrompt}
        self.instrument(self.model.body, self.model.text, self.model.head)
        TinyReference.last = self


def request(**question):
    return json.dumps({"state": "s", "questions": {"q": {"instructions": "pick one", **question}}})


class ProductionPathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)

    def run_main(self, cases, then=lambda call: None):
        for case_id, text in cases.items():
            (self.temp / f"{case_id}.json").write_text(json.dumps({"id": case_id, "request_text": text}))
        out, log = self.temp / "out", self.temp / "run.json"

        def reference(args, pins):
            made = TinyReference(args, pins)
            made.model.then = then
            return made

        with mock.patch.object(run_reference, "Reference", reference), mock.patch.object(
            run_reference, "verify_snapshot", lambda *a, **k: []
        ), mock.patch.dict(os.environ, {}), mock.patch("sys.stderr"), mock.patch("sys.stdout"):
            code = run_reference.main(
                [str(self.temp / f"{case_id}.json") for case_id in cases]
                + ["--out", str(out), "--run-log", str(log), "--check-f32-parity", "--cache", str(self.temp)]
            )
        return code, out, log

    def assertWroteNothing(self, code, out, log):
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())
        self.assertFalse(log.exists())

    def test_two_cases_pass_with_every_module_verified(self):
        code, out, log = self.run_main({"first": request(), "second": request(instructions="another")})
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["first.json", "second.json"])
        proof = json.loads(log.read_text())["float32_parity"]
        self.assertEqual(proof, {"calls": 54, "bit_identical": 54, "non_float32_inputs": 0, "modules_hooked": 27,
                                 "modules_verified": 27})

    def test_failure_in_the_second_cases_comparison_writes_nothing(self):
        # the review's regression: the first case verifies all 27 modules, then the comparison itself fails
        real = ParityCheck.compare

        def compare(parity, recorded, again, where):
            if TinyReference.last.model.calls >= 2:
                raise MemoryError("comparison failed")
            return real(parity, recorded, again, where)

        with mock.patch.object(ParityCheck, "compare", compare):
            self.assertWroteNothing(*self.run_main({"good": request(), "bad": request(instructions="another")}))

    def test_failure_in_any_hook_of_the_second_case_writes_nothing(self):
        for owner, name in ((Float32Compute, "widen"), (Float32Compute, "restore"), (ParityCheck, "before"),
                            (ParityCheck, "bits"), (ParityCheck, "require_float32")):
            real = getattr(owner, name)

            def failing(*args, real=real, **kwargs):
                if TinyReference.last.model.calls >= 2:
                    raise MemoryError("a hook failed")
                return real(*args, **kwargs)

            with self.subTest(hook=f"{owner.__name__}.{name}"), mock.patch.object(owner, name, failing):
                shutil.rmtree(self.temp / "out", ignore_errors=True)
                self.assertWroteNothing(*self.run_main({"good": request(), "bad": request(instructions="another")}))

    def test_what_the_model_raises_once_it_ran_writes_nothing(self):
        # a ValueError, the type of the reference's own refusals: still no refusal once the model ran
        self.assertWroteNothing(*self.run_main(
            {"good": request(), "bad": request(instructions="another")}, then=lambda call: "fail" if call == 2 else None
        ))

    def test_hook_failure_the_reference_swallows_writes_nothing(self):
        real = Float32Compute.restore

        def restore(*args, **kwargs):
            if TinyReference.last.model.calls >= 2:
                raise MemoryError("a hook failed")
            return real(*args, **kwargs)

        with mock.patch.object(Float32Compute, "restore", restore):
            self.assertWroteNothing(*self.run_main(
                {"good": request(), "bad": request(instructions="another")},
                then=lambda call: "swallow" if call == 2 else None,
            ))

    def test_refusals_before_the_model_runs_are_recorded(self):
        code, out, log = self.run_main(
            {"good": request(), "checked": request(refuse=True), "typed": request(type="text")},
            then=lambda call: "refuse" if call == 2 else None,
        )
        self.assertEqual(code, 0)
        checked = json.loads((out / "checked.json").read_text())
        self.assertEqual(checked["hf"]["parse"]["refusal"]["exception"], "ValueError")
        typed = json.loads((out / "typed.json").read_text())
        self.assertEqual(typed["detail"]["typed"]["refusal"]["exception"], "ValueError")

    def test_a_limit_before_the_model_runs_is_no_refusal(self):
        self.assertWroteNothing(*self.run_main({"good": request(), "exhausted": request(exhaust=True)}))


if __name__ == "__main__":
    unittest.main()
