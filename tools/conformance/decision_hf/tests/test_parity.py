"""Unit tests for run_reference.py's float32 parity gate (the pinned environment's torch; no weights).

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


if __name__ == "__main__":
    unittest.main()
