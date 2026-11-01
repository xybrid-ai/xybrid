"""Unit tests for provenance.py's refusals (standard library only).

Run with `python3 -B -m unittest discover -s tools/conformance/decision_hf/tests -p test_provenance.py`, or with the
rest through `tools/conformance/decision_hf/run.sh tests`.
"""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import provenance  # noqa: E402
from provenance import ProvenanceError  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Temp(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)

    def write(self, name: str, text: str) -> Path:
        path = self.temp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class InputsTests(Temp):
    def setUp(self):
        super().setUp()
        self.write("run_reference.py", "print('a')\n")
        self.write("run.sh", "exec true\n")
        self.sources = ("run_reference.py", "run.sh")
        self.current = {"pins_sha256": "p" * 64, "requirements_sha256": "r" * 64, "snapshot": {"a.py": "1" * 64}}
        self.recorded = {
            "pins_sha256": "p" * 64,
            "requirements_sha256": "r" * 64,
            "snapshot_files": [{"file": "a.py", "sha256": "1" * 64}],
            "recipe": provenance.recipe(self.sources, self.temp),
        }

    def check(self, recorded):
        provenance.check_inputs("observations", recorded, self.current, self.sources, base=self.temp)

    def test_matching_inputs_pass(self):
        self.check(self.recorded)

    def test_changed_lock_refuses(self):
        with self.assertRaisesRegex(ProvenanceError, "requirements.txt differs"):
            self.check({**self.recorded, "requirements_sha256": "x" * 64})

    def test_changed_pins_refuse(self):
        with self.assertRaisesRegex(ProvenanceError, "pins.json differs"):
            self.check({**self.recorded, "pins_sha256": "x" * 64})

    def test_other_snapshot_files_refuse(self):
        with self.assertRaisesRegex(ProvenanceError, "snapshot files"):
            self.check({**self.recorded, "snapshot_files": [{"file": "a.py", "sha256": "2" * 64}]})

    def test_changed_recipe_source_refuses(self):
        self.write("run_reference.py", "print('b')\n")  # the float32 hooks changed after the run
        with self.assertRaisesRegex(ProvenanceError, "recipe sources differ: run_reference.py"):
            self.check(self.recorded)

    def test_weight_free_runs_verify_all_but_the_weights(self):
        self.current["snapshot"]["model.safetensors"] = "w" * 64
        provenance.check_inputs(
            "synthetic", self.recorded, self.current, self.sources, without=("model.safetensors",), base=self.temp
        )
        with self.assertRaisesRegex(ProvenanceError, "snapshot files"):
            self.check(self.recorded)


class ParityTests(unittest.TestCase):
    GOOD = {"calls": 54, "bit_identical": 54, "non_float32_inputs": 0, "modules_hooked": 27, "modules_verified": 27}

    def test_passed_proof(self):
        provenance.check_parity("observations", self.GOOD)

    def test_missing_proof_refuses(self):
        with self.assertRaisesRegex(ProvenanceError, "no float32 parity proof"):
            provenance.check_parity("observations", None)

    def test_failed_empty_or_partial_proofs_refuse(self):
        for bad in (
            {**self.GOOD, "bit_identical": 53},
            {**self.GOOD, "calls": 0, "bit_identical": 0},
            {**self.GOOD, "non_float32_inputs": 1},
            {**self.GOOD, "modules_verified": 26},
            {**self.GOOD, "modules_hooked": 0, "modules_verified": 0},
        ):
            with self.subTest(proof=bad), self.assertRaisesRegex(ProvenanceError, "failed or verified nothing"):
                provenance.check_parity("observations", bad)


class OutputsTests(Temp):
    def setUp(self):
        super().setUp()
        self.directory = self.temp / "observations"
        self.write("observations/a.json", "{}\n")
        self.log = {"outputs": {"a": sha("{}\n")}}

    def test_written_bytes_pass(self):
        provenance.check_outputs("observations", self.log, self.directory, {"a"})

    def test_edited_observation_refuses(self):
        self.write("observations/a.json", '{"edited": true}\n')
        with self.assertRaisesRegex(ProvenanceError, "not what the run wrote"):
            provenance.check_outputs("observations", self.log, self.directory, {"a"})

    def test_unrun_file_refuses(self):
        self.write("observations/b.json", "{}\n")
        with self.assertRaisesRegex(ProvenanceError, "beyond the run"):
            provenance.check_outputs("observations", self.log, self.directory, None)

    def test_unobserved_case_refuses(self):
        with self.assertRaisesRegex(ProvenanceError, "do not match the cases"):
            provenance.check_outputs("observations", self.log, self.directory, {"a", "b"})


class RunTests(unittest.TestCase):
    def log(self, settings="hf-cpu-f32-t1", requests=None):
        return {
            "inputs": {"requests": requests or {"a": "1" * 64}},
            "numerics": {"settings": settings},
            "float32_parity": ParityTests.GOOD,
        }

    def check(self, log, requests):
        with mock.patch.object(provenance, "check_inputs"):
            provenance.check_run("observations", log, {}, requests)

    def test_matching_run_passes(self):
        self.check(self.log(), {"a": "1" * 64})

    def test_other_settings_refuse(self):
        with self.assertRaisesRegex(ProvenanceError, "not hf-cpu-f32-t1"):
            self.check(self.log(settings="hf-cpu-f32-t4"), {"a": "1" * 64})

    def test_changed_request_refuses(self):
        with self.assertRaisesRegex(ProvenanceError, "request changed"):
            self.check(self.log(), {"a": "2" * 64})

    def test_removed_case_refuses(self):
        with self.assertRaisesRegex(ProvenanceError, "no longer a case"):
            self.check(self.log(), {})


class ComparisonTests(Temp):
    def setUp(self):
        super().setUp()
        self.write("observations/a.json", "{}\n")
        patcher = mock.patch.object(provenance, "FIXTURES", self.temp)
        patcher.start()
        self.addCleanup(patcher.stop)
        script = provenance.sha256_file(provenance.HERE / "compare_observations.py")
        self.comparison = {
            "recipe": {"compare_observations.py": script},
            "a": {"files": {"a.json": sha("{}\n")}},
            "b": {"files": {"a.json": "3" * 64}},
        }

    def test_current_comparison_passes(self):
        provenance.check_comparison("threads", self.comparison, {"a.json": "3" * 64})

    def test_other_script_refuses(self):
        self.comparison["recipe"] = {"compare_observations.py": "0" * 64}
        with self.assertRaisesRegex(ProvenanceError, "another compare_observations.py"):
            provenance.check_comparison("threads", self.comparison, None)

    def test_stale_observations_refuse(self):
        self.write("observations/a.json", '{"regenerated": true}\n')
        with self.assertRaisesRegex(ProvenanceError, "other observations"):
            provenance.check_comparison("threads", self.comparison, None)

    def test_other_run_refuses(self):
        with self.assertRaisesRegex(ProvenanceError, "not the run's output"):
            provenance.check_comparison("threads", self.comparison, {"a.json": "4" * 64})


class UpdateRefusesTests(Temp):
    def test_missing_run_log_writes_nothing(self):
        target = self.temp / "provenance.json"
        cases = self.temp / "fixtures"
        (cases / "cases").mkdir(parents=True)
        with mock.patch.object(provenance, "PROVENANCE", target), mock.patch.object(
            provenance, "FIXTURES", cases
        ), mock.patch.object(provenance.make_cases, "check", lambda: []), mock.patch("sys.stderr"):
            code = provenance.main(["--update", "--runs", str(self.temp / "runs")])
        self.assertEqual(code, 1)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
