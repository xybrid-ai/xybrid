"""Fail-closed independent-reference inputs (no model or ML dependency needed)."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hf_reference", HERE / "hf_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceInputs(unittest.TestCase):
    def test_corpus_has_disjoint_calibration_and_held_out_inputs(self):
        cases = reference.corpus(None)
        self.assertEqual(len(cases), 63)
        self.assertEqual(sum(entry["split"] == "calibration" for entry, _ in cases), 42)
        self.assertEqual(sum(entry["split"] == "held_out" for entry, _ in cases), 21)

    def test_unknown_selection_fails_instead_of_running_nothing(self):
        with self.assertRaisesRegex(ValueError, "unknown --only"):
            reference.corpus(["not-a-case"])

    def test_changed_case_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "hf/cases").mkdir(parents=True)
            manifest = json.loads((reference.FIXTURES / "hf/corpus.json").read_text())
            (root / "hf/corpus.json").write_text(json.dumps(manifest))
            first = manifest["cases"][0]["id"]
            original = reference.FIXTURES / "hf/cases" / f"{first}.json"
            (root / "hf/cases" / original.name).write_text(original.read_text() + "\n")
            with patch.object(reference, "FIXTURES", root), self.assertRaisesRegex(ValueError, "digest mismatch"):
                reference.corpus(None)

    def test_corrupt_or_missing_assets_fail_before_loading_remote_code(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = {"assets": {"config.json": {"bytes": 2, "sha256": "0" * 64}}}
            with self.assertRaisesRegex(ValueError, "missing or corrupt"):
                reference.verify_assets(root, manifest)
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing or corrupt"):
                reference.verify_assets(root, manifest)


if __name__ == "__main__":
    unittest.main()
