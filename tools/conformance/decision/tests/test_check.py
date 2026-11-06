"""The input gate must fail when evidence identity or coverage changes."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check as evidence


class EvidenceInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixtures = Path(self.temporary.name) / "decision"
        shutil.copytree(evidence.FIXTURES, self.fixtures)
        self.patches = [
            patch.object(evidence, "FIXTURES", self.fixtures),
            patch.object(evidence.hf_reference, "FIXTURES", self.fixtures),
            patch.object(evidence.subprocess, "check_output", return_value=f"160000 {evidence.PIN} 0\tvendor/llama-cpp\n"),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def test_committed_input_set_passes(self):
        evidence.check()

    def test_changed_fixture_fails(self):
        path = next((self.fixtures / "cases").glob("*.json"))
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "fixture digest mismatch"):
            evidence.check()

    def test_removed_fixture_cannot_narrow_the_gate(self):
        next((self.fixtures / "cases").glob("*.json")).unlink()
        with self.assertRaisesRegex(ValueError, "143-case manifest"):
            evidence.check()

    def test_changed_native_pin_requires_reviewed_evidence(self):
        with patch.object(evidence.subprocess, "check_output", return_value="160000 " + "0" * 40 + " 0\tvendor/llama-cpp\n"):
            with self.assertRaisesRegex(ValueError, "changed: regenerate and review"):
                evidence.check()

    def test_reviewed_observation_cannot_be_removed_or_added_silently(self):
        path = next((self.fixtures / "hf/observations").rglob("*.json"))
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "observations differ"):
            evidence.check()
        path.write_bytes(original)
        (path.parent / "unreviewed.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "observations differ"):
            evidence.check()

    def test_missing_and_corrupt_required_model_fail(self):
        path = self.fixtures / "missing.gguf"
        with self.assertRaises(FileNotFoundError):
            evidence.check(path)
        path.write_bytes(b"not the pinned model")
        with self.assertRaisesRegex(ValueError, "artifact identity mismatch"):
            evidence.check(path)

    def test_changed_hf_collector_cannot_reuse_old_observations(self):
        digest = evidence.hf_reference.digest
        def changed(path):
            return "0" * 64 if Path(path).name == "hf_reference.py" else digest(path)
        with patch.object(evidence.hf_reference, "digest", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "reference source changed: collector_sha256"):
                evidence.check()


if __name__ == "__main__":
    unittest.main()
