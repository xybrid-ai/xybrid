"""Matched-engine staging refuses missing, changed or unsupported inputs."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage_native as staging


class MatchedEngine(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.build = self.root / "build"
        self.output = self.root / "prefix"
        for name in staging.ARCHIVES:
            path = self.build / "llama.cpp" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"!<arch>\nfixture")
        (self.build / "bin").mkdir()
        for name in ("decision-oracle", "llama-server"):
            (self.build / "bin" / name).write_bytes(b"fixture executable")
        source = self.root / "source"
        (source / "include").mkdir(parents=True)
        for name in ("llama.h", "llama-cpp.h"):
            (source / "include" / name).write_text("fixture header")
        self.identity = {"llama_cpp_dir": str(source), "fingerprint": "fixture",
                         "build": {"system": "Linux", "processor": "x86_64", "build_type": "Release", "backends": ["cpu"]}}
        self.identity_patch = patch.object(staging.oracle, "build_identity", return_value=(self.identity, []))
        self.identity_patch.start()

    def tearDown(self):
        self.identity_patch.stop()
        self.temporary.cleanup()

    def test_complete_stage_has_hashes_and_preserves_existing_destination(self):
        receipt = staging.stage(self.build, self.output, staging.TARGET)
        self.assertEqual(staging.verify(self.build, self.output, staging.TARGET), receipt)
        self.assertEqual(len(receipt["files"]), 8)
        with self.assertRaisesRegex(ValueError, "must not exist"):
            staging.stage(self.build, self.output, staging.TARGET)

    def test_missing_vision_archive_cannot_fall_back_to_another_build(self):
        (self.build / "llama.cpp/tools/mtmd/libmtmd.a").unlink()
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            staging.stage(self.build, self.output, staging.TARGET)
        self.assertFalse(self.output.exists())

    def test_modified_archive_is_rejected_after_staging(self):
        staging.stage(self.build, self.output, staging.TARGET)
        (self.output / staging.TARGET / "lib/libllama.a").write_bytes(b"!<arch>\nchanged")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            staging.verify(self.build, self.output, staging.TARGET)

    def test_profile_and_target_must_match_the_qualified_comparison_lane(self):
        with self.assertRaisesRegex(ValueError, "Linux x86_64 GNU"):
            staging.stage(self.build, self.output, "aarch64-apple-darwin")
        self.identity["build"]["build_type"] = "Debug"
        with self.assertRaisesRegex(ValueError, "Release CPU"):
            staging.stage(self.build, self.output, staging.TARGET)

    def test_non_archive_input_is_rejected(self):
        (self.build / "llama.cpp/src/libllama.a").write_bytes(b"not an archive")
        with self.assertRaisesRegex(ValueError, "regular static archive"):
            staging.stage(self.build, self.output, staging.TARGET)


if __name__ == "__main__":
    unittest.main()
