"""Native staging and generator pin checks, without a Rust build."""

import subprocess
import sys
import sysconfig
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_python_bolt import stage_native_artifacts
from gen_python_bolt import check_boltffi_version


class NativeStagingTests(unittest.TestCase):
    def test_stages_matching_pair_and_removes_only_obsolete_native_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "package"
            destination.mkdir()
            obsolete = destination / "_native.cpython-999-other.so"
            obsolete.write_bytes(b"old bridge")
            (destination / "xybrid_bolt.dll").write_bytes(b"foreign library")
            (destination / "_native.c").write_text("generated source")
            wheel = root / "package.whl"
            bridge = f"_native{sysconfig.get_config_var('EXT_SUFFIX')}"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(f"xybrid_bolt/{bridge}", b"new bridge")
                archive.writestr("xybrid_bolt/libxybrid_bolt.dylib", b"new library")

            stage_native_artifacts(wheel, destination, "libxybrid_bolt.dylib")

            self.assertEqual((destination / bridge).read_bytes(), b"new bridge")
            self.assertEqual((destination / "libxybrid_bolt.dylib").read_bytes(), b"new library")
            self.assertFalse(obsolete.exists())
            self.assertFalse((destination / "xybrid_bolt.dll").exists())
            self.assertEqual((destination / "_native.c").read_text(), "generated source")

    def test_incomplete_or_mismatched_wheel_preserves_existing_artifacts(self):
        bridge = f"_native{sysconfig.get_config_var('EXT_SUFFIX')}"
        for members in (
            {f"xybrid_bolt/{bridge}": b"bridge"},
            {"xybrid_bolt/libxybrid_bolt.dylib": b"library",
             "xybrid_bolt/_native.cpython-999-other.so": b"wrong interpreter"},
        ):
            with self.subTest(members=list(members)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                destination = root / "package"
                destination.mkdir()
                existing = destination / "libxybrid_bolt.dylib"
                existing.write_bytes(b"keep me")
                wheel = root / "package.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    for name, content in members.items():
                        archive.writestr(name, content)

                with self.assertRaisesRegex(RuntimeError, "missing"):
                    stage_native_artifacts(wheel, destination, "libxybrid_bolt.dylib")
                self.assertEqual(existing.read_bytes(), b"keep me")


class GeneratorVersionTests(unittest.TestCase):
    def test_rejects_mismatched_and_prefix_matching_cli_versions(self):
        for version in ("0.29.3", "0.30.10"):
            with self.subTest(version=version), patch(
                "gen_python_bolt.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout=f"boltffi {version}\n"),
            ):
                with self.assertRaises(SystemExit):
                    check_boltffi_version()

    def test_accepts_pinned_cli(self):
        with patch(
            "gen_python_bolt.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="boltffi 0.30.1\n"),
        ):
            check_boltffi_version()


if __name__ == "__main__":
    unittest.main()
