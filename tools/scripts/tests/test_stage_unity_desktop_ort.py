import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from stage_unity_desktop_ort import (
    NUPKG_URL,
    ORT_VERSION,
    RUNTIMES,
    RuntimeSpec,
    stage_from_archive,
)


class StageUnityDesktopOrtTests(unittest.TestCase):
    def test_both_platforms_come_from_the_pinned_nuget_package(self):
        self.assertTrue(NUPKG_URL.startswith("https://api.nuget.org/"))
        self.assertIn(f"/microsoft.ml.onnxruntime/{ORT_VERSION}/", NUPKG_URL)
        for platform, spec in RUNTIMES.items():
            with self.subTest(platform=platform):
                self.assertEqual(spec.url, NUPKG_URL)
                self.assertRegex(spec.sha256, r"^[0-9a-f]{64}$")
                self.assertTrue(spec.archive_member.startswith("runtimes/"))
        self.assertEqual(RUNTIMES["linux"].sha256, RUNTIMES["windows"].sha256)

    def test_linux_archive_stages_runtime_under_the_load_name(self):
        payload = b"linux-onnxruntime"
        member = "runtimes/linux-x64/native/libonnxruntime.so"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "ort.nupkg"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr(member, payload)

            spec = self.spec("linux", member, "libonnxruntime.so", archive)
            output = root / "Linux"
            stage_from_archive(spec, archive, output)

            self.assertEqual(payload, (output / "libonnxruntime.so").read_bytes())
            meta = (output / "libonnxruntime.so.meta").read_text()
            self.assertIn("Linux64:", meta)
            self.assertIn("CPU: x86_64", meta)
            self.assertIn("Exclude WebGL: 1", meta)

    def test_windows_archive_stages_runtime_and_windows_metadata(self):
        payload = b"windows-onnxruntime"
        member = "runtimes/win-x64/native/onnxruntime.dll"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "ort.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr(member, payload)

            spec = self.spec("windows", member, "onnxruntime.dll", archive)
            output = root / "Windows"
            stage_from_archive(spec, archive, output)

            self.assertEqual(payload, (output / "onnxruntime.dll").read_bytes())
            meta = (output / "onnxruntime.dll.meta").read_text()
            self.assertIn("OS: Windows", meta)
            self.assertIn("Win64:", meta)
            self.assertIn("Exclude WebGL: 1", meta)

    def test_hash_mismatch_fails_without_staging_a_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "ort.zip"
            archive.write_bytes(b"not-the-pinned-archive")
            spec = RuntimeSpec(
                platform="windows",
                url="https://example.invalid/ort.zip",
                sha256="0" * 64,
                archive_member="onnxruntime.dll",
                runtime_name="onnxruntime.dll",
            )
            output = root / "Windows"

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                stage_from_archive(spec, archive, output)

            self.assertFalse((output / "onnxruntime.dll").exists())

    def test_missing_runtime_member_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "ort.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("README.md", "no runtime here")
            spec = self.spec(
                "windows",
                "runtimes/win-x64/native/onnxruntime.dll",
                "onnxruntime.dll",
                archive,
            )
            output = root / "Windows"

            with self.assertRaisesRegex(ValueError, "does not contain"):
                stage_from_archive(spec, archive, output)

            self.assertFalse((output / "onnxruntime.dll").exists())

    def test_replace_failure_removes_temporary_runtime(self):
        payload = b"windows-onnxruntime"
        member = "runtimes/win-x64/native/onnxruntime.dll"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "ort.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr(member, payload)

            spec = self.spec("windows", member, "onnxruntime.dll", archive)
            output = root / "Windows"

            with mock.patch.object(
                Path, "replace", side_effect=OSError("rename failed")
            ):
                with self.assertRaisesRegex(OSError, "rename failed"):
                    stage_from_archive(spec, archive, output)

            self.assertFalse((output / "onnxruntime.dll").exists())
            self.assertFalse((output / ".onnxruntime.dll.tmp").exists())

    @staticmethod
    def spec(
        platform: str, member: str, runtime_name: str, archive: Path
    ) -> RuntimeSpec:
        return RuntimeSpec(
            platform=platform,
            url="https://example.invalid/ort",
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            archive_member=member,
            runtime_name=runtime_name,
        )


if __name__ == "__main__":
    unittest.main()
