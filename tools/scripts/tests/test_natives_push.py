"""Check that incomplete native slices never reach the write-once publisher."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "natives-push.sh"


class NativesPushTests(unittest.TestCase):
    def publish(self, target, features, include_hash):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "bin"
            binaries.mkdir()
            push_log = root / "push.log"
            stubs = {
                "oras": (
                    '#!/bin/sh\n'
                    'case "$1" in\n'
                    '  manifest) exit 1 ;;\n'
                    '  push) printf "%s\\n" "$@" > "$XYBRID_TEST_ORAS_LOG" ;;\n'
                    '  *) exit 2 ;;\n'
                    'esac\n'
                ),
                "cmake": "#!/bin/sh\necho 'cmake version 3.31.0'\n",
                "cc": "#!/bin/sh\necho 'test cc 1.0'\n",
            }
            for name, source in stubs.items():
                binary = binaries / name
                binary.write_text(source)
                binary.chmod(0o755)

            export = root / "export"
            slice_dir = export / target
            (slice_dir / "include").mkdir(parents=True)
            libraries = slice_dir / "lib"
            libraries.mkdir()
            names = ["llama", "ggml", "ggml-base", "ggml-cpu"]
            if features in ("vision", "vision-vulkan"):
                names.append("mtmd")
                if include_hash:
                    names.append("vendor-hash")
            if features in ("vulkan", "vision-vulkan"):
                names.append("ggml-vulkan")
            prefix, suffix = ("", ".lib") if "windows-msvc" in target else ("lib", ".a")
            for name in names:
                (libraries / f"{prefix}{name}{suffix}").write_bytes(b"archive")

            result = subprocess.run(
                ["bash", str(SCRIPT), target, features, str(export)],
                env=dict(
                    os.environ,
                    PATH=f"{binaries}{os.pathsep}{os.environ['PATH']}",
                    CC="cc",
                    XYBRID_TEST_ORAS_LOG=str(push_log),
                    XYBRID_NATIVES_PKG="ghcr.io/xybrid-ai/test-natives",
                ),
                capture_output=True,
                text=True,
                timeout=30,
            )
            return (
                result,
                push_log.read_text() if push_log.exists() else None,
                (slice_dir / "native.tar.gz").exists(),
            )

    def test_vision_without_hash_is_rejected_before_publication(self):
        for target in ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"):
            for features in ("vision", "vision-vulkan"):
                with self.subTest(target=target, features=features):
                    result, push, packed = self.publish(target, features, include_hash=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("vendor-hash", result.stderr)
                    self.assertIn("refusing to publish incomplete slice", result.stderr)
                    self.assertIsNone(push)
                    self.assertFalse(packed)

    def test_complete_vision_slice_is_published(self):
        for target in ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"):
            for features in ("vision", "vision-vulkan"):
                with self.subTest(target=target, features=features):
                    result, push, packed = self.publish(target, features, include_hash=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIsNotNone(push)
                    self.assertEqual(push.splitlines()[0], "push")
                    self.assertIn(f"dev.xybrid.features={features}", push)
                    self.assertTrue(packed)

    def test_base_slice_does_not_require_vision_archives(self):
        result, push, packed = self.publish("x86_64-unknown-linux-gnu", "base", include_hash=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(push)
        self.assertTrue(packed)


if __name__ == "__main__":
    unittest.main()
