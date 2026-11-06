"""Exercise the public Apple notice assembly without native compilation."""

import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "native_apple_notices", ROOT / "bindings/apple/pack_native_notices.py"
)
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


class NativeAppleNoticeTests(unittest.TestCase):
    def test_archive_preserves_native_bytes_and_contains_complete_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zip"
            output = Path(directory) / "output.zip"
            entries = {
                "XybridFFI.xcframework/Info.plist": b"fixture plist",
                "XybridFFI.xcframework/ios-arm64/XybridFFI.framework/XybridFFI": b"fixture native",
            }
            with zipfile.ZipFile(source, "w") as archive:
                for name, data in entries.items():
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    archive.writestr(info, data)
            original = source.read_bytes()
            notice = ROOT / "crates/llama-cpp-sys/NOTICE"
            PACK.package(source, notice, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    entries | {PACK.NOTICE_PATH: notice.read_bytes()},
                    {name: archive.read(name) for name in archive.namelist()},
                )
            self.assertEqual(original, source.read_bytes())

    def test_gradle_and_bazel_android_asset_matches_the_canonical_notice(self):
        self.assertEqual(
            (ROOT / "crates/llama-cpp-sys/NOTICE").read_bytes(),
            (ROOT / "bindings/kotlin/src/main/assets/xybrid/llama/NOTICE").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
