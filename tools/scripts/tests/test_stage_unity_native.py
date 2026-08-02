import stat
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from stage_unity_native import stage_native


class StageUnityNativeTests(unittest.TestCase):
    def test_staged_native_is_owner_writable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "bazel-bin" / "libxybrid_bolt.so"
            native.parent.mkdir()
            native.write_bytes(b"linux-native")
            native.chmod(0o555)

            destination = stage_native(
                native,
                "x86_64-unknown-linux-gnu",
                plugins_root=root / "Plugins",
            )

            self.assertNotEqual(
                0,
                destination.stat().st_mode & stat.S_IWUSR,
                "staged natives must be writable by post-processing tools",
            )

    def test_macos_native_is_staged_with_editor_enabled_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "bazel-bin" / "libxybrid_bolt.dylib"
            native.parent.mkdir()
            native.write_bytes(b"macos-native")

            destination = stage_native(
                native,
                "aarch64-apple-darwin",
                plugins_root=root / "Plugins",
            )

            self.assertEqual(
                root / "Plugins" / "macOS" / native.name, destination
            )
            self.assertEqual(b"macos-native", destination.read_bytes())
            metadata = destination.with_name(
                f"{destination.name}.meta"
            ).read_text()
            self.assertIn("OSXUniversal:", metadata)
            self.assertIn("Editor:\n      enabled: 1", metadata)
            self.assertIn("Exclude OSXUniversal: 0", metadata)
            self.assertTrue((root / "Plugins" / "macOS.meta").is_file())

    def test_android_native_and_dependencies_use_the_target_abi(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "bazel-bin" / "libxybrid_bolt.so"
            native.parent.mkdir()
            native.write_bytes(b"android-native")
            dependencies = root / "ort-android" / "x86_64"
            dependencies.mkdir(parents=True)
            (dependencies / "libonnxruntime.so").write_bytes(b"ort")
            (dependencies / "libc++_shared.so").write_bytes(b"cxx")

            destination = stage_native(
                native,
                "x86_64-linux-android",
                plugins_root=root / "Plugins",
                android_dependencies=root / "ort-android",
            )

            output = root / "Plugins" / "Android" / "x86_64"
            self.assertEqual(output / native.name, destination)
            self.assertEqual(b"ort", (output / "libonnxruntime.so").read_bytes())
            self.assertEqual(b"cxx", (output / "libc++_shared.so").read_bytes())
            for library in (
                "libxybrid_bolt.so",
                "libonnxruntime.so",
                "libc++_shared.so",
            ):
                metadata = (output / f"{library}.meta").read_text()
                self.assertIn("Android:\n      enabled: 1", metadata)
                self.assertIn("CPU: x86_64", metadata)
            self.assertTrue((root / "Plugins" / "Android.meta").is_file())
            self.assertTrue(
                (root / "Plugins" / "Android" / "x86_64.meta").is_file()
            )

    def test_existing_metadata_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "libxybrid_bolt.dylib"
            native.write_bytes(b"new-native")
            output = root / "Plugins" / "macOS"
            output.mkdir(parents=True)
            metadata = output / "libxybrid_bolt.dylib.meta"
            metadata.write_text("existing-guid")

            stage_native(
                native,
                "aarch64-apple-darwin",
                plugins_root=root / "Plugins",
            )

            self.assertEqual("existing-guid", metadata.read_text())

    def test_generated_guid_does_not_depend_on_the_checkout_path(self):
        metadata = []
        with tempfile.TemporaryDirectory() as first_temp:
            with tempfile.TemporaryDirectory() as second_temp:
                for temp in (first_temp, second_temp):
                    root = Path(temp)
                    native = root / "libxybrid_bolt.dylib"
                    native.write_bytes(b"native")
                    destination = stage_native(
                        native,
                        "aarch64-apple-darwin",
                        plugins_root=root / "Plugins",
                    )
                    metadata.append(
                        destination.with_name(
                            f"{destination.name}.meta"
                        ).read_text()
                    )

        self.assertEqual(metadata[0], metadata[1])

    def test_missing_native_fails_without_creating_plugin_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plugins = root / "Plugins"

            with self.assertRaisesRegex(FileNotFoundError, "native library"):
                stage_native(
                    root / "missing" / "libxybrid_bolt.so",
                    "aarch64-linux-android",
                    plugins_root=plugins,
                    android_dependencies=root / "ort-android",
                )

            self.assertFalse(plugins.exists())

    def test_unknown_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "libxybrid_bolt.so"
            native.write_bytes(b"native")

            with self.assertRaisesRegex(ValueError, "unsupported target"):
                stage_native(
                    native,
                    "wasm32-unknown-unknown",
                    plugins_root=root / "Plugins",
                )


if __name__ == "__main__":
    unittest.main()
