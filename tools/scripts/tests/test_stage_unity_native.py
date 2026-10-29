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

    def test_ios_archive_is_staged_with_accelerate_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "bazel-bin" / "libxybrid_bolt.a"
            native.parent.mkdir()
            native.write_bytes(b"ios-native")

            destination = stage_native(
                native,
                "aarch64-apple-ios",
                plugins_root=root / "Plugins",
            )

            self.assertEqual(root / "Plugins" / "iOS" / native.name, destination)
            self.assertEqual(b"ios-native", destination.read_bytes())
            metadata = destination.with_name(f"{native.name}.meta").read_text()
            self.assertIn("Editor:\n      enabled: 0", metadata)
            self.assertIn(
                "iOS:\n      enabled: 1\n      settings:\n"
                "        AddToEmbeddedBinaries: false\n"
                "        CPU: ARM64\n"
                "        CompileFlags:\n"
                "        FrameworkDependencies: Accelerate\n",
                metadata,
            )
            self.assertTrue((root / "Plugins" / "iOS.meta").is_file())

    def test_existing_ios_metadata_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "libxybrid_bolt.a"
            native.write_bytes(b"new-ios-native")
            output = root / "Plugins" / "iOS"
            output.mkdir(parents=True)
            metadata = output / f"{native.name}.meta"
            existing = (
                b"fileFormatVersion: 2\n"
                b"guid: 0123456789abcdef0123456789abcdef\n"
                b"PluginImporter:\n"
                b"  platformData:\n"
                b"    iOS:\n"
                b"      enabled: 1\n"
                b"      settings:\n"
                b"        FrameworkDependencies: Metal;CustomFramework\n"
            )
            metadata.write_bytes(existing)

            stage_native(
                native,
                "aarch64-apple-ios",
                plugins_root=root / "Plugins",
            )

            self.assertEqual(existing, metadata.read_bytes())
            self.assertEqual(b"new-ios-native", (output / native.name).read_bytes())

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
            # The Unity library links libc++ statically; a shared copy would
            # collide with the one Unity's GameActivity packages.
            self.assertFalse((output / "libc++_shared.so").exists())
            self.assertFalse((output / "libc++_shared.so.meta").exists())
            for library in ("libxybrid_bolt.so", "libonnxruntime.so"):
                metadata = (output / f"{library}.meta").read_text()
                self.assertIn("Android:\n      enabled: 1", metadata)
                self.assertIn("CPU: x86_64", metadata)
            self.assertTrue((root / "Plugins" / "Android.meta").is_file())
            self.assertTrue(
                (root / "Plugins" / "Android" / "x86_64.meta").is_file()
            )

    def test_stale_android_cxx_runtime_is_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "bazel-bin" / "libxybrid_bolt.so"
            native.parent.mkdir()
            native.write_bytes(b"android-native")
            dependencies = root / "ort-android" / "arm64-v8a"
            dependencies.mkdir(parents=True)
            (dependencies / "libonnxruntime.so").write_bytes(b"ort")
            output = root / "Plugins" / "Android" / "arm64-v8a"
            output.mkdir(parents=True)
            (output / "libc++_shared.so").write_bytes(b"stale-cxx")
            (output / "libc++_shared.so.meta").write_text("stale-guid")

            stage_native(
                native,
                "aarch64-linux-android",
                plugins_root=root / "Plugins",
                android_dependencies=root / "ort-android",
            )

            self.assertFalse((output / "libc++_shared.so").exists())
            self.assertFalse((output / "libc++_shared.so.meta").exists())
            self.assertEqual(b"ort", (output / "libonnxruntime.so").read_bytes())
            self.assertTrue((output / "libonnxruntime.so.meta").is_file())

    def test_stale_android_cxx_runtime_is_removed_without_dependencies(self):
        # armeabi-v7a has no vendored ORT, so staging returns early there.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "bazel-bin" / "libxybrid_bolt.so"
            native.parent.mkdir()
            native.write_bytes(b"android-native")
            output = root / "Plugins" / "Android" / "armeabi-v7a"
            output.mkdir(parents=True)
            (output / "libc++_shared.so").write_bytes(b"stale-cxx")
            (output / "libc++_shared.so.meta").write_text("stale-guid")

            stage_native(
                native,
                "armv7-linux-androideabi",
                plugins_root=root / "Plugins",
                android_dependencies=root / "ort-android",
            )

            self.assertFalse((output / "libc++_shared.so").exists())
            self.assertFalse((output / "libc++_shared.so.meta").exists())
            self.assertEqual(
                b"android-native", (output / "libxybrid_bolt.so").read_bytes()
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
