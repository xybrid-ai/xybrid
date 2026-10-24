"""Offline feature-graph checks; no Bazel fetch, native build or engine bytes."""

import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMPANION = "tts-zzz-android-companion"
STATIC = "tts-zzz"


def cargo_features(crate):
    return tomllib.loads((ROOT / f"crates/{crate}/Cargo.toml").read_text())["features"]


def configured_targets(static, companion):
    settings = {}
    flags = {}
    values = {":tts_zzz": str(static).lower(), ":tts_zzz_android_companion": str(companion).lower()}
    root = (ROOT / "BUILD.bazel").read_text().split("# The macOS GPU lane toggle", 1)[0]
    exec(root, {"load": lambda *a, **kw: None, "exports_files": lambda *a: None,
                "bool_flag": lambda **kw: flags.update({kw["name"]: kw}),
                "config_setting": lambda **kw: settings.update({kw["name"]: kw})})
    enabled = {"//:" + name for name, setting in settings.items()
               if all(values.get(key) == value for key, value in setting["flag_values"].items())}

    def select(arms):
        matches = [value for name, value in arms.items() if name in enabled]
        if len(matches) > 1:
            raise AssertionError("ambiguous speech feature select")
        return matches[0] if matches else arms.get("//conditions:default", [])

    targets = {}
    def capture(**kwargs):
        targets[kwargs["name"]] = kwargs
    namespace = {
        "load": lambda *a, **kw: None, "glob": lambda *a, **kw: [],
        "select": select, "DEP_DATA": {}, "XYBRID_VERSION": "test",
        "all_crate_deps": lambda **kw: [], "_all_crate_deps": lambda *a, **kw: [],
        "rust_library": capture, "rust_shared_library": capture,
        "rust_static_library": capture, "rust_test": capture,
        "cc_library": capture, "exports_files": lambda *a: None,
        "alias": lambda **kw: None,
    }
    for crate in ("xybrid-core", "xybrid-sdk", "xybrid-bolt", "zzz-sys"):
        path = ROOT / f"crates/{crate}/BUILD.bazel"
        exec(compile(path.read_text(), str(path), "exec"), namespace)
    return flags, targets


class SpeechFeatureWiringTests(unittest.TestCase):
    def test_companion_forwarders_do_not_enable_the_static_feature(self):
        core = cargo_features("xybrid-core")
        self.assertEqual(["dep:xybrid-zzz-sys", "xybrid-zzz-sys/android-companion"], core[COMPANION])
        self.assertEqual(["dep:xybrid-zzz-sys", "xybrid-zzz-sys/bindings"], core[STATIC])
        self.assertEqual(["xybrid-core/" + COMPANION], cargo_features("xybrid-sdk")[COMPANION])
        self.assertEqual(["xybrid-sdk/" + COMPANION], cargo_features("xybrid-bolt")[COMPANION])
        for crate in ("xybrid-core", "xybrid-sdk", "xybrid-bolt"):
            for name, expansion in cargo_features(crate).items():
                if name == "default" or name.startswith("platform-"):
                    self.assertFalse(any(COMPANION in value or value.endswith("/tts-zzz") for value in expansion))

    def test_all_bazel_flag_combinations_select_one_transport_with_static_precedence(self):
        for static, companion in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(static=static, companion=companion):
                flags, targets = configured_targets(static, companion)
                self.assertFalse(flags["tts_zzz_android_companion"]["build_setting_default"])
                expected = [STATIC] if static else [COMPANION] if companion else []
                for target in ("xybrid-core", "xybrid-sdk", "xybrid_bolt_cdylib", "xybrid_bolt_staticlib"):
                    selected = [f for f in targets[target]["crate_features"] if f in (STATIC, COMPANION)]
                    self.assertEqual(expected, selected, target)
                expected_dep = ["//crates/zzz-sys:xybrid_zzz_sys"] if static else ["//crates/zzz-sys:xybrid_zzz_companion_client"] if companion else []
                self.assertEqual(expected_dep, [d for d in targets["xybrid-core"]["deps"] if "/zzz-sys:" in d])
                self.assertEqual(["android-companion"], targets["xybrid_zzz_companion_client"]["crate_features"])
                self.assertEqual([], targets["xybrid_zzz_companion_client"]["deps"])

    def test_production_run_stream_and_runtime_gates_accept_either_transport(self):
        for name in (
            "crates/xybrid-core/src/runtime_adapter/mod.rs",
            "crates/xybrid-core/src/execution/executor.rs",
            "crates/xybrid-core/src/features.rs",
            "crates/xybrid-sdk/src/model.rs",
        ):
            source = (ROOT / name).read_text().split("#[cfg(test)]", 1)[0]
            self.assertNotRegex(source, r'cfg!?\((?:not\()?feature = "tts-zzz"\)')
            self.assertIn('feature = "tts-zzz-android-companion"', source)
        client = (ROOT / "crates/zzz-sys/src/companion.rs").read_text()
        self.assertIn("EXPECTED_CONTRACT: Option<[u8; 32]> = None", client)
        self.assertNotIn(COMPANION, (ROOT / "bindings/flutter/rust/Cargo.toml").read_text())
