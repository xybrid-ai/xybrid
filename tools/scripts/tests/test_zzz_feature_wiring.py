"""Offline feature-graph checks; no Bazel fetch, native build or engine bytes."""

import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATIC = "tts-zzz"


def cargo_features(crate):
    return tomllib.loads((ROOT / f"crates/{crate}/Cargo.toml").read_text())["features"]


def configured_targets(static):
    settings = {}
    flags = {}
    values = {":tts_zzz": str(static).lower()}
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
    def test_static_forwarders_stay_opt_in(self):
        self.assertEqual(["dep:xybrid-zzz-sys", "xybrid-zzz-sys/bindings"], cargo_features("xybrid-core")[STATIC])
        self.assertEqual(["xybrid-core/" + STATIC], cargo_features("xybrid-sdk")[STATIC])
        self.assertEqual(["xybrid-sdk/" + STATIC], cargo_features("xybrid-bolt")[STATIC])
        for crate in ("xybrid-core", "xybrid-sdk", "xybrid-bolt"):
            for name, expansion in cargo_features(crate).items():
                if name == "default" or name.startswith("platform-"):
                    self.assertFalse(any(value == STATIC or value.endswith("/" + STATIC) for value in expansion), (crate, name))

    def test_bazel_flag_selects_the_static_feature_only_when_set(self):
        for static in (False, True):
            with self.subTest(static=static):
                flags, targets = configured_targets(static)
                self.assertFalse(flags["tts_zzz"]["build_setting_default"])
                expected = [STATIC] if static else []
                for target in ("xybrid-core", "xybrid-sdk", "xybrid_bolt_cdylib", "xybrid_bolt_staticlib"):
                    selected = [f for f in targets[target]["crate_features"] if f == STATIC]
                    self.assertEqual(expected, selected, target)
                self.assertEqual(["bindings"] if static else [], targets["xybrid_zzz_sys"]["crate_features"])
                expected_dep = ["//crates/zzz-sys:xybrid_zzz_sys"] if static else []
                self.assertEqual(expected_dep, [d for d in targets["xybrid-core"]["deps"] if "/zzz-sys:" in d])
                self.assertEqual({"xybrid_zzz_sys"}, {name for name in targets if "zzz" in name})
