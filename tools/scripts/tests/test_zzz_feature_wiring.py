"""Offline, platform-aware checks of the speech (tts-zzz) feature graph.

No Bazel fetch, native build or engine bytes. The real BUILD files run with a
recording `select()`; every site that forwards tts-zzz is then resolved per
platform with Bazel's select rules (constraint-subset matching, a match_any
group acting as its first matching member, and the winner having to refine
every other match). That catches what nothing else does before a release:

- a site whose select differs from the others. xybrid-sdk compiles without
  xybrid-core's zzz module, so a core/sdk mismatch builds fine and misroutes
  text-to-speech at runtime;
- a platform that silently gains or loses speech, or an ambiguous select;
- the Kotlin AAR's ABIs: arm64-v8a links the engine, armeabi-v7a and x86_64
  build without it under the same --//:tts_zzz=true invocation.
"""

import ast
import subprocess
import tomllib
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATIC = "tts-zzz"
ZZZ_SYS = "//crates/zzz-sys:xybrid_zzz_sys"
DEFAULT = "//conditions:default"
FLAG = "//:tts_zzz"
GROUP = "//bazel/zzz:android_without_slice"
ENABLED = "//:tts_zzz_enabled"
INCOMPATIBLE = "@platforms//:incompatible"
CANONICAL_KEYS = [GROUP, ENABLED, DEFAULT]
# The BUILD files that may forward tts-zzz; each must use the canonical select.
FORWARDING_FILES = (
    "crates/xybrid-core/BUILD.bazel",
    "crates/xybrid-sdk/BUILD.bazel",
    "crates/xybrid-bolt/BUILD.bazel",
    "integration-tests/BUILD.bazel",
)
# (file, target, attribute, value of the //:tts_zzz_enabled arm)
SITES = [
    ("crates/xybrid-core/BUILD.bazel", "xybrid-core", "crate_features", STATIC),
    ("crates/xybrid-core/BUILD.bazel", "xybrid-core", "deps", ZZZ_SYS),
    ("crates/xybrid-sdk/BUILD.bazel", "xybrid-sdk", "crate_features", STATIC),
    ("crates/xybrid-bolt/BUILD.bazel", "xybrid_bolt_cdylib", "crate_features", STATIC),
    ("crates/xybrid-bolt/BUILD.bazel", "xybrid_bolt_staticlib", "crate_features", STATIC),
    ("integration-tests/BUILD.bazel", "tts_abort", "crate_features", STATIC),
]

_OS, _CPU = "@platforms//os:", "@platforms//cpu:"
_RS = "@rules_rs//rs/platforms/constraints:"
_LIBC = "@llvm//constraints/libc:"
# Constraint values of the rules_rs platforms (rules_rs 0.0.95 and its pinned
# rules_rust triple mapping), as printed by
# `bazel query --output=build @rules_rs//rs/platforms:<triple>`.
RULES_RS_PLATFORMS = {
    "aarch64-linux-android": {_CPU + "aarch64", _OS + "android"},
    "armv7-linux-androideabi": {_CPU + "armv7", _OS + "android", _RS + "softfloat"},
    "x86_64-linux-android": {_CPU + "x86_64", _OS + "android"},
    "i686-linux-android": {_CPU + "x86_32", _OS + "android"},
    "x86_64-unknown-linux-gnu": {_CPU + "x86_64", _OS + "linux", _RS + "glibc", _LIBC + "unconstrained"},
    "aarch64-unknown-linux-gnu": {_CPU + "aarch64", _OS + "linux", _RS + "glibc", _LIBC + "unconstrained"},
    "aarch64-unknown-linux-musl": {_CPU + "aarch64", _OS + "linux", _RS + "musl", _LIBC + "musl",
                                   "@llvm//constraints/pie:off"},
    "aarch64-apple-darwin": {_CPU + "aarch64", _OS + "osx"},
    "x86_64-apple-darwin": {_CPU + "x86_64", _OS + "osx"},
    "aarch64-apple-ios": {_CPU + "aarch64", _OS + "ios", "@build_bazel_apple_support//constraints:device"},
    "aarch64-apple-ios-sim": {_CPU + "aarch64", _OS + "ios", "@build_bazel_apple_support//constraints:simulator"},
    "x86_64-apple-ios": {_CPU + "x86_64", _OS + "ios", "@build_bazel_apple_support//constraints:simulator"},
    "x86_64-pc-windows-gnu": {_CPU + "x86_64", _OS + "windows", "@llvm//constraints/windows/abi:gnu",
                              "@llvm//constraints/windows/crt:msvcrt"},
}
# @platforms aliases that config_settings may name.
ALIASES = {_OS + "macos": _OS + "osx"}
NO_SPEECH = "no speech"
# Expected flag-on outcome per platform: the engine slice that zzz-sys links,
# NO_SPEECH (built without tts-zzz), or INCOMPATIBLE (tts-zzz plus a zzz-sys
# that cannot build, so an explicit request fails loudly).
EXPECTED = {
    "aarch64-linux-android": "@zzz_android//:zzz",
    "//bazel/ggml:android_arm64_cpu_variants": "@zzz_android//:zzz",
    "armv7-linux-androideabi": NO_SPEECH,
    "x86_64-linux-android": NO_SPEECH,
    "i686-linux-android": INCOMPATIBLE,
    "x86_64-unknown-linux-gnu": "@zzz_linux//:zzz",
    "aarch64-unknown-linux-gnu": "@zzz_linux_arm64//:zzz",
    "aarch64-unknown-linux-musl": INCOMPATIBLE,
    "aarch64-apple-darwin": "@zzz_macos//:zzz",
    "x86_64-apple-darwin": INCOMPATIBLE,
    "aarch64-apple-ios": "@zzz_ios//:zzz",
    "aarch64-apple-ios-sim": "@zzz_ios_sim//:zzz",
    "x86_64-apple-ios": INCOMPATIBLE,
    "x86_64-pc-windows-gnu": INCOMPATIBLE,
}


class Select:
    """A recorded select(); `+` with lists or other selects keeps the parts."""

    def __init__(self, arms):
        self.arms = arms

    def __add__(self, other):
        return Concat([self, other])

    def __radd__(self, other):
        return Concat([other, self])


class Concat:
    def __init__(self, parts):
        self.parts = []
        for part in parts:
            self.parts.extend(part.parts if isinstance(part, Concat) else [part])

    def __add__(self, other):
        return Concat([self, other])

    def __radd__(self, other):
        return Concat([other, self])


def selects_in(value):
    parts = value.parts if isinstance(value, Concat) else [value]
    return [part for part in parts if isinstance(part, Select)]


def exec_build(relative, **names):
    """Runs a BUILD file; returns {target name: kwargs} plus the namespace."""
    targets = {}

    def capture(**kwargs):
        targets[kwargs["name"]] = kwargs

    namespace = {
        "load": lambda *a, **kw: None, "glob": lambda *a, **kw: [], "select": Select,
        "DEP_DATA": {}, "XYBRID_VERSION": "test", "all_crate_deps": lambda **kw: [],
        "_all_crate_deps": lambda *a, **kw: [], "exports_files": lambda *a, **kw: None,
        "alias": lambda **kw: None,
    }
    for rule in ("rust_library", "rust_shared_library", "rust_static_library", "rust_test",
                 "cc_library", "cc_import", "cc_binary", "filegroup", "pkg_files", "pkg_zip",
                 "kt_jvm_library", "define_kt_toolchain", "platform_transition_filegroup",
                 "zzz_notice_files"):
        namespace[rule] = capture
    namespace.update(names)
    path = ROOT / relative
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return targets, namespace


def root_speech_rules():
    """The root BUILD's flag and setting, read without running the whole file."""
    calls = {}
    for node in ast.walk(ast.parse((ROOT / "BUILD.bazel").read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            name = kwargs.get("name")
            if isinstance(name, ast.Constant) and name.value in ("tts_zzz", "tts_zzz_enabled"):
                calls[name.value] = (node.func.id, {k: ast.literal_eval(v) for k, v in kwargs.items()
                                                    if k != "visibility"})
    return calls


def canonical(label, package):
    label = ALIASES.get(label, label)
    return package + label if label.startswith(":") else label


def speech_settings():
    """Every setting a speech select names: //:tts_zzz_enabled and //bazel/zzz:*."""
    settings, groups = {}, []
    namespace = {
        "load": lambda *a, **kw: None,
        "config_setting": lambda **kw: settings.update({kw["name"]: kw}),
        "selects": types.SimpleNamespace(config_setting_group=lambda **kw: groups.append(kw)),
    }
    path = ROOT / "bazel/zzz/BUILD.bazel"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    model = {}
    for name, kw in settings.items():
        model["//bazel/zzz:" + name] = types.SimpleNamespace(
            members=None,
            constraints=frozenset(canonical(c, "//bazel/zzz") for c in kw.get("constraint_values", [])),
            flags=frozenset((canonical(k, "//bazel/zzz"), v) for k, v in kw.get("flag_values", {}).items()))
    for group in groups:
        # skylib resolves a match_all group to its last member, which does not
        # refine //:tts_zzz_enabled; only match_any groups can be modelled.
        if "match_any" not in group:
            raise AssertionError(f"//bazel/zzz:{group['name']} must be a match_any group")
        model["//bazel/zzz:" + group["name"]] = types.SimpleNamespace(
            members=[canonical(m, "//bazel/zzz") for m in group["match_any"]])
    kind, enabled = root_speech_rules()["tts_zzz_enabled"]
    assert kind == "config_setting"
    model[ENABLED] = types.SimpleNamespace(
        members=None, constraints=frozenset(),
        flags=frozenset((canonical(k, "//"), v) for k, v in enabled["flag_values"].items()))
    return model, settings, groups


class Ambiguous(AssertionError):
    pass


def match(model, key, constraints, flags):
    """The config_setting a select key resolves to on this platform, or None."""
    setting = model[key]
    if setting.members is not None:  # skylib match_any: the first matching member
        return next((resolved for member in setting.members
                     if (resolved := match(model, member, constraints, flags))), None)
    if setting.constraints <= constraints and all(flags.get(k) == v for k, v in setting.flags):
        return key
    return None


def refines(model, left, right):
    a, b = model[left], model[right]
    return (a.constraints >= b.constraints and a.flags >= b.flags
            and (a.constraints, a.flags) != (b.constraints, b.flags))


def choose(model, arms, constraints, flags):
    matched = [(key, resolved) for key in arms if key != DEFAULT
               if (resolved := match(model, key, constraints, flags))]
    if not matched:
        if DEFAULT not in arms:
            raise AssertionError(f"no select arm matches and there is no default: {list(arms)}")
        return arms[DEFAULT]
    winners = [key for key, resolved in matched
               if all(refines(model, resolved, other) for key2, other in matched if key2 != key)]
    if len(winners) != 1:
        raise Ambiguous(f"ambiguous select: {[key for key, _ in matched]} all match")
    return arms[winners[0]]


def resolve(model, value, constraints, flags):
    """Resolves the speech selects in an attribute; other selects contribute nothing."""
    parts = value.parts if isinstance(value, Concat) else [value]
    result = []
    for part in parts:
        if isinstance(part, Select):
            if all(key in model or key == DEFAULT for key in part.arms):
                result += choose(model, part.arms, constraints, flags)
        else:
            result += part
    return result


def platform_constraints():
    """Constraint sets per platform, including the AAR's arm64 child platform."""
    platforms = {triple: frozenset(values) for triple, values in RULES_RS_PLATFORMS.items()}
    ggml = {}
    exec_build("bazel/ggml/BUILD.bazel", platform=lambda **kw: ggml.update({kw["name"]: kw}),
               constraint_setting=lambda **kw: None, constraint_value=lambda **kw: None,
               config_setting=lambda **kw: None)
    child = ggml["android_arm64_cpu_variants"]
    parent = child["parents"][0].rsplit(":", 1)[1]
    platforms["//bazel/ggml:android_arm64_cpu_variants"] = platforms[parent] | {
        canonical(value, "//bazel/ggml") for value in child["constraint_values"]}
    return platforms


def sites():
    """{(file, target, attribute): the recorded speech Select}."""
    found = {}
    for relative in FORWARDING_FILES:
        targets, _ = exec_build(relative)
        for name, kwargs in targets.items():
            for attribute in ("crate_features", "deps"):
                for part in selects_in(kwargs.get(attribute, [])):
                    values = [v for arm in part.arms.values() for v in arm]
                    if STATIC in values or ZZZ_SYS in values or any(k in part.arms for k in (GROUP, ENABLED)):
                        key = (relative, name, attribute)
                        if key in found:
                            raise AssertionError(f"{key} has two speech selects")
                        found[key] = part
    return found


def zzz_sys():
    targets, _ = exec_build("crates/zzz-sys/BUILD.bazel")
    return targets["xybrid_zzz_sys"]


class SpeechFeatureWiringTests(unittest.TestCase):
    def test_static_forwarders_stay_opt_in(self):
        def cargo_features(crate):
            return tomllib.loads((ROOT / f"crates/{crate}/Cargo.toml").read_text())["features"]

        self.assertEqual(["dep:xybrid-zzz-sys", "xybrid-zzz-sys/bindings"], cargo_features("xybrid-core")[STATIC])
        self.assertEqual(["xybrid-core/" + STATIC], cargo_features("xybrid-sdk")[STATIC])
        self.assertEqual(["xybrid-sdk/" + STATIC], cargo_features("xybrid-bolt")[STATIC])
        for crate in ("xybrid-core", "xybrid-sdk", "xybrid-bolt"):
            for name, expansion in cargo_features(crate).items():
                if name == "default" or name.startswith("platform-"):
                    self.assertFalse(any(value == STATIC or value.endswith("/" + STATIC) for value in expansion), (crate, name))

    def test_flag_is_off_by_default_and_reset_in_exec(self):
        # rules_rust configures every Rust dep a second time in exec
        # (proc_macro_deps). A "universal" flag would make a flag-on Android
        # build resolve the build host's engine slice as well.
        rules = root_speech_rules()
        self.assertEqual(("bool_flag", {"name": "tts_zzz", "build_setting_default": False, "scope": "target"}),
                         rules["tts_zzz"])
        self.assertEqual({":tts_zzz": "true"}, rules["tts_zzz_enabled"][1]["flag_values"])
        self.assertNotIn("constraint_values", rules["tts_zzz_enabled"][1])

    def test_android_without_slice_members_refine_the_flag(self):
        model, settings, groups = speech_settings()
        self.assertEqual("supported_enabled", groups[0]["name"])
        group = next(g for g in groups if g["name"] == "android_without_slice")
        self.assertNotIn("match_all", group)
        self.assertEqual({"//visibility:public"}, set(group["visibility"]))
        for member in model[GROUP].members:
            setting = model[member]
            self.assertIsNone(setting.members, f"{member} must be a plain config_setting")
            self.assertEqual({(FLAG, "true")}, set(setting.flags))
            self.assertIn(_OS + "android", setting.constraints)
            self.assertTrue(refines(model, member, ENABLED))
        # No platform that ships an engine slice may be in the group.
        supported = model["//bazel/zzz:supported_enabled"].members
        for member in model[GROUP].members:
            for slice_setting in supported:
                self.assertFalse(model[member].constraints <= model[slice_setting].constraints
                                 or model[slice_setting].constraints <= model[member].constraints,
                                 (member, slice_setting))

    def test_every_forwarder_uses_the_identical_select(self):
        found = sites()
        self.assertEqual({site[:3] for site in SITES}, set(found))
        for relative, target, attribute, value in SITES:
            with self.subTest(target=target, attribute=attribute):
                arms = found[(relative, target, attribute)].arms
                self.assertEqual(CANONICAL_KEYS, list(arms))
                self.assertEqual({GROUP: [], ENABLED: [value], DEFAULT: []}, arms)

    def test_no_other_build_file_forwards_the_speech_feature(self):
        # A new forwarder must join FORWARDING_FILES/SITES (and the canonical
        # select); otherwise it would escape the per-platform checks below.
        tracked = subprocess.run(["git", "ls-files", "-z", "--", "*BUILD.bazel", "*.bzl"], cwd=ROOT,
                                 capture_output=True, check=True).stdout.decode().split("\0")
        self.assertIn("crates/xybrid-core/BUILD.bazel", tracked)
        for relative in filter(None, tracked):
            text = (ROOT / relative).read_text(errors="replace")
            if f'"{STATIC}"' in text or ZZZ_SYS in text:
                self.assertIn(relative, FORWARDING_FILES)

    def test_resolution_per_platform(self):
        model, _, _ = speech_settings()
        found = sites()
        engine = zzz_sys()
        platforms = platform_constraints()
        self.assertEqual(set(EXPECTED), set(platforms))
        for flag in (False, True):
            flags = {FLAG: "true" if flag else "false"}
            for platform, constraints in platforms.items():
                expected = EXPECTED[platform] if flag else NO_SPEECH
                with self.subTest(flag=flag, platform=platform):
                    for (relative, target, attribute, value) in SITES:
                        resolved = resolve(model, found[(relative, target, attribute)], constraints, flags)
                        self.assertEqual([] if expected == NO_SPEECH else [value], resolved, (target, attribute))
                    if expected == NO_SPEECH:
                        continue
                    # Reached through xybrid-core's dep: zzz-sys links the slice
                    # or is incompatible, never a silent no-op build.
                    self.assertEqual(["bindings"], resolve(model, engine["crate_features"], constraints, flags))
                    compatible = resolve(model, engine["target_compatible_with"], constraints, flags)
                    slices = [dep for dep in resolve(model, engine["deps"], constraints, flags)
                              if dep.startswith("@zzz_")]
                    if expected == INCOMPATIBLE:
                        self.assertEqual(([INCOMPATIBLE], []), (compatible, slices))
                    else:
                        self.assertEqual(([], [expected]), (compatible, slices))
            # With the flag off zzz-sys stays a buildable no-op shell everywhere.
            if not flag:
                for constraints in platforms.values():
                    self.assertEqual([], resolve(model, engine["target_compatible_with"], constraints, flags))
                    self.assertEqual([], resolve(model, engine["deps"], constraints, flags))

    def test_kotlin_aar_links_speech_into_arm64_only(self):
        _, namespace = exec_build("bindings/kotlin/BUILD.bazel", ANDROID_ARM64_CPU_VARIANTS=[],
                                  ANDROID_ARM64_GGML_LIBS=[])
        model, _, _ = speech_settings()
        found = sites()
        platforms = platform_constraints()
        abis = {abi: platform.split(":", 1)[1] if platform.startswith("@rules_rs") else platform
                for abi, platform, _, _ in namespace["ANDROID_ABIS"]}
        self.assertEqual({"arm64-v8a", "armeabi-v7a", "x86_64"}, set(abis))
        flags = {FLAG: "true"}
        for abi, platform in abis.items():
            with self.subTest(abi=abi):
                # A new ABI either gets a pinned slice or joins
                # //bazel/zzz:android_without_slice; an incompatible ABI
                # would fail every flag-on AAR build.
                self.assertIn(platform, platforms, "add the new ABI's platform constraints to this test")
                features = resolve(model, found[SITES[0][:3]], platforms[platform], flags)
                self.assertEqual([STATIC] if abi == "arm64-v8a" else [], features)
                self.assertEqual("@zzz_android//:zzz" if abi == "arm64-v8a" else NO_SPEECH, EXPECTED[platform])

    def test_resolver_rejects_a_member_that_does_not_refine_the_flag(self):
        # Guards the model itself: a constraint-only member (or a match_all
        # group resolving to one) is not more specific than the flag-only
        # setting, which Bazel reports as an ambiguous select.
        model, _, _ = speech_settings()
        model["//test:armv7_only"] = types.SimpleNamespace(
            members=None, constraints=frozenset({_OS + "android", _CPU + "armv7"}), flags=frozenset())
        model["//test:group"] = types.SimpleNamespace(members=["//test:armv7_only"])
        arms = {"//test:group": [], ENABLED: [STATIC], DEFAULT: []}
        with self.assertRaises(Ambiguous):
            choose(model, arms, frozenset(RULES_RS_PLATFORMS["armv7-linux-androideabi"]), {FLAG: "true"})


if __name__ == "__main__":
    unittest.main()
