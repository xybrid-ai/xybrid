"""The Flutter plugin declares the platform floors its native libraries are built for.

The podspecs' `s.platform` and the Android module's `minSdkVersion` are what an
app is checked against; the build pins and release-prep's payload gates decide
what the precompiled libraries need. If a declaration drops below its build
pin, an app at the declared minimum builds and then cannot load the library.
macOS has no build pin (hermetic-llvm's default decides), so its floor is held
by release-prep running verify_apple_min_os.py, which these tests keep wired.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify_apple_min_os import format_version, parse_version, podspec_platform

REPO = Path(__file__).resolve().parents[3]
IOS_PODSPEC = "bindings/flutter/ios/xybrid_flutter.podspec"
MACOS_PODSPEC = "bindings/flutter/macos/xybrid_flutter.podspec"
ANDROID_GRADLE = "bindings/flutter/android/build.gradle"
RELEASE_PREP = ".github/workflows/release-prep.yml"


def read(path):
    return (REPO / path).read_text()


def only(pattern, text, what):
    """The single capture of `pattern` in `text`."""
    found = re.findall(pattern, text, re.M)
    if len(found) != 1:
        raise AssertionError(f"expected one {what}, found {found}")
    return found[0]


def workflow_step(name):
    """A release-prep step's text, with shell line continuations joined."""
    text = read(RELEASE_PREP)
    start = text.index(f"      - name: {name}\n")
    after = re.search(r"^      - ", text[start + 1:], re.M)
    step = text[start:start + 1 + after.start()] if after else text[start:]
    return " ".join(re.sub(r"\\\n", " ", step).split())


class ApplePlatformTests(unittest.TestCase):
    def test_ios_podspec_matches_the_bazel_minimum(self):
        os_name, floor = podspec_platform(read(IOS_PODSPEC))
        self.assertEqual(os_name, "ios")
        bazelrc = read(".bazelrc")
        for config in ("ios", "ios-sim"):
            for flag in ("--ios_minimum_os=", "--action_env=IPHONEOS_DEPLOYMENT_TARGET="):
                with self.subTest(config=config, flag=flag):
                    pinned = only(rf"^build:{config} {re.escape(flag)}(\S+)$", bazelrc, flag)
                    self.assertEqual(parse_version(pinned), floor,
                                     f"{IOS_PODSPEC} declares iOS {format_version(floor)}")

    def test_release_prep_checks_every_darwin_library_against_its_podspec(self):
        step = workflow_step("Verify Flutter darwin payloads")
        lib = "prebuilt/{}/libxybrid_flutter_ffi.a".format
        for podspec, triples in ((MACOS_PODSPEC, ["aarch64-apple-darwin"]),
                                 (IOS_PODSPEC, ["aarch64-apple-ios", "aarch64-apple-ios-sim"])):
            with self.subTest(podspec=podspec):
                command = " ".join(["python3 tools/scripts/verify_apple_min_os.py",
                                    "--podspec", podspec, *map(lib, triples)])
                self.assertTrue(command in step, f"the darwin payload step must run: {command}")


class AndroidMinSdkTests(unittest.TestCase):
    def test_min_sdk_matches_the_linked_api_level(self):
        gradle = read(ANDROID_GRADLE)
        min_sdk = only(r"^\s*minSdkVersion (\d+)\s*$", gradle, "minSdkVersion")
        pins = {
            "CARGOKIT_RUST_MIN_SDK_VERSION (cargo builds)": only(
                r"ext\.set\('CARGOKIT_RUST_MIN_SDK_VERSION', '(\d+)'\)", gradle, "Rust min SDK"),
            "release-prep's Bazel library gate": only(
                r"verify_android_api_level\.py --api (\d+)",
                workflow_step("Verify Flutter payloads"), "API level gate"),
            "release-prep's cargokit i686 build": only(
                r"--android-min-sdk-version (\d+)",
                workflow_step("Sign and upload Flutter linux/android binaries (cargokit)"),
                "cargokit min SDK"),
        }
        for what, level in pins.items():
            with self.subTest(pin=what):
                self.assertEqual(level, min_sdk, f"{ANDROID_GRADLE} declares minSdkVersion {min_sdk}")


if __name__ == "__main__":
    unittest.main()
