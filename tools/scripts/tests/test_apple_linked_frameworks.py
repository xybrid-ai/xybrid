"""The Apple manifests link the same system frameworks, and CI checks the archives.

The XybridFFI archive is linked through three manifests: the root
`Package.swift`, the `Package.swift` that release-publish.yml writes to the
`swift` branch, and the React Native podspec. Each must link every framework
the archive references, so they share one list. The Flutter macOS pod names
its frameworks twice (`OTHER_LDFLAGS` and `s.frameworks`); those agree too.

MetalPerformanceShaders is linked by none of them (#807). build-apple.yml and
release-prep.yml scan the archives with `nm -u` and fail on an MPS symbol, so
a dependency that starts using it is caught before an app fails to link. These
tests keep both scans wired and their pattern able to match.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGE_SWIFT = "Package.swift"
RELEASE_PUBLISH = ".github/workflows/release-publish.yml"
RN_PODSPEC = "bindings/react-native/react-native-xybrid.podspec"
FLUTTER_MACOS_PODSPEC = "bindings/flutter/macos/xybrid_flutter.podspec"
BUILD_APPLE = ".github/workflows/build-apple.yml"
RELEASE_PREP = ".github/workflows/release-prep.yml"


def read(path):
    return (REPO / path).read_text()


def only(pattern, text, what):
    """The single capture of `pattern` in `text`."""
    found = re.findall(pattern, text, re.M)
    if len(found) != 1:
        raise AssertionError(f"expected one {what}, found {found}")
    return found[0]


def swift_link_settings(text):
    """(libraries, frameworks) a Swift manifest links, in order."""
    return (re.findall(r'\.linkedLibrary\("([^"]+)"\)', text),
            re.findall(r'\.linkedFramework\("([^"]+)"\)', text))


def swift_branch_manifest():
    """The Package.swift release-publish.yml writes to the swift branch."""
    text = read(RELEASE_PUBLISH)
    body = text.split("cat > Package.swift << SWIFT_EOF\n", 1)[1]
    return body.split("SWIFT_EOF", 1)[0]


def podspec_list(text, attribute):
    """The quoted names assigned to `s.<attribute>`, continuation lines joined."""
    value = only(rf"^\s*s\.{attribute}\s*=\s*((?:.*,[ \t]*\n)*.*)$", text, f"s.{attribute}")
    return re.findall(r"""["']([^"']+)["']""", value)


def workflow_step(path, name):
    """A workflow step's text, with shell line continuations joined."""
    text = read(path)
    start = text.index(f"      - name: {name}\n")
    after = re.search(r"^      - ", text[start + 1:], re.M)
    step = text[start:start + 1 + after.start()] if after else text[start:]
    return " ".join(re.sub(r"\\\n", " ", step).split())


class LinkedFrameworkParityTests(unittest.TestCase):
    def test_package_swift_links_a_framework_list(self):
        libraries, frameworks = swift_link_settings(read(PACKAGE_SWIFT))
        self.assertEqual(libraries, ["c++"])
        self.assertIn("Metal", frameworks)

    def test_swift_branch_manifest_links_what_package_swift_links(self):
        self.assertEqual(swift_link_settings(swift_branch_manifest()),
                         swift_link_settings(read(PACKAGE_SWIFT)))

    def test_react_native_podspec_links_what_package_swift_links(self):
        podspec = read(RN_PODSPEC)
        libraries, frameworks = swift_link_settings(read(PACKAGE_SWIFT))
        self.assertEqual(podspec_list(podspec, "frameworks"), frameworks)
        self.assertEqual(podspec_list(podspec, "libraries"), libraries)

    def test_flutter_macos_ldflags_match_its_frameworks(self):
        podspec = read(FLUTTER_MACOS_PODSPEC)
        ldflags = only(r"'OTHER_LDFLAGS' => '([^']*)'", podspec, "OTHER_LDFLAGS")
        self.assertEqual(set(re.findall(r"-framework (\w+)", ldflags)),
                         set(podspec_list(podspec, "frameworks")))


class MpsArchiveScanTests(unittest.TestCase):
    STEPS = {
        BUILD_APPLE: "Verify XCFramework",
        RELEASE_PREP: "Verify Flutter darwin payloads",
    }

    def test_ci_scans_the_archives_for_mps_symbols(self):
        for path, name in self.STEPS.items():
            with self.subTest(step=name):
                step = workflow_step(path, name)
                self.assertIn("nm -u", step)
                self.assertIn("_MPS", step)

    def test_the_scan_pattern_matches_mps_and_nothing_else(self):
        for path, name in self.STEPS.items():
            with self.subTest(step=name):
                pattern = only(r"grep -c?E '(\(\^\| \)_\(MPS[^']*)'",
                               workflow_step(path, name), "MPS pattern")
                for symbol in ("_MPSSupportsMTLDevice", "_OBJC_CLASS_$_MPSGraph",
                               "                 U _MPSImageGaussianBlur"):
                    self.assertTrue(re.search(pattern, symbol, re.M), symbol)
                for symbol in ("_MTLCreateSystemDefaultDevice", "_OBJC_CLASS_$_MLModel",
                               "__ZN3MPS4testEv"):
                    self.assertIsNone(re.search(pattern, symbol, re.M), symbol)

    def test_build_apple_links_the_device_slice(self):
        step = workflow_step(BUILD_APPLE, "Link the device slice (iOS, no signing)")
        self.assertIn("build-for-testing", step)
        self.assertIn("-destination 'generic/platform=iOS'", step)


if __name__ == "__main__":
    unittest.main()
