"""Tests for install_pins.py, the docs install-snippet pins version-sync.sh owns.

PinTests run the real pin patterns over small docs in a temporary directory,
each pin registered for the files a test writes. RepoPinsTests read the
committed docs: a bump must change nothing but their pins, and every install
snippet in them must belong to a pin, including one in a shape no template
knows. CI's Version sync job runs this file on every pull request, docs-only
ones included, since docs are where these snippets are edited.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "scripts" / "install_pins.py"
SPEC = importlib.util.spec_from_file_location("install_pins", SCRIPT)
pins = importlib.util.module_from_spec(SPEC)
# dataclasses resolves the module's string annotations through sys.modules.
sys.modules[SPEC.name] = pins
SPEC.loader.exec_module(pins)

# Changelogs quote the install snippets of their own releases.
HISTORY = re.compile(r"(^|/)CHANGELOG[^/]*\.md$")

# A package a user installs, as install syntax names it. A version shortly
# after one, on the same line, is an install snippet that some pin must own,
# whatever its shape: this is the net for a form no PINS template knows.
PACKAGE = re.compile(
    r"\bxybrid_flutter\b|\bxybrid-kotlin\b|@xybrid/[\w-]+|\bai\.xybrid\.sdk\b"
    r"|github\.com/xybrid-ai/xybrid(?:\.git)?(?![\w/.-])|^[ \t]*xybrid(?:-\w+)?[ \t]*=|\bxybrid(?:-\w+)?@",
    re.M,
)
# The first version-like token within 60 characters, wide enough for an inline
# Cargo table that lists its features before its version.
NEARBY_VERSION = re.compile(r"[^\n]{0,60}?(?<![\d.])(?P<version>\d+\.\d+)")


def workspace_version() -> str:
    """The Cargo workspace version, which every committed pin names."""
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    return re.search(r'^\[workspace\.package\]\n(?:.*\n)*?version = "([^"]+)"', cargo, re.M)[1]


def copy_registered_docs(root: Path) -> list[str]:
    """Copy every file a pin is registered for into `root`; return their paths."""
    paths = sorted({path for _, path in pins.registrations()})
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, root / path)
    return paths


def pin(label: str, *paths: str):
    """The real pin `label`, registered for `paths` instead of its own files."""
    real = next(candidate for candidate in pins.PINS if candidate.label == label)
    return dataclasses.replace(real, paths=paths)


def every_pin(path: str):
    """Every real pin, registered for `path` alone."""
    return [dataclasses.replace(real, paths=(path,)) for real in pins.PINS]


def unowned_versions(text: str) -> list[str]:
    """Each line where a package name is followed by a version no pin owns."""
    owned = {match.start("version") for real in pins.PINS for match in real.pattern.finditer(text)}
    lines = {}  # by start offset: a line can name a package twice
    for package in PACKAGE.finditer(text):
        version = NEARBY_VERSION.match(text, package.end())
        if version and version.start("version") not in owned:
            start = text.rfind("\n", 0, package.start()) + 1
            end = text.find("\n", package.end())
            lines[start] = text[start:end if end != -1 else len(text)].strip()
    return list(lines.values())


# Every snippet shape in PINS, as the docs write it, at 0.16.0.
ALL_SNIPPETS = """\
dependencies:
  xybrid_flutter: ^0.16.0

    implementation("ai.xybrid:xybrid-kotlin:0.16.0")

    .package(url: "https://github.com/xybrid-ai/xybrid.git", from: "0.16.0")
    .package(url: "https://github.com/xybrid-ai/xybrid", from: "0.16.0")

3. Set **Dependency Rule** to **Up to Next Major Version** → `0.16.0`

npm install @xybrid/react-native@0.16.0

    "ai.xybrid.sdk": "0.16.0"
    "ai.xybrid.sdk": "file:../path/to/xybrid/bindings/unity"

https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity#v0.16.0
https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity
Pin a version by appending `#v0.16.0` to the git URL.

[dependencies]
xybrid = "0.16.0"
serde = "1.0"
"""

# The other spellings each template accepts, also at 0.16.0.
VARIANTS = """\
  xybrid_flutter: "^0.16.0"
  xybrid_flutter:  '>=0.16.0 <1.0.0'
  xybrid_flutter:
    version: ^0.16.0

    implementation 'ai.xybrid:xybrid-kotlin:0.16.0'
xybrid-kotlin = { module = "ai.xybrid:xybrid-kotlin", version = "0.16.0" }
    implementation(group = "ai.xybrid", name = "xybrid-kotlin", version = "0.16.0")
    <artifactId>xybrid-kotlin</artifactId>
    <version>0.16.0</version>

    .package(url: "https://github.com/xybrid-ai/xybrid", exact: "0.16.0")
    .package(url: "https://github.com/xybrid-ai/xybrid", .upToNextMinor(from: "0.16.0"))
    .package(url: "https://github.com/xybrid-ai/xybrid", "0.16.0"..<"1.0.0")
    .package(
        url: "https://github.com/xybrid-ai/xybrid.git",
        .exact("0.16.0")
    )

Set the rule to **Up to Next Minor Version** -> `0.16.0`.

    "@xybrid/react-native": "^0.16.0"

openupm add ai.xybrid.sdk@0.16.0
    "ai.xybrid.sdk" : "0.16.0"
https://github.com/xybrid-ai/xybrid.git?path=bindings/unity#v0.16.0

xybrid="0.16.0"
xybrid = { version = "0.16.0", features = ["platform-macos"] }
xybrid = { features = ["llm-llamacpp"], version = "=0.16.0" }

[dependencies.xybrid]
features = ["platform-macos"]

version = "0.16.0"

cargo add xybrid@0.16.0
"""

PROSE = """\
Releases 0.16.0 and later require Android 10 (API 29).
Releases before 0.16.0 require API 24. Tool calling shipped in 0.6.0.
| API 29+ | Android 10 | 0.16.0 and later |
npm install @xybrid/react-native@next
"""


class PinTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, path: str, text: str) -> None:
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)

    def read(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    def test_check_reports_one_line_per_pin_and_file(self):
        self.write("README.md", 'xybrid = "0.16.0"\n')
        self.write("crates/xybrid/README.md", 'xybrid = "0.16.0"\n')
        ok, lines = pins.check(self.root, "0.16.0", [pin("Cargo", "README.md", "crates/xybrid/README.md")])
        self.assertTrue(ok)
        self.assertEqual(lines, [
            "  Cargo pin (README.md): 0.16.0 ✓",
            "  Cargo pin (crates/xybrid/README.md): 0.16.0 ✓",
        ])

    def test_every_snippet_shape_is_checked_and_set(self):
        for snippets in (ALL_SNIPPETS, VARIANTS):
            with self.subTest(snippets=snippets.splitlines()[1]):
                self.write("README.md", snippets + PROSE)
                table = every_pin("README.md")
                ok, lines = pins.check(self.root, "0.16.0", table)
                self.assertTrue(ok, lines)

                pins.set_version(self.root, "0.17.0", table)

                self.assertEqual(self.read("README.md"), snippets.replace("0.16.0", "0.17.0") + PROSE)
                ok, lines = pins.check(self.root, "0.17.0", table)
                self.assertTrue(ok, lines)

    def test_a_stale_variant_next_to_a_current_pin_fails_check(self):
        self.write("crates/xybrid/README.md",
                   'xybrid = "0.16.0"\n\n'
                   'xybrid = { version = "0.2", features = ["platform-macos"] }\n')
        ok, lines = pins.check(self.root, "0.16.0", [pin("Cargo", "crates/xybrid/README.md")])
        self.assertFalse(ok)
        self.assertEqual(lines, ["  Cargo pin (crates/xybrid/README.md): 0.16.0 0.2 ✗ (expected 0.16.0)"])

    def test_a_versioned_package_no_pin_owns_is_flagged(self):
        """unowned_versions, the repo-wide net, on snippets no template knows."""
        self.assertEqual(unowned_versions(ALL_SNIPPETS + VARIANTS + PROSE), [])
        unknown = (
            'xybrid = { git = "https://github.com/xybrid-ai/xybrid", tag = "v0.6.0" }\n'
            'xybrid-sdk = "0.6.0"\n'
            "npm install @xybrid/web@0.6.0\n"
            "[xybrid](https://github.com/xybrid-ai/xybrid) 0.6.0\n"
        )
        self.assertEqual(unowned_versions(ALL_SNIPPETS + unknown), unknown.splitlines())
        # Prose about a package that names no version nearby is not a snippet.
        self.assertEqual(unowned_versions(
            "`@xybrid/react-native` wraps the same Rust core as the Swift and Kotlin SDKs, before 1.0.\n"
            "npm install @xybrid/react-native@next\n"
            "libxybrid_flutter_ffi.a.gz: 14.2 MB of 33.6 MB\n"
            "https://github.com/xybrid-ai/xybrid/actions/runs/123 (v0.3.0)\n"), [])

    def test_prose_naming_another_version_does_not_fail_check(self):
        self.write("README.md", PROSE + 'xybrid = "0.17.0"\n')
        ok, lines = pins.check(self.root, "0.17.0", [pin("Cargo", "README.md")])
        self.assertTrue(ok, lines)

    def test_stale_pin_fails_check(self):
        self.write("bindings/flutter/README.md", "dependencies:\n  xybrid_flutter: ^0.6.0\n")
        self.write("README.md", "dependencies:\n  xybrid_flutter: ^0.16.0\n")
        ok, lines = pins.check(self.root, "0.16.0", [pin("Flutter pubspec", "README.md", "bindings/flutter/README.md")])
        self.assertFalse(ok)
        self.assertEqual(lines, [
            "  Flutter pubspec pin (README.md): 0.16.0 ✓",
            "  Flutter pubspec pin (bindings/flutter/README.md): 0.6.0 ✗ (expected 0.16.0)",
        ])

    def test_stale_duplicate_fails_check(self):
        self.write("bindings/unity/README.md",
                   "https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity#v0.16.0\n"
                   "https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity#v0.13.1\n")
        ok, lines = pins.check(self.root, "0.16.0", [pin("Unity git tag", "bindings/unity/README.md")])
        self.assertFalse(ok)
        self.assertEqual(lines, ["  Unity git tag pin (bindings/unity/README.md): 0.13.1 0.16.0 ✗ (expected 0.16.0)"])

    def test_missing_snippet_fails_check_and_set_writes_nothing(self):
        # README.md lost its SwiftPM snippet; its Cargo pin and the other file
        # are fine, but a failed set must leave both untouched.
        self.write("README.md", PROSE + 'xybrid = "0.16.0"\n')
        self.write("crates/xybrid/README.md", 'xybrid = "0.16.0"\n')
        table = [pin("SwiftPM", "README.md"), pin("Cargo", "README.md", "crates/xybrid/README.md")]

        ok, lines = pins.check(self.root, "0.16.0", table)
        self.assertFalse(ok)
        self.assertEqual(lines[0], "  SwiftPM pin (README.md): NOT FOUND ✗ (expected 0.16.0)")

        with self.assertRaisesRegex(pins.PinError, r"SwiftPM install pin not found in README\.md"):
            pins.set_version(self.root, "0.17.0", table)
        self.assertEqual(self.read("README.md"), PROSE + 'xybrid = "0.16.0"\n')
        self.assertEqual(self.read("crates/xybrid/README.md"), 'xybrid = "0.16.0"\n')

    def test_missing_file_fails_check_and_set(self):
        table = [pin("Cargo", "crates/xybrid/README.md")]
        ok, lines = pins.check(self.root, "0.16.0", table)
        self.assertFalse(ok)
        self.assertEqual(lines, ["  Cargo pin (crates/xybrid/README.md): FILE MISSING ✗ (expected 0.16.0)"])
        with self.assertRaisesRegex(pins.PinError, r"Cargo install pin: crates/xybrid/README\.md is missing"):
            pins.set_version(self.root, "0.17.0", table)

    def test_prerelease_versions(self):
        self.write("bindings/unity/README.md",
                   '"ai.xybrid.sdk": "0.1.0-beta12"\n'
                   "https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity#v0.1.0-beta12\n")
        table = [pin("Unity OpenUPM", "bindings/unity/README.md"),
                 pin("Unity git tag", "bindings/unity/README.md")]
        self.assertTrue(pins.check(self.root, "0.1.0-beta12", table)[0])

        for version in ("0.2.0-rc.1", "1.0.0+build.5", "1.0.0"):
            pins.set_version(self.root, version, table)
            ok, lines = pins.check(self.root, version, table)
            self.assertTrue(ok, lines)
        self.assertEqual(self.read("bindings/unity/README.md"),
                         '"ai.xybrid.sdk": "1.0.0"\n'
                         "https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity#v1.0.0\n")

    def test_a_closing_period_is_not_part_of_the_version(self):
        self.write("README.md", "Pin it: https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity#v0.1.0-rc.2.\n")
        table = [pin("Unity git tag", "README.md")]
        self.assertEqual(pins.check(self.root, "0.1.0-rc.2", table)[0], True)
        pins.set_version(self.root, "0.2.0", table)
        self.assertEqual(self.read("README.md"),
                         "Pin it: https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity#v0.2.0.\n")

    def test_set_refuses_a_malformed_version(self):
        self.write("README.md", 'xybrid = "0.16.0"\n')
        for version in ("v0.17.0", "", "0.17.0.", "0.17 .0"):
            with self.subTest(version=version), self.assertRaisesRegex(pins.PinError, "is not a version"):
                pins.set_version(self.root, version, [pin("Cargo", "README.md")])
        self.assertEqual(self.read("README.md"), 'xybrid = "0.16.0"\n')

    def test_set_keeps_line_endings_and_non_ascii_text(self):
        text = "0.16.0 及更高版本需要 Android 10。\r\n\r\n```toml\r\nxybrid = \"0.16.0\"\r\n```\r\n"
        self.write("README.zh-CN.md", text)
        pins.set_version(self.root, "0.17.0", [pin("Cargo", "README.zh-CN.md")])
        with (self.root / "README.zh-CN.md").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(stream.read(), text.replace('"0.16.0"', '"0.17.0"'))


class CommandLineTests(unittest.TestCase):
    """The entry points version-sync.sh calls, on a copy of the real docs."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        copy_registered_docs(self.root)
        self.version = workspace_version()

    def run_pins(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPT), *args, "--root", str(self.root)],
                              capture_output=True, text=True, encoding="utf-8", timeout=60)

    def test_check_and_set(self):
        self.assertEqual(self.run_pins("check", self.version).returncode, 0)

        result = self.run_pins("set", "99.0.0-rc.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_pins("check", "99.0.0-rc.1")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(len(result.stdout.splitlines()), len(list(pins.registrations())))

        stale = self.root / "crates/xybrid/README.md"
        stale.write_text(stale.read_text(encoding="utf-8").replace('"99.0.0-rc.1"', '"0.2"'), encoding="utf-8")
        result = self.run_pins("check", "99.0.0-rc.1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("  Cargo pin (crates/xybrid/README.md): 0.2 ✗ (expected 99.0.0-rc.1)\n", result.stdout)

    def test_set_rejects_a_tag(self):
        result = self.run_pins("set", "v99.0.0")
        self.assertEqual(result.returncode, 1)
        self.assertIn("'v99.0.0' is not a version", result.stderr)
        self.assertEqual(self.run_pins("check", self.version).returncode, 0)


class RepoPinsTests(unittest.TestCase):
    def test_set_changes_only_install_pin_versions(self):
        """Prose that names the current release, such as "Releases 0.16.0 and
        later", keeps it; only the version groups of registered pins change."""
        current, new = workspace_version(), "99.0.0-rc.1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = copy_registered_docs(root)
            pins.set_version(root, new)
            for path in paths:
                with self.subTest(path=path):
                    before = (ROOT / path).read_text(encoding="utf-8")
                    spans = {match.span("version") for real in pins.PINS if path in real.paths
                             for match in real.pattern.finditer(before)}
                    self.assertEqual({before[start:end] for start, end in spans}, {current})
                    expected = before
                    for start, end in sorted(spans, reverse=True):
                        expected = expected[:start] + new + expected[end:]
                    self.assertEqual((root / path).read_text(encoding="utf-8"), expected)

    def test_every_install_snippet_in_the_docs_is_registered(self):
        """A snippet in a file its pin does not list would never be bumped."""
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "*.md", "*.mdx"],
                                 capture_output=True, text=True, check=True, timeout=60).stdout
        unregistered, unowned = [], []
        for path in filter(None, tracked.split("\0")):
            if HISTORY.search(path) or not (ROOT / path).is_file():
                continue
            text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
            for real in pins.PINS:
                if path not in real.paths and real.pattern.search(text):
                    unregistered.append(f"{path}: {real.label}")
            unowned += [f"{path}: {line}" for line in unowned_versions(text)]
        self.assertEqual(unregistered, [], "add these files to their pin in install_pins.PINS")
        self.assertEqual(unowned, [], "a package named with a version no pin owns: teach a PINS "
                                      "template this snippet shape, or reword prose that is not one")


if __name__ == "__main__":
    unittest.main()
