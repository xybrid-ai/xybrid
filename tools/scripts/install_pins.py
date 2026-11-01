#!/usr/bin/env python3
"""Read or set the SDK version in the install snippets of user-facing docs.

A user who copies `xybrid_flutter: ^X`, `ai.xybrid:xybrid-kotlin:X` or
`xybrid = "X"` from a README installs release X. A bump used to rewrite only
the Unity README's pins, so during 0.16.0 the pub.dev page still said ^0.6.0
and the crates.io page "0.2". PINS lists every snippet shape and the files that
carry it; version-sync.sh runs `check` for --check and `set` in both of its set
modes.

Each (file, pattern) pin:

  * reports every distinct version it finds, so a stale duplicate fails;
  * fails when it finds nothing, so a docs refactor that drops or reshapes a
    snippet surfaces here instead of leaving the pin unowned;
  * rewrites only its `version` group. The patterns carry the snippet's own
    syntax, so prose such as "Releases 0.16.0 and later" never matches.

Pins follow the workspace version, prereleases included: an rc cut points every
snippet at the rc, and since its release PR merges to master, the READMEs and
docs.xybrid.dev name the rc until the final release moves them on.

A new install snippet goes into PINS. tests/test_install_pins.py fails while a
tracked doc carries a snippet its pin does not list, or names a package with a
version that no pin owns; CI's Version sync job runs it on every pull request.
Standard library only, and Python 3.9 compatible: version-sync.sh runs it on
bare CI runners and laptops.

Usage:
    python3 tools/scripts/install_pins.py check VERSION   # exit 1 on drift
    python3 tools/scripts/install_pins.py set VERSION     # rewrite every pin
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# 0.16.0, 0.1.0-beta12, 1.0.0-rc.1: a leading digit, so `set` refuses a tag
# such as v0.16.0 that no pattern would find again, and dot-separated parts
# that are never empty, so a sentence's closing period stays out of it.
VERSION = r"[0-9][0-9A-Za-z+-]*(?:\.[0-9A-Za-z+-]+)*"

_README = "README.md"
_README_ZH = "README.zh-CN.md"
_README_JA = "README.ja-JP.md"
_QUICKSTART = "docs/content/docs/quickstart.mdx"
_QUICKSTART_ZH = "docs/content/docs/quickstart.zh.mdx"
_SDKS = "docs/content/docs/(integration)/sdks/"


@dataclass(frozen=True)
class Pin:
    """One install-snippet shape and the files that must carry it."""

    label: str
    # A regex with a `version` group; `{version}` is replaced by VERSION.
    template: str
    paths: Tuple[str, ...]

    @property
    def pattern(self) -> re.Pattern:
        return re.compile(self.template.replace("{version}", f"(?P<version>{VERSION})"), re.M)


# Each template takes the spellings a user could paste or a docs edit could
# produce: quoted or bare, any spacing, and every dependency form the tool
# accepts. test_install_pins.py flags a package name followed by a version that
# no pin owns, so a form these miss surfaces instead of drifting.
PINS: Tuple[Pin, ...] = (
    # `xybrid_flutter: ^X`, quoted or not, or under a `version:` key.
    Pin(
        "Flutter pubspec",
        r"""^[ \t]*xybrid_flutter:[ \t]*(?:\n[ \t]+version:[ \t]*)?["']?(?:\^|>=[ \t]*)?{version}""",
        (_README, _README_ZH, _README_JA, "bindings/flutter/README.md",
         _SDKS + "flutter.mdx", _SDKS + "flutter.zh.mdx", _QUICKSTART, _QUICKSTART_ZH),
    ),
    # The coordinates, a version-catalog or map entry, or a Maven <version>.
    Pin(
        "Kotlin Gradle",
        r"""(?:ai\.xybrid:xybrid-kotlin(?::|["'][ \t]*,[ \t]*version[ \t]*[=:][ \t]*["'])"""
        r"""|\bname[ \t]*[=:][ \t]*["']xybrid-kotlin["'][ \t]*,[ \t]*version[ \t]*[=:][ \t]*["']"""
        r"""|<artifactId>xybrid-kotlin</artifactId>\s*<version>){version}""",
        (_README, _README_ZH, _README_JA, "bindings/kotlin/README.md",
         _SDKS + "android.mdx", _SDKS + "android.zh.mdx", _QUICKSTART, _QUICKSTART_ZH),
    ),
    # from:, exact:, .upToNextMajor/.upToNextMinor(from:), .exact() or the
    # lower bound of a range, on one line or several.
    Pin(
        "SwiftPM",
        r'\.package\(\s*(?:name:\s*"[^"\n]*"\s*,\s*)?url:\s*"https://github\.com/xybrid-ai/xybrid(?:\.git)?"\s*,\s*'
        r'(?:(?:from|exact):\s*|\.(?:upToNextMajor|upToNextMinor)\(\s*from:\s*|\.exact\(\s*)?"{version}"',
        (_README, _README_ZH, _README_JA, "bindings/apple/README.md",
         _SDKS + "ios.mdx", _SDKS + "ios.zh.mdx", _QUICKSTART, _QUICKSTART_ZH),
    ),
    Pin(
        "Xcode dependency rule",
        r"(?:Up to Next (?:Major|Minor)|Exact) Version(?:\*\*)?[ \t]*(?:→|->)[ \t]*`{version}`",
        ("bindings/apple/README.md",),
    ),
    # `npm install @xybrid/react-native@X`, or a package.json dependency.
    Pin(
        "React Native npm",
        r"""@xybrid/react-native(?:@|["'][ \t]*:[ \t]*["'][\^~]?){version}""",
        (_README, _SDKS + "react-native.mdx", _QUICKSTART, _QUICKSTART_ZH),
    ),
    # The manifest dependency or `openupm add ai.xybrid.sdk@X`. The README's
    # local-development `"ai.xybrid.sdk": "file:..."` dependency does not start
    # with a digit, so it is not a pin.
    Pin(
        "Unity OpenUPM",
        r"""(?:"ai\.xybrid\.sdk"[ \t]*:[ \t]*"[\^~]?|\bai\.xybrid\.sdk@){version}""",
        ("bindings/unity/README.md",),
    ),
    # The git URL with its tag, or the tag alone in prose that says to append
    # it to the URL (the quickstart's Unity tab).
    Pin(
        "Unity git tag",
        r"(?:\?path=/?bindings/unity/?|`)#v{version}",
        ("bindings/unity/README.md", _QUICKSTART, _QUICKSTART_ZH),
    ),
    # `xybrid = "X"`, an inline table, a [dependencies.xybrid] table, or
    # `cargo add xybrid@X`.
    Pin(
        "Cargo",
        r'(?:^[ \t]*xybrid[ \t]*=[ \t]*(?:\{[^}\n]*?\bversion[ \t]*=[ \t]*)?"[\^~=]?'
        r'|^\[dependencies\.xybrid\][ \t]*\n(?:(?!\[).*\n)*?[ \t]*version[ \t]*=[ \t]*"[\^~=]?'
        r"|\bcargo add xybrid@){version}",
        (_README, _README_ZH, _README_JA, "crates/xybrid/README.md", _QUICKSTART, _QUICKSTART_ZH),
    ),
)


class PinError(Exception):
    """A pin could not be rewritten; nothing was written."""


def registrations(pins: Sequence[Pin] = PINS) -> Iterator[Tuple[Pin, str]]:
    """Every (pin, path) pair, in report order."""
    for pin in pins:
        for path in pin.paths:
            yield pin, path


def _read(path: Path) -> str:
    # newline="": CRLF and other line endings survive a rewrite.
    with path.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def found_versions(text: str, pin: Pin) -> List[str]:
    """Every distinct version the pin names in `text`, sorted."""
    return sorted({match.group("version") for match in pin.pattern.finditer(text)})


def check(root: Path, expected: str, pins: Sequence[Pin] = PINS) -> Tuple[bool, List[str]]:
    """Return (all pins name `expected`, one version-sync.sh report line per pin)."""
    ok, lines = True, []
    for pin, path in registrations(pins):
        file = root / path
        if not file.is_file():
            versions, found = [], "FILE MISSING"
        else:
            versions = found_versions(_read(file), pin)
            found = " ".join(versions) or "NOT FOUND"
        name = f"{pin.label} pin ({path})"
        if versions == [expected]:
            lines.append(f"  {name}: {found} ✓")
        else:
            ok = False
            lines.append(f"  {name}: {found} ✗ (expected {expected})")
    return ok, lines


def rewrite(root: Path, version: str, pins: Sequence[Pin] = PINS) -> Dict[str, str]:
    """The new text of every registered file, with each pin set to `version`.

    Raises PinError, naming every problem, when `version` is malformed or a
    pin finds nothing in a file it is registered for.
    """
    if not re.fullmatch(VERSION, version):
        raise PinError(f"install pin: {version!r} is not a version (no leading v)")

    def splice(match: re.Match) -> str:
        start, end = match.span("version")
        return match.string[match.start():start] + version + match.string[end:match.end()]

    texts: Dict[str, Optional[str]] = {}
    problems = []
    for pin, path in registrations(pins):
        if path not in texts:
            file = root / path
            texts[path] = _read(file) if file.is_file() else None
        text = texts[path]
        if text is None:
            problems.append(f"{pin.label} install pin: {path} is missing")
            continue
        texts[path], count = pin.pattern.subn(splice, text)
        if count == 0:
            problems.append(f"{pin.label} install pin not found in {path}")
    if problems:
        raise PinError("\n".join(problems))
    return {path: text for path, text in texts.items() if text is not None}


def set_version(root: Path, version: str, pins: Sequence[Pin] = PINS) -> None:
    """Set every pin to `version`. Writes nothing unless every pin is found."""
    for path, text in rewrite(root, version, pins).items():
        file = root / path
        if _read(file) != text:
            with file.open("w", encoding="utf-8", newline="") as stream:
                stream.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "set"))
    parser.add_argument("version")
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="repository root (default: the one holding this script)")
    args = parser.parse_args(argv)
    if args.command == "check":
        ok, lines = check(args.root, args.version)
        # The ✓/✗ marks, whatever the locale's encoding.
        sys.stdout.reconfigure(encoding="utf-8")
        print("\n".join(lines))
        return 0 if ok else 1
    try:
        set_version(args.root, args.version)
    except PinError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
