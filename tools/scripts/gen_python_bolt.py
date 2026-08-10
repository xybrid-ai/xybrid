#!/usr/bin/env python3
"""Regenerate the committed BoltFFI Python bindings for the Python SDK.

boltffi 0.29's Python target emits a real package — pure-Python wire layer plus
a compiled CPython bridge (`_native.c`) — so `bindings/python/xybrid/_bolt.py`
is no longer hand-written. Before 0.29 it had to be: the 0.25.3 generator could
not express handle types or fallible functions, so the wire layer was hand-ported
against the generated Swift and drifted on every core change.

This script is the Python sibling of `bindings/apple/scripts/gen-bolt-bindings.sh`
and `gen_unity_bolt_csharp.py`: it runs the generator and syncs the result into
the SDK package, so the checked-in bindings always match the crate.

What it does:

  1. `boltffi generate python` -> crates/xybrid-bolt/dist/python (git-ignored).
  2. Copies the generated package into bindings/python/xybrid/_bolt/, which the
     hand-written `xybrid/__init__.py` wrapper re-exports.
  3. Prunes stale files so a removed export cannot linger.

The compiled extension is NOT built here — `tools/scripts/build-python-bolt.sh`
runs `boltffi pack python`, which compiles `_native` and stages the cdylib.

Usage:
    python3 tools/scripts/gen_python_bolt.py            # regenerate + write
    python3 tools/scripts/gen_python_bolt.py --check    # fail on drift
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOLT_DIR = REPO_ROOT / "crates" / "xybrid-bolt"
RAW_DIR = BOLT_DIR / "dist" / "python" / "xybrid_bolt"
DEST_DIR = REPO_ROOT / "bindings" / "python" / "xybrid" / "_bolt"
PINNED_BOLTFFI = "0.29.3"

# Generated sources to publish. `_native.c` is the CPython bridge that
# build-python-bolt.sh compiles; the rest is the pure-Python wire layer.
TRACKED_SUFFIXES = (".py", ".pyi", ".c", ".typed")

# Binaries build-python-bolt.sh stages into the same directory: the compiled
# bridge and the cdylib it dlopens. They are git-ignored build outputs, not
# generator output, so they must survive both the prune and the drift check —
# otherwise regenerating silently breaks `import xybrid`.
STAGED_SUFFIXES = (".dylib", ".so", ".dll", ".pyd")


def check_boltffi_version() -> None:
    try:
        out = subprocess.run(
            ["boltffi", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        sys.exit(
            "error: `boltffi` CLI not found. Install the pinned version:\n"
            f"  cargo install boltffi_cli --version {PINNED_BOLTFFI} --locked\n"
            f"({exc})"
        )
    if PINNED_BOLTFFI not in out:
        print(
            f"warning: expected boltffi {PINNED_BOLTFFI}, got '{out}'. "
            "Generated output may differ from the committed sources.",
            file=sys.stderr,
        )


def generate() -> list[Path]:
    subprocess.run(["boltffi", "generate", "python"], cwd=BOLT_DIR, check=True)
    sources = sorted(
        p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix in TRACKED_SUFFIXES
    )
    if not sources:
        sys.exit(f"error: boltffi produced no Python sources in {RAW_DIR}")
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed bindings differ from freshly generated ones",
    )
    args = parser.parse_args()

    check_boltffi_version()
    sources = generate()
    names = {p.name for p in sources}

    if args.check:
        stale = {p.name for p in DEST_DIR.glob("*") if p.is_file() and p.suffix not in STAGED_SUFFIXES} - names
        differing = [
            p.name
            for p in sources
            if not (DEST_DIR / p.name).exists()
            or not filecmp.cmp(p, DEST_DIR / p.name, shallow=False)
        ]
        if stale or differing:
            print(
                "error: committed Python bindings are out of date "
                f"(changed: {sorted(differing)}, stale: {sorted(stale)}).\n"
                "Run: python3 tools/scripts/gen_python_bolt.py",
                file=sys.stderr,
            )
            return 1
        print(f"Python bindings up to date ({len(sources)} files)")
        return 0

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for path in DEST_DIR.glob("*"):
        if path.is_file() and path.name not in names and path.suffix not in STAGED_SUFFIXES:
            path.unlink()
    for src in sources:
        shutil.copy2(src, DEST_DIR / src.name)

    print(f"Wrote {len(sources)} Python files to {DEST_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
