#!/usr/bin/env python3
"""Build and stage a matching BoltFFI bridge/library pair for this interpreter.

The shell entry point selects Python via PYTHON; that interpreter also builds
the bridge, extracts the wheel and verifies the installed SDK. Temporary wheel
output keeps unrelated interpreter builds out of artifact selection.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOLT_DIR = REPO_ROOT / "crates" / "xybrid-bolt"
SDK_DIR = REPO_ROOT / "bindings" / "python"
LIBRARIES = {"libxybrid_bolt.dylib", "libxybrid_bolt.so", "xybrid_bolt.dll"}


def stage_native_artifacts(wheel: Path, destination: Path, library: str) -> None:
    """Validate both wheel members before replacing any staged binaries."""

    bridge = f"_native{sysconfig.get_config_var('EXT_SUFFIX')}"
    names = (bridge, library)
    with zipfile.ZipFile(wheel) as archive:
        members = [f"xybrid_bolt/{name}" for name in names]
        missing = set(members) - set(archive.namelist())
        if missing:
            raise RuntimeError(f"wheel is missing artifacts for {sys.executable}: {sorted(missing)}")
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".native-", dir=destination) as temporary:
            staged = Path(temporary)
            for name, member in zip(names, members):
                with archive.open(member) as source, (staged / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
                (staged / name).chmod(0o755)
            for name in names:
                (staged / name).replace(destination / name)

    # A bridge for another Python version may resolve against incompatible
    # symbols after the shared library changes. Keep exactly this build's pair.
    for path in destination.iterdir():
        is_bridge = path.name.startswith("_native.") and path.suffix in {".so", ".pyd"}
        if path.is_file() and path.name not in names and (is_bridge or path.name in LIBRARIES):
            path.unlink()
            print(f"Removed obsolete build output: {path.name}", flush=True)


def main() -> None:
    if sys.platform == "darwin":
        default_features, library = "platform-macos", "libxybrid_bolt.dylib"
    elif sys.platform.startswith("linux"):
        default_features, library = "platform-desktop", "libxybrid_bolt.so"
    else:
        raise RuntimeError(f"unsupported host OS: {sys.platform}; expected macOS or Linux")
    if sys.implementation.name != "cpython" or sys.version_info < (3, 10):
        raise RuntimeError("the Python SDK requires CPython 3.10 or newer")

    # Prove the committed source matches the Rust surface and pinned CLI before
    # compiling, including the SDK's deterministic compatibility transforms.
    print("Checking committed Python bindings with the pinned BoltFFI CLI", flush=True)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/scripts/gen_python_bolt.py"), "--check"],
        cwd=REPO_ROOT,
        check=True,
    )
    features = os.environ.get("XYBRID_FEATURES", default_features)
    print(f"Building Python natives for {sys.executable} ({features})", flush=True)
    with tempfile.TemporaryDirectory(prefix="xybrid-python-") as temporary:
        output = Path(temporary)
        wheelhouse = output / "wheels"
        overlay = output / "boltffi.toml"
        overlay.write_text(
            f"[targets.python.wheel]\noutput = {json.dumps(str(wheelhouse))}\n",
            encoding="utf-8",
        )
        command = [
            "boltffi", "pack", "python", "--deny-skipped",
            "--python", sys.executable, "--overlay", str(overlay),
            "--cargo-arg=--features", f"--cargo-arg={features}",
        ]
        if os.environ.get("DEBUG") != "1":
            command.append("--release")
        subprocess.run(command, cwd=BOLT_DIR, check=True)
        wheels = sorted(wheelhouse.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel for {sys.executable}, found {len(wheels)}")
        stage_native_artifacts(wheels[0], SDK_DIR / "xybrid" / "_bolt", library)

    # Use a new process so a loaded extension cannot hide stale symbols.
    subprocess.run(
        [sys.executable, "-c", "import xybrid; print('Staged xybrid', xybrid.version())"],
        cwd=SDK_DIR,
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        sys.exit(f"error: {exc}")
