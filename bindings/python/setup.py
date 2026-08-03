"""Packaging shim: the wheel bundles a platform-specific native library.

Without this, setuptools treats the package as pure Python and emits a
universal ``py3-none-any`` wheel that either carries a single-platform
``xybrid/_native`` library (broken everywhere else) or, from a fresh
checkout, no library at all. Wheels are therefore tagged ``py3-none-<plat>``
(ctypes code is Python-version independent) and the build fails fast when
``xybrid/_native`` is empty. Editable installs are unaffected — the loader
falls back to the workspace ``target/`` directory during development.
"""

import os
from pathlib import Path

from setuptools import setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1 keeps the command in the wheel package
    from wheel.bdist_wheel import bdist_wheel

_NATIVE_DIR = Path(__file__).resolve().parent / "xybrid" / "_native"
_NATIVE_SUFFIXES = {".dylib", ".so", ".dll"}


class _NativeBdistWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, plat = super().get_tag()
        # Honor an explicit --plat-name (e.g. a delocate/auditwheel repair flow
        # in CI) verbatim.
        if getattr(self, "plat_name_supplied", False):
            return "py3", "none", plat
        # Otherwise don't pin the wheel to the build host's OS version:
        # bdist_wheel derives the macOS tag from the running system (e.g.
        # macosx_26_0), so an otherwise-compatible dylib becomes un-installable
        # on older clients. Clamp to the deployment-target floor
        # (MACOSX_DEPLOYMENT_TARGET, default 11.0 — rustc's aarch64-apple-darwin
        # default, which is what build-python-bolt.sh links against). Full
        # multi-version / multi-distro portability still needs delocate (macOS)
        # and auditwheel (manylinux) in CI — see the packaging follow-up.
        if plat.startswith("macosx_"):
            arch = plat.rsplit("_", 1)[-1]
            target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")
            major, _, minor = target.partition(".")
            plat = f"macosx_{major}_{minor or '0'}_{arch}"
        return "py3", "none", plat

    def run(self) -> None:
        if not any(p.suffix in _NATIVE_SUFFIXES for p in _NATIVE_DIR.iterdir()):
            raise RuntimeError(
                "xybrid: no native library in xybrid/_native/ — run "
                "tools/scripts/build-python-bolt.sh before building a wheel"
            )
        super().run()


setup(cmdclass={"bdist_wheel": _NativeBdistWheel})
