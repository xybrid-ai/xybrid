"""Packaging shim: the wheel bundles a platform-specific native library.

Without this, setuptools treats the package as pure Python and emits a
universal ``py3-none-any`` wheel that either carries a single-platform
``xybrid/_native`` library (broken everywhere else) or, from a fresh
checkout, no library at all. Wheels are therefore tagged ``py3-none-<plat>``
(ctypes code is Python-version independent) and the build fails fast when
``xybrid/_native`` is empty. Editable installs are unaffected — the loader
falls back to the workspace ``target/`` directory during development.
"""

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
        return "py3", "none", plat

    def run(self) -> None:
        if not any(p.suffix in _NATIVE_SUFFIXES for p in _NATIVE_DIR.iterdir()):
            raise RuntimeError(
                "xybrid: no native library in xybrid/_native/ — run "
                "tools/scripts/build-python-bolt.sh before building a wheel"
            )
        super().run()


setup(cmdclass={"bdist_wheel": _NativeBdistWheel})
