"""Packaging shim: the wheel bundles a compiled bridge plus a native library.

``xybrid/_bolt/`` holds boltffi's generated wire layer, the ``_native`` CPython
extension it imports, and the ``xybrid-bolt`` cdylib that extension dlopens.
Both binaries are staged by ``tools/scripts/build-python-bolt.sh``.

Without this shim setuptools sees only Python sources and emits a universal
``py3-none-any`` wheel — wrong twice over, since the payload is specific to
both the platform *and* the interpreter ABI that compiled ``_native``. Wheels
are therefore tagged ``cp3XX-cp3XX-<plat>``, which is why the SDK requires
Python >= 3.10 and ships one wheel per interpreter. The build fails fast when
the binaries are missing. Editable installs from a checkout are unaffected.
"""

import os
import sys
import sysconfig
from pathlib import Path

from setuptools import setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1 keeps the command in the wheel package
    from wheel.bdist_wheel import bdist_wheel

_BOLT_DIR = Path(__file__).resolve().parent / "xybrid" / "_bolt"
_NATIVE_SUFFIXES = {".dylib", ".so", ".dll", ".pyd"}


class _NativeBdistWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        # With root_is_pure False this is already the running interpreter's
        # (cp314, cp314, <plat>) — keep the impl/ABI, only clamp the platform.
        impl, abi, plat = super().get_tag()
        # Honor an explicit --plat-name (e.g. a delocate/auditwheel repair flow
        # in CI) verbatim.
        if getattr(self, "plat_name_supplied", False):
            return impl, abi, plat
        # Otherwise don't pin the wheel to the build host's OS version:
        # bdist_wheel derives the macOS tag from the running system (e.g.
        # macosx_26_0), so an otherwise-compatible dylib becomes un-installable
        # on older clients. Clamp to the deployment-target floor
        # (MACOSX_DEPLOYMENT_TARGET, default 11.0 — rustc's aarch64-apple-darwin
        # default, which is what build-python-bolt.sh links against). Full
        # multi-version / multi-distro portability still needs delocate (macOS)
        # and auditwheel (manylinux) in CI — see the packaging follow-up.
        if plat.startswith("macosx_"):
            arch = plat.split("_", 3)[-1]
            target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")
            major, _, minor = target.partition(".")
            plat = f"macosx_{major}_{minor or '0'}_{arch}"
        return impl, abi, plat

    def run(self) -> None:
        libraries = {
            "darwin": "libxybrid_bolt.dylib",
            "linux": "libxybrid_bolt.so",
            "win32": "xybrid_bolt.dll",
        }
        if sys.implementation.name != "cpython" or sys.platform not in libraries:
            raise RuntimeError("xybrid wheels require CPython on macOS, Linux or Windows")
        expected = {libraries[sys.platform], f"_native{sysconfig.get_config_var('EXT_SUFFIX')}"}
        staged = {
            path.name for path in _BOLT_DIR.glob("*")
            if path.is_file() and path.suffix in _NATIVE_SUFFIXES
        }
        missing, stale = expected - staged, staged - expected
        if missing or stale:
            raise RuntimeError(
                f"xybrid: missing or mismatched native artifacts for interpreter {sys.executable} "
                f"(missing: {sorted(missing)}, stale: {sorted(stale)}). "
                "Run tools/scripts/build-python-bolt.sh with PYTHON set to this interpreter "
                "before building a wheel."
            )
        super().run()


setup(cmdclass={"bdist_wheel": _NativeBdistWheel})
