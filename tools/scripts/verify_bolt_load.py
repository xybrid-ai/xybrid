#!/usr/bin/env python3
"""Load a deployed bolt native library and exercise a real boltffi call.

Proves a native plugin is *functional* — loads under the OS loader and completes
a live FFI round-trip (boltffi_version -> owned buffer -> free) — not merely
present on disk. Used by the Windows Unity verify to guard the shipped
cargo/MSVC `xybrid_bolt.dll`; the Bazel windows-gnu DLL gets a fuller managed C#
smoke in `.github/workflows/bazel.yml`.
"""

from __future__ import annotations

import ctypes
import os
import sys


class FfiBuf(ctypes.Structure):
    """Mirrors the boltffi FfiBuf (bindings/unity/Runtime/Bolt/XybridBolt.cs)."""

    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("cap", ctypes.c_size_t),
    ]


def verify(lib_path: str) -> str:
    """Load `lib_path`, call boltffi_version, and return the version string."""
    directory = os.path.dirname(os.path.abspath(lib_path))
    # Let the OS loader resolve sibling deps (e.g. a co-located onnxruntime.dll)
    # if the library references any. No-op on platforms without the API.
    if hasattr(os, "add_dll_directory") and os.path.isdir(directory):
        os.add_dll_directory(directory)

    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.boltffi_version.restype = FfiBuf
    lib.boltffi_free_buf.argtypes = [FfiBuf]
    lib.boltffi_free_buf.restype = None

    buf = lib.boltffi_version()
    try:
        if not buf.ptr or not buf.len:
            raise RuntimeError("boltffi_version returned an empty buffer")
        version = ctypes.string_at(buf.ptr, buf.len).decode("utf-8")
        if not version:
            raise RuntimeError("boltffi_version decoded to an empty string")
        return version
    finally:
        lib.boltffi_free_buf(buf)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_bolt_load.py <path-to-native-lib>", file=sys.stderr)
        return 2
    try:
        version = verify(sys.argv[1])
    except (OSError, RuntimeError) as error:
        print(f"verify-bolt-load: {error}", file=sys.stderr)
        return 1
    print(f"bolt native loads; boltffi_version -> {version!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
