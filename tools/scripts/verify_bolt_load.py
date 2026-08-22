#!/usr/bin/env python3
"""Load a deployed bolt native library and exercise a real boltffi call.

Proves a native plugin is *functional* — loads under the OS loader and completes
a live FFI round-trip (boltffi_function_xybrid_bolt_version -> owned wire buffer -> decode -> free) —
not merely present on disk. Used by the Windows Unity verify to guard the shipped
cargo/MSVC `xybrid_bolt.dll`; the Bazel windows-gnu DLL gets a fuller managed C#
smoke in `.github/workflows/bazel.yml`.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys


class FfiBuf(ctypes.Structure):
    """Mirrors the boltffi FfiBuf (bindings/unity/Runtime/Bolt/XybridBolt.cs)."""

    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("cap", ctypes.c_size_t),
        ("align", ctypes.c_size_t),
    ]


def _decode_wire_string(raw: bytes) -> str:
    """Decode a boltffi wire string: an i32 LE length prefix, then UTF-8 bytes.

    Mirrors WireReader.ReadString in XybridBolt.cs — the FfiBuf is wire-encoded,
    NOT a raw C string.
    """
    if len(raw) < 4:
        raise RuntimeError(f"undersized wire buffer ({len(raw)} bytes)")
    length = int.from_bytes(raw[0:4], "little", signed=True)
    if length < 0 or 4 + length > len(raw):
        raise RuntimeError(f"corrupt wire buffer (prefix len={length}, total={len(raw)})")
    return raw[4 : 4 + length].decode("utf-8")


def verify(lib_path: str) -> str:
    """Load `lib_path`, call the version export, and return the decoded version."""
    directory = os.path.dirname(os.path.abspath(lib_path))
    # Let the OS loader resolve sibling deps (e.g. a co-located onnxruntime.dll)
    # if the library references any. Keep the returned handle alive through the
    # native call; closing/discarding it removes the directory from Windows'
    # process search path. No-op on platforms without the API.
    dll_directory = (
        os.add_dll_directory(directory)
        if hasattr(os, "add_dll_directory") and os.path.isdir(directory)
        else contextlib.nullcontext()
    )
    with dll_directory:
        lib = ctypes.CDLL(os.path.abspath(lib_path))
        lib.boltffi_function_xybrid_bolt_version.restype = FfiBuf
        lib.boltffi_free_buf.argtypes = [FfiBuf]
        lib.boltffi_free_buf.restype = None

        buf = lib.boltffi_function_xybrid_bolt_version()
        try:
            if not buf.ptr or not buf.len:
                raise RuntimeError("the version export returned an empty buffer")
            raw = ctypes.string_at(buf.ptr, int(buf.len))
            version = _decode_wire_string(raw)
            if not version:
                raise RuntimeError("the version export decoded to an empty string")
            return version
        finally:
            lib.boltffi_free_buf(buf)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_bolt_load.py <path-to-native-lib>", file=sys.stderr)
        return 2
    try:
        version = verify(sys.argv[1])
    except (OSError, RuntimeError, UnicodeDecodeError) as error:
        print(f"verify-bolt-load: {error}", file=sys.stderr)
        return 1
    print(f"bolt native loads; version -> {version!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
