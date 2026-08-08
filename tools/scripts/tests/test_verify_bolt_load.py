"""Unit tests for the boltffi smoke helpers (no native library required)."""

import ctypes
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify_bolt_load import FfiBuf, _decode_wire_string, verify


def wire(text: str) -> bytes:
    payload = text.encode("utf-8")
    return len(payload).to_bytes(4, "little", signed=True) + payload


class DecodeWireStringTests(unittest.TestCase):
    def test_decodes_length_prefixed_utf8(self):
        # The exact bytes the CI smoke first returned: [i32 5]["0.0.0"].
        self.assertEqual(_decode_wire_string(bytes([5, 0, 0, 0]) + b"0.0.0"), "0.0.0")

    def test_roundtrips_typical_version(self):
        self.assertEqual(_decode_wire_string(wire("0.1.0-beta12")), "0.1.0-beta12")

    def test_zero_length_is_empty(self):
        self.assertEqual(_decode_wire_string(bytes([0, 0, 0, 0])), "")

    def test_undersized_buffer_rejected(self):
        with self.assertRaises(RuntimeError):
            _decode_wire_string(bytes([1, 0]))

    def test_overlong_prefix_rejected(self):
        # prefix claims 9 bytes but only 5 follow → corrupt.
        with self.assertRaises(RuntimeError):
            _decode_wire_string(bytes([9, 0, 0, 0]) + b"0.0.0")

    def test_negative_prefix_rejected(self):
        with self.assertRaises(RuntimeError):
            _decode_wire_string(bytes([0xFF, 0xFF, 0xFF, 0xFF]) + b"junk")

    def test_ignores_trailing_bytes(self):
        # A well-formed prefix + payload is decoded even if the buffer is longer.
        self.assertEqual(_decode_wire_string(wire("ok") + b"\x00\x00"), "ok")


class FfiBufLayoutTests(unittest.TestCase):
    def test_matches_four_word_boltffi_abi(self):
        self.assertEqual(
            [name for name, _field_type in FfiBuf._fields_],
            ["ptr", "len", "cap", "align"],
        )
        self.assertEqual(ctypes.sizeof(FfiBuf), 4 * ctypes.sizeof(ctypes.c_size_t))


class VerifyLibraryTests(unittest.TestCase):
    def test_keeps_windows_dll_search_directory_active_through_call(self):
        raw = ctypes.create_string_buffer(wire("0.1.0"))

        class FakeFunction:
            def __init__(self, result=None):
                self.result = result

            def __call__(self, *_args):
                return self.result

        # `spec` pins the export names: a plain MagicMock answers to any
        # attribute, so this test kept passing across the boltffi 0.29 symbol
        # rename while the script could no longer load a real DLL.
        fake_lib = MagicMock(spec=["boltffi_function_xybrid_bolt_version", "boltffi_free_buf"])
        fake_lib.boltffi_function_xybrid_bolt_version = FakeFunction(
            FfiBuf(ctypes.addressof(raw), len(raw.raw) - 1, len(raw.raw) - 1, 1)
        )
        fake_lib.boltffi_free_buf = FakeFunction()
        directory_handle = MagicMock()

        with tempfile.TemporaryDirectory() as directory:
            library = str(Path(directory) / "xybrid_bolt.dll")
            with (
                patch.object(
                    os,
                    "add_dll_directory",
                    return_value=directory_handle,
                    create=True,
                ),
                patch.object(ctypes, "CDLL", return_value=fake_lib),
            ):
                self.assertEqual(verify(library), "0.1.0")

        directory_handle.__enter__.assert_called_once_with()
        directory_handle.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
