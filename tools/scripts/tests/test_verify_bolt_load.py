"""Unit tests for the boltffi smoke helpers (no native library required)."""

import ctypes
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify_bolt_load import FfiBuf, _decode_wire_string


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


if __name__ == "__main__":
    unittest.main()
