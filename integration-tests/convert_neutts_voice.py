#!/usr/bin/env python3
"""Convert a NeuTTS voice .pt file to xybrid PrecomputedCodes binary format.

Format: u32 count (little-endian) followed by `count` i32 values (little-endian).
This matches what `TtsVoiceLoader::load_reference_codes` expects.

Usage: convert_neutts_voice.py <input.pt> <output.bin>
"""
import struct
import sys
from pathlib import Path

import torch


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if not src.is_file():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 1

    tensor = torch.load(src, map_location="cpu", weights_only=True)
    codes = tensor.flatten().tolist()

    with dst.open("wb") as f:
        f.write(struct.pack("<I", len(codes)))
        for c in codes:
            f.write(struct.pack("<i", int(c)))

    print(f"  → {dst.name}: {len(codes)} codes ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
