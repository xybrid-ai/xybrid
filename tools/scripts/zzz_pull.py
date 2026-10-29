#!/usr/bin/env python3
"""Workspace entry point for the verifier distributed with xybrid-zzz-sys."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "crates/zzz-sys/zzz_pull.py"),
        run_name="__main__",
    )
