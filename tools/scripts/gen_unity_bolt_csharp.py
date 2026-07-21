#!/usr/bin/env python3
"""Regenerate the committed BoltFFI C# bindings for the Unity SDK.

The Unity SDK is migrating off the pre-bolt C ABI (crates/xybrid-ffi +
csbindgen) onto xybrid-bolt, the same FFI every other binding already uses
(chapter A of the Bazel/bolt migration; this is step A0 — land the generated
sources). Consumers of those bindings — like the committed Swift in
bindings/apple/Sources/Xybrid/xybrid_bolt.swift — check the generated wire
layer into the repo alongside a hand-written wrapper. This script is the C#
equivalent of the xtask copy step that stages the Swift.

What it does:

  1. `boltffi generate csharp` -> crates/xybrid-bolt/dist/csharp (git-ignored).
  2. Applies ONE deterministic post-process: boltffi 0.25.3 emits
     `NativeMemory.Alloc` in FfiBuf.FromBytes, a .NET 6 API that does not exist
     in Unity's Mono/IL2CPP scripting profile. That method also hands an owned
     buffer to Rust, which frees it with its global allocator in
     boltffi_free_buf (std::alloc::dealloc) — a free no C# allocator matches on
     every platform (on Windows, malloc/AllocHGlobal use the CRT heap while
     Rust frees on GetProcessHeap(): a cross-heap free = UB). No generated
     entry point passes an owned FfiBuf into Rust today, so rather than ship a
     latent cross-allocator bug the body is rewritten to fail closed
     (NotSupportedException). This keeps the file compiling and defers the
     correct fix (a Rust-side allocator) to the point a call site is added.
  3. Writes deterministic Unity .meta files (GUID = sha256(asset path)[:32],
     the same scheme as stage_unity_desktop_ort.py).
  4. Syncs the result into bindings/unity/Runtime/Bolt, pruning stale files.

Usage:
    python3 tools/scripts/gen_unity_bolt_csharp.py            # regenerate + write
    python3 tools/scripts/gen_unity_bolt_csharp.py --check    # fail on drift

Requires the pinned `boltffi` CLI (0.25.3) on PATH:
    cargo install boltffi_cli --version 0.25.3 --locked
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOLT_DIR = REPO_ROOT / "crates" / "xybrid-bolt"
RAW_DIR = BOLT_DIR / "dist" / "csharp"
DEST_DIR = REPO_ROOT / "bindings" / "unity" / "Runtime" / "Bolt"
# Path used to key deterministic GUIDs and to build the folder .meta location.
DEST_REL = "bindings/unity/Runtime/Bolt"
PINNED_BOLTFFI = "0.25.3"

# The exact non-empty-input block boltffi 0.25.3 emits, and its Unity-safe
# replacement. Kept as a literal so a boltffi output change trips the assertion
# below instead of silently shipping a file that will not compile under Unity.
# The whole block (alloc + copy + return) is replaced so no unreachable code is
# left behind the `throw`.
SHIM_TARGET = (
    "            void* allocated = NativeMemory.Alloc((nuint)bytes.Length);\n"
    "            Marshal.Copy(bytes, 0, (IntPtr)allocated, bytes.Length);\n"
    "            return new FfiBuf\n"
    "            {\n"
    "                ptr = (IntPtr)allocated,\n"
    "                len = (UIntPtr)bytes.Length,\n"
    "                cap = (UIntPtr)bytes.Length,\n"
    "                align = (UIntPtr)1,\n"
    "            };\n"
)
SHIM_REPLACEMENT = (
    "            // xybrid Unity shim (tools/scripts/gen_unity_bolt_csharp.py):\n"
    "            // boltffi 0.25.3 emits `NativeMemory.Alloc` here, a .NET 6 API\n"
    "            // absent from Unity's Mono/IL2CPP scripting profile. It also\n"
    "            // hands an owned buffer to Rust, which frees it with its global\n"
    "            // allocator in boltffi_free_buf (std::alloc::dealloc) -- a free\n"
    "            // no C# allocator matches on Windows (malloc/AllocHGlobal use\n"
    "            // the CRT heap; Rust frees on GetProcessHeap() = cross-heap free\n"
    "            // = UB). No generated entry point passes an owned FfiBuf into\n"
    "            // Rust today, so we fail closed rather than ship latent UB. When\n"
    "            // a call site is added, expose a Rust-side allocator (e.g.\n"
    "            // boltffi_alloc_buf) so alloc and free share one allocator.\n"
    "            throw new NotSupportedException(\n"
    "                \"FfiBuf.FromBytes: passing an owned buffer from C# into Rust\"\n"
    "                + \" is not supported by the Unity binding; boltffi_free_buf\"\n"
    "                + \" would free it with Rust's allocator, which no C#\"\n"
    "                + \" allocator matches on Windows. Expose a Rust-side\"\n"
    "                + \" allocator before using this path.\");\n"
)
SHIM_FILE = "XybridBolt.cs"


def unity_guid(asset_rel_path: str) -> str:
    """Deterministic 32-hex Unity GUID keyed on the repo-relative asset path."""
    return hashlib.sha256(asset_rel_path.encode()).hexdigest()[:32]


def script_meta(asset_rel_path: str) -> str:
    """A minimal Unity script .meta, matching the repo's existing .cs.meta form."""
    return f"fileFormatVersion: 2\nguid: {unity_guid(asset_rel_path)}"


def folder_meta(asset_rel_path: str) -> str:
    return (
        "fileFormatVersion: 2\n"
        f"guid: {unity_guid(asset_rel_path)}\n"
        "folderAsset: yes\n"
        "DefaultImporter:\n"
        "  externalObjects: {}\n"
        "  userData: \n"
        "  assetBundleName: \n"
        "  assetBundleVariant: \n"
    )


def check_boltffi() -> None:
    try:
        out = subprocess.run(
            ["boltffi", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        sys.exit(
            "error: `boltffi` CLI not found. Install the pinned version:\n"
            f"  cargo install boltffi_cli --version {PINNED_BOLTFFI} --locked\n"
            f"({exc})"
        )
    if PINNED_BOLTFFI not in out:
        print(
            f"warning: expected boltffi {PINNED_BOLTFFI}, got '{out}'. "
            "Generated output may differ from the committed sources.",
            file=sys.stderr,
        )


def generate() -> dict[str, str]:
    """Run the generator and return {relative filename: shimmed contents}."""
    subprocess.run(["boltffi", "generate", "csharp"], cwd=BOLT_DIR, check=True)

    sources = sorted(RAW_DIR.glob("*.cs"))
    if not sources:
        sys.exit(f"error: boltffi produced no .cs files in {RAW_DIR}")

    tree: dict[str, str] = {}
    for src in sources:
        content = src.read_text(encoding="utf-8")
        if src.name == SHIM_FILE:
            if SHIM_TARGET not in content:
                sys.exit(
                    f"error: expected `NativeMemory.Alloc` line not found in "
                    f"{src.name}. boltffi output changed — update SHIM_TARGET in "
                    "tools/scripts/gen_unity_bolt_csharp.py."
                )
            content = content.replace(SHIM_TARGET, SHIM_REPLACEMENT, 1)
        tree[src.name] = content
        tree[src.name + ".meta"] = script_meta(f"{DEST_REL}/{src.name}")
    return tree


def do_check(tree: dict[str, str]) -> int:
    drift: list[str] = []
    expected_files = set(tree)
    committed_files = {
        p.name for p in DEST_DIR.iterdir() if p.is_file()
    } if DEST_DIR.is_dir() else set()

    for name, expected in tree.items():
        path = DEST_DIR / name
        if not path.exists():
            drift.append(f"  missing: {DEST_REL}/{name}")
        elif path.read_text(encoding="utf-8") != expected:
            drift.append(f"  stale:   {DEST_REL}/{name}")
    for name in sorted(committed_files - expected_files):
        drift.append(f"  extra:   {DEST_REL}/{name}")

    folder_meta_path = REPO_ROOT / f"{DEST_REL}.meta"
    if (not folder_meta_path.exists()
            or folder_meta_path.read_text(encoding="utf-8") != folder_meta(DEST_REL)):
        drift.append(f"  stale:   {DEST_REL}.meta")

    if drift:
        print(
            "Unity bolt C# bindings are out of date. Run:\n"
            "  python3 tools/scripts/gen_unity_bolt_csharp.py\n"
            "Drift:\n" + "\n".join(drift),
            file=sys.stderr,
        )
        return 1
    print("Unity bolt C# bindings are up to date.")
    return 0


def do_write(tree: dict[str, str]) -> int:
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True)
    for name, content in sorted(tree.items()):
        (DEST_DIR / name).write_text(content, encoding="utf-8")
    (REPO_ROOT / f"{DEST_REL}.meta").write_text(
        folder_meta(DEST_REL), encoding="utf-8"
    )
    print(
        f"Wrote {len([n for n in tree if n.endswith('.cs')])} C# files "
        f"(+ .meta) to {DEST_REL}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed sources are current; exit 1 on drift",
    )
    args = parser.parse_args()

    check_boltffi()
    tree = generate()
    return do_check(tree) if args.check else do_write(tree)


if __name__ == "__main__":
    sys.exit(main())
