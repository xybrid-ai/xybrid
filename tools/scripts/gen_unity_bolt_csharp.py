#!/usr/bin/env python3
"""Regenerate the committed BoltFFI C# bindings for the Unity SDK.

The Unity SDK moved off the pre-bolt C ABI (the removed crates/xybrid-ffi +
csbindgen) onto xybrid-bolt, the same FFI every other binding already uses
(chapter A of the Bazel/bolt migration). Consumers of those bindings — like the
committed Swift in bindings/apple/Sources/Xybrid/xybrid_bolt.swift — check the
generated wire layer into the repo alongside a hand-written wrapper. This script
is the C# equivalent of the xtask copy step that stages the Swift.

boltffi 0.25.3's C# generator targets modern C#/.NET (C# 10, .NET 5+ BCL), but
Unity's scripting profile is frozen at C# 9 + netstandard2.1. So this script
down-levels the generated output to the largest subset Unity accepts. Each
transform below is guarded by a count assertion: if boltffi's output shape
changes, the script fails loudly rather than silently shipping code that will
not compile under Unity. The netstandard2.1 compile gate
(tools/unity-bolt-compile-check) verifies the result on every relevant PR.

What it does:

  1. `boltffi generate csharp` -> crates/xybrid-bolt/dist/csharp (git-ignored).
  2. Down-levels the four Unity incompatibilities boltffi 0.25.3 emits:
     a. `readonly record struct` (C# 10) -> plain `readonly struct` with an
        explicit ctor + get-only auto-properties. The generated structs are
        positional and use no record features (with/==/Deconstruct), so this is
        semantics-preserving.
     b. `BinaryPrimitives.WriteSingle/DoubleLittleEndian` (.NET 5 methods, absent
        from netstandard2.1) -> the integer variants over
        BitConverter.SingleToInt32Bits / DoubleToInt64Bits (both in
        netstandard2.1), a byte-identical little-endian encode.
     c. `init` accessors lower against System.Runtime.CompilerServices.
        IsExternalInit, a .NET 5 type Unity lacks -> a one-file internal
        polyfill is emitted (needed by the C# 9 record hierarchy in
        XybridError, whose positional records still synthesize `init`).
     d. `class XybridModel` -> `partial class XybridModel`, so the hand-ported
        inference path in bindings/unity/Runtime/BoltSupplement (which boltffi
        0.25.3 drops entirely -- run(), run_stream(), stream_result(), and the
        XybridEnvelope/EnvelopeKind/Result types) can
        extend it. That folder is hand-written and NOT touched by this script.
     e. Keeps the generated model wrapper alive across blocking `StreamNext`,
        preventing its finalizer from freeing the native model handle while
        Rust is waiting for the next token.
     f. `NativeMemory.Alloc` (.NET 6) in FfiBuf.FromBytes -> fail closed. That
        method hands an owned buffer to Rust, which frees it with its global
        allocator in boltffi_free_buf (std::alloc::dealloc) — a free no C#
        allocator matches on Windows (malloc/AllocHGlobal use the CRT heap;
        Rust frees on GetProcessHeap() = cross-heap free = UB). No generated
        entry point passes an owned FfiBuf into Rust today, so the body throws
        NotSupportedException; the correct fix if a call site is ever added is
        a Rust-side allocator (e.g. boltffi_alloc_buf).
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
import re
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

# --- Transform (a): readonly record struct (C# 10) -> plain readonly struct ---
# Matches a positional `public readonly record struct Name( params ) {`. The
# params never contain parentheses (simple `Type Name`, incl. `T?` / `T[]`), so
# `[^)]*` is a safe, greedy-free body match up to the closing paren.
RECORD_STRUCT_RE = re.compile(
    r"public readonly record struct (?P<name>\w+)\((?P<params>[^)]*)\)\s*\{",
    re.DOTALL,
)
# 9 since the speculative-cloud surface added XybridDownloadStatus (the
# download progress + state snapshot hosts poll while a model streams from the
# gateway). Bump this deliberately: the count is a tripwire for unreviewed
# boltffi output drift, not a value to auto-sync.
EXPECTED_RECORD_STRUCTS = 9

# --- Transform (b): .NET 5 float LE writers -> netstandard2.1-safe equivalents.
# BinaryPrimitives has the integer LE writers in netstandard2.1 but not the
# float ones; BitConverter.SingleToInt32Bits / DoubleToInt64Bits are both in
# netstandard2.1, so bit-reinterpret then write the integer. Byte-identical.
LE_FLOAT_REWRITES = (
    (
        "BinaryPrimitives.WriteSingleLittleEndian(_buffer.AsSpan(_pos), v)",
        "BinaryPrimitives.WriteInt32LittleEndian(_buffer.AsSpan(_pos), "
        "BitConverter.SingleToInt32Bits(v))",
    ),
    (
        "BinaryPrimitives.WriteDoubleLittleEndian(_buffer.AsSpan(_pos), v)",
        "BinaryPrimitives.WriteInt64LittleEndian(_buffer.AsSpan(_pos), "
        "BitConverter.DoubleToInt64Bits(v))",
    ),
)

# --- Transform (g): Unsafe.SizeOf<T>() -> Marshal.SizeOf<T>(). boltffi's wire
# codec sizes blittable arrays with System.Runtime.CompilerServices.Unsafe, whose
# assembly Unity's scripting profile does not reference -- the Unity 6.3 editor
# rejects it with `CS0103: The name 'Unsafe' does not exist`. The codecs are
# `where T : unmanaged`, so Marshal.SizeOf<T>() (core BCL, already used
# throughout the generated wire layer) returns the identical size. Verified with
# an in-editor `unity test` compile on 6.3.
UNSAFE_SIZEOF_TARGET = "Unsafe.SizeOf<T>()"
UNSAFE_SIZEOF_REPLACEMENT = "Marshal.SizeOf<T>()"
EXPECTED_UNSAFE_SIZEOF = 3

# --- Transform (d): make XybridModel partial. boltffi 0.25.3's C# generator
# drops the entire inference path -- no `run`, and no XybridEnvelope /
# XybridEnvelopeKind / XybridResult (a data-carrying enum used as a function
# INPUT, which the 0.25.3 C# lowering can't express; same class the Python
# generator drops). bindings/unity/Runtime/BoltSupplement/ hand-ports that path
# (wire codecs + a `partial class XybridModel` adding Run() / RunStreaming()), mirroring
# bindings/python/xybrid/_bolt.py. Marking the generated class partial lets the
# supplement extend it and reach its private _handle.
MODEL_FILE = "XybridModel.cs"
MODEL_PARTIAL_TARGET = "public sealed class XybridModel : IDisposable"
MODEL_PARTIAL_REPLACEMENT = "public sealed partial class XybridModel : IDisposable"

# XybridConversationContext is likewise extended in BoltSupplement with the
# envelope-input methods the generator drops (push / set_system), so it too
# must be partial.
CONTEXT_FILE = "XybridConversationContext.cs"
CONTEXT_PARTIAL_TARGET = "public sealed class XybridConversationContext : IDisposable"
CONTEXT_PARTIAL_REPLACEMENT = (
    "public sealed partial class XybridConversationContext : IDisposable"
)

# --- Transform (e): keep the managed model alive across blocking StreamNext.
STREAM_NEXT_FREE_TARGET = (
    "        public XybridStreamEvent StreamNext(ulong streamId)\n"
    "        {\n"
    "            ThrowIfDisposed();\n"
    "            FfiBuf _buf = NativeMethods.XybridModelStreamNext(_handle, streamId);\n"
    "            try\n"
    "            {\n"
    "                var reader = new WireReader(_buf);\n"
    "                if (reader.ReadU8() != 0) throw new XybridErrorException(XybridError.Decode(reader));\n"
    "                return XybridStreamEvent.Decode(reader);\n"
    "            }\n"
    "            finally\n"
    "            {\n"
    "                NativeMethods.FreeBuf(_buf);\n"
    "            }\n"
    "        }\n"
)
STREAM_NEXT_FREE_REPLACEMENT = STREAM_NEXT_FREE_TARGET.replace(
    "                NativeMethods.FreeBuf(_buf);\n",
    "                NativeMethods.FreeBuf(_buf);\n"
    "                // StreamNext can block for the full inter-token gap. Keep\n"
    "                // this wrapper alive so its finalizer cannot free _handle\n"
    "                // while the native call is still using it.\n"
    "                GC.KeepAlive(this);\n",
)

# --- Transform (c): IsExternalInit polyfill (a Unity-only supplement) ---
POLYFILL_FILE = "IsExternalInit.cs"
POLYFILL_CONTENT = (
    "// Unity C# 9 polyfill (written by tools/scripts/gen_unity_bolt_csharp.py;\n"
    "// not a BoltFFI output). The generated positional records use `init`\n"
    "// accessors, which the compiler lowers against\n"
    "// System.Runtime.CompilerServices.IsExternalInit -- a type .NET 5+ ships\n"
    "// but Unity's netstandard2.1 scripting profile does not. This empty shim\n"
    "// satisfies the reference so the records compile under Unity.\n"
    "namespace System.Runtime.CompilerServices\n"
    "{\n"
    "    internal static class IsExternalInit { }\n"
    "}\n"
)


def _downlevel_record_structs(content: str) -> tuple[str, int]:
    """Rewrite positional `readonly record struct`s to plain readonly structs."""
    count = 0

    def repl(match: "re.Match[str]") -> str:
        nonlocal count
        count += 1
        name = match.group("name")
        fields = [
            p.strip().rsplit(" ", 1)
            for p in match.group("params").split(",")
            if p.strip()
        ]
        ctor_params = ", ".join(f"{ty} {pn}" for ty, pn in fields)
        assigns = " ".join(f"this.{pn} = {pn};" for _, pn in fields)
        props = "\n".join(f"        public {ty} {pn} {{ get; }}" for ty, pn in fields)
        return (
            f"public readonly struct {name}\n"
            "    {\n"
            f"        public {name}({ctor_params}) {{ {assigns} }}\n"
            f"{props}\n"
        )

    return RECORD_STRUCT_RE.sub(repl, content), count


def _rewrite_le_floats(content: str) -> tuple[str, int]:
    """Rewrite the two .NET 5 float little-endian writers to netstandard2.1."""
    count = 0
    for old, new in LE_FLOAT_REWRITES:
        if old in content:
            content = content.replace(old, new, 1)
            count += 1
    return content, count


def _rewrite_unsafe_sizeof(content: str) -> tuple[str, int]:
    """Rewrite Unsafe.SizeOf<T>() to Marshal.SizeOf<T>().

    The Unsafe assembly is unreferenceable in Unity's scripting profile; the
    codecs are `where T : unmanaged`, so the two sizes are identical.
    """
    count = content.count(UNSAFE_SIZEOF_TARGET)
    if count:
        content = content.replace(UNSAFE_SIZEOF_TARGET, UNSAFE_SIZEOF_REPLACEMENT)
    return content, count


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


def _drift(condition: bool, message: str) -> None:
    """Fail loudly when a transform's expectation no longer matches the output."""
    if not condition:
        sys.exit(
            f"error: {message}\nboltffi output changed — review the transforms in "
            "tools/scripts/gen_unity_bolt_csharp.py before regenerating."
        )


def generate() -> dict[str, str]:
    """Run the generator, down-level for Unity, and return {filename: contents}."""
    subprocess.run(["boltffi", "generate", "csharp"], cwd=BOLT_DIR, check=True)

    sources = sorted(RAW_DIR.glob("*.cs"))
    if not sources:
        sys.exit(f"error: boltffi produced no .cs files in {RAW_DIR}")

    tree: dict[str, str] = {}
    record_structs = 0
    le_floats = 0
    unsafe_sizeof = 0
    native_memory_shimmed = False
    model_made_partial = False
    context_made_partial = False
    stream_next_kept_alive = False
    for src in sources:
        content = src.read_text(encoding="utf-8")
        content, n = _downlevel_record_structs(content)
        record_structs += n
        content, n = _rewrite_le_floats(content)
        le_floats += n
        content, n = _rewrite_unsafe_sizeof(content)
        unsafe_sizeof += n
        if src.name == SHIM_FILE:
            _drift(
                SHIM_TARGET in content,
                f"expected `NativeMemory.Alloc` block not found in {src.name}",
            )
            content = content.replace(SHIM_TARGET, SHIM_REPLACEMENT, 1)
            native_memory_shimmed = True
        if src.name == MODEL_FILE:
            _drift(
                MODEL_PARTIAL_TARGET in content,
                f"expected XybridModel class declaration not found in {src.name}",
            )
            content = content.replace(
                MODEL_PARTIAL_TARGET, MODEL_PARTIAL_REPLACEMENT, 1
            )
            model_made_partial = True
            _drift(
                STREAM_NEXT_FREE_TARGET in content,
                f"expected StreamNext body not found in {src.name}",
            )
            content = content.replace(
                STREAM_NEXT_FREE_TARGET, STREAM_NEXT_FREE_REPLACEMENT, 1
            )
            stream_next_kept_alive = True
        if src.name == CONTEXT_FILE:
            _drift(
                CONTEXT_PARTIAL_TARGET in content,
                f"expected XybridConversationContext class declaration not found in {src.name}",
            )
            content = content.replace(
                CONTEXT_PARTIAL_TARGET, CONTEXT_PARTIAL_REPLACEMENT, 1
            )
            context_made_partial = True
        tree[src.name] = content
        tree[src.name + ".meta"] = script_meta(f"{DEST_REL}/{src.name}")

    # Transform (c): emit the IsExternalInit polyfill next to the sources.
    tree[POLYFILL_FILE] = POLYFILL_CONTENT
    tree[POLYFILL_FILE + ".meta"] = script_meta(f"{DEST_REL}/{POLYFILL_FILE}")

    _drift(
        record_structs == EXPECTED_RECORD_STRUCTS,
        f"down-leveled {record_structs} record structs, expected "
        f"{EXPECTED_RECORD_STRUCTS}",
    )
    _drift(
        le_floats == len(LE_FLOAT_REWRITES),
        f"rewrote {le_floats} float LE writers, expected {len(LE_FLOAT_REWRITES)}",
    )
    _drift(
        unsafe_sizeof == EXPECTED_UNSAFE_SIZEOF,
        f"rewrote {unsafe_sizeof} Unsafe.SizeOf calls, expected "
        f"{EXPECTED_UNSAFE_SIZEOF}",
    )
    _drift(native_memory_shimmed, f"{SHIM_FILE} not found in boltffi output")
    _drift(model_made_partial, f"{MODEL_FILE} not found in boltffi output")
    _drift(stream_next_kept_alive, f"StreamNext not found in {MODEL_FILE}")
    _drift(context_made_partial, f"{CONTEXT_FILE} not found in boltffi output")
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
