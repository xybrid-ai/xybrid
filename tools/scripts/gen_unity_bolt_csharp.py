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
PINNED_BOLTFFI = "0.29.3"


# --- Transform (a): readonly record struct (C# 10) -> plain readonly struct ---
# Matches a positional `public readonly record struct Name( params ) {`. The
# params never contain parentheses (simple `Type Name`, incl. `T?` / `T[]`), so
# `[^)]*` is a safe, greedy-free body match up to the closing paren.
RECORD_STRUCT_RE = re.compile(
    r"public readonly record struct (?P<name>\w+)\((?P<params>[^)]*)\)\s*\{",
    re.DOTALL,
)
# 11 on boltffi 0.29, which emits the whole inference path the 0.25.3 C#
# lowering dropped (XybridResult / XybridEnvelope / XybridStreamEvent / …),
# plus 3 for the tool-calling records (XybridToolDefinition / XybridToolCall /
# XybridToolResult).
# Bump this deliberately: the count is a tripwire for unreviewed boltffi output
# drift, not a value to auto-sync.
EXPECTED_RECORD_STRUCTS = 14


# --- Transform (g): Unsafe.SizeOf<T>() -> Marshal.SizeOf<T>(). boltffi's wire
# codec sizes blittable arrays with System.Runtime.CompilerServices.Unsafe, whose
# assembly Unity's scripting profile does not reference -- the Unity 6.3 editor
# rejects it with `CS0103: The name 'Unsafe' does not exist`. The codecs are
# `where T : unmanaged`, so Marshal.SizeOf<T>() (core BCL, already used
# throughout the generated wire layer) returns the identical size. Verified with
# an in-editor `unity test` compile on 6.3.
UNSAFE_SIZEOF_TARGET = (
    "global::System.Runtime.CompilerServices.Unsafe.SizeOf<T>()"
)
UNSAFE_SIZEOF_REPLACEMENT = (
    "global::System.Runtime.InteropServices.Marshal.SizeOf<T>()"
)
EXPECTED_UNSAFE_SIZEOF = 1

# --- Transform (d): make XybridModel partial. boltffi 0.25.3's C# generator
# drops the entire inference path -- no `run`, and no XybridEnvelope /
# XybridEnvelopeKind / XybridResult (a data-carrying enum used as a function
# INPUT, which the 0.25.3 C# lowering can't express; same class the Python
# generator drops). bindings/unity/Runtime/BoltSupplement/ hand-ports that path
# (wire codecs + a `partial class XybridModel` adding Run() / RunStreaming()), mirroring
# bindings/python/xybrid/_bolt.py. Marking the generated class partial lets the
# supplement extend it and reach its private _handle.
MODEL_FILE = "XybridModel.cs"
MODEL_PARTIAL_TARGET = (
    "public sealed class XybridModel : global::System.IDisposable"
)
MODEL_PARTIAL_REPLACEMENT = (
    "public sealed partial class XybridModel : global::System.IDisposable"
)

# XybridConversationContext is likewise extended in BoltSupplement with the
# envelope-input methods the generator drops (push / set_system), so it too
# must be partial.
CONTEXT_FILE = "XybridConversationContext.cs"
CONTEXT_PARTIAL_TARGET = (
    "public sealed class XybridConversationContext : global::System.IDisposable"
)
CONTEXT_PARTIAL_REPLACEMENT = (
    "public sealed partial class XybridConversationContext : global::System.IDisposable"
)

# --- Transform (e): keep the managed model alive across blocking StreamNext.
# Anchored on the 0.29 body, which returns the error and result buffers through
# separate out-params rather than one tagged buffer.
STREAM_NEXT_FREE_TARGET = (
    "            finally\n"
    "            {\n"
    "                NativeMethods.FreeBuf(boltffiResultBuffer);\n"
    "            }\n"
    "        }\n"
)
STREAM_NEXT_FREE_REPLACEMENT = (
    "            finally\n"
    "            {\n"
    "                NativeMethods.FreeBuf(boltffiResultBuffer);\n"
    "                // StreamNext can block for the full inter-token gap. Keep\n"
    "                // this wrapper alive so its finalizer cannot free the\n"
    "                // native handle while the call is still using it.\n"
    "                global::System.GC.KeepAlive(this);\n"
    "            }\n"
    "        }\n"
)

# --- Transform (i): un-collide the `Text` envelope variant. boltffi emits
# `sealed record Text(string Text)`, but C# forbids a member sharing its
# enclosing type's name (CS0542 / CS8866), so the generated file does not
# compile as-is. Rename the positional parameter to `Value` — which is also the
# name Unity's Runtime/Api already reads (`case XybridEnvelopeKind.Text t =>
# t.Value`), and leaves the positional constructor callers untouched.
ENVELOPE_KIND_FILE = "XybridEnvelopeKind.cs"
TEXT_VARIANT_REWRITES = (
    (
        "public sealed record Text(string Text) : XybridEnvelopeKind;",
        "public sealed record Text(string Value) : XybridEnvelopeKind;",
    ),
    ("writer.WriteString(value.Text);", "writer.WriteString(value.Value);"),
)

# --- Transform (j): the Guid wire codecs use the .NET 8 `bigEndian:` overloads
# of `new Guid(...)` / `Guid.TryWriteBytes(...)`, which netstandard2.1 lacks. No
# generated type reads or writes a Guid (only these two definitions mention
# them), so fail closed rather than hand-roll an endian swap nothing exercises —
# the same call the script already makes for unreachable wire paths. Add a real
# down-level here if a Guid ever reaches the surface.
GUID_REWRITES = (
    (
        "            return new global::System.Guid(bytes, bigEndian: true);",
        "            throw new global::System.NotSupportedException(\n"
        "                \"WireReader.ReadGuid: the netstandard2.1 profile Unity targets has no\"\n"
        "                + \" big-endian Guid constructor, and no xybrid type crosses the wire\"\n"
        "                + \" as a Guid. Add a down-level in\"\n"
        "                + \" tools/scripts/gen_unity_bolt_csharp.py before using one.\");",
    ),
    (
        "            if (!value.TryWriteBytes(bytes, bigEndian: true, out _))\n"
        "                throw new global::System.InvalidOperationException(\"Guid conversion failed\");",
        "            throw new global::System.NotSupportedException(\n"
        "                \"WireWriter.WriteGuid: the netstandard2.1 profile Unity targets has no\"\n"
        "                + \" big-endian Guid writer, and no xybrid type crosses the wire as a\"\n"
        "                + \" Guid. Add a down-level in\"\n"
        "                + \" tools/scripts/gen_unity_bolt_csharp.py before using one.\");",
    ),
)

# --- Transform (h): rename the generated static class. 0.29 derives it from the
# Cargo package name, and `xybrid_bolt` pascal-cases to `Xybrid_bolt` (the casing
# does not split on underscores). Unity's package has always called this
# `XybridBolt`, and 117 call sites in Runtime/ depend on that name, so rename it
# here rather than churn the public surface for a generator artifact.
BOLT_CLASS_FILE = "Xybrid_bolt.cs"
BOLT_CLASS_DEST = "XybridBolt.cs"
BOLT_CLASS_TARGET = "public static class Xybrid_bolt"
BOLT_CLASS_REPLACEMENT = "public static class XybridBolt"

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
    unsafe_sizeof = 0
    bolt_class_renamed = False
    text_variant_fixed = False
    guid_fenced = False
    model_made_partial = False
    context_made_partial = False
    stream_next_kept_alive = False
    for src in sources:
        content = src.read_text(encoding="utf-8")
        content, n = _downlevel_record_structs(content)
        record_structs += n
        content, n = _rewrite_unsafe_sizeof(content)
        unsafe_sizeof += n
        if src.name == ENVELOPE_KIND_FILE:
            for target, replacement in TEXT_VARIANT_REWRITES:
                _drift(
                    target in content,
                    f"expected `{target}` in {src.name}",
                )
                content = content.replace(target, replacement, 1)
            text_variant_fixed = True
        if src.name == BOLT_CLASS_FILE:
            for target, replacement in GUID_REWRITES:
                _drift(target in content, f"expected a Guid codec body in {src.name}")
                content = content.replace(target, replacement, 1)
            guid_fenced = True
            _drift(
                BOLT_CLASS_TARGET in content,
                f"expected the generated static class declaration in {src.name}",
            )
            content = content.replace(BOLT_CLASS_TARGET, BOLT_CLASS_REPLACEMENT, 1)
            bolt_class_renamed = True
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
        out_name = BOLT_CLASS_DEST if src.name == BOLT_CLASS_FILE else src.name
        tree[out_name] = content
        tree[out_name + ".meta"] = script_meta(f"{DEST_REL}/{out_name}")

    # Transform (c): emit the IsExternalInit polyfill next to the sources.
    tree[POLYFILL_FILE] = POLYFILL_CONTENT
    tree[POLYFILL_FILE + ".meta"] = script_meta(f"{DEST_REL}/{POLYFILL_FILE}")

    _drift(
        record_structs == EXPECTED_RECORD_STRUCTS,
        f"down-leveled {record_structs} record structs, expected "
        f"{EXPECTED_RECORD_STRUCTS}",
    )
    _drift(
        unsafe_sizeof == EXPECTED_UNSAFE_SIZEOF,
        f"rewrote {unsafe_sizeof} Unsafe.SizeOf calls, expected "
        f"{EXPECTED_UNSAFE_SIZEOF}",
    )
    _drift(bolt_class_renamed, f"{BOLT_CLASS_FILE} not found in boltffi output")
    _drift(guid_fenced, "Guid codec bodies not found in boltffi output")
    _drift(
        text_variant_fixed, f"{ENVELOPE_KIND_FILE} not found in boltffi output"
    )
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
