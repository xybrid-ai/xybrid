#!/usr/bin/env python3
"""Verify a Unity Android plugin tree carries no C++ runtime of its own.

Unity's GameActivity packages its own libc++_shared.so for libgame.so, built
by whatever NDK the editor uses, and Gradle rejects a second copy at the same
APK path. So libxybrid_bolt.so must link libc++ statically, keep that runtime
private, and the package must ship no libc++_shared.so. It must also be linked
for Android 9 (API 28): only the Kotlin AAR's arm64 slice requires Android 10.
Each ABI directory (arm64-v8a, armeabi-v7a, x86_64) is checked from
`readelf -W` text; GNU readelf and llvm-readelf both read every ELF
architecture and both formats parse here. Every failure is reported, not just
the first.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

# The `.note.android.ident` parser the AAR and Flutter payload checks use too.
# Loaded from beside this file, so it works however this script is started.
_SPEC = importlib.util.spec_from_file_location(
    "verify_android_api_level", Path(__file__).with_name("verify_android_api_level.py"))
_API_LEVEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_API_LEVEL)
parse_api_levels = _API_LEVEL.parse_api_levels

DEFAULT_ROOT = Path("bindings/unity/Runtime/Plugins/Android")
BOLT = "libxybrid_bolt.so"
CXX_SHARED = "libc++_shared.so"
LIBC = "libc.so"

# ELF machine names as GNU and LLVM readelf print them; the first is the one
# reported in a mismatch.
ABI_MACHINES: dict[str, tuple[str, ...]] = {
    "arm64-v8a": ("AArch64", "EM_AARCH64"),
    "armeabi-v7a": ("ARM", "EM_ARM"),
    "x86_64": ("Advanced Micro Devices X86-64", "X86-64", "EM_X86_64"),
}

# Libraries every Android app process can resolve from the platform. Anything
# else a shipped library NEEDs must sit beside it in the same ABI directory.
SYSTEM_LIBS = frozenset(
    {"libc.so", "libm.so", "libdl.so", "liblog.so", "libandroid.so", "libz.so"}
)

# The API level libxybrid_bolt.so must be linked for: @androidndk's (MODULE.bazel),
# which --config=android-* builds use. Unity's Android floor stays Android 9;
# a library linked for Android 10 can fail to load below it (native ELF TLS).
ANDROID_API = 28

# Android 15+ 16 KB-page devices refuse to map a LOAD segment aligned below
# this; the cdylib links with -z max-page-size=16384 on every Android ABI.
MIN_LOAD_ALIGN = 0x4000

# Mangled C++ names, libc++abi's C entry points, and the unwinder libc++abi
# needs (libunwind; ARM EHABI's personality routines on armeabi-v7a).
# Undefined and unversioned, nothing in an app process provides them except a
# shared C++ runtime, and the NDK's libc++_shared.so does not export the
# unwinder either: a static link that dropped libunwind.a fails to load.
CXX_RUNTIME_PREFIXES = (
    "_Z", "__cxa_", "__gxx_personality", "__dynamic_cast",
    "_Unwind_", "__aeabi_unwind_cpp_pr", "__gnu_Unwind_",
)
# Bionic implements these itself, so every Android .so imports them from libc
# whether or not its C++ runtime is static.
BIONIC_CXA = frozenset({"__cxa_atexit", "__cxa_finalize", "__cxa_thread_atexit_impl"})
# Bionic's symbol versions (LIBC, LIBC_N, LIBC_P, ...); libm and libdl use them too.
BIONIC_VERSION_PREFIX = "LIBC"
# A thread_local's TLS init function, referenced weakly by code that may run
# before it exists; when absent the reference is null and the init is skipped.
# Not a runtime import: the vendored static-libc++ libonnxruntime.so has one.
TLS_INIT_PREFIX = "_ZTH"

# How many offending symbol names to print before summarising the rest.
MAX_LISTED = 8

# readelf --dyn-syms -W: "Num: Value Size Type Bind Vis [other] Ndx Name[@ver] [(n)]".
# GNU readelf may put an st_other annotation such as [VARIANT_PCS] after Vis.
_SYMBOL_RE = re.compile(
    r"^\s*\d+:\s+[0-9a-fA-F]+\s+\S+\s+(?P<type>\S+)\s+(?P<bind>\S+)\s+(?P<vis>\S+)"
    r"(?:\s+\[[^\]]*\])?\s+(?P<ndx>\S+)(?:\s+(?P<name>\S+))?"
)
_MACHINE_RE = re.compile(r"^\s*Machine:\s*(.+?)\s*$", re.MULTILINE)
_NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library:\s*\[([^\]]+)\]")


@dataclass(frozen=True)
class Symbol:
    name: str
    version: str | None
    kind: str
    bind: str
    visibility: str
    ndx: str

    @property
    def defined(self) -> bool:
        return self.ndx != "UND"


def parse_machine(text: str) -> str | None:
    match = _MACHINE_RE.search(text)
    return match.group(1) if match else None


def parse_needed(text: str) -> list[str]:
    return _NEEDED_RE.findall(text)


def parse_load_aligns(text: str) -> list[int]:
    aligns = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "LOAD":
            continue
        # Type Offset VirtAddr PhysAddr FileSiz MemSiz Flg... Align. Without -W
        # GNU readelf wraps 64-bit rows, and the last field is no longer Align.
        if len(fields) < 8:
            raise ValueError(f"unparseable LOAD program header (is readelf -W set?): {line.strip()}")
        aligns.append(int(fields[-1], 0))
    return aligns


def parse_symbols(text: str) -> list[Symbol]:
    symbols = []
    for line in text.splitlines():
        match = _SYMBOL_RE.match(line)
        if not match or not match.group("name"):
            continue
        # GNU prints "name@LIBC (2)", LLVM "name@LIBC"; a default version is "@@".
        name, _, version = match.group("name").partition("@")
        symbols.append(Symbol(
            name=name,
            version=version.lstrip("@") or None,
            kind=match.group("type"),
            bind=match.group("bind"),
            visibility=match.group("vis"),
            ndx=match.group("ndx"),
        ))
    return symbols


def cxx_runtime_imports(symbols: list[Symbol]) -> list[str]:
    """Undefined symbols only a shared C++ runtime would satisfy."""
    found = set()
    for symbol in symbols:
        if symbol.defined or symbol.name in BIONIC_CXA:
            continue
        if symbol.version and symbol.version.startswith(BIONIC_VERSION_PREFIX):
            continue
        if symbol.bind == "WEAK" and symbol.name.startswith(TLS_INIT_PREFIX):
            continue
        if "__ndk1" in symbol.name or symbol.name.startswith(CXX_RUNTIME_PREFIXES):
            found.add(symbol.name)
    return sorted(found)


def cxx_exports(symbols: list[Symbol]) -> list[str]:
    """Defined, dynamically visible C++ runtime symbols, mangled or not."""
    return sorted({
        symbol.name for symbol in symbols
        if symbol.defined
        and ("__ndk1" in symbol.name or symbol.name.startswith(CXX_RUNTIME_PREFIXES))
        and symbol.bind in {"GLOBAL", "WEAK", "UNIQUE", "GNU_UNIQUE"}
        and symbol.visibility in {"DEFAULT", "PROTECTED"}
    })


def _listing(names: list[str]) -> str:
    shown = ", ".join(names[:MAX_LISTED])
    extra = len(names) - MAX_LISTED
    return shown + (f", ... (+{extra} more)" if extra > 0 else "")


def check_abi(abi: str, elves: Mapping[str, str | None]) -> tuple[list[str], str]:
    """Check one ABI directory.

    `elves` maps each .so file name in the directory to its readelf text
    (`-W -h -d -l`, plus `-n --dyn-syms` for libxybrid_bolt.so), or None when
    readelf could not read it (already reported by the caller). Returns the
    failures and, when there are none, a one-line summary.
    """
    failures = []
    machines = ABI_MACHINES[abi]
    shipped = set(elves)

    if BOLT not in shipped:
        failures.append(f"{abi}/{BOLT}: missing")
    if CXX_SHARED in shipped:
        failures.append(
            f"{abi}/{CXX_SHARED}: shipped, but the Unity package must not carry a C++ "
            "runtime (Unity's GameActivity packages its own; Gradle rejects two)"
        )

    allowed_needed = (SYSTEM_LIBS | shipped) - {CXX_SHARED}
    bolt_needed: list[str] = []
    min_align = None
    bolt_api = None
    for name in sorted(shipped):
        text = elves[name]
        if text is None:
            continue
        where = f"{abi}/{name}"
        machine = parse_machine(text)
        if machine is None:
            failures.append(f"{where}: no ELF machine in readelf output")
        elif machine not in machines:
            failures.append(f"{where}: ELF machine is {machine}, expected {machines[0]}")

        needed = parse_needed(text)
        for dep in needed:
            if dep == CXX_SHARED:
                failures.append(f"{where}: NEEDs {CXX_SHARED}; libc++ must be linked statically")
            elif dep not in allowed_needed:
                failures.append(
                    f"{where}: NEEDs {dep}, which is neither an Android system "
                    f"library nor shipped in {abi}/"
                )

        try:
            aligns = parse_load_aligns(text)
        except ValueError as error:
            failures.append(f"{where}: {error}")
            aligns = None
        if aligns == []:
            failures.append(f"{where}: no LOAD program headers in readelf output")
        elif aligns and min(aligns) < MIN_LOAD_ALIGN:
            failures.append(
                f"{where}: LOAD segment aligned to {min(aligns):#x}, below 16 KB "
                f"({MIN_LOAD_ALIGN:#x}); it will not load on 16 KB-page devices"
            )

        if name != BOLT:
            continue
        bolt_needed = needed
        min_align = min(aligns) if aligns else None
        # Linking the NDK's whole static runtime (every sysroot archive) drops
        # libc.so from NEEDED and gives the library its own bionic and
        # allocator inside the app's process. Only libc++ may be static.
        if LIBC not in needed:
            failures.append(
                f"{where}: does not NEED {LIBC}, so bionic is linked into it "
                "statically; link only libc++ statically"
            )

        levels = parse_api_levels(text)
        if not levels:
            failures.append(
                f"{where}: no .note.android.ident note in readelf output; cannot "
                "tell which Android API level it was linked for"
            )
        elif set(levels) != {ANDROID_API}:
            found = ", ".join(str(level) for level in sorted(set(levels)))
            failures.append(
                f"{where}: linked for Android API {found}, expected {ANDROID_API}; "
                "the Unity plugin must keep loading on Android 9"
            )
        else:
            bolt_api = ANDROID_API

        symbols = parse_symbols(text)
        if not symbols:
            failures.append(f"{where}: no dynamic symbols in readelf output")
        imports = cxx_runtime_imports(symbols)
        if imports:
            failures.append(
                f"{where}: imports {len(imports)} C++ runtime symbol(s), so libc++ "
                f"is not linked statically: {_listing(imports)}"
            )
        exports = cxx_exports(symbols)
        if exports:
            failures.append(
                f"{where}: exports {len(exports)} C++ symbol(s); the static runtime "
                f"must stay hidden: {_listing(exports)}"
            )


    # An unreadable library was reported by the caller; don't call the ABI OK.
    if failures or None in elves.values():
        return failures, ""
    summary = (
        f"{abi}: OK - ships {', '.join(sorted(shipped))}; {BOLT} NEEDs "
        f"{', '.join(bolt_needed) or 'nothing'}, no C++ runtime imports or "
        f"exports, LOAD align {min_align:#x}, Android API {bolt_api}"
    )
    return failures, summary


def readelf_text(readelf: str, path: Path) -> str:
    flags = ["-W", "-h", "-d", "-l"]
    if path.name == BOLT:
        flags += ["-n", "--dyn-syms"]
    return subprocess.run(
        [readelf, *flags, str(path)], capture_output=True, text=True, check=True
    ).stdout


def check_tree(
    root: Path,
    read: Callable[[Path], str],
    require_abis: bool = False,
) -> tuple[list[str], list[str]]:
    """Check every ABI directory under `root`; returns (failures, summaries)."""
    if not root.is_dir():
        return [f"{root}: not a directory"], []

    failures: list[str] = []
    summaries: list[str] = []
    present = sorted(abi for abi in ABI_MACHINES if (root / abi).is_dir())
    if not present:
        failures.append(f"{root}: no ABI directories ({', '.join(ABI_MACHINES)})")
    if require_abis:
        failures += [f"{abi}: ABI directory missing" for abi in ABI_MACHINES if abi not in present]

    # Unity only packages <abi>/*.so for the ABIs above; anything elsewhere is
    # either unchecked or unshippable.
    for lib in sorted(root.rglob("*.so")):
        if lib.parent.parent != root or lib.parent.name not in ABI_MACHINES:
            failures.append(f"{lib.relative_to(root)}: native library outside a known ABI directory")

    for abi in present:
        elves: dict[str, str | None] = {}
        for lib in sorted((root / abi).glob("*.so")):
            if not lib.is_file():
                continue
            try:
                elves[lib.name] = read(lib)
            except subprocess.CalledProcessError as error:
                failures.append(f"{abi}/{lib.name}: readelf failed: {(error.stderr or '').strip()}")
                elves[lib.name] = None
        abi_failures, summary = check_abi(abi, elves)
        failures += abi_failures
        if summary:
            summaries.append(summary)
    return failures, summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root", nargs="?", type=Path, default=DEFAULT_ROOT,
        help="Unity Android plugin directory (default: %(default)s)",
    )
    parser.add_argument(
        "--readelf", default="readelf",
        help="GNU readelf or llvm-readelf (default: %(default)s)",
    )
    parser.add_argument(
        "--require-abis", action="store_true",
        help="fail unless every ABI directory (" + ", ".join(ABI_MACHINES) + ") is present",
    )
    args = parser.parse_args(argv)

    try:
        failures, summaries = check_tree(
            args.root, lambda path: readelf_text(args.readelf, path), args.require_abis
        )
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for summary in summaries:
        print(summary)
    if failures:
        print(f"\n{len(failures)} problem(s) in {args.root}:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"{args.root}: Unity Android natives OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
