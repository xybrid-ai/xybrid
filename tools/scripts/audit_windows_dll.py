#!/usr/bin/env python3
"""Audit a Windows PE DLL's import table and exports.

The windows-gnu bolt cdylib statically links the C++ runtime trio
(libc++/libc++abi/libunwind), so a correct build imports ONLY Windows system
DLLs — no MinGW/MSVC redistributables and no link-time onnxruntime.dll (ORT is
loaded dynamically at runtime). This gate fails the build if any non-system DLL
appears in the import table, or if an expected boltffi export is missing.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# Windows DLLs guaranteed present on a stock machine (system32). MinGW/MSVC
# runtime carriers are deliberately absent so they surface as failures.
SYSTEM_DLLS = frozenset(
    name.lower()
    for name in (
        "kernel32.dll",
        "kernelbase.dll",
        "ntdll.dll",
        "advapi32.dll",
        "user32.dll",
        "gdi32.dll",
        "shell32.dll",
        "shlwapi.dll",
        "ole32.dll",
        "oleaut32.dll",
        "combase.dll",
        "propsys.dll",
        "ws2_32.dll",
        "wsock32.dll",
        "iphlpapi.dll",
        "netapi32.dll",
        "secur32.dll",
        "sspicli.dll",
        "crypt32.dll",
        "bcrypt.dll",
        "bcryptprimitives.dll",
        "ncrypt.dll",
        "psapi.dll",
        "userenv.dll",
        "powrprof.dll",
        "pdh.dll",
        "dxgi.dll",
        "d3d11.dll",
        "d3d12.dll",
        "dxcore.dll",
        "winmm.dll",
        "version.dll",
        "rpcrt4.dll",
        "cfgmgr32.dll",
        "setupapi.dll",
        "dbghelp.dll",
        # The legacy C runtime that ships with Windows itself (MinGW links it).
        # NOT to be confused with msvcp140.dll / vcruntime140.dll, which are the
        # redistributable MSVC runtimes and are intentionally NOT allow-listed.
        "msvcrt.dll",
    )
)

# API-set contract stubs ("api-ms-win-*", "ext-ms-*") always resolve on modern
# Windows via apisetschema — treat any of them as system.
SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-")

# Runtime carriers that MUST NOT appear — their presence means the C++ runtime
# (or an MSVC redist) leaked out as a dynamic dependency instead of being
# statically contained. Reported explicitly for a clearer failure message.
FORBIDDEN_HINTS = (
    "libc++",
    "libc++abi",
    "libunwind",
    "libwinpthread",
    "libgcc",
    "libstdc++",
    "msvcp",
    "vcruntime",
    "libgomp",
    "libssp",
)


@dataclass(frozen=True)
class AuditResult:
    ok: bool
    lines: tuple[str, ...]


DLL_IMPORT_ENTRY_POINT = re.compile(
    r'(?m)^\s*\[DllImport\([^\r\n]*?\bEntryPoint\s*=\s*"([^"]+)"'
)


def extract_csharp_entry_points(source: str) -> list[str]:
    """Return unique native entry points declared by active C# DllImport attributes."""
    return list(dict.fromkeys(DLL_IMPORT_ENTRY_POINT.findall(source)))


def is_system_import(dll: str, *, extra_allowed: frozenset[str]) -> bool:
    """Return True if `dll` is an allowed (system or explicitly-permitted) import."""
    name = dll.lower()
    if name in SYSTEM_DLLS or name in extra_allowed:
        return True
    return any(name.startswith(prefix) for prefix in SYSTEM_PREFIXES)


def audit(
    imports: list[str],
    exports: list[str],
    *,
    required_exports: list[str],
    required_imports: tuple[str, ...] = (),
    extra_allowed: frozenset[str] = frozenset(),
) -> AuditResult:
    """Check imports against the allowlist and confirm required symbols exist."""
    lines: list[str] = []
    export_set = set(exports)
    import_set = {name.lower() for name in imports}

    forbidden: list[str] = []
    unexpected: list[str] = []
    for dll in sorted({d.lower() for d in imports}):
        if is_system_import(dll, extra_allowed=extra_allowed):
            continue
        if any(hint in dll for hint in FORBIDDEN_HINTS):
            forbidden.append(dll)
        else:
            unexpected.append(dll)

    missing = [name for name in required_exports if name not in export_set]
    missing_imports = [
        name for name in required_imports if name.lower() not in import_set
    ]

    lines.append(f"imports: {len(set(i.lower() for i in imports))} distinct DLLs")
    lines.append(f"exports: {len(export_set)} symbols")
    if forbidden:
        lines.append(
            "FAIL: runtime-carrier DLLs imported (should be statically linked): "
            + ", ".join(forbidden)
        )
    if unexpected:
        lines.append(
            "FAIL: non-system DLLs imported (not on the allowlist): "
            + ", ".join(unexpected)
        )
    if missing:
        lines.append("FAIL: required exports missing: " + ", ".join(missing))
    if missing_imports:
        lines.append("FAIL: required imports missing: " + ", ".join(missing_imports))
    failed = bool(forbidden or unexpected or missing or missing_imports)
    if not failed:
        lines.append(
            f"OK: import table is Windows-system-only; {len(required_exports)} "
            "required exports present"
        )

    return AuditResult(ok=not failed, lines=tuple(lines))


def read_pe(path: str) -> tuple[list[str], list[str]]:
    """Extract (imported DLL names, exported symbol names) from a PE file."""
    import pefile  # imported lazily so the pure logic stays unit-testable

    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(
        directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
        ]
    )
    imports = [
        entry.dll.decode("ascii", "replace")
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
    ]
    exports = [
        exp.name.decode("ascii", "replace")
        for exp in getattr(
            getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []
        )
        if exp.name
    ]
    return imports, exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a Windows PE DLL.")
    parser.add_argument("--dll", required=True)
    parser.add_argument(
        "--require-export",
        action="append",
        default=[],
        help="An export that must be present (repeatable).",
    )
    parser.add_argument(
        "--require-import",
        action="append",
        default=[],
        help=(
            "A DLL that MUST appear in the import table, and is allowed by "
            "implication (repeatable). The windows-MSVC lane uses it to assert "
            "the MSVC runtime is linked, which is the opposite of what the "
            "windows-GNU lane wants."
        ),
    )
    parser.add_argument(
        "--allow-import",
        action="append",
        default=[],
        help="An extra non-system DLL to permit, e.g. onnxruntime.dll (repeatable).",
    )
    parser.add_argument(
        "--require-exports-from-csharp",
        action="append",
        default=[],
        metavar="PATH",
        help="Require every DllImport EntryPoint declared in a C# source (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        imports, exports = read_pe(args.dll)
    except Exception as error:  # noqa: BLE001 - surface any pefile failure cleanly
        print(f"audit-windows-dll: cannot read {args.dll}: {error}", file=sys.stderr)
        return 2

    required_exports = list(args.require_export)
    for source_path in args.require_exports_from_csharp:
        try:
            source = Path(source_path).read_text(encoding="utf-8")
        except OSError as error:
            print(
                f"audit-windows-dll: cannot read {source_path}: {error}",
                file=sys.stderr,
            )
            return 2
        entry_points = extract_csharp_entry_points(source)
        if not entry_points:
            print(
                f"audit-windows-dll: no DllImport entry points found in {source_path}",
                file=sys.stderr,
            )
            return 2
        required_exports.extend(entry_points)
    required_exports = list(dict.fromkeys(required_exports))

    result = audit(
        imports,
        exports,
        required_exports=required_exports,
        required_imports=tuple(args.require_import),
        extra_allowed=frozenset(
            a.lower() for a in list(args.allow_import) + list(args.require_import)
        ),
    )
    print(f"== PE audit: {args.dll} ==")
    for line in result.lines:
        print(f"  {line}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
