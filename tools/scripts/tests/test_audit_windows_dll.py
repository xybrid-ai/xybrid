"""Unit tests for the Windows PE audit logic (no real DLL required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_windows_dll import audit, is_system_import


SYSTEM_IMPORTS = ["KERNEL32.dll", "ntdll.dll", "ws2_32.dll", "bcrypt.dll"]
BOLT_EXPORTS = ["boltffi_version", "boltffi_free_buf", "boltffi_configure_runtime"]


class AuditWindowsDllTests(unittest.TestCase):
    def test_clean_system_only_imports_pass(self):
        result = audit(
            SYSTEM_IMPORTS + ["api-ms-win-core-synch-l1-2-0.dll"],
            BOLT_EXPORTS,
            required_exports=["boltffi_version", "boltffi_free_buf"],
        )
        self.assertTrue(result.ok)
        self.assertTrue(any("Windows-system-only" in line for line in result.lines))

    def test_forbidden_runtime_carrier_fails(self):
        result = audit(
            SYSTEM_IMPORTS + ["libc++.dll"],
            BOLT_EXPORTS,
            required_exports=["boltffi_version"],
        )
        self.assertFalse(result.ok)
        self.assertTrue(
            any("runtime-carrier" in line and "libc++.dll" in line for line in result.lines)
        )

    def test_mingw_pthread_and_gcc_flagged(self):
        for carrier in ("libwinpthread-1.dll", "libgcc_s_seh-1.dll", "libstdc++-6.dll"):
            result = audit(
                SYSTEM_IMPORTS + [carrier],
                BOLT_EXPORTS,
                required_exports=[],
            )
            self.assertFalse(result.ok, carrier)
            self.assertTrue(
                any("runtime-carrier" in line for line in result.lines), carrier
            )

    def test_msvc_redist_flagged(self):
        # msvcrt.dll is system; msvcp140/vcruntime140 are redistributables.
        result = audit(
            ["msvcrt.dll", "vcruntime140.dll"],
            BOLT_EXPORTS,
            required_exports=[],
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("vcruntime140.dll" in line for line in result.lines))

    def test_unexpected_non_system_import_fails(self):
        result = audit(
            SYSTEM_IMPORTS + ["random_third_party.dll"],
            BOLT_EXPORTS,
            required_exports=[],
        )
        self.assertFalse(result.ok)
        self.assertTrue(
            any(
                "non-system" in line and "random_third_party.dll" in line
                for line in result.lines
            )
        )

    def test_missing_required_export_fails(self):
        result = audit(
            SYSTEM_IMPORTS,
            ["boltffi_version"],
            required_exports=["boltffi_version", "boltffi_run"],
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("boltffi_run" in line for line in result.lines))

    def test_extra_allowed_import_permitted(self):
        result = audit(
            SYSTEM_IMPORTS + ["onnxruntime.dll"],
            BOLT_EXPORTS,
            required_exports=[],
            extra_allowed=frozenset({"onnxruntime.dll"}),
        )
        self.assertTrue(result.ok)

    def test_is_system_import_prefixes_and_case(self):
        allowed = frozenset()
        self.assertTrue(is_system_import("API-MS-Win-Core-Heap-L1-1-0.dll", extra_allowed=allowed))
        self.assertTrue(is_system_import("ext-ms-win-foo.dll", extra_allowed=allowed))
        self.assertTrue(is_system_import("KeRnEl32.DLL", extra_allowed=allowed))
        self.assertFalse(is_system_import("libunwind.dll", extra_allowed=allowed))


if __name__ == "__main__":
    unittest.main()
