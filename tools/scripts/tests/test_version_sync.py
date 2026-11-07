"""The committed manifests and docs install pins match the Cargo version.

CI's Version sync job runs `version-sync.sh --check` on every pull request, and
release-prep runs it again on the release branch. This keeps it in the tooling
suite, so a change to version-sync.sh or install_pins.py that breaks the check
on the committed tree fails locally too.
"""

from pathlib import Path
import json
import shutil
import tempfile
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]


class VersionSyncCheck(unittest.TestCase):
    def test_committed_versions_are_in_sync(self):
        result = subprocess.run(
            ["bash", str(ROOT / "tools/scripts/version-sync.sh"), "--check"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class JsVersionSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in (
            "tools/scripts/version-sync.sh", "bindings/web/package.json",
            "bindings/web/src/version.ts", "bindings/observability-js/package.json",
            "bindings/observability-js/src/version.ts",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        # Load functions without entering the script's release/bump main routine.
        script = (self.root / "tools/scripts/version-sync.sh").read_text().split("# Main\n")[0]
        (self.root / "tools/scripts/version-sync-functions.sh").write_text(script)

    def shell(self, expression):
        return subprocess.run(
            ["bash", "-c", 'source "$0"; ' + expression,
             str(self.root / "tools/scripts/version-sync-functions.sh")],
            capture_output=True, text=True, timeout=10,
        )

    def test_bump_updates_package_dependency_and_telemetry_sources(self):
        for version in ("0.17.0", "0.18.0-rc.1"):
            with self.subTest(version=version):
                result = self.shell(f"set_observability_version {version}; set_web_version {version}")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                observability = json.loads((self.root / "bindings/observability-js/package.json").read_text())
                web = json.loads((self.root / "bindings/web/package.json").read_text())
                self.assertEqual(observability["version"], version)
                self.assertEqual(web["version"], version)
                self.assertEqual(web["dependencies"]["@xybrid/observability"], version)
                result = self.shell("get_observability_version; get_observability_source_version; "
                                    "get_web_version; get_web_source_version; get_web_observability_version")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [version] * 5)

    def test_check_rejects_independent_drift_on_each_js_surface(self):
        # Stub other surfaces: the real check's list must include all three new
        # getters, and no install-pin or binding generator is needed here.
        for getter in ("get_observability_version", "get_observability_source_version",
                       "get_web_observability_version"):
            with self.subTest(getter=getter):
                result = self.shell(
                    'get_cargo_version() { echo 0.16.0; }; '
                    'set_observability_version 0.16.0; set_web_version 0.16.0; '
                    + getter + '() { echo 0.15.0; }; check_versions'
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("0.15.0 ✗ (expected 0.16.0)", result.stdout)

    def test_both_sync_entry_points_update_observability(self):
        script = (ROOT / "tools/scripts/version-sync.sh").read_text().split("# Main\n")[1]
        self.assertEqual(script.count('set_observability_version "$VERSION"'), 2)


if __name__ == "__main__":
    unittest.main()
