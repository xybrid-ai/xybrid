"""The committed manifests and docs install pins match the Cargo version.

CI's Version sync job runs `version-sync.sh --check` on every pull request, and
release-prep runs it again on the release branch. This keeps it in the tooling
suite, so a change to version-sync.sh or install_pins.py that breaks the check
on the committed tree fails locally too.
"""

from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
