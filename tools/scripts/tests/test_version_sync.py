"""The committed manifests and Unity README install pins match the Cargo version.

`version-sync.sh --check` otherwise runs only on release branches and tags, so a
hand edit that breaks a pin would first fail while a release is being cut.
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
