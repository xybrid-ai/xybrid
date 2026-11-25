"""Exercise the iOS example installer without CocoaPods or a network."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "bindings/react-native/scripts/pod-install.sh"
VERSION = "250829098.0.10"
ARCHIVE = f"hermes-ios-{VERSION}-hermes-ios-debug.tar.gz"
DIGEST = hashlib.sha256(b"archive").hexdigest()


class ReactNativePodInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = self.root / "example"
        self.ios = self.app / "ios"
        self.ios.mkdir(parents=True)
        (self.ios / "Podfile").touch()
        properties = self.app / "node_modules/react-native/sdks/hermes-engine"
        properties.mkdir(parents=True)
        (properties / "version.properties").write_text(
            f"HERMES_VERSION_NAME=0.16.0\nHERMES_V1_VERSION_NAME={VERSION}\n",
            encoding="utf-8",
        )
        self.stub("curl", '''
            printf '%s\\n' "${@: -1}" >> "$CURL_LOG"
            if [[ "${@: -1}" == *.sha256 ]]; then
                printf '%s\\n' "$DIGEST"
            else
                while [[ "$1" != --output ]]; do shift; done
                printf '%s' archive > "$2"
            fi
        ''')
        self.stub("pod", '''
            set -eu
            [[ "$*" == "install" && -f Podfile && ! -e build && ! -e Pods ]]
            [[ "$(cat "$HERMES_ENGINE_TARBALL_PATH")" == archive ]]
            printf '%s\\n' "$HERMES_ENGINE_TARBALL_PATH" >> "$POD_LOG"
            mkdir Pods build
            printf '  - hermes-engine (%s)\\n' "$VERSION" > Podfile.lock
            if (( $(wc -l < "$POD_LOG") <= FALLBACK_ATTEMPTS )); then
                echo '[ReactNativeCore] No prebuilt artifacts found, reverting to building from source.'
            fi
            echo '[Hermes] Using pre-built binary from local path defined by HERMES_ENGINE_TARBALL_PATH'
        ''')

    def stub(self, name, script):
        path = self.root / name
        path.write_text(f"#!/usr/bin/env bash\n{script}", encoding="utf-8")
        path.chmod(0o755)

    def install(self, digest=DIGEST, fallback_attempts=0):
        return subprocess.run(
            ["bash", str(SCRIPT), str(self.app)],
            env=dict(os.environ, PATH=f"{self.root}{os.pathsep}{os.environ['PATH']}",
                     XYBRID_HERMES_DIR=str(self.root / "cache"), XYBRID_POD_RETRY_DELAY="0",
                     CURL_LOG=str(self.root / "curl.log"), POD_LOG=str(self.root / "pod.log"),
                     GITHUB_ACTIONS="true", ENTERPRISE_REPOSITORY="https://maven.example.test/repository",
                     RCT_HERMES_V1_ENABLED="1", DIGEST=digest, VERSION=VERSION,
                     FALLBACK_ATTEMPTS=str(fallback_attempts)),
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_pod_uses_the_verified_tarball_and_cache_skips_archive_download(self):
        url = f"https://maven.example.test/repository/com/facebook/hermes/hermes-ios/{VERSION}/{ARCHIVE}"
        cached = self.root / "cache" / DIGEST / ARCHIVE
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"archive")
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Hermes {VERSION} pinned", result.stdout)
        self.assertEqual((self.root / "curl.log").read_text().splitlines(), [f"{url}.sha256"])
        self.assertEqual((self.root / "pod.log").read_text().strip(), str(cached))

    def test_checksum_mismatch_never_runs_pod(self):
        result = self.install(digest="0" * 64)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertFalse((self.root / "pod.log").exists())

    def test_discards_fallback_install_and_retries(self):
        result = self.install(fallback_attempts=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("attempt 1", result.stderr)
        self.assertEqual(len((self.root / "pod.log").read_text().splitlines()), 2)

    def test_fails_after_three_fallback_installs(self):
        result = self.install(fallback_attempts=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("::error::pod install failed after 3 attempts", result.stderr)
        self.assertEqual(len((self.root / "pod.log").read_text().splitlines()), 3)
        self.assertFalse((self.ios / "Podfile.lock").exists())

    def test_source_ios_lanes_use_the_installer(self):
        for path in (".github/workflows/build-react-native.yml",
                     "tools/scripts/observability_staging/react-native-lane.sh"):
            source = (ROOT / path).read_text()
            self.assertIn("scripts/pod-install.sh", source)
            self.assertNotRegex(source, r"(?:run:\s+|\bcd ios && |\$xcframework\" )pod install\b")

    def test_published_smoke_uses_plain_pod_install(self):
        # A consumer app has no repo installer; it must resolve the release zip.
        source = (ROOT / ".github/workflows/verify-react-native-published.yml").read_text()
        self.assertIn("run: pod install", source)
        self.assertNotIn("scripts/pod-install.sh", source)


if __name__ == "__main__":
    unittest.main()
