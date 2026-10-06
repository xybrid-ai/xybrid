"""Exercise the Linux SDK installer with real curl, checksums and tar extraction."""

import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "install-vulkan-sdk-linux.sh"
VERSION = "1.4.999.0"


@unittest.skipUnless(sys.platform == "linux", "Linux Vulkan SDK installer")
class VulkanSdkInstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directory = self.root / "SDK cache"
        self.directory.mkdir()
        self.archive = self.directory / f"vulkansdk-{VERSION}.tar.xz"
        self.sdk = self.directory / VERSION / "x86_64"
        self.env_file = self.root / "github-env"
        self.path_file = self.root / "github-path"
        self.env_file.touch()
        self.path_file.touch()
        self.payload = self.sdk_archive()
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.requests = []
        self.release_stall = threading.Event()

        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                test.requests.append(self.path)
                status, body, length = test.responses[min(len(test.requests) - 1, len(test.responses) - 1)]
                self.send_response(status)
                self.send_header("Content-Length", str(length))
                self.end_headers()
                if body is None:
                    test.release_stall.wait(60)
                else:
                    self.wfile.write(body)
                self.close_connection = True

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(self.release_stall.set)
        self.url = f"http://127.0.0.1:{server.server_port}/sdk.tar.xz"
        self.responses = [(200, self.payload, len(self.payload))]

    @staticmethod
    def sdk_archive(missing=None, executable=True):
        files = {
            "bin/glslc": b"#!/bin/sh\nexit 0\n",
            "include/vulkan/vulkan_core.h": b"// Vulkan headers\n",
            "include/vk_video/vulkan_video_codecs_common.h": b"// Video headers\n",
            "include/spirv/unified1/spirv.hpp": b"// SPIR-V headers\n",
            "lib/VulkanLoader/lib/libvulkan.so.1": b"loader fixture\n",
        }
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
            root = tarfile.TarInfo(f"{VERSION}/x86_64")
            root.type = tarfile.DIRTYPE
            root.mode = 0o755
            archive.addfile(root)
            for path, contents in files.items():
                if path == missing:
                    continue
                member = tarfile.TarInfo(f"{VERSION}/x86_64/{path}")
                member.mode = 0o755 if path == "bin/glslc" and executable else 0o644
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
        return buffer.getvalue()

    def install(self, timeout=30):
        return subprocess.run(
            ["bash", str(SCRIPT), VERSION, self.sha, str(self.directory), self.url],
            env=dict(os.environ, GITHUB_ENV=str(self.env_file), GITHUB_PATH=str(self.path_file),
                     LD_LIBRARY_PATH="/existing/lib"),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def assert_installed(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.archive.read_bytes(), self.payload)
        self.assertEqual(self.sdk.joinpath("lib/libvulkan.so").read_bytes(), b"loader fixture\n")
        self.assertEqual(self.sdk.joinpath("lib/libvulkan.so").readlink(), Path("libvulkan.so.1"))
        values = dict(line.split("=", 1) for line in self.env_file.read_text().splitlines())
        self.assertEqual(values["VULKAN_SDK"], str(self.sdk))
        self.assertEqual(values["VULKAN_VERSION"], VERSION)
        self.assertEqual(values["VK_LAYER_PATH"], str(self.sdk / "share/vulkan/explicit_layer.d"))
        self.assertEqual(values["LD_LIBRARY_PATH"], f"{self.sdk}/lib/VulkanLoader/lib:{self.sdk}/lib:/existing/lib")
        self.assertEqual(self.path_file.read_text(), f"{self.sdk}/bin\n")
        self.assertFalse(Path(str(self.archive) + ".part").exists())

    def assert_not_exposed(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.env_file.read_text(), "")
        self.assertEqual(self.path_file.read_text(), "")
        self.assertFalse(Path(str(self.archive) + ".part").exists())

    def test_transient_http_failure_is_retried(self):
        self.responses.insert(0, (503, b"busy", 4))
        self.assert_installed(self.install())
        self.assertEqual(len(self.requests), 2)

    def test_interrupted_transfer_restarts_without_appending_partial_bytes(self):
        self.responses.insert(0, (200, self.payload[:16], len(self.payload)))
        self.assert_installed(self.install())
        self.assertEqual(len(self.requests), 2)

    def test_stalled_response_body_times_out_and_is_retried(self):
        self.responses.insert(0, (200, None, len(self.payload)))
        self.assert_installed(self.install(timeout=50))
        self.assertEqual(len(self.requests), 2)

    def test_exhausted_retries_fail_without_caching_or_exporting_a_partial_sdk(self):
        self.responses = [(503, b"busy", 4)]
        self.assert_not_exposed(self.install())
        self.assertEqual(len(self.requests), 3)
        self.assertFalse(self.archive.exists())

    def test_checksum_mismatch_is_rejected_before_extraction(self):
        self.responses = [(200, b"bad archive", 11)]
        result = self.install()
        self.assert_not_exposed(result)
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertFalse(self.sdk.exists())
        self.assertFalse(self.archive.exists())

    def test_corrupt_cache_is_downloaded_again(self):
        self.archive.write_bytes(b"truncated cached archive")
        self.assert_installed(self.install())
        self.assertEqual(len(self.requests), 1)

    def test_verified_cache_repairs_an_interrupted_install_without_network(self):
        self.archive.write_bytes(self.payload)
        header = self.sdk / "include/vulkan/vulkan_core.h"
        header.parent.mkdir(parents=True)
        header.write_bytes(b"partial install")
        self.assert_installed(self.install())
        self.assertEqual(header.read_bytes(), b"// Vulkan headers\n")
        self.assertEqual(self.requests, [])

    def test_incomplete_sdk_is_not_exported(self):
        self.payload = self.sdk_archive(missing="include/spirv/unified1/spirv.hpp")
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.archive.write_bytes(self.payload)
        result = self.install()
        self.assert_not_exposed(result)
        self.assertIn("Incomplete Vulkan SDK", result.stderr)

    def test_glslc_must_be_executable(self):
        self.payload = self.sdk_archive(executable=False)
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.archive.write_bytes(self.payload)
        result = self.install()
        self.assert_not_exposed(result)
        self.assertIn("glslc is not executable", result.stderr)


if __name__ == "__main__":
    unittest.main()
