import contextlib
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import zzz_pull as zzz  # noqa: E402 (scripts directory must be added first)


class ZzzPullTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_data = json.loads(zzz.DEFAULT_MANIFEST.read_text())
        self.manifest = self.root / "manifest.json"
        self.target = "aarch64-apple-darwin"
        self.environment = {
            "ZZZ_RELEASE_REPOSITORY": "example/engine-artifacts",
            "ZZZ_GITHUB_TOKEN": "fake-private-token",
        }

    def make_archive(
        self,
        *,
        target=None,
        receipt_update=None,
        metadata_update=None,
        extra=None,
        omit=None,
    ):
        target = target or self.target
        entry = next(e for e in self.manifest_data["archives"] if e["target"] == target)
        resolved = {
            "aarch64-apple-darwin": "aarch64-macos.13.0...15.6-none",
            "aarch64-linux-android": "aarch64-linux.5.10...6.19-android.29",
            "x86_64-unknown-linux-gnu": "x86_64-linux.5.10...6.19-gnu.2.28",
        }[target]
        receipt = {
            k: copy.deepcopy(entry[k])
            for k in (
                "slice",
                "profile",
                "models",
                "apis",
                "library",
                "headers",
                "backend",
                "cpu",
                "deployment_minimum",
            )
        }
        receipt.update(
            {
                "schema_version": 3,
                "release_tag": self.manifest_data["release_tag"],
                "engine_version": self.manifest_data["release_tag"][1:],
                "abi_version": 1,
                "target": zzz.ZIG_TARGETS[target],
                "resolved_target": resolved,
                "header": entry["headers"][0],
                "position_independent": True,
                "dependencies": {
                    "system_libraries": entry["system_libraries"],
                    "compiler_runtime": "bundled",
                },
                "source_repository": "example/engine-artifacts",
                "source_commit": "synthetic-test-revision",
            }
        )
        if target == "aarch64-linux-android":
            receipt.update(android_api=29, elf_max_page_size=16384)
        receipt.update(receipt_update or {})
        metadata = {
            k: copy.deepcopy(entry[k]) for k in ("apis", "models", "headers", "backend")
        }
        metadata.update(abi_version=1, target=resolved, optimize="ReleaseFast")
        metadata.update(metadata_update or {})
        payload = {
            "RECEIPT.json": json.dumps(receipt).encode(),
            "share/zzz_embed.json": json.dumps(metadata).encode(),
            "lib/libzzz_embed.a": b"!<arch>\nsynthetic-library",
            "include/zzz_embed.h": b"/* synthetic Kitten TTS 2 header */\n",
        }
        archive_root = entry["archive"].removesuffix(".tar.gz")
        archive = self.root / entry["archive"]
        with tarfile.open(archive, "w:gz") as output:
            for directory in (
                archive_root,
                *(f"{archive_root}/{d}" for d in ("include", "lib", "share")),
            ):
                member = tarfile.TarInfo(directory)
                member.type = tarfile.DIRTYPE
                output.addfile(member)
            for name, contents in payload.items():
                if name == omit:
                    continue
                member = tarfile.TarInfo(f"{archive_root}/{name}")
                member.size = len(contents)
                output.addfile(member, io.BytesIO(contents))
            if extra is not None:
                name, kind = extra
                member = tarfile.TarInfo(name.replace("{root}", archive_root))
                member.type = kind
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    member.linkname = "../../outside"
                output.addfile(member, io.BytesIO(b""))
        entry["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.manifest.write_text(json.dumps(self.manifest_data))
        return archive, zzz.select_pin(target, self.manifest)

    def release_response(self, pin, archive):
        return {
            "tag_name": pin.release_tag,
            "draft": False,
            "assets": [
                {
                    "id": 123,
                    "name": pin.entry["archive"],
                    "size": archive.stat().st_size,
                    "state": "uploaded",
                }
            ],
        }

    def fake_gh(self, release, archive):
        def run(command, **kwargs):
            if command[-1].endswith("/releases/assets/123"):
                kwargs["stdout"].write(archive.read_bytes())
                return subprocess.CompletedProcess(command, 0)
            return subprocess.CompletedProcess(
                command, 0, json.dumps(release).encode(), b""
            )

        return run

    def test_committed_manifest_has_exactly_three_kitten_targets(self):
        self.assertEqual(
            set(zzz.ZIG_TARGETS), {e["target"] for e in self.manifest_data["archives"]}
        )
        self.assertEqual(3, len(self.manifest_data["archives"]))
        for target in zzz.ZIG_TARGETS:
            self.assertEqual([zzz.PROFILE], zzz.select_pin(target).entry["models"])

    def test_all_supported_slices_stage_and_verify_without_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            for target in zzz.ZIG_TARGETS:
                with self.subTest(target=target):
                    archive, pin = self.make_archive(target=target)
                    staged = zzz.stage_archive(archive, pin, self.root / "staged")
                    self.assertEqual((self.root / "staged").joinpath(*pin.key), staged)
                    self.assertEqual(staged, zzz.verify_staged(staged, pin))
                    self.assertEqual(
                        b"!<arch>\nsynthetic-library",
                        (staged / pin.entry["library"]).read_bytes(),
                    )
                    for item in (staged, *staged.rglob("*")):
                        self.assertEqual(0, item.stat().st_mode & 0o077)

    def test_checksum_is_checked_before_tar_is_opened_or_staging_created(self):
        archive, pin = self.make_archive()
        archive.write_bytes(b"not the pinned archive")
        with mock.patch.object(zzz.tarfile, "open") as open_tar:
            with self.assertRaisesRegex(zzz.StageError, "checksum"):
                zzz.stage_archive(archive, pin, self.root / "staged")
        open_tar.assert_not_called()
        self.assertFalse((self.root / "staged").exists())

    def test_incompatible_receipts_fail_even_with_correct_archive_checksums(self):
        cases = {
            "schema_version": 2,
            "release_tag": "v0.0.0",
            "engine_version": "0.0.0",
            "abi_version": 2,
            "slice": "other-slice",
            "profile": "other-profile",
            "models": ["other-model"],
            "apis": {"zzz_embed": True},
            "library": "lib/other.a",
            "headers": ["include/other.h"],
            "header": "include/other.h",
            "backend": "other-backend",
            "cpu": "other-cpu",
            "deployment_minimum": "12.0",
            "target": "x86_64-linux-gnu",
            "resolved_target": "aarch64-macos.12.0-none",
            "position_independent": False,
            "dependencies": {
                "system_libraries": ["other"],
                "compiler_runtime": "bundled",
            },
        }
        for key, value in cases.items():
            with self.subTest(field=key):
                archive, pin = self.make_archive(receipt_update={key: value})
                with self.assertRaisesRegex(zzz.StageError, "receipt"):
                    zzz.inspect_archive(archive, pin)
        archive, pin = self.make_archive(
            receipt_update={
                "dependencies": {
                    "system_libraries": ["System"],
                    "compiler_runtime": "external",
                }
            }
        )
        with self.assertRaisesRegex(zzz.StageError, "compiler_runtime"):
            zzz.inspect_archive(archive, pin)

    def test_android_requires_api_and_16k_metadata(self):
        for update in (
            {"android_api": 28},
            {"android_api": True},
            {"elf_max_page_size": 4096},
        ):
            with self.subTest(update=update):
                archive, pin = self.make_archive(
                    target="aarch64-linux-android", receipt_update=update
                )
                with self.assertRaisesRegex(zzz.StageError, "receipt"):
                    zzz.inspect_archive(archive, pin)

    def test_embedded_metadata_must_match_receipt_and_pins(self):
        for update in (
            {"models": ["other-model"]},
            {"target": "other-target"},
            {"apis": {"zzz_embed": 2}},
            {"backend": "other"},
        ):
            with self.subTest(update=update):
                archive, pin = self.make_archive(metadata_update=update)
                with self.assertRaisesRegex(zzz.StageError, "embedded metadata"):
                    zzz.inspect_archive(archive, pin)

    def test_unsafe_unexpected_and_duplicate_archive_members_are_rejected(self):
        cases = [
            ("/absolute", tarfile.REGTYPE),
            ("{root}/../outside", tarfile.REGTYPE),
            ("{root}/include/link", tarfile.SYMTYPE),
            ("{root}/include/hardlink", tarfile.LNKTYPE),
            ("{root}/lib/device", tarfile.CHRTYPE),
            ("{root}/lib/pipe", tarfile.FIFOTYPE),
            ("{root}/extra", tarfile.REGTYPE),
            ("{root}/lib/libzzz_embed.a", tarfile.REGTYPE),
            ("other-root", tarfile.DIRTYPE),
            ("include", tarfile.DIRTYPE),
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                archive, pin = self.make_archive(extra=extra)
                with self.assertRaises(zzz.StageError):
                    zzz.inspect_archive(archive, pin)
        self.assertFalse((self.root / "outside").exists())

    def test_missing_payload_is_rejected(self):
        archive, pin = self.make_archive(omit="include/zzz_embed.h")
        with self.assertRaisesRegex(zzz.StageError, "missing archive payload"):
            zzz.inspect_archive(archive, pin)

    def test_archive_size_limits_are_enforced(self):
        archive, pin = self.make_archive()
        with mock.patch.object(zzz, "MAX_ARCHIVE_BYTES", 1):
            with self.assertRaisesRegex(zzz.StageError, "size limit"):
                zzz.inspect_archive(archive, pin)
        with mock.patch.object(zzz, "MAX_PAYLOAD_BYTES", 1):
            with self.assertRaisesRegex(zzz.StageError, "payload size"):
                zzz.inspect_archive(archive, pin)
        with mock.patch.object(zzz, "MAX_JSON_BYTES", 1):
            with self.assertRaisesRegex(zzz.StageError, "metadata"):
                zzz.inspect_archive(archive, pin)

    def test_staged_payload_and_archive_tampering_is_detected(self):
        for name in (
            "lib/libzzz_embed.a",
            "include/zzz_embed.h",
            "RECEIPT.json",
            "share/zzz_embed.json",
            zzz.STAGED_ARCHIVE,
        ):
            with self.subTest(name=name):
                archive, pin = self.make_archive()
                staged = zzz.stage_archive(
                    archive, pin, self.root / name.replace("/", "-")
                )
                (staged / name).write_bytes(b"tampered")
                with self.assertRaisesRegex(zzz.StageError, "checksum|differs"):
                    zzz.verify_staged(staged, pin)

    def test_staged_links_and_extra_directories_are_rejected(self):
        archive, pin = self.make_archive()
        staged = zzz.stage_archive(archive, pin, self.root / "staged")
        library = staged / pin.entry["library"]
        library.unlink()
        library.symlink_to(archive)
        with self.assertRaisesRegex(zzz.StageError, "links"):
            zzz.verify_staged(staged, pin)
        library.unlink()
        zzz.inspect_archive(archive, pin, staged / "recovery")
        # Verification rejects additional content even when all required files exist.
        library.write_bytes(b"!<arch>\nsynthetic-library")
        with self.assertRaisesRegex(zzz.StageError, "unexpected staged directory"):
            zzz.verify_staged(staged, pin)

    def test_missing_anchor_cannot_be_replaced_with_a_forged_stamp(self):
        archive, pin = self.make_archive()
        staged = zzz.stage_archive(archive, pin, self.root / "staged")
        (staged / zzz.STAGED_ARCHIVE).unlink()
        with self.assertRaisesRegex(zzz.StageError, "missing staged"):
            zzz.verify_staged(staged, pin)
        (staged / "verified.json").write_text(
            json.dumps({"sha256": pin.entry["sha256"]})
        )
        (staged / "verified.json").chmod(0o600)
        with self.assertRaisesRegex(zzz.StageError, "unexpected staged file"):
            zzz.verify_staged(staged, pin)

    def test_stage_reuse_and_failed_input_leave_existing_stage_unchanged(self):
        archive, pin = self.make_archive()
        staged = zzz.stage_archive(archive, pin, self.root / "staged")
        self.assertEqual(staged, zzz.stage_archive(archive, pin, self.root / "staged"))
        original = (staged / zzz.STAGED_ARCHIVE).read_bytes()
        archive.write_bytes(b"corrupt")
        with self.assertRaises(zzz.StageError):
            zzz.stage_archive(archive, pin, self.root / "staged")
        self.assertEqual(original, (staged / zzz.STAGED_ARCHIVE).read_bytes())
        self.assertEqual(staged, zzz.verify_staged(staged, pin))
        self.assertFalse(list(staged.parent.glob(".stage-*")))

    def test_interrupted_extraction_leaves_no_partial_slice_and_preserves_previous_pin(
        self,
    ):
        archive, previous_pin = self.make_archive()
        previous_stage = zzz.stage_archive(archive, previous_pin, self.root / "staged")
        archive, new_pin = self.make_archive(receipt_update={"validation": "synthetic"})
        self.assertNotEqual(previous_pin.entry["sha256"], new_pin.entry["sha256"])
        inspect_archive = zzz.inspect_archive

        def fail_during_extraction(path, pin, destination=None):
            if destination is not None:
                (destination / "partial").write_bytes(b"partial")
                raise OSError("synthetic interrupted write")
            return inspect_archive(path, pin)

        with mock.patch.object(
            zzz, "inspect_archive", side_effect=fail_during_extraction
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                zzz.stage_archive(archive, new_pin, self.root / "staged")
        self.assertEqual(
            previous_stage, zzz.verify_staged(previous_stage, previous_pin)
        )
        self.assertFalse((self.root / "staged").joinpath(*new_pin.key).exists())
        self.assertFalse(list(previous_stage.parent.glob(".stage-*")))

    def test_authenticated_download_uses_exact_tag_asset_and_dedicated_token(self):
        archive, pin = self.make_archive()
        release = self.release_response(pin, archive)
        environment = {
            **self.environment,
            "GH_DEBUG": "api",
            "GH_HOST": "other-host",
            "GH_TOKEN": "wrong-token",
            "CI": "true",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            with mock.patch.object(
                zzz.subprocess, "run", side_effect=self.fake_gh(release, archive)
            ) as run:
                cached = zzz.fetch_archive(pin, self.root / "cache")
        self.assertEqual(archive.read_bytes(), cached.read_bytes())
        self.assertEqual(2, run.call_count)
        calls = run.call_args_list
        self.assertEqual(
            f"repos/example/engine-artifacts/releases/tags/{pin.release_tag}",
            calls[0].args[0][-1],
        )
        self.assertEqual(
            "repos/example/engine-artifacts/releases/assets/123", calls[1].args[0][-1]
        )
        self.assertIn("Accept: application/octet-stream", calls[1].args[0])
        for call in calls:
            self.assertEqual("fake-private-token", call.kwargs["env"]["GH_TOKEN"])
            self.assertNotIn("fake-private-token", " ".join(call.args[0]))
            for name in (
                "GH_DEBUG",
                "GH_HOST",
                "ZZZ_GITHUB_TOKEN",
                "ZZZ_RELEASE_REPOSITORY",
            ):
                self.assertNotIn(name, call.kwargs["env"])
            self.assertEqual(subprocess.PIPE, call.kwargs["stderr"])

    def test_cached_archive_is_verified_without_auth_or_network(self):
        archive, pin = self.make_archive()
        with mock.patch.dict(os.environ, self.environment, clear=True):
            with mock.patch.object(
                zzz.subprocess,
                "run",
                side_effect=self.fake_gh(self.release_response(pin, archive), archive),
            ):
                cached = zzz.fetch_archive(pin, self.root / "cache")
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(zzz.subprocess, "run") as run,
        ):
            self.assertEqual(cached, zzz.fetch_archive(pin, self.root / "cache"))
            run.assert_not_called()
            cached.write_bytes(b"corrupt cache")
            with self.assertRaisesRegex(zzz.StageError, "checksum"):
                zzz.fetch_archive(pin, self.root / "cache")
            run.assert_not_called()

    def test_missing_or_ambiguous_assets_never_download(self):
        archive, pin = self.make_archive()
        for variant in ("missing", "duplicate", "wrong-tag", "draft"):
            with self.subTest(variant=variant):
                release = self.release_response(pin, archive)
                if variant == "missing":
                    release["assets"][0]["name"] = "other-archive.tar.gz"
                elif variant == "duplicate":
                    release["assets"] *= 2
                elif variant == "wrong-tag":
                    release["tag_name"] = "v0.0.0"
                else:
                    release["draft"] = True
                with mock.patch.dict(os.environ, self.environment, clear=True):
                    with mock.patch.object(
                        zzz.subprocess,
                        "run",
                        side_effect=self.fake_gh(release, archive),
                    ) as run:
                        with self.assertRaises(zzz.StageError):
                            zzz.fetch_archive(pin, self.root / variant)
                        self.assertEqual(1, run.call_count)

    def test_failed_download_never_publishes_cache_or_leaves_temporary_files(self):
        archive, pin = self.make_archive()
        response = self.release_response(pin, archive)
        archive.write_bytes(b"bad download")
        response["assets"][0]["size"] = archive.stat().st_size
        with mock.patch.dict(os.environ, self.environment, clear=True):
            with mock.patch.object(
                zzz.subprocess, "run", side_effect=self.fake_gh(response, archive)
            ):
                with self.assertRaisesRegex(zzz.StageError, "checksum"):
                    zzz.fetch_archive(pin, self.root / "cache")
        self.assertFalse(list((self.root / "cache").rglob("archive.tar.gz")))
        self.assertFalse(list((self.root / "cache").rglob(".fetch-*")))

    def test_local_gh_auth_is_explicit_and_ci_requires_dedicated_token(self):
        with mock.patch.dict(
            os.environ,
            {"ZZZ_RELEASE_REPOSITORY": "example/engine-artifacts"},
            clear=True,
        ):
            with self.assertRaisesRegex(zzz.StageError, "ZZZ_GITHUB_TOKEN"):
                zzz.github_environment(False)
            repository, environment = zzz.github_environment(True)
            self.assertEqual("example/engine-artifacts", repository)
            self.assertNotIn("GH_TOKEN", environment)
            os.environ["CI"] = "true"
            with self.assertRaisesRegex(zzz.StageError, "ZZZ_GITHUB_TOKEN"):
                zzz.github_environment(True)
        for locator in (
            "",
            "owner",
            "owner/repo/extra",
            "https://example.com/repo",
            "owner/repo?token=secret",
        ):
            with mock.patch.dict(
                os.environ,
                {**self.environment, "ZZZ_RELEASE_REPOSITORY": locator},
                clear=True,
            ):
                with self.assertRaisesRegex(zzz.StageError, "ZZZ_RELEASE_REPOSITORY"):
                    zzz.github_environment(False)

    def test_public_errors_never_echo_private_api_errors_or_paths(self):
        archive, pin = self.make_archive()
        failures = [
            subprocess.CompletedProcess(
                [],
                1,
                b"example/engine-artifacts",
                b"fake-private-token example/engine-artifacts",
            ),
            OSError("example/engine-artifacts fake-private-token"),
            subprocess.TimeoutExpired(
                "example/engine-artifacts fake-private-token", 120
            ),
            subprocess.CompletedProcess(
                [], 0, b"example/engine-artifacts fake-private-token", b""
            ),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                stderr = io.StringIO()
                with (
                    mock.patch.dict(os.environ, self.environment, clear=True),
                    contextlib.redirect_stderr(stderr),
                ):
                    with mock.patch.object(zzz, "select_pin", return_value=pin):
                        with mock.patch.object(
                            zzz.subprocess,
                            "run",
                            side_effect=failure
                            if isinstance(failure, Exception)
                            else None,
                            return_value=failure,
                        ):
                            self.assertEqual(
                                1,
                                zzz.main(
                                    [
                                        "--target",
                                        self.target,
                                        "--cache-dir",
                                        str(self.root / "cache"),
                                    ]
                                ),
                            )
                self.assertNotIn("example/engine-artifacts", stderr.getvalue())
                self.assertNotIn("fake-private-token", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stderr(stderr),
        ):
            with mock.patch.object(zzz, "select_pin", return_value=pin):
                self.assertEqual(
                    1,
                    zzz.main(
                        [
                            "--target",
                            self.target,
                            "--archive",
                            str(self.root / "example/engine-artifacts/missing"),
                        ]
                    ),
                )
        self.assertNotIn("example/engine-artifacts", stderr.getvalue())

    def test_override_and_verify_only_validate_the_staged_bytes(self):
        archive, pin = self.make_archive()
        staged = zzz.stage_archive(archive, pin, self.root / "staged")
        with mock.patch.dict(
            os.environ, {"XYBRID_ZZZ_PREBUILT_DIR": str(staged)}, clear=True
        ):
            with (
                mock.patch.object(zzz, "select_pin", return_value=pin),
                mock.patch.object(zzz, "fetch_archive") as fetch,
            ):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(
                        0, zzz.main(["--target", self.target, "--verify-only"])
                    )
                self.assertEqual(str(staged), stdout.getvalue().strip())
                fetch.assert_not_called()
                (staged / pin.entry["library"]).write_bytes(b"tampered")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        1, zzz.main(["--target", self.target, "--verify-only"])
                    )
                fetch.assert_not_called()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(zzz, "select_pin", return_value=pin),
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    1, zzz.main(["--target", self.target, "--verify-only"])
                )

    def test_unsupported_and_duplicate_pins_are_rejected(self):
        with self.assertRaisesRegex(zzz.StageError, "unsupported"):
            zzz.select_pin("wasm32-unknown-unknown")
        self.make_archive()
        self.manifest_data["archives"].append(self.manifest_data["archives"][0])
        self.manifest.write_text(json.dumps(self.manifest_data))
        with self.assertRaisesRegex(zzz.StageError, "duplicate"):
            zzz.select_pin(self.target, self.manifest)

    def test_links_and_public_directories_are_rejected(self):
        archive, pin = self.make_archive()
        link = self.root / "linked-archive"
        link.symlink_to(archive)
        with self.assertRaisesRegex(zzz.StageError, "links"):
            zzz.inspect_archive(link, pin)
        destination = self.root / "public"
        destination.mkdir(mode=0o755)
        with self.assertRaisesRegex(zzz.StageError, "private permissions"):
            zzz.stage_archive(archive, pin, destination)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO inputs are POSIX-only")
    def test_fifo_input_is_rejected_without_blocking(self):
        archive, pin = self.make_archive()
        fifo = self.root / "fifo-input"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(zzz.StageError, "regular input"):
            zzz.inspect_archive(fifo, pin)


if __name__ == "__main__":
    unittest.main()
