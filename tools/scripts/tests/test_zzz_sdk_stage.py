"""Offline checks for the trusted Apple SDK candidate producer."""

import copy
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("zzz_sdk_stage", ROOT / "tools/scripts/zzz_sdk_stage.py")
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)
GIT = shutil.which("git", path=os.defpath)
POSIX = STAGE.PRODUCER.posix_host() and hasattr(os, "O_DIRECTORY") and os.scandir in os.supports_fd


def github():
    return {"GITHUB_REPOSITORY": "xybrid-ai/xybrid", "GITHUB_REF": "refs/heads/master",
            "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_SHA": "a" * 40,
            "GITHUB_WORKFLOW_SHA": "a" * 40,
            "GITHUB_WORKFLOW_REF": "xybrid-ai/xybrid/.github/workflows/trusted-zzz-sdk.yml@refs/heads/master",
            "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}


class ProducerGuardTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_exact_master_dispatch_uses_runtime_sha_and_attempt_one(self):
        context = STAGE.github_context(github())
        self.assertEqual(context["workflow_commit"], "a" * 40)
        self.assertEqual(context["run_id"], 123)
        self.assertEqual(context["run_attempt"], 1)

    def test_forged_repository_ref_event_workflow_sha_and_attempt_fail(self):
        cases = [("GITHUB_REPOSITORY", "fork/xybrid"), ("GITHUB_REF", "refs/tags/master"),
                 ("GITHUB_REF", "refs/heads/release/v0.14.0"), ("GITHUB_EVENT_NAME", "push"),
                 ("GITHUB_EVENT_NAME", "pull_request_target"), ("GITHUB_SHA", "master"),
                 ("GITHUB_WORKFLOW_SHA", "b" * 40), ("GITHUB_RUN_ATTEMPT", "2"),
                 ("GITHUB_RUN_ATTEMPT", "01"), ("GITHUB_RUN_ID", "0"),
                 ("GITHUB_WORKFLOW_REF", "xybrid-ai/xybrid/.github/workflows/other.yml@refs/heads/master")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                changed = dict(github(), **{key: value})
                with self.assertRaises(STAGE.Error):
                    STAGE.github_context(changed)
        with self.assertRaises(STAGE.Error):
            STAGE.github_context({})

    def test_every_credential_name_is_refused_without_echoing_values(self):
        for name in STAGE.FORBIDDEN_ENV:
            with self.subTest(name=name), patch.dict(os.environ, {name: "DO-NOT-ECHO"}, clear=True):
                with self.assertRaises(STAGE.Error) as error:
                    STAGE.credential_free()
                self.assertNotIn("DO-NOT-ECHO", str(error.exception))

    def test_schema4_missing_ios_or_changed_notice_contract_cannot_reach_readiness(self):
        manifest = {"schema_version": 1, "engine": "zzz", "release_tag": "v0.1.0-rc.9",
                    "archives": [{"target": target, "receipt_schema": 5,
                                  "notice_files": dict(STAGE.PULL.SCHEMA5_NOTICE_FILES)}
                                 for target in STAGE.STAGING_TARGETS]}
        STAGE.schema5_apple_manifest(manifest)
        for alter in ["schema4", "missing-target", "notice-change", "duplicate-target", "frozen"]:
            with self.subTest(alter=alter):
                changed = copy.deepcopy(manifest)
                if alter == "schema4":
                    changed["archives"][0]["receipt_schema"] = 4
                elif alter == "missing-target":
                    changed["archives"].pop()
                elif alter == "notice-change":
                    changed["archives"][0]["notice_files"]["RUNTIME_TERMS.md"] = "0" * 64
                elif alter == "duplicate-target":
                    changed["archives"].append(changed["archives"][0])
                else:
                    changed["release_tag"] = "v0.1.0-rc.8"
                with self.assertRaises(STAGE.Error):
                    STAGE.schema5_apple_manifest(changed)

    def test_actual_reviewed_manifest_fails_closed_before_fetch_build_or_secret_step(self):
        import json
        manifest = json.loads((ROOT / "crates/zzz-sys/natives-manifest.json").read_bytes())
        with self.assertRaisesRegex(STAGE.Error, "schema-5"):
            STAGE.schema5_apple_manifest(manifest)

    @unittest.skipUnless(POSIX, "Apple command specification uses POSIX absolute paths")
    def test_isolated_build_plan_uses_actual_target_and_no_mutable_rc_or_remote_cache(self):
        command, environment = STAGE.apple_build_plan(Path("/isolated/private"), Path("/isolated/staged"))
        self.assertEqual(command[0], "bazelisk")
        self.assertIn("//bindings/apple:XybridFFI_zzz", command)
        self.assertIn("--nosystem_rc", command)
        self.assertIn("--nohome_rc", command)
        self.assertIn("--noworkspace_rc", command)
        self.assertIn("--bazelrc=/dev/null", command)
        self.assertIn("--//:tts_zzz=true", command)
        for flag in ["--remote_executor=", "--remote_cache=", "--bes_backend=", "--disk_cache=",
                     "--noremote_accept_cached", "--noremote_upload_local_results"]:
            self.assertIn(flag, command)
        self.assertFalse(any("--config=" in arg for arg in command))
        self.assertIn("--@rules_rust//rust/settings:lto=fat", command)
        self.assertIn("--ios_minimum_os=16.0", command)
        self.assertEqual(environment["CARGO_TARGET_DIR"], "/isolated/private/cargo-target")
        self.assertEqual(environment["BAZELISK_HOME"], "/isolated/private/bazelisk")
        self.assertFalse(set(environment) & set(STAGE.FORBIDDEN_ENV))
        self.assertNotIn("RUSTC_WRAPPER", environment)
        self.assertNotIn("SCCACHE_DIR", environment)
        self.assertTrue(all("import" not in arg for arg in command))

    @unittest.skipUnless(POSIX, "upload filesystem fixtures require POSIX descriptors")
    def test_upload_allowlist_contains_only_named_final_sdk_and_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            package = directory / "packages/xybrid-apple-v0.14.0.xcframework.zip"
            package.parent.mkdir()
            package.write_bytes(b"PK\x03\x04synthetic final SDK container")
            (directory / "source-binding.json").write_bytes(b"{}\n")
            members = STAGE.upload_members(directory, "0.14.0")
            self.assertEqual(set(members), {"packages/xybrid-apple-v0.14.0.xcframework.zip", "source-binding.json"})
            for name in ["libzzz_embed.a", "RECEIPT.json", "private/staged", "xybrid.rlib", "unexpected.zip"]:
                with self.subTest(name=name):
                    extra = directory / name
                    extra.parent.mkdir(parents=True, exist_ok=True)
                    extra.write_bytes(b"never upload")
                    with self.assertRaises(STAGE.Error):
                        STAGE.upload_members(directory, "0.14.0")
                    extra.unlink()
                    if extra.parent != directory:
                        extra.parent.rmdir()

    @unittest.skipUnless(POSIX, "upload filesystem fixtures require POSIX descriptors")
    def test_upload_bounds_links_and_wrong_package_container_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            package = directory / "packages/xybrid-apple-v0.14.0.xcframework.zip"
            package.parent.mkdir()
            package.write_bytes(b"!<arch>\nraw engine")
            (directory / "source-binding.json").write_bytes(b"{}\n")
            with self.assertRaises(STAGE.Error):
                STAGE.upload_members(directory, "0.14.0")
            package.write_bytes(b"PK\x03\x04sdk")
            with patch.object(STAGE, "MAX_UPLOAD_BYTES", 3), self.assertRaises(STAGE.Error):
                STAGE.upload_members(directory, "0.14.0")
            package.unlink()
            package.symlink_to(directory / "source-binding.json")
            with self.assertRaises((STAGE.Error, OSError)):
                STAGE.upload_members(directory, "0.14.0")

    def test_preupload_rechecks_context_before_staging_or_final_package_reads(self):
        with patch.object(STAGE, "upload_members") as inspect:
            with self.assertRaises(STAGE.Error):
                STAGE.pre_upload(Path("/tooling"), Path("/repo"), "b" * 40, "0.14.0", "2026-10-24",
                                 dict(github(), GITHUB_REF="refs/tags/master"), Path("/private"))
            inspect.assert_not_called()

    def test_preflight_refuses_credentials_before_any_git_or_candidate_input(self):
        with patch.dict(os.environ, {"ZZZ_GITHUB_TOKEN": "private-value"}, clear=True), \
                patch.object(STAGE.SOURCE, "Git") as git:
            with self.assertRaisesRegex(STAGE.Error, "credential-free"):
                STAGE.preflight(Path("/private-tooling"), Path("/private-data"), "b" * 40,
                                "0.14.0", "2026-10-24", github())
            git.assert_not_called()

    @unittest.skipUnless(GIT and POSIX, "fixed-origin command specification requires POSIX system Git")
    def test_fixed_origin_object_fetch_has_no_candidate_checkout_or_inherited_credentials(self):
        with patch.object(STAGE.subprocess, "run") as run:
            STAGE.fetch_candidate(Path("/data-only"), "b" * 40)
        command = run.call_args.args[0]
        self.assertIn("https://github.com/xybrid-ai/xybrid.git", command)
        self.assertEqual(command[-1], "b" * 40)
        self.assertIn("--no-recurse-submodules", command)
        self.assertIn("--no-write-fetch-head", command)
        self.assertIn("maintenance.auto=false", command)
        self.assertNotIn("checkout", command)
        self.assertFalse(set(run.call_args.kwargs["env"]) & set(STAGE.FORBIDDEN_ENV))
        self.assertEqual(run.call_args.kwargs["timeout"], 180)
        with patch.object(STAGE.subprocess, "run") as run, self.assertRaises(STAGE.SOURCE.PolicyError):
            STAGE.fetch_candidate(Path("/data-only"), "master; do-not-run")
        run.assert_not_called()

    def test_cli_private_errors_have_no_paths_input_values_or_success_record(self):
        arguments = ["stage", "prepare", "--tooling", "/private-sentinel/tooling", "--repo", "/private-sentinel/repo",
                     "--candidate", "DO-NOT-ECHO", "--version", "0.14.0", "--release-date", "2026-10-24",
                     "--root", "/private-sentinel/root"]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", arguments), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            result = STAGE.main()
        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("private-sentinel", stderr.getvalue())
        self.assertNotIn("DO-NOT-ECHO", stderr.getvalue())
        self.assertLess(len(stderr.getvalue()), 512)

    def test_unsupported_host_fails_before_git_or_candidate_data(self):
        with patch.object(STAGE.PRODUCER, "posix_host", return_value=False), \
                patch.object(STAGE.SOURCE, "Git") as git:
            with self.assertRaisesRegex(STAGE.Error, "POSIX"):
                STAGE.preflight(Path("/private-tooling"), Path("/private-data"), "b" * 40,
                                "0.14.0", "2026-10-24", github())
            git.assert_not_called()

    @unittest.skipUnless(GIT and POSIX, "actual pin ordering check requires POSIX system Git")
    def test_current_pins_block_real_preflight_before_candidate_fetch_or_source_check(self):
        master = STAGE.SOURCE.Git(ROOT).read("rev-parse", "HEAD", limit=64).decode().strip()
        context = dict(github(), GITHUB_SHA=master, GITHUB_WORKFLOW_SHA=master)
        with patch.object(STAGE, "fetch_candidate") as fetch, \
                patch.object(STAGE.SOURCE, "source_binding") as binding:
            with self.assertRaisesRegex(STAGE.Error, "schema-5"):
                STAGE.preflight(ROOT, ROOT, "b" * 40, "0.14.0", "2026-10-24", context)
            fetch.assert_not_called()
            binding.assert_not_called()

    def test_workflow_has_one_success_upload_immediately_after_guard_and_one_token_scope(self):
        text = (ROOT / ".github/workflows/trusted-zzz-sdk.yml").read_text()
        self.assertNotIn("download-artifact", text)
        self.assertIn("environment: zzz-engine", text)
        self.assertEqual(text.count("secrets.ZZZ_GITHUB_TOKEN"), 1)
        self.assertEqual(text.count("secrets.ZZZ_RELEASE_REPOSITORY"), 1)
        self.assertEqual(text.count("uses: actions/upload-artifact@"), 1)
        self.assertLess(text.index('zzz_sdk_stage.py" prepare'), text.index("secrets.ZZZ_GITHUB_TOKEN"))
        self.assertLess(text.index('zzz_sdk_stage.py" download'), text.index('zzz_sdk_stage.py" build'))
        self.assertLess(text.index('zzz_sdk_stage.py" pre-upload'), text.index("uses: actions/upload-artifact@"))
        self.assertIn("name: zzz-sdk-apple-${{ github.run_id }}-attempt-1", text)
        self.assertIn("compression-level: 0", text)
        self.assertEqual(text.count("persist-credentials: false"), 2)
        self.assertIn("ref: ${{ github.sha }}", text)
        self.assertIn("path: trusted-tooling", text)
        self.assertIn("path: source-objects", text)
        self.assertIn("python3 -I", text)
        self.assertNotIn("if: always()", text)
        self.assertNotIn("continue-on-error:", text)


@unittest.skipUnless(GIT and POSIX, "local source-content fixtures require POSIX Git")
class ExactSourceTreeTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "candidate"
        self.source.mkdir()
        self.init(self.source)
        (self.source / "reviewed.txt").write_bytes(b"reviewed bytes")
        (self.source / ".gitignore").write_text("ignored\n")
        for path in STAGE.SUBMODULES:
            sub = self.source / path
            sub.mkdir(parents=True)
            self.init(sub)
            (sub / "source.c").write_text("/* raw source, never executed */\n")
            self.git(sub, "add", ".")
            self.git(sub, "commit", "-qm", "submodule")
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-qm", "superproject")
        self.commit = self.git(self.source, "rev-parse", "HEAD")

    def git(self, root, *args):
        return subprocess.check_output([GIT, "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                       env={"PATH": os.defpath, "GIT_CONFIG_NOSYSTEM": "1",
                                            "GIT_CONFIG_GLOBAL": os.devnull}, timeout=20).decode().strip()

    def init(self, root):
        self.git(root, "init", "-q")
        for key, value in {"core.hooksPath": os.devnull, "gc.auto": "0", "maintenance.auto": "false",
                           "user.name": "Stage Fixture", "user.email": "stage@example.invalid"}.items():
            self.git(root, "config", key, value)

    def verify(self):
        return STAGE.verify_materialized_source(self.source, self.source, self.commit)

    def test_exact_source_and_both_submodule_contents_pass_without_execution(self):
        result = self.verify()
        self.assertEqual(set(result), set(STAGE.SUBMODULES))

    def test_changed_superproject_ignored_or_executable_content_fails(self):
        for change in ["bytes", "ignored", "mode", "symlink"]:
            with self.subTest(change=change):
                original = self.source / "reviewed.txt"
                if change == "bytes":
                    original.write_bytes(b"changed bytes")
                elif change == "ignored":
                    (self.source / "ignored").write_bytes(b"untracked output")
                elif change == "mode":
                    original.chmod(0o755)
                else:
                    original.unlink()
                    original.symlink_to(self.root / "outside")
                with self.assertRaises((STAGE.Error, OSError)):
                    self.verify()
                if original.is_symlink():
                    original.unlink()
                original.write_bytes(b"reviewed bytes")
                original.chmod(0o644)
                if (self.source / "ignored").exists():
                    (self.source / "ignored").unlink()

    def test_submodule_dirty_content_extra_file_or_wrong_head_fails(self):
        sub = self.source / next(iter(STAGE.SUBMODULES))
        original = (sub / "source.c").read_bytes()
        (sub / "source.c").write_bytes(b"tampered")
        with self.assertRaises(STAGE.Error):
            self.verify()
        (sub / "source.c").write_bytes(original)
        (sub / "ignored").write_bytes(b"extra ignored bytes")
        with self.assertRaises(STAGE.Error):
            self.verify()
        (sub / "ignored").unlink()
        self.git(sub, "commit", "--allow-empty", "-qm", "wrong head")
        with self.assertRaises(STAGE.Error):
            self.verify()

    def test_missing_or_symlinked_submodule_is_not_a_gitlink_content_proof(self):
        sub = self.source / next(iter(STAGE.SUBMODULES))
        moved = self.root / "moved-submodule"
        sub.rename(moved)
        with self.assertRaises((STAGE.Error, OSError)):
            self.verify()
        sub.symlink_to(moved, target_is_directory=True)
        with self.assertRaises((STAGE.Error, OSError)):
            self.verify()

    def test_real_source_policy_runs_on_candidate_objects_without_candidate_code(self):
        spec = importlib.util.spec_from_file_location("stage_source_fixtures", ROOT / "tools/scripts/tests/test_zzz_source_policy.py")
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        files = fixtures.synthetic_files()
        files["MODULE.bazel"] = "\n".join('zzz_prebuilt_repository(name = "' + name + '", target = "' + target + '")'
                                                for name, target in STAGE.STAGING_REPOSITORIES.items()) + "\n"
        import json
        manifest = json.loads(STAGE.PULL.DEFAULT_MANIFEST.read_bytes())
        template = manifest["archives"][0]
        manifest["release_tag"] = "v0.1.0-rc.9"
        manifest["archives"] = []
        for target in STAGE.STAGING_TARGETS:
            entry = copy.deepcopy(template)
            if target in STAGE.PULL.NEW_TARGET_CONTRACTS:
                slice_name, cpu, minimum, libraries = STAGE.PULL.NEW_TARGET_CONTRACTS[target]
            else:
                original = next(row for row in json.loads(STAGE.PULL.DEFAULT_MANIFEST.read_bytes())["archives"]
                                if row["target"] == target)
                slice_name, cpu, minimum, libraries = (original[key] for key in
                                                      ("slice", "cpu", "deployment_minimum", "system_libraries"))
            entry.update(target=target, slice=slice_name, cpu=cpu, deployment_minimum=minimum,
                         system_libraries=libraries, receipt_schema=5, notice_files=STAGE.PULL.SCHEMA5_NOTICE_FILES,
                         archive=f"zzz-{STAGE.PULL.PROFILE}-{slice_name}-v0.1.0-rc.9.tar.gz")
            manifest["archives"].append(entry)
        files[STAGE.SOURCE.MANIFEST] = json.dumps(manifest)
        fixture_root = self.root / "source-policy-fixture"
        fixture_root.mkdir()
        repo = fixtures.Repository(fixture_root, files)
        repo.write(fixtures.canonical_bump(files, fixtures.OLD, fixtures.NEW, checksum=None))
        candidate = repo.commit()
        context = dict(github(), GITHUB_SHA=repo.base, GITHUB_WORKFLOW_SHA=repo.base)
        # Tooling must stay at the master object; a separate fresh checkout is
        # represented by the same repository with HEAD restored to that master.
        repo.git("checkout", "--detach", repo.base)
        with patch.object(STAGE, "fetch_candidate") as fetch, \
                patch.object(STAGE.SOURCE, "source_binding", wraps=STAGE.SOURCE.source_binding) as bind:
            binding = STAGE.preflight(repo.path, repo.path, candidate, fixtures.NEW, fixtures.DATE, context)
            self.assertEqual(binding["candidate_commit"], candidate)
            fetch.assert_called_once_with(repo.path, candidate)
            self.assertEqual(bind.call_count, 1)
            self.assertEqual(bind.call_args.kwargs, {"checksum": None})


@unittest.skipUnless(GIT and POSIX, "functional producer fixtures require POSIX Git")
class MaterializationTests(unittest.TestCase):
    setUp = ExactSourceTreeTests.setUp
    git = ExactSourceTreeTests.git
    init = ExactSourceTreeTests.init
    def test_raw_materialization_preserves_exact_bytes_and_two_submodules_without_checkout_filters(self):
        output = self.root / "materialized"
        private = self.root / "private"
        private.mkdir()
        sub_repos = {path: self.source / path for path in STAGE.SUBMODULES}
        STAGE.materialize_candidate(self.source, self.commit, output, sub_repos)
        facts = STAGE.verify_materialized_source(output, self.source, self.commit, sub_repos)
        self.assertEqual(set(facts), set(STAGE.SUBMODULES))
        self.assertEqual((output / "reviewed.txt").read_bytes(), b"reviewed bytes")
        self.assertTrue((output / ".git").read_text().startswith("gitdir: "))
        with self.assertRaises((STAGE.Error, OSError)):
            STAGE.materialize_candidate(self.source, self.commit, output, sub_repos)

    def test_materialization_refuses_escaping_tracked_symlinks_before_any_build(self):
        (self.source / "escaping").symlink_to("../../outside")
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-qm", "bad reviewed fixture")
        commit = self.git(self.source, "rev-parse", "HEAD")
        with self.assertRaises(STAGE.Error):
            STAGE.materialize_candidate(self.source, commit, self.root / "materialized", {})


class FunctionalPhaseTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_schema4_blocks_download_before_pull_or_secret_values_are_read(self):
        with patch.object(STAGE.PULL, "fetch_archive") as fetch, \
                patch.object(STAGE, "trusted_pins", side_effect=STAGE.Error("schema-5 required")):
            with self.assertRaisesRegex(STAGE.Error, "schema-5"):
                STAGE.download_engine(Path("/private"), github())
            fetch.assert_not_called()

    def test_token_phase_environment_passes_only_fixed_context_and_mirror_names(self):
        with patch.dict(os.environ, {"ZZZ_GITHUB_TOKEN": "not-a-real-token",
                                    "ZZZ_RELEASE_REPOSITORY": "fixture/release", "SENTINEL_OTHER": "private"}, clear=True):
            environment = STAGE.download_environment(Path("/private"))
        self.assertEqual(environment["ZZZ_GITHUB_TOKEN"], "not-a-real-token")
        self.assertEqual(environment["ZZZ_RELEASE_REPOSITORY"], "fixture/release")
        self.assertNotIn("SENTINEL_OTHER", environment)
        self.assertNotIn("GITHUB_TOKEN", environment)
        self.assertEqual(environment["HOME"], "/private/mirror-home")

    def test_build_environment_never_inherits_any_token_or_sccache_configuration(self):
        with patch.dict(os.environ, {"ZZZ_GITHUB_TOKEN": "not-a-real-token", "GITHUB_TOKEN": "other",
                                    "RUSTC_WRAPPER": "sccache", "BAZEL_OPTS": "remote-secret"}, clear=True):
            _command, environment = STAGE.apple_build_plan(Path("/private"), Path("/staged"))
        self.assertFalse(set(environment) & set(STAGE.FORBIDDEN_ENV))
        self.assertNotIn("RUSTC_WRAPPER", environment)
        self.assertNotIn("BAZEL_OPTS", environment)
        self.assertNotIn("not-a-real-token", str(environment))

    @unittest.skipUnless(POSIX, "isolated public mirror configuration uses POSIX paths")
    def test_sdk_mirror_is_isolated_data_and_rejects_credentials_or_config_injection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = ["bazelisk", "build", STAGE.APPLE_TARGET]
            result = STAGE.configure_sdk_mirror(command, root, "mirror.example/CLTools.pkg")
            self.assertEqual(result[-1], STAGE.APPLE_TARGET)
            self.assertIn("--downloader_config=" + str(root / "downloader.cfg"), result)
            self.assertIn("mirror.example/CLTools.pkg", (root / "downloader.cfg").read_text())
            for mirror in ["user:password@mirror.example/sdk", "mirror.example/sdk\nbuild --remote_cache=evil", "https://mirror.example/sdk"]:
                with self.subTest(mirror=mirror), self.assertRaises(STAGE.Error):
                    STAGE.configure_sdk_mirror(command, root, mirror)

    def test_closed_six_target_set_is_exactly_the_actual_module_repositories(self):
        import re
        declarations = re.findall(r'^zzz_prebuilt_repository\(name = "([^"]+)", target = "([^"]+)"\)$',
                                  (ROOT / "MODULE.bazel").read_text(), re.MULTILINE)
        self.assertEqual(len(declarations), 6)
        self.assertEqual(dict(declarations), STAGE.STAGING_REPOSITORIES)
        self.assertEqual(set(STAGE.STAGING_TARGETS), set(STAGE.PULL.ZIG_TARGETS))

    @unittest.skipUnless(POSIX, "private binding fixtures require POSIX descriptors")
    def test_saved_binding_reads_reject_links_writes_and_unbounded_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "binding"
            path.write_bytes(b"{}\n")
            with self.assertRaises(STAGE.Error):
                STAGE.read_binding(path)
            path.chmod(0o400)
            self.assertEqual(STAGE.read_binding(path), b"{}\n")
            link = Path(temporary) / "linked"
            os.link(path, link)
            with self.assertRaises(STAGE.Error):
                STAGE.read_binding(path)
            link.unlink()
            path.chmod(0o600)
            path.write_bytes(b"x" * (STAGE.MAX_BINDING + 1))
            path.chmod(0o400)
            with self.assertRaises(STAGE.Error):
                STAGE.read_binding(path)


@unittest.skipUnless(GIT and POSIX, "offline functional phase fixtures require POSIX Git")
class OfflineProducerFlowTests(unittest.TestCase):
    setUp_source = ExactSourceTreeTests.setUp
    git = ExactSourceTreeTests.git
    init = ExactSourceTreeTests.init

    def setUp(self):
        self.setUp_source()
        self.private = self.root / "private-state"
        self.binding = {"schema_version": 1, "candidate_commit": self.commit,
                        "producer_context": STAGE.github_context(github())}
        spec = importlib.util.spec_from_file_location("stage_pull_fixtures", ROOT / "tools/scripts/tests/test_zzz_pull.py")
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        fixture_module.zzz = STAGE.PULL
        self.pull = fixture_module.ZzzPullTests()
        self.pull.setUp()
        self.addCleanup(self.pull.doCleanups)
        self.pull.use_schema5()
        self.pull.add_future_target_pins()
        self.archives = {}
        for target in STAGE.STAGING_TARGETS:
            self.archives[target], _pin = self.pull.make_archive(target=target)
        manifest = patch.object(STAGE.PULL, "DEFAULT_MANIFEST", self.pull.manifest)
        manifest.start()
        self.addCleanup(manifest.stop)
        source_facts = patch.object(STAGE, "preflight", return_value=self.binding)
        source_facts.start()
        self.addCleanup(source_facts.stop)
        self.events = []
        self.native_calls = []
        pack_spec = importlib.util.spec_from_file_location("stage_package_fixtures", ROOT / "tools/scripts/tests/test_zzz_apple_package.py")
        pack_module = importlib.util.module_from_spec(pack_spec)
        pack_spec.loader.exec_module(pack_module)
        self.pack = pack_module.ApplePackageTests()
        self.pack.setUp()
        self.addCleanup(self.pack.doCleanups)
        contract = patch.object(pack_module.PACK.PULL, "SCHEMA5_NOTICE_FILES", STAGE.PULL.SCHEMA5_NOTICE_FILES)
        contract.start()
        self.addCleanup(contract.stop)
        for directory in ("device", "simulator"):
            for name, data in self.pull.notice_payload.items():
                (self.pack.root / directory / name).write_bytes(data)
        self.pack.package()
        self.packager = pack_module.PACK

    def prepare(self):
        origins = {origin: self.source / path for path, origin in STAGE.SUBMODULES.items()}
        def local_fetch(repo, commit, origin):
            self.events.append("fetch-submodule")
            STAGE.SOURCE.Git(repo).read("-c", "protocol.file.allow=always", "fetch", "--quiet", "--no-tags",
                                      "--no-recurse-submodules", str(origins[origin]), commit, limit=4096)
        with patch.object(STAGE, "fetch_objects", side_effect=local_fetch):
            STAGE.prepare(ROOT, self.source, self.commit, "0.14.0", "2026-10-24", github(), self.private)
        self.events.append("source-materialized-and-verified")

    def broker(self):
        def pull(command, **kwargs):
            self.assertIn("source-materialized-and-verified", self.events)
            self.assertEqual(command[:2], [sys.executable, "-I"])
            self.assertEqual(Path(command[2]).name, "zzz_pull.py")
            self.assertNotIn("candidate", command[2])
            environment = kwargs["environment"]
            self.assertEqual(environment["ZZZ_GITHUB_TOKEN"], "fake-unit-token")
            self.assertFalse(set(environment) & {"GITHUB_TOKEN", "BUILDBUDDY_API_KEY", "RUSTC_WRAPPER"})
            target = command[command.index("--target") + 1]
            pin = STAGE.PULL.select_pin(target, self.pull.manifest)
            STAGE.PULL.stage_archive(self.archives[target], pin, self.private / "download-stage")
            self.events.append("trusted-pull-only")
        with patch.dict(os.environ, {"ZZZ_GITHUB_TOKEN": "fake-unit-token",
                                    "ZZZ_RELEASE_REPOSITORY": "fixture/release"}, clear=True), \
                patch.object(STAGE, "run_checked", side_effect=pull):
            STAGE.download_engine(self.private, github())

    def fake_native(self, command, **kwargs):
        self.native_calls.append(list(command))
        environment = kwargs["environment"]
        self.assertFalse(set(environment) & set(STAGE.FORBIDDEN_ENV))
        self.assertNotIn("RUSTC_WRAPPER", environment)
        self.assertNotIn("SCCACHE_DIR", environment)
        STAGE.verify_materialized_source(self.private / "candidate", self.source, self.commit,
                                         STAGE.source_stores(self.private))
        STAGE.verify_staging(self.private)
        if "build" in command:
            self.events.append("isolated-fake-build")
        elif "cquery" in command:
            relative = "bazel-out/ios-opt/bin/bindings/apple/" + STAGE.APPLE_ARCHIVE
            archive = self.private / "build/bazel-output/execroot/_main" / relative
            archive.parent.mkdir(mode=0o700, parents=True)
            archive.write_bytes(self.pack.output.read_bytes())  # Real notice overlay; only native bytes are fixtures.
            return (relative + "\n").encode()
        elif "info" in command:
            self.assertEqual(command[-1], "execution_root")
            return (str(self.private / "build/bazel-output/execroot/_main") + "\n").encode()
        else:
            self.assertEqual(command[2], str(ROOT / "bindings/apple/pack_zzz_notices.py"))
            self.assertEqual(command[3:5], ["verify", "--archive"])
            self.packager.verify_archive(Path(command[-1]))  # Actual trusted verifier body, no native execution.
            self.events.append("trusted-final-package-verifier")
        return b""

    def build(self, native=None):
        plan = STAGE.apple_build_plan
        which = STAGE.shutil.which
        def installed_plan(*args):
            command, environment = plan(*args)
            environment["DEVELOPER_DIR"] = str(self.root)  # Test-only installed-Xcode stand-in.
            return command, environment
        with patch.object(STAGE.sys, "platform", "darwin"), \
                patch.object(STAGE.shutil, "which", side_effect=lambda name, **kwargs: which(name, **kwargs)
                             if name == "git" else "/trusted/preinstalled-tool"), \
                patch.object(STAGE, "apple_build_plan", side_effect=installed_plan), \
                patch.object(STAGE, "run_checked", side_effect=native or self.fake_native):
            STAGE.build_candidate(ROOT, self.source, self.commit, "0.14.0", "2026-10-24", github(), self.private)

    def test_source_then_token_broker_then_fresh_verified_stage_and_isolated_final_package(self):
        self.prepare()
        self.broker()
        self.build()
        self.assertLess(self.events.index("source-materialized-and-verified"), self.events.index("trusted-pull-only"))
        self.assertLess(self.events.index("trusted-pull-only"), self.events.index("isolated-fake-build"))
        with patch.object(STAGE, "run_checked", side_effect=self.fake_native):
            name, members = STAGE.pre_upload(ROOT, self.source, self.commit, "0.14.0", "2026-10-24", github(), self.private)
        self.assertEqual(name, "zzz-sdk-apple-123-attempt-1")
        self.assertEqual(set(members), {"packages/xybrid-apple-v0.14.0.xcframework.zip", "source-binding.json"})
        self.assertEqual(self.events.count("trusted-pull-only"), 6)
        self.assertEqual(sum("build" in command for command in self.native_calls), 1)
        self.assertEqual(self.events.count("trusted-final-package-verifier"), 2)

    def test_corrupted_download_stage_is_rejected_before_any_fake_native_command(self):
        self.prepare()
        self.broker()
        pin = next(iter(STAGE.trusted_pins().values()))
        downloaded = (self.private / "download-stage").joinpath(*pin.key)
        (downloaded / "RUNTIME_TERMS.md").write_bytes(b"changed notice")
        with self.assertRaises(STAGE.PULL.StageError):
            self.build()
        self.assertEqual(self.native_calls, [])

    def test_source_mutation_during_fake_build_blocks_final_acceptance(self):
        self.prepare()
        self.broker()
        def mutate(command, **kwargs):
            if "build" in command:
                result = self.fake_native(command, **kwargs)
                (self.private / "candidate/reviewed.txt").write_bytes(b"changed during build")
                return result
            if "cquery" in command:
                # Return output without invoking the fixture's independent source check.
                relative = "bazel-out/ios-opt/bin/bindings/apple/" + STAGE.APPLE_ARCHIVE
                archive = self.private / "build/bazel-output/execroot/_main" / relative
                archive.parent.mkdir(mode=0o700, parents=True)
                archive.write_bytes(b"PK\x03\x04synthetic")
                return (relative + "\n").encode()
            return self.fake_native(command, **kwargs)
        with self.assertRaisesRegex(STAGE.Error, "materialized candidate bytes differ"):
            self.build(native=mutate)
        self.assertNotIn("trusted-final-package-verifier", self.events)

    def test_saved_binding_cannot_authorize_changed_submodule_or_extra_upload_member(self):
        self.prepare()
        self.broker()
        self.build()
        submodule = self.private / "candidate" / next(iter(STAGE.SUBMODULES)) / "source.c"
        original = submodule.read_bytes()
        submodule.write_bytes(b"changed after building")
        with self.assertRaisesRegex(STAGE.Error, "submodule bytes differ"):
            STAGE.pre_upload(ROOT, self.source, self.commit, "0.14.0", "2026-10-24", github(), self.private)
        submodule.write_bytes(original)
        (self.private / "upload/RECEIPT.json").write_bytes(b"never upload")
        with patch.object(STAGE, "run_checked", side_effect=self.fake_native), self.assertRaisesRegex(STAGE.Error, "allowlist"):
            STAGE.pre_upload(ROOT, self.source, self.commit, "0.14.0", "2026-10-24", github(), self.private)

    def test_final_verifier_failure_prevents_a_candidate_result(self):
        self.prepare()
        self.broker()
        self.build()
        with patch.object(STAGE, "run_checked", side_effect=STAGE.Error("synthetic final notice mismatch")):
            with self.assertRaisesRegex(STAGE.Error, "final notice mismatch"):
                STAGE.pre_upload(ROOT, self.source, self.commit, "0.14.0", "2026-10-24", github(), self.private)

    def test_actual_final_notice_verifier_rejects_rewritten_notice_with_valid_zip_crc(self):
        self.prepare()
        self.broker()
        self.build()
        package = STAGE.final_package(self.private, "0.14.0")
        with zipfile.ZipFile(package) as archive:
            members = {info.filename: archive.read(info) for info in archive.infolist()}
        members[self.packager.NOTICE_ROOT + "RUNTIME_TERMS.md"] = b"changed terms, recomputed ZIP CRC"
        package.chmod(0o600)
        with zipfile.ZipFile(package, "w") as archive:
            for name, data in members.items():
                archive.writestr(self.packager.output_info(name), data)
        package.chmod(0o400)
        with patch.object(STAGE, "run_checked", side_effect=self.fake_native), self.assertRaises(self.packager.PackageError):
            STAGE.pre_upload(ROOT, self.source, self.commit, "0.14.0", "2026-10-24", github(), self.private)


if __name__ == "__main__":
    unittest.main()
