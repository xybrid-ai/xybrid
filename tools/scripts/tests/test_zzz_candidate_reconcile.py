"""Synthetic external policy + real local Git/receiver/source checks; no network."""

import copy
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REC = module("reconciliation", HERE.parent / "zzz_candidate_reconcile.py")
FIX = module("reconcile_receiver_fixtures", HERE / "test_zzz_candidate_extract.py")
SRC = module("reconcile_source_fixtures", HERE / "test_zzz_source_policy.py")


@unittest.skipUnless(SRC.GIT and REC.PRODUCER.posix_host() and hasattr(os, "O_DIRECTORY") and
                     getattr(REC.RECEIVER.shutil.rmtree, "avoids_symlink_attacks", False),
                     "reconciliation fixtures require controlled Git and POSIX descriptors")
class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        # Keep only this boundary's real workspace/source/checksum inputs.
        # Full cross-SDK metadata rendering is covered by the unchanged source
        # checker suite; duplicating its unrelated manifests slows every Git proof.
        needed = {"Cargo.toml", "Cargo.lock", "crates/demo/Cargo.toml", "crates/other/Cargo.toml",
                  REC.SOURCE.MANIFEST, REC.SOURCE.VERIFIER, "Package.swift",
                  "bindings/react-native/package.json", "README.md", "src/frozen.rs"}
        files = {name: data for name, data in SRC.synthetic_files().items() if name in needed}
        files[REC.SOURCE.WORKFLOW] = FIX.FIXTURES.WORKFLOW_BYTES.decode()
        repo_path = self.root / "objects"
        repo_path.mkdir()
        self.repo = SRC.Repository(repo_path, files)
        self.native_files = SRC.canonical_bump(files, SRC.OLD, SRC.NEW, checksum=None)
        self.repo.write(self.native_files)
        self.native = self.repo.commit()
        self.context = {"repository": "xybrid-ai/xybrid", "workflow_path": REC.SOURCE.WORKFLOW,
                        "workflow_commit": self.repo.base, "ref": "refs/heads/master",
                        "event": "workflow_dispatch", "run_id": 300, "run_attempt": 1}
        self.package = "packages/xybrid-apple-v" + SRC.NEW + ".xcframework.zip"
        self.members = {self.package: FIX.sdk_fixture(), "source-binding.json": REC.SOURCE.canonical(
            REC.SOURCE.source_binding(self.repo.path, self.repo.base, self.native, SRC.NEW, SRC.DATE,
                                      self.context, checksum=None))}
        self.source = {"schema_version": 1, "producer_context": self.context,
                       "trusted_master_commit": self.repo.base, "native_candidate_commit": self.native,
                       "publication_candidate_commit": None, "release_version": SRC.NEW,
                       "release_date": SRC.DATE, "sdk_package_member": self.package}
        self.archive = self.root / "artifact.zip"
        self.destination = self.root / "reconciled"
        self.publish()

    def publish(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name, data in self.members.items():
                info = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data)
        raw = stream.getvalue()
        self.archive.write_bytes(raw)
        self.policy, responses, _ = FIX.FIXTURES.fixtures()
        self.policy.update(approved_head_sha=self.repo.base, artifact_sha256=hashlib.sha256(raw).hexdigest())
        def update(value):
            if type(value) is dict:
                if "head_sha" in value:
                    value["head_sha"] = self.repo.base
                if "size_in_bytes" in value:
                    value.update(size_in_bytes=len(raw), digest="sha256:" + self.policy["artifact_sha256"])
                for child in value.values():
                    update(child)
            elif type(value) is list:
                for child in value:
                    update(child)
        update(responses)
        responses = {key.replace("?ref=" + "a" * 40, "?ref=" + self.repo.base): value
                     for key, value in responses.items()}
        self.api = FIX.FIXTURES.FakeAPI(responses)
        self.extraction = {"schema_version": 1, "artifact_role": "apple",
                           "trusted_master_commit": self.repo.base, "candidate_commit": self.native,
                           "source_binding_member": "source-binding.json", "members": {
                               name: {"kind": "source-binding" if name.endswith(".json") else "sdk-package",
                                      "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                               for name, data in self.members.items()}}

    def reconcile(self):
        return REC.reconcile(self.policy, self.extraction, self.source, 300, 400, self.archive,
                             self.destination, self.repo.path, api=self.api, now=FIX.FIXTURES.NOW)

    def rejected(self, message=None, *, received=False):
        with self.assertRaisesRegex((REC.Error, REC.SOURCE.PolicyError, OSError), message or ".*"):
            self.reconcile()
        self.assertEqual(received, self.destination.exists())
        self.assertEqual([], list(self.root.glob(".zzz-receive-*")))

    def publication(self, *, checksum=None, changes=None):
        checksum = checksum or hashlib.sha256(self.members[self.package]).hexdigest()
        files = SRC.canonical_bump(self.native_files, SRC.NEW, SRC.NEW, checksum=checksum)
        files.update(changes or {})
        self.repo.write(files)
        self.source["publication_candidate_commit"] = self.repo.commit()

    def test_real_receiver_and_git_source_binding_reconcile_native_source(self):
        result = self.reconcile()
        self.assertEqual(self.native, result["native_candidate_commit"])
        self.assertEqual(self.repo.base, result["trusted_master_commit"])
        self.assertIsNone(result["publication_candidate_commit"])
        self.assertEqual(hashlib.sha256(self.members[self.package]).hexdigest(), result["sdk_package_sha256"])
        self.assertEqual("NOT_AUTHORIZED", result["activation"])
        self.assertEqual("NOT_QUALIFIED", result["installed_sdk"])
        self.assertEqual(6, self.api.paths.count(REC.PRODUCER.API_ROOT + "/actions/runs/300"))

    def test_checksum_only_publication_keeps_c_and_d_distinct(self):
        self.publication()
        result = self.reconcile()
        self.assertEqual(self.native, result["native_candidate_commit"])
        self.assertEqual(self.source["publication_candidate_commit"], result["publication_candidate_commit"])
        self.assertNotEqual(result["native_tree_sha256"], result["publication_tree_sha256"])
        self.assertEqual(self.members["source-binding.json"], (self.destination / "source-binding.json").read_bytes())

    def test_binding_cannot_self_claim_d_as_native_or_select_another_context(self):
        binding = REC.PRODUCER.strict_json(self.members["source-binding.json"], REC.RECEIVER.MAX_BINDING)
        for field, value in (("candidate_commit", "b" * 40), ("producer_context", dict(self.context, run_id=999)),
                             ("engine_manifest_sha256", "0" * 64)):
            with self.subTest(field=field):
                self.destination = self.root / ("rejected-binding-" + field)
                changed = dict(binding, **{field: value})
                self.members["source-binding.json"] = REC.SOURCE.canonical(changed)
                self.publish()  # Even independently authenticated wrong binding bytes must fail local source proof.
                self.rejected("extracted source binding differs", received=True)

    def test_publication_rejects_wrong_checksum_and_any_unrelated_byte(self):
        self.publication(checksum="f" * 64)
        # Either checksum file may be checked first (set order); both reject.
        self.rejected("outside (approved version/checksum values|exact trusted metadata rendering)",
                      received=True)
        self.destination = self.root / "unrelated-publication"
        self.publication(changes={"README.md": "unapproved publication edit\n"})
        self.rejected("outside the two checksum files", received=True)

    def test_publication_rejects_added_path_or_mode_change(self):
        self.publication(changes={"extra.txt": "not allowed"})
        self.rejected("source paths or modes", received=True)

    def test_publication_rejects_executable_checksum_file(self):
        self.publication()
        (self.repo.path / "Package.swift").chmod(0o755)
        self.source["publication_candidate_commit"] = self.repo.commit()
        self.rejected("source paths or modes", received=True)

    def test_publication_must_descend_from_c_even_when_b_to_d_metadata_would_pass(self):
        self.repo.git("checkout", "--detach", self.repo.base)
        self.publication()
        self.rejected("descend from built source", received=True)

    def test_missing_ids_source_facts_and_policy_disagreement_fail_before_api(self):
        original = copy.deepcopy(self.source)
        for field, value in (("trusted_master_commit", "b" * 40), ("native_candidate_commit", "c" * 40),
                             ("publication_candidate_commit", self.native), ("release_version", None),
                             ("producer_context", dict(self.context, run_id=True))):
            with self.subTest(field=field):
                self.source = dict(copy.deepcopy(original), **{field: value})
                self.rejected()
                self.assertEqual([], self.api.paths)
        self.source = original
        self.policy["repository_id"] = None
        self.rejected()
        self.assertEqual([], self.api.paths)

    def test_local_missing_native_objects_are_not_replaced_by_binding_claims(self):
        self.source["native_candidate_commit"] = "c" * 40
        self.extraction["candidate_commit"] = "c" * 40
        self.rejected("local Git object request failed")
        self.assertEqual([], self.api.paths)

    def test_native_code_change_cannot_pass_as_approved_version_metadata(self):
        self.repo.write({"src/frozen.rs": "unapproved code"})
        changed = self.repo.commit()
        self.source["native_candidate_commit"] = changed
        self.extraction["candidate_commit"] = changed
        self.rejected("code or other nonmetadata source changed")
        self.assertEqual([], self.api.paths)

    def test_current_api_attempt_change_at_final_use_preserves_unreconciled_output(self):
        path = REC.PRODUCER.API_ROOT + "/actions/runs/300"
        valid = copy.deepcopy(self.api.responses[path])
        self.api.responses[path] = lambda count: dict(valid, run_attempt=2) if count == 6 else valid
        self.rejected("attempt mismatch", received=True)

    def test_final_producer_check_follows_last_local_source_proof(self):
        path = REC.PRODUCER.API_ROOT + "/actions/runs/300"
        valid = copy.deepcopy(self.api.responses[path])
        with patch.object(REC.SOURCE, "source_binding", wraps=REC.SOURCE.source_binding) as binding:
            def observe(count):
                if count >= 5:  # Third real producer verification, after receiver's two.
                    self.assertEqual(3, binding.call_count)
                return valid
            self.api.responses[path] = observe
            self.reconcile()

    def test_replacement_raw_archive_is_reverified_despite_saved_success(self):
        result = self.reconcile()
        self.assertEqual("FRESH_PRODUCER_ARCHIVE_MEMBERS_AND_LOCAL_SOURCE", result["reconciliation"])
        self.destination = self.root / "second-use"
        self.archive.write_bytes(self.archive.read_bytes() + b"replacement")
        self.rejected("archive snapshot digest or identity changed")

    def test_member_mutation_at_final_boundary_fails_even_with_unchanged_report(self):
        original = REC.SOURCE.source_binding
        count = 0
        def mutate(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                package = self.destination / self.package
                package.chmod(0o600)
                package.write_bytes(b"PK\x03\x04changed after verification")
                package.chmod(0o400)
            return original(*args, **kwargs)
        with patch.object(REC.SOURCE, "source_binding", side_effect=mutate):
            self.rejected("output member metadata changed", received=True)

    def test_post_receive_directory_replacement_is_never_recursively_deleted(self):
        receive = REC.RECEIVER.receive
        original_output = self.root / "original-receiver-output"
        def replace(*args, **kwargs):
            result = receive(*args, **kwargs)
            self.destination.rename(original_output)
            self.destination.mkdir(mode=0o700)
            (self.destination / "unrelated-sentinel").write_bytes(b"preserve unrelated bytes")
            return result
        with patch.object(REC.RECEIVER, "receive", side_effect=replace):
            self.rejected(received=True)
        self.assertEqual(b"preserve unrelated bytes", (self.destination / "unrelated-sentinel").read_bytes())
        self.assertEqual(self.members["source-binding.json"], (original_output / "source-binding.json").read_bytes())

    def test_uncommitted_worktree_bytes_are_never_executed_or_used_as_object_proof(self):
        (self.repo.path / "src/frozen.rs").write_text("changed working tree, not candidate objects")
        self.assertEqual(self.native, self.reconcile()["native_candidate_commit"])

    def test_extra_report_input_cannot_supply_missing_external_authority(self):
        self.source["saved_success"] = {"activation": "approved"}
        self.rejected("unexpected policy fields")
        self.assertEqual([], self.api.paths)

    def test_credentials_are_refused_before_source_or_archive_reads(self):
        with patch.dict(os.environ, {"ZZZ_GITHUB_TOKEN": "never-echo"}), patch.object(REC.SOURCE, "Git") as git:
            self.rejected("credential-free")
        git.assert_not_called()
        self.assertEqual([], self.api.paths)

    def test_existing_destination_is_never_overwritten_or_removed(self):
        self.destination.mkdir()
        (self.destination / "keep").write_text("keep")
        with self.assertRaisesRegex(REC.Error, "destination already exists"):
            self.reconcile()
        self.assertEqual("keep", (self.destination / "keep").read_text())


class ReconcileCliTests(unittest.TestCase):
    def test_private_errors_have_no_paths_tracebacks_or_success_json(self):
        arguments = ["reconcile", "--trusted-producer-policy", "/private/customer/policy",
                     "--trusted-extraction-contract", "/private/customer/extraction",
                     "--trusted-source-contract", "/private/customer/source", "--run-id", "1",
                     "--artifact-id", "2", "--archive", "/private/customer/archive",
                     "--destination", "/private/customer/output", "--repo", "/private/customer/git"]
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", arguments), \
                patch("sys.stdout", output), patch("sys.stderr", errors):
            self.assertEqual(1, REC.main())
        self.assertEqual("", output.getvalue())
        self.assertNotIn("customer", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
