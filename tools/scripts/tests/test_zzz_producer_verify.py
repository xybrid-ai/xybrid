"""Tiny offline fixtures; no adopted producer contract, credentials or payload downloads."""

import base64
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "zzz_producer_verify.py"
SPEC = importlib.util.spec_from_file_location("zzz_producer_verify", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
NOW = datetime.datetime(2026, 10, 22, tzinfo=datetime.timezone.utc)
# This synthetic contract is not an adopted workflow or an actual action pin.
WORKFLOW_BYTES = b'''name: fixture-only
on: workflow_dispatch
jobs:
  produce:
    runs-on: ubuntu-latest
    steps:
      - run: test "$GITHUB_REF" = refs/heads/master
      - uses: actions/upload-artifact@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
'''


def archive_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(zipfile.ZipInfo("sdk.bin", (2000, 1, 1, 0, 0, 0)), b"tiny final SDK fixture")
        archive.writestr(zipfile.ZipInfo("source-binding.json", (2000, 1, 1, 0, 0, 0)),
                         b'{"producer_authentication":"UNTRUSTED_SELF_CLAIM"}')
    return stream.getvalue()


def fixtures():
    raw = archive_bytes()
    policy = {"schema_version": 1, "repository": VERIFY.REPOSITORY, "repository_id": 100,
              "workflow_id": 200, "workflow_path": VERIFY.WORKFLOW, "approved_head_sha": "a" * 40,
              "full_ref": "refs/heads/master", "event": "workflow_dispatch",
              "producer_contract": {"kind": "ref-master-before-artifact-upload-v1",
                                    "workflow_sha256": hashlib.sha256(WORKFLOW_BYTES).hexdigest()},
              "run_id": 300, "run_attempt": 1, "artifact_id": 400, "artifact_role": "apple",
              "artifact_sha256": hashlib.sha256(raw).hexdigest()}
    repo = {"id": 100, "full_name": VERIFY.REPOSITORY}
    run = {"id": 300, "workflow_id": 200, "run_attempt": 1, "head_sha": "a" * 40,
           "head_branch": "master", "event": "workflow_dispatch", "status": "completed",
           "conclusion": "success", "path": VERIFY.WORKFLOW, "repository": repo,
           "head_repository": repo}
    artifact = {"id": 400, "name": VERIFY.artifact_name(policy), "size_in_bytes": len(raw),
                "expired": False, "digest": "sha256:" + policy["artifact_sha256"],
                "expires_at": "2099-01-01T00:00:00Z",
                "workflow_run": {"id": 300, "repository_id": 100, "head_repository_id": 100,
                                 "head_sha": "a" * 40, "head_branch": "master"}}
    root = VERIFY.API_ROOT
    responses = {
        root: repo,
        root + "/actions/workflows/200": {"id": 200, "path": VERIFY.WORKFLOW, "state": "active"},
        root + "/contents/" + VERIFY.WORKFLOW + "?ref=" + policy["approved_head_sha"]: {
            "type": "file", "path": VERIFY.WORKFLOW, "encoding": "base64", "size": len(WORKFLOW_BYTES),
            "content": base64.b64encode(WORKFLOW_BYTES).decode() + "\n",
            "sha": hashlib.sha1(b"blob " + str(len(WORKFLOW_BYTES)).encode() + b"\0" + WORKFLOW_BYTES).hexdigest()},
        root + "/actions/runs/300": run,
        root + "/actions/runs/300/attempts/1": run,
        root + "/actions/artifacts/400": artifact,
        root + "/actions/runs/300/artifacts?per_page=100&page=1": {
            "total_count": 1, "artifacts": [artifact]},
    }
    return policy, responses, raw


class FakeAPI:
    def __init__(self, responses):
        self.responses = {path: copy.deepcopy(value) for path, value in responses.items()}
        self.paths = []

    def get(self, path):
        self.paths.append(path)
        value = self.responses[path]
        if callable(value):
            value = value(self.paths.count(path))
        return copy.deepcopy(value)


@unittest.skipUnless(VERIFY.posix_host() and callable(getattr(os, "mkfifo", None)),
                     "POSIX archive fixtures require real O_NOFOLLOW/O_NONBLOCK and mkfifo")
class ProducerVerificationTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.archive = self.directory / "artifact.zip"
        self.policy, responses, raw = fixtures()
        self.api = FakeAPI(responses)
        self.archive.write_bytes(raw)
        self.run_path = VERIFY.API_ROOT + "/actions/runs/300"
        self.artifact_path = VERIFY.API_ROOT + "/actions/artifacts/400"
        self.list_path = self.run_path + "/artifacts?per_page=100&page=1"

    def verify(self, **kwargs):
        return VERIFY.verify(self.policy, 300, 400, self.archive, api=self.api, now=NOW, **kwargs)

    def test_fixture_policy_requires_live_metadata_reads_and_exact_archive_bytes(self):
        value = self.verify()
        self.assertEqual(value["archive_sha256"], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        self.assertEqual(value["full_ref_assurance"], "EXTERNALLY_REVIEWED_IMMUTABLE_PRODUCER_CONTRACT")
        self.assertEqual(value["api_head_branch"], "master")
        self.assertEqual(value["source_binding"], "NOT_VERIFIED_BY_THIS_HELPER")
        self.assertEqual(value["activation"], "NOT_AUTHORIZED")
        self.assertEqual(self.api.paths.count(self.run_path), 2)
        self.assertEqual(self.api.paths[-1], self.run_path)
        self.assertEqual(self.api.paths.count(self.artifact_path), 2)
        self.assertIn(self.run_path + "/attempts/1", self.api.paths)
        self.assertEqual(list(self.directory.iterdir()), [self.archive])  # No extraction.

    def test_candidate_cannot_replace_trusted_run_or_artifact_selectors(self):
        for run, artifact in ((301, 400), (300, 401), (True, 400)):
            with self.subTest(run=run, artifact=artifact), self.assertRaises(VERIFY.VerificationError):
                VERIFY.verify(self.policy, run, artifact, self.archive, api=self.api)
        self.assertEqual(self.api.paths, [])

    def test_missing_or_self_declared_contract_cannot_substitute_for_external_anchor(self):
        for value in (None, {"reviewed": True}, {"kind": "claimed-safe", "workflow_sha256": "b" * 64}):
            self.policy["producer_contract"] = value
            with self.subTest(value=value), self.assertRaises(VERIFY.VerificationError):
                self.verify()
        self.assertEqual(self.api.paths, [])

    def test_policy_types_paths_refs_heads_and_ids_fail_closed(self):
        changes = {"schema_version": True, "repository": "attacker/fork", "repository_id": True,
                   "workflow_id": 200.0, "workflow_path": ".github/workflows/release-prep.yml",
                   "approved_head_sha": "master", "full_ref": "refs/tags/master", "event": "push",
                   "run_id": 0, "run_attempt": 2, "artifact_id": "400", "artifact_role": "../raw",
                   "artifact_sha256": "SHA256:wrong", "extra": "not permitted"}
        for field, value in changes.items():
            supplied = copy.deepcopy(self.policy)
            supplied[field] = value
            with self.subTest(field=field), self.assertRaises(VERIFY.VerificationError):
                VERIFY.verify(supplied, 300, 400, self.archive, api=self.api)
        self.assertEqual(self.api.paths, [])

    def test_repository_workflow_and_historical_workflow_blob_are_independent_anchors(self):
        paths = [VERIFY.API_ROOT, VERIFY.API_ROOT + "/actions/workflows/200",
                 VERIFY.API_ROOT + "/contents/" + VERIFY.WORKFLOW + "?ref=" + "a" * 40]
        mutations = [(paths[0], "id", 101), (paths[0], "full_name", "attacker/fork"),
                     (paths[1], "path", ".github/workflows/other.yml"), (paths[1], "id", True),
                     (paths[1], "state", "disabled_manually"), (paths[2], "sha", "b" * 40),
                     (paths[2], "content", base64.b64encode(b"different code").decode()),
                     (paths[2], "encoding", "none"), (paths[2], "type", "symlink"),
                     (paths[2], "size", VERIFY.MAX_WORKFLOW + 1), (paths[2], "content", "***")]
        for path, field, value in mutations:
            api = FakeAPI(self.api.responses)
            api.responses[path][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(VERIFY.VerificationError):
                VERIFY.verify(self.policy, 300, 400, self.archive, api=api, now=NOW)

    def test_run_fork_event_head_result_or_attempt_mismatch_rejects(self):
        changes = {"id": 301, "workflow_id": 201, "run_attempt": 2, "head_sha": "b" * 40,
                   "head_branch": "release/v1", "event": "pull_request_target", "status": "in_progress",
                   "conclusion": "failure", "path": VERIFY.WORKFLOW + "@attacker",
                   "repository": {"id": 100, "full_name": "attacker/fork"},
                   "head_repository": {"id": 101, "full_name": VERIFY.REPOSITORY}}
        for field, value in changes.items():
            api = FakeAPI(self.api.responses)
            api.responses[self.run_path][field] = value
            with self.subTest(field=field), self.assertRaises(VERIFY.VerificationError):
                VERIFY.verify(self.policy, 300, 400, self.archive, api=api, now=NOW)

    def test_attempt_endpoint_must_independently_match_and_final_run_reread_rejects_rerun_race(self):
        self.api.responses[self.run_path + "/attempts/1"]["run_attempt"] = 2
        with self.assertRaises(VERIFY.VerificationError):
            self.verify()
        self.policy, responses, _ = fixtures()
        original = responses[self.run_path]
        changed = dict(original, run_attempt=2)
        self.api = FakeAPI(responses)
        self.api.responses[self.run_path] = lambda count: original if count == 1 else changed
        with self.assertRaises(VERIFY.VerificationError):
            self.verify()
        self.assertEqual(self.api.paths[-1], self.run_path)

    def test_artifact_missing_digest_expired_wrong_size_or_wrong_membership_rejects(self):
        changes = {"id": 401, "name": "zzz-sdk-apple-300-attempt-2", "size_in_bytes": True,
                   "expired": True, "digest": None, "expires_at": "2020-01-01T00:00:00Z",
                   "workflow_run": {"id": 301, "repository_id": 100, "head_repository_id": 100,
                                    "head_sha": "a" * 40, "head_branch": "master"}}
        for field, value in changes.items():
            api = FakeAPI(self.api.responses)
            api.responses[self.artifact_path][field] = value
            with self.subTest(field=field), self.assertRaises(VERIFY.VerificationError):
                VERIFY.verify(self.policy, 300, 400, self.archive, api=api, now=NOW)
        for field in ("id", "repository_id", "head_repository_id", "head_sha", "head_branch"):
            api = FakeAPI(self.api.responses)
            api.responses[self.artifact_path]["workflow_run"][field] = 999 if field.endswith("id") else "other"
            with self.subTest(nested=field), self.assertRaises(VERIFY.VerificationError):
                VERIFY.verify(self.policy, 300, 400, self.archive, api=api, now=NOW)

    def test_duplicate_expected_name_missing_id_duplicate_id_and_incomplete_pages_reject(self):
        artifact = self.api.responses[self.artifact_path]
        other = dict(artifact, id=401)
        cases = [{"total_count": 2, "artifacts": [artifact, other]},
                 {"total_count": 1, "artifacts": [other]},
                 {"total_count": 2, "artifacts": [artifact, artifact]},
                 {"total_count": 101, "artifacts": [artifact]},
                 {"total_count": True, "artifacts": [artifact]},
                 {"total_count": VERIFY.MAX_PAGES * 100 + 1, "artifacts": []}]
        for value in cases:
            api = FakeAPI(self.api.responses)
            api.responses[self.list_path] = value
            with self.subTest(value=value["total_count"]), self.assertRaises(VERIFY.VerificationError):
                VERIFY.verify(self.policy, 300, 400, self.archive, api=api, now=NOW)

    def test_complete_pagination_accepts_late_match_and_rejects_count_or_id_race(self):
        filler = [{"id": 1000 + i, "name": "unrelated-" + str(i)} for i in range(100)]
        self.api.responses[self.list_path] = {"total_count": 101, "artifacts": filler}
        page2 = self.run_path + "/artifacts?per_page=100&page=2"
        self.api.responses[page2] = {"total_count": 101, "artifacts": [self.api.responses[self.artifact_path]]}
        self.assertEqual(self.verify()["artifact"]["id"], 400)
        self.api.responses[page2]["total_count"] = 102
        with self.assertRaises(VERIFY.VerificationError):
            self.verify()
        self.api.responses[page2] = {"total_count": 101, "artifacts": [filler[0]]}
        with self.assertRaises(VERIFY.VerificationError):
            self.verify()

    def test_endpoint_list_and_final_artifact_reread_must_agree(self):
        self.api.responses[self.list_path]["artifacts"][0]["expires_at"] = "2098-01-01T00:00:00Z"
        with self.assertRaises(VERIFY.VerificationError):
            self.verify()
        self.policy, responses, _ = fixtures()
        original = responses[self.artifact_path]
        changed = dict(original, expired=True)
        self.api = FakeAPI(responses)
        self.api.responses[self.artifact_path] = lambda count: original if count == 1 else changed
        with self.assertRaises(VERIFY.VerificationError):
            self.verify()

    def test_hashes_wrapper_not_extracted_members_or_self_declared_receipt(self):
        self.policy["artifact_sha256"] = hashlib.sha256(b"tiny final SDK fixture").hexdigest()
        for value in (self.api.responses[self.artifact_path],
                      self.api.responses[self.list_path]["artifacts"][0]):
            value["digest"] = "sha256:" + self.policy["artifact_sha256"]
        with self.assertRaisesRegex(VERIFY.VerificationError, "actual local archive digest"):
            self.verify()

    def test_archive_size_digest_and_regular_file_requirements(self):
        self.archive.write_bytes(self.archive.read_bytes().replace(b"tiny", b"evil"))
        with self.assertRaisesRegex(VERIFY.VerificationError, "archive digest"):
            self.verify()
        for path in (self.directory, self.directory / "link", self.directory / "pipe"):
            if path.name == "link":
                path.symlink_to(self.archive)
            if path.name == "pipe":
                os.mkfifo(path)
            with self.subTest(path=path.name), self.assertRaises((VERIFY.VerificationError, OSError)):
                VERIFY.archive_digest(path, 1)
        for content in (b"", b"oversize"):
            self.archive.write_bytes(content)
            with patch.object(VERIFY, "MAX_ARCHIVE", 4), self.assertRaises(VERIFY.VerificationError):
                VERIFY.archive_digest(self.archive, len(content))

    def test_archive_post_stat_mutation_is_detected_on_the_same_open_descriptor(self):
        real = os.fstat
        calls = []

        def changed(descriptor):
            result = real(descriptor)
            calls.append(descriptor)
            if len(calls) == 1:
                return result
            return types.SimpleNamespace(st_dev=result.st_dev, st_ino=result.st_ino, st_size=result.st_size,
                                         st_mtime_ns=result.st_mtime_ns + 1, st_ctime_ns=result.st_ctime_ns)

        with patch.object(VERIFY.os, "fstat", changed), self.assertRaisesRegex(VERIFY.VerificationError, "changed"):
            VERIFY.archive_digest(self.archive, self.archive.stat().st_size)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])

    def test_credentials_refused_before_policy_file_or_api_or_archive_reads(self):
        for name in VERIFY.FORBIDDEN_ENV:
            with patch.dict(os.environ, {name: "DO-NOT-READ-OR-PRINT"}), \
                    patch.object(VERIFY.Path, "open") as opened, patch.object(VERIFY, "GitHub") as api, \
                    patch.object(VERIFY.sys, "argv", [str(SCRIPT), "--trusted-policy", "missing.json",
                                                    "--run-id", "300", "--artifact-id", "400",
                                                    "--archive", "missing.zip"]), \
                    patch.object(VERIFY.sys, "stderr", io.StringIO()) as errors:
                self.assertEqual(VERIFY.main(), 1)
                opened.assert_not_called()
                api.assert_not_called()
                self.assertNotIn("DO-NOT-READ-OR-PRINT", errors.getvalue())


class JsonAndCliTests(unittest.TestCase):
    def test_cli_parse_errors_never_echo_inputs_or_read_policy_archive_or_api(self):
        required = ["--trusted-policy", "/private/DO-NOT-ECHO/policy-secretlike.json",
                    "--run-id", "300", "--artifact-id", "400",
                    "--archive", "/private/DO-NOT-ECHO/archive-secretlike.zip"]
        malformed_integer = list(required)
        malformed_integer[3] = "/private/TOKEN-LIKE-RUN-ID"
        unknown_option = required + ["--unknown-sensitive-option", "TOKEN-LIKE-VALUE-/private/path"]
        for arguments in (malformed_integer, unknown_option):
            with self.subTest(arguments=arguments), patch.object(VERIFY.Path, "open") as opened, \
                    patch.object(VERIFY, "archive_digest") as archive, \
                    patch.object(VERIFY, "GitHub") as api, \
                    patch.object(VERIFY.sys, "argv", [str(SCRIPT), *arguments]), \
                    patch.object(VERIFY.sys, "stdout", io.StringIO()) as output, \
                    patch.object(VERIFY.sys, "stderr", io.StringIO()) as errors:
                self.assertEqual(VERIFY.main(), 1)
                self.assertEqual(output.getvalue(), "")
                self.assertEqual(errors.getvalue(),
                                 "producer verification failed: invalid command-line arguments\n")
                opened.assert_not_called()
                archive.assert_not_called()
                api.assert_not_called()

    def test_json_types_duplicates_nonfinite_encoding_and_depth_are_bounded(self):
        for data in (b'{"x":1,"x":2}', b"NaN", b"1.0", b"1e999", b'"\xff"',
                     b"[" * 2000 + b"0" + b"]" * 2000, b'\xff\xfe{\x00}\x00'):
            with self.subTest(data=data[:24]), self.assertRaises(VERIFY.VerificationError):
                VERIFY.strict_json(data, 8192)
        quoted = json.dumps("[" * 2000 + '\\"' + "]" * 2000 + "\\").encode()
        self.assertEqual(VERIFY.strict_json(quoted, 8192), json.loads(quoted))
        self.assertIsNotNone(VERIFY.strict_json(b"[" * 64 + b"0" + b"]" * 64, 8192))
        with self.assertRaises(VERIFY.VerificationError):
            VERIFY.strict_json(b"{}", 1)

    def test_cli_has_no_offline_metadata_or_api_origin_override(self):
        with patch.object(VERIFY.sys, "argv", [str(SCRIPT), "--help"]), \
                patch.object(VERIFY.sys, "stdout", io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                VERIFY.main()
        self.assertEqual(result.exception.code, 0)
        self.assertNotIn("api-base", output.getvalue())
        self.assertNotIn("metadata-json", output.getvalue())


class PlatformBoundaryTests(unittest.TestCase):
    def test_missing_or_invalid_posix_flags_fail_before_any_cli_input_or_api_read(self):
        cases = [{"name": "nt", "O_NOFOLLOW": 256, "O_NONBLOCK": 4},
                 {"name": "posix", "O_NONBLOCK": 4},
                 {"name": "posix", "O_NOFOLLOW": 256}]
        cases += [{"name": "posix", "O_NOFOLLOW": flag, "O_NONBLOCK": 4}
                  for flag in (None, False, True, 0, -1, 256.0, "256", 4)]
        cases += [{"name": "posix", "O_NOFOLLOW": 256, "O_NONBLOCK": flag}
                  for flag in (None, False, True, 0, -1, 4.0, "4")]
        for flags in cases:
            system = types.SimpleNamespace(environ={}, **flags)
            with self.subTest(flags=flags), patch.object(VERIFY, "os", system), \
                    patch.object(VERIFY.Path, "open") as opened, \
                    patch.object(VERIFY, "archive_digest") as archive, \
                    patch.object(VERIFY, "GitHub") as api, \
                    patch.object(VERIFY.sys, "argv", [str(SCRIPT), "--trusted-policy", "missing.json",
                                                    "--run-id", "300", "--artifact-id", "400",
                                                    "--archive", "missing.zip"]), \
                    patch.object(VERIFY.sys, "stdout", io.StringIO()) as output, \
                    patch.object(VERIFY.sys, "stderr", io.StringIO()) as errors:
                self.assertEqual(VERIFY.main(), 1)
                self.assertIn("requires POSIX", errors.getvalue())
                self.assertEqual(output.getvalue(), "")
                opened.assert_not_called()
                archive.assert_not_called()
                api.assert_not_called()

    def test_imported_verification_and_archive_open_share_the_platform_gate(self):
        system = types.SimpleNamespace(name="nt", O_NOFOLLOW=256, O_NONBLOCK=4, environ={})
        with patch.object(VERIFY, "os", system), patch.object(VERIFY, "policy_value") as policy, \
                patch.object(VERIFY, "GitHub") as api, patch.object(VERIFY.Path, "open") as opened:
            with self.assertRaisesRegex(VERIFY.VerificationError, "requires POSIX"):
                VERIFY.verify({}, 300, 400, "not-opened", api=FakeAPI({}))
            with self.assertRaisesRegex(VERIFY.VerificationError, "requires POSIX"):
                VERIFY.archive_digest("not-opened", 1)
            policy.assert_not_called()
            api.assert_not_called()
            opened.assert_not_called()


class Response:
    def __init__(self, body=b'{"ok":true}', status=200, **headers):
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = {"Content-Type": "application/json", "Content-Length": str(len(body)), **headers}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, size):
        return self.body.read(size)


class Connection:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.closed = False

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class TransportTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def request(self, response):
        connection = Connection(response)
        with patch.object(VERIFY.http.client, "HTTPSConnection", return_value=connection) as factory:
            result = VERIFY.GitHub().get(VERIFY.API_ROOT)
        return result, connection, factory

    def test_fixed_https_origin_and_no_credentials_proxies_or_response_urls_are_followed(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "https://user:password@evil.invalid"}):
            result, connection, factory = self.request(Response())
        self.assertEqual(result, {"ok": True})
        self.assertEqual(factory.call_args.args, ("api.github.com",))
        self.assertEqual(len(connection.requests), 1)
        args, kwargs = connection.requests[0]
        self.assertEqual(args, ("GET", VERIFY.API_ROOT))
        self.assertEqual(set(kwargs["headers"]),
                         {"Accept", "User-Agent", "X-GitHub-Api-Version", "Accept-Encoding", "Cache-Control"})
        self.assertTrue(connection.closed)

    def test_redirect_error_non_json_encoding_size_and_truncation_reject(self):
        cases = [Response(status=302, Location="https://evil.invalid"), Response(status=401),
                 Response(**{"Content-Type": "text/html"}), Response(**{"Content-Encoding": "gzip"}),
                 Response(**{"Content-Length": str(VERIFY.MAX_JSON + 1)}),
                 Response(**{"Content-Length": "999"}), Response(body=b'{"x":1,"x":2}'),
                 Response(body=b"{" * (VERIFY.MAX_JSON + 1), **{"Content-Length": None})]
        for response in cases:
            with self.subTest(status=response.status), self.assertRaises(VERIFY.VerificationError):
                self.request(response)

    def test_fixed_repository_paths_request_count_and_total_deadline_are_bounded(self):
        api = VERIFY.GitHub()
        for path in ("https://evil.invalid", "/repos/attacker/fork", VERIFY.API_ROOT + "/x\nBad: yes"):
            with self.subTest(path=path), self.assertRaises(VERIFY.VerificationError):
                api.get(path)
        api.deadline = 0
        with patch.object(VERIFY.http.client, "HTTPSConnection") as connection, \
                self.assertRaises(VERIFY.VerificationError):
            api.get(VERIFY.API_ROOT)
        connection.assert_not_called()
        api.deadline = VERIFY.time.monotonic() + 60
        api.requests = VERIFY.MAX_PAGES + 9
        with self.assertRaises(VERIFY.VerificationError):
            api.get(VERIFY.API_ROOT)

    def test_request_deadline_bounds_even_a_stalled_header_worker(self):
        released = threading.Event()
        closed = threading.Event()
        connection = Connection(Response())
        connection.getresponse = lambda: (released.wait(1), Response())[1]
        connection.close = lambda: closed.set()
        api = VERIFY.GitHub()
        try:
            with patch.object(VERIFY.http.client, "HTTPSConnection", return_value=connection) as factory, \
                    patch.object(VERIFY, "REQUEST_SECONDS", 0.01):
                with self.assertRaisesRegex(VERIFY.VerificationError, "deadline"):
                    api.get(VERIFY.API_ROOT)
                self.assertFalse(closed.is_set())  # No cross-thread buffered close.
                with self.assertRaisesRegex(VERIFY.VerificationError, "cannot be reused"):
                    api.get(VERIFY.API_ROOT)
                self.assertEqual(factory.call_count, 1)
        finally:
            released.set()
        self.assertTrue(closed.wait(1))


if __name__ == "__main__":
    unittest.main()
