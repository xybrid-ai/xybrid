#!/usr/bin/env python3
"""Draft credential-free API/archive verification under an external trusted policy."""

import argparse
import base64
import binascii
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import re
import ssl
import stat
import sys
import threading
import time


REPOSITORY = "xybrid-ai/xybrid"
WORKFLOW = ".github/workflows/trusted-zzz-sdk.yml"  # Proposed; no adopted policy ships here.
API_HOST = "api.github.com"
API_ROOT = "/repos/" + REPOSITORY
MAX_JSON = 1024 * 1024
MAX_POLICY = 16 * 1024
MAX_DEPTH = 64
MAX_WORKFLOW = 128 * 1024
MAX_ARCHIVE = 512 * 1024 * 1024
MAX_PAGES = 10
CHUNK = 64 * 1024
REQUEST_SECONDS = 10
TOTAL_SECONDS = 120
FORBIDDEN_ENV = ("ZZZ_GITHUB_TOKEN", "ZZZ_RELEASE_REPOSITORY", "XYBRID_ZZZ_USE_GH_AUTH",
                 "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
                 "SSL_CERT_FILE", "SSL_CERT_DIR")


class VerificationError(Exception):
    """A required binding could not be independently established."""


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        raise VerificationError("invalid command-line arguments")


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def refuse_credentials():
    # Names only: never read, log, interpolate or forward these values.
    require(not any(name in os.environ for name in FORBIDDEN_ENV),
            "verification requires a credential-free environment without TLS overrides")


def posix_host():
    flags = (getattr(os, "O_NOFOLLOW", None), getattr(os, "O_NONBLOCK", None))
    return (os.name == "posix" and all(type(flag) is int and flag > 0 for flag in flags)
            and flags[0] != flags[1])


def require_posix_host():
    require(posix_host(), "producer verification requires POSIX O_NOFOLLOW and O_NONBLOCK")


def integer(value):
    require(type(value) is int and 0 < value < 2**63, "invalid positive integer")
    return value


def hex_string(value, length):
    require(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value),
            "invalid immutable digest")
    return value


def object_fields(value, fields):
    require(type(value) is dict and set(value) == set(fields), "unexpected policy fields")


def strict_json(data, limit):
    require(len(data) <= limit, "JSON byte bound exceeded")
    try:
        text = data.decode("utf-8")
        depth, quoted, escaped = 0, False, False
        for character in text:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in "[{":
                depth += 1
                require(depth <= MAX_DEPTH, "JSON depth bound exceeded")
            elif character in "]}":
                depth -= 1
                require(depth >= 0, "invalid JSON")

        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, "duplicate JSON key")
                result[key] = value
            return result

        def reject_number(_value):
            raise VerificationError("floating/nonfinite JSON number is outside this policy")

        return json.loads(text, object_pairs_hook=pairs, parse_float=reject_number,
                          parse_constant=reject_number)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise VerificationError("invalid JSON") from error


def policy_value(value):
    fields = {"schema_version", "repository", "repository_id", "workflow_id", "workflow_path",
              "approved_head_sha", "full_ref", "event", "producer_contract", "run_id",
              "run_attempt", "artifact_id", "artifact_role", "artifact_sha256"}
    object_fields(value, fields)
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "unsupported policy schema")
    require(value["repository"] == REPOSITORY and value["workflow_path"] == WORKFLOW,
            "unapproved repository/workflow path")
    require(value["full_ref"] == "refs/heads/master" and value["event"] == "workflow_dispatch",
            "only reviewed master-dispatch contracts are supported")
    for field in ("repository_id", "workflow_id", "run_id", "artifact_id", "run_attempt"):
        integer(value[field])
    require(value["run_attempt"] == 1, "rerun artifacts cannot be authenticated by this helper")
    hex_string(value["approved_head_sha"], 40)
    hex_string(value["artifact_sha256"], 64)
    contract = value["producer_contract"]
    object_fields(contract, {"kind", "workflow_sha256"})
    require(contract["kind"] == "ref-master-before-artifact-upload-v1",
            "an external reviewed ref-before-upload contract is required")
    hex_string(contract["workflow_sha256"], 64)
    role = value["artifact_role"]
    require(type(role) is str and re.fullmatch(r"[a-z][a-z0-9-]{0,31}", role),
            "invalid approved artifact role")
    return value


class GitHub:
    """Fixed-origin, anonymous REST reads; no URL from a response is followed."""

    def __init__(self):
        refuse_credentials()
        self.deadline = time.monotonic() + TOTAL_SECONDS
        self.requests = 0
        self.failed = False

    def get(self, path):
        require(not self.failed, "failed API transport cannot be reused")
        require((path == API_ROOT or path.startswith(API_ROOT + "/")) and
                not any(c in path for c in "\r\n#"),
                "API path is outside the fixed repository")
        self.requests += 1
        remaining = self.deadline - time.monotonic()
        require(self.requests <= MAX_PAGES + 9 and remaining > 0, "API request/time bound exceeded")
        result = queue.Queue(maxsize=1)
        connection = http.client.HTTPSConnection(API_HOST, timeout=min(5, remaining),
                                                context=ssl.create_default_context())

        def request():
            outcome = (False, None)
            try:
                outcome = (True, self._read(connection, path))
            except Exception:
                pass
            finally:
                try:
                    connection.close()
                except Exception:
                    outcome = (False, None)
            result.put(outcome)

        # Bound DNS/header/trickle reads as well as socket inactivity. The worker
        # owns closure: closing a buffered response from this thread could block
        # on its read lock and defeat the deadline.
        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        try:
            okay, value = result.get(timeout=min(REQUEST_SECONDS, remaining))
        except queue.Empty as error:
            self.failed = True
            raise VerificationError("GitHub API request deadline exceeded") from error
        if not okay or time.monotonic() > self.deadline:
            self.failed = True
            raise VerificationError("GitHub API request failed")
        return value

    def _read(self, connection, path):
        connection.request("GET", path, headers={
            "Accept": "application/vnd.github+json", "User-Agent": "xybrid-producer-verifier-draft",
            "X-GitHub-Api-Version": "2022-11-28", "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
        })
        response = connection.getresponse()
        require(response.status == 200, "GitHub API status is not successful")
        require(response.getheader("Content-Type", "").split(";", 1)[0] == "application/json",
                "GitHub API response is not JSON")
        require(response.getheader("Content-Encoding", "identity") == "identity",
                "encoded API responses are unsupported")
        length = response.getheader("Content-Length")
        require(length is None or (length.isascii() and length.isdigit() and int(length) <= MAX_JSON),
                "API response byte bound exceeded")
        chunks, size = [], 0
        while True:
            require(time.monotonic() <= self.deadline, "API time bound exceeded")
            block = response.read1(min(CHUNK, MAX_JSON + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            require(size <= MAX_JSON, "API response byte bound exceeded")
        require(length is None or size == int(length), "truncated API response")
        return strict_json(b"".join(chunks), MAX_JSON)


def repository(value, policy):
    require(type(value) is dict and integer(value.get("id")) == policy["repository_id"] and
            value.get("full_name") == REPOSITORY, "API repository identity mismatch")


def run_value(value, policy):
    require(type(value) is dict, "invalid API run")
    for field in ("id", "workflow_id", "run_attempt"):
        expected = "run_id" if field == "id" else field
        require(integer(value.get(field)) == policy[expected], "API run identity/attempt mismatch")
    require(value.get("head_sha") == policy["approved_head_sha"] and
            value.get("head_branch") == "master" and value.get("event") == policy["event"] and
            value.get("status") == "completed" and value.get("conclusion") == "success",
            "API run head/branch/event/result mismatch")
    require(value.get("path") in (WORKFLOW, WORKFLOW + "@master", WORKFLOW + "@refs/heads/master"),
            "API run workflow path mismatch")
    repository(value.get("repository"), policy)
    repository(value.get("head_repository"), policy)


def workflow_bytes(value, policy):
    require(type(value) is dict and value.get("type") == "file" and
            value.get("path") == WORKFLOW and value.get("encoding") == "base64",
            "historical workflow file metadata mismatch")
    size = integer(value.get("size"))
    require(size <= MAX_WORKFLOW, "workflow byte bound exceeded")
    encoded = value.get("content")
    require(type(encoded) is str and len(encoded) <= 2 * MAX_WORKFLOW, "invalid workflow encoding")
    try:
        raw = base64.b64decode(encoded.replace("\n", ""), validate=True)
    except (ValueError, binascii.Error) as error:
        raise VerificationError("invalid workflow encoding") from error
    require(len(raw) == size and hashlib.sha256(raw).hexdigest() ==
            policy["producer_contract"]["workflow_sha256"], "reviewed workflow bytes mismatch")
    blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0" + raw).hexdigest()
    require(hex_string(value.get("sha"), 40) == blob, "workflow Git blob identity mismatch")


def artifact_name(policy):
    return "zzz-sdk-" + policy["artifact_role"] + "-" + str(policy["run_id"]) + "-attempt-1"


def artifact_value(value, policy, now):
    require(type(value) is dict and integer(value.get("id")) == policy["artifact_id"] and
            value.get("name") == artifact_name(policy), "API artifact ID/name mismatch")
    size = integer(value.get("size_in_bytes"))
    require(size <= MAX_ARCHIVE and type(value.get("expired")) is bool and not value["expired"],
            "expired/oversized API artifact")
    require(value.get("digest") == "sha256:" + policy["artifact_sha256"], "API artifact digest mismatch")
    expires = value.get("expires_at")
    require(type(expires) is str and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", expires),
            "invalid artifact expiry")
    try:
        expiry = datetime.datetime.strptime(expires, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
    except ValueError as error:
        raise VerificationError("invalid artifact expiry") from error
    require(expiry > now, "API artifact has expired")
    run = value.get("workflow_run")
    require(type(run) is dict and integer(run.get("id")) == policy["run_id"] and
            integer(run.get("repository_id")) == policy["repository_id"] and
            integer(run.get("head_repository_id")) == policy["repository_id"] and
            run.get("head_sha") == policy["approved_head_sha"] and run.get("head_branch") == "master",
            "API artifact run/repository/head mismatch")
    return {key: value[key] for key in ("id", "name", "size_in_bytes", "digest", "expires_at")}


def artifact_membership(api, policy, now):
    total, entries, seen = None, [], set()
    for page in range(1, MAX_PAGES + 1):
        result = api.get(API_ROOT + "/actions/runs/" + str(policy["run_id"]) +
                         "/artifacts?per_page=100&page=" + str(page))
        require(type(result) is dict and type(result.get("total_count")) is int and
                0 < result["total_count"] <= MAX_PAGES * 100, "invalid/beyond-bound artifact count")
        if total is None:
            total = result["total_count"]
        require(total == result["total_count"], "artifact pagination changed")
        batch = result.get("artifacts")
        require(type(batch) is list and len(batch) == min(100, total - len(entries)),
                "incomplete artifact pagination")
        for value in batch:
            require(type(value) is dict, "invalid artifact list entry")
            identifier = integer(value.get("id"))
            require(identifier not in seen, "duplicate artifact ID across pages")
            seen.add(identifier)
            entries.append(value)
        if len(entries) == total:
            matching = [v for v in entries if v.get("name") == artifact_name(policy)]
            require(len(matching) == 1 and policy["artifact_id"] in seen,
                    "artifact membership/name is missing or ambiguous")
            return artifact_value(matching[0], policy, now)
    raise VerificationError("artifact pagination bound exceeded")


def archive_digest(path, expected_size):
    require_posix_host()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_ARCHIVE and
                before.st_size == expected_size, "archive must be a bounded regular file of exact API size")
        digest, count = hashlib.sha256(), 0
        while True:
            block = stream.read(min(CHUNK, before.st_size + 1 - count))
            if not block:
                break
            count += len(block)
            require(count <= before.st_size, "archive changed during hashing")
            digest.update(block)
        after = os.fstat(stream.fileno())
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        require(count == before.st_size and identity(before) == identity(after),
                "archive changed during hashing")
        return digest.hexdigest()


def verify(policy, run_id, artifact_id, archive, api=None, now=None):
    refuse_credentials()
    require_posix_host()
    policy = policy_value(policy)
    require(integer(run_id) == policy["run_id"] and integer(artifact_id) == policy["artifact_id"],
            "candidate selectors differ from external trusted policy")
    api = GitHub() if api is None else api  # Injected transports are for offline unit tests only.
    now = now or datetime.datetime.now(datetime.timezone.utc)
    repository(api.get(API_ROOT), policy)
    workflow = api.get(API_ROOT + "/actions/workflows/" + str(policy["workflow_id"]))
    require(type(workflow) is dict and integer(workflow.get("id")) == policy["workflow_id"] and
            workflow.get("path") == WORKFLOW and workflow.get("state") == "active",
            "API workflow identity/path mismatch")
    workflow_bytes(api.get(API_ROOT + "/contents/" + WORKFLOW + "?ref=" + policy["approved_head_sha"]), policy)
    run_path = API_ROOT + "/actions/runs/" + str(policy["run_id"])
    run_value(api.get(run_path), policy)
    run_value(api.get(run_path + "/attempts/1"), policy)
    artifact_path = API_ROOT + "/actions/artifacts/" + str(policy["artifact_id"])
    artifact = artifact_value(api.get(artifact_path), policy, now)
    require(artifact == artifact_membership(api, policy, now), "artifact endpoint/list disagree")
    digest = archive_digest(archive, artifact["size_in_bytes"])
    require(digest == policy["artifact_sha256"], "actual local archive digest mismatch")
    # The artifact API exposes no attempt field. Accept only a run that has
    # never been rerun, and check again after every other input has been read.
    require(artifact == artifact_value(api.get(artifact_path), policy,
                                      max(now, datetime.datetime.now(datetime.timezone.utc))),
            "artifact changed during verification")
    run_value(api.get(run_path), policy)
    return {"schema_version": 1, "verification": "GITHUB_API_AND_LOCAL_ARCHIVE_UNDER_EXTERNAL_POLICY",
            "repository": REPOSITORY, "repository_id": policy["repository_id"],
            "workflow_id": policy["workflow_id"], "workflow_path": WORKFLOW,
            "workflow_sha256": policy["producer_contract"]["workflow_sha256"],
            "approved_head_sha": policy["approved_head_sha"], "run_id": run_id, "run_attempt": 1,
            "api_head_branch": "master", "full_ref": policy["full_ref"],
            "full_ref_assurance": "EXTERNALLY_REVIEWED_IMMUTABLE_PRODUCER_CONTRACT",
            "artifact": artifact, "archive_sha256": digest,
            "source_binding": "NOT_VERIFIED_BY_THIS_HELPER", "activation": "NOT_AUTHORIZED"}


def main():
    parser = PrivateArgumentParser(description=__doc__)
    parser.add_argument("--trusted-policy", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--artifact-id", required=True, type=int)
    parser.add_argument("--archive", required=True, type=Path)
    try:
        arguments = parser.parse_args()
        refuse_credentials()
        require_posix_host()
        with arguments.trusted_policy.open("rb") as stream:
            policy = strict_json(stream.read(MAX_POLICY + 1), MAX_POLICY)
        value = verify(policy, arguments.run_id, arguments.artifact_id, arguments.archive)
        sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except (VerificationError, OSError, ValueError) as error:
        # Do not echo response bodies, URLs, local paths or environment values.
        message = str(error) if isinstance(error, VerificationError) else "input/transport failure"
        sys.stderr.write("producer verification failed: " + message + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
