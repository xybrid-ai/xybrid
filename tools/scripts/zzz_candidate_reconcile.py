#!/usr/bin/env python3
"""Credential-free Apple candidate source reconciliation under external policy."""

import importlib.util
import json
import os
from pathlib import Path
import sys


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


RECEIVER = sibling("xybrid_reconcile_receiver", "zzz_candidate_extract.py")
SOURCE = sibling("xybrid_reconcile_source", "zzz_source_policy.py")
PRODUCER = RECEIVER.PRODUCER
Error, require = PRODUCER.VerificationError, PRODUCER.require
MAX_SOURCE_CONTRACT = 32 * 1024
CHECKSUM_PATHS = {"Package.swift", "bindings/react-native/package.json"}


def source_contract_value(value, policy, extraction):
    PRODUCER.object_fields(value, {"schema_version", "producer_context", "trusted_master_commit",
                                   "native_candidate_commit", "publication_candidate_commit",
                                   "release_version", "release_date", "sdk_package_member"})
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "unsupported source reconciliation contract")
    for field in ("trusted_master_commit", "native_candidate_commit"):
        SOURCE.sha(value[field])
    publication = value["publication_candidate_commit"]
    if publication is not None:
        SOURCE.sha(publication)
        require(publication != value["native_candidate_commit"], "publication candidate must be distinct from built source")
    require(value["trusted_master_commit"] == policy["approved_head_sha"] == extraction["trusted_master_commit"] and
            value["native_candidate_commit"] == extraction["candidate_commit"],
            "independently approved source commits disagree")
    context = SOURCE.producer_context(value["producer_context"], value["trusted_master_commit"])
    expected = {"repository": policy["repository"], "workflow_path": policy["workflow_path"],
                "workflow_commit": policy["approved_head_sha"], "ref": policy["full_ref"],
                "event": policy["event"], "run_id": policy["run_id"], "run_attempt": policy["run_attempt"]}
    require(SOURCE.canonical(context) == SOURCE.canonical(expected), "approved producer/source contexts disagree")
    SOURCE.version(value["release_version"])
    # The source checker validates the calendar date independently during every
    # recomputation. It must be supplied externally, never taken from a receipt.
    require(type(value["release_date"]) is str, "approved release date is required")
    package = value["sdk_package_member"]
    require(extraction["artifact_role"] == "apple" and policy["artifact_role"] == "apple" and
            type(package) is str and package.endswith(".xcframework.zip") and
            set(extraction["members"]) == {package, "source-binding.json"} and
            extraction["members"][package]["kind"] == "sdk-package",
            "reconciliation requires the exact approved Apple package and source binding")
    return value


def publication_proof(repo, contract, checksum):
    """Prove C→D is only the existing approved Swift/RN checksum rendering."""
    native, publication = contract["native_candidate_commit"], contract["publication_candidate_commit"]
    if publication is None:
        return None
    git = SOURCE.Git(repo)
    before, after = git.entries(native), git.entries(publication)
    require(set(before) == set(after) and all(before[path][:2] == after[path][:2] for path in before),
            "publication candidate changed source paths or modes")
    require(git.read("merge-base", native, publication, limit=64).decode().strip() == native,
            "publication candidate must descend from built source")
    require(all(before[path] == after[path] for path in before if path not in CHECKSUM_PATHS),
            "publication candidate changed bytes outside the two checksum files")
    for path in CHECKSUM_PATHS:
        require(path in before and before[path][0] == "100644", "regular checksum source file missing")
        # old == new forbids a second version/date change. The existing checker
        # renders only each named checksum and independently rejects other bytes.
        SOURCE.check_metadata(path, git.blob(before[path]), git.blob(after[path]),
                              contract["release_version"], contract["release_version"],
                              contract["release_date"], checksum, {})
    # Independently apply the complete unchanged B→D source policy as well.
    return SOURCE.source_binding(repo, contract["trusted_master_commit"], publication,
                                 contract["release_version"], contract["release_date"],
                                 contract["producer_context"], checksum=checksum)


def source_facts(repo, contract, output_fd, checksum):
    native = SOURCE.source_binding(repo, contract["trusted_master_commit"], contract["native_candidate_commit"],
                                   contract["release_version"], contract["release_date"],
                                   contract["producer_context"], checksum=None)
    # Read through the held output descriptor so these are the bytes the
    # output checks authenticate, not whatever the path names now.
    actual = RECEIVER.read_json("source-binding.json", RECEIVER.MAX_BINDING, dir_fd=output_fd)
    require(SOURCE.canonical(actual) == SOURCE.canonical(native),
            "extracted source binding differs from independently recomputed native source")
    return native, publication_proof(repo, contract, checksum)


def reconcile(policy, extraction, source, run_id, artifact_id, archive, destination, repo, api=None, now=None):
    """Reverify at every use; reports and previously extracted bytes grant no authority.

    API transport/time injection is for offline tests only. Production uses the
    existing anonymous fixed-origin verifier. Git objects must already exist in
    a caller-owned independently fetched local store; this tool never fetches.
    """
    PRODUCER.refuse_credentials()
    RECEIVER.require_host()
    policy = PRODUCER.policy_value(PRODUCER.strict_json(json.dumps(policy).encode(), PRODUCER.MAX_POLICY))
    extraction = RECEIVER.contract_value(PRODUCER.strict_json(json.dumps(extraction).encode(), RECEIVER.MAX_CONTRACT), policy)
    source = source_contract_value(PRODUCER.strict_json(json.dumps(source).encode(), MAX_SOURCE_CONTRACT), policy, extraction)
    # Fail on missing/unapproved local source facts before touching the archive.
    SOURCE.source_binding(repo, source["trusted_master_commit"], source["native_candidate_commit"],
                          source["release_version"], source["release_date"], source["producer_context"], checksum=None)
    destination = Path(destination)
    parent_fd = RECEIVER.parent_directory(destination)
    output_fd, result = None, None
    try:
        RECEIVER.receive(policy, extraction, run_id, artifact_id, archive, destination, api=api, now=now)
        # The receiver does not transfer its held output descriptor/identity.
        # Authenticate all returned bytes afresh, but never recursively delete
        # a directory first observed after that ownership handoff gap. A later
        # failure leaves the output in place, unmarked, with no success report:
        # only exit status 0 and the stdout result signal reconciliation.
        output_fd = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                            dir_fd=parent_fd)
        RECEIVER.same_directory(output_fd, parent_fd, destination.name)
        checksum = extraction["members"][source["sdk_package_member"]]["sha256"]
        # Receiver checked the actual nested package bytes against this digest;
        # no checksum is accepted from the candidate's binding or saved report.
        source_facts(repo, source, output_fd, checksum)
        RECEIVER.same_directory(output_fd, parent_fd, destination.name)
        RECEIVER.verify_output(output_fd, extraction)
        native, publication = source_facts(repo, source, output_fd, checksum)
        # Local Git/tree proofs can be substantial. Refresh remote attempt,
        # expiry and raw archive facts after them, immediately before the final
        # bounded member/identity pass rather than before another source read.
        PRODUCER.verify(policy, run_id, artifact_id, archive, api=api, now=now)
        RECEIVER.verify_output(output_fd, extraction)
        RECEIVER.same_directory(output_fd, parent_fd, destination.name)
        result = {"schema_version": 1, "reconciliation": "FRESH_PRODUCER_ARCHIVE_MEMBERS_AND_LOCAL_SOURCE",
                  "run_id": run_id, "artifact_id": artifact_id, "archive_sha256": policy["artifact_sha256"],
                  "trusted_master_commit": source["trusted_master_commit"],
                  "native_candidate_commit": source["native_candidate_commit"],
                  "native_tree_sha256": native["candidate_tree_sha256"],
                  "publication_candidate_commit": source["publication_candidate_commit"],
                  "publication_tree_sha256": publication["candidate_tree_sha256"] if publication else None,
                  "sdk_package_member": source["sdk_package_member"], "sdk_package_sha256": checksum,
                  "producer_context": source["producer_context"],
                  "release_version": source["release_version"], "release_date": source["release_date"],
                  "installed_sdk": "NOT_QUALIFIED", "activation": "NOT_AUTHORIZED"}
    finally:
        if output_fd is not None:
            os.close(output_fd)
        os.close(parent_fd)
    return result


def main():
    parser = PRODUCER.PrivateArgumentParser(description=__doc__)
    parser.add_argument("--trusted-producer-policy", required=True, type=Path)
    parser.add_argument("--trusted-extraction-contract", required=True, type=Path)
    parser.add_argument("--trusted-source-contract", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--artifact-id", required=True, type=int)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    try:
        arguments = parser.parse_args()
        PRODUCER.refuse_credentials()
        RECEIVER.require_host()
        result = reconcile(RECEIVER.read_json(arguments.trusted_producer_policy, PRODUCER.MAX_POLICY),
                           RECEIVER.read_json(arguments.trusted_extraction_contract, RECEIVER.MAX_CONTRACT),
                           RECEIVER.read_json(arguments.trusted_source_contract, MAX_SOURCE_CONTRACT),
                           arguments.run_id, arguments.artifact_id, arguments.archive,
                           arguments.destination, arguments.repo)
        sys.stdout.write(SOURCE.canonical(result).decode())
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, (Error, SOURCE.PolicyError)) else "invalid or unavailable reconciliation input"
        sys.stderr.write("candidate reconciliation failed: " + message + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
