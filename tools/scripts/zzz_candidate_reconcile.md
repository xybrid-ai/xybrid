# Apple candidate source reconciliation

`zzz_candidate_reconcile.py` joins the existing real
[producer verifier](zzz_producer_verify.md), [bounded receiver](zzz_candidate_extract.md)
and [source checker](zzz_source_policy.md). These checkers and their policies are
unchanged. This adds no live producer policy, numeric service IDs, artifact pin,
download, native build, feature/default change, publication or activation.

The protected caller independently supplies all three contracts below. They
must come from reviewed protected code or another approved trusted channel,
outside the candidate and its artifact. JSON files do not authenticate their
own authority. Missing fields, IDs, approved context or local source objects
fail closed. The artifact, extracted binding and previous success reports cannot
select or fill any trust anchor.

1. The existing producer policy: exact immutable reviewed workflow/head, real
   repository/workflow/run/artifact IDs, attempt 1, Apple role and raw uploaded
   Actions ZIP digest. The existing full-ref review obligation still applies.
2. The existing extraction contract: trusted master B, native candidate C and
   exact member kinds, sizes and hashes. This initial reconciliation boundary
   accepts exactly one independently approved `.xcframework.zip` SDK member and
   `source-binding.json`, with the existing Apple role/path/container guards.
3. A new source contract with exactly these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `producer_context` | Independently expected source-policy context: exact repository, workflow path/commit, full ref, event, run ID and attempt |
| `trusted_master_commit` | Full lowercase master B SHA; equals the producer head and extraction contract |
| `native_candidate_commit` | Full lowercase built-source C SHA; equals the extraction contract |
| `publication_candidate_commit` | `null`, or an independently approved full lowercase D SHA distinct from C |
| `release_version`, `release_date` | Independently approved source-policy version and calendar date |
| `sdk_package_member` | Exact SDK member path already pinned by the extraction contract |

The expected producer context must also agree with all corresponding producer
policy fields. The caller independently fetches the needed B/C/D Git objects
into a trusted local object store before invoking this tool. The tool performs
only controlled local raw Git reads, with the existing source-checker limits,
no replacement objects/filters/hooks, no candidate imports and no Git fetch.
Uncommitted working-tree files cannot replace those immutable source objects.
As in the existing source checker, gitlinks bind exact submodule commit IDs;
this is not another native build or a submodule/runtime qualification run.

```sh
python3 -I tools/scripts/zzz_candidate_reconcile.py \
  --trusted-producer-policy "$PROTECTED_PRODUCER_POLICY" \
  --trusted-extraction-contract "$PROTECTED_EXTRACTION_CONTRACT" \
  --trusted-source-contract "$PROTECTED_SOURCE_CONTRACT" \
  --run-id "$APPROVED_RUN_ID" --artifact-id "$APPROVED_ARTIFACT_ID" \
  --archive "$RAW_ACTIONS_ARCHIVE" --repo "$LOCAL_INDEPENDENT_GIT_OBJECTS" \
  --destination "$NEW_PRIVATE_DESTINATION"
```

This is a caller template, not populated policy or an adopted live handoff. The
CLI uses the existing anonymous fixed-origin GitHub verifier; it offers no API
override, saved metadata mode or offline success switch. Offline tests inject a
transport only through the Python API. Credentials and TLS override names are
refused before policy, archive or Git reads.

## C remains the built source; D is a separate proof

The actual receiver verifies the raw artifact through the real API verifier
before and after exact extraction. The reconciliation tool independently
recomputes B→C with the complete existing source policy and `checksum=None`,
then compares the extracted binding's complete canonical JSON with those facts.
A self-claimed source digest, candidate SHA, context or manifest/verifier hash
cannot override independent local Git facts.

If D is supplied, it must descend from C and retain the exact complete tree
paths and Git modes. Every other path must have the same raw Git entry as C.
Only `Package.swift` and `bindings/react-native/package.json` may differ. The
existing `check_metadata` renderer, with old and new version both set to the
approved version, proves those two files contain only their already-permitted
XCFramework checksum substitutions. The approved checksum is the actual nested
SDK ZIP's SHA-256, independently pinned by the extraction contract and checked
against actual member bytes by the receiver. It is never read from the binding
or accepted from a separate candidate checksum argument. An unchanged checksum
literal is acceptable if already equal to that digest.

The complete existing B→D policy is then recomputed independently using the
same nested ZIP digest. There is no expansion of metadata policy or general
code-change exception. The result records `native_candidate_commit` C and
`publication_candidate_commit` D separately, with separate tree digests. The
extracted original binding continues to name C. D is never relabeled as the
native source or treated as another built artifact.

## Fresh checks at use; reports grant no authority

After extraction and initial source/D proofs, the tool checks the complete
extracted inventory and held file bytes again and recomputes source/D facts
again. It then repeats the actual producer/archive verifier after the potentially
expensive local Git reads, immediately before the final bounded output-byte and
directory-identity pass. Newly observed reruns, expiry, changed raw
ZIPs, changed members or changed binding/source facts fail. Existing destinations
are never replaced or deleted. The receiver retains its own internal rollback
on extraction failure. After it returns, its API does not transfer the original
held output descriptor/identity; the orchestration therefore never recursively
deletes output on a later reconciliation failure. This preserves unrelated
replacement directories and leaves any received bytes in place under the
destination name, unmarked and without a success report. They look like a
successful output: only exit status 0 and the JSON result on stdout signal
reconciliation, so callers must check both and never treat the directory's
existence as success. A failed destination must not be consumed or used for a
retry; the protected caller must choose a new private destination.

The receiver's existing archive/count/depth/byte bounds and single-link regular
file checks remain in force. The source contract is at most 32 KiB; the extracted
binding remains at most 16 KiB. Subprocess/API diagnostics and input values are
not emitted on failure. All runtime assumptions of the existing checkers still
apply, including trusted stable local Git objects, OS/Python/clock/filesystem
and protected policy delivery. Checks are observations, not atomic filesystem
or remote transactions, and are not a sandbox against this UID/root.

Every later consumer must call this boundary again on the approved raw archive
with a new destination and independently supplied policy. A saved success JSON
is an unsigned point-in-time report; the API accepts no report as authority for
replacement raw archives, extracted files or source facts. The report explicitly
states `installed_sdk: NOT_QUALIFIED` and `activation: NOT_AUTHORIZED`.

Nested package notice/native contents, deployment-floor execution, installed
Swift/model behavior, streaming/cancellation and publication approval remain
separate gates. This boundary authenticates outer bytes and source provenance
under the supplied policies; it does not execute or install the nested package.

## Offline validation

```sh
python3 -m unittest discover -s tools/scripts/tests -p test_zzz_candidate_reconcile.py -v
```

Fixtures use real controlled local Git repositories, the actual source checker,
actual ZIP receiver and actual API verifier with synthetic service responses.
They use a tiny opaque nested ZIP, not native SDK code or installed qualification.
No real IDs/policy anchor, credentials, API request, private download or native
execution are generated or used.
