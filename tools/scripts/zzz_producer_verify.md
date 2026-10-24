# Draft producer metadata and archive verifier

`zzz_producer_verify.py` is a bounded prerequisite for a future trusted SDK
handoff. It independently reads official GitHub REST metadata over anonymous
HTTPS and hashes an already downloaded local Actions artifact archive. It is
separate from [the source-policy checker](zzz_source_policy.md), whose
`producer_authentication` field remains `NOT_AUTHENTICATED_BY_THIS_HELPER`.

**No producer workflow or external policy is adopted by this change.** The
proposed `.github/workflows/trusted-zzz-sdk.yml` does not exist here. Positive
tests use an explicitly synthetic reviewed-contract fixture. They establish no
live handoff eligibility. There are no workflow, environment, native pin, API,
feature, credential policy, terms or release changes, and no public activation.

## Supported verification hosts

Producer verification is **POSIX-only**. It requires `os.name == "posix"` and
genuine OS-provided `O_NOFOLLOW` and `O_NONBLOCK` flags that are distinct positive
integers; absent, zero, boolean or otherwise invalid flags fail closed. There is
no Windows producer support or weaker file-opening fallback. The CLI checks
this boundary before policy/archive/API reads. Imported `verify()` and
`archive_digest()` enforce the same boundary, including with injected transports.
The trusted system Python/OS must provide the real primitives; manufacturing
numeric flags cannot make an unsupported runtime safe.

The POSIX archive-fixture test class also requires `os.mkfifo`. Hosts lacking
these primitives skip that class, including positive archive/producer checks and
symlink/FIFO/mutation cases. Independent JSON/CLI-help, unsupported-platform and
HTTP transport tests remain portable and run there. Those tests or skips do not
qualify Windows producer execution; no Windows test-run claim is made here.

## Trust anchors and the unresolved full-ref boundary

The future protected caller must supply an **independently reviewed external
policy**, with the exact immutable producer head and workflow bytes, numeric
repository/workflow IDs, run/artifact IDs, artifact role and archive digest.
It must load this policy and this verifier from reviewed protected code, outside
the producer/candidate's control. A policy JSON file does not authenticate its
own origin. Passing a producer receipt or candidate-owned file as
`--trusted-policy` violates the trust model; a `reviewed: true` field or arbitrary
claimed workflow checksum is no substitute for that external review. The helper
does not decide which commits or policies an organization should trust.

REST run metadata exposes `head_branch`, a **short** branch name. It does not
expose an authenticated full `GITHUB_REF`. A `workflow_dispatch` can target a
branch or tag. This helper therefore never describes the full ref as API-proven.
Its `full_ref_assurance` is explicitly
`EXTERNALLY_REVIEWED_IMMUTABLE_PRODUCER_CONTRACT`.

The external anchor must cover a real producer implementation that fails before
**every accepted artifact upload** unless `GITHUB_REF == refs/heads/master`.
Review includes all relevant steps, conditions, reusable workflows, local scripts,
checkout source choices and upload paths. Locally referenced code must be bound
to the approved tree; externally referenced actions/workflows must be pinned and
included in that review. An upload before the guard, a guard whose failure is
ignored, or an unreviewed upload path invalidates the contract. The policy's
`kind` string identifies this required contract; it does not prove its semantics.
The helper independently fetches the workflow at the exact approved head and
checks its SHA-256 and Git blob ID. It does not parse YAML to certify control flow
or audit the rest of the approved tree. Until such a reviewed contract is adopted,
the full-ref gate remains unresolved and no live positive claim is justified.

GitHub's current workflow-ID endpoint establishes current identity/path/state;
it is not a historical workflow source proof. Exact historical workflow bytes
are fetched separately using the approved 40-hex commit, never a branch ref.
The system Python/TLS installation, local filesystem/object store, external
policy delivery, and accurate local clock are trusted caller infrastructure.

## Exact inputs and checks

The trusted policy is strict UTF-8 JSON with exactly these fields:

| Field | Required value or meaning |
| --- | --- |
| `schema_version` | Integer `1`, never boolean/float |
| `repository` | `xybrid-ai/xybrid` |
| `repository_id`, `workflow_id` | Independently approved positive numeric GitHub IDs |
| `workflow_path` | `.github/workflows/trusted-zzz-sdk.yml` |
| `approved_head_sha` | Exact approved lowercase 40-hex master commit |
| `full_ref`, `event` | `refs/heads/master`, `workflow_dispatch` |
| `producer_contract` | Exactly `kind` = `ref-master-before-artifact-upload-v1` and `workflow_sha256` = approved lowercase 64-hex workflow-byte hash |
| `run_id`, `run_attempt` | Independently approved run ID; attempt must be integer `1` |
| `artifact_id`, `artifact_role` | Independently approved artifact ID and role matching `[a-z][a-z0-9-]{0,31}` |
| `artifact_sha256` | Independently approved lowercase 64-hex uploaded archive digest |

Untrusted candidate inputs are the run/artifact selectors and local archive.
Selectors must equal the trusted policy. The only accepted artifact name is
`zzz-sdk-<role>-<run_id>-attempt-1`. The name is an allowlist convention, not an
authentication primitive. The helper independently checks:

1. The fixed repository name/numeric ID; current workflow numeric ID/path/state;
   and exact historical workflow contents/hash at `approved_head_sha`.
2. Both the current run and attempt-1 endpoint: numeric run/workflow IDs, approved
   head, short branch `master`, dispatch event, completed status, successful
   conclusion, workflow path, and both repository identities. Fork heads fail.
3. The exact artifact-ID endpoint and the **complete** run artifact list: unique
   exact expected name, exact ID membership, run/repository/head identity,
   nonexpired state, future UTC expiry, bounded size, and exact `sha256:` digest.
   Missing digests, duplicates, incomplete pagination or endpoint disagreement
   fail. Unrelated artifacts are never downloaded or followed.
4. The exact local archive file: no final symlink, a positive bounded regular
   file with the API size, streamed SHA-256 through one open file descriptor, and
   unchanged device/inode/size/mtime/ctime before and after reading. FIFO/device/
   directory inputs fail. The archive is not extracted or executed.
5. The artifact-ID endpoint and current run again after hashing. Expiry/deletion,
   changed metadata, failure or a new rerun observed then fails closed.

Artifact REST metadata has **no `run_attempt` field**. A run ID survives reruns,
and attempt-scoped run metadata does not prove an artifact's upload attempt.
This initial helper deliberately rejects attempts above 1. An attempt in a name
or a timestamp cannot repair that gap. Recovery must use a fresh dispatch/run,
with a newly approved external policy. Final current-run rereading detects a
rerun that begins during verification; the REST calls are not an atomic
transaction, so a receiving consumer must repeat verification at use time.

The GitHub artifact digest describes the uploaded archive stream, not a tarball
inside it, extracted SDK library, member tree or source-binding JSON. Digest
agreement is a GitHub-service/trusted-uploader assertion and byte check, not an
attestation that the claimed build ran or that its code is safe.

## Transport and resource limits

The only origin is `https://api.github.com`; there is no API-base override or
offline metadata input mode. `http.client.HTTPSConnection` uses system TLS
validation directly and does not read proxy settings, `.netrc`, credential
helpers or GitHub tokens. Only fixed GET paths constructed from validated local
anchors are requested. Response URLs and redirect locations are never followed;
non-200 responses, rate limits, missing/private resources and malformed JSON
fail closed. No archive downloads are performed by this helper.

Requests have a 5-second socket inactivity timeout, a 10-second caller deadline
covering DNS/headers/body, a 120-second total budget and at most 19 requests.
The request worker is a daemon: a deadline failure poisons that transport,
discards all later results and exits the CLI without another request. The worker
owns connection closure; cross-thread buffered closure could itself block.
An OS DNS/network operation may outlive the failed call in that daemon until it
returns or the CLI exits; this is not a reusable background transport or a
cancellation guarantee for the remote server. Metadata reads request HTTP cache
revalidation, which does not make the observations atomic or linearizable.

Other bounds: 16 KiB policy, 1 MiB per API response, 64 nested JSON containers,
128 KiB workflow bytes, 10 pages of 100 artifact entries, 512 MiB archive and
64 KiB stream chunks. JSON requires UTF-8 and rejects duplicate keys, all floating
or nonfinite numbers, invalid numeric types and excessive depth before parsing.
All stored API data remains bounded by those limits. A valid future release
outside these deliberately conservative bounds needs a separately reviewed
change, not a silent fallback.

Before any policy/archive/API read, the CLI refuses the presence of
`ZZZ_GITHUB_TOKEN`, `ZZZ_RELEASE_REPOSITORY`, `XYBRID_ZZZ_USE_GH_AUTH`, `GH_TOKEN`,
`GITHUB_TOKEN`, `GH_ENTERPRISE_TOKEN`, `GITHUB_ENTERPRISE_TOKEN`, `SSL_CERT_FILE`
or `SSL_CERT_DIR`. It checks names only and never reads or prints their values.
This is defense in depth; it does not authorize running in a secret-bearing
process. Errors contain no response bodies, local paths or credentials.

## Future use and remaining gates

Once an actual producer contract and protected caller policy have been reviewed
and adopted, the proposed invocation is:

```sh
python3 tools/scripts/zzz_producer_verify.py \
  --trusted-policy /protected/caller/policy.json \
  --run-id 123 --artifact-id 456 --archive /isolated/download/artifact.zip
```

Without all required API/archive/anchor checks the CLI returns nonzero and no
success JSON. Success is an unsigned, point-in-time record of API facts and the
exact archive digest under the external policy. `source_binding` is
`NOT_VERIFIED_BY_THIS_HELPER` and
`activation` is `NOT_AUTHORIZED`.

The future receiver must safely select members from those **same verified
archive bytes**, validate the source-binding schema, recompute the source-policy
binding against independent trusted context, and reconcile candidate/source/
manifest/verifier digests and any later checksum commit. No separate loose JSON
file or self-declared receipt becomes authenticated by this helper. The receiver
must reverify the archive at use time; a mutable pathname or saved success JSON
is not a capability to read different bytes. Extraction safety, member allowlists,
source reconciliation, private build/cache isolation, terms/notices, platform
qualification and final artifact publication remain separate gates. Raw engine
archives/libraries must not enter public Actions artifacts or remote caches.

## Offline validation

```sh
python3 -m unittest discover -s tools/scripts/tests -p test_zzz_producer_verify.py -v
```

The tests inject API transports inside Python, never through a production CLI
metadata flag. They use tiny synthetic ZIPs and workflow bytes, and do not use
GitHub credentials, fetch real payloads, copy SDK trees, run native builds or
prove live producer adoption. Transport fixtures check origin/headers, redirect
rejection, response/count/deadline bounds and no proxy credential forwarding.

Primary references: [GitHub artifact REST schema](https://docs.github.com/en/rest/actions/artifacts),
[workflow runs and attempts](https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run-attempt),
[dispatch branch/tag ref](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event),
[run ID and attempt semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/variables),
[upload-artifact stream finalization](https://github.com/actions/toolkit/blob/main/packages/artifact/src/internal/upload/upload-artifact.ts).
