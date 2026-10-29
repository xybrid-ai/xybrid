# Draft trusted SDK source policy

`zzz_source_policy.py` is an offline source checker for a future trusted SDK
staging workflow. This patch does not add a workflow, download or stage an
engine, build an SDK, change native pins or enable `tts-zzz` in public artifacts.
It does not establish redistribution rights or supply terms, notices, schema 5
receipts, Android qualification, CPU qualification or final SDK qualification.

The checker runs from reviewed master code in a credential-free step. It reads
immutable local Git objects from an explicitly supplied repository and permits
only a narrow rendering of version/checksum metadata over a reviewed master
commit. It never executes or imports a candidate blob, runs a release branch's
version helper, checks out files, or invokes a build. A release branch may put
executable code in `Package.swift`, Gradle, MDX, generated Python or JSON package
hooks; merely allowing those paths would be unsafe. Every byte outside the
specified literal replacements must remain from the trusted baseline.

## Inputs and output

Python 3.11+ and system Git are required. The CLI accepts full lowercase 40-hex
commit IDs only. Both commits must already exist locally; the candidate must
descend from the supplied trusted master. Tags, branch names, abbreviated IDs,
Git expressions, automatic fetches and replacement objects are excluded.

```sh
python3 -I tools/scripts/zzz_source_policy.py \
  --repo /path/to/local/repository \
  --trusted-master <full-reviewed-master-commit> \
  --candidate <full-candidate-commit> \
  --version 0.14.0 --release-date 2026-10-22 \
  --producer-context /path/to/caller-context.json \
  --xcframework-sha256 <independently-approved-sdk-zip-sha256> \
  > /path/to/source-binding.json
```

The checksum flag is optional. When present, it allows exactly the existing
Swift and React Native checksum values to change to that SHA-256; it does not
verify an artifact or make that caller-supplied digest trustworthy. When absent,
those checksum literals must remain unchanged. A subsequent invocation with
`--compare-binding /path/to/source-binding.json` recomputes all source facts and
requires equality of the complete binding, including context and metadata.
Unknown or missing fields fail comparison. The checker writes only the binding
JSON to stdout and bounded, content-free errors to stderr.

The context file has exactly these fields (example commit/run values are
placeholders):

```json
{
  "repository": "xybrid-ai/xybrid",
  "workflow_path": ".github/workflows/trusted-zzz-sdk.yml",
  "workflow_commit": "0000000000000000000000000000000000000000",
  "ref": "refs/heads/master",
  "event": "workflow_dispatch",
  "run_id": 123,
  "run_attempt": 1
}
```

The workflow path is a proposed fixed identity, not an existing activated gate.
`workflow_commit` must equal the explicit trusted master input. The positive
integer run ID/attempt and master dispatch fields receive shape/equality checks
only. The output always says
`"producer_authentication":"NOT_AUTHENTICATED_BY_THIS_HELPER"`.

The version-1 binding records the two commit IDs, candidate Git tree ID, a
streamed SHA-256 tree digest, SHA-256 of the exact engine manifest and packaged
verifier, producer context, and approved release version/date/checksum inputs.
The tree digest hashes sorted UTF-8 paths with a four-byte big-endian path
length, mode, object type and content SHA-256 under the
`xybrid-zzz-source-tree-v1` domain. An unchanged gitlink hashes its pinned object
ID under the `gitlink` domain; it does not claim to hash submodule contents.
Symlink contents are hashed as raw blobs, never followed. Candidate paths,
modes, types, symlinks and gitlink IDs must match the baseline exactly, except
regular metadata blobs can contain the approved changes.

## Deliberately exact metadata policy

- Cargo workspace/member package versions and version-constrained internal path
  dependencies may change to the requested version. Internal destinations and
  package names must match the trusted workspace. Only source-less workspace
  package versions may change in `Cargo.lock`; external versions, dependency
  records, source URLs and checksums stay byte/structure constrained.
- SDK package versions, the generated Python `PACKAGE_VERSION`, the web version
  constant, Flutter's plugin lock entry, Kotlin/RN version literals, Apple
  version strings and the enumerated installation examples receive fixed
  replacements. Other code, dependency features, hooks and package scripts
  cannot change. Swift's remote/local mode and package code remain exact.
- The root changelog may only insert a fixed new-version/date heading after
  `[Unreleased]`; the Flutter changelog may only insert its fixed version
  heading. New prose, rearrangement, whitespace rewrites and arbitrary Markdown
  or MDX changes are rejected. For example, the manually edited 0.13.1 root
  changelog fails comparison with the earlier reviewed 0.13.0 baseline; it is
  not normalized into an allowed change. Once those notes land in reviewed
  master, their existing bytes become trusted baseline content. This draft is
  not a gate retrofitted to the 0.13.1 release.
- The engine manifest, packaged verifier and all other tracked code/configuration
  must have identical object IDs. Added, removed or renamed paths, executable
  metadata, changed symlinks and changed gitlinks fail closed.

The baseline itself must be reviewed and trusted; this policy constrains changes
from it rather than proving its code is safe. No arbitrary file allowlist or
textual Git diff is accepted as evidence. TOML/JSON structural checks and exact
literal rendering are both required; duplicate JSON keys and nonfinite values
are rejected. JSON must be UTF-8 and contain at most 64 nested arrays/objects.
A quote/escape-aware scan enforces that bound before parsing, independently of
Python's platform-dependent parser recursion limit; string contents do not count
as containers. The policy is intentionally conservative: a valid release outside
these forms needs a separately reviewed policy or trusted baseline change.

## Resource and credential boundary

Git runs with `--no-replace-objects`, network protocols disabled, no lazy fetch,
no inherited Git/credential environment, no global/system configuration, hooks
or filesystem monitor. Only local object commands run; checkout filters and diff
drivers are never used. The repository/object store and system Git are trusted
local inputs; a malicious `.git/config`, modified Git binary or object store is
outside this helper's isolation boundary. The helper does not certify ignored
files or the mutable worktree a later build might use.

It rejects the presence of `ZZZ_GITHUB_TOKEN`, `ZZZ_RELEASE_REPOSITORY` or
`XYBRID_ZZZ_USE_GH_AUTH`, even with an empty value, before reading inputs or
starting Git. It checks names only and never reads or logs those values. Other
environment variables are not inherited by Git. This is defense in depth, not
permission to run the checker in a secret-bearing job: the future caller must
keep credentials out of the entire source-validation process and use a separate
minimal download step afterward.

Bounds are explicit: 20,000 tree entries, 4,096 bytes per path, 8 MiB of tree
listing output, 2 MiB per metadata blob, 128 MiB per raw blob and 512 MiB total
unique blob content. Content hashing streams 64 KiB chunks. The initial reviewed
0.13.0 baseline (`aad08da1`) contains an 83,134,752-byte ORT archive and
272,602,448 unique blob bytes, so lower 64/256 MiB limits would reject legitimate
source. Context and comparison JSON are limited to 8/16 KiB. No candidate path
is used to open a working-tree file. Gitlinks do not trigger network or submodule
access.

## Required future workflow and API gate

A local context JSON file, receipt, branch name, artifact filename or successful
checker exit is **not producer authentication**. The caller must not use a
receipt's self-declared fields as both the claim and the expected trusted input.
Before any credentials, download, public upload or release decision, a future
protected workflow must independently establish all of the following:

1. The reviewed master commit, protected workflow path/ID and workflow source
   commit are approved. Execute the checker and staging code from that master
   commit; do not read or execute release-branch scripts under secrets.
2. GitHub API data independently confirms the fixed repository, producer run ID,
   attempt, workflow identity, workflow/head SHA, ref, event and successful
   conclusion. Reconcile the exact separately bound candidate source and its
   locally recomputed manifest/verifier/tree digests with that authenticated run.
3. GitHub API data confirms an exact artifact ID belongs to that exact producer
   run/attempt and has the expected digest. Verify the actual downloaded bytes;
   archive wrapper digest and inner SDK file digest are distinct. Names and
   self-asserted receipt hashes alone cannot establish membership or trust.
4. The build uses a fresh exact candidate tree with the approved source binding,
   rather than an arbitrary local checkout. Stage private engine bytes before
   Bazel analysis using the separately approved receipt/schema/rights policy.
   Keep raw engine archives, libraries and derived private outputs out of public
   artifacts and remote execution/cache/BES uploads; preserve a private output
   root and control local/global Bazel RC imports and Cargo target caches.
5. Only the intended final SDK candidates and their approved terms/notices cross
   to public preparation. The receiver repeats API membership/digest checks and
   recomputes the source binding against independently trusted context. A
   post-build checksum commit is a new immutable candidate; source/producer
   reconciliation must explicitly account for it, not silently normalize code.

These workflow/API, build-isolation and distribution gates are not implemented
here. This helper alone is insufficient to activate public SDK staging.

## Validation

```sh
python3 -m unittest discover -s tools/scripts/tests -p test_zzz_source_policy.py -v
```

The suite edits local temporary Git fixtures, exercises the actual repository's
metadata layout and generated Python constant, and streams the current immutable
source tree including its large tracked archives. Negative cases cover executable
metadata injection, external dependency/checksum changes, source/verifier/manifest
changes, mode/symlink/gitlink changes, replacement objects, filters, credential
presence, forged context/bindings and input/output bounds. No candidate scripts,
native builds, engine downloads or GitHub writes run.
