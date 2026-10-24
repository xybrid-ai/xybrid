# Draft bounded final-SDK candidate receiver

`zzz_candidate_extract.py` adds only an extraction boundary after the actual
[producer verifier](zzz_producer_verify.md). No producer workflow, policy,
artifact ID, native pin, release, platform qualification or SDK feature is
adopted. No payload is executed. The existing producer and source-policy
helpers are unchanged.

The helper receives an already downloaded Actions ZIP. It uses the real
producer verifier before reading member data and repeats that verifier after
extraction, before successful return. CLI verification uses the existing fixed
anonymous GitHub HTTPS transport, with no offline-metadata flag, credentials,
download URL or API-base override. Offline tests inject transports in Python.

```sh
python3 -I tools/scripts/zzz_candidate_extract.py \
  --trusted-producer-policy /protected/caller/producer-policy.json \
  --trusted-extraction-contract /protected/caller/extraction-contract.json \
  --run-id 123 --artifact-id 456 \
  --archive /isolated/download/candidate.zip \
  --destination /isolated/receiver/new-candidate
```

The producer-policy shape and trust obligations remain exactly those of the
producer verifier. Its full-ref boundary still requires independent review of
the immutable producer implementation; API metadata does not prove the full
ref, build execution or candidate source. No live eligible producer is supplied
here. A saved success JSON, candidate receipt, archive filename or policy file
cannot authenticate its own authority.

## Independent extraction contract

The protected caller supplies this contract from reviewed code or a separately
approved trusted channel, outside the candidate and ZIP. It is strict UTF-8
JSON with exactly these fields:

| Field | Required value |
| --- | --- |
| `schema_version` | Integer `1` |
| `artifact_role` | One of `apple`, `android`, `python`, `web`, `react-native`, `unity`, `flutter`, `rust`; must equal the independently approved producer-policy role |
| `trusted_master_commit` | Full lowercase 40-hex commit, equal to the producer policy's approved head |
| `candidate_commit` | Independently approved full lowercase 40-hex candidate commit |
| `source_binding_member` | Exactly `source-binding.json` |
| `members` | Exact map from relative member path to an object with exactly `kind`, `size` and `sha256` |

Each size is a positive integer, never a boolean or float. Each digest is a
lowercase 64-hex SHA-256. The binding member has kind `source-binding` and is
limited to 16 KiB. Every other member has kind `sdk-package`. SDK basenames must
start with `xybrid` (case-insensitive), with these role-specific suffixes:

| Role | Package suffix |
| --- | --- |
| Apple, Unity | `.zip` |
| Android | `.aar` |
| Python | `.whl` |
| Web, React Native | `.tgz` |
| Flutter | `.tar.gz` |
| Rust | `.crate` |

These are deliberately narrow draft receiver conventions, not an adopted
production upload contract. ZIP-based SDK containers must start with a ZIP
local-header signature; gzip-based packages must start with gzip's signature.
Raw engine roles, engine archive names, static libraries, rlibs, standalone
native libraries, receipt JSON and private staging paths are not SDK package
members. Components named `private`, `staging`, `natives`, `target` or `zig-out`
are excluded; paths cannot start with a dot. Renaming an ar archive to an AAR
does not bypass the container-signature check.

**Role review and inner-package qualification remain necessary.** Filename and
container checks do not establish what an opaque SDK package contains. An
independent allowlist must identify the actual intended final SDK packages,
never relabel private engine archives as SDKs. The receiver does not recursively
extract, inspect, execute or certify the inner SDK containers, including their
terms/notices, native contents, privacy or installability.

The binding member is also opaque: its exact bytes are authenticated against
the externally supplied hash/size, and cannot choose a role, candidate ID or
member allowlist. The output labels source policy
`NOT_RECOMPUTED_BY_THIS_HELPER`. It does **not** compare a binding's self-claims
with Git, establish candidate ancestry, or reconcile manifest/verifier/tree
digests or later checksum commits. The protected consumer must freshly run
[source-policy reconciliation](zzz_source_policy.md) against independently
trusted context before using these files for publication or activation.

## Same ZIP bytes and filesystem boundary

Before any input read, the receiver checks the producer verifier's complete
credential/TLS-override refusal list and POSIX prerequisites. It additionally
requires real `O_DIRECTORY`, directory-descriptor operations and Python's
symlink-resistant `rmtree`. There is no Windows fallback or host qualification.

The original ZIP is opened once with `O_NOFOLLOW | O_NONBLOCK`. It must be a
positive bounded regular file with one hard link. The helper copies it through
that held descriptor into an exclusively created private directory, checking
the original device/inode/mode/link-count/size/mtime/ctime and byte count before
and after copying. The private snapshot is mode `0400`, and its directory is
mode `0700` under the destination's trusted parent.

A read-only snapshot descriptor stays open across both real producer checks
and extraction. Before/after those checks and after extraction, the receiver
rehashes that held file and checks its named snapshot's exact identity, mode,
owner and link count. The actual producer verifier independently opens/hashes
that snapshot path on each invocation. `ZipFile` reads the held snapshot
descriptor, never the original mutable path. Replacing the original download
after the copy cannot redirect extraction. Replacing the named private snapshot
or changing its bytes fails. No producer-helper refactor or weaker archive
verification path is introduced.

The compiler/OS/Python installation, independently delivered contracts and
**local filesystem must be trusted and stable**. Read-only permissions protect
against other users; they are not immutability against this UID/root. Stat/hash
checks are repeated observations, not an atomic filesystem or GitHub
transaction. A concurrent owner can mutate files between observations. Subsequent
consumers must not treat a mutable extracted pathname or saved success record
as authorization; repeat producer/archive/member checks and source reconciliation
at the next use boundary.

The destination must be a new absolute path under existing nonlinked directory
components. It is reserved with exclusive `mkdir`; existing files, directories,
symlinks and special destinations are never replaced. All output files are
created exclusively using held directory descriptors, become mode `0400`, and
are never executable. Created directories are `0700`. Under the trusted stable
filesystem assumption, any extraction/hash/API failure removes the entire owned
output and private snapshot. Cleanup retains both directory descriptors and
checks each named directory's identity before recursive deletion. A failed
descriptor acquisition after `mkdir` never authorizes recursive deletion.
Cleanup never removes a scratch directory that this invocation did not create.
If an owner renames/replaces either directory during processing, identity checks
fail and the helper refuses to delete an unrelated replacement; it reports
cleanup/rollback failure, emits no success and may leave moved or unacquired
owned directories for trusted cleanup. The check followed by removal is still
subject to the stable-filesystem assumption. OS/permission failures can likewise
make cleanup incomplete; they are errors, never successful reception.

After the second producer/API verification, immediately before success, the
receiver walks the output through held directory descriptors. It rejects any
missing or extra file/directory and checks directories' `0700` mode/owner and
every file's single-link regular type, `0400` mode, owner, exact size and SHA-256.
Directory enumeration stops on the first beyond-allowlist entry. The root and
member identities are checked again around this bounded validation. This detects
changes during the potentially long API recheck; it does not make the returned
output immutable or authorize later use without fresh verification.

## ZIP and resource bounds

The 512 MiB original wrapper limit is inherited from producer verification.
The extraction contract is at most 128 KiB. Policy/contract parsing uses the
producer's duplicate-key, integer-only, depth-64 strict JSON parser.

The ZIP end record and central-directory bytes are checked before constructing
`ZipFile`: at most 256 members, 1 MiB central metadata, 512 MiB total uncompressed
bytes, 512 bytes per ASCII path, eight path components and 100:1 declared
per-member compression ratio. Every member must match its independently pinned
size/hash. Streaming member reads are at most 64 KiB, and stop at approved size
plus one. ZIP CRC checks also apply. Each complete producer invocation retains
its own HTTP request/time bounds; the default receiver performs two such
invocations. File I/O has byte bounds, not a guarantee against a stalled trusted
filesystem.

Only stored and deflated ZIP32 data, with explicit Unix regular-file metadata,
are accepted. Standard streaming data descriptors are supported. Duplicate,
case-alias, prefix-overlapping, absolute, traversal, backslash, drive and NUL
paths fail. Directory, symlink, device, socket, FIFO, unspecified/DOS type and
hardlink-related metadata fail. All extra fields and member/archive comments
are rejected, including ZIP64 and Unix link extensions. Encryption, unsupported
compression/flags, multidisk forms, ZIP64, self-extracting prefixes, trailing
data, local/central disagreement, gaps and overlapping member data fail closed.
No `extract()`/`extractall()` operation is used. A legitimate future upload
outside these conservative forms needs separate review, rather than a fallback.

Success is an unsigned point-in-time result, with `activation: NOT_AUTHORIZED`.
There is no workflow adoption, public artifact publication, remote cache access,
native build or platform/installed-package proof in this helper.

## Offline validation

```sh
python3 -m unittest discover -s tools/scripts/tests -p test_zzz_candidate_extract.py -v
python3 -m unittest discover -s tools/scripts/tests -p test_zzz_producer_verify.py -v
```

Synthetic fixtures exercise the actual producer checks with injected API
transports, safe stored/streamed extraction, allowlist/metadata/container limits,
CRC/hash failures, input/snapshot/output mutation, API recheck failure,
directory replacement, no-overwrite races, rollback and credential/CLI privacy.
They run no payload code, real API
requests, native builds or downloads, and establish no adopted producer or
final SDK qualification.
