# Trusted Apple SDK candidate producer

`.github/workflows/trusted-zzz-sdk.yml` is a manual, protected-master producer
for the existing Apple device/ARM-simulator SDK, using the actual opt-in
notice-bearing Bazel target. The current six engine pins use published `v0.1.0-rc.9` schema-5 archives.
All six passed actual inspection, private staging and staged-byte verification.
This enables producer preflight to accept their notice contract; default
features, the public release workflow and publication path remain unchanged.

This implementation has offline fixture/static evidence. No hosted dispatch,
private producer download, native SDK build or installed SDK qualification was
performed for this change. A produced artifact remains a candidate until independently
approved API/archive/member/source reconciliation and installed-package tests.

## Runtime identity and immutable tooling

The only trigger is `workflow_dispatch`. The job condition and trusted runtime
checks require `xybrid-ai/xybrid`, `refs/heads/master`, dispatch event and attempt
1. The full runtime `GITHUB_SHA` must equal `GITHUB_WORKFLOW_SHA`, with the exact
workflow-ref path. Candidate SHA/version/date are data inputs; none can choose
the trusted master. Invalid job contexts are skipped and produce no artifact;
invalid runtime contexts fail before candidate reads or uploads.

Both external checkout actions are pinned to the repository's existing v4
commit, use the runtime master SHA/full history, disable submodules and set
`persist-credentials: false`. `trusted-tooling` supplies all scripts; separate
`source-objects` supplies candidate Git data. `python3 -I` and exact sibling paths
prevent candidate/sys.path imports. The trusted checkout HEAD and manifest
bytes must match immutable master objects. The unchanged actual
[source checker](zzz_source_policy.md) constrains candidate C to its existing
exact metadata policy; source, verifier, manifest, workflow, code/configuration
and gitlinks cannot change in C.

## Ordered phases

1. **Prepare without credentials.** Refuse producer-verifier credential/TLS
   override names plus remote-cache/cloud credential names. Check the full
   schema-5 notice map and strict packaged `select_pin` result for the closed
   six-target staging set below. Fetch C's raw Git objects only from the fixed
   public SDK origin; recompute the actual source-policy binding with no
   approved-checksum substitution. Initialize fresh private Git stores for the
   two fixed public submodule origins, fetch each exact gitlink commit, and
   materialize C plus both submodules from raw blobs. No `git checkout`, filter,
   hook, candidate script or dependency installer runs. Verify actual source
   contents/modes and both complete submodule contents before returning.
2. **Minimal mirror broker.** Only this step references `ZZZ_GITHUB_TOKEN` and
   `ZZZ_RELEASE_REPOSITORY` from the existing `zzz-engine` environment. Repeat
   runtime context and strict schema-5 pins before constructing a token-bearing
   child. Each child executes only the trusted packaged `zzz_pull.py` with
   `-I`, explicit private cache/stage roots, fixed target and a small environment
   containing the dedicated mirror variables. Its HOME/GH config are private;
   no inherited GitHub/cache credentials, candidate code or installer enters
   that child. No local gh-auth opt-in is used. The workflow does not configure
   or certify the environment's independently reviewed master-only service rule.
3. **Fresh stage and build without credentials.** Freshly recompute source
   binding and actual source/submodule contents. Verify every downloaded staged
   archive and payload, then restage its pinned archive into a new private
   target directory and verify it again. Only after this token-free boundary
   can the existing public `cmake`, `pkg-config`, `xz` Homebrew setup run if
   tools are missing. Bazelisk must already be installed. Installers and builds
   receive a fixed environment, never mirror/publish/signing credentials.
4. **Final package and use checks.** Invoke the actual
   `//bindings/apple:XybridFFI_zzz` target with explicit `--//:tts_zzz=true`.
   Resolve its single `XybridFFI_zzz.xcframework.zip` through configured `cquery`
   and actual private `bazel info execution_root`, copy its held regular bytes
   into the exact upload directory, and verify with trusted
   `bindings/apple/pack_zzz_notices.py verify --archive PATH`. Recompute source
   policy and actual source/submodule/staged/final-package facts after building.
5. **Sole upload.** A separate success-only step immediately before the pinned
   upload action repeats those complete checks and runtime master/attempt-1
   context. Any failure prevents upload. There are no failure/`always()` uploads,
   `continue-on-error` paths, aggregate uploads or broad artifact downloads.

Bindings are freshly recomputed from independent runtime context and immutable
Git objects at each use; a saved binding JSON cannot authorize changed bytes.
Prepared and final binding reads require bounded read-only single-link regular
files, and the final binding must exactly equal freshly rendered source facts.
No permanent policy, repository/workflow numeric ID or artifact anchor is
fabricated here. The runtime run ID is actual GitHub context.

## Closed private staging set

| Repository | Target |
| --- | --- |
| `zzz_linux` | `x86_64-unknown-linux-gnu` |
| `zzz_macos` | `aarch64-apple-darwin` |
| `zzz_android` | `aarch64-linux-android` |
| `zzz_ios` | `aarch64-apple-ios` |
| `zzz_ios_sim` | `aarch64-apple-ios-sim` |
| `zzz_linux_arm64` | `aarch64-unknown-linux-gnu` |

Preflight binds this fixed map to the actual repository declarations in reviewed
`MODULE.bazel`; offline tests check the exact same six. All six require genuine
schema-5 pins and the immutable 14-file notice map through the existing packaged
verifier. This conservatively supports loading of unselected `select`-arm
repositories without relaxing the existing strict repository checks. Extra
archives remain private verified data: none executes or appears in the upload.
**Only the two actual iOS slices are linked and packaged by this Apple target.**
The extra stage does not build or qualify Android/Linux/macOS SDK packages.

## Source and build isolation

Git data operations use system Git with no inherited credentials, global/system
configuration, hooks, replacement objects, lazy fetch or prompts. Fixed-origin
fetches validate full SHAs, disable recursion/FETCH_HEAD and automatic
maintenance, and never accept URLs from `.gitmodules` or candidate metadata.
Raw materialization writes each blob exclusively with its tracked executable
mode; source symlinks cannot escape the new tree. Both fixed submodule paths
must be real directories with exact HEADs and complete raw-tree byte/mode
agreement; ignored/untracked files, missing files, extra/nested submodules,
links/devices substituted for regular files and source changes fail. Only
freshly controlled top-level `.git` metadata is excluded from source contents.
The source helper's entry/blob/byte limits and depth-bounded enumeration apply.

Bazel uses the reviewed iOS platform, iOS16 floor, local foreign-tool toolchains
and fat Rust LTO options underlying the existing Apple entrypoint, stated
directly. System/home/workspace RC loading is disabled; no mutable RC/import is
read. The existing common public NDK repository metadata is copied explicitly
for module resolution; no Android target is built. Remote execution/cache
reads/writes, BES and shared disk cache are disabled.
Bazel output/user/repository cache, Bazelisk, HOME, XDG/temp and Cargo roots are
fresh private directories. sccache and inherited Bazel/Cargo wrapper settings
are absent. Convenience symlinks and lockfile updates are disabled to preserve
the exact source tree. The configured query uses the same isolated build
options, as supported by [Bazel's configured-query contract](https://bazel.build/query/cquery).

If the existing public `MACOS_SDK_MIRROR` variable is configured, the clean build
phase writes the same pinned-SDK rewrite form as the existing cache setup into
its own downloader config. Only a host/path without credentials or whitespace
is accepted; no shared cache action or mutable workspace RC is adopted. Bazel
still checks the upstream pinned SDK checksum. Cold dependency/mirror/Xcode
availability and real Bazel analysis remain hosted qualification gates.
Subprocess output is private, failures expose fixed safe messages, and the
workflow/process deadlines bound failures. No compiler/candidate/Git stderr or
private source paths are emitted as public failure detail.

## Exact upload and later release handoff

```text
packages/xybrid-apple-v<VERSION>.xcframework.zip
source-binding.json
```

The upload name is exactly `zzz-sdk-apple-<runtime run ID>-attempt-1`.
Descriptor-relative bounded inventory and streamed hashes reject extra members,
links, special files, oversized/changing inputs and renamed raw ar containers.
Binding is at most 16 KiB; total members are at most 500 MiB. Upload compression
is explicitly disabled, leaving over 12 MiB below the receiver's 512 MiB raw
Actions-wrapper limit. Receivers must still verify the actual service ZIP bytes.
No raw engine archive, static library, `.rlib`, private receipt or stage directory
is selected for upload.

The real opt-in packager preserves the nine existing device/ARM-simulator files
and includes all 14 canonical notices at
`XybridFFI.xcframework/Notices/zzz/<canonical-path>`, for exactly 23 regular
members. Both engine inventories and final notice bytes must match the reviewed
contract. This establishes package structure/notice inclusion, not installed SDK
behavior, deployment-floor execution, native symbol correctness or speech.

Native bytes derive from candidate C. A later checksum commit D remains separate:
independently prove C→D changes only the permitted Swift/RN checksum literals,
and recompute the binding using that approved checksum. This workflow does not
modify `release-prep.yml`, adopt candidates for publication or silently call D
the source built. Existing [API verifier](zzz_producer_verify.md) and
[receiver](zzz_candidate_extract.md) require externally approved real
IDs/digests, fresh API/archive/member verification and source reconciliation.
Installed Swift/model, streaming/cancellation and deployment-floor tests remain
required before release adoption.

The OS/Python/Git/toolchain, reviewed source/dependency code, protected caller,
fresh Git metadata and local filesystem must be trusted and stable. Repeated
stat/hash checks are observations, not an atomic filesystem/API transaction or
a sandbox against this UID/root. Output paths and saved success records remain
mutable and cannot replace fresh verification at the next boundary.

## Offline evidence

```sh
python3 -m unittest discover -s tools/scripts/tests -p test_zzz_sdk_stage.py -v
actionlint .github/workflows/trusted-zzz-sdk.yml
python3 tools/scripts/gen_workflow_index.py --check
```

The flow fixtures use real local Git materialization, actual source-policy
comparison, actual tiny synthetic pinned-archive inspection/restaging and the
actual final notice packager/verifier bodies. They stub only external mirror
subprocess transport and native Bazel/Xcode execution. Synthetic native files,
notices and hashes are test-only; no production pin or license hash is changed.
They exercise ordering, token/environment isolation, six-target staging,
source/submodule and notice corruption, final API-facing member allowlists and
failures. No real API, private download, native code, workflow dispatch or
installed package is executed by these tests.
