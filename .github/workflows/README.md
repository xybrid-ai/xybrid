# GitHub Actions workflows

GitHub reads only top-level files in this directory. `ci.yml` and `bazel.yml`
validate pull requests; everything else is release, artifact, security,
community or manual automation.
This page is the map: what each workflow is for, when it runs, what it can
write, and which ones depend on each other.

## Conventions

[`tools/scripts/gen_workflow_index.py`](../../tools/scripts/gen_workflow_index.py)
`--check` (in CI job `Repo checks`) enforces the first three, and the first two
rules under [Adding CI](#adding-ci). The last two are decisions to keep.

- **Display name** is `"Group: Subject"`, quoted, with a group from the
  inventory below. Groups sort together in the Actions sidebar. A dispatch-only
  workflow ends its Subject with `(manual)`.
- **Purpose**: the line directly above `name:` is `# Purpose: <one sentence>`,
  starting `Manual:` for a dispatch-only workflow. It becomes the Purpose column.
- **Concurrency groups are literal** and start with the file stem:
  `<file-stem>-${{ github.ref }}`, the tag or issue number for event-driven
  workflows, or the bare stem when every run must queue behind the last. Never
  `${{ github.workflow }}`: the display name can then change without changing
  the group.
- **Rename files rarely.** GitHub ties run history and code-scanning
  configurations to the file path (CodeQL is
  `.github/workflows/codeql.yml:analyze`, Scorecard is
  `.github/workflows/scorecard.yml:analysis`), and `release-publish.yml` and the
  docs call `gh workflow run <file>`. Prefer changing `name:`. When a file name
  is genuinely misleading, rename it on purpose and plan for the new history and
  the new code-scanning configuration.
- **Job names are the required checks.** Branch protection on `master` requires
  `CI Success` (`ci.yml`) and
  `Bazel graph + RBE targets` (`bazel.yml`), matched by job name.
  `tools/scripts/llamacpp_update.py` waits on more job names before it cuts a
  llama.cpp release (its `REQUIRED_CHECKS`). A workflow's `name:` can change
  freely; those job names cannot. A lane file's checks are named
  `<caller job name> / <job name>`, so the caller job's `name:` in `ci.yml` is
  part of every check below it.

## Adding CI

CI used to grow one workflow per feature, each with its own triggers, path
list, setup block and runner. `ci.yml` is now the pull-request pipeline, and
new coverage extends it. Rules 1, 2 and 7 fail `Repo checks`; the rest are
how to follow them.

1. **No new workflow file** other than a lane file (rule 6). `WORKFLOW_FILES`
   in `gen_workflow_index.py` lists every file in this directory, and an
   unlisted file fails the check. A new lane file, or a file the repository
   really needs (a release, artifact, security or community workflow), is
   added there by the PR that says why.
2. **Only `ci.yml` and `bazel.yml` validate pull requests.**
   `PULL_REQUEST_FILES` names the files that may declare `pull_request` or
   `pull_request_target`: those two, the security scans, the release dry-run
   and publish, and the community bots. A lane file declares `workflow_call`
   (plus `workflow_dispatch`, plus `schedule` for a nightly), never `push` or a
   pull-request event, and a workflow must call it.
3. **A check that takes seconds and needs no toolchain is a test** under
   `tools/scripts/tests/`, not a job.
4. **A new cargo invocation is a step** in the `ci.yml` job with the same
   runner and feature family.
5. **A new backend or real-model suite is one `ci.yml` job** built from the
   shared actions and gated on its flag, one row in `LANE_PATHS` with its
   `OPTIONAL_JOBS` entry (`tools/scripts/ci_gate.py`), its line in
   `tooling-changes`' outputs, and one entry in `CI Success`'s `needs`.
   `test_ci_gate.py` fails until they agree.
6. **A new binding or platform is a lane file called from `ci.yml`.**
7. **macOS runs 5 jobs at once for the whole organization.** Add steps to an
   existing macOS job; a new one waits behind every other pull request's.
   `MACOS_JOB_BUDGET` in `gen_workflow_index.py` is the number of macOS jobs
   `ci.yml` and its lane files hold, matrix legs counted. A new one raises it
   in the same PR and says why; removing one lowers it.

## Shared setup

Jobs are built from the composite actions in [`../actions/`](../actions), not
from copied setup blocks: `setup-rust` (toolchain and cargo cache),
`install-boltffi` (the pinned boltffi CLI, cached), `fetch-model` (a pinned
test model, cached and checksummed), `free-disk-space`, `setup-vulkan-sdk`,
`bazel-repo-cache` and `empty-registry` (a loopback model registry that lists
nothing, so tests never ask the production one). A change under
`.github/actions/` selects every gated job, in `ci.yml` and in `bazel.yml`.

**Caches.** A cache saved by a pull request is visible to that pull request
alone but counts against the repository's 10 GB, and every pull request used to
save its own Rust caches, evicting master's. `setup-rust`, `install-boltffi` and
`fetch-model` restore on pull requests and save only on `master` pushes. Other
caches still save from pull requests: the release and zzz workflows' own setup
blocks (their tests pin them), `build-flutter.yml`, `build-android.yml`'s NDK,
the conformance lanes' model caches, the `web` job, `bazel-repo-cache` and
`setup-vulkan-sdk`.

Dependabot's `github-actions` entry lists each action directory, because `/`
covers only `.github/workflows/`; `tools/scripts/tests/test_ci_actions.py`
fails when one is missing.

## Preserving execution

Display-name and purpose edits must preserve workflow filenames, triggers and
path filters, job IDs and check names, matrices, steps and test commands,
dependencies, permissions, secret references and cancellation settings. The
literal concurrency keys stay stable when a display name changes.

The index check covers this page and the naming conventions. Moving a test
command between jobs or files is a change of its own, made on purpose: the
command keeps its arguments, and the registry keeps the paths that ran it.

## Inventory

<!-- BEGIN GENERATED: workflow inventory (tools/scripts/gen_workflow_index.py) -->

### CI

The pull-request pipeline (ci.yml, bazel.yml) and suites that run only on master.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`bazel.yml`](bazel.yml) | CI: Bazel | push to `master`, pull_request to `master`, workflow_dispatch | none | Bazel graph analysis and remote-execution targets; a required check that skips when no Bazel input changed. |
| [`test-choice-conformance.yml`](test-choice-conformance.yml) | CI: Choice conformance | called by `ci.yml`, schedule (`23 5 * * *`), workflow_dispatch | none | Regenerate and verify choice-scoring conformance artifacts against pinned sha256s, then score FORMS through core, the Rust SDK and the CLI; daily cold rebuild. |
| [`test-decision-conformance.yml`](test-decision-conformance.yml) | CI: Decision conformance | called by `ci.yml`, schedule (`41 5 * * *`), workflow_dispatch | none | Verify the decision corpus and run isolated preparation, finalization, matched CPU inference and SDK lifecycle checks against pinned d1 weights. |
| [`ci.yml`](ci.yml) | CI: Workspace | push to `master`, pull_request to `master`, workflow_dispatch | checks | The pull-request pipeline: repository checks, Rust format, lint, tests and feature matrix, binding drift, API contract, observability, Python and Web SDKs, the backend lanes, and the SDK and conformance lanes it calls. Gates merges. |
| [`test-zzz-android.yml`](test-zzz-android.yml) | CI: zzz Android AAR | push to `master`, workflow_dispatch | none | Build the arm64-v8a speech AAR on master with the pinned engine, gate it and its Gradle repackaging, then dlopen its x86_64 slice on an emulator. |
| [`test-zzz-engine.yml`](test-zzz-engine.yml) | CI: zzz engine | push to `master`, workflow_dispatch | none | Fetch and verify the pinned Phonon/Kitten-only engine slices and link them into Rust tests on each pinned target. |

### SDK

Per-platform SDK builds, wrapper tests and example apps.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`build-android.yml`](build-android.yml) | SDK: Android | called by `ci.yml`, workflow_dispatch | none | Build the Android .so files for every ABI, test the Kotlin wrapper, and gate dlopen and the observability scenario on an x86_64 emulator. |
| [`build-apple.yml`](build-apple.yml) | SDK: Apple | called by `ci.yml`, workflow_dispatch | none | Build and verify the Apple XCFramework, then unit-test the Swift wrapper and run the observability runtime scenario on an iOS Simulator. |
| [`build-flutter.yml`](build-flutter.yml) | SDK: Flutter | called by `ci.yml`, workflow_dispatch | none | Analyze and test the Dart wrapper, build the native libraries, run the observability runtime scenario on the Linux library, and build a consumer app against the packaged layout. |
| [`build-react-native.yml`](build-react-native.yml) | SDK: React Native | called by `ci.yml`, workflow_dispatch | none | Test and npm-pack the JS package; build the iOS and Android example apps. |
| [`trusted-zzz-sdk.yml`](trusted-zzz-sdk.yml) | SDK: Trusted zzz Apple candidate (manual) | workflow_dispatch | none | Manual: Produce a source-verified, notice-bearing Apple SDK candidate from genuine schema-5 engine pins on protected master. |
| [`unity-editor.yml`](unity-editor.yml) | SDK: Unity Editor | called by `ci.yml`, workflow_dispatch | checks | Run a real Unity Editor: EditMode tests against a Bazel-built native, an IL2CPP player smoke, then the observability runtime scenario. |
| [`web-webgpu.yml`](web-webgpu.yml) | SDK: WebGPU inference (manual) | workflow_dispatch | none | Manual: build the Web SDK and run WebGPU generation in Chromium on a self-hosted GPU runner. |

### Artifacts

Publish reusable native artifacts and their download manifests.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`build-natives.yml`](build-natives.yml) | Artifacts: Publish llama.cpp prebuilts | push to `master`, schedule (`17 6 * * 1`), workflow_dispatch | contents, packages, pull-requests | Publish prebuilt llama.cpp slices to ghcr.io and open the natives-manifest PR that lets plain cargo builds download them. |

### Release

Cut, validate, publish and announce a release.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`release-notify.yml`](release-notify.yml) | Release: Announce on Discord | release (published) | none | Announce a published release on Discord. |
| [`release-dryrun.yml`](release-dryrun.yml) | Release: Dry-run | push to `release/**`, pull_request to `master`, workflow_dispatch | none | Validate release packages before tagging: version sync, changelog, pub.dev and Kotlin/Maven dry-runs. |
| [`llamacpp-update.yml`](llamacpp-update.yml) | Release: llama.cpp stable update | push to `master`, schedule (`43 7 * * *`), workflow_dispatch | none | Daily: open a PR for each new llama.cpp stable release; once it is validated and its natives are published, push the release branch. |
| [`release-prep.yml`](release-prep.yml) | Release: Prepare | push to `release/v*`, workflow_dispatch | attestations, contents, id-token, pull-requests | Release step 1: build every artifact on a release/v* branch, create the draft release, open the release PR. |
| [`release-publish.yml`](release-publish.yml) | Release: Publish | pull_request (closed) to `master`, workflow_dispatch | contents, id-token | Release step 2: when the release PR merges, publish the release and the language packages. |
| [`verify-react-native-published.yml`](verify-react-native-published.yml) | Release: React Native published-package smoke | schedule (`0 5 * * 1`), workflow_dispatch | none | Qualify a published @xybrid/react-native release on current Expo by running FunctionGemma in Release apps on an Android emulator and an iOS Simulator; weekly against npm latest. |
| [`build-unity.yml`](build-unity.yml) | Release: Unity bundles | push of tags `v*`, workflow_dispatch | contents | On a v* tag, build the Unity native libraries for every platform and upload the bundles to the release. |

### Security

Static analysis and supply-chain checks.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`codeql.yml`](codeql.yml) | Security: CodeQL | push to `master`, pull_request to `master`, schedule (`0 4 * * 3`) | security-events | CodeQL analysis of the workflow files themselves (language: actions). |
| [`gradle-wrapper-validation.yml`](gradle-wrapper-validation.yml) | Security: Gradle wrapper | push to `master`, pull_request to `master` | none | Validate every gradle-wrapper.jar against Gradle's published checksums. |
| [`scorecard.yml`](scorecard.yml) | Security: OpenSSF Scorecard | schedule (`0 6 * * 1`), workflow_dispatch | id-token, security-events | Weekly OpenSSF Scorecard run; publishes results and uploads SARIF to code scanning. |

### Maintenance

Manual utilities, run on demand.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`test-ci.yml`](test-ci.yml) | Maintenance: Packaging tools (manual) | workflow_dispatch | none | Manual: dry-run the speech-free Kotlin packaging. |
| [`unity-activation.yml`](unity-activation.yml) | Maintenance: Unity activation (manual) | workflow_dispatch | none | Manual: emit the Unity license activation request (.alf); re-run when Editor activation starts failing. |

### Community

Contributor-facing automation.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`discord-notify.yml`](discord-notify.yml) | Community: Contributor activity | pull_request_target (opened), issues (opened) | none | Post newly opened good-first-issue and help-wanted issues, and first-time contributors' pull requests, to Discord. |
| [`star-history.yml`](star-history.yml) | Community: Star history | schedule (`17 3 * * *`), workflow_dispatch | contents | Record the daily star count and redraw the README chart on the star-history branch. |
| [`welcome.yml`](welcome.yml) | Community: Welcome contributors | pull_request_target (opened), issues (opened) | issues, pull-requests | Welcome first-time contributors on their first issue or pull request. |

<!-- END GENERATED: workflow inventory -->

Refresh the table after editing a workflow with
`python3 tools/scripts/gen_workflow_index.py`. Triggers show each event with its
branch, tag and type filters. Writes lists the permission scopes granted `write`
at workflow or job level. `id-token` is the OIDC token for trusted publishing,
not a repository write, and `star-history.yml` writes to the unprotected
`star-history` branch, never `master`.

## How they fit together

**Docs-only changes.** `ci.yml` runs on every change. Its first job classifies
the changed paths with `tools/scripts/ci_gate.py`. When every path is
documentation, the build matrix is skipped. `CI Success` still reports, and it
checks that each skipped job had a recorded reason. `Repo checks` has no path
gate and runs on every change: the workflow index, `version-sync.sh` (the docs
carry the install snippets it keeps on the current release) and every test
under `tools/scripts/tests/`, which take about two minutes and no toolchain.

**Lanes.** A lane validates one area (a backend, a binding) and runs only when
that area's paths change. The backend lanes are `ci.yml` jobs: `test-candle`,
`test-whispercpp-integration`, `test-policy-routing` and `llamacpp-validate`.
The SDK lanes are files `ci.yml` calls (`workflow_call`): `build-android.yml`,
`build-apple.yml`, `build-flutter.yml`, `build-react-native.yml` and
`unity-editor.yml`, whose checks read `Android / Build Android .so (all ABIs)`
and so on. The conformance lanes are called files too:
`test-choice-conformance.yml` (when its paths change; its Qwen GGUF job, a macOS
runner, only when the paths that job reads change) and
`test-decision-conformance.yml` (on every change that is not docs-only). Each
keeps its nightly cold rebuild as its own `schedule`: on `ci.yml` it would put
skipped checks on master's HEAD over the ones the llama.cpp release reads.
`tools/scripts/ci_gate.py` holds each lane's paths (`LANE_PATHS`, in GitHub's
`paths:` syntax), and an edit to `ci.yml`, `.github/actions/` or `ci_gate.py`
runs them all. `CI Success` requires every lane a change selects, so one green
check means each of them passed. The exception is a pull request without secrets
(a fork's, or Dependabot's): the Unity Editor lane passes there without running
the Editor, which the master push after the merge does. A lane that turns flaky
goes back to `ADVISORY_JOBS` (out of `CI Success`'s `needs`) while it is fixed.
To run lanes by hand, dispatch a lane file, or dispatch `ci.yml` with `lanes`
(comma-separated; empty runs every lane). On `master`, or a branch whose tip is
master's HEAD, `ci_gate.py` ignores `lanes` and runs every lane: a skipped lane
would put a newer, skipped check on HEAD under a name the llama.cpp release
requires to succeed. To retry one lane there, re-run the push run's failed jobs.

**Bazel.** `bazel.yml` runs on every pull request so its required job can
report; its `changes` job skips the build unless a path in `BAZEL_PATHS`
(`ci_gate.py bazel`) changed. Its push trigger lists the same paths, and
`test_ci_gate.py` keeps the two equal.

**Natives.** `build-natives.yml` compiles the llama.cpp static slices once per
target and feature set and pushes them to `ghcr.io/xybrid-ai/llama-natives`. It
runs on pushes to `master` that touch its inputs (its own file included) and
weekly. The SDK builds (`build-android.yml`, `build-apple.yml`, ...) download the
slices instead of recompiling them. Its `publish-manifest` job opens a PR that
updates `crates/llama-cpp-sys/natives-manifest.txt`, which lets a plain
`cargo build` download a slice too. Its `NDK_VERSION` must equal
`build-android.yml`'s, or every Android slice is invalidated.

**llama.cpp updates.** `llamacpp-update.yml` looks for a new llama.cpp stable
release daily and opens one update PR with `RELEASE_PAT`, so the PR triggers
`ci.yml` (its `llama.cpp stable validation` lane included) and the other
required checks. Once that PR merges and
`build-natives.yml` has published a matching natives manifest, it pushes
`release/v<version>`, which starts the release flow below. Setup and dry runs
are in [docs/development/llamacpp-updates.md](../../docs/development/llamacpp-updates.md).

**Release.** Pushing a `release/v<version>` branch starts two workflows:
`release-prep.yml` (builds every artifact, patches the checksums, creates the
draft release, opens the release PR) and `release-dryrun.yml` (version sync,
changelog, pub.dev and Kotlin dry-runs; it builds no artifacts and does not
exercise prep or publish). Merging the release PR starts `release-publish.yml`,
which publishes the release and the language packages. Two steps re-dispatch by
file name. pub.dev accepts only a dispatch from the tag, so publish re-runs
itself with `flutter_only`. `build-unity.yml` triggers on `push: tags: v*`,
which a tag created with `GITHUB_TOKEN` does not fire, so publish dispatches it.
Both need `RELEASE_PAT`; without it the job summary prints the
`gh workflow run` command. `release-notify.yml` announces on the
`release: published` event. The full flow is in [AGENTS.md](../../AGENTS.md),
Releases.

**Android speech.** The Kotlin AAR links the private Phonon/Kitten TTS 2 engine
into its arm64-v8a slice. `release-prep.yml`'s Build Android job binds the
`zzz-release` environment to stage the engine, and fails without it;
`tools/scripts/verify_zzz_android.py` gates the AAR and the
`xybrid-android-<tag>.zip` there, its Verify Android Gradle packaging job
replays the Gradle publish on that zip before the draft exists, and
`release-publish.yml` gates both again before Maven Central. The environment's
branch rule admits `release/v*` refs, and GitHub checks only the run's ref: any
workflow on a `release/v*` branch can bind it. Who may push `release/v*` is
therefore what guards the engine credentials; a repository ruleset should
limit it to maintainers and the release automation. `test-zzz-android.yml` proves the same
build, gates and Gradle packaging on `master` with the master-only
`zzz-engine` environment, which `test-zzz-engine.yml` and `trusted-zzz-sdk.yml`
also use.

**Manual only.** `test-ci.yml` dry-runs the speech-free Kotlin packaging; the
speech publish path is rehearsed by dispatching `test-zzz-android.yml` on
`master`, and `release-prep.yml` precompiles the Flutter binaries. `unity-activation.yml` emits the Unity license activation request; it is
kept for when the license expires or the GameCI image changes.
`web-webgpu.yml` runs WebGPU generation on a self-hosted runner labelled
`webgpu`, because hosted runners have no GPU adapter with `shader-f16`.
