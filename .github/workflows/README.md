# GitHub Actions workflows

One file per pipeline; GitHub reads only top-level files in this directory. This
page is the map: what each workflow is for, when it runs, what it can write, and
which ones depend on each other.

## Conventions

[`tools/scripts/gen_workflow_index.py`](../../tools/scripts/gen_workflow_index.py)
`--check` (CI job `Workflow index`) enforces the first three. The last two are
decisions to keep.

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
  `CI Success` (reported by both `ci.yml` and `ci-docs.yml`) and
  `Bazel graph + RBE targets` (`bazel.yml`), matched by job name.
  `tools/scripts/llamacpp_update.py` waits on more job names before it cuts a
  llama.cpp release (its `REQUIRED_CHECKS`). A workflow's `name:` can change
  freely; those job names cannot.

## Preserving execution

Display-name and purpose edits must preserve workflow filenames, triggers and
path filters, job IDs and check names, matrices, steps and test commands,
dependencies, permissions, secret references and cancellation settings. The
literal concurrency keys stay stable when a display name changes.

The `Workflow index` job supplements the existing checks. Every original test
command continues to run in its existing job; the index job checks this page and
the naming conventions.

## Inventory

<!-- BEGIN GENERATED: workflow inventory (tools/scripts/gen_workflow_index.py) -->

### CI

Workspace validation, pull-request gates and backend test suites.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`bazel.yml`](bazel.yml) | CI: Bazel | push to `master`, pull_request to `master`, workflow_dispatch | none | Bazel graph analysis and remote-execution targets; a required check that skips when no Bazel input changed. |
| [`test-candle.yml`](test-candle.yml) | CI: Candle backend | push to `master`, pull_request to `master`, workflow_dispatch | none | Test the Candle backend (Metal runs only on macOS), scoped to the paths that can affect it. |
| [`test-choice-conformance.yml`](test-choice-conformance.yml) | CI: Choice conformance | push to `master`, pull_request to `master`, schedule (`23 5 * * *`), workflow_dispatch | none | Regenerate and verify choice-scoring conformance artifacts against pinned sha256s; daily cold rebuild. |
| [`clean-scripts.yml`](clean-scripts.yml) | CI: Clean scripts | push to `master`, pull_request to `master` | none | Test the root and per-folder clean.sh scripts and clean-lib.sh; every path they list must be one git ignores. |
| [`ci-docs.yml`](ci-docs.yml) | CI: Docs-only gate | push to `master`, pull_request to `master` | none | Report the CI Success gate for docs-only changes, which ci.yml skips. |
| [`llamacpp-validate.yml`](llamacpp-validate.yml) | CI: llama.cpp backend | push to `master`, pull_request to `master`, workflow_dispatch | none | Check the llama.cpp pins and generated bindings, then run real text and vision models on a source build. |
| [`test-policy-routing.yml`](test-policy-routing.yml) | CI: Policy routing | push to `master`, pull_request to `master`, workflow_dispatch | none | Drive the xybrid binary through a policy-routed hybrid stage: local GGUF leg, fake DeepSeek cloud leg. |
| [`test-whispercpp.yml`](test-whispercpp.yml) | CI: whisper.cpp backend | push to `master`, pull_request to `master`, workflow_dispatch | none | Exercise whisper.cpp with real multilingual weights: translation, per-request language, long windows, non-speech suppression. |
| [`ci.yml`](ci.yml) | CI: Workspace | push to `master`, pull_request to `master` | none | Rust format, lint, tests and feature matrix, plus binding drift, API contract, Python and Web SDK checks. Gates merges. |
| [`test-zzz-engine.yml`](test-zzz-engine.yml) | CI: zzz engine | push to `master`, pull_request to `master`, workflow_dispatch | none | Fetch and verify the pinned zzz (Kitten TTS) engine slices and link them into Rust tests on each pinned target. |

### SDK

Per-platform SDK builds, wrapper tests and example apps.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`build-android.yml`](build-android.yml) | SDK: Android | push to `master`, pull_request to `master`, workflow_dispatch | none | Build the Android .so files for every ABI, test the Kotlin wrapper, and gate dlopen on an x86_64 emulator. |
| [`build-apple.yml`](build-apple.yml) | SDK: Apple | push to `master`, pull_request to `master`, workflow_dispatch | none | Build and verify the Apple XCFramework, then compile and unit-test the Swift wrapper on an iOS Simulator. |
| [`build-flutter.yml`](build-flutter.yml) | SDK: Flutter | push to `master`, pull_request to `master`, workflow_dispatch | none | Analyze and test the Dart wrapper, build the native libraries, and build a consumer app against the packaged layout. |
| [`build-react-native.yml`](build-react-native.yml) | SDK: React Native | push to `master`, pull_request to `master`, workflow_dispatch | none | Test and npm-pack the JS package; build the iOS and Android example apps. |
| [`unity-editor.yml`](unity-editor.yml) | SDK: Unity Editor | push to `master`, pull_request to `master`, workflow_dispatch | checks | Run a real Unity Editor: EditMode tests against a Bazel-built native, then an IL2CPP player smoke. |
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
| [`test-ci.yml`](test-ci.yml) | Maintenance: Packaging tools (manual) | workflow_dispatch | contents | Manual: precompile and upload Flutter binaries per platform, and dry-run the Kotlin publish. |
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

**Docs-only changes.** `ci.yml` skips itself when a change touches only docs
paths (its `paths-ignore`), which would leave the required `CI Success` check
pending forever. `ci-docs.yml` runs on exactly those paths and reports the same
check. Keep the two path lists in sync and the job names identical.

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
`llamacpp-validate.yml` and the other required checks. Once that PR merges and
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

**Manual only.** `test-ci.yml` precompiles the Flutter binaries per platform and
uploads them to GitHub Releases (`contents: write`), and dry-runs the Kotlin
publish. `unity-activation.yml` emits the Unity license activation request; it is
kept for when the license expires or the GameCI image changes.
`web-webgpu.yml` runs WebGPU generation on a self-hosted runner labelled
`webgpu`, because hosted runners have no GPU adapter with `shader-f16`.
