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
  `CI Success` (`ci.yml`) and
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
| [`test-choice-conformance.yml`](test-choice-conformance.yml) | CI: Choice conformance | push to `master`, pull_request to `master`, schedule (`23 5 * * *`), workflow_dispatch | none | Regenerate and verify choice-scoring conformance artifacts against pinned sha256s, then score FORMS through core, the Rust SDK and the CLI; daily cold rebuild. |
| [`clean-scripts.yml`](clean-scripts.yml) | CI: Clean scripts | push to `master`, pull_request to `master` | none | Test the root and per-folder clean.sh scripts and clean-lib.sh; every path they list must be one git ignores. |
| [`llamacpp-validate.yml`](llamacpp-validate.yml) | CI: llama.cpp backend | push to `master`, pull_request to `master`, workflow_dispatch | none | Check the llama.cpp pins and generated bindings, then run real text and vision models on a source build. |
| [`test-policy-routing.yml`](test-policy-routing.yml) | CI: Policy routing | push to `master`, pull_request to `master`, workflow_dispatch | none | Drive the xybrid binary through a policy-routed hybrid stage: local GGUF leg, fake DeepSeek cloud leg. |
| [`test-whispercpp.yml`](test-whispercpp.yml) | CI: whisper.cpp backend | push to `master`, pull_request to `master`, workflow_dispatch | none | Exercise whisper.cpp with real multilingual weights: translation, per-request language, long windows, non-speech suppression. |
| [`ci.yml`](ci.yml) | CI: Workspace | push to `master`, pull_request to `master` | none | Rust format, lint, tests and feature matrix, plus binding drift, API contract, Python and Web SDK checks. Gates merges. |
| [`test-zzz-android.yml`](test-zzz-android.yml) | CI: zzz Android AAR | push to `master`, workflow_dispatch | none | Build the arm64-v8a speech AAR on master with the pinned engine, gate it and its Gradle repackaging, then dlopen its x86_64 slice on an emulator. |
| [`test-zzz-engine.yml`](test-zzz-engine.yml) | CI: zzz engine | push to `master`, workflow_dispatch | none | Fetch and verify the pinned Phonon/Kitten-only engine slices and link them into Rust tests on each pinned target. |

### SDK

Per-platform SDK builds, wrapper tests and example apps.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`build-android.yml`](build-android.yml) | SDK: Android | push to `master`, pull_request to `master`, workflow_dispatch | none | Build the Android .so files for every ABI, test the Kotlin wrapper, and gate dlopen on an x86_64 emulator. |
| [`build-apple.yml`](build-apple.yml) | SDK: Apple | push to `master`, pull_request to `master`, workflow_dispatch | none | Build and verify the Apple XCFramework, then compile and unit-test the Swift wrapper on an iOS Simulator. |
| [`build-flutter.yml`](build-flutter.yml) | SDK: Flutter | push to `master`, pull_request to `master`, workflow_dispatch | none | Analyze and test the Dart wrapper, build the native libraries, and build a consumer app against the packaged layout. |
| [`build-react-native.yml`](build-react-native.yml) | SDK: React Native | push to `master`, pull_request to `master`, workflow_dispatch | none | Test and npm-pack the JS package; build the iOS and Android example apps. |
| [`trusted-zzz-sdk.yml`](trusted-zzz-sdk.yml) | SDK: Trusted zzz Apple candidate (manual) | workflow_dispatch | none | Manual: Produce a source-verified, notice-bearing Apple SDK candidate from genuine schema-5 engine pins on protected master. |
| [`unity-editor.yml`](unity-editor.yml) | SDK: Unity Editor | push to `master`, pull_request to `master`, workflow_dispatch | checks | Run a real Unity Editor: EditMode tests against a Bazel-built native, then an IL2CPP player smoke. |
| [`web-webgpu.yml`](web-webgpu.yml) | SDK: WebGPU inference (manual) | workflow_dispatch | none | Manual: build the Web SDK and run WebGPU generation in Chromium on a self-hosted GPU runner. |

### Artifacts

Publish reusable native artifacts and their download manifests.

| File | Name | Triggers | Writes | Purpose |
| --- | --- | --- | --- | --- |
| [`build-mlx-xcframework.yml`](build-mlx-xcframework.yml) | Artifacts: MLX XCFramework | push to `master` of tags `mlx-v*`, pull_request to `master`, workflow_dispatch | contents | Build the pinned MLX XCFramework for validation and publish it on mlx-v* tags. |
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
| [`test-ci.yml`](test-ci.yml) | Maintenance: Packaging tools (manual) | workflow_dispatch | contents | Manual: precompile and upload Flutter binaries per platform, and dry-run the speech-free Kotlin packaging. |
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
checks that each skipped job had a recorded reason.

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

**Manual only.** `test-ci.yml` precompiles the Flutter binaries per platform and
uploads them to GitHub Releases (`contents: write`), and dry-runs the
speech-free Kotlin packaging; the speech publish path is rehearsed by
dispatching `test-zzz-android.yml` on `master`. `unity-activation.yml` emits the Unity license activation request; it is
kept for when the license expires or the GameCI image changes.
`web-webgpu.yml` runs WebGPU generation on a self-hosted runner labelled
`webgpu`, because hosted runners have no GPU adapter with `shader-f16`.

---

## `build-mlx-xcframework.yml`

Builds a fat `mlx.xcframework` from upstream
[`ml-explore/mlx`](https://github.com/ml-explore/mlx) and
[`ml-explore/mlx-c`](https://github.com/ml-explore/mlx-c) so that downstream
consumers (the `mlx-c-sys` crate and `vendor/mlx-apple/`) can link against a
prebuilt artifact instead of carrying CMake + a Metal toolchain.

### When it runs

| Trigger | Behaviour |
|---|---|
| `workflow_dispatch` | Manual rebuild (e.g. after bumping `vendor/mlx-apple/UPSTREAM_VERSIONS.txt`). |
| `push` to `master` touching `vendor/mlx-apple/UPSTREAM_VERSIONS.txt` or this workflow | Validate that pinned SHAs still build cleanly. |
| `pull_request` to `master` (same paths) | Same validation, in PR context. |
| `push` of a tag matching `mlx-v*` | Builds **and** uploads `mlx-{version}.xcframework.zip` + `.sha256` to the GitHub Release for that tag. |

The release tag namespace (`mlx-v*`) is intentionally distinct from the main
`v*` release stream — the xcframework cadence is decoupled from xybrid SDK
releases.

### Runner & toolchain

- `runs-on: macos-15` with `Xcode_16.app` selected explicitly.
- `sccache` caches the CMake build to keep cold-rebuild time bounded.

### Pinning model

Upstream commits are read from `vendor/mlx-apple/UPSTREAM_VERSIONS.txt` in
`key=value` format:

```
mlx=<full sha>
mlx-c=<full sha>
```

The workflow refuses to build if that file is missing or either pin is empty,
so the source-of-truth lives next to the vendored xcframework rather than in
workflow YAML. The pins file also records whether a downloadable xcframework is
published; when it is, the fetch script verifies that artifact's SHA256.

### Build steps

1. Clone `ml-explore/mlx` and `ml-explore/mlx-c` at the pinned SHAs.
2. Build three CMake slices via `tools/scripts/build-mlx-slice.sh`:
   - `iphoneos-arm64` (`SDK=iphoneos`, `arch=arm64`)
   - `iphonesimulator-arm64` (`SDK=iphonesimulator`, `arch=arm64`)
   - `macos-arm64` (`SDK=macosx`, `arch=arm64`)

   The build requests `MLX_BUILD_METAL=ON` for every slice, but pinned upstream
   MLX forces Metal off when `CMAKE_SYSTEM_NAME=iOS`. The iOS slices therefore
   exist for link-layout compatibility only until upstream supports iOS Metal
   builds. Tests, examples, benchmarks, and the Python bindings are disabled.
3. Combine `libmlx.a` + `libmlxc.a` into `libmlx-combined.a` per slice (via
   `libtool -static`) so the xcframework consumes a single archive per slice.
4. Run `xcodebuild -create-xcframework` against the three combined archives
   plus their merged header tree.
5. Copy each slice's compiled `*.metallib` into the matching slice's
   `Resources/` directory inside `mlx.xcframework`. Consumers (the
   `mlx-c-sys` crate's `build.rs`) resolve them relative to the slice library
   path. Missing macOS `*.metallib` is an error; missing iOS `*.metallib` is
   expected with the current upstream pins and is emitted as a notice.
6. Zip the framework as `mlx-{version}.xcframework.zip` and write a sibling
   `.sha256` file.
7. Upload both files as a workflow artifact (`mlx-xcframework`).
8. On a `mlx-v*` tag, attach both files to the matching GitHub Release.

The main CI workflow also has a macOS `llm-mlx-runtime` job. It first runs
`tools/scripts/fetch-mlx-xcframework.sh --check-published`. When
`vendor/mlx-apple/UPSTREAM_VERSIONS.txt` contains a download pin, CI fetches
and verifies that artifact. When the pin still says
`release=unpublished` or `sha256=unpublished`, CI builds a local macOS-only
`mlx.xcframework` from the pinned upstream SHAs instead, then runs the same
linked runtime checks. This keeps runtime CI independent of artifact publication
state. The job checks that iOS still accepts only the non-linking `llm-mlx`
tier and that `llm-mlx-runtime` fails with the Apple Silicon macOS-only
compile error. The iOS non-linking build writes to `RUNNER_TEMP`, and the gate
script uses its own temporary Cargo target, so those transient artifacts are
removed before the linked Rust checks without wiping the runtime job's main
Cargo target. The local MLX source-build script also deletes its temporary
source/build tree by default to keep macOS runner disk use bounded. It then runs
the `mlx-c-sys`, `xybrid-mlx`, `xybrid-core`, `xybrid-sdk`, CLI, LLM and
embedding benchmark compile checks, and Apple-platform link checks. The
`xybrid-core` runtime test steps run the full `tests/mlx_llm_chat.rs` harness
and the `tests/mlx_embedding.rs` harness.
Synthetic Qwen/Gemma/LFM smoke fixtures always run, while real-bundle
Qwen/Gemma/LFM and Nomic embedding checks skip unless their staged env vars
are present. Qwen real smokes accept either the legacy
`XYBRID_MLX_QWEN_DIR` or the manifest-declared `XYBRID_MLX_QWEN_4B_DIR`;
Gemma, LFM2, LFM2.5, and Nomic use `XYBRID_MLX_GEMMA_DIR`,
`XYBRID_MLX_LFM_DIR`, `XYBRID_MLX_LFM25_DIR`, and `XYBRID_MLX_NOMIC_DIR`. The
staged bundle directories must contain the files declared in
`integration-tests/fixtures/models/models.json`; indexed SafeTensors fixtures must
also include every shard referenced by `model.safetensors.index.json`.
Benchmark-only exact fixture rows use their own manifest-declared env vars,
such as `XYBRID_MLX_QWEN_4B_DIR` and `XYBRID_MLX_LFM25_DIR`, so CI/runtime
smoke fixtures cannot accidentally feed a stale LFM3.5 placeholder into the benchmark.
The Linux non-linking MLX job also runs `tools/scripts/test-bench-scripts.sh` to
pin benchmark-wrapper host-gate behavior: normal mode treats unsupported hosts
as skips, while `--strict` exits non-zero for those same skips.

### Version resolution

- On a `mlx-v*` tag the version is the tag minus the `mlx-v` prefix
  (`mlx-v0.18.1` → `0.18.1`).
- On all other triggers the version is `0.0.0+sha.<7-char mlx sha>` so PR
  artifacts are unambiguously identifiable but cannot be confused with a
  published xcframework.

### Local validation

```bash
# Lint the workflow with the same tool CI uses:
actionlint .github/workflows/build-mlx-xcframework.yml

# Lint the slice script:
shellcheck tools/scripts/build-mlx-slice.sh
```

### Updating the pinned MLX version

1. Edit `vendor/mlx-apple/UPSTREAM_VERSIONS.txt` with the new SHAs.
2. Push to `master` (or open a PR) — the workflow rebuilds and uploads an
   artifact you can sanity-check.
3. When ready, tag `mlx-vX.Y.Z` to publish the artifact + SHA256 to a
   GitHub Release.
4. Update `release=` and `sha256=` in
   `vendor/mlx-apple/UPSTREAM_VERSIONS.txt` with the published release tag and
   artifact SHA256. The fetch script derives the URL from those pins.
5. Update `vendor/mlx-apple/README.md` if the upstream version or operational
   instructions changed.
