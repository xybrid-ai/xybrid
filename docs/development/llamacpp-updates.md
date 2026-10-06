# llama.cpp stable update automation

The POC checks for an official llama.cpp release once a day at 07:43 UTC.
It follows GraphQL cursors through the upstream release history and selects
the highest published `vX.Y.Z` version, skipping development builds, drafts,
prereleases, and malformed tags. GitHub's `releases/latest` can point at a
`b*` development build, so it does not identify the stable channel reliably.
With no qualifying stable release, discovery succeeds with no update and
release preparation can still run. Annotated tags resolve to immutable commit
SHAs. The committed submodule, Cargo fallback pin, and
`.github/llamacpp-version.json` must agree.
Cursor pagination avoids the REST release list's 10,000-result limit and
stops when `hasNextPage` is false, including when the final page is full.

## Update review

`llamacpp-update.yml` prepares one `chore/llamacpp-vX.Y.Z` PR against master.
It updates the source pins and changelog, compiles the C++ wrapper, and
regenerates the committed vision-capable Rust bindings through the existing
Cargo build script. If the wrapper does not compile, it still opens a draft
PR so a maintainer can adapt it. Binding generation errors remain visible in
the intake workflow log; the validation check also fails until they are fixed.

The bot never rewrites an existing update branch or a PR under review. Each
upstream version has its own branch, so merged branches do not block future
updates. If you close an update PR to reject it, leave its branch in place to
suppress retries for that version; delete it when you want another attempt.
Re-running discovery without a new upstream commit is a no-op.

The update PR runs the existing Cargo, Bazel, SDK streaming, policy-routing,
and Whisper workflows. `llamacpp-validate.yml` additionally verifies the pins
and binding snapshot, runs real grammar-constrained text generation, and runs
the existing LFM2-VL caption test with both model files present. Missing model
files fail validation. Whisper consumes llama.cpp's ggml, so its real-model
suite is a release prerequisite too.

One update triggers several workflows for platform builds and integration
tests. These are checks on the same PR, not repeated discoveries. The single
`CI Success` check belongs to `ci.yml`; its build matrix is skipped only when
every changed path is documentation. A changelog alongside a native update
still requires the complete matrix.

SDK builds disable llama.cpp's application and CLI tools and enable standalone
mtmd for vision. Static vision builds also stage and link upstream's internal
`vendor-hash` archive. Cargo, Bazel, and the native publisher must all carry
that dependency; an incomplete prebuilt slice falls back to a source build.

The choice-conformance provenance pins the llama.cpp converter as well as the
runtime submodule. An update will report that drift. Review its conversion
recipe and regenerate the derived artifacts deliberately; do not replace
expected digests merely to make the check green.
The Qwen fixture excludes the optional MTP prediction head with `--no-mtp`,
preserving the text trunk covered by the reference goldens. Record new artifact
digests only after rebuilding from the pinned inputs and checking the embedded
chat template.

## Release preparation

After the update PR merges, the existing `build-natives.yml` publishes fresh
native slices and opens its generated manifest PR. Review and merge that PR.
The bot waits for the manifest's llama.cpp commit and all three source hashes
to match, and requires every base/vision target in the native build matrix.
It also verifies that the slices are anonymously reachable before cutting a
release branch.

The daily run checks validation on the exact master commit it will release:
CI Success, Bazel, stable-update validation, policy routing, Whisper, and
choice-conformance provenance must all succeed. Any other pending or failing
check also blocks preparation. The intake/preparation jobs themselves are
excluded so the bot does not wait on its own running check.

For a stable `0.x` workspace with a classified patch or minor upstream update,
the default target is the next minor version (currently `0.11.0`). This includes
all unreleased master changes and avoids claiming patch compatibility from
the upstream version alone. The first tracked update is classified as
`bootstrap`: without a previous upstream tag, its compatibility is unknown.
Bootstrap updates, upstream major upgrades, and SDK versions >=1.0 require an
explicit `sdk_version` input.
Automatic patch classification is outside this POC.
A prerelease SDK workspace waits for its current release to finish before
preparing another release branch; upstream discovery can still open an update PR.

Preparation updates internal Cargo path-dependency constraints, runs the
existing package-version sync and Python binding generator, moves the root
and Flutter Unreleased changelogs into the release entry, and sets SPM to
remote natives. It pushes `release/v<version>`; `release-prep.yml` then builds
the SDK artifacts, patches checksums, creates the draft release, and opens
the release PR. A human reviews that PR. Merging it invokes the existing
publishing workflow.

Release branches record the SDK version and exact llama.cpp commit they
include. The normal version-sync script stamps manually prepared release
branches too. A version bump alone never marks an upstream update as shipped;
an older release branch can have been built before that update landed.
Preparation skips when this exact upstream commit is recorded as released,
another release PR or upcoming release branch exists, or the target tag exists.
If master advances during generation, it aborts before pushing and retries
on a later run. A failed release build leaves a reviewable release branch
that a maintainer can repair and push again.

## Setup and rehearsal

Use the existing `RELEASE_PAT` secret with repository contents and pull-request
write access, plus permission to read checks. It must belong to an identity
that can push the bot and release branches. `GITHUB_TOKEN` can perform discovery
but cannot be used to create these branches/PRs: its events do not trigger the
downstream validation and release workflows. The automation refuses those
writes if `RELEASE_PAT` is absent. Scheduled writes run only in `xybrid-ai/xybrid`.

Dispatch **Update llama.cpp stable** with `dry_run: true` (the default) to
report the upstream candidate and release readiness without creating a branch,
PR, tag, or release. To accelerate preparation after validation finishes,
dispatch it with `dry_run: false`; supply `sdk_version` for the first tracked
update, upstream major upgrades, or SDK versions >=1.0. Both jobs check out
master even if dispatched from another ref. Deployment of the workflows
themselves still requires merging this POC into master.

For a local read-only rehearsal:

```bash
python3 tools/scripts/llamacpp_update.py discover
python3 tools/scripts/llamacpp_update.py release-plan
python3 -m unittest discover -s tools/scripts/tests -p test_llamacpp_update.py
```

Discovery needs an authenticated `gh` and Git access to origin. Binding
regeneration uses Cargo, rustfmt, CMake, and libclang. Release preparation also
needs the pinned BoltFFI CLI; the workflow installs it automatically.
