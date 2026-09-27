# GitHub Actions Workflows

Reference for the workflows under `.github/workflows/`.

## Continuous integration

| Workflow | File | Purpose |
|---|---|---|
| CI | `ci.yml` | Format, clippy, and per-platform feature checks on every push/PR to `master`. |
| CodeQL | `codeql.yml` | GitHub-hosted code scanning. |
| Scorecard | `scorecard.yml` | OpenSSF supply-chain scorecard. |
| Test CI | `test-ci.yml` | Workspace test runs. |

## Per-platform builds (push / PR / manual)

| Workflow | File | Purpose |
|---|---|---|
| Build Apple | `build-apple.yml` | XCFramework build on `macos-14`. |
| Build Android | `build-android.yml` | Android `.so` build with NDK. |
| Build Flutter | `build-flutter.yml` | flutter_rust_bridge regeneration + smoke build. |
| Build Unity | `build-unity.yml` | Unity native plugin packaging. |
| **Build MLX XCFramework** | **`build-mlx-xcframework.yml`** | **See dedicated section below.** |

## Release

| Workflow | File | Purpose |
|---|---|---|
| Release | `release.yml` | Triggered on `v*` tags; builds, packages, publishes to pub.dev / Maven / Swift branch / GitHub Release. |

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
