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
workflow YAML. (US-002 establishes the pins file and a fetch script that
verifies the published artifact's SHA256.)

### Build steps

1. Clone `ml-explore/mlx` and `ml-explore/mlx-c` at the pinned SHAs.
2. Build three CMake slices via `tools/scripts/build-mlx-slice.sh`:
   - `iphoneos-arm64` (`SDK=iphoneos`, `arch=arm64`)
   - `iphonesimulator-arm64` (`SDK=iphonesimulator`, `arch=arm64`)
   - `macos-arm64` (`SDK=macosx`, `arch=arm64`)

   All slices build with `MLX_BUILD_METAL=ON`, `BUILD_SHARED_LIBS=OFF`. Tests,
   examples, benchmarks, and the Python bindings are disabled.
3. Combine `libmlx.a` + `libmlxc.a` into `libmlx-combined.a` per slice (via
   `libtool -static`) so the xcframework consumes a single archive per slice.
4. Run `xcodebuild -create-xcframework` against the three combined archives
   plus their merged header tree.
5. Copy each slice's compiled `*.metallib` into the matching slice's
   `Resources/` directory inside `mlx.xcframework`. Consumers (the
   `mlx-c-sys` crate's `build.rs`) resolve them relative to the slice library
   path. A warning is emitted if a slice produced no `.metallib`.
6. Zip the framework as `mlx-{version}.xcframework.zip` and write a sibling
   `.sha256` file.
7. Upload both files as a workflow artifact (`mlx-xcframework`).
8. On a `mlx-v*` tag, attach both files to the matching GitHub Release.

### Version resolution

- On a `mlx-v*` tag the version is the tag minus the `mlx-v` prefix
  (`mlx-v0.18.1` → `0.18.1`).
- On all other triggers the version is `0.0.0+sha.<7-char mlx sha>` so PR
  artifacts are unambiguously identifiable but cannot be confused with a
  release artifact.

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
4. Update `vendor/mlx-apple/README.md` with the new published SHA256.
5. Bump the URL/SHA in `tools/scripts/fetch-mlx-xcframework.sh` so fresh
   clones pull the published artifact.
