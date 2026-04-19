# MLX Apple XCFramework

This directory vendors a prebuilt `mlx.xcframework` built from upstream
[ml-explore/mlx](https://github.com/ml-explore/mlx) and
[ml-explore/mlx-c](https://github.com/ml-explore/mlx-c). Downstream consumers
(the `mlx-c-sys` and `xybrid-mlx` crates, and the Flutter / Apple SDK bindings)
link against this xcframework so they do not need CMake or a Metal toolchain
locally.

## Layout

```
vendor/mlx-apple/
├── README.md                 # this file (tracked)
├── UPSTREAM_VERSIONS.txt     # pinned upstream SHAs + release metadata (tracked)
├── .gitignore                # excludes mlx.xcframework/ and .installed-sha256 (tracked)
├── mlx.xcframework/          # populated by fetch-mlx-xcframework.sh (gitignored)
│   ├── ios-arm64/
│   │   ├── libmlx-combined.a
│   │   ├── Headers/
│   │   └── Resources/*.metallib
│   ├── ios-arm64-simulator/
│   │   └── ... (same layout)
│   └── macos-arm64/
│       └── ... (same layout)
└── .installed-sha256         # tracks the SHA of the currently-installed artifact (gitignored)
```

## Upstream Sources

| Component | Upstream | Role |
|-----------|----------|------|
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | Apple's array framework, Metal kernels, auto-diff engine | Produces `libmlx.a` + `*.metallib` |
| [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c) | C API shim over mlx's C++ surface | Produces `libmlxc.a` |

Both are MIT-licensed. See the upstream `LICENSE` files for the canonical
attribution text. This directory redistributes the compiled binaries under the
same MIT terms; the license text travels with the xcframework's
`Info.plist`/metadata produced by `xcodebuild -create-xcframework`.

## Pinned Version

The current pin lives in `UPSTREAM_VERSIONS.txt` as two full-length commit
SHAs (`mlx=` and `mlx-c=`) plus a `release=` tag and `sha256=` pair identifying
the published artifact. The initial pins are:

- `mlx` at tag `v0.31.1`
- `mlx-c` at tag `v0.6.0`

These are starting points — bump them in a dedicated PR when a newer MLX
release is required, retrigger the build workflow, and update `release=` +
`sha256=` from the resulting release page.

## Fresh-Clone Setup

Run the fetch helper from the repo root:

```bash
./tools/scripts/fetch-mlx-xcframework.sh
```

The script:

1. Reads `UPSTREAM_VERSIONS.txt` for the pinned `release=` and `sha256=`.
2. If `mlx.xcframework/` is already present **and** `.installed-sha256` matches
   the pin, exits `0` (idempotent — safe to call repeatedly).
3. Otherwise downloads
   `https://github.com/xybrid-ai/xybrid/releases/download/<release>/mlx-<version>.xcframework.zip`,
   verifies the SHA256, and unpacks into `vendor/mlx-apple/mlx.xcframework/`.

If the pin is still `release=unpublished` the script exits with instructions
to run the `Build MLX XCFramework` workflow first.

## Build Reproduction (CI path)

The same xcframework is built reproducibly in
`.github/workflows/build-mlx-xcframework.yml`. To rebuild manually on a local
macOS machine matching the CI configuration (macos-15, Xcode 16):

```bash
# Clone upstream at the pinned SHAs.
mkdir -p build/upstream
git clone --filter=blob:none https://github.com/ml-explore/mlx.git build/upstream/mlx
git -C build/upstream/mlx checkout "$(grep '^mlx=' vendor/mlx-apple/UPSTREAM_VERSIONS.txt | cut -d= -f2)"
git clone --filter=blob:none https://github.com/ml-explore/mlx-c.git build/upstream/mlx-c
git -C build/upstream/mlx-c checkout "$(grep '^mlx-c=' vendor/mlx-apple/UPSTREAM_VERSIONS.txt | cut -d= -f2)"

# Build each slice via the shared helper.
for slice in iphoneos-arm64 iphonesimulator-arm64 macos-arm64; do
  case "$slice" in
    iphoneos-arm64)          sysroot=iphoneos;         system=iOS;    target=15.0 ;;
    iphonesimulator-arm64)   sysroot=iphonesimulator;  system=iOS;    target=15.0 ;;
    macos-arm64)             sysroot=macosx;           system=Darwin; target=13.3 ;;
  esac
  ./tools/scripts/build-mlx-slice.sh \
    --slice "$slice" \
    --sysroot "$sysroot" \
    --arch arm64 \
    --system-name "$system" \
    --deployment-target "$target" \
    --mlx-src build/upstream/mlx \
    --mlx-c-src build/upstream/mlx-c \
    --out "build/slices/$slice"
done

# Assemble and package (mirrors the workflow's `Assemble` + `Package` steps).
```

## How to Update

1. Bump `mlx=` and `mlx-c=` in `UPSTREAM_VERSIONS.txt` to the new commits.
2. Commit and push to `master`; the workflow rebuilds on path-change.
3. Tag a release `mlx-vX.Y.Z` matching the semver bump — the workflow attaches
   `mlx-X.Y.Z.xcframework.zip` + `.sha256` to the GitHub Release.
4. Update `release=` and `sha256=` in `UPSTREAM_VERSIONS.txt` to point at the
   new release, commit, push.
5. Downstream consumers re-run `./tools/scripts/fetch-mlx-xcframework.sh` to
   pull the new artifact.
