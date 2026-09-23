# MLX Apple XCFramework

This directory materializes an `mlx.xcframework` built from upstream
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
├── .gitignore                # excludes generated xcframework/markers (tracked)
├── mlx.xcframework/          # populated by fetch or local source build (gitignored)
│   ├── ios-arm64/
│   │   ├── libmlx-combined.a
│   │   ├── Headers/
│   │   └── Resources/          # iOS Metal libraries are absent with current pins
│   ├── ios-arm64-simulator/
│   │   └── ... (same layout)
│   └── macos-arm64/
│       └── ... (same layout)
├── .installed-sha256         # tracks the SHA of the currently-installed artifact (gitignored)
└── .installed-source-pin      # tracks local source-build pins when no release artifact is used
```

The macOS slice must contain `Resources/mlx.metallib`. Current pinned upstream
MLX disables Metal for `CMAKE_SYSTEM_NAME=iOS`, so iOS slices are packaged for
link-layout compatibility only and are not evidence of iOS MLX runtime
readiness.

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
SHAs (`mlx=` and `mlx-c=`) plus an optional `release=` tag and `sha256=` pair
identifying the published download artifact. The source pins are sufficient for
runtime CI and local Apple Silicon validation; the release fields are only
needed for download-based installs through `fetch-mlx-xcframework.sh`.

The initial pins are:

- `mlx` at tag `v0.31.1`
- `mlx-c` at tag `v0.6.0`

These are starting points — bump them in a dedicated PR when a newer MLX source
revision is required. CI can validate the runtime from source pins before a
published xcframework artifact exists.

## Fresh-Clone Setup

When `release=` and `sha256=` point at a published artifact, run the fetch
helper from the repo root:

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

If the pin is still `release=unpublished`, the fetch helper reports that no
download artifact is available. To validate the runtime without a published
release artifact on Apple Silicon macOS, source-build the macOS runtime slice
from the pinned upstream commits:

```bash
./tools/scripts/build-local-mlx-xcframework.sh
```

That fallback builds only the `macos-arm64` slice, requires Xcode command-line
tools plus the MLX build dependencies used by CI, writes
`.installed-source-pin`, removes any stale `.installed-sha256`, and deletes its
temporary build directory by default.

## Build Reproduction

Runtime CI uses `tools/scripts/build-local-mlx-xcframework.sh` whenever the
download pin is unpublished, so the MLX runtime checks are not blocked on a
GitHub Release. The full multi-slice xcframework is built reproducibly in
`.github/workflows/build-mlx-xcframework.yml`. To rebuild that full artifact
manually on a local macOS machine matching the workflow configuration
(macos-15, Xcode 16):

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
    iphoneos-arm64)          sysroot=iphoneos;         system=iOS;    target=16.4 ;;
    iphonesimulator-arm64)   sysroot=iphonesimulator;  system=iOS;    target=16.4 ;;
    macos-arm64)             sysroot=macosx;           system=Darwin; target=14.0 ;;
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
2. Validate with `./tools/scripts/build-local-mlx-xcframework.sh` or let runtime
   CI run the same release-free source-build path.
3. Optionally trigger `.github/workflows/build-mlx-xcframework.yml` for the full
   iOS + macOS packaging artifact.
4. If a downloadable artifact is needed, tag `mlx-vX.Y.Z`; the workflow attaches
   `mlx-X.Y.Z.xcframework.zip` + `.sha256` to the GitHub Release.
5. Update `release=` and `sha256=` only when you want
   `./tools/scripts/fetch-mlx-xcframework.sh` to pull that published artifact.
