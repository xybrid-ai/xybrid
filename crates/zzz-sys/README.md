# Kitten TTS 2 engine binary pins

`natives-manifest.json` pins the published zzz CPU archives for Kitten TTS 2
accepted by xybrid. The engine release version is independent of the xybrid SDK
version. This directory currently holds the distribution contract; downloading,
Rust bindings, native linking, and runtime adapters are subsequent integration
steps.

## Manifest format

The manifest identifies the engine as `zzz` and pins release `v0.1.0-rc.3`.
Public fields describe the accepted binaries and their compatibility
requirements. Source repository locations, source revisions, credentials, and
workflow URLs belong to private build records.

| Field | Meaning |
| --- | --- |
| `schema_version` | Consumer manifest format; currently `1` |
| `engine` | Engine identity; `zzz` |
| `release_tag` | Exact engine release tag |
| `archives` | Entries selected by `(target, profile)` |

Each entry records the Rust target, platform slice, exact asset name, SHA-256
of the compressed archive, receipt schema, C API versions, compiled model
family, library/header paths, backend, CPU requirement, deployment minimum, and
required system libraries. The Rust target spelling differs from the engine's
Zig target spelling.

Every entry has profile `kitten-tts2`, model family `kitten-tts2`, and
`zzz_embed` ABI version `1`. Receipt schema `3` is a separate version from both
the consumer manifest format and the C API. Model weights and prepared voices
are supplied separately. A compiled family does not imply support for every
checkpoint or quantization.

## Accepted binary targets

| Rust target | Slice | Requirement |
| --- | --- | --- |
| `aarch64-apple-darwin` | `macos-arm64` | Apple Silicon, macOS 13+ |
| `aarch64-linux-android` | `android-arm64` | ARM64, Android API 29+ |
| `x86_64-unknown-linux-gnu` | `linux-x86_64` | x86-64-v3 CPU, glibc 2.28+ compiled baseline |

The Android consumer's final shared library must retain
`-Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384` and pass 16 KB ELF
alignment checks. A static archive itself has no ELF load segments. Linux
consumers must enforce the x86-64-v3 CPU requirement; linking with a newer
consumer sysroot can raise the final application's glibc minimum. Minimums
describe compiled targets, not certification on every older OS.

## Private resolution and staging contract

The downloader will select the exact `kitten-tts2` entry for the Rust target.
It will receive `ZZZ_RELEASE_REPOSITORY` (`OWNER/REPO`) and `ZZZ_GITHUB_TOKEN`
through private CI configuration or the local environment. Keep both values
out of committed files and public logs. The manifest has no repository-location
fallback. Default/fork builds without zzz require no private assets.

Resolve the pinned tag and asset name, then verify the compressed archive
against its committed SHA-256 before extracting. Validate the archive layout
and receipt compatibility before staging inputs for Cargo or Bazel. The
receipt's release tag, slice, profile, models, APIs, header/library paths,
backend, CPU requirement, and deployment minimum must match the selected entry.
Schema 3 receipt `engine_version` must match `release_tag` without its leading
`v`, and `abi_version` must match `apis.zzz_embed`. Its
`dependencies.system_libraries` must match `system_libraries`; compiler runtime
must be bundled and the library must be position independent.

Current schema-3 receipts contain source repository and revision information.
Raw archives and their receipts remain private verification inputs. Keep them
out of public SDK packages, build artifacts, logs, and caches. Retain verified
receipts in private staging alongside the source-to-artifact mapping. SDK users
receive the linked engine. Stage engine headers and libraries privately, and
audit final SDK package contents and linked binaries for private identifiers
before publication. Public raw engine downloads require a distribution format
that keeps source mapping private.

`XYBRID_ZZZ_PREBUILT_DIR` is the planned local/Cargo staging override. The same
archive and receipt verification requirements apply to staged inputs. An
enabled zzz build must fail if its target/profile is unsupported or its required
artifact is missing or incompatible. Native SDK release jobs fetch before
Bazel analysis; build actions link declared staged inputs. Link the selected
`lib/libzzz_embed.a` and the manifest's required system libraries.

These environment names define the intended consumer contract. Downloading,
staging verification, feature gates, and native linking are still to be
implemented.

## Updating the pins

1. Verify the published Kitten TTS 2 archives privately.
2. Derive entries from verified bytes and receipts. Retain source mapping in
   private build records and export only the public compatibility fields.
3. Update `release_tag` and the three Kitten TTS 2 entries in a reviewed xybrid
   PR. Keep entries sorted by `(target, profile)` and unique by that pair.
4. Confirm every checksum and compatibility field against the downloaded
   release, then validate consumer paths before shipping an SDK with it.

Published archive bytes remain immutable. A changed archive requires a new
engine release and newly reviewed pins.
