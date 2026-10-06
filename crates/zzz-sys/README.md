# zzz engine binary pins

`natives-manifest.json` pins the published zzz CPU engine archives accepted by
xybrid. The engine release version is independent of the xybrid SDK version.
This directory currently holds the distribution contract; native fetching,
Rust bindings, build linking, and runtime adapters are subsequent integration
steps.

## Manifest format

The manifest identifies the engine as `zzz`. Its public fields describe the
accepted binaries and their compatibility requirements. Source repository
locations, source revisions, download credentials, and workflow URLs belong to
private build records.

| Field | Meaning |
| --- | --- |
| `schema_version` | Version of this consumer manifest format; currently `1` |
| `engine` | Engine identity; `zzz` |
| `release_tag` | Exact engine release tag; currently `v0.1.0-rc.3` |
| `archives` | Entries selected by the pair `(target, profile)` |

Every archive entry records:

- `target`: the Cargo/Rust target triple, which differs from the engine's Zig
  target spelling.
- `slice` and `profile`: the platform slice and compiled engine selection.
- `archive` and `sha256`: the exact asset name and SHA-256 of its compressed
  bytes. The committed hash is the expected value for download verification.
- `receipt_schema`: the archive receipt format; currently `3`. This is separate
  from the consumer manifest format and the C API versions.
- `apis` and `models`: required C API versions and compiled model families.
- `library`, `headers`, and `system_libraries`: paths inside the archive root
  and system libraries needed by the consumer linker.
- `backend`, `cpu`, and `deployment_minimum`: runtime compatibility requirements.

An entry is unique by `(target, profile)`. The three alternative profiles are
`combined`, `phonon`, and `kitten-tts2`. Combined contains the implemented CPU
engine families and exposes `zzz_embed` ABI 1, `zzz` ABI 2, and `zzz_engine`
ABI 1. The selective profiles expose `zzz_embed` ABI 1. A compiled family does
not imply support for every checkpoint or quantization. Model weights and
prepared voices are supplied separately. Metal LLM support is excluded.

## Supported targets

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

The downloader will select an exact entry by Rust target and requested profile.
It will receive `ZZZ_RELEASE_REPOSITORY` (`OWNER/REPO`) and `ZZZ_GITHUB_TOKEN`
through private CI configuration or the local environment. Repository location
and token values must stay out of committed files and public logs. There is no
repository-location fallback in this manifest. CI and release builds obtain
these settings through Actions secrets; public/fork builds without zzz require
no private assets.

The downloader must resolve the pinned tag and exact asset name, verify the
compressed archive against the committed SHA-256 before extracting, validate
the archive layout and receipt compatibility, then stage the verified payload
for Cargo or Bazel. Validate the receipt's release tag, slice, profile, models,
APIs, header/library paths, and CPU/platform requirements against the selected
entry. Schema 3 receipt `engine_version` must match `release_tag` without its
leading `v`, and `abi_version` must match `apis.zzz_embed`. The receipt's
`dependencies.system_libraries` must match `system_libraries`.

Current receipt-schema-3 archives also contain source repository and revision
information. Raw archives and their receipts are private verification inputs:
keep them out of SDK packages, public build artifacts, logs, and caches. Retain
the verified receipts in private staging alongside the full source-to-artifact
mapping. Public downloads of raw engine archives need a future distribution
format with an engine/build identity and private source mapping.

`XYBRID_ZZZ_PREBUILT_DIR` is the planned local/Cargo staging override. Staged
payloads must satisfy the same verification contract. An enabled zzz build must
fail if its target/profile is unsupported or its required artifact is missing
or incompatible. Native SDK release jobs will fetch before Bazel analysis;
build actions link declared staged inputs. Link exactly one profile's
`libzzz_embed.a` into each SDK binary. SDK users receive the linked engine.

These environment names describe the agreed consumer contract. Fetching,
staging verification, feature gates, and native linking are still to be
implemented.

## Updating the pins

1. Publish the engine release and verify the complete archive set privately.
2. Derive archive entries from the verified bytes and receipts. Retain the source
   mapping in private build records; export only the public manifest fields.
3. Update `release_tag` and the applicable entries in a reviewed xybrid PR.
   Keep entries sorted by `(target, profile)` and unique by that pair.
4. Confirm every new checksum and compatibility field against the downloaded
   release, then validate the consumer paths before shipping an SDK with it.

Previously published archive bytes remain immutable. A changed archive requires
a new engine release and newly reviewed pins.
