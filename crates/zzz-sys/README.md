# zzz-sys

[natives-manifest.json](natives-manifest.json) pins the zzz CPU binaries for
Phonon and Kitten TTS 2, including release versions, archive checksums, ABI requirements,
and linker metadata.

| Pinned binary target | Compiled baseline |
| --- | --- |
| macOS ARM64 | macOS 13+ |
| Android ARM64 | Android API 29+ |
| Linux x86_64 | x86-64-v3 CPU, glibc 2.28+ |
| iOS ARM64 device / simulator | iOS 16+ |
| Linux ARM64 | baseline ARM64 CPU, glibc 2.28+ |

On Linux x86_64, Rust checks the complete x86-64-v3 CPU feature set and
OS-enabled AVX state before every engine FFI call, including capability
probes and native defaults. Unsupported CPUs return ABI version `0`, report
no supported models, and reject session work with a `LoadFailed` error naming
the required CPU floor. `availability()` reports the same condition as
`UnavailableReason::UnsupportedCpu` without opening model files. ABI `0`
means the runtime is unavailable; it does not mean the pinned artifact failed
ABI verification. Native defaults return zeroed settings with `struct_size`
set in this case; opening a session checks availability before validating
settings. Other target slices retain their existing behavior. The Rust host
must itself be built for its intended CPU baseline; this check cannot make a
host binary compiled for v3 run on v2.

Privately stage a pinned slice before native builds with
`python3 tools/scripts/zzz_pull.py --target <target>` from the repository root;
see `--help` for authentication, offline archives and staged verification.
The tool prints the verified slice path for `XYBRID_ZZZ_PREBUILT_DIR`. Keep its
archives, receipts and staging directories private. Model weights are supplied
separately. The same verifier is included in the published crate as
`zzz_pull.py`, beside `natives-manifest.json`; run it with
`python3 <crate-directory>/zzz_pull.py --target <target>`. It works without this
workspace and does not include the engine binary or private credentials.

## Rust bindings

`xybrid-zzz-sys` carries the ABI-1 FFI for `libzzz_embed.a` plus a thin safe
surface (`PhononSession`, `KittenSession`, typed `ZzzError`s). Both share the
same streaming, cancellation and ownership machinery. Default builds are a no-op shell;
enabling `bindings` links the pinned slice for the build target through the
same `zzz_pull.py` verification (cache hits are reverified; `XYBRID_ZZZ_USE_GH_AUTH`
opts local builds into stored gh credentials for activation) and fails the
build when the target is unsupported or the pinned slice is missing or fails
verification.

The `phonon-kitten-tts2` profile supplies exactly those two models in one
archive and exposes only the `zzz_embed` API. Single-model archives export
overlapping symbols and cannot be linked together. Staging verifies all 18
payload files, the exact model/API declarations and the archive's SHA-256
before either Cargo or Bazel links it. This SDK pins engine `v0.1.0-rc.9`
with receipt schema 5. The full `combined` engine is not accepted.

Receipt schema 4 remains accepted only for the frozen `rc.8` release. The
current schema-5 pins require the immutable 14-file notice
inventory, including runtime terms and upstream license texts. Each archive's
manifest entry must pin the exact schema-5 `notice_files` map; the receipt map
and every notice's bytes must match it before staging. Changing notice bytes
requires a new receipt schema. All six current archives were independently
compared with their published counterparts and inspected, staged and reverified
through the packaged helper. Pin adoption alone does not enable speech in
standard SDK binaries or qualify installed SDK applications. Only the Android
SDK's arm64-v8a library opts in; see [Android SDK](#android-sdk).

### Additional static platforms

Cargo and Bazel recognize the following pinned schema-5 targets. All use
the baseline ARM64 CPU. Enabling static `bindings` requires the exact verified
staged slice. Schema 4 and frozen release tags cannot supply these new slices. No x86 iOS simulator
or Mac Catalyst support is added.

| Rust target | Receipt Zig target | Slice | Minimum | System libraries |
| --- | --- | --- | --- | --- |
| `aarch64-apple-ios` | `aarch64-ios-none` | `ios-arm64` | iOS 16.0 | `System` |
| `aarch64-apple-ios-sim` | `aarch64-ios-simulator` | `ios-arm64-simulator` | iOS 16.0 | `System` |
| `aarch64-unknown-linux-gnu` | `aarch64-linux-gnu` | `linux-arm64` | glibc 2.28 | `c`, `m`, `pthread`, `dl` |

These requirements follow the engine's release slice contract. Pins and
receipts must preserve that CPU, minimum, system-library list and device or
simulator ABI. Bazel uses the existing iOS 16 configurations and pinned Apple
toolchains: its two ARM64 selects distinguish apple_support's
[`device` and `simulator` constraints](https://github.com/bazelbuild/apple_support/blob/2.2.0/constraints/BUILD),
as emitted by [rules_rs 0.0.95's pinned rules_rust mapping](https://github.com/hermeticbuild/rules_rust/blob/f19661ba71f77c993ce1eb082053692610fe4313/rust/platform/triple_mappings.bzl).
Linux ARM64 additionally requires the GNU libc constraint.

Bazel loads packages from unselected `select` arms. For only these three new
targets, a genuinely absent target row produces an inert repository whose
`zzz` rule fails when selected for analysis. Resolving that unused repository
does not require a staged slice. Existing targets always retain strict
verification, and any attempted new target row (including a malformed or
duplicate pin) takes the verifier path. A selected target cannot silently omit
the engine.

Outside the Android SDK's arm64-v8a library, native SDK Cargo/Bazel linking
and installed SDK runtime qualification remain required. Pin adoption does not
change defaults, activate standard binaries, or qualify minimum iOS/glibc
execution. Offline tests use synthetic archives and notices; actual published
archive staging was verified separately.

`PhononSettings::default()` reads the linked engine's defaults. Temperature `0`
selects the GGUF's default; a positive finite value overrides it. Seed `0` is
valid. Omitting the voice lets the engine select a bundled voice; this surface
does not load external voice files or clone voices. Keep mapped model files
unchanged until all session/cancellation handles are gone.

## Android SDK

The Android SDK (the `ai.xybrid:xybrid-kotlin` AAR, which React Native also
uses) is the only standard SDK binary that links the engine. Its release build
passes `--//:tts_zzz=true` and stages only the pinned `aarch64-linux-android`
slice, which is linked statically into the AAR's arm64-v8a
`libxybrid_bolt.so` (Android 10, API 29, the AAR's floor). Swift, Flutter,
Unity, Python and the CLI still ship without it.

- armeabi-v7a and x86_64 have no pinned slice. Under the flag they build
  without speech through `//bazel/zzz:android_without_slice`, so one flag-on
  AAR build covers every ABI. Every crate that forwards `tts-zzz` uses the same
  select with that group first; `tools/scripts/tests/test_zzz_feature_wiring.py`
  checks this per platform. Any other target without a slice (Windows, macOS
  x86_64, Linux musl, x86 Android) still fails as incompatible when the flag is
  set.
- The flag is target-scoped (`scope = "target"`), so build tools in the exec
  configuration never select the engine. A Bazel build therefore resolves only
  its target platform's slice, not the build host's.
- A custom flag-on `--config=android-arm64` build (the Flutter library or the
  Unity plugin) links the same slice, but with the API-28 toolchain those
  libraries use. The engine still needs Android 10, so such a build must not
  claim a lower floor.
- The fourteen notices travel in the AAR at `assets/xybrid/zzz/<path>`
  (`//bazel/zzz:notices.bzl`). Analysis fails unless the verified slice
  exports exactly those files. Android merges library assets into the APK, so
  every app that ships the SDK carries them.
