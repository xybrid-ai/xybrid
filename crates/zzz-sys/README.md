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
the required CPU floor. ABI `0` means the runtime is unavailable; it does not
mean the pinned artifact failed ABI verification. Native defaults return
zeroed settings with `struct_size` set in this case; opening a session checks
availability before validating settings. Other target slices retain their
existing behavior. The Rust host must itself be built for its intended CPU
baseline; this check cannot make a host binary compiled for v3 run on v2.

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
through the packaged helper. Pin adoption does not enable speech in standard
SDK binaries or qualify installed SDK applications.

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

Native SDK Cargo/Bazel linking and installed SDK runtime qualification remain
required. Pin adoption does not change defaults, activate standard binaries,
or qualify minimum iOS/glibc execution. Offline tests use synthetic archives
and notices; actual published archive staging was verified separately.

`PhononSettings::default()` reads the linked engine's defaults. Temperature `0`
selects the GGUF's default; a positive finite value overrides it. Seed `0` is
valid. Omitting the voice lets the engine select a bundled voice; this surface
does not load external voice files or clone voices. Keep mapped model files
unchanged until all session/cancellation handles are gone.


## Optional Android companion client (inert preparation)

The additive `android-companion` feature exposes the same safe sessions through
an SDK-owned table instead of requiring a static engine archive. **No production
companion contract is configured.** This feature currently advertises no models,
returns ABI `0`, and rejects opens with `LoadFailed`; it does not enable speech in
any shipped SDK. Defaults and the static `bindings` contract remain unchanged by this client. When both features are selected, static `bindings` take precedence.
The Bazel-only opt-in client target is `:xybrid_zzz_companion_client`; it has no
private archive dependency and is not activated in standard SDK bindings.

Core, SDK and Bolt also expose the off-by-default Cargo forwarder
`tts-zzz-android-companion`. It enables only this client transport, using the
same existing run/audio-streaming methods and template adapters. It does not
enable static `tts-zzz` or `bindings`, and it currently advertises neither
`tts-zzz` nor `zzz-phonon-31m`: the production identity remains unconfigured.
Availability completes the lazy platform/table/identity validation before the
core's capability list is cached. Unsupported or unavailable components reject
execution with the loader's bounded reason while other backends remain usable.

For Bazel the separate flag is `--//:tts_zzz_android_companion=true`; defaults
and standard package targets remain unchanged. Core selects exactly one
zzz-sys dependency: the client for companion-only mode, or the original static
target when `--//:tts_zzz=true` is also set. SDK and Bolt mirror that selection.
Cargo feature unification likewise retains static bindings precedence. Flutter
forwarding, production identity/pins, package inclusion and installed-device
qualification remain separate work.

The private ABI-1 getter contract in `src/companion.rs` takes a requested ABI,
exact output size and caller-owned C-layout table. Its fixed header contains
size, companion ABI, engine ABI, Android API minimum, selected-model bits,
reserved zero and a 32-byte independently expected contract identity. Eight
nullable C function slots cover ABI/model/default probes, the two opens,
synthesis, cancellation and close. All metadata and slots must match before
native probes; ABI 1 and generic Phonon support alone cannot establish the public
Phonon-31m contract. A future reviewed adoption must supply the actual qualified,
notice-bearing engine identity; this preparation fabricates none.

Only Android ARM64 has a native loader. Other OS/ABI combinations return explicit
unavailable reasons. Before opening the fixed `libxybrid_zzz_tts.so` SONAME with
local visibility, the loader dynamically resolves bionic's actual-device API
query and requires API 29+. The API-29 query is not a strong linker import;
unknown/older OS versions cannot attempt the companion load. Getter lookup uses
that owned library handle. No caller path, authentication, model directory,
network fetch or environment activation is accepted. The unconfigured identity
fails before any OS query or library operation.

`availability()` returns a bounded typed `UnavailableReason`; probes/defaults
and opens share the same lazy initialization. Success and failure are permanent
for the process, so an optional library installed later requires app restart.
The process cache retains a successful library reference. Every opened session
and cancellation owner also retains its exact table/library, and final close
runs before that ownership is released. Existing Linux v3 checks still precede
all eight native operations, including cancel/close. Companion loading, symbol
imports, package minima and actual Android execution remain unqualified until
future platform/package work and installed-app tests; mock tests prove dispatch,
validation, ordering and lifetime only.
