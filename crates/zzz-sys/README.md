# zzz-sys

[natives-manifest.json](natives-manifest.json) pins the zzz CPU binaries for
Phonon and Kitten TTS 2, including release versions, archive checksums, ABI requirements,
and linker metadata.

| Binary target | Compiled baseline |
| --- | --- |
| macOS ARM64 | macOS 13+ |
| Android ARM64 | Android API 29+ |
| Linux x86_64 | x86-64-v3 CPU, glibc 2.28+ |

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
overlapping symbols and cannot be linked together. Staging verifies all four
payload files, the exact model/API declarations and the archive's SHA-256
before either Cargo or Bazel links it. This SDK pins engine `v0.1.0-rc.8`
with receipt schema 4. The full `combined` engine is not accepted.

Receipt schema 4 is accepted only for that frozen `rc.8` release. For future
schema-5 pins, verification also requires the immutable 14-file notice
inventory, including runtime terms and upstream license texts. Each archive's
manifest entry must pin the exact schema-5 `notice_files` map; the receipt map
and every notice's bytes must match it before staging. Changing notice bytes
requires a new receipt schema. This verifier support does not change the
current engine pins or enable additional SDK targets or build features.

`PhononSettings::default()` reads the linked engine's defaults. Temperature `0`
selects the GGUF's default; a positive finite value overrides it. Seed `0` is
valid. Omitting the voice lets the engine select a bundled voice; this surface
does not load external voice files or clone voices. Keep mapped model files
unchanged until all session/cancellation handles are gone.
