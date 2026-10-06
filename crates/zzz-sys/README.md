# zzz-sys

[natives-manifest.json](natives-manifest.json) pins the zzz CPU binaries for
Kitten TTS 2, including release versions, archive checksums, ABI requirements,
and linker metadata.

| Binary target | Compiled baseline |
| --- | --- |
| macOS ARM64 | macOS 13+ |
| Android ARM64 | Android API 29+ |
| Linux x86_64 | x86-64-v3 CPU, glibc 2.28+ |

Privately stage a pinned slice before native builds with
`python3 tools/scripts/zzz_pull.py --target <target>` from the repository root;
see `--help` for authentication, offline archives and staged verification.
The tool prints the verified slice path for `XYBRID_ZZZ_PREBUILT_DIR`. Keep its
archives, receipts and staging directories private. Model weights are supplied
separately.

## Rust bindings

`xybrid-zzz-sys` carries the ABI-1 FFI for `libzzz_embed.a` plus a thin safe
surface (`KittenSession`, typed `ZzzError`s). Default builds are a no-op shell;
enabling `bindings` links the pinned slice for the build target through the
same `zzz_pull.py` verification (cache hits are reverified; `XYBRID_ZZZ_USE_GH_AUTH`
opts local builds into stored gh credentials for activation) and fails the
build when the target is unsupported or the pinned slice is missing or fails
verification.
