# zzz-sys

[natives-manifest.json](natives-manifest.json) pins the zzz CPU binaries for
Kitten TTS 2, including release versions, archive checksums, ABI requirements,
and linker metadata.

| Binary target | Compiled baseline |
| --- | --- |
| macOS ARM64 | macOS 13+ |
| Android ARM64 | Android API 29+ |
| Linux x86_64 | x86-64-v3 CPU, glibc 2.28+ |

This directory currently contains only binary pins. Downloading, Rust bindings,
and native linking will be added during integration. Model weights are supplied
separately.
