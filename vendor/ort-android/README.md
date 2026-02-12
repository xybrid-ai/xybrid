# ORT Android Shared Libraries

This directory contains the ONNX Runtime Android shared libraries used for on-device inference.

## Current Version

- **Version**: 1.23.x (matches iOS vendor — see `vendor/ort-ios/README.md`)
- **Source**: Copied from Flutter jniLibs (`bindings/flutter/android/src/main/jniLibs/`)

## Contents

```
ort-android/
├── arm64-v8a/
│   ├── libonnxruntime.so    # 18 MB — ORT shared library (ARM aarch64)
│   └── libc++_shared.so     # 1.7 MB — C++ standard library
└── x86_64/
    ├── libonnxruntime.so    # 22 MB — ORT shared library (x86-64)
    └── libc++_shared.so     # 1.7 MB — C++ standard library
```

## Usage

These libraries are referenced by symlinks from:
- `bindings/flutter/android/src/main/jniLibs/{abi}/` — Flutter SDK
- `bindings/kotlin/libs/{abi}/` — Kotlin SDK

The `cargo xtask build-android` command also copies these libraries to the Kotlin SDK output.

## Updating the Libraries

1. Download the new ORT Android release from [GitHub onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
2. Extract `libonnxruntime.so` for each ABI (`arm64-v8a`, `x86_64`)
3. Replace the files in this directory
4. Update symlinks in `bindings/flutter/` and `bindings/kotlin/` if paths change
5. Update this README with the new version number
6. Test with `cargo xtask build-android`
