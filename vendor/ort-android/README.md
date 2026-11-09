# ORT Android Shared Libraries

This directory contains the ONNX Runtime Android shared libraries used for on-device inference.

## Current Version

- **Version**: 1.23.x (matches iOS vendor — see `vendor/ort-ios/README.md`)
- **Source**: Copied from Flutter jniLibs (`bindings/flutter/android/src/main/jniLibs/`)

## Contents

```
ort-android/
├── arm64-v8a/
│   └── libonnxruntime.so    # 18 MB — ORT shared library (ARM aarch64)
└── x86_64/
    └── libonnxruntime.so    # 22 MB — ORT shared library (x86-64)
```

No C++ runtime is vendored here. Where an Android artifact needs
`libc++_shared.so`, it comes from the NDK that linked it: the Kotlin AAR takes
it from `@androidndk`'s sysroot, and the Flutter plugin's Gradle build copies it
from the NDK. The Unity Android library links libc++ statically and ships none.

## Usage

These libraries are referenced by:
- symlinks in `bindings/flutter/android/src/main/jniLibs/{abi}/` — Flutter SDK
- the `//:ort_android_*` filegroups — the Bazel AAR
  (`bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar`) bundles them into
  its `jni/<abi>/` payload
- `tools/scripts/stage_unity_native.py` and `cargo xtask deploy-unity-native`,
  which stage them beside the Unity Android library

## Updating the Libraries

1. Download the new ORT Android release from [GitHub onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
2. Extract `libonnxruntime.so` for each ABI (`arm64-v8a`, `x86_64`)
3. Replace the files in this directory
4. Update the symlinks in `bindings/flutter/` if paths change
5. Update this README with the new version number
6. Test with `bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar`
