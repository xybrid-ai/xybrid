# ORT iOS Static Library

This directory contains the ONNX Runtime iOS static library used for on-device inference.

## Current Version

- **Version**: 1.23.2
- **Execution Provider**: CoreML EP included
- **Source**: [HuggingFace csukuangfj/ios-onnxruntime](https://huggingface.co/csukuangfj/ios-onnxruntime)

## Contents

```
onnxruntime.xcframework/
└── ios-arm64/
    ├── libonnxruntime.a    # Static library for iOS arm64
    └── Headers/            # C/C++ headers
```

## Usage

The library is automatically detected by xtask commands that build for iOS targets. The `resolve_ort_lib_location()` function checks this location when `ORT_LIB_LOCATION` is not set.

## Updating the Library

1. Download the new version from [HuggingFace](https://huggingface.co/csukuangfj/ios-onnxruntime)
2. Extract and replace the contents of `onnxruntime.xcframework/`
3. Update this README with the new version number
4. Test with `cargo xtask build-xcframework`

## Symlinks

For Flutter compatibility, a symlink exists at:
- `bindings/flutter/ios/Frameworks/onnxruntime.xcframework` -> `../../../../vendor/ort-ios/onnxruntime.xcframework`

This allows the Flutter podspec to reference the framework without changing its vendored_frameworks path.
