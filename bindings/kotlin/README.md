# Xybrid Kotlin Binding (Android)

> **Status**: Placeholder - Future Implementation

This directory will contain the Android library for Xybrid, providing native Kotlin/Java support via UniFFI-generated bindings.

## Planned Structure

```
kotlin/
├── build.gradle.kts          # Gradle build configuration
├── src/main/kotlin/ai/xybrid/  # Generated Kotlin code (from uniffi-bindgen)
└── libs/                     # Prebuilt native libraries
    ├── armeabi-v7a/
    │   └── libxybrid_uniffi.so
    ├── arm64-v8a/
    │   └── libxybrid_uniffi.so
    └── x86_64/
        └── libxybrid_uniffi.so
```

## FFI Strategy

The Kotlin bindings will be generated from `crates/xybrid-uniffi/` using UniFFI:
- Single Rust source generates both Swift and Kotlin
- Kotlin coroutines support for async operations
- Memory-safe wrappers

## Distribution

- **Maven Central**: Primary distribution as an Android library (.aar)

## Supported Android Versions

| Android API | Version Name |
|-------------|--------------|
| API 24+ | Android 7.0 (Nougat) |

## NDK ABIs

| Architecture | ABI |
|--------------|-----|
| ARMv7 | armeabi-v7a |
| ARM64 | arm64-v8a |
| x86_64 | x86_64 |

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
