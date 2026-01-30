# Xybrid Apple Binding (Swift)

> **Status**: Placeholder - Future Implementation

This directory will contain the Swift package for Xybrid, providing native iOS and macOS support via UniFFI-generated bindings.

## Planned Structure

```
apple/
├── Package.swift             # Swift Package manifest
├── Sources/Xybrid/           # Generated Swift code (from uniffi-bindgen)
├── Xybrid.podspec           # Optional CocoaPods support
└── XCFrameworks/            # Prebuilt binaries
    └── XybridFFI.xcframework
```

## FFI Strategy

The Swift bindings will be generated from `crates/xybrid-uniffi/` using UniFFI:
- Single Rust source generates both Swift and Kotlin
- Native async/await support
- Memory-safe wrappers

## Distribution

- **SwiftPM**: Primary distribution via Swift Package Manager
- **CocoaPods**: Optional for legacy projects

## Supported Platforms

| Platform | Minimum Version |
|----------|-----------------|
| iOS | 13.0 |
| macOS | 10.15 |

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
