# Xybrid Unity Binding (C#)

> **Status**: Placeholder - Future Implementation

This directory will contain the Unity package for Xybrid, providing C# bindings for Unity game development via the C FFI from `crates/xybrid-ffi/`.

## Planned Structure

```
unity/
├── package.json              # Unity package manifest
├── Runtime/                  # C# scripts
│   └── Xybrid.cs            # P/Invoke wrappers
└── Plugins/                  # Prebuilt native libraries
    ├── iOS/
    │   └── libxybrid.a
    ├── Android/
    │   └── libxybrid.so
    ├── macOS/
    │   └── libxybrid.dylib
    └── Windows/
        └── xybrid.dll
```

## FFI Strategy

Unity uses C#'s P/Invoke for native interop. The bindings will call into the C FFI exposed by `crates/xybrid-ffi/`:
- C header: `crates/xybrid-ffi/include/xybrid.h`
- Synchronous and callback-based async patterns
- Memory managed via explicit allocation/deallocation

## Distribution

- **Unity Asset Store**: Primary distribution as a Unity package
- **OpenUPM**: Alternative package registry
- **Manual**: Direct import via `package.json`

## Supported Platforms

| Platform | Architecture |
|----------|--------------|
| iOS | arm64 |
| Android | armeabi-v7a, arm64-v8a |
| macOS | arm64, x86_64 |
| Windows | x86_64 |

## Unity Versions

| Unity Version | Status |
|---------------|--------|
| 2021.3 LTS | Planned |
| 2022.3 LTS | Planned |
| 2023.x | Planned |

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
