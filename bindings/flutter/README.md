# Xybrid Flutter Binding

> **Status**: Placeholder

This directory will contain the Flutter plugin for Xybrid, providing Dart bindings via flutter_rust_bridge (FRB).

## Current Location

The Flutter SDK currently lives in a separate repository:
- **Repository**: [xybrid-flutter](https://github.com/xybrid-ai/xybrid-flutter)

## Planned Structure

```
flutter/
├── pubspec.yaml          # Flutter plugin manifest
├── lib/                  # Dart API wrappers
├── rust/                 # FRB bridge (Rust code)
│   ├── Cargo.toml        # depends on xybrid-sdk
│   └── src/api/          # Thin #[frb] wrappers (~100-150 LOC)
├── ios/
├── android/
├── macos/
├── windows/
└── linux/
```

## Migration Plan

As part of the monorepo restructure, the Flutter binding will be migrated here with:
1. Rust business logic moved to `crates/xybrid-sdk/`
2. Flutter-specific code kept thin (FRB wrappers only)
3. Platform-specific code in respective directories

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
