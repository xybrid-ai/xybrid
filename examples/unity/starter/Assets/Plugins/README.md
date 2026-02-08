# Native Plugins

This directory contains platform-specific native libraries for the Xybrid SDK.

## Building Native Libraries

Before the SDK will work, you need to build the native libraries from the Rust source:

```bash
# From the repository root
cd repos/xybrid

# Build for all platforms
cargo xtask build-xcframework   # macOS/iOS
cargo xtask build-android       # Android
```

## Plugin Structure

After building, copy the native libraries to their respective directories:

```
Assets/Plugins/
├── iOS/
│   └── libxybrid.a           # iOS static library
├── Android/
│   ├── arm64-v8a/
│   │   └── libxybrid.so
│   └── armeabi-v7a/
│       └── libxybrid.so
├── macOS/
│   └── libxybrid.dylib       # macOS dynamic library
└── Windows/
    └── xybrid.dll            # Windows dynamic library (future)
```

## Alternative: Symlink to SDK Plugins

You can also symlink to the SDK's Plugins directory:

```bash
# From examples/unity/Assets
rm -rf Plugins
ln -s ../../../bindings/unity/Plugins Plugins
```

Note: Unity may need to re-import after creating symlinks.
