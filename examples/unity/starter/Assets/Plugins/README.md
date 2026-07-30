# Native Plugins

This directory contains platform-specific native libraries for the Xybrid SDK.

## Building Native Libraries

Before the SDK will work, you need to build the native libraries from the Rust source:

```bash
# From the repository root
cd repos/xybrid

# Build + stage per platform
# Host platform — builds and stages into the Unity package (editor testing)
cargo xtask build-ffi --deploy-unity

# Device targets mirror .github/workflows/build-unity.yml: build the bolt
# native with Bazel, then stage it (with its Unity .meta) into
# bindings/unity/Runtime/Plugins/<Platform>/ — no manual copying:
bazel build --config=ios //crates/xybrid-bolt:xybrid_bolt_staticlib               # iOS
bazel build --config=macos-metal -c opt //crates/xybrid-bolt:xybrid_bolt_cdylib   # macOS
bazel build --config=android-arm64 //crates/xybrid-bolt:xybrid_bolt_cdylib        # Android (also android-armv7, android-x86_64)
python3 tools/scripts/stage_unity_native.py --lib <bazelisk cquery --output=files path> --target <triple>
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
