# react-native-xybrid

React Native binding for the Xybrid SDK. Wraps the same UniFFI-generated
Swift/Kotlin surface that powers the standalone iOS and Android SDKs, exposed
to JavaScript through a TurboModule.

## Status

**Spike / pre-release.** Surface covers loader → `load()` → `run()` plus voice
introspection and platform-state push. Streaming (ASR partials, LLM token
streams) is not yet wired through — see "Open work" below.

## Architecture

```
JS / TS
  └── react-native-xybrid (this package)
        ├── ios/      Swift TurboModule → bundled Xybrid.swift wrapper → XybridFFI.xcframework
        └── android/  Kotlin TurboModule → bundled ai.xybrid.* wrapper → libxybrid_uniffi.so + ORT
                                                          └── xybrid-uniffi (Rust)
                                                                └── xybrid-sdk (Rust)
                                                                      └── xybrid-core (Rust)
```

Native artifacts are staged by `cargo xtask build-react-native` from the
existing per-platform builds (`build-xcframework`, `build-android`). No new
Rust code — the bridge is purely a thin layer above the UniFFI bindings.

## Layout

```
bindings/react-native/
├── package.json             # npm + RN codegen config
├── react-native-xybrid.podspec
├── src/
│   ├── index.ts             # Public TS facade (Xybrid, ModelLoader, Model)
│   ├── NativeXybrid.ts      # TurboModule spec (codegen input)
│   └── types.ts
├── ios/
│   ├── XybridModule.{h,mm}  # ObjC++ TurboModule registration
│   ├── XybridModuleImpl.swift  # Actual work, calls bundled Xybrid.swift
│   ├── XybridSwift/         # ← staged by xtask: Xybrid.swift + xybrid_uniffi.swift
│   └── Frameworks/          # ← staged by xtask: XybridFFI.xcframework
└── android/
    ├── build.gradle
    ├── libs/{abi}/          # ← staged by xtask: libxybrid_uniffi.so + ORT libs
    └── src/main/java/ai/xybrid/
        ├── Xybrid.kt        # ← staged by xtask: copy of bindings/kotlin Xybrid.kt
        ├── xybrid_uniffi.kt # ← staged by xtask: UniFFI generated Kotlin
        └── reactnative/
            ├── XybridModule.kt
            └── XybridPackage.kt
```

The staged paths are gitignored — they're regenerated from the Rust core
on every build, and shipped vendored inside the npm tarball.

## Local development

```bash
# 1. Build native artifacts (XCFramework on macOS, .so on Linux/macOS)
cargo xtask build-react-native --release

# 2. Use a yarn link or relative path in a sample app
cd ../my-sample-rn-app
yarn add ../yangon-v1/bindings/react-native
cd ios && pod install && cd ..

# 3. Wrap the app entry
import { Xybrid, ModelLoader } from 'react-native-xybrid';

await Xybrid.initialize();
const model = await ModelLoader.fromRegistry('whisper-tiny').load();
const result = await model.run({ kind: 'audio', bytesBase64, sampleRate: 16000, channels: 1 });
console.log(result.text);
await model.release();
```

## Requirements

- React Native ≥ 0.74 (TurboModules + codegen).
- iOS 13+, Android API 24+ (matches xybrid-kotlin and xybrid-apple).
- **Apple Silicon Mac for iOS development.** The staged XCFramework
  intentionally omits `ios-x86_64-simulator` and `macos-x86_64` slices —
  ort-sys ships no prebuilt ONNX Runtime for Intel Mac / Intel iOS
  Simulator, so the podspec excludes those archs explicitly. Apps built
  for real iOS devices (arm64) work everywhere; only the simulator
  workflow is constrained.
- New Architecture enabled (`newArchEnabled=true` in `gradle.properties`,
  `RCT_NEW_ARCH_ENABLED=1` in the iOS Podfile env).

## Open work for GA

1. **Streaming.** ASR partial results and LLM token streams currently terminate
   at the UniFFI boundary's poll-based `XybridStream`. Surfacing them to JS
   needs an `EventEmitter` (legacy) or a JSI `HostObject` wrapper for low-jitter
   token delivery. The Rust side already exposes everything required.
2. **Binary payloads.** Audio bytes ride as base64 strings today. Move to
   `ArrayBuffer` via JSI to drop the encode/decode hop on every chunk.
3. **TypeScript codegen.** The `Spec` interface is hand-written; once the
   surface stabilizes, generate `NativeXybrid.ts` from the same source the
   UniFFI UDL is derived from to keep all four bindings in lockstep.
4. **Example app + automated smoke test.** No `example/` directory yet.
   The CI workflow builds and lints the package but does not yet run a
   sample app end-to-end.
