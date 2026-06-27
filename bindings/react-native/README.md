# react-native-xybrid

React Native binding for the Xybrid SDK. Wraps the same BoltFFI-generated
Swift/Kotlin surface that powers the standalone iOS and Android SDKs, exposed
to JavaScript through a TurboModule.

## Status

**Pre-release.** The synchronous surface is at 1:1 parity with the Apple and
Kotlin SDKs: loader → `run()` with full `RunOptions` (sampling config plus
cloud fallback / abort-on-stress / correlation ID), `warmup`/`unload`,
`GenerationConfigs` presets, voice introspection, and platform-state push.
Streaming (ASR partials, LLM token streams) is the remaining gap — see
"Open work" below.

## Architecture

```
JS / TS
  └── react-native-xybrid (this package)
        ├── ios/      Swift TurboModule → bundled Xybrid.swift wrapper → XybridFFI.xcframework
        └── android/  Kotlin TurboModule → ai.xybrid:xybrid-kotlin AAR (Maven; bundles .so + ORT)
                                                          └── xybrid-bolt (Rust, BoltFFI)
                                                                └── xybrid-ffi-facade (Rust)
                                                                      └── xybrid-sdk → xybrid-core (Rust)
```

The two platforms consume the bolt core differently:

- **iOS** vendors the `XybridFFI.xcframework` + the bolt Swift wrapper sources,
  staged into this package by `cargo xtask stage-react-native` (from the same
  `build-xcframework` output the standalone Apple SDK uses).
- **Android** depends on the published `ai.xybrid:xybrid-kotlin` Maven AAR,
  which bundles `libxybrid-bolt.so` + the ONNX Runtime alongside the
  `ai.xybrid.*` Kotlin classes. Nothing is staged per-package.

No new Rust code — the bridge is purely a thin layer above the bolt bindings.

## Layout

```
bindings/react-native/
├── package.json             # npm + RN codegen config
├── react-native-xybrid.podspec
├── src/
│   ├── index.ts             # Public TS facade (Xybrid, ModelLoader, Model)
│   ├── NativeXybrid.ts      # TurboModule spec (codegen input)
│   ├── presets.ts           # GenerationConfigs.greedy() / .creative()
│   └── types.ts
├── ios/
│   ├── XybridModule.{h,mm}  # ObjC++ TurboModule registration
│   ├── XybridModuleImpl.swift  # Actual work, calls bundled Xybrid.swift
│   ├── XybridSwift/         # ← staged by xtask: Xybrid.swift + xybrid_bolt.swift
│   └── Frameworks/          # ← staged by xtask: XybridFFI.xcframework
└── android/
    ├── build.gradle         # depends on ai.xybrid:xybrid-kotlin (Maven AAR)
    └── src/main/java/ai/xybrid/reactnative/
        ├── XybridModule.kt  # Kotlin TurboModule → ai.xybrid.* (from the AAR)
        └── XybridPackage.kt
```

The staged iOS paths are gitignored — they're regenerated from the Rust core
on every build and shipped vendored inside the npm tarball. Android pulls its
binding + natives from Maven, so there is nothing to stage there.

## Local development

```bash
# 1. Stage the iOS native artifacts (XCFramework + Swift wrapper). macOS only.
#    Android needs nothing — gradle resolves the Maven AAR.
cargo xtask stage-react-native --release

# 2. Use a yarn link or relative path in a sample app
cd ../my-sample-rn-app
yarn add ../xybrid/bindings/react-native
cd ios && pod install && cd ..

# 3. Wrap the app entry
import { Xybrid, ModelLoader } from 'react-native-xybrid';

await Xybrid.initialize();
const model = await ModelLoader.fromRegistry('whisper-tiny').load();
const result = await model.run({ kind: 'audio', bytesBase64, sampleRate: 16000, channels: 1 });
console.log(result.text);
await model.release();
```

> The JS `ModelLoader.fromRegistry(id).load()` facade is preserved for API
> stability even though the native bolt layer collapsed the loader into the
> `XybridModel` factories — `index.ts` maps the old shape onto the new calls.

### Run options, warmup/unload, presets

`run()` takes a `RunOptions` second argument mirroring the bolt
`XybridRunOptions` the Apple/Kotlin SDKs expose — sampling config plus the
platform-plane knobs. A bare `GenerationConfig` is still accepted as shorthand
for `{ generationConfig }`.

```ts
import { ModelLoader, GenerationConfigs } from 'react-native-xybrid';

const model = await ModelLoader.fromRegistry('llama-3.2-1b').load();

// Optional: prime the model so first-token latency is inference, not cold start.
await model.warmup();

const result = await model.run(
  { kind: 'text', text: 'Write a haiku about the sea.' },
  {
    generationConfig: GenerationConfigs.creative(),
    fallbackToCloud: true,                 // allow cloud under device stress
    abortOn: ['thermalCritical'],          // bail early if the device overheats
    maxGraceTokens: 16,
    correlationId: 'req-42',               // threaded into telemetry
  },
);
console.log(result.text);

// Shed weights under memory pressure; the handle stays valid and reloads on next run.
await model.unload();
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

1. **Streaming.** ASR partial results and LLM token streams aren't surfaced to
   JS yet — they need an `EventEmitter` (legacy) or a JSI `HostObject` wrapper
   for low-jitter token delivery. This is the last synchronous-surface parity
   gap with the native SDKs.
2. **Binary payloads.** Audio bytes ride as base64 strings today. Move to
   `ArrayBuffer` via JSI to drop the encode/decode hop on every chunk.
3. **TypeScript codegen.** The `Spec` interface and the native shim mappers are
   hand-written, so RN is the one binding not generated from the bolt
   `#[data]`/facade source of truth — every new core field must be hand-wired
   here (as `RunOptions` / `warmup` / `unload` just were). Generate them from
   the same definitions the other bindings derive from to keep parity
   structural rather than a manual chase. See the JSI re-architecture plan.
4. **End-to-end smoke test.** The `example/` Expo app and CI build/lint the
   package, but CI does not yet run inference end-to-end on a device/emulator.
