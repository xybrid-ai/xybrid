# react-native-xybrid

React Native binding for the Xybrid SDK. Wraps the same BoltFFI-generated
Swift/Kotlin surface that powers the standalone iOS and Android SDKs, exposed
to JavaScript through a TurboModule.

## Status

**Pre-release.** At 1:1 parity with the Apple and Kotlin SDKs: loader →
`run()` with full `RunOptions` (sampling config plus cloud fallback /
abort-on-stress / correlation ID), `warmup`/`unload`, `GenerationConfigs`
presets, voice introspection, platform-state push, and **token streaming** via
`model.runStreaming()` (pull-based, aborts on break/completion).

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
  staged into this package from the Bazel build (the same
  `//bindings/apple:XybridFFI` target the standalone Apple SDK ships from —
  see Local development below for the commands).
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
bazel build --config=ios //bindings/apple:XybridFFI
rm -rf ios/Frameworks/XybridFFI.xcframework ios/XybridSwift
mkdir -p ios/Frameworks ios/XybridSwift
unzip -o -q ../../bazel-bin/bindings/apple/XybridFFI.xcframework.zip -d ios/Frameworks
cp ../apple/Sources/Xybrid/{Xybrid.swift,xybrid_bolt.swift} ios/XybridSwift/

# 2. Use a yarn link or relative path in a sample app
cd ../my-sample-rn-app
yarn add ../xybrid/bindings/react-native
cd ios && pod install && cd ..

# 3. Wrap the app entry
import { Xybrid, ModelLoader } from 'react-native-xybrid';

await Xybrid.initialize();
const model = await ModelLoader.fromRegistry('whisper-tiny-ggml').load();
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

### Structured output (JSON Schema / GBNF)

Constrain a local LLM to schema-valid JSON by setting
`generationConfig.grammar` — produce a grammar from a JSON Schema with
`jsonSchemaToGbnf()` (the same native converter every other binding uses), or
pass raw GBNF. Local llama backend only.

```ts
import { jsonSchemaToGbnf } from 'react-native-xybrid';

const grammar = await jsonSchemaToGbnf({
  type: 'object',
  properties: { name: { type: 'string' }, total: { type: 'number' } },
  required: ['name', 'total'],
});
const result = await model.run(
  { kind: 'text', text: 'Extract: 2x espresso, 8.40 EUR total' },
  { generationConfig: { grammar, maxTokens: 128 } },
);
JSON.parse(result.text!); // guaranteed schema-valid
```

### Streaming

`model.runStreaming()` returns an async generator that yields each token as it
is generated. It is pull-based: the underlying native run is **aborted
automatically** when iteration ends — it completes, you `break`, or an error is
thrown (each of these runs the generator's cleanup) — so stopping a generation
early never keeps the device busy. Unmounting a component does **not** stop a
running `for await` loop by itself: break out of the loop (or call
`gen.return()`) on unmount. A generator that is simply abandoned mid-stream is
never released until its model is. It takes the same `RunOptions` second
argument as `run()`.

```ts
for await (const token of model.runStreaming(
  { kind: 'text', text: 'Write a haiku about the sea.' },
  { generationConfig: GenerationConfigs.creative() },
)) {
  setOutput((prev) => prev + token.token); // token.cumulativeText also available
}
```

The final `InferenceResult` (latency, metrics) is the generator's return value;
non-LLM models emit a single token carrying the full result, then complete.
Errors raised mid-stream throw from the loop with the same typed `xybrid_*`
codes as `run()`.

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

1. **Binary payloads.** Audio bytes ride as base64 strings today. Move to
   `ArrayBuffer` via JSI to drop the encode/decode hop on every chunk. This is
   also where a *push* (`HostObject`) streaming path would land for high-rate
   binary; the current token streaming is pull-based, which is right for text.
2. **TypeScript codegen.** The `Spec` interface and the native shim mappers are
   hand-written, so RN is the one binding not generated from the bolt
   `#[data]`/facade source of truth — every new core field must be hand-wired
   here (as `RunOptions` / `warmup` / `unload` just were). Generate them from
   the same definitions the other bindings derive from to keep parity
   structural rather than a manual chase. See the JSI re-architecture plan.
3. **End-to-end smoke test.** The `example/` Expo app and CI build/lint the
   package, but CI does not yet run inference end-to-end on a device/emulator.
