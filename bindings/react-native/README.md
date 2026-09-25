# @xybrid/react-native

On-device AI for React Native — LLMs, speech recognition, text-to-speech,
embeddings and vision — backed by the same Rust SDK as Xybrid's Swift, Kotlin,
Flutter and Unity bindings. Runs offline with no account; add an API key to
light up cloud fallback and telemetry.

```ts
import { Envelope, ModelLoader } from '@xybrid/react-native';

const model = await ModelLoader.fromRegistry('qwen3.5-0.8b').load();
const result = await model.run(Envelope.text('Write a haiku about the sea.'));
console.log(result.text);
await model.release();
```

> **Status: preview.** The API mirrors the Swift and Kotlin SDKs and is
> covered by tests on every change, but it may still move before 1.0.

## Requirements

| | |
|---|---|
| React Native | 0.76 or newer, **New Architecture** (the default since 0.76; the only option since 0.82) |
| iOS | 16.0+. Simulator builds need an Apple Silicon Mac (there is no x86_64 simulator slice) |
| Android | API 24+, `arm64-v8a`, `armeabi-v7a`, `x86_64` |
| Expo | SDK 52+ with a [development build](https://docs.expo.dev/develop/development-builds/introduction/) — not Expo Go |

## Install

```sh
npm install @xybrid/react-native
cd ios && pod install
```

- **iOS** — `pod install` downloads the Rust core (`XybridFFI.xcframework`)
  from this version's GitHub Release, checks it against the SHA-256 pinned in
  the package, and caches it in `~/.xybrid/cache/xcframework`. Behind a proxy or
  offline, point `XYBRID_NATIVES_BASE_URL` at a mirror, or
  `XYBRID_XCFRAMEWORK_PATH` at a local copy (directory or `.zip`). If your app
  targets iOS below 16.0, raise it in the Podfile (`platform :ios, '16.0'`).
- **Android** — nothing to do: Gradle pulls `ai.xybrid:xybrid-kotlin` (the
  Kotlin SDK with the native libraries) from Maven Central.
- **Expo** — `npx expo install @xybrid/react-native`, then
  `npx expo prebuild` / `npx expo run:ios|android`. No config plugin needed.

## Quick tour

### Configure (optional)

Local inference needs no setup. Call `initialize` first only when you have
options — an API key enables the platform features on top of the same local
runtime: cloud fallback, speculative cloud serving and dashboard telemetry.

```ts
import { Xybrid } from '@xybrid/react-native';

await Xybrid.initialize({ apiKey: XYBRID_API_KEY });
```

Options apply once per app process; calling again with different options
rejects with `xybrid_config_error` (restart the app to change them).

### Load and run

```ts
import { Envelope, GenerationConfigs, ModelLoader } from '@xybrid/react-native';

const model = await ModelLoader.fromRegistry('qwen3.5-0.8b').load();
await model.warmup(); // optional: pay cold start now, not on the first run

const result = await model.run(Envelope.text('Summarise: …'), {
  generationConfig: GenerationConfigs.greedy({ maxTokens: 128 }),
});
result.text;              // the answer
result.reasoningContent;  // <think> text, if the model emitted any
result.metrics;           // totalMs, ttftMs, tokensPerSecond, …
result.executionTarget;   // 'local' or 'cloud'
```

Other sources: `ModelLoader.fromBundle(path)`, `fromDirectory(path)`,
`fromHuggingFace('org/repo', { revision })`, `fromModelFile('model.gguf')`
(`file://` URLs work too). `model.info()` returns the model id, output type
and capabilities; `model.unload()` frees the weights but keeps the handle;
`model.release()` frees everything.

### Stream tokens, with a stop button

```ts
const controller = new AbortController();

for await (const token of model.runStreaming(Envelope.text('Tell me a story'), {
  signal: controller.signal,
})) {
  setText((previous) => previous + token.token);
}
// Elsewhere (a Stop button, unmount): controller.abort();
```

Generation stops natively when the loop ends — completion, `break`, an
error, or `abort()`. An aborted run rejects with `xybrid_cancelled`. A loop that
is simply abandoned is never cleaned up (JavaScript runs no `finally` on
garbage collection), so abort on unmount. The final `InferenceResult` is the
generator's return value.

### Conversations

```ts
import { ConversationContext, Envelope } from '@xybrid/react-native';

const chat = await ConversationContext.create();
await chat.setSystem('You are concise.');

const question = Envelope.user('What is the capital of France?');
const reply = await model.run(question, { context: chat });
await chat.push(question);       // the run never changes the context:
await chat.push(reply.envelope); // push both turns yourself, after the run
```

### Structured output

```ts
import { jsonSchemaToGbnf } from '@xybrid/react-native';

const grammar = await jsonSchemaToGbnf({
  type: 'object',
  properties: { name: { type: 'string' }, total: { type: 'number' } },
  required: ['name', 'total'],
});
const result = await model.run(Envelope.text('Extract: 2x espresso, 8.40 EUR'), {
  generationConfig: { grammar, maxTokens: 128 },
});
JSON.parse(result.text!);
```

### Tool calling

One `run` is one model turn; the loop is yours.

```ts
const tools = [{ name: 'weather', description: 'Current weather', parameters: { type: 'object', properties: { city: { type: 'string' } } } }];
const prompt = 'Weather in Paris?';

const first = await model.run(Envelope.text(prompt), { generationConfig: { tools } });
if (first.toolCalls.length > 0) {
  const results = await Promise.all(first.toolCalls.map(async (call) => ({
    callId: call.id,
    name: call.name,
    content: await runTool(call.name, JSON.parse(call.argumentsJson)),
  })));
  const next = await Envelope.toolResults(prompt, first.text ?? '', results);
  const answer = await model.run(next, { generationConfig: { tools } });
}
```

When streaming, the terminal token carries `toolCalls` and `rawText` (use
`rawText` as `priorAssistantText`).

### Vision

```ts
const message = Envelope.userMessage('What is in this photo?', [
  Envelope.image(photoBase64, 'jpeg'),
]);
const result = await visionModel.run(message);
```

### Text-to-speech

```ts
const tts = await ModelLoader.fromRegistry('kokoro-82m').load();
const voices = await tts.voices();
const speech = await tts.run(Envelope.text('Hello!', { voiceId: voices[0].id, speed: 1.1 }));
speech.audioBytesBase64; // WAV/PCM, base64
```

### Speech recognition

Batch — transcribe a recording:

```ts
const asr = await ModelLoader.fromRegistry('whisper-tiny-ggml').load();
const { text } = await asr.run(Envelope.audio(wavBase64)); // 16 kHz mono by default
```

Live — feed microphone audio as it arrives (Float32, mono, 16 kHz):

```ts
const session = await asr.stream({ language: 'en' });
(async () => {
  for await (const partial of session.partials()) setCaption(partial.text);
})();
await session.feed(float32Samples);       // from your audio callback
const transcript = await session.flush(); // when the user stops talking
await session.release();
```

### Pipelines

```ts
import { Pipeline } from '@xybrid/react-native';

const pipeline = await Pipeline.fromFile(`${documentDirectory}voice-assistant.yaml`);
const result = await pipeline.run(Envelope.audio(recordingBase64));
const heard = result.stages.find((stage) => stage.stageId === 'asr')?.text;
play(result.audioBytesBase64);
```

### Downloads, speculative cloud, cache and memory

```ts
// Download with a progress bar, then load instantly from cache.
const loader = ModelLoader.fromRegistry('qwen3.5-0.8b');
const download = await loader.download();
for await (const status of download!.progress()) setProgress(status.progress);
const model = await loader.load();

// Answer from the cloud while the weights download (needs an API key).
const fast = await ModelLoader.fromRegistrySpeculative('qwen3.5-0.8b').load();
await fast.isCloudServing(); // true until the local weights land

await Xybrid.modelCacheStatus();       // bytes and models on disk
await Xybrid.removeCachedModel('qwen3.5-0.8b');
Xybrid.releaseMemoryOnWarning();       // free idle models on OS memory warnings
```

### Errors

Every rejection has a stable `code` (`XybridErrorCodes` lists them):

```ts
import { isRetryable, isXybridError } from '@xybrid/react-native';

try {
  await model.run(envelope);
} catch (error) {
  if (isXybridError(error) && error.code === 'xybrid_model_not_found') { /* … */ }
  if (isRetryable(error)) { /* network, rate limit, timeout, offline */ }
}
```

`xybrid_handle` means the object was released; `xybrid_invalid_argument`
means a malformed argument (bad envelope, options, base64).

## How it works

```
JS / TS (src/) ── TurboModule (src/NativeXybrid.ts, Codegen)
  ├── iOS      ios/XybridModule.mm → XybridModuleImpl.swift → Swift SDK (Xybrid.swift) ─┐
  └── Android  XybridModule.kt → Kotlin SDK (ai.xybrid:xybrid-kotlin)                 ─┤
                                                                   xybrid-bolt (BoltFFI)
                                                                   → xybrid-ffi-facade → xybrid-sdk → xybrid-core
```

- Native objects (models, contexts, streams, sessions…) are opaque handles in
  one registry per platform; `release()` frees them, and a model's release
  also stops the streams started from it.
- Envelopes cross as a payload plus string metadata, exactly like the Rust
  type; convenience fields (`voiceId`, `sampleRate`, `role`…) are folded in on
  the JS side (`src/wire.ts`), so both native shims stay generic.
- Binary payloads (audio, images) cross as base64 strings: TurboModules have no
  zero-copy buffer type. Fine for prompts, clips and TTS output; a JSI buffer
  path is on the roadmap for high-rate audio.

## Developing in this repo

```sh
npm install && npm test          # types, unit tests, spec conformance, SDK parity
npm run test:ios                 # Swift type-check + codec tests (macOS, Xcode)
npm run test:android             # Kotlin codec tests on the JVM (kotlinc + java)
```

Run the example app (`example/`) against the working tree:

```sh
# iOS: build the Rust core; `pod install` picks up bazel-bin automatically.
bazel build --config=ios //bindings/apple:XybridFFI
cd example && npm install && npx expo run:ios

# Android: build the AAR, publish the Kotlin SDK to mavenLocal (the example
# prefers mavenLocal), then run.
bazel build -c opt --config=remote //bindings/kotlin:xybrid_kotlin_aar
#   stage its jni/* into bindings/kotlin/libs/ — see bindings/kotlin/README.md
(cd ../kotlin && ./gradlew publishToMavenLocal)
cd example && npm install && npx expo run:android
```

The example runs a smoke test on launch and logs each step with a
`[xybrid-smoke]` prefix.

**Adding SDK surface.** `tests/parity.test.mjs` reads every export of
`crates/xybrid-bolt/src/lib.rs` and fails until each new function, method,
field, enum variant or error variant is mapped in `parity.json` (or excluded
with a reason). To map one: add the method to `src/NativeXybrid.ts`, implement
it in `ios/XybridModuleImpl.swift` (+ its forwarder in `ios/XybridModule.mm`)
and `android/.../XybridModule.kt`, expose it in `src/`, and record it in
`parity.json`. `tests/spec-conformance.test.mjs` checks the iOS selectors and
Kotlin overrides against the spec.
