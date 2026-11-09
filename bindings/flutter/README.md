# Xybrid Flutter SDK

Run LLMs, ASR, and TTS natively in Flutter apps — private, offline, no cloud required.

[![pub package](https://img.shields.io/pub/v/xybrid_flutter.svg)](https://pub.dev/packages/xybrid_flutter)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation

```bash
flutter pub add xybrid_flutter
```

Or add to your `pubspec.yaml`:

```yaml
dependencies:
  xybrid_flutter: ^0.17.0
```

<details>
<summary>Alternative installation (git / local path)</summary>

**From git** (unreleased changes):

```yaml
dependencies:
  xybrid_flutter:
    git:
      url: https://github.com/xybrid-ai/xybrid.git
      ref: master
      path: bindings/flutter
```

</details>

## Quick Start

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Runs locally with no key. Configure observability explicitly to export.
  await Xybrid.init();

  // Load a TTS model from the registry
  final model = await XybridModelLoader.fromRegistry('kokoro-82m').load();

  // Run text-to-speech
  final result = await model.run(XybridEnvelope.text('Hello from Xybrid!'));
  print('Audio: ${result.audioBytes?.length} bytes');
}
```

Inference runs locally without an account. `apiKey` configures cloud authentication;
export is off until you explicitly configure observability.

## Observability

```dart
await Xybrid.init(
  apiKey: const String.fromEnvironment('XYBRID_API_KEY'), // optional cloud auth
  observability: ObservabilityConfig.xybrid(
    apiKey: const String.fromEnvironment('XYBRID_API_KEY'),
    metadataOnly: true,
  ),
);
// Or: ObservabilityConfig.otlp(endpoint: 'http://localhost:4318',
//   serviceName: 'my-app', metadataOnly: true)

final result = await model.run(
  XybridEnvelope.text('Hello'),
  options: const RunOptions(observation: ObservationOptions(
    conversationId: 'conversation-1',
    // Optional W3C traceparent/tracestate link this run to a caller's span.
  )),
);
final reference = result.observation; // null when untraced/suppressed
if (reference != null) {
  XybridObservability.recordFeedback(reference, score: 1.0);
}
await XybridObservability.flush(timeoutMs: 5000);
await XybridObservability.shutdown(timeoutMs: 5000);
```

Xybrid's preset exports traces only. A Collector receives traces, metrics, and
logs; feedback/evaluation returns `false` if logs are unavailable. Explicit export
configuration captures prompts, outputs, and tool payloads by default unless
`metadataOnly` is set or the native environment disables content. No observability
credentials are needed for offline inference. Export configuration is independent of cloud
authentication, and changing either does not change the other's key.
Legacy `ingestUrl` and `initTelemetry` calls preserve metadata-only export;
use explicit observability configuration to enable content capture.

The same `RunOptions.observation` works with `runWithContext`, `runStreaming`,
`runStreamingWithContext`, `runStreamingWithFallback`, and `runTtsStreaming`.
`suppressTrace: true` skips detailed spans while retaining aggregate metrics.
An invalid explicit `traceparent` is rejected before any work starts: `run`
and `XybridPipeline.run` throw `XybridException`, token streams end with an
error token, `runTtsStreaming` fails its stream with a `StateError`, and
`XybridStreamSession.fromModel`, `withObservation` and the manual `start*`
calls throw. Normal and fallback
streams merge `observation` and metrics into one terminal token, preserving its
native index and finish reason. Context streams retain their separate
metrics/completion token. Tool calls are delivered once.

Every traced surface takes observation controls and returns a reference
(`null` when untraced or suppressed):

| Surface | Controls | Reference |
|---|---|---|
| `model.run(...)` and token streams | `RunOptions.observation` | `result.observation`; the terminal `StreamToken.observation` |
| Model load | `XybridModelLoader.withObservation(options)` | `model.loadObservationRef` |
| Pipeline | `pipeline.run(envelope, observation: options)` | `result.observation` (the workflow); `result.metrics.stageLatenciesMs[i].observation` (each stage's inference) |
| Live ASR | `XybridStreamSession.fromModel(model, observation: options)` | `(await session.flush()).observation` (the session); `partial.decoderObservation` (each decoder run) |
| Streamed TTS | `runTtsStreaming(..., options: RunOptions(observation: options))` | `onComplete: (result) => result.observation` |

```dart
final pipeline = XybridPipeline.fromYaml(yaml);
final run = await pipeline.run(
  XybridEnvelope.audio(bytes: audioBytes, sampleRate: 16000),
  observation: const ObservationOptions(conversationId: 'conversation-1'),
);
final workflow = run.observation;
final stages = run.metrics.stageLatenciesMs.map((stage) => stage.observation);
```

`XybridObservability.init(config)` replaces an exporter and propagates config
errors; corrected config can be retried. `shutdown` disables native exporter
state and is idempotent; a later `init` restarts export. Flush/shutdown run off
the UI isolate, have bounded timeouts, and surface native lifecycle errors. Counters
and state are available through `XybridObservability.stats`.

Migration: `Xybrid.init(apiKey: ...)` now configures authentication only. Pass
`observability:` to enable export. The explicit legacy
`Xybrid.init(apiKey: ..., ingestUrl: ...)` and `Xybrid.initTelemetry` still start
a real OTLP trace exporter: bare URLs map to `/v1/traces`, full endpoint paths
are preserved. Existing custom receivers must accept OTLP instead of legacy
platform events. Legacy initialization now replaces prior config (the obsolete
process-wide once-guard is gone); `isTelemetryInitialized` reports actual native
state, including shutdown. Dedicated `initTelemetry` preserves the current cloud
key; combined `init(apiKey: ..., ingestUrl: ...)` configures both with the supplied
key. `resourceTelemetry` keeps configuring CPU/memory
sampling independently of export (`off`, `boundary`, `summary[:ms]`,
`debug_local[:ms]`). Invalid modes and malformed intervals now fail before
export is enabled or replaced.

Migration (breaking): `XybridStreamSession.flush()` returns an
`FfiTranscriptionResult` instead of a `String`. Read `.text` for the
transcript; `durationMs`, `chunksProcessed` and `observation` come with it.
`FfiTtsStreamEvent.complete` now carries an `FfiTtsStreamResult` (status,
sample rate, channels, sample and chunk counts, `observation`).
`runTtsStreaming` passes it to the new `onComplete` callback and, when the
status is `limited`, fails the stream with a `StateError` after the partial
audio. A `traceparent` that is not valid W3C now fails the call instead of
being ignored.

### Manual operations

Record work Xybrid does not run itself (a tool, another runtime's inference, a
model load) with a manual handle. `context()` returns the W3C context to pass
to children, `observationRef()` the reference for feedback. The first terminal
call (`complete`, `fail` or `cancel`) wins. Disposing an unfinished handle
cancels it without flushing.

```dart
final workflow = await XybridObservability.startWorkflow('answer-question',
    options: const ObservationOptions(conversationId: 'conversation-1'));
final parent = await workflow.context(); // null when untraced or suppressed
final child = parent == null
    ? null
    : ObservationOptions(
        traceparent: parent.traceparent, tracestate: parent.tracestate);

final tool = await XybridObservability.startTool(
  const ToolCallInfo(name: 'search', callId: 'call-1', argumentsJson: '{"q":"rivers"}'),
  options: child,
);
try {
  await tool.recordResult(resultJson: '{"hits":3}');
  await tool.complete();
} catch (error) {
  await tool.fail(errorType: 'tool_error', message: '$error');
}

final answer = await model.run(
  XybridEnvelope.text('Name three rivers.'),
  options: RunOptions(observation: child),
);
await workflow.complete();
tool.dispose();
workflow.dispose();
```

`XybridObservability.startInference(InferenceRequest(...))` records another
runtime's inference (`recordInputMessages`, `recordOutputMessages`,
`recordUsage`, `recordFirstToken`, `recordChunk`, `recordFinishReasons`), and
`startModelLoad(ModelLoadInfo(model: ...))` a load (`recordCacheHit`). Content
arguments are JSON strings; they go through the same capture mode, redaction
rules and size budgets as the SDK's own content.

## Features

### Model Loading

Load models from the Xybrid registry or local bundles:

```dart
// From registry (downloads + caches automatically)
final model = await XybridModelLoader.fromRegistry('kokoro-82m').load();

// From local bundle
final model = await XybridModelLoader.fromBundle('path/to/model.xyb').load();

// Check if already cached
if (Xybrid.isModelCached('kokoro-82m')) {
  print('Model ready, no download needed');
}
```

### Download Progress

Track model downloads with progress events:

```dart
final loader = XybridModelLoader.fromRegistry('kokoro-82m');

await for (final event in loader.loadWithProgress()) {
  switch (event) {
    case LoadProgress(:final progress, :final downloadedBytes, :final totalBytes):
      print('Downloading: ${(progress * 100).toInt()}% '
          '($downloadedBytes / ${totalBytes ?? '?'} bytes)');
    case LoadComplete():
      print('Model ready!');
    case LoadError(:final message):
      print('Error: $message');
  }
}
```

`progress` spans every file the model needs and never moves backwards.
`totalBytes` is `null` when the source publishes no size; `downloadedBytes` is
exact either way, so megabytes, speed and time remaining are all derivable.

### Input Envelopes

Type-safe inputs for different model types:

```dart
// Text (for TTS or LLM)
final textInput = XybridEnvelope.text('Hello world');

// Text with TTS voice selection
final ttsInput = XybridEnvelope.text('Hello', voiceId: 'af_heart', speed: 1.0);

// Audio (for ASR / speech-to-text)
final audioInput = XybridEnvelope.audio(
  bytes: wavBytes,
  sampleRate: 16000,
  channels: 1,
);

// Embedding vector
final embeddingInput = XybridEnvelope.embedding([0.1, 0.2, 0.3]);

// Vision-language prompt with an encoded image
final image = XybridEnvelope.image(bytes: pngBytes, format: 'png');
final visionInput = XybridEnvelope.userMessage(
  text: 'Describe this image.',
  images: [image],
);
```

### Inference Results

```dart
final result = await model.run(envelope);

if (result.success) {
  // Text output (ASR transcription or LLM response)
  print(result.text);

  // Audio output (TTS) — get as WAV for playback
  final wav = result.audioAsWav(sampleRate: 24000, channels: 1);

  // Embedding output
  print(result.embedding);

  // Inference timing
  print('Latency: ${result.latencyMs}ms');
}
```

### Reasoning (thinking models)

Reasoning models (metadata `reasoning: true`, e.g. `lfm2.5-1.2b-thinking`)
produce a chain-of-thought before their answer. Xybrid keeps it out of the
answer text and surfaces it on `reasoningContent` — `null` for non-thinking
models. Nothing to enable; just read it if you want it.

```dart
final result = await model.run(XybridEnvelope.text(
    'Is 97 a prime number? Reason, then answer.'));

if (result.text != null) print('Answer: ${result.text}');
if (result.reasoningContent != null) print('Reasoning: ${result.reasoningContent}');
```

### Structured Output (JSON Schema)

Constrain a local llama model so its output is always schema-valid — no
retry loop, no parse failures. `jsonSchemaToGbnf` turns a JSON Schema into the
GBNF grammar `GenerationConfig.grammar` expects:

```dart
final grammar = jsonSchemaToGbnf(
  schemaJson: '{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}',
);

final result = await model.run(
  XybridEnvelope.text('Extract the city: "I flew into Paris last night."'),
  config: GenerationConfig.greedy(grammar: grammar),
);
// result.text is guaranteed to parse against the schema
```

`GenerationConfig.greedy` is the usual pairing — deterministic decoding plus a
grammar is the standard extraction shape. Raw GBNF works too: pass it to
`grammar` directly.

### Tool Calling

One `run` is one model turn, so the loop lives in your code: run a request
carrying tools, execute the calls the model asks for, then run a continuation
envelope that feeds the outcomes back.

```dart
final tools = [
  ToolDefinition(
    name: 'get_weather',
    description: 'Current weather for a city',
    parametersJson: '{"type":"object","properties":{"city":{"type":"string"}}}',
  ),
];
final config = GenerationConfig.greedy(tools: tools);

const question = 'Weather in Paris?';
final first = await model.run(XybridEnvelope.text(question), config: config);

if (first.hasToolCalls) {
  final results = [
    for (final call in first.toolCalls)
      ToolResult(callId: call.id, name: call.name, contentJson: runTool(call)),
  ];

  final answer = await model.run(
    XybridEnvelope.toolResults(
      userText: question,
      priorAssistantText: first.text!,  // raw output, tool-call block included
      results: results,
    ),
    config: config,  // same tools as the first turn
  );
}
```

Two rules the loop depends on: run the continuation with the **same tools** as
the original turn so the executor rebuilds an identical chat prefix, and pass
`priorAssistantText` **verbatim** — tool-call block included.

Gate your tool UI on the bundle's advisory flag:

```dart
if (model.supportsToolCalling == true) showToolsToggle();
```

`null` means the bundle says nothing and never implies support. Tool calling is
llama.cpp-only; unsupported paths reject tool-bearing requests rather than
silently generating without them.

### LLM Streaming

Stream tokens in real-time as the LLM generates:

```dart
final model = await XybridModelLoader.fromRegistry('qwen-2.5-0.5b').load();
print('Token streaming: ${model.supportsTokenStreaming()}');

await for (final token in model.runStreaming(XybridEnvelope.text('What is ML?'))) {
  stdout.write(token.token);

  if (token.isFinal) {
    print('\n--- Done (${token.finishReason}) ---');
  }
}
```

### Conversation Memory

Multi-turn LLM conversations with automatic history management:

```dart
final model = await XybridModelLoader.fromRegistry('qwen-2.5-0.5b').load();
final context = ConversationContext();
context.setSystem('You are a helpful assistant.');

// Turn 1
context.pushText('What is Rust?', MessageRole.user);
final result = await model.runWithContext(
  XybridEnvelope.text('What is Rust?'),
  context,
);
context.pushText(result.text!, MessageRole.assistant);

// Turn 2 — the model remembers Turn 1
context.pushText('How does it compare to Go?', MessageRole.user);
final result2 = await model.runWithContext(
  XybridEnvelope.text('How does it compare to Go?'),
  context,
);
```

Streaming with context also supported:

```dart
await for (final token in model.runStreamingWithContext(envelope, context)) {
  stdout.write(token.token);
}
```

## Platform Support

| Platform | ONNX Runtime | Candle | LLM (llama.cpp) | MLX | Notes |
|----------|:---:|:---:|:---:|:---:|-------|
| **macOS** | ✅ | ✅ Metal | ✅ | ✅ | Apple Silicon only (M1+); MLX auto-selected for supported LLMs when `llm-mlx-runtime` is enabled |
| **iOS** | ✅ CoreML | ✅ Metal | ✅ | Metadata only | arm64; MLX runtime remains staged until upstream ships a Metal-enabled iOS slice |
| **Android** | ✅ | — | ✅ | — | arm64-v8a, x86_64; ORT from Maven Central |
| **Linux** | ✅ | ✅ CPU | ✅ | — | x86_64 |
| **Windows** | ✅ | ✅ CPU | ✅ | — | x86_64 |

### MLX on Apple Silicon

MLX is a third LLM + embedding backend for `aarch64-apple-darwin`. iOS keeps the non-linking MLX skeleton for metadata and registry parity, but real MLX inference is not treated as device-ready until upstream ships a Metal-enabled iOS slice. **No API changes are needed** — `Xybrid.model(id).load()` and `XybridModelLoader.fromRegistry(id).load()` automatically route to MLX when the model has an `mlx` registry variant, the build includes `llm-mlx-runtime`, and the host is Apple Silicon macOS. On non-Apple platforms, Intel Macs, and current iOS builds, calls fall through to llama.cpp or the registry default with no change in behaviour.

Supported MLX models today: Qwen 3 (end-to-end chat + streaming), Gemma 4 and LFM 2 / 2.5 (text generation with env-gated real-bundle smoke and informational benchmark rows), LFM 3.5 (synthetic coverage only until a public text-generation SafeTensors fixture is available), and BERT / nomic-bert (embeddings).

To force a specific backend per pipeline stage:

```yaml
stages:
  - model: qwen3-4b
    backend: mlx          # "auto" (default) | "mlx" | "llamacpp"
```

See [`docs/backends/mlx.md`](../../docs/backends/mlx.md) in the main repo for the full selection rules, xcframework setup, and iOS OOM troubleshooting.

### Model Support

| Model | Type | All Platforms |
|-------|------|:---:|
| Kokoro 82M | TTS | ✅ |
| KittenTTS Nano | TTS | ✅ |
| Whisper Tiny (Candle) | ASR | ✅ |
| Wav2Vec2 (ONNX) | ASR | ✅ |
| SmolLM2 360M | LLM | ✅ |
| Qwen 2.5 0.5B | LLM | ✅ |
| Qwen 3.5 0.8B | LLM | ✅ |
| Qwen 3.5 2B | LLM | ✅ |
| Gemma 3 1B | LLM | ✅ |
| Llama 3.2 1B | LLM | ✅ |

### Platform Requirements

- **macOS**: 14.0+, Xcode 15+, Apple Silicon
- **iOS**: 16.0+, Xcode 15+
- **Android**: Android 9 (API 28)+, NDK r25+, 64-bit only
- **Linux/Windows**: x86_64

The precompiled libraries are built for these versions, and the plugin
declares them, so an app that targets an older one fails at build time:
`pod install` rejects the plugin, or the Android manifest merge fails. Raise
your app's minimums:

- **Android**: `minSdk = 28` in `defaultConfig` of
  `android/app/build.gradle.kts` (Flutter's default, `flutter.minSdkVersion`,
  is lower).
- **iOS**: `platform :ios, '16.0'` in `ios/Podfile`, and an iOS Deployment
  Target of 16.0 for the Runner target in Xcode.
- **macOS**: `platform :osx, '14.0'` in `macos/Podfile`, and a macOS
  Deployment Target of 14.0 for the Runner target in Xcode.

## Native Libraries

Native ML runtimes are resolved automatically at build time:

- **Android**: ONNX Runtime pulled from Maven Central (`com.microsoft.onnxruntime:onnxruntime-android`)
- **iOS**: ONNX Runtime is part of the precompiled library, so nothing extra is downloaded or installed. (Monorepo source builds fetch an ONNX Runtime xcframework from HuggingFace into `~/.xybrid/cache/ort-ios/`; simulator source builds also need `xz`.)
- **macOS/Linux/Windows**: ONNX Runtime downloaded by the `ort` Rust crate at compile time

The Rust library itself ships as a precompiled, signature-verified binary for
every supported platform via [cargokit](https://github.com/nicklocking/cargokit),
downloaded at build time. No Rust toolchain is required, and having one
installed does not change anything — the published package is precompiled-only
and cannot be built from source, because its Rust crate lives in the xybrid
monorepo workspace.

The first build of an app downloads that binary, gzip-compressed: roughly
10 MB per Android ABI and 30–50 MB for iOS and macOS, where it is a static
library of over 100 MB once unpacked. A slow first build is this download, not
a Rust compile.

The download is kept in a shared cache at `~/.xybrid/cache/precompiled/`, so
`flutter clean` and other projects on the same machine reuse it instead of
downloading again. Every reuse re-verifies the binary's signature against the
key pinned in this package, and entries unused for 90 days are removed. Set
`XYBRID_PRECOMPILED_CACHE_DIR` to move the cache — for example into a directory
your CI persists between runs — or to an empty value to turn it off.

Flutter hides native build-step output unless you pass `-v`; with it (or in
the Xcode / Android Studio build log) cargokit reports what it is doing:

```
INFO: Downloading precompiled aarch64-apple-ios_libxybrid_flutter_ffi.a.gz (33.6 MB) from https://github.com/…
INFO: aarch64-apple-ios_libxybrid_flutter_ffi.a.gz: 14.2 MB of 33.6 MB (42%)
INFO: Downloaded aarch64-apple-ios_libxybrid_flutter_ffi.a.gz: 33.6 MB in 21s (1.6 MB/s)
INFO: Unpacked aarch64-apple-ios_libxybrid_flutter_ffi.a.gz to 119.2 MB
INFO: Using precompiled xybrid_flutter for aarch64-apple-ios (downloaded)
```

Later builds print `(cached)`; after `flutter clean`, or in another project,
`(shared cache)`. A line starting
`Building xybrid_flutter for` means a source build, which only happens inside
the monorepo.

Building from source is for monorepo development, where the workspace root and
the `xybrid-*` crates are present. There it is the default whenever a Rust
toolchain is installed, because cargokit's crate hash only covers
`bindings/flutter/rust` — a precompiled binary would not pick up edits to
`xybrid-core`, `xybrid-sdk` or `xybrid-ffi-facade`.

Either default can be overridden with a `cargokit_options.yaml` next to your
app's `pubspec.yaml`:

```yaml
use_precompiled_binaries: false
```

## Example App

A full example app with 8 demo screens (TTS, ASR, LLM chat, pipelines, device info) is available:

```sh
https://github.com/xybrid-ai/xybrid/tree/master/examples/flutter
```

## API Reference

| Class | Purpose |
|-------|---------|
| `Xybrid` | SDK initialization, cache checking, factory methods |
| `XybridModelLoader` | Load models from registry or local bundle |
| `XybridModel` | Run inference (batch, streaming, with context) and inspect token streaming support |
| `XybridEnvelope` | Type-safe inputs: audio, text, embedding |
| `XybridResult` | Inference output: text, audio, embedding, latency |
| `StreamToken` | Individual LLM token during streaming |
| `ConversationContext` | Multi-turn conversation history with FIFO pruning |
| `XybridPipeline` | Multi-stage pipeline execution from YAML |
| `MessageRole` | Enum: `system`, `user`, `assistant` |
| `LoadEvent` | Download progress events: `LoadProgress`, `LoadComplete`, `LoadError` |

Full API documentation: [pub.dev/documentation/xybrid_flutter](https://pub.dev/documentation/xybrid_flutter/latest/)

## Telemetry

The plugin reports `binding=flutter` in a small `X-Xybrid-Client` header attached to registry metadata calls. See [docs/telemetry/registry.md](../../docs/telemetry/registry.md) for the exact wire format and the opt-out switch (`XYBRID_TELEMETRY_OPTOUT=1`).

## License

Apache 2.0 — see [LICENSE](LICENSE)
