# Xybrid Flutter SDK Example

## Quick Start

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

Future<void> main() async {
  // Initialize the SDK (call once at app startup). Runs locally with no
  // key; pass an apiKey to light up the dashboard:
  //   await Xybrid.init(apiKey: const String.fromEnvironment('XYBRID_API_KEY'));
  await Xybrid.init();

  // Load a TTS model from the registry
  final loader = XybridModelLoader.fromRegistry('kokoro-82m');
  final model = await loader.load();

  // Run text-to-speech inference
  final result = await model.run(XybridEnvelope.text('Hello from Xybrid!'));

  // Access the audio output
  if (result.audioBytes != null) {
    print('Generated ${result.audioBytes!.length} bytes of audio');
  }
}
```

## Loading with Progress

```dart
final loader = XybridModelLoader.fromRegistry('kokoro-82m');

await for (final event in loader.loadWithProgress()) {
  switch (event) {
    case LoadProgress(:final progress):
      print('Downloading: ${(progress * 100).toInt()}%');
    case LoadComplete():
      final model = await loader.load();
      print('Model ready!');
    case LoadError(:final message):
      print('Error: $message');
  }
}
```

## LLM Chat with Streaming

```dart
final model = await XybridModelLoader.fromRegistry('qwen-2.5-0.5b').load();
final context = ConversationContext();

// Stream tokens as they're generated
await for (final token in model.chat(
  'What is machine learning?',
  context: context,
)) {
  stdout.write(token);
}
```

## Full Example App

See the complete Flutter example app with 8 demo screens:
https://github.com/xybrid-ai/xybrid/tree/main/examples/flutter
