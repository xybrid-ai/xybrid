/// SDK initialization for Xybrid.
///
/// This provides the main entry point for initializing the Xybrid runtime.
library;

import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:xybrid_flutter/src/rust/api/sdk_client.dart';
import 'package:path_provider/path_provider.dart';
import 'rust/frb_generated.dart';

import '../xybrid.dart';

/// Main entry point for the Xybrid SDK.
///
/// Call [Xybrid.init] once before using any other Xybrid functionality.
///
/// ```dart
/// void main() async {
///   await Xybrid.init();
///
///   // Now you can use Xybrid
///   final loader = XybridModelLoader.fromRegistry('kokoro-82m');
///   final model = await loader.load();
///   // ...
/// }
/// ```
class Xybrid {
  static bool _initialized = false;
  static final Completer<void> _initCompleter = Completer<void>();
  static bool _initializing = false;

  /// Private constructor to prevent instantiation.
  Xybrid._();

  /// Initialize the Xybrid runtime.
  ///
  /// This must be called once before using any Xybrid functionality.
  /// It is safe to call this multiple times - subsequent calls are no-ops.
  ///
  /// Example:
  /// ```dart
  /// void main() async {
  ///   await Xybrid.init();
  ///   // SDK is ready to use
  /// }
  /// ```
  ///
  /// Throws an exception if initialization fails (e.g., native library not found).
  static Future<void> init() async {
    // Fast path: already initialized
    if (_initialized) {
      return;
    }

    // Handle concurrent initialization attempts
    if (_initializing) {
      return _initCompleter.future;
    }

    _initializing = true;

    try {
      // On iOS and macOS, we use static linking with -force_load.
      // The Rust symbols are linked directly into the main executable,
      // so we must use DynamicLibrary.process() to look them up.
      ExternalLibrary? externalLibrary;
      if (Platform.isIOS || Platform.isMacOS) {
        externalLibrary = ExternalLibrary.process(iKnowHowToUseIt: true);
      }

      await XybridRustLib.init(externalLibrary: externalLibrary);

      if (Platform.isAndroid) {
        final appDir = await getApplicationSupportDirectory();
        final cacheDir = '${appDir.path}/xybrid/models';
        XybridSdkClient.initSdkCacheDir(cacheDir: cacheDir);
      }

      _initialized = true;
      _initCompleter.complete();
    } catch (e) {
      _initCompleter.completeError(e);
      _initializing = false;
      rethrow;
    }
  }

  /// Check if the SDK has been initialized.
  ///
  /// Returns `true` if [init] has been called successfully.
  static bool get isInitialized => _initialized;

  static void setApiKey(String apiKey) {
    XybridSdkClient.setApiKey(apiKey: apiKey);
  }

  static void initTelemetry() {
    // TODO - Implement telemetry
    // XybridSdkClient.enableTelemetry();
    throw UnimplementedError();
  }

  /// Create a ModelLoader for the specified model.
  ///
  /// This is the entry point for the **Loader → Model → Run** pattern.
  ///
  /// ## From Registry (recommended for production)
  /// ```dart
  /// final loader = Xybrid.model(modelId: 'whisper-tiny');
  /// final model = await loader.load();
  /// ```
  static XybridModelLoader model(String modelId) =>
      XybridModelLoader.fromRegistry(modelId);

  /// Create a PipelineRef for multi-stage inference pipelines.
  ///
  /// Pipelines orchestrate multiple models in sequence (e.g., ASR → LLM → TTS).
  ///
  /// ## From YAML Content
  /// ```dart
  /// final yaml = '''
  /// name: "Voice Assistant"
  /// stages:
  ///   - whisper-tiny
  ///   - llm-stage
  ///   - kokoro-tts
  /// ''';
  /// final ref = Xybrid.pipeline(yaml: yaml);
  /// final pipeline = await ref.load();
  /// ```
  ///
  /// ## From File
  /// ```dart
  /// final ref = Xybrid.pipeline(filePath: 'pipelines/voice-assistant.yaml');
  /// ```
  static XybridPipeline pipeline({String? yaml, String? filePath}) {
    final hasYaml = yaml != null;
    final hasFile = filePath != null;

    if (!hasYaml && !hasFile) {
      throw ArgumentError('Must provide either yaml or filePath');
    }

    if (hasYaml && hasFile) {
      throw ArgumentError('Only one source can be specified: yaml or filePath');
    }

    if (hasYaml) {
      return XybridPipeline.fromYaml(yaml);
    }

    return XybridPipeline.fromFile(filePath!);
  }
}
