/// Model loading API for Xybrid.
///
/// This class wraps the FRB-generated [FfiModelLoader] and [FfiModel]
/// with a clean, idiomatic Dart API.
library;

import 'dart:async';

import 'envelope.dart';
import 'llm.dart';
import 'result.dart';
import 'rust/api/model.dart';

/// Exception thrown when Xybrid operations fail.
class XybridException implements Exception {
  /// The error message.
  final String message;

  /// Creates a new [XybridException] with the given [message].
  XybridException(this.message);

  @override
  String toString() => 'XybridException: $message';
}

/// Event emitted during model loading with progress tracking.
sealed class LoadEvent {
  const LoadEvent._();
}

/// Download progress update (0.0 to 1.0).
class LoadProgress extends LoadEvent {
  /// Progress value from 0.0 to 1.0
  final double progress;

  const LoadProgress(this.progress) : super._();

  /// Progress as a percentage (0-100).
  int get percentage => (progress * 100).round();
}

/// Model loading completed successfully.
class LoadComplete extends LoadEvent {
  const LoadComplete() : super._();
}

/// Model loading failed with an error.
class LoadError extends LoadEvent {
  /// The error message.
  final String message;

  const LoadError(this.message) : super._();
}

/// Prepares a model for loading from registry or local bundle.
class XybridModelLoader {
  /// The underlying FRB model loader.
  final FfiModelLoader _inner;

  XybridModelLoader._(this._inner);

  /// Create a loader for a model from the Xybrid registry.
  ///
  /// The [modelId] should match a model ID in the registry (e.g., "kokoro-82m").
  factory XybridModelLoader.fromRegistry(String modelId) {
    return XybridModelLoader._(
      FfiModelLoader.fromRegistry(modelId: modelId),
    );
  }

  /// Create a loader for a model from a local bundle path.
  ///
  /// The [path] should point to a directory containing model_metadata.json.
  /// Throws if the bundle is invalid.
  factory XybridModelLoader.fromBundle(String path) {
    return XybridModelLoader._(
      FfiModelLoader.fromBundle(path: path),
    );
  }

  /// Load the model asynchronously.
  ///
  /// Downloads the model if loading from registry and not cached.
  /// Returns a ready-to-use [XybridModel] instance.
  Future<XybridModel> load() async {
    final ffiModel = await _inner.load();
    return XybridModel._(ffiModel);
  }

  /// Load the model with download progress updates.
  ///
  /// Returns a stream of [LoadEvent]:
  /// - [LoadProgress] with download progress (0.0 to 1.0)
  /// - [LoadComplete] when the model is ready
  /// - [LoadError] if loading fails
  ///
  /// After receiving [LoadComplete], call [load] to get the cached model.
  ///
  /// Example:
  /// ```dart
  /// final loader = Xybrid.model(modelId: 'kokoro-82m');
  /// await for (final event in loader.loadWithProgress()) {
  ///   switch (event) {
  ///     case LoadProgress(:final progress):
  ///       print('Downloading: ${(progress * 100).toInt()}%');
  ///     case LoadComplete():
  ///       final model = await loader.load();
  ///       print('Model ready!');
  ///     case LoadError(:final message):
  ///       print('Error: $message');
  ///   }
  /// }
  /// ```
  Stream<LoadEvent> loadWithProgress() {
    return _inner.loadWithProgress().map((ffiEvent) {
      return switch (ffiEvent) {
        FfiLoadEvent_Progress(:final field0) => LoadProgress(field0),
        FfiLoadEvent_Complete() => const LoadComplete(),
        FfiLoadEvent_Error(:final field0) => LoadError(field0),
      };
    });
  }
}

/// A loaded model ready for inference.
class XybridModel {
  /// The underlying FRB model.
  final FfiModel inner;

  XybridModel._(this.inner);

  /// Run inference with the given envelope.
  ///
  /// Returns [XybridResult] containing output text, audio, or embeddings
  /// depending on the model type.
  ///
  /// Throws [XybridException] if inference fails.
  Future<XybridResult> run(XybridEnvelope envelope) async {
    try {
      final ffiResult = await inner.run(envelope: envelope.inner);
      return XybridResult.fromFfi(ffiResult);
    } catch (e) {
      throw XybridException('Inference failed: $e');
    }
  }

  /// Run inference with streaming output.
  ///
  /// This method runs batch inference and then emits tokens progressively,
  /// simulating a streaming experience. Returns a [Stream] of [StreamToken].
  ///
  /// Each [StreamToken] contains the generated token text and metadata.
  /// The stream completes when generation finishes.
  ///
  /// # Arguments
  /// * [envelope] - The input envelope (typically text for LLMs)
  ///
  /// # Example
  /// ```dart
  /// final stream = model.runStreaming(XybridEnvelope.text('Tell me a story'));
  /// await for (final token in stream) {
  ///   stdout.write(token.token);
  ///   if (token.isFinal) {
  ///     print('\n\nDone: ${token.finishReason}');
  ///   }
  /// }
  /// ```
  Stream<StreamToken> runStreaming(XybridEnvelope envelope) async* {
    try {
      // Run batch inference
      final result = await run(envelope);
      final fullText = result.text;

      if (fullText == null || fullText.isEmpty) {
        // Empty response
        yield StreamToken(
          token: '',
          index: 0,
          cumulativeText: '',
          isFinal: true,
          finishReason: 'stop',
        );
        return;
      }

      // Split into words and emit progressively
      // Use regex to preserve whitespace in output
      final pattern = RegExp(r'(\S+)(\s*)');
      final matches = pattern.allMatches(fullText).toList();

      if (matches.isEmpty) {
        yield StreamToken(
          token: '',
          index: 0,
          cumulativeText: '',
          isFinal: true,
          finishReason: 'stop',
        );
        return;
      }

      var cumulative = '';
      var tokenIndex = 0;

      for (var i = 0; i < matches.length; i++) {
        final match = matches[i];
        final word = match.group(1) ?? '';
        final trailing = match.group(2) ?? '';
        final chunk = word + trailing;

        // Build cumulative text
        cumulative += chunk;

        // Emit token
        final isLast = i == matches.length - 1;
        yield StreamToken(
          token: chunk,
          index: tokenIndex++,
          cumulativeText: cumulative.trimRight(),
          isFinal: isLast,
          finishReason: isLast ? 'stop' : null,
        );

        // Small delay for streaming effect
        if (!isLast) {
          await Future.delayed(const Duration(milliseconds: 20));
        }
      }
    } catch (e) {
      // Emit error token
      yield StreamToken(
        token: '',
        index: 0,
        cumulativeText: '',
        isFinal: true,
        finishReason: 'error: $e',
      );
    }
  }
}
