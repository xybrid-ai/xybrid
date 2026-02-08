/// Model loading API for Xybrid.
///
/// This class wraps the FRB-generated [FfiModelLoader] and [FfiModel]
/// with a clean, idiomatic Dart API.
library;

import 'dart:async';

import 'context.dart';
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

  /// Run inference with conversation context.
  ///
  /// The context provides conversation history which is formatted into
  /// the prompt using the model's chat template.
  ///
  /// **Note:** The context is NOT automatically updated with the result.
  /// Call `context.pushText(result.text, MessageRole.assistant)` to add
  /// the response to the history.
  ///
  /// ## Example
  ///
  /// ```dart
  /// final context = ConversationContext();
  /// context.setSystem('You are a helpful assistant.');
  ///
  /// context.pushText('Hello!', MessageRole.user);
  /// final result = await model.runWithContext(
  ///   XybridEnvelope.text('Hello!'),
  ///   context,
  /// );
  /// context.pushText(result.text ?? '', MessageRole.assistant);
  /// ```
  Future<XybridResult> runWithContext(
    XybridEnvelope envelope,
    ConversationContext context,
  ) async {
    try {
      final ffiResult = await inner.runWithContext(
        envelope: envelope.inner,
        context: context.inner,
      );
      return XybridResult.fromFfi(ffiResult);
    } catch (e) {
      throw XybridException('Inference with context failed: $e');
    }
  }

  /// Run inference with streaming output.
  ///
  /// This method uses native token-by-token streaming for LLM models,
  /// providing real-time token output as they are generated.
  /// Returns a [Stream] of [StreamToken].
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
      // Use native streaming from FFI
      final stream = inner.runStream(envelope: envelope.inner);

      var emittedFinal = false;

      await for (final event in stream) {
        switch (event) {
          case FfiStreamEvent_Token(:final field0):
            final isFinal = field0.finishReason != null;
            if (isFinal) emittedFinal = true;
            yield StreamToken(
              token: field0.token,
              index: field0.index,
              cumulativeText: field0.cumulativeText,
              isFinal: isFinal,
              finishReason: field0.finishReason,
            );
          case FfiStreamEvent_Complete(:final field0):
            // Only emit if we haven't already emitted a final token
            if (!emittedFinal) {
              yield StreamToken(
                token: '',
                index: 0,
                cumulativeText: field0.text ?? '',
                isFinal: true,
                finishReason: 'stop',
              );
            }
          case FfiStreamEvent_Error(:final field0):
            if (!emittedFinal) {
              yield StreamToken(
                token: '',
                index: 0,
                cumulativeText: '',
                isFinal: true,
                finishReason: 'error: $field0',
              );
            }
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

  /// Run inference with streaming output and conversation context.
  ///
  /// Combines streaming output with multi-turn conversation memory.
  /// The model sees the full conversation history when generating responses.
  ///
  /// **Note:** The context is NOT automatically updated with the result.
  /// You should push the final response to the context after streaming completes.
  ///
  /// ## Example
  ///
  /// ```dart
  /// final context = ConversationContext();
  /// context.setSystem('You are a helpful assistant.');
  /// context.pushText('Tell me a joke', MessageRole.user);
  ///
  /// final buffer = StringBuffer();
  /// await for (final token in model.runStreamingWithContext(
  ///   XybridEnvelope.text('Tell me a joke'),
  ///   context,
  /// )) {
  ///   stdout.write(token.token);
  ///   buffer.write(token.token);
  /// }
  /// context.pushText(buffer.toString(), MessageRole.assistant);
  /// ```
  Stream<StreamToken> runStreamingWithContext(
    XybridEnvelope envelope,
    ConversationContext context,
  ) async* {
    try {
      final stream = inner.runStreamWithContext(
        envelope: envelope.inner,
        context: context.inner,
      );

      await for (final event in stream) {
        switch (event) {
          case FfiStreamEvent_Token(:final field0):
            yield StreamToken(
              token: field0.token,
              index: field0.index,
              cumulativeText: field0.cumulativeText,
              isFinal: field0.finishReason != null,
              finishReason: field0.finishReason,
            );
          case FfiStreamEvent_Complete(:final field0):
            yield StreamToken(
              token: '',
              index: 0,
              cumulativeText: field0.text ?? '',
              isFinal: true,
              finishReason: 'stop',
            );
          case FfiStreamEvent_Error(:final field0):
            yield StreamToken(
              token: '',
              index: 0,
              cumulativeText: '',
              isFinal: true,
              finishReason: 'error: $field0',
            );
        }
      }
    } catch (e) {
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
