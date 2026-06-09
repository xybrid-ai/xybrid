/// Model loading API for Xybrid.
///
/// This class wraps the FRB-generated [FfiModelLoader] and [FfiModel]
/// with a clean, idiomatic Dart API.
library;

import 'dart:async';
import 'dart:typed_data';

import 'context.dart';
import 'envelope.dart';
import 'generation_config.dart';
import 'llm.dart';
import 'result.dart';
import 'rust/api/model.dart';
import 'run_options.dart';

/// Exception thrown when Xybrid operations fail.
class XybridException implements Exception {
  /// The error message.
  final String message;

  /// Creates a new [XybridException] with the given [message].
  XybridException(this.message);

  @override
  String toString() => 'XybridException: $message';
}

/// Cooperative cancel handle for an in-flight streaming run.
///
/// Construct one, pass it into [XybridModel.runStreaming],
/// [XybridModel.runStreamingWithContext], or
/// [XybridModel.runStreamingWithFallback], and call [cancel] to halt Rust
/// generation. Cancellation is cooperative: it takes effect at the next token
/// boundary (it never interrupts mid-token), and releases the model write lock
/// promptly so a follow-up run can start.
///
/// A token is single-use in spirit — create a fresh one per run.
///
/// ## Example
/// ```dart
/// final cancel = CancellationToken();
/// final stream = model.runStreaming(
///   XybridEnvelope.text('Tell me a long story'),
///   cancellationToken: cancel,
/// );
/// final sub = stream.listen((token) { ... });
/// // Later, to stop generation:
/// cancel.cancel();
/// await sub.cancel();
/// ```
class CancellationToken {
  /// The underlying FRB cancellation handle (shared with the Rust run).
  final FfiCancellationToken inner;

  /// Source of truth once [cancel] is called, so [isCancelled] stays correct
  /// after the FRB handle is reclaimed.
  bool _isCancelled = false;

  /// Create a fresh, un-cancelled token.
  CancellationToken() : inner = FfiCancellationToken();

  CancellationToken._(this.inner);

  /// Wrap an existing FRB handle.
  factory CancellationToken.fromFfi(FfiCancellationToken handle) =>
      CancellationToken._(handle);

  /// Request cooperative cancellation of the associated run.
  ///
  /// Takes effect at the next token boundary. Safe to call more than once and
  /// safe to call after the run has already finished (a no-op in that case).
  void cancel() {
    _isCancelled = true;
    try {
      inner.cancel();
    } catch (_) {
      // The run already finished and flutter_rust_bridge reclaimed the opaque
      // handle, so encoding it for the FFI call throws DroppableDisposedException.
      // Per the contract above, cancelling a completed run is a no-op — swallow
      // the use-after-dispose guard rather than surfacing it to callers (e.g. a
      // stream's `finally`/`onCancel` cleanup that races the run's completion).
    }
  }

  /// Whether cancellation has been requested. Safe to read after the run ends:
  /// a reclaimed handle returns false instead of throwing.
  bool get isCancelled {
    if (_isCancelled) return true;
    try {
      return inner.isCancelled();
    } catch (_) {
      return false;
    }
  }
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
    return XybridModelLoader._(FfiModelLoader.fromRegistry(modelId: modelId));
  }

  /// Create a loader for a model from a local bundle path.
  ///
  /// The [path] should point to a directory containing model_metadata.json.
  /// Throws if the bundle is invalid.
  factory XybridModelLoader.fromBundle(String path) {
    return XybridModelLoader._(FfiModelLoader.fromBundle(path: path));
  }

  /// Create a loader for a model from a local directory path.
  ///
  /// The [path] should point to a directory containing model_metadata.json
  /// and all required model files.
  /// Throws if the directory doesn't exist or metadata is missing/invalid.
  factory XybridModelLoader.fromDirectory(String path) {
    return XybridModelLoader._(FfiModelLoader.fromDirectory(path: path));
  }

  /// Create a loader for a model from a HuggingFace Hub repository.
  ///
  /// Downloads model files from HuggingFace and caches them locally.
  /// The [repo] should be a HuggingFace repo ID (e.g., "xybrid-ai/kokoro-82m").
  /// Model metadata is auto-generated if not present in the repo.
  ///
  /// Requires the `huggingface` feature flag to be enabled in the Rust SDK.
  factory XybridModelLoader.fromHuggingFace(String repo) {
    return XybridModelLoader._(FfiModelLoader.fromHuggingface(repo: repo));
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
/// One chunk of streamed TTS audio: raw 16-bit little-endian PCM ([pcm]) and
/// its [sampleRate] in Hz. Wrap into a WAV (header + [pcm]) to play. Emitted by
/// [XybridModel.runTtsStreaming] one sentence-chunk at a time.
class TtsAudioChunk {
  const TtsAudioChunk({required this.pcm, required this.sampleRate});

  final Uint8List pcm;
  final int sampleRate;
}

class XybridModel {
  /// The underlying FRB model.
  final FfiModel inner;

  XybridModel._(this.inner);

  /// Run inference with the given envelope.
  ///
  /// Returns [XybridResult] containing output text, audio, or embeddings
  /// depending on the model type.
  ///
  /// Pass an optional [config] to control generation parameters (temperature,
  /// top-p, etc.). When `null`, the model's default parameters are used.
  ///
  /// Throws [XybridException] if inference fails.
  Future<XybridResult> run(
    XybridEnvelope envelope, {
    GenerationConfig? config,
  }) async {
    try {
      final ffiResult = await inner.run(
        envelope: envelope.inner,
        config: config?.toFfi(),
      );
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
    ConversationContext context, {
    GenerationConfig? config,
  }) async {
    try {
      final ffiResult = await inner.runWithContext(
        envelope: envelope.inner,
        context: context.inner,
        config: config?.toFfi(),
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
  /// * [cancellationToken] - Optional cooperative cancel handle. Call
  ///   [CancellationToken.cancel] to halt generation at the next token boundary
  ///   and release the model write lock. Unsubscribing from the returned stream
  ///   also cancels the in-flight Rust run.
  /// * [preempt] - When `true` (latest-frame-wins) **and** a
  ///   [cancellationToken] is supplied, this run cancels the model's previously
  ///   in-flight streaming run before acquiring the model write lock, so a new
  ///   frame's stream does not head-of-line block behind a still-running one.
  ///   This is the seam a continuous live-capture loop flips to switch from
  ///   drop-if-busy to cancel-and-replace. Defaults to `false` (drop-if-busy /
  ///   serialized) — chat and one-shot callers leave it unset and are
  ///   unaffected. A `preempt` with no [cancellationToken] is a no-op.
  /// * [frameSessionId] - Optional caller-supplied UUID identifying one
  ///   continuous live-capture session. When set, the SDK tags every inference
  ///   in the session and rate-limits its telemetry to ~1 wire row/sec instead
  ///   of one row per frame. Leave unset for chat / one-shot runs (telemetry is
  ///   plain per-run rows, unchanged).
  ///
  /// # Example
  /// ```dart
  /// final cancel = CancellationToken();
  /// final stream = model.runStreaming(
  ///   XybridEnvelope.text('Tell me a story'),
  ///   cancellationToken: cancel,
  /// );
  /// await for (final token in stream) {
  ///   stdout.write(token.token);
  ///   if (token.isFinal) {
  ///     print('\n\nDone: ${token.finishReason}');
  ///   }
  /// }
  /// ```
  Stream<StreamToken> runStreaming(
    XybridEnvelope envelope, {
    GenerationConfig? config,
    CancellationToken? cancellationToken,
    bool preempt = false,
    String? frameSessionId,
  }) async* {
    try {
      // Use native streaming from FFI
      final stream = inner.runStream(
        envelope: envelope.inner,
        config: config?.toFfi(),
        cancellationToken: cancellationToken?.inner,
        preempt: preempt,
        frameSessionId: frameSessionId,
      );

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
                metrics: XybridInferenceMetrics.fromFfi(field0.metrics),
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

  /// Streaming TTS: synthesize [envelope]'s text sentence-chunk by
  /// sentence-chunk and yield each chunk's PCM (with its sample rate) as it is
  /// produced, instead of one batched WAV — so playback can start after the
  /// first sentence. Pass a [cancellationToken] (or unsubscribe from the
  /// stream) to stop synthesis at the next chunk boundary (barge-in).
  Stream<TtsAudioChunk> runTtsStreaming(
    XybridEnvelope envelope, {
    CancellationToken? cancellationToken,
  }) async* {
    final stream = inner.runTtsStream(
      envelope: envelope.inner,
      config: null,
      cancellationToken: cancellationToken?.inner,
    );
    await for (final event in stream) {
      switch (event) {
        case FfiTtsStreamEvent_AudioChunk(:final pcm, :final sampleRate):
          yield TtsAudioChunk(pcm: pcm, sampleRate: sampleRate);
        case FfiTtsStreamEvent_Complete():
          return;
        case FfiTtsStreamEvent_Error(:final field0):
          throw StateError('TTS streaming failed: $field0');
      }
    }
  }

  /// Run streaming inference with local abort and cloud fallback.
  ///
  /// Pass an optional [cancellationToken] to make the run user-cancellable in
  /// addition to the resource-pressure abort policy: calling
  /// [CancellationToken.cancel] (or unsubscribing from the stream) halts the
  /// local run at the next token boundary. User cancellation is terminal and
  /// never triggers cloud fallback.
  Stream<StreamToken> runStreamingWithFallback(
    XybridEnvelope envelope, {
    required RunOptions options,
    GenerationConfig? config,
    CancellationToken? cancellationToken,
  }) async* {
    try {
      final stream = inner.runStreamWithFallback(
        envelope: envelope.inner,
        options: options.toFfi(),
        config: config?.toFfi(),
        cancellationToken: cancellationToken?.inner,
      );

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
            if (!emittedFinal) {
              yield StreamToken(
                token: '',
                index: 0,
                cumulativeText: field0.text ?? '',
                isFinal: true,
                finishReason: 'stop',
                metrics: XybridInferenceMetrics.fromFfi(field0.metrics),
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
  ///
  /// Pass an optional [cancellationToken] to make the run cancellable: calling
  /// [CancellationToken.cancel] (or unsubscribing from the stream) halts
  /// generation at the next token boundary and releases the model write lock.
  ///
  /// Pass [preempt] `= true` (latest-frame-wins) together with a
  /// [cancellationToken] to cancel the model's previously in-flight streaming
  /// run before acquiring the write lock — see [runStreaming] for the full
  /// semantics. Defaults to `false` (drop-if-busy / serialized); chat leaves it
  /// unset and is unaffected.
  ///
  /// Pass an optional [frameSessionId] (a caller-supplied UUID) to tag the run
  /// as part of a continuous live-capture session — see [runStreaming] for the
  /// telemetry rate-limit semantics. Leave unset for chat / one-shot runs.
  Stream<StreamToken> runStreamingWithContext(
    XybridEnvelope envelope,
    ConversationContext context, {
    GenerationConfig? config,
    CancellationToken? cancellationToken,
    bool preempt = false,
    String? frameSessionId,
  }) async* {
    try {
      final stream = inner.runStreamWithContext(
        envelope: envelope.inner,
        context: context.inner,
        config: config?.toFfi(),
        cancellationToken: cancellationToken?.inner,
        preempt: preempt,
        frameSessionId: frameSessionId,
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
              metrics: XybridInferenceMetrics.fromFfi(field0.metrics),
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
