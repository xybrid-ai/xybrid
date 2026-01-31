/// Inference result types for Xybrid.
///
/// This class wraps the FRB-generated [FfiResult] with a clean,
/// idiomatic Dart API.
library;

import 'dart:typed_data';

import 'rust/api/result.dart';

/// Result of a model inference operation.
///
/// Access the output using the appropriate getter based on model type:
/// - [text] for ASR (speech-to-text) results
/// - [audioBytes] for TTS (text-to-speech) results
/// - [embedding] for embedding model results
class XybridResult {
  /// The underlying FRB result.
  final FfiResult _inner;

  /// Internal constructor from FRB result.
  /// @nodoc
  XybridResult.fromFfi(this._inner);

  /// Whether the inference completed successfully.
  bool get success => _inner.success;

  /// Text output (for ASR models).
  ///
  /// Returns null if the model doesn't produce text output.
  String? get text => _inner.text;

  /// Audio bytes output (for TTS models).
  ///
  /// Returns null if the model doesn't produce audio output.
  Uint8List? get audioBytes => _inner.audioBytes;

  /// Embedding vector output (for embedding models).
  ///
  /// Returns null if the model doesn't produce embeddings.
  List<double>? get embedding => _inner.embedding?.toList();

  /// Inference latency in milliseconds.
  int get latencyMs => _inner.latencyMs;
}
