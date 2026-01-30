/// Inference result types for Xybrid.
///
/// This class wraps the FRB-generated [FfiResult] with a clean,
/// idiomatic Dart API.
library;

// TODO: Import FRB-generated bindings when available
// import '../src/rust/api/result.dart';

/// Result of a model inference operation.
///
/// Access the output using the appropriate getter based on model type:
/// - [text] for ASR (speech-to-text) results
/// - [audioBytes] for TTS (text-to-speech) results
/// - [embedding] for embedding model results
class XybridResult {
  // TODO: Replace with actual FRB type when generated
  // final FfiResult _inner;

  /// Internal constructor - not for public use.
  /// @nodoc
  XybridResult.internal();

  /// Whether the inference completed successfully.
  bool get success {
    // TODO: return _inner.success();
    return false;
  }

  /// Text output (for ASR models).
  ///
  /// Returns null if the model doesn't produce text output.
  String? get text {
    // TODO: return _inner.text();
    return null;
  }

  /// Audio bytes output (for TTS models).
  ///
  /// Returns null if the model doesn't produce audio output.
  List<int>? get audioBytes {
    // TODO: return _inner.audioBytes();
    return null;
  }

  /// Embedding vector output (for embedding models).
  ///
  /// Returns null if the model doesn't produce embeddings.
  List<double>? get embedding {
    // TODO: return _inner.embedding()?.cast<double>();
    return null;
  }

  /// Inference latency in milliseconds.
  int get latencyMs {
    // TODO: return _inner.latencyMs();
    return 0;
  }
}
