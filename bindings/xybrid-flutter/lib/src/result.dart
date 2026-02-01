/// Inference result types for Xybrid.
///
/// This class wraps the FRB-generated [FfiResult] with a clean,
/// idiomatic Dart API.
library;

import 'dart:typed_data';

import 'rust/api/result.dart';
import 'utils/audio.dart';

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
  /// Returns raw PCM audio bytes (16-bit signed, little-endian).
  /// Use [audioAsWav] for playback-ready WAV format.
  ///
  /// Returns null if the model doesn't produce audio output.
  Uint8List? get audioBytes => _inner.audioBytes;

  /// Audio output as WAV format (for TTS models).
  ///
  /// Wraps the raw PCM bytes in a WAV header for easy playback.
  /// Default format is 24kHz mono (Kokoro TTS output).
  ///
  /// Example:
  /// ```dart
  /// final result = await model.run(envelope);
  /// final wavBytes = result.audioAsWav();
  /// // Play with just_audio or save as .wav file
  /// ```
  ///
  /// Returns null if the model doesn't produce audio output.
  Uint8List? audioAsWav({int sampleRate = 24000, int channels = 1}) {
    final bytes = audioBytes;
    if (bytes == null) return null;
    return wrapInWavHeader(bytes, sampleRate: sampleRate, channels: channels);
  }

  /// Embedding vector output (for embedding models).
  ///
  /// Returns null if the model doesn't produce embeddings.
  List<double>? get embedding => _inner.embedding?.toList();

  /// Inference latency in milliseconds.
  int get latencyMs => _inner.latencyMs;
}
