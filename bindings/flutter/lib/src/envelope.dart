/// Input envelope types for Xybrid inference.
///
/// This class wraps the FRB-generated [FfiEnvelope] with a clean,
/// idiomatic Dart API.
library;

// TODO: Import FRB-generated bindings when available
// import '../src/rust/api/envelope.dart';

/// Envelope containing input data for model inference.
///
/// Create envelopes using the factory constructors for different input types:
/// - [XybridEnvelope.audio] for speech recognition
/// - [XybridEnvelope.text] for text-to-speech
/// - [XybridEnvelope.embedding] for embedding models
class XybridEnvelope {
  // TODO: Replace with actual FRB type when generated
  // final FfiEnvelope _inner;

  XybridEnvelope._();

  /// Create an audio envelope for speech recognition.
  ///
  /// [bytes] - Raw audio bytes (e.g., WAV file contents)
  /// [sampleRate] - Audio sample rate in Hz (typically 16000)
  /// [channels] - Number of audio channels (typically 1 for mono)
  factory XybridEnvelope.audio({
    required List<int> bytes,
    required int sampleRate,
    int channels = 1,
  }) {
    // TODO: return XybridEnvelope._()..._inner = FfiEnvelope.audio(bytes, sampleRate, channels);
    return XybridEnvelope._();
  }

  /// Create a text envelope for text-to-speech.
  ///
  /// [text] - The text to synthesize
  /// [voiceId] - Optional voice identifier (model-specific)
  /// [speed] - Optional speed multiplier (default 1.0)
  factory XybridEnvelope.text(
    String text, {
    String? voiceId,
    double? speed,
  }) {
    // TODO: return XybridEnvelope._()..._inner = FfiEnvelope.text(text, voiceId, speed);
    return XybridEnvelope._();
  }

  /// Create an embedding envelope from float vector.
  ///
  /// [data] - The embedding vector
  factory XybridEnvelope.embedding(List<double> data) {
    // TODO: return XybridEnvelope._()..._inner = FfiEnvelope.embedding(data.cast<double>());
    return XybridEnvelope._();
  }
}
