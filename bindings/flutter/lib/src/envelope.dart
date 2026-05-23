/// Input envelope types for Xybrid inference.
///
/// This class wraps the FRB-generated [FfiEnvelope] with a clean,
/// idiomatic Dart API.
library;

import 'context.dart';
import 'rust/api/envelope.dart';

/// Envelope containing input data for model inference.
///
/// Create envelopes using the factory constructors for different input types:
/// - [XybridEnvelope.audio] for speech recognition
/// - [XybridEnvelope.text] for text-to-speech
/// - [XybridEnvelope.embedding] for embedding models
/// - [XybridEnvelope.image] for encoded image input
/// - [XybridEnvelope.userMessage] for vision-language prompts
class XybridEnvelope {
  /// The underlying FRB envelope.
  final FfiEnvelope inner;

  final _EnvelopeModality _modality;

  XybridEnvelope._(this.inner, this._modality);

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
    return XybridEnvelope._(
      FfiEnvelope.audio(
        bytes: bytes,
        sampleRate: sampleRate,
        channels: channels,
      ),
      _EnvelopeModality.audio,
    );
  }

  /// Create a text envelope for text-to-speech.
  ///
  /// [text] - The text to synthesize
  /// [voiceId] - Optional voice identifier (model-specific)
  /// [speed] - Optional speed multiplier (default 1.0)
  factory XybridEnvelope.text(String text, {String? voiceId, double? speed}) {
    return XybridEnvelope._(
      FfiEnvelope.text(text: text, voiceId: voiceId, speed: speed),
      _EnvelopeModality.text,
    );
  }

  /// Create an embedding envelope from float vector.
  ///
  /// [data] - The embedding vector
  factory XybridEnvelope.embedding(List<double> data) {
    return XybridEnvelope._(
      FfiEnvelope.embedding(data: data),
      _EnvelopeModality.embedding,
    );
  }

  /// Create an encoded image envelope for vision models.
  ///
  /// [bytes] - Encoded PNG, JPEG, or WebP image bytes
  /// [format] - Image format hint: `png`, `jpeg`, `jpg`, or `webp`
  factory XybridEnvelope.image({
    required List<int> bytes,
    required String format,
  }) {
    final normalizedFormat = _normalizeImageFormat(format);
    return XybridEnvelope._(
      FfiEnvelope.image(bytes: bytes, format: normalizedFormat),
      _EnvelopeModality.image,
    );
  }

  /// Create a user-role multi-part message with image attachments.
  ///
  /// [images] must contain envelopes created by [XybridEnvelope.image].
  factory XybridEnvelope.userMessage({
    required String text,
    List<XybridEnvelope> images = const [],
  }) {
    final nonImageIndex = images.indexWhere(
      (image) => image._modality != _EnvelopeModality.image,
    );
    if (nonImageIndex != -1) {
      throw ArgumentError.value(
        images,
        'images',
        'all attachments must be image envelopes',
      );
    }

    return XybridEnvelope._(
      FfiEnvelope.userMessage(
        text: text,
        images: images.map((image) => image.inner).toList(growable: false),
      ),
      _EnvelopeModality.multipart,
    );
  }

  /// Create a text envelope with a specific message role.
  ///
  /// This is used for building conversation context with proper role tagging.
  ///
  /// [text] - The message text
  /// [role] - The message role (system, user, or assistant)
  factory XybridEnvelope.textWithRole(String text, MessageRole role) {
    return XybridEnvelope._(
      FfiEnvelope.textWithRole(text: text, role: role.toFfi()),
      _EnvelopeModality.text,
    );
  }

  /// Set the message role on this envelope.
  ///
  /// Returns a new envelope with the role set.
  XybridEnvelope withRole(MessageRole role) {
    return XybridEnvelope._(inner.withRole(role: role.toFfi()), _modality);
  }

  /// Get the message role of this envelope, if set.
  MessageRole? get role {
    final ffiRole = inner.role();
    return ffiRole != null ? MessageRole.fromFfi(ffiRole) : null;
  }

  static String _normalizeImageFormat(String format) {
    switch (format.trim().toLowerCase()) {
      case 'png':
        return 'png';
      case 'jpg':
      case 'jpeg':
        return 'jpeg';
      case 'webp':
        return 'webp';
      default:
        throw ArgumentError.value(
          format,
          'format',
          'expected png, jpeg, jpg, or webp',
        );
    }
  }
}

enum _EnvelopeModality { audio, text, embedding, image, multipart }
